"""Chronos-Bolt wrapper for probabilistic forecasting.

This module exposes :func:`predict_chronos`, a convenience wrapper that
leverages AutoGluon's Chronos integration to deliver zero-shot forecasts and an
optional light fine-tuning loop against the provided historical data. The
implementation keeps all computation local (CPU or CUDA) and returns tidy
quantile forecasts ready to be consumed by the Streamlit UI.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration placeholders (can be overridden via kwargs)
# ---------------------------------------------------------------------------
DEFAULT_CHRONOS_MODEL_ID = "amazon/chronos-bolt-small"
DEFAULT_CONTEXT_LENGTH_MULT = 4
DEFAULT_QUANTILES = ("0.1", "0.5", "0.9")
DEFAULT_MAX_STEPS_FINETUNE = 200
DEFAULT_DEVICE = "cpu"


@dataclass
class ChronosConfig:
    """Runtime configuration for the Chronos forecaster."""

    model_id: str = DEFAULT_CHRONOS_MODEL_ID
    context_length_mult: int = DEFAULT_CONTEXT_LENGTH_MULT
    quantiles: Sequence[str] = DEFAULT_QUANTILES
    device: str = DEFAULT_DEVICE
    max_steps_finetune: int = DEFAULT_MAX_STEPS_FINETUNE
    num_samples: int = 200


def predict_chronos(
    df_long: pd.DataFrame,
    prediction_length: int,
    freq: str,
    *,
    zero_shot: bool = True,
    do_finetune: bool = False,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Generate Chronos-Bolt forecasts for multiple time series.

    Parameters
    ----------
    df_long:
        Historical data in tidy/long format with columns ``item_id``,
        ``timestamp`` and ``target``.
    prediction_length:
        Horizon length in the same units as ``freq``.
    freq:
        Pandas offset alias (e.g. ``"D"``).
    zero_shot:
        When ``True`` (default) the model performs inference without training.
    do_finetune:
        When ``True`` a lightweight fine-tuning stage is executed using
        AutoGluon's Chronos integration. Requires the AutoGluon timeseries
        module.
    logger:
        Optional logger instance. By default the module-level logger is used.
    **kwargs:
        Optional overrides for :class:`ChronosConfig` fields and additional
        AutoGluon parameters (for example ``model_id``, ``context_length_mult``,
        ``quantiles``, ``device``, ``max_steps_finetune``, ``num_samples``).

    Returns
    -------
    pandas.DataFrame
        Forecast frame with columns ``item_id``, ``timestamp``, ``p10``,
        ``p50`` and ``p90``.
    """

    log = logger or LOGGER
    cfg = _build_config(kwargs)

    # ------------------------------------------------------------------
    # Input validation & preparation
    # ------------------------------------------------------------------
    required_columns = {"item_id", "timestamp", "target"}
    missing = required_columns - set(df_long.columns)
    if missing:
        raise ValueError(f"predict_chronos expected columns {sorted(required_columns)}; missing {sorted(missing)}")

    df = df_long.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df.sort_values(["item_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_series = df["item_id"].nunique()
    context_length = max(cfg.context_length_mult * prediction_length, prediction_length)

    log.info(
        "Chronos request received | rows=%d series=%d | horizon=%d | freq=%s | model=%s",
        len(df),
        n_series,
        prediction_length,
        freq,
        cfg.model_id,
    )

    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError as exc:  # pragma: no cover - safety guard
        raise ImportError(
            "AutoGluon[tabular, timeseries] is required for Chronos integration. "
            "Install it via `pip install autogluon>=1.4.0`"
        ) from exc

    time_series = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # ------------------------------------------------------------------
    # Build AutoGluon predictor
    # ------------------------------------------------------------------
    quantile_levels = [float(q) for q in cfg.quantiles]
    chronos_params: MutableMapping[str, Any] = {
        "model_path": cfg.model_id,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "freq": freq,
        "quantile_levels": quantile_levels,
        "device": cfg.device,
        "num_samples": cfg.num_samples,
        "use_fine_tuning": bool(do_finetune),
        "max_steps_finetune": cfg.max_steps_finetune,
        "use_zero_shot": bool(zero_shot),
    }

    # Allow caller overrides for advanced knobs (e.g. learning rate)
    extra_model_args = kwargs.get("chronos_model_args", {})
    chronos_params.update(extra_model_args)

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq=freq,
        eval_metric=kwargs.get("eval_metric", "MAPE"),
    )

    fit_kwargs: Dict[str, Any] = {}
    time_limit = kwargs.get("time_limit")
    if time_limit is not None:
        fit_kwargs["time_limit"] = time_limit

    log.debug("Chronos hyperparameters: %s", chronos_params)
    start_train = time.perf_counter()
    predictor.fit(
        train_data=time_series,
        hyperparameters={"Chronos": chronos_params},
        presets=None,
        **fit_kwargs,
    )
    train_duration = time.perf_counter() - start_train
    log.info(
        "Chronos fine-tune duration: %.2fs (do_finetune=%s)",
        train_duration,
        do_finetune,
    )

    # ------------------------------------------------------------------
    # Forecast & reshape output
    # ------------------------------------------------------------------
    start_infer = time.perf_counter()
    predictions = predictor.predict(time_series)
    inference_duration = time.perf_counter() - start_infer
    log.info("Chronos inference duration: %.2fs", inference_duration)

    forecast_df = _format_forecast_frame(predictions, cfg.quantiles)
    log.info("Chronos forecast completed | rows=%d", len(forecast_df))

    return forecast_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config(overrides: Mapping[str, Any]) -> ChronosConfig:
    cfg_kwargs = {k: overrides.get(k, getattr(ChronosConfig(), k)) for k in ChronosConfig().__dataclass_fields__.keys()}
    quantiles = overrides.get("quantiles") or cfg_kwargs["quantiles"]
    cfg_kwargs["quantiles"] = tuple(str(q) for q in quantiles)
    cfg = ChronosConfig(**cfg_kwargs)
    return cfg


def _format_forecast_frame(forecast, quantiles: Sequence[str]) -> pd.DataFrame:
    """Convert AutoGluon ``TimeSeriesDataFrame`` predictions to tidy frame."""

    forecast_pd = forecast.reset_index()

    quantile_column_map = _resolve_quantile_columns(forecast_pd, quantiles)
    missing = set(quantiles) - set(quantile_column_map)
    if missing:
        raise KeyError(f"Forecast output missing quantiles: {sorted(missing)}")

    tidy = forecast_pd[["item_id", "timestamp"]].copy()
    for q in quantiles:
        alias = _quantile_alias(q)
        tidy[alias] = forecast_pd[quantile_column_map[q]]

    tidy.sort_values(["item_id", "timestamp"], inplace=True)
    tidy.reset_index(drop=True, inplace=True)
    return tidy


def _resolve_quantile_columns(df: pd.DataFrame, quantiles: Sequence[str]) -> Dict[str, Any]:
    """Map requested quantiles to DataFrame column labels."""

    col_map: Dict[str, Any] = {}
    for column in df.columns:
        if column in {"item_id", "timestamp"}:
            continue
        label = _normalise_column_label(column)
        if label in quantiles and label not in col_map:
            col_map[label] = column
    return col_map


def _normalise_column_label(column: Any) -> str:
    if isinstance(column, tuple):
        column = column[-1]
    if isinstance(column, float):
        return f"{column:.1f}".rstrip("0").rstrip(".")
    text = str(column)
    if text.startswith("quantile_"):
        return text.split("_", 1)[1]
    if text.startswith("p") and text[1:].isdigit():
        return f"0.{text[1:]}"
    return text


def _quantile_alias(q: str) -> str:
    percentage = int(round(float(q) * 100))
    return f"p{percentage}"


__all__ = [
    "ChronosConfig",
    "predict_chronos",
]
