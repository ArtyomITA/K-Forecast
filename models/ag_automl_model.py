"""AutoGluon TimeSeriesPredictor wrapper for K-Forecast."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple
import math
import re

import pandas as pd

LOGGER = logging.getLogger(__name__)

ARTIFACT_DIR = Path("artifacts/agts")
DEFAULT_ID_COL = "item_id"
DEFAULT_TS_COL = "timestamp"
DEFAULT_TARGET_COL = "target"
DEFAULT_QUANTILE_LEVELS = (0.1, 0.5, 0.9)


@dataclass
class AutoGluonConfig:
    prediction_length: int
    freq: str
    quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS
    presets: str = "medium_quality"
    time_limit: int | None = 120
    use_covariates: bool = False
    id_column: str = DEFAULT_ID_COL
    timestamp_column: str = DEFAULT_TS_COL
    target_column: str = DEFAULT_TARGET_COL
    split_date: str | None = None


def fit_predict_agts(
    df_long: pd.DataFrame,
    prediction_length: int,
    freq: str,
    **opts: Any,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Train AutoGluon TimeSeriesPredictor and return forecasts and metrics."""

    cfg = _build_config(prediction_length, freq, opts)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "AutoGluon training start | rows=%d | series=%d | freq=%s | horizon=%d | presets=%s",
        len(df_long),
        df_long[cfg.id_column].nunique(),
        cfg.freq,
        cfg.prediction_length,
        cfg.presets,
    )

    df_prepared = _prepare_dataframe(df_long, cfg)
    train_df, holdout_df = _split_data(df_prepared, cfg)

    time_series_train = _to_timeseries(train_df, cfg)

    from autogluon.timeseries import TimeSeriesPredictor

    predictor = TimeSeriesPredictor(
        prediction_length=cfg.prediction_length,
        freq=cfg.freq,
        quantile_levels=list(cfg.quantile_levels),
        path=ARTIFACT_DIR.as_posix(),
    )

    fit_kwargs: Dict[str, Any] = {}
    if cfg.time_limit:
        fit_kwargs["time_limit"] = cfg.time_limit

    start_train = time.perf_counter()
    predictor.fit(
        train_data=time_series_train,
        presets=cfg.presets,
        **fit_kwargs,
    )
    training_time = time.perf_counter() - start_train

    leaderboard = predictor.leaderboard(time_series_train, silent=True)
    LOGGER.info("AutoGluon training completed | models=%d | duration=%.2fs", len(leaderboard), training_time)
    LOGGER.debug("Leaderboard head:\n%s", leaderboard.head())

    forecast_df = _predict_to_frame(predictor, df_prepared, cfg)

    metrics: Dict[str, Any] = {
        "training_time": training_time,
        "leaderboard_models": len(leaderboard),
        "best_model": leaderboard.iloc[0]["model"] if not leaderboard.empty else None,
    }

    if not holdout_df.empty:
        LOGGER.info("Evaluating AutoGluon on holdout (%d rows)", len(holdout_df))
        holdout_ts = _to_timeseries(holdout_df, cfg)
        evaluation = predictor.evaluate(holdout_ts)
        metrics.update({f"eval_{k}": float(v) for k, v in evaluation.items()})

    artifact_payload = {
        "config": cfg.__dict__,
        "metrics": metrics,
        "leaderboard": leaderboard.head(10).to_dict(orient="records"),
    }
    artifact_path = ARTIFACT_DIR / "report.json"
    artifact_path.write_text(json.dumps(artifact_payload, indent=2, default=str))
    LOGGER.info("AutoGluon artifacts saved to %s", artifact_path)

    return forecast_df, metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(prediction_length: int, freq: str, overrides: Mapping[str, Any]) -> AutoGluonConfig:
    quantile_override = overrides.get("quantile_levels")
    if quantile_override is None:
        quantile_override = overrides.get("quantiles", DEFAULT_QUANTILE_LEVELS)

    quantile_levels = tuple(float(q) for q in quantile_override)

    return AutoGluonConfig(
        prediction_length=prediction_length,
        freq=freq,
        quantile_levels=quantile_levels,
        presets=str(overrides.get("presets", "medium_quality")),
        time_limit=overrides.get("time_limit"),
        use_covariates=bool(overrides.get("use_covariates", False)),
        id_column=str(overrides.get("id_column", DEFAULT_ID_COL)),
        timestamp_column=str(overrides.get("timestamp_column", DEFAULT_TS_COL)),
        target_column=str(overrides.get("target_column", DEFAULT_TARGET_COL)),
        split_date=overrides.get("split_date"),
    )


def _prepare_dataframe(df: pd.DataFrame, cfg: AutoGluonConfig) -> pd.DataFrame:
    required = {cfg.id_column, cfg.timestamp_column, cfg.target_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df_copy = df.copy()
    df_copy[cfg.timestamp_column] = pd.to_datetime(df_copy[cfg.timestamp_column], utc=False)
    df_copy.sort_values([cfg.id_column, cfg.timestamp_column], inplace=True)

    rename_map = {}
    if cfg.id_column != DEFAULT_ID_COL:
        rename_map[cfg.id_column] = DEFAULT_ID_COL
    if cfg.timestamp_column != DEFAULT_TS_COL:
        rename_map[cfg.timestamp_column] = DEFAULT_TS_COL
    if cfg.target_column != DEFAULT_TARGET_COL:
        rename_map[cfg.target_column] = DEFAULT_TARGET_COL

    df_copy = df_copy.rename(columns=rename_map)

    if cfg.use_covariates:
        df_copy = _add_covariates(df_copy)

    return df_copy


def _split_data(df: pd.DataFrame, cfg: AutoGluonConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.split_date:
        return df, pd.DataFrame(columns=df.columns)

    split_ts = pd.to_datetime(cfg.split_date, utc=False)
    train_mask = df[DEFAULT_TS_COL] < split_ts
    holdout = df.loc[~train_mask]
    train = df.loc[train_mask]
    return train, holdout


def _to_timeseries(df: pd.DataFrame, cfg: AutoGluonConfig):
    from autogluon.timeseries import TimeSeriesDataFrame

    return TimeSeriesDataFrame.from_data_frame(
        df,
        id_column=DEFAULT_ID_COL,
        timestamp_column=DEFAULT_TS_COL,
    )


def _predict_to_frame(predictor, df: pd.DataFrame, cfg: AutoGluonConfig) -> pd.DataFrame:
    from autogluon.timeseries import TimeSeriesDataFrame

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column=DEFAULT_ID_COL,
        timestamp_column=DEFAULT_TS_COL,
    )

    forecasts = predictor.predict(ts_df)
    if hasattr(forecasts, "reset_index"):
        forecasts = forecasts.reset_index()
    else:
        forecasts = forecasts.to_dataframe().reset_index()

    forecast_df = _harmonise_forecast_columns(forecasts)
    forecast_df = forecast_df[[DEFAULT_ID_COL, DEFAULT_TS_COL, "p10", "p50", "p90"]]
    forecast_df[DEFAULT_TS_COL] = pd.to_datetime(forecast_df[DEFAULT_TS_COL], utc=False)
    forecast_df.sort_values([DEFAULT_ID_COL, DEFAULT_TS_COL], inplace=True)
    forecast_df.reset_index(drop=True, inplace=True)
    return forecast_df


def _harmonise_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    rename_map: Dict[Any, str] = {}
    for column in result.columns:
        if column in {DEFAULT_ID_COL, DEFAULT_TS_COL}:
            continue
        normalized = _extract_quantile_label(column)
        if normalized and str(column) != normalized and normalized not in result.columns:
            rename_map[column] = normalized

    if rename_map:
        result.rename(columns=rename_map, inplace=True)

    if "p50" not in result.columns:
        for candidate in ("median", "mean"):
            if candidate in result.columns:
                result["p50"] = result[candidate]
                LOGGER.warning("AutoGluon output missing p50; using '%s' column as substitute.", candidate)
                break

    required = {"p10", "p90"}
    for missing in required - set(result.columns):
        if "p50" in result.columns:
            result[missing] = result["p50"]
            LOGGER.warning("AutoGluon output missing %s; duplicating p50 as fallback.", missing)

    if "p50" not in result.columns:
        raise ValueError("AutoGluon forecast missing central tendency (p50/mean/median).")

    return result


def _extract_quantile_label(column: Any) -> str | None:
    if column in {DEFAULT_ID_COL, DEFAULT_TS_COL}:
        return None
    if isinstance(column, tuple):
        label = "_".join(str(part) for part in column if part is not None)
    else:
        label = str(column)

    lowered = label.lower()
    if "quantile" in lowered:
        lowered = lowered.replace("quantile", "")

    match = re.search(r"(\d+(?:\.\d+)?)", lowered)
    if not match:
        return None

    value = float(match.group(1))
    if value <= 1:
        value *= 100

    if math.isclose(value, 10.0, rel_tol=1e-9, abs_tol=1e-6):
        return "p10"
    if math.isclose(value, 50.0, rel_tol=1e-9, abs_tol=1e-6):
        return "p50"
    if math.isclose(value, 90.0, rel_tol=1e-9, abs_tol=1e-6):
        return "p90"

    return None


def _add_covariates(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[DEFAULT_TS_COL]
    df = df.copy()
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_month_end"] = ts.dt.is_month_end.astype(int)
    return df


__all__ = ["fit_predict_agts", "AutoGluonConfig"]
