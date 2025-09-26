"""DeepAR (GluonTS) forecaster for the K-Forecast demo."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from gluonts.dataset.common import ListDataset
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation.backtest import Evaluator
from gluonts.torch import DeepAREstimator
from pandas.tseries.frequencies import to_offset
from pytorch_lightning.callbacks import EarlyStopping

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default hyper-parameters (can be overridden)
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_CONTEXT_LENGTH_MULT = 4
DEFAULT_NUM_SAMPLES = 100
DEFAULT_WORKERS = 0
DEFAULT_DEVICE = "cpu"
DEFAULT_FREQUENCY = "D"


@dataclass
class DeepARConfig:
    prediction_length: int
    freq: str = DEFAULT_FREQUENCY
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    context_length_mult: int = DEFAULT_CONTEXT_LENGTH_MULT
    num_samples: int = DEFAULT_NUM_SAMPLES
    num_workers: int = DEFAULT_WORKERS
    device: str = DEFAULT_DEVICE
    use_progress_bar: bool = False
    early_stop_patience: int = 5

    @property
    def context_length(self) -> int:
        return max(self.context_length_mult * self.prediction_length, self.prediction_length)


def train_and_predict_deepar(
    df_long: pd.DataFrame,
    prediction_length: int,
    freq: str,
    **hparams: Any,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Train DeepAR locally and produce probabilistic forecasts."""

    cfg = _build_config(prediction_length, freq, hparams)
    required_columns = {"item_id", "timestamp", "target"}
    missing = required_columns - set(df_long.columns)
    if missing:
        raise ValueError(f"DeepAR expects columns {sorted(required_columns)}; missing {sorted(missing)}")

    df = df_long.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df.sort_values(["item_id", "timestamp"], inplace=True)

    _warn_if_irregular(df, cfg)

    train_df = _trim_tail(df, cfg.prediction_length)
    train_dataset = _to_gluonts_dataset(train_df, cfg)
    full_dataset = _to_gluonts_dataset(df, cfg)

    stats = calculate_dataset_statistics(full_dataset)
    LOGGER.info(
        "DeepAR dataset stats | series=%d | freq=%s | prediction_length=%d",
        stats.num_time_series,
        cfg.freq,
        cfg.prediction_length,
    )

    estimator = _build_estimator(cfg)

    start_train = time.perf_counter()
    predictor = estimator.train(training_data=train_dataset, validation_data=full_dataset)
    training_time = time.perf_counter() - start_train

    metrics = {
        "training_time": training_time,
        "epochs": cfg.epochs,
    }

    LOGGER.info(
        "DeepAR training completed | epochs=%d | duration=%.2fs",
        cfg.epochs,
        training_time,
    )

    start_pred = time.perf_counter()
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=full_dataset,
        predictor=predictor,
        num_samples=cfg.num_samples,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    prediction_time = time.perf_counter() - start_pred

    LOGGER.info(
        "DeepAR prediction completed | samples=%d | duration=%.2fs",
        cfg.num_samples,
        prediction_time,
    )

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(tss), iter(forecasts))
    metrics.update({
        "RMSE": float(agg_metrics.get("RMSE", math.nan)),
        "MAPE": float(agg_metrics.get("MAPE", math.nan)),
        "mean_wQuantileLoss": float(agg_metrics.get("mean_wQuantileLoss", math.nan)),
    })

    LOGGER.info(
        "DeepAR final mean_wQuantileLoss=%.4f | MAPE=%.4f",
        metrics["mean_wQuantileLoss"],
        metrics["MAPE"],
    )

    forecast_df = _forecasts_to_frame(df, forecasts, cfg)

    return forecast_df, metrics


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_config(prediction_length: int, freq: str, overrides: Mapping[str, Any]) -> DeepARConfig:
    cfg_kwargs = {
        "prediction_length": prediction_length,
        "freq": freq,
        "epochs": int(overrides.get("epochs", DEFAULT_EPOCHS)),
        "batch_size": int(overrides.get("batch_size", DEFAULT_BATCH_SIZE)),
        "learning_rate": float(overrides.get("learning_rate", DEFAULT_LEARNING_RATE)),
        "context_length_mult": int(overrides.get("context_length_mult", DEFAULT_CONTEXT_LENGTH_MULT)),
        "num_samples": int(overrides.get("num_samples", DEFAULT_NUM_SAMPLES)),
        "num_workers": int(overrides.get("num_workers", DEFAULT_WORKERS)),
        "device": str(overrides.get("device", DEFAULT_DEVICE)),
        "use_progress_bar": bool(overrides.get("use_progress_bar", True)),
        "early_stop_patience": int(overrides.get("early_stop_patience", 5)),
    }
    return DeepARConfig(**cfg_kwargs)


def _warn_if_irregular(df: pd.DataFrame, cfg: DeepARConfig) -> None:
    irregular_series = []
    for item_id, group in df.groupby("item_id"):
        timestamps = group["timestamp"].sort_values().to_series(index=None)
        diffs = timestamps.diff().dropna()
        if diffs.empty:
            continue
        expected_interval = (
            pd.date_range("2000-01-01", periods=2, freq=cfg.freq)
            .to_series(index=None)
            .diff()
            .dropna()
            .iloc[0]
        )
        if (diffs != expected_interval).any():
            irregular_series.append(item_id)
    if irregular_series:
        LOGGER.warning(
            "Serie irregolari rilevate (%s). Si consiglia di eseguire un resample a %s.",
            ", ".join(map(str, irregular_series)),
            cfg.freq,
        )


def _trim_tail(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    trimmed_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("item_id"):
        if len(group) > horizon:
            trimmed_parts.append(group.iloc[:-horizon])
        else:
            trimmed_parts.append(group)
    return pd.concat(trimmed_parts, ignore_index=True)


def _to_gluonts_dataset(df: pd.DataFrame, cfg: DeepARConfig) -> Any:
    freq_offset = to_offset(cfg.freq)
    start_times = {}
    targets = {}
    for item_id, group in df.groupby("item_id"):
        group_sorted = group.sort_values("timestamp")
        start_times[item_id] = group_sorted["timestamp"].iloc[0]
        targets[item_id] = group_sorted["target"].to_numpy(dtype=float)

    return ListDataset(
        data_iter=[
            {
                "item_id": item_id,
                "start": start_times[item_id],
                "target": targets[item_id],
            }
            for item_id in targets
        ],
        freq=cfg.freq,
    )


def _build_estimator(cfg: DeepARConfig) -> DeepAREstimator:
    trainer_kwargs = {
        "max_epochs": cfg.epochs,
        "accelerator": cfg.device,
        "enable_progress_bar": cfg.use_progress_bar,
        "enable_model_summary": False,
        "log_every_n_steps": 50,
        "callbacks": [EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience)],
    }

    estimator = DeepAREstimator(
        prediction_length=cfg.prediction_length,
        context_length=cfg.context_length,
        freq=cfg.freq,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        num_parallel_samples=cfg.num_samples,
        trainer_kwargs=trainer_kwargs,
    )

    return estimator


def _forecasts_to_frame(df: pd.DataFrame, forecasts: Iterable, cfg: DeepARConfig) -> pd.DataFrame:
    records = []
    for forecast in forecasts:
        item_id = forecast.item_id
        index = pd.date_range(
            start=forecast.start_date,
            periods=cfg.prediction_length,
            freq=cfg.freq,
        )
        p10 = forecast.quantile(0.1)
        p50 = forecast.quantile(0.5)
        p90 = forecast.quantile(0.9)

        for ts, q10, q50, q90 in zip(index, p10, p50, p90):
            records.append(
                {
                    "item_id": item_id,
                    "timestamp": ts,
                    "p10": q10,
                    "p50": q50,
                    "p90": q90,
                }
            )

    forecast_df = pd.DataFrame(records)
    forecast_df.sort_values(["item_id", "timestamp"], inplace=True)
    forecast_df.reset_index(drop=True, inplace=True)
    return forecast_df


__all__ = ["DeepARConfig", "train_and_predict_deepar"]
