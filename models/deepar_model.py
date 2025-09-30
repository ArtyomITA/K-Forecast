"""DeepAR (GluonTS) forecaster for the K-Forecast demo."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation.backtest import Evaluator
from gluonts.torch import DeepAREstimator
from lightning.pytorch.callbacks import EarlyStopping

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default hyper-parameters (can be overridden)
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_CONTEXT_LENGTH_MULT = 4
DEFAULT_NUM_SAMPLES = 200
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
    use_covariates: bool = True,
    static_cols: Sequence[str] = ("sku", "customer", "area"),
    dynamic_cols: Sequence[str] = ("promo_flag", "price_idx"),
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

    available_columns = set(df.columns)
    dynamic_full: Dict[str, np.ndarray] | None = None
    dynamic_train: Dict[str, np.ndarray] | None = None
    static_features: Dict[str, np.ndarray] | None = None
    cardinality: Sequence[int] | None = None
    embedding_dimension: Sequence[int] | None = None
    num_dynamic = 0
    num_static = 0

    covariates_active = bool(use_covariates)
    if covariates_active:
        missing_dynamic = sorted(set(dynamic_cols) - available_columns)
        missing_static = sorted(set(static_cols) - available_columns)
        if missing_dynamic or missing_static:
            reasons = []
            if missing_dynamic:
                reasons.append(f"dynamic {missing_dynamic}")
            if missing_static:
                reasons.append(f"static {missing_static}")
            LOGGER.warning("DeepAR covariate fallback: missing %s", "; ".join(reasons))
            covariates_active = False

    if covariates_active and static_cols:
        dynamic_full, dynamic_feature_names = _prepare_dynamic_features(
            df,
            static_cols=static_cols,
            dynamic_cols=dynamic_cols,
        )
        static_features, cardinality = _prepare_static_features(df, static_cols)
        num_dynamic = len(dynamic_feature_names)
        num_static = len(static_cols)
        if cardinality is None:
            raise ValueError("Failed to compute static cardinality for DeepAR covariates")
        embedding_dimension = _compute_embedding_dimensions(cardinality)
        cards_label = (
            ", ".join(
                f"{col}={cardinality[idx]}"
                for idx, col in enumerate(static_cols)
            )
            if cardinality and len(cardinality) == len(static_cols)
            else "n/a"
        )
        LOGGER.info("[SD] feat_dynamic_real=%s", list(dynamic_feature_names))
        LOGGER.info("[SD] feat_static_cat=%s | cardinality={%s}", list(static_cols), cards_label)
        LOGGER.info(
            "[SD] DeepAR covariate: dyn=%d, static=%d | card={%s}",
            num_dynamic,
            num_static,
            cards_label,
        )
    elif not covariates_active and use_covariates:
        LOGGER.debug("DeepAR covariate mode disabled; proceeding with target-only input.")

    train_df = _trim_tail(df, cfg.prediction_length)
    if covariates_active and dynamic_full is not None:
        train_lengths = train_df.groupby("item_id").size().to_dict()
        dynamic_train = {
            item_id: features[:, :train_lengths[item_id]]
            for item_id, features in dynamic_full.items()
            if item_id in train_lengths
        }

    train_dataset = _to_gluonts_dataset(train_df, cfg, dynamic_train, static_features if covariates_active else None)
    full_dataset = _to_gluonts_dataset(df, cfg, dynamic_full if covariates_active else None, static_features if covariates_active else None)

    stats = calculate_dataset_statistics(full_dataset)
    LOGGER.info(
        "DeepAR dataset stats | series=%d | freq=%s | prediction_length=%d",
        stats.num_time_series,
        cfg.freq,
        cfg.prediction_length,
    )

    estimator = _build_estimator(
        cfg,
        num_feat_dynamic_real=num_dynamic if covariates_active else 0,
        num_feat_static_cat=num_static if covariates_active else 0,
        cardinality=cardinality if covariates_active else None,
        embedding_dimension=embedding_dimension if covariates_active else None,
    )

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


def train_and_predict_deepar_f110(
    df_long: pd.DataFrame,
    prediction_length: int,
    freq: str,
    **hparams: Any,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """DeepAR wrapper pre-configured for AP F110 runs."""

    static_cols = ("company_code", "bank_acct")
    dynamic_cols = ("run_flag", "volume_due_next_7d")
    dyn_with_calendar = list(dynamic_cols) + ["dow_sin", "dow_cos", "eom_flag"]
    LOGGER.info(
        "[AP][F110] feat_static_cat=%s | dyn=%s",
        list(static_cols),
        dyn_with_calendar,
    )
    return train_and_predict_deepar(
        df_long,
        prediction_length,
        freq,
        use_covariates=True,
        static_cols=static_cols,
        dynamic_cols=dynamic_cols,
        **hparams,
    )


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
        timestamps = group["timestamp"].sort_values().reset_index(drop=True)
        diffs = timestamps.diff().dropna()
        if diffs.empty:
            continue
        sample_index = pd.date_range("2000-01-01", periods=2, freq=cfg.freq)
        expected_interval = sample_index[1] - sample_index[0]
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


def _prepare_dynamic_features(
    df: pd.DataFrame,
    *,
    static_cols: Sequence[str],
    dynamic_cols: Sequence[str],
) -> tuple[Dict[Any, np.ndarray], list[str]]:
    working = df.copy()

    feature_names: list[str] = []

    processed_dynamic: set[str] = set()

    if "promo_flag" in dynamic_cols:
        working["promo_flag"] = pd.to_numeric(working["promo_flag"], errors="coerce").fillna(0.0)
        feature_names.append("promo_flag")
        processed_dynamic.add("promo_flag")

    if "price_idx" in dynamic_cols:
        if "sku" not in static_cols or "sku" not in working.columns:
            raise ValueError("price_idx normalization requires 'sku' column")
        price_series = pd.to_numeric(working["price_idx"], errors="coerce")
        price_series = price_series.groupby(working["sku"]).transform(
            lambda s: s.fillna(s.mean(skipna=True))
        )
        global_mean = float(price_series.mean(skipna=True))
        if math.isnan(global_mean):
            global_mean = 1.0
        price_series = price_series.fillna(global_mean)
        price_series = price_series.replace([np.inf, -np.inf], global_mean)
        working["price_idx_norm"] = price_series.groupby(working["sku"]).transform(_zscore_series)
        feature_names.append("price_idx_norm")
        processed_dynamic.add("price_idx")

    for col in dynamic_cols:
        if col in processed_dynamic:
            continue
        if col not in working.columns:
            continue
        series = pd.to_numeric(working[col], errors="coerce")
        if series.isna().all():
            series = pd.Series(0.0, index=series.index)
        else:
            series = series.ffill().bfill().fillna(0.0)
        working[col] = series.astype(float)
        feature_names.append(col)
        processed_dynamic.add(col)

    dow = working["timestamp"].dt.weekday.astype(float)
    working["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    working["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    working["eom_flag"] = working["timestamp"].dt.is_month_end.astype(int)

    feature_names.extend(["dow_sin", "dow_cos", "eom_flag"])

    dynamic_features: Dict[Any, np.ndarray] = {}
    for item_id, group in working.groupby("item_id"):
        group_sorted = group.sort_values("timestamp")
        matrix = np.vstack([
            group_sorted[name].to_numpy(dtype=float)
            for name in feature_names
        ])
        dynamic_features[item_id] = matrix

    return dynamic_features, feature_names


def _prepare_static_features(
    df: pd.DataFrame,
    static_cols: Sequence[str],
) -> tuple[Dict[Any, np.ndarray], list[int]]:
    working = df.copy()
    encoders: Dict[str, Dict[str, int]] = {}
    cardinality: list[int] = []

    for col in static_cols:
        col_values = working[col].fillna("__missing__").astype(str)
        working[col] = col_values
        categories = sorted(col_values.unique())
        encoders[col] = {value: idx for idx, value in enumerate(categories)}
        cardinality.append(len(categories))

    static_features: Dict[Any, np.ndarray] = {}
    for item_id, group in working.groupby("item_id"):
        group_sorted = group.sort_values("timestamp")
        encoded = [encoders[col][group_sorted[col].iloc[0]] for col in static_cols]
        static_features[item_id] = np.asarray(encoded, dtype=int)

    return static_features, cardinality


def _compute_embedding_dimensions(cardinality: Sequence[int]) -> list[int]:
    dims: list[int] = []
    for card in cardinality:
        base = max(1, int(round(pow(card, 0.25))))
        dims.append(int(min(50, max(1, base * 4))))
    return dims


def _zscore_series(series: pd.Series) -> pd.Series:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 1e-8:
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def _to_gluonts_dataset(
    df: pd.DataFrame,
    cfg: DeepARConfig,
    dynamic_features: Mapping[Any, np.ndarray] | None = None,
    static_features: Mapping[Any, np.ndarray] | None = None,
) -> Any:
    data_iter = []
    for item_id, group in df.groupby("item_id"):
        group_sorted = group.sort_values("timestamp")
        record = {
            "item_id": item_id,
            "start": group_sorted["timestamp"].iloc[0],
            "target": group_sorted["target"].to_numpy(dtype=float),
        }

        if dynamic_features and item_id in dynamic_features:
            record["feat_dynamic_real"] = dynamic_features[item_id]

        if static_features and item_id in static_features:
            record["feat_static_cat"] = static_features[item_id]

        data_iter.append(record)

    return ListDataset(data_iter=data_iter, freq=cfg.freq)


def _build_estimator(
    cfg: DeepARConfig,
    *,
    num_feat_dynamic_real: int = 0,
    num_feat_static_cat: int = 0,
    cardinality: Sequence[int] | None = None,
    embedding_dimension: Sequence[int] | None = None,
) -> DeepAREstimator:
    trainer_kwargs = {
        "max_epochs": cfg.epochs,
        "accelerator": cfg.device,
        "enable_progress_bar": cfg.use_progress_bar,
        "enable_model_summary": False,
        "log_every_n_steps": 50,
        "callbacks": [EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience)],
    }

    estimator_kwargs: Dict[str, Any] = {
        "prediction_length": cfg.prediction_length,
        "context_length": cfg.context_length,
        "freq": cfg.freq,
        "batch_size": cfg.batch_size,
        "lr": cfg.learning_rate,
        "num_parallel_samples": cfg.num_samples,
        "trainer_kwargs": trainer_kwargs,
    }

    if num_feat_dynamic_real:
        estimator_kwargs["num_feat_dynamic_real"] = int(num_feat_dynamic_real)

    if num_feat_static_cat:
        if not cardinality:
            raise ValueError("cardinality is required when num_feat_static_cat > 0")
        estimator_kwargs["num_feat_static_cat"] = int(num_feat_static_cat)
        estimator_kwargs["cardinality"] = list(cardinality)
        if embedding_dimension is not None:
            estimator_kwargs["embedding_dimension"] = list(embedding_dimension)

    estimator = DeepAREstimator(**estimator_kwargs)

    return estimator


def _forecasts_to_frame(df: pd.DataFrame, forecasts: Iterable, cfg: DeepARConfig) -> pd.DataFrame:
    records = []
    for forecast in forecasts:
        item_id = forecast.item_id
        start = forecast.start_date
        if isinstance(start, pd.Period):
            start_ts = start.to_timestamp(freq=cfg.freq, how="start")
        else:
            start_ts = pd.Timestamp(start)
        index = pd.date_range(
            start=start_ts,
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


__all__ = ["DeepARConfig", "train_and_predict_deepar", "train_and_predict_deepar_f110"]
