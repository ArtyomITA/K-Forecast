"""Utility script to generate mock EBS daily row counts per bank account.

The generator creates long-format daily series with explicit weekly seasonality,
month-end workload increases, and payroll spikes. The output is designed to
feed forecasting demos that require discrete, count-based targets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EBSDatasetConfig:
    min_series: int = 3
    max_series: int = 5
    min_months: int = 18
    max_months: int = 24
    freq: str = "D"
    seed: int = 101
    start_date: str = "2023-01-01"
    base_lambda_low: float = 140.0
    base_lambda_high: float = 320.0
    weekday_boost: float = 0.15  # +15% lunedì e venerdì
    weekend_drop: float = 0.60   # -60% sabato/domenica
    month_end_boost: float = 0.30
    payroll_boost: float = 0.50


def _build_calendar(config: EBSDatasetConfig, rng: np.random.Generator) -> pd.DatetimeIndex:
    months = int(rng.integers(config.min_months, config.max_months + 1))
    start = pd.Timestamp(config.start_date)
    end = start + pd.DateOffset(months=months) - pd.Timedelta(days=1)
    return pd.date_range(start, end, freq=config.freq)


def _weekly_multiplier(dates: pd.DatetimeIndex, config: EBSDatasetConfig) -> np.ndarray:
    weekday = dates.weekday
    multipliers = np.ones(len(dates))
    multipliers[(weekday == 0) | (weekday == 4)] *= 1.0 + config.weekday_boost
    weekend_mask = (weekday == 5) | (weekday == 6)
    multipliers[weekend_mask] *= 1.0 - config.weekend_drop
    return multipliers


def _month_end_mask(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    periods = dates.to_period("M")
    for period in periods.unique():
        month_idx = np.flatnonzero(periods == period)
        last_days = int(rng.integers(2, 4))  # 2 o 3 giorni finali
        mask[month_idx[-last_days:]] = True
    return mask


def _payroll_mask(
    dates: pd.DatetimeIndex,
    periods: pd.PeriodIndex,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for period in periods.unique():
        month_idx = np.flatnonzero(periods == period)
        chosen = int(rng.choice(month_idx))
        mask[chosen] = True
    return mask


def _generate_series(
    bank_id: str,
    dates: pd.DatetimeIndex,
    weekly: np.ndarray,
    month_end_mask: np.ndarray,
    config: EBSDatasetConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    periods = dates.to_period("M")
    payroll_mask = _payroll_mask(dates, periods, rng)

    base_lambda = rng.uniform(config.base_lambda_low, config.base_lambda_high)
    lam = np.full(len(dates), base_lambda, dtype=float)
    lam *= weekly
    lam = np.where(month_end_mask, lam * (1.0 + config.month_end_boost), lam)
    lam = np.where(payroll_mask, lam * (1.0 + config.payroll_boost), lam)

    counts = rng.poisson(lam)
    counts = np.maximum(counts, 1)

    df = pd.DataFrame(
        {
            "item_id": bank_id,
            "timestamp": dates,
            "target": counts.astype(int),
        }
    )

    payroll_dates = dates[payroll_mask]
    LOGGER.debug(
        "Serie %s | base λ=%.1f | payroll=%s",
        bank_id,
        base_lambda,
        ", ".join(str(ts.date()) for ts in payroll_dates),
    )
    return df


def generate_mock_ebs_counts(config: EBSDatasetConfig = EBSDatasetConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    dates = _build_calendar(config, rng)
    weekly = _weekly_multiplier(dates, config)
    month_end_mask = _month_end_mask(dates, rng)

    n_series = int(rng.integers(config.min_series, config.max_series + 1))
    series_ids = [f"BANK_{idx:03d}" for idx in range(1, n_series + 1)]

    frames: list[pd.DataFrame] = []
    for series_id in series_ids:
        frames.append(
            _generate_series(series_id, dates, weekly, month_end_mask, config, rng)
        )

    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["item_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    expected_rows = len(series_ids) * len(dates)
    if len(df) != expected_rows:
        raise AssertionError("Unexpected number of rows generated")
    if df["target"].isna().any():
        raise AssertionError("NaN values detected in generated data")

    return df


def _validate_contiguity(df: pd.DataFrame) -> None:
    groups = df.groupby("item_id")
    expected = len(df["timestamp"].unique())
    for item_id, group in groups:
        if len(group) != expected:
            raise AssertionError(f"Serie {item_id} con numero di date inatteso ({len(group)})")
        diffs = group["timestamp"].diff().dropna()
        if not (diffs == pd.Timedelta(days=1)).all():
            raise AssertionError(f"Serie {item_id} con buchi temporali")


def _write_output(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "mock_ebs_counts.csv"
    export = df.assign(timestamp=df["timestamp"].dt.strftime("%Y-%m-%d"))
    export.to_csv(out_path, index=False)
    LOGGER.info("Salvati %d record in %s", len(export), out_path)


def _log_summary(df: pd.DataFrame, dates: pd.DatetimeIndex, config: EBSDatasetConfig) -> None:
    series_count = df["item_id"].nunique()
    days = len(dates)
    mean_day = df["target"].mean()
    LOGGER.info(
        "[MOCK][EBS] series=%d | days=%d | mean/day=%.1f | month-end boost=+%d%% | payroll spikes=1/mese",
        series_count,
        days,
        mean_day,
        int(config.month_end_boost * 100),
    )


def main(args: Iterable[str] | None = None) -> None:  # noqa: ARG001
    config = EBSDatasetConfig()
    df = generate_mock_ebs_counts(config)
    _validate_contiguity(df)
    dates = df["timestamp"].unique()
    _log_summary(df, pd.DatetimeIndex(dates), config)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data"
    _write_output(df, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
