"""Utility script to generate mock bank flow time series data.

This module synthesizes multiple daily time series with weekly seasonality,
monthly outflow spikes, and occasional cash-collection delays. Outputs include
both the full dataset and a short out-of-sample slice for quick experiments.
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
class MockDataConfig:
    series_ids: tuple[str, ...] = (
        "CASH_MAIN",
        "CASH_ALT",
        "AR_CLIENT_A",
        "AR_CLIENT_B",
        "AR_CLIENT_C",
        "AR_CLIENT_D",
    )
    freq: str = "D"
    n_months: int = 24
    seed: int = 7
    d_stipendi: int = 27
    d_iva: int = 16
    horizon_days: int = 30
    start_date: str = "2023-01-01"
    base_low: float = 7_500.0
    base_high: float = 40_000.0
    base_sigma_ratio: float = 0.12
    stipendio_drop_ratio: float = 2.2
    iva_drop_ratio: float = 1.5
    delay_drop_factor: float = 0.4


def _weekly_multipliers(dates: pd.DatetimeIndex) -> np.ndarray:
    weekday = dates.weekday
    multipliers = np.ones(len(dates))
    multipliers[(weekday >= 0) & (weekday <= 3)] = 1.2  # Mon-Thu +20%
    multipliers[weekday == 4] = 1.0  # Fri baseline
    multipliers[(weekday == 5) | (weekday == 6)] = 0.6  # Sat/Sun -40%
    return multipliers


def _sample_delay_windows(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    n_windows: int = 2,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    monday_mask = dates.weekday == 0
    monday_dates = dates[monday_mask]
    if len(monday_dates) < n_windows:
        raise ValueError("Not enough Mondays to sample delay windows.")

    # Avoid first month and final fortnight for more realistic placement.
    valid = monday_dates[
        (monday_dates >= dates[0] + pd.Timedelta(days=30))
        & (monday_dates <= dates[-1] - pd.Timedelta(days=21))
    ]
    if len(valid) < n_windows:
        valid = monday_dates

    chosen = rng.choice(valid, size=n_windows, replace=False)
    windows = []
    for start in np.sort(chosen):
        end = start + pd.Timedelta(days=6)
        windows.append((pd.Timestamp(start), pd.Timestamp(end)))
    return windows


def _generate_single_series(
    series_id: str,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    config: MockDataConfig,
) -> pd.DataFrame:
    base_level = rng.uniform(config.base_low, config.base_high)
    sigma = base_level * config.base_sigma_ratio
    noise = rng.normal(0.0, sigma, size=len(dates))

    weekly = _weekly_multipliers(dates)
    values = (base_level + noise) * weekly

    # Monthly outflow spikes.
    stipendi_mask = dates.day == config.d_stipendi
    iva_mask = dates.day == config.d_iva
    values[stipendi_mask] -= base_level * config.stipendio_drop_ratio
    values[iva_mask] -= base_level * config.iva_drop_ratio

    delay_windows = _sample_delay_windows(rng, dates)
    for start, end in delay_windows:
        window_mask = (dates >= start) & (dates <= end)
        values[window_mask] *= config.delay_drop_factor

    floor = base_level * 0.05
    values = np.clip(values, floor, None)

    series_df = pd.DataFrame(
        {
            "item_id": series_id,
            "timestamp": dates,
            "target": values.round(2),
        }
    )

    LOGGER.info(
        "Serie %s | base %.0f | spikes: stipendi giorno %d, IVA giorno %d | ritardi: %s",
        series_id,
        base_level,
        config.d_stipendi,
        config.d_iva,
        ", ".join(f"{start.date()}→{end.date()}" for start, end in delay_windows),
    )
    return series_df


def generate_mock_bank_flows(config: MockDataConfig = MockDataConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    start = pd.Timestamp(config.start_date)
    end = start + pd.DateOffset(months=config.n_months) - pd.Timedelta(days=1)
    dates = pd.date_range(start, end, freq=config.freq)

    LOGGER.info(
        "Generazione %d serie con frequenza %s su intervallo %s → %s (%d giorni)",
        len(config.series_ids),
        config.freq,
        start.date(),
        end.date(),
        len(dates),
    )
    LOGGER.info("Stagionalità settimanale: +20%% lun-giov, -40%% sab/dom")
    LOGGER.info(
        "Eventi mensili: STIPENDI giorno %d, IVA giorno %d (uscite negative)",
        config.d_stipendi,
        config.d_iva,
    )

    frames: list[pd.DataFrame] = []
    for series_id in config.series_ids:
        frames.append(_generate_single_series(series_id, dates, rng, config))

    df = pd.concat(frames, ignore_index=True)

    expected_rows = len(config.series_ids) * len(dates)
    if len(df) != expected_rows:
        raise AssertionError("Unexpected number of rows generated.")
    if df["target"].isna().any():
        raise AssertionError("NaN values detected in generated data.")

    return df


def _write_outputs(df: pd.DataFrame, config: MockDataConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "mock_bank_flows.csv"
    test_path = output_dir / "mock_bank_flows_test_slice.csv"

    df_sorted = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    df_out = df_sorted.assign(timestamp=df_sorted["timestamp"].dt.strftime("%Y-%m-%d"))
    df_out.to_csv(full_path, index=False)

    cutoff = df_sorted["timestamp"].max() - pd.Timedelta(days=config.horizon_days * 2 - 1)
    test_slice = df_sorted[df_sorted["timestamp"] >= cutoff]
    test_out = test_slice.assign(timestamp=test_slice["timestamp"].dt.strftime("%Y-%m-%d"))
    test_out.to_csv(test_path, index=False)

    LOGGER.info("Salvati %d righe in %s", len(df_out), full_path)
    LOGGER.info(
        "Slice di test (%d giorni per serie) salvata in %s",
        config.horizon_days * 2,
        test_path,
    )


def _validate_no_gaps(df: pd.DataFrame, config: MockDataConfig) -> None:
    expected_counts = len(df["timestamp"].unique())
    expected_step = pd.Timedelta(days=1)
    for series_id, group in df.groupby("item_id"):
        if len(group) != expected_counts:
            raise AssertionError(f"Serie {series_id} con numero di date inatteso: {len(group)}")
        gaps = group["timestamp"].diff().dropna()
        if (gaps != expected_step).any():
            raise AssertionError(f"Serie {series_id} ha buchi temporali.")


def main(args: Iterable[str] | None = None) -> None:  # noqa: ARG001
    config = MockDataConfig()
    df = generate_mock_bank_flows(config)
    _validate_no_gaps(df, config)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data"
    _write_outputs(df, config, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
