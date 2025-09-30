"""Generate synthetic F110 payout data with covariates for mock forecasting.

This script creates a long-format dataset with engineered calendar features that mimic
AP payment runs driven by SAP F110. Runs typically occur on Tuesdays and Fridays,
with occasional skips around holidays and a visible boost on end-of-month dates.

Usage
-----
python scripts/generate_mock_f110_payouts.py --output data/mock_f110_payouts.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("k_forecast.mock.f110")


DEFAULT_COMBINATIONS: Sequence[Tuple[str, str]] = (
    ("CC_1000", "BANK_001"),
    ("CC_1000", "BANK_002"),
    ("CC_2000", "BANK_001"),
    ("CC_2000", "BANK_003"),
    ("CC_3000", "BANK_002"),
)


@dataclass
class GeneratorConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    combinations: Sequence[Tuple[str, str]]
    seed: int


def _build_run_schedule(dates: pd.DatetimeIndex, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    weekday = dates.weekday
    candidates = np.isin(weekday, (1, 4))  # Tuesday and Friday
    run_flag = np.zeros(len(dates), dtype=bool)
    total_candidates = int(candidates.sum())
    skipped_total = 0

    if total_candidates == 0:
        return run_flag, 0.0

    periods = dates.to_period("M")
    for period in periods.unique():
        month_mask = periods == period
        month_candidates = np.flatnonzero(candidates & month_mask)
        if month_candidates.size == 0:
            continue
        skip_count = int(rng.integers(1, 3))
        skip_count = min(skip_count, month_candidates.size)
        skip_idx = ()
        if skip_count > 0:
            skip_idx = tuple(rng.choice(month_candidates, size=skip_count, replace=False))
            skipped_total += len(skip_idx)
        run_flag[month_candidates] = True
        if skip_count > 0:
            run_flag[list(skip_idx)] = False

    skip_pct = skipped_total / total_candidates if total_candidates > 0 else 0.0
    return run_flag, skip_pct


def _simulate_series(
    dates: pd.DatetimeIndex,
    company_code: str,
    bank_acct: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_days = len(dates)
    run_flag, skip_pct = _build_run_schedule(dates, rng)
    eom_flag = dates.is_month_end.astype(int)

    base_run = rng.uniform(250_000, 620_000)
    eom_boost = rng.uniform(0.20, 0.35)
    phase = rng.uniform(0, 2 * np.pi)

    day_of_year = dates.dayofyear.to_numpy()
    seasonal = 1.0 + 0.06 * np.sin(2 * np.pi * day_of_year / 365.25 + phase)

    daily_inflow = base_run * (0.30 + 0.22 * rng.random(n_days)) * seasonal
    backlog = base_run * rng.uniform(0.4, 0.9)

    target = np.zeros(n_days, dtype=float)
    volume_due = np.zeros(n_days, dtype=float)

    for idx in range(n_days):
        backlog += daily_inflow[idx]
        if eom_flag[idx]:
            backlog += base_run * eom_boost * (0.6 + 0.4 * rng.random())
        due_proj = backlog * (1.0 + rng.normal(0.0, 0.04))
        volume_due[idx] = max(due_proj, 0.0)

        if run_flag[idx]:
            payout_factor = 0.72 + 0.22 * rng.random()
            payout = backlog * payout_factor
            if eom_flag[idx]:
                payout *= 1.0 + eom_boost
            payout *= max(0.9 + 0.1 * rng.random(), 0.01)
            target[idx] = max(payout, 0.0)
            backlog = max(backlog - payout, base_run * 0.15 * rng.random())
        else:
            target[idx] = 0.0

    data = pd.DataFrame(
        {
            "item_id": f"{company_code}|{bank_acct}",
            "timestamp": dates,
            "target": target,
            "company_code": company_code,
            "bank_acct": bank_acct,
            "run_flag": run_flag.astype(int),
            "eom_flag": eom_flag,
            "volume_due_next_7d": volume_due,
        }
    )

    run_mask = data["run_flag"] == 1
    eom_mask = (data["run_flag"] == 1) & (data["eom_flag"] == 1)
    non_eom_mask = run_mask & ~eom_mask

    if data.loc[eom_mask, "target"].sum() > 0 and data.loc[non_eom_mask, "target"].sum() > 0:
        current_ratio = data.loc[eom_mask, "target"].mean() / data.loc[non_eom_mask, "target"].mean()
        desired_ratio = 1.0 + eom_boost
        if current_ratio > 0:
            scale = desired_ratio / current_ratio
            data.loc[eom_mask, "target"] *= scale

    runs = int(data["run_flag"].sum())
    months = max(len(pd.period_range(dates[0], dates[-1], freq="M")), 1)
    runs_per_month = runs / months

    run_values = data.loc[run_mask, "target"]
    eom_values = data.loc[eom_mask, "target"]
    if not run_values.empty and not eom_values.empty:
        boost_pct = (eom_values.mean() / run_values.mean() - 1.0) * 100.0
    else:
        boost_pct = eom_boost * 100.0

    LOGGER.info(
        "[MOCK][F110] series=%s | runs/mese=%.1f | eom boost=+%.1f%% | skip runs=%.1f%%",
        data.loc[0, "item_id"],
        runs_per_month,
        boost_pct,
        skip_pct * 100.0,
    )

    return data


def generate_mock_f110_payouts(config: GeneratorConfig) -> pd.DataFrame:
    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    rng = np.random.default_rng(config.seed)

    frames: List[pd.DataFrame] = []
    for company_code, bank_acct in config.combinations:
        series_df = _simulate_series(dates, company_code, bank_acct, rng)
        frames.append(series_df)

    result = pd.concat(frames, ignore_index=True)
    dow = result["timestamp"].dt.weekday
    result["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Ensure non-run days have zero target
    result.loc[result["run_flag"] == 0, "target"] = 0.0

    result.sort_values(["item_id", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock F110 payout data with covariates.")
    parser.add_argument("--start-date", type=pd.Timestamp, default=pd.Timestamp("2023-10-01"))
    parser.add_argument("--end-date", type=pd.Timestamp, default=pd.Timestamp("2025-09-30"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mock_f110_payouts.csv"),
        help="Destination CSV file for the generated dataset.",
    )
    return parser.parse_args(args=args)


def main(cli_args: Sequence[str] | None = None) -> None:
    args = parse_args(cli_args)
    output_path: Path = args.output

    config = GeneratorConfig(
        start_date=args.start_date.normalize(),
        end_date=args.end_date.normalize(),
        combinations=DEFAULT_COMBINATIONS,
        seed=args.seed,
    )

    if config.end_date < config.start_date:
        raise ValueError("end-date must be on or after start-date")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = generate_mock_f110_payouts(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_write = df.copy()
    df_to_write["timestamp"] = df_to_write["timestamp"].dt.strftime("%Y-%m-%d")
    df_to_write.to_csv(output_path, index=False)

    LOGGER.info("Wrote %d rows to %s", len(df_to_write), output_path)


if __name__ == "__main__":
    main()
