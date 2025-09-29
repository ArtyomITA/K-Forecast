"""Generate mock SD sales dataset with promotional and price covariates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_OUTPUT = Path("data/mock_sd_sales.csv")
START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2024-12-31")

SKU_LIST: Sequence[str] = ("SKU_001", "SKU_002")
CUSTOMERS: Sequence[str] = ("CUST_A", "CUST_B")
AREAS: Sequence[str] = ("AREA_NORTH", "AREA_SOUTH")

WEEKLY_MULTIPLIERS = {
    0: 1.20,  # Monday
    1: 1.20,  # Tuesday
    2: 1.20,  # Wednesday
    3: 1.20,  # Thursday
    4: 1.00,  # Friday
    5: 0.60,  # Saturday
    6: 0.60,  # Sunday
}


@dataclass(frozen=True)
class SeriesDescriptor:
    sku: str
    customer: str
    area: str

    @property
    def item_id(self) -> str:
        return f"{self.sku}|{self.customer}|{self.area}"


def _build_price_index(dates: Iterable[pd.Timestamp], rng: np.random.Generator) -> np.ndarray:
    values: list[float] = []
    level = rng.uniform(0.9, 1.1)
    for _ in dates:
        shock = rng.normal(loc=0.0, scale=0.01)
        level = float(np.clip(level + shock * 0.3, 0.8, 1.2))
        values.append(level)
    return np.asarray(values, dtype=float)


def _promo_flags(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    flags = pd.Series(0, index=index, dtype=int)
    month_periods = pd.period_range(index.min(), index.max(), freq="M")

    for period in month_periods:
        month_mask = (index.year == period.year) & (index.month == period.month)
        month_dates = index[month_mask]
        if month_dates.empty:
            continue

        if rng.random() < 0.45:
            continue  # no promo this month

        num_promos = 1 if rng.random() < 0.7 else 2
        for _ in range(num_promos):
            length = int(rng.integers(7, 11))
            if len(month_dates) <= length:
                start_idx = 0
            else:
                high = len(month_dates) - length + 1
                start_idx = int(rng.integers(0, high))
            start_date = month_dates[start_idx]
            promo_range = pd.date_range(start_date, periods=length, freq="D")
            promo_range = promo_range[(promo_range >= index.min()) & (promo_range <= index.max())]
            flags.loc[promo_range] = 1

    return flags.to_numpy(dtype=int)


def _generate_series(descriptor: SeriesDescriptor, dates: pd.DatetimeIndex, sku_price_idx: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    base_level = rng.uniform(150, 260)
    customer_factor = rng.uniform(0.8, 1.2)
    area_factor = rng.uniform(0.9, 1.1)
    promo_flags = _promo_flags(dates, rng)

    rows: list[dict[str, object]] = []
    promo_multipliers = rng.uniform(1.25, 1.4, size=len(dates))

    for idx, current_date in enumerate(dates):
        weekday = current_date.weekday()
        weekly_mult = WEEKLY_MULTIPLIERS.get(weekday, 1.0)
        promo_flag = promo_flags[idx]
        promo_effect = promo_multipliers[idx] if promo_flag else 1.0

        price_idx = sku_price_idx[idx]
        price_effect = 1.0 + (1.0 - price_idx) * 0.4

        noise = rng.normal(loc=0.0, scale=base_level * 0.08)
        demand = base_level * weekly_mult * customer_factor * area_factor
        demand *= price_effect * promo_effect
        demand += noise
        demand = max(demand, 0.0)

        rows.append(
            {
                "item_id": descriptor.item_id,
                "timestamp": current_date.date().isoformat(),
                "target": round(float(demand), 2),
                "sku": descriptor.sku,
                "customer": descriptor.customer,
                "area": descriptor.area,
                "promo_flag": int(promo_flag),
                "price_idx": round(float(price_idx), 4),
            }
        )

    return pd.DataFrame(rows, columns=[
        "item_id",
        "timestamp",
        "target",
        "sku",
        "customer",
        "area",
        "promo_flag",
        "price_idx",
    ])


def generate_dataset(output_path: Path, seed: int = 20250929) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    price_index_per_sku = {
        sku: _build_price_index(dates, rng)
        for sku in SKU_LIST
    }

    frames: list[pd.DataFrame] = []
    for sku in SKU_LIST:
        price_idx = price_index_per_sku[sku]
        for customer in CUSTOMERS:
            for area in AREAS:
                descriptor = SeriesDescriptor(sku=sku, customer=customer, area=area)
                frame = _generate_series(descriptor, dates, price_idx, rng)
                frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)
    dataset.sort_values(["item_id", "timestamp"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock SD sales dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Percorso file CSV di output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20250929,
        help="Seed per il generatore pseudo-casuale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(args.output, seed=args.seed)
    print(f"Dataset salvato in {args.output} con {len(dataset)} righe e {dataset['item_id'].nunique()} serie.")


if __name__ == "__main__":
    main()
