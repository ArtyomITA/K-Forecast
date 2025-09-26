"""Mock data generation utilities for the K-Forecast demo (skeleton).

TODO: Provide implementation for make_mock_flows that creates synthetic cash-flow
series with weekly seasonality, monthly spikes, and negative delays. The function
must persist the dataset to ``data/mock_bank_flows.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "mock_bank_flows.csv"


def make_mock_flows(
    *,
    series_ids: Sequence[str],
    start: str,
    periods: int,
    freq: str,
    seed: int,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Skeleton for the synthetic flow generator.

    TODO: Implement logic to:
    - Create multi-series daily data with trends, seasonality, spikes, and delays.
    - Return the generated DataFrame with columns ["item_id", "timestamp", "target"].
    - Save the DataFrame to ``output_path`` or ``DEFAULT_OUTPUT_PATH`` when omitted.
    """

    raise NotImplementedError("make_mock_flows needs implementation in future prompt")
