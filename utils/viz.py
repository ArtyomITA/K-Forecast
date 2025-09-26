"""Visualization and evaluation helpers for the K-Forecast demo."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REQUIRED_FORECAST_COLUMNS = ("item_id", "timestamp", "p10", "p50", "p90")


def build_forecast_chart(actual: pd.DataFrame, forecast: pd.DataFrame, *, item_id: str) -> go.Figure:
    """Create a Plotly figure comparing actuals and forecast quantiles."""

    _validate_columns(actual, {"item_id", "timestamp", "target"}, name="actual")
    _validate_columns(forecast, set(REQUIRED_FORECAST_COLUMNS), name="forecast")

    actual_series = actual[actual["item_id"] == item_id].sort_values("timestamp")
    forecast_series = forecast[forecast["item_id"] == item_id].sort_values("timestamp")

    if actual_series.empty and forecast_series.empty:
        raise ValueError(f"No data available for item_id='{item_id}'")

    fig = go.Figure()

    if not actual_series.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_series["timestamp"],
                y=actual_series["target"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=5),
            )
        )

    if not forecast_series.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_series["timestamp"],
                y=forecast_series["p50"],
                mode="lines",
                name="Forecast p50",
                line=dict(color="#ff7f0e", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_series["timestamp"],
                y=forecast_series["p90"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_series["timestamp"],
                y=forecast_series["p10"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(255, 127, 14, 0.2)",
                line=dict(width=0),
                name="Int. p10-p90",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Forecast vs Actual · {item_id}",
        xaxis_title="Data",
        yaxis_title="Valore",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def summarise_quantiles(forecast: pd.DataFrame) -> pd.DataFrame:
    """Return tidy quantile snapshot sorted by item and timestamp."""

    _validate_columns(forecast, set(REQUIRED_FORECAST_COLUMNS), name="forecast")
    summary = forecast[list(REQUIRED_FORECAST_COLUMNS)].copy()
    summary.sort_values(["item_id", "timestamp"], inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def compute_metrics(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Compute MAPE, sMAPE, pinball loss (0.1/0.9) and interval coverage."""

    _validate_columns(actual, {"item_id", "timestamp", "target"}, name="actual")
    _validate_columns(forecast, set(REQUIRED_FORECAST_COLUMNS), name="forecast")

    merged = pd.merge(
        actual,
        forecast,
        on=["item_id", "timestamp"],
        how="inner",
    )

    metrics = []
    for item_id, group in merged.groupby("item_id"):
        y_true = group["target"].to_numpy(dtype=float)
        p10 = group["p10"].to_numpy(dtype=float)
        p50 = group["p50"].to_numpy(dtype=float)
        p90 = group["p90"].to_numpy(dtype=float)

        metrics.append(
            {
                "item_id": item_id,
                "MAPE": _mape(y_true, p50),
                "sMAPE": _smape(y_true, p50),
                "Pinball@0.1": _pinball_loss(y_true, p10, 0.1),
                "Pinball@0.9": _pinball_loss(y_true, p90, 0.9),
                "Coverage[p10,p90]": _coverage(y_true, p10, p90),
            }
        )

    return pd.DataFrame(metrics)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: Sequence[str] | set[str], *, name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} dataframe missing columns: {sorted(missing)}")


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator < 1e-8, 1.0, denominator)
    return float(np.mean(200 * np.abs(y_true - y_pred) / denominator))


def _pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, q: float) -> float:
    diff = y_true - y_quantile
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within) * 100)
