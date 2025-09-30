"""Visualization and evaluation helpers for the K-Forecast demo."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

from utils.theme import KT_GREY_STROKE, KT_ORANGE, KT_SUCCESS, KT_TEXT_MUTED, PLOTLY_TEMPLATE

LOGGER = logging.getLogger("k_forecast.viz")


@dataclass(frozen=True)
class EventBand:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp

REQUIRED_FORECAST_COLUMNS = ("item_id", "timestamp", "p10", "p50", "p90")


def _dt_array(s: pd.Series) -> np.ndarray:
    return np.array(pd.to_datetime(s))


def build_count_bar_chart(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    item_id: str,
    events: Sequence[EventBand] | None = None,
    bar_name: str = "Righe FEBEP",
    freq: str | None = "D",
    sap_note: str | None = None,
) -> go.Figure:
    """Render a bar + fan chart tailored for EBS daily counts."""

    _validate_columns(actual, {"item_id", "timestamp", "target"}, name="actual")
    _validate_columns(forecast, set(REQUIRED_FORECAST_COLUMNS), name="forecast")

    actual_series = actual[actual["item_id"] == item_id].sort_values("timestamp").copy()
    forecast_series = forecast[forecast["item_id"] == item_id].sort_values("timestamp").copy()

    actual_series["timestamp"] = pd.to_datetime(actual_series["timestamp"])
    forecast_series["timestamp"] = pd.to_datetime(forecast_series["timestamp"])

    x_a = _dt_array(actual_series["timestamp"]) if not actual_series.empty else np.array([])
    x_f = _dt_array(forecast_series["timestamp"]) if not forecast_series.empty else np.array([])

    if actual_series.empty and forecast_series.empty:
        raise ValueError(f"No data available for item_id='{item_id}'")

    fig = go.Figure()

    if not actual_series.empty:
        fig.add_trace(
            go.Bar(
                x=x_a,
                y=actual_series["target"].astype(float),
                name=bar_name,
                marker_color=_hex_to_rgba(KT_GREY_STROKE, 0.85),
                hovertemplate="%{x|%Y-%m-%d}: %{y:,} righe<extra></extra>",
            )
        )

    if not forecast_series.empty:
        fig.add_trace(
            go.Scatter(
                x=x_f,
                y=forecast_series["p90"],
                mode="lines",
                line=dict(width=0, color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_f,
                y=forecast_series["p10"],
                mode="lines",
                fill="tonexty",
                fillcolor=_hex_to_rgba(KT_ORANGE, 0.20),
                line=dict(width=0, color="rgba(0,0,0,0)"),
                name="Banda p10–p90",
                hovertemplate="%{x|%Y-%m-%d}: %{y:,}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_f,
                y=forecast_series["p50"],
                mode="lines+markers",
                name="Forecast p50",
                line=dict(color=KT_ORANGE, width=3),
                marker=dict(size=5, color=KT_ORANGE),
                hovertemplate="%{x|%Y-%m-%d}: %{y:,} righe<extra></extra>",
            )
        )

    if events:
        for event in events:
            fig.add_vrect(
                x0=event.start,
                x1=event.end,
                fillcolor=_hex_to_rgba(KT_ORANGE, 0.10),
                line=dict(color=_hex_to_rgba(KT_ORANGE, 0.4), width=1),
                layer="below",
                annotation=dict(
                    text=event.label,
                    textangle=270,
                    font=dict(color=KT_ORANGE, size=11),
                    x=event.start,
                    xshift=-6,
                    y=1,
                    yref="paper",
                    showarrow=False,
                ),
            )

    y_axis_title = "Conteggi giornalieri"
    fig.update_layout(
        title=f"EBS workload · {item_id}",
        xaxis_title="Data",
        yaxis_title=y_axis_title,
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        barmode="overlay",
    )

    if sap_note:
        fig.add_annotation(
            text=sap_note,
            xref="paper",
            yref="paper",
            x=0,
            y=1.15,
            showarrow=False,
            font=dict(size=11, color=KT_TEXT_MUTED),
        )

    if freq:
        fig.update_xaxes(type="date")

    return fig


def build_forecast_chart(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    item_id: str,
    events: Sequence[EventBand] | None = None,
    sap_note: str | None = None,
) -> go.Figure:
    """Create a Plotly figure comparing actuals and forecast quantiles."""

    _validate_columns(actual, {"item_id", "timestamp", "target"}, name="actual")
    _validate_columns(forecast, set(REQUIRED_FORECAST_COLUMNS), name="forecast")

    actual_series = actual[actual["item_id"] == item_id].sort_values("timestamp").copy()
    forecast_series = forecast[forecast["item_id"] == item_id].sort_values("timestamp").copy()

    actual_series["timestamp"] = pd.to_datetime(actual_series["timestamp"])
    forecast_series["timestamp"] = pd.to_datetime(forecast_series["timestamp"])

    x_actual = _dt_array(actual_series["timestamp"]) if not actual_series.empty else np.array([])
    x_forecast = _dt_array(forecast_series["timestamp"]) if not forecast_series.empty else np.array([])

    if actual_series.empty and forecast_series.empty:
        raise ValueError(f"No data available for item_id='{item_id}'")

    fig = go.Figure()

    if not actual_series.empty:
        fig.add_trace(
            go.Scatter(
                x=x_actual,
                y=actual_series["target"],
                mode="lines+markers",
                name="Actual",
                line=dict(color=KT_SUCCESS, width=2.5),
                marker=dict(size=5, color=KT_SUCCESS),
            )
        )

    if not forecast_series.empty:
        fig.add_trace(
            go.Scatter(
                x=x_forecast,
                y=forecast_series["p50"],
                mode="lines",
                name="Forecast p50",
                line=dict(color=KT_ORANGE, width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_forecast,
                y=forecast_series["p90"],
                mode="lines",
                line=dict(width=0, color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_forecast,
                y=forecast_series["p10"],
                mode="lines",
                fill="tonexty",
                fillcolor=_hex_to_rgba(KT_ORANGE, 0.18),
                line=dict(width=0, color="rgba(0,0,0,0)"),
                name="Intervallo p10-p90",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
            )
        )

    if events:
        for event in events:
            fig.add_vrect(
                x0=event.start,
                x1=event.end,
                fillcolor=_hex_to_rgba(KT_ORANGE, 0.08),
                line=dict(color=_hex_to_rgba(KT_ORANGE, 0.35), width=1),
                layer="below",
                annotation=dict(
                    text=event.label,
                    textangle=270,
                    font=dict(color=KT_ORANGE, size=11),
                    x=event.start,
                    xshift=-6,
                    y=1,
                    yref="paper",
                    showarrow=False,
                ),
            )

    fig.update_layout(
        title=f"Forecast vs Actual · {item_id}",
        xaxis_title="Data",
        yaxis_title="Valore",
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    if sap_note:
        fig.add_annotation(
            text=sap_note,
            xref="paper",
            yref="paper",
            x=0,
            y=1.15,
            showarrow=False,
            font=dict(size=11, color=KT_TEXT_MUTED),
        )

    fig.update_xaxes(
        rangeselector=dict(
            bgcolor=_hex_to_rgba(KT_GREY_STROKE, 0.35),
            activecolor=_hex_to_rgba(KT_ORANGE, 0.5),
            buttons=[
                dict(count=7, label="7g", step="day", stepmode="backward"),
                dict(count=30, label="30g", step="day", stepmode="backward"),
                dict(count=90, label="90g", step="day", stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
        rangeslider=dict(visible=True, bgcolor=_hex_to_rgba(KT_GREY_STROKE, 0.25)),
        type="date",
    )

    return fig


def build_comparison_chart(
    actual: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    *,
    item_id: str,
    sap_note: str | None = None,
) -> go.Figure:
    """Render comparison of model medians for a single item."""

    _validate_columns(actual, {"item_id", "timestamp", "target"}, name="actual")

    fig = go.Figure()

    actual_series = actual[actual["item_id"] == item_id].sort_values("timestamp").copy()
    actual_series["timestamp"] = pd.to_datetime(actual_series["timestamp"])
    x_actual = _dt_array(actual_series["timestamp"]) if not actual_series.empty else np.array([])

    if not actual_series.empty:
        fig.add_trace(
            go.Scatter(
                x=x_actual,
                y=actual_series["target"],
                mode="lines",
                name="Actual",
                line=dict(color=KT_GREY_STROKE, width=2, dash="dot"),
            )
        )

    for model_name, forecast in forecasts.items():
        _validate_columns(forecast, set(REQUIRED_FORECAST_COLUMNS), name=f"forecast[{model_name}]")
        series = forecast[forecast["item_id"] == item_id].sort_values("timestamp").copy()
        if series.empty:
            continue
        series["timestamp"] = pd.to_datetime(series["timestamp"])
        x_forecast = _dt_array(series["timestamp"])
        fig.add_trace(
            go.Scatter(
                x=x_forecast,
                y=series["p50"],
                mode="lines",
                name=model_name,
                line=dict(width=3),
            )
        )

    fig.update_layout(
        title=f"Confronto modelli · {item_id}",
        xaxis_title="Data",
        yaxis_title="Valore",
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )

    if sap_note:
        fig.add_annotation(
            text=sap_note,
            xref="paper",
            yref="paper",
            x=0,
            y=1.15,
            showarrow=False,
            font=dict(size=11, color=KT_TEXT_MUTED),
        )

    fig.update_xaxes(type="date")
    return fig


def build_step_run_chart(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    item_id: str,
    events: Sequence[EventBand] | None = None,
    sap_note: str | None = None,
) -> go.Figure:
    """Render step-style forecast focussed on payment run days only."""

    _validate_columns(actual, {"item_id", "timestamp", "target"}, name="actual")
    _validate_columns(forecast, set(REQUIRED_FORECAST_COLUMNS), name="forecast")

    actual_series = actual[actual["item_id"] == item_id].sort_values("timestamp").copy()
    forecast_series = forecast[forecast["item_id"] == item_id].sort_values("timestamp").copy()

    actual_series["timestamp"] = pd.to_datetime(actual_series["timestamp"])
    forecast_series["timestamp"] = pd.to_datetime(forecast_series["timestamp"])

    if actual_series.empty and forecast_series.empty:
        raise ValueError(f"No data available for item_id='{item_id}'")

    if "run_flag" in actual_series.columns:
        run_wd = (
            sorted(
                actual_series.loc[actual_series["run_flag"].astype(int) == 1, "timestamp"]
                .dt.weekday.unique()
                .tolist()
            )
            or [1, 4]
        )
    else:
        run_wd = [1, 4]

    mask_a = actual_series["timestamp"].dt.weekday.isin(run_wd)
    mask_f = forecast_series["timestamp"].dt.weekday.isin(run_wd)

    actual_runs = actual_series.loc[mask_a]
    forecast_runs = forecast_series.loc[mask_f]

    x_forecast_runs = _dt_array(forecast_runs["timestamp"]) if not forecast_runs.empty else np.array([])
    x_actual_runs = _dt_array(actual_runs["timestamp"]) if not actual_runs.empty else np.array([])

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "[CHK][F110] item=%s | forecast_run=%d | forecast_skip=%d",
            item_id,
            len(forecast_runs),
            len(forecast_series) - len(forecast_runs),
        )

    fig = go.Figure()

    if not forecast_runs.empty and {"p10", "p90"}.issubset(forecast_runs.columns):
        fig.add_scatter(
            x=x_forecast_runs,
            y=forecast_runs["p90"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            line_shape="hv",
        )
        fig.add_scatter(
            x=x_forecast_runs,
            y=forecast_runs["p10"],
            mode="lines",
            fill="tonexty",
            fillcolor=_hex_to_rgba(KT_ORANGE, 0.18),
            line=dict(width=0),
            name="Banda p10-p90",
            hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}€<extra></extra>",
            line_shape="hv",
        )

    if not forecast_runs.empty:
        fig.add_scatter(
            x=x_forecast_runs,
            y=forecast_runs["p50"],
            mode="lines+markers",
            name="Forecast p50",
            line=dict(color=KT_ORANGE, width=3),
            marker=dict(size=8, color=KT_ORANGE, symbol="square"),
            hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}€<extra></extra>",
            line_shape="hv",
        )

    if not actual_runs.empty:
        fig.add_scatter(
            x=x_actual_runs,
            y=actual_runs["target"],
            mode="markers",
            name="Pagamenti",
            marker=dict(size=10, color=KT_SUCCESS, symbol="circle", line=dict(width=1, color="white")),
            hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}€<extra></extra>",
        )

    if events:
        for event in events:
            fig.add_vrect(
                x0=event.start,
                x1=event.end,
                fillcolor=_hex_to_rgba(KT_ORANGE, 0.10),
                line=dict(color=_hex_to_rgba(KT_ORANGE, 0.4), width=1),
                layer="below",
                annotation=dict(
                    text=event.label,
                    textangle=270,
                    font=dict(color=KT_ORANGE, size=11),
                    x=event.start,
                    xshift=-6,
                    y=1,
                    yref="paper",
                    showarrow=False,
                ),
            )

    fig.update_layout(
        title=f"F110 payout runs · {item_id}",
        xaxis_title="Data di run",
        yaxis_title="€ pagati",
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )

    if sap_note:
        fig.add_annotation(
            text=sap_note,
            xref="paper",
            yref="paper",
            x=0,
            y=1.15,
            showarrow=False,
            font=dict(size=11, color=KT_TEXT_MUTED),
        )

    fig.update_xaxes(type="date")

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


def compute_count_metrics(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    seasonal_period: int = 7,
) -> pd.DataFrame:
    """Compute MAE and optional MASE for daily count series."""

    _validate_columns(actual, {"item_id", "timestamp", "target"}, name="actual")
    _validate_columns(forecast, set(REQUIRED_FORECAST_COLUMNS), name="forecast")

    merged = pd.merge(actual, forecast, on=["item_id", "timestamp"], how="inner")

    rows: list[dict[str, Any]] = []
    for item_id, group in merged.groupby("item_id"):
        y_true = group["target"].to_numpy(dtype=float)
        y_pred = group["p50"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))

        mase_denominator = None
        if len(y_true) > seasonal_period:
            seasonal_naive = y_true[:-seasonal_period]
            shifted = y_true[seasonal_period:]
            mase_denominator = float(np.mean(np.abs(shifted - seasonal_naive)))

        mase = float(mae / mase_denominator) if mase_denominator and mase_denominator > 1e-8 else float("nan")
        rows.append({"item_id": item_id, "MAE": mae, "MASE": mase})

    return pd.DataFrame(rows)


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


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"
