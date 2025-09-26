"""Brand theming utilities for the K-Forecast UI."""

from __future__ import annotations

from typing import Literal

import plotly.graph_objects as go

# Keytech color system (replace placeholders with official brand HEX if available)
KT_ORANGE = "#FF7A00"
KT_DARK_BG = "#0F1218"
KT_CARD_BG = "#1B1F2A"
KT_TEXT = "#E6E8EC"
KT_TEXT_MUTED = "#AEB4C0"
KT_GREY_STROKE = "#2A3040"
KT_SUCCESS = "#22C55E"
KT_WARNING = "#F59E0B"
KT_DANGER = "#EF4444"


PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor=KT_CARD_BG,
        plot_bgcolor=KT_CARD_BG,
        font=dict(color=KT_TEXT, family="Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"),
        margin=dict(t=40, r=24, b=32, l=32),
        xaxis=dict(
            gridcolor=KT_GREY_STROKE,
            linecolor=KT_GREY_STROKE,
            zerolinecolor=KT_GREY_STROKE,
            tickfont=dict(color=KT_TEXT_MUTED),
            titlefont=dict(color=KT_TEXT),
        ),
        yaxis=dict(
            gridcolor=KT_GREY_STROKE,
            linecolor=KT_GREY_STROKE,
            zerolinecolor=KT_GREY_STROKE,
            tickfont=dict(color=KT_TEXT_MUTED),
            titlefont=dict(color=KT_TEXT),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=KT_GREY_STROKE,
            borderwidth=0,
            font=dict(color=KT_TEXT),
        ),
        colorway=[KT_ORANGE, KT_SUCCESS, KT_WARNING, KT_TEXT],
    )
)

# Semi-transparent fill for predictive intervals
PLOTLY_TEMPLATE.data.scatter = [
    go.Scatter(marker=dict(color=KT_ORANGE), line=dict(color=KT_ORANGE)),
]


Tone = Literal["accent", "success", "warning", "danger"]


def brand_badge(text: str, tone: Tone = "accent") -> str:
    """Return HTML markup for a pill badge aligned with the Keytech brand."""

    tone_map = {
        "accent": KT_ORANGE,
        "success": KT_SUCCESS,
        "warning": KT_WARNING,
        "danger": KT_DANGER,
    }
    color = tone_map.get(tone, KT_ORANGE)
    return (
        f"<span class='kt-badge kt-badge--{tone}' style=\""
        f"background: {color}20; border: 1px solid {color}80; color: {color};\">{text}</span>"
    )


__all__ = [
    "PLOTLY_TEMPLATE",
    "brand_badge",
    "KT_ORANGE",
    "KT_DARK_BG",
    "KT_CARD_BG",
    "KT_TEXT",
    "KT_TEXT_MUTED",
    "KT_GREY_STROKE",
    "KT_SUCCESS",
    "KT_WARNING",
    "KT_DANGER",
]
