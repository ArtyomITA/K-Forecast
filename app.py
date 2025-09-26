"""Minimal Streamlit UI that unifies Chronos-Bolt, DeepAR and AutoML (AG-TS).

The app exposes the three forecasting wrappers behind a compact interface:

* Sidebar → model selection, forecast horizon and debug logging toggle.
* Main view → dataset source choice, upload handling, execution trigger and
  visual/metric output.

Edge cases such as irregular timestamps and missing quantiles are handled with
forward-filling resampling and informative degradations so that the app can
operate against a variety of inputs without surprising the user.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import streamlit as st

from models.ag_automl_model import fit_predict_agts
from models.chronos_model import predict_chronos
from models.deepar_model import train_and_predict_deepar
from utils import viz

# ---------------------------------------------------------------------------
# Application-wide constants
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES = ("Chronos-Bolt", "DeepAR", "AutoML (AG-TS)")
MODEL_KEY_MAP = {
    "Chronos-Bolt": "chronos",
    "DeepAR": "deepar",
    "AutoML (AG-TS)": "automl",
}
HORIZON_OPTIONS = (7, 30, 90)
DEFAULT_HORIZON = 30
DEFAULT_FREQ = "D"
MOCK_DATA_PATH = Path("data/mock_bank_flows.csv")

# Default hyper-parameters passed to model wrappers.
DEEPAR_DEFAULTS: dict[str, Any] = {
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "context_length_mult": 4,
    "num_samples": 100,
    "num_workers": 0,
    "device": "cpu",
    "early_stop_patience": 5,
}
AUTOML_DEFAULTS: dict[str, Any] = {
    "time_limit": 120,
    "presets": "medium_quality",
    "use_covariates": True,
    "split_date": None,
}

LOGGER = logging.getLogger("k_forecast.app")


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def configure_logging(debug_mode: bool) -> None:
    """Initialise logging according to the sidebar toggle."""

    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(name)s: %(message)s")
    LOGGER.setLevel(level)
    LOGGER.info("Log level set to %s", logging.getLevelName(level))


# ---------------------------------------------------------------------------
# Streamlit UI assembly
# ---------------------------------------------------------------------------

def render_sidebar() -> dict[str, Any]:
    """Return sidebar selections (model, horizon, debug flag)."""

    st.sidebar.header("Configurazione")
    model_display = st.sidebar.selectbox("Modello", MODEL_DISPLAY_NAMES, index=0)
    horizon = st.sidebar.select_slider("Orizzonte (giorni)", options=HORIZON_OPTIONS, value=DEFAULT_HORIZON)
    debug_mode = st.sidebar.checkbox("DEBUG logs", value=False)

    return {
        "model_display": model_display,
        "model_key": MODEL_KEY_MAP[model_display],
        "horizon": int(horizon),
        "debug_mode": debug_mode,
    }


def render_main_area(options: dict[str, Any]) -> None:
    """Compose the main panel with dataset selection and outputs."""

    st.title("K-Forecast · Forecast demo")
    st.caption("Scegli l'origine dei dati, seleziona il modello e lancia il forecast.")

    data_choice = st.radio("Sorgente dati", ("Usa mock data", "Carica CSV"), horizontal=True)
    uploaded_file = None
    if data_choice == "Carica CSV":
        uploaded_file = st.file_uploader("Carica dataset long (item_id,timestamp,target)", type=["csv"])

    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    download_placeholder = st.empty()

    dataset_df, dataset_origin = _resolve_dataset(data_choice, uploaded_file)
    if dataset_df is None:
        if data_choice == "Carica CSV":
            st.info("Carica un CSV per abilitare il forecast.")
        else:
            st.warning("Dataset mock non trovato. Generarlo con scripts/generate_mock_bank_flows.py.")
        return

    dataset_df, resampled = _ensure_daily_frequency(dataset_df, DEFAULT_FREQ)
    if resampled:
        st.warning("Timestamp irregolari rilevati: effettuato resample giornaliero con forward-fill.")

    freq = _infer_frequency(dataset_df) or DEFAULT_FREQ

    date_min = dataset_df["timestamp"].min()
    date_max = dataset_df["timestamp"].max()
    st.caption(f"Origine: {dataset_origin} · Range: {date_min:%Y-%m-%d} → {date_max:%Y-%m-%d}")

    run_button = st.button("Esegui Forecast", type="primary", use_container_width=True)
    if not run_button:
        return

    LOGGER.debug("Dataset shape=%s", dataset_df.shape)
    st.toast("Training…")
    with st.spinner("Esecuzione modello in corso…"):
        handle_forecast(
            model_key=options["model_key"],
            horizon=int(options["horizon"]),
            dataset_df=dataset_df,
            freq=freq,
            debug_mode=bool(options["debug_mode"]),
            chart_placeholder=chart_placeholder,
            metrics_placeholder=metrics_placeholder,
            download_placeholder=download_placeholder,
        )
    st.toast("Previsioni pronte ✅")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Streamlit entrypoint."""

    options = render_sidebar()
    configure_logging(options["debug_mode"])
    LOGGER.info("Modello selezionato: %s | Horizon=%d", options["model_display"], options["horizon"])
    render_main_area(options)


# ---------------------------------------------------------------------------
# Forecast orchestration
# ---------------------------------------------------------------------------

def handle_forecast(
    *,
    model_key: Literal["chronos", "deepar", "automl"],
    horizon: int,
    dataset_df: pd.DataFrame,
    freq: str,
    debug_mode: bool,
    chart_placeholder,
    metrics_placeholder,
    download_placeholder,
) -> None:
    """Run the selected model and display outputs."""

    start_time = time.perf_counter()
    train_df, holdout_df = _train_holdout_split(dataset_df, horizon)
    LOGGER.debug("Train shape=%s | Holdout shape=%s", train_df.shape, holdout_df.shape)

    try:
        if model_key == "chronos":
            forecast_df = predict_chronos(train_df, horizon, freq, zero_shot=True, do_finetune=False, logger=LOGGER)
        elif model_key == "deepar":
            forecast_df, training_metrics = train_and_predict_deepar(dataset_df, horizon, freq, **DEEPAR_DEFAULTS)
            LOGGER.debug("DeepAR training metrics=%s", training_metrics)
        else:  # automl
            forecast_df, automl_metrics = fit_predict_agts(dataset_df, horizon, freq, **AUTOML_DEFAULTS)
            LOGGER.debug("AutoML leaderboard metrics=%s", automl_metrics)
    except Exception as exc:  # pragma: no cover - Streamlit UI path
        LOGGER.exception("Forecast %s failed: %s", model_key, exc)
        st.error(
            "Si è verificato un errore durante l'esecuzione del modello {}. "
            "Dettagli: {}".format(model_key, exc)
        )
        return

    duration = time.perf_counter() - start_time
    LOGGER.info("Forecast %s completato in %.2fs", model_key, duration)

    forecast_df = _sanitize_forecast_frame(forecast_df)
    forecast_df, quantile_msg = _ensure_quantiles(forecast_df)
    if quantile_msg is not None:
        st.info(quantile_msg)

    _render_outputs(
        chart_placeholder=chart_placeholder,
        metrics_placeholder=metrics_placeholder,
        download_placeholder=download_placeholder,
        forecast_df=forecast_df,
        train_df=train_df,
        holdout_df=holdout_df,
        model_label=model_key,
    )


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def _resolve_dataset(choice: str, upload) -> tuple[pd.DataFrame | None, str]:
    if choice == "Usa mock data":
        return _load_mock_dataset()
    if upload is None:
        return None, ""
    LOGGER.info("Carico dataset da upload (%s bytes)", upload.size)
    return _standardise_dataframe(pd.read_csv(upload)), "upload"


@st.cache_data(show_spinner=False)
def _load_mock_dataset() -> tuple[pd.DataFrame | None, str]:
    if not MOCK_DATA_PATH.exists():
        LOGGER.warning("Mock dataset non presente in %s", MOCK_DATA_PATH)
        return None, ""
    df = pd.read_csv(MOCK_DATA_PATH)
    return _standardise_dataframe(df), MOCK_DATA_PATH.as_posix()


def _standardise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"item_id", "timestamp", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Il dataset deve contenere le colonne {sorted(required)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df.sort_values(["item_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _ensure_daily_frequency(df: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, bool]:
    resampled_frames: list[pd.DataFrame] = []
    touched = False

    for item_id, group in df.groupby("item_id", sort=True):
        group = group.groupby("timestamp", as_index=False).mean(numeric_only=True)
        group_sorted = group.sort_values("timestamp")
        full_range = pd.date_range(group_sorted["timestamp"].min(), group_sorted["timestamp"].max(), freq=freq)
        current_index = pd.DatetimeIndex(group_sorted["timestamp"])
        if len(full_range) != len(group_sorted) or not current_index.equals(full_range):
            touched = True
            series = (
                group_sorted.set_index("timestamp")["target"].resample(freq).ffill().bfill().reset_index()
            )
        else:
            series = group_sorted[["timestamp", "target"]].copy()
        series["item_id"] = item_id
        resampled_frames.append(series)

    result = pd.concat(resampled_frames, ignore_index=True)
    result.sort_values(["item_id", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result, touched


def _train_holdout_split(df: pd.DataFrame, prediction_length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("item_id", sort=True):
        group_sorted = group.sort_values("timestamp")
        if len(group_sorted) > prediction_length:
            holdout_parts.append(group_sorted.tail(prediction_length))
            train_parts.append(group_sorted.iloc[:-prediction_length])
        else:
            train_parts.append(group_sorted)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    holdout_df = pd.concat(holdout_parts, ignore_index=True) if holdout_parts else pd.DataFrame(columns=df.columns)
    return train_df, holdout_df


def _infer_frequency(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    ref_series = df[df["item_id"] == df["item_id"].iloc[0]].sort_values("timestamp")["timestamp"]
    return pd.infer_freq(ref_series)


def _sanitize_forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=False)
    result.sort_values(["item_id", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _ensure_quantiles(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    result = df.copy()
    if "p50" not in result.columns:
        raise ValueError("Il forecast deve contenere almeno la colonna p50.")

    missing = [q for q in ("p10", "p90") if q not in result.columns]
    if not missing:
        return result, None

    for q in missing:
        result[q] = result["p50"]
    message = "Quantili mancanti ({}) sostituiti con p50.".format(
        ", ".join(sorted(missing))
    )
    LOGGER.warning(message)
    return result, message


def _render_outputs(
    *,
    chart_placeholder,
    metrics_placeholder,
    download_placeholder,
    forecast_df: pd.DataFrame,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    model_label: str,
) -> None:
    all_actual = pd.concat([train_df, holdout_df], ignore_index=True) if not train_df.empty else holdout_df

    available_items = sorted(forecast_df["item_id"].dropna().unique())
    if not available_items:
        st.warning("Nessuna serie disponibile nelle previsioni.")
        return

    selected_item = cast(
        str,
        st.selectbox("Serie da visualizzare", options=available_items, key=f"series_{model_label}"),
    )
    chart = viz.build_forecast_chart(all_actual, forecast_df, item_id=selected_item)
    chart_placeholder.plotly_chart(chart, use_container_width=True)

    if holdout_df.empty:
        metrics_placeholder.info("Nessun holdout disponibile per il calcolo delle metriche.")
    else:
        metrics = viz.compute_metrics(holdout_df, forecast_df)
        metrics_placeholder.dataframe(metrics, use_container_width=True)

    csv_buffer = StringIO()
    forecast_df.to_csv(csv_buffer, index=False)
    download_placeholder.download_button(
        label="Scarica CSV previsioni",
        data=csv_buffer.getvalue(),
        file_name=f"forecast_{model_label}.csv",
        mime="text/csv",
    )


def _maybe_launch_streamlit() -> None:
    """Bootstraps Streamlit when the script is executed via `python app.py`."""

    if os.environ.get("K_FORECAST_STREAMLIT_ACTIVE") == "1":
        main()
        return

    os.environ["K_FORECAST_STREAMLIT_ACTIVE"] = "1"
    from streamlit.web import cli as stcli  # Imported lazily to avoid overhead during standard run

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())


if __name__ == "__main__":
    _maybe_launch_streamlit()
