"""Streamlit UI showcasing DeepAR (SD), DeepAR (F110) and ARIMA/SARIMA (EBS).

The app exposes the three forecasting wrappers behind a compact interface:

* Sidebar → model selection, forecast horizon and debug logging toggle.
* Main view → dataset source choice, upload handling, execution trigger and
    visual/metric output.

Edge cases such as irregular timestamps and missing quantiles are handled with
forward-filling resampling and informative degradations so that the app can
operate against a variety of inputs without surprising the user.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import numpy as np
import pandas as pd
import streamlit as st

from models.arima_model import fit_predict_arima_ebs
from models.deepar_model import train_and_predict_deepar, train_and_predict_deepar_f110
from utils import viz
from utils.viz import EventBand
from utils.theme import (
    KT_CARD_BG,
    KT_DANGER,
    KT_DARK_BG,
    KT_GREY_STROKE,
    KT_ORANGE,
    KT_SUCCESS,
    KT_TEXT,
    KT_TEXT_MUTED,
    KT_WARNING,
    brand_badge,
)

st.set_page_config(page_title="K-Forecast", layout="wide")


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CSS_FILE = ASSETS_DIR / "app.css"


def inject_brand_css() -> None:
    """Inject Keytech-inspired dark theme styles."""

    font_stack = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
    try:
        css_template = CSS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.warning("CSS file '%s' not found; skipping CSS injection", CSS_FILE)
        return

    replacements = {
        "__KT_ORANGE__": KT_ORANGE,
        "__KT_DARK_BG__": KT_DARK_BG,
        "__KT_CARD_BG__": KT_CARD_BG,
        "__KT_TEXT__": KT_TEXT,
        "__KT_TEXT_MUTED__": KT_TEXT_MUTED,
        "__KT_GREY_STROKE__": KT_GREY_STROKE,
        "__KT_SUCCESS__": KT_SUCCESS,
        "__KT_WARNING__": KT_WARNING,
        "__KT_DANGER__": KT_DANGER,
        "__FONT_STACK__": font_stack,
    }

    css_filled = css_template
    for placeholder, value in replacements.items():
        css_filled = css_filled.replace(placeholder, value)

    st.markdown(f"<style>{css_filled}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Application-wide constants
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES = (
    "DeepAR (F110)",
    "DeepAR (SD)",
    "ARIMA/SARIMA (EBS)",
)
MODEL_KEY_MAP = {
    "DeepAR (F110)": "deepar_f110",
    "DeepAR (SD)": "deepar_sd",
    "ARIMA/SARIMA (EBS)": "arima_ebs",
}
MODEL_TAB_ORDER = ["deepar_f110", "deepar_sd", "arima_ebs"]
TAB_TITLES = list(MODEL_DISPLAY_NAMES)
TAB_INDEX = {key: idx for idx, key in enumerate(MODEL_TAB_ORDER)}
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "deepar_sd": {
        "label": "Mock SD (vendite)",
        "path": "data/mock_sd_sales.csv",
        "expected": [
            "item_id",
            "timestamp",
            "target",
            "sku",
            "customer",
            "area",
            "promo_flag",
            "price_idx",
        ],
    },
    "deepar_f110": {
        "label": "Mock F110 (cash-out)",
        "path": "data/mock_f110_payouts.csv",
        "expected": [
            "item_id",
            "timestamp",
            "target",
            "company_code",
            "bank_acct",
            "run_flag",
            "eom_flag",
            "volume_due_next_7d",
        ],
    },
    "arima_ebs": {
        "label": "Mock EBS (conteggi)",
        "path": "data/mock_ebs_counts.csv",
        "expected": [
            "item_id",
            "timestamp",
            "target",
        ],
    },
}
USECASE_COPY: dict[str, dict[str, Any]] = {
    "deepar_sd": {
        "title": "Use case: Domanda commerciale (SD)",
        "bullets": [
            "Previsioni vendite SKU×Cliente×Area",
            "Covariate: promo, prezzo, calendario",
            "Output per serie con intervalli p10–p90",
        ],
    },
    "deepar_f110": {
        "title": "Use case: Payment run F110 (AP)",
        "bullets": [
            "Prevediamo il cash-out per ogni run (mar/ven)",
            "Covariate: run_flag, volume_due_next_7d, calendario",
            "Output su giorni-run con intervalli p10–p90",
        ],
        "sap_tables": "Tabelle SAP: BSIK/BSAK, BSEG (ZFBDT, ZTERM, NETDT/SK1DT/SK2DT), BKPF, T001",
    },
    "arima_ebs": {
        "title": "Use case: EBS workload (Bank Ops)",
        "bullets": [
            "Conteggio righe FEBEP/FEBKO per banca",
            "Stagionalità settimanale e picchi fine mese",
            "Grafico a barre con banda p10–p90",
        ],
        "sap_tables": "Tabelle SAP: FEBKO, FEBEP",
    },
}
USECASE_HERO: dict[str, dict[str, str]] = {
    "deepar_sd": {
        "scope": "Prevediamo le vendite per SKU×Cliente×Area con incertezza p10–p90.",
        "sap": "VBRK/VBRP, VBAK/VBAP, KONV, KNA1/KNVV, MARA/MARC",
        "fields": "item_id (SKU|Cliente|Area), timestamp, target, promo_flag, price_idx",
    },
    "deepar_f110": {
        "scope": "Prevediamo il cash-out di ciascun run F110 (mar/ven).",
        "sap": "BSIK/BSAK, BSEG (ZFBDT,ZTERM,NETDT), BKPF, T001",
        "fields": "item_id (CC|Banca), timestamp, target, run_flag, volume_due_next_7d",
    },
    "arima_ebs": {
        "scope": "Prevediamo quante righe FEBEP arrivano al giorno (workload EBS).",
        "sap": "FEBKO (testata), FEBEP (righe)",
        "fields": "item_id (Banca/Conto), timestamp, target",
    },
}
COLUMN_ALIASES: dict[str, dict[str, str]] = {
    "deepar_sd": {
        "item_id": "Serie (SKU|Cliente|Area)",
        "timestamp": "Data",
        "target": "Vendite (€)",
        "promo_flag": "Promozione (0/1)",
        "price_idx": "Indice prezzo (rel.)",
        "dow_sin": "Settimana (sin)",
        "dow_cos": "Settimana (cos)",
        "eom_flag": "Fine mese (0/1)",
    },
    "deepar_f110": {
        "item_id": "Serie (CC|Banca)",
        "timestamp": "Data",
        "target": "Pagato nel run (€)",
        "run_flag": "Giorno di run F110 (0/1)",
        "volume_due_next_7d": "Fatture con NETDT ≤7g (€)",
        "eom_flag": "Fine mese (0/1)",
        "dow_sin": "Settimana (sin)",
        "dow_cos": "Settimana (cos)",
        "company_code": "Company code",
        "bank_acct": "Conto bancario",
    },
    "arima_ebs": {
        "item_id": "Serie (Banca/Conto)",
        "timestamp": "Data",
        "target": "Righe EBS/giorno (FEBEP)",
    },
}
COLUMN_TIPS: dict[str, str] = {
    "run_flag": "1 sui giorni di run programmati (es. martedì/venerdì).",
    "volume_due_next_7d": "Somma importi con scadenza (NETDT) entro 7 giorni, da BSEG/ZTERM.",
    "promo_flag": "Periodo promozionale attivo (1) / non attivo (0).",
    "price_idx": "Indice relativo del prezzo (0.8–1.2).",
    "eom_flag": "Fine mese (1) nei giorni di chiusura.",
    "dow_sin": "Feature calendario (seno del giorno-settimana).",
    "dow_cos": "Feature calendario (coseno del giorno-settimana).",
}
HORIZON_OPTIONS = (7, 30, 90)
DEFAULT_HORIZON = 30
DEFAULT_FREQ = "D"
MOCK_DATA_PATH = Path("data/mock_bank_flows.csv")
MOCK_SD_PATH = Path("data/mock_sd_sales.csv")
PRICE_DRIFT_THRESHOLD = 0.03
SESSION_FORECAST_CACHE_KEY = "kt_forecast_cache"

# Default hyper-parameters passed to model wrappers.
DEEPAR_DEFAULTS: dict[str, Any] = {
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "context_length_mult": 4,
    "num_samples": 200,
    "num_workers": 0,
    "device": "cpu",
    "early_stop_patience": 5,
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

    model_key = MODEL_KEY_MAP[model_display]
    mock_options = list(DATASET_REGISTRY.items())

    source_choice = st.sidebar.radio(
        "Sorgente dati",
        (
            "Mock specifico",
            "Mock alternativo",
            "Carica CSV",
        ),
        index=0,
        key="dataset_source_radio",
    )

    alt_mock_label = None
    alt_mock_key = None
    if source_choice == "Mock alternativo":
        labels = [meta["label"] for _, meta in mock_options]
        default_index = next((idx for idx, (key, _) in enumerate(mock_options) if key != model_key), 0)
        alt_mock_label = st.sidebar.selectbox(
            "Seleziona mock",
            options=labels,
            index=default_index,
        )
        for key, meta in mock_options:
            if meta["label"] == alt_mock_label:
                alt_mock_key = key
                break
        if alt_mock_key is None and mock_options:
            alt_mock_key = mock_options[default_index][0]
    else:
        alt_mock_key = model_key if source_choice == "Mock specifico" else None

    dataset_source_key = {
        "Mock specifico": "mock_specific",
        "Mock alternativo": "mock_alt",
        "Carica CSV": "upload",
    }[source_choice]
    st.session_state["dataset_source"] = dataset_source_key
    st.session_state["dataset_choice"] = alt_mock_key

    LOGGER.info(
        "[UI] dataset_source=%s | model_key=%s | mock_choice=%s",
        dataset_source_key,
        model_key,
        alt_mock_key,
    )

    debug_mode = st.sidebar.checkbox("DEBUG logs", value=False)

    return {
        "model_display": model_display,
        "model_key": model_key,
        "horizon": int(horizon),
        "debug_mode": debug_mode,
        "dataset_source": dataset_source_key,
        "dataset_choice": alt_mock_key,
    }


def _render_model_overview_tab(model_key: str) -> None:
    if model_key == "deepar_f110":
        st.markdown("**Focus:** cash-out su run F110 programmati (martedì/venerdì).")
        st.markdown("**Covariate:** run_flag, volume_due_next_7d, calendario (sin/cos, eom).")
        st.markdown(
            "**Dati richiesti:** item_id, timestamp, target, company_code, bank_acct, run_flag, eom_flag, volume_due_next_7d."
        )
        st.markdown("**Output:** intervalli p10/p50/p90 sui soli giorni di run.")
    elif model_key == "deepar_sd":
        st.markdown("**Focus:** domanda commerciale SKU×Cliente in area SD.")
        st.markdown("**Covariate:** promo_flag, price_idx più feature calendario.")
        st.markdown(
            "**Dati richiesti:** storico giornaliero con item_id, timestamp, target, sku, customer, area, promo_flag, price_idx."
        )
        st.markdown("**Output:** previsioni p10/p50/p90 per orizzonte selezionato.")
    elif model_key == "arima_ebs":
        st.markdown("**Focus:** carico FEBEP/FEBKO per banca (EBS).")
        st.markdown("**Covariate:** modello statistico puro con fallback stagionale.")
        st.markdown("**Dati richiesti:** item_id, timestamp, target (conteggi giornalieri).")
        st.markdown("**Output:** barre storiche + fascia p10–p90 per pianificazione operativa.")
    else:
        st.caption("Seleziona un modello dalla sidebar per visualizzare il dettaglio use case.")


def display_df(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    aliases = COLUMN_ALIASES.get(model_key, {})
    rename_map = {col: aliases.get(col, col) for col in df.columns}
    return df.rename(columns=rename_map)


def build_column_dictionary(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    aliases = COLUMN_ALIASES.get(model_key, {})
    rows: list[dict[str, str]] = []
    for column in df.columns:
        gui_label = aliases.get(column, column)
        tooltip = COLUMN_TIPS.get(column, "—")
        rows.append(
            {
                "GUI Label": gui_label,
                "Colonna tecnica": column,
                "Tooltip/Descrizione": tooltip,
            }
        )
    return pd.DataFrame(rows)


def render_main_area(options: dict[str, Any]) -> None:
    """Compose the main panel with dataset selection and outputs."""

    with st.expander("Cos'è K-Forecast?"):
        st.markdown(
            "\n".join(
                [
                    "- Prevede i flussi di cassa a 7/30/90g con incertezza (p10–p90).",
                    "- Funziona su più serie (conti/cliente) e confronta 3 modelli.",
                    "- Esporta CSV e mostra metriche di qualità.",
                ]
            )
        )
        st.markdown(brand_badge(f"Modello: {options['model_display']}"), unsafe_allow_html=True)

    header_left, header_right = st.columns([3, 2], gap="large")

    with header_left:
        st.title("K-Forecast · Forecast demo")
        st.caption("Scegli l'origine dei dati, seleziona il modello e lancia il forecast.")

    with header_right:
        status_badges = " ".join(
            [
                brand_badge(f"Modello: {options['model_display']}", tone="accent"),
                brand_badge(f"Orizzonte: {options['horizon']}g", tone="warning"),
            ]
        )
        st.markdown(
            (
                "<div class='kt-header-badges'>"
                f"{brand_badge('Keytech AI', tone='accent')}"
                f"<div class='kt-header-badges__row'>{status_badges}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class='kt-callout'>
            <h4>Come leggere il grafico</h4>
            <ul>
                <li><strong>Forecast p50</strong> = curva attesa</li>
                <li><strong>Intervallo p10–p90</strong> = fascia d'incertezza</li>
                <li><strong>Actual</strong> = storico reale</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(TAB_TITLES)
    active_index = int(st.session_state.get("active_tab_index", TAB_INDEX.get(options["model_key"], 0)))
    active_index = max(0, min(active_index, len(tabs) - 1))

    for idx, tab in enumerate(tabs):
        with tab:
            if idx == active_index:
                _render_model_overview_tab(options["model_key"])
            else:
                st.caption("Seleziona il modello dalla sidebar per visualizzare questo pannello.")

    with st.expander("Assunzioni & limiti"):
        st.markdown("**Assunzioni:** {{ASSUNZIONI_TXT}}")
        st.markdown("**Limiti:** {{LIMITI_TXT}}")

    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    download_placeholder = st.empty()

    dataset_df = _load_dataset_for_model(options["model_key"])

    freq = _infer_frequency(dataset_df) or DEFAULT_FREQ

    date_min = dataset_df["timestamp"].min()
    date_max = dataset_df["timestamp"].max()
    origin_label = st.session_state.get("dataset_origin_label", "Sorgente sconosciuta")
    st.caption(f"Origine: {origin_label} · Range: {date_min:%Y-%m-%d} → {date_max:%Y-%m-%d}")

    hero_copy = USECASE_HERO.get(options["model_key"])
    if hero_copy:
        with st.container(border=True):
            st.markdown(f"**Scope**: {hero_copy['scope']}")
            st.markdown(f"**Tabelle SAP**: {hero_copy['sap']}")
            st.markdown(f"**Campi principali**: {hero_copy['fields']}")

    registry_entry = DATASET_REGISTRY.get(options["model_key"], {})
    expected_cols = registry_entry.get("expected")
    if expected_cols:
        aliases = COLUMN_ALIASES.get(options["model_key"], {})
        expected_alias = [str(aliases.get(col, col)) for col in expected_cols]
        found_alias = [str(aliases.get(col, col)) for col in dataset_df.columns]
        missing_alias = [alias for alias in expected_alias if alias not in found_alias]

        missing_display = (
            ", ".join(f"**{alias}**" for alias in missing_alias)
            if missing_alias
            else "nessuna"
        )
        st.info(
            "**Colonne attese:** {expected} | **Trovate:** {found} | **Mancanti:** {missing}".format(
                expected=", ".join(expected_alias),
                found=", ".join(sorted(set(found_alias))),
                missing=missing_display,
            )
        )

    usecase = USECASE_COPY.get(options["model_key"], {})
    title = usecase.get("title", "Use case")
    bullets = usecase.get("bullets", [])
    sap_tables = usecase.get("sap_tables")
    with st.expander(title, expanded=False):
        for bullet in bullets:
            st.markdown(f"• {bullet}")
        if sap_tables:
            st.caption(sap_tables)
        st.caption("Schema rilevato: " + ", ".join(sorted(map(str, dataset_df.columns.tolist()))))
        preview_df = display_df(dataset_df.head(5), options["model_key"])
        st.dataframe(preview_df, use_container_width=True)
        dict_df = build_column_dictionary(dataset_df, options["model_key"])
        st.caption("Dizionario campi (GUI → colonna tecnica)")
        st.table(dict_df)

    event_columns = {"promo_flag", "due_flag", "event_flag"}
    st.session_state["has_events"] = bool(event_columns & set(dataset_df.columns))

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

    inject_brand_css()
    options = render_sidebar()
    st.session_state["selected_model_key"] = options["model_key"]
    st.session_state["active_tab_index"] = TAB_INDEX.get(options["model_key"], 0)
    configure_logging(options["debug_mode"])
    LOGGER.info("Modello selezionato: %s | Horizon=%d", options["model_display"], options["horizon"])
    render_main_area(options)


# ---------------------------------------------------------------------------
# Forecast orchestration
# ---------------------------------------------------------------------------

def handle_forecast(
    *,
    model_key: Literal["deepar_sd", "deepar_f110", "arima_ebs"],
    horizon: int,
    dataset_df: pd.DataFrame,
    freq: str,
    debug_mode: bool,
    chart_placeholder,
    metrics_placeholder,
    download_placeholder,
) -> None:
    """Run the selected model and display outputs."""

    train_df, holdout_df = _train_holdout_split(dataset_df, horizon)
    LOGGER.debug("Train shape=%s | Holdout shape=%s", train_df.shape, holdout_df.shape)

    run_signature = _compute_run_signature(dataset_df, horizon, freq)
    cache: dict[str, dict[str, Any]] = cast(
        dict[str, dict[str, Any]],
        st.session_state.get(SESSION_FORECAST_CACHE_KEY, {}),
    )

    cache = {
        key: value for key, value in cache.items() if value.get("signature") == run_signature
    }
    st.session_state[SESSION_FORECAST_CACHE_KEY] = cache

    start_time = time.perf_counter()
    try:
        if model_key == "deepar_sd":
            forecast_raw, _training_metrics = train_and_predict_deepar(
                dataset_df,
                horizon,
                freq,
                use_covariates=True,
                static_cols=("sku", "customer", "area"),
                dynamic_cols=("promo_flag", "price_idx"),
                **DEEPAR_DEFAULTS,
            )
        elif model_key == "deepar_f110":
            forecast_raw, _training_metrics = train_and_predict_deepar_f110(
                dataset_df,
                horizon,
                freq,
                **DEEPAR_DEFAULTS,
            )
        elif model_key == "arima_ebs":
            forecast_raw, _arima_metrics = fit_predict_arima_ebs(
                dataset_df,
                horizon,
                freq,
            )
        else:
            raise ValueError(f"Modello non supportato: {model_key}")
    except Exception as exc:  # pragma: no cover - Streamlit UI path
        LOGGER.exception("Forecast %s failed: %s", model_key, exc)
        st.error(
            "Si è verificato un errore durante l'esecuzione del modello {}. "
            "Dettagli: {}".format(model_key, exc)
        )
        return

    duration = time.perf_counter() - start_time
    forecast_df_sanitized = _sanitize_forecast_frame(forecast_raw)
    forecast_df_sanitized, quantile_msg = _ensure_quantiles(forecast_df_sanitized)

    LOGGER.info("[INFO][MODEL] %s: durata_inferenza=%.2fs, quantili=[p10,p50,p90]", model_key, duration)

    cache[model_key] = {
        "forecast": forecast_df_sanitized.copy(deep=True),
        "duration": duration,
        "quantile_message": quantile_msg,
        "signature": run_signature,
        "updated_at": time.time(),
    }
    st.session_state[SESSION_FORECAST_CACHE_KEY] = cache

    primary = cache.get(model_key)
    if not primary:
        st.error("Risultato del modello selezionato non disponibile.")
        return

    comparison_payload: dict[str, dict[str, Any]] = {
        key: {
            "forecast": entry.get("forecast", pd.DataFrame()),
            "duration": entry.get("duration"),
        }
        for key, entry in cache.items()
    }
    quantile_messages: dict[str, str] = {
        key: entry["quantile_message"]
        for key, entry in cache.items()
        if entry.get("quantile_message")
    }

    if message := quantile_messages.get(model_key):
        st.info(message)

    _render_outputs(
        chart_placeholder=chart_placeholder,
        metrics_placeholder=metrics_placeholder,
        download_placeholder=download_placeholder,
        forecast_df=primary["forecast"],
        train_df=train_df,
        holdout_df=holdout_df,
        model_label=model_key,
        comparison_payload=comparison_payload,
        quantile_messages=quantile_messages,
        horizon=horizon,
        source_df=dataset_df,
    )


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def _load_dataset_for_model(model_key: str) -> pd.DataFrame:
    dataset_source = st.session_state.get("dataset_source", "mock_specific")
    registry = DATASET_REGISTRY

    if dataset_source == "upload":
        uploaded = st.file_uploader(
            "Carica CSV (long: item_id,timestamp,target + extra opz.)",
            type=["csv"],
            key="dataset_upload",
        )
        if uploaded is None:
            st.stop()
        df_upload = pd.read_csv(uploaded)
        st.session_state["dataset_origin_label"] = "Upload CSV"
        return _validate_and_fix(df_upload, model_key)

    target_key = model_key
    if dataset_source == "mock_alt":
        alt_key = st.session_state.get("dataset_choice", model_key)
        if alt_key not in registry:
            LOGGER.warning("Chiave mock alternativo non presente, uso %s", model_key)
            alt_key = model_key
        target_key = alt_key

    entry = registry.get(target_key)
    if not entry:
        LOGGER.warning("Dataset non configurato per la chiave %s", target_key)
        st.warning("Dataset non configurato nella registry. Aggiorna DATASET_REGISTRY.")
        st.stop()

    dataset_path = Path(entry["path"])
    if not dataset_path.exists():
        st.warning(f"Mock non trovato: {dataset_path}. Genera i dati con lo script dedicato.")
        st.stop()

    LOGGER.info("Carico dataset mock '%s' da %s", entry["label"], dataset_path)
    df_mock = pd.read_csv(dataset_path)
    cleaned = _validate_and_fix(df_mock, target_key, expected_cols=entry.get("expected"))
    st.session_state["dataset_origin_label"] = entry["label"]

    if target_key == "deepar_sd":
        _log_sd_mock_dataset(cleaned)

    return cleaned


def _log_sd_mock_dataset(df: pd.DataFrame) -> None:
    series_count = int(df["item_id"].nunique()) if "item_id" in df.columns else 0
    promo_series = df.get("promo_flag")
    promo_ratio = 0.0
    if promo_series is not None and len(promo_series) > 0:
        promo_numeric = pd.to_numeric(promo_series, errors="coerce").fillna(0)
        promo_ratio = float((promo_numeric > 0).mean() * 100)
    areas: list[str] = []
    if "area" in df.columns:
        areas = sorted({str(value) for value in df["area"].dropna().unique()})
    areas_label = ", ".join(areas) if areas else "n/d"
    LOGGER.info(
        "[INFO][SD] mock dataset detected | series=%d | promo_days=%.1f%% | areas={%s}",
        series_count,
        promo_ratio,
        areas_label,
    )


def _validate_and_fix(
    df: pd.DataFrame,
    model_key: str,
    *,
    expected_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    required_base = {"item_id", "timestamp", "target"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        st.error(
            "Dataset privo delle colonne obbligatorie: {}".format(
                ", ".join(sorted(missing_base))
            )
        )
        st.stop()

    df_work = df.copy()

    expected_missing: list[str] = []
    if expected_cols:
        expected_set = set(expected_cols)
        present_set = set(df_work.columns)
        expected_missing = sorted(expected_set - present_set)
        extra_cols = sorted(present_set - expected_set)
        summary = (
            f"Colonne attese: {', '.join(expected_cols)} | "
            f"Trovate: {', '.join(sorted(df_work.columns))} | "
            f"Mancanti: {', '.join(expected_missing) if expected_missing else 'nessuna'}"
        )
        st.info(summary)
        if extra_cols:
            st.caption(f"Colonne extra rilevate: {', '.join(extra_cols)}")

    df_work["item_id"] = df_work["item_id"].astype(str)
    df_work["timestamp"] = pd.to_datetime(df_work["timestamp"], utc=False, errors="coerce")
    invalid_ts = df_work["timestamp"].isna()
    if invalid_ts.any():
        st.warning(f"Righe con timestamp non valido rimosse: {int(invalid_ts.sum())}")
        df_work = df_work[~invalid_ts]

    df_work.sort_values(["item_id", "timestamp"], inplace=True)
    df_work.reset_index(drop=True, inplace=True)

    df_resampled, touched = _ensure_daily_frequency(df_work, DEFAULT_FREQ)
    if touched:
        st.caption("Timestamp irregolari rilevati: effettuato resample giornaliero (ffill).")

    if model_key in {"deepar_sd", "deepar_f110"}:
        dow = df_resampled["timestamp"].dt.weekday.astype(float)
        if "dow_sin" not in df_resampled.columns:
            df_resampled["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        if "dow_cos" not in df_resampled.columns:
            df_resampled["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        if "eom_flag" not in df_resampled.columns:
            df_resampled["eom_flag"] = df_resampled["timestamp"].dt.is_month_end.astype(int)

    LOGGER.info(
        "[DATA] model=%s | rows=%d | series=%d | missing=%s",
        model_key,
        len(df_resampled),
        df_resampled["item_id"].nunique(),
        expected_missing,
    )

    return df_resampled


def _compute_run_signature(df: pd.DataFrame, horizon: int, freq: str) -> str:
    if df.empty:
        base = "empty"
    else:
        relevant = df[["item_id", "timestamp", "target"]].copy()
        relevant["timestamp"] = pd.to_datetime(relevant["timestamp"], utc=False)
        hash_series = pd.util.hash_pandas_object(relevant, index=False)
        hashed_bytes = hash_series.to_numpy(dtype="uint64", copy=False).tobytes()
        base = hashlib.sha1(hashed_bytes).hexdigest()
    return f"{base}|h={horizon}|f={freq}"


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
    comparison_payload: dict[str, dict[str, Any]],
    quantile_messages: dict[str, str],
    horizon: int,
    source_df: pd.DataFrame,
) -> None:
    all_actual = pd.concat([train_df, holdout_df], ignore_index=True) if not train_df.empty else holdout_df

    available_items = sorted(forecast_df["item_id"].dropna().unique())
    if not available_items:
        st.warning("Nessuna serie disponibile nelle previsioni.")
        return

    is_sd = model_label == "deepar_sd"
    is_f110 = model_label == "deepar_f110"
    is_ebs = model_label == "arima_ebs"
    sap_note: str | None = None
    if is_sd:
        _render_sd_use_case_panel(source_df)
        sap_note = "Tabelle SAP: VBRK/VBRP, VBAK/VBAP, KONV · KNA1/KNVV, MARA/MARC"
    elif is_f110:
        _render_f110_use_case_panel(source_df)
        sap_note = "Tabelle SAP: BSIK/BSAK, BSEG (ZFBDT,ZTERM,NETDT) · BKPF, T001"
    elif is_ebs:
        sap_note = "Tabelle SAP: FEBKO, FEBEP · campi: bank_acct, valuta, posting_date"

    display_name_by_key = {value: key for key, value in MODEL_KEY_MAP.items()}

    selected_item = cast(
        str,
        st.selectbox("Serie da visualizzare", options=available_items, key=f"series_{model_label}"),
    )
    LOGGER.info("[INFO][UI] Render grafico: item=%s, horizon=%d", selected_item, horizon)

    events = (
        _generate_f110_event_bands(all_actual, forecast_df, selected_item)
        if is_f110
        else _generate_event_bands(all_actual, forecast_df, selected_item)
    )

    metric_column = "MAPE"
    metric_label = "MAPE"
    metric_tooltip = "Errore percentuale medio (più basso = meglio)"
    if is_f110:
        metric_label = "MAPE (run)"
        metric_tooltip = "Errore percentuale medio calcolato sui soli giorni di run F110"
    if is_ebs:
        metric_column = "MAE"
        metric_label = "MAE (conteggi)"
        metric_tooltip = "Errore medio assoluto sui conteggi giornalieri dell'holdout"

    if is_f110:
        main_fig = viz.build_step_run_chart(
            all_actual,
            forecast_df,
            item_id=selected_item,
            events=events,
            sap_note=sap_note,
        )
        holdout_for_metrics = _filter_run_days(holdout_df)
    elif is_ebs:
        main_fig = viz.build_count_bar_chart(
            all_actual,
            forecast_df,
            item_id=selected_item,
            events=events,
            sap_note=sap_note,
        )
        holdout_for_metrics = holdout_df
    else:
        main_fig = viz.build_forecast_chart(
            all_actual,
            forecast_df,
            item_id=selected_item,
            events=events,
            sap_note=sap_note,
        )
        holdout_for_metrics = holdout_df

    LOGGER.debug(
        "[CHK] item=%s | actual=%d pts | holdout=%d pts | forecast=%d pts",
        selected_item,
        len(all_actual.loc[all_actual["item_id"] == selected_item]) if not all_actual.empty else 0,
        len(holdout_for_metrics.loc[holdout_for_metrics["item_id"] == selected_item])
        if not holdout_for_metrics.empty
        else 0,
        len(forecast_df.loc[forecast_df["item_id"] == selected_item]) if not forecast_df.empty else 0,
    )

    metrics = viz.compute_metrics(holdout_for_metrics, forecast_df) if not holdout_for_metrics.empty else pd.DataFrame()
    if is_ebs and not metrics.empty:
        mae_df = viz.compute_count_metrics(holdout_for_metrics, forecast_df)
        metrics = metrics.merge(mae_df[["item_id", "MAE"]], on="item_id", how="left")
    metrics_display = metrics.rename(
        columns={
            "item_id": "Serie",
            "Coverage[p10,p90]": "Copertura p10–p90",
        }
    ) if not metrics.empty else pd.DataFrame()

    selected_metrics = (
        metrics_display[metrics_display["Serie"] == selected_item].iloc[0]
        if not metrics_display.empty and selected_item in metrics_display["Serie"].values
        else None
    )

    if selected_metrics is not None:
        metric_value = float(selected_metrics.get(metric_column, float("nan")))
        LOGGER.info(
            "[INFO][METRICS] %s=%.3f sMAPE=%.3f Coverage=%.2f",
            metric_label,
            metric_value,
            selected_metrics["sMAPE"],
            selected_metrics["Copertura p10–p90"],
        )
    else:
        metric_value = float("nan")

    chips_html = _render_kpi_chips(
        metric_value=metric_value,
        coverage=float(selected_metrics["Copertura p10–p90"]) if selected_metrics is not None else float("nan"),
        horizon=horizon,
        series_count=len(available_items),
        metric_label=metric_label,
    )

    with chart_placeholder.container():
        st.markdown(chips_html, unsafe_allow_html=True)
        st.plotly_chart(main_fig, use_container_width=True)
        if events and any(event.label == "PROMO" for event in events):
            st.caption("Bande PROMO = periodi promozionali considerati come covariata")
        if is_f110:
            st.caption("Metriche calcolate sui soli giorni di run F110 (mar/ven disponibili).")

    tab_single, tab_compare = st.tabs(["Singolo modello", "Confronto modelli"])

    with tab_single:
        _render_single_model_section(
            metrics_placeholder=metrics_placeholder,
            download_placeholder=download_placeholder,
            forecast_df=forecast_df,
            metrics_display=metrics_display,
            model_label=model_label,
            chart_fig=main_fig,
            selected_item=selected_item,
            coverage=float(selected_metrics["Copertura p10–p90"]) if selected_metrics is not None else float("nan"),
            metric_value=metric_value,
            metric_label=metric_label,
            metric_tooltip=metric_tooltip,
            metric_column=metric_column,
        )

    with tab_compare:
        _render_comparison_section(
            all_actual=all_actual,
            comparison_payload=comparison_payload,
            holdout_df=holdout_for_metrics,
            selected_item=selected_item,
            quantile_messages=quantile_messages,
            display_name_by_key=display_name_by_key,
            sap_note=sap_note,
        )


def _render_kpi_chips(
    *,
    metric_value: float,
    coverage: float,
    horizon: int,
    series_count: int,
    metric_label: str = "MAPE",
) -> str:
    def _chip(label: str, value: str, tone: str | None = None) -> str:
        classes = ["kt-chip"]
        if tone:
            classes.append(f"kt-chip--{tone}")
        return f"<span class='{' '.join(classes)}'><span>{label}</span><span>{value}</span></span>"

    def _tone_from_metric(value: float) -> str | None:
        if math.isnan(value):
            return None
        label_upper = metric_label.upper()
        if "MAPE" in label_upper:
            if value <= 10:
                return "success"
            if value <= 20:
                return "warning"
            return "danger"
        return None

    def _tone_from_coverage(value: float) -> str | None:
        if math.isnan(value):
            return None
        if value >= 85:
            return "success"
        if value >= 70:
            return "warning"
        return "danger"

    label_upper = metric_label.upper()
    if math.isnan(metric_value):
        metric_display = "n/d"
    elif "MAPE" in label_upper:
        metric_display = f"{metric_value:.1f}%"
    elif "MAE" in label_upper:
        metric_display = f"{metric_value:,.0f}"
    else:
        metric_display = f"{metric_value:.2f}"
    coverage_display = "n/d" if math.isnan(coverage) else f"{coverage:.1f}%"

    chips = [
        _chip(metric_label, metric_display, _tone_from_metric(metric_value)),
        _chip("Coverage", coverage_display, _tone_from_coverage(coverage)),
        _chip("Orizzonte", f"{horizon}g"),
        _chip("Serie", str(series_count)),
    ]

    return f"<div class='kt-chip-row'>{''.join(chips)}</div>"


def _render_single_model_section(
    *,
    metrics_placeholder,
    download_placeholder,
    forecast_df: pd.DataFrame,
    metrics_display: pd.DataFrame,
    model_label: str,
    chart_fig,
    selected_item: str,
    coverage: float,
    metric_value: float,
    metric_label: str,
    metric_tooltip: str,
    metric_column: str,
) -> None:
    with metrics_placeholder.container():
        if metrics_display.empty:
            st.info("Nessun holdout disponibile per il calcolo delle metriche.")
        else:
            metric_series = metrics_display.get(metric_column)
            baseline = float(metric_series.median(skipna=True)) if isinstance(metric_series, pd.Series) and not metric_series.empty else float("nan")

            def _info_header(label: str, tooltip: str) -> str:
                return f"{label} <span class='kt-info' title=\"{tooltip}\">ℹ</span>"

            def _format_percent(value: float, decimals: int = 2) -> str:
                if pd.isna(value):
                    return "—"
                return f"{value:.{decimals}f}%"

            def _format_pinball(value: float) -> str:
                if pd.isna(value):
                    return "—"
                return f"{value:.3f}"

            def _format_primary(value: float) -> str:
                if pd.isna(value):
                    return "—"
                label_upper = metric_label.upper()
                if "MAPE" in label_upper:
                    return f"{value:.2f}%"
                if "MAE" in label_upper:
                    return f"{value:,.0f}"
                return f"{value:.2f}"

            def _trend_badge(value: float) -> str:
                if pd.isna(value) or pd.isna(baseline) or baseline <= 0:
                    return brand_badge("≈ baseline", tone="warning")
                delta = (value - baseline) / baseline * 100
                if abs(delta) < 1.5:
                    return brand_badge("≈ baseline", tone="warning")
                if delta < 0:
                    return brand_badge(f"▼ {abs(delta):.1f}%", tone="success")
                return brand_badge(f"▲ {delta:.1f}%", tone="danger")

            def _coverage_badge(value: float) -> str:
                if pd.isna(value):
                    return brand_badge("n/d", tone="warning")
                if value >= 85:
                    tone = "success"
                elif value >= 70:
                    tone = "warning"
                else:
                    tone = "danger"
                return brand_badge(f"{value:.1f}%", tone=tone)

            table_rows = []
            for _, row in metrics_display.iterrows():
                primary_value = row.get(metric_column, float("nan"))
                table_rows.append(
                    {
                        "Serie": row["Serie"],
                        metric_column: _format_primary(primary_value),
                        "Trend": _trend_badge(primary_value),
                        "sMAPE": _format_percent(row["sMAPE"]),
                        "Pinball@0.1": _format_pinball(row["Pinball@0.1"]),
                        "Pinball@0.9": _format_pinball(row["Pinball@0.9"]),
                        "Copertura p10–p90": _coverage_badge(row["Copertura p10–p90"]),
                    }
                )

            metrics_table = pd.DataFrame(table_rows, columns=[
                "Serie",
                metric_column,
                "Trend",
                "sMAPE",
                "Pinball@0.1",
                "Pinball@0.9",
                "Copertura p10–p90",
            ])
            metrics_table.columns = [
                "Serie",
                _info_header(metric_label, metric_tooltip),
                "Trend",
                _info_header("sMAPE", "Versione simmetrica della MAPE"),
                _info_header("Pinball@0.1", "Accuratezza sui quantili (bassa = meglio)"),
                _info_header("Pinball@0.9", "Accuratezza sui quantili (bassa = meglio)"),
                _info_header("Copertura p10–p90", "% punti reali dentro la banda"),
            ]
            metrics_html = metrics_table.to_html(index=False, escape=False, classes="kt-metric-table")
            st.markdown(metrics_html, unsafe_allow_html=True)

    csv_buffer = StringIO()
    forecast_df.to_csv(csv_buffer, index=False)

    png_bytes: bytes | None = None
    png_error: str | None = None
    try:
        png_bytes = chart_fig.to_image(format="png", width=1280, height=640, scale=2)
    except Exception as exc:  # pragma: no cover - optional feature
        png_error = str(exc)

    insight_text = _build_insight_text(
        item_id=selected_item,
        metric_value=metric_value,
        metric_label=metric_label,
        coverage=coverage,
        peak_date=_estimate_risk_peak(forecast_df, selected_item),
    )

    escaped_insight = json.dumps(insight_text)
    copy_button = f"<button class='kt-copy' onclick=\"navigator.clipboard.writeText({escaped_insight})\">Copia insight</button>"

    with download_placeholder.container():
        col_csv, col_png, col_copy = st.columns([1, 1, 1])
        with col_csv:
            st.download_button(
                label="Scarica CSV previsioni",
                data=csv_buffer.getvalue(),
                file_name=f"forecast_{model_label}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_png:
            if png_bytes is not None:
                st.download_button(
                    label="Scarica PNG grafico",
                    data=png_bytes,
                    file_name=f"forecast_{model_label}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    "<button class='kt-copy kt-download-disabled' disabled>Scarica PNG grafico</button>",
                    unsafe_allow_html=True,
                )
                if png_error:
                    st.caption("Installare `kaleido` per abilitare l'esportazione PNG.")
        with col_copy:
            st.markdown(copy_button, unsafe_allow_html=True)


def _render_sd_use_case_panel(df: pd.DataFrame) -> None:
    price_threshold_pct = PRICE_DRIFT_THRESHOLD * 100
    bullet_html = f"""
    <div class="kt-callout">
        <h4>Use case: Domanda commerciale (SD)</h4>
        <ul>
            <li>SKU × Cliente × Area</li>
        	<li>Promo (<code>promo_flag</code>)</li>
        	<li>Indice prezzo (<code>price_idx</code>)</li>
            <li><strong>Valori consigliati (demo)</strong>
                <ul>
                    <li>Orizzonte default: 30 giorni</li>
                    <li>Epoche: 30 (demo) con early_stop_patience=5</li>
                    <li>Num samples: 200 per quantili più stabili</li>
                    <li>Embedding dimension: euristica radice quarta × 4 (prompt 2)</li>
                    <li>Soglia drift prezzo: ±{price_threshold_pct:.0f}%</li>
                </ul>
            </li>
        </ul>
    </div>
    """

    st.markdown(bullet_html, unsafe_allow_html=True)

    columns_to_check = ["sku", "customer", "area", "promo_flag", "price_idx"]
    rows = []
    for column in columns_to_check:
        present = column in df.columns
        example = ""
        if present:
            sample_value = df[column].dropna().astype(str).head(1).tolist()
            example = sample_value[0] if sample_value else ""
        rows.append({
            "Colonna": column,
            "Presente": "✅" if present else "—",
            "Esempio": example,
        })

    table_df = pd.DataFrame(rows)
    st.table(table_df)


def _render_f110_use_case_panel(df: pd.DataFrame) -> None:
    bullet_html = """
    <div class="kt-callout">
        <h4>Use case: Pagamenti AP (F110)</h4>
        <ul>
            <li>Run programmati martedì/venerdì via transazione F110</li>
            <li>Covariate dinamiche: <code>run_flag</code>, <code>volume_due_next_7d</code>, feature calendario</li>
            <li>Covariate statiche: <code>company_code</code>, <code>bank_acct</code></li>
            <li>Boost end-of-month e salti festivi gestiti nel mock</li>
        </ul>
    </div>
    """
    st.markdown(bullet_html, unsafe_allow_html=True)

    columns_to_check = [
        "company_code",
        "bank_acct",
        "run_flag",
        "eom_flag",
        "volume_due_next_7d",
        "dow_sin",
        "dow_cos",
    ]
    rows = []
    for column in columns_to_check:
        present = column in df.columns
        example = ""
        if present:
            sample_value = df[column].dropna().astype(str).head(1).tolist()
            example = sample_value[0] if sample_value else ""
        rows.append({
            "Colonna": column,
            "Presente": "✅" if present else "—",
            "Esempio": example,
        })

    table_df = pd.DataFrame(rows)
    st.table(table_df)


def _filter_run_days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    mask = pd.Series(False, index=working.index)
    if "target" in working.columns:
        mask |= pd.to_numeric(working["target"], errors="coerce").fillna(0.0) > 0
    if "run_flag" in working.columns:
        mask |= working["run_flag"].astype(int) == 1
    return working.loc[mask].copy()


def _render_comparison_section(
    *,
    all_actual: pd.DataFrame,
    comparison_payload: dict[str, dict[str, Any]],
    holdout_df: pd.DataFrame,
    selected_item: str,
    quantile_messages: dict[str, str],
    display_name_by_key: dict[str, str],
    sap_note: str | None = None,
) -> None:
    forecasts_for_chart: dict[str, pd.DataFrame] = {}
    for key, payload in comparison_payload.items():
        forecast = payload.get("forecast")
        if forecast is None or forecast.empty:
            continue
        display_name = display_name_by_key.get(key, key.title())
        forecasts_for_chart[display_name] = forecast

    if not forecasts_for_chart:
        st.info("Nessun forecast disponibile per il confronto.")
        return

    comparison_fig = viz.build_comparison_chart(
        all_actual,
        forecasts_for_chart,
        item_id=selected_item,
        sap_note=sap_note,
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

    if len(forecasts_for_chart) == 1:
        st.info("Esegui un altro modello dal menu per popolare il confronto.")

    missing_messages = [
        f"{display_name_by_key.get(key, key.title())}: {msg}"
        for key, msg in quantile_messages.items()
        if key in comparison_payload and (comparison_payload[key].get("forecast") is not None)
    ]
    if missing_messages:
        st.caption(" · ".join(missing_messages))

    if holdout_df.empty:
        st.info("Nessun holdout disponibile per confrontare le metriche.")
        return

    comparison_rows: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {
        "MAPE": [],
        "sMAPE": [],
        "Pinball@0.1": [],
        "Pinball@0.9": [],
        "Coverage[p10,p90]": [],
    }

    for key, payload in comparison_payload.items():
        forecast = payload.get("forecast")
        duration = payload.get("duration")
        if forecast is None or forecast.empty:
            continue

        model_metrics = viz.compute_metrics(holdout_df, forecast)
        if "item_id" not in model_metrics.columns:
            if model_metrics.index.name == "item_id":
                model_metrics = model_metrics.reset_index()
            else:
                model_metrics = model_metrics.assign(item_id=selected_item)
        row = model_metrics[model_metrics["item_id"] == selected_item]
        if row.empty:
            row = model_metrics.mean(numeric_only=True)
            if isinstance(row, pd.Series):
                metrics_dict = row.to_dict()
            else:
                continue
        else:
            metrics_dict = row.iloc[0].to_dict()

        display_name = display_name_by_key.get(key, key.title())
        comparison_rows.append(
            {
                "Modello": display_name,
                "MAPE": metrics_dict.get("MAPE", float("nan")),
                "sMAPE": metrics_dict.get("sMAPE", float("nan")),
                "Pinball@0.1": metrics_dict.get("Pinball@0.1", float("nan")),
                "Pinball@0.9": metrics_dict.get("Pinball@0.9", float("nan")),
                "Coverage": metrics_dict.get("Coverage[p10,p90]", float("nan")),
                "Durata (s)": duration,
            }
        )

        for metric_name in metric_values.keys():
            metric_values[metric_name].append(metrics_dict.get(metric_name, float("nan")))

    if not comparison_rows:
        st.info("Impossibile calcolare le metriche di confronto.")
        return

    best_low = {
        key: np.nanmin(values) if values else float("nan")
        for key, values in metric_values.items()
        if key != "Coverage[p10,p90]"
    }
    coverage_values = metric_values.get("Coverage[p10,p90]", [])
    best_high = np.nanmax(coverage_values) if coverage_values else float("nan")

    table_rows = []
    for row in comparison_rows:
        formatted = {"Modello": row["Modello"]}
        for metric in ("MAPE", "sMAPE", "Pinball@0.1", "Pinball@0.9"):
            value = row[metric]
            display = "n/d" if pd.isna(value) else f"{value:.2f}%" if metric.endswith("MAPE") else f"{value:.3f}"
            if not pd.isna(value) and math.isclose(value, best_low.get(metric, float("nan")), rel_tol=1e-9, abs_tol=1e-6):
                display = f"<span class='kt-highlight'>{display}</span>"
            formatted[metric] = display
        coverage_value = row["Coverage"]
        coverage_display = "n/d" if pd.isna(coverage_value) else f"{coverage_value:.2f}%"
        if not (pd.isna(coverage_value) or math.isnan(best_high)) and math.isclose(
            coverage_value,
            best_high,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            coverage_display = f"<span class='kt-highlight'>{coverage_display}</span>"
        formatted["Coverage"] = coverage_display
        duration_value = row["Durata (s)"]
        formatted["Durata (s)"] = "n/d" if duration_value is None else f"{duration_value:.2f}"
        table_rows.append(formatted)

    comparison_table = pd.DataFrame(table_rows, columns=[
        "Modello",
        "MAPE",
        "sMAPE",
        "Pinball@0.1",
        "Pinball@0.9",
        "Coverage",
        "Durata (s)",
    ])
    table_html = comparison_table.to_html(index=False, escape=False, classes="kt-metric-table")
    st.markdown(table_html, unsafe_allow_html=True)


def _generate_f110_event_bands(
    all_actual: pd.DataFrame,
    forecast_df: pd.DataFrame,
    item_id: str,
) -> list[EventBand]:
    def _collect_timestamps() -> pd.Series:
        actual_dates = (
            all_actual[all_actual["item_id"] == item_id]["timestamp"]
            if not all_actual.empty
            else pd.Series(dtype="datetime64[ns]")
        )
        forecast_dates = forecast_df[forecast_df["item_id"] == item_id]["timestamp"]
        return (
            pd.concat([actual_dates, forecast_dates], ignore_index=True)
            .dropna()
            .sort_values()
            .reset_index(drop=True)
        )

    combined = _collect_timestamps()
    if combined.empty:
        return []

    diffs = combined.sort_values().diff().dropna()
    if diffs.empty:
        half_window = pd.Timedelta(hours=12)
    else:
        min_delta = diffs.min()
        half_window = min_delta / 2 if isinstance(min_delta, pd.Timedelta) and min_delta > pd.Timedelta(0) else pd.Timedelta(hours=12)

    item_actual = all_actual[all_actual["item_id"] == item_id].sort_values("timestamp") if not all_actual.empty else pd.DataFrame()
    if item_actual.empty:
        return []

    events: list[EventBand] = []

    if "eom_flag" in item_actual.columns:
        eom_dates = pd.to_datetime(item_actual.loc[item_actual["eom_flag"].astype(int) == 1, "timestamp"], errors="coerce")
        for ts in sorted({ts for ts in eom_dates.dropna()}):
            events.append(EventBand(label="FINE MESE", start=ts - half_window, end=ts + half_window))

    if "run_flag" in item_actual.columns:
        candidate_mask = item_actual["timestamp"].dt.weekday.isin([1, 4])
        skipped = item_actual.loc[candidate_mask & (item_actual["run_flag"].astype(int) == 0)]
        skipped_dates = pd.to_datetime(skipped["timestamp"], errors="coerce")
        for ts in sorted({ts for ts in skipped_dates.dropna()}):
            events.append(EventBand(label="NO RUN", start=ts - half_window, end=ts + half_window))

    return events


def _generate_event_bands(all_actual: pd.DataFrame, forecast_df: pd.DataFrame, item_id: str):
    promo_col = "promo_flag"

    def _collect_timestamps() -> pd.Series:
        actual_dates = (
            all_actual[all_actual["item_id"] == item_id]["timestamp"]
            if not all_actual.empty
            else pd.Series(dtype="datetime64[ns]")
        )
        forecast_dates = forecast_df[forecast_df["item_id"] == item_id]["timestamp"]
        return pd.concat([actual_dates, forecast_dates], ignore_index=True).dropna().sort_values().reset_index(drop=True)

    combined = _collect_timestamps()
    if combined.empty:
        return []

    diffs = combined.sort_values().diff().dropna()
    if diffs.empty:
        half_window = pd.Timedelta(hours=12)
    else:
        min_delta = diffs.min()
        if isinstance(min_delta, pd.Timedelta) and min_delta > pd.Timedelta(0):
            half_window = min_delta / 2
        else:
            half_window = pd.Timedelta(hours=12)

    item_actual = all_actual[all_actual["item_id"] == item_id].sort_values("timestamp") if not all_actual.empty else pd.DataFrame()

    if not item_actual.empty and promo_col in item_actual.columns and item_actual[promo_col].fillna(0).ne(0).any():
        bands: list[EventBand] = []
        start_ts: pd.Timestamp | None = None
        last_ts: pd.Timestamp | None = None

        for row in item_actual.itertuples():
            ts = getattr(row, "timestamp", None)
            if pd.isna(ts):
                continue
            if not isinstance(ts, pd.Timestamp):
                try:
                    ts = pd.Timestamp(ts)
                except (TypeError, ValueError):
                    continue

            raw_flag = getattr(row, promo_col)
            flag_active = False if pd.isna(raw_flag) else bool(raw_flag)

            if flag_active:
                if start_ts is None:
                    start_ts = ts
                last_ts = ts
            elif start_ts is not None and last_ts is not None:
                bands.append(EventBand(label="PROMO", start=start_ts - half_window, end=last_ts + half_window))
                start_ts = None
                last_ts = None

        if start_ts is not None and last_ts is not None:
            bands.append(EventBand(label="PROMO", start=start_ts - half_window, end=last_ts + half_window))

        return bands

    # Fallback to existing demo labels when promo_flag is unavailable.
    labels = ["STIPENDI", "IVA", "Ritardo incassi"]
    unique_dates = combined.drop_duplicates()
    count = min(len(labels), len(unique_dates))
    if count == 0:
        return []

    indices = np.linspace(0, len(unique_dates) - 1, count, dtype=int)
    bands = []
    for label, idx in zip(labels, indices):
        ts = pd.to_datetime(unique_dates.iloc[idx])
        bands.append(EventBand(label=label, start=ts - half_window, end=ts + half_window))
    return bands


def _estimate_risk_peak(forecast_df: pd.DataFrame, item_id: str) -> str:
    series = forecast_df[forecast_df["item_id"] == item_id].sort_values("timestamp")
    if series.empty or {"p90", "p10", "timestamp"} - set(series.columns):
        return "n/d"
    band = series["p90"] - series["p10"]
    if band.empty or band.isna().all():
        return "n/d"
    idx = band.idxmax()
    if pd.isna(idx):
        return "n/d"
    ts = series.loc[idx, "timestamp"]
    if pd.isna(ts):
        return "n/d"
    if not isinstance(ts, pd.Timestamp):
        try:
            ts = pd.Timestamp(cast(Any, ts))
        except (TypeError, ValueError):
            return "n/d"
    return ts.strftime("%Y-%m-%d")


def _build_insight_text(
    *,
    item_id: str,
    metric_value: float,
    metric_label: str,
    coverage: float,
    peak_date: str,
) -> str:
    label_upper = metric_label.upper()
    if math.isnan(metric_value):
        metric_display = "n/d"
    elif "MAPE" in label_upper:
        metric_display = f"{metric_value:.1f}%"
    elif "MAE" in label_upper:
        metric_display = f"{metric_value:,.0f}"
    else:
        metric_display = f"{metric_value:.2f}"

    coverage_display = "n/d" if math.isnan(coverage) else f"{coverage:.1f}%"
    return (
        f"Per {item_id}, {metric_label}={metric_display}, coverage={coverage_display}. "
        f"Picco rischio cassa: {peak_date}."
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
