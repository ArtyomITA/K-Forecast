# K-Forecast · Technical README

Documento di riferimento per sviluppatori e data scientist che contribuiscono
alla piattaforma. Integra le informazioni operative, gli script di supporto e le
note d'implementazione più recenti.

## Sommario

- [Stack e prerequisiti](#stack-e-prerequisiti)
- [Setup ambiente](#setup-ambiente)
- [Avvio applicazione](#avvio-applicazione)
- [Struttura del repository](#struttura-del-repository)
- [Modelli supportati](#modelli-supportati)
- [Gestione dataset e mock](#gestione-dataset-e-mock)
- [Interfaccia utente e branding](#interfaccia-utente-e-branding)
- [Logging e diagnostica](#logging-e-diagnostica)
- [Test e qualità](#test-e-qualità)
- [Troubleshooting](#troubleshooting)
- [Roadmap tecnica](#roadmap-tecnica)

## Stack e prerequisiti

- **Python 3.10 – 3.12** (supporto verificato). Python 3.13 non è ancora
  validato con la pipeline DeepAR/Lightning.
- **Streamlit 1.37+** per l'interfaccia.
- **GluonTS**, **PyTorch Lightning**, **statsmodels**, **Plotly** come dipendenze
  principali (pin disponibili in `requirements.txt`).
- Sistema operativo testato: Windows 11, macOS Sonoma, Ubuntu 22.04.

## Setup ambiente

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Suggerito installare opzionalmente `kaleido` per abilitare il download PNG dei
plot.

## Avvio applicazione

Metodo consigliato (include bootstrap Streamlit interno):

```bash
python app.py
```

In alternativa è sempre possibile usare:

```bash
streamlit run app.py
```

L'applicazione espone log CLI e apre il browser sul percorso locale.

## Struttura del repository

```
K-Forecast/
├── app.py                  # Entry point Streamlit + bootstrap CLI
├── assets/
│   └── app.css             # Tema Keytech, caricato dinamicamente
├── models/
│   ├── arima_model.py      # Wrapper ARIMA/SARIMA per EBS
│   ├── deepar_model.py     # Pipeline DeepAR (Lightning) per SD e F110
│   ├── ag_automl_model.py  # Wrapper AutoGluon legacy (disabilitato in UI)
│   └── chronos_model.py    # Wrapper Chronos-Bolt legacy (disabilitato in UI)
├── utils/
│   ├── theme.py            # Palette brand e template Plotly
│   └── viz.py              # Grafici, metriche, badge e download CSV
├── data/                   # Dataset mock e file demo
├── scripts/
│   ├── generate_mock_f110_payouts.py
│   ├── generate_mock_ebs_counts.py
│   └── generate_mock_sd_sales.py
├── requirements.txt
└── README.md / README-tech.md
```

### Componenti chiave

- `app.py`: orchestrazione UI, gestione stato, cache sessione, validazione
  dataset, routing verso i modelli.
- `assets/app.css`: stylesheet brandizzato (novità: l'iniezione inline è stata
  rimossa in favore di questo file esterno, caricato con placeholder sostituiti
  a runtime).
- `utils/theme.py`: definizione palette e funzioni per i badge.
- `utils/viz.py`: costruzione dei grafici Plotly, tabella metriche, generazione
  CSV esportabile.
- `models/*`: moduli indipendenti per training/inference dei tre motori.

## Modelli supportati

| Modello | Libreria | Note operative |
|---------|----------|----------------|
| DeepAR (SD) | GluonTS / PyTorch Lightning | Covariate dinamiche (promo_flag, price_idx) e feature calendario (sin/cos, eom). |
| DeepAR (F110) | GluonTS / PyTorch Lightning | Run programmati martedì/venerdì con covariate run_flag e volume_due_next_7d. |
| ARIMA/SARIMA (EBS) | statsmodels | Auto-selezione p/d/q con controllo stagionalità settimanale; output p10/p50/p90 sintetizzato dal residuo. |

I forecast vengono normalizzati su tre colonne `p10`, `p50`, `p90` per uniformare
la UI. I wrapper Chronos/AutoML restano nel repository ma non sono caricati nella
versione demo.

## Gestione dataset e mock

- **Registry**: in `app.py` (`DATASET_REGISTRY`) sono mappati i mock disponibili
  con metadati e schema previsto.
- **Upload CSV**: convalida sulle colonne `item_id`, `timestamp`, `target`; il
  formato datetime viene inferito e normalizzato a frequenza giornaliera.
- **Mock generator**:
  - `python scripts/generate_mock_f110_payouts.py`
  - `python scripts/generate_mock_sd_sales.py`
  - `python scripts/generate_mock_ebs_counts.py`

Gli script salvano i CSV in `data/` e coprono i casi Tesoreria, Retail promo ed
elaborazioni giornaliere EBS. Il vecchio `generate_mock_bank_flows.py` resta come
reference per scenari legacy Chronos.

## Interfaccia utente e branding

- Il tema Keytech è governato da `assets/app.css` che utilizza placeholder
  (`__KT_ORANGE__`, `__FONT_STACK__`, ecc.) sostituiti da `inject_brand_css()`.
- La cache di confronto modelli è session-based e viene invalidata quando cambia
  dataset, frequenza o orizzonte.
- I grafici mostrano bande evento tramite `utils.viz.EventBand` (promo o altri
  eventi di business).

## Logging e diagnostica

- Logger principale: `k_forecast.app` (INFO di default, DEBUG attivabile da
  sidebar).
- I wrapper modello loggano parametri principali, tempi di esecuzione e metriche.
- Gli errori vengono catturati e mostrati in Streamlit con messaggi user-friendly
  senza interrompere la sessione.

## Test e qualità

- Verifica sintattica rapida:

  ```bash
  python -m py_compile app.py
  ```

- È possibile aggiungere unit test (PyTest) per moduli specifici; ad oggi non
  sono inclusi.

## Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| `ModuleNotFoundError: statsmodels` | Installare le dependences con `pip install -r requirements.txt` (statsmodels è obbligatoria per ARIMA). |
| Grafico vuoto dopo upload CSV | Verificare colonne richieste e presenza di dati nell'orizzonte selezionato; il log riporta eventuali colonne mancanti. |
| Metriche NaN nel pannello KPI | Controllare che il dataset copra almeno 30 giorni utili; in caso contrario ridurre l'orizzonte o aumentare lo storico. |
| Stile non applicato | Verificare esistenza di `assets/app.css`; in caso di assenza il log avverte e il fallback è il tema Streamlit standard. |

## Roadmap tecnica

- Job di training batch schedulati + storage forecast storico.
- Estensione UI per scenari what-if multi-variabile.
- Hardening layer API per ingest da sistemi esterni.

---

Ultimo aggiornamento: ottobre 2025 (allineamento lineup DeepAR F110/SD e ARIMA EBS,
mock generator aggiornati e dismissione wrapper Chronos/AutoML dalla UI).
