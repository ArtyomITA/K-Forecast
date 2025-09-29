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

- **Python 3.10 – 3.12** (supporto verificato). Python 3.13 non è attualmente
  compatibile con AutoGluon/Chronos per assenza del wheel `tf-keras`.
- **Streamlit 1.37+** per l'interfaccia.
- **AutoGluon TimeSeries**, **GluonTS**, **PyTorch**, **Plotly** come dipendenze
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
│   ├── ag_automl_model.py  # Wrapper AutoGluon TimeSeries
│   ├── chronos_model.py    # Wrapper Chronos-Bolt
│   └── deepar_model.py     # Pipeline DeepAR (Lightning)
├── utils/
│   ├── theme.py            # Palette brand e template Plotly
│   └── viz.py              # Grafici, metriche, badge e download CSV
├── data/                   # Dataset mock e file demo
├── scripts/
│   ├── generate_mock_bank_flows.py
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
| Chronos-Bolt | AutoGluon TimeSeries | Inference zero-shot con quantili 0.1/0.5/0.9. `TRANSFORMERS_NO_TF=1` impostato per evitare dipendenza TensorFlow. |
| DeepAR | GluonTS / PyTorch Lightning | Training locale con early stopping (`patience=5`), supporto covariate statiche e dinamiche, esportazione quantili multipli. |
| AutoML (AG-TS) | AutoGluon TimeSeries | Leaderboard completa, armonizzazione automatica quantili; fallback duplicando `p50` se `p10/p90` assenti. |

I forecast vengono normalizzati su tre colonne `p10`, `p50`, `p90` per uniformare
la UI.

## Gestione dataset e mock

- **Registry**: in `app.py` (`DATASET_REGISTRY`) sono mappati i mock disponibili
  con metadati e schema previsto.
- **Upload CSV**: convalida sulle colonne `item_id`, `timestamp`, `target`; il
  formato datetime viene inferito e normalizzato a frequenza giornaliera.
- **Mock generator**:
  - `python scripts/generate_mock_bank_flows.py`
  - `python scripts/generate_mock_sd_sales.py`

Entrambi generano output sotto `data/` e sono utilizzati rispettivamente per i
casi Tesoreria e Retail promo.

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
| `ModuleNotFoundError: tf_keras` | Utilizzare Python ≤ 3.12 e assicurarsi che `TRANSFORMERS_NO_TF=1` rimanga settato (gestito in app). |
| Grafico vuoto dopo upload CSV | Verificare colonne richieste e presenza di dati nell'orizzonte scelto. |
| Messaggio "Quantili mancanti" in AutoML | Alcuni modelli restituiscono solo `median`; l'app replica `p50` sui quantili mancanti. |
| Stile non applicato | Verificare esistenza di `assets/app.css`; in caso di assenza il log avverte e il fallback è il tema Streamlit standard. |

## Roadmap tecnica

- Job di training batch schedulati + storage forecast storico.
- Estensione UI per scenari what-if multi-variabile.
- Hardening layer API per ingest da sistemi esterni.

---

Ultimo aggiornamento: settembre 2025 (include estrazione CSS in `assets/app.css`
e registri dataset centralizzati in `app.py`).
