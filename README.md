# K-Forecast

Applicazione **Streamlit** per il forecasting probabilistico di flussi di cassa
multi-serie. L'applicazione integra tre percorsi modellistici (Chronos-Bolt,
DeepAR e AutoGluon AutoML), produce visualizzazioni interattive, calcola
metriche di accuratezza e permette il download dei risultati.

## Sommario

- [Caratteristiche principali](#caratteristiche-principali)
- [Requisiti e compatibilità](#requisiti-e-compatibilità)
- [Setup rapido](#setup-rapido)
- [Come usare l'app](#come-usare-lapp)
- [Datasets e mock generator](#datasets-e-mock-generator)
- [Use case SD (DeepAR)](#use-case-sd-deepar)
- [Architettura del progetto](#architettura-del-progetto)
- [Modelli supportati](#modelli-supportati)
- [Debug e log](#debug-e-log)
- [Suggerimenti di sviluppo](#suggerimenti-di-sviluppo)
- [Troubleshooting](#troubleshooting)

## Caratteristiche principali

- **Interfaccia unica**: dashboard Streamlit con controlli per modello,
	orizzonte di previsione, scelta dataset (mock vs upload CSV) e attivazione dei
	log di debug.
- **Flux multi-serie**: supporto a dataset "long" con colonne
	`item_id`, `timestamp`, `target`, resampling automatico a frequenza giornaliera
	e split train/holdout.
- **Percorsi modellistici multipli**:
  - Chronos-Bolt (Amazon) via AutoGluon TimeSeries con quantili 0.1/0.5/0.9.
  - DeepAR (GluonTS/PyTorch Lightning) con fine-tuning locale.
  - AutoML ensemble (AutoGluon TimeSeries) con leaderboard e valutazione.
- **Esecuzioni one-shot**: ogni click su "Esegui Forecast" avvia esclusivamente il modello selezionato,
  così da ridurre i tempi di attesa e permettere confronti manuali.
- **Confronto progressivo**: i risultati vengono memorizzati in cache di sessione (per dataset, frequenza e horizon) e
  popolano la tab "Confronto modelli" man mano che si lanciano gli altri modelli.
- **Visualizzazioni interattive**: grafici Plotly con intervalli di confidenza e
	tabella metriche (MAPE, sMAPE, Pinball Loss, Coverage) sull'holdout.
- **Download immediato**: esportazione CSV delle previsioni direttamente dalla
	UI.
- **Avvio semplificato**: `python app.py` apre automaticamente Streamlit grazie
	al bootstrap interno.

## Requisiti e compatibilità

- **Python 3.10 – 3.12** consigliato. Python 3.13 non è attualmente supportato
	da AutoGluon/Chronos (dipendenza `tf-keras` non disponibile) e causerà errori.
- Ambiente virtuale suggerito: `python -m venv .venv && source .venv/bin/activate`
	(su Windows: `.venv\Scripts\activate`).
- Dipendenze principali: Streamlit, AutoGluon TimeSeries, GluonTS, Torch,
	Plotly. Tutte elencate in `requirements.txt`.

## Setup rapido

```bash
pip install -r requirements.txt
python app.py  # avvia Streamlit automaticamente
```

In alternativa è sempre possibile usare `streamlit run app.py`, ma il bootstrap
incorporato permette di lavorare con il solo comando Python (utile ad esempio
per integrazioni con task runner o debugger).

> Suggerimento: per abilitare il download PNG dei grafici installare anche
> `kaleido` (`pip install kaleido`). In assenza del pacchetto il bottone viene
> disabilitato e compare un messaggio informativo.

## Come usare l'app

1. **Avviare** l'app con `python app.py` e aprire l'URL indicato dalla CLI.
2. **Selezionare il modello** (Chronos-Bolt, DeepAR o AutoML) dalla sidebar.
3. **Impostare l'orizzonte** di previsione (7, 30 o 90 giorni) e, se necessario,
	 attivare il toggle "DEBUG logs" per maggiori dettagli.
4. **Scegliere il dataset**:
	 - "Usa mock data" carica `data/mock_bank_flows.csv` generato dallo script
		 fornito.
	 - "Carica CSV" accetta file long con `item_id,timestamp,target`.
5. **Eseguire il forecast** premendo "Esegui Forecast". Il grafico e le metriche
	 compariranno in pagina; è disponibile anche il bottone per scaricare il CSV
	 delle previsioni. Ad ogni run viene eseguito solo il modello selezionato.
6. **Confrontare i modelli (opzionale)**: lanciare gli altri modelli con gli
	 stessi dati/orizzonte. I forecast recenti vengono riutilizzati nella tab
	 "Confronto modelli". Se viene mostrato il messaggio "Esegui un altro modello..."
	 significa che è presente soltanto un forecast in cache.

> ℹ️ La cache di confronto viene azzerata automaticamente quando cambia uno tra
> dataset, frequenza o horizon, così da evitare mix di contesti differenti.

## Datasets e mock generator

- La cartella `data/` ospita il dataset mock principale (`mock_bank_flows.csv`)
	e una slice di test.
- Per rigenerare i dati sintetici:

	```bash
	python scripts/generate_mock_bank_flows.py
	```

	Lo script crea serie giornaliere con stagionalità settimanale, spike mensili
	e finestre di ritardo controllate. I file vengono salvati automaticamente in
	`data/`.
- Per caricare dati propri, assicurarsi che il CSV contenga esattamente le
	colonne richieste con timestamp ISO (`YYYY-MM-DD` o full datetime).

## Use case SD (DeepAR)

- **Dataset dedicato**: selezionando DeepAR con "Usa mock data" l'app carica automaticamente `data/mock_sd_sales.csv`, generato dallo script `scripts/generate_mock_sd_sales.py`. Ogni riga rappresenta le vendite giornaliere (`target`) per una coppia SKU × Cliente × Area (`item_id` è composto concatenando i tre campi). Le colonne supplementari includono:
	- `promo_flag`: indicatore binario (0/1) della presenza di una campagna promozionale nel giorno.
	- `price_idx`: indice di prezzo nominale che viene normalizzato a livello SKU (`price_idx_norm`).
	- `sku`, `customer`, `area`: attributi statici usati per le embedding categoriali.
- **Lettura delle bande PROMO**: nel grafico principale, le bande arancioni etichettate "PROMO" evidenziano gli intervalli in cui `promo_flag` è attivo. Sono le stesse finestre considerate come covariata dinamica dal modello, così è immediato confrontare picchi di domanda e calendario promozionale.
- **Perché DeepAR**: il dataset SD contiene più serie correlate (SKU×Cliente×Area) con pattern stagionali condivisi e covariate esplicative. DeepAR sfrutta embedding statiche per modellare la gerarchia e covariate dinamiche (`promo_flag`, `price_idx_norm`, `dow_*`, `eom_flag`) per apprendere gli effetti promozionali, mantenendo coerenza tra serie e migliorando la qualità delle previsioni probabilistiche.
- **Valori consigliati (placeholder)**:
	- Orizzonte default: **30 giorni**.
	- Epoche: **30** per le demo, con `early_stop_patience=5` per fermare il training in anticipo.
	- Num samples: **200** campioni per ottenere quantili più stabili.
	- Embedding dimension: euristica già implementata (`pow(cardinalità, 0.25) × 4`) come indicato nel prompt 2.
	- Soglia drift prezzo: variazioni oltre **±3 %** sull'indice `price_idx` meritano una review dei listini.

## Architettura del progetto

```
K-Forecast/
├── app.py                     # Entry point Streamlit + bootstrap CLI
├── models/
│   ├── chronos_model.py       # Wrapper Chronos-Bolt (AutoGluon TimeSeries)
│   ├── deepar_model.py        # Pipeline DeepAR con PyTorch Lightning
│   └── ag_automl_model.py     # AutoML ensemble + report artefatti
├── utils/
│   ├── viz.py                 # Grafici Plotly e metriche di valutazione
│   └── mock_data.py           # (placeholder) helper futuri per dati sintetici
├── scripts/
│   └── generate_mock_bank_flows.py
├── data/                      # Dataset mock generati
└── requirements.txt
```

Componenti chiave:

- `app.py` orchestration: gestione UI, caricamento dataset, preprocess, split,
  logging, invocazione modelli, cache di sessione per i confronti e rendering risultati.
- `utils/viz.py`: costruzione grafici, metriche e download CSV.
- `models/*`: singoli wrapper per incapsulare setup, training, previsione e
	logging di ciascun approccio.

## Modelli supportati

| Modello         | Libreria                 | Note principali |
|-----------------|--------------------------|-----------------|
| Chronos-Bolt    | AutoGluon TimeSeries     | Inference zero-shot con fallback fine-tuning. TensorFlow disabilitato di default per evitare dipendenza `tf-keras`. |
| DeepAR          | GluonTS + PyTorch        | Training locale con EarlyStopping, metriche aggregate e campionamento quantili. |
| AutoML (AG-TS)  | AutoGluon TimeSeries     | Ensemble multi-modello, salvataggio artefatti e valutazione su holdout. L'app armonizza automaticamente i nomi dei quantili e replica `p50` se `p10/p90` non sono forniti. |

Le previsioni riportano sempre le colonne `p10`, `p50`, `p90`; l'app effettua
controlli e fallback su quantili mancanti.

## Debug e log

- Ogni modello utilizza il logger `k_forecast.app` (livello INFO di default).
- Attivando "DEBUG logs" nella sidebar si ottengono messaggi extra su split,
	parametri, leaderboard e metriche.
- Gli errori runtime (es. mancato training di un modello) vengono intercettati
	e mostrati in Streamlit senza interrompere la sessione.

## Suggerimenti di sviluppo

- Preferire Python ≤ 3.12 per compatibilità con AutoGluon/Chronos.
- Per aggiornare le dipendenze, valutare il pinning delle versioni in
	`requirements.txt`, soprattutto per ambienti di produzione.
- Lo script `generate_mock_bank_flows.py` è ideale per creare nuovi sandbox
	rapidi (modificare seed, numero mesi o serie).
- `utils/mock_data.py` contiene un placeholder da estendere se servono altri
	generatori custom.

## Troubleshooting

| Problema | Possibile causa / soluzione |
|----------|-----------------------------|
| `ModuleNotFoundError: tf_keras` durante Chronos | Usare Python ≤ 3.12. L'app imposta già `TRANSFORMERS_NO_TF=1`, ma su Python 3.13 la dipendenza non è installabile. |
| Errore "dataset missing columns" | Il CSV caricato non rispetta le colonne richieste (`item_id`, `timestamp`, `target`). Controllare header e formato timestamp. |
| Forecast vuoto o grafico mancante | Le serie selezionate non contengono dati nell'intervallo scelto: verificare che `item_id` esista sia nello storico che nelle previsioni. |
| UI non parte con `python app.py` | Assicurarsi che `streamlit` sia installato nell'ambiente attivo. In alternativa eseguire `streamlit run app.py`. |
| Messaggio "Quantili mancanti" nei log AutoML | Alcuni modelli AutoGluon potrebbero restituire solo `mean/median`. L'app duplica automaticamente `p50` sui quantili mancanti; il messaggio è informativo. |

Per segnalare bug o proporre miglioramenti, aprire una issue o PR indicando
ambiente, log e dataset utilizzato.
