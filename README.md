# K-Forecast

K-Forecast è il cruscotto che mette in mano ai soci una vista unica sulle
previsioni operative AP/AR e sul carico dei team di tesoreria. Una singola
interfaccia, tre motori previsivi specializzati e un linguaggio visivo in linea
con l'identità Keytech.

> **A chi è rivolto**: board, responsabili finance e stakeholder che vogliono
> valutare rapidamente la maturità della soluzione, il valore sui processi SAP e
> le evidenze ottenibili.

## Cosa offre in pratica

- **Tre motori mirati**: DeepAR (F110) per i run di pagamento, DeepAR (SD) per
  la domanda commerciale SKU×Cliente e ARIMA/SARIMA (EBS) per il workload di
  riconciliazione bancaria.
- **Focus su business KPI**: KPI di accuratezza tradotti in metri di sala
  (MAPE run-day, MAE per il workload, coverage p10–p90) con tooltip che li
  riportano al contesto SAP.
- **Esperienza brandizzata**: layout dark in linea con il brand Keytech (tema
  separato in `assets/app.css`).
- **Accesso guidato**: percorso in tre step “Seleziona dataset → Scegli modello
  → Esegui Forecast” che mantiene il racconto sotto controllo.

## Come leggere il menu

1. **Sidebar** – selezione del motore, orizzonte (7/30/90 giorni) e origine
   dati (mock dedicato, mock alternativo o upload proprietario).
2. **Vista principale** – grafico interattivo con intervalli di confidenza,
   callout sugli eventi (promo, run programmati) e pulsanti di export.
3. **Tab "Confronto Modelli"** – confronta gli ultimi run per misurare gap fra
   approcci, con badge che evidenziano il best performer.

## Use case da mostrare in demo

| Linea | Modello | Situazione | Valore per il socio |
|-------|---------|------------|---------------------|
| Accounts Payable | DeepAR (F110) | Cash-out per run martedì/venerdì, covariate run_flag e volume_due_next_7d | Anticipa tensioni di liquidità e supporta le decisioni di copertura |
| Sales & Distribution | DeepAR (SD) | Vendite giornaliere SKU×Cliente, promo flag e price index | Misura uplift promozionale e valuta saturazione commerciale |
| Bank Operations | ARIMA/SARIMA (EBS) | Conteggio giornaliero FEBEP/FEBKO per banca | Pianifica la capacità operativa sui team EBS |

Durante la demo consigliamo di aprire con **DeepAR (F110)** per l'impatto
sulla tesoreria, proseguire con **DeepAR (SD)** per mostrare bande promo e
chiudere con **EBS** per la vista Ops.

### Mock e script rapidi

- `data/mock_f110_payouts.csv` · generato da `scripts/generate_mock_f110_payouts.py`
- `data/mock_sd_sales.csv` · generato da `scripts/generate_mock_sd_sales.py`
- `data/mock_ebs_counts.csv` · generato da `scripts/generate_mock_ebs_counts.py`

Gli script producono dataset già pronti all'uso (timestamp ISO, item_id, target
e covariate necessarie) e possono essere rigenerati con `python scripts/<nome_script>.py`.

### SAP cheat sheet

- **F110** → tabelle BSIK/BSAK, BSEG (campi ZFBDT, ZTERM, NETDT/SK1DT/SK2DT), BKPF, T001
- **SD** → anagrafiche da VBAK/VBAP, condizioni di prezzo da KONV, calendario promo da
  tavole di customizing
- **EBS** → FEBKO/FEBEP con ribaltamento giornaliero degli statement

## Esperienza attesa in sala

- I grafici reagiscono in tempo reale al cambio modello e mantengono la storia
  nella tab di confronto.
- Le metriche principali mostrano MAPE run-day o MAE a seconda del modello e la
  coverage p10–p90.
- Il pulsante "Scarica previsioni" abilita l'approfondimento offline.

## Come organizzare la prova live

1. Avviare l'app con `python app.py` (Streamlit si apre da solo in browser).
2. Presentare la sidebar come “pannello di controllo”.
3. Alternare mock già forniti e file proprietari per evidenziare la flessibilità.
4. Chiudere con un export CSV delle previsioni e proposta di follow-up.

Per i dettagli tecnici completi (setup ambienti, architettura, troubleshooting)
consultare il documento [`README-tech.md`](README-tech.md).

## Roadmap breve

- Collegamento a fonti dati esterne (ERP/BI) via API.
- Scenario planning con shock macro e stress test automatici.
- Salvataggio e confronto versioni forecast nel tempo.

## Contatti

- **Referente Prodotto**: adrian.iancu@keytech.srl
- **TeamLeader FullStack AI**: sandro.rossano@keytech.srl

Per feedback e priorità aggiuntive scrivere al team o aprire una issue
indicando contesto e aspettative della demo.
