# K-Forecast

K-Forecast è il cruscotto che mette in mano ai soci una vista unica sulle
previsioni di flussi, vendite e scenari promozionali. Una singola interfaccia,
tre motori previsivi e un linguaggio visivo in linea con l'identità Keytech.

> **A chi è rivolto**: board e stakeholder che vogliono valutare in pochi minuti
> la maturità della soluzione, le aree di applicazione e le evidenze ottenibili.

## Cosa offre in pratica

- **Un'unica cabina di regia**: scegliere il motore (Chronos-Bolt, DeepAR,
  AutoML) e confrontare i risultati fianco-a-fianco senza cambiare schermata.
- **Focus su business KPI**: la dashboard mette al centro _cash flow_,
  vendite promo e scenari what-if, con metriche di accuratezza leggibili.
- **Esperienza brandizzata**: layout dark in linea con il brand Keytech
  (nuova versione con stile separato in `assets/app.css`).
- **Accesso guidato**: il percorso “Seleziona dataset → Scegli modello → Esegui
  Forecast” mantiene il controllo in tre click.

## Come leggere il menu

1. **Sidebar** – selezione del motore, orizzonte (7/30/90 giorni) e origine
   dati (mock pronto o upload proprietario).
2. **Vista principale** – grafico interattivo con intervalli di confidenza,
   callout sugli eventi (promo, anomalie) e pulsanti di export.
3. **Tab "Confronto Modelli"** – accumula gli ultimi run per valutare gap fra
   approcci, con badge che evidenziano il best performer.

## Use case da mostrare in demo

| Linea | Situazione | Valore per il socio |
|-------|------------|---------------------|
| Tesoreria | Proiezione flussi di cassa mensili multi-filiale | Anticipa tensioni di liquidità e pianifica coperture |
| Retail promo | Vendite SKU × Cliente con calendario promozioni | Confronta uplift promo e identifica saturazione promozionale |
| Scenario planning | Mix modelli per stress test orizzonte 90 giorni | Mettere a confronto modelli e scegliere quello più aderente al rischio |

Durante la demo suggeriamo di partire dal caso **Retail promo** (dataset SD) per
mostrare bande promozionali e badge di evidenza.

## Esperienza attesa in sala

- Il grafico reagisce in tempo reale al cambio modello e mantiene la storia
  nella tab di confronto.
- Le metriche principali (MAPE, sMAPE, Pinball Loss) sono spiegate con tooltip
  comprensibili.
- Il pulsante "Scarica previsioni" abilita l'approfondimento offline.

## Come organizzare la prova live

1. Avviare l'app con `python app.py` (streamlit si apre da solo in browser).
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

- **Referente Prodotto**: Alessia Rossi – alessia.rossi@keytech.it
- **Team Data & AI**: data.ai@keytech.it

Per feedback e priorità aggiuntive scrivere al team o aprire una issue
indicando contesto e aspettative della demo.
