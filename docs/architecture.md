# Architettura — Milestone 1

Il flusso è unidirezionale:

```text
Candle validata -> indicatori -> regime -> strategia -> Risk Engine
                 -> Order Manager -> Broker simulato -> Fill/Position/EventStore
```

La strategia riceve soltanto modelli interni e non importa adapter. `MarketDataProvider`,
`Broker`, `InstrumentRepository`, `AccountProvider`, `Clock`, `EventStore` e `RiskEngine` sono
contratti tipizzati iniettati dall'orchestratore. Backtest e futuri adapter condividono la stessa
`AdaptiveRangeStrategy`.

I modelli Pydantic rifiutano campi inattesi. Eventi e clock sono UTC; prezzi e quantità che
attraversano execution/risk usano `Decimal`. Pandas/NumPy sono confinati agli indicatori.

SQLite conserva eventi idempotenti con primary key; Parquet conserva dati storici e DuckDB li
interroga. La Milestone 1 non include networking, reconciliation completa o PostgreSQL.

La dashboard operativa legge soltanto il report JSON prodotto dal backtester. Il server usa la
standard library, accetta connessioni esclusivamente dal loopback locale e non espone comandi di
trading. Il browser aggiorna telemetria, calcoli e decisioni ogni due secondi.

L'adapter Alpaca Paper converte barre, quote, account, ordini e posizioni nei modelli comuni. Le
API sincrone dell'SDK ufficiale sono isolate con `asyncio.to_thread`; dopo un timeout ambiguo
l'ordine viene cercato tramite client order ID e non viene reinviato alla cieca. Shadow mode
aggrega le barre minute a 15 minuti e riusa il backtester senza inviare ordini.
Il WebSocket account converte ogni trading update in aggiornamenti `Order` e, per fill o partial
fill, in eventi `Fill` distinti destinati allo store e alla riconciliazione.

Il runtime paper aggrega le barre minute soltanto dopo la chiusura del bucket, calcola indicatori e
regime, interroga account/posizioni/ordini Alpaca e sottopone ogni entry al Risk Engine. Gli ingressi
approvati usano bracket atomici; il kill switch impedisce nuove entry e richiede flatten paper per
perdita oltre soglia o stop protettivo assente. I report paper usano lo stesso schema della dashboard.

Bitunix è limitato a BTCUSDT perpetual futures con margine isolated USDT. Il calendario NYSE non
viene applicato alle candele crypto 24/7. Non esiste un percorso dal runtime agli endpoint privati
Bitunix: tutti gli ordini restano nel broker simulato locale.

Gli ordini seguono una state machine esplicita. Il client order ID è un hash deterministico di
strumento, azione, timestamp, correlation ID e scopo. Un retry restituisce l'ordine esistente.
# Meme application boundary

`adaptive_bot.meme` è una seconda applicazione nello stesso pacchetto. Riusa modelli di dominio,
indicatori e vincoli monetari, ma non condivide database, capitale, configurazione o dashboard con
BTC. Il flusso è `CoinGecko category ∩ Bitunix USDT futures → REST/WebSocket recorder → scanner →
strategy experts → quantitative checks → Luna policy/review → risk sizing → paper ledger → dashboard`.

Policy, richieste Luna Low e review sono file JSON atomici e persistenti. Il cambio della policy o
l'arrivo di una review fanno parte della firma osservata dal paper engine, quindi provocano
immediatamente un replay deterministico senza attendere un'altra modifica dei dati di mercato.
Il sidecar richiede inoltre una nuova policy quando almeno un simbolo passa a `ELIGIBLE`; gli stati
`ELIGIBLE_REDUCED` non attivano il refresh. Un cooldown di 15 minuti e il limite giornaliero
impediscono rivalutazioni ripetute quando lo scanner oscilla attorno alle soglie.

I raw event vengono conservati in `data/meme/raw`, le feature Parquet in `data/meme/processed` e i
report in `data/meme/reports`. Lo scanner è fail-closed: assenza catalogo, stream stale o metadata
ambigui non producono un universo alternativo implicito.

Luna è un sidecar host separato dai container. `codex exec` riceve soltanto JSON sanitizzato in una
directory temporanea read-only. Luna Max può consultare fonti web allowlisted e promuove una policy
solo dopo validazione Pydantic; Luna Low opera senza web su una coda di setup deterministica. Il
runtime legge esclusivamente artefatti validati e resta fermo se il sidecar o l'autenticazione non
sono disponibili. Il sidecar non importa né possiede un adapter di execution.
