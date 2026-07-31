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

Gli ordini seguono una state machine esplicita. Il client order ID è un hash deterministico di
strumento, azione, timestamp, correlation ID e scopo. Un retry restituisce l'ordine esistente.
