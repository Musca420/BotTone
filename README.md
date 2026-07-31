# Adaptive Range Trading Bot

Bot deterministico e fail-closed per mean reversion adaptive-range. La Milestone 1 supporta
backtest QQQ a 15 minuti con broker simulato, senza leva e con massimo una posizione. Alpaca
Paper, shadow e paper operativo appartengono alla Milestone 2 e non sono ancora disponibili.

## Rischi e limitazioni

Il software non garantisce profitto né assenza di perdite. Un backtest OHLCV non ricostruisce
la sequenza intrabar: applica spread, slippage, volume disponibile, gap e l'ipotesi peggiore se
stop e target sono entrambi toccati. Dati incompleti o non UTC fermano il run. La modalità live
è disabilitata nel codice e non va usata come sistema operativo di trading.

## Installazione

Servono Python 3.12 e `uv`:

```powershell
uv sync --all-groups
uv run adaptive-bot --help
```

Per Docker:

```powershell
docker compose build
docker compose run --rm bot --help
```

## Configurazione e credenziali

`configs/backtest.yaml` contiene i limiti QQQ. Copiare `.env.example` in `.env` senza mai
versionarlo. Le chiavi Alpaca devono essere paper-only, senza permessi di prelievo e, quando
possibile, limitate per IP. `configs/alpaca_qqq_paper.yaml` resta `enabled: false` fino alla
Milestone 2. Nessun comando della Milestone 1 contatta Alpaca.

La modalità live richiederà insieme `TRADING_MODE=live` e
`ALLOW_LIVE_TRADING=I_ACKNOWLEDGE_THE_RISK`; oggi viene comunque rifiutata perché manca un
adapter live verificato.

## Dati

Il downloader Alpaca sarà introdotto nella Milestone 2. Per ora fornire un Parquet con colonne:
`timestamp`, `open`, `high`, `low`, `close`, `volume`. I timestamp rappresentano la chiusura
della candela, devono includere timezone e vengono normalizzati UTC. Per azioni/ETF usare dati
split-adjusted; salti oltre il 20% richiedono la colonna booleana `corporate_action`.

```powershell
uv run adaptive-bot validate-data --input data/raw/qqq_15m.parquet
```

## Test e backtest

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run adaptive-bot backtest --config configs/backtest.yaml
```

Il report JSON predefinito è `data/reports/backtest.json`: include PnL lordo/netto, fee,
slippage, drawdown, equity curve, segnali e fill. Non è ancora il report HTML completo previsto
per la Milestone 3 e non costituisce evidenza di live readiness.

## Paper, recovery ed emergenze

Paper e shadow non sono operativi in questa milestone. Il simulated broker è asincrono e usa
la stessa strategia broker-agnostic, ma non simula disponibilità o latenza delle API Alpaca.
Prima di ogni futuro riavvio paper il sistema dovrà riconciliare ordini e posizioni; la procedura
è descritta in `docs/runbook.md`.

In caso di stop mancante, posizione sconosciuta, perdita oltre soglia, dati stale o divergenza:
bloccare nuovi ordini, preservare/ripristinare la protezione, cancellare ordini non protettivi e
seguire il runbook. Il kill switch richiede reset manuale motivato.

Approfondimenti: `docs/architecture.md`, `docs/risk-model.md`, `docs/strategy.md`,
`docs/backtesting.md` e `docs/live-readiness-checklist.md`.
