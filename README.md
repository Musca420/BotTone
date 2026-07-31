# Adaptive Range Trading Bot

Bot deterministico e fail-closed per mean reversion adaptive-range. La Milestone 1 supporta
backtest QQQ a 15 minuti con broker simulato, senza leva e con massimo una posizione. La Milestone
2 include ora dati, shadow mode, account, ordini bracket e riconciliazione Alpaca Paper; il loop di
invio paper automatico resta intenzionalmente bloccato fino al completamento dei test di recovery.

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

## Dashboard operativa

La dashboard locale read-only mostra equity, drawdown, regime, ATR, ADX, VWAP, z-score, bande,
decisioni, fill, kill switch e avanzamento del progetto. L'interfaccia è in inglese e rilegge il
report ogni due secondi:

```powershell
uv run adaptive-bot dashboard --report data/reports/backtest.json
```

Aprire `http://127.0.0.1:8080`. Il server rifiuta bind non-loopback per non pubblicare telemetria
operativa senza autenticazione.

## Alpaca Paper e shadow mode

La configurazione `configs/alpaca_qqq_paper.yaml` accetta esclusivamente Alpaca Paper. Impostare
le credenziali e l'account senza inserirli nei file versionati:

```powershell
$env:ALPACA_API_KEY="..."
$env:ALPACA_API_SECRET="..."
$env:ALPACA_ACCOUNT_ID="..."
```

Scaricare e validare i dati QQQ corretti per split:

```powershell
uv run adaptive-bot download-data --config configs/alpaca_qqq_paper.yaml
```

Prima di ogni sessione verificare account e posizioni:

```powershell
uv run adaptive-bot reconcile --config configs/alpaca_qqq_paper.yaml
```

Shadow mode riceve barre e quote ma non invia ordini. Il report è leggibile dalla dashboard:

```powershell
uv run adaptive-bot shadow --config configs/alpaca_qqq_paper.yaml
uv run adaptive-bot dashboard --report data/reports/shadow.json
```

L'invio paper automatico resta bloccato finché trading update, partial fill, cancellazione
concorrente e recovery non superano gli integration test. Le emergenze sono in `docs/runbook.md`.

In caso di stop mancante, posizione sconosciuta, perdita oltre soglia, dati stale o divergenza:
bloccare nuovi ordini, preservare/ripristinare la protezione, cancellare ordini non protettivi e
seguire il runbook. Il kill switch richiede reset manuale motivato.

Approfondimenti: `docs/architecture.md`, `docs/risk-model.md`, `docs/strategy.md`,
`docs/backtesting.md` e `docs/live-readiness-checklist.md`.
