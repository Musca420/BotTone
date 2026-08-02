# Adaptive Range Trading Bot

Bot deterministico e fail-closed per mean reversion adaptive-range. La Milestone 1 supporta
backtest QQQ a 15 minuti con broker simulato, senza leva e con massimo una posizione. La Milestone
2 include dati, shadow mode e trading Alpaca Paper con denaro simulato, ordini bracket e
riconciliazione fail-closed. La modalità live resta disabilitata.

Il profilo crypto operativo corrente è esclusivamente BTCUSDT perpetual futures con margine USDT
isolated; gli adapter delle milestone precedenti restano disponibili soltanto come codice storico e
testato.

## Rischi e limitazioni

Il software non garantisce profitto né assenza di perdite. Un backtest OHLCV non ricostruisce
la sequenza intrabar: applica spread, slippage, volume disponibile, gap e l'ipotesi peggiore se
stop e target sono entrambi toccati. Dati incompleti o non UTC fermano il run. La modalità live
è disabilitata nel codice e non va usata come sistema operativo di trading.

La policy di rischio corrente limita ogni trade all'1% dell'equity, blocca nuovi ingressi dopo una
perdita giornaliera del 2% o settimanale del 10% e mantiene il drawdown massimo all'8%. Il blocco è
persistente e richiede reset manuale motivato; non forza da solo la chiusura di posizioni protette.

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
decisioni, fill, kill switch e avanzamento del progetto. Mostra inoltre il feed pubblico BTCUSDT
raccolto in tempo reale, il grafico delle barre chiuse, l'età del feed e il warm-up degli indicatori.
L'interfaccia è in inglese e si aggiorna ogni due secondi:

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

Per eseguire la strategia con il saldo fittizio dell'account Alpaca Paper:

```powershell
uv run adaptive-bot paper --config configs/alpaca_qqq_paper.yaml
uv run adaptive-bot dashboard --report data/reports/paper.json
```

Il comando invia soltanto ordini all'endpoint paper. Ogni ingresso è un bracket atomico con target
e stop; quantità e rischio sono ricalcolati sul saldo paper. Dati stale, spread eccessivo, account
non autorizzato, posizione sconosciuta o stop mancante bloccano nuovi ingressi. In caso di stop
mancante o perdita oltre soglia viene richiesta la chiusura paper cancellando prima gli ordini.

Al riavvio, una posizione o un ordine non associabile allo stato locale blocca il processo invece
di essere adottato automaticamente: verificare il portale Alpaca ed eseguire `reconcile`. Le
procedure di emergenza sono in `docs/runbook.md`.

Dopo una verifica manuale sul portale paper, lo stato broker può essere adottato esplicitamente:

```powershell
uv run adaptive-bot reconcile --config configs/alpaca_qqq_paper.yaml --accept-broker-state
```

Una posizione priva di stop protettivo non può essere adottata. Lo stato riconciliato è persistito
in SQLite e viene verificato nuovamente prima di consentire nuovi ordini. Anche i riferimenti di
equity giornaliera/settimanale e il kill switch sono persistenti: un riavvio non azzera i limiti.

In caso di stop mancante, posizione sconosciuta, perdita oltre soglia, dati stale o divergenza:
bloccare nuovi ordini, preservare/ripristinare la protezione, cancellare ordini non protettivi e
seguire il runbook. Il kill switch richiede reset manuale motivato.

## Bitunix futures con denaro simulato

Il solo mercato operativo crypto è BTCUSDT perpetual futures, margine isolated USDT. Usa dati
storici pubblici Bitunix e il broker simulato locale: non servono API key e nessun ordine raggiunge
l'exchange.

```powershell
uv run adaptive-bot download-data --config configs/bitunix_btc_futures_simulated.yaml `
  --output data/raw/bitunix_btcusdt_futures_5m.parquet
uv run adaptive-bot backtest --config configs/bitunix_btc_futures_simulated.yaml
```

Il collector autonomo archivia per sette giorni le candele pubbliche chiuse da 5 minuti in JSONL,
deduplicandole a ogni riavvio. Il file grezzo viene poi convertito e validato prima del backtest:

```powershell
uv run adaptive-bot collect-bitunix --config configs/bitunix_btc_futures_simulated.yaml `
  --output data/raw/bitunix_btcusdt_mark_futures_5m.jsonl --duration-hours 168 --poll-seconds 60
uv run adaptive-bot paper-bitunix --config configs/bitunix_btc_futures_simulated.yaml `
  --input data/raw/bitunix_btcusdt_mark_futures_5m.jsonl `
  --output data/reports/bitunix_paper.json --duration-hours 168
```

Il paper engine persiste il timestamp di avvio, usa lo storico precedente soltanto come warm-up e
può simulare ordini esclusivamente sulle candele successive. La dashboard deve leggere
`data/reports/bitunix_paper.json` per mostrare decisioni e operazioni BTCUSDT.

L'esecuzione privata Bitunix resta disabilitata finché il testnet non dispone di endpoint ufficiali
verificati. I limiti strumento locali sono conservativi e dovranno essere confrontati con i metadati
pubblici correnti prima di una futura modalità paper collegata all'exchange. Il downloader rifiuta
deviazioni OHLC superiori a 1 bps; entro tale soglia il dataset processato espande conservativamente
high/low per includere open e close, registra il conteggio e conserva immutato il raw originale.

La simulazione usa candele da 5 minuti e VWAP rolling su 288 barre (24 ore), leva 10×,
esposizione massima del 20% dell'equity e quindi margine previsto del
2%, sotto il cap del 10%. Stop e target sono simmetrici all'1% del prezzo, equivalenti a circa
−10%/+10% ROE prima di fee e slippage.

Approfondimenti: `docs/architecture.md`, `docs/risk-model.md`, `docs/strategy.md`,
`docs/backtesting.md` e `docs/live-readiness-checklist.md`.

## Meme Futures Lab — Bitunix paper 24/7

Il secondo bot usa lo stesso core di dominio e rischio ma possiede configurazione, dati, report,
equity simulata e dashboard indipendenti. Il bot BTC non viene riconfigurato. L'universo è
l'intersezione tra perpetual USDT Bitunix e categoria `meme-token` CoinGecko; ticker ambigui,
stream stale, listing con meno di sette giorni, spread, depth, funding o mark divergence fuori
soglia vengono esclusi.

La strategia deterministica abilita long breakout/pullback nei regimi rialzisti e short breakdown
nei regimi distribution/ribassisti. Usa momentum normalizzato ATR, volume, liquidità, spread,
funding, mark/index divergence e manipulation score. Il conto paper parte da 100 USDT: rischio
normale 0,25 USDT, anticipato 0,125 USDT, cap assoluto 0,30 USDT, notional massimo 40 USDT,
margine massimo 20 USDT e leva minima necessaria fino a 3× isolated. I limiti sono 1,5%
giornaliero, 4% settimanale, 8% drawdown, due posizioni e cooldown di otto barre dopo tre perdite.

Prima dell'avvio autenticare una volta la CLI Codex con l'abbonamento ChatGPT e scaricare lo storico
in uno script separato (può restare in esecuzione mentre il collector lavora):

```powershell
codex.cmd login
uv run adaptive-bot meme-download-history --config configs/bitunix_meme_paper.yaml --weeks 52
```

Avviare quattro terminali:

```powershell
uv run adaptive-bot meme-collect --config configs/bitunix_meme_paper.yaml --duration-hours 168
uv run adaptive-bot meme-luna-sidecar --config configs/bitunix_meme_paper.yaml --duration-hours 168
uv run adaptive-bot meme-paper --config configs/bitunix_meme_paper.yaml --duration-hours 168
uv run adaptive-bot meme-dashboard --config configs/bitunix_meme_paper.yaml
```

Aprire `http://127.0.0.1:8081`. L'interfaccia, interamente in inglese, mostra feed, scanner,
motivazioni di esclusione, liquidity/manipulation score, policy Luna, posizione, operazioni, equity
e audit. Con Tailscale Serve si può
pubblicare la porta come percorso `/meme`, mantenendo la dashboard BTC sulla porta 8080.

Il recorder usa REST per il bootstrap storico e WebSocket pubblici per kline 5m/1h, trade, book e
mark/index/funding. Per costruire il dataset shadow:

```powershell
uv run adaptive-bot meme-build-dataset --config configs/bitunix_meme_paper.yaml
uv run adaptive-bot meme-backtest --config configs/bitunix_meme_paper.yaml
```

Le triple-barrier label e le feature vengono generate offline. I modelli restano shadow-only:
servono almeno 20 settimane, 1.000 setup e dieci simboli prima di poter passare da
`paper_bootstrap` a `paper_validated`. Open interest, liquidazioni, social e on-chain restano
`UNKNOWN` finché
non viene scelto e verificato un provider ufficiale.

Il paper usa dati pubblici Bitunix reali e simula localmente ordini, costi, stop, target ed equity.
L'esecuzione privata Bitunix con denaro reale resta bloccata anche se vengono fornite credenziali.

Luna Max usa `codex exec` con ricerca web e produce una Market Policy valida al massimo sei ore,
usando soltanto domini autorizzati. Luna Low non usa il web e revisiona ogni setup. Entrambi possono
solo bloccare o ridurre il rischio: senza login, policy, fonti ammesse o review valida il bot resta
fail-closed. Codex opera in una directory temporanea con unicamente snapshot sanitizzato e schema;
non riceve credenziali, repository o facoltà di inviare ordini.
