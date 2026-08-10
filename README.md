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
`data/reports/bitunix_paper.json` per mostrare decisioni e operazioni BTCUSDT. Ogni candela reale
alimenta otto portafogli simulati indipendenti: cinque con soglia range ADX 20, 22, 23, 24 e 25,
`MR SCORE v1`, `MR SCORE v1.1` e `MR SCORE v2`. V1.1 conserva l'ingresso rapido di v1 ma
evita il counter-trend, prende profitto vicino al VWAP e applica invalidazione e cooldown. V2
richiede tre candele di rientro, opera soltanto con
`1,5 <= abs(Z) <= 4`, evita ingressi contro trend e usa soglia score 0,70. Il selettore
`Strategy profile` cambia tra equity, PnL, posizione, score e operazioni degli otto report.

L'esecuzione privata Bitunix resta disabilitata finché il testnet non dispone di endpoint ufficiali
verificati. I limiti strumento locali sono conservativi e dovranno essere confrontati con i metadati
pubblici correnti prima di una futura modalità paper collegata all'exchange. Il downloader rifiuta
deviazioni OHLC superiori a 100 bps; entro tale soglia il dataset processato espande conservativamente
high/low per includere open e close, registra il conteggio e conserva immutato il raw originale.

La simulazione usa candele da 5 minuti e VWAP rolling su 288 barre (24 ore), leva 10×,
esposizione massima del 20% dell'equity e quindi margine previsto del
2%, sotto il cap del 10%. Stop e target sono simmetrici all'1% del prezzo, equivalenti a circa
−10%/+10% ROE prima di fee e slippage.

Approfondimenti: `docs/architecture.md`, `docs/risk-model.md`, `docs/strategy.md`,
`docs/backtesting.md` e `docs/live-readiness-checklist.md`.

## Ricerca deterministica dei parametri

Il laboratorio research e separato dagli otto profili paper: usa sempre la stessa
`AdaptiveRangeStrategy` della V1.1 (`weighted_reversion_v11`), soltanto il
broker simulato e non puo inviare ordini, cambiare configurazioni operative o promuovere da solo un
candidato. Rischio per trade 1%, leva 10x, BTCUSDT e timeframe 5 minuti restano fissi.

L'inizializzazione scarica 12 mesi di candele Bitunix `MARK_PRICE`, valida il Parquet, valuta 60
combinazioni V1.1 core e un campione deterministico di 200 combinazioni V1.1 ampie, quindi salva
ogni tentativo e gli eventuali champion in DuckDB. La selezione applica CSCV, mostra la probabilita
di backtest overfitting (PBO) e calcola il Deflated Sharpe Ratio (DSR) tenendo conto di tutti i
tentativi. Gli orizzonti recenti 1d/7d/30d sono diagnostici e non partecipano alla classifica:

```powershell
uv run adaptive-bot research-init --config configs/bitunix_btc_futures_simulated.yaml
uv run adaptive-bot research-status --config configs/bitunix_btc_futures_simulated.yaml
uv run adaptive-bot research-worker --config configs/bitunix_btc_futures_simulated.yaml
uv run adaptive-bot research-pin --config configs/bitunix_btc_futures_simulated.yaml CANDIDATE_ID
```

Per prove rapide locali usare `research-init --skip-download --core-only --candidate-limit 3`. Il
worker aggiorna dati e shadow dei migliori candidati ogni cinque minuti, esegue il core ogni notte
alle 00:15 UTC e include la ricerca ampia la domenica. La dashboard
espone la classifica read-only con filtri per famiglia, stato e orizzonte 1d/7d/30d/all. Un candidato
resta `insufficient` sotto 30 trade OOS, `provisional` tra 30 e 299 e puo diventare `validated` da
300 trade soltanto se supera anche expectancy, profit factor, drawdown, costi 2x, maggioranza delle
finestre, sicurezza, stabilita del vicinato, DSR >= 95% e PBO <= 20%. Queste ultime due soglie sono
gate prudenziali del progetto, non garanzie di profitto futuro.

## Machine learning Adaptive Range (solo ricerca/shadow)

Il protocollo `scientific_v2` separa long e short e confronta timeframe 5m/15m/30m, VWAP rolling,
ATR, ADX, regime, conferma, stop, target, time stop e cooldown. La ricerca usa 3.000 strategie,
porta 120 configurazioni alla validazione completa e applica Logistic Regression e XGBoost CUDA ai
12 finalisti. Il modello puo soltanto rifiutare un segnale deterministico: rischio 1%, leva 10x,
sizing e ordini non sono apprendibili.

L'archivio usa candele 1m `LAST_PRICE`/`MARK_PRICE`, volumi e funding osservati dalle REST ufficiali
Bitunix. I timeframe superiori sono aggregati soltanto dopo la chiusura. Funding assente e spread
storico restano `unavailable`; non vengono stimati. La selezione usa walk-forward cronologico
purgato, PBO, Deflated Sharpe, bootstrap e costi 1x/2x/3x. Il holdout 5 maggioâ€“3 agosto 2025 resta
sigillato fino al comando esplicito `ml-finalize` e puo essere aperto una sola volta per run.

Protocollo corrente: nested purged walk-forward, Reality Check e replay event-driven; il holdout è
la coda più recente di 12 settimane, determinata automaticamente prima della selezione.

```powershell
uv sync --extra gpu
uv run adaptive-bot ml-download-data --config configs/bitunix_btc_futures_simulated.yaml --all-available
uv run adaptive-bot collect-bitunix-microstructure --config configs/bitunix_btc_futures_simulated.yaml --duration-hours 168
uv run adaptive-bot ml-research --config configs/bitunix_btc_futures_simulated.yaml
uv run adaptive-bot ml-status --config configs/bitunix_btc_futures_simulated.yaml --watch
uv run adaptive-bot ml-research --config configs/bitunix_btc_futures_simulated.yaml --resume
uv run adaptive-bot ml-finalize --config configs/bitunix_btc_futures_simulated.yaml --run-id RUN_ID --open-holdout
```

Il candidato, anche se approvato, resta sotto `data/models/candidates/RUN_ID/` e non sostituisce i
profili paper. Dettagli e gate: `docs/ml-research-protocol.md`.

### Multi-Expert v5

La pipeline v5 importa le 2.847 configurazioni valide congelate dallo screen scientifico, le
separa LONG/SHORT e aggiunge 3.072 azioni preregistrate (8.766 al massimo). Costruisce esiti
controfattuali dal minuto successivo, seleziona al massimo 24 esperti sul solo train e stima EV
netto con due ensemble XGBoost CUDA calibrati; `FLAT=0` prevale se EV o limite inferiore 95% non
sono positivi. Il bundle non viene mai promosso automaticamente a paper/live.

```powershell
uv run adaptive-bot ml-expert-train --config configs/bitunix_btc_futures_simulated.yaml
uv run adaptive-bot ml-expert-status --config configs/bitunix_btc_futures_simulated.yaml --watch
uv run adaptive-bot ml-expert-train --config configs/bitunix_btc_futures_simulated.yaml --resume
uv run adaptive-bot ml-expert-finalize --config configs/bitunix_btc_futures_simulated.yaml --run-id RUN_ID --open-holdout
```

Il protocollo maker-first V6 e il comando di raccolta BTC/ETH sono documentati in
[`docs/expert-policy-v6.md`](docs/expert-policy-v6.md). V6 richiede otto settimane reali prima del
fit di sviluppo e mantiene sigillate le quattro settimane finali.

Il percorso ML corrente è V8 Hybrid: Alpha trasferibile con audit leave-one-exchange-out e
Execution calibrata esclusivamente su Bitunix, documentati in
[`docs/hybrid-policy-v8.md`](docs/hybrid-policy-v8.md). V7 resta diagnostica; nessun bundle V8 può
attivare automaticamente denaro reale.

Matrice, bundle e report sono rispettivamente in `data/ml/counterfactual_v5/`,
`data/models/expert_policy/` e `data/reports/ml_expert_research_v5.json`. Il protocollo è
descritto in `docs/expert-policy-training-v5.md`. L'apertura holdout è
atomica e monouso; un verdetto `NO_DEPLOYABLE_POLICY` la lascia sigillata.

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

Per richiedere esplicitamente una nuova policy senza cancellare file:

```powershell
uv run adaptive-bot meme-luna-sidecar --config configs/bitunix_meme_paper.yaml --once --refresh-max
```

I setup inviati a Luna Low restano nella coda persistente e il paper engine si aggiorna appena
compare la review; la dashboard mostra quante revisioni sono ancora in attesa.

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
