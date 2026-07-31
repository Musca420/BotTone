# Progetto: Adaptive Range Trading Bot

Agisci come un senior quantitative developer, software architect e reliability engineer specializzato in sistemi di trading algoritmico.

Devi progettare e implementare un bot di trading modulare, testabile e sicuro per quanto tecnicamente possibile. “Sicuro” non significa privo di rischio: il sistema deve limitare rigorosamente il rischio, impedire liquidazioni, rifiutare operazioni non valide e restare inattivo quando non dispone di dati affidabili.

Non limitarti a produrre una descrizione. Crea il repository, implementa il codice, scrivi i test, esegui i test disponibili e documenta ciò che non hai potuto verificare.

## 1. Obiettivi iniziali

Implementa tre modalità operative:

1. `backtest`
2. `paper`
3. `live`

La modalità `live` deve essere disabilitata per impostazione predefinita.

Gli adapter iniziali saranno:

* Alpaca Paper, per QQQ e successivamente BTC/USD spot.
* OKX Demo, per BTC-USDT-SWAP.
* Interactive Brokers Paper, per MNQ, da implementare dopo il completamento dell’MVP Alpaca.

La prima milestone funzionante deve essere:

* strumento: QQQ;
* broker: Alpaca Paper;
* timeframe: 15 minuti;
* sessione: regular trading hours statunitensi;
* strategia: adaptive range mean reversion;
* nessuna leva;
* massimo una posizione aperta;
* rischio massimo per operazione: 0,25% dell’equity simulata.

Non implementare machine learning nella prima versione.

Non utilizzare un LLM per prendere decisioni operative. Tutte le decisioni di trading devono essere deterministiche, riproducibili e verificabili.

## 2. Stack tecnologico

Usa:

* Python 3.12 o versione stabile compatibile;
* `uv` per dipendenze e ambiente virtuale;
* `asyncio` per market data ed execution;
* `pydantic` e `pydantic-settings` per configurazione e validazione;
* `numpy` e `pandas` per indicatori e ricerca;
* Parquet per i dati storici;
* DuckDB per analisi e interrogazioni sui dati;
* SQLite per l’MVP operativo;
* PostgreSQL come opzione di produzione;
* `pytest`, `pytest-asyncio` e `hypothesis`;
* logging JSON strutturato;
* Docker e Docker Compose;
* type checking con `mypy` o `pyright`;
* linting e formatting con `ruff`.

Per prezzi, quantità monetarie, tick size e lot size non utilizzare floating point binario nel modulo di esecuzione. Usa `Decimal`.

## 3. Struttura del repository

Crea almeno questa struttura:

```text
adaptive-range-bot/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── configs/
│   ├── alpaca_qqq_paper.yaml
│   ├── okx_btc_demo.yaml
│   ├── ibkr_mnq_paper.yaml
│   └── backtest.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── src/
│   └── adaptive_bot/
│       ├── cli.py
│       ├── config.py
│       ├── clock.py
│       ├── domain/
│       │   ├── models.py
│       │   ├── events.py
│       │   ├── enums.py
│       │   └── exceptions.py
│       ├── data/
│       │   ├── interfaces.py
│       │   ├── validation.py
│       │   ├── resampling.py
│       │   └── repository.py
│       ├── indicators/
│       │   ├── atr.py
│       │   ├── adx.py
│       │   ├── vwap.py
│       │   ├── volatility.py
│       │   └── slope.py
│       ├── strategy/
│       │   ├── interfaces.py
│       │   ├── adaptive_range.py
│       │   ├── regime.py
│       │   └── signals.py
│       ├── risk/
│       │   ├── engine.py
│       │   ├── position_sizing.py
│       │   ├── liquidation.py
│       │   ├── limits.py
│       │   └── kill_switch.py
│       ├── execution/
│       │   ├── interfaces.py
│       │   ├── order_manager.py
│       │   ├── state_machine.py
│       │   ├── reconciliation.py
│       │   └── idempotency.py
│       ├── adapters/
│       │   ├── alpaca/
│       │   ├── okx/
│       │   ├── ibkr/
│       │   └── simulated/
│       ├── backtest/
│       │   ├── engine.py
│       │   ├── fills.py
│       │   ├── costs.py
│       │   ├── funding.py
│       │   ├── slippage.py
│       │   ├── metrics.py
│       │   └── walk_forward.py
│       ├── monitoring/
│       │   ├── health.py
│       │   ├── metrics.py
│       │   └── alerts.py
│       └── services/
│           ├── trading_service.py
│           ├── market_data_service.py
│           └── recovery_service.py
└── tests/
    ├── unit/
    ├── property/
    ├── integration/
    ├── replay/
    ├── chaos/
    └── fixtures/
```

## 4. Principi architetturali

La strategia non deve conoscere Alpaca, OKX o IBKR.

Definisci interfacce astratte per:

* `MarketDataProvider`;
* `Broker`;
* `InstrumentRepository`;
* `AccountProvider`;
* `Clock`;
* `EventStore`;
* `RiskEngine`.

La stessa implementazione della strategia deve essere utilizzata nel backtest, nel paper trading e nel live.

Non duplicare la logica della strategia tra backtest e trading reale.

Usa dependency injection.

Gli adapter dei broker devono convertire i dati specifici della piattaforma nei modelli interni comuni.

## 5. Modelli di dominio

Implementa modelli tipizzati per:

* Instrument;
* Candle;
* Quote;
* Trade;
* OrderBookSnapshot;
* Signal;
* OrderRequest;
* Order;
* Fill;
* Position;
* AccountSnapshot;
* RiskDecision;
* StrategyState;
* MarketRegime;
* HealthStatus.

Ogni evento deve contenere:

* timestamp exchange;
* timestamp di ricezione;
* source;
* instrument;
* sequence number, quando disponibile;
* correlation ID;
* schema version.

Tutti i timestamp interni devono essere UTC.

Per il Nasdaq utilizza un calendario di mercato affidabile e gestisci correttamente DST e festività. Non codificare manualmente date annuali.

## 6. Strategia Adaptive Range

### 6.1 Centro del range

Per QQQ e MNQ durante la sessione regolare:

```text
center = session VWAP
```

Per il mercato crypto 24/7:

```text
center = rolling VWAP delle ultime 96 candele da 15 minuti
```

Rendi configurabile la finestra.

### 6.2 Volatilità

Calcola:

```text
ATR = ATR(14)
```

Non utilizzare dati futuri.

Il range teorico è:

```text
lower_band = center - range_multiplier * ATR
upper_band = center + range_multiplier * ATR
```

Valore iniziale:

```text
range_multiplier = 2.0
```

La distanza normalizzata dal centro è:

```text
z = (close - center) / ATR
```

Gestisci esplicitamente ATR nullo o insufficiente.

### 6.3 Classificazione del regime

Implementa almeno:

* `RANGE`;
* `TREND_UP`;
* `TREND_DOWN`;
* `SHOCK`;
* `UNKNOWN`.

Parametri iniziali:

```text
ADX period = 14
RANGE se ADX < 20
TREND se ADX > 25
SHOCK se ATR percentile > 90
```

Aggiungi anche:

* pendenza normalizzata della EMA 50;
* variazione dell’ATR;
* spread corrente;
* quantità di dati mancanti;
* movimento cumulato delle ultime tre candele.

Il regime deve essere `UNKNOWN` se i dati sono insufficienti o non affidabili.

Non aprire operazioni in regime `UNKNOWN` o `SHOCK`.

Aggiungi isteresi per evitare passaggi continui tra regimi. Una nuova classificazione deve essere confermata per un numero configurabile di candele.

### 6.4 Segnali

Long iniziale:

```text
regime == RANGE
z <= -1.5
spread accettabile
nessuna posizione aperta
nessun cooldown
risk engine approva
```

Short iniziale:

```text
regime == RANGE
z >= +1.5
short consentito dallo strumento e dal broker
spread accettabile
nessuna posizione aperta
nessun cooldown
risk engine approva
```

Per l’MVP QQQ puoi abilitare soltanto il long. Lo short deve essere una configurazione separata.

### 6.5 Uscite

Take profit principale:

```text
ritorno verso center
```

Uscita parziale opzionale:

```text
50% della posizione quando abs(z) <= 0.5
resto quando z attraversa 0
```

Stop iniziale:

```text
2.5 ATR dall’ingresso
```

Time stop:

```text
8 candele da 15 minuti
```

Uscita immediata se:

* regime passa a SHOCK;
* integrità dati compromessa;
* rischio giornaliero superato;
* posizione reale non riconciliabile;
* stop protettivo assente;
* liquidation buffer non più valido.

Dopo l’ingresso, lo stop non può essere allontanato dal prezzo d’ingresso.

Può soltanto:

* restare invariato;
* essere stretto;
* essere portato a break-even secondo regole configurate.

Non implementare martingala.

Non implementare averaging down illimitato.

Non incrementare la posizione quando il trade è in perdita nella prima versione.

## 7. Risk engine

Il Risk Engine deve essere indipendente dalla strategia e deve poter rifiutare qualsiasi segnale.

Limiti iniziali:

```text
risk_per_trade = 0.25% equity
max_daily_loss = 1.0% equity iniziale della giornata
max_weekly_loss = 2.5%
max_strategy_drawdown = 8%
max_open_positions = 1
max_correlated_positions = 1
max_consecutive_losses = 3
cooldown_after_losses = 8 candele
```

Formula di sizing:

```text
risk_budget = account_equity * risk_per_trade

quantity =
risk_budget /
(abs(entry_price - stop_price) * point_value + estimated_costs)
```

La quantità deve essere arrotondata verso il basso rispettando:

* tick size;
* lot size;
* minimum quantity;
* minimum notional;
* buying power;
* broker limits.

Dopo l’arrotondamento, ricalcola il rischio effettivo.

Se il rischio effettivo supera il budget, rifiuta l’ordine.

Se la quantità minima dello strumento comporta un rischio eccessivo, non operare.

### 7.1 Liquidazione

Per strumenti a leva:

* utilizza mark price;
* recupera dall’exchange i dati reali della posizione e del margine;
* non applicare una formula universale hard-coded;
* crea comunque un estimatore indipendente per controllo e test;
* confronta stima interna e valore restituito dal broker;
* genera errore se la differenza supera una soglia configurabile.

Vincolo iniziale:

```text
distance_to_liquidation >= 3 * distance_to_stop
```

Aggiungi un buffer per:

* fee;
* funding;
* slippage;
* variazione del maintenance margin;
* latenza;
* gap;
* rounding.

Per il demo crypto:

```text
max leverage = 2x
margin mode = isolated, quando disponibile
```

Se non è possibile determinare in modo affidabile il prezzo di liquidazione, il Risk Engine deve rifiutare l’operazione.

Una liquidazione deve essere classificata come failure critica del sistema.

## 8. Kill switch

Implementa kill switch indipendenti per:

* perdita giornaliera;
* drawdown;
* dati stale;
* WebSocket disconnesso;
* REST API non disponibile;
* clock drift;
* troppe risposte 429;
* ordini duplicati;
* posizione sconosciuta;
* stop protettivo mancante;
* divergenza tra stato locale e broker;
* slippage estremo;
* spread estremo;
* frequenza operativa anomala;
* errore persistente del database.

Quando il kill switch si attiva:

1. impedisci nuovi ordini;
2. cancella gli ordini non protettivi;
3. conserva o ripristina gli stop protettivi;
4. chiudi la posizione soltanto quando la policy configurata lo richiede;
5. registra causa, stato e timestamp;
6. richiedi reset manuale.

Il bot non deve riattivarsi automaticamente dopo una violazione di rischio grave.

## 9. Order management

Implementa una state machine esplicita:

```text
CREATED
VALIDATED
SUBMITTED
ACKNOWLEDGED
PARTIALLY_FILLED
FILLED
CANCEL_REQUESTED
CANCELED
REJECTED
EXPIRED
UNKNOWN
```

Ogni ordine deve avere un client order ID deterministico e univoco.

Le richieste devono essere idempotenti.

Gestisci:

* partial fills;
* timeout;
* retry con exponential backoff;
* risposta ambigua;
* ordine accettato ma risposta REST persa;
* riavvio del processo;
* cancellazione concorrente;
* fill ricevuto durante la cancellazione.

Dopo ogni risposta ambigua, interroga il broker prima di ripetere un ordine.

Non inviare ciecamente lo stesso ordine dopo un timeout.

Gli stop protettivi devono essere `reduce-only` quando il broker lo supporta.

## 10. Credenziali e sicurezza

Non inserire segreti nel codice.

Usa variabili d’ambiente:

```text
TRADING_MODE
ALPACA_API_KEY
ALPACA_API_SECRET
OKX_API_KEY
OKX_API_SECRET
OKX_API_PASSPHRASE
IBKR_ACCOUNT_ID
DATABASE_URL
```

Crea `.env.example` senza valori reali.

Le API key live devono:

* essere separate dalle chiavi demo;
* avere solo permessi di trading necessari;
* non avere permessi di prelievo;
* essere limitate per IP quando supportato;
* essere periodicamente ruotate.

La modalità live deve richiedere contemporaneamente:

```text
TRADING_MODE=live
ALLOW_LIVE_TRADING=I_ACKNOWLEDGE_THE_RISK
```

Aggiungi inoltre:

* hard cap del notional;
* lista consentita di strumenti;
* lista consentita di account;
* controllo che nessuna chiave demo sia usata in live e viceversa.

Non stampare mai segreti nei log.

## 11. Adapter Alpaca

Usa l’SDK ufficiale Alpaca.

Implementa:

* download storico;
* streaming WebSocket;
* account snapshot;
* invio ordine;
* cancellazione ordine;
* lettura ordini;
* lettura posizioni;
* riconciliazione;
* paper mode obbligatorio per l’MVP.

Strumento MVP:

```text
QQQ
```

Sessione iniziale:

```text
09:30–16:00 America/New_York
```

Non operare nei primi 15 minuti della sessione nella configurazione iniziale.

Non mantenere posizioni overnight nell’MVP.

Chiudi eventuali posizioni prima della fine della sessione usando un orario configurabile.

## 12. Adapter OKX Demo

Implementa soltanto dopo il completamento dell’MVP Alpaca.

Usa le API REST e WebSocket documentate ufficialmente.

In modalità demo aggiungi l’indicazione richiesta per simulated trading.

Sottoscrivi almeno:

* ticker;
* trades;
* candles;
* mark price;
* order book;
* orders;
* fills;
* positions;
* account;
* position risk warning, se disponibile.

Strumento iniziale:

```text
BTC-USDT-SWAP
```

Recupera dinamicamente:

* tick size;
* lot size;
* contract value;
* leverage consentita;
* funding;
* mark price;
* index price;
* margin ratio;
* liquidation price;
* position mode;
* margin mode.

Non assumere che i metadati demo e live siano identici.

## 13. Adapter Interactive Brokers

Implementa come terza milestone.

Usa l’API ufficiale IBKR, TWS o IB Gateway.

Strumento iniziale:

```text
MNQ
```

Non hard-codificare la scadenza del future.

Implementa la selezione del contratto front-month secondo una policy esplicita.

Evita l’apertura di nuove posizioni durante la finestra di rollover configurata.

Gestisci:

* contract ID;
* multiplier;
* minimum tick;
* trading hours;
* liquid hours;
* scadenza;
* market data permissions;
* sessione API;
* pacing limits;
* riconnessione a TWS o IB Gateway.

Per la prima versione MNQ opera soltanto nella regular trading session definita dalla configurazione.

## 14. Backtester event-driven

Crea un backtester event-driven.

Non utilizzare semplicemente il prezzo di chiusura della candela come prezzo di esecuzione.

Il backtest deve:

* generare il segnale dopo la chiusura della candela;
* eseguire non prima dell’evento successivo disponibile;
* distinguere bid e ask quando disponibili;
* modellare spread;
* modellare slippage;
* modellare commissioni;
* modellare partial fills;
* modellare funding per perpetual;
* modellare tick e lot size;
* modellare latenze configurabili;
* modellare stop gap;
* modellare ordini non eseguiti;
* evitare qualsiasi look-ahead bias.

Se sono disponibili soltanto dati OHLCV, usa ipotesi conservative e documentale chiaramente.

Per una candela nella quale vengono toccati sia stop sia target e non è possibile conoscere l’ordine temporale, usa l’ipotesi peggiore per la posizione.

## 15. Validazione dei dati

Prima di un backtest verifica:

* timestamp duplicati;
* timestamp fuori ordine;
* candele mancanti;
* prezzi negativi o nulli;
* high inferiore a low;
* open o close fuori da high-low;
* volume negativo;
* timezone;
* cambio sessione;
* stock split e corporate action per ETF e azioni;
* contract roll per futures;
* salti anomali.

Il backtest deve interrompersi se la qualità dei dati non supera una soglia configurabile.

## 16. Test obbligatori

### Unit test

Scrivi test per:

* ATR;
* VWAP;
* ADX;
* z-score normalizzato;
* regime detection;
* sizing;
* arrotondamento quantità;
* fee;
* slippage;
* liquidation buffer;
* daily loss;
* state machine;
* time stop;
* trailing dello stop;
* divieto di allontanare lo stop.

### Property-based test

Verifica proprietà come:

* il rischio dopo rounding non supera il budget;
* la quantità non è negativa;
* uno stop long è inferiore all’ingresso;
* uno stop short è superiore all’ingresso;
* il kill switch impedisce sempre nuovi ordini;
* lo stesso evento non produce due ordini;
* un retry non duplica una posizione;
* lo stop non può aumentare il rischio;
* con equity zero non viene aperta alcuna posizione.

### Integration test

Usa mock server o fake broker per simulare:

* order accepted;
* order rejected;
* partial fill;
* timeout;
* 429;
* disconnessione;
* fill arrivato in ritardo;
* ordine sconosciuto;
* posizione modificata manualmente;
* dati stale;
* database momentaneamente indisponibile.

### Replay test

Registra sessioni demo e riproducile offline.

A parità di input, configurazione e versione del codice, il risultato deve essere deterministico.

### Chaos test

Durante il paper trading simula:

* riavvio del bot;
* interruzione della rete;
* duplicazione di un messaggio;
* perdita di messaggi WebSocket;
* dati fuori ordine;
* clock locale errato;
* latenza elevata;
* API REST indisponibile;
* database bloccato.

Il sistema deve ripartire riconciliando account, ordini e posizioni prima di consentire nuovi trade.

## 17. Walk-forward analysis

Non ottimizzare i parametri sull’intero dataset.

Implementa una walk-forward analysis configurabile.

Configurazione iniziale indicativa:

```text
training window = 12 settimane
validation window = 4 settimane
test window = 4 settimane
rolling step = 4 settimane
```

Mantieni completamente separato il test finale.

Per QQQ utilizza dati comprendenti differenti condizioni di mercato.

Per crypto includi:

* volatilità elevata;
* periodi laterali;
* trend rialzisti;
* trend ribassisti;
* flash movement;
* differenti condizioni di funding.

Non selezionare i parametri soltanto in base al profitto.

Valuta la stabilità dei parametri vicini. Se soltanto un valore molto preciso produce un risultato positivo, considera la strategia overfitted.

## 18. Metriche

Genera un report HTML e JSON con:

* rendimento netto;
* rendimento annualizzato;
* volatilità;
* Sharpe;
* Sortino;
* Calmar;
* max drawdown;
* durata del drawdown;
* profit factor;
* expectancy per trade;
* numero di trade;
* win rate;
* payoff ratio;
* average holding time;
* exposure;
* turnover;
* fee totali;
* slippage totale;
* funding totale;
* MAE;
* MFE;
* percentuale di ordini non eseguiti;
* partial fill rate;
* numero di kill switch;
* distanza minima dalla liquidazione;
* perdita massima giornaliera;
* performance per regime;
* performance per ora e giorno della settimana.

Mostra sempre risultati lordi e netti separatamente.

## 19. Stress test

Esegui automaticamente almeno questi scenari:

```text
commissioni normali
commissioni 2x
commissioni 3x
slippage normale
slippage 2x
slippage 3x
latenza 100 ms
latenza 500 ms
latenza 2 secondi
20% degli ordini limit non eseguiti
spread 2x
spread 3x
ATR shock
gap oltre lo stop
funding sfavorevole
WebSocket intermittente
```

Esegui anche bootstrap o Monte Carlo sulla sequenza dei trade.

Mostra distribuzione di:

* drawdown;
* rendimento;
* serie di perdite;
* probabilità di raggiungere il limite giornaliero;
* probabilità di rovina sotto le ipotesi definite.

## 20. Criteri minimi di accettazione

Questi sono gate ingegneristici, non garanzie di profitto.

La strategia non può passare al paper trading finché:

* tutti i test unitari e di integrazione non passano;
* non è stato rilevato look-ahead bias;
* il rischio massimo teorico è rispettato;
* non esiste un percorso che apra un ordine dopo il kill switch;
* restart e reconciliation funzionano.

Non può passare dal paper al piccolo live finché non presenta almeno:

```text
almeno 100 operazioni paper
almeno 300 operazioni complessive out-of-sample
expectancy out-of-sample positiva dopo i costi
profit factor out-of-sample >= 1.15
max drawdown <= 10%
nessuna liquidazione
nessuna violazione del risk budget
nessun ordine duplicato
nessuna posizione orfana
maggioranza delle finestre walk-forward positiva
risultato positivo o accettabile con costi 2x
```

Se una condizione non è soddisfatta, non aggirarla automaticamente modificando le soglie.

## 21. Paper trading

Nel paper trading registra:

* segnale teorico;
* prezzo teorico;
* ordine inviato;
* acknowledgment;
* fill;
* slippage;
* posizione;
* rischio previsto;
* rischio effettivo;
* motivo di ingresso;
* motivo di uscita;
* regime;
* configurazione;
* versione Git;
* latenza.

Confronta giornalmente:

* fill simulato;
* bid-ask disponibile;
* prezzo previsto;
* prezzo effettivo;
* backtest parallelo sulla stessa sessione.

Crea una modalità shadow nella quale il bot calcola le operazioni ma non invia ordini.

## 22. Live trading

Non attivare automaticamente il live.

Quando verrà implementato, applica inizialmente:

```text
risk_per_trade = 0.05%
max_daily_loss = 0.25%
max_open_positions = 1
minimum practical notional
```

Crea un hard cap monetario indipendente dall’equity restituita dal broker.

Non aumentare automaticamente il rischio in base ai profitti.

Ogni aumento deve richiedere una modifica manuale e versionata della configurazione.

## 23. CLI

Implementa comandi come:

```bash
uv run adaptive-bot download-data --config configs/alpaca_qqq_paper.yaml

uv run adaptive-bot validate-data \
  --input data/raw/qqq_15m.parquet

uv run adaptive-bot backtest \
  --config configs/backtest.yaml

uv run adaptive-bot walk-forward \
  --config configs/backtest.yaml

uv run adaptive-bot stress-test \
  --config configs/backtest.yaml

uv run adaptive-bot shadow \
  --config configs/alpaca_qqq_paper.yaml

uv run adaptive-bot paper \
  --config configs/alpaca_qqq_paper.yaml

uv run adaptive-bot reconcile \
  --config configs/alpaca_qqq_paper.yaml

uv run adaptive-bot report \
  --run-id RUN_ID
```

Il comando live deve rifiutarsi di partire senza tutti gli acknowledgement richiesti.

## 24. Osservabilità

Esponi metriche per:

* bot alive;
* WebSocket connected;
* last market event age;
* account equity;
* current exposure;
* realized PnL;
* unrealized PnL;
* daily PnL;
* drawdown;
* open orders;
* current position;
* order latency;
* API errors;
* rejected orders;
* current regime;
* current ATR;
* current z;
* spread;
* liquidation distance;
* risk usage.

Implementa alert per eventi critici.

Per l’MVP è sufficiente logging JSON e un health endpoint locale. Prepara le interfacce per Prometheus.

## 25. Documentazione

Il `README.md` deve spiegare:

* scopo del progetto;
* rischi e limitazioni;
* installazione;
* creazione ambiente;
* configurazione Alpaca Paper;
* download dati;
* esecuzione dei test;
* backtest;
* paper trading;
* recovery;
* interpretazione dei report;
* procedure di emergenza;
* differenze tra paper e live.

Crea inoltre:

```text
docs/architecture.md
docs/risk-model.md
docs/strategy.md
docs/backtesting.md
docs/runbook.md
docs/live-readiness-checklist.md
```

Il runbook deve includere procedure per:

* bot bloccato;
* ordine sconosciuto;
* posizione non riconciliata;
* API down;
* chiave compromessa;
* stop mancante;
* perdita oltre soglia;
* database corrotto;
* disattivazione definitiva.

## 26. Modalità di lavoro

Procedi per milestone:

### Milestone 1

* repository;
* modelli di dominio;
* indicatori;
* strategia;
* risk engine;
* simulated broker;
* backtester;
* unit test.

### Milestone 2

* Alpaca Paper;
* download storico;
* WebSocket;
* execution;
* reconciliation;
* paper mode;
* integration test.

### Milestone 3

* walk-forward;
* stress test;
* report;
* shadow mode;
* chaos test.

### Milestone 4

* OKX Demo;
* perpetual;
* mark price;
* funding;
* margin;
* liquidation buffer.

### Milestone 5

* IBKR Paper;
* MNQ;
* contract selection;
* rollover;
* session management.

Non iniziare la milestone successiva se i test della precedente non passano.

Alla fine di ogni milestone:

1. esegui formatter e linter;
2. esegui type checker;
3. esegui tutti i test;
4. mostra i comandi eseguiti;
5. riporta test passati e falliti;
6. descrivi le limitazioni residue;
7. aggiorna la documentazione;
8. crea un commit Git locale descrittivo, se il repository è configurato.

Quando una scelta API non è chiara, consulta esclusivamente la documentazione ufficiale corrente della piattaforma.

Non inventare endpoint o parametri.

Non inserire workaround che disabilitano i controlli di rischio per far passare i test.

Inizia ora dalla Milestone 1 e prosegui fino a ottenere una versione funzionante e testata dell’MVP Alpaca Paper.
