# SPECIFICA PER LA REVISIONE DEL BOT BTC/USDT VWAP / ANCHORED VWAP

L'obiettivo **non** è riscrivere completamente il bot, perché la logica direzionale attuale sembra già avere un certo valore. Il problema principale osservato nel paper trading è che il sistema spesso identifica correttamente o quasi correttamente la direzione del prezzo, ma apre trade su movimenti il cui potenziale è troppo piccolo rispetto a commissioni, spread e slippage.

Inoltre, il bot tende a chiudere troppo rapidamente quando compare un segnale contrario di breve durata. Questo genera churn: entra, esce e rientra nella stessa zona VWAP, accumulando costi.

Il sistema deve quindi evolvere da:

**Segnale VWAP/AVWAP → BUY/SELL → chiudi quando cambia il segnale**

a:

**Market State → Candidate Setup → Stima del movimento futuro → Stima dei costi → Valutazione dell'edge → Risk Check → Execution Decision → Gestione dinamica del trade → Exit**

Un segnale VWAP o Anchored VWAP non deve più generare automaticamente un ordine.

Deve invece creare un **Candidate Setup**, che viene successivamente valutato.

---

## 1. Aggiungere MFE e MAE come obiettivi di analisi e training

Per ogni Candidate Setup e per ogni trade reale o paper devono essere calcolati:

- Maximum Favorable Excursion, MFE
- Maximum Adverse Excursion, MAE

su più orizzonti temporali, almeno:

- `MFE_5m`
- `MAE_5m`
- `MFE_15m`
- `MAE_15m`
- `MFE_30m`
- `MAE_30m`
- `MFE_60m`
- `MAE_60m`

L'obiettivo è insegnare al modello non soltanto se la direzione era corretta, ma **quanto movimento favorevole era realmente disponibile dopo l'ingresso**.

Il modello futuro non dovrebbe quindi produrre soltanto:

`LONG / SHORT`

ma qualcosa come:

- `P_long = 0.73`
- `expected_MFE_5m = +0.15%`
- `expected_MFE_15m = +0.31%`
- `expected_MFE_30m = +0.44%`
- `expected_MAE_15m = -0.12%`

Questo è fondamentale per calcolare target realistici.

---

## 2. Aggiungere TP-before-SL e probabilità di barriera

MFE e MAE da sole non sono sufficienti, perché non mostrano **l'ordine temporale** con cui vengono raggiunti i movimenti.

Per ogni Candidate devono essere calcolate etichette come:

- `hit_20bps_before_stop`
- `hit_30bps_before_stop`
- `hit_50bps_before_stop`

e nel modello:

- `P(TP before SL)`

Esempio:

`P(+0.30% before -0.20%) = 0.68`

Questa probabilità deve diventare una delle variabili principali per il calcolo dell'expectancy.

---

## 3. Calcolare il costo completo prima dell'ingresso

Prima di accettare un trade, il sistema deve stimare:

- `entry_fee`
- `expected_exit_fee`
- `spread_cost`
- `expected_slippage_entry`
- `expected_slippage_exit`
- `expected_funding`, se rilevante

e da questi:

- `expected_round_trip_cost`

Il break-even di un trade non deve coincidere semplicemente con il prezzo di ingresso.

Per un Long:

**break-even economico = Entry + costi totali stimati**

Per uno Short, l'equivalente inverso.

Le commissioni non devono essere hardcoded.

Devono essere configurabili perché possono cambiare in base a:

- VIP level
- Maker/Taker
- Exchange
- condizioni operative

---

## 4. Introdurre un vero Minimum Edge Filter

Un segnale direzionalmente corretto non deve automaticamente essere tradato.

Il bot deve poter decidere:

**NO TRADE**

se il movimento atteso è troppo piccolo rispetto ai costi.

Non basta verificare:

`expected_MFE > fees`

Bisogna calcolare una vera expectancy netta:

**EV = P(win) × NetWin − P(loss) × NetLoss**

eventualmente aggiungendo una penalizzazione per tail risk.

Esempio:

- `P(TP before SL) = 0.68`
- `NetWin = +0.28%`
- `NetLoss = -0.24%`

Quindi:

`0.68 × 0.28% − 0.32 × 0.24% = +0.1136%`

Il trade viene accettato soltanto se:

`expected_net_EV > minimum_required_EV`

Questa soglia deve essere configurabile.

---

## 5. Introdurre WAIT / NO TRADE come comportamento centrale

Il sistema non deve essere obbligato a cercare continuamente un trade.

Deve poter scegliere tra:

- BUY
- SELL
- HOLD / MANAGE
- CLOSE
- WAIT / NO TRADE

Idealmente l'architettura dovrebbe essere gerarchica invece di trattare tutte queste azioni come classi equivalenti.

Pipeline suggerita:

### Stage 1 — Setup Detector

Esiste un setup VWAP/AVWAP valido?

### Stage 2 — Alpha Model

Qual è la direzione probabile e quale movimento è plausibile?

### Stage 3 — Economic Gate

Il movimento previsto è economicamente tradabile dopo tutti i costi?

### Stage 4 — Risk Engine

Il trade è permesso dalle regole di rischio?

Solo dopo questi quattro passaggi può essere inviato un ordine.

---

## 6. Introdurre un target dinamico

La logica:

**chiudi appena il trade è verde**

deve essere eliminata.

Anche un Take Profit fisso uguale per tutti i trade non è ideale.

Il target deve dipendere da:

- Expected MFE
- volatilità
- ATR
- Market Regime
- struttura VWAP
- Order Flow
- Timeframe
- probabilità TP-before-SL

Esempio:

`expected_MFE = +0.46%`

Il sistema potrebbe scegliere:

`TP = +0.34%`

invece di tentare sempre di catturare tutto il movimento massimo previsto.

Deve inoltre esistere un:

`minimum_profitable_exit`

per impedire al bot di considerare economicamente positivo un trade che ha soltanto un Gross P&L leggermente sopra zero.

Il profitto reale deve sempre essere:

**Net P&L = Gross P&L − Fees − Spread − Slippage − Funding**

---

## 7. Aggiungere Time-to-Target

Per ogni Candidate Setup devono essere calcolati:

- `time_to_10bps`
- `time_to_20bps`
- `time_to_30bps`
- `time_to_50bps`

se raggiunti.

Il modello deve inoltre stimare:

- `expected_time_to_target`
- `P(target within 5m)`
- `P(target within 15m)`
- `P(target within 30m)`
- `P(target within 60m)`

Per il bot è molto diverso sapere che +0,30% è probabile entro 12 minuti oppure entro 4 ore.

---

## 8. Classificare il regime di volatilità

Target e Stop non devono essere identici in tutte le condizioni di mercato.

Aggiungere almeno:

- `ATR_1m`
- `ATR_5m`
- `ATR_15m`
- `ATR_30m`
- `realized_volatility`
- `range_N`
- `volatility_percentile`
- `volume_percentile`
- `spread_percentile`
- `depth_percentile`

Il regime può essere classificato, per esempio, come:

- `LOW_VOL`
- `NORMAL`
- `HIGH_VOL`
- `STRESS`

In bassa volatilità, un TP da +0,50% può essere irrealistico.

In alta volatilità, lo stesso TP potrebbe essere troppo piccolo.

Il modello deve poter decidere:

`NO_TRADE_ABNORMAL_REGIME`

---

## 9. Ampliare molto il contesto Rolling VWAP

Non usare soltanto:

- `price_above_vwap`
- `price_below_vwap`

Aggiungere:

- `distance_from_vwap_bps`
- `VWAP_slope`
- `VWAP_slope_change`
- `number_of_VWAP_tests`
- `number_of_VWAP_rejections`
- `time_since_last_VWAP_cross`
- `volume_on_VWAP_test`
- `rejection_strength`
- `VWAP_band_position`

ed eventualmente bande basate su deviazione standard.

Il bot deve poter distinguere, per esempio, tra:

- Trend Continuation
- Mean Reversion
- VWAP Rejection
- Fake Breakout
- VWAP Reclaim
- VWAP Failure

---

## 10. Utilizzare Anchored VWAP come feature strutturale, non solo come trigger

Anchored VWAP deve essere mantenuto.

Va però arricchito con:

`anchor_type`

Possibili tipi:

- `SESSION_OPEN`
- `DAY_HIGH`
- `DAY_LOW`
- `SWING_HIGH`
- `SWING_LOW`
- `BREAKOUT`
- `BREAKDOWN`
- `VOLUME_SPIKE`
- `LIQUIDITY_SWEEP`
- `VOLATILITY_EXPANSION`
- `MAJOR_IMPULSE`

Aggiungere anche:

- `anchor_timestamp`
- `anchor_age_seconds`
- `anchor_age_bars`
- `anchor_price`
- `volume_since_anchor`
- `return_since_anchor`
- `distance_from_AVWAP_bps`
- `AVWAP_slope`
- `AVWAP_slope_change`
- `AVWAP_tests`
- `AVWAP_rejections`
- `AVWAP_rejection_strength`
- `rolling_VWAP_vs_AVWAP_distance`
- `rolling_VWAP_AVWAP_convergence`
- `rolling_VWAP_AVWAP_divergence`

Un AVWAP ancorato al massimo o minimo giornaliero non deve essere trattato come uno ancorato a un piccolo swing di pochi minuti.

---

## 11. Separare statisticamente i diversi tipi di setup

Non aggregare tutti i trade VWAP insieme.

Salvare e analizzare separatamente almeno:

- `AVWAP_FAILURE_REVERSAL`
- `VWAP_PULLBACK_CONTINUATION`
- `ROLLING_VWAP_REENTRY`

oltre agli altri setup già esistenti.

Per ogni tipo di setup calcolare:

- `trade_count`
- `gross_win_rate`
- `net_win_rate`
- `mean_MFE`
- `median_MFE`
- `mean_MAE`
- `median_MAE`
- `mean_time_to_target`
- `median_time_to_target`
- `mean_cost`
- `mean_net_PnL`
- `profit_factor`
- `expectancy`
- `TP_before_SL_probability`

Un setup può avere una Win Rate elevata e restare netto negativo se i movimenti sono troppo piccoli rispetto ai costi.

Il bot deve imparare precisamente questa differenza.

---

## 12. Migliorare in modo sostanziale Order Flow e Order Book

La regola attuale:

**Order Flow contrario due volte → chiudi**

è troppo reattiva.

Un cambio molto breve dell'Order Flow può essere semplice rumore.

Aggiungere feature Multi-Level e Multi-Window.

### Order Flow

- `OFI_1s`
- `OFI_3s`
- `OFI_5s`
- `OFI_15s`
- `OFI_30s`

### Order Book

- `bid_depth_5bps`
- `ask_depth_5bps`
- `bid_depth_10bps`
- `ask_depth_10bps`
- `bid_depth_20bps`
- `ask_depth_20bps`
- `depth_imbalance_5bps`
- `depth_imbalance_10bps`
- `depth_imbalance_20bps`

Aggiungere inoltre:

- `aggressive_buy_volume`
- `aggressive_sell_volume`
- `aggressor_ratio`
- `market_buy_sell_delta`
- `trade_arrival_rate`
- `bid_cancel_rate`
- `ask_cancel_rate`
- `spread_bps`
- `spread_change`

Il bot deve distinguere tra:

**Micro-Flip temporaneo**

e:

**cambiamento persistente dell'Order Flow**

Prima di uscire da un trade per Order Flow contrario devono essere valutati almeno:

- intensità
- durata
- volume
- profondità
- persistenza
- conferma multi-window
- relazione con VWAP/AVWAP

---

# MODIFICHE AGGIUNTIVE NECESSARIE

Oltre ai dodici punti principali devono essere implementati anche i seguenti componenti.

---

## Simulatore di esecuzione realistico

Il Paper Trading non deve eseguire automaticamente un Market Order al Last Price.

Per ogni BUY simulato deve essere utilizzato il lato ASK dell'Order Book.

Per ogni SELL deve essere utilizzato il lato BID.

La size deve consumare progressivamente i livelli del book:

`ask1 → ask2 → ask3 → ...`

oppure:

`bid1 → bid2 → bid3 → ...`

fino al completo riempimento dell'ordine.

Calcolare:

- `execution_VWAP`
- `realized_slippage`

Il vero prezzo simulato di esecuzione deve essere l'Execution VWAP, non il Last Trade.

La stessa logica vale per l'uscita.

---

## Maker vs Taker come decisione di esecuzione

Il bot deve poter scegliere:

`TAKER`

quando il setup è urgente e il rischio di perdere il movimento è superiore al maggiore costo di esecuzione.

Oppure:

`MAKER_POST_ONLY`

quando il setup consente di attendere.

Per gli ordini Maker non assumere un fill garantito.

Devono essere modellati almeno:

- `fill_probability`
- `time_to_fill`
- `partial_fill`
- `adverse_selection_after_fill`
- `cancel_if_not_filled_after`

Il vantaggio potenziale delle fee Maker deve essere confrontato con il rischio di non essere eseguiti.

---

## Funding

Aggiungere:

- `current_funding_rate`
- `time_to_next_funding`
- `expected_funding_cost`

Nei trade molto brevi sarà spesso zero, ma deve essere incluso se la posizione attraversa un momento di Funding.

---

## Stop tecnico e Position Sizing

Lo Stop non deve essere deciso principalmente in base alla quantità di denaro che si vuole perdere.

Prima deve essere individuato il punto tecnico di invalidazione in base a:

- struttura
- swing
- VWAP
- AVWAP
- ATR
- Liquidity Level
- invalidazione Order Flow

Dopo:

`technical_stop_distance_pct`

Il Risk Engine calcola quindi la Position Size compatibile con il rischio consentito.

Formula concettuale:

**position size = risk budget / stop distance**

adattata a:

- leva
- fee
- slippage
- caratteristiche del contratto

Se la size risultante supera il massimo consentito:

**ridurre la size**

oppure:

**NO TRADE**

Non stringere artificialmente uno Stop tecnicamente corretto solo per poter utilizzare una posizione più grande.

---

## Hard Catastrophic Stop

Oltre allo Stop tecnico deve esistere uno Stop di sicurezza assoluto e non modificabile dal modello.

Questo deve proteggere da:

- Flash Crash
- Feed Error
- errore del modello
- Execution Failure
- improvvisa espansione della volatilità

Il modello non deve poter rimuovere o allargare autonomamente questo Stop.

---

## Trailing Stop solo dopo un vero profitto netto

Non iniziare a proteggere la posizione appena il Gross P&L diventa leggermente positivo.

Definire zone:

- `ZONE_A`: sotto break-even economico
- `ZONE_B`: sopra break-even ma sotto Minimum Net Profit
- `ZONE_C`: Minimum Net Profit raggiunto
- `ZONE_D`: forte espansione favorevole

Trailing e Profit Protection devono iniziare principalmente nelle Zone C e D.

---

## Re-Entry Penalty basata sullo stato di mercato

Ridurre il churn attorno alla VWAP.

Non usare necessariamente:

`cooldown = N minutes`

Usare invece una condizione basata sul cambiamento reale dello stato.

Esempi:

- nuovo Swing
- nuova Rejection
- nuovo VWAP Reclaim
- forte cambiamento OFI
- Breakout
- cambio di volatilità
- nuovo Volume Impulse
- Setup Score significativamente superiore al precedente

Possibili regole:

`reentry_allowed = meaningful_state_change == true`

oppure:

`new_setup_score > previous_setup_score + threshold`

---

## Regime Multi-Timeframe

Il sistema deve utilizzare almeno:

- 1m
- 5m
- 15m
- 30m

Opzionalmente:

- 1h come contesto

Non usare semplicemente una votazione a maggioranza tra timeframe.

Salvare invece:

- `trend_1m`
- `trend_5m`
- `trend_15m`
- `trend_30m`
- `VWAP_state_1m`
- `VWAP_state_5m`
- ecc.

Il modello deve distinguere, per esempio, tra:

**Mean-Reversion Scalp 1m contro un trend 30m ribassista**

e:

**Long Continuation con allineamento 1m/5m/15m/30m**

Questo deve influenzare soprattutto:

- Target
- durata prevista
- Confidence
- Stop
- tipo di trade

---

## Reward Function basata sul NET P&L

Nel training non deve essere premiato il Gross P&L.

Usare sempre:

**Net P&L = Gross P&L − Fees − Spread − Slippage − Funding**

Una possibile Reward Function:

**Reward = ΔNetEquity − λ_drawdown × DrawdownPenalty − λ_tail × TailRiskPenalty − λ_invalid × InvalidTradePenalty**

Opzionalmente può essere aggiunta una piccola:

`turnover_penalty`

Prestare attenzione a non penalizzare due volte le commissioni.

Se il Net P&L include già i costi, la Turnover Penalty deve rappresentare soltanto il churn indesiderato aggiuntivo.

---

## Separare il Risk Engine dal modello ML

Il modello ML non deve avere controllo assoluto sul trading.

Deve esistere un **Risk Engine deterministico con diritto di veto**.

Deve poter rifiutare un trade per:

- `expected_edge_too_low`
- `spread_too_high`
- `slippage_too_high`
- `depth_too_low`
- `daily_drawdown_limit`
- `position_limit`
- `max_account_risk`
- `stale_market_data`
- `desynchronized_orderbook`
- `abnormal_volatility`
- `execution_failure`
- `exchange_connection_problem`

Il modello non deve poter bypassare queste regole.

---

## Data Integrity

Aggiungere controlli:

- `last_trade_update_age_ms`
- `last_book_update_age_ms`
- `sequence_gap_detected`
- `book_is_synced`
- `trade_feed_alive`
- `orderbook_feed_alive`
- `clock_drift_ms`

Se i dati non sono affidabili:

`NO TRADE`

Se esiste già una posizione aperta, deve attivarsi una procedura di gestione sicura.

---

# DATASET DI TRAINING

Il Dataset non deve essere costituito soltanto dai trade realmente aperti.

Questo è fondamentale.

Ogni potenziale setup VWAP/AVWAP deve essere registrato come:

`candidate_setup`

anche se il bot decide di non entrare.

Per ogni Candidate Setup deve comunque essere registrato ciò che il mercato fa successivamente.

Questo riduce il Selection Bias e permette al modello di imparare:

**trade presi che avrebbero dovuto essere evitati**

e:

**trade non presi che invece avevano un edge reale**

Una struttura minima potrebbe essere:

    timestamp
    symbol

    candidate_side
    setup_type
    setup_score

    price
    mid_price
    best_bid
    best_ask
    spread_bps

    rolling_vwap
    distance_from_rolling_vwap_bps
    rolling_vwap_slope
    rolling_vwap_tests
    rolling_vwap_rejections

    anchored_vwap
    anchor_type
    anchor_timestamp
    anchor_age_seconds
    anchor_age_bars
    anchor_price
    distance_from_avwap_bps
    avwap_slope
    avwap_tests
    avwap_rejections
    avwap_rejection_strength
    volume_since_anchor

    atr_1m
    atr_5m
    atr_15m
    atr_30m

    realized_volatility
    volatility_percentile
    volume_percentile
    spread_percentile

    ofi_1s
    ofi_5s
    ofi_15s
    ofi_30s

    bid_depth_5bps
    ask_depth_5bps
    bid_depth_10bps
    ask_depth_10bps
    bid_depth_20bps
    ask_depth_20bps

    depth_imbalance_5bps
    depth_imbalance_10bps
    depth_imbalance_20bps

    aggressive_buy_volume
    aggressive_sell_volume
    aggressor_ratio
    trade_arrival_rate

    expected_entry_fee
    expected_exit_fee
    expected_slippage
    expected_funding
    expected_total_cost

    technical_stop_pct

    MFE_5m
    MAE_5m
    MFE_15m
    MAE_15m
    MFE_30m
    MAE_30m
    MFE_60m
    MAE_60m

    time_to_10bps
    time_to_20bps
    time_to_30bps
    time_to_50bps

    hit_20bps_before_SL
    hit_30bps_before_SL
    hit_50bps_before_SL

    execution_type
    execution_price
    execution_vwap
    realized_slippage

    realized_gross_pnl
    realized_net_pnl
    exit_reason

---

# TRAINING: PREVENIRE DATA LEAKAGE

Tutte le feature utilizzate per una decisione al timestamp `t` devono essere realmente disponibili al timestamp `t`.

Non usare informazioni future.

Prestare particolare attenzione a:

- High/Low di una candela ancora aperta
- Close futuro
- ATR calcolato usando barre non ancora concluse
- VWAP normalizzato usando dati successivi
- Scaling sull'intero Dataset
- selezione degli Anchor usando informazioni future
- fill simulato a un prezzo che ha generato il segnale ma che non sarebbe stato realmente eseguibile

Tutte le normalizzazioni Rolling devono utilizzare esclusivamente dati passati.

---

## Train / Validation / Test

Non usare split casuali.

Usare split cronologici:

`TRAIN → VALIDATION → TEST`

e preferibilmente Walk-Forward Validation.

Mantenere inoltre un periodo finale completamente Out-of-Sample che non venga mai utilizzato per:

- Training
- Hyperparameter Tuning
- Threshold Selection
- Feature Selection

---

# METRICHE DI VALUTAZIONE

Non valutare il modello principalmente tramite accuracy direzionale.

La sola BUY/SELL Accuracy non è sufficiente.

Monitorare almeno:

- `Net PnL`
- `Net expectancy per trade`
- `Profit factor`
- `Max drawdown`
- `Sharpe / Sortino`, se utili
- `Average winner net`
- `Average loser net`
- `Win rate net`
- `Average round-trip cost`
- `Average MFE`
- `Average MAE`
- `MFE capture ratio`
- `Average holding time`
- `Trade count`
- `Turnover`
- `Re-entry frequency`
- `Cost / gross profit ratio`
- `Maker fill rate`
- `Average slippage`
- `TP-before-SL accuracy`
- `Calibration of predicted probabilities`

Una metrica particolarmente importante è:

**Cost / Gross Profit**

Se il sistema possiede un Gross Edge ma questa metrica rimane troppo alta, il sistema continua a essere economicamente inefficiente.

---

# OUTPUT DESIDERATO DEL MODELLO

L'obiettivo finale non deve essere semplicemente:

`BUY`

ma una risposta strutturata come:

    SETUP:
    VWAP_PULLBACK_CONTINUATION

    DIRECTION:
    LONG

    CONFIDENCE:
    0.74

    MARKET REGIME:
    NORMAL_VOL / BULLISH_15M / NEUTRAL_30M

    ENTRY:
    64,365

    EXPECTED MFE:
    5m  +0.14%
    15m +0.29%
    30m +0.43%

    EXPECTED MAE:
    5m  -0.06%
    15m -0.11%
    30m -0.17%

    P(+0.30% BEFORE SL):
    0.67

    EXPECTED TIME TO TP:
    14 minuti

    EXPECTED COST:
    0.14%

    TECHNICAL STOP:
    -0.21%

    PROPOSED TARGET:
    +0.34%

    EXPECTED NET WIN:
    +0.20%

    EXPECTED NET LOSS:
    -0.35%

    EXPECTED VALUE:
    +0.022%

    EXECUTION:
    POST_ONLY_MAKER

    ESTIMATED FILL PROBABILITY:
    71%

    POSITION SIZE:
    calculated from risk budget

    DECISION:
    TRADE

Oppure:

    DECISION:
    NO TRADE

    REASON:
    Expected MFE 0.18%
    Expected round-trip cost 0.15%
    Insufficient net edge after risk adjustment

La seconda decisione deve essere considerata **corretta e desiderabile**, non un'occasione persa.

---

# ORDINE DI IMPLEMENTAZIONE

Non implementare tutto contemporaneamente senza poter misurare l'effetto delle singole modifiche.

Ordine consigliato:

## 1. MFE / MAE + TP-before-SL Labels

Prima bisogna sapere cosa succede realmente dopo ogni Candidate Setup.

## 2. Net EV Filter + NO TRADE

Il bot deve smettere di tradare setup economicamente inutili.

## 3. Cost Model realistico

Fees, Spread, Execution VWAP e Slippage.

## 4. Order-Flow Persistence + Multi-Level Depth

Sostituire la logica troppo semplice "Order Flow contrario due volte".

## 5. Dynamic TP + Technical SL

Il Target deve dipendere dalla distribuzione prevista del movimento.

## 6. Position Sizing dal rischio

Prima Stop tecnico, poi Position Size.

## 7. State-Based Re-Entry / Churn Control

Ridurre Round Trip inutili.

## 8. Maker/Taker Execution Decision

Ottimizzare i costi di esecuzione.

## 9. Multi-Timeframe Regime

Usare 1m/5m/15m/30m come contesto, non come segnali isolati.

## 10. Anti-Leakage + Walk-Forward Validation

Obbligatorio prima di fidarsi delle metriche ottenute durante il Training GPU.

## 11. Salvare tutti i Candidate Setup, inclusi i NO TRADE

Ridurre Selection Bias.

## 12. Risk Engine indipendente e deterministico

Il modello propone; il Risk Engine autorizza.

---

# RISULTATO ATTESO

Non cercare immediatamente di massimizzare la Win Rate.

Il primo comportamento desiderato nel Paper Trading dopo queste modifiche è:

- meno trade
- meno entra-esci-rientra
- durata media leggermente maggiore quando esiste un edge reale
- Gross Profit medio per Winner più alto
- costi complessivi molto più bassi rispetto al Gross Profit
- riduzione della Cost/Gross-Profit Ratio
- maggiore differenza tra Average Winner Net e costo medio
- frequenti decisioni NO TRADE sui setup marginali

Se il numero di trade diminuisce molto dopo queste modifiche, non deve essere automaticamente considerato un problema.

L'obiettivo non è produrre il maggior numero possibile di previsioni corrette sulla direzione.

L'obiettivo è:

> **Eseguire soltanto i setup nei quali il movimento futuro atteso è abbastanza grande, abbastanza probabile e abbastanza liquido da produrre una expectancy positiva dopo tutti i costi, mantenendo il rischio entro limiti deterministici.**

La logica attuale basata su VWAP + Anchored VWAP + Order Flow **non deve essere eliminata**.

Deve diventare il **generatore dei Candidate Setup**.

Il nuovo layer basato su MFE/MAE, probabilità di barriera, Cost Model, Execution, Risk Management e Net EV deve decidere **quali di questi setup meritano realmente capitale**.