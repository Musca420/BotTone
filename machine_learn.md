Specifica tecnica: addestramento del modello per Adaptive Range Bot

Stato implementazione: il protocollo attivo e `scientific_v2`. Il precedente esperimento da 20
trial e classificato come baseline non valida per la selezione perche riutilizzava payoff del
periodo valutato e un holdout gia consultato. La specifica eseguibile, i comandi e i gate correnti
sono documentati in `docs/ml-research-protocol.md`.

Obiettivo

Costruire un sistema di ricerca quantitativa e machine learning che, usando VWAP come centro dinamico, impari a stimare quando un’entrata long o short ha probabilità favorevole di ritorno verso la media.

Il modello non deve prevedere semplicemente se la prossima candela sarà verde o rossa. Deve stimare:

“Dato lo stato attuale del mercato, qual è la probabilità che il prezzo raggiunga un obiettivo di mean reversion prima dello stop, al netto dei costi?”

Il sistema deve essere sviluppato inizialmente in paper trading, senza ordini reali.

1. Acquisizione dei dati

Scaricare almeno 12 mesi di candele a 5 minuti, preferibilmente 18–24 mesi se disponibili, per ogni coppia analizzata.

Bitunix offre endpoint ufficiali per:

elenco delle coppie futures;
dati Kline storici;
ticker;
order book;
dati WebSocket in tempo reale.

Per ogni candela salvare:

timestamp
symbol
open
high
low
close
volume
quote_volume, se disponibile

Per il funzionamento live salvare anche:

best_bid
best_ask
spread_bps
order_book_depth
trade_count o flusso trade, se disponibile

Lo spread deve essere calcolato così:

mid_price = (best_ask + best_bid) / 2
spread_bps = (best_ask - best_bid) / mid_price * 10_000
Limitazione fondamentale

Non inventare lo spread storico usando soltanto OHLCV.

Se Bitunix non fornisce uno storico bid/ask completo per l’anno passato:

usare un modello prudenziale dei costi nel backtest;
raccogliere da ora in avanti bid/ask e profondità reali;
marcare chiaramente lo spread storico come estimated, non observed.

Le candele permettono di ricalcolare VWAP, ATR, ADX e bande, ma non permettono di ricostruire con precisione il book storico.

2. Controlli di qualità del dataset

Prima dell’addestramento:

- ordinare rigorosamente per timestamp;
- rimuovere duplicati;
- verificare barre mancanti;
- non riempire volume o prezzi mancanti con valori arbitrari;
- registrare eventuali interruzioni del feed;
- verificare che ogni feature usi soltanto dati disponibili fino a quel momento;
- usare UTC internamente;
- mantenere separati i dati dei diversi simboli.

Non usare interpolazione futura per riempire le barre: provocherebbe data leakage.

3. Feature engineering

Tutte le feature alla candela t devono essere calcolate esclusivamente con dati disponibili entro la chiusura di t.

Feature centrali VWAP

Calcolare:

vwap
distance_from_vwap_pct
distance_from_vwap_atr
vwap_zscore
vwap_slope_3
vwap_slope_6
vwap_slope_12
bars_above_vwap
bars_below_vwap
crossed_vwap_last_n_bars

Esempi:

distance_from_vwap_pct = (close - vwap) / vwap
distance_from_vwap_atr = (close - vwap) / atr
vwap_zscore = (close - vwap) / rolling_std

Il valore dello Z-score deve mantenere il segno:

negativo: prezzo sotto VWAP;
positivo: prezzo sopra VWAP.

Non usare soltanto abs(z) perché long e short devono essere analizzati separatamente.

Bande adattive
upper_band = vwap + band_multiplier * volatility_measure
lower_band = vwap - band_multiplier * volatility_measure
band_width_pct
position_inside_band
distance_from_upper_band
distance_from_lower_band

La volatilità può essere ATR, deviazione standard o una combinazione, ma la formula deve essere configurabile.

Volatilità
ATR
ATR_pct
ATR_percentile
rolling_std
realized_volatility
high_low_range_pct
candle_body_pct
upper_wick_pct
lower_wick_pct
Regime
ADX
ADX_slope
EMA_20_slope
EMA_50_slope
VWAP_slope
trend_strength
volatility_regime
range_regime
shock_regime
Volume e liquidità
relative_volume
volume_zscore
volume_change
spread_bps
book_depth_bid
book_depth_ask
book_imbalance
Momentum e inversione
return_1
return_3
return_6
return_12
RSI
close_position_in_candle
previous_high_break
previous_low_break
bullish_reversal_flag
bearish_reversal_flag

Evitare centinaia di indicatori ridondanti. Il modello deve partire da un set compatto e interpretabile, indicativamente 20–50 feature.

4. Definizione corretta delle etichette

Non addestrare il modello a prevedere semplicemente close[t+1] > close[t].

Creare due dataset o due target distinti:

target_long
target_short
Target long

Alla candela t, ipotizzare un ingresso long al prezzo realistico di esecuzione.

Definire:

profit barrier = ritorno verso VWAP oppure +X ATR
stop barrier   = -Y ATR
time barrier   = massimo N barre

Il target long vale:

1 = profit barrier raggiunta prima dello stop
0 = stop raggiunto prima del target

Se nessuna barriera viene raggiunta entro N barre:

classificare come neutral;
oppure chiudere al time barrier e assegnare il PnL netto.
Target short

Stessa logica invertita:

profit barrier = ritorno verso VWAP oppure -X ATR
stop barrier   = +Y ATR
time barrier   = massimo N barre
Ambiguità intrabar

Se nella stessa candela high e low toccano sia target sia stop e non sono disponibili dati più granulari:

assumere l’esito peggiore;
oppure scartare il campione;
non assumere automaticamente che sia stato raggiunto prima il target.

Questo rende il backtest prudente.

5. Costi e realismo dell’esecuzione

Ogni etichetta e ogni backtest devono includere:

commissione apertura
commissione chiusura
spread
slippage
funding, se la posizione lo attraversa

Usare:

net_pnl = gross_pnl - fees - spread_cost - slippage - funding

Non scegliere il modello usando profitto lordo.

Se lo spread storico non è disponibile, eseguire almeno tre scenari:

ottimistico
base
stress

Per esempio, con costi e slippage progressivamente maggiori.

6. Separazione temporale dei dati

È vietato usare split casuali.

Scikit-learn specifica che per dati ordinati temporalmente bisogna usare suddivisioni che impediscano di addestrare sul futuro e testare sul passato; TimeSeriesSplit è progettato proprio per questo caso.

Usare una divisione iniziale simile:

Train:      primi 60%
Validation: successivi 20%
Test:       ultimi 20%

Il test finale deve restare completamente inutilizzato fino alla selezione conclusiva.

Walk-forward validation

Implementare inoltre una validazione walk-forward:

Fold 1: train mesi 1–4, validate mese 5
Fold 2: train mesi 1–5, validate mese 6
Fold 3: train mesi 1–6, validate mese 7
...

Tra train e validation applicare un gap pari almeno all’orizzonte massimo dell’etichetta, per evitare che trade sovrapposti trasferiscano informazioni tra i set.

Esempio:

TimeSeriesSplit(
    n_splits=5,
    test_size=validation_size,
    gap=max_holding_bars
)
7. Modelli da addestrare

Partire con modelli interpretabili e tabellari.

Baseline obbligatoria
Logistic Regression regolarizzata

Serve come riferimento. Scikit-learn applica regolarizzazione alla regressione logistica e consente di ottenere probabilità tramite predict_proba.

Modello principale

Usare uno dei seguenti:

XGBoost
oppure
HistGradientBoostingClassifier

Per XGBoost:

profondità limitata;
regolarizzazione;
learning rate contenuto;
early stopping;
probabilità, non soltanto classi.

La documentazione ufficiale XGBoost prevede l’uso di un validation set e dell’early stopping per interrompere l’addestramento quando la metrica non migliora.

Non iniziare con reti neurali o LSTM: su un solo anno di barre a 5 minuti il rischio di overfitting è elevato e l’interpretabilità è inferiore.

8. Long e short separati

Addestrare preferibilmente due modelli:

long_model
short_model

Perché:

i ribassi possono essere più rapidi dei rialzi;
volatilità e liquidità possono essere asimmetriche;
le soglie ottimali possono essere diverse;
il ritorno al VWAP può avere distribuzioni differenti.

Output richiesto:

p_long_success
p_short_success

Non consentire simultaneamente un long e uno short sullo stesso simbolo.

9. Ottimizzazione degli iperparametri

Non fare una griglia enorme.

Usare Optuna con ricerca TPE o equivalente e pruning dei trial non promettenti. La documentazione ufficiale di Optuna descrive sia il campionamento efficiente sia la possibilità di interrompere anticipatamente i trial deboli.

Ottimizzare, ad esempio:

max_depth
learning_rate
n_estimators
min_child_weight
subsample
colsample_bytree
reg_alpha
reg_lambda
entry_probability_threshold
zscore_min
ADX_max
ATR_percentile_min
ATR_percentile_max
max_holding_bars
stop_ATR
target_ATR
Regola

Gli iperparametri devono essere scelti solo sui fold train/validation.

Il test finale non deve essere usato da Optuna.

10. Funzione obiettivo

Non massimizzare l’accuracy.

La maggior parte delle barre non produrrà un buon segnale e l’accuracy potrebbe essere fuorviante.

Ottimizzare una funzione economica, per esempio:

score = (
    median_walk_forward_net_return
    - 1.5 * max_drawdown
    + 0.5 * profit_factor_bonus
    - instability_penalty
    - low_trade_count_penalty
)

Richiedere anche:

numero minimo di trade
profitto netto positivo in più fold
drawdown entro il limite
stabilità tra mesi
risultato positivo nello scenario costi stress

Il candidato migliore non deve essere quello con il profitto massimo assoluto, ma quello più robusto.

11. Metriche ML

Registrare:

precision
recall
PR-AUC
ROC-AUC
Brier score
calibration curve
confusion matrix

La precisione misura quanti segnali positivi generati siano realmente corretti; la curva precision-recall mostra il compromesso tra precisione e recall al variare della soglia.

Nel trading, dare priorità alla precisione dei segnali rispetto alla quantità.

12. Calibrazione delle probabilità

Una probabilità prevista del 70% dovrebbe corrispondere, approssimativamente, a un successo osservato vicino al 70%.

Usare:

CalibratedClassifierCV

con metodo:

sigmoid

oppure isotonic soltanto quando ci sono abbastanza campioni.

Scikit-learn fornisce strumenti specifici per calibrare le probabilità e produrre reliability diagram.

La calibrazione deve essere effettuata su dati temporalmente successivi a quelli usati per addestrare il modello base.

13. Decisione operativa

Il modello non deve entrare solo perché p > 0.5.

Calcolare l’expected value:

ev = (
    p_success * expected_win
    - (1 - p_success) * expected_loss
    - estimated_costs
)

Entrare soltanto se:

EV > soglia minima
probabilità calibrata > soglia
regime ammesso
spread valido
liquidità valida
nessun risk guard attivo

Esempio long:

long_allowed = (
    p_long_success >= 0.67
    and expected_value_long > 0
    and close < vwap
    and zscore <= -z_threshold
    and spread_is_valid
    and regime in {"range", "controlled_volatility"}
)

La soglia 0.67 è solo un valore iniziale da validare, non un valore garantito.

14. Risk management separato dal modello

Il modello deve decidere la qualità del setup, non la leva.

Tenere separati:

Signal Engine
Risk Manager
Execution Engine

Il Risk Manager deve imporre:

rischio massimo per trade
esposizione massima
numero massimo di posizioni
perdita giornaliera massima
drawdown massimo
cooldown dopo perdite consecutive
blocco in caso di feed incompleto

La leva 10× non deve significare automaticamente rischio 10×. La size deve essere calcolata dalla distanza dello stop:

risk_amount = equity * risk_per_trade
position_notional = risk_amount / stop_distance_pct

Poi applicare i limiti:

position_notional = min(
    position_notional,
    equity * max_exposure * leverage
)
15. Backtest event-driven

Non limitarsi a confrontare segnali con rendimenti futuri.

Realizzare un backtester sequenziale che simuli:

segnale
ordine
fill
commissioni
spread
slippage
posizione
stop
target
uscita VWAP
time stop
funding
equity
drawdown

Il backtest deve impedire:

più posizioni quando il limite è 1;
uso di capitale non disponibile;
fill irrealistici;
utilizzo di dati della chiusura prima che la candela sia terminata.
16. Criteri minimi per accettare il modello

Non promuovere il modello al paper trading live finché non soddisfa tutti questi requisiti:

1. Profitto netto positivo sul test mai visto.
2. Profitto positivo in più finestre walk-forward.
3. Profit factor superiore a 1 dopo i costi.
4. Drawdown compatibile con il limite stabilito.
5. Numero sufficiente di trade.
6. Nessuna dipendenza da un solo mese o da un solo simbolo.
7. Risultati accettabili con costi maggiorati.
8. Probabilità ragionevolmente calibrate.
9. Prestazioni migliori della strategia VWAP deterministica di base.

Non fissare valori rigidi come garanzia di successo. Registrarli nel report e confrontarli con la baseline.

17. Paper trading e retraining

Dopo il backtest:

- eseguire paper trading live per almeno 4–8 settimane;
- raccogliere spread e slippage reali;
- confrontare risultato previsto e risultato osservato;
- rilevare drift delle feature;
- non riaddestrare automaticamente ogni giorno.

Retraining iniziale consigliato:

mensile
oppure
quando viene rilevato un cambiamento significativo nella distribuzione

Ogni nuovo modello deve superare lo stesso test walk-forward prima di sostituire quello attivo.

18. Output richiesti nella dashboard

Mostrare per ogni decisione:

symbol
timestamp
side valutato
p_success calibrata
expected value
close
VWAP
z-score
ATR percentile
ADX
spread reale
regime
target
stop
holding horizon
decisione finale
motivo dell’eventuale rifiuto
model_version

Esempio:

LONG candidate: BTCUSDT
P(return to VWAP before stop): 71.4%
Expected value after costs: +0.18%
Z-score: -2.31
ADX: 17.8
ATR percentile: 64
Spread: 0.42 bps
Decision: ACCEPTED

Oppure:

Decision: REJECTED
Reason: spread unavailable

Lo spread mancante non deve apparire come 0.00 bps: usare esplicitamente:

N/A
INVALID
STALE
19. Struttura software richiesta
data/
    bitunix_client.py
    historical_loader.py
    websocket_collector.py
    data_validator.py

features/
    vwap_features.py
    volatility_features.py
    regime_features.py
    liquidity_features.py

labels/
    triple_barrier.py

models/
    baseline_logistic.py
    xgboost_long.py
    xgboost_short.py
    calibration.py

validation/
    purged_walk_forward.py
    metrics.py
    stability_tests.py

backtest/
    event_engine.py
    cost_model.py
    portfolio.py
    execution_simulator.py

optimization/
    optuna_objective.py

live/
    signal_engine.py
    risk_manager.py
    paper_execution.py

reports/
    model_card.py
    walk_forward_report.py
