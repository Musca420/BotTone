# Musca V5 — gap analysis definitivo rispetto a `fina_model.md`

Data audit: 2026-08-08. Ambito: BTCUSDT, Alpha Binance e paper execution
Bitunix VIP0. Musca V2 e i relativi dati non sono stati modificati.

## Verdetto economico verificato

La pipeline richiesta è implementata, ma il run corrente è
`NO_ECONOMIC_ALPHA`. Non è un errore dovuto a `FLAT`: sulle famiglie frequenti il
payoff lordo medio è vicino a zero e il round trip Bitunix VIP0 taker/taker costa
12 bps di sole fee, più spread/slippage/funding osservati o riservati.

Il precedente label confondeva ogni mancato target con uno stop pieno. È stato
corretto con tre esiti mutuamente esclusivi: `TARGET`, `STOP`, `TIMEOUT`. Un
timeout usa il close osservato a 60 minuti; stop e target nella stessa candela
assegnano lo stop. Il limite prudenziale dei residui viene ora stimato sulla
calibrazione, mai sull'audit.

Run corretto:

- sorgente: 1.209.599 minuti Binance BTC perpetual + spot;
- candidati: 339.178;
- fit: 213.813; calibrazione: 45.805; audit: 45.822;
- holdout sigillato: 33.704 eventi, dal 2026-05-11, non aperto;
- champion: Ridge; XGBoost CUDA non batte Ridge su MAE del payoff e Brier;
- policy OOS accettata: 0 trade;
- miglior 0,5% ordinato dal modello: 229 eventi, expectancy -11,73 bps,
  PF 0,568, bootstrap LCB -18,01 bps;
- stato: `NO_ECONOMIC_ALPHA`; nessun bundle operativo e denaro reale vietato.

## Copertura dei requisiti

| Requisito | Stato | Implementazione / limite reale |
|---|---|---|
| MFE/MAE 5/15/30/60m | COMPLETO | Label causali per ogni candidato. |
| TP-before-SL 20/30/50 bps | COMPLETO | Ordine temporale, stop wins same minute, tempo al target. |
| Target/stop/timeout | COMPLETO | Payoff specifico per piano; timeout non è più uno stop inventato. |
| Cost model | COMPLETO LIVE | Fee VIP0–VIP5 configurabili, spread, book-walk, slippage, funding. Lo storico senza book non viene imputato. |
| Net EV e NO TRADE | COMPLETO | Alpha lordo separato dall'execution corrente; EV e limite prudenziale devono essere positivi. `FLAT=0`, mai profitto. |
| Target dinamico | COMPLETO PREREGISTRATO | Selezione fra 20/30/50 bps in base al payoff previsto; nessuna griglia opportunistica. |
| Time-to-target | COMPLETO | 10/20/30/50 nei dataset dettagliati; probabilità 5/15/30/60 e tempo condizionato. |
| Regime multi-timeframe | COMPLETO | ATR, trend e stato VWAP 1/5/15/30m, volatilità e volume percentile. |
| Rolling VWAP strutturale | COMPLETO | Distanza, slope/change, test, rejection, ultimo cross e bande. |
| Anchored VWAP strutturale | COMPLETO | Anchor causale, età, prezzo, volume, ritorno, slope, test/rejection e convergenza. |
| Statistiche per setup | COMPLETO | Continuation, reversion, re-entry e AVWAP failure separati. |
| Order flow / book multi-window | COMPLETO LIVE | Binance L2 top-20 e Bitunix L2 5/10/20 bps; storico L2 non viene inventato. |
| Execution VWAP | COMPLETO | BUY consuma ask, SELL consuma bid, Decimal, partial depth e slippage realizzato. |
| Maker/taker | PARZIALE PER DATI | Taker realistico attivo. Maker resta disabilitato finché fill/partial/adverse privati non sono osservati su giornate indipendenti. |
| Funding | COMPLETO | Mark/funding Bitunix osservato, applicato solo quando attraversato. |
| Stop tecnico e sizing | COMPLETO | Prima invalidazione tecnica, poi 1% risk budget, 10% margin cap e leva massima 10x. |
| Catastrophic stop | COMPLETO | 200 bps, non rimovibile dal modello. |
| Trailing netto | COMPLETO | Si arma solo oltre costi più 8 bps e non allarga mai il rischio. |
| Re-entry state based | COMPLETO | Richiede cambiamento osservato di setup/stato, non un semplice timer. |
| Risk Engine / integrity | COMPLETO | Veto deterministico per depth, stale data, sync, spread, size e rischio. |
| Tutti i candidati, inclusi NO TRADE | COMPLETO | Matrice controfattuale, non soltanto trade eseguiti. |
| Anti-leakage | COMPLETO | Entry al minuto successivo, `available_at`, purge sull'uscita, split cronologici e holdout chiuso. |

## Errori corretti nell'ultimo audit

1. `non target = stop`: sostituito con competing outcomes target/stop/timeout.
2. LCB calcolato sull'audit: spostato sulla calibrazione cronologica.
3. Gate drawdown/stress veri con zero trade: ora falliscono se non esistono trade.
4. Mapping live `ANCHOR_FAILURE`: prima veniva codificato come reversion generica;
   ora usa la stessa codifica AVWAP del training.
5. Stato UI basato sulla sola esistenza del file bundle: ora legge il verdetto del
   report (`NO_ECONOMIC_ALPHA`).

## Dove sparisce l'edge

Sull'audit corretto, i piani 20/30/50 bps delle famiglie frequenti hanno tutti
expectancy netta circa -13 bps: il loro lordo è circa zero e le fee spiegano quasi
interamente la perdita. `AVWAP_FAILURE_REVERSAL LONG` è l'unica nicchia con
movimento potenziale (220 eventi), ma il miglior target scelto con conoscenza del
futuro produce +8,23 bps: è un oracle, non una decisione eseguibile. Selezionarla
dopo aver visto l'audit sarebbe data snooping.

La V10 deterministica resta un'ipotesi discovery promettente (+14,97 bps su 27
trade pre-holdout), ma non supera la numerosità e non costituisce conferma
indipendente. Rimane shadow/research-only.

## Gap informativo residuo e percorso corretto

Gli studi primari sull'order-flow mostrano che la previsione di breve periodo usa
sequenze di quote, queue imbalance e più livelli del limit order book. Le sole
candele a un minuto e il taker imbalance aggregato non contengono l'ordine degli
eventi né cancellazioni, assorbimento e ritiro della liquidità necessari a
distinguere gli eventi VWAP vincenti.

Disponibilità reale al momento dell'audit:

- Binance L2: 151.441 righe, 2 giornate UTC (2026-08-06–07);
- Bitunix L2: 452.965 righe, 6 giornate UTC (2026-08-03–08);
- forward selector post-protocollo: 4 decisioni su 320 richieste;
- holdout execution: minimo 10 giorni e 100 trade selezionati;
- storico pluriennale L2 vicino al best: non disponibile, non imputato.

Quindi il prossimo modello autorizzabile non è una nuova ricerca su soglie candle.
È il meta-filtro forward sui candidati preregistrati, con MLOFI/queue imbalance,
persistenza, cancellazioni, spread e book depth realmente osservati. Ridge resta
champion; XGBoost CUDA è challenger sulle stesse split. Il maker model richiede
fill privati reali e resta fail-closed.

## File attivi

- `fina_model.md` — specifica funzionale.
- `src/adaptive_bot/btc_vwap_alpha.py` — dataset pluriennale, competing outcomes,
  Ridge/XGBoost, EV e audit.
- `src/adaptive_bot/musca_v5_market_state.py` — feature VWAP/AVWAP e regimi causali.
- `src/adaptive_bot/musca_v5_event_policy.py` — label multi-orizzonte e gestione.
- `src/adaptive_bot/musca_v5_execution.py` — costi, book-walk e risk sizing.
- `src/adaptive_bot/binance_l2_collector.py` e `binance_l2_dataset.py` — L2 Binance.
- `src/adaptive_bot/bitunix_l2_dataset.py` — L2/execution feature Bitunix.
- `src/adaptive_bot/btc_cross_exchange_forward_audit.py` — replay shadow e selector.
- `src/adaptive_bot/musca_v5_shadow.py` — conto paper da 10.000 USD.
- `data/reports/btc_vwap_alpha_v1.json` — risultato GPU verificabile.
- `data/reports/btc_cross_exchange_forward_audit.json` — stato forward/UI.

## Fonti primarie

- Binance Public Data: https://github.com/binance/binance-public-data/blob/master/README.md
- Bitunix depth: https://www.bitunix.com/api-docs/futures/market/get_depth.html
- Bitunix funding: https://www.bitunix.com/api-docs/futures/market/get_funding_rate.html
- Bitunix fee table: https://www.bitunix.com/service/handling-fee
- Cont, Kukanov, Stoikov, *The Price Impact of Order Book Events*:
  https://doi.org/10.2139/ssrn.1712822
- Xu, Gould, Howison, *Multi-Level Order-Flow Imbalance*:
  https://arxiv.org/abs/1907.06230
- Gould, Bonart, *Queue Imbalance as a One-Tick-Ahead Price Predictor*:
  https://arxiv.org/abs/1512.03492
- Zhang, Zohren, Roberts, *DeepLOB*:
  https://arxiv.org/abs/1808.03668

## Decisione operativa

Musca V5 continua esclusivamente in shadow. Non è stato abbassato alcun gate, non
è stato aperto l'holdout e non è stato promosso un modello negativo. Il bot può
simulare trade soltanto quando la base shadow produce un setup; l'Alpha appena
addestrato resta un filtro diagnostico non operativo finché non esiste evidenza
forward netta positiva dopo costi VIP0.
