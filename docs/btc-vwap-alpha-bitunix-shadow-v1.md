# BTC VWAP Alpha → Bitunix shadow v1

## Stato

- Modalità: `RESEARCH_ONLY`, denaro reale vietato.
- Alpha: BTC soltanto, storico ufficiale Binance perpetual + spot 1m.
- Execution: BTCUSDT Bitunix, replay del book osservato a 15 livelli.
- Profilo operativo costi: Bitunix futures VIP0 taker/taker.
- Scenari diagnostici: VIP0–VIP5 sugli stessi fill e sulla stessa size.
- Holdout storico: ultime 12 settimane sigillate e non aperte.

## Dati e split

- Minuti Binance: 1.209.599, dal 15 aprile 2024 al 3 agosto 2026.
- Eventi VWAP deterministici: 231.639.
- Fit: 146.488 eventi.
- Calibrazione: 31.383 eventi.
- Audit cronologico pre-holdout: 31.393 eventi.
- Holdout sigillato: 22.355 eventi dal 11 maggio 2026.
- Bitunix L2 al primo avvio: 6 giorni e oltre 440.000 snapshot.

Gli eventi sono `PULLBACK_CONTINUATION` e `VWAP_REVERSION`, separati per LONG e
SHORT e deduplicati ogni 10 minuti per famiglia/lato. Il segnale usa soltanto la
candela completata; l'entry storica è l'open del minuto successivo. Se stop e
target sono raggiunti nella stessa candela da un minuto, vince lo stop.

## Modelli

Ridge/logistic è il champion predefinito. XGBoost usa CUDA ma viene promosso
soltanto se migliora contemporaneamente MAE del rendimento e Brier score della
probabilità target sullo stesso audit.

Risultato corrente:

- Champion: Ridge.
- Ridge MAE EV: 36,496 bps; Brier P(30 bps prima dello stop): 0,209856.
- XGBoost MAE EV: 36,570 bps; Brier: 0,209490.
- XGBoost non promosso perché peggiora l'errore EV.

## Copertura dei 12 punti di `fina_model.md`

1. MFE e MAE: label e regressioni a 5/15/30/60 minuti.
2. TP-before-SL: probabilità calibrate per 20/30/50 bps.
3. Costi: fee entry/exit, book VWAP, riserva slippage e funding osservato.
4. Minimum edge: target ammesso soltanto con EV VIP0 netto positivo e MFE
   atteso almeno 3× il costo round-trip.
5. Pipeline: generatore setup → Alpha → execution/risk veto → TRADE/FLAT →
   gestione HOLD/CLOSE dinamica.
6. Target: scelta tra 20/30/50 bps usando EV probabilistico VIP0.
7. Tempo: probabilità entro 5/15/30/60 minuti e regressione del tempo al target.
8. Regime: stress/volatilità anomala provoca veto fail-closed.
9. Rolling VWAP: VWAP 5m da trade/volume, distanza e slope causali.
10. Anchored VWAP: usato dal forward engine per conferma/fallimento, non come
    direzione autonoma dell'Alpha storico.
11. Metriche per setup: continuation/reversion e lato sono registrati separatamente.
12. Order flow/book: imbalance multi-finestra, microprice, depth e cancellazioni
    nel replay Bitunix; le feature senza storia comune non vengono inventate.

## Costi VIP

Tariffe futures ufficiali in bps per lato (maker/taker): VIP0 2,0/6,0; VIP1
2,0/5,0; VIP2 1,6/5,0; VIP3 1,4/4,0; VIP4 1,2/3,75; VIP5 1,0/3,5.

Il conto usa sempre VIP0 per decidere e dimensionare. VIP1–VIP5 cambiano solo
la commissione nello scenario controfattuale; non cambiano segnale, fill, size,
funding o slippage. I fill maker restano disabilitati finché non esistono label
private osservate.

## Stato shadow iniziale

- Conto: 10.000 USDT.
- Margine massimo: 10% del saldo.
- Leva massima: 10×.
- Perdita pianificata massima: 1% del saldo per trade.
- Candidati Bitunix completati e valutati: 54.
- Candidati con EV VIP0 positivo: 0.
- Ordini eseguiti dal nuovo Alpha: 0; saldo 10.000 USDT.

Questo non è un errore operativo: il candidato più recente aveva probabilità
target circa 26,5%, MFE atteso circa 15,8 bps e EV circa -18,7 bps a VIP0
(-13,7 bps a VIP5). Forzarlo avrebbe violato il Cost/EV Gate. Il worker continua
a valutare nuovi eventi e aprirà ordini shadow soltanto quando il setup completo
ha EV netto VIP0 positivo.

## File collegati

- `src/adaptive_bot/btc_vwap_alpha.py`: dataset, label, training GPU, calibrazione e scoring.
- `src/adaptive_bot/btc_cross_exchange_forward_audit.py`: replay Bitunix, risk engine, conto e scenari VIP.
- `src/adaptive_bot/bitunix_fees.py`: tabella ufficiale delle fee.
- `src/adaptive_bot/dashboard/static/index.html`: pannelli Alpha/VIP.
- `src/adaptive_bot/dashboard/static/app.js`: rendering delle decisioni e dei trade.
- `scripts/run_btc_vwap_forward_audit.ps1`: worker con lock PID anti-duplicato.
- `tests/unit/test_btc_vwap_alpha.py`: causalità, deduplica, esclusione ETH e mapping live.
- `data/ml/btc_vwap_alpha_v1/events.parquet`: matrice storica.
- `data/models/btc_vwap_alpha_v1/bundle.joblib`: bundle research congelato.
- `data/reports/btc_vwap_alpha_v1.json`: audit Alpha.
- `data/reports/btc_cross_exchange_forward_audit.json`: stato shadow aggiornato.

## Prossima calibrazione

I trade shadow Bitunix non entrano nel training Alpha iniziale. Dopo almeno 10
giorni e 100 trade completati si esegue una calibrazione cronologica separata
di execution/transfer; non si apre l'holdout storico e non si sostituiscono
osservazioni mancanti con simulazioni.
