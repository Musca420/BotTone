# MUSCA V2 — report di implementazione e audit

## Verdetto

`RESEARCH_ONLY_DISCOVERY_EDGE`. MUSCA V2 è disponibile in shadow nella dashboard
`http://127.0.0.1:8080`, profilo `MUSCA V2 · Trend + VWAP`.

Non è autorizzata al denaro reale. La storia usata per progettare il protocollo è ormai
discovery; la conferma deve arrivare dai dati shadow cronologicamente successivi.

## Domanda posta al modello

La direzione non deriva dal solo VWAP. La policy determina prima il trend BTC con:

```text
trend_strength = 0.60 × rendimento 7 giorni + 0.40 × rendimento 30 giorni
```

Il VWAP giornaliero è poi il benchmark di ingresso. LONG richiede reclaim rialzista del VWAP
con trend positivo; SHORT richiede reclaim ribassista con trend negativo. Il segnale nasce
dopo la chiusura 5m e l'ingresso avviene non prima dell'open 5m successivo.

La protezione non è una distanza fissa:

- stop iniziale oltre lo swing dei sette giorni precedenti, con buffer 0,1 ATR;
- aggiornamento giornaliero dopo due giorni;
- lo stop può solo stringersi;
- uscita per stop, inversione persistente del trend dopo almeno un giorno o timeout a 14 giorni;
- rischio per trade 0,25%, massimo una posizione;
- costo composito base 8 bps e stress 16 bps round-trip.

## Dati

- Binance BTCUSDT perpetual 5m: gennaio 2020–luglio 2026;
- Binance BTCUSDT spot 5m: gennaio 2020–luglio 2026;
- 691.890 timestamp comuni;
- archivi ufficiali mensili, ognuno verificato tramite file `.CHECKSUM` Binance;
- ETH completamente escluso;
- il collector Bitunix alimenta soltanto lo shadow corrente della UI, non il training Alpha.

Fonte dati: [Binance public data](https://github.com/binance/binance-public-data/blob/master/README.md).
Il ruolo del VWAP come benchmark/prezzo relativo è coerente con
[Busseti e Boyd, VWAP Optimal Execution](https://web.stanford.edu/~boyd/papers/pdf/vwap_opt_exec.pdf);
la direzione multi-orizzonte segue l'ipotesi verificabile di
[Moskowitz, Ooi e Pedersen, Time Series Momentum](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf).

## Risultati

| Audit | Trade | EV netta | PF | Win rate | Max drawdown a rischio 0,25% |
|---|---:|---:|---:|---:|---:|
| Base discovery, costi 8 bps | 575 | +64,25 bps | 1,46 | 32,87% | 4,29% |
| Walk-forward 2023–2026, costi 8 bps | 227 | +39,14 bps | 1,31 | 30,40% | 3,83% |
| Walk-forward, stress 16 bps | 227 | +31,14 bps | 1,23 | 29,96% | 3,83% |

Expectancy annuale base in bps: 2020 `+263,42`; 2021 `+75,34`; 2022 `-7,42`;
2023 `+90,97`; 2024 `+39,99`; 2025 `+2,22`; 2026 `+2,65` fino a luglio.

Nel walk-forward il champion è stato XGBoost GPU nel 2023, la base deterministica nel 2024
e Ridge nel 2025–2026. Ridge rimane il meta-filtro finale; non sostituisce la base finché
non dimostra stabilmente un vantaggio.

## Gate

- PASS: expectancy OOS positiva;
- PASS: PF OOS ≥ 1,15;
- PASS: drawdown ≤ 8%;
- PASS: stress 2× non negativo;
- PASS: maggioranza degli anni positiva;
- FAIL: 227 trade OOS, richiesti almeno 300;
- FAIL operativo: manca ancora un holdout futuro non osservato dopo la definizione del protocollo.

Per questi due fallimenti il bundle resta `RESEARCH_ONLY` e contiene
`real_capital_allowed=false` e `live_orders_enabled=false`.

## File collegati

- Training, download checksum, feature, replay e walk-forward:
  `src/adaptive_bot/musca_v2_research.py`
- Worker shadow persistente e simulazione ordini:
  `src/adaptive_bot/musca_v2.py`
- Integrazione profilo API/UI:
  `src/adaptive_bot/dashboard/server.py`
- Etichette e rendering browser:
  `src/adaptive_bot/dashboard/static/app.js`
- Test mirati:
  `tests/unit/test_musca_v2.py`
- Launcher shadow:
  `scripts/run_musca_v2_shadow.ps1`
- Dataset BTC allineato:
  `data/ml/musca_v2/btc_spot_perp_5m.parquet`
- Trade controfattuali della policy:
  `data/ml/musca_v2/trades.parquet`
- Decisioni walk-forward:
  `data/ml/musca_v2/oos_trades.parquet`
- Bundle congelato research-only:
  `data/models/musca_v2/bundle.joblib`
- Audit JSON:
  `data/reports/musca_v2_research.json`
- Report consumato dalla dashboard:
  `data/reports/musca_v2_shadow.json`
- Stato persistente del paper shadow:
  `data/research/musca_v2_shadow_state.json`

## Prossimo protocollo senza data snooping

Non si cambiano pesi 7d/30d, holding o stop osservando i prossimi risultati. Si accumulano
trade shadow futuri; a 300 trade OOS complessivi si rieseguono gate, bootstrap a blocchi,
stabilità temporale e confronto base/Ridge/XGBoost. Qualsiasi nuova variante deve avere un
nuovo protocol hash e un holdout successivo distinto.
