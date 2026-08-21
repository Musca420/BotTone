# V25 VWAP Definitive Bot — implementation report

## Verdetto

`NO_META_EDGE`.

La V25 è stata implementata ed eseguita senza modificare V21–V24. I dati, gli eventi e gli outcome hanno superato i controlli tecnici; il meta-modello non ha però identificato una policy OOS che superi i gate preregistrati. Il bundle prodotto è quindi soltanto `RESEARCH_ONLY`: può essere caricato in `research`, `paper` o `shadow`, rifiuta il live e non autorizza capitale reale.

`FLAT` non è conteggiato come profitto: con zero trade expectancy, profit factor, LCB e drawdown non superano i gate finanziari.

## Dati realmente usati

| Asset | Mercati | Intervallo UTC | Minuti allineati | Copertura obbligatoria | Mark | Funding | OI opzionale |
|---|---|---|---:|---:|---:|---:|---:|
| BTCUSDT | Binance spot + perpetual UM | 2024-04-15 20:28 — 2026-08-03 20:28 | 1.209.599 | 100% | 100% | 100% | 94,80% |
| ETHUSDT | Binance spot + perpetual UM | 2024-04-15 20:28 — 2026-08-03 20:28 | 1.209.599 | 100% | 100% | 100% | 0% |

Gli archivi ufficiali Binance sono verificati con i file `.CHECKSUM`. OI, bid/ask storico, L2 e liquidazioni non vengono inventati: l’OI è opzionale e non fa parte delle 45 feature obbligatorie del modello quando manca copertura comune.

## Eventi e outcome

- 6.834 eventi indipendenti: 3.326 BTC e 3.508 ETH.
- 3.682 LONG e 3.152 SHORT.
- Una riga per evento; nessuna moltiplicazione per configurazione.
- 2.274 candidati respinti per stop economicamente non valido.
- 6.834/6.834 outcome costruiti; zero violazioni temporali e zero violazioni della monotonicità dei costi.
- Exit: 4.850 stop, 1.735 invalidazioni flow, 232 timeout, 17 stop gap.
- 729 barre ambigue stop/target risolte sempre con stop, come worst case.
- Base 4 bps: EV −3,8222 bps, PF 0,8122, 35,43% positivi.
- Stress 8 bps: EV −7,8222 bps, PF 0,6591.
- Variante TP1/flow: EV −3,8565 bps, PF 0,7837.
- Oracle ex-post fra le sole due exit preregistrate: +0,1260 bps; è una diagnostica, non una policy utilizzabile.

Il base negativo non ha bloccato arbitrariamente il meta-labeling: la specifica V25 autorizza il modello a cercare un sottoinsieme positivo, e il training è stato eseguito.

## Nested walk-forward e risultato

- 5 outer fold: 52 settimane train, 8 calibrazione, 8 test, step 8 settimane, embargo 6 ore.
- 3 split cronologici interni per selezione.
- Ridge/Logistic e ranking deterministico come baseline.
- Otto configurazioni XGBoost come unico challenger, GPU `cuda:0`, early stopping soltanto sulla validation interna.
- Quantili separati Q25/Q50/Q75.
- XGBoost ha battuto le baseline interne in tutti i cinque fold; è stata scelta sempre la variante flow.
- 2.071 eventi OOS; zero trade hanno superato tutti i gate.
- `expected_gross_bps >= 12`: 14 eventi.
- `calibrated_probability >= threshold`: 0 eventi; la calibrazione non ha trovato almeno 20 trade con PF ≥ 1, quindi la soglia fail-closed è 1,0.
- `Q25 >= -2 bps`: 0 eventi; il miglior Q25 OOS è −13,6983 bps.
- Top 10% per score: EV −4,0114 bps.
- Top 5%: EV −0,1035 bps.
- Top 1%: EV +2,7347 bps, ma solo 21 eventi; non soddisfa numerosità, stabilità o gate prudenziali.

Controlli multipli:

- Stationary Bootstrap SPA contro FLAT: `p = 0,562` — nessuna evidenza di superiorità.
- PBO diagnostico: `0,1825`; da solo non autorizza la policy.
- DSR: `0`, perché non esiste una sequenza di trade accettata.
- Holdout BTC sigillato: 2026-05-11 11:30 — 2026-08-03 11:30 UTC, 346 righe escluse, `opened=false`.
- Train BTC → test ETH: zero eventi selezionati in tutti i fold.

## Gate

Superati:

- almeno 500 eventi OOS;
- data audit;
- event dataset;
- outcome audit;
- causalità, costi e holdout sigillato.

Falliti:

- almeno 150 trade OOS;
- expectancy positiva;
- PF ≥ 1,10;
- maggioranza dei fold positiva;
- concentrazione PnL ≤ 40%;
- tutti i gate paper avanzati.

Il bundle non è operativo perché il modello non dimostra un meta-edge OOS e nessun candidato supera contemporaneamente edge lordo, probabilità calibrata e downside Q25. Abbassare questi gate dopo aver visto il test sarebbe data snooping; non è stato fatto.

## File di codice

- `src/adaptive_bot/hybrid_policy_v25/protocol.py` — protocollo, hash, freeze e stato.
- `src/adaptive_bot/hybrid_policy_v25/data.py` — archivi ufficiali, checksum, disponibilità e feature causali.
- `src/adaptive_bot/hybrid_policy_v25/events.py` — evento trend/VWAP, stop, simulatore e outcome.
- `src/adaptive_bot/hybrid_policy_v25/models.py` — baseline, XGBoost GPU, quantili, calibrazione, walk-forward e controlli statistici.
- `src/adaptive_bot/hybrid_policy_v25/bundle.py` — bundle, manifest, scoring e divieto live.
- `src/adaptive_bot/hybrid_policy_v25/reporting.py` — audit JSON/Markdown.
- `src/adaptive_bot/hybrid_policy_v25/cli.py` — comandi V25.
- `tests/unit/test_hybrid_policy_v25.py` — test mirati.
- `scripts/run_hybrid_v25.ps1` e `scripts/watch_hybrid_v25.ps1` — worker e monitor.

## Artefatti

- `data/reports/ml_hybrid_v25_data_audit.json`
- `data/reports/ml_hybrid_v25_candidate_audit.json`
- `data/reports/ml_hybrid_v25_outcome_audit.json`
- `data/reports/ml_hybrid_v25_model_audit.json`
- `data/reports/ml_hybrid_v25_policy_audit.json`
- `data/reports/ml_hybrid_v25.status.json`
- `data/ml/hybrid_v25/events.parquet`
- `data/ml/hybrid_v25/events_with_outcomes.parquet`
- `data/ml/hybrid_v25/oos_predictions.parquet`
- `data/ml/hybrid_v25/oos_decisions.parquet`
- `data/models/expert_policy/v25/protocol.json`
- `data/models/expert_policy/v25_research_bundle/bundle.joblib`
- `data/models/expert_policy/v25_research_bundle/manifest.json`

## Dipendenze

Nessuna dipendenza aggiunta. Sono state riutilizzate NumPy, pandas, scikit-learn, XGBoost, `arch` e le utility già presenti. CatBoost/LightGBM/PyTorch/TensorFlow non sono stati aggiunti.

## Verifiche eseguite

- `ruff check src tests`: passato.
- `mypy src/adaptive_bot/hybrid_policy_v25 --ignore-missing-imports`: passato.
- `pytest tests/unit/test_hybrid_policy_v25.py -q`: 10 passati.
- `pytest -q`: 256 passati, 2 falliti per componenti preesistenti estranei alla V25:
  - health check Hypothesis lento in `tests/property/test_risk_properties.py`;
  - varianti dashboard inattese in `tests/unit/test_data_backtest.py`.
- `mypy src --ignore-missing-imports`: tre errori preesistenti `unused-ignore` in `gpu_features.py`, `expert_policy.py` e `hybrid_policy_v9.py`; nessuno nella V25.
- Hash implementazione congelato: verificato con gli stessi percorsi assoluti usati dalla CLI.
- Bundle `shadow`: caricato e manifest verificato.
- Bundle `live`: rifiutato con `PermissionError`.

Le failure globali non sono state corrette perché la richiesta vietava modifiche a componenti estranei al training V25.
