# Musca BTC Binance — canonical policy challenger

## Correzione dello spazio d'azione — 2026-08-11

I dieci piani ereditati (`LONG/SHORT x 1m/5m/15m/1h/6h`) non sono piu' l'universo
operativo. I cinque orizzonti rimangono soltanto ancore di previsione per gli esperti OOF.

Per ogni stato e lato, un gate robusto combina le previsioni di tutti gli esperti per vista e
orizzonte. Da quella combinazione genera un piano parametrizzato con:

- durata dinamica tra 60 e 21.600 secondi;
- target 1 e target 2 dinamici;
- quota dinamica da chiudere al primo target;
- stop iniziale e trailing non allargabile;
- ID deterministico del piano e contributori osservabili.

Il critic contestuale viene addestrato dopo che il piano e' stato simulato con il percorso a un
secondo. Le foglie del critic non definiscono il piano e hanno un ID separato. La matrice precedente
resta soltanto un controllo storico congelato. La motivazione, le fonti primarie e i criteri di
falsificazione sono in `docs/musca-btc-action-space-audit.md`.

## Correzione precedente (contesto storico, spazio d'azione sostituito)

I punti seguenti descrivono il run precedente e restano nel registro anti-ripetizione. Non
descrivono piu' il generatore di piani attivo:

1. every outer-fold fit generates thousands of context experts from all XGBRF tree leaves for
   each LONG/SHORT and managed horizon; no terminal-return economics filter is used;
2. every leaf is evaluated on the exact one-second TP1/TP2/stop/trailing/timeout outcome already
   used by replay;
3. exclusion happens only after managed evaluation and only when support is below 100 fit
   opportunities; losing experts remain valid candidates because they can describe regimes to
   avoid;
4. probability, conditional-return and calibration heads are trained independently for LONG and
   SHORT;
5. each decision identifies a fold-local expert, side, horizon and dynamic management plan;
6. the four-week selection window maximizes compounded net equity subject only to the Risk Engine
   and the 8% drawdown ceiling; it does not apply final PF, LCB or positive-day gates;
7. LCB, PF, drawdown, active-day, SPA/Reality Check, PBO and DSR controls are applied only to the
   concatenated, untouched outer tests;
8. losing trades and losing days are permitted. An entry needs positive calibrated EV and risk
   approval, not certainty that the individual trade will win.

The selector no longer reads the old `expert_*` columns derived from terminal outcomes. Fold
catalogs and resumable models are written under `data/ml/musca_btc_policy/fold_experts/`.

Promotion requires at least 300 aggregate OOS trades and at least 100 OOS trades for an enabled
side. These are sample-size safeguards, not a claim that crypto returns must be perfectly stable.

## Scope and status

This is the only active challenger after the frozen Auto-MoE discovery control. It trades only
Binance USD-M `BTCUSDT`; ETH, Bybit, OKX and Bitunix are outside this training. It can produce a
`RESEARCH_PAPER_READY` bundle, but it cannot enable real orders. Confirmation requires data after
the sealed holdout boundary (`2026-08-10T00:00:00Z`).

The frozen control remains in:

- `src/adaptive_bot/musca_btc_auto_moe.py`;
- `data/reports/musca_btc_auto_moe.json`;
- `data/models/musca_btc_auto_moe/research_bundle.joblib`.

The challenger never writes those files and records the control report SHA-256 before and after a
run.

## Structural correction

The former generic MoE trained the selector on terminal returns, then replayed selected actions
with TP1, TP2, stop and trailing. Auto-MoE also discarded a leaf on terminal economics before its
managed path was evaluated. The canonical pipeline removes that mismatch:

1. it reads the existing out-of-fold action plans, without their old return labels;
2. it reconstructs official Binance aggregate trades at one-second resolution;
3. it enters on the first observed trade after the decision;
4. it computes TARGET-before-STOP, STOP-before-TARGET or TIMEOUT and the exact managed return;
5. that same return is used by training, walk-forward audit and sequential replay.

No missing trade, spread, queue position or maker fill is simulated. Taker execution is the only
baseline. The signed Binance `commissionRate` is used when account credentials are configured;
otherwise the explicitly labelled official configuration fallback is used. The decision cost is
exactly the observed/configured taker commission for entry and exit; no unsupported fixed spread
or slippage number is added. Cost stress at 1.5x and 2x is diagnostic and never confused with 10x
leverage.

## Models and decisions

The statistical heads estimate:

- `P(TARGET before STOP)` — exposed as `target_probability`;
- `P(STOP before TARGET)`;
- `P(TIMEOUT)`;
- net return conditional on each event;
- MFE/MAE residual quantiles and expected target time;
- calibrated net EV after actual 1x Binance costs.

Logistic regression plus Ridge is the default champion. XGBoost CUDA is promoted only if it is
strictly better on the same chronological inner audit in multiclass Brier score, EV calibration
error, EV MAE and decision regret.

The controller exposes `WAIT`, `ENTER_LONG`, `ENTER_SHORT`, `HOLD`, `CLOSE` and `TIGHTEN_STOP`.
It allows one position at a time and multiple sequential trades. A position is not closed at UTC
midnight. Daily P&L and residual daily risk are controller state and the daily loss limit is a
Risk Engine veto. Individual losing trades and losing days are allowed.

## Validation

Every outer fold has an expanding fit, two weeks of inner probability calibration, a subsequent
two-week model audit, four weeks of final calibration, four weeks of policy/equity selection and
four weeks of test. Every boundary purges on the actual managed exit timestamp. Thresholds are
preregistered. LONG and SHORT choose their thresholds independently by maximum compounded net
equity while respecting risk; no final statistical gate or trade quota is imposed on this short
selection window. A fold that selects no positive-utility threshold is a valid FLAT period.

The report includes daily and weekly block-bootstrap lower bounds, the complete threshold
frontier, SPA/Reality Check over that frontier, PBO across chronological slices and DSR adjusted
for the global historical trial count. All previously inspected periods and all V7–V25/FT runs
are declared contaminated in `data/ml/musca_btc_policy/research_registry.json`.

Verdicts are intentionally specific:

- `NO_ECONOMIC_ACTION_SET`;
- `NO_PREDICTABLE_EDGE`;
- `NO_CALIBRATED_POLICY`;
- `NO_STABLE_OOS_POLICY`;
- `RESEARCH_PAPER_READY`;
- `HOLDOUT_CONFIRMED` (not produced by this command because the holdout stays sealed).

## Commands

```powershell
uv run adaptive-bot musca-btc-policy-train --resume
uv run adaptive-bot musca-btc-policy-status --watch
```

The status file reports the exact month/action/fold, completed blocks, PID, CPU time, process RAM,
GPU/VRAM and an estimated remaining time. Monthly one-second sources and state-action partitions
are atomic checkpoints, so `--resume` does not repeat completed work.

## Outputs

- `src/adaptive_bot/musca_btc_policy.py` — canonical labels, probability heads, controller and
  scientific audit;
- `tests/unit/test_musca_btc_policy.py` — causal, path, semantics, sequential and CPU/GPU checks;
- `data/ml/musca_btc_policy/state_actions/` — canonical state-action partitions;
- `data/ml/musca_btc_policy/research_registry.json` — global contamination/trial registry;
- `data/ml/musca_btc_policy/audit_trades.parquet` — OOS trades;
- `data/ml/musca_btc_policy/audit_decisions.parquet` — WAIT/ENTER/HOLD/CLOSE/TIGHTEN trace;
- `data/reports/musca_btc_policy.json` — complete reproducible report;
- `data/reports/musca_btc_policy.status.json` — live progress;
- `data/models/musca_btc_policy/research_bundle.joblib` — research-paper model, calibrators and
  selected threshold.

Fold-local expert checkpoints and their complete managed-outcome catalogs are stored in
`data/ml/musca_btc_policy/fold_experts/`.

## Anti-repetition ledger

The challenger does not repeat the terminal-label MoE, the terminal-prefilter Auto-MoE, the
hindsight daily oracle, or the UTC-bounded FQI. It does not add another model family or search
more Optuna trials. Any failure is classified at the first failed stage so that the next change
must address evidence rather than lower a gate.

Primary references used by the protocol:

- Binance USD-M commission rate: <https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/User-Commission-Rate>
- Binance public archives: <https://github.com/binance/binance-public-data/blob/master/README.md?plain=1>
- `arch` multiple comparisons: <https://bashtage.github.io/arch/multiple-comparison/multiple-comparison-reference.html>
- White Reality Check: <https://doi.org/10.1111/1468-0262.00152>
- Probability of Backtest Overfitting: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>
- Deflated Sharpe Ratio: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
