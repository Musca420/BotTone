# Musca BTC Auto-MoE — risultato congelato

Data del run: 2026-08-10  
Protocol hash: `cbc1fa33352722e8a6e02b698c8532a8cbe9d7607aebf1bdb38534aafd32af4c`  
Verdetto: `NO_DEPLOYABLE_POLICY`

## Risposta alla nuova domanda

Il training è stato realmente separato in due fasi. Non sono stati imposti 125 esperti.

1. Un generatore GPU ha appreso regole/foglie su tutto il 2025.
2. Gennaio e febbraio 2026 hanno selezionato e congelato la libreria.
3. Soltanto dopo, un gate ha imparato a scegliere gli esperti usando marzo–aprile, con maggio
   separato per la calibrazione.
4. Giugno e luglio sono stati letti soltanto dall'audit finale.

Sono stati valutati 9.831 candidati. 235 hanno superato i controlli economici prima della
diversità e 96 esperti distinti sono stati congelati. Il numero finale è quindi un risultato dei
dati, non un parametro del training.

## Libreria scoperta

| Azione | Foglie valide | Esperti congelati |
|---|---:|---:|
| LONG 1 minuto | 0 | 0 |
| SHORT 1 minuto | 0 | 0 |
| LONG 5 minuti | 0 | 0 |
| SHORT 5 minuti | 0 | 0 |
| LONG 15 minuti | 0 | 0 |
| SHORT 15 minuti | 18 | 5 |
| LONG 1 ora | 38 | 19 |
| SHORT 1 ora | 20 | 15 |
| LONG 6 ore | 22 | 13 |
| SHORT 6 ore | 137 | 44 |

La discovery ha quindi respinto automaticamente le micro-operazioni da uno e cinque minuti: non
coprivano stabilmente i 9 bps round-trip nei due mesi successivi al fit annuale. Non sono state
eliminate con una regola manuale. La prova SPA della libreria durante il primo periodo del gate ha
prodotto `p = 0,044`, quindi esisteva evidenza preliminare, ma questa non è rimasta significativa
nell'audit finale.

## Gate

Aprile 2026 ha confrontato i modelli sulle stesse 33.962 azioni:

| Modello | MAE EV | Brier | Regret |
|---|---:|---:|---:|
| Ridge | 76,453 bps | 0,35256 | 42,902 bps |
| XGBoost GPU | 81,238 bps | 0,41034 | 41,961 bps |

XGBoost ha migliorato soltanto il regret. Poiché il protocollo richiedeva di migliorare tutte e
tre le metriche, Ridge è rimasto champion. Il gate ha poi visto 91.221 azioni di esperti in giugno
e luglio e ha trovato almeno un candidato con EV calibrata positiva nel 49,47% dei timestamp. Il
vincolo di una posizione ha trasformato questi candidati in 178 trade.

## Audit economico giugno–luglio

- 178 trade, 2,92 al giorno;
- expectancy netta `−0,031 bps` per trade;
- profit factor `0,9992`;
- win rate 53,93%;
- movimento lordo medio `+8,920 bps`;
- funding medio `+0,049 bps`;
- costo round-trip `9 bps`;
- max drawdown 12,85%;
- bootstrap LCB 95% `−15,49 bps`;
- SPA finale `p = 0,482`;
- zero violazioni del budget di rischio.

Rispetto al precedente sistema da 125 componenti (`−15,22 bps`, PF 0,671, 1,85 trade/giorno),
la separazione discovery/gating ha quasi raggiunto il pareggio ed è più frequente. Non ha però
prodotto un vantaggio stabile dopo costi.

### Stabilità mensile

| Mese OOS | Trade/giorno | Expectancy | PF | Win rate |
|---|---:|---:|---:|---:|
| Giugno 2026 | 4,63 | +5,632 bps | 1,162 | 56,83% |
| Luglio 2026 | 1,26 | −20,215 bps | 0,580 | 43,59% |

Giugno era economicamente positivo a costi reali e anche a costi 1,5× (`+1,132 bps`), ma il suo
LCB era ancora negativo. Luglio ha perso già al lordo dei costi (`−11,353 bps`): il problema non è
stato la sola commissione, bensì un cambio di regime che gli esperti SHORT a lunga durata non
hanno generalizzato.

### Durata e lato

- 6 ore: 118 trade, `+5,751 bps`, PF 1,148;
- 1 ora: 55 trade, `−9,365 bps`, PF 0,733;
- 15 minuti: 5 trade, `−33,801 bps`, PF 0,179;
- LONG: 17 trade, `+3,376 bps`, PF 1,135;
- SHORT: 161 trade, `−0,391 bps`, PF 0,990.

Il segnale migliore è stato quello a sei ore, ma non ha raggiunto PF 1,15 e non è stabile tra i
mesi. Selezionarlo ora isolatamente significherebbe usare giugno/luglio per correggere il modello,
violando il protocollo.

## Gate falliti

Sono falliti: 300 trade, 3 trade/giorno, expectancy positiva, PF 1,15, maggioranza dei giorni
positiva, drawdown 10%, LCB positivo e SPA 0,05. È passato soltanto il gate di rischio operativo:
nessuna violazione del budget.

Il bundle è quindi `RESEARCH_ONLY`; `orders_enabled`, paper e live restano `false`. `FLAT` non è
stato contato come profitto e nessun risultato è stato forzato.

## Artefatti

- protocollo: `docs/musca-btc-auto-moe.md`;
- questo risultato: `docs/musca-btc-auto-moe-results.md`;
- discovery, gating e audit: `src/adaptive_bot/musca_btc_auto_moe.py`;
- test: `tests/unit/test_musca_btc_auto_moe.py`;
- monitor: `scripts/run_musca_btc_auto_moe_training.ps1`;
- catalogo completo: `data/ml/musca_btc_auto_moe/candidates.parquet`;
- libreria: `data/ml/musca_btc_auto_moe/expert_library.joblib`;
- azioni forward: `data/ml/musca_btc_auto_moe/expert_actions.parquet`;
- trade audit: `data/ml/musca_btc_auto_moe/audit_trades.parquet`;
- report macchina: `data/reports/musca_btc_auto_moe.json`;
- bundle non operativo: `data/models/musca_btc_auto_moe/research_bundle.joblib`.

## Verifiche

- Ruff: superato;
- mypy strict: superato;
- 23 test mirati: superati;
- future holdout letto: 0 righe;
- vecchio protocollo da 125 componenti invariato;
- nessuna nuova dipendenza;
- nessun ordine paper/live autorizzato.
