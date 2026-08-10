# Musca BTC Mixture of Experts — protocollo preregistrato

Data di congelamento: 2026-08-10

## Obiettivo

Apprendere una policy intraday BTC su Binance `BTCUSDT` USD-M senza imporre regole di ingresso
FADE/FOLLOW. Il training osserva ogni stato causale a cadenza di cinque minuti e apprende:

- direzione LONG o SHORT;
- orizzonte 5, 15, 30 o 60 minuti;
- primo target, secondo target, stop e trailing dipendenti dallo stato;
- probabilità di risultato netto positivo ed EV netta dopo costi;
- quale esperto ascoltare nel regime corrente.

VWAP rolling 5/15/60/240 minuti resta un centro e una famiglia di feature, non determina da solo
la direzione. `FLAT` vale zero e non è contato come profitto. Una policy può accettare singoli
trade negativi; deve essere positiva soltanto in aggregato OOS.

## Dati e disponibilità

Si usano esclusivamente archivi ufficiali Binance perpetual e spot a un minuto già verificati con
checksum, dal 15 aprile 2024 al 3 agosto 2026. Le feature comprendono prezzo, VWAP, momentum,
volatilità, volume, taker flow, trade count, spot/perpetual basis, mark, funding e open interest.
Ogni feature deve avere `available_at` non successivo alla decisione. L'ingresso avviene all'open
del minuto seguente. Valori mancanti non vengono sostituiti con zero.

Il final holdout comincia il 10 agosto 2026, dopo la preregistrazione, e richiede nuovi dati futuri.
Non viene aperto dal training storico. Gli intervalli precedenti sono ricerca cronologica, non una
conferma finale indipendente:

- OOF meta-fit: gennaio-dicembre 2025;
- confronto Ridge/XGBoost: gennaio 2026;
- calibrazione: febbraio 2026;
- selezione della copertura: marzo-aprile 2026;
- audit storico congelato: maggio-3 agosto 2026;
- final holdout futuro: dal 10 agosto 2026.

Il purge è 60 minuti. Il training di ogni previsione usa esclusivamente dati precedenti.

## Esperti generati

Il pool finale contiene 100 componenti XGBoost GPU: cinque bootstrap temporali per ciascuna delle
20 combinazioni `orizzonte × vista`.

Orizzonti: 5, 15, 30 e 60 minuti.

Viste apprese:

1. stato completo;
2. struttura VWAP e volatilità;
3. momentum, efficienza e posizione nel range;
4. taker flow, volume, trade intensity e price action;
5. regime, open interest, basis, funding e ora.

Altri 24 modelli quantile apprendono, per lato e orizzonte, mediana e 75° percentile
dell'escursione favorevole e 75° percentile dell'escursione avversa. Insieme generano target e
stop entro soli limiti di sicurezza, senza una griglia di segnali manuali.

## Gating e meta-modello

Le previsioni OOF degli esperti diventano input di un gating model. Ridge/logistic è il champion
predefinito; un ensemble di cinque XGBoost GPU può sostituirlo soltanto se migliora sulle stesse
righe cronologiche errore EV, Brier score e regret decisionale.

Per ogni decisione il gating confronta otto azioni: LONG/SHORT per quattro orizzonti. La policy
sceglie l'azione con EV calibrata più alta e può restare FLAT. Il livello di copertura è scelto
soltanto su marzo-aprile e congelato prima dell'audit.

L'uscita simulata usa:

- 50% al primo target quantile;
- restante 50% al secondo target quantile;
- stop quantile strutturale;
- dopo il primo target, stop mai allargato e trailing basato sull'escursione avversa prevista;
- timeout all'orizzonte scelto;
- stop prevalente se stop e target sono toccati nella stessa candela.

Queste sono meccaniche esecutive; direzione, durata e livelli provengono dai modelli.

## Economia e rischio

Scenario operativo iniziale: Binance USD-M VIP0 taker/taker, 10 bps round-trip più 1 bp di riserva
slippage, da sostituire nel paper con la commissione account restituita dall'endpoint ufficiale.
Il primo target deve coprire il costo reale e almeno 2 bps netti. Costi 1,5× e 2× sono soltanto
diagnostici.

Simulazione: equity 10.000 USDT, rischio massimo 1% per trade, leva massima 10×, una posizione
BTC, nessun averaging down e nessuna liquidazione ammessa.

Gate dell'audit storico:

- almeno 300 trade e almeno 3 trade/giorno;
- expectancy netta e bootstrap LCB 95% positive;
- profit factor almeno 1,15;
- drawdown massimo 10%;
- maggioranza dei giorni di calendario positiva;
- SPA contro FLAT con `p <= 0,05`;
- nessun look-ahead, sovrapposizione o violazione del rischio.

Un pass storico produce soltanto `RESEARCH_ONLY`, mai live. L'operatività richiede almeno dieci
giorni e 100 trade del final holdout futuro, oltre ai gate completi.

## Output

- `data/ml/musca_btc_moe/matrix.parquet`;
- `data/ml/musca_btc_moe/checkpoints/`;
- `data/models/musca_btc_moe/research_bundle.joblib`;
- `data/reports/musca_btc_moe.json`;
- `data/reports/musca_btc_moe.status.json`.

Fonti ufficiali: [Binance Public Data](https://github.com/binance/binance-public-data/blob/master/README.md),
[XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/),
[XGBoost quantile regression](https://xgboost.readthedocs.io/en/stable/tutorials/quantile.html)
e [XGBoost learning to rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).
