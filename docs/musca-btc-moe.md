# Musca BTC Mixture of Experts — protocollo preregistrato

Data di congelamento: 2026-08-10

## Obiettivo

Apprendere una policy intraday BTC su Binance `BTCUSDT` USD-M senza imporre regole di ingresso
FADE/FOLLOW. Il training osserva ogni stato causale a cadenza di un minuto, usando il flusso
ufficiale Binance aggregato a cinque secondi, e apprende:

- direzione LONG o SHORT;
- orizzonte 1, 5, 15, 60 minuti o 6 ore;
- primo target, secondo target, stop e trailing dipendenti dallo stato;
- probabilità di risultato netto positivo ed EV netta dopo costi;
- quale esperto ascoltare nel regime corrente.

VWAP rolling 5/15/60/240 minuti resta un centro e una famiglia di feature, non determina da solo
la direzione. `FLAT` vale zero e non è contato come profitto. Una policy può accettare singoli
trade negativi; deve essere positiva soltanto in aggregato OOS.

I cinque passaggi del sistema sono congelati così:

1. costruzione causale dello stato BTC ogni minuto;
2. addestramento degli esperti specializzati per vista e durata;
3. generazione delle loro previsioni esclusivamente OOF;
4. apprendimento del gating che decide quali esperti ascoltare e quale azione preferire;
5. replay sequenziale con target, stop, trailing, costi e una sola posizione.

Il report registra anche quanta importanza il gate assegna realmente alle uscite degli esperti e
quali venti segnali usa di più. In questo modo “ascoltare gli esperti” è verificabile, non soltanto
una descrizione dell'architettura.

## Dati e disponibilità

Si usano gli archivi ufficiali Binance USD-M `aggTrades` a cinque secondi, verificati con il
checksum pubblicato da Binance, da gennaio 2025 a luglio 2026. Il contesto causale a un minuto
aggiunge VWAP, momentum, volatilità, volume, spot/perpetual basis, mark, funding e open interest.
Il flusso a cinque secondi aggiunge imbalance degli aggressori, persistenza, intensità, velocità e
assorbimento. Ogni feature deve avere `available_at` non successivo alla decisione. L'ingresso
avviene nel primo bucket da cinque secondi successivo. Valori mancanti non vengono sostituiti con
zero. Un intervallo assente in un archivio `aggTrades` il cui checksum ufficiale è valido viene
rappresentato esplicitamente come `no_trade_bucket`: volume e conteggio sono realmente zero e
OHLC resta all'ultimo prezzo già osservato. Non è interpolazione; distingue causalmente “nessun
trade” da “dato non disponibile”. Una vera discontinuità della fonte resta fail-closed.

Il final holdout comincia il 10 agosto 2026, dopo la preregistrazione, e richiede nuovi dati futuri.
Non viene aperto dal training storico. Gli intervalli precedenti sono ricerca cronologica, non una
conferma finale indipendente:

- OOF meta-fit: aprile-dicembre 2025, con base train precedente e tre fold trimestrali;
- confronto Ridge/XGBoost: gennaio 2026;
- calibrazione: febbraio 2026;
- selezione della copertura: marzo-aprile 2026;
- audit storico congelato: maggio-luglio 2026;
- final holdout futuro: dal 10 agosto 2026.

Il purge è 6 ore. Il training di ogni previsione usa esclusivamente dati precedenti.

## Esperti generati

Il pool finale contiene 125 componenti XGBoost GPU: cinque bootstrap temporali per ciascuna delle
25 combinazioni `orizzonte × vista`.

Orizzonti: 1, 5, 15 e 60 minuti e 6 ore. Gli orizzonti sono alternative apprese, non scadenze
imposte a ogni trade: target, stop o trailing possono chiudere prima.

Viste apprese:

1. stato completo;
2. struttura VWAP e volatilità;
3. momentum, efficienza e posizione nel range;
4. taker flow, volume, trade intensity, assorbimento e price action a 5 secondi;
5. regime, open interest, basis, funding e ora.

Altri 30 modelli quantile apprendono, per lato e orizzonte, mediana e 75° percentile
dell'escursione favorevole e 75° percentile dell'escursione avversa. Insieme generano target e
stop entro soli limiti di sicurezza, senza una griglia di segnali manuali.

Il target supervisionato del gating è il rendimento terminale netto dell'azione allo specifico
orizzonte. La selezione della copertura e l'audit economico non usano quel terminale teorico:
riproducono invece in sequenza sul percorso a cinque secondi target parziali, stop e trailing.
Questo mantiene il training trattabile senza sostituire il risultato economico con un'etichetta
irrealizzabile.

## Gating e meta-modello

Le previsioni OOF degli esperti diventano input di un gating model insieme al solo contesto di
regime. Il gating non riceve nuovamente tutte le feature alpha: in questo modo deve scegliere quali
esperti ascoltare invece di riapprendere direttamente il rendimento. Ridge/logistic è il champion
EV predefinito; un ensemble di cinque XGBoost GPU può sostituirlo soltanto se migliora sulle stesse
righe cronologiche errore EV, Brier score e regret decisionale. Un ensemble separato di cinque
`XGBRanker` con loss pairwise può determinare l'ordinamento delle otto azioni soltanto se riduce il
regret OOS rispetto al champion EV.

Per ogni decisione il gating confronta dieci azioni: LONG/SHORT per cinque orizzonti. La policy
sceglie l'azione con EV calibrata più alta e può restare FLAT. Il livello di copertura è scelto
soltanto su marzo-aprile e congelato prima dell'audit.

L'uscita simulata usa:

- 50% al primo target quantile;
- restante 50% al secondo target quantile;
- stop quantile strutturale;
- dopo il primo target, stop mai allargato e trailing basato sull'escursione avversa prevista;
- timeout all'orizzonte scelto;
- stop prevalente se stop e target sono toccati nello stesso bucket da cinque secondi.

Queste sono meccaniche esecutive; direzione, durata e livelli provengono dai modelli. Gli esperti
sono modelli numerici specializzati, non agenti linguistici: tutti vengono interrogati allo stesso
timestamp e il gating impara OOS quale combinazione ascoltare nel regime osservato.

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
