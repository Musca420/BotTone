# Musca Altcoin Micro — protocollo Binance

Ultimo aggiornamento: 2026-08-10.

## Obiettivo

Valutare una sola pipeline riproducibile su tre perpetual Binance USD-M indipendenti:
`ETHUSDT`, `XRPUSDT` e `DOGEUSDT`. L'obiettivo è massimizzare la frequenza di operazioni
intraday sotto vincoli economici, non forzare un numero di trade né garantire un profitto ogni
giorno. BTCUSDT resta fuori dalle azioni e viene usato esclusivamente come contesto causale.

La linea BTC/Musca V2/V5 resta conservata e immutata. Denaro reale è disabilitato.

## Ipotesi preregistrata

I rendimenti e la volatilità delle principali criptovalute presentano dipendenze e correlazioni
dinamiche. La policy non assume che ogni altcoin segua sempre Bitcoin. Per ogni decisione misura:

- rendimenti BTC a 1, 5, 15 e 60 minuti;
- volatilità e taker flow BTC;
- beta e correlazione altcoin/BTC stimate soltanto con osservazioni già disponibili;
- rendimento residuo dell'altcoin rispetto a BTC;
- concordanza, divergenza e shock BTC.

Uno shock BTC può quindi sostenere, contraddire o invalidare un'azione; non è una conferma fissa.

Fonti metodologiche:

- Binance Public Data, archivi mensili con checksum e campi OHLCV, trade count e taker-buy:
  https://github.com/binance/binance-public-data/blob/master/README.md
- Katsiampa (2019), correlazioni condizionali crypto dinamiche e asimmetriche:
  https://doi.org/10.1016/j.ribaf.2019.06.004
- Sensoy et al. (2021), spillover ad alta frequenza tra criptovalute:
  https://doi.org/10.1080/00036846.2021.1899119

## Dati e cronologia sigillata

Si usano esclusivamente klines ufficiali Binance USD-M a un minuto, verificate con SHA-256.
Ogni feature è disponibile solo dopo la chiusura della candela; l'ingresso avviene non prima
dell'open del minuto successivo.

- fit: 2026-01-01 / 2026-03-01;
- calibrazione: 2026-03-01 / 2026-03-15;
- confronto Ridge-XGBoost: 2026-03-15 / 2026-04-01;
- selezione della copertura: 2026-04-01 / 2026-05-01;
- audit riutilizzabile: 2026-05-01 / 2026-07-01;
- holdout sigillato: dal 2026-07-01, mai letto in questa ricerca.

Il purge è pari al massimo orizzonte dell'azione. Mancanze, timestamp futuri o disallineamenti
BTC/altcoin vengono esclusi senza imputazione a zero.

## Stato e azioni

Le feature dell'altcoin includono rendimenti multi-orizzonte, ATR/volatilità, taker imbalance,
volume relativo, trade intensity, VWAP giornaliero e rolling, distanza e pendenza VWAP. Le feature
BTC sono affiancate a beta, correlazione e rendimenti residui.

Il modello non valuta ogni minuto. Genera una decisione soltanto sul fronte iniziale di uno di
questi eventi causali: attraversamento, touch o rifiuto del rolling VWAP; restart momentum con
flow; shock BTC con correlazione osservata; inversione del rendimento residuo; impulso di volume.
Eventi entro tre minuti vengono accorpati. Questo evita di diluire il segnale con minuti privi di
setup e conserva comunque una frontiera abbastanza ampia per cercare più trade quotidiani.

Azioni LONG e SHORT preregistrate:

| Piano | Orizzonte | Stop causale | Target minimo |
|---|---:|---:|---:|
| MICRO_5 | 5 min | 1,5 ATR, 12–30 bps | max(1,5× costo, 1,4× stop) |
| MICRO_15 | 15 min | 2 ATR, 15–40 bps | max(1,5× costo, 1,5× stop) |
| MICRO_30 | 30 min | 2,5 ATR, 18–50 bps | max(1,5× costo, 1,6× stop) |
| MICRO_60 | 60 min | 3 ATR, 22–65 bps | max(1,5× costo, 1,75× stop) |

Se stop e target sono toccati nello stesso minuto, vince lo stop. Il risultato a timeout usa il
close osservato. Una sola posizione per asset; nessun averaging down.

Costi taker round-trip preregistrati, comprensivi di una riserva non-fee crescente con la minore
liquidità: ETH 9 bps, XRP 10,5 bps, DOGE 11,5 bps. Nel paper saranno sostituiti da commissione
account, spread, profondità e slippage osservati; i costi 2× sono soltanto diagnostici.

## Modelli e selezione

Ridge è baseline e XGBoost `hist` su CUDA è challenger. Entrambi vedono le stesse split e le
stesse azioni. Il modello stima rendimento lordo e probabilità di risultato netto positivo; la
regressione viene calibrata cronologicamente. XGBoost diventa champion soltanto se migliora Ridge
nelle decisioni OOS, non perché usa la GPU.

La regressione serve a ordinare gli eventi, non a imporre un limite prudenziale positivo a ogni
singolo trade. La regolarizzazione può contrarre tutte le stime verso la media. Un percentile può
essere negoziato soltanto quando la sequenza completa, scelta su dati precedenti, supera expectancy,
PF, drawdown e bootstrap; altrimenti la policy intera resta FLAT. Questo consente normali trade
perdenti senza attribuire un profitto a FLAT.

La copertura viene scelta una sola volta su aprile, massimizzando i trade/giorno tra i punti
preregistrati che superano tutti i gate. La soglia viene poi congelata per maggio-giugno.

Gate per autorizzare un profilo paper:

- almeno 3 trade per giorno di calendario e 100 trade di audit;
- expectancy netta e bootstrap lower bound al 95% positivi;
- profit factor almeno 1,15;
- drawdown massimo 8%;
- maggioranza dei giorni attivi positiva;
- nessun look-ahead, sovrapposizione o violazione del rischio.

Un asset che fallisce resta `RESEARCH_ONLY_FLAT` senza impedire agli altri di passare. Testnet
serve a verificare API e stato degli ordini; la redditività è valutata con dati reali Binance e
un simulatore conservativo. Nessun risultato apre automaticamente l'holdout o autorizza live.
