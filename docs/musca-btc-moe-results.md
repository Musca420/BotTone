# Musca BTC Mixture of Experts — risultato congelato

Data del run: 2026-08-10  
Protocol hash: `73563d1aed16e4f796d18d446dc9033946473429e48f18fe15c1a52ffc193ddd`  
Verdetto: `NO_HISTORICALLY_STABLE_BTC_MOE_ALPHA`

## Cosa è stato provato

Il run ha costruito una policy BTCUSDT Binance a cadenza di un minuto su dati `aggTrades` a
cinque secondi. A ogni stato sono state confrontate dieci azioni: LONG e SHORT con durata massima
di 1, 5, 15, 60 minuti o 6 ore. VWAP è una famiglia di feature e un benchmark, non una regola
obbligatoria di ingresso.

Il sistema comprende:

- 125 esperti XGBoost GPU: cinque viste, cinque orizzonti e cinque bootstrap temporali;
- 30 modelli quantile per escursione favorevole e avversa;
- Ridge/logistic come champion predefinito;
- cinque regressori, cinque classificatori e cinque ranker XGBoost come challenger;
- target, stop, parziale, trailing, timeout, funding osservato e replay con una posizione;
- 172 componenti finali nel bundle di ricerca.

Il gating ha assegnato il 44,76% della propria importanza alle uscite degli esperti. Non è quindi
corretto descrivere il fallimento come un gate che li ha ignorati.

## Dati realmente usati

- Binance USD-M BTCUSDT `aggTrades` ufficiali: gennaio 2025–luglio 2026;
- 9.970.560 bucket regolari da cinque secondi;
- 1.024 intervalli senza trade rappresentati causalmente a volume zero;
- nessun timestamp duplicato o fuori ordine;
- 830.341 stati validi nella matrice;
- 3.948.000 azioni OOF per il meta-fit;
- 3.049.200 azioni successive per audit, calibrazione e selezione;
- hash combinato dei Parquet e del contesto:
  `16d8aa8647e3190057a6db75f74aba2ee14b77ae1d3f586e7e39d84bbd01411d`.

Gli archivi sono stati verificati contro i checksum pubblicati da Binance. Il contesto a un
minuto contiene VWAP, prezzo, volatilità, taker flow, spot/perpetual basis, mark, open interest e
funding. Il funding entra nel risultato soltanto quando il trade attraversa l'evento osservato.

## Opportunità lorde

I movimenti non mancano. La percentuale di stati nei quali il percorso futuro supera il target
minimo di 11 bps in almeno una direzione è:

| Orizzonte | Stati con percorso > 11 bps | Miglior percorso mediano |
|---|---:|---:|
| 1 minuto | 12,05% | 4,12 bps |
| 5 minuti | 46,77% | 10,34 bps |
| 15 minuti | 77,11% | 18,66 bps |
| 1 ora | 96,82% | 38,50 bps |
| 6 ore | 99,99% | 101,26 bps |

Questo è un oracle non negoziabile: usa il futuro soltanto per dimostrare che esistono escursioni,
non che siano prevedibili prima dell'ingresso.

## Modelli e stabilità

Sul model-audit di gennaio 2026:

| Modello | MAE EV | Brier | Regret decisionale |
|---|---:|---:|---:|
| Ridge | 23,4729 bps | 0,155531 | 63,8948 bps |
| XGBoost | 23,4593 bps | 0,154947 | 63,9889 bps |
| Ranker | — | — | 72,1821 bps |

XGBoost ha migliorato MAE e Brier, ma non il regret sulle stesse righe. Per il protocollo
preregistrato Ridge è quindi rimasto champion.

Nel periodo di selezione marzo–aprile alcuni ranking XGBoost apparivano positivi:

- top 1% a 1 ora: +4,31 bps netti terminali;
- top 1% a 6 ore: +90,36 bps;
- top 5% a 6 ore: +13,27 bps.

Nel periodo successivo maggio–luglio non hanno generalizzato:

- top 1% a 1 ora: −20,04 bps;
- top 1% a 6 ore: −54,04 bps;
- solo il top 5% a 6 ore è rimasto +3,29 bps, senza stabilità tra coperture.

Scegliere ora quel singolo livello significherebbe riusare l'audit per il tuning.

## Replay economico

La calibrazione ha stimato EV positiva per l'azione migliore soltanto nell'1,41% degli stati di
selezione. Tutte le coperture hanno quindi prodotto gli stessi 159 trade dopo il vincolo `EV > 0`.

Selezione marzo–aprile:

- 159 trade, 2,61 al giorno;
- expectancy −8,33 bps;
- profit factor 0,794;
- win rate 55,97%;
- giorni positivi 39,34%;
- max drawdown 8,44%;
- bootstrap LCB 95% −17,74 bps;
- 2 violazioni del budget per gap oltre lo stop.

Il win rate sopra il 50% non è bastato perché la vincita mediana era +43,87 bps e la perdita
mediana −77,34 bps. Target e stop non sono però l'unica causa: anche il rendimento terminale
reale dei candidati con EV prevista positiva era −8,42 bps. Il segnale direzionale era già
instabile prima della gestione.

Audit maggio–luglio:

- 170 trade, 1,85 al giorno;
- expectancy −15,22 bps;
- profit factor 0,671;
- win rate 49,41%;
- giorni positivi 26,09%;
- max drawdown 21,32%;
- bootstrap LCB 95% −22,12 bps;
- SPA p-value 0,519;
- 2 violazioni del budget per gap.

Sono falliti tutti i gate economici e di rischio. Nessun ordine paper o live è autorizzato.

## Cosa si può e non si può concludere

Il training ha generato e interrogato numerosi esperti come richiesto, ma non ha trovato una
relazione causale stabile tra le feature disponibili e la direzione futura netta. Aggiungere altri
seed, trial o soglie sugli stessi intervalli non crea informazione e trasformerebbe la ricerca in
data snooping.

La prossima evidenza indipendente può provenire soltanto da:

1. nuovi dati successivi al 10 agosto 2026, mantenuti come holdout futuro;
2. informazione storica realmente nuova, per esempio L2/queue/latency completa e verificata, non
   ricostruita da candele o aggTrades;
3. un altro asset, con protocollo congelato prima di leggerne l'audit.

Non è scientificamente corretto promettere che uno di questi ingressi produrrà profitto. Il
risultato corrente è un bundle `RESEARCH_ONLY` e resta `FLAT`.

## File correlati

- protocollo: `docs/musca-btc-moe.md`;
- questo risultato: `docs/musca-btc-moe-results.md`;
- training, dataset, gating e audit: `src/adaptive_bot/musca_btc_moe.py`;
- test: `tests/unit/test_musca_btc_moe.py`;
- monitor: `scripts/run_musca_btc_moe_training.ps1`;
- report macchina: `data/reports/musca_btc_moe.json`;
- stato: `data/reports/musca_btc_moe.status.json`;
- matrice: `data/ml/musca_btc_moe/matrix.parquet`;
- checkpoint: `data/ml/musca_btc_moe/checkpoints/`;
- bundle non operativo: `data/models/musca_btc_moe/research_bundle.joblib`;
- baseline precedente a un minuto: `data/reports/musca_btc_moe_1m_baseline.json`.

## Verifiche

- Ruff: superato;
- mypy: superato;
- 33 test pertinenti: superati;
- future holdout letto: 0 righe;
- violazioni `available_at > entry_timestamp`: 0;
- stop prevalente nello stesso bucket: verificato;
- una sola posizione: verificato;
- nessuna nuova dipendenza aggiunta.

Fonti ufficiali: [Binance Public Data](https://github.com/binance/binance-public-data/blob/master/README.md),
[commission rate USD-M](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account#user-commission-rate),
[XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/),
[XGBoost quantile regression](https://xgboost.readthedocs.io/en/stable/tutorials/quantile.html),
[XGBoost learning to rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).
