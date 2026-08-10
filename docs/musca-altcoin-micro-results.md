# Musca Altcoin Micro — risultato dell'audit Binance

Data: 2026-08-10  
Branch: `codex/multiasset-micro`  
Protocol hash: `6a047d6aad8349c59dc4b03b9392078e4d4e1a22e1b2243e493ea4d21525ead4`

## Verdetto

`NO_SUSTAINABLE_ALTCOIN_MICRO_ALPHA`.

Nessuna delle policy ETHUSDT, XRPUSDT o DOGEUSDT supera i gate economici sull'audit
cronologico maggio-giugno 2026. Nessun profilo paper/testnet è stato attivato e il denaro reale
resta vietato. L'holdout dal 1 luglio 2026 non è stato letto.

Questo è l'ultimo tentativo della linea, come richiesto. Il risultato non è stato trasformato in
positivo abbassando i gate, cambiando i costi dopo l'osservazione o scegliendo un'altra finestra.

## Risultati congelati

I costi netti preregistrati sono 9,0 bps round-trip per ETH, 10,5 bps per XRP e 11,5 bps per
DOGE. Il test costi 2× è diagnostico e non sostituisce i costi normali.

| Asset | Champion | Righe | Trade | Trade/giorno | EV netta | PF | Win rate | Giorni attivi positivi | Max DD | LCB 95% | EV a costi 2× |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETHUSDT | XGBoost GPU | 42.905 | 163 | 2,67 | -3,30 bps | 0,902 | 41,1% | 36,1% | 19,9% | -14,20 bps | -12,30 bps |
| XRPUSDT | XGBoost GPU | 41.812 | 524 | 8,59 | -9,64 bps | 0,679 | 38,0% | 27,6% | 66,5% | -12,35 bps | -20,14 bps |
| DOGEUSDT | Ridge | 42.834 | 1.246 | 20,43 | -9,80 bps | 0,514 | 37,5% | 4,9% | 96,9% | -11,09 bps | -21,30 bps |

ETH è il caso meno negativo, ma fallisce frequenza, expectancy, PF, stabilità giornaliera,
drawdown e limite bootstrap. XRP e DOGE raggiungono ampiamente la frequenza richiesta, ma
perdono dopo i costi. Questo separa il problema della frequenza dal problema dell'edge: il
generatore trova occasioni, ma il ranking non le distingue in modo stabile su dati successivi.

La selezione di aprile conteneva un punto ETH all'1% di copertura con EV +5,60 bps e PF 1,19,
ma solo 22 trade, 0,73 trade/giorno e LCB -27,08 bps. Congelata quella copertura, maggio-giugno
ha prodotto i 163 trade negativi riportati sopra. Non è quindi una conferma riutilizzabile.

### Diagnosi del disegno

Sull'audit maggio-giugno nessuna delle otto azioni fisse, presa senza selezione ML, ha expectancy
netta positiva. La migliore è ETH `MICRO_60_SHORT` a -6,55 bps; le altre vanno da -7,72 a
-14,48 bps su ETH, da -8,55 a -11,99 bps su XRP e da -8,13 a -14,48 bps su DOGE.

L'oracle riportato nel JSON non prova che il 95% degli eventi sia prevedibilmente profittevole:
per ogni evento sceglie *dopo* aver osservato il futuro il massimo fra otto azioni e orizzonti.
È un upper bound affetto da selezione multipla, utile per verificare che esistano movimenti ma non
per stimare l'edge realizzabile. Il compito affidato al modello è quindi troppo compresso: deve
ricavare da klines 1m il raro sottoinsieme positivo di azioni strutturalmente negative, oltre a
scegliere insieme lato, durata e barriera.

Una libreria più complessa non aggiunge l'informazione assente. XGBoost già rappresenta soglie e
interazioni non lineari e usa la GPU. Gli strumenti potenzialmente appropriati per una ricerca
successiva sarebbero label da `aggTrades`/trade sub-minute, dati causali di spread e book, e una
formulazione separata per probabilità di barriera, tempo all'evento e payoff condizionato. XGBoost
supporta già [learning to rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html),
[quantile regression](https://xgboost.readthedocs.io/en/stable/tutorials/quantile.html) e
[survival AFT](https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html):
non è necessario introdurre subito un framework deep-learning. Questa è una futura ipotesi di
ricerca, non una correzione autorizzata o una promessa di profitto.

## Gate

| Gate audit | ETH | XRP | DOGE |
|---|---:|---:|---:|
| Numero minimo di trade | passa | passa | passa |
| Almeno 3 trade/giorno | fallisce | passa | passa |
| Expectancy netta positiva | fallisce | fallisce | fallisce |
| Profit factor ≥ 1,15 | fallisce | fallisce | fallisce |
| Maggioranza giorni attivi positiva | fallisce | fallisce | fallisce |
| Drawdown ≤ 8% | fallisce | fallisce | fallisce |
| Bootstrap LCB 95% positivo | fallisce | fallisce | fallisce |

## Cosa è stato verificato

- BTCUSDT è stato usato soltanto come contesto causale: rendimenti, volatilità, taker flow,
  correlazione e beta rolling, shock e rendimento residuo. Non è stata imposta una conferma BTC
  fissa e BTC non è stato tradato.
- Gli eventi sono front-edge causali: VWAP cross/touch/rejection, restart momentum, shock BTC,
  inversione residua e impulso di volume, con cooldown di tre minuti.
- Le azioni LONG/SHORT usano orizzonti 5, 15, 30 e 60 minuti, target e stop dinamici; ingresso al
  minuto successivo e stop vincente quando stop e target sono toccati nella stessa candela.
- Ridge e XGBoost GPU vedono le stesse split. XGBoost è champion solo su ETH e XRP; DOGE resta
  correttamente su Ridge.
- La previsione continua ordina i candidati; l'isotonic calibra l'EV visualizzata senza appiattire
  il ranking. FLAT è applicato alla policy aggregata fallita, non contato come trade vincente.
- Controlli causali: zero violazioni di feature future, zero righe dell'holdout lette, zero righe
  con feature mancanti nella matrice finale.

## Limiti dichiarati

La ricerca usa klines ufficiali Binance USD-M a un minuto, con archivi e checksum secondo
[Binance Public Data](https://github.com/binance/binance-public-data/blob/master/README.md).
Non dispone, per l'intero periodo comune, di percorso tick/aggTrades sub-minute, storico L2,
spread osservato e funding sincronizzato. Per questo:

- i casi stop-target nello stesso minuto sono valutati conservativamente come stop;
- spread/slippage/fee sono inclusi in riserve fisse preregistrate, non ricostruiti da book storico;
- un paper testnet sarebbe utile per API, ordini e stato, ma non renderebbe positivo questo audit;
- dati di esecuzione più realistici aggiungerebbero normalmente costi e incertezza, quindi non
  possono essere usati come giustificazione per promuovere risultati già negativi.

Le dipendenze dinamiche con BTC sono state modellate perché la letteratura trova correlazioni
crypto variabili e spillover ad alta frequenza, non una relazione direzionale costante:
[Katsiampa (2019)](https://doi.org/10.1016/j.ribaf.2019.06.004) e
[Sensoy et al. (2021)](https://doi.org/10.1080/00036846.2021.1899119).

## Artefatti e integrità

| Artefatto | SHA-256 |
|---|---|
| `data/reports/musca_altcoin_micro.json` | `7e356cc673b7ca35bd8e9a8eb2742ca13308f8dbe04312e1a462960bd62bd863` |
| `data/ml/musca_altcoin_micro/ethusdt_matrix.parquet` | `5481288582caa36ca34118b3ec881ad9e194e18e4bc8d27ee9ce6cb4f1cbafc5` |
| `data/ml/musca_altcoin_micro/xrpusdt_matrix.parquet` | `84265236647e97a2f92c1ac7c2abb6ab7706b8bc12bb9be2d35c332a44aabb30` |
| `data/ml/musca_altcoin_micro/dogeusdt_matrix.parquet` | `d8d648f8aa9a486bdcba5faf0a6c6f09e19862fd899e1c7d5f223b0209680230` |

File correlati:

- `src/adaptive_bot/musca_altcoin_micro.py`: dati, feature BTC-aware, simulazione, modelli e audit;
- `tests/unit/test_musca_altcoin_micro.py`: test causali ed economici mirati;
- `docs/musca-altcoin-micro.md`: protocollo preregistrato;
- `data/reports/musca_altcoin_micro.json`: report macchina completo;
- `data/ml/musca_altcoin_micro/*_matrix.parquet`: matrici controfattuali riproducibili.

## Cronologia Git della linea

- `8d23acf` — protocollo preregistrato;
- `e64f90c` — audit multiasset con contesto BTC;
- `8714b73` — generazione event-driven;
- `46de329` — valutazione aggregata della policy prima di FLAT;
- `587a1cf` — ranking continuo separato dalla calibrazione isotonic.

Le tre correzioni non sono ottimizzazioni di profitto: risolvono rispettivamente la diluizione su
minuti senza evento, l'asimmetria FLAT/per-trade e i plateau della calibrazione. Dopo tali
correzioni il test cronologico resta negativo, quindi la linea viene fermata.
