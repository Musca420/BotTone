# Musca BTC Auto-MoE — protocollo ufficiale di ricerca

## Stato congelato

La policy corrente è `RESEARCH_PAPER`: può emettere ordini nel simulatore Binance,
ma non può inviare denaro reale. Il protocollo congelato è identificato da:

```text
a195b75e41bf7bfd40dd2ca23103720cee1ea5dfbc7333cd34dcca4c7171a360
```

Artefatti principali:

- `src/adaptive_bot/musca_btc_auto_moe.py`: generazione esperti, gate e audit;
- `src/adaptive_bot/musca_v8_binance.py`: adattatore paper Binance fail-closed;
- `src/adaptive_bot/musca_v5_paper.py`: conto, ordini e gestione posizione simulati;
- `data/reports/musca_btc_auto_moe.json`: report riproducibile completo;
- `data/models/musca_btc_auto_moe/research_bundle.joblib`: policy congelata;
- `data/research/musca_btc_auto_moe_paper_state.json`: conto paper persistente;
- `tests/unit/test_musca_btc_auto_moe.py`: causalità, selezione e audit;
- `tests/unit/test_musca_v8_binance.py`: parità storico/live e fail-closed;
- `tests/unit/test_musca_v5_paper.py`: TP parziali e trailing non allargabile.

## Cosa è stato adottato dai bot pubblici maturi

I bot pubblici funzionanti non offrono un Alpha BTC universale. Binance separa
Grid, DCA, rebalancing e arbitraggio; Hummingbot separa la logica di quotazione
dalla gestione di inventory, spread e rischio; Freqtrade/FreqAI richiede
backtest, dry-run e riaddestramento temporale. La conseguenza applicata qui è:

1. nessun grid sempre acceso quando spread, fee e adverse selection lo rendono
   negativo;
2. candidati generati soltanto da eventi causali realmente osservati;
3. molte strategie locali apprese, poi un gate adattivo sceglie esperto o FLAT;
4. stessa gestione di stop/target/trailing in label, audit e paper;
5. riaddestramento prequentiale: ogni mese usa soltanto dati precedenti;
6. esecuzione e rischio restano separati dall'Alpha.

Riferimenti ufficiali:

- <https://academy.binance.com/ur-PK/articles/your-guide-to-binance-trading-bots>
- <https://hummingbot.org/strategies/v1-strategies/pure-market-making/>
- <https://hummingbot.org/strategies/v1-strategies/avellaneda-market-making/>
- <https://www.freqtrade.io/en/stable/strategy-101/>
- <https://www.freqtrade.io/en/stable/freqai-running/>

## Architettura del training

### Fase 1 — generatore automatico di esperti

La GPU genera foglie decisionali sui lati LONG/SHORT e sugli orizzonti 1 minuto,
5 minuti, 15 minuti, 1 ora e 6 ore. Ogni foglia è valutata come strategia con il
percorso economico realmente gestito: ingresso sul bucket successivo, metà
posizione a q50, seconda metà a q75, stop avverso q75, trailing che non aumenta
mai il rischio, funding e costo round-trip di 9 bps.

Non è imposto un numero finale di esperti. La generazione si arresta per
saturazione e rimuove segnali quasi duplicati con Jaccard massimo 0,90. Delle
7.691 azioni-esperto valutate, 65 sono risultate economicamente valide prima
della diversificazione e 29 sono state congelate:

| Lato | Orizzonte | Esperti |
|---|---:|---:|
| LONG | 1 ora | 5 |
| SHORT | 1 ora | 0 |
| LONG | 6 ore | 9 |
| SHORT | 6 ore | 15 |

Gli orizzonti 1/5/15 minuti hanno prodotto zero esperti validi dopo i costi. Il
protocollo non li forza: questo è il risultato che ha respinto la precedente
ipotesi delle microtransazioni continue su BTC.

### Fase 2 — selettore adattivo

Il gate usa le attivazioni OOS degli esperti e il contesto di mercato. Ridge è
il champion predefinito; XGBoost GPU può sostituirlo soltanto battendolo sulle
stesse righe in MAE EV, Brier e regret decisionale. Nel run congelato:

| Modello | MAE bps | Brier | Regret bps |
|---|---:|---:|---:|
| Ridge | 81,088 | 0,31681 | 35,544 |
| XGBoost | 87,554 | 0,38881 | 39,284 |

Ridge rimane quindi il modello corretto. FLAT vale zero ed è neutro: viene scelto
solo quando nessun esperto attivo presenta EV calibrato positivo.

## Contratto causale dei dati

Il modello usa 57 feature disponibili sia nello storico sia nel paper: ritorni,
VWAP multi-orizzonte, distanza/velocità/test del VWAP, volatilità, volumi,
taker flow, struttura della candela, efficiency/range, order-flow L2 e quattro
feature derivatives. Le ultime sono ricostruite esclusivamente dagli endpoint
pubblici ufficiali Binance:

- open interest 1h: `/futures/data/openInterestHist?period=5m`;
- basis: ultima candela chiusa mark price contro spot;
- funding z-score: `/fapi/v1/fundingRate` ricostruito sulla griglia al minuto;
- interazione ritorno 5m × variazione OI.

Ogni decisione usa l'ultimo minuto completato. Timestamp futuri, buchi recenti,
book stale o warm-up insufficiente invalidano l'intero candidato; i mancanti non
sono sostituiti con zero.

Documentazione Binance:

- <https://github.com/binance/binance-public-data/blob/master/README.md?plain=1>
- <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data>

## Risultato storico OOS

Audit prequentiale 1 giugno–31 luglio 2026, dopo costo round-trip di 9 bps:

| Metrica | Risultato |
|---|---:|
| Trade | 72 |
| Frequenza | 1,1803 per giorno di calendario |
| Expectancy netta | +19,5415 bps/trade |
| Profit factor | 1,5511 |
| Win rate | 62,50% |
| Max drawdown | 2,438% |
| Bootstrap LCB 95% | +11,3991 bps |
| SPA p-value | 0,040 |
| Expectancy con costi 1,5× | +15,0415 bps |
| Expectancy con costi 2× | +10,5415 bps |
| Giorni attivi positivi | 63,16% |

Il risultato non è uniforme: giugno ha 67 trade, expectancy +21,312 bps e PF
1,605; luglio ha soltanto 5 trade, expectancy −4,187 bps e PF 0,892. È un motivo
esplicito per mantenere il sistema in paper e non presentare il backtest come una
garanzia di redditività futura.

## Gate e decisione operativa

Passano i gate paper: almeno 50 trade di ricerca, expectancy positiva, PF,
drawdown e assenza di violazioni del risk budget. Non passano i gate live:

- meno di 300 trade storici OOS;
- frequenza inferiore a 3 trade/giorno;
- holdout futuro ancora sigillato e vuoto.

Il simulatore parte da 10.000 USDT, rischia al massimo l'1% per trade, usa leva
massima 10× solo come limite di notional, ammette una posizione e applica
profondità L2, fee, slippage e funding osservati. Leva e stress dei costi sono
concetti distinti. Il denaro reale rimane vietato fino al completamento di almeno
300 osservazioni OOS complessive, dell'holdout futuro e dei gate ingegneristici.

## Riproduzione

Training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_musca_btc_auto_moe_training.ps1
```

Paper persistente e dashboard:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_musca_runtime.ps1
```

La pagina operativa resta su `http://127.0.0.1:8080/?profile=musca-v5-binance`;
il valore interno del profilo è mantenuto per compatibilità con i bookmark, ma
l'interfaccia lo identifica come `MUSCA BTC · AUTO-MoE PAPER`.
