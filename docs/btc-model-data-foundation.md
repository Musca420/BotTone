# BTC model data foundation

## Risultato della ricerca

Le sole candele OHLCV non contengono queue position, cancellazioni, aggressor flow,
open interest, basis, volatilita implicita o fill. Un modello piu grande non puo
ricostruire causalmente variabili mai osservate. La pipeline viene quindi divisa in:

1. alpha lento (1-48 ore): prezzo cross-venue, VWAP, momentum, funding, basis, open
   interest e regime di volatilita;
2. alpha veloce (secondi-minuti): microprice, order-flow imbalance, trade imbalance,
   profondita e variazioni/cancellazioni del book;
3. execution Bitunix: queue ahead, probabilita/tempo di fill POST_ONLY, fill parziali,
   maker/taker, fee, latenza e adverse selection realmente osservata;
4. risk engine: EV netto, size indipendente, una posizione, stop e kill switch.

VWAP resta un centro e un benchmark di esecuzione, non una garanzia di ritorno. Il
gating deve scegliere fra mean reversion, trend continuation e FLAT; FLAT vale zero e
non conta come profitto.

## Evidenze e fonti primarie

- Il microprice usa spread e imbalance per stimare il prossimo prezzo meglio del
  midpoint: [Stoikov, Micro-Price](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694).
- L'order-flow imbalance deve includere market order, nuovi limit e cancellazioni:
  [Cont, Kukanov e Stoikov](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822).
- La dinamica della queue richiede eventi di book, non OHLCV:
  [Queue-reactive model](https://arxiv.org/abs/1312.0563).
- Il trend following ha evidenza su orizzonti piu lenti:
  [Time Series Momentum](https://www.sciencedirect.com/science/article/pii/S0304405X11002613).
- Funding e basis rappresentano una famiglia alpha/carry distinta:
  [Crypto Carry](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2024.05069).
- Endpoint usati dal collector: [Binance Futures](https://developers.binance.com/docs/derivatives),
  [Bybit OI](https://bybit-exchange.github.io/docs/v5/market/open-interest),
  [OKX API V5](https://www.okx.com/docs-v5/en/),
  [Bitunix MarketPrice](https://www.bitunix.com/api-docs/futures/websocket/public/MarketPrice%20Channel.html),
  [Bitunix funding](https://www.bitunix.com/api-docs/futures/market/get_funding_rate.html),
  [Deribit DVOL](https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data).

## Inventario verificato il 4 agosto 2026

- Binance, Bybit, OKX e Bitunix BTC: circa 1,21 milioni di minuti per venue,
  aprile 2024-agosto 2026.
- Binance aggressor flow: 1,21 milioni di minuti con quote volume, trade count e
  taker-buy quote.
- Bitunix book/trade: due giornate live; sufficiente per testare la pipeline, non per
  un audit execution indipendente.
- Execution privata Bitunix: nessun archivio di ordini/fill osservati; la matrice V8
  ricostruita non puo sostituirli.
- OI/basis/DVOL comune: assente dallo storico locale; il nuovo collector lo registra
  da ora senza backfill inventato.

## Errori di metodo trovati

1. Il vecchio flag `data_valid` Bitunix invalida 62.393 righe che la regola corrente
   considera correzioni OHLC accettabili entro 5 bps; soltanto 111 righe superano la
   soglia. Le versioni congelate non vengono riscritte.
2. Applicare `one_position` separatamente a venue correlate produce sequenze diverse
   e puo creare un falso edge. I nuovi audit devono confrontare prima le stesse azioni
   agli stessi timestamp.
3. V18 TSMOM SHORT sembrava positivo sulle tre venue esterne, ma sui 3.993 segnali
   sincronizzati l'EV lordo e negativo su tutte e quattro. Non e un modello operativo.
4. Funding carry a 8 ore non e stabile: nel 30% cronologico recente Binance e
   negativo e nessuna configurazione trasferibile supera in modo robusto PF 1,15.
5. L'imbalance aggressivo Binance a 4 ore mostrava +7,95 bps e PF 1,156 in una
   prima label close-to-close. Con ingresso next-event, stop 2 ATR e costi reali,
   l'EV diventa da -0,0038 R a +0,0059 R e PF 0,99-1,02 sulle quattro venue; a
   costi 2x e negativo ovunque. Il candidato e stato respinto prima della GPU.
6. La matrice microstrutturale corrente contiene 891 istanti distinti di una sola
   giornata. Le correlazioni di book/microprice con adverse selection cambiano tra
   prima e seconda parte e non autorizzano un modello.

## Regola per il prossimo training

Nessun V19 viene lanciato finche l'audit non registra esattamente quali feature hanno
copertura. Ridge resta champion; XGBoost GPU e challenger. DeepLOB o reti neurali si
valutano solo dopo un archivio LOB ampio e dopo che baseline lineari/tree semplici
sono state battute OOS. Un risultato negativo resta `FLAT`.
