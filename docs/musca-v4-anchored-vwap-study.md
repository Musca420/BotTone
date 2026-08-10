# Studio Musca V4 — uso corretto dell’Anchored VWAP

Data: 5 agosto 2026. Ambito: BTCUSDT, Musca V2/V3, Binance storico e Bitunix shadow.

## Verdetto

Musca V3 non sta testando correttamente una strategia Anchored VWAP. Il problema non è una singola soglia troppo severa: esistono prima due incompatibilità nei dati e poi quattro errori nella formulazione della strategia.

L’AVWAP deve rappresentare il costo medio scambiato **da un evento economico o strutturale significativo**. In V3 nasce invece da un reclaim del Daily VWAP, viene poi sostituito dal timestamp del touch e, dopo il trade, continua da quel nuovo punto. Il riferimento perde quindi la memoria dell’impulso che avrebbe dovuto misurare.

La soluzione proposta non è allentare le condizioni per generare ordini. È costruire una V4 separata, event-driven, nella quale:

1. la direzione proviene dalla struttura del prezzo e dall’order flow;
2. gli anchor provengono da eventi causali e restano invariati;
3. più AVWAP definiscono una zona, non un prezzo esatto;
4. il pullback arma il setup, ma l’ingresso avviene soltanto sulla ripartenza;
5. il modello ML filtra setup già sensati e non inventa la direzione.

## Evidenze ufficiali

- TradingView definisce l’AVWAP come media ponderata per volume da un punto scelto e, nell’indicatore Auto Anchored, usa anchor come `Highest High`, `Lowest Low`, `Highest Volume`, sessione e settimana. La sorgente tradizionale è `HLC3` e sono previste bande di deviazione standard: <https://www.tradingview.com/support/solutions/43000669764-anchored-vwap-drawing-tool/> e <https://www.tradingview.com/support/solutions/43000652199-vwap-auto-anchored/>.
- La presentazione CMT Association di Brian Shannon usa AVWAP da eventi, massimi/minimi significativi e barre ad alto volume. Specifica che non tutti i touch producono un rimbalzo e sintetizza l’ingresso come forza dopo il pullback/debolezza dopo il rimbalzo, non come acquisto o vendita immediata del touch: <https://cmtassociation.org/wp-content/uploads/2024/01/Shannon-Specific-Anchored-VWAP-Strategies-1.pdf>.
- TrendSpider documenta anchor da breakout, swing e high-volume event, l’uso di più AVWAP per individuare confluence zone e l’uso con price action/volume, non come segnale autonomo: <https://trendspider.com/learning-center/anchored-vwap-trading-strategies/>.
- Binance rende disponibili klines, trades e aggregate trades ufficiali. Le klines distinguono volume base, volume quote, trade count e taker-buy base/quote; gli archivi hanno checksum: <https://github.com/binance/binance-public-data/blob/master/README.md?plain=1>.
- Bitunix distingue `LAST_PRICE` e `MARK_PRICE` nelle klines e usa `LAST_PRICE` come default. L’esempio ufficiale mostra `quoteVol=1` e `baseVol=60000` per BTC a 60.000 USDT; il canale trade espone direttamente prezzo riempito e quantità: <https://www.bitunix.com/api-docs/futures/market/get_kline.html> e <https://www.bitunix.com/api-docs/futures/websocket/public/Trade%20Channel.html>.

Queste fonti descrivono strumenti e pratiche, non dimostrano un profitto. L’edge deve ancora essere verificato OOS.

## Errori trovati nel flusso attuale

### 1. Il live usa la serie di prezzo sbagliata per il VWAP

`src/adaptive_bot/adapters/bitunix/collector.py` richiede esplicitamente klines `MARK_PRICE`. Il mark serve al controllo del rischio e della liquidazione; un VWAP di trading deve essere costruito da prezzi realmente scambiati (`LAST_PRICE` o trade tick).

V3 usa invece OHLC mark come prezzo, con volumi di scambio, e simula anche entry/stop sulle stesse barre mark. Questo mescola benchmark di rischio e mercato eseguibile.

### 2. Le unità Bitunix sono interpretate al contrario

`read_collected_candles()` in `src/adaptive_bot/services/bitunix_paper_service.py` assegna `baseVol` a `volume`. Successivamente `_live_features()` in `src/adaptive_bot/musca_v2.py` calcola `perp_quote_volume = close * volume`.

In base all’esempio ufficiale Bitunix e ai valori osservati nel file live:

- `quoteVol` è circa 171,843 BTC;
- `baseVol` è circa 11.127.280,65 USDT;
- il VWAP esatto della barra/periodo è `sum(baseVol) / sum(quoteVol)`.

Il codice corrente tratta invece gli 11 milioni di USDT come BTC e li moltiplica ancora per il prezzo. Il rapporto finale resta numericamente plausibile, ma non è il VWAP degli scambi.

Audit sul 5 agosto, 222 barre chiuse disponibili:

| Calcolo | Valore |
|---|---:|
| Daily VWAP usato da V3 | 64.324,2007 |
| VWAP dagli importi Bitunix | 64.319,0156 |
| differenza | 0,8062 bps |
| AVWAP V3 da 13:45 | 64.510,2791 |
| AVWAP corretto dagli importi | 64.503,1950 |
| differenza | 1,0983 bps |

Lo scarto corrente non basta da solo a spiegare zero trade, ma invalida la parità storico/live e può cambiare un reclaim o un touch vicino alla soglia.

### 3. L’anchor non rappresenta un evento indipendente

In `src/adaptive_bot/musca_v3.py`:

- ogni reclaim del Daily VWAP sostituisce `continuation_anchor_at`;
- un ingresso continuation salva come `anchor_at` il timestamp del touch, non quello dell’impulso originale;
- alla chiusura, il nuovo continuation anchor viene ricavato dall’anchor della posizione, ormai sostituito dal touch.

Questo è circolare: un evento generato da un VWAP crea un secondo VWAP, che viene poi riavviato quando il prezzo lo tocca. Non misura più il costo medio dei partecipanti dall’origine del movimento.

### 4. Un solo anchor e una linea troppo precisa

V3 conserva un solo AVWAP e richiede che la chiusura sia entro `0,5 ATR`. Le implementazioni documentate usano spesso:

- session/day e week VWAP;
- AVWAP dall’impulso o breakout;
- AVWAP da swing high/low significativo;
- bande volume-weighted;
- confluence tra più riferimenti.

Il mercato non deve toccare un singolo numero. Deve entrare in una zona di costo e mostrare accettazione o rifiuto.

### 5. La pendenza AVWAP viene usata come conferma quasi circolare

V3 richiede slope AVWAP positiva/negativa a 15 minuti e un’ora. Poiché l’AVWAP incorporaora proprio i prezzi successivi all’anchor, questa slope è in gran parte una versione ritardata dello stesso movimento già richiesto da return 15m/1h e close sopra/sotto AVWAP. Aggiunge filtri correlati, non una conferma indipendente.

La slope AVWAP può descrivere lo stato del riferimento, ma la direzione deve provenire da struttura, breakout, spot/perpetual e aggressione reale.

### 6. “V3 intraday” usa ancora una direzione da 7 e 30 giorni

`trend_strength = 0,6 × return_7d + 0,4 × return_30d` proviene da V2. Era coerente con il suo orizzonte fino a 14 giorni, ma non con un bot che cerca più swing intraday.

Questo spiega perché V3 può restare LONG durante un microtrend ribassista o SHORT durante un recupero intraday. Il Daily VWAP e l’AVWAP non correggono questo problema: non devono inventare la direzione.

### 7. Il passaggio Binance→Bitunix non è una serie continua

Lo storico termina il 31 luglio 23:55 UTC; il live successivo usa un’altra venue e prezzi mark. Le finestre 7d/30d attraversano quindi contemporaneamente un cambio di exchange, di price type e il confine tra due dataset. Prima di giudicare i segnali live, la V4 deve avere un contratto dati identico tra backtest e shadow.

## Cosa dicono i risultati già prodotti

- V2 ha trovato un edge discovery su un orizzonte lungo: 227 trade OOS, +39,14 bps medi netti e PF 1,306. Tuttavia il fold 2026 è sceso a +1,36 bps e PF 1,011: non dimostra che l’edge sia ancora sufficiente.
- V3 ha 255 eventi anchored tradeable su 9.854 candidati. La sequenza base ha +1,65 bps, PF 1,127 e drawdown 60,77%; nessun trade è stato selezionato OOS. Non è una policy validata.
- V24/V25 mostrano che aggiungere condizioni non risolve automaticamente il problema. Il collo di bottiglia è la definizione economica dell’evento, non il numero di feature o i trial GPU.
- Lo shadow V3 corrente ha zero trade, ma non può essere usato per validare o bocciare AVWAP a causa delle incompatibilità sopra.

## Protocollo proposto per Musca V4

### A. Contratto dati unico

Per segnale e backtest:

- prezzo: `LAST_PRICE`/trade price;
- quantità base e notional quote conservati in colonne separate;
- bar VWAP: `quote_notional / base_quantity` quando entrambi sono ufficiali;
- mark e index conservati solo come feature e per rischio/liquidazione;
- bid/ask reale per simulare l’esecuzione shadow;
- nessuna concatenazione silenziosa Binance-last con Bitunix-mark.

Per lo storico principale si usano Binance spot e perpetual ufficiali, inclusi quote volume, taker volume e trade count. Bitunix resta la verifica shadow della trasferibilità e dell’esecuzione.

### B. Piccolo insieme preregistrato di anchor

Mantenere al massimo tre riferimenti attivi, senza una griglia estesa:

1. `SESSION_UTC`: VWAP giornaliero, reset 00:00 UTC;
2. `IMPULSE`: prima barra 15m chiusa che rompe una struttura già nota con volume/taker flow concorde;
3. `STRUCTURAL_SWING`: ultimo swing high/low confermato causalmente; l’anchor ha il timestamp dello swing ma diventa disponibile soltanto al timestamp di conferma.

Lo stesso anchor non viene mai spostato al touch o all’ingresso. Viene invalidato solo da una rottura strutturale opposta o dalla scadenza preregistrata. L’anchor più vecchio può essere sostituito solo da un nuovo evento dello stesso tipo più recente.

La variante `HIGHEST_VOLUME` documentata da TradingView va studiata come diagnostica separata; non va aggiunta contemporaneamente alla prima V4 per evitare un’altra ricerca a griglia.

### C. Zona AVWAP

Per ogni anchor calcolare causalmente:

```text
avwap = sum(quote_notional) / sum(base_quantity)
sigma_vw = sqrt(sum(base_quantity * (price - avwap)^2) / sum(base_quantity))
```

La zona primaria preregistrata è AVWAP ± 1 deviazione standard volume-weighted. Daily/Impulse/Swing che si sovrappongono aumentano il `confluence_score`; non generano da soli un ordine.

### D. Macchina a stati, non controlli indipendenti ogni cinque minuti

```text
NO_SETUP
  -> IMPULSE_CONFIRMED
  -> PULLBACK_IN_PROGRESS
  -> AVWAP_ZONE_REACHED
  -> REACCELERATION_CONFIRMED
  -> ENTRY_NEXT_EVENT
  -> MANAGE_POSITION
```

Ogni stato ha `detected_at`, `available_at`, direzione, anchor immutabile, scadenza e motivo di invalidazione. Un cross isolato non salta direttamente a `ENTRY`.

### E. Direzione indipendente dall’AVWAP

Usare solo dati chiusi e disponibili:

- struttura 4h/1h (massimi/minimi e slope normalizzata);
- breakout 15m di una struttura precedente;
- spot e perpetual concordi;
- taker imbalance e intensità trade concordi;
- open interest soltanto dove la copertura storica è reale; altrimenti feature shadow, fail-closed.

L’AVWAP risponde alla domanda “dove è ragionevole fare affari rispetto al costo medio dall’evento?”, non “qual è il trend?”.

### F. Pullback e ripartenza

Il pullback arma il setup quando entra nella zona di uno o più anchor e mostra partecipazione contraria inferiore all’impulso. L’ingresso avviene al minuto successivo soltanto quando:

- la barra chiusa rompe il micro swing nella direzione dell’impulso;
- il taker flow torna concorde;
- spot e perpetual non divergono;
- il prezzo non è già esteso oltre la prima banda;
- il movimento lordo disponibile verso il prossimo livello è almeno tre volte il costo round-trip.

Questo implementa “comprare forza dopo il pullback / vendere debolezza dopo il rimbalzo”, non comprare il semplice touch.

### G. Stop e uscita dinamici

“Nessuno stop fisso” deve significare nessuna distanza arbitraria fissa, non assenza di protezione.

- stop iniziale oltre l’estremo strutturale del pullback e oltre la zona AVWAP;
- ordine protettivo sempre presente;
- lo stop non può mai allargarsi;
- dopo 1R o una nuova struttura favorevole, trailing sullo swing 15m confermato;
- uscita per accettazione opposta: chiusura 15m oltre l’AVWAP operativo con flow contrario, non un singolo cross 5m;
- uscita parziale a 1R/1,5R solo se i costi reali lasciano spazio; resto verso la successiva zona AVWAP o trailing;
- timeout intraday massimo 6 ore per V4, separato dalla V2 long-horizon.

### H. Training corretto

Prima si valuta la base deterministica. Il dataset contiene solo eventi completi, non tutti i minuti.

Target separati:

- probabilità di raggiungere 1R prima dell’invalidazione;
- MFE e MAE netti;
- tempo a 1R;
- rendimento netto con costi reali e stress 2×.

Ridge resta baseline. XGBoost GPU è challenger soltanto sulle stesse split. Il modello può rifiutare o ordinare i candidati; non può cambiare anchor, direzione, stop o generare un evento che la base non ha prodotto.

Walk-forward, purge sull’exit reale, holdout futuro e confronto contro FLAT restano obbligatori. FLAT vale zero e non è profitto; entra in gioco solo dopo che esiste un candidato economico completo.

## Sequenza di implementazione che evita un altro training inutile

1. Correggere il contratto volume/prezzo in un unico reader condiviso e mantenere mark separato.
2. Aggiungere test di parità formula Bitunix/Binance e snapshot storico/live.
3. Costruire il registro immutabile degli anchor e visualizzare in UI tipo, origine, età, zona e stato.
4. Eseguire un audit deterministico: eventi/giorno, transizioni di stato, MFE/MAE, costi, esiti per tipo di anchor e confluence.
5. Fermarsi se la base netta è negativa o se il movimento lordo non supera 3× i costi.
6. Solo allora addestrare Ridge e XGBoost GPU per il filtro.
7. Avviare `MUSCA V4` come profilo shadow separato; non sovrascrivere V2/V3.

## Test obbligatori V4

- `quoteVol`/`baseVol` con l’esempio ufficiale Bitunix produce VWAP 60.000;
- LAST, mark e index non vengono confusi;
- un touch non modifica l’anchor originale;
- uno swing non è disponibile prima della conferma;
- bande e AVWAP usano solo dati fino alla barra chiusa;
- nessun ingresso al solo touch;
- entry sempre sull’evento successivo;
- stop strutturale presente e mai allargato;
- identico snapshot produce identiche feature in backtest e shadow;
- gap, costi 2× e barra ambigua sono conservativi;
- caso senza reaccelerazione resta FLAT, ma non viene conteggiato come vincita.

## Decisione

Non va riaddestrata l’attuale V3. Prima va corretta la semantica dei dati e va implementata V4 come linea separata. Musca V2 può continuare in shadow come benchmark long-horizon; lo shadow V3 può continuare a raccogliere osservazioni, ma i suoi segnali/PnL precedenti non devono essere usati come evidenza finché LAST_PRICE, volumi e anchor non sono coerenti.

