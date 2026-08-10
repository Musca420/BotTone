# Musca DOGE Auto-MoE — protocollo di ricerca

## Obiettivo e stato di sicurezza

Questa linea applica a `DOGEUSDT` le due fasi congelate per BTC senza riusare
gli esperti BTC: prima genera strategie locali dalla storia DOGE, poi addestra un
selettore adattivo che decide quale strategia attiva ascoltare oppure resta
`FLAT`. Il bundle è sempre `RESEARCH_ONLY`; il denaro reale resta disabilitato.

## Evidenze adottate

- Binance pubblica archivi giornalieri/mensili con checksum per klines e
  `aggTrades`; questi sono la fonte storica primaria.
- Hummingbot separa controller direzionali, market making ed executor e permette
  più controller nello stesso bot. La nostra separazione esperti/gating/gestione
  posizione segue lo stesso confine di responsabilità.
- Gli executor Hummingbot applicano stop, take profit, time limit e trailing come
  gestione distinta dal segnale. I label DOGE riproducono quindi l'intero
  percorso gestito e non il solo prezzo terminale.
- FreqAI supporta coppie correlate e timeframe informativi. BTC è quindi una
  variabile informativa causale per DOGE, mai un veto fisso all'ingresso.
- La letteratura su DOGE documenta spillover BTC variabile e pump/crowd-pump nei
  quali numero di trade e volumi aggressivi cambiano in pochi secondi. Per questo
  il modello include beta/correlazione mobili, residui DOGE-vs-BTC, shock BTC,
  trade intensity, aggressive flow, OFI e assorbimento.
- I controlli di look-ahead seguono lo stesso principio dell'analisi ufficiale
  Freqtrade: una feature può influire soltanto dopo il proprio `available_at`.

Fonti:

- <https://github.com/binance/binance-public-data/blob/master/README.md>
- <https://hummingbot.org/strategies/v2-strategies/controllers/>
- <https://hummingbot.org/strategies/v2-strategies/examples/>
- <https://docs.freqtrade.io/en/2025.5/freqai-feature-engineering/>
- <https://www.freqtrade.io/en/stable/lookahead-analysis/>
- <https://www.mdpi.com/1911-8074/16/1/41>
- <https://arxiv.org/abs/2105.00733>

Queste fonti descrivono architettura, dati e rischi; nessuna costituisce prova
che un bot pubblico sia profittevole. La redditività viene giudicata soltanto
dall'audit cronologico locale dopo costi.

## Dati DOGE realmente usati

Periodo discovery/audit storico: 1 gennaio 2025–31 luglio 2026.

- 19 mesi di `DOGEUSDT` perpetual 1m;
- 19 mesi di `DOGEUSDT` spot 1m;
- 19 mesi di mark price 1m;
- 19 mesi di funding osservato;
- 19 mesi di perpetual `aggTrades`, aggregati causalmente a 5 secondi;
- contesto informativo BTC chiuso e disponibile allo stesso timestamp.

Ogni ZIP è validato con il checksum ufficiale Binance. Open interest DOGE è
escluso: non esiste un archivio ufficiale multi-anno con copertura comune al
protocollo. Non viene riempito con zero e non viene simulato. Anche sentiment e
social media restano esclusi finché non esisterà una raccolta causale,
riproducibile e utilizzabile nello storico e nel paper.

## Fase 1 — esperti appresi dalla storia DOGE

Un generatore XGBoost GPU produce foglie decisionali LONG e SHORT sugli
orizzonti 1m, 5m, 15m, 1h e 6h. Non esiste una griglia manuale di strategie né
un limite finale preregistrato al numero di esperti. Ogni candidato deve avere
opportunità sufficienti e superare fit, validazione cronologica, stabilità
mensile e replay economico gestito.

Target 1, target 2, stop avverso e trailing derivano dai quantili del percorso
del candidato. Il trailing non può allargare il rischio; se stop e target sono
toccati nello stesso bucket, vince lo stop. L'ingresso avviene nel bucket
successivo alla decisione.

## Fase 2 — gating adattivo

Il gate usa soltanto attivazioni forward-OOS degli esperti congelati e contesto
causale. Ridge resta champion predefinito. XGBoost GPU lo sostituisce soltanto
se migliora contemporaneamente MAE dell'EV, Brier score e regret decisionale
sulle stesse righe. `FLAT=0` è neutro e non conta come profitto.

Il costo storico prudenziale è 11,5 bps round-trip: 4 bps taker per lato più
3,5 bps di riserva complessiva per spread/slippage. In paper la commissione
account-specifica dovrà essere letta dall'endpoint firmato Binance; leva e costi
restano grandezze separate.

## Cronologia sigillata

- discovery fit: prima del 1 gennaio 2026;
- validazione e congelamento libreria: gennaio–febbraio 2026;
- tuning/fitting/calibrazione gate: marzo–maggio 2026;
- audit storico prequentiale: giugno–luglio 2026;
- holdout futuro: dal 10 agosto 2026, non letto dal training.

## Risultato congelato del 10 agosto 2026

Protocollo DOGE: `38784331694aa884b5773f523701eb736f641e2e287cc39659557115b70e13d6`.
Il run ha letto 828.612 stati causali e ha valutato 6.248 foglie candidate:
100 candidati erano economicamente validi prima del filtro di diversità e 45
esperti sono stati congelati dopo il filtro. La distribuzione è:

- 7 esperti LONG a 1 ora;
- 16 esperti LONG a 6 ore;
- 22 esperti SHORT a 6 ore;
- nessun esperto a 1, 5 o 15 minuti e nessun esperto SHORT a 1 ora ha superato
  i gate economici preregistrati.

Il replay forward ha prodotto 380.737 righe azione. Ridge è rimasto champion:

| Modello | MAE EV (bps) | Brier | Regret decisionale (bps) |
|---|---:|---:|---:|
| Ridge | 101,01 | 0,2992 | 56,43 |
| XGBoost GPU | 107,85 | 0,3352 | 58,20 |

XGBoost non ha migliorato nessuna delle tre metriche richieste e non è stato
promosso. Nell'audit prequentiale giugno-luglio 2026 la policy combinata ha
generato 23 trade (22 LONG, 1 SHORT), tutti sull'orizzonte 6 ore:

- expectancy netta: -33,14 bps per trade;
- expectancy lorda: -21,56 bps per trade;
- profit factor: 0,668;
- win rate: 43,48%;
- max drawdown: 4,94%;
- bootstrap LCB 95%: -50,11 bps;
- SPA p-value: 0,937;
- stress costi 1,5x: -38,89 bps;
- stress costi 2x: -44,64 bps;
- tre violazioni del budget dell'1% dovute a uscite oltre lo stop nel percorso
  simulato conservativo.

È passato soltanto il gate drawdown. Sono falliti numerosità, frequenza,
expectancy, profit factor, maggioranza dei giorni attivi, stress costi,
bootstrap, SPA e integrità del risk budget. Il verdetto è quindi
`NO_DEPLOYABLE_POLICY`. Il bundle esiste esclusivamente per riproduzione della
ricerca con `research_only=true`; `orders_enabled`, `paper_orders_enabled` e
`live_orders_enabled` sono tutti `false`. Il holdout futuro letto è pari a zero
righe.

Questo risultato localizza l'assenza di edge sulle micro-operazioni DOGE nei
label economici, non in un veto artificiale di `FLAT`: nessuna foglia a 1, 5 o
15 minuti ha superato fit, validazione e replay dopo 11,5 bps di costo
round-trip. Gli esperti a 6 ore esistono, ma il gate non li trasferisce con EV
positivo nel periodo successivo. I gate non sono stati abbassati.

## Correzione di coerenza OI e ripresa

Il primo tentativo si è fermato dopo la discovery perché la vista downstream
`regime/gating` richiedeva ancora `oi_change_1h`, nonostante l'Open Interest
DOGE fosse escluso dal dataset. La correzione rimuove OI da tutte le viste DOGE
e conserva invariato il contratto BTC. Matrice e libreria sono state migrate
atomicamente dai vecchi hash soltanto dopo avere verificato che la discovery
usasse già l'elenco feature senza OI; nessun candidato è stato rigenerato o
alterato. Test dedicati impediscono che una feature OI torni in una vista DOGE.

## File correlati

- `src/adaptive_bot/musca_doge_data.py`: archivi ufficiali e contesto BTC;
- `src/adaptive_bot/musca_doge_auto_moe.py`: ingresso asset-specifico;
- `src/adaptive_bot/musca_doge_pipeline.py`: pipeline dati + training;
- `src/adaptive_bot/musca_btc_moe.py`: motore MoE condiviso e causalità;
- `src/adaptive_bot/musca_btc_auto_moe.py`: generazione e gating condivisi;
- `src/adaptive_bot/musca_v5_microstructure.py`: aggTrades 5s verificati;
- `scripts/run_musca_doge_auto_moe_training.ps1`: monitor terminale;
- `tests/unit/test_musca_doge_auto_moe.py`: test DOGE mirati;
- `data/reports/musca_doge_data_audit.json`: audit dati;
- `data/reports/musca_doge_auto_moe.json`: report training;
- `data/models/musca_doge_auto_moe/research_bundle.joblib`: bundle ricerca.

## Riproduzione

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_musca_doge_auto_moe_training.ps1
```

Il risultato numerico e i gate vengono riportati soltanto dopo il completamento
del run congelato. Un esito negativo produce `NO_DEPLOYABLE_POLICY`, senza
abbassare costi, frequenza minima o gate.
