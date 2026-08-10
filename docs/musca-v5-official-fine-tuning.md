# Musca V5 — protocollo ufficiale di fine tuning

Ultimo aggiornamento: 2026-08-09. Questo documento è la fonte di verità della linea Musca V5.
Va riletto prima di riprendere il lavoro dopo una compattazione del contesto. Il diario degli
esperimenti è in `docs/musca-v5-fine-tuning-log.md`.

## Obiettivo corrente — Binance only (decisione 2026-08-09)

Musca V5 usa esclusivamente Binance BTCUSDT per Alpha storica, segnali, book, mark/index,
funding, costi, simulazione paper e — soltanto dopo tutti i gate — futuro adapter live. Bitunix è
fuori dalla policy e dal runtime: i dati già raccolti sono conservati, ma potranno essere usati
solo in una futura prova di trasferimento dopo la scelta del miglior modello Binance.

Percorso attivo e riproducibile:

- configurazione paper: `configs/binance_btcusdt_paper.yaml`;
- policy congelata: `src/adaptive_bot/musca_v8_multi_horizon.py`;
- worker Binance: `src/adaptive_bot/musca_v8_binance.py`;
- motore paper condiviso: `src/adaptive_bot/musca_v5_paper.py`;
- report operativo: `data/reports/musca_v8_binance_paper.json`;
- stato persistente: `data/research/musca_v8_binance_paper_state.json`;
- profilo UI: `musca-v5-binance` sulla porta 8080;
- live reale: disabilitato, senza eccezioni automatiche.

## Obiettivo economico

L'obiettivo finale non è replicare la frequenza bassa della V8. È trovare, su Binance BTCUSDT,
la **massima frequenza giornaliera economicamente sostenibile** di micro-operazioni. Il modello può
e deve accettare trade negativi; si giudica sul saldo netto dopo fee, spread, slippage e funding.
Non viene imposto che ogni giornata sia positiva, condizione impossibile da garantire senza
overfitting: i gate richiedono expectancy OOS positiva, maggioranza robusta di giornate e finestre
positive, PF almeno 1,15, drawdown entro l'8% e stabilità al variare dei periodi.

La frequenza non è una soglia arbitraria da forzare: è la variabile da massimizzare sotto questi
gate. La V8 resta il controllo positivo a bassa frequenza. La policy ufficiale sarà sempre una sola,
Musca V5 Binance; esperimenti respinti non diventano nuove versioni operative.

Le analisi successive devono prima riusare funnel, upper bound e report Binance già calcolati.
Sono vietate nuove griglie brute-force costose se una diagnosi causale o statistica più piccola può
rispondere alla domanda. Ogni modifica deve spiegare quale collo di bottiglia economico rimuove.

Musca V5 usa come unica base Alpha la policy deterministica positiva V8
`IMPULSE_PULLBACK_MULTI_HORIZON`. Il fine tuning deve capire perché la base genera pochi trade e
aumentare in modo sostanziale gli eventi eseguibili senza forzare ingressi, ridurre artificialmente
i costi, usare il test per scegliere i parametri o trasformare FLAT in un profitto.

Il bot può e deve avere trade perdenti. Il risultato richiesto è un vantaggio aggregato netto:
expectancy positiva, profit factor e drawdown accettabili, più giornate positive che negative su un
campione sufficiente. Non è possibile garantire più vincite che perdite in ogni singola giornata.

Musca V2 e tutti i dati che alimentano V2/V5 devono essere conservati. Non creare nuove linee V9,
V10, ecc.: le modifiche accettate confluiscono in Musca V5. Denaro reale resta disabilitato.

## Baseline congelata

- Protocollo: `musca_v8_multi_horizon_impulse_pullback_v1`.
- Protocol hash: `0983acebf9a849c175259abf7f289b99841886d1032752bf1de0565922de7ad7`.
- Report: `data/reports/musca_v8_multi_horizon.json`.
- SHA-256 report: `3FAF76E207BBCF12FF3501B1A2160273A67745F4D84BF54F3C52913A5339252B`.
- Dataset realmente letto dalla V8: `data/ml/hybrid_v24/bars_5m.parquet`.
- SHA-256 dataset V8: `9BEB34C48F366F31CA2E4DFB466745C1224177421EABE2B220417A7782A834A0`.
- Il dataset V25 ha le stesse 241.921 righe e lo stesso intervallo temporale, ma schema diverso;
  non sostituisce automaticamente V24 perché cambierebbe la provenienza della baseline.
- Commit osservato: `4d8c110a6b8aa0c70f08079043a6639a659ed08b` con worktree già modificato;
  nessuna modifica esistente deve essere scartata.
- Holdout finale: sigillato, inizio `2026-05-11T11:30:00Z`.

Risultati reali del report corrente:

| Periodo | Trade | EV netto | PF | Win rate | Max drawdown |
|---|---:|---:|---:|---:|---:|
| 2024–2025 combinato | 92 | +5,43 bps | 1,155 | 50,0% | 5,99% |
| 2026 pre-holdout | 24 | +14,50 bps | 1,432 | 62,5% | 3,22% |
| 2026, costi 2× | 24 | +6,50 bps | 1,178 | 62,5% | 3,22% |

La frequenza corretta è circa 3,8 trade al mese nel 2024–2025, non 2,8. È ancora troppo bassa.
Il solo gate V8 fallito è la numerosità OOS: 24 trade contro 50 richiesti. Questi numeri sono
ricerca promettente, non una garanzia di profitti futuri.

## Stato verificato del fine tuning

La frontiera stateful corretta ha eliminato un errore di lineage dei primi audit: rimuovere il
filtro room cambiava lo stato del generatore e non produceva un superset della V8. Ogni soglia viene
ora rigenerata come percorso indipendente e il controllo 24 bps riproduce esattamente la baseline.

Con sizing coerente con rischio massimo 1%, margine 10% e leva 10×, il miglior candidato resta
VIP4, room `2× costo reale 1×`, H24: 152 trade nel 2024–2025, EV +7,015 bps, PF 1,229,
LCB +0,119 e drawdown 4,868%; nel 2026 pre-holdout produce 33 trade, EV +6,994 e PF 1,215,
ma fallisce count e LCB. H1/H2 non aggiungono frequenza robusta. Il paper attivo non cambia.

Il prossimo possibile vantaggio strutturale è l'esecuzione POST_ONLY, non un altro ritocco alle
stesse soglie. Al 2026-08-09 esistono 6 giornate Bitunix L2 materializzate più una in raccolta, ma
zero ordini/fill privati. Il report `data/reports/musca_v5_maker_feasibility.json` mantiene quindi
il maker fail-closed: il book pubblico non viene trasformato in un fill inventato e le fee maker
non entrano nell'economia del bot. Prima dell'addestramento Execution servono almeno 30 giornate e
100 esiti POST_ONLY completi con classi fill/non-fill, latenza, ruolo e adverse selection osservati.

## Strategia che non deve essere snaturata

1. Trend e direzione da 1h/4h con conferma Binance spot/perpetual.
2. Impulso causale su barra 5m chiusa: breakout 3/6/12/24/48, volume relativo e taker flow.
3. Pullback verso VWAP giornaliero, VWAP dell'impulso o anchored VWAP di swing.
4. Entrata soltanto su una nuova ripartenza causale coerente.
5. Spazio lordo sufficiente rispetto ai costi del profilo.
6. Stop strutturale oltre il pullback, mai allargato.
7. Uscita 50% a 1,5R, protezione dei costi, trailing 15m e timeout massimo 6h.
8. Una sola posizione, rischio 1%, margine massimo 10%, leva massima 10×.

Il modello generico Ridge/XGBoost con esito `NO_ECONOMIC_ALPHA` resta challenger diagnostico e non
ha autorità sugli ordini. FLAT significa soltanto nessun trade e ha P&L zero.

## Diagnosi FT-001

L'audit senza modifiche ha localizzato il problema: 2.668 restart confermati non sovrapposti in
756 giorni, cioè 3,53 candidati al giorno, ma il filtro di spazio fisso lascia passare circa
l'1,5–2,2%. Il MFE mediano dei candidati è 31,29 bps, quindi i micromovimenti esistono. Il loro
trading indiscriminato con target 1,5R è però negativo (EV -8,35 bps, PF 0,714). Il fine tuning deve
quindi apprendere **quale target economico è raggiungibile e in quanto tempo**, non inventare più
segnali né rimuovere FLAT.

I 3,53 candidati/giorno sono già successivi alla grammatica V8 completa e alla rimozione delle
sovrapposizioni; non rappresentano tutte le interazioni VWAP osservabili. Rimuovendo soltanto
dall'audit i filtri finali di spazio e range dello stop, ma conservando trend, impulso, conferme,
pullback e restart, risultano 6.809 decisioni uniche in 756 giorni: **9,01 candidati/giorno**. FT-002
deve ora misurare la frontiera frequenza–rendimento. Non viene imposto un numero arbitrario di
trade: si sceglie la frequenza massima alla quale i gate economici OOS restano validi.

## Perché la frequenza può sparire

Non si modificheranno parametri finché un audit a imbuto non avrà contato, separatamente LONG e
SHORT, quanti eventi sopravvivono a ciascun passaggio:

1. tutti i minuti osservabili;
2. trend 1h/4h concorde;
3. breakout per ciascun orizzonte;
4. conferma spot/perpetual;
5. volume relativo;
6. taker flow;
7. pullback verso ogni famiglia VWAP;
8. ripartenza entro la finestra causale;
9. spazio lordo minimo specifico del profilo fee;
10. eliminazione dei duplicati tra orizzonti;
11. trade rimossi perché una posizione precedente è ancora aperta;
12. trade rimossi da dati mancanti o non freschi.

L'audit deve inoltre distinguere: assenza di setup, setup sovrapposti, setup con movimento lordo
insufficiente, edge perso per i costi ed edge perso dalla gestione dell'uscita.

## Fine tuning preregistrato

Si parte da poche ipotesi economiche, non da una griglia:

1. **Re-entry sullo stesso impulso.** Dopo la chiusura di un trade, consentire un nuovo pullback e
   una nuova ripartenza indipendente sullo stesso impulso, con segnale nuovo e cooldown minimo.
   Non è consentito aggiungere a una posizione in perdita.
2. **Zona VWAP adattiva.** Misurare la zona con volatilità e spread osservati, mantenendo daily,
   impulse e swing VWAP; non trasformare ogni semplice attraversamento in un ordine.
3. **Ammissione economica per profilo.** Nessun multiplo rigido del costo. Un trade è ammissibile se
   `P(win) × gain − P(loss) × loss − costi` e il suo limite prudenziale sono positivi. Il rapporto
   movimento/costi viene riportato, mentre i costi 2× restano uno stress test diagnostico separato,
   senza autorità sugli ingressi o sulla promozione paper. VIP0 e VIP5
   non devono condividere una soglia fittizia.
4. **Uscita multi-orizzonte.** Verificare se la durata effettiva della posizione blocca nuovi setup;
   TP1, trailing e invalidazione possono liberare capitale prima, ma solo se migliorano EV netto.
5. **Conferme non ridondanti.** La conferma spot sull'impulso resta obbligatoria. La seconda
   conferma spot al restart, già individuata come ridondante in discovery, resta shadow finché dati
   cronologicamente nuovi non la confermano.

## Economia della policy, non perfezione del singolo trade

Il fine tuning non richiede che ogni ingresso sia previsto come vincente. Una strategia valida può
avere stop e giornate negative; viene accettata soltanto se l'intera sequenza OOS, con una posizione
alla volta, conserva expectancy netta positiva, PF, drawdown, stress costi e stabilità.

La gestione ufficiale da verificare è a due stadi:

1. TP1 lordo legato al costo reale del profilo (`1,5 ×` oppure `2 ×` il round-trip), raggiungibile
   in qualunque minuto; lo scadere dell'orizzonte è un timeout, non il momento obbligatorio di
   chiusura.
2. Al TP1, nuova decisione causale: chiudere tutto oppure realizzare una quota, proteggere costi e
   lasciare il resto verso TP2/trailing. Questa decisione usa stato, volatilità e order flow
   disponibili soltanto in quel momento.

Il filtro ML è valutato a livello di copertura/ranking della policy. Non può usare il futuro per
scegliere i trade, ma non deve imporre un LCB positivo a ogni singola operazione se tale vincolo
annulla una policy aggregata robusta.

## Diagnosi della classe di strategia frequente

L'audit diretto `musca_v5_strategy_class_diagnostic_v1` ha separato disponibilità del movimento e
prevedibilità. Su 168.959 decisioni causali a un minuto, l'oracle futuro VIP5 trova un'azione netta
positiva nel 66,37% dei casi: le oscillazioni non mancano. Momentum, taker flow, lato del VWAP e
gli otto stati combinati non producono però EV netto positivo prima dell'evento. Anche il ritorno
event-driven verso un rolling VWAP congelato resta negativo a tutte le distanze economiche
preregistrate. Il problema non è quindi il numero di target osservabili, ma l'assenza di
discriminazione causale nelle feature aggregate.

È respinto anche il market making ingenuo al best quote: lo spread Bitunix mediano osservato è
0,0155 bps contro 2 bps di fee maker round-trip VIP5. La componente frequente non può guadagnare
semplicemente lo spread visibile.

La linea ufficiale diventa ibrida senza creare un'altra versione:

1. V8 resta Alpha direzionale rara e immutata.
2. Il percorso frequente è execution-aware e centrato sul VWAP/fair value: decide se e dove
   quotare POST_ONLY, non cerca una direzione universale ogni minuto.
3. Il modello deve stimare fill, queue, adverse selection e probabilità di ritorno dalla quota al
   centro; una quota è valida soltanto se l'EV netto osservabile supera fee e rischio inventario.
4. Trend forte/V8 agisce da bias o kill switch; inventory skew e limiti di rischio impediscono a
   una griglia di accumulare contro trend.
5. Senza fill privati e giornate L2 indipendenti sufficienti, questa componente resta raccolta
   shadow e nessun costo maker viene assunto nel paper.

Sono già state falsificate e non vanno ripetute alla cieca:

- semplice estensione della finestra di restart da 6 a 12 barre: più eventi ma 2024/2025 negativi;
- eventi generici a 1 minuto: 123–194 eventi in tre mesi ma EV negativo;
- abbassamento indiscriminato di volume, confluence o profondità del pullback;
- ottimizzazione del modello generico per superare FLAT nonostante label negative.

## Raccolta execution-aware attiva

Dal 9 agosto 2026 il collector Bitunix conserva il book completo causale e un tape compatto per il
replay POST_ONLY. Il protocollo congelato
`2c1566922a582bb5572bcfd67a765610052a572836b7f6bfa084165457b4c0a5` usa fair value VWAP
dei trade a cinque minuti, quote reali full-depth, costo maker in ingresso più taker in uscita,
quantità coerente con il portafoglio paper, latenza 250 ms, coda conservativa e funding osservato
nella corretta unità frazionaria. VIP0–VIP5 restano esperimenti separati. Il replay pubblico è un
proxy di ricerca; non sostituisce ordini e fill privati Bitunix.

La prima valutazione è autorizzata separatamente per profilo dopo 10 giornate e 100 trade proxy
chiusi. L'Execution model resta disabilitato fino a 30 giornate e label private complete. I file
autorevoli sono `src/adaptive_bot/adapters/bitunix/collector.py`,
`src/adaptive_bot/musca_v5_post_only_replay.py`,
`data/raw/bitunix_microstructure/btcusdt_post_only_tape_YYYY-MM-DD.jsonl` e
`data/reports/musca_v5_post_only_replay.json`.

Gli audit upper-bound FT-023–FT-026 impediscono di trasformare questa raccolta in una quotazione
VWAP frequente automatica. Perfino con fill al primo touch, coda nulla e uscita maker regalata,
VIP5 ha prodotto 1.391 trade in sei giorni con EV −5,50 bps e nessuna giornata positiva. Il target
vale circa +1 bps, mentre un mancato rientro perde in media −11,33 bps; servirebbe una probabilità
target del 92,96%. Ridge e XGBoost su 23 feature L2 causali non hanno trovato discriminazione OOS.
Questa classe resta quindi respinta; il collector continua soltanto per acquisire verità privata
su fill/coda/adverse selection e non modifica Musca V5 paper.

La diagnosi e la scelta degli esperimenti sono responsabilità diretta di Codex. Il PC dell'utente
può eseguire soltanto estrazioni deterministiche, controlli mirati e un training finale già
preregistrato: niente brute force, sweep o tentativi ripetuti per cercare un risultato positivo.

La documentazione Bitunix consultata il 9 agosto 2026 espone soltanto il dominio REST reale
`https://fapi.bitunix.com` e i WebSocket pubblico/privato reali; l'“OpenAPI Demo” è un esempio che
richiede API key, non un dominio sandbox documentato. Fonti:
https://www.bitunix.com/api-docs/futures/common/introduction.html e
https://www.bitunix.com/api-docs/futures/websocket/prepare/WebSocket.html. Le label private
`POST_ONLY` possono quindi provenire soltanto da ordini realmente osservati e riconciliati con
`get_history_orders`/`get_history_trades`; nessun ordine reale viene inviato senza autorizzazione
esplicita e nessuna simulazione viene spacciata per fill dell'account.

Gli audit FT-028/FT-029 hanno inoltre respinto sia il lead–lag Binance→Bitunix immediatamente prima
del touch, sia la capacità delle quattro giornate Binance L2 di battere la previsione zero sui
ritorni a 5/30/60 secondi. Questi campioni non autorizzano un nuovo modello; il protocollo full-depth
continua immutato fino a giornate indipendenti sufficienti.

## Protocollo di valutazione

- La selezione usa solo dati anteriori al periodo valutato, con purge basato sull'uscita effettiva.
- Il 2026 già osservato è discovery/audit, non nuova conferma indipendente.
- Nessun risultato decide di aprire l'holdout sigillato.
- La frequenza è un vincolo secondario: prima economia positiva, poi più eventi.
- Requisito del generatore: registrare tutti i candidati causali unici e tutti i rifiuti. Il numero di
  trade non è preregistrato: la policy deve massimizzare la frequenza sotto i vincoli economici, non
  massimizzare il profitto apparente scegliendo pochi casi né raggiungere una quota forzata. I 3,8
  trade/mese della baseline sono il limite da superare, non un risultato accettato in partenza.
- Ogni variante registra candidati totali, trade eseguiti, sovrapposizioni, LONG/SHORT, sessione,
  regime, durata e ragione di rifiuto.

Gate economici per accettare una modifica nella linea paper Musca V5:

- expectancy netta positiva dopo fee, spread, slippage e funding osservabili;
- PF almeno 1,15;
- max drawdown massimo 8%;
- limite bootstrap inferiore dell'expectancy maggiore di zero quando la numerosità lo consente;
- stress costi 2× sempre riportato come diagnostica di robustezza, ma non usato come costo reale o
  gate operativo; fee, spread e slippage osservati 1× determinano l'economia della policy;
- maggioranza delle finestre cronologiche e delle giornate con trade positiva;
- frequenza scelta sulla frontiera OOS: nessuna policy a frequenza maggiore deve conservare gli
  stessi gate economici; il conteggio usa il periodo completo, non giorni scelti a posteriori;
- almeno 50 trade OOS per una decisione di ricerca e almeno 100 trade paper futuri prima di
  considerare il modello maturo;
- replay senza look-ahead, duplicati, posizioni sovrapposte, stop allargati o rischio oltre 1%.

I profili VIP sono valutati separatamente. Un fallimento VIP0 non invalida automaticamente VIP5;
nessun profilo può però usare costi diversi da quelli dichiarati.

## File autorizzati della linea ufficiale

- `src/adaptive_bot/musca_v8_multi_horizon.py`: base Alpha congelata e audit.
- `src/adaptive_bot/musca_v4_research.py`: generatore/label condiviso da modificare solo con
  protocollo registrato.
- `src/adaptive_bot/btc_cross_exchange_forward_audit.py`: collegamento Alpha–paper.
- `src/adaptive_bot/musca_v5_paper.py`: simulazione Bitunix persistente.
- `src/adaptive_bot/musca_v5_shadow.py`: stato e telemetria Musca V5.
- `src/adaptive_bot/dashboard/`: rappresentazione, senza autorità decisionale.
- `data/reports/musca_v8_multi_horizon.json`: baseline congelata.
- `data/reports/musca_v5_shadow.json`: stato paper corrente.
- `data/research/musca_v5_shadow_state.json`: portafoglio e posizioni persistenti.

## Ordine di lavoro dopo ogni ripresa

1. Leggere questo file e `docs/musca-v5-fine-tuning-log.md`.
2. Verificare che holdout e denaro reale siano ancora chiusi.
3. Riprendere il primo elemento `NEXT` del diario.
4. Registrare protocollo e hash prima di calcolare risultati.
5. Aggiornare il diario anche quando l'ipotesi fallisce.

## Chiusura della ricerca Binance ad alta frequenza - 9 agosto 2026

Su richiesta dell'utente la ricerca si ferma dopo l'ultimo protocollo multi-orizzonte. V8 rimane
soltanto il controllo positivo a bassa frequenza; non e' stata ripristinata come soluzione finale.

L'ultimo universo Binance contiene 521.070 decisioni causali ogni 30 secondi e 72 feature ottenute
da BTCUSDT spot/perpetual, aggTrades 5s, volume, taker flow, open interest, mark, funding, VWAP
giornaliero/rolling/anchored, volatilita' e trend. Ogni stato offre LONG e SHORT con piani a
3/5/10/15/30/60 minuti. Luglio resta sigillato. L'oracle diagnostico dimostra che lo spazio delle
azioni contiene movimenti sufficienti, ma non che siano prevedibili.

Il confronto cronologico Ridge-XGBoost seleziona XGBoost. La policy non supera la selezione di
aprile: aumentando la copertura, l'expectancy diventa negativa prima di raggiungere tre trade al
giorno. Il controllo maggio-giugno allo 0,1% trova 17 trade, 0,279/giorno, EV +14,366 bps, PF 1,409
e stress 2x +5,366 bps, ma e' troppo piccolo e non e' una policy selezionata. Non viene prodotto
alcun bundle e paper/live restano invariati.

File autorevoli:

- `src/adaptive_bot/musca_v5_binance_micro_ranker.py`;
- `tests/unit/test_musca_v5_binance_micro_ranker.py`;
- `data/ml/musca_v5/binance_30s_policy_matrix.parquet`;
- `data/reports/musca_v5_binance_policy.json`;
- `data/reports/musca_v5_binance_policy.status.json`.
