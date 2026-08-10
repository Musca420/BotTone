Verdetto tecnico

La V22-A dimostra che il problema non è:

FLAT troppo prudente;
Ridge poco potente;
XGBoost inadeguato;
una singola combinazione sbagliata di stop e target;
il costo di 8 bps troppo severo.

Il problema è precedente al modello:

Gli eventi extension, rejection e confirmed_reentry, presi nelle forme attuali, non forniscono sufficiente informazione direzionale per decidere FADE o FOLLOW.

FADE ha prodotto -0,1456 R con costi da 4 bps e zero fold positivi su 15. FOLLOW ha prodotto -0,2869 R, anch’esso con zero fold positivi.

Non avrebbe senso addestrare Tradability, Action Advantage o Exit Model sopra queste label: il modello cercherebbe soltanto di estrarre sottogruppi casualmente positivi da una popolazione strutturalmente negativa.

Cosa ha dimostrato la V22-A
1. Modificare soltanto stop, target e timeout non basta

Sono state valutate 108 configurazioni:

54 FADE;
54 FOLLOW;
selezione della configurazione esclusivamente nel training fold;
test OOS separato;
costi 4 e 8 bps;
una posizione alla volta;
ingresso dal minuto successivo;
purge sull’uscita effettiva.

Questa è una verifica abbastanza ampia e correttamente separata per concludere che non esiste una geometria di uscita semplice capace di trasformare gli eventi attuali in una strategia profittevole.

2. Anche il migliore sottogruppo resta negativo

Il risultato meno negativo è:

FADE su confirmed_reentry;
expectancy -0,1010 R;
profit factor 0,323;
bootstrap lower bound -0,1185 R.

Non è un risultato “quasi buono”. È ancora nettamente lontano dall’equilibrio economico.

Anche separando i regimi, nessuna combinazione si avvicina ai gate. La migliore indicazione relativa è FOLLOW in trend rialzista con -0,1671 R e PF 0,663, ancora insufficiente.

3. Il prezzo si muove, ma non nella direzione prevedibile dall’evento

Il dato più importante del report è la quasi simmetria tra MFE e MAE a 60 minuti:

FOLLOW: MFE 1,153 ATR, MAE 1,099 ATR;
FADE: MFE 1,099 ATR, MAE 1,153 ATR.

Questo significa che dopo gli eventi esiste movimento, ma la direzione favorevole e quella avversa hanno dimensioni molto simili. Il rapporto rischio/opportunità iniziale è vicino a uno già prima dei costi.

In termini semplici:

Il VWAP identifica una zona di attività e movimento, ma gli eventi attuali non identificano in modo sufficiente quale lato vincerà.

4. Il ritorno al VWAP non è abbastanza frequente

Nel 63,30% degli eventi il VWAP non viene raggiunto entro 60 minuti. Anche metà della distanza non viene raggiunta nel 36,81% dei casi.

Questo è particolarmente rilevante per FADE: “prezzo distante dal VWAP” non implica automaticamente “ritorno al VWAP in un orizzonte negoziabile”.

Il VWAP può essere:

un attrattore;
un livello di equilibrio ormai superato;
un riferimento che si sposta insieme al prezzo;
una zona lontana durante un nuovo regime;
una conseguenza del movimento, non una causa del ritorno.
Cosa non bisogna fare adesso
Non abbassare i gate

Abbassare expectancy, profit factor o bootstrap richiesto non crea edge. Autorizzerebbe soltanto una strategia già risultata negativa.

Non scegliere la migliore configurazione ex-post

Scegliere adesso FADE su confirmed_reentry perché è “la meno negativa” sarebbe data snooping. È comunque negativa e la scelta avverrebbe dopo aver visto l’OOS.

Non aggiungere subito XGBoost, reti neurali o reinforcement learning

Un modello più potente può trovare divisioni più elaborate nei dati, ma con label strutturalmente negative e pochi eventi aumenterebbe soprattutto il rischio di overfitting.

Il machine learning deve essere utilizzato dopo aver formulato una nuova ipotesi causale, non per cercare retroattivamente un angolo profittevole tra migliaia di combinazioni.

Non considerare il paper trading una validazione della V22-A

Potete creare una modalità di ricerca con denaro virtuale, ma deve essere marcata chiaramente:

RESEARCH_ONLY
NO_DEPLOYMENT
NO_REAL_CAPITAL

La V22-A non autorizza una policy operativa.

V22.1: revisione tecnica necessaria

Il report segnala due elementi incompleti:

una riga con rappresentazione incoerente tra available_at e feature_available_at;
assenza di time_to_outer_band e time-to-invalidation separati per FADE e FOLLOW.

Questi elementi devono essere corretti per completare tecnicamente il Path Label, ma non cambiano il verdetto economico della V22-A.

La V22.1 dovrebbe quindi essere una revisione di completezza, non un nuovo tentativo di ottimizzazione:

- correggere il timestamp sintetico;
- aggiungere time_to_outer_band;
- aggiungere time_to_invalidation_FADE;
- aggiungere time_to_invalidation_FOLLOW;
- aggiungere test unitari;
- mantenere congelato il risultato economico V22-A;
- non riaprire la ricerca delle configurazioni sulla stessa OOS.
La nuova strada deve partire dall’ingresso

La V22-A ha separato correttamente la geometria dell’uscita e ha scoperto che il problema è principalmente quando e perché entrare.

La prossima versione non dovrebbe chiedere:

“È avvenuta un’extension: faccio FADE o FOLLOW?”

Dovrebbe chiedere:

“Quale conferma osservabile rende questa specifica extension un’extension continuativa oppure terminale?”

L’evento VWAP deve diventare un generatore di candidati, non il segnale d’ingresso definitivo.

Nuova architettura consigliata
Fase 1 — Evento VWAP

Gli eventi attuali rimangono utili per individuare situazioni interessanti:

extension;
rejection;
confirmed reentry.

Ma non aprono direttamente una posizione.

Producono uno stato:

VWAP_EVENT_DETECTED
WAITING_FOR_CONFIRMATION
Fase 2 — Finestra di conferma causale

Dopo l’evento, il sistema osserva per esempio da 1 a 5 minuti, usando esclusivamente informazioni disponibili in tempo reale.

La conferma deve distinguere tra:

FADE confermato

Possibili segnali:

fallimento di un nuovo massimo/minimo;
riduzione dell’aggressione taker;
ritorno dentro la banda superata;
divergenza tra movimento del prezzo e volume;
open interest che diminuisce durante l’estensione;
forte aggressione senza avanzamento del prezzo;
chiusura della candela verso il VWAP;
diminuzione della velocità di allontanamento;
passaggio della distanza da accelerazione positiva a negativa.
FOLLOW confermato

Possibili segnali:

permanenza oltre la banda;
nuovo massimo/minimo accompagnato da volume;
aggressione taker persistente;
open interest crescente;
mancato ritorno nella banda;
VWAP inclinato nella stessa direzione;
espansione concorde tra spot e perpetual;
pullback debole seguito da nuova accelerazione;
superamento dell’estremo dell’evento.

Questa conferma deve modificare l’entry timestamp. Il trade non parte più automaticamente al minuto successivo all’evento.

Ipotesi V23 proposte

La prossima versione dovrebbe testare poche ipotesi economiche preregistrate.

Ipotesi A — FADE dopo fallimento dell’estensione

Un FADE viene autorizzato soltanto quando:

il prezzo supera una banda VWAP;
forma un’estensione significativa;
non riesce a produrre un nuovo estremo;
rientra nella banda;
l’aggressione diminuisce o si inverte.

Ingresso:

al rientro causale nella banda;
oppure alla rottura del minimo/massimo della candela di fallimento.

Target:

prima quota a metà distanza;
seconda quota verso VWAP.

Invalidazione:

nuovo estremo oltre l’extension.

Questa è un’ipotesi diversa dal semplice “prezzo lontano dal VWAP”.

Ipotesi B — FOLLOW dopo accettazione oltre la banda

Un FOLLOW viene autorizzato soltanto quando:

il prezzo supera una banda;
rimane oltre la banda per un tempo minimo;
viene negoziato volume sufficiente oltre il livello;
il pullback non riesce a rientrare stabilmente;
il movimento riparte nella direzione dell’estensione.

Ingresso:

superamento del massimo/minimo del pullback;
non all’extension iniziale.

Invalidazione:

accettazione nuovamente dentro la banda.

Questa struttura tenta di evitare l’ingresso nel punto di massima estensione.

Ipotesi C — VWAP come filtro, non come sorgente principale

Il segnale principale può provenire da:

trend;
breakout;
order flow;
volatilità;
cross-market confirmation.

Il VWAP determina soltanto:

se il prezzo è troppo esteso;
se l’ingresso è vicino a un livello di equilibrio;
dove posizionare invalidazione o target;
se usare logica FADE o FOLLOW.

Questa potrebbe essere la strada più robusta:

Il bot non deve necessariamente “tradare il VWAP”; può tradare una struttura di mercato usando il VWAP come riferimento contestuale.

Dati mancanti che limitano la ricerca

La V22-A utilizza percorsi OHLCV e mark price a un minuto. Non dispone di:

sequenza tick;
aggregate trade;
spread storico;
bid/ask storico;
book L2;
latenza e fill reali.

Questi dati non sono stati simulati, scelta corretta, ma limitano la capacità di identificare assorbimento, aggressione e microstruttura della conferma.

Di conseguenza ci sono due percorsi possibili.

Percorso 1 — Continuare con dati a un minuto

Usare conferme osservabili da OHLCV:

chiusura dentro/fuori banda;
fallimento di nuovi estremi;
volume relativo;
taker imbalance già disponibile;
variazione OI;
slope VWAP;
persistenza;
pullback;
breakout del pullback.

È il percorso più rapido, ma con informazione limitata.

Percorso 2 — Costruire una vera ricerca microstrutturale

Acquisire o registrare:

aggregate trades Binance;
aggressor flow;
spread;
book L2;
profondità;
cancellazioni e consumo del book;
spot e perpetual sincronizzati.

Questo consentirebbe di distinguere:

assorbimento;
esaurimento;
continuazione;
liquidazione;
breakout sostenuto;
fake breakout.

È più lungo, ma maggiormente coerente con l’obiettivo di un bot VWAP serio.

Paper trading utile anche senza modello promosso

Potete iniziare una policy paper esplorativa, ma non usando la V22-A come sistema profittevole.

La policy dovrebbe registrare per ogni evento:

evento VWAP
conferma osservata
azione teorica
entry proposta
stop e target
MFE
MAE
time-to-VWAP
time-to-invalidation
risultato con 4 bps
risultato con 8 bps
spread al momento dell’ingresso
stato del book
order flow
open interest
decisione FADE/FOLLOW/NO_CONFIRMATION

Per raccogliere esempi, può effettuare un numero limitato di trade virtuali anche senza gate economico, per esempio:

massimo 1 FADE e 1 FOLLOW per sessione;
esposizione esclusivamente virtuale;
nessuna selezione successiva sulla stessa finestra senza nuovo protocollo;
log anche dei segnali non eseguiti.

Il vantaggio è che iniziate a produrre dati realistici di conferma ed esecuzione senza rischiare denaro.

Prossima sequenza corretta
V22.1 — Completamento tecnico
Correzione timestamp.
time_to_outer_band.
Invalidazione separata FADE/FOLLOW.
Test aggiuntivi.
Nessuna riapertura del gate economico.
V23-A — Confirmed-entry audit
Nuove regole preregistrate di conferma.
Nessun machine learning.
Ingresso non automatico.
Confronto tra entry immediata ed entry confermata.
Walk-forward congelato.
V23-B — Tradability model

Autorizzato soltanto se almeno una famiglia di entry confermata presenta:

expectancy positiva;
profit factor accettabile;
fold positivi non isolati;
stabilità sotto stress costi.
V23-C — Action model

Soltanto dopo il gate economico:

classificazione FADE/FOLLOW;
regressione di ΔEV;
quantili MFE/MAE;
probabilità di invalidazione.
V23-D — Paper research
Trading virtuale;
raccolta microstrutturale;
nessuna promozione automatica.
V23-E — Conferma temporale futura

La validazione finale deve avvenire su dati successivi al 5 agosto 2026, senza riutilizzare le stesse finestre per correggere l’ipotesi.

Conclusione

La V22-A ha lavorato correttamente e ha fornito una risposta importante:

Non esiste un vantaggio sufficiente nell’operare automaticamente FADE o FOLLOW quando viene rilevato uno degli attuali eventi VWAP.

Questo non dimostra che un bot VWAP sia impossibile. Dimostra che:

il VWAP da solo non è un segnale direzionale;
l’evento deve generare un candidato, non un ordine;
serve una conferma causale dell’ingresso;
FADE e FOLLOW devono essere definiti attraverso strutture di conferma differenti;
modelli ML ed exit adattive devono essere introdotti soltanto dopo aver trovato una famiglia economicamente valida.

La prossima evoluzione corretta non è una V22-B con più machine learning. È una V23 dedicata alla qualità e alla conferma dell’ingresso.