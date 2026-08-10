# Musca V5 — diario persistente di fine tuning

Questo file è append-only per gli esperimenti. Va letto insieme a
`docs/musca-v5-official-fine-tuning.md` dopo ogni compattazione del contesto.

## Stato corrente

- `OFFICIAL_BASE`: Musca V8 multi-horizon, integrata nella linea paper Musca V5.
- `STATUS`: `RESEARCH_BASE_ALPHA_READY`.
- `REAL_CAPITAL`: vietato.
- `FINAL_HOLDOUT`: sigillato.
- `CURRENT_LIMIT`: 92 trade nel 2024–2025 e 24 nel 2026 pre-holdout; frequenza inaccettabile.
- `FREQUENCY_REQUIREMENT`: massimo numero di trade compatibile con EV netto, PF, drawdown e stress
  OOS; nessuna quota giornaliera arbitraria.
- `CANDIDATE_REQUIREMENT`: registrare l'intero pool causale, inclusi tutti i rifiutati.
- `GENERIC_ML`: `NO_ECONOMIC_ALPHA`, challenger senza autorità sugli ordini.
- `ANALYSIS_OWNERSHIP`: Codex esegue direttamente l'analisi causale dei dataset e decide quali
  ipotesi meritano verifica. Il PC dell'utente non va usato per ricerca brute-force o tentativi
  ciechi: serve soltanto per estrazione/riproduzione indispensabile, test mirati e training finale
  preregistrato già giustificato dall'analisi. Ogni training costoso deve avere domanda, split,
  metrica economica e criterio di arresto definiti prima dell'avvio.
- `NEXT`: lasciare immutato il protocollo POST_ONLY FT-022 mentre il tape full-depth raccoglie
  giornate indipendenti. Gli upper bound FT-023–FT-026 hanno già falsificato la quotazione VWAP
  frequente non direzionale: non ripeterla né integrarla nel paper. La prossima analisi ammessa deve
  cambiare l'asimmetria economica delle uscite oppure aggiungere informazione causale realmente
  osservata. Anche lead–lag pre-fill e predizione Binance L2 FT-028/029 sono respinte. Prima lettura
  del replay a 10 giorni/100 trade proxy e training Execution soltanto a 30 giorni con fill privati.
  V8 resta Alpha direzionale rara.

## Registro

### 2026-08-09 — FT-000: congelamento della baseline

**Domanda.** Qual è l'unica base che ha mostrato economia positiva senza abbassare i costi?

**Protocollo.** Consolidamento dei risultati già prodotti; nessun nuovo training e nessuna apertura
dell'holdout.

**Risultato.** V8 multi-horizon: 92 trade 2024–2025, EV +5,43 bps, PF 1,155; 24 trade 2026
pre-holdout, EV +14,50 bps, PF 1,432; stress 2× +6,50 bps. Fallisce soltanto il gate di numerosità
nel report aggregato. VIP0 è positivo nello scenario normale ma negativo a costi 2×; VIP1–VIP5
sono positivi nel test normale e VIP1–VIP5 non negativi nello stress 2026, con margini differenti.

**Decisione.** Questa è l'unica base ufficiale per il fine tuning. Non promossa a denaro reale.

**Correzione di lineage.** La V8 importa `BARS` da `musca_v4_research.py` e legge
`data/ml/hybrid_v24/bars_5m.parquet` (241.921 righe, 2024-04-15–2026-08-03), non il parquet V25.
V25 ha lo stesso numero di righe e intervallo ma schema differente; nessuna sostituzione implicita.

**Prossimo passo.** FT-001: contare la caduta incrementale di eventi a ogni filtro, i duplicati tra
orizzonti e i segnali persi per posizione ancora aperta. L'audit non deve cambiare la strategia.

### 2026-08-09 — FT-001: audit a imbuto della frequenza

**Ipotesi preregistrata.** Determinare se la bassa frequenza nasce da assenza di setup, conferme,
spazio economico, duplicati tra orizzonti o posizione occupata. Nessun parametro modificato.

**Protocol hash.** `0b3e8bdc6a1e2a13f3c0d1c6b596ce93ba6d20918c0d726fd375ba9b52ef1bc3`.

**Risultato.** Per ciascun orizzonte vengono confermati 5.084–9.163 restart, ma soltanto 114–136
superano lo spazio fisso di 24 bps: la soglia conserva appena l'1,48–2,24% dei restart confermati.
I cinque orizzonti producono 547 righe primarie ma solo 120 segnali unici; 427 sono duplicati dello
stesso evento. La posizione occupata elimina soltanto 4 dei 120 segnali primari.

Senza il filtro di spazio esistono 3.360 segnali unici etichettati e 2.668 non sovrapposti in 756
giorni, pari a 3,53 candidati/giorno. Il MFE mediano è 31,29 bps; il 58,47% raggiunge almeno 24 bps
favorevoli prima dell'uscita osservata. Tuttavia la gestione V8 a 1,5R applicata indiscriminatamente
a questi candidati produce EV -8,35 bps e PF 0,714.

Le famiglie già disponibili non risolvono il problema: `IMPULSE_REENTRY` ha 12 trade, EV -22,22
bps e PF 0,50; `SESSION_SWING_PULLBACK` ha 201 trade, EV -7,14 bps e PF 0,80.

**Decisione.** Il problema non è la generazione di eventi. È l'uso dell'estremo dell'impulso come
proxy rigido dello spazio e l'uscita 1,5R unica per restart con dinamiche differenti. Vietato
rimuovere semplicemente il filtro o abilitare le famiglie negative.

**Prossimo passo.** FT-002: costruire label causali target-before-stop a più orizzonti e target
economici legati a volatilità/costi; la direzione resta quella V8. Il modello potrà filtrare e
scegliere target/uscita, non inventare LONG/SHORT.

### 2026-08-09 — FT-002A: materializzazione del pool causale completo

**Ipotesi preregistrata.** Rimuovere soltanto dall'audit i filtri finali `room_bps` e range dello
stop, conservando trend, impulso, spot/perpetual, volume, taker flow, pullback VWAP e restart.
Nessun cambiamento alla policy paper attiva.

**Risultato.** 24.290 azioni-esperto tra i cinque orizzonti diventano 6.809 decisioni uniche in 756
giorni: 9,01 candidati/giorno. Periodi: 2.449 nel 2024, 3.309 nel 2025 e 1.051 nel 2026
pre-holdout. Famiglie: 5.312 primi pullback e 1.497 re-entry. Dataset:
`data/ml/musca_v5/fine_tuning_candidates.parquet`.

**Decisione.** Il generatore V8 contiene già un pool giornaliero ampio. Non servono quote di trade
né nuovi segnali inventati; va stimata la frontiera economica dei target/stop/uscite causali.

**Prossimo passo.** FT-002B: label MFE/MAE, target-before-stop e payoff dei piani preregistrati,
quindi frontiera tra copertura dei candidati ed EV netto per ogni livello VIP.

### 2026-08-09 — FT-002B: correzione della barriera economica

**Problema.** Il vincolo rigido `movimento programmato >= 3 × costo round-trip` elimina setup che
possono avere EV positivo grazie a probabilità elevata e perdita contenuta, e duplica lo stress a
costi 2×.

**Decisione preregistrata.** Rimuovere il multiplo 3× dall'ammissione. Il criterio diventa
`P(win) × gain − P(loss) × loss − costi > 0`, con limite prudenziale positivo. Il rapporto fra
movimento e costi resta telemetria; i costi raddoppiati restano un gate di stress separato.

### 2026-08-09 — FT-002C: esito del filtro statico

**Protocol hash.** `592d4b77aa921882ce8517f707f325e6d99719f04b91c723630858ccd39e8a41`.

**Correzioni verificate.** Un mancato TP1 non viene più trasformato obbligatoriamente in perdita:
la decomposizione usa il rendimento firmato del ramo `no_hit`. Sono state aggiunte, con join solo
backward e fail-closed, 242.200 osservazioni Binance ufficiali di OI, order flow, metriche e
profondità aggregata. Copertura matrice 93,34%; 6.180 righe escluse; zero violazioni temporali.

**Risultato 2025 H2.** Il pool contiene 1.645 segnali unici. XGBoost trova per VIP5 34 trade con
EV +5,44 bps, PF 1,256 e drawdown 6,73%; per VIP3 33 trade con EV +4,42 bps e PF 1,198. Il gate di
50 trade non viene superato.

**Audit 2026 pre-holdout.** La stessa soglia VIP5, scelta soltanto sul 2025 H2, produce 43 trade,
EV −4,26 bps, PF 0,861, drawdown 13,40% e stress 2× −12,26 bps. Tutti i profili falliscono
l'audit. Il filtro statico è respinto e non viene collegato al paper.

**Decisione.** La V8 congelata resta fallback ufficiale. Il verdetto corretto del report è
`BASE_ONLY_NO_FREQUENCY_GAIN`, non `NO_ALPHA`.

### 2026-08-09 — FT-003: economia di policy e uscita a due stadi

**Chiarimento dell'obiettivo.** La policy non deve prevedere soltanto trade individualmente
vincenti. Deve eseguire un insieme causale di trade, accettare stop e perdite, e risultare positivo
in aggregato dopo costi. FLAT resta disponibile quando non esiste un setup, ma il limite
prudenziale non deve cancellare ogni singolo trade prima che la strategia possa esprimersi.

**Protocollo preregistrato.** Per ogni profilo fee, testare prima due TP1 derivati economicamente:
`1,5 × costo round-trip` e `2 × costo round-trip`. Il trade può chiudersi in qualunque minuto al
TP1 o allo stop strutturale; 60 minuti sono soltanto il timeout massimo. Dopo un TP1, il secondo
stadio confronta `chiudi tutto` contro `parziale + TP2/trailing` usando soltanto feature disponibili
al momento del TP1. I gate sono applicati alla sequenza OOS completa.

**Prossimo passo.** Etichettare e valutare la policy deterministica TP1 prima di autorizzare il
filtro ML o la decisione di continuazione.

### 2026-08-09 — FT-003A: esito TP1 e correzione del gate costi

**Protocol hash TP1.** `162a739bf9e071be045604b09ce36b76ac65cd0be3f09e32ef1d5485e6a7876e`.

**Risultato.** La matrice contiene 77.904 esiti per 6.492 candidati unici. La policy larga esegue
circa 6–8 trade/giorno, ma il rendimento lordo medio è vicino a zero: il netto perde quindi circa
un round-trip per trade. VIP5 a TP1 `2×` produce nel 2025 2.930 trade, EV −8,19 bps e PF 0,336;
nel 2026 pre-holdout 914 trade, EV −8,64 bps. Stop medio 21,2 bps contro target 16 bps e hit rate
51,5%: il payoff non raggiunge il break-even.

**Walk-forward.** Protocol hash
`b3579334158b555ecb0b9d64784b9d44a1029c2b81812651d5ddf61ad63e6c7b`; 18 fold, copertura causale
93,38%. Nessuna frontiera è stata autorizzata e nessun trade è stato inviato al paper. Verdetto:
`BASE_ONLY_NO_WALK_FORWARD_GAIN`.

**Incongruenza scoperta.** Chiudere tutto a TP1 `1,5×` o `2×` e richiedere contemporaneamente EV
non negativo con costi `2×` rende il gate quasi impossibile: un target `1,5×` è già negativo nello
stress, un target `2×` vale zero prima di qualunque perdita. I costi realmente simulati restano
fee, spread e slippage 1× del profilo. Il 2× viene mantenuto nel report come scenario diagnostico,
non come gate operativo e non è collegato alla leva 10×.

**Decisione.** La policy full-TP1 è respinta; V8 resta fallback ufficiale. Il prossimo esperimento
è il secondo stadio già preregistrato: parziale a TP1, protezione dei costi e TP2/trailing usando
stato causale aggiornato di volatilità, volume e order flow.

### 2026-08-09 — FT-004: runner cost-linked preregistrato

**Ipotesi preregistrata.** Il problema della policy larga non è la scarsità dei candidati ma il
payoff della chiusura completa a TP1. Senza cambiare direzione o generatore, realizzare il 50% a
TP1 e lasciare il 50% restante verso un TP2 può aumentare il vincitore medio senza richiedere che
ogni trade sia previsto positivo.

**Piani.** Due soli piani, già derivati dai due TP1 FT-003: `TP1=1,5× costo, TP2=3× costo` e
`TP1=2× costo, TP2=4× costo`; quota TP1 50%, timeout massimo 60 minuti. Dopo TP1 lo stop viene
alzato al minimo che protegge l'intero costo 1× più 1 bps sul trade complessivo e non viene mai
allargato. Il trailing conserva esattamente quella distanza iniziale dal TP1. Stop avverso nella
stessa candela continua a vincere. Il costo 2× resta soltanto nel report.

**Decisione TP1.** La matrice registra separatamente `chiudi tutto a TP1` e il rendimento
incrementale del runner. Volatilità, volume e order flow all'ingresso restano nel filtro; lo stato
aggiornato al TP1 sarà unito soltanto con timestamp disponibile dopo la chiusura della candela che
ha toccato TP1, quindi senza usare dati futuri.

**Gate.** Economia, PF, drawdown, LCB e stabilità sono calcolati ai costi reali 1×. Nessun
collegamento al paper prima del walk-forward; holdout e denaro reale restano chiusi.

**Risultato deterministico.** Protocol hash
`9bb17ca25389a0c84fe9ed3f4893ac5bd6fc3da8a14ea73a77dd9fb49c778887`. VIP5 `2×→4×` esegue
4.921 trade nel 2024–2025 con lordo medio −0,10 bps e netto −8,10 bps; nel 2026 pre-holdout 888
trade con lordo −0,64 e netto −8,64 bps. Il runner aumenta alcuni vincitori ma non crea edge nel
pool largo. Anche l'oracle di sola uscita `max(chiudi TP1, runner)` resta negativo: −6,74 bps
prima del 2026 e −7,23 bps nel 2026 per VIP5.

**Risultato walk-forward.** Dopo aver separato i costi 1× dallo stress 2×, è stato scoperto e
corretto un secondo errore: la soglia sull'EV isotonic espandeva il 10% sui molti pareggi. Il
ranking ora usa il punteggio grezzo continuo e l'EV calibrata resta una stima economica. Protocol
hash corretto `9530b6b0cc5ed289e546620d27b6ddbb40917382043cbba16578c69af85b6e0d`.
Nessuno dei 18 fold trova una frontiera valida. Il miglior fold VIP5 ha EV −3,95 bps; negli ultimi
fold 2026 i migliori decili restano tra circa −5 e −6 bps. Ridge è champion in 18/18; la
correlazione di ranking OOS è soltanto 0,026 grezza e 0,062 calibrata. Il decile previsto migliore
del fold finale ha lordo medio +1,75 bps, sotto gli 8 bps di costo VIP5.

**Decisione.** FT-004 è respinto e non cambia il paper. Il problema non è più il gate 2× né il
pareggio isotonic: lo stato a 1–15 minuti non discrimina causalmente i micromovimenti netti.

### 2026-08-09 — FT-005: discriminazione microstrutturale a un secondo

**Ipotesi preregistrata.** Usare esclusivamente i `aggTrades` Binance ufficiali già archiviati per
aggiungere OFI 1/3/5/15/30s, persistenza, intensità, accelerazione del flusso, assorbimento,
velocità del prezzo e volatilità 30s allo stesso generatore e agli stessi piani FT-004. Nessuna
nuova direzione, nessun book storico inventato.

**Periodo.** Gennaio–10 maggio 2026, con join backward su `available_at`; maggio viene letto con
filtro parquet `< 2026-05-11T11:30Z`. Luglio e ogni evento successivo all'holdout restano esclusi.
Split mensili cronologici con fit, calibrazione, selezione e test separati. Il campione è ricerca,
non conferma finale.

**Criterio.** Ridge resta champion; XGBoost GPU è challenger sulle stesse righe. Costi reali 1×,
stress 2× diagnostico. Anche un risultato positivo rimane `RESEARCH_ONLY` e non modifica il paper
finché non esistono numerosità e conferma cronologica futura sufficienti.

**Risultato.** Protocol hash
`b9a567d18cc9ebb3216483562250f4f15ca5f77c8fe23968652993914f86299e`; 11.688 righe con copertura
micro 100% e zero look-ahead. Nel fold di selezione marzo, il miglior decile VIP5 ha 22 trade,
EV +0,43 bps, PF 1,14 e LCB −3,86: insufficiente. Nella selezione aprile il segnale sparisce e
l'intero pool resta negativo. Ridge batte XGBoost in MAE in entrambi i fold; nessun modello trova
una frontiera valida. Verdetto `BASE_ONLY_NO_MICRO_ENTRY_GAIN`.

**Decisione.** Fotografie microstrutturali all'istante del segnale respinte; nessun cambiamento al
paper. Il segnale marginale non è una conferma e non giustifica l'abbassamento di PF, LCB o count.

### 2026-08-09 — FT-006: timing causale della ripartenza a 5 secondi

**Ipotesi preregistrata.** Il segnale V8 a 5 minuti definisce il setup, ma non obbliga l'ingresso
immediato. Entro cinque minuti si attende una ripartenza concorde già preregistrata nel motore
Musca V5: velocità prezzo 15s ≥ 0,5 bps, OFI 15s ≥ 0,05, OFI 1m ≥ −0,02 e intensità trade ≥ 0,8,
tutte firmate nella direzione V8. L'entrata avviene al bucket 5s successivo; se la conferma non
arriva, il setup scade senza ordine.

**Gestione.** Stop causale sul recente percorso 5s, minimo 12 e massimo 60 bps; piani cost-linked
FT-004 per ciascun VIP; 50% a TP1, costo 1× protetto, TP2/trailing, timeout 60 minuti. Stop vince
nello stesso bucket e un gap viene riempito al prezzo peggiore. I dati dopo l'holdout restano
esclusi tramite filtro parquet.

**Valutazione.** Stessi due fold cronologici FT-005 e stessi gate 1×. Il confronto misura se il
timing cambia realmente i label; non ottimizza le quattro soglie micro. Risultato sempre
`RESEARCH_ONLY`, senza autorità sul paper.

**Risultato.** Protocol hash
`21b017b9ed09aa2d2c0e566267905c82accd32234b42d04a7c18689aecb80a36`. Sono confermati 1.000 dei
1.050 setup (95,2%): il filtro quasi non discrimina. Lordo medio VIP5 −0,21 bps per TP1 1,5× e
−0,35 bps per TP1 2×; i migliori decili di marzo e aprile restano negativi. Nessuna fascia di
ritardo 0–300s è netta positiva; la fascia 60–120s ha lordo +2,04 bps, ancora sotto gli 8 bps di
costo VIP5. Nessuna frontiera Ridge/XGBoost valida.

**Decisione.** FT-006 respinto; vietato ottimizzare a posteriori le quattro soglie di conferma.

### 2026-08-09 — FT-007: target con floor costo e rischio

**Causa preregistrata.** Lo stop micro medio è 18,04 bps; nel 38,3% dei trade supera il TP1 VIP5
da 16 bps. Il target legato soltanto alle commissioni può quindi avere reward/risk inferiore a uno
anche prima del modello.

**Piani.** Per entrambi i floor costo FT-004, `TP1=max(1,5× o 2× costo, 1R)` e
`TP2=max(3× o 4× costo, 2R)`. Il trailing dopo TP1 usa la distanza fra il TP1 effettivo e lo stop
che protegge costo 1× + 1 bps; non resta fisso quando la volatilità richiede un target più ampio.

**Timing.** Due sole azioni preregistrate sullo stesso setup: ingresso immediato al prossimo bucket
5s e ingresso dopo la conferma FT-006. La seconda resta controllo, non viene ritoccata. Modello e
gate scelgono la copertura su finestre precedenti; nessun esito modifica il paper.

**Risultato.** Protocol hash
`97f2cf092f9bd46735626d9769d6b24488752d69a6a9ee44225c70ddb39f574a`; 24.312 azioni e zero
look-ahead. Per VIP5, ingresso immediato/TP1 1,5× ha lordo −0,08 bps; confermato −0,24. Il floor in
R non cambia il fatto che il pool medio è privo di edge. Nel miglior punto della selezione marzo
VIP5 ha 47 trade, EV −3,69 bps e PF 0,48; aprile è peggiore. Nessuna frontiera valida.

**Decisione.** FT-007 respinto; target, trailing e timing non vengono più modificati su questi
fold. La matrice 5s resta il dataset ufficiale per verificare la capacità del modello.

### 2026-08-09 — FT-008: EV probabilistica condizionata

**Ipotesi preregistrata.** La regressione diretta comprime una distribuzione asimmetrica e molto
rumorosa. Sulle stesse identiche righe stimare separatamente `P(net>0)`, rendimento lordo
condizionato al netto positivo e rendimento lordo condizionato al netto non positivo; comporre e
calibrare `EV gross = P×gain + (1−P)×miss`. Il costo del profilo è una feature nota e resta
sottratto una sola volta.

**Confronto.** Ridge/logistic baseline; XGBoost GPU challenger. Calibrazione isotonic separata per
probabilità ed EV, ranking continuo non calibrato per evitare pareggi. Stessi split e stessi gate;
nessun nuovo target, soglia micro o evento.

**Protocol hash.** `0ea3449f8522bcdd00d6f395c9bafc125d7a47ea6aceff920ac439f9e0e2e631`;
timing matrix invariata, hash
`c41d2cab02e8f9139adac85799d1c03e640c9214f62bd750942aa5a667d94bb1`.

**Risultato.** 24.312 azioni, copertura micro 100% e zero look-ahead. Nessuna frontiera valida.
Nel fold 1 Ridge batte XGBoost sia in MAE (17,318 contro 17,643) sia in Brier sul netto positivo
(0,2455 contro 0,2560). Nel fold 2 XGBoost migliora soltanto la MAE (16,244 contro 16,600), ma
peggiora Brier (0,2483 contro 0,2434); Ridge resta quindi champion. Il miglior punto VIP5 ha EV
−7,31 bps nel primo fold e −5,88 bps nel secondo, con LCB rispettivamente −9,68 e −9,78 bps.

**Causa economica.** Sul piano VIP5 più favorevole, il 55,0% dei trade è netto positivo, ma il
vincitore medio vale soltanto +4,72 bps e la perdita media −23,73 bps. Il pareggio richiederebbe
un hit rate dell'83,4%. Il modello non può correggere un payoff che realizza troppo poco quando ha
ragione. Aggiungere AFT o quantile regression prima di correggere questa asimmetria modellerebbe
meglio il tempo o la distribuzione di un'azione ancora negativa, senza crearne l'edge.

**Decisione.** FT-008 respinto; nessuna modifica al paper e nessuna apertura dell'holdout.

### 2026-08-09 — FT-009: frontiera room/risk con gestione V8 congelata

**Ipotesi preregistrata.** La V8 positiva combina gestione a 1,5R con un filtro `room_bps` fisso
di 24 bps, indipendente dal profilo fee, dallo stop e dalla volatilità. FT-009 non modifica stop,
uscite o direzione: etichetta l'intero pool già generato con la gestione V8 esatta e verifica se lo
spazio minimo pari a `1,5×`, `2×` o `3×` il costo reale del profilo conserva l'edge aumentando la
frequenza. `3×` è il controllo V8; `1,5×` e `2×` rispondono alla sola incoerenza economica del
filtro fisso, non sono quote di trade.

**Dati e split.** Binance BTC perpetual/spot già congelati. Entry al primo minuto eseguibile,
stop strutturale, 50% a 1,5R, protezione del costo, trailing 15 minuti e timeout 6 ore identici a
V8; stop peggiore nella stessa candela. Esperti preregistrati: cinque orizzonti e le due famiglie
già esistenti `IMPULSE_PULLBACK`/`IMPULSE_REENTRY`. Un esperto è ammissibile soltanto se positivo
con PF almeno 1,05 sia nel 2024 sia nel 2025. La policy sceglie sul 2024–2025 la massima frequenza
che conserva PF aggregato 1,15 e drawdown massimo 8%; il 2026 pre-holdout è solo audit discovery.
Holdout finale chiuso, costi 1× operativi, 2× diagnostico.

**FT-009A invalidato prima della decisione.** Il primo materializzatore ha riutilizzato il pool
FT-002 costruito con `minimum_room_bps=None`. Il confronto di controllo ha dimostrato che non era
un superset stateful della V8: H3 riproduceva soltanto 46 dei 129 eventi congelati e perdeva 83
restart. La causa è che accettare un restart precoce senza filtro resetta lo stato, mentre la V8 lo
rifiuta e può accettarne uno successivo. Il report hash
`5359ee2cd6c9f35ba7bfe1b37e3c7ff79cd8b7f8f8e62511c9d802e2013ebdea` è quindi diagnostica
invalida e non fornisce evidenza economica. Non ha modificato il paper.

**Protocollo corretto FT-009B.** Ogni combinazione `soglia room × orizzonte` viene rigenerata come
traiettoria stateful indipendente con il generatore originale. Prima di calcolare la frontiera, il
controllo 24 bps deve uguagliare per tutti i cinque orizzonti conteggio ed expectancy 2024/2025 del
report V8 congelato; in caso contrario il run fallisce chiuso. Le parti sono checkpoint atomici e
le feature di mercato vengono calcolate una sola volta per worker.

**Risultato FT-009B.** Protocol hash
`0ed5d8c17fb4362bfb752ddaefe565315c564eef511606efb3d9831aed77be0f`; 14.081 esiti. Il controllo
24 bps riproduce esattamente conteggi ed expectancy V8 per H3/H6/H12/H24/H48. L'hash del report
V8 resta `3FAF76E207BBCF12FF3501B1A2160273A67745F4D84BF54F3C52913A5339252B`.

VIP5 a `room=2×costo` aumenta il prior da 92 a 187 trade con EV +3,53 bps, ma PF 1,11; nel 2026
pre-holdout produce 46 trade, EV +2,65 e PF 1,08. VIP4 a `2×` produce 163 trade, EV +3,80 e PF
1,12 nel prior; nel 2026 produce 40 trade, EV +8,79, PF 1,29 e LCB +0,59 bps. È la prima
frontiera corretta che aumenta sostanzialmente la frequenza senza rendere negativa l'expectancy,
ma fallisce ancora PF 1,15 sul prior e la numerosità 50 nel 2026. Nessun collegamento al paper.

**Incompletezza del selettore.** Il protocollo richiede la massima frequenza sotto i gate, ma
FT-009B aggregava obbligatoriamente tutti gli esperti individualmente idonei. FT-009C valuta, sul
solo 2024–2025, i prefissi annidati degli esperti ordinati per robustezza minima 2024/2025 e sceglie
il prefisso con più trade che supera i gate. Non valuta subset arbitrari: al massimo dieci policy
annidate per soglia, con conteggio esplicito. Il 2026 viene letto soltanto dopo la scelta e non può
modificare il prefisso.

**Risultato FT-009C e preregistro FT-010.** I prefissi robusti rendono il prior più solido, ma il
calcolo del drawdown usa ancora `net_return_R × 1%` per ogni trade. Questo presume notional oltre
il 100% del patrimonio quando lo stop è inferiore all'1%, in conflitto con le regole paper già
fissate: rischio massimo 1%, margine massimo 10% e leva massima 10×, quindi notional massimo 100%.
Per il prefisso H24 VIP3 a room 2×, la metrica vecchia riporta drawdown 8,55%; applicando la formula
deterministica già prevista dal Risk Engine — `notional=min(100%, 1%/stop_fraction)` — il drawdown
composto è 4,61%. FT-010 corregge soltanto sizing e curva equity nei gate; rendimento in bps, PF,
eventi, stop e uscite restano identici. La scelta continua a usare soltanto 2024–2025 e l'audit
2026 viene calcolato una volta sola dopo la scelta.

**Risultato FT-010.** Protocol hash
`fa4828dc002cabdfdc7f667fc09a3d29f863810a7a19f488b90eaa93221769af`. Con sizing coerente,
VIP4 `room=2×costo`, esperto H24, produce 152 trade nel 2024–2025, EV +7,01 bps, PF 1,229,
bootstrap LCB +0,119 bps e drawdown 4,87%. Senza modifiche, nel 2026 pre-holdout produce 33 trade,
EV +6,99, PF 1,215, drawdown 4,13% e 57,7% di giornate attive positive; fallisce soltanto count 50
e LCB (+ campione insufficiente). VIP3 e VIP5 conservano economia positiva ma audit più debole.
Nessun profilo viene promosso; il risultato è il primo candidato di frequenza economicamente
coerente da raccogliere in paper dopo il completamento del protocollo.

**Diagnosi occupazione.** Per H24/VIP4 la posizione aperta elimina soltanto 4 dei 156 segnali raw
nel 2024–2025 e 2 dei 35 nel 2026. Durata mediana 62/73 minuti nei due anni prior e 44 minuti nel
2026. Accorciare il timeout non può trasformare la frequenza e non viene testato.

### 2026-08-09 — FT-011: esperti impulso micro H1/H2

**Ipotesi preregistrata.** Il collo di bottiglia residuo precede il trade: l'impulso più corto V8
è H3, cioè 15 minuti. Aggiungere soltanto H1 e H2 (breakout 5 e 10 minuti) può rilevare più cicli
impulso–pullback indipendenti mantenendo invariati trend 1h/4h, conferma spot/perpetual, volume,
taker flow, zone VWAP, restart, stop strutturale e gestione V8.

**Protocollo.** Gli outcome stateful restano separati per soglia room e orizzonte. Il controllo
H3/H6/H12/H24/H48 deve continuare a riprodurre V8. H1/H2 sono selezionabili soltanto con EV positivo
e PF ≥1,05 sia nel 2024 sia nel 2025; entrano negli stessi prefissi annidati robusti. La scelta usa
solo il prior; il 2026 è letto una volta dopo la scelta. Nessun nuovo target, modello o parametro.

**Protocol hash.** `32d211e4772156837a852faffcc1fe2cb801565f7b36d7a0aba7772a6aa77f21`;
outcome hash `433b30f4e9215caed0bda8681fff9e7a0a4ac8e934bde2d4faaabd3d6dda87aa`.

**Risultato.** H1/H2 producono alcuni esperti individualmente idonei, ma nessuno entra nel prefisso
robusto di massima frequenza: aggiungerli fa perdere PF o LCB. Il controllo congelato continua a
riprodurre esattamente H3/H6/H12/H24/H48 e l'hash V8 resta
`3FAF76E207BBCF12FF3501B1A2160273A67745F4D84BF54F3C52913A5339252B`.

Le policy selezionate restano quelle FT-010. VIP4 a room `2×` e H24 ha 152 trade nel
2024–2025, EV +7,015 bps, PF 1,229, LCB +0,119 e drawdown 4,868%; nel 2026 pre-holdout ha
33 trade, EV +6,994, PF 1,215, drawdown 4,131% e LCB −6,09. VIP5 H48+H24 ha 165 trade nel
prior e 36 nell'audit, ma PF audit 1,145 e LCB −8,23. Nessuna policy supera count e LCB audit.

**Decisione.** FT-011 respinto; i breakout 5/10 minuti non aumentano stabilmente la frequenza e
non cambiano il paper. Verdetto `BASE_ONLY_NO_ROOM_FRONTIER_GAIN`.

### 2026-08-09 — FT-012: disponibilità reale del modello maker Bitunix

**Ipotesi preregistrata.** Prima di attribuire al bot commissioni maker inferiori, verificare che
esistano fill POST_ONLY, parziali, latenza, ruolo e adverse selection realmente osservati. Il book
pubblico è una feature; non è verità di fill. Nessun ordine reale viene inviato.

**Protocol hash.** `9030739949b98b23291aee3c77c2a9d71f69dc0656c9f2b0b591ea6f2fef4d2e`.
Report: `data/reports/musca_v5_maker_feasibility.json`.

**Dati.** Bitunix BTC L2: 480.802 righe materializzate in 6 giornate, 426.498 valide (88,71%);
la settima giornata è in raccolta raw. Ordini POST_ONLY privati completi: 0; fill maker: 0;
giornate execution indipendenti: 0. Nel processo non sono configurate credenziali Bitunix.
L'archivio Binance ufficiale locale `bookDepth` contiene snapshot a ±1–5%, non la coda al best;
il percorso USD-M `bookTicker` verificato per una giornata restituisce 404. Non può quindi
sostituire i fill reali.

**Gate.** 30 giornate indipendenti, 100 ordini POST_ONLY completi, almeno 20 fill e 20 non-fill,
ruolo maker verificato, latenza e adverse selection 1/5/30s osservate, timestamp causali. Passa
soltanto il controllo temporale; tutti i gate di campione falliscono.

**Decisione.** `COLLECTING_NO_OBSERVED_MAKER_MODEL`. Costi maker vietati in Alpha e paper; costi
taker 1× restano operativi e 2× resta sola diagnostica. Il codice non simula fill mancanti.

**Prossimo passo.** FT-013 calcola soltanto un limite superiore a fill maker perfetto sugli outcome
stateful. Se neppure tale limite produce frequenza sostanziale, il maker non è la soluzione; se la
produce, si continua la raccolta ma nessuna policy viene promossa senza FT-012.

### 2026-08-09 — FT-013: limite superiore con fill maker perfetto

**Ipotesi preregistrata.** Verificare se l'esecuzione maker possa, almeno in teoria, trasformare la
frequenza. Il test attribuisce fill certo, adverse selection nulla e costo maker in ingresso più
taker in uscita, mantenendo invariati segnali, stop, gestione, sizing e split V8. È un limite
superiore impossibile, non una simulazione di fill e non ha autorità operativa.

**Protocol hash.** `517f73fb94843c29c49724b8dafbc3e6e1e8d28124c69f7fab0f77fd6204c8ae`.
Report: `data/reports/musca_v5_maker_perfect_fill_upper_bound.json`.

**Risultato.** Anche regalando fill maker perfetto, la massima frequenza robusta nel prior è 177
trade in due anni, circa 7,4 al mese. Il massimo audit 2026 pre-holdout è 42 trade. VIP4 è l'unico
punto con audit economicamente pieno — 36 trade, EV +9,50 bps, PF 1,294 e LCB +0,77 bps — ma
fallisce comunque il minimo di 50 trade. VIP5 arriva a 42 trade, ma LCB resta −3,62 bps.

**Decisione.** `PERFECT_FILL_NO_FREQUENCY_SOLUTION`. I costi e il tipo di fill non spiegano la
bassa frequenza; nessuna modifica al paper, nessuna apertura dell'holdout.

### 2026-08-09 — FT-014: VWAP del ciclo di funding Binance (preregistrato)

**Ipotesi preregistrata.** La V8 consuma pochi cicli indipendenti perché usa VWAP UTC giornaliero,
impulso e swing. Il perpetual Binance ha timestamp di funding osservati ufficialmente; ogni evento
può definire un nuovo benchmark intraday economicamente motivato senza inventare sessioni. La
direzione resta quella V8 (trend 1h/4h, EMA e conferma spot); il nuovo VWAP serve soltanto come zona
di pullback.

**Regola fissa.** Dopo un'estensione concorde oltre la banda del VWAP del ciclo di funding, con
volume relativo almeno 1 e flusso perpetual/spot concorde, si attende un pullback di almeno 0,25 ATR
verso la banda e una ripartenza entro sei barre da 5 minuti. Lo stop resta oltre pullback e banda;
gestione e label sono V8 congelate. Sono consentiti nuovi cicli estensione–pullback nello stesso
intervallo di funding, ma una sola posizione alla volta nel replay.

**Selezione.** Per ciascun VIP si valutano soltanto room pari a 1,5×, 2× e 3× il costo reale 1×.
La scelta usa esclusivamente 2024–2025 e richiede positività e PF almeno 1,05 in entrambi gli anni,
PF aggregato 1,15, LCB positivo e drawdown massimo 8%. Il 2026 pre-holdout viene letto una volta
dopo la scelta. Il 2× dei costi è registrato soltanto come stress diagnostico e non decide ingressi
o promozione. Holdout chiuso, paper invariato.

**Protocol hash e dati.** `06fc88b3de9770821642e5c34b58325adea95169897bae945c609ab009cd886f`;
2.267 settlement BTCUSDT osservati dall'endpoint Binance ufficiale, dal 16 aprile 2024 all'11 maggio
2026. Report: `data/reports/musca_v5_funding_vwap_frontier.json`.

**Risultato.** La famiglia non è bloccata: produce 1.171 segnali unici e 9.864 outcome stateful.
Per VIP5, room 1,5× genera 365 trade nel 2024 e 406 nel 2025, ma EV rispettivamente −8,44 e
−7,42 bps, PF 0,807 e 0,805. Tutte le 18 policy `VIP × room` falliscono il gate della famiglia in
entrambi gli anni. Le policy V8 base selezionate restano invariate; il 2026 letto per esse coincide
con FT-011.

**Decisione.** `NO_FUNDING_VWAP_FREQUENCY_GAIN`. Un anchor aggiuntivo aumenta la frequenza grezza,
ma non corregge il payoff; nessuna modifica al paper.

### 2026-08-09 — FT-015: ingresso sul touch del pullback (preregistrato)

**Motivazione.** L'utente autorizza una deviazione controllata dalla grammatica V8, mantenendo
fermo l'obiettivo di più micro-operazioni giornaliere. FT-002/003 ha già dimostrato che, dopo la
ripartenza, target 1,5×/2× costo con stop medio 21 bps ha payoff insufficiente. FT-015 sposta
l'ingresso alla prima zona di pullback dell'impulso, prima della ripartenza: prezzo migliore e
target economico naturale pari al ritorno verso l'estremo dell'impulso.

**Regola fissa.** Direzione, breakout H3/H6/H12/H24/H48, volume relativo, taker flow e conferma
spot restano V8. Dopo un impulso si entra al primo minuto eseguibile successivo al touch di daily,
impulse o swing VWAP con profondità almeno 0,25 ATR e volume più quieto dell'impulso. Non è richiesta
la conferma di restart. Stop causale oltre estremo del pullback e banda VWAP, mai allargato; target
completo all'estremo dell'impulso; timeout 60 minuti; stop vince nella stessa candela. Un nuovo
touch può essere valutato soltanto dopo la conclusione della traiettoria precedente.

**Frontiera.** Room minima 1,5× o 2× il costo reale 1× del VIP; nessun 3× obbligatorio. Selezione
solo 2024–2025 con gli stessi gate FT-010, poi singolo audit 2026 pre-holdout. Costi 2× soltanto
diagnostici. V8 resta controllo e fallback; nessuna promozione automatica.

**Protocol hash e risultato.** `bc9929c1f9f9936f28b4451f5f3aa879ba84e823883d098d8330ba4a529f7973`;
326.037 outcome, 12.304 timestamp di segnale distinti. L'ipotesi risolve la frequenza ma non
l'economia: nessun esperto è positivo con PF almeno 1,05 sia nel 2024 sia nel 2025. Il caso meno
negativo, VIP5 H48 con room `1,5×`, produce 1.950/2.501 trade con EV −8,16/−7,08 bps e PF
0,54/0,54. H24 con room `2×`, usato per la diagnosi del payoff, produce 4.324 trade nel prior:
lordo medio +0,57 bps ma netto −7,43 bps. I vincenti medi valgono +19,63 bps netti, i perdenti
−36,07; 1.240 stop perdono in media −42,72 bps e 1.036 timeout −20,04 bps.

**Decisione.** `NO_TOUCH_ENTRY_FREQUENCY_GAIN`. Il semplice touch ha lordo circa nullo e non può
essere trasformato in Alpha dalle fee. La conferma di ripartenza resta necessaria; la frequenza va
aumentata creando più episodi direzionali locali, non anticipando indiscriminatamente l'ingresso.
V8 paper invariata, holdout chiuso.

### 2026-08-09 — FT-016: direzione locale e restart VWAP a un minuto (preregistrato)

**Ipotesi preregistrata.** V8 dimostra che il restart dopo il pullback contiene edge, mentre FT-015
dimostra che il touch da solo non lo contiene. Il collo di bottiglia è la direzione rigida 1h/4h:
FT-016 sostituisce soltanto quel contesto con una direzione locale causale e genera più cicli
indipendenti `impulso → pullback VWAP → forza dopo il pullback`. La presentazione CMT di Brian
Shannon usa l'AVWAP come zona e richiede forza dopo il dip/debolezza dopo il rimbalzo; non giustifica
l'ingresso sul solo touch. I campi OHLCV, numero trade e taker-buy derivano dalle kline Binance
ufficiali e sono disponibili alla chiusura della barra.

**Regola fissa.** BTC Binance perpetual e spot, barre chiuse a un minuto. Direzione per maggioranza
di rendimento 15/60 minuti, slope VWAP 60 minuti, rendimento spot 15 minuti e imbalance taker
perpetual/spot a cinque minuti. Impulsi preregistrati: breakout 5, 15 o 30 minuti con volume relativo
almeno uno e flow perpetual/spot concorde. Pullback verso rolling VWAP 15/60 minuti, daily VWAP o
impulse AVWAP; ingresso soltanto dopo rottura del micro-swing precedente e flow concorde. Stop oltre
pullback/anchor, mai allargato; target all'estremo congelato dell'impulso; invalidazione alla chiusura
oltre il VWAP nella direzione opposta; timeout massimo 60 minuti. Stop vince se stop e target sono
toccati nella stessa barra.

**Economia e split.** Sono richiesti room almeno `2×` il costo reale 1× e reward/risk lordo almeno
1,5; non esiste una quota di trade. Si sceglie la massima frequenza che conserva positività e PF
almeno 1,05 separatamente nel 2024 e 2025, poi i gate FT-010 aggregati. Il 2026 pre-holdout è audit
discovery già riutilizzato e non conferma indipendente; l'holdout finale resta chiuso. Il meta-modello
GPU è vietato finché la base deterministica non ha EV positivo e stabilità nei due anni prior.

**Protocol hash e risultato.** `8948d31abe6d4332859d44b9815f5f86561901bfab55ed53ca8b0d88e9025edc`;
62.064 outcome e 25.630 timestamp distinti. Nessuno dei tre esperti è positivo in entrambi gli
anni. VIP5 M5 è il meno negativo: 201 trade nel 2024, EV −10,81 bps e PF 0,26; 274 nel 2025,
EV −8,98 e PF 0,34. Nel prior aggregato il lordo è −1,75 bps e il netto −9,75 bps.

**Diagnosi delle uscite.** Su 475 trade VIP5 M5, 281 (59,2%) escono per invalidazione VWAP dopo
5,44 minuti medi e perdono −15,79 bps netti; i 86 target rendono +23,39, i 105 stop perdono
−21,04. L'invalidazione appena dopo il restart taglia il normale retest e domina la distribuzione.

**Decisione.** `NO_LOCAL_RESTART_FREQUENCY_GAIN`; nessuna modifica al paper, nessun ML e holdout
chiuso.

### 2026-08-09 — FT-017: rimozione dell'invalidazione VWAP immediata (preregistrato)

**Ipotesi preregistrata.** Si riusano esattamente i 62.064 candidati FT-016. È rimossa soltanto
l'uscita al primo close oltre l'operating VWAP: dopo un restart il VWAP resta benchmark/anchor, ma
un singolo recross non invalida il trade. Entry al minuto successivo, stop strutturale, target
all'estremo dell'impulso, room `2×` costo, reward/risk 1,5, stop-wins e timeout massimo 60 minuti
restano invariati. Nessun nuovo segnale, soglia o modello viene valutato.

**Split.** Stessa selezione 2024–2025 e audit discovery 2026 pre-holdout di FT-016; costi 1×
operativi e 2× diagnostici. Se la base resta negativa, target/stop non vengono ritoccati sulla stessa
matrice: il prossimo cambiamento dovrà riguardare l'informazione direzionale, non l'uscita.

**Protocol hash e risultato.** `63b0f1f0c0c2cecf4930ca6cf8ccc69b6436534471ccf9f27f9c48dff2089d87`.
L'uscita hold peggiora i risultati: VIP5 M15 produce 206 trade nel 2024 con EV −11,37 bps e PF
0,35; 263 nel 2025 con EV −9,97 e PF 0,38. FT-017 è respinto e l'invalidazione FT-016 resta la
label meno negativa, senza autorità paper.

### 2026-08-09 — FT-018: meta-modello probabilità + EV sui restart locali (preregistrato)

**Oracle operativo.** Sui candidati FT-016 non filtrati, scegliendo a posteriori la migliore fra
M5/M15/M30, VIP5 contiene 995 timestamp netti positivi nel 2024 e 1.101 nel 2025: 3–4 esiti
positivi al giorno, ma soltanto 9–10% dei segnali. Il filtro deterministico room `2×` + RR 1,5
riduce l'oracle a 42/56. La frequenza esiste nel generatore; manca discriminazione causale.

**Modello preregistrato.** Dataset FT-016 con invalidazione, senza filtro room/RR. Due teste pooled:
`P(target prima delle altre uscite)` e `EV lordo = P×E(gain|target)+(1−P)×E(gross|miss)`.
Feature soltanto disponibili al segnale: lato, orizzonte, trend score, rendimenti 15/60 minuti,
conferma spot, slope VWAP, volume relativo, taker flow 1/5 minuti, profondità pullback, room e rischio
calcolati sul close del segnale, distanza direzionale dal VWAP e ora ciclica. Nessun prezzo del minuto
successivo entra nelle feature.

**Split e confronto.** Fit fino al 31 agosto 2024; calibrazione settembre–ottobre; selezione soglia
novembre–dicembre. Il 2025 è validazione intoccata durante scelta di modello/soglia; il 2026
pre-holdout è audit discovery soltanto se la validazione passa. Ridge è champion predefinito;
XGBoost `hist`/CUDA può sostituirlo solo se su selection migliora sia MAE EV sia Brier probability e
produce una policy economicamente valida. Coperture preregistrate 10%, 5%, 2,5%, 1%; FLAT=0,
una posizione alla volta. Costi 1× decidono, 2× è diagnostico. Holdout chiuso e nessuna promozione
automatica.

**Protocol hash e risultato.** `7fffba5785b70357373d0801ca1f948a6f88ec47e34ee4a27ad2ee88136a7b89`.
Fit 12.137 righe, calibrazione 5.198, selection 5.565. Ridge è migliore del challenger XGBoost
sia per MAE lordo (7,0999 contro 7,2281 bps) sia per Brier target (0,21345 contro 0,21350), quindi
XGBoost non può diventare champion. Nessuna frontiera passa neppure su selection: VIP5 Ridge
seleziona 57 trade con EV −9,91 bps e PF 0,286; XGBoost 28 trade con EV −12,70 e PF 0,328.
Il 2025 e il 2026 non vengono usati per correggere il modello.

**Decisione.** `NO_LOCAL_RESTART_MODEL`. L'oracle dimostra esiti positivi, ma le feature FT-018
non li discriminano causalmente. Nessun bundle e nessuna modifica paper.

### 2026-08-09 — FT-019: qualità micro del restart (preregistrato)

**Ipotesi preregistrata.** Candidati, entry, stop, target, invalidazione e outcome FT-016 restano
identici. Si aggiungono soltanto feature causali già contenute nelle kline Binance ufficiali:
ATR in bps e rapporto con la mediana precedente, rendimento/accelerazione 1–5 minuti, posizione
del close e wick nella barra di restart, ampiezza della rottura del micro-swing, z-score precedente
di trade count e quote volume, variazione taker 1m–5m, basis mark-perpetual e variazione a cinque
minuti, divergenza spot/perpetual 1–5 minuti, durata impulso→restart, rapporto volume e variazione
taker fra impulso e restart. Nessuna imputazione con zero; coverage incompleta fallisce chiusa.

**Modello e split.** Stesse due teste, Ridge/XGBoost, split, coverages e regole champion FT-018.
La matrice outcome non viene rietichettata. Se selection resta negativa, non si apre il 2025 e non
si aumenta capacità/tuning del modello: la famiglia restart locale è respinta.

**Protocol hash e risultato.** `8525b5251fdf5e0bf569558de70c87a973eb58f153020c9895871af918076fbc`.
Ridge conserva MAE migliore (7,0910 contro 7,1269 bps), mentre XGBoost migliora soltanto Brier
(0,21181 contro 0,21545). VIP5 Ridge resta negativo: 20 trade, EV −6,37 bps e PF 0,52.
XGBoost individua 7 trade con EV +23,22 e PF 5,84, ma fallisce numerosità e il requisito champion
perché non batte Ridge sul MAE. Il 2025 non viene aperto. Nessun bundle.

### 2026-08-09 — FT-020: Q delle azioni target × orizzonte (preregistrato)

**Ipotesi preregistrata.** Il target unico all'estremo congelato dell'impulso mescola movimenti e
tempi differenti. Per ogni candidato FT-016 si costruiscono dodici azioni fisse: target lordo VIP5
`1,5×`, `2×` o `3×` il costo reale (12/16/24 bps) e timeout 5/15/30/60 minuti. Entry, stop
strutturale, invalidazione VWAP, funding, stop-wins e percorso sottostante 1m restano identici.
Ogni target viene simulato una volta e i quattro orizzonti sono snapshot cronologici dello stesso
percorso; nessun esito viene ricostruito da MFE/MAE.

**Modello.** Le feature FT-019 più target, multiplo costo e orizzonte alimentano le stesse teste
probabilità/EV. A ogni timestamp la policy sceglie una sola azione con EV netto previsto maggiore di
zero oppure FLAT. Stessi split, champion Ridge, challenger XGBoost e coperture FT-018. La selection
resta novembre–dicembre 2024; niente lettura del 2025 se fallisce. I livelli VIP inferiori sono
stress economici dello stesso set di azioni; VIP5 è il caso principale preregistrato.

**Protocol hash e risultato.** `d0bf3c21cdb5044857f44411c7e519cd91b7269eeeadba508f07804a3f661cf7`;
744.768 righe, 25.630 segnali e dodici azioni per segnale. L'oracle VIP5 contiene almeno un'azione
netta positiva nel 36,17% dei segnali 2024 e nel 34,80% del 2025, con EV oracle FLAT-inclusive
+4,50/+4,20 bps. Questo vantaggio usa il futuro e non è tradabile. Ridge mantiene MAE migliore
(10,525 contro 10,558 bps); XGBoost migliora soltanto marginalmente il Brier (0,18351 contro
0,18390). Nessun profilo produce un champion sulla selection 2024; il 2025 non viene aperto.

**Decisione.** `NO_LOCAL_ACTION_MODEL`. Orizzonte e target multipli aumentano l'oracle, ma le
feature di barra non discriminano causalmente le azioni. Nessun bundle e nessuna modifica paper.

### 2026-08-09 — FT-021: diagnosi diretta della classe di strategia

**Domanda.** Le micro-oscillazioni possono essere monetizzate frequentemente con una direzione
derivata da VWAP, momentum e taker flow, oppure con un market maker al best bid/ask?

**Protocollo.** Una sola analisi, non tuning: 2.073.600 intervalli Binance futures a cinque secondi
dal 1 gennaio al 30 aprile 2026, 168.959 decisioni valide a un minuto, ingresso non prima del
successivo intervallo a cinque secondi, target 12 bps, stop 6 bps, timeout 15 minuti e stop-wins.
Una sola posizione alla volta. Gennaio-febbraio definiscono gli otto stati
`trend 5m × flow 1m × lato del VWAP`; marzo e aprile sono validation. Maggio-luglio non vengono
letti. Costi Bitunix 1×: VIP5 maker-taker 4,5078 bps usando lo spread mediano osservato. Vengono
inoltre misurati, senza selezione a posteriori, fade verso il rolling VWAP congelato alle distanze
2/4/6/8/12/16/24/32 bps e fattibilità del market making al best quote.

**Protocol hash e report.** `392a9f699818a0465840c14cc8a25ebe7d5412cc2c42ec49da766a04e133f89e`;
`data/reports/musca_v5_strategy_class_audit.json`.

**Risultati.** L'oracle sceglie a posteriori un'azione netta positiva nel 66,37% dei minuti e vale
+4,68 bps FLAT-inclusive: i movimenti esistono. Non sono però prevedibili con gli stati disponibili.
Momentum 1m/5m, flow 1m, VWAP fade e VWAP follow hanno tutti EV VIP5 fra circa -3,97 e -4,50 bps
in ciascun mese. Tutti gli otto stati Jan-Feb hanno EV LONG e SHORT negativo, quindi la policy
corretta non apre artificialmente marzo-aprile. Il fade event-driven verso il rolling VWAP resta
negativo a ogni distanza; sul totale l'EV varia da -4,32 a -5,15 bps e nessuna soglia è stabile.

Lo spread Bitunix mediano è 0,0155 bps. Anche VIP5 paga 2 bps maker round-trip: quotare best
bid/best ask parte con un deficit di 1,9845 bps prima dell'adverse selection; VIP0 parte con un
deficit di 3,9845 bps. Il replay maker semplice già disponibile è coerentemente negativo.

**Decisione.** `NO_FREQUENT_POLICY_IN_AVAILABLE_FEATURES`. Non significa che BTC non oscilli:
significa che candele, VWAP e flow aggregato non contengono informazione sufficiente a scegliere
prima quale oscillazione pagherà i costi. Sono respinti sia il direzionale ogni minuto sia il
market maker ingenuo al best quote. La direzione architetturale è ora unica: V8 resta Alpha rara;
la componente frequente deve apprendere su L2 la probabilità di fill, l'adverse selection e il
ritorno da una quota POST_ONLY economicamente distante dal fair value/VWAP. Finora mancano fill
privati Bitunix e giornate L2 indipendenti sufficienti, quindi nessuna autorità paper o reale.

### 2026-08-09 — FT-022: fondazione full-depth Bitunix e replay POST_ONLY

**Causa strutturale.** Il collector storico usava `depth_book15`: a BTC circa 65.000 USDT i primi
15 livelli coprono soltanto pochi decimi di basis point. Non possono osservare coda e liquidità
alle distanze necessarie per superare le commissioni. Il canale `depth_books` reale invia inoltre
refresh quasi completi da circa 13.000 livelli bid e 15.000 ask: salvarli come delta integrali
produceva oltre 1 MB per evento e conservava livelli rimossi. Il file del 9 agosto era già vicino
a 2 GB. La causa è stata corretta, non aggirata: snapshot REST `limit=max`, stato full-depth in
memoria, confronto refresh→stato precedente, salvataggio delle sole differenze reali, checkpoint
causale ogni 15 minuti e nessuna compattazione sincrona durante la raccolta.

Fonti ufficiali: `GET /api/v1/futures/market/depth` con `limit=max` e WebSocket pubblico
`depth_books`/`depth_book15`:
https://www.bitunix.com/api-docs/futures/market/get_depth.html e
https://www.bitunix.com/api-docs/futures/websocket/public/depth%20channel.html.

**Protocollo preregistrato.** Il fair value è il VWAP dei trade pubblici Bitunix osservati negli
ultimi cinque minuti. Ogni profilo VIP0–VIP5 quota un bid e un ask POST_ONLY alla prima profondità
reale distante almeno `1,5 × (fee maker ingresso + fee taker uscita)` dal fair value. L'uscita
taker rende l'economia verificabile senza inventare un secondo fill maker. Ordine per un nozionale
massimo di 10.000 USDT, coerente con equity 10.000, margine 10% e leva 10×, arrotondato al lotto
0,001 BTC. Latenza d'invio 250 ms; lifetime ordine 30 secondi dall'attivazione; target al fair value
eseguibile; time-stop cinque minuti; stop catastrofico 100 bps; una sola posizione per profilo. La
coda iniziale è quella visibile e le cancellazioni non migliorano mai artificialmente la priorità.
Il funding Bitunix viene applicato, nella corretta unità frazionaria, soltanto se la posizione
attraversa l'istante osservato di settlement. Un trade pubblico è soltanto proxy conservativo di
fill, mai verità dell'account.

**Protocol hash.** `2c1566922a582bb5572bcfd67a765610052a572836b7f6bfa084165457b4c0a5`.

**Artefatti.** `src/adaptive_bot/musca_v5_post_only_replay.py`, tape append-only
`data/raw/bitunix_microstructure/btcusdt_post_only_tape_YYYY-MM-DD.jsonl` e report
`data/reports/musca_v5_post_only_replay.json`.

**Verifica live.** Collector connesso, nessun reconnect dopo l'ultimo avvio; snapshot osservato di
13.071 bid e 14.965 ask. Le distanze minime reali risultano 12 bps per VIP0 e 6,75 bps per VIP5,
con prezzo e coda espliciti. Il primo smoke replay copre soltanto circa un minuto (223 eventi,
due coppie di quote per profilo, nessun fill): prova il flusso, non l'economia.

**Aggiornamento dopo FT-029.** Il tape corretto copre 52 minuti, 10.822 eventi e 76 cicli di quota
per profilo; zero fill proxy anche VIP5. Il protocollo e le distanze non vengono cambiati: un'ora
senza touch non è un campione economico e non giustifica fill simulati.

**Gate.** Risultato per singolo profilo soltanto dopo almeno 10 giornate e 100 trade proxy chiusi;
training Execution ancora vietato senza 30 giornate e fill privati Bitunix. Musca V8/V5 paper,
holdout e denaro reale restano invariati.

**Decisione.** `COLLECTING_EXECUTION_AWARE_SHADOW`. Nessuna promozione; la raccolta ora contiene
per la prima volta le variabili necessarie a decidere causalmente se la componente frequente può
esistere.

**Correzione funding.** La sorgente Bitunix pubblica `fundingRate` in punti percentuali, mentre
Binance lo espone come frazione. Il live precedente conservava `-0.008083` come frazione anziché
`-0.00008083`, creando un errore potenziale 100× al settlement. `btc_context.py` ora normalizza
Bitunix dividendo per 100 e registra esplicitamente unità sorgente e unità canonica. Evidenza:
documentazione ufficiale Bitunix con esempio `0.0005` e limiti `±0.3`, più controllo di coerenza
con mark-index basis osservato. I file vecchi non vengono riscritti; il replay li normalizza in
lettura e i nuovi record sono marcati `fraction_of_notional`.

### 2026-08-09 — FT-023: limite superiore POST_ONLY con uscita taker

**Ipotesi preregistrata.** Prima di attendere nuovi fill privati, verificare se il semplice ritorno
da una quota economicamente distante al VWAP possa essere positivo perfino assumendo fill certo al
primo touch, nessuna coda e percorso L2 osservato.

**Dati.** 480.802 snapshot/eventi Bitunix L2, sei giornate dal 3 all'8 agosto; 426.498 righe con
feature causali valide. Decisione ogni 30 secondi, latenza 250 ms, quota valida 30 secondi, posizione
massima cinque minuti, un solo inventario per profilo.

**Protocol hash e report.** `febdd4a727fccb4968b6185b5f9f17e41e0c3144e8f7b856bc87507296b54812`;
`data/reports/musca_v5_post_only_touch_upper_bound.json`.

**Risultato.** La frequenza teorica esiste, da 30,17 trade/giorno VIP0 a 95,50 VIP5, ma tutti i
profili perdono. VIP5: 573 trade, EV −5,99 bps, PF 0,142, 35,78% positivi e nessuna giornata
positiva; stress costi −10,49 bps. I 187 target rendono circa +2,96 bps, i 386 timeout perdono in
media −10,32 bps. Poiché il test regala il fill, il replay reale non può migliorarlo senza nuova
informazione o diversa gestione dell'inventario.

**Decisione.** `FROZEN_VWAP_QUOTING_REJECTED`.

### 2026-08-09 — FT-024: centro VWAP dinamico

**Ipotesi preregistrata.** Verificare se aggiornare il centro durante la posizione riduca il costo
dei timeout senza selezione di parametri.

**Protocol hash e report.** `d0ea3fa2dfdc5ea7fd3b421c72b595837ff265d94478ce3780e0b97beccc84ed`;
`data/reports/musca_v5_post_only_dynamic_vwap_upper_bound.json`.

**Risultato.** VIP5 sale a 834 trade, 139/giorno, ma peggiora a EV −6,76 bps e PF 0,048; nessuna
giornata positiva. Il VWAP mobile non salva l'inventario: quando il centro attraversa l'ingresso,
la chiusura realizza una perdita anziché un ritorno economicamente sufficiente.

**Decisione.** `DYNAMIC_VWAP_QUOTING_REJECTED`.

### 2026-08-09 — FT-025: confluence L2 e target maker

**Ipotesi preregistrata.** Testare, senza ricerca di soglie, sia quattro segni L2 concordi
(depth, microprice, flow aggressivo e slope VWAP), sia il limite superiore con uscita maker al
target.

**Protocol hash e report.** Filtro `7dfd788e3c5c58bee85148e8007a0a68358aae41c395a3ecc8f9ab077f750463`,
`data/reports/musca_v5_post_only_l2_filter_audit.json`; maker target
`7c34eca4c0d75091b8af99791791b719807d878834a7de64cf2a4202f9b0fb3b`,
`data/reports/musca_v5_post_only_maker_exit_upper_bound.json`.

**Risultato.** Il filtro VIP5 resta negativo sia in discovery (132 trade, EV −7,45 bps, PF 0,099)
sia in audit (81 trade, EV −4,18 bps, PF 0,215). Anche regalando l'uscita maker, VIP5 realizza
1.391 trade (231,83/giorno) ma EV −5,50 bps, PF 0,079 e nessuna giornata positiva. I target sono il
47,30% e rendono circa +1 bps; i mancati target perdono in media −11,33 bps.

**Decisione.** `STATIC_L2_SIGNS_AND_MAKER_TARGET_REJECTED`. La frequenza non è il collo di
bottiglia: lo è l'adverse selection dopo il touch.

### 2026-08-09 — FT-026: modello causale di tossicità L2

**Ipotesi preregistrata.** Stabilire se le feature causali già osservate separino i touch che
rientrano al VWAP da quelli che lasciano inventario tossico. Nessun Optuna, nessuna soglia cercata:
Ridge alpha 10 campione predefinito, XGBoost fisso challenger, probabilità target con regressione
logistica; train 3–4 agosto, selezione 5 agosto, audit intoccato 6–8 agosto.

**Protocol hash e report.** `e1955fa4702f3fff4ae7d04d77c22d2d3d78750eb177dc47b4683286ef61d495`;
`data/reports/musca_v5_l2_toxicity_model.json`.

**Risultato.** 485/302/604 trade nei tre periodi. Il target deve essere raggiunto con probabilità
almeno 92,96% per compensare payoff +1 bps e perdita media molto maggiore. Ridge accetta un solo
evento in selezione (+1 bps) e zero in audit; XGBoost zero in entrambi ed errore MAE superiore a
Ridge. Brier circa 0,249 in selezione e audit: nessuna discriminazione economica utilizzabile.

**Decisione.** `NO_CAUSAL_L2_TOXICITY_MODEL`; nessuna modifica al paper. Il fallimento non chiede
più GPU: richiede label private di coda/fill oppure una struttura di payoff diversa, valutata prima
come limite superiore. L'analisi dei dataset e la scelta di tale struttura restano responsabilità
diretta di Codex; il PC non verrà usato per una nuova ricerca combinatoria.

### 2026-08-09 — FT-027: curva temporale dell'inventario

**Ipotesi preregistrata.** Verificare se la perdita della quota maker nasce dall'attesa fissa di
cinque minuti. Unico caso analizzato: VIP5, già il più favorevole per commissioni; time-stop
30/60/120/180/300 secondi dichiarati prima del calcolo. Ogni punto è descrittivo e non può essere
selezionato per il paper.

**Protocol hash e report.** `4eecf952746275c5169b518c1fced233289f8d8c3706a7e40bdb16fb676c8a4f`;
`data/reports/musca_v5_post_only_inventory_hazard.json`.

**Risultato.** EV/PF: 30 s −6,64 bps/0,011; 60 s −6,29/0,026; 120 s −5,82/0,048;
180 s −5,67/0,062; 300 s −5,50/0,079. Le frequenze teoriche sono 673–232 trade/giorno, ma
nessuna giornata è positiva e nessun gate passa. Accorciare aumenta il riciclo delle quote ma
riduce drasticamente la probabilità del target (7,4% a 30 s contro 47,3% a 300 s) senza eliminare
il costo dell'uscita taker.

**Decisione.** `NO_INVENTORY_HORIZON_UPPER_BOUND`. Il problema è già presente subito dopo il
touch; non deriva da un time-stop troppo lento.

### 2026-08-09 — FT-028: lead–lag pre-fill Binance→Bitunix (preregistrato)

**Ipotesi preregistrata.** FT-026 usava soltanto lo stato Bitunix al momento di emissione della
quota. Durante i successivi 0,25–30 secondi Binance può anticipare un movimento avverso. Verificare
se lo stato più recente realmente disponibile almeno 250 ms prima del touch discrimina le quote da
cancellare. Questo è un limite superiore: il live non conosce in anticipo l'istante del touch; un
eventuale segnale dovrà poi superare un replay continuo che può soltanto peggiorarlo.

**Dati e split congelati.** Maker-target VIP5 FT-025; join `backward` sui timestamp `available_at`,
mai exchange timestamp futuro. Binance massimo cinque secondi di età, Bitunix massimo due secondi;
feature mancanti fail-closed. Train 6 agosto, selezione 7 agosto, audit 8 agosto. Ridge alpha 10
campione predefinito, XGBoost fisso challenger, regressione logistica della probabilità target;
nessun Optuna o ricerca di soglie.

**Protocol hash.** `7a5f879e6b1562ffd9931a3031bc9ba40770e816824d1cda6c2fec9d4706949f`.

**Decisione prima del calcolo.** Se selezione e audit non hanno almeno 20 trade, EV positivo e PF
1,15, la lead–lag viene respinta. Anche in caso positivo non modifica il paper: autorizza soltanto
il successivo replay continuo causale.

**Risultato.** 1.391 touch sorgente; 494 completi dopo join fail-closed, ripartiti 186/258/50 tra
train, selezione e audit; zero violazioni causali. La probabilità target di pareggio nel train è
90,83%. Ridge seleziona 14 casi il 7 agosto con EV −5,90 bps e PF 0,088, poi zero l'8 agosto.
XGBoost seleziona due casi con EV −12,86 bps e zero nell'audit; non batte Ridge secondo il criterio
champion. Report `data/reports/musca_v5_prefill_lead_lag.json`.

**Decisione.** `NO_PREFILL_LEAD_LAG_SIGNAL`. Nemmeno l'upper bound che conosce a posteriori il
momento del touch trova informazione economica; un replay live continuo non può essere autorizzato.

### 2026-08-09 — FT-029: validità strumentale Binance L2 (preregistrato)

**Domanda.** Le nuove feature L2 contengono almeno informazione predittiva sul proprio mercato? Se
non prevedono il future Binance a breve, non possono essere usate per spiegare Bitunix.

**Protocollo.** Campionamento causale non sovrapposto; orizzonti fissi 5/30/60 secondi. Train 6
agosto, selezione 7 agosto, audit 8 agosto con purge tramite `label_available_at`. Feature complete
fail-closed. Ridge alpha 10 campione; XGBoost fisso challenger. Nessuna soglia cercata: il trade
upper-bound viene ammesso soltanto quando il valore assoluto previsto supera i 3 bps VIP5
maker→taker; costo 1× sottratto al rendimento firmato. Ogni orizzonte deve battere previsione zero
in MAE e produrre in selezione e audit almeno 50 trade, EV netto positivo e PF 1,15. Nessuna
autorità paper anche in caso positivo.

**Protocol hash.** `25c833a63081195f865ed2b80e8a912f8abd8d6eb67ef4844762cdccd4aeb19b`.

**Risultato.** 236.478 righe complete. Split: 2.012/2.880/2.880 righe a 5 s,
2.010/2.879/2.878 a 30 s e 1.004/1.439/1.438 a 60 s. Ridge e XGBoost hanno MAE peggiore
della previsione zero in selezione e audit a ogni orizzonte. A 5 s nessun modello produce trade
OOS economicamente numerosi; a 30 s XGBoost trova quattro casi positivi in selezione ma zero in
audit e non è champion; a 60 s Ridge passa da due casi in selezione a 88 trade audit con EV
−2,71 bps e PF 0,0068. Report `data/reports/musca_v5_binance_l2_signal_validity.json`.

**Decisione.** `NO_BINANCE_L2_INSTRUMENT_SIGNAL`. Le quattro giornate L2 disponibili non
giustificano un timing direzionale né un altro modello; il prossimo dato informativo deve arrivare
dal tape Bitunix full-depth corretto e da giornate indipendenti aggiuntive.

### 2026-08-09 — FT-030: disponibilità delle label private Bitunix

**Verifica ufficiale.** Bitunix espone `get_history_orders`, `get_history_trades` e il WebSocket
privato `order` sul solo dominio reale. La firma REST usa doppio SHA-256 su nonce, timestamp, API
key, parametri ordinati e secret. Non è documentato un dominio testnet/sandbox separato.

**Stato locale.** Nessuna delle variabili `BITUNIX_API_KEY`, `BITUNIX_SECRET_KEY` o
`BITUNIX_API_SECRET` è configurata. Nessun ordine reale è autorizzato. Non esistono quindi dati
privati da interrogare o un collector autenticato verificabile da avviare. Il builder fail-closed
`src/adaptive_bot/adapters/bitunix/execution_observer.py` resta pronto a trasformare soltanto
ordini/trade finali realmente osservati.

**Runtime pubblico.** Collector connesso, full-depth sincronizzato (13.076 bid, 14.965 ask), zero
reconnect e supervisor con tutti gli otto worker attivi. Il tape pubblico continua a crescere e il
protocollo resta `2c1566922a582bb5572bcfd67a765610052a572836b7f6bfa084165457b4c0a5`.

**Decisione.** Nessun altro training o modello è autorizzato sul campione attuale. Ripresa
automatica dell'analisi a 10 giornate/100 trade proxy; Execution privata soltanto quando esistono
credenziali read-only e ordini osservati, senza mai inviare ordini reali implicitamente.

### 2026-08-09 — FT-031: conversione Binance-only

**Decisione preregistrata.** Binance BTCUSDT diventa l'unica venue di Alpha, simulazione paper e
futuro live. Bitunix non partecipa più a feature, decisioni, book, costi o funding; i suoi file
restano intatti per un eventuale transfer test futuro.

**Correzione strutturale.** La policy positiva V8 era addestrata su Binance, mentre il runtime V5
richiedeva book e freschezza Bitunix. È stato introdotto un solo account paper Binance persistente,
riusando sizing e gestione V5: 10.000 USDT, rischio massimo 1%, margine 10%, leva massima 10x,
una posizione. Entry sul primo book Binance successivo al segnale; fee taker account-specific via
endpoint firmato quando le credenziali esistono, altrimenti fallback ufficiale esplicitamente
etichettato; mark/index e funding Binance osservati.

**Baseline economica cost-matched.** Con 4 bps taker per lato e riserva non-fee 1 bp, il costo
round-trip è 9 bps. Il report V8 già congelato contiene questo scenario: 92 trade 2024–2025,
EV +4,429 bps, PF 1,125; 24 trade 2026 pre-holdout, EV +13,499 bps, PF 1,398; stress costi 2x
2026 EV +4,499 bps. Holdout sigillato e denaro reale vietato.

**Primo forward Binance.** Alpha, execution L2 e funding risultano freschi; il primo stato è
`WAIT_NEW_IMPULSE_PULLBACK_RESTART`, non un blocco dati. Report:
`data/reports/musca_v8_binance_paper.json`. Stato:
`data/research/musca_v8_binance_paper_state.json`.

**Decisione.** `BINANCE_PAPER_RESEARCH_ONLY`. Nessun nuovo training costoso: si registra prima il
paper forward. Audit dopo almeno 10 giorni distinti e 100 trade; live reale resta disabilitato.

### 2026-08-09 — FT-032: obiettivo permanente di frequenza Binance

**Obiettivo.** Massimizzare il numero di micro-trade giornalieri che resta economicamente
sostenibile OOS su Binance. Sono ammessi trade e giornate negative; il requisito è vantaggio netto
aggregato e maggioranza robusta di giornate positive dopo fee, spread, slippage e funding. Nessun
numero di ingressi viene forzato e nessuna perdita viene nascosta tramite FLAT.

**Metodo.** V8 è il controllo positivo a bassa frequenza. Le analisi FT-001–FT-029 sono evidence
reuse: prima si ricostruisce da esse la frontiera frequenza–movimento–costo; poi si modifica un solo
meccanismo causale alla volta nella policy ufficiale Musca V5 Binance. Niente nuove linee V9/V10,
niente ricerca finché appare un profitto, niente live prima dei gate.

### 2026-08-09 - FT-033: ultimo audit Binance multi-orizzonte

**Ipotesi preregistrata.** Sostituire il generatore raro V8 e i 10.273 eventi VWAP prefiltrati con
decisioni causali ogni 30 secondi. Usare soltanto Binance BTCUSDT spot/perpetual e tutte le feature
locali disponibili. Lasciare al modello la scelta fra LONG/SHORT, sei orizzonti 3/5/10/15/30/60
minuti e FLAT implicito. Questo e' l'ultimo tentativo autorizzato: in caso di fallimento non si
modificano altre soglie o piani.

**Dati e split.** 521.070 righe, 72 feature complete senza imputazione, 2.879 decisioni/giorno.
Fit gennaio-febbraio; calibrazione 1-14 marzo; selezione modello 15-31 marzo; selezione policy
aprile; audit discovery riutilizzato maggio-giugno; luglio sigillato. Purge 60 minuti. Entry al
primo open 5s causalmente disponibile; stop prevale quando stop e target sono nella stessa barra;
funding osservato. Costi Binance normali 9 bps e stress diagnostico 18 bps.

**Protocol hash.** Vedere `protocol_hash` in
`data/reports/musca_v5_binance_policy.json`. Codice:
`src/adaptive_bot/musca_v5_binance_micro_ranker.py`.

**Controllo dei label.** L'oracle futuro, non negoziabile, ha EV netto medio +30,78 bps e almeno
un'azione positiva nel 92,76% degli stati. Lo spazio d'azione contiene quindi movimenti economici;
il limite e' la loro prevedibilita' causale.

**Risultato modello.** XGBoost batte Ridge ma fallisce il model-selection economico. In aprile le
coperture 0,75%, 1%, 1,5% e 2% producono rispettivamente 0,30, 0,47, 0,87 e 1,33 trade/giorno con
EV -0,175, -7,010, -5,038 e -4,687 bps. Nessun punto raggiunge contemporaneamente frequenza,
expectancy, PF, maggioranza di giornate positive e LCB.

**Diagnostica audit non selezionabile.** Applicando soltanto per diagnosi la copertura 0,1% a
maggio-giugno: 17 trade, 0,279/giorno, EV +14,366 bps, PF 1,409, win rate 47,1%, drawdown 2,98%,
stress costi 2x +5,366 bps. La numerosita' e la frequenza sono insufficienti e il punto non era
passato dalla selezione di aprile; non puo' diventare bundle.

**Decisione.** `NO_SUSTAINABLE_BINANCE_HIGH_FREQUENCY_ALPHA`. Nessun bundle, nessuna modifica al
paper V8 e live disabilitato. La ricerca si ferma qui come richiesto dall'utente; non esiste un
prossimo esperimento autorizzato.

### 2026-08-10 — FT-034: costo-opportunità lineare della durata

**Ipotesi preregistrata.** Penalizzare l'EV calibrata del MoE positivo in proporzione alle ore
durante le quali l'esperto occupa l'unica posizione, senza modificare esperti o label.

**Risultato.** In giugno la penalità da 1 bps/ora porta la frequenza da 2,23 a 2,33 trade/giorno,
ma riduce l'EV da +21,31 a +16,64 bps. Da 2 bps/ora la policy favorisce gli esperti da un'ora e
diventa negativa: EV −9,62 bps, PF 0,77 e 0,93 trade/giorno. Luglio resta quasi interamente senza
segnali. La calibrazione cronologica di maggio non produce alcuna azione positiva, quindi non
esiste neppure un parametro selezionabile senza leggere giugno.

**Decisione.** Respinta e vietata da ripetere. Un costo lineare non stima il valore delle
opportunità future; serve un target sequenziale a orizzonte finito. Protocollo successivo:
`docs/musca-btc-daily-portfolio-challenger.md`.

### 2026-08-10 — FT-035: advantage giornaliero semi-Markov

**Ipotesi preregistrata.** Apprendere il vantaggio di LONG/SHORT/FLAT rispetto al valore della
prossima decisione libera, includendo esplicitamente la durata che blocca la posizione. Nessuna
soglia verrà scelta sul 2026.

**Dati e split.** OOF Binance 2025 già esistente: Q2 fit, Q3 model selection, Q2–Q3 refit e Q4
calibrazione. Gennaio–luglio 2026 è audit discovery; dal 10 agosto l'holdout resta chiuso.

**Decisione.** In corso. Tutti i dettagli, gate e riferimenti sono congelati in
`docs/musca-btc-daily-portfolio-challenger.md`.

**Risultato FT-035.** `NO_INCREMENTAL_DAILY_Q_ALPHA`, hash
`995bcdb6fbbdef3c7d40d4e39a928f44d0388ac574915257e3860ac3e904e21d`. Il target era positivo
solo nello 0,125–0,127% delle azioni. L'errore è nel teacher: il valore di `FLAT` usava il massimo
futuro realizzato e disponeva quindi di hindsight che nessuna azione causale possedeva. Il modello
ha emesso correttamente zero trade. Target vietato da ripetere.

### 2026-08-10 — FT-036: Fitted Q Iteration con FLAT appreso

**Ipotesi preregistrata.** Sostituire l'oracle onnisciente con quattro aggiornamenti di Bellman. Il
valore futuro arriva esclusivamente dalla stima dell'iterazione precedente; `FLAT` è una vera
azione di attesa con transizione causale al minuto seguente.

**Dati e split.** Invariati rispetto a FT-035. Nessun risultato 2026 seleziona modello,
calibrazione o soglia. Ridge champion, XGBoost GPU challenger; gestione finale identica a 5 s.

**Decisione.** In corso. Report previsto:
`data/reports/musca_btc_daily_portfolio_fqi.json`.

**Risultato FT-036.** Hash
`1f4cbfc845453c542ace123ba40c7c8b4dc76d7d821c47393524ef8a781bffaa`, verdetto
`NO_INCREMENTAL_FQI_DAILY_ALPHA`. I giorni parziali di fine split sono esclusi. Nel Q3 2025 Ridge
produce 1,01 trade/giorno con EV −11,21 bps, PF 0,63 e drawdown 13,70%. XGBoost riduce il TD MAE
ma a 2,63 trade/giorno produce EV −14,18 bps, PF 0,52 e drawdown 40,29%; non può diventare
champion. Ridge rifittata e calibrata sul Q4
emette zero azioni nel 2026. Nessun gate economico, tranne il rispetto del risk budget, passa.

**Decisione.** Respinta e vietata da ripetere sugli stessi action label. La policy paper positiva
Auto-MoE resta invariata. Il risultato dimostra che l'obiettivo giornaliero corretto non crea edge
nei candidati frequenti già negativi dopo i costi; un nuovo tentativo richiede nuovi dati o un
nuovo meccanismo economico osservabile, non un'altra loss, soglia o rete.

## Template per il prossimo esperimento

### YYYY-MM-DD — FT-NNN: titolo

**Ipotesi preregistrata.**

**Dati e split.**

**Protocol hash.**

**Candidati e funnel.**

**Risultati netti per VIP.**

**Gate superati/falliti.**

**Decisione:** accettata / respinta / solo shadow.

**Prossimo passo.**
