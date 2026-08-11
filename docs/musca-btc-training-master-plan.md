# Musca BTC Binance — piano master del training

Aggiornato: 2026-08-11  
Stato: punti 1, 2 e 5 verificati; action-space corretta, nuovo preflight in preparazione
Ambito: solo Binance USD-M `BTCUSDT`, training e replay Musca BTC

## Stato implementazione dopo il blackout

Protocollo challenger corrente: `e5560023f084bde0e2b8356112f303b983ff4a65ccec595b2bdc76527bd92f9d`.
Il vecchio run non è stato ripreso. Le caselle restano non spuntate finché il nuovo walk-forward non
produce anche gli artefatti OOS; il codice già completato è elencato qui per evitare di ripeterlo.

- punto 1 implementato: critic expanding cross-fitted con purge e test che ogni encoding usa
  esclusivamente history con uscita precedente alla riga codificata;
- punto 2 implementato: `sized_portfolio_return`, `log_utility`, ranking e contabilità equity;
- punto 3 implementato per l'ingresso: fitted value semi-Markov causale con
  `Q(ENTER)=utility+V(next_free)` e `Q(WAIT)=V(next_decision)`; nessun oracle entra nel test;
- punto 4 implementato: perturbazioni locali one-family-at-a-time, audit past-only e abilitazione
  per fold soltanto quando il regret supera 2 bps; nessuna griglia globale di azioni;
- punto 5 implementato: testa decomposta e direct-net/direct-utility sulle stesse split, Ridge
  champion e XGBoost CUDA promosso soltanto con miglioramento congiunto;
- punto 6 implementato: ingresso `Q(action)>Q(WAIT)`; la vecchia curva di soglie è soltanto audit;
- controlli P0 implementati: ledger persistente, random/shift/permutation/FULL/best-active-expert/
  equal-weight/no-gate/train-median-constant-plan, LONG/SHORT/momentum/mean-reversion/always-WAIT e
  bucket di calibrazione economica;
- audit punto 8 predisposto: correlazione prediction/error, residuo di FULL, leave-one-view-out e
  relazione dispersione→errore; LIQUIDITY resta shadow-only;
- collector Binance aggiornato senza cambiare i consumer: conserva anche timestamp exchange e
  receive-time di ogni aggTrade, latenza book e ritardo di aggregazione separati.
- label execution aggiornati: gli archivi ufficiali vengono materializzati in Parquet ordinati per
  `timestamp_ms,event_id`; entry, target, stop, gap e trailing usano il primo evento compatibile e
  conservano timestamp, ID, prezzo e stop slippage. I conflitti non risolvibili restano esclusi
  fail-closed.
- errore E-12 corretto: `exit_seconds` conta i bucket includendo quello d'ingresso; il vecchio
  raffinatore cercava quindi il crossing nel secondo successivo. Il preflight di aprile ha ora
  validato 86.310 righe, 20.164 exit event-level e nessun secondo di fill privo di trade.
- diagnostica P1 implementata: `expected_ev_bps_per_minute` e
  `expected_log_utility_per_hour`; non partecipano alla promozione o alla scelta dei parametri.

Verifiche locali correnti: 53 test Musca verdi; la suite globale ha superato 388 test e l'unico
health-check Hypothesis `too_slow` estraneo al training è passato al rerun isolato. Ruff e mypy sono
verdi sul modulo policy modificato. Il mypy globale conserva errori preesistenti nei moduli legacy
fuori ambito; non sono stati nascosti né modificati.
`data/reports/musca_btc_execution_contract.json` rileva 16/16 archivi event-level disponibili ma
soltanto 6 giornate L2 osservate. I label storici restano quindi dichiarati proxy e non possono
autorizzare capitale reale: bid/ask, profondità e partial fill storici non esistono. La gestione usa
ancora il percorso 1s tra gli eventi d'uscita; l'ordine event-level corregge i fill critici ma non
inventa un order book. Resta condizionale l'attivazione della policy intra-trade.

Il protocollo `4ec0733e45ed33d2aa8798a6a652c931ddd17579651975ce3e62af32bc9a0e11`
è fallito durante il preflight, prima di qualsiasi fold, con 19 secondi d'uscita apparentemente
senza aggTrade. La causa era E-12 e non assenza di dati Binance; quel run non è un risultato di
training e non viene ripreso.

## Esito del protocollo ec1937 e correzione vincolante

Il protocollo `ec1937faa8a1a694e71a213ed24878060741a607ea8748fb036841d37fd2dc05` ha
completato tutti i dieci outer fold e ha prodotto `NO_STABLE_OOS_POLICY`. Non è un risultato
ambiguo: 1.820 trade OOS, 6,5 trade/giorno, −8,2669 bps/trade, PF 0,3419, drawdown 96,87% e tutti i
dieci fold negativi. Il movimento lordo scelto era −0,2667 bps medio contro 8 bps round-trip.

L'audit ha localizzato tre errori nuovi, registrati per non ripeterli:

- **E-13 — continuation rescue non valido.** `Q_ENTER≈0,349` e `Q_WAIT≈0,330` venivano calibrati
  separatamente; la loro differenza aveva correlazione −0,0008 con l'utility realizzata. Il 98,96%
  degli ingressi aveva utility immediata prevista negativa, ma la differenza dei Q lo promuoveva
  ugualmente. Il controller sceglieva quasi sempre `TRAIL_TIGHTER`, pagando fee su movimenti nulli.
- **E-14 — sizing senza stop-overrun reserve.** 184 stop su 185 oltrepassavano il budget di pochi
  millesimi perché la leva consumava esattamente l'1% senza riserva per lo slippage osservato.
- **E-15 — drawdown solo misurato, non vietato.** Il replay continuava a entrare oltre l'8%; inoltre
  le metriche per lato ereditavano il contatore globale delle violazioni.

La correzione congelata per il prossimo protocollo usa una regressione diretta del paired advantage
e il vincolo di dominanza
`decision_advantage = min(predicted_paired_advantage, immediate_expected_log_utility)`. In un mercato
esogeno WAIT conserva tutte le opportunità future: il continuation value può quindi rifiutare un
ingresso, ma non rendere conveniente una perdita immediata prevista. Il Risk Engine usa una riserva
past-only al percentile 99,9% dell'overrun osservato e veta nuovo rischio prima del drawdown 8%.
I test mirati sono 41/41 verdi; il nuovo walk-forward è necessario prima di spuntare di nuovo il
punto 3.

## Esito del protocollo c474e38c e secondo audit causale

Il protocollo `c474e38c5466047df44e840431c01329e3cd4946e8fe9644fa8deb4be7b4e497` ha
completato tutti i dieci outer fold senza crash e ha prodotto `NO_STABLE_OOS_POLICY`. La correzione
del rischio ha funzionato, ma la policy economica no: 8 trade OOS, tutti LONG, −13,8404 bps/trade,
PF equity 0,3138, drawdown 2,08%, nessuna violazione di rischio e soltanto 20% dei giorni attivi
positivi. Otto fold non hanno aperto alcun trade; i due attivi sono entrambi negativi.

Il run ha falsificato la vecchia implementazione del punto 3 e ha localizzato quattro errori nuovi:

- **E-16 — EV e utility economicamente contraddittori.** Cinque degli otto ingressi avevano EV
  netto calibrato negativo ma utility calibrata positiva. Con leva nota, Jensen impone
  `E[log(1+L·R)] <= log(1+L·E[R])`: una utility positiva non può autorizzare un EV netto negativo.
- **E-17 — backup ancora oracle.** I target ricorsivi usavano `max` del miglior futuro realizzato:
  `Q_ENTER` e `Q_WAIT` arrivavano a circa 0,30 log-equity contro utility immediata nell'ordine di
  0,0001. La sign accuracy del 94–95% era dominata dalla classe negativa e non dimostrava valore
  decisionale.
- **E-18 — azioni locali fuori supporto nel fit.** I local plan erano presenti in audit,
  calibrazione e test, ma non nel fit principale. Tutti gli otto trade scelti erano perturbazioni
  locali (`TRAIL_TIGHTER` o `TARGETS_TIGHTER`), quindi la policy stava estrapolando su azioni non
  etichettate nella parte più ampia del training.
- **E-19 — audit del piano incompleto.** Il regret locale era materiale in tutti i fold (81–83% degli
  stati aveva un piano vicino migliore), ma mancavano perturbazioni della quota parziale e il report
  non verificava la coerenza utility–EV.

Le viste non sono la prima correzione: le correlazioni degli errori OOS sono 0,97–0,99 e tutti i
controlli aggregati (`FULL`, equal-weight, no-gate, momentum, mean-reversion e best expert) restano
negativi. Aggiungere expert o un gate più flessibile prima di correggere E-16…E-19 aumenterebbe solo
la capacità di overfit.

Il protocollo preregistrato successivo è
`5561e5afdb53b1b1ea38adf42e69a23d3ce52f750b077d9a587574832457fc63` e mantiene invariato
l'hash dei label canonici. Le sole modifiche ammesse sono:

1. utility calibrata limitata dal bound coerente derivato dall'EV netto e dalla leva nota;
2. un backup semi-Markov fitted, prodotto da un modello della precedente iterazione addestrato su
   storia strettamente antecedente, mai dal miglior futuro osservato;
3. supporto locale deterministico e past-only nel fit e nell'inner calibration prima di consentire
   le stesse perturbazioni in test;
4. perturbazioni della quota parziale e diagnostiche su target/prediction positive, advantage MAE e
   violazioni della coerenza EV–utility.

Prima dei dieci fold viene eseguito un preflight limitato e dichiarato discovery. Se non mostra
coerenza contabile, supporto, almeno il numero proporzionale di trade e metriche economiche positive
con gli stessi gate, il training completo non parte. Questo impedisce un altro run costoso noto in
partenza come non economico; non trasforma il preflight in un holdout indipendente.

```powershell
uv run adaptive-bot musca-btc-policy-train --resume --preflight-only
uv run adaptive-bot musca-btc-policy-train --resume  # consentito solo dopo PREFLIGHT_PASSED
```

## Esito preflight 5561e5af e terzo audit causale

Il preflight `5561e5afdb53b1b1ea38adf42e69a23d3ce52f750b077d9a587574832457fc63`
ha completato due fold in 1.739 secondi senza crash, leakage, lettura dell'holdout o violazioni di
rischio. Ha correttamente bloccato il training completo: zero trade in entrambi i fold. La
correzione E-16…E-19 è verificata (`utility_ev_consistency_violations=0`, supporto locale nel fit
399.714/399.705 righe, target continuation past-only), ma il controller ha annullato ogni azione.

Il fallimento non equivale a assenza totale di opportunità immediate. Nel primo test erano presenti
965 righe-azione con EV previsto oltre 20 bps, EV realizzato medio +3,5879 bps; nel secondo test
quattro righe tra 12 e 20 bps hanno realizzato +123,2101 bps. Sono diagnostiche non trade
indipendenti, ma provano che il blocco è avvenuto dopo la testa immediata. Tutte le 80.618 decisioni
compatte sono state `WAIT / EXPECTED_EQUITY_UTILITY_BELOW_THRESHOLD`.

Il terzo audit ha localizzato cinque errori aggiuntivi:

- **E-20 — continuation imposto senza benchmark.** Il piano richiedeva il confronto sulle stesse
  split tra controller miope e continuation; il codice sostituiva sempre l'utility immediata con il
  paired advantage, anche quando il challenger non aveva prodotto alcun trade nella selection.
- **E-21 — maximization bias nel backup.** Lo stesso regressore sceglieva e valutava il massimo fra
  tutte le azioni del timestamp. Con molte perturbazioni rumorose questo sovrastima il valore di
  WAIT. La correzione usa due regressori addestrati su settimane temporali disgiunte: uno seleziona
  l'azione e l'altro la valuta, simmetricamente.
- **E-22 — supporto temporale insufficiente del continuation critic.** Il fold 1 aveva un solo blocco
  cross-fitted e il fold 2 soltanto due; nel secondo test la frazione di advantage predetto positivo
  era 0%. Il blocco passa da quattro a una settimana e la promozione ne richiede almeno quattro.
- **E-23 — durata ausiliaria fuori dominio.** `expected_holding_seconds` arrivava a 428.693 secondi
  nonostante il massimo piano fosse 21.600 secondi. Tempo a target e holding vengono ora limitati
  causalmente all'orizzonte del piano.
- **E-24 — decision audit opaco e replay largo.** Il log conservava il motivo WAIT ma non EV,
  advantage, Q, piano e durata del candidato respinto; inoltre ordinava copie dell'intero frame. Il
  replay ora ordina soltanto un indice compatto, conserva i campi economici del candidato e libera i
  frame a fine fold.

Il nuovo protocollo preregistrato è
`4735bba967f512b2de21ac2f5550b906b83c59ded098b64da609fa39c2530aa9`; l'hash dei
label canonici resta `64de949ad7b21c124bb5e6a8234df91a0b84766ca95388a390a2b10aade06a5c`.
La regola del controller è ora: il controller miope coerente è champion; il continuation Double-Q è
challenger e viene promosso soltanto se, sulla selection antecedente, ha almeno 30 trade, log-equity
positiva, rischio valido, almeno quattro blocchi temporali e utility superiore al miope. Se nessuno
dei due è economicamente valido, il lato resta disabilitato. Il test outer non partecipa alla scelta.

## Esito preflight 4735bba9 e quarto audit causale

Il preflight `4735bba967f512b2de21ac2f5550b906b83c59ded098b64da609fa39c2530aa9`
ha completato due fold e ha correttamente vietato il run completo. Tutti i gate causali e di rischio
sono passati, ma non è stato eseguito alcun trade. Nel selection set del primo fold il controller
miope LONG ha prodotto 29 trade, −6,9815 bps/trade e PF 0,7268; SHORT ha prodotto 5 trade,
−22,2924 bps/trade e PF 0,3281. Nel secondo fold entrambi i lati hanno prodotto zero trade.

Il clamp delle durate e il confronto myopic/Double-Q hanno quindi funzionato: il fallimento si trova
prima del controller. Nel primo outer test 965 righe previste oltre +20 bps hanno realizzato soltanto
+3,5879 bps medi; nel fold seguente quasi tutte le 885.769 righe sono state calibrate sotto zero.
Un audit diagnostico sull'intera matrice canonica ha inoltre misurato che gli score ereditati
scelgono il lato ex-post migliore soltanto nel 49,9–50,4% degli stati. I percentili estremi talvolta
sono positivi aggregati, ma cambiano segno tra mesi e non costituiscono una policy causale.

Sono registrati tre errori nuovi:

- **E-25 — calibrazione prima dell'argmax.** EV e utility venivano calibrati su tutte le righe, mentre
  il replay usa soltanto il massimo fra lati e perturbazioni. La buona calibrazione media nascondeva
  l'ottimismo del vincitore selezionato.
- **E-26 — nessuna validazione globale del vincitore.** I champion erano confrontati per lato e per
  riga; nessun set antecedente verificava il candidato finale risultante dalla competizione congiunta
  LONG/SHORT/piani.
- **E-27 — controlli negativi con copie simultanee.** Tredici copie larghe del test portavano il
  processo oltre 24 GB RAM senza aggiungere informazione.

Il protocollo `e5560023f084bde0e2b8356112f303b983ff4a65ccec595b2bdc76527bd92f9d`
divide ora le quattro settimane di calibrazione in due finestre disgiunte: due settimane per la
calibrazione delle righe e due per scegliere causalmente un solo vincitore per timestamp e calibrare
il suo effettivo bias post-selezione. Selection e outer test restano successivi. Gli altri piani non
vengono valorizzati dal loro outcome e restano non selezionabili in quel timestamp. Il report misura
ottimismo, errore prima/dopo e frazione di lati realmente scelti correttamente. I controlli negativi
sono valutati uno alla volta e liberati immediatamente. Nessun gate, costo o label è stato cambiato.

Gli artefatti del protocollo precedente sono preservati in:

- `data/reports/archive/4735bba967f512b2/musca_btc_policy.preflight.json`;
- `data/ml/musca_btc_policy/audits/4735bba967f512b2/preflight_decisions.parquet`;
- `data/ml/musca_btc_policy/audits/4735bba967f512b2/preflight_trades.parquet`.

## Esito preflight e5560023 e causa strutturale dell'action-space

Il preflight `e5560023f084bde0e2b8356112f303b983ff4a65ccec595b2bdc76527bd92f9d`
ha completato due fold in 1.871 secondi. Tutti i gate di causalita, rischio, coerenza EV-utility,
continuation, selezione del controller e supporto locale sono passati. Il calibratore sul vincitore
ha pero disabilitato entrambi i lati e il replay ha prodotto zero trade. Non e un nuovo guasto di
`FLAT`: ha correttamente rifiutato un ranking non informativo.

Nel periodo winner-only antecedente al test, il candidato scelto era realmente la migliore azione
soltanto nel 7,062% degli stati del fold 1 e nel 6,540% del fold 2. L'EV realizzata dei vincitori era
rispettivamente -7,8292 e -7,9692 bps. Gli audit aggiuntivi hanno escluso due correzioni isolate:
aggiungere le 29 feature causali omesse e usare un ranker pairwise non ha prodotto EV OOS positiva.

La causa nuova e nel passaggio expert -> piano:

- **E-28 - compressione multimodale degli orizzonti.** Cinque viste per cinque orizzonti vengono
  mediate in un unico orizzonte geometrico per lato. Il 97,7% dei piani risultanti cade tra 5 e 60
  minuti, mentre almeno una vista propone sei ore nell'85,53% degli stati. Le proposte specialistiche
  vengono quindi distrutte prima che la policy possa confrontarle.
- **E-29 - foglie chiamate impropriamente esperti.** I 2.343-2.574 candidati dei due fold sono leaf
  context dello stesso piano per lato; non sono strategie eseguibili con gestione distinta.
- **E-30 - critic non consapevole del piano.** Il generatore dei leaf legge soltanto il contesto di
  mercato. Horizon, target, stop, trailing e quota parziale non entrano nelle sue feature; le
  perturbazioni locali ereditano quindi la stessa rappresentazione del piano base.
- **E-31 - media di quantili incompatibili.** Quantili MFE/MAE appartenenti a distribuzioni e
  orizzonti differenti vengono mediati per costruire un piano virtuale. Questo non conserva ne la
  distribuzione first-passage ne una strategia specialistica osservabile.

La correzione preregistrata non aggiunge soglie economiche e non osserva il test: ogni vista propone
causalmente il proprio orizzonte migliore, le proposte identiche vengono deduplicate, il consenso e
registrato come supporto ma non puo cancellare uno specialista. Ogni piano mantiene i quantili del
proprio orizzonte e viene etichettato con la stessa gestione event-level. Il critic diventa
plan-aware. Solo dopo un audit antecedente positivo questa action-space puo raggiungere il preflight.

Gli artefatti e556 sono preservati in:

- `data/reports/archive/e5560023f084bde0/musca_btc_policy.preflight.json`;
- `data/ml/musca_btc_policy/audits/e5560023f084bde0/preflight_decisions.parquet`;
- `data/ml/musca_btc_policy/audits/e5560023f084bde0/preflight_trades.parquet`.

## E-32 - secondi senza trade usati come prezzi eseguibili

Il primo tentativo con le proposte expert distinte non ha raggiunto i fold: la costruzione di marzo
2026 si e fermata su `2026-03-09T21:38:25+00:00`. L'archivio ufficiale Binance contiene eventi nel
secondo 24 e nel secondo 26, ma nessun aggregate trade nel secondo 25. Il frame regolarizzato 1s
riportava comunque OHLC forward-filled e il simulatore permetteva a quel bucket non osservato di
attivare trailing, stop o timeout. Il raffinatore event-level ha correttamente respinto il fill.

La correzione vincolante e:

- bucket con `observed_trade=False` non sono eseguibili e non attivano stop, target o trailing;
- il timeout viene eseguito sul primo bucket con trade osservato a partire dall'orizzonte;
- se quel trade futuro non e presente, la riga fallisce closed;
- sulla GPU i bucket vuoti usano sentinel neutrali (`open=NaN`, `high=-inf`, `low=+inf`) che rendono
  impossibile ogni crossing senza modificare il kernel congelato; il timeout viene poi corretto sul
  primo trade osservato con la stessa formula CPU;
- il protocollo dei label e versionato nell'albero `state_actions/<label-hash>`: gli undici mesi
  parziali costruiti con la semantica precedente restano preservati ma non possono essere riusati.

Il protocollo corrente e `11a4748b8fce81417f20e1eeb09fddaa953b18f3797285f585275711f5103ac7`;
l'hash dei label e `215813660251767695c66181c72e8b24712983a78b47d1ebde39fc658be06e1f`.
I test mirati verificano il bucket senza trade, il timeout successivo, il versionamento e la parita
CPU/GPU. Il fallimento precedente e un errore di execution, non un risultato economico.

## E-33 - cross-fit locale conservato come copie wide

Il primo preflight E-32 ha completato 16/16 partizioni e superato marzo, ma nel fold 1-local
conservava in `pieces` ogni blocco trasformato con tutte le colonne, insieme a `ordered`, history e
modelli. Il working set e salito a 23,68 GB con meno di 2 GB fisici liberi. Il worker e stato fermato
prima dell'OOM; nessun risultato economico e stato prodotto e tutte le partizioni atomiche restano
riutilizzabili.

Il cross-fit conserva ora durante i blocchi soltanto chiavi e feature del critic, libera history e
library a ogni blocco, scrive il cache encoded e fa una sola merge finale. Il fit del catalogo finale
avviene prima della merge. Split, righe, target, modelli e protocol hash sono invariati; cambia
soltanto il picco di memoria. Un nuovo preflight riparte dalle 16 partizioni gia verificate.

Il primo resume ha inoltre rivelato una seconda duplicazione: prima di sapere se le varianti locali
fossero abilitate, il codice applicava il critic base a inner calibration, model audit, calibration,
selection e test; se l'audit risultava materiale, conservava quelle cinque copie e le sostituiva con
cinque nuove copie locali. Il picco ha raggiunto 24,23 GB. L'audit di efficienza non usa feature del
critic, quindi viene ora eseguito sui raw label; solo dopo si materializza una delle due famiglie,
mai entrambe. I frame `augmented_*` e raw vengono liberati appena consumati.

Questo è il documento persistente da rileggere prima di ogni modifica al training. Le caselle degli
otto interventi si spuntano soltanto dopo implementazione, test e produzione dell'artefatto indicato.
Una modifica parziale non conta come completamento.

## Obiettivo invariabile

Costruire una policy causale che, dato stato di mercato, expert, posizione e rischio residuo,
massimizzi l'utility attesa dell'equity netta dopo costi Binance reali. Sono ammessi trade e periodi
negativi; non sono ammessi edge inventati, frequenza imposta o scelta di parametri sul test.

- una posizione contemporanea, più posizioni sequenziali;
- rischio massimo 1% per trade, leva massima 10x e veto giornaliero del Risk Engine;
- taker execution come baseline finché non esistono label maker osservate;
- VWAP/AVWAP restano centro, benchmark e contesto anche se l'audit dovesse respingere `VWAP` come
  expert autonomo;
- costi 1x nella decisione; 1,5x e 2x soltanto diagnostici;
- holdout futuro sigillato e nessun denaro reale.

## Run parametrico interrotto e congelato

Il run con protocol hash
`b3bec96c5e3d662e9ab7a20e7ae240778cdc13116e864042fe52ab5d0c49cbac` è stato interrotto da un
blackout al fold 10/10, fase `model_fit SHORT xgboost_cuda`, 78%. Non viene ripreso perché contiene
ancora gli errori strutturali E-01…E-11 e non deve consumare altro calcolo come se fosse una prova
definitiva. Lo stato è `INTERRUPTED_POWER_LOSS`, non `NO_POLICY` e non un risultato OOS completo.

Log e stato antecedenti al blackout sono preservati in:

- `data/logs/musca-btc-policy.stdout.pre-outage-20260811-093518.log`;
- `data/logs/musca-btc-policy.stderr.pre-outage-20260811-093518.log`;
- `data/reports/musca_btc_policy.status.pre-outage-20260811-093518.json`.

Il prossimo training parte soltanto dopo l'implementazione e il congelamento di questo piano.
Nessun risultato parziale del run interrotto può scegliere retroattivamente formule, viste o soglie.

## Gate P0 — verità dell'execution prima del training

Il nuovo commento tecnico ha identificato una lacuna confermata nel codice: gli archivi mensili
Binance contengono `aggTrades` con timestamp in millisecondi, ma `_aggregate()` li comprime in OHLC
da un secondo. Il replay perde quindi l'ordine degli eventi dentro il secondo. Inoltre il training
entra/esce su prezzi trade aggregati, mentre il paper live cammina bid/ask e profondità osservati.
Training e paper non hanno oggi lo stesso contratto d'esecuzione.

Prima del punto 1 devono essere completate queste attività P0:

1. preservare ordine, ID, prezzo, quantità, aggressore e timestamp degli `aggTrades` event-level;
2. risolvere first-passage, gap, stop, target e trailing sull'ordine reale degli eventi, non su OHLC;
3. separare esplicitamente `alpha_path_return` da `executable_return`;
4. per i periodi con L2 osservato, usare bid/ask, book walking, size disponibile, latency, slippage,
   partial fill, stop slippage e gap; nessun mid/last come fill dichiarato reale;
5. per i periodi senza L2 storico, etichettare l'esecuzione come proxy e vietarne la promozione a
   prova operativa; nessuna depth/spread/partial fill inventata;
6. aggiungere stress market-impact dipendente dal notional, chiaramente separato dal costo 1x;
7. rendere identico il motore condiviso di label, audit, replay e shadow paper;
8. simulare regole USD-M: mark price, margin disponibile, maintenance margin, leverage bracket,
   liquidazione, precision/minimum notional, rejection, conditional-order failure e data gap.

Il paper attuale possiede già book walking, fee, slippage osservato e rifiuto per profondità
insufficiente: queste funzioni vanno riutilizzate, non riscritte. La copertura L2 storica nota è
soltanto di pochi giorni; fino a nuova copertura, il modello execution resta separato dall'Alpha e
fail-closed fuori distribuzione.

Artefatto P0: `data/reports/musca_btc_execution_contract.json`, contenente copertura temporale,
risoluzione, fonti, hash, latency distribution, fill assumptions e differenze training/paper pari a
zero sui periodi osservabili.

## Controlli P0 trasversali

### Research-overfitting ledger

Prima di ogni esperimento, `research_registry.json` deve registrare in append-only:

- `experiment_id`, ipotesi falsificabile e modifica unica;
- commit/protocol/data hash;
- periodi fit/calibration/selection/test osservati;
- numero globale di modelli, viste, soglie e policy confrontati;
- risultato, gate, decisione accepted/rejected e motivo;
- contaminazione permanente di ogni periodo visualizzato.

PBO, DSR e SPA/Reality Check useranno il numero globale dei tentativi, non soltanto i modelli del
run corrente. Modificare il protocollo dopo aver visto un outer-OOS rende quel periodo discovery.

### Negative controls obbligatori

Ogni nuovo protocollo deve essere confrontato, sulle stesse split, con:

- prediction casuale, prediction temporalmente shiftata e label permutation;
- `FULL` only, miglior singolo expert, media uniforme, no gating e constant/simple plan;
- LONG only, SHORT only, momentum semplice, mean reversion semplice e always WAIT.

Un componente viene mantenuto soltanto se mostra valore incrementale paired OOS. Se
`equal_weight > gate`, il gate fallisce; se `MoE ≈ FULL`, il MoE non ha dimostrato valore.

## Diagnosi congelata

| ID | Problema verificato | Evidenza nel codice corrente |
|---|---|---|
| E-01 | Target encoding nel critic | Le statistiche dei leaf del fit sono calcolate con gli stessi outcome poi assegnati alle righe del fit. |
| E-02 | Objective mismatch | Il ranking usa EV in bps, mentre replay e sizing operano su equity con leva variabile. |
| E-03 | `WAIT = 0` miope | Non viene stimato il valore dell'occasione successiva né il costo di occupare la posizione. |
| E-04 | Piano non ottimizzato | Il gate produce un solo LONG e un solo SHORT con formule deterministiche; il critic può rifiutarli ma non correggerli. |
| E-05 | Catena EV fragile | `P(event) × E[return|event]` più due calibrazioni può accumulare errore; manca il controllo direct-EV/direct-utility. |
| E-06 | Soglia potenzialmente rumorosa | Un vincitore scelto su quattro settimane può cambiare fortemente tra fold con pochi trade. |
| E-07 | Gestione non appresa | `HOLD/REDUCE/TIGHTEN/CLOSE` sono esecuzione del piano iniziale e log post-hoc, non action value controfattuali. |
| E-08 | Viste poco ortogonali | `full/VWAP/trend/flow/regime` sono price-centric; il gate a consenso può sopprimere una specializzazione utile. |
| E-09 | Execution train/paper divergente | Training usa OHLC 1s/last trade; paper usa book L2 osservato e book walking. |
| E-10 | Research overfitting | Decine di protocolli e periodi OOS già osservati richiedono un ledger globale, non solo nested folds. |
| E-11 | Mancanza di controlli negativi completi | Non è ancora dimostrato quale componente batta casuale, FULL, equal-weight e regole semplici. |
| E-12 | Exit bucket spostato di un secondo | `exit_seconds=1` valutava il bucket d'ingresso ma il raffinatore interrogava `entry+1s`, producendo falsi eventi mancanti. |
| E-13 | Differenza di due Q sovrastimata | Due calibratori separati permettevano al continuation value di promuovere utility immediata negativa. |
| E-14 | Stop overrun fuori sizing | Lo slippage event-level, pur piccolo, non aveva alcuna riserva nel budget 1%. |
| E-15 | Drawdown senza veto | Il replay misurava l'8% solo a posteriori e duplicava le violazioni nelle metriche LONG/SHORT. |
| E-16 | EV netto e utility contraddittori | Cinque degli otto ingressi c474 avevano EV calibrato negativo ma utility calibrata positiva. |
| E-17 | Continuation target ancora oracle | Il backup ricorsivo usava il massimo futuro realizzato e generava livelli Q circa mille volte l'utility immediata. |
| E-18 | Local plan fuori supporto | Le perturbazioni erano etichettate da model-audit in poi ma non nel fit principale; tutti i trade c474 erano locali. |
| E-19 | Audit locale incompleto | Mancavano quota parziale e controllo esplicito della coerenza EV–utility. |
| E-20 | Continuation imposto | Il paired advantage sostituiva sempre il champion miope senza evidenza incrementale sulla selection. |
| E-21 | Massimo dello stesso regressore | Lo stesso modello sceglieva e valutava l'azione futura, sovrastimando WAIT. |
| E-22 | 1–2 blocchi continuation | Il critic dell'advantage non aveva supporto temporale indipendente sufficiente. |
| E-23 | Durata prevista fuori piano | Il regressore ausiliario poteva prevedere giorni per un piano massimo di sei ore. |
| E-24 | WAIT opaco e replay largo | Mancavano valori del candidato respinto e venivano ordinate copie di tutti i campi. |
| E-25 | Calibrazione prima dell'argmax | La calibrazione media delle righe non correggeva l'ottimismo del massimo realmente scelto. |
| E-26 | Vincitore globale non validato | Il confronto dei modelli era per lato/riga, non sul candidato finale LONG/SHORT/piano. |
| E-27 | Copie simultanee dei negative controls | Tredici frame larghi portavano il processo oltre 24 GB RAM. |

## Ordine vincolante degli otto interventi

### [x] 1. Cross-fitting temporale del critic

**Ipotesi.** Il critic appare più informativo nel fit perché ogni campione contribuisce alle
statistiche del proprio leaf.

**Implementazione.**

- produrre le feature dei leaf del fit con cross-fitting cronologico expanding e purge sull'uscita;
- nessuna riga può contribuire al modello o alla statistica che genera le sue feature;
- applicare lo stesso divieto a ogni encoding derivato da return, TARGET/STOP/TIMEOUT, MFE, MAE,
  durata o utility, non soltanto ai leaf del critic;
- le righe iniziali senza storia sufficiente vengono escluse o marcate fail-closed, mai riempite con
  statistiche future;
- calibration, selection e test ricevono statistiche prodotte esclusivamente dal fit antecedente;
- conservare critic attuale come controllo `same-sample`, ma vietarlo nella policy.

**Verifica/uscita.** Test di non-appartenenza del campione, timestamp, purge e determinismo; confronto
train/inner-OOS della distribuzione delle feature e del regret. Artefatto:
`data/reports/musca_btc_policy_critic_crossfit.json`.

**Completato quando.** Nessun sample vede il proprio outcome nel critic encoding e l'intera suite
causale passa.

### [x] 2. Obiettivo unico sull'equity

**Ipotesi.** Ordinare per `calibrated_ev_bps` non ordina correttamente piani con stop e leva diversi.

**Implementazione.**

- calcolare per ogni outcome l'effettivo `portfolio_return` dopo sizing, fee e funding;
- aggiungere come target primario `log_utility = log1p(portfolio_return)`;
- confrontare LONG, SHORT e in seguito WAIT sulla stessa unità economica;
- mantenere bps/notional come diagnostica, non come criterio finale;
- rendere coerenti expectancy, PF e log-growth: pubblicare sia metriche unlevered sia equity-weighted.
- verificare le curve di calibrazione economica sull'utility dimensionata, non soltanto la
  calibrazione della classe evento.

**Verifica/uscita.** Casi con uguale EV in bps e stop diversi devono produrre ranking economico
diverso; identità contabili trade→giorno→equity. Artefatto:
`data/reports/musca_btc_policy_equity_objective.json`.

**Completato quando.** Training, ranking, threshold/WAIT e replay ottimizzano la stessa utility e il
report non mescola più vettori diversi senza etichettarli.

### [ ] 3. Continuation value causale di WAIT

**Ipotesi.** Un trade positivo ma lungo può essere peggiore dell'attesa perché blocca opportunità
future migliori.

**Implementazione.**

- prima del modello sequenziale, registrare come diagnostica `EV/minuto` e
  `expected_log_growth/expected_holding_time` per quantificare il costo di occupazione;
- formulazione semi-Markov: WAIT transisce alla prossima decisione; ENTER transisce al primo stato
  libero successivo all'uscita;
- target `Q(action)=utility realizzata + V(next_free_state)` e
  `Q(WAIT)=V(next_decision_state)`;
- il continuation value proviene sempre da un modello cross-fitted dell'iterazione precedente, mai
  dal miglior futuro realizzato;
- nessuna chiusura obbligatoria a mezzanotte UTC; rimuovere le code di split senza copertura del
  massimo orizzonte;
- il primo benchmark mantiene deterministica la gestione intra-trade per isolare il solo valore di
  WAIT.

**Non ripetere FT-035/FT-036.** FT-035 assegnava a FLAT un futuro onnisciente; FT-036 usava i vecchi
piani fissi, episodio UTC e obiettivo non ancora coerente con la nuova utility. Il nuovo test deve
usare piani parametrizzati, cross-fitting, transizioni `next_free` e objective del punto 2.

**Verifica/uscita.** Test sintetico in cui un +3 bps lungo viene correttamente rifiutato a favore di
un'opportunità successiva migliore; confronto miope vs continuation sulle identiche split.
Artefatto: `data/reports/musca_btc_policy_wait_value.json`.

**Completato quando.** WAIT ha un valore predetto OOS e non è più una costante zero.

**Esito ec1937.** WAIT non era costante, ma l'ipotesi operativa è stata respinta: sottrarre due Q
calibrati indipendentemente ha creato un advantage non economico. Il nuovo paired advantage con
dominance cap è implementato e testato, ma resta non spuntato fino alla nuova evidenza OOS.

### [ ] 4. Audit di efficienza e correzione del plan generator

**Ipotesi.** Il downstream rifiuta piani localmente inefficienti che non può modificare.

**Implementazione.**

- generare, soltanto dentro fit/inner-audit, piccole perturbazioni supportate di horizon, TP1, TP2,
  stop, trailing e quota parziale attorno al piano proposto;
- simulare ogni perturbazione con lo stesso percorso a un secondo e gli stessi costi;
- misurare local oracle regret, coverage e distanza dal supporto per stato/lato;
- se il regret è materiale, addestrare un critic parametrico e scegliere il massimo previsto tra
  proposte locali supportate; il test non determina perturbazioni o limiti;
- non trasformare lo stencil diagnostico in un'altra grande griglia fissa.

**Verifica/uscita.** Distribuzione del regret locale e confronto piano base/piano scelto OOS; stop
non allargabile e medesimo replay label/paper. Artefatto:
`data/reports/musca_btc_policy_plan_efficiency.json`.

**Completato quando.** Il generatore produce piani localmente efficienti OOS oppure viene dimostrato
che il collo di bottiglia non è nei parametri del piano.

### [x] 5. Benchmark direct-EV e direct-utility

**Ipotesi.** La decomposizione TARGET/STOP/TIMEOUT perde segnale lungo la catena di modelli e
calibratori.

**Implementazione.**

- conservare la testa decomposta per interpretabilità;
- addestrare sulle stesse identiche righe/split un Ridge direct-net-return e direct-log-utility;
- usare un solo XGBoost CUDA challenger, senza nuovi trial o nuove librerie;
- confrontare MAE, calibrazione, decision regret, Brier della testa eventi e log-growth replay;
- produrre bucket OOS di predicted EV/utility (`<0`, `0–2`, `2–4`, `4–8`, `8–12`, `12–20`,
  `>20 bps`) con valore realizzato, supporto e intervallo temporale;
- richiedere relazione monotona prediction→realizzato e diagnosticare separatamente
  overconfidence nelle code;
- promozione soltanto inner-OOS; nessuna scelta sul test outer.

**Verifica/uscita.** Tabella paired per lato e fold, con intervalli a blocchi temporali. Artefatto:
`data/reports/musca_btc_policy_value_heads.json`.

**Completato quando.** È congelata la testa che ordina meglio l'utility OOS, oppure entrambe vengono
respinte con una diagnosi distinta `NO_PREDICTABLE_UTILITY`.

**Esito ec1937.** Ridge è rimasto champion in 19 confronti su 20; direct è stato scelto soltanto in
due lati/fold. I bucket erano monotoni solo in 3 fold su 10 e tutti i candidati con EV previsto
positivo hanno realizzato −5,17 bps medi aggregati. Le due teste sono quindi respinte come
`NO_PREDICTABLE_UTILITY` per quel protocollo; non si aggiungono modelli.

### [ ] 6. Stabilità della regola di ingresso

**Ipotesi.** L'argmax della log-equity su una sola finestra di quattro settimane seleziona rumore.

**Implementazione.**

- registrare per lato/fold/soglia: candidati sopra soglia, trade eseguiti, occupazione, log-growth,
  expectancy, drawdown e sottoperiodi;
- misurare turnover della soglia tra fold adiacenti;
- con il punto 3 attivo, preferire `Q(action) > Q(WAIT)` e rimuovere la soglia duplicata;
- se resta un margine prudenziale, stimarlo con inner rolling windows e criterio robusto
  preregistrato, non con il massimo di una singola finestra;
- la frequenza è un risultato della frontiera P&L/rischio, non un quota.

**Verifica/uscita.** Curva completa frequenza–utility e stabilità per sottoperiodo. Artefatto:
`data/reports/musca_btc_policy_entry_stability.json`.

**Completato quando.** La regola d'ingresso è identica in training/replay/paper e non dipende da un
threshold winner instabile.

### [ ] 7. Policy intra-trade semi-Markov

**Prerequisito.** Si esegue soltanto dopo i punti 1–6; non si usa per nascondere un ingresso senza
edge.

**Implementazione.**

- stati event-driven: nuovi setup, TP1, deterioramento EV, avvicinamento stop, attivazione trail e
  checkpoint temporali preregistrati;
- azioni con posizione: `HOLD`, `CLOSE`, `REDUCE`, `TIGHTEN_STOP`, `UPDATE_TRAIL`;
- controfattuali sul percorso a un secondo con reward in log-equity e continuation value dopo
  l'uscita;
- fitted-Q/backward induction supervisionato con pessimismo fuori copertura, non PPO/SAC;
- stop mai allargabile, una posizione e Risk Engine esterno invariati.

**Verifica/uscita.** Ogni azione ha label e action value propri; confronto gestione deterministica
vs appresa sulle stesse split. Artefatto:
`data/reports/musca_btc_policy_intratrade.json`.

**Completato quando.** `HOLD/REDUCE/TIGHTEN/CLOSE` sono decisioni OOS riproducibili e non semplici
righe di log derivate dall'esito.

### [ ] 8. Audit delle viste e del gating

**Ipotesi.** Le viste attuali sono semanticamente sovrapposte e la penalità di disaccordo può
spegnere lo specialista proprio quando è utile.

**Audit obbligatorio per ciascun orizzonte.**

- correlazione delle prediction OOS e, soprattutto, correlazione degli errori;
- informazione residuale: capacità di ogni vista di predire il residuo di `FULL`;
- leave-one-view-out sulle metriche di calibrazione, utility, drawdown, frequenza, turnover,
  stabilità e lati separati;
- relazione OOS tra dispersione degli expert, errore e rendimento dei trade;
- ablation paired: singolo expert, media uniforme, gate robusto corrente e sparse gate appreso.
- includere i negative controls P0 e verificare che il risultato non sopravviva a shift/permutation.

**Candidati semantici, non decisione anticipata.**

- `FULL`: generalista/benchmark, da mantenere;
- `TREND`: prezzo, momentum, struttura e VWAP/AVWAP;
- `FLOW`: taker/aggressor activity;
- `LIQUIDITY`: spread, depth, book slope, replenishment, cancellation, microprice e impatto;
- `DERIVATIVES`: basis, funding, mark-index e OI;
- `REGIME/VOL`: candidato soprattutto per gating/critic, non necessariamente direzionale.

Se si resta a cinque viste, l'ipotesi da testare è
`FULL/TREND/FLOW/LIQUIDITY/DERIVATIVES`, con VWAP dentro FULL/TREND e regime nel gate. Non viene
adottata senza ablation OOS. Conoscenza importante: FT-029 disponeva di sole quattro giornate L2;
se la copertura storica resta insufficiente, `LIQUIDITY` sarà shadow-only e non entrerà nel
backtest. Nessuna depth, cancellation o fill verrà inventata. Derivatives usa solo serie Binance
con semantica, copertura e `available_at` verificati; nessuna imputazione a zero.

**Verifica/uscita.** Matrice di ridondanza e ablation paired con protocollo congelato. Artefatto:
`data/reports/musca_btc_policy_view_audit.json`.

**Completato quando.** Il numero e l'identità delle viste derivano da informazione incrementale OOS,
non da una preferenza semantica, e il gating è promosso soltanto se migliora la baseline paired.

## Sequenza e stop scientifici

```text
run interrotto congelato
  → P0 execution event-level + ledger + negative controls
  → 1 cross-fitting
  → 2 equity objective
  → 3 WAIT continuation
  → 4 plan efficiency
  → 5 value heads
  → 6 entry stability
  → 7 intra-trade
  → 8 views/gating
  → nuovo protocollo congelato
  → nested walk-forward una volta
  → paper research
  → holdout futuro sigillato
```

Ogni punto può produrre `IPOTESI_RESPINTA`: in quel caso si conserva il controllo precedente e si
registra l'evidenza. Non si procede a un nuovo tentativo della stessa famiglia con soglie diverse.
Un risultato positivo sui periodi contaminati resta discovery e non autorizza denaro reale.

## Prove già effettuate — registro anti-ripetizione

| Famiglia | Prove | Evidenza | Conclusione da conservare |
|---|---|---|---|
| Baseline VWAP positiva ma rara | FT-000–FT-002 | `docs/musca-v5-fine-tuning-log.md` | Esisteva un controllo promettente, ma allargare staticamente il pool non aumentava la frequenza in modo OOS. |
| Target fee-linked, TP1/runner e room/risk | FT-003–FT-011 | stesso log | Target collegati ai costi, soglie micro, timing 1s/5s e nuovi expert impulso non hanno creato una frontiera stabile. |
| Maker, anchor e restart VWAP | FT-012–FT-021 | stesso log | Maker non osservato, funding VWAP, touch/restart, rimozione invalidazione e modelli locali non hanno fornito causalità frequente. |
| Bitunix POST_ONLY/L2 | FT-022–FT-030 | stesso log | Molte opportunità teoriche ma negative dopo adverse selection; L2 aveva copertura insufficiente e label private mancanti. |
| Binance high-frequency | FT-031–FT-033 | `docs/musca-btc-daily-portfolio-challenger.md` | Aumentare copertura/frequenza non ha prodotto Alpha netto sostenibile. |
| Costo opportunità e Q giornaliero | FT-034–FT-036 | stesso documento | Penalità lineare durata, oracle WAIT e FQI UTC sui vecchi piani sono falliti; FT-035 aveva FLAT onnisciente. |
| MoE 125 componenti | protocol hash `73563d…` | `docs/musca-btc-moe-results.md` | 170 trade audit, −15,22 bps, PF 0,671: più modelli non correggono label/policy. |
| Auto-MoE discovery/gate | protocol hash `cbc1fa…` | `docs/musca-btc-auto-moe-results.md` | 178 trade, quasi pareggio aggregato ma forte instabilità giugno/luglio. |
| Piani fissi 2×5 | precedente canonical policy | `docs/musca-btc-action-space-audit.md` | Gli expert contestuali valutavano piani ereditati; spazio d'azione sostituito. |
| Piani parametrizzati multi-expert | run `b3bec9…`, interrotto dal blackout | log pre-outage preservati | Prima baseline con 25 contributi e piani dinamici; non riprendere perché contiene ancora E-01…E-11. |

Il dettaglio append-only FT-000–FT-036 resta in `docs/musca-v5-fine-tuning-log.md`; questo piano non
lo duplica riga per riga e non lo sostituisce.

## Errori vietati

- usare outer test o aggregato dei dieci fold per scegliere una modifica e poi chiamarlo conferma;
- interpretare oracle, MFE o miglior futuro realizzato come segnale disponibile live;
- calcolare target encoding con la stessa riga che lo riceve;
- dedurre l'ordine stop/target da OHLC 1s quando sono disponibili eventi ordinati;
- usare OHLC forward-filled di un secondo senza aggregate trade come prezzo eseguibile;
- convertire il numero di bucket in timestamp con `entry + exit_seconds` invece di
  `entry_bucket + exit_seconds - 1`;
- dichiarare fill eseguibili usando last/mid senza bid/ask e profondità osservati;
- confrontare EV in bps quando sizing e obiettivo sono sull'equity;
- trattare WAIT/FLAT come vincita oppure come valore costante se esistono opportunità esclusive;
- consentire al continuation value di rendere positivo un ingresso con utility immediata prevista
  negativa in un mercato esogeno a una posizione;
- sottrarre due Q calibrati indipendentemente senza un paired-advantage auditato;
- calibrare EV netto e utility come due autorizzazioni indipendenti quando violano il bound di Jensen;
- costruire continuation target con il massimo futuro realizzato invece del precedente modello fitted;
- autorizzare in test una perturbazione di piano che non possiede esempi etichettati nel fit;
- imporre il continuation controller senza un confronto paired con il controller miope sulla
  selection antecedente;
- usare lo stesso regressore per selezionare e valutare il massimo futuro fra molte azioni;
- promuovere un continuation critic con meno di quattro blocchi temporali cross-fitted;
- accettare una durata prevista maggiore dell'orizzonte del piano;
- dimensionare esattamente sullo stop senza una riserva causalmente stimata per l'overrun;
- continuare ad aprire nuovo rischio quando il budget di drawdown residuo è inferiore al worst risk;
- chiamare `HOLD/REDUCE/CLOSE` azioni apprese quando sono soltanto esecuzioni del piano;
- aggiungere expert, viste, seed, Optuna trial o librerie prima dell'ablation che ne dimostra il bisogno;
- conservare copie wide di ogni blocco cross-fitted invece delle sole feature encoded;
- usare consenso tra expert come sinonimo di accuratezza senza il test disaccordo→errore OOS;
- creare dati L2, depth, spread, maker fill, OI o funding mancanti tramite simulazione o zero-fill;
- riaprire FT-000–FT-036 cambiando soltanto soglie, target o mesi;
- imporre un numero di trade, abbassare gate/costi o usare stress 2x nella decisione reale;
- mutare Auto-MoE, Musca V2/V5 paper o il run attivo mentre si lavora sul challenger;
- promuovere una policy perché un solo lato, mese o fold appare positivo;
- riaddestrare automaticamente su drift senza alert, audit e nuovo protocollo congelato;
- aprire l'holdout futuro prima del congelamento di codice, protocollo e hash dati.

## Gate del protocollo finale

Il nuovo walk-forward viene lanciato una sola volta dopo gli otto punti e i test locali. Il report
deve separare LONG/SHORT e dichiarare almeno:

- numero di trade e giorni/settimane indipendenti;
- expectancy e PF sia notional-bps sia equity-weighted;
- log-growth medio e bootstrap LCB temporale;
- drawdown massimo ≤ 8%, nessuna violazione rischio;
- maggioranza dei giorni attivi positiva;
- calibrazione, regret, turnover, occupazione e stabilità tra fold;
- SPA/Reality Check, PBO e DSR corretti per il registro globale dei tentativi;
- costi 1x decisionali e stress 1,5x/2x separati;
- divergenza train→inner-OOS→outer-OOS;
- parità label/replay/shadow su timestamp, bid/ask, quantità, fee, slippage e gestione;
- controllo liquidation/bracket/margin/rejection/data-gap, anche quando non scatta;
- verdict causale preciso, senza trasformare FLAT in profitto.

Anche `RESEARCH_PAPER_READY` autorizza soltanto lo shadow. Prima del capitale reale il modello resta
congelato e riceve stream Binance live, feature, expert, gate, piani e decisioni completi. Lo shadow
confronta prediction live/replay e fill assunto/osservabile e verifica timestamp, latency, warm-up,
reconnect, precision, minimum quantity e rejection. Il capitale reale richiede inoltre holdout
futuro non riutilizzato e verifiche operative separate.

Il runtime pubblica drift per expert e lato: distribuzione prediction, dispersione, gate entropy,
predicted/realized utility, calibration slope, MFE/MAE, stop/timeout rate, holding time e feature
drift. Il drift genera un alert e un audit; non autorizza retraining automatico.

## File sorgente e artefatti da preservare

- `src/adaptive_bot/musca_btc_policy.py` — pipeline challenger;
- `src/adaptive_bot/musca_btc_moe.py` — expert/view parent e feature;
- `src/adaptive_bot/musca_btc_auto_moe.py` — controllo congelato;
- `tests/unit/test_musca_btc_policy.py` — test causali/policy;
- `data/ml/musca_btc_policy/` — matrici, cataloghi e registro;
- `data/reports/musca_btc_policy*.json` — baseline e audit dei punti;
- `data/models/musca_btc_policy/` — bundle research;
- `docs/musca-btc-action-space-audit.md` — diagnosi spazio d'azione;
- `docs/musca-v5-fine-tuning-log.md` — registro completo delle prove precedenti;
- `docs/musca-btc-training-master-plan.md` — questo piano e unica checklist operativa.

## Fonti primarie già adottate

- Mixture of Experts: <https://www.cs.toronto.edu/~hinton/absps/jacobs.pdf>
- Parameterized actions: <https://ojs.aaai.org/index.php/AAAI/article/view/10226>
- Conservative Q-Learning: <https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html>
- Average-cost semi-Markov decision processes: <https://doi.org/10.2307/3211944>
- Offline RL limits: <https://proceedings.mlr.press/v178/foster22a.html>
- Direct trading utility with costs: <https://pubmed.ncbi.nlm.nih.gov/18249919/>
- Binance public data: <https://github.com/binance/binance-public-data/blob/master/README.md?plain=1>
- Binance commission rate: <https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/User-Commission-Rate>
- Binance aggregate trade stream: <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Aggregate-Trade-Streams>
- Binance diff depth stream: <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Diff-Book-Depth-Streams>
- Binance local order book contract: <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>
- Binance notional/leverage brackets: <https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/Notional-and-Leverage-Brackets>
- `arch` multiple comparisons: <https://bashtage.github.io/arch/multiple-comparison/multiple-comparison-reference.html>
