# Musca BTC Binance — piano master del training

Aggiornato: 2026-08-11  
Stato: protocollo di lavoro; nessuno degli otto interventi è ancora completato  
Ambito: solo Binance USD-M `BTCUSDT`, training e replay Musca BTC

## Stato implementazione dopo il blackout

Protocollo challenger corrente: `24caa8b9311c901320f2fe45fc206bdb8bf4c3aba82df77d59854965949456da`.
Il vecchio run non è stato ripreso. Le caselle restano non spuntate finché il nuovo walk-forward non
produce anche gli artefatti OOS; il codice già completato è elencato qui per evitare di ripeterlo.

- punto 1 implementato: critic expanding cross-fitted con purge e test che ogni encoding usa
  esclusivamente history con uscita precedente alla riga codificata;
- punto 2 implementato: `sized_portfolio_return`, `log_utility`, ranking e contabilità equity;
- punto 3 implementato per l'ingresso: fitted value semi-Markov causale con
  `Q(ENTER)=utility+V(next_free)` e `Q(WAIT)=V(next_decision)`; nessun oracle entra nel test;
- punto 5 implementato: testa decomposta e direct-net/direct-utility sulle stesse split, Ridge
  champion e XGBoost CUDA promosso soltanto con miglioramento congiunto;
- punto 6 implementato: ingresso `Q(action)>Q(WAIT)`; la vecchia curva di soglie è soltanto audit;
- controlli P0 implementati: ledger persistente, random/shift/permutation/FULL/equal-weight/no-gate,
  LONG/SHORT/momentum/mean-reversion/always-WAIT e bucket di calibrazione economica;
- audit punto 8 predisposto: correlazione prediction/error, residuo di FULL, leave-one-view-out e
  relazione dispersione→errore; LIQUIDITY resta shadow-only;
- collector Binance aggiornato senza cambiare i consumer: conserva anche timestamp exchange e
  receive-time di ogni aggTrade, latenza book e ritardo di aggregazione separati.

Verifiche locali correnti: 33 test mirati e 369 test completi verdi, Ruff verde e mypy verde sui
due moduli modificati.
`data/reports/musca_btc_execution_contract.json` rileva 16/16 archivi event-level disponibili ma
soltanto 6 giornate L2 osservate. I label storici restano quindi dichiarati proxy e non possono
autorizzare capitale reale. Mancano ancora l'uso completo dell'ordine event-level nei label,
l'audit/correzione locale del plan generator e l'attivazione condizionale della policy intra-trade.

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

## Ordine vincolante degli otto interventi

### [ ] 1. Cross-fitting temporale del critic

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

### [ ] 2. Obiettivo unico sull'equity

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

### [ ] 5. Benchmark direct-EV e direct-utility

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
- dichiarare fill eseguibili usando last/mid senza bid/ask e profondità osservati;
- confrontare EV in bps quando sizing e obiettivo sono sull'equity;
- trattare WAIT/FLAT come vincita oppure come valore costante se esistono opportunità esclusive;
- chiamare `HOLD/REDUCE/CLOSE` azioni apprese quando sono soltanto esecuzioni del piano;
- aggiungere expert, viste, seed, Optuna trial o librerie prima dell'ablation che ne dimostra il bisogno;
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
- Offline RL limits: <https://proceedings.mlr.press/v178/foster22a.html>
- Direct trading utility with costs: <https://pubmed.ncbi.nlm.nih.gov/18249919/>
- Binance public data: <https://github.com/binance/binance-public-data/blob/master/README.md?plain=1>
- Binance commission rate: <https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/User-Commission-Rate>
- Binance aggregate trade stream: <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Aggregate-Trade-Streams>
- Binance diff depth stream: <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Diff-Book-Depth-Streams>
- Binance local order book contract: <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>
- Binance notional/leverage brackets: <https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/Notional-and-Leverage-Brackets>
- `arch` multiple comparisons: <https://bashtage.github.io/arch/multiple-comparison/multiple-comparison-reference.html>
