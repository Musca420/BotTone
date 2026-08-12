# Musca BTC Binance — handoff completo del contesto

Aggiornato: 2026-08-12, dopo l'audit metodologico E-42--E-53

Repository: `C:\Users\david\Desktop\progetti\BotTone`

Branch: `codex/doge-auto-moe`

HEAD di partenza: `04db03378f1262e5bb9f67afba2a7beabfff8f50`

Protocollo del codice non committato: `e149e2cc97c22d5af2f95e34df49112577de4d958fc7e8b97af4cd6e59d188dc`

## Istruzione per la nuova chat

Continuare esclusivamente Musca BTC Binance da questo documento e da
`docs/musca-btc-training-master-plan.md`. Non ricominciare la ricerca, non creare una nuova versione,
non cambiare costi o gate e non avviare subito il preflight. Prima completare nell'ordine la sezione
"Prossimi passi obbligatori". Conservare tutte le modifiche e tutti i dati esistenti.

## Obiettivo invariato

Costruire una policy Musca per Binance USD-M `BTCUSDT` che:

- usa VWAP/AVWAP come centro e contesto, senza obbligare una sola famiglia di trade;
- combina expert contestuali e piani LONG/SHORT gestiti dinamicamente;
- può accettare trade e giornate negative, ma deve avere equity netta OOS sostenibile dopo costi
  Binance reali;
- confronta `ENTER_LONG`, `ENTER_SHORT` e `WAIT` quando flat, poi applica una gestione coerente della
  singola posizione;
- usa rischio massimo 1% per trade, leva massima 10x, una posizione contemporanea e veto del Risk
  Engine;
- non impone artificialmente un numero di trade; ricerca la massima frequenza economicamente
  sostenibile;
- produce al massimo `RESEARCH_PAPER_READY` prima del nuovo holdout futuro. Nessun denaro reale.

Non esiste autorizzazione a falsificare un risultato positivo. Un fallimento economico deve essere
diagnosticato con precisione, non nascosto abbassando fee, gate o incertezza.

## Stato operativo reale al passaggio di consegne

- Nessun worker di training vivo: `data/reports/musca_btc_policy.status.json` indica PID `12116`, ma
  quel PID non esiste più. È uno stato orfano fermo a `critic_crossfit`, fold 1 block 3.
- Non usare quello status come prova che il training stia avanzando.
- La dashboard sulla porta 8080 è attiva; `netstat` mostrava i PID `12604` e `12828`. Non fermarli.
- Altri processi Python appartengono ai servizi/collector esistenti e non vanno terminati alla cieca.
- Il nuovo preflight non è stato autorizzato né completato dopo le modifiche correnti.
- L'ultimo preflight concluso resta il protocollo `e38fe797...`: `PREFLIGHT_FAILED` economico, non
  crash software, con 0 trade. La frontiera diagnostica mostrava LONG fold 1: 11 trade,
  -1,0466 bps/trade; LONG fold 2: 3 trade, +114,6091 bps/trade, campione troppo piccolo. Entrambi i
  lati erano stati disabilitati nella selection antecedente.

## Stato Git e modifiche da preservare

Il worktree è intenzionalmente sporco. Non usare `git reset`, `git checkout --`, `git clean` o
cancellazioni ricorsive.

File modificati:

- `src/adaptive_bot/musca_btc_policy.py`
- `tests/unit/test_musca_btc_policy.py`
- `docs/musca-btc-training-master-plan.md`
- questo handoff

`data/tmp/` e le directory che Git non riesce a enumerare per `Permission denied` sono dati utente e
artefatti storici: non eliminarli e non cercare di cambiare proprietario/permessi come scorciatoia.

Pubblicazione GitHub richiesta ma non ancora possibile al momento dell'handoff:

- il checkout non ha alcun remote configurato;
- GitHub CLI è installato (`2.96.0`), ma `gh auth status` dichiara non valido il token dell'account
  `Musca420`;
- dopo la nuova autenticazione creare/usare un repository privato, aggiungere `origin`, pubblicare
  questo branch e aprire una draft PR;
- includere anche `data/logs/musca-btc-policy*.log` e
  `data/reports/musca_btc_policy*.json` con `git add -f`: sono ignorati globalmente da `.gitignore`,
  sono stati controllati per pattern di segreti e servono all'audit;
- non includere gli archivi `data/ml`, modelli, database, `.env` o altri log non pertinenti.

## Diagnosi del problema che ha prodotto zero trade

La correzione già presente nel commit `04db033` risolve quattro errori collegati:

1. veniva conservato un solo vincitore globale LONG/SHORT prima di sapere quale lato fosse
   economicamente abilitato; se SHORT vinceva il ranking e poi veniva disabilitato, il miglior LONG
   era già perso;
2. il ranker relativo sceglieva il piano "meno peggio" anche quando tutte le azioni avevano valore
   assoluto negativo;
3. una calibrazione comune mescolava distribuzioni LONG e SHORT;
4. la frontiera diagnostica del test veniva calcolata dopo `DISABLED`, nascondendo i candidati e
   producendo uno zero fuorviante.

Contratto corretto ora atteso:

- conservare il miglior candidato LONG e SHORT a ogni timestamp;
- calibrare e applicare controller/gate separatamente per lato;
- confrontare soltanto dopo i gate i lati ancora abilitati;
- mantenere un modello di valore assoluto oltre al ranker relativo;
- calcolare la frontiera OOS diagnostica prima di `DISABLED`;
- usare la stessa risoluzione dei lati in walk-forward, bundle finale e replay.

## Audit metodologico successivo già implementato nel worktree

Le modifiche non committate correggono problemi ulteriori scoperti dopo `04db033`:

1. **Loss del valore medio.** Le teste che devono stimare EV netto e log-utility condizionali ora
   usano squared loss. Pseudo-Huber resta soltanto per MFE, MAE e tempi evento. La media condizionale,
   non una mediana robusta implicita, è la quantità economica usata dalla decisione.
2. **Calibrazione con pochi giorni.** Isotonic è stata rimossa dalle calibrazioni di valore. È stato
   aggiunto `MonotoneAffineCalibrator`, separato per lato e pesato affinché ogni giorno UTC abbia lo
   stesso peso. La calibrazione di probabilità resta parametrica/logistica.
3. **Oracle locale.** L'audit delle piccole perturbazioni di stop/target/trailing mostrava che oltre
   il 93% dei piani aveva una variante futura migliore. Questo è un oracle e non una policy causale:
   ora resta diagnostico e non può abilitare varianti nel fit/test o nel bundle.
4. **Soglie selezionate su quattro settimane.** Le soglie non-zero restano curve diagnostiche. La
   sola regola autorizzante è `Q(action) > Q(WAIT)` a margine extra zero, con supporto minimo e
   log-equity positiva sulla selection passata. Una soglia positiva non può salvare un controller
   negativo a zero.
5. **SPA/PBO/DSR.** Sono stati corretti ambito e unità della Reality Check/SPA, combinazioni simmetriche
   del PBO e benchmark del Deflated Sharpe. Questi gate ora possono realmente impedire la promozione;
   prima erano calcolati dopo il verdict.
6. **Preflight fail-closed.** Audit LONG/SHORT mancanti o malformati, calibrazione con meno di dieci
   giorni indipendenti, metodo di calibrazione errato, soglia autorizzante non-zero o oracle locale
   abilitato fanno fallire il preflight.
7. **Contratto dello stato sequenziale.** Il modello di ingresso apprende dalle feature di mercato
   dichiarate in `MODEL_FEATURES`. P&L giornaliero, rischio residuo, posizione e drawdown sono usati
   dal replay/Risk Engine deterministico; non vengono più descritti falsamente come feature
   supervisionate indipendenti. Una policy intra-trade appresa richiederà label controfattuali e FQI
   separati, non un semplice append di colonne.
8. **Inferenza GPU.** È stato introdotto `_predict_array` per passare array CuPy ai modelli XGBoost
   CUDA ed eliminare il fallback `GPU model / CPU input`. Dopo la modifica Ruff, mypy, 66 test
   specifici e 403 test globali sono verdi; il warning `mismatched devices` non compare più.

## Evidenza sui dati e sui label già verificata

Matrice canonica esaminata:
`data/ml/musca_btc_policy/state_actions/2158136602517676/*.parquet`.

- 3.609.776 righe state-action e 699.720 stati;
- nessuna feature con `available_at` successivo all'entry;
- nessuna entry precedente alla decisione o alla disponibilità delle feature;
- nessun path/exit non positivo e nessun duplicato `(entry, side, plan_id)`;
- entry uguale al primo aggregate trade ufficiale osservato dopo la decisione;
- contabilizzazione verificata: `net = gross + funding - round_trip_cost`;
- 14 conflitti stop/target nello stesso secondo sono marcati invalidi e rimossi fail-closed;
- ritardo entry osservato 0--27,622 secondi, dovuto al primo trade successivo disponibile.

Quindi la causa attuale non è una matrice corrotta. Il segnale dei parent expert resta però debole:
le correlazioni con il rendimento netto sono circa -0,01--+0,05 e i candidati con gross previsto
oltre 8 bps risultavano mediamente negativi. L'oracle positivo dimostra opportunità ex-post, non
predicibilità live.

## Limiti reali ancora aperti

Questi punti non sono bug da nascondere e possono ancora impedire una policy stabile:

- **Predicibilità causale debole.** L'oracle trova piani profittevoli, ma il modello non ha ancora
  dimostrato di identificarli OOS prima del movimento.
- **Generatore del piano.** L'elevato local regret segnala piani spesso inefficienti, ma le varianti
  non possono essere eseguite finché un selettore causale past-only non batte il piano base OOS.
- **Execution storica.** Gli aggregate trade a un secondo permettono il path degli eventi, ma non
  ricostruiscono coda, bid/ask storico L2, partial fill o maker fill. La baseline resta taker.
- **Copertura L2 live insufficiente.** L'ultimo execution contract dichiarava 16/16 archivi event-level
  ma circa 6--7 giornate L2, sotto il minimo di 30. Nessun execution model reale autorizzato.
- **Account state appreso.** La decisione di ingresso non è ancora un RL/FQI completo dipendente dalla
  policy. Il Risk Engine gestisce lo stato di portafoglio; non chiamare il sistema "RL end-to-end".
- **Conferma indipendente.** I periodi già osservati sono discovery/contaminati. Anche un esito OOS
  positivo del nuovo protocollo autorizza solo paper/shadow, seguito da holdout futuro sigillato.

## Verifiche concluse sul worktree dell'handoff

- Ruff: verde.
- mypy sul modulo policy: verde.
- test mirati: 66/66 verdi.
- suite globale: 403/403 verdi.
- strict JSON del protocollo: PASS.
- round-trip joblib di `MonotoneAffineCalibrator`: PASS.
- RTX 4060 Ti rilevata: 16 GB totali, circa 14,88 GB liberi al controllo.
- `git diff --check`: verde.
- il primo passaggio globale ha incontrato soltanto l'health-check intermittente Hypothesis
  `too_slow`; il test isolato con il seed registrato è passato e il rerun globale è terminato
  403/403.

## Prossimi passi obbligatori, in questo ordine

### 1. Rileggere e non riscrivere

Leggere integralmente:

1. questo documento;
2. `docs/musca-btc-training-master-plan.md`;
3. il diff dei tre file modificati;
4. `data/reports/musca_btc_policy.preflight.json`;
5. `data/reports/musca_btc_policy.status.json`, ricordando che è stale.

### 2. Completare il contratto esplicito del percorso GPU

- aggiungere un test mirato per `_predict_array` su modello sklearn CPU e XGBoost CUDA;
- verificare equivalenza delle prediction CPU/GPU entro una tolleranza dichiarata;
- conservare il risultato già ottenuto: stderr non contiene più `mismatched devices`;
- non modificare loss, gate o feature durante questo controllo.

### 3. Eseguire i controlli statici

```powershell
.\.venv\Scripts\ruff.exe check src/adaptive_bot/musca_btc_policy.py tests/unit/test_musca_btc_policy.py
.\.venv\Scripts\mypy.exe src/adaptive_bot/musca_btc_policy.py
git diff --check
```

### 4. Eseguire i test mirati e globali

Usare directory temporanee esterne a `data/tmp`, perché vecchie directory pytest hanno ACL Windows
non leggibili:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_musca_btc_policy.py -q `
  --basetemp=C:\Users\david\Documents\Codex\pytest-musca-handoff `
  -o cache_dir=C:\Users\david\Documents\Codex\pytest-cache-musca-handoff

.\.venv\Scripts\python.exe -m pytest -q `
  --basetemp=C:\Users\david\Documents\Codex\pytest-full-handoff `
  -o cache_dir=C:\Users\david\Documents\Codex\pytest-cache-full-handoff
```

### 5. Audit locale finale senza training

- serializzare `PROTOCOL` con `allow_nan=False`;
- round-trip joblib del calibratore e del contratto bundle minimo;
- controllare assenza di `IsotonicRegression`;
- controllare che `local_plan_variants_enabled` sia sempre falso nel walk-forward e bundle;
- controllare che la sola soglia autorizzante sia `0.0`;
- controllare parità della funzione che risolve LONG/SHORT tra walk-forward e bundle/replay;
- verificare che SPA/PBO/DSR entrino nel verdict prima della promozione;
- aggiornare le checklist del master plan con il nuovo numero di test.

### 6. Versionare soltanto dopo tutti i verdi

Esaminare il diff e creare un commit intenzionale. Non aggiungere `data/tmp/`, dataset, log o bundle.
Il commit deve includere soltanto codice, test e documentazione del protocollo.

### 7. Lanciare un solo preflight visibile

Solo se i punti 2--6 sono verdi:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_musca_btc_policy_training.ps1 `
  -PreflightOnly
```

Usare resume soltanto dopo aver verificato che ogni cache abbia lo stesso protocol hash e che il
contenuto sia compatibile. Poiché lo status `e149...` è orfano, non interpretare `--resume` come un
worker vivo. Non lanciare contemporaneamente un secondo monitor/worker.

### 8. Interpretare il preflight senza scorciatoie

Il preflight di due fold è un controllo causale/economico preliminare, non una conferma finale.
Deve mostrare almeno:

- frontiera prima di `DISABLED` per LONG e SHORT;
- zero-margin metrics e threshold diagnostiche separate;
- numero di candidati, selezionati ed eseguiti;
- prediction media vs rendimento realizzato e calibration slope;
- controller scelto e motivo preciso di disabilitazione;
- nessuna variante locale autorizzata dall'oracle;
- nessuna lettura del test nella selection;
- nessun mismatch GPU/CPU e nessun valore JSON non finito.

Soltanto `full_training_authorized=true` autorizza il walk-forward completo a dieci fold. Il full run
può ancora produrre `NO_STABLE_OOS_POLICY`: i test garantiscono coerenza metodologica, non inventano
un edge di mercato.

## File principali da consultare

- `src/adaptive_bot/musca_btc_policy.py` — pipeline, label, modelli, calibrazione, controller,
  walk-forward, replay, gate e report.
- `tests/unit/test_musca_btc_policy.py` — contratto causale/economico e regressioni.
- `docs/musca-btc-training-master-plan.md` — checklist persistente, E-01--E-53 e prove vietate.
- `docs/musca-btc-action-space-audit.md` — diagnosi dello spazio d'azione.
- `docs/musca-btc-daily-portfolio-challenger.md` — tentativi daily/FQI precedenti.
- `docs/musca-v5-fine-tuning-log.md` — FT-000--FT-036; non ripeterli.
- `docs/musca-btc-moe-results.md` e `docs/musca-btc-auto-moe-results.md` — controlli congelati.
- `src/adaptive_bot/musca_btc_moe.py` — parent expert/view e feature.
- `src/adaptive_bot/musca_btc_auto_moe.py` — controllo discovery congelato.
- `scripts/run_musca_btc_policy_training.ps1` — worker unico e monitor visibile.
- `src/adaptive_bot/cli.py` — comandi `musca-btc-policy-train/status`.

## Dati e artefatti da non toccare

- `data/ml/musca_v5/aggtrades/` — archivi Binance ufficiali event-level/1s;
- `data/ml/musca_btc_policy/` — matrici, cataloghi fold-local, registry e checkpoint;
- `data/reports/musca_btc_policy*.json` — report storici e diagnosi;
- `data/models/musca_btc_policy/` — bundle research;
- report e bundle Musca V2/V5;
- report Auto-MoE congelato;
- holdout futuro;
- collector e dashboard live;
- qualunque contenuto di `data/tmp/` non esplicitamente creato dalla nuova chat.

## Cose vietate per il prossimo agente

- promettere o forzare una policy positiva;
- abbassare fee Binance 1x, PF, LCB, drawdown o altri gate;
- scegliere soglie osservando outer test o aggregato dei fold;
- trattare WAIT/FLAT come profitto;
- usare l'oracle delle varianti locali come segnale disponibile;
- aggiungere expert, librerie, seed o trial prima di localizzare un errore misurato;
- reintrodurre isotonic su finestre piccole;
- mescolare calibrazione LONG e SHORT;
- eliminare il lato alternativo prima dei gate separati;
- chiamare `HOLD/REDUCE/TIGHTEN/CLOSE` azioni apprese se sono regole deterministiche;
- inventare book, maker fill, depth, OI, funding o spread storici;
- aprire l'holdout, promuovere live o toccare Musca V2/V5;
- cancellare dati o pulire il worktree senza esaminare ogni target.

## Fonti primarie già usate

- XGBoost parameter/objective documentation:
  <https://xgboost.readthedocs.io/en/stable/parameter.html>
- scikit-learn probability calibration:
  <https://scikit-learn.org/stable/modules/calibration.html>
- Fitted Q Iteration, Ernst et al.:
  <https://jmlr.org/papers/volume6/ernst05a/ernst05a.pdf>
- Double Q-learning, van Hasselt:
  <https://papers.nips.cc/paper/3964-double-q-learning.pdf>
- `arch` SPA/Reality Check:
  <https://bashtage.github.io/arch/multiple-comparison/multiple-comparisons.html>
- Deflated Sharpe Ratio:
  <https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf>
- Binance official public data:
  <https://github.com/binance/binance-public-data/blob/master/README.md?plain=1>
- Binance USD-M commission rate, aggregate trades and funding endpoints:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/User-Commission-Rate>

Le guide Fiscal.ai e il generico `crypto-trading-bot-playbook` di MCP Market sono state valutate e
non sono state integrate: trattano rispettivamente fondamentali societari e un playbook CCXT
generico, non correggono label, selection bias, microstruttura o validazione di questa pipeline.
