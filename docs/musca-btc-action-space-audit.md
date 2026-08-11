# Musca BTC Binance — audit definitivo dello spazio d'azione

Data: 2026-08-11  
Stato: decisione architetturale congelata prima della prossima modifica al training

## Verdetto

Il vincolo principale del challenger corrente e' stato identificato: il training non genera
veramente piani di trading. Genera esperti di contesto che possono soltanto valutare dieci piani
ereditati, cioe' LONG/SHORT moltiplicato per cinque orizzonti fissi (60, 300, 900, 3.600 e 21.600
secondi).

Target, stop e trailing sono calcolati dinamicamente, ma restano subordinati a una delle dieci
righe. Ogni foglia XGBRF chiamata `expert` e' una partizione dello stato per una coppia gia' fissata
di lato e orizzonte; non propone un nuovo piano. Anche `HOLD`, `CLOSE` e `TIGHTEN_STOP` sono
attualmente registrazioni dell'esito gia' determinato dal simulatore, non azioni scelte da una
policy durante la posizione.

La matrice a dieci piani rimane disponibile soltanto come controllo congelato. Non sara' piu'
l'universo operativo del challenger.

## Evidenza nel codice corrente

- `src/adaptive_bot/musca_btc_moe.py:42` definisce i cinque orizzonti.
- `src/adaptive_bot/musca_btc_moe.py:47` definisce i due lati.
- `src/adaptive_bot/musca_btc_moe.py:949` crea il prodotto cartesiano dei cinque orizzonti e dei
  due lati.
- `src/adaptive_bot/musca_btc_policy.py:1167` trasforma le foglie degli alberi in esperti, ma ogni
  foglia conserva `side` e `horizon_seconds` della riga madre.
- `src/adaptive_bot/musca_btc_policy.py:1666` riduce tutte le righe dello stesso timestamp a una
  sola azione classificata.
- `src/adaptive_bot/musca_btc_policy.py:1769` registra `HOLD` quando la posizione e' gia' aperta.
- `src/adaptive_bot/musca_btc_policy.py:1824` e `:1841` ricavano `TIGHTEN_STOP` e `CLOSE` dal
  percorso gia' simulato; non esiste una stima separata del valore di queste decisioni.

Quindi il nome multi-expert descrive il classificatore del contesto, non un sistema nel quale piu'
esperti compongono e gestiscono un piano.

## Verifica su fonti primarie

### Mixture of Experts

Jacobs, Jordan, Nowlan e Hinton descrivono reti esperte separate e una gating network. Il gate non
richiede che una sola foglia determini tutto il comportamento: la formulazione combina le uscite
degli esperti tramite pesi del gate. La variante sparse MoE usa esplicitamente una combinazione
pesata e sparsa di pochi esperti attivi.

Riferimenti:

- <https://www.cs.toronto.edu/~hinton/absps/jacobs.pdf>
- <https://www.cs.toronto.edu/~hinton/mixex.html>
- <https://www.cs.toronto.edu/~hinton/absps/Outrageously.pdf>

Conclusione applicata: selezionare sempre una singola foglia e copiarne il piano madre non e'
obbligatorio e non sfrutta la proprieta' utile del MoE. Musca deve aggregare le distribuzioni
predette da piu' esperti realmente differenti.

### Azioni parametrizzate

Masson, Ranchod e Konidaris formalizzano azioni composte da un tipo discreto e da parametri
continui specifici di quel tipo. L'agente sceglie sia l'azione sia i suoi parametri. Questa e' la
formulazione coerente con un ordine: `ENTER_LONG` non e' un piano completo; servono anche rischio,
stop, target, trailing, scadenza e gestione parziale.

Riferimento:

- <https://ojs.aaai.org/index.php/AAAI/article/view/10226>

Conclusione applicata: LONG e SHORT sono soltanto due tipi di ingresso. Lo spazio corretto include
azioni di ingresso e gestione con parametri appresi dallo stato.

### Decisioni temporali e trading netto

Le options di Sutton, Precup e Singh descrivono politiche chiuse nel tempo che possono essere
interrotte. Moody e Saffell formulano il trading come controllo stocastico e ottimizzano
direttamente un rendimento aggiustato per il rischio includendo i costi di transazione.

Riferimenti:

- <https://www.sciencedirect.com/science/article/pii/S0004370299000521>
- <https://pubmed.ncbi.nlm.nih.gov/18249919/>

Conclusione applicata: entrata, mantenimento e uscita devono far parte dello stesso problema
sequenziale; il reward e' equity netta, non precisione della direzione o percentuale di trade
vincenti.

### Limite dell'apprendimento offline

Creare piani arbitrari senza copertura produce sovrastima fuori distribuzione. CQL e i risultati
teorici sull'offline RL richiedono pessimismo e copertura sufficiente. Il generatore deve quindi
essere flessibile ma non illimitato: puo' proporre soltanto parametri valutabili causalmente dal
simulatore a un secondo e sostenuti dai dati del fold.

Riferimenti:

- <https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html>
- <https://proceedings.mlr.press/v178/foster22a.html>

## Architettura sostitutiva congelata

### 1. Esperti indipendenti

Le foglie della stessa regressione non saranno piu' chiamate libreria di esperti. La libreria deve
contenere componenti con compiti e target differenti:

1. esperti direzionali: distribuzione dei rendimenti futuri e probabilita' LONG/SHORT per regime;
2. esperti di percorso: distribuzioni MFE, MAE e tempo agli eventi;
3. esperti VWAP/AVWAP: accettazione, rifiuto, ritorno al centro e continuazione;
4. esperti di volatilita': ampiezza e durata economicamente negoziabili;
5. esperti di order flow: aggressione, esaurimento e conferma;
6. esperto costi/funding: costo netto osservabile dell'azione;
7. esperti di gestione: valore condizionato di mantenere, ridurre, chiudere o stringere il rischio.

Ogni esperto produce una distribuzione o una stima calibrata, non un ordine completo copiato da
una tabella.

### 2. Gating multi-esperto

Il gate assegna pesi normalizzati a piu' esperti compatibili con lo stato. La composizione e'
soft/top-k, non winner-takes-all. Il numero di esperti attivi e' determinato dalla copertura e dalla
diversita' OOS; non e' fissato a uno.

Un gruppo di esperti altamente correlati non conta come pluralita' informativa. La selezione deve
misurare correlazione degli errori, specializzazione per regime e calibrazione cronologica.

### 3. Generatore di piani parametrizzati

Il generatore riceve stato e uscite pesate degli esperti. Produce candidati diversi a ogni
decisione.

Tipi discreti senza posizione:

- `WAIT`;
- `ENTER_LONG`;
- `ENTER_SHORT`.

Tipi discreti con posizione:

- `HOLD`;
- `REDUCE`;
- `CLOSE`;
- `TIGHTEN_STOP`;
- `UPDATE_TRAIL`.

Parametri continui o ordinali:

- rischio/notional entro il veto del Risk Engine;
- stop strutturale;
- primo e secondo obiettivo;
- percentuali da chiudere ai target;
- trailing e regola di attivazione;
- durata massima e prossimo tempo di riesame.

`REVERSE` non e' atomica: viene rappresentata come `CLOSE` e, a una decisione successiva, un nuovo
ingresso. Lo stop non puo' mai essere allargato.

I valori non provengono da una nuova griglia fissa. Sono proposte condizionate ottenute dai
quantili di MFE/MAE/tempo e dai segnali combinati degli esperti attivi. Il numero di proposte nasce
dai dati e dalla diversita', non da dieci template.

### 4. Critic separato dal proponente

Il proponente genera piani; un critic distinto stima la distribuzione dell'equity netta per
`(stato, tipo azione, parametri)`. Tutti i candidati vengono etichettati con lo stesso simulatore a
un secondo usato nel replay.

Il critic valuta:

- EV netta con costo Binance 1x e funding;
- probabilita' e tempo di target, stop, trailing e timeout;
- quantili di perdita e rendimento;
- valore di continuazione quando esiste una posizione;
- incertezza e distanza dalla copertura del training.

Un piano fuori copertura viene respinto, non premiato dalla semplice estrapolazione del modello.
Ridge/logistica restano baseline; XGBoost GPU e' challenger.

### 5. Obiettivo

La policy massimizza l'equity netta attesa sulla sequenza, con una sola posizione contemporanea e
piu' trade nella stessa giornata. Sono ammessi trade e giornate in perdita. Il limite giornaliero e'
un veto di rischio, non il target del modello.

Il costo Binance 1x entra nell'obiettivo. Stress 1,5x e 2x restano diagnostici. La leva cambia il
rendimento e il rischio sul capitale, non cancella o raddoppia le commissioni in basis point sul
notional.

## Costruzione causale senza ricerca infinita

Per ciascun outer fold:

1. gli esperti vengono addestrati soltanto sul fit;
2. il gate viene stimato su predizioni OOF del fit;
3. le distribuzioni degli esperti generano parametri supportati dal fit;
4. il simulatore a un secondo etichetta quei piani senza usare il test;
5. proposer e critic vengono selezionati su inner audit purged;
6. calibrazione, decisione e replay usano periodi cronologicamente distinti;
7. il test outer viene letto una volta;
8. tutti i risultati gia' osservati restano discovery e non vengono riciclati come conferma.

L'assunzione necessaria e' che il notional simulato sia sufficientemente piccolo da non modificare
il mercato. Maker fill e queue position restano esclusi finche' non sono osservati.

## Test che separano strategia e modello

1. **Oracle dei piani generati:** se neppure il miglior piano causale dopo costi e' positivo, il
   generatore non offre un action set economico.
2. **Regret del critic:** se l'oracle e' positivo ma il critic non lo ordina correttamente, il
   problema e' predittivo/calibrazione.
3. **Ablation gate:** confronto singolo esperto, media uniforme, soft top-k e gate appreso sulle
   stesse split.
4. **Ablation gestione:** confronto uscita statica contro gestione sequenziale appresa.
5. **Copertura:** nessun parametro fuori dal supporto del fold puo' essere promosso.
6. **Replay identico:** target, stop, trailing, costi e funding devono coincidere tra label e paper.

## Evidenza del run appena terminato

Il run non e' stato bocciato soltanto da un gate severo. Sui 220 trade OOS effettivamente scelti:

- lordo medio: `+1,513 bps`;
- costo round-trip: `8 bps`;
- netto medio: `-6,489 bps`;
- previsione EV media: `+18,398 bps`;
- correlazione previsione-risultato: `0,142`;
- trade vincenti: `53,64%`;
- payoff medio vincita/perdita: `0,752`;
- win rate di pareggio richiesto da quel payoff: `57,09%`.

Solo quattro fold su dieci hanno prodotto trade e tutti e quattro hanno expectancy netta negativa.
Questa evidenza dimostra contemporaneamente un action set troppo rigido e un critic mal calibrato
OOS. Togliere il solo gate finale non avrebbe reso positiva la policy.

## Criterio conclusivo

La prossima implementazione e' valida soltanto se:

- nessun ciclo operativo dipende dal prodotto fisso `2 lati x 5 orizzonti`;
- un test prova che lo stesso stato puo' generare parametri diversi;
- un test prova che piu' esperti contribuiscono allo stesso piano;
- gestione e uscita sono decisioni valutate, non log post-hoc;
- il report separa `NO_ECONOMIC_ACTION_SET`, `NO_PREDICTABLE_EDGE` e
  `NO_STABLE_OOS_POLICY`;
- i risultati OOS restano netti dei costi reali 1x.

Non e' possibile promettere onestamente un risultato positivo. E' invece possibile eliminare
questa limitazione strutturale e rendere il prossimo risultato una prova reale della domanda
corretta, senza cucire i parametri sul test.

## Preflight dell'implementazione

Il 2026-08-11 la sostituzione e' stata verificata sul mese OOF aprile 2025, senza usare il risultato
per scegliere parametri:

- 86.310 piani parametrizzati, uno LONG e uno SHORT per stato;
- 86.310 `plan_id` distinti;
- 6.774 durate distinte tra 60 e 21.600 secondi;
- costo taker round-trip 1x: 8 bps;
- expectancy se si esegue indiscriminatamente ogni LONG: -7,217 bps;
- expectancy se si esegue indiscriminatamente ogni SHORT: -8,339 bps;
- oracle diagnostico tra i due piani: positivo nel 68,26% degli stati;
- oracle medio netto: +9,875 bps.

L'oracle non e' una strategia e non puo' essere tradato: legge il futuro. Il suo unico uso e'
falsificare `NO_ECONOMIC_ACTION_SET`. Questo controllo e' passato; il prossimo run deve stabilire
se il critic riesce a ordinare causalmente i due piani OOS. Il periodo resta contaminato/discovery e
non puo' confermare la policy.
