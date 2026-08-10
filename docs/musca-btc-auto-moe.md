# Musca BTC Auto-MoE — protocollo preregistrato

Data di congelamento: 2026-08-10

## Domanda

Su Binance USD-M `BTCUSDT`, quali strategie causali complete possono essere scoperte dai dati e
quale di esse, se una, conviene eseguire nello stato corrente dopo costi e rischio?

Il training è composto da due fasi indipendenti. La prima crea gli esperti; la seconda può
soltanto scegliere tra gli esperti congelati e `FLAT`. Il precedente run da 125 componenti resta
immutato ed è una baseline negativa, non una sorgente di conferma.

## Fase 1 — scoperta della libreria

Un generatore di alberi randomizzati GPU apprende direttamente regole condizionali dai 56 input
causali già verificati. Ogni foglia è un candidato esperto completo con:

- regola di attivazione appresa;
- lato `LONG` o `SHORT`;
- orizzonte massimo 1, 5, 15, 60 minuti oppure 6 ore;
- primo e secondo target, stop e trailing ricavati esclusivamente dalle escursioni osservate nel
  fit precedente;
- identità deterministica derivata da protocollo, lato, orizzonte, albero e foglia.

Non esiste un massimo preregistrato di esperti nella libreria. Il generatore valuta alberi in
blocchi e si arresta quando due blocchi consecutivi non aggiungono un esperto nuovo e robusto. Un
tetto di emergenza di 128 alberi per lato e orizzonte limita soltanto il calcolo in caso di mancata
convergenza; non tronca la libreria dopo la selezione.

Un candidato viene congelato soltanto se:

- ha almeno 1.000 attivazioni nel fit e 90 nella validazione;
- expectancy terminale netta positiva sia nel fit sia nella validazione;
- profit factor di validazione almeno 1,05;
- expectancy non negativa con costi 1,5×;
- entrambi i mesi di validazione hanno expectancy positiva;
- non duplica un esperto già scelto e ha Jaccard dei segnali inferiore a 0,90 rispetto agli esperti
  dello stesso lato e orizzonte.

La selezione è greedy per robust score, stabilità mensile e diversità. Il report registra tutti i
candidati valutati, quelli respinti, la saturazione e la prova SPA della libreria rispetto a
`FLAT`. I costi 1,5× sono uno stress diagnostico della discovery; il replay principale usa sempre
i costi reali 1×.

## Fase 2 — gating

Dopo il congelamento, ogni esperto viene applicato soltanto a periodi successivi. Per ogni sua
attivazione viene costruito il percorso economico reale con ingresso nel bucket successivo da
cinque secondi, funding osservato, fee, target parziali, stop, trailing e timeout.

Il gate riceve esclusivamente:

- le feature di regime causali disponibili alla decisione;
- lato, durata e livelli dell'esperto;
- statistiche congelate dell'esperto;
- l'uscita del generatore dell'esperto sullo stato corrente.

Ridge/logistic resta il champion predefinito. XGBoost GPU lo sostituisce soltanto se migliora sulle
stesse righe cronologiche MAE dell'EV, Brier score e regret decisionale. La calibrazione isotonic
usa un mese separato e non cambia il ranking. A ogni timestamp vince l'esperto con EV calibrata
più alta; se la migliore EV non è positiva, l'azione è `FLAT`, con valore neutro zero.

## Cronologia sigillata

- generazione degli alberi: tutto il 2025;
- selezione e congelamento libreria: gennaio–febbraio 2026;
- primo fit del gate: marzo 2026;
- confronto Ridge/XGBoost: aprile 2026;
- refit del champion su marzo–aprile 2026;
- calibrazione: maggio 2026;
- audit storico mensile e aggregato: giugno–luglio 2026;
- holdout futuro intoccabile: dal 10 agosto 2026.

Il purge è pari a sei ore. I mesi usati per provare la libreria non partecipano alla generazione
degli esperti; aprile confronta i modelli ma non modifica la libreria; giugno e luglio non possono
cambiare libreria, champion, calibrazione, soglia `EV > 0` o regole di gestione.

## Economia e gate

Scenario principale: commissione Binance USD-M taker di 4 bps per lato più 1 bp round-trip di
riserva esecutiva, quindi 9 bps complessivi. Funding osservato, capitale simulato 10.000 USDT,
rischio massimo 1% per trade, leva massima 10×, una sola posizione, nessun averaging down.

L'audit richiede:

- almeno 300 trade e almeno 3 trade al giorno;
- expectancy e bootstrap LCB 95% positive;
- profit factor almeno 1,15;
- drawdown massimo 10%;
- maggioranza dei giorni di calendario positiva;
- SPA contro `FLAT` con `p <= 0,05`;
- zero violazioni del budget di rischio.

Qualunque risultato resta `RESEARCH_ONLY`; ordini paper/live rimangono disabilitati finché il
futuro holdout non contiene almeno dieci giorni e 100 trade e supera gli stessi gate. Il risultato
corretto può essere `NO_DISCOVERED_EXPERT_LIBRARY` oppure `NO_DEPLOYABLE_POLICY`.

## Fonti e implementazione

La matrice riusa esclusivamente i dati Binance ufficiali e il contratto causale già testato. Gli
alberi rappresentano regole piecewise-constant apprese dai dati; profondità e foglie minime ne
controllano la complessità. La calibrazione isotonic è usata soltanto con un campione separato.

- [Binance Public Data](https://github.com/binance/binance-public-data/blob/master/README.md)
- [Binance USD-M commission rate](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account#user-commission-rate)
- [XGBoost random forests](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html)
- [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/)
- [scikit-learn decision trees](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn isotonic regression](https://scikit-learn.org/stable/modules/isotonic.html)
- [arch SPA](https://bashtage.github.io/arch/multiple-comparison/multiple-comparison-reference.html)

## Output separati

- `data/ml/musca_btc_auto_moe/`;
- `data/models/musca_btc_auto_moe/research_bundle.joblib`;
- `data/reports/musca_btc_auto_moe.json`;
- `data/reports/musca_btc_auto_moe.status.json`.
