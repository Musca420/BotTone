# MUSCA BTC — challenger sequenziale sull'equity giornaliera

## Domanda congelata

> Dato lo stato del mercato, gli esperti disponibili, il P&L giornaliero, il rischio residuo e
> l'eventuale posizione aperta, quale azione massimizza l'equity netta attesa a fine giornata
> dopo i costi Binance?

La policy positiva `musca_btc_auto_moe` con protocol hash
`a195b75e41bf7bfd40dd2ca23103720cee1ea5dfbc7333cd34dcca4c7171a360` resta invariata. Il
challenger non può sostituirla, aprire l'holdout futuro o abilitare denaro reale.

## Registro anti-ripetizione

Queste linee sono già state eseguite e non possono essere riproposte come nuove:

| Tentativo | Evidenza | Esito | Motivo per non ripeterlo |
|---|---|---|---|
| Ranker giornaliero dei setup V5 | `src/adaptive_bot/musca_v5_binance_daily_ranker.py`, hash `f86d848cb315ba8f44e3eb68b2e0d003511d03d82d0b9004cc9624017cd18442` | `NO_DAILY_RANKING_ALPHA` | A 3,07 trade/giorno EV −10,95 bps e PF 0,40. Raggruppare o ordinare per giorno non crea edge. |
| Policy 30 s multi-orizzonte | FT-033, `data/reports/musca_v5_binance_policy.json` | `NO_SUSTAINABLE_BINANCE_HIGH_FREQUENCY_ALPHA` | L'oracle vede movimento, ma la causalità non lo predice a copertura elevata. |
| MoE generico 125 componenti | `data/reports/musca_btc_moe.json`, hash `73563d1aed16e4f796d18d446dc9033946473429e48f18fe15c1a52ffc193ddd` | non deployable | Circa 2,61 trade/giorno, EV −8,33 bps e PF 0,79. |
| Tutti i minuti / soglie di copertura | FT-003/FT-004 e FT-033 | negativo | I costi trasformano un movimento lordo vicino a zero in perdita netta. |
| Target collegati soltanto alle fee e TP1/runner | FT-004/FT-006 | negativo | Non aggiungono prevedibilità causale. |
| Timing L2, restart 5 s e microstruttura corrente | FT-005–FT-008, FT-029 | negativo o campione insufficiente | Non vanno rilanciati sugli stessi giorni. |
| Penalità lineare per durata sul MoE positivo | diagnostica 2026-08-10 | respinta | Da 0 a 1 bps/ora la frequenza di giugno sale 2,23→2,33 ma l'EV scende 21,31→16,64 bps; privilegiando davvero gli esperti da un'ora l'EV diventa −9,62 bps. |

Ogni nuovo fallimento viene aggiunto qui e nel log append-only
`docs/musca-v5-fine-tuning-log.md` prima di formulare un'altra ipotesi.

## Errore strutturale che resta da testare

Le policy precedenti stimano o ordinano il rendimento del singolo trade. Non attribuiscono al
trade il valore delle occasioni che una posizione lunga impedisce di prendere. Il nuovo target è
quindi un vantaggio d'azione a orizzonte finito:

```text
Q(t, azione) = rendimento netto dell'azione + V(primo stato libero dopo l'uscita)
Q(t, FLAT)   = V(prossima decisione)
advantage    = Q(t, azione) - Q(t, FLAT)
```

`V` è calcolato all'indietro per ogni giornata usando soltanto i label controfattuali del periodo
di training. È un insegnante non negoziabile; in test il modello vede esclusivamente feature
disponibili alla decisione. Questo è diverso sia dalla regressione EV per trade sia dal ranker che
raggruppava azioni per giorno senza modellare il tempo occupato.

## Protocollo preregistrato

- Mercato: soltanto Binance USD-M `BTCUSDT`.
- Dati: `oof_actions.parquet` aprile–dicembre 2025 e `future_actions.parquet` gennaio–luglio
  2026 già prodotti e hashati dal protocollo MoE; nessun download e nessuna rigenerazione.
- Azioni: LONG/SHORT agli orizzonti 1, 5, 15, 60 e 360 minuti, più `FLAT` implicito.
- Posizioni: una sola posizione Binance per volta; sono ammesse più posizioni successive nello
  stesso giorno. Una posizione aperta continua con stop, TP1, TP2 e trailing non allargabile.
- Economia: fee taker 4 bps per lato più 1 bp round-trip di riserva execution, totale 9 bps;
  scenario 2× soltanto come stress separato. Leva massima 10×, rischio 1% per trade e perdita
  giornaliera massima 2%.
- Obiettivo: log-rendimento del portafoglio netto, dimensionato dallo stop. Le azioni che non
  possono terminare entro la giornata UTC sono escluse dal target giornaliero.
- Split: fit 2025-Q2; confronto Ridge/XGBoost 2025-Q3; refit 2025-Q2–Q3; calibrazione 2025-Q4;
  audit discovery 2026-01-01–2026-08-01. Il futuro da 2026-08-10 resta chiuso.
- Champion: Ridge. XGBoost GPU è accettato soltanto se batte Ridge sullo stesso 2025-Q3 in MAE,
  regret decisionale e P&L giornaliero replay.
- Nessuna ricerca di soglie: si entra soltanto quando l'advantage calibrato è maggiore di zero e
  il risk engine approva.
- Promozione: il risultato deve migliorare il controllo miope sugli stessi dati e passare EV>0,
  PF≥1,15, drawdown≤8%, maggioranza dei giorni attivi positiva, stress 2× non negativo e assenza
  di violazioni. Anche in caso positivo il bundle è `RESEARCH_ONLY`.

## Fonti primarie

- Conservative Q-Learning: <https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html>
- Safe Policy Improvement with Baseline Bootstrapping: <https://proceedings.mlr.press/v97/laroche19a.html>
- Doubly Robust off-policy evaluation: <https://proceedings.mlr.press/v48/jiang16.html>
- OPE in ambienti non stazionari: <https://proceedings.mlr.press/v206/liu23d.html>
- XGBoost learning-to-rank: <https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html>
- Binance commission rate: <https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/User-Commission-Rate>
- Binance order book: <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Order-Book>
- Binance mark price: <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price>
- Binance funding history: <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History>

Il principio applicato è conservativo: non si valuta un'azione fuori dal supporto dei dati, non si
usa l'oracle come segnale live e una policy nuova deve battere la baseline sul periodo successivo,
non soltanto adattarsi al training.

## Risultato FT-035 e correzione preregistrata

FT-035, hash
`995bcdb6fbbdef3c7d40d4e39a928f44d0388ac574915257e3860ac3e904e21d`, ha prodotto
`NO_INCREMENTAL_DAILY_Q_ALPHA`. Il target era positivo soltanto nello 0,125–0,127% delle azioni:
la mediana variava da −255 a −313 bps di portafoglio. Il motivo è identificato: `V(t+1)` usava il
miglior risultato futuro realizzato. `FLAT` riceveva quindi informazione onnisciente e non era un
confronto equo con un'azione causale. Ridge e il calibratore hanno correttamente restituito zero
trade. XGBoost non ha battuto Ridge. Questo target è vietato da ripetere.

FT-036 sostituisce esclusivamente il teacher, non dati, costi o azioni:

- Fitted Q Iteration a orizzonte giornaliero finito;
- il valore successivo è la previsione del modello dell'iterazione precedente, non il massimo
  ritorno futuro realizzato;
- `FLAT` ha un proprio modello di attesa, addestrato sulla transizione al minuto successivo;
- quattro iterazioni preregistrate, corrispondenti a una profondità massima di quattro decisioni
  apprese senza imporre quattro trade;
- Ridge resta champion; un solo XGBoost GPU per le stesse quattro iterazioni è challenger;
- scelta modello su 2025-Q3, refit Q2–Q3, calibrazione Q4 e audit 2026;
- il controller riceve P&L e rischio residuo come vincoli causali; una posizione aperta è soltanto
  gestita e non consente un secondo ingresso.

Il report separato è `data/reports/musca_btc_daily_portfolio_fqi.json`. Anche un risultato positivo
resta research-only e non modifica il paper corrente.

## Risultato FT-036

Protocol hash: `1f4cbfc845453c542ace123ba40c7c8b4dc76d7d821c47393524ef8a781bffaa`.
I giorni UTC parziali creati dal purge al termine degli split sono esclusi.

Sul model-selection 2025-Q3:

| Modello | TD MAE | Trade/giorno | EV netto | PF | Drawdown | Daily return medio |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 49,76 bps | 1,01 | −11,21 bps | 0,63 | 13,70% | −0,140% |
| XGBoost CUDA | 43,00 bps | 2,63 | −14,18 bps | 0,52 | 40,29% | −0,518% |

XGBoost riduce l'errore di Bellman ma peggiora il risultato giornaliero, quindi non supera la
regola champion. Ridge viene rifittata su Q2–Q3. La calibrazione Q4 applicata invariata al 2026 non
produce azioni con advantage positivo: audit a zero trade, `NO_INCREMENTAL_FQI_DAILY_ALPHA`.

La conclusione tecnica è circoscritta: cambiare l'obiettivo da EV per trade a P&L giornaliero non
trasforma gli esperti generici frequenti in azioni economicamente valide. Sono ammesse perdite
realizzate, ma un controller razionale non deve accettare azioni con perdita *attesa*. I 29 esperti
filtrati della policy positiva restano necessari; il test lineare FT-034 ha già mostrato che
privilegiare i suoi esperti da un'ora riduce l'edge. Rimuovere ancora i gate o sovrapporre più
posizioni ripeterebbe esperimenti negativi o aumenterebbe il rischio, non risolverebbe la causalità.

Nessun paper bundle è stato sostituito, l'holdout dal 10 agosto resta chiuso e il denaro reale resta
disabilitato. Il bundle FT-036 conserva i modelli soltanto come evidenza `RESEARCH_ONLY`, con ordini
paper e live esplicitamente disabilitati.
