# Modello di rischio

Ogni strumento opera con una sola posizione e rischio nominale massimo dell'1% dell'equity.

```text
risk_budget = equity * risk_per_trade
risk_per_unit = abs(entry - stop) * point_value + estimated_round_trip_cost
quantity = floor_to_lot(min(risk_budget / risk_per_unit,
                            buying_power / entry,
                            hard_notional_cap / entry))
```

Dopo il rounding vengono ricalcolati rischio, notional, minimum quantity e minimum notional.
Qualsiasi superamento o dato mancante rifiuta l'ordine. Equity zero non produce quantità.

Limiti iniziali: 2% perdita giornaliera, 10% settimanale, 8% drawdown, una posizione/correlazione,
tre perdite consecutive e cooldown di otto barre. Il cap indipendente è 25.000 USD nella config
simulata.

Il kill switch è latched: impedisce entrate e ammette soltanto cancel, ordini reduce-only e stop
protettivi. Registra causa, timestamp e policy di flatten; il reset richiede attore e motivazione.
Il raggiungimento del limite giornaliero o settimanale blocca le nuove entrate senza liquidare
automaticamente una posizione già protetta. Il riferimento di equity viene persistito: un riavvio
non azzera il conteggio. Le altre cause mantengono la propria policy di uscita d'emergenza.

Per strumenti a leva il motore rifiuta l'operazione finché non riceve liquidazione broker e stima
indipendente concordi. Il buffer minimo è tre volte la distanza dello stop dopo fee, funding,
slippage e altri buffer. Non esiste una formula universale hard-coded.

La sola eccezione è il backtest simulato BTCUSDT: leva 10×, esposizione massima 20% dell'equity,
margine teorico 2% e cap margine 10%. Stop e target distano entrambi l'1% dal prezzo d'ingresso,
equivalenti a circa −10%/+10% ROE prima dei costi. Il percorso reale resta bloccato senza prezzo di
liquidazione e margine restituiti dal broker.
# Profilo meme separato

Il paper meme parte da 100 USDT e usa rischio base 0,50% (0,50 USDT), con cap assoluto 0,60%.
La perdita massima è 1,5% giornaliera, 4% settimanale e 8% di drawdown. È consentita
una sola posizione nell'intero universo e dopo tre perdite scatta un cooldown di otto barre.
Il notional è calcolato dallo stop e dai costi, poi limitato a 40 USDT e 20% di margine. Fra 1× e
2× viene scelta la leva minima sufficiente; la leva non aumenta mai il budget monetario. Dopo il
rounding il rischio viene ricalcolato e qualsiasi violazione rifiuta l'ordine.

Il profilo è isolato dal bot BTC. Il live resta disabilitato e richiederà subaccount, chiavi e gate
separati oltre alla modifica versionata della configurazione.

Market Policy, Luna Low e score quantitativi sono filtri addizionali: non possono creare segnali né
aumentare rischio/leva. Dati assenti non vengono convertiti in zero; il risultato è un blocco o
`UNKNOWN`. In `paper_bootstrap` la stima probabilistica è mostrata ma non calibrata e resta shadow.

Il funding Bitunix è letto come frazione decimale per intervallo di settlement e normalizzato a otto
ore (`rate * 8 / interval_hours`). Ad esempio `0.005` ogni quattro ore equivale a `1% / 8h`.
Funding mancante, non finito o con intervallo invalido diventa `UNKNOWN` e blocca il contratto. Nel
profilo long-only un valore positivo è un costo long; un valore negativo oltre soglia resta bloccato
come dislocazione di mercato, non come costo short.
