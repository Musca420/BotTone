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

Il paper meme usa cap 1% per trade, 3% giornaliero, 10% settimanale e 15% drawdown. È consentita
una sola posizione nell'intero universo e dopo tre perdite scatta un cooldown di otto barre.
Il notional è prima calcolato dallo stop e dai costi, poi limitato al 10% di margine. Fra 2×, 3× e
5× viene scelta la leva minima sufficiente entro il tetto manuale; la leva non aumenta mai il budget
monetario. Quantità minima o rounding incompatibili causano rifiuto.

Il profilo è isolato dal bot BTC. Il live resta disabilitato e richiederà subaccount, chiavi e gate
separati oltre alla modifica versionata della configurazione.
