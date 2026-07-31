# Modello di rischio

QQQ opera senza leva, long-only, una posizione e rischio nominale massimo dello 0,25% equity.

```text
risk_budget = equity * risk_per_trade
risk_per_unit = abs(entry - stop) * point_value + estimated_round_trip_cost
quantity = floor_to_lot(min(risk_budget / risk_per_unit,
                            buying_power / entry,
                            hard_notional_cap / entry))
```

Dopo il rounding vengono ricalcolati rischio, notional, minimum quantity e minimum notional.
Qualsiasi superamento o dato mancante rifiuta l'ordine. Equity zero non produce quantità.

Limiti iniziali: 1% perdita giornaliera, 2,5% settimanale, 8% drawdown, una posizione/correlazione,
tre perdite consecutive e cooldown di otto barre. Il cap indipendente è 25.000 USD nella config
simulata.

Il kill switch è latched: impedisce entrate e ammette soltanto cancel, ordini reduce-only e stop
protettivi. Registra causa, timestamp e policy di flatten; il reset richiede attore e motivazione.

Per strumenti a leva il motore rifiuta l'operazione finché non riceve liquidazione broker e stima
indipendente concordi. Il buffer minimo è tre volte la distanza dello stop dopo fee, funding,
slippage e altri buffer. Non esiste una formula universale hard-coded.
