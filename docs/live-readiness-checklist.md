# Live readiness checklist

Stato Milestone 1: **NON PRONTO**.

- [x] Modelli UTC/Decimal e configurazione fail-closed
- [x] Strategia deterministica senza LLM/ML
- [x] Risk budget e kill switch testati
- [x] Simulated broker e backtest next-event
- [ ] Adapter Alpaca Paper e streaming ufficiale
- [ ] Restart e reconciliation broker verificati
- [ ] Integration, replay e chaos test paper
- [ ] Almeno 100 operazioni paper
- [ ] Almeno 300 operazioni out-of-sample
- [ ] Expectancy OOS positiva netta e profit factor ≥ 1,15
- [ ] Max drawdown ≤ 10%, zero liquidazioni e zero violazioni rischio
- [ ] Nessun ordine duplicato o posizione orfana
- [ ] Maggioranza finestre walk-forward positiva
- [ ] Costi 2x accettabili
- [ ] Revisione manuale di account allowlist, cap e credenziali live

Non modificare automaticamente le soglie per spuntare un gate.
# Meme-specific live gate

- [ ] Subaccount Bitunix e API key dedicati, senza prelievo e con IP allowlist.
- [ ] Almeno 100 operazioni paper e 300 out-of-sample dopo costi.
- [ ] Profit factor almeno 1,15, expectancy positiva e max drawdown massimo 10%.
- [ ] Nessuna violazione rischio, duplicazione, posizione orfana o liquidazione.
- [ ] Costi 2× e gap/slippage stress ancora accettabili.
- [ ] Modello probabilistico, se presente, calibrato e validato; shadow non può inviare ordini.
- [ ] Attivazione live eseguita come modifica manuale, revisionata e separata da questo MVP.
