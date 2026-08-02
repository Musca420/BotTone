# Strategia Adaptive Range

Il center QQQ è il VWAP cumulativo della sessione regular. Per BTCUSDT futures viene usato il rolling
VWAP configurabile a 96 barre su mercato continuo 24/7. ATR e ADX usano smoothing Wilder
con periodo 14; EMA slope usa EMA50 e lookback 5.

Il regime grezzo segue questa precedenza:

1. dato mancante o spread oltre 5 bps: `UNKNOWN`;
2. percentile ATR oltre 90, crescita ATR oltre 50% o movimento a tre barre oltre 3 ATR: `SHOCK`;
3. ADX sotto 20: `RANGE`;
4. ADX sopra 25 con slope e movimento concordi: `TREND_UP`/`TREND_DOWN`;
5. altrimenti `UNKNOWN`.

I cambi ordinari richiedono tre conferme; `SHOCK` è immediato. Non si entra in `UNKNOWN` o
`SHOCK`.

Entrata long: `RANGE`, z ≤ -1,5, spread valido, nessuna posizione/cooldown, finestra sessione e
approvazione del Risk Engine. Stop iniziale 2,5 ATR; target al center noto al momento del segnale;
time stop otto barre. Lo stop può soltanto restare invariato o stringersi.

Non si opera nei primi 15 minuti. Il flatten avviene 15 minuti prima della chiusura effettiva,
incluse early close. Short e uscita parziale sono disabilitati di default. Martingala, pyramiding
in perdita e averaging down non esistono.

Il profilo operativo Bitunix è esclusivamente BTCUSDT perpetual futures, USDT isolated e leva 10×.
Long e short mantengono ingresso adaptive-range, ma usano stop e target simmetrici all'1% del
prezzo d'ingresso: circa −10%/+10% ROE prima dei costi. Il ritorno al center non chiude in anticipo
questo profilo; restano attive le uscite di sicurezza e il time stop di otto barre.
# Meme momentum experts

La strategia meme non usa Adaptive Range ed è long-only. Richiede trend 1h confermato da
EMA20/EMA50, slope e ADX. L'expert principale cerca il breakout delle 20 barre precedenti su 5m con
volume z-score almeno 2; la candela corrente è esclusa dal Donchian e il retest deve arrivare entro
tre barre. Il secondo expert cerca un pullback confermato sulla EMA20 5m con volume sopra la media.

Una candela oltre 2,5 ATR o un movimento a tre barre oltre 4 ATR è `SHOCK`. In tal caso nessun
ingresso è consentito. Stop, uscita 1R, trailing e time stop sono descritti nella configurazione e
si applicano al long. Ogni segnale passa poi scanner, score quantitativi, Market Policy, Luna Low e
risk engine; nessuno di questi stadi può creare un segnale autonomamente.

Il dataset probabilistico contiene soltanto setup deterministici e label `target prima dello stop`,
con stop prevalente quando entrambe le barriere sono toccate nella stessa candela. Il modello futuro
potrà soltanto rifiutare o ridurre il rischio, mai creare un segnale.
