# Strategia Adaptive Range

Il center QQQ è il VWAP cumulativo della sessione regular. Per crypto è disponibile un rolling
VWAP configurabile a 96 barre, non usato in questa milestone. ATR e ADX usano smoothing Wilder
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
