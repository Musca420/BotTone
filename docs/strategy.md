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

La strategia meme non usa Adaptive Range. Classifica `bullish expansion`, `euphoric pump`,
`sideways`, `distribution`, `bearish expansion`, `panic crash`, `illiquid` e `unknown`. Long
breakout/pullback sono ammessi nei regimi rialzisti; gli short breakdown, con rischio ridotto del
25%, nei regimi distribution/ribassisti. Il momentum minimo è 1,5 ATR, il volume z-score minimo è
zero e il retest deve arrivare entro tre barre.

Una candela oltre 2,5 ATR o un movimento a tre barre oltre 4 ATR è `SHOCK`. In tal caso nessun
ingresso è consentito. La gestione riduce il 25% a 1R long/0,75R short, un altro 25% al target e
applica trailing alla quota residua; lo short ha un time stop più breve. Ogni segnale passa poi
scanner hard/soft, score quantitativi, Market Policy, Luna Low e
risk engine; nessuno di questi stadi può creare un segnale autonomamente.

Il dataset probabilistico contiene soltanto setup deterministici e label `target prima dello stop`,
con stop prevalente quando entrambe le barriere sono toccate nella stessa candela. Il modello futuro
potrà soltanto rifiutare o ridurre il rischio, mai creare un segnale.

## Adaptive Range Scanner meme

Ogni ora il collector scarica le ultime 200 candele da 5 minuti per tutte le perpetual meme USDT
presenti nell'intersezione Bitunix/CoinGecko. Calcola ATR(14), ADX(14), rolling VWAP(96), distanza
normalizzata `z` e filtri shock; privilegia `ADX < 20` e seleziona dieci coppie per lo streaming
profondo. Volume, spread, depth, funding e manipulation restano filtri operativi indipendenti: una
coppia può essere monitorata senza essere autorizzata al trading.

L'expert Adaptive Range entra soltanto in regime `SIDEWAYS` con `abs(z) >= 2`, target al rolling
VWAP e stop a 1,25 ATR. Il rapporto reward/risk deve essere almeno 1,6. Gli expert momentum restano
separati e non possono trasformare un regime laterale in un segnale trend.
