# Backtesting

Il backtester valida il dataset prima di creare eventi. Controlla timezone, ordine e unicità dei
timestamp, gap RTH, OHLC, volume, prezzi, corporate action dichiarate e salti anomali. La soglia
MVP è strict: un errore critico interrompe il run.

Sequenza per ogni barra:

1. riempire ordini creati su barre precedenti al prossimo open disponibile;
2. applicare prima entry, poi stop protettivo, poi target/uscite;
3. aggiornare account e limiti;
4. calcolare indicatori e segnale solo dopo la chiusura.

Con soli OHLCV, bid/ask sono sintetizzati dallo spread configurato. Market fill iniziale:
prossimo open più mezzo spread e slippage avverso, arrotondato al tick. La quantità è limitata
dalla partecipazione al volume e può essere parziale. Un limit non attraversato resta inevaso.
Un gap oltre stop viene eseguito al prezzo peggiore; stop e target nella stessa barra significano
stop prima del target.

Default: spread 2 bps, slippage 1 bp, commissione 0,005 USD/unità, partecipazione massima 10%.
Funding è modellabile tramite eventi/rate deterministici, ma non applicato a QQQ.

Il JSON della Milestone 1 è diagnostico. Walk-forward, stress, Monte Carlo e report HTML completi
arrivano nella Milestone 3.
