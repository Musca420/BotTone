# Musca V5 / ricerca V5–V10 — report tecnico

Data report: 2026-08-06. Mercato: BTCUSDT perpetual. Exchange di esecuzione previsto:
Binance; Bitunix resta soltanto feed/shadow durante la transizione. Denaro reale: disabilitato.

## Domanda posta

Dato uno stato causale del mercato BTC, il VWAP deve essere usato come benchmark/area di
pullback. La direzione deriva da trend, breakout, spot/perpetual e flusso aggressivo. Si entra
LONG o SHORT soltanto dopo pullback e ripartenza; FLAT vale zero e non viene contato come
profitto. La policy deve produrre EV OOS positivo dopo 8 bps round-trip, senza usare l'holdout
sigillato dall'11 maggio 2026.

## Dati realmente usati

- Binance BTCUSDT perpetual e spot, 1m/5m, 2024-04-15–2026-08-03.
- Binance USD-M BTCUSDT `aggTrades`, gennaio–marzo 2026: circa 180 milioni di trade,
  aggregati in 1.554.990 bucket da 5 secondi. ZIP verificati con checksum ufficiali.
- I parquet a 5 secondi conservano ora open/high/low/close, base volume, quote volume,
  aggressor flow, trade count e `available_at`.
- Nessun book storico è stato inventato. Il dataset pubblico ufficiale Binance distribuisce
  aggTrades/trades/klines; L2 viene quindi raccolto da ora tramite WebSocket ufficiale.

Percorsi dati:

- `data/ml/hybrid_v25/asset=BTCUSDT/bars_5m.parquet`
- `data/ml/hybrid_v25/asset=BTCUSDT/minutes.parquet`
- `data/ml/musca_v5/aggtrades/`
- `data/research/binance_l2/`

## Risultati delle ipotesi scartate

### V5/V6 — eventi frequenti e order flow

Le famiglie DAILY/ROLLING/SWING hanno rendimento lordo medio vicino a zero e netto negativo.
L'uso del percorso a 5 secondi ha eliminato gran parte dell'ambiguità intrabar, senza creare
edge. La V6 a eventi microstrutturali 1m ha prodotto un falso positivo in febbraio
(XGBoost +13,26 bps) e ha perso in marzo (-6,46 bps, PF 0,53). Ridge è rimasta vicina a zero.

Report:

- `data/reports/musca_v5_research.json`
- `data/reports/musca_v5_micro_model.json`
- `data/reports/musca_v6_flow_research.json`

### V7 — aumento della frequenza della V4

Allungare la finestra di ripartenza da 6 a 12 barre ha aumentato gli eventi da circa 116 a
193, ma ha reso negativi 2024 e 2025. Ridurre volume relativo o profondità del pullback non ha
creato un plateau robusto. Report: `data/reports/musca_v7_frequency_audit.json`.

### V8 — esperti multi-orizzonte

Breakout preregistrati a 3/6/12/24/48 barre. Gli esperti 6–48 hanno EV positivo e stress 2×
positivo sia nel 2024 sia nel 2025. L'ensemble scelto senza il 2026 ha ottenuto nel test 2026
pre-holdout:

- 24 trade;
- expectancy +14,50 bps;
- PF 1,43;
- max drawdown 3,22%;
- costi 2×: expectancy +6,50 bps, PF 1,18.

Fallisce il gate di numerosità (24 < 50/100). Gli orizzonti generano in gran parte gli stessi
eventi, quindi non possono essere sommati artificialmente. Report:
`data/reports/musca_v8_multi_horizon.json`.

### V9 — stessa struttura a un minuto

Con VWAP esatto e percorso 5s, 123–194 eventi in tre mesi. Tutti gli orizzonti 5/15/30/60
minuti sono negativi già in gennaio e febbraio (EV tra -6,81 e -15,23 bps). È rumore
microstrutturale, non un problema di FLAT. Report: `data/reports/musca_v9_micro_pullback.json`.

### V10 — ablation delle conferme V8

Il fattoriale completo mostra che ampliare la confluence aumenta gli eventi ma perde nel 2026.
L'unica modifica stabile è rimuovere la seconda conferma spot alla ripartenza; la conferma spot
sull'impulso resta obbligatoria:

| periodo | trade | EV netto | PF | EV stress 2× |
|---|---:|---:|---:|---:|
| 2024 fit | 57 | +9,80 bps | 1,33 | +1,80 bps |
| 2025 validation | 43 | +6,44 bps | 1,17 | -1,56 bps |
| 2026 pre-holdout | 27 | +14,97 bps | 1,40 | +6,97 bps |

Sono 127 trade complessivi, ma soltanto 27 nel test e circa 1,5 trade/settimana. La variante è
stata identificata dopo l'audit, quindi è **discovery**, non conferma indipendente. È autorizzata
solo in shadow futuro. Report: `data/reports/musca_v10_filter_ablation.json`.

## Policy shadow Musca V5

- timeframe decisionale 5m;
- impulso: breakout delle precedenti 24 barre, volume relativo, taker flow e spot/perpetual;
- direzione: trend 1h/4h, EMA e conferma spot/perpetual;
- pullback: zona daily/impulse/swing anchored VWAP;
- restart: price action + taker flow, senza ripetere la conferma spot già richiesta all'impulso;
- spazio residuo minimo: 24 bps, pari a 3× il costo round-trip;
- stop strutturale oltre pullback/AVWAP, mai allargato;
- 50% a 1,5R, protezione costi, trailing swing 15m, timeout 6h;
- rischio simulato 1%, leva cap 10×, una posizione, nessun ordine reale.

Codice e stato:

- `src/adaptive_bot/musca_v4_research.py` — generatore causale condiviso;
- `src/adaptive_bot/musca_v10_filter_ablation.py` — audit completo;
- `src/adaptive_bot/musca_v5_shadow.py` — profilo shadow;
- `src/adaptive_bot/musca_v4.py` — motore condiviso di replay/esecuzione simulata;
- `data/reports/musca_v5_shadow.json` — report corrente;
- `data/research/musca_v5_shadow_state.json` — stato persistente;
- UI: `http://127.0.0.1:8080/?profile=musca-v5`.

## Raccolta L2 Binance

`src/adaptive_bot/binance_l2_collector.py` registra una volta al secondo top-20 bid/ask,
update ID, depth imbalance, buy/sell aggressive quote volume e trade count. Output e stato:

- `data/research/binance_l2/btcusdt_YYYY-MM-DD.jsonl`
- `data/reports/binance_l2_collector.status.json`

Questi dati serviranno a misurare queue/depth, assorbimento, ritiro della liquidità e adverse
selection. Il collector non usa credenziali e non invia ordini.

## Dipendenze

Nessuna nuova dipendenza. Sono stati riutilizzati pandas, NumPy, scikit-learn, XGBoost GPU,
`arch` e `websockets`, già presenti. Aggiungere deep learning ora non risolverebbe l'assenza di
L2 storico e aumenterebbe il rischio di overfitting.

## Verdetto

Musca V5 è un candidato `RESEARCH_ONLY` positivo nei periodi osservati, operativo in shadow ma
non validato per denaro reale. Il limite è la frequenza/numerosità, non FLAT. L'holdout finale non
è stato aperto. La prossima conferma valida deve essere cronologicamente futura e deve includere
L2 realmente osservato; live resta vietato finché i gate OOS e paper non passano.

## Fonti primarie

- Binance public data: https://github.com/binance/binance-public-data/blob/master/README.md
- Binance developer documentation: https://developers.binance.com/en/docs/introduction
- CMT Association, Anchored VWAP (Brian Shannon):
  https://cmtassociation.org/video/interactive-session-anchored-volume-weighted-average-price-analysis-techniques/
- Wen et al., intraday momentum/reversal:
  https://doi.org/10.1016/j.frl.2022.103048
- Deep Learning for Digital Asset Limit Order Books: https://arxiv.org/abs/2010.01241
- Multi-Level Order-Flow Imbalance: https://arxiv.org/abs/1907.06230
