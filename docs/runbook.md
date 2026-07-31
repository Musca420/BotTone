# Runbook operativo

## Bot bloccato

Non riavviare alla cieca. Salvare log/database, leggere il kill switch, verificare dati, account,
ordini e posizioni. Resettare manualmente soltanto dopo avere rimosso la causa e registrato attore
e motivazione.

## Ordine sconosciuto

Bloccare nuove entry. Cercare il client order ID sul broker prima di qualsiasi retry. Se lo stato
resta ambiguo, mantenere il kill switch e gestire manualmente l'esposizione.

## Posizione non riconciliata

Considerare il broker fonte autorevole, bloccare entry, cancellare ordini non protettivi e
verificare quantità, lato, average price e stop. Non correggere automaticamente senza audit.
Per Alpaca Paper eseguire `uv run adaptive-bot reconcile --config configs/alpaca_qqq_paper.yaml`;
un exit code `2` mantiene il sistema bloccato e richiede verifica manuale nel portale Alpaca.

## API down o rate limiting

Attivare il kill switch REST/WebSocket/429. Non inviare retry ciechi. Preservare gli stop già al
broker e attendere conferma esplicita del recupero prima del reset manuale.

## Chiave compromessa

Disabilitare immediatamente la chiave dal broker, fermare il bot, verificare ordini/posizioni,
ruotare le credenziali e controllare i log. Le nuove chiavi devono essere paper/live separate,
senza prelievo e IP-limited quando disponibile.

## Stop protettivo mancante

Bloccare entry e cancellare ordini non protettivi. Ripristinare uno stop reduce-only se lo stato è
affidabile; altrimenti applicare la policy di flatten manuale. Non riattivare finché stop e
posizione non sono riconciliati.

## Perdita oltre soglia

Lasciare latched il kill switch, impedire nuovi ordini e applicare la policy configurata alla
posizione. Non alzare le soglie per riattivare il sistema.

## Database bloccato o corrotto

Fermare le scritture, copiare database e file WAL/SHM, verificare `PRAGMA integrity_check` su una
copia e ricostruire lo stato dal broker/eventi. Non eliminare il database originale durante
l'incidente.

## Disattivazione definitiva

Fermare il processo, attivare il kill switch, cancellare ordini non protettivi, chiudere o
trasferire la gestione della posizione secondo policy, revocare le chiavi e archiviare database,
configurazione, commit Git e log dell'ultima sessione.
