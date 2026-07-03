# Tesi - Analisi predittiva FC Barcelona

Questo repository contiene script Python per analizzare dati di partite e passaggi dal dataset `open-data-master`, con un focus sulla previsione delle vittorie del FC Barcelona in base alla distribuzione dei passaggi nelle 9 macro-zone del campo.

## Struttura del repository

- `Codice/`: contiene tutti gli script Python utilizzati per l'elaborazione dei dati, il training del modello e la visualizzazione dei risultati.
- `Immagini/`: cartella per salvare grafici e immagini generate dalla tesi.
- `Pdf/`: eventuali documenti PDF correlati alla tesi.

## File principale da leggere

- `Codice/predizioneBarca526PartiteConMacroZone.py`

Questo script fa quanto segue:

1. **Estrae i risultati delle partite**
   - Scorre i file JSON in `BASE_DIR/matches/11`
   - Cerca le partite in cui gioca il `Barcelona`
   - Calcola se il Barcelona ha vinto la partita (1) o no (0)

2. **Genera la rappresentazione delle macro-zone**
   - Per ogni partita del Barcelona carica il file eventi corrispondente da `BASE_DIR/events`
   - Filtra solo i passaggi di tipo `Pass` eseguiti dal team `Barcelona`
   - Estrae le coordinate `x` e `y` del passaggio
   - Raggruppa i passaggi in una griglia 3x3 per ottenere 9 macro-zone
   - Restituisce un vettore di 9 valori che rappresentano la densità dei passaggi in ogni macro-zona

3. **Costruisce il dataset**
   - Assegna a ogni partita il vettore delle macro-zone come `X`
   - Usa il risultato della partita come target `y`

4. **Addestra un modello di machine learning**
   - Usa `RandomForestClassifier` con `class_weight='balanced'`
   - Divide i dati in training e test con `train_test_split`
   - Calcola l'accuratezza sui dati di test

5. **Stampa le metriche di valutazione**
   - `accuracy_score`
   - `confusion_matrix`
   - `classification_report`
   - interpretazione dettagliata della matrice di confusione

6. **Mostra l'importanza delle macro-zone**
   - Elenca le 9 zone del campo ordinate per importanza del modello
   - Produce un grafico a barre salvato come `Importanza_MacroZone_Barcellona.png`

7. **Opzionale**
   - Se `seaborn` è installato, salva anche una heatmap della matrice di confusione come `Matrice_Confusione_Barcellona.png`

## Come leggere il risultato

- `ACCURATEZZA DEL MODELLO`: percentuale di partite correttamente previste nel set di test.
- `Matrice di Confusione`: mostra quanti match veri e propri sono stati classificati correttamente o erroneamente.
  - `Veri Negativi`: non vittorie previste correttamente
  - `Falsi Positivi`: non vittorie previste come vittorie
  - `Falsi Negativi`: vittorie previste come non vittorie
  - `Veri Positivi`: vittorie previste correttamente
- `Report di Classificazione`: contiene precision, recall e f1-score per ogni classe.
- `Importanza delle zone`: indica quali macro-zone dei passaggi influenzano maggiormente la previsione della vittoria.
- Grafici salvati: `Importanza_MacroZone_Barcellona.png` e, se disponibile, `Matrice_Confusione_Barcellona.png`.

## Dettagli utili per la tesi

- Le 9 macro-zone sono:
  1. Difesa Fascia Sx
  2. Difesa Centro
  3. Difesa Fascia Dx
  4. Centrocampo Fascia Sx
  5. Centrocampo Centro
  6. Centrocampo Fascia Dx
  7. Attacco Fascia Sx
  8. Attacco Centro
  9. Attacco Fascia Dx

- Il modello usa la distribuzione dei passaggi di `Barcelona` come input.
- Il target è la vittoria del Barcelona, non il punteggio esatto.

## Esecuzione

Apri un terminale nella cartella `Codice` e lancia:

```bash
python predizioneBarca526PartiteConMacroZone.py
```

Assicurati di avere installati i pacchetti Python necessari:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Se vuoi la heatmap della matrice di confusione, installa anche:

```bash
pip install seaborn
```

## Note finali

- Controlla che `BASE_DIR` punti al percorso corretto dei dati JSON.
- Se i file JSON non esistono o i nomi delle cartelle cambiano, aggiorna `BASE_DIR`, `EVENTS_DIR` e `MATCHES_FOLDER` nel file.
- La funzione `crea_macrozone_from_match` restituisce `None` se non trova dati validi per una partita.
