import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. CONFIGURAZIONE PERCORSI ---
BASE_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data"
EVENTS_DIR = os.path.join(BASE_DIR, "events")

# Ora puntiamo all'INTERA CARTELLA, non più al singolo file
MATCHES_FOLDER = os.path.join(BASE_DIR, "matches", "11") 

# --- 2. ESTRAZIONE MASSIVA DEI RISULTATI ---
def estrai_risultati_barcellona_massivo(folder_path):
    print(f"Scansione di tutti i file JSON nella cartella: {folder_path} ...")
    
    # Trova tutti i file che finiscono in .json in quella cartella
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    risultati = {}
    
    for filepath in json_files:
        with open(filepath, encoding="utf-8") as f:
            matches_data = json.load(f)

        for match in matches_data:
            match_id = match['match_id']
            home_team = match['home_team']['home_team_name']
            away_team = match['away_team']['away_team_name']
            home_score = match['home_score']
            away_score = match['away_score']

            if home_team == 'Barcelona':
                win = 1 if home_score > away_score else 0
                risultati[match_id] = win
            elif away_team == 'Barcelona':
                win = 1 if away_score > home_score else 0
                risultati[match_id] = win

    vittorie = sum(1 for v in risultati.values() if v == 1)
    non_vittorie = len(risultati) - vittorie
    print(f"Trovate {len(risultati)} partite TOTALI del Barcellona.")
    print(f"Bilancio Storico -> Vittorie: {vittorie} | Pareggi/Sconfitte: {non_vittorie}\n")
    
    return risultati

# --- 3. GENERAZIONE HEATMAP ---
def crea_heatmap_from_match(match_id):
    file_path = os.path.join(EVENTS_DIR, f"{match_id}.json")

    if not os.path.exists(file_path):
        return None 

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    
    # Filtro rigoroso: solo i passaggi e solo del Barcellona
    if "type.name" in df.columns and "team.name" in df.columns:
        df = df[(df["type.name"] == "Pass") & (df["team.name"] == "Barcelona")]
    else:
        return None

    if len(df) == 0:
        return None

    df["x"] = df["location"].apply(lambda x: x[0] if isinstance(x, list) else np.nan)
    df["y"] = df["location"].apply(lambda x: x[1] if isinstance(x, list) else np.nan)
    df = df.dropna(subset=['x', 'y'])

    heatmap, _, _ = np.histogram2d(
        df["x"], df["y"],
        bins=[25, 25],
        range=[[0, 120], [0, 80]]
    )
    return heatmap

# --- 4. PIPELINE PRINCIPALE ---
match_dict = estrai_risultati_barcellona_massivo(MATCHES_FOLDER)

X = [] # Feature (Heatmap)
y = [] # Target (Risultato)

print("Generazione delle heatmap in corso...")
print("ATTENZIONE: L'estrazione di centinaia di heatmap impiegherà alcuni minuti. Non chiudere il programma...")

partite_processate = 0
for match_id, esito in match_dict.items():
    h = crea_heatmap_from_match(match_id)
    if h is not None:
        X.append(h.flatten()) 
        y.append(esito)
    
    partite_processate += 1
    # Stampa un aggiornamento ogni 50 partite per farti capire che sta lavorando
    if partite_processate % 50 == 0:
        print(f"... processate {partite_processate}/{len(match_dict)} partite ...")

X = np.array(X)
y = np.array(y)

print(f"\nDataset finale pronto: {X.shape[0]} heatmap estratte con successo.\n")

# --- 5. MACHINE LEARNING AVANZATO ---
# Divisione in Train (80%) e Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Addestramento del Modello (con bilanciamento delle classi) ---")
# Il parametro class_weight="balanced" forza il modello a studiare bene anche le sconfitte
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

print("--- Valutazione sui dati di Test (non visti dal modello) ---")
predizioni = model.predict(X_test)

accuratezza = accuracy_score(y_test, predizioni)
print(f"ACCURATEZZA DEL MODELLO: {accuratezza * 100:.2f}%\n")

print("Dettaglio (Matrice di Confusione):")
print("[[Veri Negativi (Non-Vitt. indovinate) , Falsi Positivi (Non-Vitt. scambiate per Vitt.)]")
print(" [Falsi Negativi (Vitt. scambiate per Non-Vitt.), Veri Positivi (Vittorie indovinate)]]")
print(confusion_matrix(y_test, predizioni))

print("\nReport di Classificazione Completo:")
print(classification_report(y_test, predizioni, target_names=["Non-Vittoria (0)", "Vittoria (1)"]))