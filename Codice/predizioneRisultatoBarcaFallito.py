import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. CONFIGURAZIONE PERCORSI ---
BASE_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data"
EVENTS_DIR = os.path.join(BASE_DIR, "events")

# Sostituisci "1.json" con il nome esatto del file che hai trovato nella cartella 11
MATCHES_FILE = os.path.join(BASE_DIR, "matches", "11", "1.json") 

# --- 2. ESTRAZIONE AUTOMATICA DEI RISULTATI ---
def estrai_risultati_barcellona(filepath):
    print(f"Leggendo le partite dal file: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        matches_data = json.load(f)

    risultati = {}
    for match in matches_data:
        match_id = match['match_id']
        home_team = match['home_team']['home_team_name']
        away_team = match['away_team']['away_team_name']
        home_score = match['home_score']
        away_score = match['away_score']

        # Verifica se il Barcellona gioca in casa o in trasferta e calcola il risultato
        if home_team == 'Barcelona':
            win = 1 if home_score > away_score else 0
            risultati[match_id] = win
        elif away_team == 'Barcelona':
            win = 1 if away_score > home_score else 0
            risultati[match_id] = win

    vittorie = sum(1 for v in risultati.values() if v == 1)
    non_vittorie = len(risultati) - vittorie
    print(f"Trovate {len(risultati)} partite del Barcellona (Vittorie: {vittorie}, Pareggi/Sconfitte: {non_vittorie})")
    
    return risultati

# --- 3. GENERAZIONE HEATMAP ---
def crea_heatmap_from_match(match_id):
    file_path = os.path.join(EVENTS_DIR, f"{match_id}.json")

    if not os.path.exists(file_path):
        return None  # Restituiamo None se il file manca

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    
    # Prendi solo i passaggi della squadra Barcellona
    df = df[(df["type.name"] == "Pass") & (df["team.name"] == "Barcelona")]

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
# Recupero del dizionario {match_id: 1 o 0}
match_dict = estrai_risultati_barcellona(MATCHES_FILE)

X = [] # Feature (Heatmap)
y = [] # Target (Risultato)

print("\nGenerazione delle heatmap in corso... (potrebbe richiedere qualche minuto)")
for match_id, esito in match_dict.items():
    h = crea_heatmap_from_match(match_id)
    if h is not None:
        X.append(h.flatten()) # Appiattimento a 1D
        y.append(esito)

X = np.array(X)
y = np.array(y)

print(f"Dataset pronto: {X.shape[0]} partite valide analizzate.\n")

# --- 5. MACHINE LEARNING (TRAIN & TEST) ---
# Dividiamo i dati: 80% per addestrare il modello, 20% per l'esame finale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Addestramento del Modello ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("--- Valutazione sui dati di Test (non visti dal modello) ---")
predizioni = model.predict(X_test)

# Calcolo accuratezza
accuratezza = accuracy_score(y_test, predizioni)
print(f"ACCURATEZZA DEL MODELLO: {accuratezza * 100:.2f}%\n")

print("Dettaglio (Matrice di Confusione):")
print("[[Veri Negativi (Non-Vitt. indovinate) , Falsi Positivi (Non-Vitt. scambiate per Vitt.)]")
print(" [Falsi Negativi (Vitt. scambiate per Non-Vitt.), Veri Positivi (Vittorie indovinate)]]")
print(confusion_matrix(y_test, predizioni))