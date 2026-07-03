import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

EVENTS_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data\events"

# I tuoi match
selected_matches = [69243, 69276, 69328, 266015]

# -------- 1. FUNZIONE HEATMAP (Invariata) --------
def crea_heatmap_from_match(match_id):
    file_path = os.path.join(EVENTS_DIR, f"{match_id}.json")

    if not os.path.exists(file_path):
        return np.zeros((25, 25))

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    df = df[df["type.name"] == "Pass"]

    if len(df) == 0:
        return np.zeros((25, 25))

    df["x"] = df["location"].apply(lambda x: x[0])
    df["y"] = df["location"].apply(lambda x: x[1])

    heatmap, _, _ = np.histogram2d(
        df["x"], df["y"],
        bins=[25, 25],
        range=[[0, 120], [0, 80]]
    )
    return heatmap

# -------- 2. PREPARAZIONE DATI PER IL MACHINE LEARNING --------
print("Estrazione delle feature (Heatmap in array 1D)...")

X = [] # Matrice delle features (le heatmap appiattite)
y = [] # Array dei target (i risultati)

# ⚠️ ATTENZIONE: Per addestrare il modello ti servono i risultati REALI.
# Qui creo un dizionario fittizio per farti girare il codice. 
# 1 = Vittoria, 0 = Pareggio/Sconfitta
risultati_reali = {
    69243: 1, 
    69276: 1, 
    69328: 1, 
    266015: 1
}

for mid in selected_matches:
    h = crea_heatmap_from_match(mid)
    
    # FLATTEN: Trasforma la matrice 25x25 in un array di 625 elementi
    X.append(h.flatten())
    y.append(risultati_reali[mid])

X = np.array(X)
y = np.array(y)

# -------- 3. ADDESTRAMENTO E PREDIZIONE --------
print("\n--- Modello Predittivo ---")
if len(selected_matches) < 10:
    print("Nota: Stai usando solo 4 partite. Per la tesi dovrai caricare molte più partite (es. 50-100) affinché il modello sia statisticamente valido.\n")

# Inizializziamo il classificatore Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Siccome abbiamo solo 4 dati, per questo test addestriamo e facciamo predizioni sugli stessi dati.
# Nella tesi finale dovrai usare la funzione train_test_split per dividere i dati in Train e Test!
model.fit(X, y)

# Prediciamo i risultati in base alle heatmap
predizioni = model.predict(X)
probabilita = model.predict_proba(X)

for i, mid in enumerate(selected_matches):
    esito_reale = "Vittoria" if y[i] == 1 else "Non Vittoria"
    esito_predetto = "Vittoria" if predizioni[i] == 1 else "Non Vittoria"
    
    # La probabilità calcolata dal modello per la classe predetta
    confidenza = np.max(probabilita[i]) * 100
    
    print(f"Match {mid} | Reale: {esito_reale} | Predizione: {esito_predetto} (Confidenza: {confidenza:.1f}%)")