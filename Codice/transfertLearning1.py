import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pandas as pd
import numpy as np
import glob
import random
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURAZIONE PERCORSI ---
BASE_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data"
EVENTS_DIR = os.path.join(BASE_DIR, "events")
MATCHES_DIR = os.path.join(BASE_DIR, "matches") 
SQUADRA_TARGET = "Barcelona"

# --- 2. ESTRAZIONE E BILANCIAMENTO PERFETTO (UNDERSAMPLING) ---
def estrai_risultati_bilanciati(matches_dir, team_name):
    print("Scansione partite in corso...")
    vittorie, pareggi, sconfitte = [], [], []
    json_files = glob.glob(os.path.join(matches_dir, "**", "*.json"), recursive=True)
    
    for filepath in json_files:
        with open(filepath, encoding="utf-8") as f:
            matches_data = json.load(f)
            
        for match in matches_data:
            match_id = match['match_id']
            home_team = match['home_team']['home_team_name']
            away_team = match['away_team']['away_team_name']
            home_score = match['home_score']
            away_score = match['away_score']
            
            if home_team == team_name:
                if home_score > away_score: vittorie.append(match_id)
                elif home_score == away_score: pareggi.append(match_id)
                else: sconfitte.append(match_id)
            elif away_team == team_name:
                if away_score > home_score: vittorie.append(match_id)
                elif away_score == home_score: pareggi.append(match_id)
                else: sconfitte.append(match_id)
                
    # Troviamo il numero minimo tra le tre classi
    min_len = min(len(vittorie), len(pareggi), len(sconfitte))
    print(f"Partite totali trovate -> Vittorie: {len(vittorie)} | Pareggi: {len(pareggi)} | Sconfitte: {len(sconfitte)}")
    print(f"Bilanciamento: Seleziono esattamente {min_len} partite per ogni categoria.")
    
    # Selezioniamo casualmente lo stesso numero di partite per ogni classe
    random.seed(42)
    vittorie_bilanciate = random.sample(vittorie, min_len)
    pareggi_bilanciati = random.sample(pareggi, min_len)
    sconfitte_bilanciate = random.sample(sconfitte, min_len)
    
    risultati_finali = {}
    for m in vittorie_bilanciate: risultati_finali[m] = 2
    for m in pareggi_bilanciati: risultati_finali[m] = 1
    for m in sconfitte_bilanciate: risultati_finali[m] = 0
    
    return risultati_finali

# --- 3. GENERAZIONE HEATMAP SQUADRA ---
def crea_heatmap_squadra(match_id, team_name):
    file_path = os.path.join(EVENTS_DIR, f"{match_id}.json")
    if not os.path.exists(file_path): return None

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    
    if "type.name" not in df.columns or "team.name" not in df.columns: return None
    df_team = df[(df["type.name"] == "Pass") & (df["team.name"] == team_name)].copy()
    if len(df_team) == 0: return None

    df_team["x"] = df_team["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    df_team["y"] = df_team["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
    df_team = df_team.dropna(subset=['x', 'y'])

    heatmap, _, _ = np.histogram2d(df_team["x"], df_team["y"], bins=[50, 50], range=[[0, 120], [0, 80]])
    return heatmap

# --- 4. PIPELINE E TRANSFER LEARNING ---
match_dict = estrai_risultati_bilanciati(MATCHES_DIR, SQUADRA_TARGET)
X_immagini, y_esiti = [], []

print("\nEstrazione heatmap in corso...")
for match_id, esito in match_dict.items():
    mappa = crea_heatmap_squadra(match_id, SQUADRA_TARGET)
    if mappa is not None:
        X_immagini.append(mappa)
        y_esiti.append(esito)

X_immagini = np.array(X_immagini)
y_esiti = np.array(y_esiti)

# Adattamento per VGG16 (3 canali)
X_cnn = X_immagini.reshape(-1, 50, 50, 1)
X_cnn_rgb = np.repeat(X_cnn, 3, axis=3) 

X_train, X_test, y_train, y_test = train_test_split(X_cnn_rgb, y_esiti, test_size=0.2, random_state=42)

print("\nScaricamento VGG16 e configurazione Transfer Learning...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(50, 50, 3))
for layer in base_model.layers: layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(3, activation='softmax')(x)

model_transfer = Model(inputs=base_model.input, outputs=output_layer)
model_transfer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nInizio addestramento della Nuova Testa...")
storia = model_transfer.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose=1)