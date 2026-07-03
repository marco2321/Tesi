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
from tensorflow.keras.layers import Dense, Flatten, Dropout, RandomFlip, RandomTranslation, RandomRotation
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURAZIONE PERCORSI ---
BASE_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data"
EVENTS_DIR = os.path.join(BASE_DIR, "events")
MATCHES_DIR = os.path.join(BASE_DIR, "matches") 
SQUADRA_TARGET = "Barcelona"

# --- 2. ESTRAZIONE E BILANCIAMENTO ---
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
                
    min_len = min(len(vittorie), len(pareggi), len(sconfitte))
    print(f"Bilanciamento in corso: seleziono {min_len} partite per categoria (Totale: {min_len * 3}).\n")
    
    random.seed(42)
    vittorie_bilanciate = random.sample(vittorie, min_len)
    pareggi_bilanciate = random.sample(pareggi, min_len)
    sconfitte_bilanciate = random.sample(sconfitte, min_len)
    
    risultati_finali = {}
    for m in vittorie_bilanciate: risultati_finali[m] = 2
    for m in pareggi_bilanciate: risultati_finali[m] = 1
    for m in sconfitte_bilanciate: risultati_finali[m] = 0
    
    return risultati_finali

# --- 3. GENERAZIONE MAPPA TATTICA (RGB) ---
def crea_heatmap_rgb(match_id, team_name):
    file_path = os.path.join(EVENTS_DIR, f"{match_id}.json")
    if not os.path.exists(file_path): return None

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    
    if "type.name" not in df.columns or "team.name" not in df.columns: return None
    df_team = df[df["team.name"] == team_name].copy()
    
    df_team["x"] = df_team["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    df_team["y"] = df_team["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
    df_team = df_team.dropna(subset=['x', 'y'])

    is_shot = df_team["type.name"] == "Shot"
    is_key_pass = df_team.get("pass.shot_assist", pd.Series(False, index=df_team.index)) == True
    df_red = df_team[is_shot | is_key_pass]
    df_green = df_team[df_team["type.name"] == "Pass"]
    df_blue = df_team[df_team["type.name"].isin(["Duel", "Interception", "Block", "Clearance"])]

    h_red, _, _ = np.histogram2d(df_red["x"], df_red["y"], bins=[50, 50], range=[[0, 120], [0, 80]])
    h_green, _, _ = np.histogram2d(df_green["x"], df_green["y"], bins=[50, 50], range=[[0, 120], [0, 80]])
    h_blue, _, _ = np.histogram2d(df_blue["x"], df_blue["y"], bins=[50, 50], range=[[0, 120], [0, 80]])

    h_red = h_red / (np.max(h_red) + 1e-9)
    h_green = h_green / (np.max(h_green) + 1e-9)
    h_blue = h_blue / (np.max(h_blue) + 1e-9)

    return np.dstack((h_red, h_green, h_blue))

# --- 4. PREPARAZIONE DATI ---
match_dict = estrai_risultati_bilanciati(MATCHES_DIR, SQUADRA_TARGET)
X_immagini, y_esiti = [], []

print("Costruzione Mappe Tattiche RGB in corso...")
for match_id, esito in match_dict.items():
    mappa = crea_heatmap_rgb(match_id, SQUADRA_TARGET)
    if mappa is not None:
        X_immagini.append(mappa)
        y_esiti.append(esito)

X_cnn_rgb = np.array(X_immagini)
y_esiti = np.array(y_esiti)
X_train, X_test, y_train, y_test = train_test_split(X_cnn_rgb, y_esiti, test_size=0.2, random_state=42)

# --- 5. TRANSFER LEARNING + DATA AUGMENTATION ---
print("\nInizializzazione Architettura con Data Augmentation...")

# Creazione del modulo di Data Augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("vertical"), # Capovolge sopra-sotto il campo
    RandomTranslation(height_factor=0.05, width_factor=0.05), # Micro-spostamenti (5%)
    RandomRotation(factor=0.02) # Micro-rotazione (2%)
])

# Importiamo la VGG16 e la congeliamo
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(50, 50, 3))
base_model.trainable = False

# Assembliamo la rete: Input -> Augmentation -> VGG16 -> Nostri Neuroni
inputs = tf.keras.Input(shape=(50, 50, 3))
x = data_augmentation(inputs) # L'immagine viene deformata qui
x = base_model(x, training=False) # L'immagine passa per la VGG16 congelata
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation='softmax')(x)

model_transfer = Model(inputs, outputs)
model_transfer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- 6. ADDESTRAMENTO ---
print("\nInizio addestramento (L'algoritmo deformerà le immagini ad ogni epoca)...")
# Aumentiamo le epoche a 30 per dare tempo alla rete di imparare dalle varianti
storia = model_transfer.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=1)

print("\n--- Valutazione Finale ---")
loss, accuracy = model_transfer.evaluate(X_test, y_test, verbose=0)
print(f"Accuratezza Modello RGB con Data Augmentation: {accuracy*100:.2f}%")