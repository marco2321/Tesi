import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# --- 1. CONFIGURAZIONE PERCORSI E DATI ---
BASE_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data"
MATCHES_DIR = os.path.join(BASE_DIR, "matches")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
GIOCATORE_TARGET = "Kevin De Bruyne"

# --- 2. TROVARE LE PARTITE DEL BELGIO (Ripristinato dal tuo codice originale) ---
def trova_partite_belgio(matches_dir):
    print("Ricerca delle partite del Belgio nel database...")
    match_ids = []
    json_files = glob.glob(os.path.join(matches_dir, "**", "*.json"), recursive=True)
    
    for filepath in json_files:
        with open(filepath, encoding="utf-8") as f:
            matches_data = json.load(f)
            
        for match in matches_data:
            home_team = match['home_team']['home_team_name']
            away_team = match['away_team']['away_team_name']
            
            if home_team == 'Belgium' or away_team == 'Belgium':
                match_ids.append(match['match_id'])
                
    return match_ids

# --- 3. ESTRAZIONE DATI DI DE BRUYNE ---
def estrai_dati_kdb(match_ids):
    print(f"Estrazione passaggi di {GIOCATORE_TARGET} in corso...")
    X_list, y_list = [], []
    
    for match_id in match_ids:
        file_path = os.path.join(EVENTS_DIR, f"{match_id}.json")
        if not os.path.exists(file_path): continue
            
        with open(file_path, encoding="utf-8") as f:
            events = json.load(f)
            
        df = pd.json_normalize(events)
        if "player.name" not in df.columns: continue
        
        # Prendiamo solo i passaggi di De Bruyne
        df_kdb = df[(df["player.name"] == GIOCATORE_TARGET) & (df["type.name"] == "Pass")].copy()
        if df_kdb.empty: continue
        
        # Calcolo Creatività
        key_passes = df_kdb.get("pass.shot_assist", pd.Series(False, index=df_kdb.index)).sum()
        assist = df_kdb.get("pass.goal_assist", pd.Series(False, index=df_kdb.index)).sum()
        creativity_index = key_passes + assist
        
        # Generazione Heatmap
        df_kdb["x"] = df_kdb["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
        df_kdb["y"] = df_kdb["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
        df_kdb = df_kdb.dropna(subset=['x', 'y'])
        
        if len(df_kdb) > 5:
            heatmap, _, _ = np.histogram2d(df_kdb["x"], df_kdb["y"], bins=[50, 50], range=[[0, 120], [0, 80]])
            heatmap = heatmap / (np.max(heatmap) + 1e-9)
            X_list.append(heatmap.T)
            y_list.append(creativity_index)
            
    return np.array(X_list).reshape(-1, 50, 50, 1), np.array(y_list)

# Esecuzione estrazione
partite_belgio = trova_partite_belgio(MATCHES_DIR)
X_data, y_data = estrai_dati_kdb(partite_belgio)
print(f"Trovate {len(X_data)} partite valide per l'addestramento.")

# --- 4. CREAZIONE E ADDESTRAMENTO MODELLO ---
inputs = Input(shape=(50, 50, 1))
x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
# QUESTO È IL LAYER CHE OSSERVEREMO CON LA GRAD-CAM
last_conv_layer = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='ultimo_conv')(x)
x = MaxPooling2D(pool_size=(2, 2))(last_conv_layer)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='linear')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Addestramento veloce della rete in corso...")
model.fit(X_data, y_data, epochs=15, verbose=0)

# --- 5. ALGORITMO GRAD-CAM ---
# Prendiamo la partita in cui De Bruyne è stato più creativo
best_idx = np.argmax(y_data)
img_tensor = X_data[best_idx:best_idx+1]
creativita_reale = y_data[best_idx]

print("Applicazione algoritmo Grad-CAM...")
# Creiamo un modello che restituisce sia l'output finale sia le attivazioni dell'ultimo layer convoluzionale
grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer('ultimo_conv').output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_tensor)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalizzazione della heatmap tra 0 e 1
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

# Ridimensionamento a 50x50 per sovrapporla all'immagine originale
heatmap_tensor = tf.expand_dims(tf.expand_dims(heatmap, 0), -1)
heatmap_resized = tf.image.resize(heatmap_tensor, (50, 50))
heatmap_resized = tf.squeeze(heatmap_resized).numpy()

# --- 6. VISUALIZZAZIONE E SALVATAGGIO ---
plt.figure(figsize=(12, 5))
plt.suptitle(f"Analisi Grad-CAM - Kevin De Bruyne (Indice Creatività: {creativita_reale})", fontsize=14, fontweight='bold')

plt.subplot(1, 2, 1)
plt.imshow(img_tensor[0, :, :, 0], cmap='gray', origin='lower')
plt.title("Input Originale (Passaggi in scala di grigi)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_tensor[0, :, :, 0], cmap='gray', origin='lower')
plt.imshow(heatmap_resized, cmap='jet', alpha=0.5, origin='lower') # Sovrapposizione
plt.title("Attivazione Grad-CAM (Cosa guarda la rete)")
plt.axis('off')

plt.savefig("Grad_CAM_DeBruyne.png", dpi=300, bbox_inches='tight')
print("\nSuccesso! Immagine salvata come 'Grad_CAM_DeBruyne.png'. L'esperimento è concluso!")
plt.show()