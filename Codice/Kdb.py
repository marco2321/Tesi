import os
# Disattiva gli avvisi informativi di TensorFlow (oneDNN) per un terminale più pulito
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURAZIONE PERCORSI ---
BASE_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data"
EVENTS_DIR = os.path.join(BASE_DIR, "events")
MATCHES_DIR = os.path.join(BASE_DIR, "matches") 

GIOCATORE_TARGET = "Kevin De Bruyne"

# --- 2. TROVARE LE PARTITE DEL BELGIO ---
def trova_partite_belgio(matches_dir):
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
                
    print(f"Trovate {len(match_ids)} partite del Belgio nel database.")
    return match_ids

# --- 3. ESTRAZIONE DATI DEL GIOCATORE (X = Heatmap, y = Creatività) ---
def estrai_dati_giocatore(match_id, player_name):
    file_path = os.path.join(EVENTS_DIR, f"{match_id}.json")
    if not os.path.exists(file_path):
        return None, None

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    
    if "player.name" not in df.columns:
        return None, None

    df_player = df[(df["player.name"] == player_name) & (df["type.name"] == "Pass")].copy()
    
    if len(df_player) == 0:
        return None, None

    # Calcolo Creatività oggettiva
    key_passes = df_player["pass.shot_assist"].sum() if "pass.shot_assist" in df_player.columns else 0
    assists = df_player["pass.goal_assist"].sum() if "pass.goal_assist" in df_player.columns else 0
    indice_creativita = int(key_passes + assists)

    # Generazione Heatmap 50x50
    df_player["x"] = df_player["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    df_player["y"] = df_player["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
    df_player = df_player.dropna(subset=['x', 'y'])

    heatmap, _, _ = np.histogram2d(
        df_player["x"], df_player["y"],
        bins=[50, 50],
        range=[[0, 120], [0, 80]]
    )
    
    return heatmap, indice_creativita

# --- 4. ESECUZIONE DELLA PIPELINE DATI ---
partite_belgio = trova_partite_belgio(MATCHES_DIR)

X_immagini = []
y_creativita = []

print(f"\nEstrazione passaggi creativi di {GIOCATORE_TARGET} in corso...")

for match_id in partite_belgio:
    mappa_visiva, punteggio = estrai_dati_giocatore(match_id, GIOCATORE_TARGET)
    
    if mappa_visiva is not None:
        X_immagini.append(mappa_visiva)
        y_creativita.append(punteggio)
        # IL PRINT RIPRISTINATO PER IL CONTROLLO SINGOLO:
        print(f"Match ID {match_id}: Trovati {punteggio} passaggi chiave/assist.")

X_immagini = np.array(X_immagini)
y_creativita = np.array(y_creativita)

print(f"\nDataset Visivo Pronto: {X_immagini.shape[0]} partite valide analizzate.")

# --- 5. VISUALIZZAZIONE DELLA HEATMAP MIGLIORE ---
if len(y_creativita) > 0:
    print("\nGenerazione dell'immagine della prestazione più creativa...")
    indice_max = np.argmax(y_creativita)
    miglior_heatmap = X_immagini[indice_max]
    punteggio_max = y_creativita[indice_max]

    plt.figure(figsize=(10, 6))
    plt.imshow(miglior_heatmap.T, cmap='magma', origin='lower', extent=[0, 120, 0, 80])
    plt.title(f"Heatmap Creatività: {GIOCATORE_TARGET}\nKey Passes & Assists: {punteggio_max}", fontsize=14, fontweight='bold')
    plt.xlabel("Lunghezza Campo (Metri)", fontsize=12)
    plt.ylabel("Larghezza Campo (Metri)", fontsize=12)
    plt.colorbar(label="Densità dei Passaggi")
    
    plt.axvline(60, color='white', linestyle='--', alpha=0.5) 
    plt.axvline(102, color='white', linestyle='-', alpha=0.3) 
    
    nome_file_mappa = f"Heatmap_{GIOCATORE_TARGET.replace(' ', '_')}_Max.png"
    plt.savefig(nome_file_mappa, dpi=300)
    print(f"Immagine salvata come '{nome_file_mappa}'. Chiudi la finestra dell'immagine per continuare.")
    plt.show()

# --- 6. ARCHITETTURA DELLA RETE NEURALE (CNN) ---
print("\nInizializzazione della Rete Neurale Convoluzionale...")

# Reshape per aggiungere il canale del colore (1) richiesto dalla CNN
X_cnn = X_immagini.reshape(-1, 50, 50, 1)

X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_creativita, test_size=0.2, random_state=42)

model_cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])

model_cnn.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 7. ADDESTRAMENTO ---
print("\nInizio addestramento della CNN...")
storia = model_cnn.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

print("\n--- Valutazione Finale sul Test Set ---")
loss, mae = model_cnn.evaluate(X_test, y_test, verbose=0)
print(f"Errore Medio Assoluto (MAE): {mae:.2f}")

# --- 8. GRAFICO DELLA CURVA DI APPRENDIMENTO ---
print("\nGenerazione della curva di apprendimento...")

plt.figure(figsize=(10, 6))
mae_train = storia.history['mae']
mae_val = storia.history['val_mae']
epoche = range(1, len(mae_train) + 1)

plt.plot(epoche, mae_train, 'o-', label='Errore Training (Studio)', color='blue')
plt.plot(epoche, mae_val, 's-', label='Errore Validazione (Esame)', color='orange')

plt.title("Curva di Apprendimento della CNN\nPredizione Creatività di De Bruyne", fontsize=14, fontweight='bold')
plt.xlabel("Epoche di Addestramento", fontsize=12)
plt.ylabel("Errore Medio Assoluto (MAE)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig("Curva_Apprendimento_DeBruyne.png", dpi=300)
print("Grafico salvato come 'Curva_Apprendimento_DeBruyne.png'.")
plt.show()