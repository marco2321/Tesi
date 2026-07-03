import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURAZIONE PERCORSI E MATCH ---
BASE_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data"
EVENTS_DIR = os.path.join(BASE_DIR, "events")

MATCH_ID = 69243  # Usiamo un match specifico del Barcellona come esempio illustrativo
SQUADRA_TARGET = "Barcelona"

file_path = os.path.join(EVENTS_DIR, f"{MATCH_ID}.json")

if not os.path.exists(file_path):
    print(f"Errore: File {file_path} non trovato.")
    exit()

with open(file_path, encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)
df_team = df[df["team.name"] == SQUADRA_TARGET].copy()

# Estrazione coordinate
df_team["x"] = df_team["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
df_team["y"] = df_team["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
df_team = df_team.dropna(subset=['x', 'y'])

# --- 2. SEPARAZIONE DEGLI EVENTI NEI 3 CANALI ---
# CANALE ROSSO (Attacco): Tiri e Passaggi Chiave
is_shot = df_team["type.name"] == "Shot"
is_key_pass = df_team.get("pass.shot_assist", pd.Series(False, index=df_team.index)) == True
df_red = df_team[is_shot | is_key_pass]

# CANALE VERDE (Costruzione): Passaggi
df_green = df_team[df_team["type.name"] == "Pass"]

# CANALE BLU (Difesa): Duelli, Intercetti, Blocchi, Spazzate
df_blue = df_team[df_team["type.name"].isin(["Duel", "Interception", "Block", "Clearance"])]

# --- 3. GENERAZIONE DELLE MATRICI 50x50 ---
# Calcoliamo gli istogrammi
h_red, _, _ = np.histogram2d(df_red["x"], df_red["y"], bins=[50, 50], range=[[0, 120], [0, 80]])
h_green, _, _ = np.histogram2d(df_green["x"], df_green["y"], bins=[50, 50], range=[[0, 120], [0, 80]])
h_blue, _, _ = np.histogram2d(df_blue["x"], df_blue["y"], bins=[50, 50], range=[[0, 120], [0, 80]])

# Normalizzazione (0-1)
h_red = h_red / (np.max(h_red) + 1e-9)
h_green = h_green / (np.max(h_green) + 1e-9)
h_blue = h_blue / (np.max(h_blue) + 1e-9)

# Trasponiamo le matrici per allinearle correttamente all'orientamento del campo (X orizzontale, Y verticale)
h_red = h_red.T
h_green = h_green.T
h_blue = h_blue.T

# Creiamo l'immagine RGB unita
heatmap_rgb = np.dstack((h_red, h_green, h_blue))

# --- 4. CREAZIONE DELL'IMMAGINE PER LA TESI ---
# Sfondo scuro per far risaltare i colori RGB
plt.style.use('dark_background')
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"Scomposizione Mappa Tattica Totale RGB (Match: {MATCH_ID})", fontsize=16, fontweight='bold')

# Plot Canale Rosso
axs[0].imshow(h_red, cmap='Reds', origin='lower')
axs[0].set_title('Canale ROSSO\n(Tiri e Key Passes)')
axs[0].axis('off')

# Plot Canale Verde
axs[1].imshow(h_green, cmap='Greens', origin='lower')
axs[1].set_title('Canale VERDE\n(Passaggi e Possesso)')
axs[1].axis('off')

# Plot Canale Blu
axs[2].imshow(h_blue, cmap='Blues', origin='lower')
axs[2].set_title('Canale BLU\n(Intercetti, Duelli, Difesa)')
axs[2].axis('off')

# Plot RGB Combinato
axs[3].imshow(heatmap_rgb, origin='lower')
axs[3].set_title('Mappa RGB Completa\n(Input per VGG16)')
axs[3].axis('off')

plt.tight_layout()

# Salvataggio dell'immagine ad alta risoluzione pronta per Word
plt.savefig("Scomposizione_RGB_Tesi.png", dpi=300, bbox_inches='tight')
print("Immagine salvata con successo: 'Scomposizione_RGB_Tesi.png'")
plt.show()