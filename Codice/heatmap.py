import json
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle

# ---- CARICA MATCH ----
with open(r"C:\Users\mg259\OneDrive\Desktop\TESI\open-data-master\open-data-master\data\events\7580.json", encoding='utf-8') as f:
    data = json.load(f)

df = pd.json_normalize(data)

# ---- FILTRA PASSAGGI ----
passes = df[df['type.name'] == 'Pass'].copy()

# ---- CONTROLLA CHE CI SIANO PASSAGGI ----
if passes.empty:
    print("Nessun passaggio trovato nel JSON!")
else:
    # ---- ESTRAI COORDINATE ----
    passes['x'] = passes['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    passes['y'] = passes['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    passes = passes.dropna(subset=['x', 'y'])

    # ---- CREA FIGURA ----
    fig, ax = plt.subplots(figsize=(12, 7))

    # ---- CAMPO BLU ----
    ax.add_patch(Rectangle((0, 0), 120, 80, edgecolor='white', facecolor='#001f4d', zorder=0))  # sfondo blu

    # LINEE CAMPO
    # Linea di centrocampo
    ax.plot([60, 60], [0, 80], color='white', linewidth=2)
    # Aree di rigore sinistra
    ax.add_patch(Rectangle((0, 20), 18, 40, edgecolor='white', facecolor='none', linewidth=2))
    # Aree di rigore destra
    ax.add_patch(Rectangle((102, 20), 18, 40, edgecolor='white', facecolor='none', linewidth=2))
    # Cerchio al centro
    ax.add_patch(Circle((60, 40), 10, edgecolor='white', facecolor='none', linewidth=2))

    # PORTE
    ax.add_patch(Rectangle((0, 30), 2, 20, edgecolor='white', facecolor='yellow', linewidth=2, zorder=1))
    ax.add_patch(Rectangle((118, 30), 2, 20, edgecolor='white', facecolor='yellow', linewidth=2, zorder=1))

    # CORNER
    corner_size = 2
    corners = [(0,0),(0,80),(120,0),(120,80)]
    for x, y in corners:
        ax.add_patch(Circle((x, y), corner_size, color='white', zorder=1))

    # ---- HEATMAP SOPRA IL CAMPO ----
    sns.kdeplot(
        x=passes['x'],
        y=passes['y'],
        fill=True,
        cmap="RdYlGn_r",  # verde → giallo → rosso
        bw_adjust=0.6,
        alpha=0.6,
        ax=ax
    )

    # LIMITE ASSI
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Pass Heatmap (TV style con campo blu)", color='white')

    # Nasconde assi
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#001f4d')

    plt.show()