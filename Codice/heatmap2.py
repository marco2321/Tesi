import json
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap

# ---- CARICA MATCH ----
with open(r"C:\Users\mg259\OneDrive\Desktop\TESI\open-data-master\open-data-master\data\events\7580.json", encoding='utf-8') as f:
    data = json.load(f)

df = pd.json_normalize(data)

# ---- FILTRA PASSAGGI ----
passes = df[df['type.name'] == 'Pass'].copy()
if passes.empty:
    print("Nessun passaggio trovato nel JSON!")
else:
    passes['x'] = passes['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    passes['y'] = passes['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    passes = passes.dropna(subset=['x','y'])

    # ---- CREA FIGURA ----
    fig, ax = plt.subplots(figsize=(12,7))
    ax.set_facecolor('#001f4d')  # blu campo TV

    # ---- COLORI E LINEE ----
    line_color = 'white'
    line_width = 1.5
    glow_widths = [6,4,2]

    def draw_glow_line(xy_start, xy_end, color=line_color, linewidth=line_width, glow_widths=glow_widths):
        for w in glow_widths:
            ax.plot([xy_start[0], xy_end[0]], [xy_start[1], xy_end[1]], color=color, linewidth=w, alpha=0.08)
        ax.plot([xy_start[0], xy_end[0]], [xy_start[1], xy_end[1]], color=color, linewidth=linewidth)

    # Contorno campo
    draw_glow_line((0,0),(120,0))
    draw_glow_line((0,80),(120,80))
    draw_glow_line((0,0),(0,80))
    draw_glow_line((120,0),(120,80))

    # Linea centrocampo
    draw_glow_line((60,0),(60,80))

    # ---- AREE DI RIGORE ----
    penalty_params = {'facecolor':'none','edgecolor':'white','linewidth':2}
    ax.add_patch(Rectangle((0,18),18,44, **penalty_params))
    ax.add_patch(Rectangle((102,18),18,44, **penalty_params))

    # Cerchio centrocampo sfumato
    for r, alpha in zip([14,12,10], [0.08,0.15,0.25]):
        ax.add_patch(Circle((60,40), r, edgecolor='white', facecolor='none', linewidth=2, alpha=alpha))
    ax.add_patch(Circle((60,40),10, edgecolor='white', facecolor='none', linewidth=2))

    # ---- PORTE ----
    ax.add_patch(Rectangle((0,30),2,20, edgecolor='white', facecolor='yellow', linewidth=2, zorder=2))
    ax.add_patch(Rectangle((118,30),2,20, edgecolor='white', facecolor='yellow', linewidth=2, zorder=2))

    # ---- ANGOLI ----
    corner_radius = 2
    for x,y in [(0,0),(0,80),(120,0),(120,80)]:
        ax.add_patch(Circle((x,y), corner_radius, color='white', zorder=2))

    # ---- HEATMAP PASSAGGI ----
    cmap = LinearSegmentedColormap.from_list('pass_cmap',['#00ff00','#ffff00','#ff0000'])
    sns.kdeplot(
        x=passes['x'],
        y=passes['y'],
        fill=True,
        cmap=cmap,
        bw_adjust=0.4,
        alpha=0.55,
        thresh=0,
        levels=100,
        zorder=1,
        ax=ax
    )

    # ---- LIMITI ASSI E TITOLI ----
    ax.set_xlim(0,120)
    ax.set_ylim(0,80)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pass Heatmap (TV style con campo blu)", color='white', fontsize=18, pad=15)

    plt.show()