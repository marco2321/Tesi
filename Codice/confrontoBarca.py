import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

EVENTS_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data\events"

# 👉 USA I TUOI MATCH GIUSTI
selected_matches = [69243, 69276, 69328, 266015]

print("Match usati:", selected_matches)


# -------- FUNZIONE HEATMAP --------
def crea_heatmap_from_match(match_id):
    file_path = os.path.join(EVENTS_DIR, f"{match_id}.json")

    if not os.path.exists(file_path):
        print("MISSING:", file_path)
        return np.zeros((25, 25))

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    # ❗ NON filtriamo squadra (già sono match Spagna)
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


# -------- CREA HEATMAP --------
heatmaps = []

for mid in selected_matches:
    h = crea_heatmap_from_match(mid)
    heatmaps.append(h)


# -------- CONFRONTO --------
def mse(h1, h2):
    return np.mean((h1 - h2) ** 2)

print("\nConfronto:\n")

for i in range(len(heatmaps)):
    for j in range(i + 1, len(heatmaps)):
        print(f"{i} vs {j} -> MSE: {mse(heatmaps[i], heatmaps[j])}")


# -------- PLOT --------
avg_heatmap = np.sum(heatmaps, axis=0)

if np.max(avg_heatmap) > 0:
    avg_heatmap = avg_heatmap / np.max(avg_heatmap)

print("MAX:", np.max(avg_heatmap))
print("MIN:", np.min(avg_heatmap))

plt.imshow(avg_heatmap, cmap="Reds", origin="lower")
plt.title("Heatmap aggregata passaggi Barca (4 partite)")
plt.colorbar()
plt.show()