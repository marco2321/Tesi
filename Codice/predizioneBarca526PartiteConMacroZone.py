import json
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. CONFIGURAZIONE PERCORSI ---
BASE_DIR = r"C:\Users\mg259\Downloads\open-data-master\open-data-master\data"
EVENTS_DIR = os.path.join(BASE_DIR, "events")
MATCHES_FOLDER = os.path.join(BASE_DIR, "matches", "11") 

# Nomi delle 9 Macro-Zone per renderle leggibili alla fine
ZONE_NAMES = [
    "Difesa Fascia Sx", "Difesa Centro", "Difesa Fascia Dx",
    "Centrocampo Fascia Sx", "Centrocampo Centro", "Centrocampo Fascia Dx",
    "Attacco Fascia Sx", "Attacco Centro", "Attacco Fascia Dx"
]

# --- 2. ESTRAZIONE MASSIVA DEI RISULTATI ---
def estrai_risultati_barcellona_massivo(folder_path):
    print(f"Scansione di tutti i file JSON nella cartella: {folder_path} ...")
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    risultati = {}
    
    for filepath in json_files:
        with open(filepath, encoding="utf-8") as f:
            matches_data = json.load(f)

        for match in matches_data:
            match_id = match['match_id']
            home_team = match['home_team']['home_team_name']
            away_team = match['away_team']['away_team_name']
            home_score = match['home_score']
            away_score = match['away_score']

            if home_team == 'Barcelona':
                win = 1 if home_score > away_score else 0
                risultati[match_id] = win
            elif away_team == 'Barcelona':
                win = 1 if away_score > home_score else 0
                risultati[match_id] = win

    vittorie = sum(1 for v in risultati.values() if v == 1)
    non_vittorie = len(risultati) - vittorie
    print(f"Trovate {len(risultati)} partite del Barcellona (V: {vittorie} | N/P: {non_vittorie})\n")
    return risultati

# --- 3. GENERAZIONE MACRO-ZONE ---
def crea_macrozone_from_match(match_id):
    file_path = os.path.join(EVENTS_DIR, f"{match_id}.json")

    if not os.path.exists(file_path):
        return None 

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    
    if "type.name" in df.columns and "team.name" in df.columns:
        df = df[(df["type.name"] == "Pass") & (df["team.name"] == "Barcelona")]
    else:
        return None

    if len(df) == 0:
        return None

    df["x"] = df["location"].apply(lambda x: x[0] if isinstance(x, list) else np.nan)
    df["y"] = df["location"].apply(lambda x: x[1] if isinstance(x, list) else np.nan)
    df = df.dropna(subset=['x', 'y'])

    # Griglia 3x3
    # X va da 0 a 120 (40, 80, 120) -> Difesa, Centro, Attacco
    # Y va da 0 a 80 (26.6, 53.3, 80) -> Fascia Sx, Centro, Fascia Dx
    macrozone, _, _ = np.histogram2d(
        df["x"], df["y"],
        bins=[3, 3],
        range=[[0, 120], [0, 80]]
    )
    
    return macrozone.flatten()

# --- 4. PIPELINE PRINCIPALE ---
match_dict = estrai_risultati_barcellona_massivo(MATCHES_FOLDER)

X = []
y = []

print("Calcolo delle densità di passaggio nelle 9 Macro-Zone...")
for match_id, esito in match_dict.items():
    mz = crea_macrozone_from_match(match_id)
    if mz is not None:
        X.append(mz) 
        y.append(esito)

X = np.array(X)
y = np.array(y)

print(f"Dataset pronto: {X.shape[0]} partite analizzate (9 variabili per partita).\n")

# --- 5. MACHINE LEARNING AVANZATO ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Addestramento del Modello (Random Forest Bilanciato) ---")
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

print("\n--- Valutazione sui dati di Test (non visti dal modello) ---")
predizioni = model.predict(X_test)

accuratezza = accuracy_score(y_test, predizioni)
print(f"ACCURATEZZA DEL MODELLO: {accuratezza * 100:.2f}%\n")

# --- MATRICE DI CONFUSIONE CON PANDAS (versione autoesplicativa) ---
cm = confusion_matrix(y_test, predizioni)

riga_index = [
    "CASO REALE: Non-Vittoria",
    "CASO REALE: Vittoria"
]
colonna_index = [
    "PREVISIONE MODELLO: Non-Vittoria",
    "PREVISIONE MODELLO: Vittoria"
]

cm_df = pd.DataFrame(cm, index=riga_index, columns=colonna_index)

n_totale_test = len(y_test)
veri_negativi = cm[0, 0]
falsi_positivi = cm[0, 1]
falsi_negativi = cm[1, 0]
veri_positivi = cm[1, 1]

print("=" * 70)
print("MATRICE DI CONFUSIONE - Conteggi assoluti")
print("=" * 70)
print(f"""
Cos'e': confronta, per ogni partita del set di TEST (dati mai visti
dal modello durante l'addestramento), il risultato REALE con quello
PREVISTO dal modello.

  ↓  Le RIGHE (leggendo dall'alto verso il basso) indicano cosa e'
     successo davvero (dato reale).
  →  Le COLONNE (leggendo da sinistra verso destra) indicano cosa
     ha previsto il modello.

- Sulla DIAGONALE PRINCIPALE (↘ dall'alto-sinistra al basso-destra)
  ci sono le previsioni CORRETTE.
- Fuori dalla diagonale (↗ angolo in alto a destra e in basso a
  sinistra) ci sono gli ERRORI del modello.

Partite totali nel set di test analizzate: {n_totale_test}
""")
print(cm_df)
print(f"""
Lettura dettagliata dei 4 valori:
  - Veri Negativi  ({veri_negativi:>3}): partite NON vinte, previste correttamente come "Non-Vittoria"
  - Falsi Positivi ({falsi_positivi:>3}): partite NON vinte, ma il modello ha previsto ERRONEAMENTE "Vittoria"
  - Falsi Negativi ({falsi_negativi:>3}): partite VINTE, ma il modello ha previsto ERRONEAMENTE "Non-Vittoria"
  - Veri Positivi  ({veri_positivi:>3}): partite VINTE, previste correttamente come "Vittoria"
""")

# --- Versione normalizzata (percentuali per classe reale) ---
cm_norm = confusion_matrix(y_test, predizioni, normalize='true')
cm_norm_df = (pd.DataFrame(cm_norm, index=riga_index, columns=colonna_index) * 100).round(1)

print("=" * 70)
print("MATRICE DI CONFUSIONE - Percentuali (normalizzata per riga)")
print("=" * 70)
print("""
Cos'e': stessa matrice di sopra, ma ogni RIGA e' convertita in percentuale
sul totale dei casi reali di quella riga (invece di numeri assoluti).
Utile per confrontare l'affidabilita' del modello sulle due classi anche
se nel dataset non sono numericamente bilanciate (es. piu' vittorie che
non-vittorie, o viceversa).

  ↓  Ogni RIGA (dato reale) somma sempre al 100%.
  →  Le COLONNE (previsione del modello) si dividono quella percentuale.

Esempio di lettura: nella riga "CASO REALE: Vittoria", il valore nella
colonna "PREVISIONE MODELLO: Vittoria" indica la percentuale di partite
VINTE che il modello ha indovinato correttamente (questo valore
corrisponde al "recall" della classe Vittoria nel report sottostante).
""")
print(cm_norm_df.astype(str) + " %")
print()

print("\nReport di Classificazione:")
print(classification_report(y_test, predizioni, target_names=["Non-Vittoria (0)", "Vittoria (1)"]))

# --- 6. IMPORTANZA DELLE ZONE (Feature Importance) ---
print("\n--- Analisi Tattica: Quali zone determinano il risultato? ---")
importances = model.feature_importances_
indici_ordinati = np.argsort(importances)[::-1]

for i in indici_ordinati:
    print(f"{ZONE_NAMES[i]:<25}: {importances[i]*100:.1f}% di influenza sul modello")


# --- 7. VISUALIZZAZIONE GRAFICA PER LA TESI ---
print("\nGenerazione del grafico a barre in corso...")

plt.figure(figsize=(10, 6))

indici_ordinati_grafico = np.argsort(importances)
zone_ordinate_grafico = [ZONE_NAMES[i] for i in indici_ordinati_grafico]
valori_ordinati_grafico = importances[indici_ordinati_grafico] * 100

bars = plt.barh(zone_ordinate_grafico, valori_ordinati_grafico, color='steelblue', edgecolor='black')

bars[-1].set_color('crimson')
bars[-1].set_edgecolor('black')

for bar in bars:
    plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.1f}%',
             va='center', ha='left', fontsize=10, fontweight='bold')

plt.xlabel('Importanza nel determinare la Vittoria (%)', fontsize=12)
plt.ylabel('Macro-Zone del Campo', fontsize=12)
plt.title("L'Importanza Tattica delle Zone di Passaggio\nModello Predittivo FC Barcelona", fontsize=14, fontweight='bold')

plt.xlim(0, max(valori_ordinati_grafico) + 3) 

plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

nome_file_grafico = "Importanza_MacroZone_Barcellona.png"
plt.savefig(nome_file_grafico, dpi=300)
print(f"Grafico salvato con successo come '{nome_file_grafico}' nella cartella del codice.")

# --- 8. HEATMAP DELLA MATRICE DI CONFUSIONE (opzionale, per la tesi) ---
try:
    import seaborn as sns

    etichette_brevi = ["Non-Vittoria", "Vittoria"]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=True,
        xticklabels=etichette_brevi, yticklabels=etichette_brevi,
        annot_kws={"size": 16, "weight": "bold"}, ax=ax
    )
    ax.set_title(
        "Matrice di Confusione - Modello Predittivo FC Barcelona\n"
        "(Righe = risultato reale, Colonne = risultato previsto dal modello)",
        fontsize=11, pad=40
    )
    ax.set_ylabel("Risultato REALE della partita", fontsize=11)
    ax.set_xlabel("Risultato PREVISTO dal modello", fontsize=11)

    # Freccia orizzontale sopra la matrice: senso di lettura delle colonne
    ax.annotate(
        '', xy=(2.05, -0.55), xytext=(-0.05, -0.55),
        xycoords=('data', 'data'), textcoords=('data', 'data'),
        arrowprops=dict(arrowstyle='-|>', color='dimgray', lw=1.8),
        annotation_clip=False
    )
    ax.text(1.0, -0.65, "senso di lettura: PREVISIONE",
            ha='center', va='bottom', fontsize=9, color='dimgray',
            style='italic')

    # Freccia verticale a sinistra della matrice: senso di lettura delle righe
    ax.annotate(
        '', xy=(-0.55, 2.05), xytext=(-0.55, -0.05),
        xycoords=('data', 'data'), textcoords=('data', 'data'),
        arrowprops=dict(arrowstyle='-|>', color='dimgray', lw=1.8),
        annotation_clip=False
    )
    ax.text(-0.75, 1.0, "senso di lettura: REALE", ha='center', va='center',
            fontsize=9, color='dimgray', style='italic', rotation=90)

    plt.tight_layout()
    plt.savefig("Matrice_Confusione_Barcellona.png", dpi=300, bbox_inches='tight')
    print("Heatmap della matrice di confusione salvata come 'Matrice_Confusione_Barcellona.png'.")
except ImportError:
    print("\n[Info] Libreria 'seaborn' non trovata: salta la heatmap della matrice di confusione.")
    print("Per installarla: pip install seaborn")

# Mostra tutti i grafici a schermo
plt.show()