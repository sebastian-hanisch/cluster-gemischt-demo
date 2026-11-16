"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und die vorgerechneten Zeilen der Messreihe."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
FROZEN_PATH = ROOT / "data" / "clg_frozen.json"

# --- Regler des Vehikels (Seitenleiste) ------------------------------------------------------------------------------------------------------
M_MIN, M_MAX, DEFAULT_M = 6, 48, 12                    # Kunden m
K_MIN, K_MAX, DEFAULT_K, K_STEP = 10, 120, 30, 10      # Container je Kunde im Mittel
SU_OPTIONS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)   # Streuung der Kundeneffekte σ_u (log-Skala)
DEFAULT_SU = 0.30
SEED_MAX, DEFAULT_SEED = 999999, 2026
ARTEN = {"half": "0/1, die Hälfte der Kunden", "two": "0/1, nur 2 Kunden", "cont": "stetig (Kundenvolumen)"}
DEFAULT_ART = "half"

# --- Regler beim Experiment (Hauptbereich) ----------------------------------------------------------------------------------------------------
MERKMALE = ("Rahmenvertrag (Kundenmerkmal)", "Zollhalt (Merkmal im Kunden)")
DEFAULT_MERKMAL = MERKMALE[0]
REPS_OPTIONS = (100, 200, 500, 1000)                   # Wiederholungen der Live-Rechnung (Zähleinheit: eine Historie)
DEFAULT_REPS = 200
DEFAULT_MIM = True                                      # Experiment 5: Kundenmerkmal im Modell

HAND_BEISPIEL = "1, 3 | 5, 9 | 8, 10"
HAND_RANDFALL = "1, 3 | 2, 6 | 4, 4"
HAND_DATENSAETZE = ("Handbeispiel", "Randfall", "Zufallsdatensatz")
HAND_MAX_KUNDEN, HAND_MAX_CONTAINER = 8, 8

# --- Vorgerechnete Messreihe (data/clg_frozen.json, erzeugt von tools/clg_make_frozen.py) -------------------------------------------------------
FROZEN_R = 1000                                         # Historien je Zeile
FROZEN_SEED_BASE = 7000
FROZEN_ROWS = (                                         # (m, k̄, σ_u, Art des Kundenmerkmals)
    (6, 30, 0.30, "half"), (12, 30, 0.30, "half"), (24, 30, 0.30, "half"), (48, 30, 0.30, "half"),
    (12, 10, 0.30, "half"), (12, 120, 0.30, "half"),
    (12, 30, 0.00, "half"), (12, 30, 0.10, "half"), (12, 30, 0.20, "half"), (12, 30, 0.45, "half"),
    (12, 30, 0.30, "two"), (12, 30, 0.30, "cont"),
    (6, 30, 0.30, "cont"), (10, 30, 0.30, "cont"), (24, 30, 0.30, "cont"),
    (6, 10, 0.30, "half"), (6, 300, 0.30, "half"), (12, 300, 0.30, "half"), (48, 10, 0.30, "half"), (48, 300, 0.30, "half"),
)
SU2_GRID_M = (6, 12, 48)                                # Experiment 5: σ̂_u² über Kundenzahl und Container je Kunde
SU2_GRID_K = (10, 30, 300)
SU2_SEED = 4900

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------
COLOR_NAIV = "#d62728"
COLOR_ROBUST = "#ff7f0e"
COLOR_MIXED = "#2F6B65"
COLOR_MEAN = "#4c78a8"
COLOR_NEUTRAL = "rgba(150,150,150,0.8)"
COLOR_TRUE = "#7f7f7f"

PRESETS = {
    "Von Hand: drei Kunden (Voreinstellung)": {},
    "Randfall (singulär)": {"hand": HAND_RANDFALL},
    "Viele Container, wenige Kunden": {"k": 120},
    "Wenige Kunden": {"m": 6},
    "Kleine Streuung zwischen Kunden": {"su": 0.10},
    "Ungleich verteiltes Kundenmerkmal": {"art": "two"},
    "Ohne Cluster": {"su": 0.0},
}
PRESET_HELP = {
    "Von Hand: drei Kunden (Voreinstellung)": "Drei Kunden mit je zwei Containern von Hand (Zahlen 1, 3 | 5, 9 | 8, 10); darunter das Vehikel mit 12 Kunden und im Mittel 30 Containern je Kunde, Streuung zwischen den Kunden 0,30.",
    "Randfall (singulär)": "Drei Kunden mit den Zahlen 1, 3 | 2, 6 | 4, 4: beide Schätzer der Kundenstreuung werden negativ, die Anpassung liegt auf dem Rand σ_u² = 0.",
    "Viele Container, wenige Kunden": "12 Kunden mit im Mittel 120 Containern je Kunde: mehr Zeilen, aber nicht mehr Kunden.",
    "Wenige Kunden": "Nur 6 Kunden mit im Mittel 30 Containern je Kunde.",
    "Kleine Streuung zwischen Kunden": "Streuung der Kundeneffekte nur 0,10: das gemischte Modell liegt oft am Rand σ̂_u² = 0.",
    "Ungleich verteiltes Kundenmerkmal": "Der Rahmenvertrag gilt nur bei 2 der Kunden (statt bei der Hälfte).",
    "Ohne Cluster": "Streuung zwischen den Kunden 0: es gibt keinen Kundeneffekt, nichts ist zu finden.",
}
