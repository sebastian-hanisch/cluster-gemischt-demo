"""Vorgerechnete Messreihe der App: 1000 Historien je Zeile, eingefroren in data/clg_frozen.json.

Aufruf (Python des Test-venv, Arbeitsordner egal):
    python tools/clg_make_frozen.py            schreibt data/clg_frozen.json neu (alle Zeilen aus clg_constants.FROZEN_ROWS, dazu die Reihen ohne und mit Kundenmerkmal)
    python tools/clg_make_frozen.py --check    rechnet neu und vergleicht mit der Datei als Band (Zählungen ± 2, Anteile ± 0,01, Mittel relativ 1e-6)
    python tools/clg_make_frozen.py --check --rows 3   nur die ersten drei Zeilen vergleichen (der Test nutzt das)

Die Rechnung ist deterministisch (SplitMix64 mit Reihen für Logarithmus und Kosinus, nur Grundrechenarten): gleiche Zufalls-Seeds geben dieselben Historien.
Die Zählungen können sich zwischen Plattformen nur durch letzte Binärstellen in der Suche nach σ̂_u² unterscheiden (Gleichstände bei einem Intervallrand); darum prüft --check Bänder."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import clg_constants as C  # noqa: E402
import clg_vehicle as V  # noqa: E402


def row_seed(i: int) -> int:
    return C.FROZEN_SEED_BASE + i


def compute_rows(limit: int | None = None) -> list[dict]:
    rows = []
    for i, (m, k, su, mode) in enumerate(C.FROZEN_ROWS):
        if limit is not None and i >= limit:
            break
        t0 = time.time()
        rows.append(V.simulate(m, k, su, C.FROZEN_R, row_seed(i), mode))
        print(f"   Zeile {i + 1}/{len(C.FROZEN_ROWS)}: m = {m}, k̄ = {k}, σ_u = {su}, Art {mode}: {time.time() - t0:.1f} s", flush=True)
    return rows


def compute_su2_merkmal() -> dict:
    out = {}
    for mit in (True, False):
        r = V.su2_series(12, 30, 0.30, C.FROZEN_R, C.SU2_SEED, "half", mit)
        r.pop("werte")
        out["mit" if mit else "ohne"] = r
    return out


def build() -> dict:
    return dict(info="Vorgerechnet von tools/clg_make_frozen.py (SplitMix64, 1000 Historien je Zeile); nicht von Hand ändern.", R=C.FROZEN_R, rows=compute_rows(),
                su2_merkmal=compute_su2_merkmal())


def _nan_zu_null(x):
    if isinstance(x, dict):
        return {k: _nan_zu_null(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nan_zu_null(v) for v in x]
    if isinstance(x, float) and x != x:
        return None
    return x


def vergleiche(alt: dict, neu: dict, band_zaehl: int = 2) -> list[str]:
    """Abweichungen zwischen zwei Zeilen von `simulate` als Liste von Meldungen (leer = gleich im Band)."""
    meld = []
    for nm in ("rv", "zoll"):
        for k, v in alt["cover"][nm].items():
            if abs(v - neu["cover"][nm][k]) > band_zaehl:
                meld.append(f"{nm}/{k}: {v} gegen {neu['cover'][nm][k]}")
    for k in ("rand_ml", "rand_re", "ok"):
        if abs(alt[k] - neu[k]) > band_zaehl:
            meld.append(f"{k}: {alt[k]} gegen {neu[k]}")
    for k in ("su2_mean", "su2_sd"):
        if abs(alt[k] - neu[k]) > 1e-6 * max(abs(alt[k]), 1e-12):
            meld.append(f"{k}: {alt[k]} gegen {neu[k]}")
    return meld


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="neu rechnen und mit data/clg_frozen.json vergleichen (Band), nichts schreiben")
    ap.add_argument("--rows", type=int, default=None, help="nur die ersten N Zeilen (mit --check)")
    args = ap.parse_args()
    if args.check:
        alt = json.loads(C.FROZEN_PATH.read_text(encoding="utf-8"))
        neu = compute_rows(args.rows)
        fehler = []
        for a, n in zip(alt["rows"], neu):
            fehler += [f"Zeile m={a['m']} k̄={a['kbar']} σ_u={a['su']} {a['mode']}: {x}" for x in vergleiche(a, n)]
        if args.rows is None:
            su2 = compute_su2_merkmal()
            for key in ("mit", "ohne"):
                a, n = alt["su2_merkmal"][key], su2[key]
                if abs(a["mean"] - n["mean"]) > 1e-6 or abs(a["sd"] - n["sd"]) > 1e-6 or abs(a["rand"] - n["rand"]) > 2:
                    fehler.append(f"su2_merkmal/{key}: {a['mean']} gegen {n['mean']}")
        print("\n".join(fehler) if fehler else "GESAMT: Messreihe stimmt mit der Datei überein (Band).")
        return 1 if fehler else 0
    daten = _nan_zu_null(build())
    C.FROZEN_PATH.parent.mkdir(exist_ok=True)
    C.FROZEN_PATH.write_text(json.dumps(daten, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {C.FROZEN_PATH} ({len(daten['rows'])} Zeilen)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
