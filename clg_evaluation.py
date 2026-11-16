"""Kennzahlen, Verdikte (reine Funktionen), Handbeispiel und vorgerechnete Messreihe.

Alle Verdikte sind `verdict_*(…) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Zeilen.
Aussagen gelten für **diese Welt**: Import-Container, Zufallsachsenabschnitt, normale Fehler, konstante Streuung, Kunden frisch gezogen, 6 bis 48 Kunden."""

import json
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import clg_constants as C
import clg_mixed as MX
import clg_vehicle as V
from clg_rng import SplitMix64


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def de_pct(share: float, nd: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Komma und Leerzeichen vor dem Zeichen."""
    return de(100.0 * share, nd) + " %"


def frac_str(f: Fraction) -> str:
    """Bruch als „11“ bzw. „20/3“."""
    text = str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    return text.replace("-", "−")


def frac_zahl(f: Fraction, nd: int = 3) -> str:
    """Bruch mit Dezimalwert: „11“ bzw. „20/3 (≈ 6,667)“."""
    return frac_str(f) if f.denominator == 1 else f"{frac_str(f)} (≈ {de(float(f), nd)})"


def pp(x: float, nd: int = 1) -> str:
    """Prozentpunkte (Anteil 0 bis 1) mit Komma."""
    return de(100.0 * x, nd)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    m: int = C.DEFAULT_M
    k: int = C.DEFAULT_K
    su: float = C.DEFAULT_SU
    art: str = C.DEFAULT_ART
    seed: int = C.DEFAULT_SEED
    hand: str = C.HAND_BEISPIEL
    merk: str = C.DEFAULT_MERKMAL
    reps: int = C.DEFAULT_REPS
    mim: bool = C.DEFAULT_MIM

    @property
    def rho(self) -> float:
        return self.su ** 2 / (self.su ** 2 + V.S_IMPORT ** 2)

    @property
    def zoll(self) -> bool:
        return self.merk == C.MERKMALE[1]


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: von Hand
# ---------------------------------------------------------------------------------------------------------------------------------------------

def parse_hand(text: str) -> list[list[Fraction]]:
    """„1, 3 | 5, 9 | 8, 10“ → drei Kunden mit je zwei Zahlen (Zahlen als Brüche: ganz, dezimal mit Punkt oder Bruch wie 1/3). Fehler als ValueError mit Meldung."""
    kunden = [k.strip() for k in text.split("|")]
    if len(kunden) < 2:
        raise ValueError("Mindestens 2 Kunden nötig (mit | trennen).")
    if len(kunden) > C.HAND_MAX_KUNDEN:
        raise ValueError(f"Höchstens {C.HAND_MAX_KUNDEN} Kunden.")
    rows: list[list[Fraction]] = []
    for i, k in enumerate(kunden, 1):
        teile = [t.strip() for t in k.replace(";", ",").split(",") if t.strip()]
        if not teile:
            raise ValueError(f"Kunde {i} hat keine Zahl.")
        if len(teile) > C.HAND_MAX_CONTAINER:
            raise ValueError(f"Kunde {i}: höchstens {C.HAND_MAX_CONTAINER} Container.")
        try:
            rows.append([Fraction(t) for t in teile])
        except (ValueError, ZeroDivisionError) as err:
            raise ValueError(f"Kunde {i}: keine Zahl in „{k}“.") from err
    return rows


def random_hand(seed: int) -> str:
    """Zufallsdatensatz aus dem Zufalls-Seed: 3 bis 6 Kunden, 2 bis 4 Container je Kunde, ganze Zahlen 0 bis 19 (wie in der Orakel-Messreihe)."""
    g = SplitMix64(seed)
    m, k = 3 + g.below(4), 2 + g.below(3)
    return " | ".join(", ".join(str(g.below(20)) for _ in range(k)) for _ in range(m))


def hand_text(rows) -> str:
    return " | ".join(", ".join(frac_str(v) for v in r) for r in rows)


@dataclass(frozen=True)
class HandAnalysis:
    rows: tuple
    m: int
    n: int
    ausgewogen: bool
    gm: Fraction
    kundenmittel: tuple
    anova: dict | None             # exakt in Brüchen, nur ausgewogen
    reml: MX.Fit
    ml: MX.Fit
    se_naiv: float
    se_gemischt: float             # REML
    halb_naiv: float               # Halbbreite 95 %: naiv mit t(N−1)
    halb_gemischt: float           # gemischt (REML) mit t(m−1)
    rho_plot: np.ndarray           # Gitter der Intraklassenkorrelation ρ = λ/(1+λ) für die Kriteriumskurve
    krit_reml: np.ndarray
    krit_ml: np.ndarray
    hinweis: str


def hand_analysis(rows) -> HandAnalysis:
    rows = [list(r) for r in rows]
    m = len(rows)
    n = sum(len(r) for r in rows)
    y = np.array([float(v) for r in rows for v in r])
    g = np.repeat(np.arange(m), [len(r) for r in rows])
    X = np.ones((n, 1))
    mod = MX.MixedModel(X, y, g, m)
    reml, ml = mod.fit(True), mod.fit(False)
    gm = sum(sum(r) for r in rows) / Fraction(n)
    kundenmittel = tuple(sum(r) / len(r) for r in rows)
    ausgewogen = len({len(r) for r in rows}) == 1 and len(rows[0]) >= 2
    anova = MX.anova_balanced(rows) if ausgewogen else None
    s2_naiv = float(np.sum((y - y.mean()) ** 2)) / (n - 1)
    se_naiv = math.sqrt(s2_naiv / n)
    rho_plot = np.linspace(0.0, 0.98, 99)
    lam = rho_plot / (1.0 - rho_plot)
    if reml.entartet:
        kr = km = np.zeros_like(lam)                       # alle Werte gleich: das Kriterium ist nicht definiert (Q = 0), die Kurve bleibt leer
    else:
        kr, km = mod.crit_grid(lam, True), mod.crit_grid(lam, False)
    hinweis = reml.hinweis
    se_g = reml.se(0)
    return HandAnalysis(tuple(map(tuple, rows)), m, n, ausgewogen, gm, kundenmittel, anova, reml, ml, se_naiv, se_g,
                        V.t_quantil(n - 1) * se_naiv, V.t_quantil(m - 1) * se_g, rho_plot, kr - kr.min(), km - km.min(), hinweis)


def verdict_hand(a: HandAnalysis) -> tuple[str, str]:
    """Verdikt zum Handbeispiel: Zahlen aus der Bruchrechnung (ausgewogen) bzw. der profilierten Rechnung (sonst)."""
    daten = " | ".join(", ".join(frac_str(v) for v in r) for r in a.rows)
    if a.reml.entartet:
        return "warning", f"Daten {daten}: alle Werte sind gleich, es gibt keine Streuung zu schätzen."
    if not a.reml.getrennt:
        return "warning", f"Daten {daten}: {a.hinweis} Es gibt keinen Schätzwert für σ_u², nur die naive Rechnung (Standardfehler des Mittels {de(a.se_naiv, 2)})."
    if a.reml.rand:
        teil = ""
        if a.anova is not None:
            teil = (f"MSB = {frac_str(a.anova['msb'])}, MSW = {frac_str(a.anova['msw'])}: der Schätzer (MSB − MSW)/k ist bei REML {frac_zahl(a.anova['reml_roh'])} und bei ML "
                    f"{frac_zahl(a.anova['ml_roh'])}, also negativ. ")
        return "warning", (f"Daten {daten}: {teil}Die Anpassung liegt auf dem Rand σ̂_u² = 0 (ein Rand, kein Absturz): der Standardfehler des gemischten Modells ist dann der der naiven Rechnung "
                           f"({de(a.se_gemischt, 2)}), aber das t(m−1)-Intervall hat mit {de(a.halb_gemischt, 2)} Halbbreite (m = {a.m}) eine andere Breite als das naive mit {de(a.halb_naiv, 2)} (t(N−1)).")
    teil = ""
    if a.anova is not None:
        teil = (f"MSB = {frac_str(a.anova['msb'])}, MSW = {frac_str(a.anova['msw'])}; REML σ̂_u² = {frac_zahl(a.anova['reml'])}, ML {frac_zahl(a.anova['ml'])}. ")
    else:
        teil = f"REML σ̂_u² = {de(a.reml.sigma_u2, 3)}, ML {de(a.ml.sigma_u2, 3)} (ungleich große Kunden: keine geschlossene Form, profilierte Rechnung). "
    faktor = a.se_gemischt / a.se_naiv
    return "info", (f"Daten {daten}: {teil}Der naive Standardfehler des Mittels ({de(a.se_naiv, 2)}) ist um den Faktor {de(faktor, 2)} kleiner als der des gemischten Modells "
                    f"({de(a.se_gemischt, 2)}); mit nur {a.m} Kunden ist das t({a.m - 1})-Intervall ±{de(a.halb_gemischt, 2)} breit (naiv ±{de(a.halb_naiv, 2)}).")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: wie viele unabhängige Container?
# ---------------------------------------------------------------------------------------------------------------------------------------------

DESIGN_ANZAHL = 300
LIVE_BUDGET_MS = 2500.0          # Rechenzeit, die eine Live-Simulation höchstens brauchen soll (Obergrenze der Wiederholungen)


def live_reps(m: int, k: int, reps: int) -> int:
    """Wiederholungen der Live-Simulation: die gewählte Zahl, begrenzt so, dass die Rechnung grob höchstens LIVE_BUDGET_MS braucht
    (gemessen: etwa 1,4 ms + 0,4 µs je Container und Historie; der Wert 1000 bei m = 48 und 120 Containern je Kunde wäre sonst fast 5 s)."""
    je_historie = 1.4 + 0.0004 * m * k
    return max(20, min(int(reps), int(LIVE_BUDGET_MS / je_historie)))


def analyse_design(s: Settings, anzahl: int = DESIGN_ANZAHL) -> dict:
    art = "zoll" if s.zoll else "rv"
    r = V.design_effects(s.m, s.k, s.su, art, s.art, s.seed, anzahl)
    r.update(art=art, m=s.m, k=s.k, su=s.su)
    return r


def verdict_design(d: dict) -> tuple[str, str]:
    """Verdikt zum Designeffekt für das gewählte Merkmal (Kundenmerkmal oder Zollhalt)."""
    n, m, k = d["n"], d["m"], d["k"]
    kopf = f"m = {m}, im Mittel {k} Container je Kunde, ρ = {de(d['rho'], 3)}: {n} Container."
    if d["art"] == "rv":
        if d["rho"] == 0:
            return "info", f"{kopf} Ohne Kundeneffekt ist der Designeffekt {de(d['median'], 2)}: alle {n} Container zählen für den Rahmenvertrag voll."
        return "warning", (f"{kopf} Designeffekt für den Rahmenvertrag {de(d['median'], 2)} (exakt, im Median über {d['anzahl']} Zuordnungen, 5 bis 95 % der Zuordnungen: "
                           f"{de(d['q05'], 2)} bis {de(d['q95'], 2)}; die Formel 1 + (k̄ − 1)ρ mit mittlerer Kundengröße gibt {de(d['formel'], 2)}): die Tabelle enthält so viel Information wie etwa "
                           f"{de(d['wirksam_median'], 0)} unabhängige Container.")
    return "info", (f"{kopf} Zollhalt (Merkmal im Kunden, je Container zufällig): wahre Varianz des OLS-Koeffizienten gegen die erwartete naive Varianz im Median {de(d['median_erwartet'], 2)} "
                    f"(über {d['anzahl']} Zuordnungen), gegen die naive Formel mit der Gesamtvarianz {de(d['median'], 2)} (5 bis 95 % der Zuordnungen: {de(d['q05'], 2)} bis {de(d['q95'], 2)}). "
                    f"Ein Verhältnis von 1 hieße: die naive Rechnung schätzt die Genauigkeit richtig; die wirksame Zahl ist dann {de(d['wirksam_median'], 0)} von {n} Containern. "
                    f"Das ist der Negativbefund dieser Demo: für ein Merkmal, das im Kunden wechselt, ist die Zeilenzahl nicht das Problem (naives Intervall für den Zollhalt: Experiment 3).")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3 und 4: Überdeckung, Rand
# ---------------------------------------------------------------------------------------------------------------------------------------------

LABELS_RV = {"naiv": "naiv (OLS, t(N − p))", "cr1z": "CR1, Normalquantil", "cr1t": "CR1, t(m − 1)", "mlz": "ML, Normalquantil", "rez": "REML, Normalquantil",
             "ret": "REML, t(m − 1)", "kmm": "Kunden-Mittel-Methode, t(m − 2)"}
LABELS_ZOLL = {"naiv": "naiv (OLS, t(N − p))", "cr1z": "CR1, Normalquantil", "cr1t": "CR1, t(m − 1)", "mlz": "ML, Normalquantil", "rez": "REML, Normalquantil",
               "ret": "REML, t(m − 1)", "fe": "feste Kunden-Effekte, t(N − m − 2)"}
WIDTH_KEY = {"naiv": "w_naiv", "cr1t": "w_cr1t", "rez": "w_rez", "ret": "w_ret", "kmm": "w_kmm", "fe": "w_fe"}


def coverage_table(row: dict, name: str = "rv") -> list[dict]:
    """Je Verfahren: Überdeckung (Anteil), Monte-Carlo-Standardfehler, Zahl der Historien und (soweit gemessen) mittlere Breite im Verhältnis zur naiven."""
    labels = LABELS_RV if name == "rv" else LABELS_ZOLL
    out = []
    for key, label in labels.items():
        p = V.share(row, name, key)
        wk = WIDTH_KEY.get(key)
        w = row["width"][name].get(wk) if wk else None
        ratio = w / row["width"][name]["w_naiv"] if w is not None and row["width"][name]["w_naiv"] else None
        out.append(dict(key=key, label=label, share=p, se=V.se_share(p, row["ok"]), ok=row["ok"], breite=ratio))
    return out


def unter_soll(p: float, n: int, soll: float = 0.95) -> bool:
    """Liegt der Anteil p aus n Historien mehr als zwei Standardfehler (bei Soll) unter dem Sollwert?"""
    return p < soll - 2.0 * V.se_share(soll, n)


def ueber_soll(p: float, n: int, soll: float = 0.95) -> bool:
    return p > soll + 2.0 * V.se_share(soll, n)


def verdict_coverage(row: dict, quelle: str) -> tuple[str, str]:
    """Verdikt: Überdeckung der Verfahren für das Kundenmerkmal. `quelle`: „vorgerechnet“ oder „live“."""
    t = {r["key"]: r for r in coverage_table(row, "rv")}
    n = row["ok"]
    kopf = (f"Rahmenvertrag, m = {row['m']}, im Mittel {row['kbar']} Container je Kunde, ρ = {de(row['rho'], 3)}, {n} Historien ({quelle}; Monte-Carlo-Fehler etwa "
            f"{pp(V.se_share(0.95, n))} Prozentpunkte bei 95 %): ")
    zahlen = (f"naiv {de_pct(t['naiv']['share'])}, CR1 mit z {de_pct(t['cr1z']['share'])}, mit t(m−1) {de_pct(t['cr1t']['share'])}, ML mit z {de_pct(t['mlz']['share'])}, "
              f"REML mit z {de_pct(t['rez']['share'])}, mit t(m−1) {de_pct(t['ret']['share'])}, Kunden-Mittel-Methode {de_pct(t['kmm']['share'])}.")
    zu_knapp = [t[k]["label"] for k in ("naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "kmm") if unter_soll(t[k]["share"], n)]
    if not unter_soll(t["naiv"]["share"], n):
        return "success", kopf + zahlen + " Das naive Intervall liegt hier nicht unter dem Sollwert 95 %: in dieser Einstellung gibt es für diese Messung keinen Clusterbefund."
    folge = ("Unter 95 % (mehr als zwei Standardfehler): " + "; ".join(zu_knapp) + ".") if zu_knapp else ""
    return "warning", kopf + zahlen + " " + folge


def verdict_rand(row: dict) -> tuple[str, str]:
    """Verdikt: Anteil der Historien mit σ̂_u² = 0 (Randlösung) und Überdeckung des REML-t-Intervalls bedingt darauf."""
    n = row["ok"]
    sr, sm = row["rand_re"], row["rand_ml"]
    kopf = f"σ_u = {de(row['su'], 2)}, m = {row['m']}, {n} Historien: Rand σ̂_u² = 0 bei REML in {sr} ({de_pct(sr / n)}), bei ML in {sm} ({de_pct(sm / n)}). "
    if sr == 0:
        return "info", kopf + "Hier liegt keine Historie am Rand."
    rd, inn = row["bedingt"]["rand"], row["bedingt"]["inner"]
    teil = ""
    if rd[0] >= 30:
        teil = (f"Am Rand deckt das REML-t(m−1)-Intervall für den Rahmenvertrag {de_pct(rd[1] / rd[0])} ({rd[0]} Historien), das naive {de_pct(rd[2] / rd[0])}, CR1 mit t(m−1) "
                f"{de_pct(rd[3] / rd[0])}")
        teil += f"; nicht am Rand {de_pct(inn[1] / inn[0])} ({inn[0]} Historien). " if inn[0] else ". "
    else:
        teil = f"Am Rand {'liegt nur' if rd[0] == 1 else 'liegen nur'} {rd[0]} {'Historie' if rd[0] == 1 else 'Historien'}, zu wenig zum Deuten. "
    if row["su"] == 0:
        return "info", kopf + teil + "Ohne Kundeneffekt (σ_u = 0) ist der Rand das richtige Ergebnis: hier gibt es nichts zu finden und nichts wird verfehlt."
    return "info", kopf + teil + "Ein Rand ist ein Messergebnis, kein Absturz."


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Vorgerechnete Messreihe
# ---------------------------------------------------------------------------------------------------------------------------------------------

_FROZEN_CACHE: dict = {}


def load_frozen() -> dict:
    """data/clg_frozen.json (erzeugt von tools/clg_make_frozen.py). Fehlt die Datei, ist die Messreihe leer (die App rechnet dann nur live)."""
    if "d" not in _FROZEN_CACHE:
        try:
            _FROZEN_CACHE["d"] = json.loads(C.FROZEN_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _FROZEN_CACHE["d"] = {"rows": [], "su2_merkmal": {}}
    return _FROZEN_CACHE["d"]


def find_frozen_row(m: int, k: int, su: float, art: str) -> dict | None:
    for r in load_frozen()["rows"]:
        if r["m"] == m and r["kbar"] == k and abs(r["su"] - su) < 1e-9 and r["mode"] == art:
            return r
    return None


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: die zwölf Kunden des Generators, Genauigkeit von σ̂_u²
# ---------------------------------------------------------------------------------------------------------------------------------------------

def generator_summary(R: int = 20000, seed: int = 5000) -> dict:
    """Die zwölf Kundeneffekte des Generators (fest): realisierte gegen nominale Streuung und Intraklassenkorrelation; Anteil von zwölf Ziehungen mit nominal 0,30,
    deren Stichproben-Streuung mindestens so groß ist."""
    uk = np.array(V.GENERATOR_UK)
    sd, var = float(np.std(uk, ddof=1)), float(np.var(uk, ddof=1))
    s2 = V.S_IMPORT ** 2
    sds = V.sd_of_normal_draws(12, 0.30, R, seed)
    return dict(sd=sd, var=var, icc_real=var / (var + s2), icc_nom=0.09 / (0.09 + s2), anteil=float(np.mean(sds >= sd)), R=R,
                mittel=float(sds.mean()), q05=float(np.quantile(sds, 0.05)), q95=float(np.quantile(sds, 0.95)))


def verdict_generator(g: dict) -> tuple[str, str]:
    return "info", (f"Die zwölf Kunden des Generators haben die Streuung {de(g['sd'], 3)} statt nominal 0,30 (Intraklassenkorrelation {de(g['icc_real'], 3)} statt {de(g['icc_nom'], 3)}). "
                    f"Zwölf frische Ziehungen mit 0,30 geben als Stichproben-Streuung im Mittel {de(g['mittel'], 3)} (5 bis 95 %: {de(g['q05'], 3)} bis {de(g['q95'], 3)}); in "
                    f"{de_pct(g['anteil'])} der {g['R']} Ziehungen ist sie mindestens {de(g['sd'], 3)}: die Abweichung des Generators ist eine einzelne Ziehung, keine Auffälligkeit.")


def su2_grid_table(frozen: dict | None = None) -> list[dict]:
    """Zeilen der vorgerechneten Messreihe für Experiment 5: Mittel und Streuung von σ̂_u² (REML) über Kundenzahl und Container je Kunde (σ_u = 0,30, σ_u² = 0,09)."""
    frozen = frozen or load_frozen()
    out = []
    for m in C.SU2_GRID_M:
        for k in C.SU2_GRID_K:
            r = next((x for x in frozen["rows"] if x["m"] == m and x["kbar"] == k and abs(x["su"] - 0.30) < 1e-9 and x["mode"] == "half"), None)
            if r:
                out.append(dict(m=m, k=k, mittel=r["su2_mean"], sd=r["su2_sd"], grenze=MX.sd_grenze_viele_container(m, 0.09), ok=r["ok"]))
    return out


def verdict_su2(live: dict, grid: list[dict]) -> tuple[str, str]:
    """Verdikt zur Genauigkeit der Varianzkomponente: Live-Reihe (mit oder ohne Kundenmerkmal) und vorgerechnete Tabelle."""
    wahr = live["su"] ** 2
    kopf = (f"m = {live['m']}, im Mittel {live['kbar']} Container je Kunde, σ_u² = {de(wahr, 3)}, {live['ok']} Historien: σ̂_u² im Mittel {de(live['mean'], 3)} mit Streuung {de(live['sd'], 3)}, "
            f"Rand in {live['rand']} Historien ({'mit' if live['mit_merkmal'] else 'ohne'} Rahmenvertrag im Modell). ")
    if not grid:
        return "info", kopf
    z = {(g["m"], g["k"]): g for g in grid}
    teile = []
    if (12, 30) in z and (12, 300) in z and (48, 30) in z:
        teile.append(f"Vorgerechnet (σ_u = 0,30, je {z[(12, 30)]['ok']} Historien): Streuung von σ̂_u² bei m = 12 {de(z[(12, 30)]['sd'], 3)} (30 Container je Kunde) und {de(z[(12, 300)]['sd'], 3)} (300), "
                     f"bei m = 48 {de(z[(48, 30)]['sd'], 3)}: mehr Kunden helfen, mehr Container je Kunde kaum.")
    return "info", kopf + " ".join(teile)


def hand_table(a: HandAnalysis) -> list[dict]:
    """Tabelle zum Handbeispiel: Schätzer von σ_u², σ² und ρ für REML und ML; im ausgewogenen Fall als Bruch (mit Dezimalwert), sonst als Zahl."""
    if a.anova is not None:
        x = a.anova
        rand_r, rand_m = x["reml_roh"] <= 0, x["ml_roh"] <= 0
        return [dict(Größe="σ̂_u² (Kundenstreuung)", REML=frac_zahl(x["reml"]) + (" (Rand)" if rand_r else ""), ML=frac_zahl(x["ml"]) + (" (Rand)" if rand_m else "")),
                dict(Größe="σ̂² (Streuung im Kunden)", REML=frac_zahl(x["s2_reml"]), ML=frac_zahl(x["s2_ml"])),
                dict(Größe="ρ̂ = σ̂_u²/(σ̂_u² + σ̂²)", REML=frac_zahl(x["rho_reml"]), ML=frac_zahl(x["rho_ml"]))]
    return [dict(Größe="σ̂_u² (Kundenstreuung)", REML=de(a.reml.sigma_u2, 3) + (" (Rand)" if a.reml.rand else ""), ML=de(a.ml.sigma_u2, 3) + (" (Rand)" if a.ml.rand else "")),
            dict(Größe="σ̂² (Streuung im Kunden)", REML=de(a.reml.sigma2, 3), ML=de(a.ml.sigma2, 3)),
            dict(Größe="ρ̂ = σ̂_u²/(σ̂_u² + σ̂²)", REML=de(a.reml.rho, 3), ML=de(a.ml.rho, 3))]
