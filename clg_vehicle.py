"""Vehikel: Import-Container der Verweilzeit-Welt, Kunden als Cluster, log-Verweilzeit als Antwort; Simulation der Intervalle.

Herkunft: der Import-Zweig des Containergenerators der `verweilzeit-demo` (Datei `vwz_generator.py`, Funktion `log_median_and_s`) ist hier **neu geschrieben**
(nichts importiert, nichts kopiert; `tests/test_vehicle.py` rechnet die Konstanten nach). Modell einer Zeile (Import-Container, Kunde k):

    y = ln 3,5 + 0,7·Zoll + 0,3·Wochenende + β_R·Rahmenvertrag_k + u_k + 0,55·ε,   u_k ~ N(0, σ_u²), ε ~ N(0, 1).

Zollhalt hat die Wahrscheinlichkeit 0,25, Wochenende 2/7, wie im Generator. **Rahmenvertrag ist erfunden** (ein Merkmal, das nur der Kunde hat; die Wirkung
β_R = −0,30 und die Aufteilung sind Annahmen). Die Streuung 0,55 ist beim Import konstant. **Die Kundeneffekte werden in jeder Wiederholung frisch gezogen**
(Annahme: die Kunden sind eine Stichprobe aus einer Kundschaft); die zwölf festen Kundeneffekte des Generators sind eine einzige Ziehung (`GENERATOR_UK`).
Der Kunde jedes Containers wird gleichverteilt gezogen, die Kunden sind also ungleich groß. Zufall nur über SplitMix64 (`clg_rng`)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats

import clg_mixed as MX
from clg_rng import exact_log, normal_from_uniform, splitmix_array, uniform_array

LN35 = exact_log(3.5)
S_IMPORT = 0.55
P_ZOLL = 0.25
P_WE = 2.0 / 7.0
BETA_R = -0.30
BETA_Z = 0.7
BETA_W = 0.3
BETA_TRUE = np.array([LN35, BETA_R, BETA_Z, BETA_W])      # Spalten von X: Achsenabschnitt, Rahmenvertrag (Kunde), Zollhalt, Wochenende

# Die zwölf Kundeneffekte des Generators (Zufalls-Seed 12345 des numpy-Generators dort, Streuung 0,30): hier als feste Zahlen übernommen, damit die
# Demo nichts aus der anderen Demo importiert. Quelle: numpy.random.default_rng(12345).normal(0.0, 0.30, 12), so wie `kunden_effekte()` in vwz_generator.py.
GENERATOR_UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
                -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)

MODES = ("half", "two", "cont")                         # Verteilung des Kundenmerkmals: halb/halb, nur 2 Kunden, stetig


def kundenmerkmal(u_unif_normals: np.ndarray, m: int, mode: str) -> np.ndarray:
    """Wert des Kundenmerkmals je Kunde: 'half' die ersten m//2 Kunden = 1, 'two' die ersten zwei, 'cont' standardnormal (Wirkung β_R je Standardabweichung)."""
    if mode == "half":
        return (np.arange(m) < m // 2).astype(float)
    if mode == "two":
        return (np.arange(m) < 2).astype(float)
    if mode == "cont":
        return u_unif_normals
    raise ValueError(f"unbekannte Art des Kundenmerkmals: {mode}")


@dataclass(frozen=True)
class History:
    X: np.ndarray       # Spalten: 1, Rahmenvertrag, Zollhalt, Wochenende
    y: np.ndarray
    g: np.ndarray       # Kundennummer je Container (0 … m_eff−1, ohne leere Kunden)
    m: int              # Zahl der Kunden mit mindestens einem Container
    u: np.ndarray       # gezogene Kundeneffekte (alle m_ziel Kunden)


def draw_history(seed: int, m: int, kbar: int, su: float, mode: str = "half", u_fixed=None, beta_r: float = BETA_R) -> History:
    """Eine Historie: m Kunden, N = m·k̄ Import-Container. Ziehungsreihenfolge (fest, damit Läufe über Plattformen gleich sind): Kunde je Container,
    Zollhalt, Wochenende, ε (Box-Muller aus je zwei Gleichverteilten), u_k, stetiges Kundenmerkmal. `u_fixed`: feste Kundeneffekte statt frischer Ziehung."""
    n = m * kbar
    u01 = uniform_array(seed, 5 * n + 4 * m)
    g_raw = np.minimum((u01[:n] * m).astype(int), m - 1)
    zoll = (u01[n:2 * n] < P_ZOLL).astype(float)
    we = (u01[2 * n:3 * n] < P_WE).astype(float)
    eps = normal_from_uniform(u01[3 * n:5 * n:2], u01[3 * n + 1:5 * n:2])
    base = 5 * n
    if u_fixed is None:
        u = su * normal_from_uniform(u01[base:base + 2 * m:2], u01[base + 1:base + 2 * m:2])
    else:
        u = np.asarray(u_fixed, dtype=float)
    rv_c = normal_from_uniform(u01[base + 2 * m:base + 4 * m:2], u01[base + 2 * m + 1:base + 4 * m:2])
    rvk = kundenmerkmal(rv_c, m, mode)
    rv = rvk[g_raw]
    y = LN35 + BETA_Z * zoll + BETA_W * we + beta_r * rv + u[g_raw] + S_IMPORT * eps
    g, m_eff = MX.compress_groups(g_raw)
    X = np.column_stack([np.ones(n), rv, zoll, we])
    return History(X, y, g, m_eff, u)


@lru_cache(maxsize=None)
def t_quantil(df: int, level: float = 0.95) -> float:
    return float(stats.t.ppf(0.5 + level / 2.0, df))


Z975 = float(stats.norm.ppf(0.975))

METHODEN_RV = ("naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "kmm")                 # Verfahren für das Kundenmerkmal
METHODEN_ZOLL = ("naiv", "cr1t", "rez", "ret", "fe")                                # Verfahren für das Merkmal innerhalb der Kunden


def analyse_history(h: History, beta_true=BETA_TRUE) -> dict:
    """Intervalle für Rahmenvertrag (Spalte 1) und Zollhalt (Spalte 2) mit allen Verfahren: trifft das 95-%-Intervall den wahren Wert, dazu die Breite.
    naiv: OLS mit t(N−p); CR1 mit z und t(m−1); ML mit z; REML mit z und t(m−1); KMM: Kunden-Mittel-Methode t(m−2); fe: feste Kunden-Effekte (nur Zollhalt)."""
    X, y, g, m = h.X, h.y, h.g, h.m
    n, p = X.shape
    b, Vn, dfn = MX.ols(X, y)
    Vc = MX.cr1_cov(X, y, g, m, b)
    mod = MX.MixedModel(X, y, g, m)
    fm, fr = mod.fit(False), mod.fit(True)
    tm, tn = t_quantil(m - 1), t_quantil(dfn)
    out: dict = {}
    for j, name in ((1, "rv"), (2, "zoll")):
        d = beta_true[j]
        se_n, se_c, se_ml, se_re = np.sqrt(Vn[j, j]), np.sqrt(Vc[j, j]), fm.se(j), fr.se(j)
        r = dict(naiv=abs(b[j] - d) < tn * se_n, cr1z=abs(b[j] - d) < Z975 * se_c, cr1t=abs(b[j] - d) < tm * se_c,
                 mlz=abs(fm.beta[j] - d) < Z975 * se_ml, rez=abs(fr.beta[j] - d) < Z975 * se_re, ret=abs(fr.beta[j] - d) < tm * se_re,
                 w_naiv=2 * tn * se_n, w_cr1t=2 * tm * se_c, w_rez=2 * Z975 * se_re, w_ret=2 * tm * se_re)
        out[name] = r
    kb, kse, kq = MX.kmm_interval(X[:, 1], y, g, m)
    out["rv"]["kmm"] = abs(kb - beta_true[1]) < kq * kse
    out["rv"]["w_kmm"] = 2 * kq * kse
    fb, fse, fdf = MX.fixed_effects_coef(X[:, 2:4], y, g, m, 0)
    out["zoll"]["fe"] = abs(fb - beta_true[2]) < t_quantil(fdf) * fse
    out["zoll"]["w_fe"] = 2 * t_quantil(fdf) * fse
    out["rand_ml"], out["rand_re"] = fm.rand, fr.rand
    out["su2_re"], out["su2_ml"] = fr.sigma_u2, fm.sigma_u2
    return out


def simulate(m: int, kbar: int, su: float, R: int, seed: int, mode: str = "half", u_fixed=None) -> dict:
    """R Historien, Zähleinheit Wiederholung (eine Historie = eine Tabelle aus m Kunden). Rückgabe: Zählungen (Zahl der Historien, in denen das Intervall den
    wahren Wert trifft) je Verfahren, mittlere Breiten, Anteil Rand, Überdeckung bedingt auf Rand, Mittel und Streuung von σ̂_u²."""
    seeds = splitmix_array(seed, R)
    keys = {"rv": ("naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "kmm"), "zoll": ("naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "fe")}
    cnt = {nm: {k: 0 for k in ks} for nm, ks in keys.items()}
    wid = {nm: {k: 0.0 for k in ("w_naiv", "w_cr1t", "w_rez", "w_ret", "w_kmm" if nm == "rv" else "w_fe")} for nm in keys}
    rand_ml = rand_re = ok = 0
    su2: list[float] = []
    su2_ml: list[float] = []
    cond = {"rand": [0, 0, 0, 0], "inner": [0, 0, 0, 0]}          # je: Zahl, REML-t trifft, naiv trifft, CR1-t trifft
    for s in seeds:
        h = draw_history(int(s), m, kbar, su, mode, u_fixed)
        try:
            o = analyse_history(h)
        except (ValueError, np.linalg.LinAlgError):
            continue
        ok += 1
        for nm, ks in keys.items():
            for k in ks:
                cnt[nm][k] += int(o[nm][k])
            for k in wid[nm]:
                wid[nm][k] += float(o[nm][k])
        rand_ml += int(o["rand_ml"])
        rand_re += int(o["rand_re"])
        su2.append(o["su2_re"])
        su2_ml.append(o["su2_ml"])
        c = cond["rand" if o["rand_re"] else "inner"]
        c[0] += 1
        c[1] += int(o["rv"]["ret"])
        c[2] += int(o["rv"]["naiv"])
        c[3] += int(o["rv"]["cr1t"])
    rho = su ** 2 / (su ** 2 + S_IMPORT ** 2)
    return dict(m=m, kbar=kbar, su=su, mode=mode, R=R, ok=ok, seed=seed, rho=rho, cover=cnt,
                width={nm: {k: v / max(ok, 1) for k, v in w.items()} for nm, w in wid.items()},
                rand_ml=rand_ml, rand_re=rand_re, bedingt=cond,
                su2_mean=float(np.mean(su2)) if su2 else float("nan"), su2_sd=float(np.std(su2)) if su2 else float("nan"),
                su2_ml_mean=float(np.mean(su2_ml)) if su2_ml else float("nan"))


def share(row: dict, name: str, method: str) -> float:
    """Überdeckung (Anteil der Historien) eines Verfahrens aus einer Zeile von `simulate`."""
    return row["cover"][name][method] / row["ok"]


def se_share(p: float, n: int) -> float:
    """Monte-Carlo-Standardfehler eines Anteils."""
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1)))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Designs (ohne Antwort): wie viele unabhängige Container hat die Tabelle?
# ---------------------------------------------------------------------------------------------------------------------------------------------

def draw_design(seed: int, m: int, kbar: int, art: str, mode: str = "half"):
    """Zufällige Zuordnung der N = m·k̄ Container zu Kunden (gleichverteilt, ungleich große Kunden) und die Spalte des Merkmals:
    art 'rv' (Kundenmerkmal Rahmenvertrag, Art nach `mode`) oder 'zoll' (Zollhalt je Container). Rückgabe X, g, m_eff."""
    n = m * kbar
    u01 = uniform_array(seed, n + 2 * m + 2 * m + n)
    g_raw = np.minimum((u01[:n] * m).astype(int), m - 1)
    if art == "rv":
        rvk = kundenmerkmal(normal_from_uniform(u01[n:n + 2 * m:2], u01[n + 1:n + 2 * m:2]), m, mode)
        col = rvk[g_raw]
    else:
        col = (u01[n + 4 * m:n + 4 * m + n] < P_ZOLL).astype(float)
    g, m_eff = MX.compress_groups(g_raw)
    return np.column_stack([np.ones(n), col]), g, m_eff


def design_effects(m: int, kbar: int, su: float, art: str, mode: str, seed: int, R: int) -> dict:
    """Exakter Designeffekt über R zufällige Zuordnungen (Matrix-Sandwich in Kundensummen) gegen die Formel mit mittlerer Kundengröße.
    Rückgabe: Median, 5-%- und 95-%-Quantil des Designeffekts (gegen die Gesamtvarianz), Median gegen die erwartete naive Varianz, Formel, wirksame Zahl."""
    s2, su2 = S_IMPORT ** 2, su ** 2
    rho = su2 / (su2 + s2)
    seeds = splitmix_array(seed, R)
    deff, deff_e, wirksam = [], [], []
    for s in seeds:
        X, g, me = draw_design(int(s), m, kbar, art, mode)
        if np.ptp(X[:, 1]) == 0:
            continue
        d = MX.design_effect(X, g, me, su2, s2, 1)
        deff.append(d["deff"])
        deff_e.append(d["deff_erwartet"])
        wirksam.append(d["wirksam"])
    deff = np.array(deff)
    return dict(median=float(np.median(deff)), q05=float(np.quantile(deff, 0.05)), q95=float(np.quantile(deff, 0.95)), median_erwartet=float(np.median(deff_e)),
                formel=MX.design_effect_formula(kbar, rho), wirksam_median=float(np.median(wirksam)), n=m * kbar, rho=rho, anzahl=len(deff))


def sd_of_normal_draws(m: int, sigma: float, R: int, seed: int) -> np.ndarray:
    """Stichproben-Standardabweichung (n−1) von je m Normalzahlen mit Streuung `sigma`, R Wiederholungen (vektorisiert, SplitMix64)."""
    u = uniform_array(seed, 2 * m * R)
    z = sigma * normal_from_uniform(u[0::2], u[1::2]).reshape(R, m)
    return z.std(axis=1, ddof=1)


def su2_series(m: int, kbar: int, su: float, R: int, seed: int, mode: str = "half", mit_merkmal: bool = True, u_fixed=None) -> dict:
    """σ̂_u² (REML) über R Historien, mit dem Kundenmerkmal im Modell (Spalten 1, Rahmenvertrag, Zollhalt, Wochenende) oder ohne (nur 1, Zollhalt, Wochenende).
    Rückgabe: Mittel, Streuung, Zahl der Randlösungen, Zahl der Historien und die Werte selbst (`werte`)."""
    seeds = splitmix_array(seed, R)
    werte: list[float] = []
    rand = 0
    for s in seeds:
        h = draw_history(int(s), m, kbar, su, mode, u_fixed)
        X = h.X if mit_merkmal else h.X[:, [0, 2, 3]]
        try:
            f = MX.MixedModel(X, h.y, h.g, h.m).fit(True)
        except (ValueError, np.linalg.LinAlgError):
            continue
        werte.append(f.sigma_u2)
        rand += int(f.rand)
    w = np.array(werte)
    return dict(m=m, kbar=kbar, su=su, mode=mode, R=R, ok=len(w), seed=seed, mit_merkmal=mit_merkmal, mean=float(w.mean()), sd=float(w.std()), rand=rand,
                werte=[float(v) for v in w])


def history_intervals(h: History, j: int = 1) -> list[dict]:
    """Die 95-%-Intervalle aller Verfahren für den Koeffizienten j (Spalte 1 = Rahmenvertrag, 2 = Zollhalt) **einer** Historie: Schätzwert, Grenzen, trifft der wahre Wert?"""
    X, y, g, m = h.X, h.y, h.g, h.m
    b, Vn, dfn = MX.ols(X, y)
    Vc = MX.cr1_cov(X, y, g, m, b)
    mod = MX.MixedModel(X, y, g, m)
    fm, fr = mod.fit(False), mod.fit(True)
    tm, tn = t_quantil(m - 1), t_quantil(dfn)
    d = BETA_TRUE[j]
    spec = [("naiv", "naiv (OLS)", b[j], tn * np.sqrt(Vn[j, j])), ("cr1z", "CR1, z", b[j], Z975 * np.sqrt(Vc[j, j])), ("cr1t", "CR1, t(m−1)", b[j], tm * np.sqrt(Vc[j, j])),
            ("mlz", "ML, z", fm.beta[j], Z975 * fm.se(j)), ("rez", "REML, z", fr.beta[j], Z975 * fr.se(j)), ("ret", "REML, t(m−1)", fr.beta[j], tm * fr.se(j))]
    if j == 1:
        kb, kse, kq = MX.kmm_interval(X[:, 1], y, g, m)
        spec.append(("kmm", "Kunden-Mittel-Methode", kb, kq * kse))
    else:
        fb, fse, fdf = MX.fixed_effects_coef(X[:, 2:4], y, g, m, 0)
        spec.append(("fe", "feste Kunden-Effekte", fb, t_quantil(fdf) * fse))
    return [dict(key=k, label=lab, est=float(e), lo=float(e - hw), hi=float(e + hw), trifft=bool(abs(e - d) < hw)) for k, lab, e, hw in spec]
