"""Kern: lineare Modelle mit Kunden als Cluster (Zufallsachsenabschnitt), eigene Rechnung nur aus numpy und scipy.

Modell: y = X·β + u_k(i) + ε, u_k ~ N(0, σ_u²), ε ~ N(0, σ²); Kunden k = 0 … m−1 (alle nicht leer). Kovarianz V = σ²·(I + λ·Z·Z'), λ = σ_u²/σ²,
Z die Zuordnung der Zeilen zu Kunden. Alle Größen stehen in Kundensummen (Woodbury): für festes λ ist die verallgemeinerte Kleinste-Quadrate-Rechnung
eine p×p-Rechnung, und das profilierte Kriterium hängt nur von λ ab (Bates u. a. 2015, Gl. 34 und 41 nach dem Quellenbericht, hier nachgerechnet).

Enthalten:
  * `MixedModel`: profiliertes ML- und REML-Kriterium, seine Ableitung nach λ (analytisch), die Anpassung über λ ≥ 0 mit Randlösung λ = 0.
    **Der Rand wird über die Ableitung bei λ = 0 festgestellt** (≥ 0 heißt: das Kriterium steigt von λ = 0 aus), nicht über das Gitter allein:
    liegt das Optimum zwischen 0 und dem ersten Gitterpunkt, ist das ein Inneres (Regressionstest dazu in tests/test_mixed.py).
  * `ols`, `cr1_cov` (cluster-robuste Varianz CR1 = Stata-Faktor m/(m−1)·(N−1)/(N−p)), `kmm_interval` (Kunden-Mittel-Methode), `fixed_effects_coef`.
  * `design_effect`: exakte Varianz des OLS-Koeffizienten bei der wahren Kovarianz (Sandwich in Kundensummen) gegen die naive Formel.
  * geschlossene Formeln des ausgewogenen Falls in Brüchen (`anova_balanced`), Singularitätswahrscheinlichkeit (F-Verteilung), Streuung des ANOVA-Schätzers.
Alle Formeln sind eigene Nachrechnung in diesem Modell (nur Zufallsachsenabschnitt, normale Fehler, konstante Streuung)."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import optimize, stats

# λ-Gitter für die Suche: 0 und 29 Punkte von 1e-3 bis 1e4 (Schrittweite 0,25 Zehnerpotenzen); die Verfeinerung löst die Ableitung nach λ.
LAMBDA_GRID = np.concatenate(([0.0], np.logspace(-3.0, 4.0, 29)))


def compress_groups(g) -> tuple[np.ndarray, int]:
    """Kunden 0 … m−1 ohne leere Kunden: (neue Kundennummern, Kundenzahl). Leere Kunden fallen aus."""
    g = np.asarray(g, dtype=int)
    used, inverse = np.unique(g, return_inverse=True)
    return inverse.astype(int), int(len(used))


@dataclass(frozen=True)
class Fit:
    beta: np.ndarray
    lam: float                 # λ = σ_u²/σ²
    sigma2: float
    sigma_u2: float
    cov: np.ndarray            # Kovarianz von β̂ (σ̂²·M⁻¹ bei λ̂)
    rand: bool                 # Randlösung λ̂ = 0 („singulär“ im Sprachgebrauch von lme4)
    reml: bool
    n: int
    p: int
    m: int
    getrennt: bool = True      # False: σ_u² und σ² sind nicht trennbar (m = 1 oder jeder Kunde mit genau einem Container)
    entartet: bool = False     # True: alle Residuen 0 (Q = 0), keine Streuung schätzbar
    hinweis: str = ""

    @property
    def rho(self) -> float:
        """Intraklassenkorrelation σ_u²/(σ_u² + σ²) des Schätzers (0, wenn beide Varianzen 0 sind)."""
        gesamt = self.sigma_u2 + self.sigma2
        return self.sigma_u2 / gesamt if gesamt > 0 else 0.0

    def se(self, j: int) -> float:
        return float(np.sqrt(self.cov[j, j]))


class MixedModel:
    """Zufallsachsenabschnitt-Modell mit Kundensummen. `g`: Kundennummer je Zeile (0 … m−1, jeder Kunde kommt vor)."""

    def __init__(self, X, y, g, m: int):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.g = np.asarray(g, dtype=int)
        self.m = int(m)
        self.n, self.p = self.X.shape
        self.ng = np.bincount(self.g, minlength=self.m).astype(float)
        if np.any(self.ng == 0):
            raise ValueError("leerer Kunde: vorher mit compress_groups entfernen")
        self.S = np.column_stack([np.bincount(self.g, weights=self.X[:, j], minlength=self.m) for j in range(self.p)])   # Kundensummen der Spalten
        self.t = np.bincount(self.g, weights=self.y, minlength=self.m)                                                    # Kundensummen von y
        self.XtX = self.X.T @ self.X
        self.Xty = self.X.T @ self.y
        self.yty = float(self.y @ self.y)

    # ---- für festes λ -------------------------------------------------------------------------------------------------------------------
    def gls(self, lam: float):
        """(M, β̂(λ), Q(λ)): M = X'W X, Q = (y − Xβ̂)'W(y − Xβ̂) mit W = (I + λZZ')⁻¹ (Woodbury)."""
        c = lam / (1.0 + self.ng * lam)
        M = self.XtX - (self.S.T * c) @ self.S
        v = self.Xty - (self.S.T * c) @ self.t
        b = np.linalg.solve(M, v)
        Q = self.yty - float(np.sum(c * self.t ** 2)) - float(b @ v)
        return M, b, Q

    def crit(self, lam: float, reml: bool) -> float:
        """Profiliertes Kriterium (−2 mal die Log-Likelihood bis auf Konstanten, halbiert): kleiner ist besser."""
        M, _b, Q = self.gls(lam)
        ld = float(np.sum(np.log1p(self.ng * lam)))
        if reml:
            return 0.5 * ((self.n - self.p) * np.log(Q / (self.n - self.p)) + ld + np.linalg.slogdet(M)[1])
        return 0.5 * (self.n * np.log(Q / self.n) + ld)

    def crit_grid(self, lams, reml: bool) -> np.ndarray:
        """Dasselbe Kriterium für viele λ auf einmal (gestapelte p×p-Rechnungen)."""
        L = np.asarray(lams, dtype=float)
        c = L[:, None] / (1.0 + self.ng[None, :] * L[:, None])                       # (G, m)
        M = self.XtX[None, :, :] - np.einsum("gm,mp,mq->gpq", c, self.S, self.S)
        v = self.Xty[None, :] - (c * self.t[None, :]) @ self.S                        # (G, p)
        b = np.linalg.solve(M, v[:, :, None])[:, :, 0]
        Q = self.yty - np.sum(c * self.t[None, :] ** 2, axis=1) - np.sum(b * v, axis=1)
        ld = np.sum(np.log1p(self.ng[None, :] * L[:, None]), axis=1)
        if reml:
            return 0.5 * ((self.n - self.p) * np.log(Q / (self.n - self.p)) + ld + np.linalg.slogdet(M)[1])
        return 0.5 * (self.n * np.log(Q / self.n) + ld)

    def deriv(self, lam: float, reml: bool) -> float:
        """Ableitung des Kriteriums nach λ, analytisch. Q'(λ) = −Σ_k (r_k/(1 + n_kλ))², r_k = Kundensumme der GLS-Residuen;
        d/dλ ln det M = −Σ_k s_k' M⁻¹ s_k/(1 + n_kλ)² mit s_k = Kundensumme der Zeilen von X; d/dλ Σ ln(1 + n_kλ) = Σ n_k/(1 + n_kλ)."""
        M, b, Q = self.gls(lam)
        w = 1.0 / (1.0 + self.ng * lam)
        r = self.t - self.S @ b
        dQ = -float(np.sum((r * w) ** 2))
        summe = float(np.sum(self.ng * w))
        if reml:
            Sw = self.S * w[:, None]
            tr = -float(np.sum(Sw * np.linalg.solve(M, Sw.T).T))
            return 0.5 * ((self.n - self.p) * dQ / Q + summe + tr)
        return 0.5 * (self.n * dQ / Q + summe)

    # ---- Anpassung ----------------------------------------------------------------------------------------------------------------------
    def _wurzel(self, lo: float, hi: float, reml: bool) -> float:
        """Nullstelle der Ableitung in [lo, hi] (Vorzeichenwechsel − nach +), sonst das Kriterium-Minimum auf dem Intervall."""
        flo, fhi = self.deriv(lo, reml), self.deriv(hi, reml)
        if flo < 0.0 < fhi:
            return float(optimize.brentq(lambda l: self.deriv(l, reml), lo, hi, xtol=1e-15, rtol=8.9e-16, maxiter=200))
        res = optimize.minimize_scalar(lambda l: self.crit(l, reml), bounds=(lo, hi), method="bounded", options=dict(xatol=1e-12))
        kandidaten = [(self.crit(lo, reml), lo), (self.crit(hi, reml), hi), (float(res.fun), float(res.x))]
        return min(kandidaten)[1]

    def fit(self, reml: bool = True) -> Fit:
        n, p, m = self.n, self.p, self.m
        dof = n - p if reml else n
        if n <= p:
            raise ValueError("zu wenige Zeilen für die Zahl der Spalten")
        if np.linalg.matrix_rank(self.X) < p:
            raise ValueError("Rangmangel: die Spalten von X sind linear abhängig")
        _M0, b0, Q0 = self.gls(0.0)
        if Q0 <= 1e-14 * max(1.0, self.yty):
            cov = np.zeros((p, p))
            return Fit(b0, 0.0, 0.0, 0.0, cov, True, reml, n, p, m, True, True, "Alle Residuen sind 0: keine Streuung schätzbar.")
        if m == 1 or self.ng.max() == 1.0:
            hinweis = ("Nur ein Kunde: σ_u² ist nicht schätzbar." if m == 1
                       else "Jeder Kunde hat genau einen Container: σ_u² und σ² sind nicht trennbar (das Kriterium ist in λ konstant).")
            M, b, Q = self.gls(0.0)
            s2 = Q / dof
            return Fit(b, 0.0, s2, 0.0, s2 * np.linalg.inv(M), True, reml, n, p, m, False, False, hinweis)
        vals = self.crit_grid(LAMBDA_GRID, reml)
        k = int(np.argmin(vals))
        if k == 0:
            lam = self._wurzel(0.0, float(LAMBDA_GRID[1]), reml) if self.deriv(0.0, reml) < 0.0 else 0.0
        else:
            lam = self._wurzel(float(LAMBDA_GRID[k - 1]), float(LAMBDA_GRID[min(k + 1, len(LAMBDA_GRID) - 1)]), reml)
        M, b, Q = self.gls(lam)
        s2 = Q / dof
        return Fit(b, lam, s2, lam * s2, s2 * np.linalg.inv(M), lam == 0.0, reml, n, p, m)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# OLS, cluster-robust, Kunden-Mittel-Methode, feste Kunden-Effekte
# ---------------------------------------------------------------------------------------------------------------------------------------------

def ols(X, y):
    """(β̂, Kovarianz der naiven Rechnung σ̂²(X'X)⁻¹ mit σ̂² = RSS/(N−p), N−p)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    A = np.linalg.inv(X.T @ X)
    b = A @ (X.T @ y)
    e = y - X @ b
    return b, float(e @ e) / (n - p) * A, n - p


def cr1_cov(X, y, g, m: int, beta) -> np.ndarray:
    """Cluster-robuste Kovarianz CR1: (X'X)⁻¹ (Σ_k s_k s_k') (X'X)⁻¹ · m/(m−1) · (N−1)/(N−p), s_k = Σ_{i∈k} x_i e_i. CR0 ist ohne den Faktor."""
    X = np.asarray(X, float)
    n, p = X.shape
    e = np.asarray(y, float) - X @ beta
    A = np.linalg.inv(X.T @ X)
    Sg = np.column_stack([np.bincount(g, weights=X[:, j] * e, minlength=m) for j in range(p)])
    return A @ (Sg.T @ Sg) @ A * (m / (m - 1.0)) * ((n - 1.0) / (n - p))


def kmm_interval(X_col, y, g, m: int, level: float = 0.95):
    """Kunden-Mittel-Methode für ein Kundenmerkmal: Regression der Kundenmittel auf den Kundenwert, t(m−2).
    Rückgabe (Steigung, Standardfehler, Quantil). Im ausgewogenen Fall (gleich große Kunden) exakt."""
    if m < 3:
        raise ValueError("Die Kunden-Mittel-Methode braucht mindestens 3 Kunden (t(m − 2) hat sonst keine Freiheitsgrade).")
    ng = np.bincount(g, minlength=m).astype(float)
    ybar = np.bincount(g, weights=y, minlength=m) / ng
    xbar = np.bincount(g, weights=X_col, minlength=m) / ng
    A = np.column_stack([np.ones(m), xbar])
    coef = np.linalg.lstsq(A, ybar, rcond=None)[0]
    res = ybar - A @ coef
    s2 = float(res @ res) / (m - 2)
    se = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1]))
    return float(coef[1]), se, float(stats.t.ppf(0.5 + level / 2.0, m - 2))


def fixed_effects_coef(X_inner, y, g, m: int, j: int):
    """Kunden als feste Effekte (Dummies ohne eigenen Achsenabschnitt) für Merkmale **innerhalb** der Kunden: (Koeffizient j, Standardfehler, Freiheitsgrade)."""
    n = len(y)
    D = np.zeros((n, m))
    D[np.arange(n), g] = 1.0
    Xf = np.column_stack([X_inner, D])
    A = np.linalg.inv(Xf.T @ Xf)
    b = A @ (Xf.T @ y)
    e = y - Xf @ b
    dof = n - m - X_inner.shape[1]
    return float(b[j]), float(np.sqrt(float(e @ e) / dof * A[j, j])), dof


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Designeffekt
# ---------------------------------------------------------------------------------------------------------------------------------------------

def design_effect(X, g, m: int, su2: float, s2: float, j: int) -> dict:
    """Exakte Varianz des OLS-Koeffizienten j bei der wahren Kovarianz V = σ²I + σ_u²ZZ' (Sandwich in Kundensummen, ohne N×N-Matrix):
    Var = [A (σ²X'X + σ_u² Σ_k s_k s_k') A]_jj mit A = (X'X)⁻¹. Verglichen mit der naiven Formel (σ²+σ_u²)·A_jj (Gesamtvarianz)
    und mit dem Erwartungswert der naiven Varianz E[σ̂²]·A_jj, E[σ̂²] = tr((I−H)V)/(N−p)."""
    X = np.asarray(X, float)
    n, p = X.shape
    A = np.linalg.inv(X.T @ X)
    S = np.column_stack([np.bincount(g, weights=X[:, c], minlength=m) for c in range(p)])
    var_true = float((A @ (s2 * (X.T @ X) + su2 * (S.T @ S)) @ A)[j, j])
    gesamt = (s2 + su2) * float(A[j, j])
    es2 = (n * (s2 + su2) - s2 * p - su2 * float(np.sum((S @ A) * S))) / (n - p)
    erwartet_naiv = es2 * float(A[j, j])
    return dict(var_true=var_true, naiv_gesamt=gesamt, naiv_erwartet=erwartet_naiv, deff=var_true / gesamt, deff_erwartet=var_true / erwartet_naiv,
                wirksam=n / (var_true / gesamt))


def design_effect_formula(kbar: float, rho: float) -> float:
    """Die Formel 1 + (k − 1)ρ des ausgewogenen Falls (Merkmal nur zwischen den Kunden)."""
    return 1.0 + (kbar - 1.0) * rho


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Geschlossene Formeln des ausgewogenen Falls (nur Achsenabschnitt), in Brüchen
# ---------------------------------------------------------------------------------------------------------------------------------------------

def anova_balanced(rows) -> dict:
    """Einwegklassifikation mit gleich vielen Containern je Kunde: `rows` = Liste der Kunden, je eine Liste von Zahlen (int, Fraction oder Dezimalzeichenfolge).
    Rückgabe in Brüchen: Gesamtmittel, Kundenmittel, SSB, SSW, MSB, MSW, die Schätzer von σ_u² für REML ((MSB−MSW)/k) und ML (((m−1)/m·MSB−MSW)/k)
    jeweils roh (kann negativ sein) und abgeschnitten bei 0, sowie ρ."""
    rows = [[Fraction(v) for v in r] for r in rows]
    m = len(rows)
    ks = {len(r) for r in rows}
    if m < 2 or len(ks) != 1 or min(ks) < 2:
        raise ValueError("ausgewogen mit mindestens 2 Kunden und 2 Containern je Kunde nötig")
    k = ks.pop()
    n = m * k
    gm = sum(sum(r) for r in rows) / n
    cm = [sum(r) / k for r in rows]
    ssb = k * sum((c - gm) ** 2 for c in cm)
    ssw = sum((v - cm[i]) ** 2 for i, r in enumerate(rows) for v in r)
    msb, msw = ssb / (m - 1), ssw / (m * (k - 1))
    reml_roh = (msb - msw) / k
    ml_roh = (Fraction(m - 1, m) * msb - msw) / k
    reml, ml = max(Fraction(0), reml_roh), max(Fraction(0), ml_roh)
    sst = ssb + ssw
    # Randlösung: σ² aus der Gesamtstreuung (REML: SST/(N−1), ML: SST/N); sonst σ² = MSW (REML) bzw. = MSW (ML, siehe README)
    s2_reml = msw if reml_roh > 0 else sst / (n - 1)
    s2_ml = msw if ml_roh > 0 else sst / n
    return dict(m=m, k=k, n=n, gm=gm, cm=cm, ssb=ssb, ssw=ssw, sst=sst, msb=msb, msw=msw, reml_roh=reml_roh, ml_roh=ml_roh, reml=reml, ml=ml,
                s2_reml=s2_reml, s2_ml=s2_ml, rho_reml=reml / (reml + s2_reml) if reml + s2_reml else Fraction(0),
                rho_ml=ml / (ml + s2_ml) if ml + s2_ml else Fraction(0))


def p_singulaer_ausgewogen(m: int, k: int, rho: float, art: str = "zwischen") -> float:
    """Wahrscheinlichkeit von σ̂_u²(REML) = 0 im ausgewogenen Fall: P(F(df_zwischen, df_innen) ≤ 1/(1 + kλ)), λ = ρ/(1 − ρ).
    art: 'zwischen' (ein Kundenmerkmal: df_zwischen = m − 2), 'innerhalb' (ein Merkmal innerhalb, df_innen um 1 kleiner), 'achsenabschnitt'."""
    lam = rho / (1.0 - rho) if rho < 1.0 else float("inf")
    if art == "zwischen":
        dfb, dfw = m - 2, m * (k - 1)
    elif art == "innerhalb":
        dfb, dfw = m - 1, m * (k - 1) - 1
    else:
        dfb, dfw = m - 1, m * (k - 1)
    return float(stats.f.cdf(1.0 / (1.0 + k * lam), dfb, dfw))


def anova_sd(m: int, k: int, su2: float, s2: float) -> float:
    """Standardabweichung des ANOVA-Schätzers (MSB − MSW)/k ohne Abschneiden: Var = (2/k²)·[(σ² + kσ_u²)²/(m−1) + σ⁴/(m(k−1))]."""
    return float(np.sqrt(2.0 / k ** 2 * ((s2 + k * su2) ** 2 / (m - 1) + s2 ** 2 / (m * (k - 1)))))


def sd_grenze_viele_container(m: int, su2: float) -> float:
    """Grenzwert der Streuung von σ̂_u² für k → ∞ im ausgewogenen Fall ohne Kundenmerkmal im Modell: σ_u²·√(2/(m−1)) (mit einem Kundenmerkmal im Modell m−2 statt m−1). Mehr Container je Kunde senken die Streuung nur bis hierher."""
    return su2 * float(np.sqrt(2.0 / (m - 1)))
