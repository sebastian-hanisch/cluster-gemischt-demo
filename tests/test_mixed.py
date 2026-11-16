"""Kern (clg_mixed): Handbeispiele in Brüchen, Mini-Instanz-Tests je Einheit, Zweige und Randfälle (DEMO-PLAYBOOK §1b und §12).

Die Handwerte stehen im Test als Rechnung (Kundenmittel, Quadratsummen, Brüche), nicht als abgeschriebene Ausgabe des Codes."""

from fractions import Fraction as F

import numpy as np
import pytest
from scipy import stats

import clg_mixed as MX
from conftest import bruch_inv, bruch_matmul, bruch_transp, bruch_V


def _modell(rows):
    y = np.array([float(v) for r in rows for v in r])
    g = np.repeat(np.arange(len(rows)), [len(r) for r in rows])
    return MX.MixedModel(np.ones((len(y), 1)), y, g, len(rows))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 1: drei Kunden mit je zwei Containern, y = (1, 3 | 5, 9 | 8, 10), nur Achsenabschnitt
# ---------------------------------------------------------------------------------------------------------------------------------------------

def test_hand1_quadratsummen_und_schaetzer_in_bruechen(hand1):
    a = MX.anova_balanced(hand1)
    # von Hand: Gesamtmittel 36/6 = 6, Kundenmittel 2, 7, 9; SSB = 2·((2−6)² + (7−6)² + (9−6)²) = 52; SSW = (1−2)²+(3−2)² + (5−7)²+(9−7)² + (8−9)²+(10−9)² = 2+8+2 = 12
    assert a["gm"] == 6 and a["cm"] == [2, 7, 9] and a["ssb"] == 52 and a["ssw"] == 12
    assert a["msb"] == F(52, 2) == 26 and a["msw"] == F(12, 3) == 4
    assert a["reml"] == (26 - 4) / F(2) == 11
    assert a["ml"] == (F(2, 3) * 26 - 4) / 2 == F(20, 3)
    assert a["rho_reml"] == F(11, 15) and a["rho_ml"] == F(5, 8)


def test_hand1_stationaritaet_des_reml_kriteriums(hand1):
    """Ableitung nach s = 1 + kλ ist proportional zu g(s) = (m−1)(s·SSW + SSB) − (N−1)·SSB (hier von Hand hergeleitet); g(13/2) = 0, in s linear steigend."""
    m, n, ssw, ssb = 3, 6, F(12), F(52)

    def g(s):
        return (m - 1) * (s * ssw + ssb) - (n - 1) * ssb

    assert g(F(13, 2)) == 0 and g(F(1)) < 0 < g(F(7))
    a = MX.anova_balanced(hand1)
    assert 1 + 2 * a["reml"] / a["msw"] == F(13, 2)


def test_hand1_gleitkommaanpassung_stimmt_mit_den_bruechen(hand1):
    mod = _modell(hand1)
    r, ml = mod.fit(True), mod.fit(False)
    assert r.sigma_u2 == pytest.approx(11.0, rel=1e-9) and r.sigma2 == pytest.approx(4.0, rel=1e-9) and r.lam == pytest.approx(11 / 4, rel=1e-9)
    assert ml.sigma_u2 == pytest.approx(20 / 3, rel=1e-9) and ml.sigma2 == pytest.approx(4.0, rel=1e-9)
    assert r.rho == pytest.approx(11 / 15, rel=1e-9) and ml.rho == pytest.approx(5 / 8, rel=1e-9)
    assert not r.rand and not ml.rand


def test_hand1_zweiter_rechenweg_dichte_matrix_in_bruechen(hand1):
    """GLS-Achsenabschnitt und Varianz mit der dichten 6×6-Matrix V = σ²I + σ_u²ZZ' in Brüchen: 6 und 13/3 = MSB/(m·k)."""
    g = [0, 0, 1, 1, 2, 2]
    Vi = bruch_inv(bruch_V(g, F(4), F(11)))
    eins = [[F(1)] for _ in g]
    y = [[F(v)] for r in hand1 for v in r]
    info = bruch_matmul(bruch_matmul(bruch_transp(eins), Vi), eins)[0][0]
    mu = bruch_matmul(bruch_matmul(bruch_transp(eins), Vi), y)[0][0] / info
    assert mu == 6 and 1 / info == F(13, 3) == F(26, 6)
    fit = _modell(hand1).fit(True)
    assert fit.beta[0] == pytest.approx(6.0) and fit.cov[0, 0] == pytest.approx(13 / 3, rel=1e-9)


def test_hand1_naive_varianz_und_intervalle(hand1):
    """Naiv: σ̂² = SST/(N−1) = 64/5, Varianz des Mittels 32/15; gemischt 13/3: Verhältnis 65/32 (Faktor √(65/32) ≈ 1,425 im Standardfehler)."""
    y = [v for r in hand1 for v in r]
    sst = sum((F(v) - 6) ** 2 for v in y)
    assert sst == 64 and sst / 5 == F(64, 5) and sst / 5 / 6 == F(32, 15) and F(13, 3) / F(32, 15) == F(65, 32)
    _b, cov, dfn = MX.ols(np.ones((6, 1)), np.array(y, float))
    assert cov[0, 0] == pytest.approx(32 / 15) and dfn == 5
    # Halbbreiten: gemischt t(2) = 4,303·√(13/3) = 8,96; naiv t(5) = 2,571·√(32/15) = 3,76 (Quantile aus scipy, nur Anzeigewerte)
    assert stats.t.ppf(0.975, 2) * np.sqrt(13 / 3) == pytest.approx(8.96, abs=0.005)
    assert stats.t.ppf(0.975, 5) * np.sqrt(32 / 15) == pytest.approx(3.755, abs=0.005)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 2: Randfall
# ---------------------------------------------------------------------------------------------------------------------------------------------

def test_hand2_negative_schaetzer_und_rand(hand2):
    a = MX.anova_balanced(hand2)
    assert a["gm"] == F(10, 3) and a["msb"] == F(8, 3) and a["msw"] == F(10, 3)
    assert a["reml_roh"] == F(-1, 3) and a["ml_roh"] == F(-7, 9) and a["reml"] == 0 and a["ml"] == 0
    assert a["s2_reml"] == F(46, 15) and a["s2_ml"] == F(23, 9)
    ssw, ssb = a["ssw"], a["ssb"]

    def g(s):
        return 2 * (s * ssw + ssb) - 5 * ssb

    assert g(F(1)) == 4 and a["msb"] / a["msw"] == F(4, 5) < 1                  # das Kriterium steigt im zulässigen Bereich s ≥ 1 überall


def test_hand2_anpassung_liegt_am_rand_und_die_ableitung_zeigt_es(hand2):
    mod = _modell(hand2)
    r, ml = mod.fit(True), mod.fit(False)
    assert r.rand and ml.rand and r.sigma_u2 == 0.0 and ml.sigma_u2 == 0.0
    assert r.sigma2 == pytest.approx(46 / 15, rel=1e-12) and ml.sigma2 == pytest.approx(23 / 9, rel=1e-12)
    assert mod.deriv(0.0, True) > 0 and mod.deriv(0.0, False) > 0
    assert r.cov[0, 0] == pytest.approx(46 / 15 / 6, rel=1e-12)                 # am Rand ist die Kovarianz die naive: σ̂²(X'X)⁻¹ = (46/15)/6


def test_hand1_ableitung_bei_null_ist_negativ_also_kein_rand(hand1):
    assert _modell(hand1).deriv(0.0, True) < 0 and _modell(hand1).deriv(0.0, False) < 0


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 3: Designeffekt (vier Kunden à drei Container, x = (0, 0, 1, 1), σ² = 1, σ_u² = 1/3)
# ---------------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def design3():
    g = np.repeat(np.arange(4), 3)
    X = np.column_stack([np.ones(12), np.repeat([0.0, 0.0, 1.0, 1.0], 3)])
    return X, g


def test_hand3_designeffekt_exakt_in_bruechen_und_in_der_funktion(design3):
    X, g = design3
    Xf = [[F(1), F(int(v))] for v in X[:, 1]]
    XtXi = bruch_inv(bruch_matmul(bruch_transp(Xf), Xf))
    V = bruch_V(list(g), F(1), F(1, 3))
    var_true = bruch_matmul(bruch_matmul(bruch_matmul(bruch_matmul(XtXi, bruch_transp(Xf)), V), Xf), XtXi)[1][1]
    assert XtXi[1][1] == F(1, 3) and var_true == F(2, 3)
    assert var_true / ((F(1) + F(1, 3)) * XtXi[1][1]) == F(3, 2) == 1 + 2 * F(1, 4)         # 1 + (k−1)ρ mit ρ = 1/4
    d = MX.design_effect(X, g, 4, 1 / 3, 1.0, 1)
    assert d["var_true"] == pytest.approx(2 / 3) and d["deff"] == pytest.approx(1.5) and d["wirksam"] == pytest.approx(8.0)
    # erwartete naive Varianz E[s²](X'X)⁻¹ = (6/5)(1/3) = 2/5, also wahr gegen naiv 5/3
    assert d["naiv_erwartet"] == pytest.approx(2 / 5) and d["deff_erwartet"] == pytest.approx(5 / 3)
    assert MX.design_effect_formula(3, 0.25) == pytest.approx(1.5)


def test_hand3_merkmal_innerhalb_der_kunden_nimmt_dem_kunden_nichts_weg():
    """x = (−1, 0, 1) in jedem Kunden: wahre Varianz = σ²(X'X)⁻¹ = 1/8, in Einheiten von σ_ges²(X'X)⁻¹ gleich 1 − ρ = 3/4."""
    g = np.repeat(np.arange(4), 3)
    X = np.column_stack([np.ones(12), np.tile([-1.0, 0.0, 1.0], 4)])
    d = MX.design_effect(X, g, 4, 1 / 3, 1.0, 1)
    assert d["var_true"] == pytest.approx(1 / 8) and d["deff"] == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 4: CR1 gegen Kunden-Mittel-Methode (dieselben vier Kunden, y = (1,2,3 | 2,3,4 | 3,5,7 | 4,6,8))
# ---------------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def daten4(design3):
    X, g = design3
    return X, g, np.array([1, 2, 3, 2, 3, 4, 3, 5, 7, 4, 6, 8], float)


def test_hand4_ols_cr0_cr1_und_kmm(daten4):
    X, g, y = daten4
    b, naiv, dfn = MX.ols(X, y)
    assert b[0] == pytest.approx(5 / 2) and b[1] == pytest.approx(3.0) and dfn == 10
    assert naiv[1, 1] == pytest.approx(23 / 30)                       # Residuenquadratsumme 23, geteilt durch N − p = 10, mal (X'X)⁻¹ = 1/3
    cr1 = MX.cr1_cov(X, y, g, 4, b)
    assert cr1[1, 1] == pytest.approx(11 / 30, rel=1e-12)             # CR0 = 1/4, Faktor 4/3 · 11/10 = 22/15
    sl, se, q = MX.kmm_interval(X[:, 1], y, g, 4)
    assert sl == pytest.approx(3.0) and se ** 2 == pytest.approx(1 / 2) and q == pytest.approx(stats.t.ppf(0.975, 2))
    # Halbbreiten 1,95 / 1,93 / 3,04
    assert stats.t.ppf(0.975, 10) * np.sqrt(23 / 30) == pytest.approx(1.95, abs=0.005)
    assert stats.t.ppf(0.975, 3) * np.sqrt(11 / 30) == pytest.approx(1.93, abs=0.005)
    assert q * se == pytest.approx(3.04, abs=0.005)


def test_beziehung_cr1_gleich_welch_der_kundenmittel_in_zufallsbeispielen():
    """Ausgewogen, 0/1-Kundenmerkmal, gleich große Gruppen aus n_g Kunden: CR1 = Welch·(n_g−1)/n_g·m/(m−1)·(N−1)/(N−2)."""
    rng = np.random.default_rng(11)
    for _ in range(60):
        m, k = [(4, 2), (4, 3), (6, 2), (6, 3), (8, 2)][int(rng.integers(0, 5))]
        g = np.repeat(np.arange(m), k)
        n = m * k
        ng = m // 2
        X = np.column_stack([np.ones(n), (g >= ng).astype(float)])
        y = rng.integers(0, 21, n).astype(float)
        b, _cov, _df = MX.ols(X, y)
        cr1 = MX.cr1_cov(X, y, g, m, b)[1, 1]
        km = y.reshape(m, k).mean(axis=1)
        welch = km[:ng].var(ddof=1) / ng + km[ng:].var(ddof=1) / ng
        assert cr1 == pytest.approx(welch * (ng - 1) / ng * m / (m - 1) * (n - 1) / (n - 2), rel=1e-10)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Konventionen und Zweige der Funktionen
# ---------------------------------------------------------------------------------------------------------------------------------------------

def test_konventionen_ml_gegen_reml_und_lambda_gegen_theta(hand1):
    a = MX.anova_balanced(hand1)
    assert a["ml"] < a["reml"] and a["ml"] / a["reml"] == F(20, 33)
    lam = a["reml"] / a["msw"]
    assert lam == F(11, 4) and np.sqrt(float(lam)) == pytest.approx(1.6583, abs=1e-4)        # θ = √λ wie in lme4 (nach dem Quellenbericht)


def test_ausgewogenheit_wird_verlangt():
    with pytest.raises(ValueError):
        MX.anova_balanced([[1, 2], [3]])
    with pytest.raises(ValueError):
        MX.anova_balanced([[1, 2]])
    with pytest.raises(ValueError):
        MX.anova_balanced([[1], [2]])


def test_streuung_des_anova_schaetzers_und_grenze_vieler_container():
    assert MX.anova_sd(12, 30, 0.09, 0.3025) == pytest.approx(0.0427, abs=5e-5)
    assert MX.anova_sd(12, 10 ** 6, 0.09, 0.3025) == pytest.approx(MX.sd_grenze_viele_container(12, 0.09), rel=1e-3)
    assert MX.sd_grenze_viele_container(12, 0.09) == pytest.approx(0.09 * np.sqrt(2 / 11))


def test_singularitaetswahrscheinlichkeit_zweige():
    p_zw = MX.p_singulaer_ausgewogen(10, 10, 0.05, "zwischen")
    p_in = MX.p_singulaer_ausgewogen(10, 10, 0.05, "innerhalb")
    p_ac = MX.p_singulaer_ausgewogen(10, 10, 0.05, "achsenabschnitt")
    assert len({round(p_zw, 6), round(p_in, 6), round(p_ac, 6)}) == 3          # drei Zweige geben drei Werte
    assert p_zw == pytest.approx(stats.f.cdf(1 / (1 + 10 * 0.05 / 0.95), 8, 90))
    assert MX.p_singulaer_ausgewogen(10, 10, 0.0, "zwischen") > 0.5            # ohne Kundeneffekt liegt mehr als die Hälfte am Rand


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Randfälle
# ---------------------------------------------------------------------------------------------------------------------------------------------

def test_ein_kunde_sigma_u_nicht_schaetzbar():
    f = MX.MixedModel(np.ones((5, 1)), np.array([1.0, 2, 4, 3, 5]), np.zeros(5, int), 1).fit(True)
    assert not f.getrennt and f.rand and f.sigma_u2 == 0.0 and "ein kunde" in f.hinweis.lower()


def test_jeder_kunde_mit_genau_einem_container_ist_nicht_trennbar():
    y = np.array([1.0, 4, 2, 7, 3, 5, 9, 2])
    mod = MX.MixedModel(np.ones((8, 1)), y, np.arange(8), 8)
    werte = [mod.crit(lam, True) for lam in (0.0, 0.01, 0.1, 1.0, 10.0)]
    assert max(werte) - min(werte) < 1e-12                                       # das Kriterium ist in λ konstant: ein Vergleich wäre Rauschen
    f = mod.fit(True)
    assert not f.getrennt and "nicht trennbar" in f.hinweis


def test_alle_werte_gleich_entartet_ohne_absturz():
    f = MX.MixedModel(np.ones((6, 1)), np.full(6, 3.0), np.repeat(np.arange(3), 2), 3).fit(True)
    assert f.entartet and f.rand and f.sigma2 == 0.0 and f.beta[0] == pytest.approx(3.0)


def test_ein_kunde_mit_fast_allen_containern():
    rng = np.random.default_rng(3)
    g = np.r_[np.zeros(28, int), [1], [2]]
    y = rng.standard_normal(30) + np.r_[np.full(28, 0.5), [2.0], [-1.0]]
    f = MX.MixedModel(np.ones((30, 1)), y, g, 3).fit(True)
    assert np.isfinite(f.sigma_u2) and np.isfinite(f.cov).all() and f.sigma2 > 0


def test_leere_kunden_fallen_aus_und_der_kern_verlangt_sie_nicht():
    g = np.array([0, 0, 2, 2, 5, 5])
    g2, m = MX.compress_groups(g)
    assert list(g2) == [0, 0, 1, 1, 2, 2] and m == 3
    with pytest.raises(ValueError):
        MX.MixedModel(np.ones((6, 1)), np.arange(6.0), g, 6)          # Kunden 1, 3, 4 sind leer
    y = np.array([1.0, 3, 5, 9, 8, 10])
    assert MX.MixedModel(np.ones((6, 1)), y, g2, m).fit(True).sigma_u2 == pytest.approx(11.0, rel=1e-9)


def test_zu_wenige_zeilen_und_rangmangel():
    with pytest.raises(ValueError):
        MX.MixedModel(np.ones((2, 2)), np.array([1.0, 2]), np.array([0, 1]), 2).fit(True)
    g = np.repeat(np.arange(4), 3)
    D = np.eye(4)[g]
    X = np.column_stack([D, np.repeat([0.0, 0.0, 1.0, 1.0], 3)])                 # Kunden-Dummies zusammen mit einem Kundenmerkmal
    assert np.linalg.matrix_rank(X) == 4 < X.shape[1]
    with pytest.raises(ValueError, match="Rangmangel"):
        MX.MixedModel(X, np.arange(12.0), g, 4).fit(True)


def test_kmm_braucht_drei_kunden_und_cr1_mit_zwei_kunden_laeuft():
    g = np.repeat(np.arange(2), 3)
    X = np.column_stack([np.ones(6), np.repeat([0.0, 1.0], 3)])
    y = np.array([1.0, 2, 3, 4, 6, 5])
    with pytest.raises(ValueError):
        MX.kmm_interval(X[:, 1], y, g, 2)
    b, _c, _d = MX.ols(X, y)
    assert np.isfinite(MX.cr1_cov(X, y, g, 2, b)).all()                         # m = 2: CR1 mit Faktor 2/1, das Quantil t(1) ist die Folge


def _alte_gitter_anpassung(mod, reml):
    """Die Suche der früheren Messung: Gitter aus 0 und 23 Punkten von 1e-3 bis 10^2,5; verfeinert nur, wenn das Gitterminimum positiv ist (die Lücke)."""
    gitter = np.r_[0.0, np.logspace(-3, 2.5, 23)]
    werte = np.array([mod.crit(lam, reml) for lam in gitter])
    k = int(np.argmin(werte))
    return 0.0 if k == 0 else float(gitter[k])


def test_optimum_zwischen_null_und_erstem_gitterpunkt_ist_kein_rand():
    """Regression: liegt λ̂ zwischen 0 und dem ersten positiven Gitterpunkt 1e-3, meldete die frühere Gittersuche den Rand λ = 0 (hier: λ̂ = 3·10⁻⁴).
    Konstruktion: m = 8 Kunden, k = 2, Abweichungen ±1 im Kunden (MSW = 2), Kundenmittel so gewählt, dass MSB/MSW = 1,0006 ist, also λ̂ = (MSB/MSW − 1)/k = 3·10⁻⁴."""
    m, k = 8, 2
    t = np.sqrt(7.0042 / 42.0)
    mu = t * (np.arange(m) - 3.5)
    y = np.column_stack([mu - 1.0, mu + 1.0]).ravel()
    mod = MX.MixedModel(np.ones((m * k, 1)), y, np.repeat(np.arange(m), k), m)
    a = MX.anova_balanced([[float(v) for v in r] for r in y.reshape(m, k)])
    lam_exakt = float((a["msb"] / a["msw"] - 1) / k)
    assert 2.5e-4 < lam_exakt < 3.5e-4
    f = mod.fit(True)
    assert not f.rand and f.lam == pytest.approx(lam_exakt, rel=1e-6) and mod.deriv(0.0, True) < 0
    assert _alte_gitter_anpassung(mod, True) == 0.0                              # die frühere Suche hätte hier „Rand“ gemeldet: der Test schlägt auf ihr fehl


def test_ableitung_stimmt_mit_der_differenz_des_kriteriums_ueberein():
    rng = np.random.default_rng(8)
    n = 40
    g, m = MX.compress_groups(rng.integers(0, 6, n))
    kundenwert = rng.standard_normal(m)
    X = np.column_stack([np.ones(n), rng.standard_normal(n), kundenwert[g]])
    y = rng.standard_normal(n) + rng.standard_normal(m)[g]
    mod = MX.MixedModel(X, y, g, m)
    for reml in (True, False):
        for lam in (0.0, 0.05, 0.7, 3.0):
            h = 1e-6 * max(lam, 1.0)
            lo = max(lam - h, 0.0)
            num = (mod.crit(lam + h, reml) - mod.crit(lo, reml)) / (lam + h - lo)
            assert mod.deriv(lam, reml) == pytest.approx(num, rel=2e-5, abs=1e-7)


def test_gestapeltes_kriterium_gleicht_dem_einzelnen():
    rng = np.random.default_rng(9)
    g = np.repeat(np.arange(5), 4)
    X = np.column_stack([np.ones(20), rng.standard_normal(20)])
    mod = MX.MixedModel(X, rng.standard_normal(20) + rng.standard_normal(5)[g], g, 5)
    lams = np.array([0.0, 0.01, 0.3, 4.0, 90.0])
    for reml in (True, False):
        assert np.allclose(mod.crit_grid(lams, reml), [mod.crit(lam, reml) for lam in lams], rtol=1e-12, atol=1e-12)


def test_cr1_fassung_mit_vektoren_gleicht_der_schleife():
    rng = np.random.default_rng(12)
    sizes = [3, 7, 5, 2, 6, 4]
    g = np.repeat(np.arange(6), sizes)
    n = len(g)
    X = np.column_stack([np.ones(n), np.repeat(rng.standard_normal(6), sizes), rng.standard_normal(n)])
    y = X @ np.array([1.0, 0.5, -0.4]) + np.repeat(rng.normal(0, 0.8, 6), sizes) + rng.standard_normal(n)
    b, _c, _d = MX.ols(X, y)
    e = y - X @ b
    A = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[g == c].T @ e[g == c], X[g == c].T @ e[g == c]) for c in range(6))
    schleife = A @ meat @ A * (6 / 5) * ((n - 1) / (n - 3))
    assert np.allclose(MX.cr1_cov(X, y, g, 6, b), schleife, rtol=1e-12, atol=1e-15)


def test_feste_kunden_effekte_geben_den_innen_koeffizienten():
    """Merkmal innerhalb der Kunden: mit Kunden-Dummies ist der Koeffizient die gepoolte Steigung innerhalb der Kunden."""
    rng = np.random.default_rng(15)
    m, k = 5, 6
    g = np.repeat(np.arange(m), k)
    x = rng.standard_normal(m * k)
    y = 2.0 * x + np.repeat(rng.standard_normal(m), k) + 0.1 * rng.standard_normal(m * k)
    b, se, df = MX.fixed_effects_coef(x[:, None], y, g, m, 0)
    xc = x - np.repeat(x.reshape(m, k).mean(axis=1), k)
    yc = y - np.repeat(y.reshape(m, k).mean(axis=1), k)
    assert b == pytest.approx(float(xc @ yc / (xc @ xc)), rel=1e-10) and df == m * k - m - 1 and se > 0
