"""Orakel-Tests: die Kernrechnung gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2, §5, §6).

Referenzen: (1) geschlossene Form des ausgewogenen Falls in Brüchen (Varianzanalyse), auch am Rand; (2) die dichte N×N-Kovarianz mit voller Determinante und
Inversen, **ohne Optimierer** (das profilierte Kriterium muss sich von der dichten Fassung nur um eine Konstante unterscheiden, so ist die Referenz nicht ungenauer
als die Toleranz); (3) CR1 mit Schleife über die Kunden; (4) Simulation gegen die exakte Sandwich-Varianz, die F-Verteilung für den Rand, die Formel der Streuung des
ANOVA-Schätzers und die exakte Überdeckung der Kunden-Mittel-Methode. Zufall in den Tests: numpy (unabhängig vom Zufallsgenerator der App)."""

import numpy as np
import pytest

import clg_mixed as MX
from conftest import dichtes_V


def test_reml_und_ml_gleichen_der_geschlossenen_form_auch_am_rand():
    rng = np.random.default_rng(2001)
    maxd_r = maxd_m = 0.0
    negativ = 0
    for _ in range(120):
        m, k = int(rng.integers(3, 9)), int(rng.integers(2, 6))
        yy = rng.integers(0, 20, (m, k))
        a = MX.anova_balanced(yy.tolist())
        negativ += int(a["reml_roh"] < 0)
        mod = MX.MixedModel(np.ones((m * k, 1)), yy.ravel().astype(float), np.repeat(np.arange(m), k), m)
        fr, fm = mod.fit(True), mod.fit(False)
        maxd_r = max(maxd_r, abs(fr.sigma_u2 - float(a["reml"])) / max(float(a["msw"]), 1.0))
        maxd_m = max(maxd_m, abs(fm.sigma_u2 - float(a["ml"])) / max(float(a["msw"]), 1.0))
        assert fr.rand == (a["reml_roh"] <= 0) and fm.rand == (a["ml_roh"] <= 0)
        assert fr.sigma2 == pytest.approx(float(a["s2_reml"]), rel=1e-7) and fm.sigma2 == pytest.approx(float(a["s2_ml"]), rel=1e-7)
    assert negativ >= 20                                   # Randfälle sind im Datensatz (hier mindestens 20 von 120)
    assert maxd_r < 1e-7 and maxd_m < 1e-7


def _dichtes_kriterium(X, y, g, lam, reml):
    """Dieselbe Größe mit der dichten N×N-Matrix (V = I + λZZ', σ² herausgekürzt): ½[(N−p)·ln(Q/(N−p)) + ln det V + ln det X'V⁻¹X] bzw. ½[N·ln(Q/N) + ln det V]."""
    n, p = X.shape
    V = dichtes_V(g, 1.0, lam)
    Vi = np.linalg.inv(V)
    M = X.T @ Vi @ X
    b = np.linalg.solve(M, X.T @ Vi @ y)
    r = y - X @ b
    Q = float(r @ Vi @ r)
    ld = np.linalg.slogdet(V)[1]
    if reml:
        return 0.5 * ((n - p) * np.log(Q / (n - p)) + ld + np.linalg.slogdet(M)[1])
    return 0.5 * (n * np.log(Q / n) + ld)


def test_profiliertes_kriterium_gleicht_der_dichten_matrix_bis_auf_eine_konstante():
    """Unausgewogen (6 Kunden, N = 27), ein Kundenmerkmal und ein Merkmal innerhalb: die Differenz der beiden Fassungen ist über λ konstant, in ML und REML."""
    rng = np.random.default_rng(2002)
    sizes = [3, 7, 5, 2, 6, 4]
    g = np.repeat(np.arange(6), sizes)
    n = len(g)
    X = np.column_stack([np.ones(n), np.repeat(rng.standard_normal(6), sizes), rng.standard_normal(n)])
    y = X @ np.array([1.0, 0.5, -0.4]) + np.repeat(rng.normal(0, 0.8, 6), sizes) + rng.standard_normal(n)
    mod = MX.MixedModel(X, y, g, 6)
    lams = [0.0, 0.02, 0.3, 1.5, 12.0]
    for reml in (True, False):
        diff = [_dichtes_kriterium(X, y, g, lam, reml) - mod.crit(lam, reml) for lam in lams]
        assert max(diff) - min(diff) < 1e-9


def test_gls_und_kovarianz_bei_festem_lambda_gleichen_der_dichten_rechnung():
    rng = np.random.default_rng(2012)
    sizes = [3, 7, 5, 2, 6, 4]
    g = np.repeat(np.arange(6), sizes)
    n = len(g)
    X = np.column_stack([np.ones(n), np.repeat(rng.standard_normal(6), sizes), rng.standard_normal(n)])
    y = X @ np.array([1.0, 0.5, -0.4]) + np.repeat(rng.normal(0, 0.8, 6), sizes) + rng.standard_normal(n)
    mod = MX.MixedModel(X, y, g, 6)
    for lam in (0.0, 0.4, 3.0):
        Vi = np.linalg.inv(dichtes_V(g, 1.0, lam))
        M = X.T @ Vi @ X
        b = np.linalg.solve(M, X.T @ Vi @ y)
        M2, b2, Q2 = mod.gls(lam)
        assert np.allclose(b, b2, rtol=1e-10, atol=1e-12) and np.allclose(M, M2, rtol=1e-10, atol=1e-12)
        r = y - X @ b
        assert Q2 == pytest.approx(float(r @ Vi @ r), rel=1e-10)


def test_angepasstes_lambda_ist_das_minimum_der_dichten_fassung():
    """Das Ergebnis der Anpassung liegt in der dichten Fassung im Minimum: Nachbarn λ·(1 ± 0,02) sind schlechter; am Rand steigt die Ableitung."""
    rng = np.random.default_rng(2013)
    for _ in range(12):
        m = int(rng.integers(4, 9))
        sizes = rng.integers(2, 7, m)
        g = np.repeat(np.arange(m), sizes)
        n = len(g)
        X = np.column_stack([np.ones(n), np.repeat(rng.standard_normal(m), sizes)])
        y = X @ np.array([1.0, 0.5]) + np.repeat(rng.normal(0, rng.uniform(0.1, 1.2), m), sizes) + rng.standard_normal(n)
        mod = MX.MixedModel(X, y, g, m)
        for reml in (True, False):
            f = mod.fit(reml)
            ref = _dichtes_kriterium(X, y, g, f.lam, reml)
            if f.rand:
                assert _dichtes_kriterium(X, y, g, 1e-4, reml) >= ref - 1e-12
            else:
                assert _dichtes_kriterium(X, y, g, f.lam * 0.98, reml) > ref - 1e-12 and _dichtes_kriterium(X, y, g, f.lam * 1.02, reml) > ref - 1e-12


def test_cr1_gleicht_der_schleife_und_dem_vorfaktor():
    rng = np.random.default_rng(2003)
    sizes = [3, 7, 5, 2, 6, 4]
    g = np.repeat(np.arange(6), sizes)
    n = len(g)
    X = np.column_stack([np.ones(n), np.repeat(rng.standard_normal(6), sizes), rng.standard_normal(n)])
    y = X @ np.array([1.0, 0.5, -0.4]) + np.repeat(rng.normal(0, 0.8, 6), sizes) + rng.standard_normal(n)
    b, _c, _d = MX.ols(X, y)
    e = y - X @ b
    A = np.linalg.inv(X.T @ X)
    cr0 = A @ sum(np.outer(X[g == c].T @ e[g == c], X[g == c].T @ e[g == c]) for c in range(6)) @ A
    assert np.allclose(MX.cr1_cov(X, y, g, 6, b), cr0 * (6 / 5) * ((n - 1) / (n - 3)), rtol=1e-12, atol=1e-15)


def test_designeffekt_exakt_gegen_simulation_bei_festem_design():
    """Exakte Sandwich-Varianz des OLS-Anstiegs (12 Kunden, Größen fest, Kundenmerkmal halb/halb) gegen die Varianz über 20 000 simulierte Antworten (3 %)."""
    rng = np.random.default_rng(2006)
    sz = np.array([14, 41, 22, 35, 28, 19, 44, 30, 26, 33, 38, 30])
    g = np.repeat(np.arange(12), sz)
    X = np.column_stack([np.ones(len(g)), np.repeat(np.r_[np.zeros(6), np.ones(6)], sz)])
    su2, s2 = 0.09, 0.55 ** 2
    exakt = MX.design_effect(X, g, 12, su2, s2, 1)["var_true"]
    A = np.linalg.inv(X.T @ X)
    sl = np.empty(20000)
    for i in range(20000):
        yy = np.repeat(rng.normal(0, np.sqrt(su2), 12), sz) + np.sqrt(s2) * rng.standard_normal(len(g))
        sl[i] = (A @ X.T @ yy)[1]
    assert np.var(sl) / exakt == pytest.approx(1.0, abs=0.03)
    V = dichtes_V(g, s2, su2)                                                    # die Matrixformel ohne Kundensummen
    assert exakt == pytest.approx(float((A @ X.T @ V @ X @ A)[1, 1]), rel=1e-12)


def test_designeffekt_ausgewogen_gleich_der_formel():
    for rho in (0.05, 0.30):
        su2 = rho / (1 - rho)
        g = np.repeat(np.arange(10), 10)
        X = np.column_stack([np.ones(100), np.repeat(np.r_[np.zeros(5), np.ones(5)], 10)])
        assert MX.design_effect(X, g, 10, su2, 1.0, 1)["deff"] == pytest.approx(1 + 9 * rho, abs=1e-9)


def _singulaer_simulation(rng, m, k, rho, art, anzahl):
    lam = rho / (1 - rho)
    g = np.repeat(np.arange(m), k)
    zaehl = 0
    for _ in range(anzahl):
        if art == "zwischen":
            X = np.column_stack([np.ones(m * k), np.repeat(rng.standard_normal(m), k)])
        elif art == "innerhalb":
            x = rng.standard_normal(m * k)
            X = np.column_stack([np.ones(m * k), x - np.repeat(x.reshape(m, k).mean(axis=1), k)])
        else:
            X = np.ones((m * k, 1))
        y = X[:, 0] + np.repeat(rng.normal(0, np.sqrt(lam), m), k) + rng.standard_normal(m * k)
        zaehl += MX.MixedModel(X, y, g, m).fit(True).rand
    return zaehl / anzahl


@pytest.mark.parametrize("art,m,k,rho", [("zwischen", 10, 10, 0.05), ("zwischen", 10, 10, 0.0), ("innerhalb", 10, 10, 0.05), ("achsenabschnitt", 10, 10, 0.05)])
def test_anteil_am_rand_gleicht_der_f_formel(art, m, k, rho):
    rng = np.random.default_rng(2007)
    anzahl = 1500
    p = MX.p_singulaer_ausgewogen(m, k, rho, art)
    sim = _singulaer_simulation(rng, m, k, rho, art, anzahl)
    assert abs(sim - p) < 4 * np.sqrt(p * (1 - p) / anzahl) + 1e-9


def test_streuung_des_anova_schaetzers_gleicht_der_formel():
    rng = np.random.default_rng(2009)
    for m, k in ((12, 30), (6, 10)):
        su2, s2 = 0.09, 0.3025
        R = 20000
        yy = rng.normal(0, np.sqrt(su2), (R, m, 1)) + rng.normal(0, np.sqrt(s2), (R, m, k))
        cm = yy.mean(axis=2)
        msb = k * np.sum((cm - cm.mean(axis=1, keepdims=True)) ** 2, axis=1) / (m - 1)
        msw = np.sum((yy - cm[:, :, None]) ** 2, axis=(1, 2)) / (m * (k - 1))
        est = (msb - msw) / k
        assert est.std() == pytest.approx(MX.anova_sd(m, k, su2, s2), rel=0.04)
        assert abs(est.mean() - su2) < 4 * est.std() / np.sqrt(R)               # kein Hinweis auf Verzerrung


@pytest.mark.parametrize("m,k,rho", [(10, 10, 0.3), (6, 30, 0.1)])
def test_kunden_mittel_methode_deckt_im_ausgewogenen_fall_95_prozent(m, k, rho):
    rng = np.random.default_rng(2008)
    lam = rho / (1 - rho)
    R = 20000
    xc = np.r_[np.zeros(m // 2), np.ones(m // 2)]
    ybar = 1.0 + 0.5 * xc[None, :] + rng.normal(0, np.sqrt(lam + 1.0 / k), (R, m))
    treffer = 0
    for i in range(R):
        sl, se, q = MX.kmm_interval(xc, ybar[i], np.arange(m), m)                # ein Kunde je „Container“: Mittel auf Mittel, df = m − 2
        treffer += abs(sl - 0.5) < q * se
    se_mc = np.sqrt(0.95 * 0.05 / R)
    assert abs(treffer / R - 0.95) < 4 * se_mc
