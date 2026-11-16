"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Beziehungen, die gelten müssen, ohne dass man das richtige Ergebnis kennt; je 20 bis 50 Zufallsinstanzen, fester Zufalls-Seed."""

import numpy as np
import pytest

import clg_mixed as MX


def _instanz(rng, m=None, ausgewogen=False, merkmale=True):
    m = int(rng.integers(4, 9)) if m is None else m
    sizes = np.full(m, int(rng.integers(3, 7))) if ausgewogen else rng.integers(2, 8, m)
    g = np.repeat(np.arange(m), sizes)
    n = len(g)
    cols = [np.ones(n)]
    if merkmale:
        cols += [np.repeat(rng.standard_normal(m), sizes), rng.standard_normal(n)]
    X = np.column_stack(cols)
    y = X @ rng.uniform(-1, 1, X.shape[1]) + np.repeat(rng.normal(0, rng.uniform(0.2, 1.2), m), sizes) + rng.standard_normal(n)
    return X, y, g, m


def test_skalierung_der_antwort_skaliert_beta_und_varianzen():
    rng = np.random.default_rng(101)
    for _ in range(25):
        X, y, g, m = _instanz(rng)
        c = float(rng.uniform(0.3, 5.0))
        a, b = MX.MixedModel(X, y, g, m).fit(True), MX.MixedModel(X, c * y, g, m).fit(True)
        assert np.allclose(b.beta, c * a.beta, rtol=1e-6, atol=1e-8)
        assert b.sigma2 == pytest.approx(c * c * a.sigma2, rel=1e-6) and b.sigma_u2 == pytest.approx(c * c * a.sigma_u2, rel=1e-5, abs=1e-9)
        assert b.lam == pytest.approx(a.lam, rel=1e-5, abs=1e-9) and b.rand == a.rand


def test_verschiebung_der_antwort_aendert_nur_den_achsenabschnitt():
    rng = np.random.default_rng(102)
    for _ in range(25):
        X, y, g, m = _instanz(rng)
        s = float(rng.uniform(-5, 5))
        a, b = MX.MixedModel(X, y, g, m).fit(True), MX.MixedModel(X, y + s, g, m).fit(True)
        assert b.beta[0] == pytest.approx(a.beta[0] + s, rel=1e-6, abs=1e-7) and np.allclose(b.beta[1:], a.beta[1:], atol=1e-7)
        assert b.sigma_u2 == pytest.approx(a.sigma_u2, rel=1e-6, abs=1e-9) and np.allclose(b.cov, a.cov, rtol=1e-6, atol=1e-10)


def test_umbenennen_und_umsortieren_aendert_nichts():
    rng = np.random.default_rng(103)
    for _ in range(25):
        X, y, g, m = _instanz(rng)
        pk = rng.permutation(m)                       # Kunden umbenennen
        pz = rng.permutation(len(y))                  # Zeilen umsortieren
        a = MX.MixedModel(X, y, g, m).fit(True)
        b = MX.MixedModel(X[pz], y[pz], pk[g][pz], m).fit(True)
        assert np.allclose(a.beta, b.beta, rtol=1e-8, atol=1e-10) and a.sigma_u2 == pytest.approx(b.sigma_u2, rel=1e-7, abs=1e-10)
        bo, _c, _d = MX.ols(X, y)
        assert np.allclose(MX.cr1_cov(X, y, g, m, bo), MX.cr1_cov(X[pz], y[pz], pk[g][pz], m, bo), rtol=1e-9, atol=1e-13)


def test_merkmal_skalieren_und_verschieben_skaliert_die_steigung():
    rng = np.random.default_rng(104)
    for _ in range(25):
        X, y, g, m = _instanz(rng)
        a_, b_ = float(rng.uniform(0.4, 3.0)) * rng.choice([-1.0, 1.0]), float(rng.uniform(-2, 2))
        X2 = X.copy()
        X2[:, 2] = a_ * X[:, 2] + b_                                        # Merkmal innerhalb der Kunden
        f1, f2 = MX.MixedModel(X, y, g, m).fit(True), MX.MixedModel(X2, y, g, m).fit(True)
        assert f2.beta[2] == pytest.approx(f1.beta[2] / a_, rel=1e-6, abs=1e-8)
        assert f2.sigma_u2 == pytest.approx(f1.sigma_u2, rel=1e-6, abs=1e-9)
        assert f2.cov[2, 2] == pytest.approx(f1.cov[2, 2] / a_ ** 2, rel=1e-6)


def test_ausgewogen_und_spalten_je_kunde_konstant_oder_mittelwertfrei_gleicht_reml_der_ols():
    """GLS = OLS für jedes λ, wenn das Design ausgewogen ist und jede Spalte je Kunde konstant oder je Kunde mittelwertfrei ist."""
    rng = np.random.default_rng(105)
    for _ in range(40):
        m, k = int(rng.integers(4, 9)), int(rng.integers(3, 7))
        g = np.repeat(np.arange(m), k)
        xi = rng.standard_normal(m * k)
        xi = xi - np.repeat(xi.reshape(m, k).mean(axis=1), k)
        X = np.column_stack([np.ones(m * k), np.repeat(rng.standard_normal(m), k), xi])
        y = X @ np.array([1.0, 0.5, -0.3]) + np.repeat(rng.normal(0, 0.7, m), k) + rng.standard_normal(m * k)
        for reml in (True, False):
            f = MX.MixedModel(X, y, g, m).fit(reml)
            assert np.allclose(f.beta, MX.ols(X, y)[0], atol=1e-9)


def test_am_rand_ist_die_kovarianz_des_gemischten_modells_die_naive():
    rng = np.random.default_rng(106)
    gefunden = 0
    for _ in range(50):
        m, k = int(rng.integers(4, 9)), int(rng.integers(3, 7))
        g = np.repeat(np.arange(m), k)
        X = np.column_stack([np.ones(m * k), np.repeat(rng.standard_normal(m), k)])
        y = X @ np.array([1.0, 0.2]) + np.repeat(rng.normal(0, 0.05, m), k) + rng.standard_normal(m * k)     # kleine Kundenstreuung: oft am Rand
        f = MX.MixedModel(X, y, g, m).fit(True)
        if f.rand:
            gefunden += 1
            _b, naiv, _df = MX.ols(X, y)
            assert np.allclose(f.cov, naiv, rtol=1e-10, atol=1e-14)
    assert gefunden >= 10                                                       # der Zweig kommt in den Zufallsinstanzen vor (nicht leer)


def test_cluster_der_groesse_eins_geben_cr1_gleich_hc1():
    rng = np.random.default_rng(107)
    for _ in range(30):
        n, p = int(rng.integers(15, 40)), 3
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 2.0, 3.0]) + rng.standard_normal(n) * (1 + np.abs(X[:, 1]))
        b, _c, _d = MX.ols(X, y)
        e = y - X @ b
        A = np.linalg.inv(X.T @ X)
        hc1 = A @ (X.T * e ** 2) @ X @ A * n / (n - p)
        assert np.allclose(MX.cr1_cov(X, y, np.arange(n), n, b), hc1, rtol=1e-10, atol=1e-15)


def test_designeffekt_ist_eins_ohne_kundeneffekt_und_waechst_in_rho_und_k():
    g10 = np.repeat(np.arange(10), 10)
    X = np.column_stack([np.ones(100), np.repeat(np.r_[np.zeros(5), np.ones(5)], 10)])
    assert MX.design_effect(X, g10, 10, 0.0, 1.0, 1)["deff"] == pytest.approx(1.0, abs=1e-12)
    werte = [MX.design_effect(X, g10, 10, su2, 1.0, 1)["deff"] for su2 in (0.0, 0.1, 0.3, 1.0, 3.0)]
    assert all(a < b for a, b in zip(werte, werte[1:]))
    nach_k = []
    for k in (2, 5, 10, 30):
        g = np.repeat(np.arange(10), k)
        Xk = np.column_stack([np.ones(10 * k), np.repeat(np.r_[np.zeros(5), np.ones(5)], k)])
        nach_k.append(MX.design_effect(Xk, g, 10, 0.3, 1.0, 1)["deff"])
    assert all(a < b for a, b in zip(nach_k, nach_k[1:]))


def test_wirksame_zahl_ist_hoechstens_n_fuer_ein_kundenmerkmal_in_zufallsdesigns():
    rng = np.random.default_rng(108)
    for _ in range(40):
        X, _y, g, m = _instanz(rng, merkmale=True)
        d = MX.design_effect(X[:, :2], g, m, float(rng.uniform(0.0, 1.0)), 1.0, 1)
        assert d["wirksam"] <= len(g) + 1e-9


def test_reml_sigma_u2_mindestens_ml_ausgewogen_nur_achsenabschnitt():
    """Nur dort belegt (Handbeispiel 1 und die geschlossene Form): REML teilt durch m − 1, ML durch m. Mit Merkmalen oder ungleichen Kunden wird es nicht behauptet."""
    rng = np.random.default_rng(109)
    for _ in range(40):
        X, y, g, m = _instanz(rng, ausgewogen=True, merkmale=False)
        assert MX.MixedModel(X, y, g, m).fit(True).sigma_u2 >= MX.MixedModel(X, y, g, m).fit(False).sigma_u2 - 1e-12


def test_kunden_dummies_mit_kundenmerkmal_haben_rangmangel():
    rng = np.random.default_rng(110)
    for _ in range(20):
        m = int(rng.integers(3, 8))
        sizes = rng.integers(2, 6, m)
        g = np.repeat(np.arange(m), sizes)
        D = np.eye(m)[g]
        X = np.column_stack([D, np.repeat(rng.standard_normal(m), sizes)])
        assert np.linalg.matrix_rank(X) == m < X.shape[1]
