"""Vehikel (clg_vehicle): Konstanten, Ziehung einer Historie, Simulation, Designs. Das Modell der Antwort wird unabhängig mit Kleinsten Quadraten zurückgelesen."""

import math

import numpy as np
import pytest

import clg_vehicle as V


def test_konstanten_des_import_zweigs():
    assert V.LN35 == pytest.approx(math.log(3.5), abs=1e-15)
    assert V.S_IMPORT == 0.55 and V.P_ZOLL == 0.25 and V.P_WE == pytest.approx(2 / 7)
    assert list(V.BETA_TRUE) == [pytest.approx(math.log(3.5)), -0.30, 0.7, 0.3]


def test_zwoelf_kundeneffekte_des_generators():
    uk = np.array(V.GENERATOR_UK)
    assert len(uk) == 12
    assert np.std(uk, ddof=1) == pytest.approx(0.3793, abs=5e-5) and np.var(uk, ddof=1) == pytest.approx(0.1439, abs=5e-5)
    ref = np.random.default_rng(12345).normal(0.0, 0.30, 12)                     # Herkunft laut Modulkopf; der numpy-Strom kann sich zwischen Versionen ändern
    if not np.allclose(uk, ref, rtol=0, atol=1e-12):
        pytest.skip("numpy erzeugt für den Zufalls-Seed 12345 eine andere Folge als bei der Übernahme der Zahlen (Versionsdrift des numpy-Stroms)")


def test_historie_ist_deterministisch_und_vom_zufalls_seed_abhaengig():
    a, b, c = V.draw_history(5, 12, 30, 0.3), V.draw_history(5, 12, 30, 0.3), V.draw_history(6, 12, 30, 0.3)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.g, b.g) and np.array_equal(a.X, b.X)
    assert not np.array_equal(a.y, c.y)


def test_historie_festwerte_des_zufallsstroms():
    """Pin des Zufallsstroms (nur Grundrechenarten, auf jeder Plattform dieselben Bits): Zufalls-Seed 5, m = 12, k̄ = 30, σ_u = 0,3. Die Werte sind die Ausgabe des Codes
    beim Bau, ein Regressionsanker (kein unabhängiger Wert): ändert sich der Strom, ändern sich alle vorgerechneten Zeilen."""
    h = V.draw_history(5, 12, 30, 0.3)
    assert h.X.shape == (360, 4) and h.m == 12 and list(h.g[:5]) == [4, 9, 2, 1, 2]
    assert list(h.y[:3]) == pytest.approx([1.4957395165524978, 1.623525080479922, 1.6439793041789044], abs=1e-9)
    assert float(h.y.sum()) == pytest.approx(524.2567363549433, abs=1e-8)


def test_aufbau_der_spalten_und_kundenmerkmal_je_kunde_konstant():
    for mode in ("half", "two", "cont"):
        h = V.draw_history(11, 12, 30, 0.3, mode)
        assert set(np.unique(h.X[:, 2])) <= {0.0, 1.0} and set(np.unique(h.X[:, 3])) <= {0.0, 1.0} and np.all(h.X[:, 0] == 1.0)
        for c in range(h.m):
            assert len(np.unique(h.X[h.g == c, 1])) == 1                       # Rahmenvertrag ist ein Kundenmerkmal
        werte = np.array([h.X[h.g == c, 1][0] for c in range(h.m)])
        if mode == "two":
            assert set(np.unique(werte)) <= {0.0, 1.0} and werte.sum() <= 2
        if mode == "half":
            assert werte.sum() <= 6 and set(np.unique(werte)) <= {0.0, 1.0}
    with pytest.raises(ValueError):
        V.draw_history(1, 12, 30, 0.3, "unbekannt")


def test_modell_der_antwort_wird_mit_kleinsten_quadraten_zurueckgelesen():
    """Große Historie (m = 12, 25 000 Container je Kunde): Kunden-Dummies, Zollhalt und Wochenende zurückgelesen, Streuung im Kunden und Anteile."""
    h = V.draw_history(77, 12, 25000, 0.3)
    n = len(h.y)
    D = np.eye(h.m)[h.g]
    Xf = np.column_stack([D, h.X[:, 2], h.X[:, 3]])
    b = np.linalg.lstsq(Xf, h.y, rcond=None)[0]
    res = h.y - Xf @ b
    assert b[-2] == pytest.approx(0.7, abs=0.01) and b[-1] == pytest.approx(0.3, abs=0.01)
    assert res.std() == pytest.approx(0.55, abs=0.004)
    assert h.X[:, 2].mean() == pytest.approx(0.25, abs=0.004) and h.X[:, 3].mean() == pytest.approx(2 / 7, abs=0.004)
    zaehl = np.bincount(h.g, minlength=h.m)
    erwartet = n / h.m
    assert ((zaehl - erwartet) ** 2 / erwartet).sum() < 31.3                    # Chi-Quadrat mit 11 Freiheitsgraden, Schwelle 0,1 % (Kunde je Container gleichverteilt)
    # Koeffizient des Kunden k ist ln 3,5 + β_R·Rahmenvertrag_k + u_k (Zollhalt und Wochenende sind herausgerechnet)
    erwartet = np.array([V.LN35 + V.BETA_R * h.X[h.g == c, 1][0] + h.u[c] for c in range(h.m)])
    assert np.max(np.abs(b[:h.m] - erwartet)) < 0.02


def test_feste_kundeneffekte_ersetzen_die_frische_ziehung():
    uk = np.array(V.GENERATOR_UK)
    h = V.draw_history(3, 12, 30, 0.99, u_fixed=uk)
    assert np.array_equal(h.u, uk)                                              # σ_u wird dann nicht benutzt


def test_leere_kunden_fallen_aus():
    h = V.draw_history(2, 48, 1, 0.3)                                           # 48 Container auf 48 Kunden: viele Kunden bleiben leer
    assert h.m < 48 and h.g.max() == h.m - 1 and len(np.unique(h.g)) == h.m


def test_intervalle_einer_historie_stimmen_mit_den_zaehlern_der_simulation_ueberein():
    for seed in range(6):
        h = V.draw_history(seed, 12, 30, 0.3)
        o = V.analyse_history(h)
        iv = {x["key"]: x for x in V.history_intervals(h, 1)}
        for key in ("naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "kmm"):
            assert iv[key]["trifft"] == o["rv"][key], key
        ivz = {x["key"]: x for x in V.history_intervals(h, 2)}
        for key in ("naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "fe"):
            assert ivz[key]["trifft"] == o["zoll"][key], key
        assert all(x["lo"] < x["est"] < x["hi"] for x in iv.values())


def test_simulation_zaehlt_historien_und_ist_deterministisch():
    r1, r2 = V.simulate(12, 30, 0.3, 40, 9), V.simulate(12, 30, 0.3, 40, 9)
    assert r1 == r2 and r1["ok"] == 40 and r1["R"] == 40
    for nm in ("rv", "zoll"):
        assert all(0 <= v <= 40 for v in r1["cover"][nm].values())
    assert sum(r1["bedingt"][k][0] for k in ("rand", "inner")) == 40 and r1["bedingt"]["rand"][0] == r1["rand_re"]
    assert V.share(r1, "rv", "naiv") == r1["cover"]["rv"]["naiv"] / 40
    assert V.se_share(0.5, 100) == pytest.approx(0.05) and V.se_share(1.0, 10) == 0.0


def test_naives_intervall_verfehlt_das_kundenmerkmal_deutlich_ohne_kundeneffekt_nicht():
    """Gegenprobe der Richtung: mit Kundeneffekt (σ_u = 0,3, 200 Historien) deckt das naive Intervall deutlich weniger als ohne (σ_u = 0)."""
    mit, ohne = V.simulate(12, 30, 0.3, 200, 21), V.simulate(12, 30, 0.0, 200, 21)
    assert V.share(mit, "rv", "naiv") < 0.65 < 0.88 < V.share(ohne, "rv", "naiv")


def test_designeffekte_im_vehikel():
    d = V.design_effects(12, 30, 0.3, "rv", "half", 5, 600)
    rho = 0.09 / (0.09 + 0.3025)
    assert d["formel"] == pytest.approx(1 + 29 * rho, rel=1e-12) and d["n"] == 360 and d["anzahl"] == 600
    assert 7.6 < d["median"] < 8.0 and d["q05"] < d["median"] < d["q95"] and 44 < d["wirksam_median"] < 48
    z = V.design_effects(12, 30, 0.3, "zoll", "half", 5, 300)
    assert 0.9 < z["median_erwartet"] < 1.05 and z["wirksam_median"] > 300             # ein Merkmal im Kunden: fast die volle Zeilenzahl
    n0 = V.design_effects(12, 30, 0.0, "rv", "half", 5, 100)
    assert n0["median"] == pytest.approx(1.0, abs=1e-9) and n0["wirksam_median"] == pytest.approx(360.0, abs=1e-6)


def test_stichproben_streuung_von_zwoelf_ziehungen():
    sds = V.sd_of_normal_draws(12, 0.30, 20000, 5000)
    assert sds.mean() == pytest.approx(0.294, abs=0.003)                          # E[s] = c4(12)·σ mit c4 = 0,9776
    assert 0.085 < float(np.mean(sds >= 0.3793)) < 0.100


def test_streuung_der_varianzkomponente_mit_und_ohne_kundenmerkmal():
    a = V.su2_series(12, 30, 0.3, 300, 4900, "half", True)
    b = V.su2_series(12, 30, 0.3, 300, 4900, "half", False)
    assert a["ok"] == b["ok"] == 300 and len(a["werte"]) == 300
    assert 0.01 < b["mean"] - a["mean"] < 0.035                                   # erwartet 0,0225 (0,30²·0,25), Standardfehler der Differenz je Reihe etwa 0,003
    assert a["mean"] == pytest.approx(0.09, abs=0.012)
