"""Auswertung (clg_evaluation): Eingabe des Handbeispiels, Tabellen, Begrenzung der Live-Rechnung, vorgerechnete Zeilen."""

from fractions import Fraction as F

import pytest

import clg_constants as C
import clg_evaluation as ev


def test_formatierung_mit_komma_und_minuszeichen():
    assert ev.de(1.5, 1) == "1,5" and ev.de(-0.25, 2) == "−0,25" and ev.de(float("nan")) == "–"
    assert ev.de_pct(0.508) == "50,8 %" and ev.pp(0.0069) == "0,7"
    assert ev.frac_str(F(20, 3)) == "20/3" and ev.frac_str(F(-1, 3)) == "−1/3" and ev.frac_str(F(11)) == "11"
    assert ev.frac_zahl(F(20, 3)) == "20/3 (≈ 6,667)" and ev.frac_zahl(F(11)) == "11"


def test_eingabe_lesen():
    assert ev.parse_hand("1, 3 | 5, 9 | 8, 10") == [[1, 3], [5, 9], [8, 10]]
    assert ev.parse_hand("1/3, 2.5 | 4") == [[F(1, 3), F(5, 2)], [4]]
    assert ev.parse_hand("1; 2 | 3") == [[1, 2], [3]]                              # Semikolon wie Komma
    for schlecht, stichwort in (("1, 2", "Mindestens 2 Kunden"), ("1, x | 2", "keine Zahl"), ("1, 2 | | 3", "keine Zahl"), ("1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9", "Höchstens"),
                                ("1, 2, 3, 4, 5, 6, 7, 8, 9 | 1", "höchstens")):
        with pytest.raises(ValueError, match=stichwort):
            ev.parse_hand(schlecht)


def test_zufallsdatensatz_ist_deterministisch_und_in_den_grenzen():
    assert ev.random_hand(2026) == ev.random_hand(2026) != ev.random_hand(2027)
    for seed in range(40):
        rows = ev.parse_hand(ev.random_hand(seed))
        assert 3 <= len(rows) <= 6 and len({len(r) for r in rows}) == 1 and 2 <= len(rows[0]) <= 4
        assert all(0 <= v < 20 and v.denominator == 1 for r in rows for v in r)


def test_handbeispiel_analyse_und_tabelle():
    a = ev.hand_analysis(ev.parse_hand(C.HAND_BEISPIEL))
    assert a.ausgewogen and a.anova["reml"] == 11 and a.m == 3 and a.n == 6
    assert a.se_naiv == pytest.approx((32 / 15) ** 0.5) and a.se_gemischt == pytest.approx((13 / 3) ** 0.5, rel=1e-9)
    assert a.halb_gemischt == pytest.approx(8.96, abs=0.005) and a.halb_naiv == pytest.approx(3.755, abs=0.005)
    t = {r["Größe"]: r for r in ev.hand_table(a)}
    assert t["σ̂_u² (Kundenstreuung)"]["REML"] == "11" and t["σ̂_u² (Kundenstreuung)"]["ML"] == "20/3 (≈ 6,667)"
    assert t["ρ̂ = σ̂_u²/(σ̂_u² + σ̂²)"]["REML"] == "11/15 (≈ 0,733)" and t["ρ̂ = σ̂_u²/(σ̂_u² + σ̂²)"]["ML"] == "5/8 (≈ 0,625)"
    assert a.krit_reml.min() == 0.0 and a.krit_ml.min() == 0.0 and len(a.rho_plot) == len(a.krit_reml)


def test_randfall_ist_gekennzeichnet():
    a = ev.hand_analysis(ev.parse_hand(C.HAND_RANDFALL))
    t = {r["Größe"]: r for r in ev.hand_table(a)}
    assert a.reml.rand and a.ml.rand
    assert t["σ̂_u² (Kundenstreuung)"]["REML"].endswith("(Rand)") and t["σ̂² (Streuung im Kunden)"]["REML"] == "46/15 (≈ 3,067)"


def test_ungleiche_kunden_haben_keine_bruchrechnung():
    a = ev.hand_analysis(ev.parse_hand("1, 2 | 3 | 5, 9, 2"))
    assert not a.ausgewogen and a.anova is None
    assert all("≈" not in str(r["REML"]) and "≈" not in str(r["ML"]) for r in ev.hand_table(a))        # Zahlen, keine Brüche
    assert a.reml.sigma_u2 > 0 and not a.reml.rand


def test_live_rechnung_wird_auf_das_zeitbudget_begrenzt():
    assert ev.live_reps(12, 30, 200) == 200 and ev.live_reps(12, 30, 1000) == 1000
    assert ev.live_reps(48, 120, 1000) < 1000 and ev.live_reps(48, 120, 100) == 100
    je = 1.4 + 0.0004 * 48 * 120
    assert ev.live_reps(48, 120, 1000) * je <= ev.LIVE_BUDGET_MS and ev.live_reps(48, 120, 1000) > 0.9 * ev.LIVE_BUDGET_MS / je
    assert ev.live_reps(48, 10 ** 6, 1000) == 20                                    # Untergrenze


def test_vorgerechnete_zeilen_passen_zu_den_konstanten():
    fz = ev.load_frozen()
    assert len(fz["rows"]) == len(C.FROZEN_ROWS) and fz["R"] == C.FROZEN_R
    for r, (m, k, su, mode) in zip(fz["rows"], C.FROZEN_ROWS):
        assert (r["m"], r["kbar"], r["su"], r["mode"]) == (m, k, su, mode) and r["R"] == C.FROZEN_R and r["ok"] == C.FROZEN_R
        assert ev.find_frozen_row(m, k, su, mode) is r
        for nm in ("rv", "zoll"):
            assert all(0 <= v <= r["ok"] for v in r["cover"][nm].values())
        assert r["bedingt"]["rand"][0] == r["rand_re"] and r["bedingt"]["rand"][0] + r["bedingt"]["inner"][0] == r["ok"]
    assert ev.find_frozen_row(7, 30, 0.3, "half") is None


def test_jede_voreinstellung_und_jedes_preset_hat_eine_vorgerechnete_zeile():
    base = dict(m=C.DEFAULT_M, k=C.DEFAULT_K, su=C.DEFAULT_SU, art=C.DEFAULT_ART)
    for name, preset in C.PRESETS.items():
        werte = {**base, **{k: v for k, v in preset.items() if k in base}}
        assert ev.find_frozen_row(werte["m"], werte["k"], werte["su"], werte["art"]) is not None, name


def test_ueberdeckungstabelle_und_sollwert():
    r = ev.find_frozen_row(12, 30, 0.3, "half")
    tab = {t["key"]: t for t in ev.coverage_table(r, "rv")}
    assert set(tab) == {"naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "kmm"}
    assert tab["naiv"]["share"] == r["cover"]["rv"]["naiv"] / 1000 and tab["naiv"]["breite"] == pytest.approx(1.0)
    assert tab["cr1z"]["breite"] is None and tab["ret"]["breite"] > tab["rez"]["breite"]            # t(m−1) ist breiter als das Normalquantil
    assert set(t["key"] for t in ev.coverage_table(r, "zoll")) == {"naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "fe"}
    assert ev.unter_soll(0.93, 1000) and not ev.unter_soll(0.94, 1000) and ev.ueber_soll(0.97, 1000) and not ev.ueber_soll(0.96, 1000)
    assert ev.unter_soll(0.91, 200) and not ev.unter_soll(0.93, 200)                  # kleinere Reihe: größerer Standardfehler, breiterer Zweifelsbereich


def test_generator_und_streuung_der_varianzkomponente():
    g = ev.generator_summary(R=5000, seed=5000)
    assert g["sd"] == pytest.approx(0.3793, abs=5e-5) and g["icc_real"] == pytest.approx(0.3224, abs=5e-4) and g["icc_nom"] == pytest.approx(0.2293, abs=5e-4)
    assert 0.07 < g["anteil"] < 0.12
    grid = ev.su2_grid_table()
    assert len(grid) == len(C.SU2_GRID_M) * len(C.SU2_GRID_K)
    sd = {(x["m"], x["k"]): x["sd"] for x in grid}
    for m in C.SU2_GRID_M:
        assert sd[(m, 10)] > sd[(m, 300)] > 0.9 * ev.MX.sd_grenze_viele_container(m, 0.09)         # mehr Container senken die Streuung, aber nicht unter die Grenze
    assert sd[(6, 30)] > sd[(12, 30)] > sd[(48, 30)]                                   # mehr Kunden senken sie weiter
