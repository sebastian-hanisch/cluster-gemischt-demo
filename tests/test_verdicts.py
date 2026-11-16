"""Verdikte (reine Funktionen `verdict_*`): Schwellen und jede Zahl im Text an handgesetzten Zeilen; je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a Punkt 7)."""

import clg_constants as C
import clg_evaluation as ev


def _zeile(cover_rv, cover_zoll=None, ok=1000, m=12, k=30, su=0.3, rand_re=0, rand_ml=0, bedingt=None):
    keys_rv = ("naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "kmm")
    keys_z = ("naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "fe")
    cover_zoll = cover_zoll or [int(0.95 * ok)] * 7
    return dict(m=m, kbar=k, su=su, rho=su ** 2 / (su ** 2 + 0.55 ** 2), ok=ok, R=ok, mode="half", rand_re=rand_re, rand_ml=rand_ml,
                cover={"rv": dict(zip(keys_rv, cover_rv)), "zoll": dict(zip(keys_z, cover_zoll))},
                width={"rv": dict(w_naiv=1.0, w_cr1t=3.0, w_rez=2.8, w_ret=3.1, w_kmm=3.2), "zoll": dict(w_naiv=1.0, w_cr1t=1.0, w_rez=0.9, w_ret=1.0, w_fe=0.9)},
                bedingt=bedingt or {"rand": [rand_re, 0, 0, 0], "inner": [ok - rand_re, 900, 500, 900]})


def test_ueberdeckungs_verdikt_nennt_jede_zahl_und_die_verfehlenden_verfahren():
    z = _zeile([508, 910, 945, 900, 936, 958, 957])
    art, text = ev.verdict_coverage(z, "vorgerechnet")
    assert art == "warning"
    for erwartet in ("naiv 50,8 %", "CR1 mit z 91,0 %", "mit t(m−1) 94,5 %", "ML mit z 90,0 %", "REML mit z 93,6 %", "mit t(m−1) 95,8 %", "Kunden-Mittel-Methode 95,7 %",
                     "1000 Historien (vorgerechnet", "m = 12", "30 Container je Kunde", "ρ = 0,229", "0,7 Prozentpunkte"):
        assert erwartet in text, erwartet
    # unter 95 % um mehr als zwei Standardfehler (> 1,38 Punkte): naiv, CR1 z, CR1 t (94,5 liegt knapp darunter, 94,5 > 93,6: nein), ML z, REML z
    assert "naiv (OLS, t(N − p))" in text and "CR1, Normalquantil" in text and "ML, Normalquantil" in text and "REML, Normalquantil" in text
    assert "REML, t(m − 1)" not in text.split("Unter 95 %")[1] and "Kunden-Mittel-Methode, t" not in text.split("Unter 95 %")[1]


def test_ueberdeckungs_verdikt_ohne_clusterbefund_ist_erfolg():
    art, text = ev.verdict_coverage(_zeile([949, 909, 943, 952, 962, 978, 959], su=0.0), "live")
    assert art == "success" and "keinen Clusterbefund" in text and "Unter 95 %" not in text and "(live;" in text


def test_schwelle_des_naiven_verdikts_haengt_an_der_zahl_der_historien():
    """Bei 200 Historien ist der Standardfehler bei 95 % 1,54 Punkte, die Schwelle liegt bei 91,9 %; bei 1000 Historien bei 93,6 %."""
    z200 = _zeile([186, 180, 185, 186, 187, 190, 188], ok=200)                 # naiv 93,0 %: über der Schwelle für 200 Historien
    assert ev.verdict_coverage(z200, "live")[0] == "success"
    z1000 = _zeile([930, 900, 940, 940, 940, 950, 950], ok=1000)
    assert ev.verdict_coverage(z1000, "vorgerechnet")[0] == "warning"


def test_rand_verdikt_mit_zahlen_und_mit_zu_wenigen_historien():
    z = _zeile([850, 894, 928, 904, 922, 959, 949], su=0.10, rand_re=125, rand_ml=219, bedingt={"rand": [125, 112, 103, 94], "inner": [875, 847, 747, 835]})
    art, text = ev.verdict_rand(z)
    assert art == "info" and "REML in 125 (12,5 %)" in text and "ML in 219 (21,9 %)" in text
    assert "89,6 % (125 Historien)" in text and "naive 82,4 %" in text and "75,2 %" in text and "nicht am Rand 96,8 % (875 Historien)" in text
    kein = _zeile([950] * 7, rand_re=0)
    assert "keine Historie am Rand" in ev.verdict_rand(kein)[1]
    wenig = _zeile([950] * 7, su=0.2, rand_re=6, bedingt={"rand": [6, 3, 2, 2], "inner": [994, 931, 601, 914]})
    assert "nur 6 Historien, zu wenig zum Deuten" in ev.verdict_rand(wenig)[1] and "%" in ev.verdict_rand(wenig)[1]
    eins = _zeile([950] * 7, rand_re=1, rand_ml=1, bedingt={"rand": [1, 0, 0, 0], "inner": [999, 958, 508, 945]})
    assert "liegt nur 1 Historie," in ev.verdict_rand(eins)[1]


def test_rand_verdikt_ohne_kundeneffekt_sagt_dass_nichts_zu_finden_ist():
    z = _zeile([949, 909, 943, 952, 962, 978, 959], su=0.0, rand_re=546, rand_ml=722, bedingt={"rand": [546, 530, 520, 495], "inner": [454, 448, 429, 448]})
    assert "nichts zu finden" in ev.verdict_rand(z)[1] and "54,6 %" in ev.verdict_rand(z)[1]


def test_designverdikt_kundenmerkmal_zollhalt_und_ohne_kundeneffekt():
    d = dict(median=7.81, q05=7.69, q95=7.99, median_erwartet=8.12, formel=7.65, wirksam_median=46.1, n=360, rho=0.229, anzahl=300, art="rv", m=12, k=30, su=0.3)
    art, text = ev.verdict_design(d)
    assert art == "warning" and "7,81" in text and "7,69 bis 7,99" in text and "7,65" in text and "etwa 46 unabhängige" in text and "360 Container" in text
    z = dict(d, art="zoll", median=0.96, median_erwartet=0.98, q05=0.86, q95=1.14, wirksam_median=374.0)
    art, text = ev.verdict_design(z)
    assert art == "info" and "0,98" in text and "0,86 bis 1,14" in text and "374 von 360" in text and "Negativbefund" in text
    art, text = ev.verdict_design(dict(d, rho=0.0, median=1.0, wirksam_median=360.0))
    assert art == "info" and "alle 360 Container zählen für den Rahmenvertrag voll" in text


def test_handverdikte():
    art, text = ev.verdict_hand(ev.hand_analysis(ev.parse_hand(C.HAND_BEISPIEL)))
    assert art == "info" and "MSB = 26, MSW = 4" in text and "REML σ̂_u² = 11" in text and "ML 20/3" in text and "1,46" in text and "2,08" in text and "1,43" in text and "±8,96" in text and "±3,75" in text
    art, text = ev.verdict_hand(ev.hand_analysis(ev.parse_hand(C.HAND_RANDFALL)))
    assert art == "warning" and "MSB = 8/3, MSW = 10/3" in text and "−1/3" in text and "−7/9" in text and "σ̂_u² = 0" in text and "kein Absturz" in text
    art, text = ev.verdict_hand(ev.hand_analysis(ev.parse_hand("4 | 5 | 6")))
    assert art == "warning" and "nicht trennbar" in text
    art, text = ev.verdict_hand(ev.hand_analysis(ev.parse_hand("2, 2 | 2, 2")))
    assert art == "warning" and "alle Werte sind gleich" in text
    art, text = ev.verdict_hand(ev.hand_analysis(ev.parse_hand("1, 2 | 3 | 5, 9, 2")))
    assert art == "info" and "ungleich große Kunden" in text


def test_generator_und_su2_verdikt():
    art, text = ev.verdict_generator(dict(sd=0.3793, icc_real=0.3224, icc_nom=0.2293, anteil=0.092, R=20000, mittel=0.294, q05=0.193, q95=0.401))
    assert art == "info" and "0,379" in text and "0,322 statt 0,229" in text and "9,2 %" in text and "20000" in text and "0,294" in text and "0,193 bis 0,401" in text
    live = dict(m=12, kbar=30, su=0.3, ok=200, mean=0.091, sd=0.048, rand=0, mit_merkmal=False)
    grid = [dict(m=12, k=30, sd=0.045, ok=1000), dict(m=12, k=300, sd=0.041, ok=1000), dict(m=48, k=30, sd=0.022, ok=1000)]
    art, text = ev.verdict_su2(live, grid)
    assert "σ_u² = 0,090" in text and "ohne Rahmenvertrag" in text and "0,091" in text and "0,048" in text and "0,045" in text and "0,041" in text and "0,022" in text
    assert "mit Rahmenvertrag" in ev.verdict_su2(dict(live, mit_merkmal=True), [])[1] and ev.verdict_su2(live, [])[0] == "info"


def test_je_preset_die_erwartete_verdikt_zeile_aus_der_vorgerechneten_messreihe():
    """Kein Preset ohne Effekt meldet einen Effekt; jedes Preset mit Effekt meldet ihn (Messreihe 1000 Historien)."""
    base = dict(m=C.DEFAULT_M, k=C.DEFAULT_K, su=C.DEFAULT_SU, art=C.DEFAULT_ART)
    for name, preset in C.PRESETS.items():
        w = {**base, **{k: v for k, v in preset.items() if k in base}}
        row = ev.find_frozen_row(w["m"], w["k"], w["su"], w["art"])
        art_c, text_c = ev.verdict_coverage(row, "vorgerechnet")
        art_r, text_r = ev.verdict_rand(row)
        if name == "Ohne Cluster":
            assert art_c == "success" and "keinen Clusterbefund" in text_c
            assert "nichts zu finden" in text_r
        else:
            assert art_c == "warning" and "Unter 95 %" in text_c, name
            assert "nichts zu finden" not in text_r, name
    row_klein = ev.find_frozen_row(12, 30, 0.10, "half")
    assert row_klein["rand_re"] / row_klein["ok"] > 0.05 and "Am Rand deckt" in ev.verdict_rand(row_klein)[1]     # kleine Streuung: der Rand kommt vor
