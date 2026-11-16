"""Claims: jede Zahl der README, soweit sie aus der vorgerechneten Messung (data/clg_frozen.json) oder aus einer Neu-Rechnung mit festem Zufalls-Seed stammt.

Die vorgerechneten Zahlen sind Zählungen (ganze Zahlen) und stehen exakt in der Datei; ihre Darstellung in der README wird als Zeichenkette gesucht. Neu-Rechnungen
(Designeffekt, Generator) prüfen Bänder, nie das letzte Bit. Die Zahlen der früheren Messung, die die README zitiert (93,0 %, 52,2 %, 12,5 %, 29,3 %, 3,76), stehen nicht im Test."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import clg_constants as C
import clg_evaluation as ev
import clg_mixed as MX
import clg_vehicle as V

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FZ = ev.load_frozen()


def row(m, k, su, mode="half"):
    r = ev.find_frozen_row(m, k, su, mode)
    assert r is not None, (m, k, su, mode)
    return r


def P(r, nm, key, nd=1):
    """Überdeckung als „50,8 %“ (aus der Zählung)."""
    return ev.de_pct(r["cover"][nm][key] / r["ok"], nd)


def Q(zaehler, n, nd=1):
    return ev.de_pct(zaehler / n, nd)


def _tabellenzeile(label, r):
    return "| " + " | ".join([label] + [P(r, "rv", k) for k in ("naiv", "cr1z", "cr1t", "mlz", "rez", "ret", "kmm")]) + " |"


TABELLE = [
    ("m = 6, k̄ = 30", row(6, 30, 0.30)), ("m = 12, k̄ = 30", row(12, 30, 0.30)), ("m = 24, k̄ = 30", row(24, 30, 0.30)), ("m = 48, k̄ = 30", row(48, 30, 0.30)),
    ("m = 12, k̄ = 10", row(12, 10, 0.30)), ("m = 12, k̄ = 120", row(12, 120, 0.30)),
    ("m = 12, k̄ = 30, σ_u = 0", row(12, 30, 0.0)), ("m = 12, k̄ = 30, σ_u = 0,10", row(12, 30, 0.10)), ("m = 12, k̄ = 30, σ_u = 0,20", row(12, 30, 0.20)),
    ("m = 12, k̄ = 30, σ_u = 0,45", row(12, 30, 0.45)),
    ("m = 12, k̄ = 30, Merkmal bei nur 2 Kunden", row(12, 30, 0.30, "two")), ("m = 12, k̄ = 30, Merkmal stetig", row(12, 30, 0.30, "cont")),
    ("m = 6, k̄ = 30, Merkmal stetig", row(6, 30, 0.30, "cont")),
]


@pytest.mark.parametrize("label,r", TABELLE, ids=[t[0] for t in TABELLE])
def test_zeile_der_ueberdeckungstabelle_steht_so_in_der_readme(label, r):
    assert r["ok"] == 1000
    assert _tabellenzeile(label, r) in README


def test_kernbotschaft_zahlen():
    r12, r120, r6, r2 = row(12, 30, 0.30), row(12, 120, 0.30), row(6, 30, 0.30), row(12, 30, 0.30, "two")
    r10 = row(12, 30, 0.10)
    assert f"deckt das **naive 95-%-Intervall** des Rahmenvertrags nur **{P(r12, 'rv', 'naiv')}**, bei im Mittel 120 Containern je Kunde **{P(r120, 'rv', 'naiv')}**" in README
    assert f"nahe an 95 % ({P(r12, 'rv', 'cr1t')} und {P(r12, 'rv', 'ret')}), die Kunden-Mittel-Methode auch ({P(r12, 'rv', 'kmm')})" in README
    assert f"deckt CR1 mit t(m−1) nur **{P(r2, 'rv', 'cr1t')}**, REML mit t(m−1) {P(r2, 'rv', 'ret')} und die Kunden-Mittel-Methode {P(r2, 'rv', 'kmm')}" in README
    assert f"in {Q(r10['rand_re'], r10['ok'])} der Historien (ML: {Q(r10['rand_ml'], r10['ok'])})" in README
    rd, inn = r10["bedingt"]["rand"], r10["bedingt"]["inner"]
    assert f"nur {Q(rd[1], rd[0])} gegen {Q(inn[1], inn[0])} sonst" in README
    assert f"deckt das naive Intervall {P(row(12, 30, 0.0), 'rv', 'naiv')}" in README
    assert r6["ok"] == 1000


def test_zollhalt_naiv_nicht_zu_schmal_in_allen_zeilen():
    naiv = [r["cover"]["zoll"]["naiv"] / r["ok"] for r in FZ["rows"]]
    ret = [r["cover"]["zoll"]["ret"] / r["ok"] for r in FZ["rows"]]
    fe = [r["cover"]["zoll"]["fe"] / r["ok"] for r in FZ["rows"]]
    assert len(naiv) == 20
    assert (min(naiv), max(naiv)) == (0.944, 0.964) and (min(ret), max(ret)) == (0.955, 0.992) and (min(fe), max(fe)) == (0.941, 0.966)
    assert "(Überdeckung 94,4 bis 96,4 %)" in README and "(95,5 bis 99,2 %)" in README and "94,1 bis 96,6 %" in README
    assert not any(ev.unter_soll(v, 1000) for v in naiv)                         # keine Zeile liegt mehr als zwei Standardfehler unter 95 %


def test_breiten_zollhalt():
    w = row(12, 30, 0.30)["width"]["zoll"]
    assert w["w_rez"] / w["w_naiv"] == pytest.approx(0.90, abs=0.005) and w["w_fe"] / w["w_naiv"] == pytest.approx(0.91, abs=0.005)
    assert w["w_ret"] / w["w_naiv"] == pytest.approx(1.01, abs=0.005)
    w45 = row(12, 30, 0.45)["width"]["zoll"]
    assert w45["w_rez"] / w45["w_naiv"] == pytest.approx(0.81, abs=0.005)
    assert "REML mit z 0,90-fach der naiven Breite, feste Kunden-Effekte 0,91-fach, REML mit t(m−1) 1,01-fach, bei σ_u = 0,45 REML z 0,81-fach" in README


def test_anteil_am_rand():
    r0, r1, r2, r6 = row(12, 30, 0.0), row(12, 30, 0.10), row(12, 30, 0.20), row(6, 30, 0.30)
    assert f"σ_u = 0: REML {Q(r0['rand_re'], 1000)}, ML {Q(r0['rand_ml'], 1000)}; σ_u = 0,10: {Q(r1['rand_re'], 1000)} und {Q(r1['rand_ml'], 1000)}" in README
    assert f"σ_u = 0,20: {Q(r2['rand_re'], 1000)} ({r2['rand_re']} Historien, zu wenige zum Deuten) und {Q(r2['rand_ml'], 1000)}" in README
    assert f"{Q(r6['rand_re'], 1000)} ({r6['rand_re']} Historien)" in README
    assert (r0["rand_re"], r0["rand_ml"], r1["rand_re"], r1["rand_ml"], r2["rand_re"], r6["rand_re"]) == (546, 722, 111, 194, 7, 22)


def test_ueberdeckung_bedingt_auf_den_rand():
    r = row(12, 30, 0.10)
    rd, inn = r["bedingt"]["rand"], r["bedingt"]["inner"]
    assert rd[0] + inn[0] == 1000
    assert f"REML t(m−1) {Q(rd[1], rd[0])} ({rd[0]} Historien am Rand) gegen {Q(inn[1], inn[0])} ({inn[0]} Historien sonst); naiv am Rand {Q(rd[2], rd[0])}, CR1 t(m−1) {Q(rd[3], rd[0])}" in README
    assert (rd, inn) == ([111, 98, 95, 92], [889, 860, 755, 850])


def test_formelwerte_fuer_den_rand():
    f1, f0, f30 = (MX.p_singulaer_ausgewogen(10, 10, 0.05, "zwischen"), MX.p_singulaer_ausgewogen(10, 10, 0.0, "zwischen"), MX.p_singulaer_ausgewogen(30, 10, 0.05, "zwischen"))
    assert (ev.de_pct(f1), ev.de_pct(f0), ev.de_pct(f30)) == ("27,1 %", "55,8 %", "9,0 %")
    assert "27,1 % (m = 10, k = 10, ρ = 0,05, Kundenmerkmal), 55,8 % (ρ = 0), 9,0 % (m = 30)" in README


def test_streuung_der_varianzkomponente():
    sd = {(g["m"], g["k"]): g["sd"] for g in ev.su2_grid_table()}
    assert [ev.de(sd[k], 3) for k in ((12, 30), (12, 10), (12, 300), (48, 30), (6, 30))] == ["0,045", "0,054", "0,041", "0,022", "0,073"]
    assert "m = 12: 0,045 (30 Container je Kunde), 0,054 (10), 0,041 (300); m = 48: 0,022 (30); m = 6: 0,073 (30)" in README
    assert (ev.de(MX.sd_grenze_viele_container(12, 0.09), 3), ev.de(MX.sd_grenze_viele_container(48, 0.09), 3)) == ("0,038", "0,019")
    assert "ist 0,038 bei m = 12 und 0,019 bei m = 48" in README
    # mehr Kunden senken die Streuung, mehr Container je Kunde senken sie nur bis zur Grenze (kein Wert unter 0,9 mal der Grenze)
    for m in C.SU2_GRID_M:
        assert sd[(m, 10)] > sd[(m, 30)] > sd[(m, 300)] > 0.9 * MX.sd_grenze_viele_container(m, 0.09)


def test_varianzkomponente_ohne_kundenmerkmal():
    mit, ohne = FZ["su2_merkmal"]["mit"], FZ["su2_merkmal"]["ohne"]
    assert (ev.de(mit["mean"], 3), ev.de(ohne["mean"], 3)) == ("0,091", "0,117") and mit["ok"] == ohne["ok"] == 1000
    assert "σ̂_u² im Mittel 0,117 statt 0,091 (mit dem Merkmal); erwartet 0,1125 (0,09 + 0,30²·0,25)" in README
    assert 0.09 + 0.30 ** 2 * 0.25 == pytest.approx(0.1125)                       # Varianz einer halb/halb verteilten 0/1-Wirkung −0,30: 0,30²·(1/2)·(1/2)


def test_zwoelf_kunden_des_generators():
    g = ev.generator_summary()
    assert (ev.de(g["sd"], 3), ev.de(g["icc_real"], 3), ev.de(g["icc_nom"], 3)) == ("0,379", "0,322", "0,229")
    assert 0.088 < g["anteil"] < 0.098 and ev.de_pct(g["anteil"]) == "9,3 %"
    assert "Streuung 0,379 statt nominal 0,30 (Intraklassenkorrelation 0,322 statt 0,229); in 9,3 % von 20000 Ziehungen" in README


def test_designeffekt_im_vehikel():
    d = V.design_effects(12, 30, 0.30, "rv", "half", 5, 2000)
    assert d["median"] == pytest.approx(7.82, abs=0.03) and d["q05"] == pytest.approx(7.69, abs=0.03) and d["q95"] == pytest.approx(7.99, abs=0.03)
    assert d["formel"] == pytest.approx(7.65, abs=0.005) and d["wirksam_median"] == pytest.approx(46, abs=1.0)
    assert "exakt 7,82 im Median (5 bis 95 % der Zuordnungen: 7,69 bis 7,99); die Formel 1 + (k̄ − 1)ρ mit mittlerer Kundengröße gibt 7,65; wirksam etwa 46 von 360" in README
    # Abweichung der Formel mit mittlerer Kundengröße vom exakten Wert über die 2000 Zuordnungen
    seeds = V.splitmix_array(5, 2000)
    abw = []
    for s in seeds:
        X, g, me = V.draw_design(int(s), 12, 30, "rv", "half")
        if np.ptp(X[:, 1]) == 0:
            continue
        abw.append(abs(d["formel"] / MX.design_effect(X, g, me, 0.09, 0.3025, 1)["deff"] - 1))
    assert 0.04 < max(abw) < 0.10 and "bis zu 7 %" in README                       # größte Abweichung (hier 7 %), Median der Abweichung unter 3 %
    assert float(np.median(abw)) < 0.03


def test_designeffekt_zollhalt_ueber_36_einstellungen():
    werte = {}
    for m in (6, 12, 48):
        for k in (10, 30, 120):
            for su in (0.0, 0.1, 0.3, 0.45):
                werte[(m, k, su)] = V.design_effects(m, k, su, "zoll", "half", 5, 1000)["median_erwartet"]
    assert len(werte) == 36 and min(werte.values()) == pytest.approx(0.95, abs=0.01) and max(werte.values()) == pytest.approx(1.0, abs=0.01)
    assert "im Median zwischen 0,95 und 1,00 über 36 Einstellungen" in README
    z = V.design_effects(12, 30, 0.30, "zoll", "half", 5, 2000)
    assert z["median_erwartet"] == pytest.approx(0.99, abs=0.01) and "0,99 der erwarteten naiven" in README


def _tools_modul():
    spec = importlib.util.spec_from_file_location("clg_make_frozen", ROOT / "tools" / "clg_make_frozen.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clg_make_frozen"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_vorgerechnete_zeilen_stimmen_mit_neuer_rechnung():
    """Das Einfrieren passt zur Suite: die ersten drei Zeilen neu gerechnet, Vergleich als Band (Zählungen ± 2). `tools/clg_make_frozen.py --check` macht es für alle 20 Zeilen."""
    mod = _tools_modul()
    neu = mod.compute_rows(3)
    assert len(neu) == 3
    for a, n in zip(FZ["rows"], neu):
        assert mod.vergleiche(a, n) == [], (a["m"], a["kbar"], a["su"])
    assert mod.vergleiche(FZ["rows"][0], FZ["rows"][1])                              # und der Vergleich schlägt bei verschiedenen Zeilen an (nicht leer)


def test_readme_nennt_die_korrekturen_der_fruehen_messung_und_ihre_neuen_werte():
    r2, r0, r1, r120 = row(12, 30, 0.30, "two"), row(12, 30, 0.0), row(12, 30, 0.10), row(12, 120, 0.30)
    assert f"früher 93,0 %, jetzt {P(r2, 'rv', 'ret')}" in README
    assert f"früher 52,2 % bei σ_u = 0 und 12,5 % bei σ_u = 0,10 (REML), jetzt {Q(r0['rand_re'], 1000)} und {Q(r1['rand_re'], 1000)}" in README
    assert f"früher 29,3 %, jetzt {P(r120, 'rv', 'naiv')}" in README


def test_readme_hat_keine_planungswoerter_und_keine_platzhalter():
    for wort in ("Stück", "Linie", "Vormessung", "TESTZAHL", "TODO:", "trägt", "tragen", "Saatwert", "läuft lokal"):
        assert wort not in README.replace("TODO nach dem Deploy", ""), wort
