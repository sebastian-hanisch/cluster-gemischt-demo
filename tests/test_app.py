"""End-to-end Smoke-Test via Streamlits AppTest: Standardeinstellungen, jeder Knopf, jeder Regler an seinen Grenzen, jedes Preset, Permalink, ungültige Eingabe
und der teuerste Fall am Regler-Maximum (mit Zeitgrenze statt Zeit-Assertion). Es darf kein Python-Fehler auftreten (insbesondere keine doppelten Diagramm-Schlüssel)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import clg_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen


def _fresh_app(query=None) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    for k, v in (query or {}).items():
        at.query_params[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_laedt_ohne_ausnahme_und_zeigt_die_kernfragen():
    at = _fresh_app()
    texte = " ".join(m.value for m in at.markdown)
    for stichwort in ("Kernfragen", "Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4", "Experiment 5", "Wo die Annahmen enden"):
        assert stichwort in texte or any(stichwort in s.value for s in at.subheader), stichwort
    assert [m.value for m in at.metric][0] == "360"
    assert next(e for e in at.expander if e.label == "📐 Mathematische Formulierung")       # Expander per Titel, nie per Index


def test_jeder_knopf_loest_keine_ausnahme_aus():
    labels = [b.label for b in _fresh_app().button]
    assert len(labels) >= len(C.PRESETS) + 4, "Presets, drei Datensatz-Knöpfe und der Zufalls-Seed-Knopf erwartet"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_jeder_regler_an_seinen_grenzen():
    at0 = _fresh_app()
    for index, s in enumerate(at0.slider):
        for kante in (s.min, s.max):
            at = _fresh_app()
            at.slider[index].set_value(kante).run(timeout=TIMEOUT)
            assert not at.exception, (s.label, kante, [str(e) for e in at.exception])
    for key, optionen in (("su_slider", C.SU_OPTIONS), ("reps_select", C.REPS_OPTIONS)):
        for opt in (optionen[0], optionen[-1]):
            at = _fresh_app()
            at.select_slider(key=key).set_value(opt).run(timeout=TIMEOUT)
            assert not at.exception, (key, opt, [str(e) for e in at.exception])


def test_auswahlfelder_und_umschalter():
    at = _fresh_app()
    for key in ("art_select",):
        for opt in C.ARTEN:
            at.selectbox(key=key).set_value(opt).run(timeout=TIMEOUT)
            assert not at.exception, (key, opt)
    for opt in C.MERKMALE:
        at.radio(key="merk_select").set_value(opt).run(timeout=TIMEOUT)
        assert not at.exception, opt
    at.toggle(key="mim_toggle").set_value(False).run(timeout=TIMEOUT)
    assert not at.exception


def test_permalink_uebernimmt_werte_begrenzt_und_eingerastet_und_ignoriert_ungueltiges():
    at = _fresh_app({"m": "999", "k": "33", "su": "0.31", "art": "two", "reps": "500", "merk": "gibt es nicht", "seed": "abc", "hand": "1, 2 | 3, 5"})
    assert at.slider(key="m_slider").value == C.M_MAX                      # begrenzt
    assert at.slider(key="k_slider").value == 30                           # auf das Raster 10 eingerastet
    assert at.select_slider(key="su_slider").value == 0.30                 # auf die nächste Auswahl eingerastet
    assert at.selectbox(key="art_select").value == "two"
    assert at.select_slider(key="reps_select").value == 500
    assert at.radio(key="merk_select").value == C.DEFAULT_MERKMAL           # ungültige Auswahl ignoriert
    assert at.number_input(key="seed_input").value == C.DEFAULT_SEED        # nicht lesbare Zahl ignoriert
    assert at.text_input(key="hand_text").value == "1, 2 | 3, 5"


def test_nicht_lesbare_eingabe_des_handbeispiels_zeigt_eine_meldung_und_keine_ausnahme():
    at = _fresh_app()
    at.text_input(key="hand_text").set_value("1, x | 2").run(timeout=TIMEOUT)
    assert not at.exception and any("nicht lesbar" in e.value for e in at.error)
    at.text_input(key="hand_text").set_value("4 | 5 | 6").run(timeout=TIMEOUT)          # Randfall der Randfälle: jeder Kunde ein Container
    assert not at.exception and not at.error


def test_handbeispiel_und_randfall_ueber_die_knoepfe():
    at = _fresh_app()
    at.button(key="hand_btn_2").click().run(timeout=TIMEOUT)
    assert at.text_input(key="hand_text").value == C.HAND_RANDFALL and not at.exception
    assert any("Rand σ̂_u² = 0" in w.value or "σ̂_u² = 0" in w.value for w in at.warning)
    at.button(key="hand_btn_1").click().run(timeout=TIMEOUT)
    assert at.text_input(key="hand_text").value == C.HAND_BEISPIEL
    at.button(key="hand_btn_3").click().run(timeout=TIMEOUT)
    assert at.text_input(key="hand_text").value not in (C.HAND_BEISPIEL, C.HAND_RANDFALL) and not at.exception


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_und_zeigt_seine_verdikte(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    meldungen = [w.value for w in list(at.warning) + list(at.success) + list(at.info)]
    assert any("Historien (vorgerechnet" in t for t in meldungen), name                  # jedes Preset hat eine vorgerechnete Zeile
    if name == "Ohne Cluster":
        assert any("keinen Clusterbefund" in t for t in meldungen)
    else:
        assert not any("keinen Clusterbefund" in t for t in meldungen)


def test_teuerster_fall_am_regler_maximum_mit_zeitgrenze():
    """m = 48, k̄ = 120, 1000 Wiederholungen gewählt: die Live-Rechnung wird begrenzt (ev.live_reps), die App antwortet innerhalb der Zeitgrenze (kein Zeit-Assert)."""
    at = _fresh_app()
    at.slider(key="m_slider").set_value(C.M_MAX)
    at.slider(key="k_slider").set_value(C.K_MAX)
    at.select_slider(key="reps_select").set_value(1000)
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("auf" in c.value and "begrenzt" in c.value for c in at.caption)
