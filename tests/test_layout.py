"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import clg_constants as C
import clg_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_und_des_hauptbereichs_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at, art) if w.key and not w.key.startswith(("preset_", "hand_btn_"))}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel


def _anderer_wert(regler, spec, state_key):
    """Ein gültiger Wert, der vom aktuellen verschieden ist und **keinem Preset entspricht** (sonst wäre dieses Preset danach aktiv)."""
    schluessel = {v: k for k, v in P.PRESET_KEYS.items()}.get(state_key)
    preset_werte = {p[schluessel] for p in C.PRESETS.values() if schluessel in p}
    if spec.options is not None:
        kandidaten = list(spec.options)
    elif isinstance(regler.value, str):
        kandidaten = ["1, 2 | 3, 5, 8", "1, 3 | 5, 9"]
    elif spec.lo is not None and spec.hi is not None:
        kandidaten = [spec.lo, spec.hi, (spec.lo + spec.hi) // 2]
    else:
        return not regler.value
    return next(o for o in kandidaten if o != regler.value and o not in preset_werte)


@pytest.mark.parametrize("state_key", [k for k in P.SETTING_SPECS if k not in ("merk_select", "reps_select", "hand_text", "mim_toggle")])
def test_jede_aenderung_eines_reglers_der_seitenleiste_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(regler, spec, state_key)).run()
    assert not at.exception
    assert "primary" not in _typen(at).values()
    # erneutes Anklicken markiert wieder
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


@pytest.mark.parametrize("key,art", [("merk_select", "radio"), ("reps_select", "select_slider"), ("hand_text", "text_input"), ("mim_toggle", "toggle")])
def test_jede_aenderung_eines_reglers_im_hauptbereich_hebt_die_hervorhebung_auf(key, art):
    """Die Regler beim Experiment stehen im Hauptbereich, aber in SETTING_SPECS: ihre Änderung hebt das aktive Preset ebenfalls auf."""
    at = _app()
    spec = P.SETTING_SPECS[key]
    regler = getattr(at, art)(key=key)
    regler.set_value(_anderer_wert(regler, spec, key)).run()
    assert not at.exception
    assert "primary" not in _typen(at).values()
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"
