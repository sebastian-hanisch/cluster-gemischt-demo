"""Presets und Permalink: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen, Einrasten."""

import clg_constants as C
import clg_evaluation as ev
import clg_presets as P


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[n] for n in C.PRESETS)
    assert set(k for p in C.PRESETS.values() for k in p) <= set(P.PRESET_KEYS)


def test_preset_values_are_valid_for_their_controls():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            assert P.snap(spec, wert) == wert, (name, key)                     # gültig und schon auf dem Raster


def test_first_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {}
    assert ev.Settings() == ev.Settings(m=C.DEFAULT_M, k=C.DEFAULT_K, su=C.DEFAULT_SU, art=C.DEFAULT_ART, seed=C.DEFAULT_SEED, hand=C.HAND_BEISPIEL)


def test_every_setting_has_a_default_and_a_url_name_and_urls_are_unique():
    assert all(s.default is not None for s in P.SETTING_SPECS.values())
    urls = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(urls) == len(set(urls))
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_snap_begrenzt_und_rastet_ein():
    assert P.snap(P.SETTING_SPECS["m_slider"], 999) == C.M_MAX and P.snap(P.SETTING_SPECS["m_slider"], 0) == C.M_MIN
    assert P.snap(P.SETTING_SPECS["k_slider"], 33) == 30 and P.snap(P.SETTING_SPECS["k_slider"], 36) == 40
    assert P.snap(P.SETTING_SPECS["su_slider"], 0.31) == 0.30 and P.snap(P.SETTING_SPECS["su_slider"], 9.0) == 0.45
    assert P.snap(P.SETTING_SPECS["art_select"], "nein") is None and P.snap(P.SETTING_SPECS["art_select"], "cont") == "cont"
    assert P.snap(P.SETTING_SPECS["seed_input"], -5) == 0
    assert P.snap(P.SETTING_SPECS["reps_select"], 250) in (200, 500)
