"""Mount-dependent reduction of the external (ATE) balance.

The ATE yaws with the model but never pitches with it, so which of its
six channels carries which model load depends on how the model is
mounted:

* full span — alpha comes from the incidence strut above the balance,
  the balance stays level, and its channels ARE the wind-axis loads.
* half span — the semispan model stands on the turntable and alpha IS
  the yaw drive, so the horizontal channels are body-fixed (they need
  the alpha rotation) and the vertical channel reads the model's SIDE
  force, not lift.

Reducing a ½-span run with the full-span algorithm is what made drag
come out backwards: the missing ``+Side*sin(a)`` term is the whole
induced-drag contribution, and 'lift' was really the span-direction
channel. These tests pin the two algorithms, the marker that selects
them, and the unit systems the OGI can be set to.

Reference: deprecated/scripts/calc_uncertainty_Extbalance.m ('Vertical'
case) for the ½-span coefficient definitions.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windtunnel.external_balance import (      # noqa: E402
    KGF_TO_N, NM_TO_INLB, N_TO_LBF, SPAN_FULL, SPAN_HALF,
    build_load_matrix_from_channels, calc_uncertainty_ext_balance,
    external_loads_to_ips, normalize_span_config, resolve_external_wrf,
    transfer_external_loads_to_mrc)
from utils.windtunnel.data_io import (                # noqa: E402
    MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION, run_span_config)
from utils.windtunnel.reduction import reduce_single_point  # noqa: E402
from utils.windtunnel.transforms import Geometry      # noqa: E402

scipy_io = pytest.importorskip("scipy.io")

CHANNELS = ("Lift", "Drag", "Side", "Roll", "Pitch", "Yaw")


def _channels(n=8, **overrides):
    """Six distinguishable load channels, one constant per channel."""
    base = {"Lift": 0.5, "Drag": 2.0, "Side": 20.0,
            "Roll": 1.0, "Pitch": 3.0, "Yaw": 7.0}
    base.update(overrides)
    return {k: np.full(n, v, dtype=float) for k, v in base.items()}


# ── the marker ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("value,expected", [
    ("half", SPAN_HALF), ("Half", SPAN_HALF), ("HALF", SPAN_HALF),
    ("semispan", SPAN_HALF), ("Vertical", SPAN_HALF),
    (b"half", SPAN_HALF),
    ("full", SPAN_FULL), ("Horizontal", SPAN_FULL),
    ("", SPAN_FULL), (None, SPAN_FULL), (3, SPAN_FULL),
])
def test_normalize_span_config(value, expected):
    assert normalize_span_config(value) == expected


def test_unknown_marker_defaults_to_full_span():
    """A missing marker must not silently change historical behaviour."""
    raw = _channels()
    assert normalize_span_config(raw.get("span_config")) == SPAN_FULL


# ── the two algorithms ──────────────────────────────────────────────────
def test_full_span_passes_channels_straight_through():
    raw = _channels()
    w = resolve_external_wrf(raw, alpha_deg=12.0, span_config="full")
    for ch in CHANNELS:
        assert np.allclose(getattr(w, ch), raw[ch]), ch


def test_half_span_matches_the_matlab_vertical_definitions():
    """Cl/Cd/Cs numerators from calc_uncertainty_Extbalance.m."""
    alpha = np.array([0.0, 5.0, 10.0, -7.5, 15.0])
    raw = {"Drag": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
           "Side": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
           "Lift": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
           "Roll": np.full(5, 1.0), "Pitch": np.full(5, 2.0),
           "Yaw": np.full(5, 3.0)}
    w = resolve_external_wrf(raw, alpha_deg=alpha, span_config="half")
    a = np.deg2rad(alpha)

    # Cl = [Fy*cos(a) - Fx*sin(a)]/QS, Fy = Side channel, Fx = Drag
    assert np.allclose(w.Lift, raw["Side"] * np.cos(a)
                       - raw["Drag"] * np.sin(a))
    # Cd = [Fx*cos(a) + Fy*sin(a)]/QS
    assert np.allclose(w.Drag, raw["Drag"] * np.cos(a)
                       + raw["Side"] * np.sin(a))
    # Cs = Fz/QS, Fz = the balance's vertical (Lift) channel
    assert np.allclose(w.Side, raw["Lift"])
    # moments permute with the axes: model pitch is about the vertical
    assert np.allclose(w.Roll, raw["Roll"])
    assert np.allclose(w.Pitch, raw["Yaw"])
    assert np.allclose(w.Yaw, raw["Pitch"])


def test_half_span_at_zero_alpha_is_pure_relabelling():
    raw = _channels()
    w = resolve_external_wrf(raw, alpha_deg=0.0, span_config="half")
    assert np.allclose(w.Lift, raw["Side"])
    assert np.allclose(w.Drag, raw["Drag"])
    assert np.allclose(w.Side, raw["Lift"])


def test_half_span_recovers_the_drag_the_pass_through_loses():
    """The reported symptom: at positive alpha a lifting model's drag
    is dominated by Side*sin(a), and dropping it can flip the sign."""
    raw = _channels(Drag=-0.4, Side=25.0)      # small axial, big normal
    w = resolve_external_wrf(raw, alpha_deg=10.0, span_config="half")
    passthrough = resolve_external_wrf(raw, alpha_deg=10.0,
                                       span_config="full")
    assert np.all(passthrough.Drag < 0)        # what the bug produced
    assert np.all(w.Drag > 0)                  # what the mount really means


def test_half_span_rotation_conserves_the_horizontal_force():
    """The alpha rotation is a rotation: the in-plane magnitude holds."""
    raw = _channels()
    for alpha in (0.0, 7.0, -13.0, 30.0):
        w = resolve_external_wrf(raw, alpha_deg=alpha, span_config="half")
        assert np.allclose(w.Lift ** 2 + w.Drag ** 2,
                           raw["Side"] ** 2 + raw["Drag"] ** 2)


def test_alpha_falls_back_to_the_channel_then_to_zero():
    raw = _channels()
    raw["Alpha"] = np.full(8, 10.0)
    from_channel = resolve_external_wrf(raw, span_config="half")
    explicit = resolve_external_wrf(raw, alpha_deg=10.0, span_config="half")
    assert np.allclose(from_channel.Drag, explicit.Drag)

    bare = _channels()
    zero = resolve_external_wrf(bare, span_config="half")
    assert np.allclose(zero.Drag, bare["Drag"])


def test_missing_channels_zero_fill_on_both_mounts():
    partial = {"Lift": np.ones(4), "Drag": np.ones(4)}
    for span in ("full", "half"):
        w = resolve_external_wrf(partial, alpha_deg=0.0, span_config=span)
        assert len(w.Yaw) == 4 and np.allclose(w.Yaw, 0.0)


# ── units ───────────────────────────────────────────────────────────────
def test_every_ogi_unit_setting_converts_to_lb_and_inlb():
    raw = _channels()
    cases = {
        "N": (N_TO_LBF, NM_TO_INLB),
        "kg": (KGF_TO_N * N_TO_LBF, KGF_TO_N * NM_TO_INLB),
        "lbft": (1.0, 12.0),        # the OGI pairs Lb with Lb*FT
        "lb": (1.0, 1.0),           # already the chain's native system
    }
    for marker, (f_scale, m_scale) in cases.items():
        d = dict(raw, load_units=marker)
        out = external_loads_to_ips(d)
        assert np.allclose(out["Drag"], raw["Drag"] * f_scale), marker
        assert np.allclose(out["Pitch"], raw["Pitch"] * m_scale), marker


def test_unmarked_and_native_dicts_are_not_copied():
    raw = _channels()
    assert external_loads_to_ips(raw) is raw
    native = dict(raw, load_units="lb")
    assert external_loads_to_ips(native) is native


def test_conversion_does_not_mutate_the_input():
    raw = _channels()
    before = raw["Drag"].copy()
    external_loads_to_ips(dict(raw, load_units="N"))
    assert np.allclose(raw["Drag"], before)


# ── MRC transfer ────────────────────────────────────────────────────────
def test_zero_mrc_is_a_no_op_on_both_mounts():
    raw = _channels()
    for span in ("full", "half"):
        w = resolve_external_wrf(raw, alpha_deg=8.0, span_config=span)
        same = transfer_external_loads_to_mrc(w, 8.0, 0.0, [0, 0, 0],
                                              span_config=span)
        assert same is w


def test_half_span_mrc_shift_moves_pitch_by_the_normal_force_arm():
    """At alpha = 0 the normal force is the Side channel, so a pure x
    shift moves the pitching moment by -Fz*mx with Fz = Side."""
    raw = _channels()
    w = resolve_external_wrf(raw, alpha_deg=0.0, span_config="half")
    mx = 1.5
    shifted = transfer_external_loads_to_mrc(w, 0.0, 0.0, [mx, 0, 0],
                                             span_config="half")
    assert np.allclose(shifted.Pitch, w.Pitch - raw["Side"] * mx)
    assert np.allclose(shifted.Lift, w.Lift)     # forces untouched


# ── uncertainty ─────────────────────────────────────────────────────────
def test_uncertainty_runs_on_both_mounts():
    raw = _channels(n=6)
    fb = build_load_matrix_from_channels(raw)
    assert fb.shape == (6, 6)
    assert np.allclose(fb[:, 0], raw["Drag"])    # EXTERNAL_CHANNEL_ORDER
    assert np.allclose(fb[:, 2], raw["Lift"])

    coeffs = {c: np.ones(6) for c in
              ("Cl", "Cd", "Cs", "CRoll", "CPitch", "CYaw")}
    for span in ("full", "half"):
        unc = calc_uncertainty_ext_balance(
            coeffs, fb, np.full(6, 5.0), np.full(6, 0.5),
            S=18.75, C=2.86, span_config=span)
        assert set(unc["total"]) == set(coeffs), span
        for name, v in unc["total"].items():
            assert np.all(np.isfinite(v)), (span, name)


def test_full_span_uncertainty_has_no_attitude_term():
    """Only the Vertical mount's alpha rotation makes Cl sensitive to
    attitude; the level mount has no pCpa at all."""
    raw = _channels(n=4)
    fb = build_load_matrix_from_channels(raw)
    coeffs = {"Cl": np.ones(4)}
    full = calc_uncertainty_ext_balance(coeffs, fb, np.full(4, 5.0),
                                        np.full(4, 0.5), S=18.75, C=2.86,
                                        span_config="full")
    half = calc_uncertainty_ext_balance(coeffs, fb, np.full(4, 5.0),
                                        np.full(4, 0.5), S=18.75, C=2.86,
                                        span_config="half")
    assert "pCpa" not in full["InfCoeffs"]["Cl"]
    assert "pCpa" in half["InfCoeffs"]["Cl"]


def test_unknown_mount_is_rejected_not_guessed():
    fb = build_load_matrix_from_channels(_channels(n=2))
    with pytest.raises(ValueError, match="unknown mounting config"):
        calc_uncertainty_ext_balance({"Cl": np.ones(2)}, fb,
                                     np.zeros(2), np.full(2, 0.5),
                                     S=1.0, C=1.0, config="Diagonal")


# ── end to end through reduce_single_point ──────────────────────────────
def _point(span, alpha=10.0, n=16):
    on = _channels(n)
    on.update({"Alpha": np.full(n, alpha), "Beta": np.zeros(n),
               "Pdiff": np.full(n, 0.8), "Ptot": np.full(n, 12.2),
               "Temp": np.full(n, 295.0),
               "balance_type": "external", "span_config": span})
    off = {k: (np.zeros(n) if k in CHANNELS else v)
           for k, v in on.items()}
    return on, off


def test_reduce_single_point_honours_the_span_marker():
    geo = Geometry(C=2.86, S=18.75, b=6.0, mshift=np.zeros(3))
    results = {}
    for span in ("full", "half"):
        on, off = _point(span)
        results[span] = reduce_single_point(
            on, off, cal=None, geo=geo, pressure_cal={}, facility='SWT')

    full, half = results["full"], results["half"]
    # full span reports the Lift channel as lift; half span reports the
    # rotated Side channel — the two must NOT agree
    assert not np.allclose(np.mean(full.wrf_aero.Lift),
                           np.mean(half.wrf_aero.Lift))
    a = np.deg2rad(10.0)
    assert np.allclose(np.mean(half.wrf_aero.Lift),
                       20.0 * np.cos(a) - 2.0 * np.sin(a))
    assert np.allclose(np.mean(full.wrf_aero.Lift), 0.5)
    # and the coefficients follow
    assert float(np.mean(half.coeffs.Cl)) > float(np.mean(full.coeffs.Cl))


def test_reduce_single_point_defaults_to_full_span_without_a_marker():
    geo = Geometry(C=2.86, S=18.75, b=6.0, mshift=np.zeros(3))
    on, off = _point("full")
    del on["span_config"]
    unmarked = reduce_single_point(on, off, cal=None, geo=geo,
                                   pressure_cal={}, facility='SWT')
    on2, off2 = _point("full")
    marked = reduce_single_point(on2, off2, cal=None, geo=geo,
                                 pressure_cal={}, facility='SWT')
    assert np.allclose(unmarked.wrf_aero.Lift, marked.wrf_aero.Lift)


def test_tare_resolves_with_its_own_alpha():
    """A ½-span tare taken at a different attitude has to be rotated by
    ITS alpha before subtraction, exactly like the internal path."""
    geo = Geometry(C=2.86, S=18.75, b=6.0, mshift=np.zeros(3))
    n = 16
    on, _ = _point("half", alpha=12.0, n=n)
    off = _channels(n)
    off.update({"Alpha": np.zeros(n), "Beta": np.zeros(n),
                "balance_type": "external", "span_config": "half"})
    red = reduce_single_point(on, off, cal=None, geo=geo,
                              pressure_cal={}, facility='SWT')
    a = np.deg2rad(12.0)
    expected = ((20.0 * np.cos(a) - 2.0 * np.sin(a))   # air-on at 12 deg
                - 20.0)                                # tare at 0 deg
    assert np.allclose(np.mean(red.wrf_aero.Lift), expected)


# ── the directory probe ─────────────────────────────────────────────────
def _write_run(directory: Path, name: str, span: str, n: int = 16) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    group = {c: np.ones(n) for c in CHANNELS}
    group["Pdiff"] = np.full(n, 0.8)
    scipy_io.savemat(str(directory / name), {
        "ATE_Balance": group,
        "Time": {"Time": np.arange(n) / 50.0},
        "meta": {"run": {"balance_type": "external", "span_config": span,
                         "air_state": "AirOn", "alpha": 0.0, "beta": 0.0},
                 "config_json": json.dumps({})},
    }, long_field_names=True)
    (directory / MANIFEST_FILENAME).write_text(json.dumps(
        {"schema_version": MANIFEST_SCHEMA_VERSION,
         "config_name": directory.name, "output_format": "mat",
         "points": []}), encoding="utf-8")
    return directory / name


def test_run_span_config_reads_the_recorded_marker(tmp_path):
    for span in ("full", "half"):
        d = tmp_path / span
        _write_run(d, "run_0001_alpha_0.0.mat", span)
        assert run_span_config(str(d)) == span


def test_run_span_config_is_quiet_when_nothing_says(tmp_path):
    d = tmp_path / "silent"
    _write_run(d, "run_0001_alpha_0.0.mat", "")
    assert run_span_config(str(d)) == ""
    assert run_span_config(str(tmp_path / "missing")) == ""


# ── slow-channel resampling ─────────────────────────────────────────────
def test_slow_channels_are_clamped_not_extrapolated(tmp_path):
    """A slow group whose block ends before the fastest one's must not be
    cubic-extrapolated into the tail.

    Found 2026-08-21: the ATE streams ~50 Hz against a DaqBook running
    several times faster, so every external run left a tail to fill.
    Cubic extrapolation diverges cubically — a Lift channel that never
    left 204..207 N came back reaching -1463 N, and the point mean fell
    from 205 N to 59 N. Nothing about the reduction can survive that.
    """
    from utils.windtunnel.data_io import _resample_channels_to_fastest
    from utils.windtunnel.data_io import RawData

    fast_t = np.linspace(0.0, 1.0, 200)
    slow_t = np.linspace(0.0, 0.18, 36)          # ends early, like the ATE
    slow_v = 205.0 + 1.5 * np.sin(2 * np.pi * 3.0 * slow_t)

    raw = RawData()
    _resample_channels_to_fastest({
        'Pdiff': {'data': np.full(200, 0.8), 'time': fast_t,
                  'group': 'DaqBook'},
        'Lift': {'data': slow_v, 'time': slow_t, 'group': 'ATE_Balance'},
    }, raw)

    out = np.asarray(raw.data['Lift'], dtype=float)
    assert len(out) == 200
    # never leaves the measured range
    assert out.min() >= slow_v.min() - 1e-6
    assert out.max() <= slow_v.max() + 1e-6
    # the tail holds the last real reading
    assert np.allclose(out[fast_t > slow_t[-1]], slow_v[-1])
    # and inside the overlap the mean is preserved (outside it the held
    # value legitimately shifts the whole-window mean — the point is
    # that it stays a number the instrument actually read)
    inside = fast_t <= slow_t[-1]
    assert abs(np.mean(out[inside]) - np.mean(slow_v)) < 0.05 * np.ptp(slow_v)

    # what the old extrapolating interpolant did, for contrast
    from scipy.interpolate import interp1d
    old = interp1d(slow_t, slow_v, kind='cubic', bounds_error=False,
                   fill_value='extrapolate')(fast_t)
    assert np.ptp(old) > 100 * np.ptp(slow_v)   # diverges, badly


def test_resampling_still_interpolates_inside_the_overlap(tmp_path):
    """Clamping the ends must not flatten the interior."""
    from utils.windtunnel.data_io import RawData, \
        _resample_channels_to_fastest

    fast_t = np.linspace(0.0, 1.0, 101)
    slow_t = np.linspace(0.0, 1.0, 11)
    raw = RawData()
    _resample_channels_to_fastest({
        'Fast': {'data': np.zeros(101), 'time': fast_t, 'group': 'g'},
        'Ramp': {'data': 2.0 * slow_t, 'time': slow_t, 'group': 'h'},
    }, raw)
    assert np.allclose(raw.data['Ramp'], 2.0 * fast_t, atol=1e-8)
