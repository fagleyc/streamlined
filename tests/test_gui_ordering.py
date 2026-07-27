"""GUI ordering / filter-toolbar regression tests.

Covers Casey's round:
  (1) The Reynolds (Re) and freestream-velocity (U_inf) FILTER combos were
      removed from the plot filter toolbar (they were confusing/incorrect).
      Alpha, Beta and Mach filters remain.
  (2) The data table and Excel export present rows ordered by tunnel speed
      (primary), beta (secondary), then alpha (tertiary) — the order the
      tunnel is actually run: every alpha for a beta at one speed, then the
      next beta, then the next speed.
  (3) Tunnel-condition columns are always present; the toolbar checkboxes
      that used to gate them (and the unsteady export option) are gone.
  (4) A hysteresis run tags its return leg (sweep_dir 'dn'), which duplicates
      every (speed, beta, alpha) triple.  Rows order speed -> beta ->
      sweep_dir -> alpha -> run_number so the two legs stay as contiguous
      blocks, outbound first.  Both per-point arrays are optional and the
      order degrades to speed -> beta -> alpha without them.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from utils.gui.models.case import TestCase as Case  # noqa: E402
from utils.gui.models.data_model import DataModel  # noqa: E402
from utils.gui.models.settings import AppSettings  # noqa: E402
from utils.gui.widgets.filter_widgets import FilterToolbar  # noqa: E402
from utils.gui.views.table_panel import (  # noqa: E402
    ExportAborted, TablePanel)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Task 1: Re / U_inf filter combos removed from the toolbar
# ---------------------------------------------------------------------------

class TestFilterToolbarNoReVelocity:
    def test_reynolds_and_velocity_combos_removed(self, qapp):
        tb = FilterToolbar()
        # The combos themselves are gone
        assert not hasattr(tb, "cmb_reynolds")
        assert not hasattr(tb, "cmb_velocity")
        # ...and so is their wiring
        for name in ("set_reynolds_values", "set_velocity_values",
                     "get_selected_reynolds", "get_selected_velocity"):
            assert not hasattr(tb, name), f"{name} should be removed"

    def test_alpha_beta_mach_filters_remain(self, qapp):
        tb = FilterToolbar()
        assert hasattr(tb, "alpha_filter")
        assert hasattr(tb, "beta_filter")
        assert hasattr(tb, "cmb_mach")
        # Core accessors still present
        assert hasattr(tb, "set_mach_values")
        assert hasattr(tb, "get_selected_mach")
        # reset_filters must not reference the removed combos
        tb.reset_filters()


# ---------------------------------------------------------------------------
# Task 2: table / Excel rows ordered speed -> beta -> alpha
# ---------------------------------------------------------------------------

def _unsorted_case():
    """A deliberately-unsorted case spanning multiple alpha/beta/mach."""
    c = Case(id="ord", name="OrderCase")
    # (alpha, beta, mach) triples in scrambled order:
    triples = [
        (2.0, 0.0, 0.30),
        (0.0, 5.0, 0.30),
        (0.0, 0.0, 0.50),
        (0.0, 0.0, 0.30),
        (2.0, 0.0, 0.50),
        (0.0, 5.0, 0.10),
    ]
    a = np.array([t[0] for t in triples])
    b = np.array([t[1] for t in triples])
    m = np.array([t[2] for t in triples])
    n = len(triples)
    c.alphas = a
    c.betas = b
    c.machs = m
    c.reynolds = np.full(n, 1.0e6)
    c.velocities = np.linspace(10.0, 20.0, n)
    c.Cl = np.arange(n, dtype=float)
    c.Cd = np.full(n, 0.1)
    c.Cs = np.zeros(n)
    c.CRoll = np.zeros(n)
    c.CPitch = np.zeros(n)
    c.CYaw = np.zeros(n)
    return c


def _fan_sweep_case():
    """A low-speed fan-Hz sweep: Mach is degenerate, velocity is not."""
    c = Case(id="fan", name="FanSweep")
    # (alpha, beta, velocity) triples in scrambled order:
    triples = [
        (4.0, 0.0, 20.0),
        (0.0, 0.0, 10.0),
        (0.0, 5.0, 20.0),
        (4.0, 0.0, 10.0),
        (0.0, 0.0, 20.0),
        (0.0, 5.0, 10.0),
    ]
    n = len(triples)
    c.alphas = np.array([t[0] for t in triples])
    c.betas = np.array([t[1] for t in triples])
    c.velocities = np.array([t[2] for t in triples])
    c.machs = np.zeros(n)           # low-speed tunnel reports Mach = 0
    c.Cl = np.arange(n, dtype=float)
    c.Cd = np.full(n, 0.1)
    c.Cs = np.zeros(n)
    c.CRoll = np.zeros(n)
    c.CPitch = np.zeros(n)
    c.CYaw = np.zeros(n)
    return c


class TestTableRowOrdering:
    def test_rows_sorted_mach_beta_alpha(self, qapp):
        panel = TablePanel(DataModel(), AppSettings())
        case = _unsorted_case()
        rows = panel._get_case_rows(case)

        # Locate column indices for Alpha, Beta, Mach
        cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}
        ia, ib, im = cols["Alpha"], cols["Beta"], cols["Mach"]

        got = [(round(r[ia], 1), round(r[ib], 1), round(r[im], 3))
               for r in rows]
        expected = [
            (0.0, 5.0, 0.10),
            (0.0, 0.0, 0.30),
            (2.0, 0.0, 0.30),
            (0.0, 5.0, 0.30),
            (0.0, 0.0, 0.50),
            (2.0, 0.0, 0.50),
        ]
        assert got == expected

    def test_speed_is_primary_key(self, qapp):
        """Mach must be non-decreasing across the whole table."""
        panel = TablePanel(DataModel(), AppSettings())
        rows = panel._get_case_rows(_unsorted_case())
        cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}
        machs = [r[cols["Mach"]] for r in rows]
        assert machs == sorted(machs)

    def test_alpha_varies_fastest_within_a_speed_beta_block(self, qapp):
        """Alpha ascends inside each (speed, beta) group."""
        panel = TablePanel(DataModel(), AppSettings())
        rows = panel._get_case_rows(_unsorted_case())
        cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}
        blocks = {}
        for r in rows:
            key = (round(r[cols["Mach"]], 3), round(r[cols["Beta"]], 1))
            blocks.setdefault(key, []).append(r[cols["Alpha"]])
        for key, alphas in blocks.items():
            assert alphas == sorted(alphas), f"alpha unsorted in {key}"

    def test_velocity_fallback_when_mach_degenerate(self, qapp):
        """A fan-Hz sweep with Mach = 0 still groups by velocity."""
        panel = TablePanel(DataModel(), AppSettings())
        rows = panel._get_case_rows(_fan_sweep_case())
        cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}
        ia, ib, iu = cols["Alpha"], cols["Beta"], cols["U_inf"]

        # U_inf is shown in the output unit system, so compare its
        # grouping rather than the raw m/s values.
        speeds = [r[iu] for r in rows]
        assert speeds == sorted(speeds)
        assert len(set(round(s, 1) for s in speeds)) == 2

        got = [(round(r[ia], 1), round(r[ib], 1)) for r in rows]
        expected = [
            (0.0, 0.0), (4.0, 0.0), (0.0, 5.0),   # slow step
            (0.0, 0.0), (4.0, 0.0), (0.0, 5.0),   # fast step
        ]
        assert got == expected

    def test_degrades_to_beta_alpha_without_speed(self, qapp):
        """No usable speed key -> order falls back to (beta, alpha)."""
        c = _unsorted_case()
        c.machs = np.array([])
        c.velocities = np.array([])
        panel = TablePanel(DataModel(), AppSettings())
        rows = panel._get_case_rows(c)
        cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}
        got = [(round(r[cols["Beta"]], 1), round(r[cols["Alpha"]], 1))
               for r in rows]
        assert got == sorted(got)


# ---------------------------------------------------------------------------
# Task 4: hysteresis legs and acquisition order
# ---------------------------------------------------------------------------

def _hysteresis_case(up_tag="up"):
    """An alpha sweep flown up and back down at one speed and beta.

    Both legs share the same (speed, beta, alpha) triples, so only the
    sweep_dir tag can keep them apart.  Cl carries a marker (1xx on the
    outbound leg, 2xx on the return leg) so a test can tell which point
    landed where.
    """
    c = Case(id="hyst", name="Hysteresis")
    # (alpha, sweep_dir) pairs in scrambled order:
    pairs = [
        (4.0, "dn"),
        (0.0, up_tag),
        (4.0, up_tag),
        (2.0, "dn"),
        (2.0, up_tag),
        (0.0, "dn"),
    ]
    n = len(pairs)
    c.alphas = np.array([p[0] for p in pairs])
    c.betas = np.zeros(n)
    c.machs = np.full(n, 0.30)
    c.velocities = np.full(n, 100.0)
    c.sweep_dirs = np.array([p[1] for p in pairs])
    c.run_numbers = np.arange(1.0, n + 1.0)
    c.Cl = np.array([(200.0 if p[1] == "dn" else 100.0) + p[0]
                     for p in pairs])
    c.Cd = np.full(n, 0.1)
    c.Cs = np.zeros(n)
    c.CRoll = np.zeros(n)
    c.CPitch = np.zeros(n)
    c.CYaw = np.zeros(n)
    return c


def _degenerate_case():
    """Repeat points at one (speed, beta, alpha) — only the run differs."""
    c = Case(id="rep", name="Repeats")
    n = 3
    c.alphas = np.full(n, 2.0)
    c.betas = np.zeros(n)
    c.machs = np.full(n, 0.30)
    c.velocities = np.full(n, 100.0)
    c.run_numbers = np.array([7.0, 5.0, 6.0])
    c.Cl = np.array([70.0, 50.0, 60.0])   # marker: Cl == 10 * run number
    c.Cd = np.full(n, 0.1)
    c.Cs = np.zeros(n)
    c.CRoll = np.zeros(n)
    c.CPitch = np.zeros(n)
    c.CYaw = np.zeros(n)
    return c


def _today_order(panel, case):
    """The (beta, alpha, Cl) rows a case yields, for before/after compares."""
    cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}
    return [(round(r[cols["Beta"]], 1), round(r[cols["Alpha"]], 1),
             r[cols["Cl"]]) for r in panel._get_case_rows(case)]


class TestHysteresisRowOrdering:
    @pytest.mark.parametrize("up_tag", ["up", ""])
    def test_legs_come_out_as_contiguous_blocks(self, qapp, up_tag):
        """Outbound block first, return block second, alpha ascending."""
        panel = TablePanel(DataModel(), AppSettings())
        rows = panel._get_case_rows(_hysteresis_case(up_tag))
        cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}

        got = [(round(r[cols["Alpha"]], 1), r[cols["Cl"]]) for r in rows]
        assert got == [
            (0.0, 100.0), (2.0, 102.0), (4.0, 104.0),   # outbound leg
            (0.0, 200.0), (2.0, 202.0), (4.0, 204.0),   # return leg
        ]

    def test_legs_are_not_interleaved(self, qapp):
        """The 'dn' marker must not appear before the last outbound row."""
        panel = TablePanel(DataModel(), AppSettings())
        rows = panel._get_case_rows(_hysteresis_case())
        cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}
        legs = [r[cols["Cl"]] > 150.0 for r in rows]
        assert legs == sorted(legs)

    def test_run_number_breaks_a_degenerate_tie(self, qapp):
        """Identical keys fall back to acquisition order, not input order."""
        panel = TablePanel(DataModel(), AppSettings())
        rows = panel._get_case_rows(_degenerate_case())
        cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}
        assert [r[cols["Cl"]] for r in rows] == [50.0, 60.0, 70.0]


class TestOrderingKeysAreOptional:
    """Agent-supplied per-point arrays may be absent or misaligned."""

    def _baseline(self, qapp):
        panel = TablePanel(DataModel(), AppSettings())
        return panel, _today_order(panel, _unsorted_case())

    def test_case_without_either_array_is_unchanged(self, qapp):
        panel, baseline = self._baseline(qapp)
        c = _unsorted_case()
        assert not hasattr(c, "sweep_dirs") or c.sweep_dirs.size == 0
        assert _today_order(panel, c) == baseline

    def test_empty_run_numbers_falls_back_to_point_order(self, qapp):
        panel, baseline = self._baseline(qapp)
        c = _unsorted_case()
        c.run_numbers = np.array([])
        c.sweep_dirs = np.array([])
        assert _today_order(panel, c) == baseline

    def test_mismatched_sweep_dirs_does_not_raise(self, qapp):
        panel, baseline = self._baseline(qapp)
        c = _unsorted_case()
        c.sweep_dirs = np.array(["up", "dn"])       # 2 tags for 6 points
        assert _today_order(panel, c) == baseline

    def test_mismatched_run_numbers_does_not_raise(self, qapp):
        panel, baseline = self._baseline(qapp)
        c = _unsorted_case()
        c.run_numbers = np.array([1.0, 2.0])        # 2 runs for 6 points
        assert _today_order(panel, c) == baseline

    def test_non_numeric_run_numbers_does_not_raise(self, qapp):
        panel, baseline = self._baseline(qapp)
        c = _unsorted_case()
        c.run_numbers = np.array(["a", "b", "c", "d", "e", "f"])
        assert _today_order(panel, c) == baseline

    def test_all_blank_sweep_dirs_changes_nothing(self, qapp):
        panel, baseline = self._baseline(qapp)
        c = _unsorted_case()
        c.sweep_dirs = np.array([""] * 6)
        assert _today_order(panel, c) == baseline

    def test_key_helpers_tolerate_missing_attributes(self, qapp):
        """The helpers are the defensive boundary — exercise them directly."""
        c = Case(id="bare", name="Bare")
        assert list(TablePanel._sweep_dir_rank(c, 3)) == [0.0, 0.0, 0.0]
        assert list(TablePanel._run_number_key(c, 3)) == [0.0, 1.0, 2.0]
        c.sweep_dirs = np.array(["", "up", "dn"])
        c.run_numbers = np.array([9.0, 8.0, 7.0])
        assert list(TablePanel._sweep_dir_rank(c, 3)) == [0.0, 0.0, 1.0]
        assert list(TablePanel._run_number_key(c, 3)) == [9.0, 8.0, 7.0]


# ---------------------------------------------------------------------------
# Task 3: tunnel columns always on, gating checkboxes removed
# ---------------------------------------------------------------------------

class TestTablePanelCleanup:
    def test_tunnel_columns_always_present(self, qapp):
        panel = TablePanel(DataModel(), AppSettings())
        keys = [k for k, _lbl in panel._columns]
        for key in ("Q", "Mach", "Re", "U_inf", "rho", "T", "P_tot"):
            assert key in keys, f"{key} column should always be present"

    def test_toolbar_checkboxes_removed(self, qapp):
        panel = TablePanel(DataModel(), AppSettings())
        for name in ("chk_tunnel_conditions", "chk_include_unsteady"):
            assert not hasattr(panel, name), f"{name} should be removed"

    def test_conditions_sidebar_removed(self, qapp):
        panel = TablePanel(DataModel(), AppSettings())
        assert not hasattr(panel, "conditions_panel")
        import utils.gui.views.table_panel as tp
        assert not hasattr(tp, "TunnelConditionsPanel")

    def test_unsteady_flag_comes_from_export_config(self, qapp):
        """The Export dialog config, not a toolbar checkbox, drives it."""
        assert TablePanel._config_wants_unsteady(None) is False
        assert TablePanel._config_wants_unsteady({}) is False
        assert TablePanel._config_wants_unsteady(
            {'include_coefficients_ts': False,
             'include_tunnel_ts': False}) is False
        assert TablePanel._config_wants_unsteady(
            {'include_coefficients_ts': True}) is True
        assert TablePanel._config_wants_unsteady(
            {'include_tunnel_ts': True}) is True

    def test_do_export_threads_config_through(self, qapp):
        """do_export must not drop the dialog config."""
        panel = TablePanel(DataModel(), AppSettings())
        seen = {}

        def _capture(filepath, format, config=None):
            seen['filepath'] = filepath
            seen['format'] = format
            seen['config'] = config

        panel._do_export = _capture
        cfg = {'filepath': 'out.h5', 'format': 'hdf5',
               'include_tunnel_ts': True}
        panel.do_export(cfg)
        assert seen['filepath'] == 'out.h5'
        assert seen['format'] == 'hdf5'
        assert seen['config'] is cfg


# ---------------------------------------------------------------------------
# The writers are separate from the dialogs, so exports run headless
# ---------------------------------------------------------------------------

class TestHeadlessExport:
    """_write_export does the file I/O; _do_export shows the dialogs.

    A modal QMessageBox blocks forever with no event loop, which made the
    export path impossible to drive from a script or a test.
    """

    def _panel_with_case(self):
        model = DataModel()
        model.add_case(_unsorted_case())
        return TablePanel(model, AppSettings())

    def test_excel_writer_opens_no_dialog(self, qapp, tmp_path,
                                          monkeypatch):
        pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        import utils.gui.views.table_panel as tp

        def _boom(*args, **kwargs):
            raise AssertionError("export writers must not open dialogs")

        for name in ("information", "warning", "critical", "question"):
            monkeypatch.setattr(tp.QMessageBox, name, staticmethod(_boom))

        panel = self._panel_with_case()
        out = tmp_path / "export.xlsx"
        panel._write_export(str(out), 'excel')
        assert out.exists()

    def test_excel_rows_keep_table_order(self, qapp, tmp_path):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")

        panel = self._panel_with_case()
        case = next(iter(panel.model.cases))
        out = tmp_path / "export.xlsx"
        panel._write_export(str(out), 'excel')

        cols = {k: i for i, (k, _lbl) in enumerate(panel._columns)}
        alpha_label = panel._columns[cols["Alpha"]][1]

        # The data table starts below the per-case summary block
        raw = pd.read_excel(out, header=None)
        header = next(i for i in range(len(raw))
                      if str(raw.iloc[i, 0]) == alpha_label)
        df = pd.read_excel(out, header=header)

        rows = panel._get_case_rows(case)
        assert len(df) == len(rows)
        for i, row in enumerate(rows):
            for key in ("Alpha", "Beta", "Mach"):
                label = panel._columns[cols[key]][1]
                assert df[label].iloc[i] == pytest.approx(row[cols[key]])

    def test_writer_raises_instead_of_warning_when_empty(self, qapp,
                                                         tmp_path):
        pytest.importorskip("pandas")
        panel = TablePanel(DataModel(), AppSettings())
        with pytest.raises(ExportAborted):
            panel._write_export(str(tmp_path / "empty.xlsx"), 'excel')
