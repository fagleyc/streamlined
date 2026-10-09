"""Hysteresis runs plot in run sequence, one trace per leg, with direction.

Freestream tags a return sweep's points sweep_dir 'up' / 'dn' (the sign of
alpha-dot).  The reduction orders points by alpha, so a 0:2:10R-style run
(0..10 up, then 8, 6 down) reaches the plot interleaved by angle.  Sorting
each trace by alpha alone drew it as 0, 2, 4, 6, 6, 8, 8, 10 - a zigzag
with no direction.  The plot now splits the two legs, orders each by the
acquisition run number, and marks them up / down.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from utils.gui.models.case import TestCase as Case  # noqa: E402
from utils.gui.models.data_model import DataModel  # noqa: E402
from utils.gui.views.plot_panel import PlotPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# Reduced point order (alpha-sorted, stable), as the reduction hands it
# over for runs 1..6 = alpha 0..10 'up', run 7 = 8 'dn', run 8 = 6 'dn'.
_ALPHA = [0.0, 2.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0]
_RUN = [1.0, 2.0, 3.0, 4.0, 8.0, 5.0, 7.0, 6.0]
_DIR = ['up', 'up', 'up', 'up', 'dn', 'up', 'dn', 'up']


def _case(dirs=_DIR, runs=_RUN, betas=None) -> Case:
    c = Case(id="h1", name="Hyst")
    c.alphas = np.array(_ALPHA)
    c.betas = np.zeros(len(_ALPHA)) if betas is None else np.array(betas)
    c.Cl = np.array(_RUN) * 0.1            # Cl tags the run: run k -> 0.k
    c.Cd = np.full(len(_ALPHA), 0.05)
    c.sweep_dirs = np.array(dirs)
    c.run_numbers = np.array(runs)
    return c


class _Harness:
    def __init__(self, case):
        self.model = DataModel()
        self.model.cases.add(case)
        self.panel = PlotPanel(self.model)
        self.traces = []
        self.panel.plot_canvas.plot = self._capture

    def _capture(self, x, y, **kw):
        self.traces.append({'label': kw.get('label'),
                            'marker': kw.get('marker'),
                            'linestyle': kw.get('linestyle'),
                            'x': np.asarray(x, float),
                            'y': np.asarray(y, float)})

    def draw(self, x_var="Alpha"):
        self.panel.plot_controls.plot_selector.set_x_var(x_var)
        self.traces.clear()
        self.panel._update_plot()
        return self.traces


def test_two_legs_in_run_sequence(qapp):
    up, dn = _Harness(_case()).draw()
    np.testing.assert_allclose(up['x'], [0, 2, 4, 6, 8, 10])
    np.testing.assert_allclose(up['y'], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(dn['x'], [8, 6])
    np.testing.assert_allclose(dn['y'], [0.7, 0.8])


def test_legs_show_direction(qapp):
    up, dn = _Harness(_case()).draw()
    assert up['marker'] == '^' and dn['marker'] == 'v'
    assert up['label'].endswith('↑') and dn['label'].endswith('↓')
    assert up['linestyle'] != dn['linestyle']


def test_blank_outbound_tag_counts_as_up(qapp):
    dirs = ['' if d == 'up' else d for d in _DIR]
    up, dn = _Harness(_case(dirs=dirs)).draw()
    np.testing.assert_allclose(up['x'], [0, 2, 4, 6, 8, 10])
    np.testing.assert_allclose(dn['x'], [8, 6])


def test_without_run_numbers_the_tag_gives_the_direction(qapp):
    up, dn = _Harness(_case(runs=[])).draw()
    np.testing.assert_allclose(up['x'], [0, 2, 4, 6, 8, 10])
    np.testing.assert_allclose(dn['x'], [8, 6])


def test_down_first_sweep_draws_down_leg_first(qapp):
    # 10..0 down (runs 1..6), then 6, 8 back up (runs 7, 8)
    c = _case(dirs=['dn', 'dn', 'dn', 'dn', 'up', 'dn', 'up', 'dn'],
              runs=[6.0, 5.0, 4.0, 3.0, 7.0, 2.0, 8.0, 1.0])
    c.alphas = np.array([0.0, 2.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0])
    first, second = _Harness(c).draw()
    assert first['marker'] == 'v'
    np.testing.assert_allclose(first['x'], [10, 8, 6, 4, 2, 0])
    np.testing.assert_allclose(second['x'], [6, 8])


def test_untagged_run_is_one_alpha_sorted_trace(qapp):
    c = _case(dirs=[''] * 8)
    (t,) = _Harness(c).draw()
    np.testing.assert_allclose(t['x'], sorted(_ALPHA))
    assert t['label'] == 'Hyst'


def test_misaligned_tags_fall_back(qapp):
    (t,) = _Harness(_case(dirs=['up', 'dn'])).draw()
    np.testing.assert_allclose(t['x'], sorted(_ALPHA))


def test_each_beta_splits_on_its_own(qapp):
    c = _case()
    c.alphas = np.array(_ALPHA * 2)
    c.betas = np.array([0.0] * 8 + [4.0] * 8)
    c.Cl = np.concatenate([np.array(_RUN) * 0.1] * 2)
    c.Cd = np.full(16, 0.05)
    c.sweep_dirs = np.array(_DIR * 2)
    c.run_numbers = np.array(_RUN + [r + 8 for r in _RUN])
    traces = _Harness(c).draw()
    assert len(traces) == 4
    assert [t['marker'] for t in traces] == ['^', 'v', '^', 'v']
    for t in traces[1::2]:
        np.testing.assert_allclose(t['x'], [8, 6])


def test_beta_return_sweep_on_beta_axis(qapp):
    # beta -4..4 up (runs 1..5), then 2, 0 down (runs 6, 7); alpha fixed
    c = Case(id="b1", name="BetaHyst")
    c.betas = np.array([-4.0, -2.0, 0.0, 0.0, 2.0, 2.0, 4.0])
    c.alphas = np.zeros(7)
    runs = [1.0, 2.0, 3.0, 7.0, 4.0, 6.0, 5.0]
    c.Cl = np.array(runs) * 0.1
    c.Cd = np.full(7, 0.05)
    c.sweep_dirs = np.array(['up', 'up', 'up', 'dn', 'up', 'dn', 'up'])
    c.run_numbers = np.array(runs)
    up, dn = _Harness(c).draw("Beta")
    np.testing.assert_allclose(up['x'], [-4, -2, 0, 2, 4])
    np.testing.assert_allclose(dn['x'], [2, 0])
