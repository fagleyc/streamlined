"""GUI-facing multi-speed (single-config, Mach-selectable) behavior.

Covers Casey's round: all speed increments live in ONE config; the Mach
filter selects which speed step to visualize; the MATLAB export lays each
series out as a squeezed (alpha, beta, mach) 3-D array.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from utils.gui.models.case import TestCase as Case, CaseCollection
from utils.gui.views.table_panel import TablePanel
from utils.gui.views.plot_panel import PlotPanel


def _multispeed_case():
    # 3 alphas x 1 beta x 2 machs, sorted (alpha, beta, mach)
    c = Case(id="c1", name="LSWT_Ex")
    c.alphas = np.array([-2., -2, 0, 0, 2, 2])
    c.betas = np.zeros(6)
    c.machs = np.array([0.030, 0.060, 0.030, 0.060, 0.030, 0.060])
    c.Cl = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    return c


class TestAllMachNumbers:
    def test_distinct_per_point_machs(self):
        coll = CaseCollection()
        coll.add(_multispeed_case())
        # both speed steps enumerated (not one averaged value per case)
        assert coll.all_mach_numbers == [0.03, 0.06]

    def test_low_speed_steps_not_merged_at_3dp(self):
        c = Case(id="c2", name="lo")
        c.alphas = np.array([0., 0.])
        c.betas = np.zeros(2)
        c.machs = np.array([0.031, 0.038])   # would merge at 2 dp
        c.Cl = np.array([0.0, 0.0])
        coll = CaseCollection()
        coll.add(c)
        assert coll.all_mach_numbers == [0.031, 0.038]


class TestMachPointMask:
    def test_selects_one_speed_step(self):
        c = _multispeed_case()
        mask = PlotPanel._mach_point_mask(c, 0.030)
        assert mask is not None
        assert mask.tolist() == [True, False, True, False, True, False]

    def test_none_when_no_selection(self):
        assert PlotPanel._mach_point_mask(_multispeed_case(), None) is None

    def test_none_when_machs_misaligned(self):
        c = _multispeed_case()
        c.machs = np.array([0.03])            # wrong length -> no filtering
        assert PlotPanel._mach_point_mask(c, 0.03) is None


class TestCase3DGrid:
    def test_alpha_beta_mach_grid_squeezes_singleton_beta(self):
        to3d, axes = TablePanel._case_3d_grid(_multispeed_case())
        assert to3d is not None
        np.testing.assert_allclose(axes['alpha'], [-2, 0, 2])
        np.testing.assert_allclose(axes['beta'], [0])
        np.testing.assert_allclose(axes['mach'], [0.03, 0.06])
        assert list(axes['dim_order']) == ['alpha', 'beta', 'mach']
        g = to3d(_multispeed_case().Cl)
        assert g.shape == (3, 2)              # (3,1,2) squeezed
        np.testing.assert_allclose(g[0], [0.1, 0.2])   # alpha=-2: m.03, m.06
        np.testing.assert_allclose(g[2], [0.5, 0.6])   # alpha=+2

    def test_single_speed_single_beta_squeezes_to_1d(self):
        c = Case(id="c3", name="one")
        c.alphas = np.array([-2., 0, 2])
        c.betas = np.zeros(3)
        c.machs = np.array([0.3, 0.3, 0.3])
        to3d, axes = TablePanel._case_3d_grid(c)
        g = to3d(np.array([1., 2., 3.]))
        assert g.shape == (3,)
        np.testing.assert_allclose(g, [1, 2, 3])

    def test_full_grid_alpha_beta_mach(self):
        # 2 alpha x 2 beta x 2 mach = 8 points
        c = Case(id="c4", name="grid")
        a, b, m, cl = [], [], [], []
        v = 0
        for av in (0., 4.):
            for bv in (0., 5.):
                for mv in (0.3, 0.5):
                    a.append(av); b.append(bv); m.append(mv); cl.append(v)
                    v += 1
        c.alphas = np.array(a); c.betas = np.array(b)
        c.machs = np.array(m); c.Cl = np.array(cl)
        to3d, axes = TablePanel._case_3d_grid(c)
        g = to3d(c.Cl)
        assert g.shape == (2, 2, 2)           # no singleton to squeeze
        assert g[0, 0, 0] == 0 and g[1, 1, 1] == 7

    def test_no_grid_without_machs(self):
        c = Case(id="c5", name="nomach")
        c.alphas = np.array([0., 2.])
        c.betas = np.zeros(2)
        # machs left empty -> not applicable
        to3d, axes = TablePanel._case_3d_grid(c)
        assert to3d is None and axes is None
