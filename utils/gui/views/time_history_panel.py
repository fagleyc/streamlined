"""
Time History Panel
==================

Panel for displaying time-series data and FFT frequency spectra
from reduced DAQ data points.

Always plots the selected point/channel for all visible cases,
enabling multi-case comparison with a color-coded legend.
"""

import numpy as np
from typing import Optional, Tuple, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QPushButton
)
from PyQt6.QtCore import Qt

from ..models.data_model import DataModel
from ..widgets.plot_canvas import PlotCanvas
from ..utils.themes import DarkTheme


def _element_names(balance_config: str = 'Force') -> list:
    """Return element display names based on balance configuration."""
    if balance_config == 'Moment':
        return ['AftPitch', 'AftYaw', 'FwdPitch', 'FwdYaw', 'Axial', 'Roll']
    return ['N1', 'N2', 'Y1', 'Y2', 'Axial', 'Roll']


class TimeHistoryPanel(QWidget):
    """Panel for viewing raw and reduced time-series data with optional FFT."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._updating = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QFrame()
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border-bottom: 1px solid {DarkTheme.BORDER};
            }}
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)

        toolbar_layout.addWidget(QLabel("Case:"))
        self.cmb_case = QComboBox()
        self.cmb_case.setMinimumWidth(180)
        toolbar_layout.addWidget(self.cmb_case)

        toolbar_layout.addWidget(QLabel("Point:"))
        self.cmb_point = QComboBox()
        self.cmb_point.setMinimumWidth(200)
        toolbar_layout.addWidget(self.cmb_point)

        toolbar_layout.addWidget(QLabel("Channel:"))
        self.cmb_channel = QComboBox()
        self.cmb_channel.setMinimumWidth(160)
        toolbar_layout.addWidget(self.cmb_channel)

        toolbar_layout.addWidget(QLabel("Domain:"))
        self.cmb_domain = QComboBox()
        self.cmb_domain.addItem("Time Domain", "time")
        self.cmb_domain.addItem("Frequency Domain (FFT)", "fft")
        toolbar_layout.addWidget(self.cmb_domain)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setToolTip("Clear all traces from the plot")
        self.btn_clear.setFixedWidth(60)
        self.btn_clear.clicked.connect(self._clear_plot)
        toolbar_layout.addWidget(self.btn_clear)

        toolbar_layout.addStretch()
        layout.addWidget(toolbar)

        self.plot_canvas = PlotCanvas(interactive=False)
        layout.addWidget(self.plot_canvas, stretch=1)

        self.info_bar = QFrame()
        self.info_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHTER};
                border-top: 1px solid {DarkTheme.BORDER};
            }}
        """)
        info_layout = QHBoxLayout(self.info_bar)
        info_layout.setContentsMargins(8, 4, 8, 4)
        self.lbl_stats = QLabel("No data")
        self.lbl_stats.setStyleSheet(
            f"color: {DarkTheme.TEXT_SECONDARY}; font-family: monospace;"
        )
        info_layout.addWidget(self.lbl_stats)
        info_layout.addStretch()
        layout.addWidget(self.info_bar)

    def _connect_signals(self):
        self.model.cases_changed.connect(self._update_case_list)
        self.model.case_visibility_changed.connect(self._on_visibility_changed)
        self.cmb_case.currentIndexChanged.connect(self._update_point_list)
        self.cmb_point.currentIndexChanged.connect(self._update_channel_list)
        self.cmb_channel.currentIndexChanged.connect(self._update_plot)
        self.cmb_domain.currentIndexChanged.connect(self._update_plot)

    def _on_visibility_changed(self, case_id, visible):
        self._update_plot()

    def _clear_plot(self):
        self.plot_canvas.clear()
        self.plot_canvas.refresh()
        self.lbl_stats.setText("No data")

    def _update_case_list(self):
        self._updating = True
        self.cmb_case.blockSignals(True)
        self.cmb_case.clear()
        for case in self.model.cases:
            self.cmb_case.addItem(case.name, case.id)
        self.cmb_case.blockSignals(False)
        self._updating = False
        self._update_point_list()

    def _update_point_list(self):
        self._updating = True
        self.cmb_point.blockSignals(True)
        self.cmb_point.clear()
        case = self._get_selected_case()
        if case is not None and hasattr(case, 'daq') and case.daq is not None:
            red = getattr(case.daq, 'red', None)
            if red:
                for i, pt in enumerate(red):
                    a = float(np.mean(pt.alpha)) if len(pt.alpha) > 0 else 0.0
                    b = float(np.mean(pt.beta)) if len(pt.beta) > 0 else 0.0
                    self.cmb_point.addItem(
                        f"Point {i}: \u03b1={a:.1f}\u00b0, \u03b2={b:.1f}\u00b0", i
                    )
        self.cmb_point.blockSignals(False)
        self._updating = False
        self._update_channel_list()

    def _update_channel_list(self):
        self._updating = True
        self.cmb_channel.blockSignals(True)
        self.cmb_channel.clear()
        pt = self._get_selected_point()
        if pt is not None:
            if pt.air_on:
                skip = {'Time', 'Alpha', 'Beta'}
                for key in pt.air_on:
                    if key not in skip:
                        self.cmb_channel.addItem(f"Raw: {key}", ('raw', key))
            for attr in ('Lift', 'Drag', 'Side', 'Roll', 'Pitch', 'Yaw'):
                val = getattr(pt.wrf_aero, attr, None)
                if val is not None and len(val) > 0:
                    self.cmb_channel.addItem(f"Force: {attr}", ('force', attr))
            for attr in ('Cl', 'Cd', 'Cs', 'CRoll', 'CPitch', 'CYaw'):
                val = getattr(pt.coeffs, attr, None)
                if val is not None and len(val) > 0:
                    self.cmb_channel.addItem(f"Coeff: {attr}", ('coeff', attr))
            if hasattr(pt, 'brf_on') and pt.brf_on is not None:
                elems = getattr(pt.brf_on, 'elements', None)
                if (elems is not None and hasattr(elems, 'ndim')
                        and elems.ndim == 2 and elems.shape[1] >= 6):
                    bal_cfg = getattr(self.model, 'balance_config', 'Force')
                    for col_idx, name in enumerate(
                            _element_names(bal_cfg)):
                        self.cmb_channel.addItem(
                            f"Element: {name}", ('element', name))
            for attr in ('Q', 'U_inf', 'Mach', 'Re', 'rho', 'T', 'T0',
                        'P_tot', 'P_static', 'a'):
                val = getattr(pt.tunnel, attr, None)
                if val is not None and len(val) > 0:
                    self.cmb_channel.addItem(f"Tunnel: {attr}", ('tunnel', attr))
        self.cmb_channel.blockSignals(False)
        self._updating = False
        self._update_plot()

    def _get_selected_case(self):
        case_id = self.cmb_case.currentData()
        if case_id is None:
            return None
        return self.model.cases.get(case_id)

    def _get_selected_point(self):
        case = self._get_selected_case()
        if case is None or not hasattr(case, 'daq') or case.daq is None:
            return None
        red = getattr(case.daq, 'red', None)
        if not red:
            return None
        idx = self.cmb_point.currentData()
        if idx is None or idx < 0 or idx >= len(red):
            return None
        return red[idx]

    def _get_selected_alpha_beta(self):
        pt = self._get_selected_point()
        if pt is None:
            return None
        alpha = float(np.mean(pt.alpha)) if len(pt.alpha) > 0 else 0.0
        beta = float(np.mean(pt.beta)) if len(pt.beta) > 0 else 0.0
        return alpha, beta

    def _extract_signal(self, pt, channel_data):
        source, key = channel_data
        time = pt.time if len(pt.time) > 0 else np.arange(0)
        if source == 'raw':
            signal = np.asarray(pt.air_on.get(key, []))
        elif source == 'force':
            signal = np.asarray(getattr(pt.wrf_aero, key, []))
        elif source == 'coeff':
            signal = np.asarray(getattr(pt.coeffs, key, []))
        elif source == 'element':
            elem_col = {
                'N1': 0, 'N2': 1, 'Y1': 2, 'Y2': 3,
                'AftPitch': 0, 'AftYaw': 1, 'FwdPitch': 2, 'FwdYaw': 3,
                'Axial': 4, 'Ax': 4, 'Roll': 5,
            }
            col_idx = elem_col.get(key)
            if col_idx is None:
                return None
            elems_on = (getattr(pt.brf_on, 'elements', None)
                        if hasattr(pt, 'brf_on') else None)
            if (elems_on is None or elems_on.ndim != 2
                    or col_idx >= elems_on.shape[1]):
                return None
            # Subtract air-off tare from air-on elements
            elems_off = (getattr(pt.brf_off, 'elements', None)
                         if hasattr(pt, 'brf_off') and pt.brf_off is not None
                         else None)
            if (elems_off is not None and elems_off.ndim == 2
                    and elems_off.shape[1] > col_idx):
                signal = elems_on[:, col_idx] - np.mean(elems_off[:, col_idx])
            else:
                signal = elems_on[:, col_idx]
        elif source == 'tunnel':
            signal = np.asarray(getattr(pt.tunnel, key, []))
        else:
            return None
        if len(signal) == 0 or len(time) == 0:
            return None
        n = min(len(time), len(signal))
        return time[:n], signal[:n], f"{source.title()}: {key}"

    def _find_matching_point(self, case, target_alpha, target_beta):
        if not hasattr(case, 'daq') or case.daq is None:
            return None
        red = getattr(case.daq, 'red', None)
        if not red:
            return None
        best_pt = None
        best_dist = float('inf')
        for pt in red:
            a = float(np.mean(pt.alpha)) if len(pt.alpha) > 0 else 0.0
            b = float(np.mean(pt.beta)) if len(pt.beta) > 0 else 0.0
            dist = abs(a - target_alpha) + abs(b - target_beta)
            if dist < best_dist:
                best_dist = dist
                best_pt = pt
        if best_dist < 1.0:
            return best_pt
        return None

    def _update_plot(self):
        if self._updating:
            return
        self.plot_canvas.clear()
        channel_data = self.cmb_channel.currentData()
        if channel_data is None:
            self.lbl_stats.setText("No data")
            self.plot_canvas.set_labels(title="No data to display")
            self.plot_canvas.refresh()
            return
        domain = self.cmb_domain.currentData()
        ab = self._get_selected_alpha_beta()
        if ab is None:
            self.lbl_stats.setText("No data")
            self.plot_canvas.set_labels(title="No data to display")
            self.plot_canvas.refresh()
            return
        target_alpha, target_beta = ab
        last_time = None
        last_signal = None
        plotted = 0
        for case in self.model.cases:
            if not case.visible or not case.has_data:
                continue
            pt = self._find_matching_point(case, target_alpha, target_beta)
            if pt is None:
                continue
            result = self._extract_signal(pt, channel_data)
            if result is None:
                continue
            time, signal, ch_label = result
            pt_a = float(np.mean(pt.alpha)) if len(pt.alpha) > 0 else 0.0
            pt_b = float(np.mean(pt.beta)) if len(pt.beta) > 0 else 0.0
            label = f"{case.name} - {ch_label} (a={pt_a:.1f}, b={pt_b:.1f})"
            self._plot_signal(time, signal, label, domain, color=case.color)
            last_time, last_signal = time, signal
            plotted += 1
        if plotted == 0:
            self.lbl_stats.setText("No data")
            self.plot_canvas.set_labels(title="No data to display")
        else:
            if last_time is not None and last_signal is not None:
                self._update_stats(last_time, last_signal)
            self.plot_canvas.add_legend()
        self.plot_canvas.refresh()

    def _plot_signal(self, time, signal, label, domain, color=None):
        kw = dict(label=label, marker='', linestyle='-')
        if color:
            kw['color'] = color
        if domain == 'fft':
            self._plot_frequency_domain(time, signal, **kw)
        else:
            self._plot_time_domain(time, signal, **kw)

    def _plot_time_domain(self, time, signal, **kwargs):
        self.plot_canvas.plot(time, signal, **kwargs)
        self.plot_canvas.set_labels(xlabel="Time [s]", ylabel="Signal")

    def _plot_frequency_domain(self, time, signal, **kwargs):
        N = len(signal)
        if N < 2:
            return
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        if dt <= 0:
            dt = 1.0
        freqs = np.fft.rfftfreq(N, d=dt)
        magnitude = 2.0 / N * np.abs(np.fft.rfft(signal - np.mean(signal)))
        self.plot_canvas.plot(freqs, magnitude, **kwargs)
        self.plot_canvas.set_labels(xlabel="Frequency [Hz]", ylabel="Magnitude")

    def _update_stats(self, time, signal):
        n = len(signal)
        mean = np.mean(signal)
        std = np.std(signal)
        smin = np.min(signal)
        smax = np.max(signal)
        duration = time[-1] - time[0] if n > 1 else 0.0
        dt = (time[1] - time[0]) if n > 1 else 0.0
        fs = 1.0 / dt if dt > 0 else 0.0
        self.lbl_stats.setText(
            f"N={n}  Mean={mean:.4f}  Std={std:.4f}  "
            f"Min={smin:.4f}  Max={smax:.4f}  "
            f"Fs={fs:.1f} Hz  Duration={duration:.3f} s"
        )
