"""
Dialogs
=======

Dialog windows for geometry settings, calibration, etc.
"""

from pathlib import Path
from typing import List

import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QDoubleSpinBox, QComboBox, QPushButton,
    QDialogButtonBox, QFrame, QTextBrowser, QTabWidget, QWidget,
    QListWidget, QListWidgetItem, QFileDialog, QAbstractItemView,
    QTreeView, QListView, QCheckBox, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..utils.themes import DarkTheme
from ..utils.icons import Icons
from .. import __version__, __app_name__


class GeometryDialog(QDialog):
    """
    Dialog for managing multiple model geometry definitions.

    Layout: geometry list on left, edit form on right.
    Each geometry has MAC, span, ref area, MRC offset, and input units.
    """

    def __init__(self, parent=None, geometries: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Model Geometry")
        self.setMinimumSize(650, 500)

        if geometries is None:
            geometries = {
                'Default': {
                    'mac': 1.0, 'ref_area': 1.0, 'span': 1.0,
                    'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'
                }
            }

        # Deep copy so edits don't modify original until accepted
        import copy
        self._geometries = copy.deepcopy(geometries)
        self._current_name = None

        self._setup_ui()
        self._populate_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Main content: list on left, form on right
        content = QHBoxLayout()

        # --- Left: geometry list ---
        left = QVBoxLayout()

        lbl_geos = QLabel("Geometries")
        lbl_geos.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        left.addWidget(lbl_geos)

        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(180)
        self.list_widget.setMinimumWidth(140)
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        left.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_add.setToolTip("Define a new geometry")
        self.btn_add.clicked.connect(self._add_geometry)
        btn_row.addWidget(self.btn_add)

        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setToolTip("Remove selected geometry")
        self.btn_remove.clicked.connect(self._remove_geometry)
        btn_row.addWidget(self.btn_remove)
        left.addLayout(btn_row)

        content.addLayout(left)

        # --- Right: edit form ---
        right = QVBoxLayout()

        # Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.txt_name = QLineEdit()
        self.txt_name.editingFinished.connect(self._on_name_edited)
        name_layout.addWidget(self.txt_name)
        right.addLayout(name_layout)

        # Reference values
        ref_group = QGroupBox("Reference Values")
        ref_layout = QFormLayout(ref_group)

        self.spn_mac = QDoubleSpinBox()
        self.spn_mac.setRange(0.001, 10000)
        self.spn_mac.setDecimals(4)
        ref_layout.addRow("Mean Aerodynamic Chord:", self.spn_mac)

        self.spn_span = QDoubleSpinBox()
        self.spn_span.setRange(0.001, 10000)
        self.spn_span.setDecimals(4)
        ref_layout.addRow("Reference Span:", self.spn_span)

        self.spn_area = QDoubleSpinBox()
        self.spn_area.setRange(0.001, 100000)
        self.spn_area.setDecimals(4)
        ref_layout.addRow("Reference Area:", self.spn_area)

        right.addWidget(ref_group)

        # MRC offset
        mrc_group = QGroupBox("Moment Reference Center Offset")
        mrc_layout = QFormLayout(mrc_group)

        self.spn_mrc_x = QDoubleSpinBox()
        self.spn_mrc_x.setRange(-1000, 1000)
        self.spn_mrc_x.setDecimals(4)
        mrc_layout.addRow("X (forward):", self.spn_mrc_x)

        self.spn_mrc_y = QDoubleSpinBox()
        self.spn_mrc_y.setRange(-1000, 1000)
        self.spn_mrc_y.setDecimals(4)
        mrc_layout.addRow("Y (right):", self.spn_mrc_y)

        self.spn_mrc_z = QDoubleSpinBox()
        self.spn_mrc_z.setRange(-1000, 1000)
        self.spn_mrc_z.setDecimals(4)
        mrc_layout.addRow("Z (down):", self.spn_mrc_z)

        right.addWidget(mrc_group)

        # Input units
        units_group = QGroupBox("Geometry Input Units")
        units_layout = QFormLayout(units_group)

        self.cmb_units = QComboBox()
        self.cmb_units.addItems(["IPS", "FPS", "MKS", "CGS"])
        units_layout.addRow("Unit System:", self.cmb_units)

        units_info = QLabel("Units for the geometry values entered above")
        units_info.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; font-size: 9pt;")
        units_layout.addRow("", units_info)

        right.addWidget(units_group)
        right.addStretch()

        content.addLayout(right, 1)
        layout.addLayout(content, 1)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_list(self):
        """Populate the geometry list widget."""
        self.list_widget.clear()
        for name in self._geometries:
            self.list_widget.addItem(name)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _on_selection_changed(self, current, previous):
        """Save previous selection, load new one."""
        if previous and self._current_name:
            self._save_current_to_dict()

        if current:
            name = current.text()
            self._current_name = name
            self._load_from_dict(name)

    def _load_from_dict(self, name: str):
        """Load geometry values into the form."""
        geo = self._geometries.get(name, {})
        self.txt_name.setText(name)
        self.spn_mac.setValue(geo.get('mac', 1.0))
        self.spn_span.setValue(geo.get('span', 1.0))
        self.spn_area.setValue(geo.get('ref_area', 1.0))
        mrc = geo.get('mrc', [0.0, 0.0, 0.0])
        self.spn_mrc_x.setValue(mrc[0])
        self.spn_mrc_y.setValue(mrc[1])
        self.spn_mrc_z.setValue(mrc[2])
        idx = self.cmb_units.findText(geo.get('units', 'IPS'))
        if idx >= 0:
            self.cmb_units.setCurrentIndex(idx)

    def _save_current_to_dict(self):
        """Save current form values to the geometries dict."""
        if not self._current_name or self._current_name not in self._geometries:
            return
        self._geometries[self._current_name] = {
            'mac': self.spn_mac.value(),
            'ref_area': self.spn_area.value(),
            'span': self.spn_span.value(),
            'mrc': [self.spn_mrc_x.value(), self.spn_mrc_y.value(), self.spn_mrc_z.value()],
            'units': self.cmb_units.currentText(),
        }

    def _on_name_edited(self):
        """Handle geometry rename."""
        new_name = self.txt_name.text().strip()
        if not new_name or not self._current_name:
            return
        if new_name == self._current_name:
            return
        if new_name in self._geometries:
            # Name already taken, revert
            self.txt_name.setText(self._current_name)
            return
        # Rename in dict
        self._geometries[new_name] = self._geometries.pop(self._current_name)
        old_name = self._current_name
        self._current_name = new_name
        # Update list item text
        current_item = self.list_widget.currentItem()
        if current_item:
            current_item.setText(new_name)

    def _add_geometry(self):
        """Add a new geometry with default values."""
        # Generate unique name
        base = "New Geometry"
        name = base
        n = 1
        while name in self._geometries:
            n += 1
            name = f"{base} {n}"

        self._save_current_to_dict()
        self._geometries[name] = {
            'mac': 1.0, 'ref_area': 1.0, 'span': 1.0,
            'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'
        }
        self.list_widget.addItem(name)
        self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def _remove_geometry(self):
        """Remove selected geometry (cannot remove the last one)."""
        if self.list_widget.count() <= 1:
            return
        current = self.list_widget.currentItem()
        if not current:
            return
        name = current.text()
        row = self.list_widget.row(current)
        self.list_widget.takeItem(row)
        if name in self._geometries:
            del self._geometries[name]
        self._current_name = None
        # Select next available
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(min(row, self.list_widget.count() - 1))

    def _on_accept(self):
        """Save current form before accepting."""
        self._save_current_to_dict()
        self.accept()

    def get_geometries(self) -> dict:
        """Get all geometry definitions."""
        return self._geometries

    # Backward compatibility: get_values returns default geometry
    def get_values(self) -> dict:
        """Get values for the first geometry (backward compat)."""
        if self._geometries:
            name = next(iter(self._geometries))
            geo = self._geometries[name]
            return {
                'mac': geo.get('mac', 1.0),
                'ref_area': geo.get('ref_area', 1.0),
                'span': geo.get('span', 1.0),
                'mrc': geo.get('mrc', [0.0, 0.0, 0.0]),
                'units': geo.get('units', 'IPS'),
            }
        return {'mac': 1.0, 'ref_area': 1.0, 'span': 1.0,
                'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'}


class OutputUnitsDialog(QDialog):
    """Dialog for setting output unit system (independent of geometry)."""

    def __init__(self, parent=None, current_units: str = "IPS"):
        super().__init__(parent)
        self.setWindowTitle("Output Units")
        self.setMinimumWidth(350)
        self._setup_ui(current_units)

    def _setup_ui(self, current_units: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        group = QGroupBox("Output Unit System")
        form = QFormLayout(group)

        self.cmb_units = QComboBox()
        self.cmb_units.addItems(["IPS", "FPS", "MKS", "CGS"])
        idx = self.cmb_units.findText(current_units)
        if idx >= 0:
            self.cmb_units.setCurrentIndex(idx)
        form.addRow("Unit System:", self.cmb_units)

        info = QLabel(
            "IPS: lbf, lb-in, psi, ft/s\n"
            "FPS: lbf, lb-ft, psf, ft/s\n"
            "MKS: N, N-m, Pa, m/s (SI)\n"
            "CGS: dyn, dyn-cm, Pa, cm/s"
        )
        info.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; font-size: 9pt;")
        form.addRow("", info)

        layout.addWidget(group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_units(self) -> str:
        """Get the selected output unit system."""
        return self.cmb_units.currentText()


class TunnelCorrectionsDialog(QDialog):
    """
    Dialog for configuring tunnel blockage / wall-effect corrections.

    The user selects an approach from a dropdown; the relevant input
    fields enable / disable based on the selection.  Defaults are
    chosen so an unchanged dialog yields zero correction.
    """

    METHODS = [
        ('none', 'None (no correction)'),
        ('pope_kirsten', 'Pope-Harper (Kirsten Wind Tunnel)'),
        ('pope_generic', 'Pope-Harper (generic facility)'),
        ('maskell', 'Maskell (stalled / bluff-body)'),
        ('glauert_closed', 'Glauert (closed test section)'),
    ]

    def __init__(self, parent=None, current_config: dict = None):
        super().__init__(parent)
        self.setWindowTitle('Tunnel Corrections')
        self.setMinimumWidth(440)
        cfg = current_config or {'method': 'none'}
        self._setup_ui(cfg)

    def _setup_ui(self, cfg: dict):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Method selector
        method_group = QGroupBox('Correction Approach')
        method_layout = QFormLayout(method_group)
        self.cmb_method = QComboBox()
        for key, label in self.METHODS:
            self.cmb_method.addItem(label, key)
        idx = self.cmb_method.findData(cfg.get('method', 'none'))
        if idx >= 0:
            self.cmb_method.setCurrentIndex(idx)
        self.cmb_method.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addRow('Method:', self.cmb_method)
        layout.addWidget(method_group)

        # Description label updates per-method
        self.lbl_description = QLabel('')
        self.lbl_description.setWordWrap(True)
        self.lbl_description.setStyleSheet(
            f'color: {DarkTheme.TEXT_SECONDARY}; font-style: italic;')
        layout.addWidget(self.lbl_description)

        # Test section geometry.  Defaults reflect the SWT (subsonic wind
        # tunnel): 36" x 36" cross-section, area = 1296 in^2.
        self.geom_group = QGroupBox('Test Section / Reference')
        geom_layout = QFormLayout(self.geom_group)

        self.spn_ts_area = QDoubleSpinBox()
        self.spn_ts_area.setRange(0.0, 1e7)
        self.spn_ts_area.setDecimals(2)
        self.spn_ts_area.setSuffix(' in^2')
        self.spn_ts_area.setValue(
            cfg.get('test_section_area_in2', 1296.0))
        geom_layout.addRow('Test section area:', self.spn_ts_area)

        self.spn_ts_w = QDoubleSpinBox()
        self.spn_ts_w.setRange(0.0, 1e4)
        self.spn_ts_w.setDecimals(2)
        self.spn_ts_w.setSuffix(' in')
        self.spn_ts_w.setValue(cfg.get('test_section_width_in', 36.0))
        geom_layout.addRow('Test section width:', self.spn_ts_w)

        self.spn_ts_h = QDoubleSpinBox()
        self.spn_ts_h.setRange(0.0, 1e4)
        self.spn_ts_h.setDecimals(2)
        self.spn_ts_h.setSuffix(' in')
        self.spn_ts_h.setValue(cfg.get('test_section_height_in', 36.0))
        geom_layout.addRow('Test section height:', self.spn_ts_h)

        self.spn_ref_area = QDoubleSpinBox()
        self.spn_ref_area.setRange(0.0, 1e7)
        self.spn_ref_area.setDecimals(2)
        self.spn_ref_area.setSuffix(' in^2')
        self.spn_ref_area.setValue(cfg.get('reference_area_in2', 1.0))
        geom_layout.addRow('Wing reference area S:', self.spn_ref_area)

        layout.addWidget(self.geom_group)

        # Pope coefficients (generic)
        self.pope_group = QGroupBox('Pope-Harper coefficients (generic)')
        pope_layout = QFormLayout(self.pope_group)
        self.spn_lambda = QDoubleSpinBox()
        self.spn_lambda.setRange(0.0, 10.0)
        self.spn_lambda.setDecimals(4)
        self.spn_lambda.setValue(cfg.get('lambda_', 1.0))
        pope_layout.addRow('lambda:', self.spn_lambda)

        self.spn_k = QDoubleSpinBox()
        self.spn_k.setRange(0.0, 10.0)
        self.spn_k.setDecimals(4)
        self.spn_k.setValue(cfg.get('k', 0.333))
        pope_layout.addRow('k:', self.spn_k)

        self.spn_delta = QDoubleSpinBox()
        self.spn_delta.setRange(0.0, 10.0)
        self.spn_delta.setDecimals(4)
        self.spn_delta.setValue(cfg.get('delta', 0.141))
        pope_layout.addRow('delta:', self.spn_delta)

        self.spn_sigma = QDoubleSpinBox()
        self.spn_sigma.setRange(0.0, 10.0)
        self.spn_sigma.setDecimals(4)
        self.spn_sigma.setValue(cfg.get('sigma', 0.011))
        pope_layout.addRow('sigma:', self.spn_sigma)

        layout.addWidget(self.pope_group)

        # Frontal area anchors (linear interp by alpha)
        self.area_group = QGroupBox(
            'Solid blockage frontal area (linear interp by alpha)')
        area_layout = QFormLayout(self.area_group)
        self.spn_alpha_lo = QDoubleSpinBox()
        self.spn_alpha_lo.setRange(-90, 90)
        self.spn_alpha_lo.setDecimals(2)
        self.spn_alpha_lo.setSuffix(' deg')
        self.spn_alpha_lo.setValue(
            cfg.get('frontal_area_alpha_low_deg', 0.0))
        area_layout.addRow('Low alpha:', self.spn_alpha_lo)

        self.spn_area_lo = QDoubleSpinBox()
        self.spn_area_lo.setRange(0.0, 1e6)
        self.spn_area_lo.setDecimals(3)
        self.spn_area_lo.setSuffix(' in^2')
        self.spn_area_lo.setValue(
            cfg.get('frontal_area_alpha_low_in2', 0.0))
        area_layout.addRow('Frontal area at low alpha:', self.spn_area_lo)

        self.spn_alpha_hi = QDoubleSpinBox()
        self.spn_alpha_hi.setRange(-90, 90)
        self.spn_alpha_hi.setDecimals(2)
        self.spn_alpha_hi.setSuffix(' deg')
        self.spn_alpha_hi.setValue(
            cfg.get('frontal_area_alpha_high_deg', 20.0))
        area_layout.addRow('High alpha:', self.spn_alpha_hi)

        self.spn_area_hi = QDoubleSpinBox()
        self.spn_area_hi.setRange(0.0, 1e6)
        self.spn_area_hi.setDecimals(3)
        self.spn_area_hi.setSuffix(' in^2')
        self.spn_area_hi.setValue(
            cfg.get('frontal_area_alpha_high_in2', 0.0))
        area_layout.addRow('Frontal area at high alpha:', self.spn_area_hi)

        layout.addWidget(self.area_group)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initial state
        self._on_method_changed()

    def _on_method_changed(self):
        method = self.cmb_method.currentData()
        descriptions = {
            'none': 'No correction applied. Original CL, CD, alpha pass '
                    'through unchanged.',
            'pope_kirsten': 'Pope-Harper for the Kirsten Wind Tunnel with '
                            'fixed coefficients lambda=1.0, k=0.333, '
                            'delta=0.141, sigma=0.011.',
            'pope_generic': 'Pope-Harper with user-supplied facility '
                            'coefficients (see the field below).',
            'maskell': "Maskell's correction for stalled / bluff bodies; "
                       'requires test section area and wing reference '
                       'area.',
            'glauert_closed': 'Classical Glauert closed-section lift '
                              'interference. Uses delta from the Pope '
                              'coefficient block (default 0.125).',
        }
        self.lbl_description.setText(descriptions.get(method, ''))

        # Enable/disable input groups based on method
        is_pope_generic = (method == 'pope_generic')
        is_pope_kirsten = (method == 'pope_kirsten')
        is_pope = is_pope_generic or is_pope_kirsten
        is_glauert = (method == 'glauert_closed')
        is_maskell = (method == 'maskell')
        any_method = method != 'none'

        # Geometry block used by all non-none methods
        self.geom_group.setEnabled(any_method)
        # Pope coefficients only for generic Pope (kirsten is locked)
        self.pope_group.setEnabled(is_pope_generic or is_glauert)
        # Frontal area only relevant to Pope (solid blockage)
        self.area_group.setEnabled(is_pope)

    def get_config(self) -> dict:
        """Return the configuration dict (JSON-serializable)."""
        return {
            'method': self.cmb_method.currentData(),
            'test_section_area_in2': self.spn_ts_area.value(),
            'test_section_width_in': self.spn_ts_w.value(),
            'test_section_height_in': self.spn_ts_h.value(),
            'reference_area_in2': self.spn_ref_area.value(),
            'lambda_': self.spn_lambda.value(),
            'k': self.spn_k.value(),
            'delta': self.spn_delta.value(),
            'sigma': self.spn_sigma.value(),
            'frontal_area_alpha_low_deg': self.spn_alpha_lo.value(),
            'frontal_area_alpha_low_in2': self.spn_area_lo.value(),
            'frontal_area_alpha_high_deg': self.spn_alpha_hi.value(),
            'frontal_area_alpha_high_in2': self.spn_area_hi.value(),
        }


class CalculatorDialog(QDialog):
    """
    Edit / manage user-defined calculator rules.

    Layout:
      Left  - list of rules with Add / Duplicate / Remove buttons.
      Right - form editor for the currently selected rule with:
              * Rule metadata (name, index var, range, description, enabled)
              * Expression text field
              * Variable picker (category dropdown + variable dropdown +
                Insert button) so users don't have to remember names
              * Math function and operator buttons that insert text
                at the expression's cursor position
              * Live preview (expanded outputs + test evaluation)
              * Refresh button to re-scan available variables after
                loading new data
    """

    def __init__(self, parent=None, rules=None,
                 available_vars=None, preview_case=None,
                 refresh_callback=None):
        super().__init__(parent)
        self.setWindowTitle('Custom Calculator')
        self.setMinimumSize(940, 640)
        self._available_vars = list(available_vars or [])
        self._preview_case = preview_case
        self._refresh_callback = refresh_callback
        self._categorized = {}

        # Local copy of rules; only committed on accept
        from utils.windtunnel.calculator import CalcRule
        self._CalcRule = CalcRule
        self._rules = [self._copy_rule(r) for r in (rules or [])]
        self._current_index: int = -1
        self._suppress_form_update = False
        self._setup_ui()
        self._recompute_categories()
        if self._rules:
            self.list_widget.setCurrentRow(0)

    def _copy_rule(self, rule):
        return self._CalcRule(
            name_template=rule.name_template,
            expression=rule.expression,
            index_var=rule.index_var,
            index_range=rule.index_range,
            enabled=rule.enabled,
            description=rule.description,
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QHBoxLayout(self)

        # --- LEFT: rule list ---
        left = QVBoxLayout()
        left.addWidget(QLabel('Rules:'))
        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(190)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        left.addWidget(self.list_widget, 1)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton('+ Add')
        self.btn_add.clicked.connect(self._add_rule)
        btn_row.addWidget(self.btn_add)
        self.btn_dup = QPushButton('Duplicate')
        self.btn_dup.clicked.connect(self._duplicate_rule)
        btn_row.addWidget(self.btn_dup)
        self.btn_remove = QPushButton('- Remove')
        self.btn_remove.clicked.connect(self._remove_rule)
        btn_row.addWidget(self.btn_remove)
        left.addLayout(btn_row)
        outer.addLayout(left, 0)

        # --- RIGHT: editor + builder + preview ---
        right = QVBoxLayout()

        # Rule metadata - compact form at top
        meta_group = QGroupBox('Rule Metadata')
        meta_form = QFormLayout(meta_group)
        meta_form.setVerticalSpacing(4)

        self.chk_enabled = QCheckBox('Enabled')
        self.chk_enabled.stateChanged.connect(self._on_form_changed)
        meta_form.addRow('', self.chk_enabled)

        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText('Cp{i}')
        self.txt_name.textChanged.connect(self._on_form_changed)
        meta_form.addRow('Output name template:', self.txt_name)

        idx_row = QHBoxLayout()
        self.txt_index_var = QLineEdit()
        self.txt_index_var.setPlaceholderText('i')
        self.txt_index_var.setMaximumWidth(60)
        self.txt_index_var.textChanged.connect(self._on_form_changed)
        idx_row.addWidget(self.txt_index_var)
        idx_row.addWidget(QLabel('  Range:'))
        self.txt_range = QLineEdit()
        self.txt_range.setPlaceholderText(
            "1..32   or   1,5,10   or   auto:P{i}   or   blank")
        self.txt_range.textChanged.connect(self._on_form_changed)
        idx_row.addWidget(self.txt_range, 1)
        idx_widget = QWidget()
        idx_widget.setLayout(idx_row)
        meta_form.addRow('Index variable:', idx_widget)

        self.txt_desc = QLineEdit()
        self.txt_desc.setPlaceholderText('Description (optional)')
        self.txt_desc.textChanged.connect(self._on_form_changed)
        meta_form.addRow('Description:', self.txt_desc)

        right.addWidget(meta_group)

        # Expression builder
        builder_group = QGroupBox('Expression Builder')
        bv = QVBoxLayout(builder_group)
        bv.setSpacing(6)

        # Expression text field
        expr_row = QHBoxLayout()
        expr_row.addWidget(QLabel('Expression:'))
        self.txt_expr = QLineEdit()
        self.txt_expr.setPlaceholderText('(P{i} - p_inf) / q_inf')
        self.txt_expr.textChanged.connect(self._on_form_changed)
        expr_row.addWidget(self.txt_expr, 1)
        bv.addLayout(expr_row)

        # Variable picker
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel('Insert variable:'))
        self.cmb_category = QComboBox()
        self.cmb_category.setMinimumWidth(140)
        self.cmb_category.currentIndexChanged.connect(self._on_category_changed)
        picker_row.addWidget(self.cmb_category)
        self.cmb_variable = QComboBox()
        self.cmb_variable.setMinimumWidth(140)
        picker_row.addWidget(self.cmb_variable)
        self.btn_insert_var = QPushButton('Insert')
        self.btn_insert_var.clicked.connect(self._on_insert_variable)
        picker_row.addWidget(self.btn_insert_var)
        self.btn_refresh = QPushButton('Refresh')
        self.btn_refresh.setToolTip(
            'Re-scan available variables from the current case')
        self.btn_refresh.clicked.connect(self._on_refresh_variables)
        picker_row.addWidget(self.btn_refresh)
        self.btn_browse_all = QPushButton('Browse All...')
        self.btn_browse_all.setToolTip(
            'Open a window listing every available variable with '
            'its current sample value')
        self.btn_browse_all.clicked.connect(self._show_variable_browser)
        picker_row.addWidget(self.btn_browse_all)
        picker_row.addStretch(1)
        bv.addLayout(picker_row)

        # Math function buttons - two rows
        from utils.windtunnel.calculator import MATH_FUNCTIONS, OPERATORS
        math_label = QLabel('Math functions:')
        math_label.setStyleSheet(
            f'color: {DarkTheme.TEXT_SECONDARY}; font-size: 9pt;')
        bv.addWidget(math_label)
        math_grid = QGridLayout()
        math_grid.setSpacing(2)
        for idx, (lbl, txt) in enumerate(MATH_FUNCTIONS):
            btn = QPushButton(lbl)
            btn.setMaximumWidth(60)
            btn.setStyleSheet('padding: 3px;')
            btn.clicked.connect(
                lambda checked=False, t=txt: self._insert_at_cursor(t))
            math_grid.addWidget(btn, idx // 10, idx % 10)
        bv.addLayout(math_grid)

        # Operators
        op_label = QLabel('Operators:')
        op_label.setStyleSheet(
            f'color: {DarkTheme.TEXT_SECONDARY}; font-size: 9pt;')
        bv.addWidget(op_label)
        op_row = QHBoxLayout()
        op_row.setSpacing(2)
        for lbl, txt in OPERATORS:
            btn = QPushButton(lbl)
            btn.setMaximumWidth(45)
            btn.setStyleSheet('padding: 3px;')
            btn.clicked.connect(
                lambda checked=False, t=txt: self._insert_at_cursor(t))
            op_row.addWidget(btn)
        op_row.addStretch(1)
        bv.addLayout(op_row)

        right.addWidget(builder_group)

        # Preview pane
        prev_group = QGroupBox('Preview')
        prev_layout = QVBoxLayout(prev_group)
        self.lbl_expanded = QLabel('Expanded outputs: -')
        self.lbl_expanded.setWordWrap(True)
        prev_layout.addWidget(self.lbl_expanded)

        self.lbl_eval = QLabel('Evaluation: -')
        self.lbl_eval.setWordWrap(True)
        self.lbl_eval.setStyleSheet(
            f'color: {DarkTheme.TEXT_SECONDARY};')
        prev_layout.addWidget(self.lbl_eval)

        self.lbl_diag = QLabel('')
        self.lbl_diag.setWordWrap(True)
        self.lbl_diag.setStyleSheet(
            f'color: {DarkTheme.TEXT_SECONDARY}; font-size: 9pt;')
        prev_layout.addWidget(self.lbl_diag)

        right.addWidget(prev_group, 1)

        # OK / Cancel
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        right.addWidget(btns)

        outer.addLayout(right, 1)
        self._refresh_list()

    # ------------------------------------------------------------------
    # Variable picker logic
    # ------------------------------------------------------------------

    def _recompute_categories(self):
        try:
            from utils.windtunnel.calculator import categorize_variables
            self._categorized = categorize_variables(self._available_vars)
        except Exception:
            self._categorized = {}

        self.cmb_category.blockSignals(True)
        self.cmb_category.clear()
        for cat, names in self._categorized.items():
            self.cmb_category.addItem(f'{cat}  ({len(names)})', cat)
        self.cmb_category.blockSignals(False)
        # Default to Pressure Ports if present, else first non-empty
        target = None
        for i in range(self.cmb_category.count()):
            if self.cmb_category.itemData(i) == 'Pressure Ports':
                target = i
                break
        if target is None and self.cmb_category.count() > 0:
            target = 0
        if target is not None:
            self.cmb_category.setCurrentIndex(target)
        self._on_category_changed()
        self._update_diagnostic()

    def _on_category_changed(self):
        cat = self.cmb_category.currentData()
        self.cmb_variable.blockSignals(True)
        self.cmb_variable.clear()
        names = self._categorized.get(cat, [])
        for n in names:
            self.cmb_variable.addItem(n, n)
        self.cmb_variable.blockSignals(False)

    def _on_insert_variable(self):
        name = self.cmb_variable.currentData()
        if not name:
            return
        self._insert_at_cursor(name)

    def _insert_at_cursor(self, text: str):
        # QLineEdit.insert() inserts at current cursor position
        self.txt_expr.setFocus()
        self.txt_expr.insert(text)

    def _on_refresh_variables(self):
        if self._refresh_callback is None:
            return
        try:
            new_vars = list(self._refresh_callback() or [])
        except Exception:
            new_vars = []
        self._available_vars = new_vars
        self._recompute_categories()
        # Re-evaluate preview against the (possibly newly-loaded) case
        if (0 <= self._current_index < len(self._rules)):
            self._update_preview(self._rules[self._current_index])

    def _update_diagnostic(self):
        """Show a friendly diagnostic when no/few variables are available."""
        n = len(self._available_vars)
        if n == 0:
            self.lbl_diag.setText(
                'No data loaded yet. Load a data directory and process '
                'it, then click Refresh.')
        elif n < 5:
            self.lbl_diag.setText(
                f'{n} variable(s) detected.  If pressure ports are '
                'missing, the channels may be named differently in '
                'the TDMS files (e.g. P_001 vs P1).')
        else:
            self.lbl_diag.setText(
                f'{n} variables available.  Click "Browse All..." to '
                "see the full list with sample values.")

    def _show_variable_browser(self):
        """Open a dialog listing every available variable + sample mean."""
        # Build a list of (category, name, sample_mean)
        rows = []
        try:
            from utils.windtunnel.calculator import (
                categorize_variables, build_namespace)
        except Exception:
            categorize_variables = None
            build_namespace = None

        ns = {}
        if build_namespace is not None and self._preview_case is not None:
            red = (getattr(self._preview_case.daq, 'red', None)
                   if getattr(self._preview_case, 'daq', None) is not None
                   else None)
            if red:
                try:
                    ns = build_namespace(red[0])
                except Exception:
                    ns = {}

        cats = (categorize_variables(self._available_vars)
                if categorize_variables else {})
        for cat, names in cats.items():
            for n in names:
                v = ns.get(n)
                if v is None:
                    sample = ''
                else:
                    try:
                        arr = np.asarray(v, dtype=float)
                        if arr.size == 0:
                            sample = '(empty)'
                        else:
                            sample = f'{float(np.mean(arr)):.6g}'
                    except Exception:
                        sample = '(not numeric)'
                rows.append((cat, n, sample))

        # Create a modal dialog with a table
        dlg = QDialog(self)
        dlg.setWindowTitle('All Available Variables')
        dlg.resize(560, 560)
        v = QVBoxLayout(dlg)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel('Filter:'))
        edt_filter = QLineEdit()
        edt_filter.setPlaceholderText('Type to filter (substring match)...')
        filter_row.addWidget(edt_filter, 1)
        v.addLayout(filter_row)

        lst = QListWidget()
        lst.setAlternatingRowColors(True)

        def _repopulate(txt=''):
            txt_lower = (txt or '').lower()
            lst.clear()
            for cat, name, sample in rows:
                if txt_lower and txt_lower not in name.lower() \
                        and txt_lower not in cat.lower():
                    continue
                tail = f'  ({cat})' if cat else ''
                if sample:
                    text = f'{name:30s}  sample={sample}{tail}'
                else:
                    text = f'{name:30s}{tail}'
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, name)
                lst.addItem(item)

        edt_filter.textChanged.connect(_repopulate)
        _repopulate()
        v.addWidget(lst, 1)

        # Double-click inserts and closes
        def _on_double_click(item):
            name = item.data(Qt.ItemDataRole.UserRole)
            if name:
                self._insert_at_cursor(name)
                dlg.accept()
        lst.itemDoubleClicked.connect(_on_double_click)

        info = QLabel(
            'Double-click a variable to insert it at the expression '
            'cursor.  Use this to verify exact channel names if a rule '
            'expansion is producing NaN.')
        info.setWordWrap(True)
        info.setStyleSheet(
            f'color: {DarkTheme.TEXT_SECONDARY}; font-size: 9pt;')
        v.addWidget(info)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        v.addWidget(btns)

        dlg.exec()

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------

    def _refresh_list(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for r in self._rules:
            display = r.name_template or '(unnamed)'
            if not r.enabled:
                display = '[off] ' + display
            self.list_widget.addItem(display)
        self.list_widget.blockSignals(False)

    def _add_rule(self):
        new_rule = self._CalcRule(
            name_template='NewVar', expression='',
            index_var='i', index_range='', enabled=True)
        self._rules.append(new_rule)
        self._refresh_list()
        self.list_widget.setCurrentRow(len(self._rules) - 1)

    def _duplicate_rule(self):
        if self._current_index < 0:
            return
        self._rules.append(self._copy_rule(self._rules[self._current_index]))
        self._refresh_list()
        self.list_widget.setCurrentRow(len(self._rules) - 1)

    def _remove_rule(self):
        if self._current_index < 0:
            return
        del self._rules[self._current_index]
        self._refresh_list()
        if self._rules:
            self.list_widget.setCurrentRow(
                min(self._current_index, len(self._rules) - 1))
        else:
            self._current_index = -1
            self._load_form(None)

    def _on_selection_changed(self, row: int):
        # Save the form into the previously-selected rule before
        # switching, so edits aren't lost on rapid clicking.
        if (self._current_index != -1
                and self._current_index < len(self._rules)):
            self._save_form_into_rule(self._rules[self._current_index])
        self._current_index = row
        if 0 <= row < len(self._rules):
            self._load_form(self._rules[row])
        else:
            self._load_form(None)

    # ------------------------------------------------------------------
    # Form <-> rule
    # ------------------------------------------------------------------

    def _load_form(self, rule):
        self._suppress_form_update = True
        if rule is None:
            self.chk_enabled.setChecked(False)
            self.txt_name.setText('')
            self.txt_expr.setText('')
            self.txt_index_var.setText('i')
            self.txt_range.setText('')
            self.txt_desc.setText('')
        else:
            self.chk_enabled.setChecked(rule.enabled)
            self.txt_name.setText(rule.name_template)
            self.txt_expr.setText(rule.expression)
            self.txt_index_var.setText(rule.index_var)
            self.txt_range.setText(rule.index_range)
            self.txt_desc.setText(rule.description)
        self._suppress_form_update = False
        self._update_preview(rule)

    def _save_form_into_rule(self, rule):
        rule.enabled = self.chk_enabled.isChecked()
        rule.name_template = self.txt_name.text().strip()
        rule.expression = self.txt_expr.text().strip()
        rule.index_var = self.txt_index_var.text().strip() or 'i'
        rule.index_range = self.txt_range.text().strip()
        rule.description = self.txt_desc.text().strip()

    def _on_form_changed(self):
        if self._suppress_form_update:
            return
        if self._current_index < 0 or self._current_index >= len(self._rules):
            return
        rule = self._rules[self._current_index]
        self._save_form_into_rule(rule)
        # Refresh list label without rebuilding from scratch
        item = self.list_widget.item(self._current_index)
        if item is not None:
            display = rule.name_template or '(unnamed)'
            if not rule.enabled:
                display = '[off] ' + display
            item.setText(display)
        self._update_preview(rule)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self, rule):
        if rule is None:
            self.lbl_expanded.setText('Expanded outputs: -')
            self.lbl_eval.setText('Evaluation: -')
            return
        try:
            from utils.windtunnel.calculator import (
                expand_rule, evaluate, build_namespace)
        except Exception:
            self.lbl_expanded.setText('Expanded outputs: (engine import error)')
            return

        expansions = expand_rule(rule, self._available_vars)
        if not expansions:
            self.lbl_expanded.setText(
                'Expanded outputs: (none - check name/range)')
            self.lbl_eval.setText('Evaluation: -')
            return

        names = [n for n, _ in expansions]
        names_shown = ', '.join(names[:20])
        if len(names) > 20:
            names_shown += f'  (+{len(names) - 20} more)'
        self.lbl_expanded.setText(
            f'Expanded outputs ({len(names)}): {names_shown}')

        # Test evaluation on the first point of preview_case
        case = self._preview_case
        red = (getattr(case.daq, 'red', None)
               if case is not None and getattr(case, 'daq', None) is not None
               else None)
        if not red:
            self.lbl_eval.setText(
                'Evaluation: load a case to preview test values.')
            return
        pt = red[0]
        ns = build_namespace(pt)

        ok = []         # list of (name, mean)
        failed = []     # list of (name, error_str)
        for name, expr in expansions:
            try:
                r = evaluate(expr, ns)
                arr = np.asarray(r, dtype=float)
                if arr.size == 0:
                    failed.append((name, 'empty array'))
                    continue
                ok.append((name, float(np.mean(arr))))
            except ValueError as e:
                # Extract the missing variable name if NameError-like
                msg = str(e)
                # Trim to a useful one-line snippet
                if "'" in msg:
                    short = msg[msg.find("'"):msg.rfind("'") + 1]
                else:
                    short = msg.splitlines()[0][:120]
                failed.append((name, short))
            except Exception as e:
                failed.append((name, f'{type(e).__name__}: {e}'))

        n_total = len(expansions)
        n_ok = len(ok)
        if n_ok == 0:
            # Show the first failure to help diagnose
            if failed:
                _, msg = failed[0]
                self.lbl_eval.setText(
                    f'Evaluation FAILED: 0/{n_total} OK.  First error '
                    f'(rule {failed[0][0]}): {msg}')
            else:
                self.lbl_eval.setText(
                    f'Evaluation: {n_total} expansion(s), 0 evaluated.')
        elif n_ok == n_total:
            n, mean = ok[0]
            self.lbl_eval.setText(
                f'Evaluation OK: {n_ok}/{n_total} succeed.  '
                f'Sample {n} mean = {mean:.6g}')
        else:
            # Partial success - show count + first failing name + reason
            failed_names = [n for n, _ in failed]
            shown = ', '.join(failed_names[:5])
            if len(failed_names) > 5:
                shown += f' (+{len(failed_names) - 5})'
            _, reason = failed[0]
            self.lbl_eval.setText(
                f'PARTIAL: {n_ok}/{n_total} succeed.  Failed: {shown}.  '
                f'First reason: {reason}')

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def get_rules(self):
        # Make sure pending form edits are committed
        if 0 <= self._current_index < len(self._rules):
            self._save_form_into_rule(self._rules[self._current_index])
        # Filter empty rules out
        return [r for r in self._rules
                if r.name_template and r.expression]


class CalibrationDialog(QDialog):
    """
    Dialog for calibration settings.
    """

    def __init__(self, parent=None, cal_type: str = "Cubic",
                 facility: str = "SWT", balance_config: str = "Force",
                 pdiff_channel: str = "220", p0_channel: str = "690"):
        super().__init__(parent)
        self.setWindowTitle("Calibration Settings")
        self.setMinimumWidth(350)

        self._setup_ui(cal_type, facility, balance_config,
                       pdiff_channel, p0_channel)

    def _setup_ui(self, cal_type: str, facility: str, balance_config: str,
                  pdiff_channel: str, p0_channel: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Calibration type
        type_group = QGroupBox("Calibration Type")
        type_layout = QFormLayout(type_group)

        self.cmb_type = QComboBox()
        self.cmb_type.addItems(["Linear", "Quadratic", "Cubic"])
        idx = self.cmb_type.findText(cal_type)
        if idx >= 0:
            self.cmb_type.setCurrentIndex(idx)
        type_layout.addRow("Fit Type:", self.cmb_type)

        info_label = QLabel(
            "Linear: V\n"
            "Quadratic: V + V²\n"
            "Cubic: V + V² + V³"
        )
        info_label.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; font-size: 9pt;")
        type_layout.addRow("", info_label)

        layout.addWidget(type_group)

        # Facility settings
        fac_group = QGroupBox("Facility Settings")
        fac_layout = QFormLayout(fac_group)

        self.cmb_facility = QComboBox()
        self.cmb_facility.addItems(["SWT", "LSWT", "TST"])
        idx = self.cmb_facility.findText(facility)
        if idx >= 0:
            self.cmb_facility.setCurrentIndex(idx)
        fac_layout.addRow("Facility:", self.cmb_facility)

        self.cmb_balance = QComboBox()
        self.cmb_balance.addItems(["Internal", "External"])
        fac_layout.addRow("Balance Type:", self.cmb_balance)

        self.cmb_config = QComboBox()
        self.cmb_config.addItems(["Force", "Moment"])
        idx = self.cmb_config.findText(balance_config)
        if idx >= 0:
            self.cmb_config.setCurrentIndex(idx)
        fac_layout.addRow("Balance Config:", self.cmb_config)

        layout.addWidget(fac_group)

        # Pressure channels
        press_group = QGroupBox("Pressure Channels")
        press_layout = QFormLayout(press_group)

        self.txt_pdiff = QLineEdit(pdiff_channel)
        press_layout.addRow("Pdiff Channel:", self.txt_pdiff)

        self.txt_p0 = QLineEdit(p0_channel)
        press_layout.addRow("P0 Channel:", self.txt_p0)

        layout.addWidget(press_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> dict:
        """Get the entered values."""
        return {
            'cal_type': self.cmb_type.currentText(),
            'facility': self.cmb_facility.currentText(),
            'balance_type': self.cmb_balance.currentText(),
            'balance_config': self.cmb_config.currentText(),
            'pdiff_channel': self.txt_pdiff.text(),
            'p0_channel': self.txt_p0.text(),
        }


class AboutDialog(QDialog):
    """
    About dialog with application information.

    Follows the shared ecosystem template: app name + version prominent,
    one-paragraph summary, author/contact line, and a compact
    version-history table.

    Note: the changelog is intentionally built from
    ``utils.gui.about.VERSION_HISTORY`` rather than inline HTML. A
    previous revision embedded literal ``{i}`` placeholders inside an
    f-string, which raised ``NameError`` on construction.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {__app_name__}")
        self.setFixedSize(520, 560)

        self._setup_ui()

    def _setup_ui(self):
        from ..about import VERSION_HISTORY, SUMMARY, AUTHOR, CONTACT

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # App name + version, prominent
        title = QLabel(__app_name__)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        layout.addWidget(title)

        version = QLabel(f"Version {__version__}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setFont(QFont("Segoe UI", 11))
        version.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
        layout.addWidget(version)

        # One-paragraph summary
        summary = QLabel(SUMMARY)
        summary.setWordWrap(True)
        summary.setAlignment(Qt.AlignmentFlag.AlignJustify)
        summary.setStyleSheet(f"color: {DarkTheme.TEXT_PRIMARY};")
        layout.addWidget(summary)

        # Author / contact
        author = QLabel(f"Author: {AUTHOR} \u2014 {CONTACT}")
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
        layout.addWidget(author)

        # Compact version-history table
        rows = "".join(
            "<tr>"
            f"<td style='padding:2px 10px 2px 0; white-space:nowrap;"
            f" color:{DarkTheme.TEXT_PRIMARY};'><b>{ver}</b></td>"
            f"<td style='padding:2px 10px 2px 0; white-space:nowrap;"
            f" color:{DarkTheme.TEXT_SECONDARY};'>{date}</td>"
            f"<td style='padding:2px 0; color:{DarkTheme.TEXT_PRIMARY};'>"
            f"{note}</td>"
            "</tr>"
            for ver, date, note in VERSION_HISTORY
        )
        history = QTextBrowser()
        history.setHtml(
            "<div style=\"font-family:'Segoe UI'; font-size:9pt;\">"
            "<table cellspacing='0' cellpadding='0'>" + rows +
            "</table></div>"
        )
        history.setStyleSheet(
            f"QTextBrowser {{"
            f" background-color: {DarkTheme.BACKGROUND_LIGHT};"
            f" border: 1px solid {DarkTheme.BORDER};"
            f" border-radius: 4px; padding: 8px; }}"
        )
        layout.addWidget(history, stretch=1)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)


def get_multiple_directories(parent=None, caption: str = "Select Directories",
                             directory: str = "") -> List[str]:
    """
    Open a file dialog that allows selecting multiple directories.

    Uses a QFileDialog with custom configuration to enable multi-selection
    of directories using Ctrl+click or Shift+click.
    """
    from PyQt6.QtWidgets import QTreeView, QListView

    dialog = QFileDialog(parent, caption, directory)
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

    # Find the tree view and list view inside the dialog and enable multi-selection
    for view in dialog.findChildren(QTreeView):
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    for view in dialog.findChildren(QListView):
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    if dialog.exec():
        return dialog.selectedFiles()
    return []


class MultiDirectoryDialog(QDialog):
    """
    Dialog for selecting multiple directories.

    Supports:
    - Multi-select with Shift+click or Ctrl+click in the file browser
    - Adding directories one at a time
    - Removing selected directories from the list
    """

    def __init__(self, parent=None, initial_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Select Data Directories")
        self.setMinimumSize(600, 500)

        self._initial_dir = initial_dir
        self._directories: List[str] = []

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Instructions
        instructions = QLabel(
            "Select directories containing TDMS data files.\n"
            "Use Ctrl+click or Shift+click to select multiple directories at once."
        )
        instructions.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; font-size: 10pt;")
        layout.addWidget(instructions)

        # Scan subfolders checkbox
        self.chk_recursive = QCheckBox("Scan subfolders")
        self.chk_recursive.setChecked(True)
        self.chk_recursive.setToolTip("When checked, TDMS files in subdirectories will also be loaded")
        layout.addWidget(self.chk_recursive)

        # Directory list
        list_group = QGroupBox("Selected Directories")
        list_layout = QVBoxLayout(list_group)

        self.dir_list = QListWidget()
        self.dir_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.dir_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {DarkTheme.BACKGROUND};
                border: 1px solid {DarkTheme.BORDER};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 6px;
                border-bottom: 1px solid {DarkTheme.BORDER};
            }}
            QListWidget::item:selected {{
                background-color: {DarkTheme.SELECTION};
            }}
        """)
        list_layout.addWidget(self.dir_list)

        # Buttons for list management
        btn_layout = QHBoxLayout()

        self.btn_browse = QPushButton("Browse... (Multi-select)")
        self.btn_browse.setIcon(Icons.folder_open())
        self.btn_browse.setToolTip("Open browser to select multiple directories (Ctrl/Shift+click)")
        self.btn_browse.clicked.connect(self._browse_directories)
        btn_layout.addWidget(self.btn_browse)

        self.btn_add = QPushButton("Add Single...")
        self.btn_add.setIcon(Icons.add())
        self.btn_add.setToolTip("Add a single directory")
        self.btn_add.clicked.connect(self._add_single_directory)
        btn_layout.addWidget(self.btn_add)

        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setIcon(Icons.delete())
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_layout.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._clear_all)
        btn_layout.addWidget(self.btn_clear)

        btn_layout.addStretch()
        list_layout.addLayout(btn_layout)

        layout.addWidget(list_group)

        # Status label
        self.lbl_status = QLabel("No directories selected")
        self.lbl_status.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
        layout.addWidget(self.lbl_status)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Disable OK until at least one directory is added
        self.btn_ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_ok.setEnabled(False)

        layout.addWidget(buttons)

    def _browse_directories(self):
        """Open multi-select directory browser."""
        directories = get_multiple_directories(
            self, "Select Data Directories", self._initial_dir
        )
        for directory in directories:
            if directory and directory not in self._directories:
                self._directories.append(directory)
                self._initial_dir = str(Path(directory).parent)

                item = QListWidgetItem(directory)
                item.setToolTip(directory)
                self.dir_list.addItem(item)

        self._update_status()

    def _add_single_directory(self):
        """Add a single directory to the list."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Data Directory",
            self._initial_dir
        )
        if directory and directory not in self._directories:
            self._directories.append(directory)
            self._initial_dir = str(Path(directory).parent)

            item = QListWidgetItem(directory)
            item.setToolTip(directory)
            self.dir_list.addItem(item)

            self._update_status()

    def _remove_selected(self):
        """Remove selected directories from the list."""
        for item in self.dir_list.selectedItems():
            directory = item.text()
            if directory in self._directories:
                self._directories.remove(directory)
            self.dir_list.takeItem(self.dir_list.row(item))

        self._update_status()

    def _clear_all(self):
        """Clear all directories."""
        self._directories.clear()
        self.dir_list.clear()
        self._update_status()

    def _update_status(self):
        """Update status label and OK button state."""
        count = len(self._directories)
        if count == 0:
            self.lbl_status.setText("No directories selected")
            self.btn_ok.setEnabled(False)
        elif count == 1:
            self.lbl_status.setText("1 directory selected")
            self.btn_ok.setEnabled(True)
        else:
            self.lbl_status.setText(f"{count} directories selected")
            self.btn_ok.setEnabled(True)

    def get_directories(self) -> List[str]:
        """Get the list of selected directories."""
        return self._directories.copy()

    @property
    def recursive(self) -> bool:
        """Whether to scan subdirectories for TDMS files."""
        return self.chk_recursive.isChecked()
