"""
Dialogs
=======

Dialog windows for geometry settings, calibration, etc.
"""

from pathlib import Path
from typing import List

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QDoubleSpinBox, QComboBox, QPushButton,
    QDialogButtonBox, QFrame, QTextBrowser, QTabWidget, QWidget,
    QListWidget, QListWidgetItem, QFileDialog, QAbstractItemView,
    QTreeView, QListView, QCheckBox
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
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {__app_name__}")
        self.setFixedSize(500, 600)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel(__app_name__)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # Version
        version = QLabel(f"Version {__version__}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
        layout.addWidget(version)

        # Description
        desc = QTextBrowser()
        desc.setOpenExternalLinks(True)
        desc.setHtml(f"""
        <div style="color: {DarkTheme.TEXT_PRIMARY}; font-family: 'Segoe UI'; line-height: 1.5;">
            <p>A professional application for wind tunnel data reduction and aerodynamic
            coefficient analysis.</p>

            <h3>Features</h3>
            <ul>
                <li>Load and process TDMS data files</li>
                <li>Apply force balance and pressure calibrations</li>
                <li>Compute aerodynamic coefficients with &plusmn;1&sigma; uncertainty</li>
                <li>Multiple geometry definitions with per-case assignment</li>
                <li>Interactive plotting with alpha/beta filtering and std dev shading</li>
                <li>Interactive save image dialog with theme, label, and trace customization</li>
                <li>Time history and FFT analysis with multi-case overlay</li>
                <li>Consolidated export to CSV/Excel/HDF5/MAT with extended data options</li>
            </ul>

            <h3>Changelog</h3>
            <p><b>v1.2.5</b></p>
            <ul>
                <li>Auto-detect thermocouple calibration vintage per sample:
                    new cal (0.1 V/&deg;C) vs old cal (0.1 V/&deg;F).
                    Optional <code>temp_cal_mode</code> override on
                    <code>calc_tunnel_conditions()</code> /
                    <code>reduce_raw()</code> for forced behavior</li>
                <li>New <code>raw</code> export group (MAT and HDF5):
                    per-point mean of pdiff, ptot, ttot, alpha, beta, and
                    BRF forces / moments (Fx, Fy, Fz, Mx, My, Mz) in the
                    selected output units, plus a <code>units</code> label</li>
                <li>User-friendly error dialogs replace silent failures:
                    pre-flight checks before processing flag missing
                    calibrations, missing data directories, no .tdms files,
                    and invalid geometry with actionable messages</li>
                <li>Calibration loading detects corrupt or wrong-format files
                    with a dedicated &ldquo;Invalid Calibration File&rdquo; dialog</li>
                <li>Export dialog blocks opening when no reduced cases exist
                    and surfaces write failures as a critical dialog rather
                    than a console traceback</li>
                <li>Configuration save / load surfaces JSON parse errors and
                    missing-file conditions as targeted dialogs</li>
                <li>Worker reports &ldquo;Processing Failed&rdquo; if every
                    configuration errors out, and notes partial-failure
                    counts in the success status message</li>
                <li>Cleaned up <code>requirements.txt</code>: dropped unused
                    sympy, moved pyinstaller to <code>requirements-dev.txt</code>;
                    added <code>INSTALL.md</code> with step-by-step setup</li>
            </ul>
            <p><b>v1.2.4</b></p>
            <ul>
                <li>Compressible isentropic tunnel conditions: dynamic pressure,
                    Mach, density, and velocity now computed from isentropic
                    pressure-ratio relations instead of incompressible Bernoulli</li>
                <li>Static temperature derived from stagnation temperature using
                    isentropic relation; density uses static P and static T</li>
                <li>Sutherland's law for dynamic viscosity replaces fixed constant</li>
                <li>New tunnel fields: P_static, T0, speed of sound available
                    in time history viewer</li>
                <li>Tare subtraction now removes only the DC mean of air-off,
                    preserving time-varying dynamics of air-on signals</li>
                <li>Moment balance support: balance config (Force/Moment)
                    now persists correctly and propagates through data reduction</li>
                <li>Calibration dialog pre-populates with current settings</li>
                <li>Balance element labels adapt to config: N1/N2/Y1/Y2 (Force)
                    or AftPitch/AftYaw/FwdPitch/FwdYaw (Moment)</li>
                <li>LabVIEW interface: standalone <code>labview_balance_cal.py</code>
                    for parsing .vol files and applying calibrations</li>
                <li>Span included in all export formats (Excel, HDF5, MAT)</li>
                <li>Help &rarr; Documentation menu opens README</li>
            </ul>
            <p><b>v1.2.3</b></p>
            <ul>
                <li>Multiple geometry definitions: define and manage named geometries
                    (e.g. &ldquo;Full-Span&rdquo;, &ldquo;Half-Span&rdquo;) with
                    independent MAC, span, area, MRC, and input units</li>
                <li>Per-case geometry assignment: right-click a case &rarr;
                    &ldquo;Assign Geometry&rdquo; to use a specific geometry definition,
                    then auto-reprocesses with the new reference values</li>
                <li>Reference span added: roll (C<sub>l</sub>) and yaw (C<sub>n</sub>)
                    moment coefficients now use span (b) for normalization;
                    pitching moment (C<sub>m</sub>) continues to use chord (MAC)</li>
                <li>Output units moved to Edit &rarr; Output Units (independent
                    of geometry definitions)</li>
                <li>Geometry definitions and per-case assignments are saved/loaded
                    in configuration files with backward compatibility</li>
            </ul>
            <p><b>v1.2.2</b></p>
            <ul>
                <li>Interactive save image dialog with live preview: customize
                    legend labels, line widths, marker sizes/types, trace colors,
                    axis labels, axis limits, export theme (White/Dark/Black/Transparent),
                    font sizes, and resolution before saving</li>
                <li>Save image settings are remembered between sessions</li>
                <li>Standard deviation shading on plots (&plusmn;1&sigma; bands
                    toggled via &ldquo;Show Std Dev&rdquo; checkbox)</li>
                <li>Table columns now sort numerically (negative values sort correctly)</li>
                <li>Consolidated export: single File &gt; Export dialog replaces
                    individual CSV/Excel/HDF5/MAT buttons with extended data options
                    for HDF5/MAT (raw data, reduced data, time-series, metadata)</li>
            </ul>
            <p><b>v1.2.0</b></p>
            <ul>
                <li>HDF5 and MAT exports now support optional unsteady (time-series)
                    data via the &ldquo;Include Unsteady&rdquo; checkbox</li>
                <li>HDF5 export restructured to per-case groups with
                    <code>averaged/</code> and <code>unsteady/</code> sub-groups</li>
                <li>Calibration and geometry metadata now included in all exports
                    (Excel, HDF5, MAT)</li>
                <li>Time history and FFT panels automatically overlay all visible
                    cases with color-coded legends</li>
            </ul>
            <p><b>v1.1.0</b></p>
            <ul>
                <li>Added &ldquo;Plot vs &beta;&rdquo; toggle to plot coefficients
                    with respect to sideslip angle (&beta;) on the x-axis</li>
                <li>MAT export now saves each configuration as a named MATLAB
                    struct with a <code>meta</code> sub-struct containing test
                    summary information</li>
                <li>Added Export to Excel in the File menu</li>
                <li>Default calibration type changed to Cubic</li>
            </ul>
            <p><b>v1.0.0</b></p>
            <ul>
                <li>Initial release</li>
                <li>TDMS data loading and multi-directory support</li>
                <li>Balance and pressure calibration</li>
                <li>BRF/WRF coordinate transforms and coefficient calculation</li>
                <li>Interactive plotting with alpha/beta/Mach/Re filtering</li>
                <li>Time history and FFT views</li>
                <li>Export to CSV, Excel, HDF5, and MAT</li>
                <li>Unit system support (IPS, FPS, MKS, CGS)</li>
            </ul>

            <h3>Credits</h3>
            <p>Based on MATLAB wind tunnel data reduction routines.</p>
            <p>Author: C. Fagley</p>

            <h3>Libraries</h3>
            <p>PyQt6 • NumPy • pyqtgraph • Matplotlib • pandas • nptdms</p>
        </div>
        """)
        desc.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border: 1px solid {DarkTheme.BORDER};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        layout.addWidget(desc)

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
