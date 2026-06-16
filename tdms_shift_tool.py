"""
TDMS Alpha/Beta Shift Tool
==========================

A small standalone utility that loads wind-tunnel TDMS files, applies a
constant shift to the Alpha and/or Beta channels, and writes the files
back out.  All other groups, channels, properties, and waveform timing
are preserved exactly.

Features
--------
- Add individual files or whole folders (optionally recursive)
- Constant Alpha / Beta shift in degrees
- Overwrite in place, or write to a separate output folder
- Optional update of the Alpha_/Beta_ values encoded in the filename
- Optional filename suffix
- Background processing with progress and a log
- Dark theme matching the Streamlined application

Run with:
    python tdms_shift_tool.py

Author: C. Fagley
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from nptdms import (TdmsFile, TdmsWriter, ChannelObject,
                        RootObject, GroupObject)
    NPTDMS_AVAILABLE = True
except ImportError:
    NPTDMS_AVAILABLE = False

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup, QLineEdit,
    QFileDialog, QProgressBar, QTextEdit, QMessageBox, QAbstractItemView,
)


APP_NAME = "TDMS Alpha/Beta Shift Tool"
APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Dark theme (compact subset matching Streamlined)
# ---------------------------------------------------------------------------

_BACKGROUND = "#1e1e1e"
_BACKGROUND_LIGHT = "#252526"
_BACKGROUND_LIGHTER = "#2d2d30"
_SURFACE = "#333333"
_TEXT_PRIMARY = "#e0e0e0"
_TEXT_SECONDARY = "#a0a0a0"
_TEXT_DISABLED = "#606060"
_ACCENT = "#0078d4"
_ACCENT_LIGHT = "#3399ff"
_ACCENT_DARK = "#005a9e"
_SUCCESS = "#4caf50"
_ERROR = "#f44336"
_BORDER = "#3f3f46"
_SELECTION = "#264f78"
_HOVER = "#3a3a3c"


def _stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {_BACKGROUND};
        color: {_TEXT_PRIMARY};
        font-family: "Segoe UI", "SF Pro Display", sans-serif;
        font-size: 10pt;
    }}
    QGroupBox {{
        background-color: {_BACKGROUND_LIGHT};
        border: 1px solid {_BORDER};
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 10px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
        color: {_TEXT_PRIMARY};
    }}
    QLabel {{ background-color: transparent; border: none; color: {_TEXT_PRIMARY}; }}
    QPushButton {{
        background-color: {_SURFACE};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        padding: 6px 16px;
        color: {_TEXT_PRIMARY};
        min-height: 22px;
    }}
    QPushButton:hover {{ background-color: {_HOVER}; border-color: {_ACCENT}; }}
    QPushButton:pressed {{ background-color: {_ACCENT_DARK}; }}
    QPushButton:disabled {{
        background-color: {_BACKGROUND_LIGHTER};
        color: {_TEXT_DISABLED}; border-color: {_BORDER};
    }}
    QPushButton[primary="true"] {{
        background-color: {_ACCENT}; border-color: {_ACCENT}; color: white;
        font-weight: bold;
    }}
    QPushButton[primary="true"]:hover {{ background-color: {_ACCENT_LIGHT}; }}
    QPushButton[primary="true"]:disabled {{
        background-color: {_BACKGROUND_LIGHTER}; color: {_TEXT_DISABLED};
    }}
    QLineEdit {{
        background-color: {_SURFACE};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        padding: 6px 10px;
        color: {_TEXT_PRIMARY};
        selection-background-color: {_ACCENT};
    }}
    QLineEdit:focus {{ border-color: {_ACCENT}; }}
    QLineEdit:disabled {{
        background-color: {_BACKGROUND_LIGHTER}; color: {_TEXT_DISABLED};
    }}
    QDoubleSpinBox {{
        background-color: {_SURFACE};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        color: {_TEXT_PRIMARY};
        min-width: 90px;
    }}
    QDoubleSpinBox:focus {{ border-color: {_ACCENT}; }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: {_BACKGROUND_LIGHTER}; border: none; width: 16px;
    }}
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {_HOVER};
    }}
    QCheckBox, QRadioButton {{ spacing: 8px; color: {_TEXT_PRIMARY}; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px; height: 18px;
        border: 1px solid {_BORDER};
        background-color: {_SURFACE};
    }}
    QCheckBox::indicator {{ border-radius: 3px; }}
    QRadioButton::indicator {{ border-radius: 9px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {_ACCENT};
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {_ACCENT}; border-color: {_ACCENT};
    }}
    QListWidget, QTextEdit {{
        background-color: {_BACKGROUND_LIGHT};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        padding: 4px;
    }}
    QListWidget::item {{ padding: 4px; border-radius: 4px; }}
    QListWidget::item:selected {{ background-color: {_SELECTION}; }}
    QListWidget::item:hover:!selected {{ background-color: {_HOVER}; }}
    QProgressBar {{
        background-color: {_SURFACE};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        text-align: center;
        color: {_TEXT_PRIMARY};
        height: 20px;
    }}
    QProgressBar::chunk {{ background-color: {_ACCENT}; border-radius: 3px; }}
    QScrollBar:vertical {{ background-color: {_BACKGROUND}; width: 12px; margin: 0; }}
    QScrollBar::handle:vertical {{
        background-color: {_SURFACE}; border-radius: 4px;
        min-height: 30px; margin: 2px;
    }}
    QScrollBar::handle:vertical:hover {{ background-color: {_HOVER}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    """


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

# Channel names treated as angle channels (case-insensitive match).
_ALPHA_NAMES = {'alpha'}
_BETA_NAMES = {'beta'}


def _detect_angle_channels(tdms: 'TdmsFile') -> Tuple[List, List]:
    """Return (alpha_refs, beta_refs) as lists of (group_name, channel_name)."""
    alpha_refs, beta_refs = [], []
    for group in tdms.groups():
        for ch in group.channels():
            lname = ch.name.lower()
            if lname in _ALPHA_NAMES:
                alpha_refs.append((group.name, ch.name))
            elif lname in _BETA_NAMES:
                beta_refs.append((group.name, ch.name))
    return alpha_refs, beta_refs


def _update_filename_angles(name: str, alpha_shift: float,
                            beta_shift: float) -> str:
    """
    Update the Alpha_X / Beta_Y values encoded in a filename by adding
    the shifts.  Returns the new stem (without extension).  If no
    Alpha_/Beta_ token is found, the name is returned unchanged.
    """
    def _bump(match, shift):
        val = float(match.group(1))
        new_val = val + shift
        # Format: keep one decimal place like the originals (e.g. -2.0)
        return f"{new_val:.1f}"

    new_name = name

    # Alpha
    m = re.search(r'(Alpha[_\s]*)(-?\d+\.?\d*)', new_name, re.IGNORECASE)
    if m and alpha_shift != 0.0:
        val = float(m.group(2))
        new_name = (new_name[:m.start(2)]
                    + f"{val + alpha_shift:.1f}"
                    + new_name[m.end(2):])

    # Beta (re-search because indices may have shifted)
    m = re.search(r'(Beta[_\s]*)(-?\d+\.?\d*)', new_name, re.IGNORECASE)
    if m and beta_shift != 0.0:
        val = float(m.group(2))
        new_name = (new_name[:m.start(2)]
                    + f"{val + beta_shift:.1f}"
                    + new_name[m.end(2):])

    return new_name


def process_file(src_path: Path, dst_path: Path,
                 alpha_shift: float, beta_shift: float) -> Tuple[int, int]:
    """
    Read src_path, shift its Alpha/Beta channels, write to dst_path.

    Returns (n_alpha_channels_shifted, n_beta_channels_shifted).
    Writes atomically by going through a temp file in the destination
    directory, then os.replace into place.
    """
    tdms = TdmsFile.read(str(src_path))
    alpha_refs, beta_refs = _detect_angle_channels(tdms)
    alpha_set = set(alpha_refs)
    beta_set = set(beta_refs)

    # Build the object list for the writer
    root = RootObject(properties=dict(tdms.properties))
    objects = [root]
    for group in tdms.groups():
        objects.append(GroupObject(group.name,
                                   properties=dict(group.properties)))
        for ch in group.channels():
            data = np.asarray(ch[:])
            ref = (group.name, ch.name)
            if ref in alpha_set and alpha_shift != 0.0:
                data = data + alpha_shift
            elif ref in beta_set and beta_shift != 0.0:
                data = data + beta_shift
            objects.append(ChannelObject(
                group.name, ch.name, data,
                properties=dict(ch.properties)))

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: temp file in the same directory, then replace
    fd, tmp_name = tempfile.mkstemp(
        suffix='.tdms', dir=str(dst_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with TdmsWriter(str(tmp_path)) as writer:
            writer.write_segment(objects)
        os.replace(str(tmp_path), str(dst_path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return len(alpha_refs), len(beta_refs)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class ProcessWorker(QObject):
    progress = pyqtSignal(int, int)          # done, total
    message = pyqtSignal(str, str)           # text, level ('info'/'ok'/'error')
    finished = pyqtSignal(int, int)          # n_ok, n_failed

    def __init__(self, files: List[Path], alpha_shift: float,
                 beta_shift: float, overwrite: bool,
                 out_dir: Optional[Path], update_filename: bool,
                 suffix: str):
        super().__init__()
        self.files = files
        self.alpha_shift = alpha_shift
        self.beta_shift = beta_shift
        self.overwrite = overwrite
        self.out_dir = out_dir
        self.update_filename = update_filename
        self.suffix = suffix
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self.files)
        n_ok = 0
        n_failed = 0
        for i, src in enumerate(self.files):
            if self._cancelled:
                self.message.emit("Cancelled by user.", 'error')
                break
            try:
                stem = src.stem
                if self.update_filename:
                    stem = _update_filename_angles(
                        stem, self.alpha_shift, self.beta_shift)
                if self.suffix:
                    stem = stem + self.suffix

                if self.overwrite and not self.suffix \
                        and not self.update_filename:
                    dst = src
                elif self.overwrite:
                    dst = src.parent / f"{stem}.tdms"
                else:
                    # New folder, mirror just the filename (flat)
                    dst = (self.out_dir or src.parent) / f"{stem}.tdms"

                n_a, n_b = process_file(
                    src, dst, self.alpha_shift, self.beta_shift)

                rel = dst.name
                detail = []
                if n_a:
                    detail.append(f"{n_a} alpha")
                if n_b:
                    detail.append(f"{n_b} beta")
                ch_info = (", ".join(detail) + " ch" if detail
                           else "no alpha/beta channels found")
                self.message.emit(
                    f"OK  {src.name}  ->  {rel}   ({ch_info})", 'ok')
                n_ok += 1
            except Exception as e:
                traceback.print_exc()
                self.message.emit(
                    f"FAIL  {src.name}: {type(e).__name__}: {e}", 'error')
                n_failed += 1
            self.progress.emit(i + 1, total)

        self.finished.emit(n_ok, n_failed)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ShiftToolWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")
        self.resize(760, 720)
        self._files: List[Path] = []
        self._thread: Optional[QThread] = None
        self._worker: Optional[ProcessWorker] = None
        self._setup_ui()
        if not NPTDMS_AVAILABLE:
            self._log("nptdms is not installed. Run: pip install nptdms",
                      'error')
            self.btn_process.setEnabled(False)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 14)

        title = QLabel(APP_NAME)
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        layout.addWidget(title)
        subtitle = QLabel(
            "Apply a constant shift to the Alpha / Beta channels of "
            "wind-tunnel TDMS files.")
        subtitle.setStyleSheet(f"color: {_TEXT_SECONDARY};")
        layout.addWidget(subtitle)

        # --- Input files ---
        in_group = QGroupBox("Input Files")
        in_layout = QVBoxLayout(in_group)
        self.list_files = QListWidget()
        self.list_files.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_files.setMinimumHeight(150)
        in_layout.addWidget(self.list_files)

        btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files...")
        self.btn_add_files.clicked.connect(self._add_files)
        btn_row.addWidget(self.btn_add_files)
        self.btn_add_folder = QPushButton("Add Folder...")
        self.btn_add_folder.clicked.connect(self._add_folder)
        btn_row.addWidget(self.btn_add_folder)
        self.chk_recursive = QCheckBox("Recursive")
        self.chk_recursive.setChecked(True)
        self.chk_recursive.setToolTip(
            "Include TDMS files in subfolders when adding a folder")
        btn_row.addWidget(self.chk_recursive)
        btn_row.addStretch(1)
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.btn_remove)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_files)
        btn_row.addWidget(self.btn_clear)
        in_layout.addLayout(btn_row)

        self.lbl_count = QLabel("0 files")
        self.lbl_count.setStyleSheet(f"color: {_TEXT_SECONDARY};")
        in_layout.addWidget(self.lbl_count)
        layout.addWidget(in_group)

        # --- Shift ---
        shift_group = QGroupBox("Shift")
        shift_layout = QGridLayout(shift_group)
        shift_layout.addWidget(QLabel("Alpha shift (deg):"), 0, 0)
        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(-180.0, 180.0)
        self.spn_alpha.setDecimals(3)
        self.spn_alpha.setSingleStep(0.1)
        shift_layout.addWidget(self.spn_alpha, 0, 1)
        shift_layout.addWidget(QLabel("Beta shift (deg):"), 0, 2)
        self.spn_beta = QDoubleSpinBox()
        self.spn_beta.setRange(-180.0, 180.0)
        self.spn_beta.setDecimals(3)
        self.spn_beta.setSingleStep(0.1)
        shift_layout.addWidget(self.spn_beta, 0, 3)
        shift_layout.setColumnStretch(4, 1)
        layout.addWidget(shift_group)

        # --- Output ---
        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout(out_group)

        self.radio_group = QButtonGroup(self)
        self.rb_overwrite = QRadioButton("Overwrite original files")
        self.rb_overwrite.setChecked(True)
        self.rb_overwrite.toggled.connect(self._update_output_state)
        self.radio_group.addButton(self.rb_overwrite)
        out_layout.addWidget(self.rb_overwrite)

        self.rb_newfolder = QRadioButton("Write to a new folder:")
        self.radio_group.addButton(self.rb_newfolder)
        nf_row = QHBoxLayout()
        nf_row.addWidget(self.rb_newfolder)
        self.txt_outdir = QLineEdit()
        self.txt_outdir.setPlaceholderText("Output folder...")
        self.txt_outdir.setEnabled(False)
        nf_row.addWidget(self.txt_outdir, 1)
        self.btn_browse_out = QPushButton("Browse...")
        self.btn_browse_out.setEnabled(False)
        self.btn_browse_out.clicked.connect(self._browse_outdir)
        nf_row.addWidget(self.btn_browse_out)
        out_layout.addLayout(nf_row)

        self.chk_update_name = QCheckBox(
            "Update Alpha_/Beta_ values in the filename")
        self.chk_update_name.setToolTip(
            "Parse the Alpha_X and Beta_Y tokens in the filename and "
            "add the shift to them")
        out_layout.addWidget(self.chk_update_name)

        suf_row = QHBoxLayout()
        suf_row.addWidget(QLabel("Filename suffix (optional):"))
        self.txt_suffix = QLineEdit()
        self.txt_suffix.setPlaceholderText("e.g. _shifted")
        self.txt_suffix.setMaximumWidth(220)
        suf_row.addWidget(self.txt_suffix)
        suf_row.addStretch(1)
        out_layout.addLayout(suf_row)

        warn = QLabel(
            "Note: overwriting with no suffix and no filename update "
            "replaces the source files in place.")
        warn.setStyleSheet(f"color: {_TEXT_SECONDARY}; font-size: 9pt;")
        warn.setWordWrap(True)
        out_layout.addWidget(warn)
        layout.addWidget(out_group)

        # --- Progress + log ---
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(130)
        self.log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log, 1)

        # --- Action buttons ---
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.btn_process = QPushButton("Process Files")
        self.btn_process.setProperty("primary", True)
        self.btn_process.clicked.connect(self._process)
        action_row.addWidget(self.btn_process)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        action_row.addWidget(self.btn_close)
        layout.addLayout(action_row)

    # ---- file management ----

    def _add_paths(self, paths: List[Path]):
        existing = set(self._files)
        added = 0
        for p in paths:
            if p.suffix.lower() == '.tdms' and p not in existing:
                self._files.append(p)
                existing.add(p)
                added += 1
        self._refresh_list()
        if added:
            self._log(f"Added {added} file(s).", 'info')

    def _refresh_list(self):
        self.list_files.clear()
        for p in self._files:
            self.list_files.addItem(QListWidgetItem(str(p)))
        n = len(self._files)
        self.lbl_count.setText(f"{n} file{'s' if n != 1 else ''}")

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select TDMS files", "",
            "TDMS Files (*.tdms);;All Files (*.*)")
        if files:
            self._add_paths([Path(f) for f in files])

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder of TDMS files")
        if not folder:
            return
        base = Path(folder)
        if self.chk_recursive.isChecked():
            found = sorted(base.rglob("*.tdms"))
        else:
            found = sorted(base.glob("*.tdms"))
        if not found:
            self._log(f"No .tdms files found in {folder}.", 'error')
            return
        self._add_paths(found)

    def _remove_selected(self):
        rows = sorted((self.list_files.row(i)
                       for i in self.list_files.selectedItems()),
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._files):
                del self._files[r]
        self._refresh_list()

    def _clear_files(self):
        self._files.clear()
        self._refresh_list()

    # ---- output state ----

    def _update_output_state(self):
        new_folder = self.rb_newfolder.isChecked()
        self.txt_outdir.setEnabled(new_folder)
        self.btn_browse_out.setEnabled(new_folder)

    def _browse_outdir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder")
        if folder:
            self.txt_outdir.setText(folder)

    # ---- logging ----

    def _log(self, text: str, level: str = 'info'):
        color = {
            'info': _TEXT_PRIMARY,
            'ok': _SUCCESS,
            'error': _ERROR,
        }.get(level, _TEXT_PRIMARY)
        self.log.append(f'<span style="color:{color};">{text}</span>')

    # ---- processing ----

    def _process(self):
        if not self._files:
            QMessageBox.information(
                self, "No Files",
                "Add one or more TDMS files first.")
            return

        alpha_shift = self.spn_alpha.value()
        beta_shift = self.spn_beta.value()
        if alpha_shift == 0.0 and beta_shift == 0.0:
            QMessageBox.information(
                self, "No Shift",
                "Both Alpha and Beta shifts are zero. Set a non-zero "
                "shift to change the data.")
            return

        overwrite = self.rb_overwrite.isChecked()
        out_dir = None
        if not overwrite:
            out_text = self.txt_outdir.text().strip()
            if not out_text:
                QMessageBox.warning(
                    self, "No Output Folder",
                    "Choose an output folder, or select "
                    "'Overwrite original files'.")
                return
            out_dir = Path(out_text)

        update_name = self.chk_update_name.isChecked()
        suffix = self.txt_suffix.text().strip()

        # Confirm in-place overwrite
        if overwrite and not suffix and not update_name:
            reply = QMessageBox.question(
                self, "Confirm Overwrite",
                f"This will overwrite {len(self._files)} original file(s) "
                "in place. Continue?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.log.clear()
        self._log(
            f"Processing {len(self._files)} file(s)  "
            f"(alpha {alpha_shift:+.3f} deg, beta {beta_shift:+.3f} deg)...",
            'info')
        self.progress.setValue(0)
        self._set_busy(True)

        self._thread = QThread()
        self._worker = ProcessWorker(
            files=list(self._files),
            alpha_shift=alpha_shift,
            beta_shift=beta_shift,
            overwrite=overwrite,
            out_dir=out_dir,
            update_filename=update_name,
            suffix=suffix)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.message.connect(self._log)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_progress(self, done: int, total: int):
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _on_finished(self, n_ok: int, n_failed: int):
        self._log(
            f"Done. {n_ok} succeeded, {n_failed} failed.",
            'ok' if n_failed == 0 else 'error')
        self._set_busy(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _set_busy(self, busy: bool):
        for w in (self.btn_process, self.btn_add_files, self.btn_add_folder,
                  self.btn_remove, self.btn_clear, self.spn_alpha,
                  self.spn_beta, self.rb_overwrite, self.rb_newfolder):
            w.setEnabled(not busy)
        if not busy:
            self._update_output_state()

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(_stylesheet())
    win = ShiftToolWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
