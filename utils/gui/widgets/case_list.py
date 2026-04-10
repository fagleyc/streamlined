"""
Case List Widget
================

Widget for displaying and managing test cases.
"""

from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFrame, QMenu, QCheckBox, QColorDialog,
    QAbstractItemView, QStyledItemDelegate, QStyle
)
from PyQt6.QtCore import pyqtSignal, Qt, QSize, QRect, QPoint
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QIcon, QPixmap

from ..models.case import TestCase
from ..utils.themes import DarkTheme
from ..utils.icons import Icons


class ColorButton(QPushButton):
    """Button that displays and allows selecting a color."""

    color_changed = pyqtSignal(str)

    def __init__(self, color: str = "#1f77b4", parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(24, 24)
        self.setToolTip("Click to change color")
        self.clicked.connect(self._pick_color)
        self._update_style()

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, value: str):
        self._color = value
        self._update_style()

    def _update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color};
                border: 2px solid {DarkTheme.BORDER};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border-color: {DarkTheme.ACCENT};
            }}
        """)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._color), self)
        if color.isValid():
            self._color = color.name()
            self._update_style()
            self.color_changed.emit(self._color)


class CaseListItem(QWidget):
    """
    Custom widget for displaying a test case in the list.
    """

    visibility_changed = pyqtSignal(str, bool)
    color_changed = pyqtSignal(str, str)
    delete_requested = pyqtSignal(str)

    def __init__(self, case: TestCase, parent=None):
        super().__init__(parent)
        self.case = case
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Visibility checkbox
        self.chk_visible = QCheckBox()
        self.chk_visible.setChecked(self.case.visible)
        self.chk_visible.setToolTip("Show/hide in plots")
        self.chk_visible.stateChanged.connect(self._on_visibility_changed)
        layout.addWidget(self.chk_visible)

        # Color button
        self.btn_color = ColorButton(self.case.color)
        self.btn_color.color_changed.connect(self._on_color_changed)
        layout.addWidget(self.btn_color)

        # Case info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        self.lbl_name = QLabel(self.case.name)
        self.lbl_name.setStyleSheet(f"font-weight: bold; color: {DarkTheme.TEXT_PRIMARY};")
        info_layout.addWidget(self.lbl_name)

        self.lbl_info = QLabel(self.case.description)
        self.lbl_info.setStyleSheet(f"font-size: 9pt; color: {DarkTheme.TEXT_SECONDARY};")
        info_layout.addWidget(self.lbl_info)

        layout.addLayout(info_layout)
        layout.addStretch()

        # Right side: geometry label + point count
        right_layout = QVBoxLayout()
        right_layout.setSpacing(1)

        self._geometry_name = getattr(self.case, 'geometry_name', 'Default')
        self._calibration_name = getattr(self.case, 'calibration_name', '')
        # Combined label: geometry / calibration
        label_text = self._geometry_name
        if self._calibration_name:
            label_text += f" / {self._calibration_name}"
        self.lbl_geometry = QLabel(label_text)
        self.lbl_geometry.setStyleSheet(
            f"font-size: 8pt; color: {DarkTheme.ACCENT}; font-style: italic;")
        self.lbl_geometry.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(self.lbl_geometry)

        if self.case.has_data:
            self.lbl_points = QLabel(f"{self.case.n_points} pts")
            self.lbl_points.setStyleSheet(f"font-size: 8pt; color: {DarkTheme.TEXT_SECONDARY};")
            self.lbl_points.setAlignment(Qt.AlignmentFlag.AlignRight)
            right_layout.addWidget(self.lbl_points)

        layout.addLayout(right_layout)

    def _update_label(self):
        """Update the combined geometry/calibration label."""
        label = self._geometry_name
        if self._calibration_name:
            label += f" / {self._calibration_name}"
        self.lbl_geometry.setText(label)

    def set_geometry_name(self, name: str):
        """Update the displayed geometry name."""
        self._geometry_name = name
        self._update_label()

    def set_calibration_name(self, name: str):
        """Update the displayed calibration name."""
        self._calibration_name = name
        self._update_label()

    def _on_visibility_changed(self, state):
        visible = state == Qt.CheckState.Checked.value
        self.case.visible = visible
        self.visibility_changed.emit(self.case.id, visible)

    def _on_color_changed(self, color: str):
        self.case.color = color
        self.color_changed.emit(self.case.id, color)

    def update_from_case(self):
        """Update widget to reflect case changes."""
        self.chk_visible.setChecked(self.case.visible)
        self.btn_color.color = self.case.color
        self.lbl_name.setText(self.case.name)
        self.lbl_info.setText(self.case.description)
        self._geometry_name = getattr(self.case, 'geometry_name', 'Default')
        self._calibration_name = getattr(self.case, 'calibration_name', '')
        self._update_label()


class CaseListWidget(QWidget):
    """
    Widget for managing test cases.

    Signals
    -------
    case_selected : pyqtSignal(str)
        Emitted when a case is selected (case ID)
    case_visibility_changed : pyqtSignal(str, bool)
        Emitted when case visibility changes
    case_color_changed : pyqtSignal(str, str)
        Emitted when case color changes
    case_deleted : pyqtSignal(str)
        Emitted when a case is deleted
    add_requested : pyqtSignal
        Emitted when add button is clicked
    """

    case_selected = pyqtSignal(str)
    case_visibility_changed = pyqtSignal(str, bool)
    case_color_changed = pyqtSignal(str, str)
    case_deleted = pyqtSignal(str)
    add_requested = pyqtSignal()
    geometry_assigned = pyqtSignal(str, str)  # (case_id, geometry_name)
    calibration_assigned = pyqtSignal(str, str)  # (case_id, calibration_name)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._case_widgets = {}  # case_id -> CaseListItem
        self._geometry_names = ['Default']  # Available geometry names
        self._calibration_names = []  # Available calibration names
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHTER};
                border: none;
                border-bottom: 1px solid {DarkTheme.BORDER};
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("Test Cases")
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Add button
        self.btn_add = QPushButton()
        self.btn_add.setIcon(Icons.add())
        self.btn_add.setToolTip("Add test case")
        self.btn_add.setFixedSize(28, 28)
        self.btn_add.clicked.connect(self.add_requested.emit)
        header_layout.addWidget(self.btn_add)

        # Delete button
        self.btn_delete = QPushButton()
        self.btn_delete.setIcon(Icons.delete())
        self.btn_delete.setToolTip("Delete selected case")
        self.btn_delete.setFixedSize(28, 28)
        self.btn_delete.clicked.connect(self._delete_selected)
        header_layout.addWidget(self.btn_delete)

        layout.addWidget(header)

        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setSpacing(2)
        self.list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {DarkTheme.BACKGROUND_LIGHT};
                border: none;
            }}
            QListWidget::item {{
                background-color: {DarkTheme.SURFACE};
                border-radius: 4px;
                margin: 2px 4px;
            }}
            QListWidget::item:selected {{
                background-color: {DarkTheme.SELECTION};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {DarkTheme.HOVER};
            }}
        """)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.list_widget)

        # Footer with quick actions
        footer = QFrame()
        footer.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkTheme.BACKGROUND_LIGHTER};
                border: none;
                border-top: 1px solid {DarkTheme.BORDER};
            }}
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(8, 4, 8, 4)

        self.btn_show_all = QPushButton("Show All")
        self.btn_show_all.setToolTip("Show all cases")
        self.btn_show_all.clicked.connect(self._show_all)
        footer_layout.addWidget(self.btn_show_all)

        self.btn_hide_all = QPushButton("Hide All")
        self.btn_hide_all.setToolTip("Hide all cases")
        self.btn_hide_all.clicked.connect(self._hide_all)
        footer_layout.addWidget(self.btn_hide_all)

        footer_layout.addStretch()

        self.lbl_count = QLabel("0 cases")
        self.lbl_count.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY};")
        footer_layout.addWidget(self.lbl_count)

        layout.addWidget(footer)

    def add_case(self, case: TestCase):
        """Add a test case to the list."""
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, case.id)
        item.setSizeHint(QSize(0, 60))

        case_widget = CaseListItem(case)
        case_widget.visibility_changed.connect(self.case_visibility_changed.emit)
        case_widget.color_changed.connect(self.case_color_changed.emit)

        self._case_widgets[case.id] = case_widget

        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, case_widget)

        self._update_count()

    def remove_case(self, case_id: str):
        """Remove a test case from the list."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == case_id:
                self.list_widget.takeItem(i)
                if case_id in self._case_widgets:
                    del self._case_widgets[case_id]
                break

        self._update_count()

    def update_case(self, case_id: str, case: TestCase):
        """Update a case in the list."""
        if case_id in self._case_widgets:
            self._case_widgets[case_id].case = case
            self._case_widgets[case_id].update_from_case()

    def clear(self):
        """Clear all cases."""
        self.list_widget.clear()
        self._case_widgets.clear()
        self._update_count()

    def get_selected_case_id(self) -> Optional[str]:
        """Get the ID of the currently selected case."""
        items = self.list_widget.selectedItems()
        if items:
            return items[0].data(Qt.ItemDataRole.UserRole)
        return None

    def _on_item_clicked(self, item: QListWidgetItem):
        case_id = item.data(Qt.ItemDataRole.UserRole)
        self.case_selected.emit(case_id)

    def _delete_selected(self):
        case_id = self.get_selected_case_id()
        if case_id:
            self.case_deleted.emit(case_id)

    def _show_all(self):
        for case_id, widget in self._case_widgets.items():
            widget.chk_visible.setChecked(True)

    def _hide_all(self):
        for case_id, widget in self._case_widgets.items():
            widget.chk_visible.setChecked(False)

    def _update_count(self):
        count = self.list_widget.count()
        self.lbl_count.setText(f"{count} case{'s' if count != 1 else ''}")

    def set_geometry_names(self, names: list):
        """Update the available geometry names for the context menu."""
        self._geometry_names = list(names) if names else ['Default']

    def set_calibration_names(self, names: list):
        """Update the available calibration names for the context menu."""
        self._calibration_names = list(names) if names else []

    def update_case_geometry_label(self, case_id: str, geometry_name: str):
        """Update the geometry label displayed on a case item."""
        if case_id in self._case_widgets:
            self._case_widgets[case_id].set_geometry_name(geometry_name)

    def update_case_calibration_label(self, case_id: str, cal_name: str):
        """Update the calibration label displayed on a case item."""
        if case_id in self._case_widgets:
            self._case_widgets[case_id].set_calibration_name(cal_name)

    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return

        case_id = item.data(Qt.ItemDataRole.UserRole)

        menu = QMenu(self)
        menu.addAction("Show/Hide", lambda: self._toggle_visibility(case_id))
        menu.addAction("Change Color", lambda: self._change_color(case_id))

        # Assign Geometry submenu
        if len(self._geometry_names) > 0:
            geo_menu = menu.addMenu("Assign Geometry")
            for geo_name in self._geometry_names:
                action = geo_menu.addAction(
                    geo_name,
                    lambda checked=False, gn=geo_name: self.geometry_assigned.emit(case_id, gn)
                )
                widget = self._case_widgets.get(case_id)
                if widget and hasattr(widget, '_geometry_name') and widget._geometry_name == geo_name:
                    action.setCheckable(True)
                    action.setChecked(True)

        # Assign Calibration submenu
        if len(self._calibration_names) > 0:
            cal_menu = menu.addMenu("Assign Calibration")
            for cal_name in self._calibration_names:
                action = cal_menu.addAction(
                    cal_name,
                    lambda checked=False, cn=cal_name: self.calibration_assigned.emit(case_id, cn)
                )
                widget = self._case_widgets.get(case_id)
                if widget and hasattr(widget, '_calibration_name') and widget._calibration_name == cal_name:
                    action.setCheckable(True)
                    action.setChecked(True)

        menu.addSeparator()
        menu.addAction("Delete", lambda: self.case_deleted.emit(case_id))

        menu.exec(self.list_widget.mapToGlobal(pos))

    def _toggle_visibility(self, case_id: str):
        if case_id in self._case_widgets:
            widget = self._case_widgets[case_id]
            widget.chk_visible.setChecked(not widget.chk_visible.isChecked())

    def _change_color(self, case_id: str):
        if case_id in self._case_widgets:
            self._case_widgets[case_id].btn_color.click()
