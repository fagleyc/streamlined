"""
Icon Resources
==============

SVG icons for the application using inline SVG data.
"""

from PyQt6.QtGui import QIcon, QPixmap, QPainter
from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtSvg import QSvgRenderer
from typing import Dict


class Icons:
    """
    Application icons using inline SVG data.
    All icons are designed for dark theme (light colored).
    """

    # Icon color for dark theme
    COLOR = "#e0e0e0"
    ACCENT = "#0078d4"

    # SVG Templates
    _FOLDER_OPEN = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M20 6h-8l-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/>
    </svg>'''

    _FILE = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6z"/>
    </svg>'''

    _ADD = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
    </svg>'''

    _REMOVE = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M19 13H5v-2h14v2z"/>
    </svg>'''

    _DELETE = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM8 9h8v10H8V9zm7.5-5l-1-1h-5l-1 1H5v2h14V4h-3.5z"/>
    </svg>'''

    _REFRESH = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
    </svg>'''

    _SETTINGS = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
    </svg>'''

    _CHART = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M3.5 18.49l6-6.01 4 4L22 6.92l-1.41-1.41-7.09 7.97-4-4L2 16.99z"/>
    </svg>'''

    _TABLE = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M3 3v18h18V3H3zm8 16H5v-6h6v6zm0-8H5V5h6v6zm8 8h-6v-6h6v6zm0-8h-6V5h6v6z"/>
    </svg>'''

    _SAVE = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M17 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm2 16H5V5h11.17L19 7.83V19zm-7-7c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3zM6 6h9v4H6z"/>
    </svg>'''

    _EXPORT = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M19 12v7H5v-7H3v7c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2v-7h-2zm-6 .67l2.59-2.58L17 11.5l-5 5-5-5 1.41-1.41L11 12.67V3h2v9.67z"/>
    </svg>'''

    _ZOOM_IN = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14zM12 10h-2v2H9v-2H7V9h2V7h1v2h2v1z"/>
    </svg>'''

    _ZOOM_OUT = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14zM7 9h5v1H7z"/>
    </svg>'''

    _HOME = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>
    </svg>'''

    _VISIBLE = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
    </svg>'''

    _HIDDEN = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M12 7c2.76 0 5 2.24 5 5 0 .65-.13 1.26-.36 1.83l2.92 2.92c1.51-1.26 2.7-2.89 3.43-4.75-1.73-4.39-6-7.5-11-7.5-1.4 0-2.74.25-3.98.7l2.16 2.16C10.74 7.13 11.35 7 12 7zM2 4.27l2.28 2.28.46.46C3.08 8.3 1.78 10.02 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l.42.42L19.73 22 21 20.73 3.27 3 2 4.27zM7.53 9.8l1.55 1.55c-.05.21-.08.43-.08.65 0 1.66 1.34 3 3 3 .22 0 .44-.03.65-.08l1.55 1.55c-.67.33-1.41.53-2.2.53-2.76 0-5-2.24-5-5 0-.79.2-1.53.53-2.2zm4.31-.78l3.15 3.15.02-.16c0-1.66-1.34-3-3-3l-.17.01z"/>
    </svg>'''

    _GRID = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M20 2H4c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM8 20H4v-4h4v4zm0-6H4v-4h4v4zm0-6H4V4h4v4zm6 12h-4v-4h4v4zm0-6h-4v-4h4v4zm0-6h-4V4h4v4zm6 12h-4v-4h4v4zm0-6h-4v-4h4v4zm0-6h-4V4h4v4z"/>
    </svg>'''

    _CALIBRATION = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M19.5 12c0-.23-.01-.45-.03-.68l1.86-1.41c.4-.3.51-.86.26-1.3l-1.87-3.23c-.25-.44-.79-.62-1.25-.42l-2.15.91c-.37-.26-.76-.49-1.17-.68l-.29-2.31c-.06-.5-.49-.88-.99-.88h-3.73c-.51 0-.94.38-1 .88l-.29 2.31c-.41.19-.8.42-1.17.68l-2.15-.91c-.46-.2-1-.02-1.25.42L2.41 8.62c-.25.44-.14.99.26 1.3l1.86 1.41c-.02.22-.03.44-.03.67s.01.45.03.68l-1.86 1.41c-.4.3-.51.86-.26 1.3l1.87 3.23c.25.44.79.62 1.25.42l2.15-.91c.37.26.76.49 1.17.68l.29 2.31c.06.5.49.88.99.88h3.73c.5 0 .93-.38.99-.88l.29-2.31c.41-.19.8-.42 1.17-.68l2.15.91c.46.2 1 .02 1.25-.42l1.87-3.23c.25-.44.14-.99-.26-1.3l-1.86-1.41c.03-.23.04-.45.04-.68zm-7.46 3.5c-1.93 0-3.5-1.57-3.5-3.5s1.57-3.5 3.5-3.5 3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5z"/>
    </svg>'''

    _PLAY = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M8 5v14l11-7z"/>
    </svg>'''

    _COMPARE = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M10 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h5v2h2V1h-2v2zm0 15H5l5-6v6zm9-15h-5v2h5v13l-5-6v9h5c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"/>
    </svg>'''

    _FILTER = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"/>
    </svg>'''

    _RULER = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M21 6H3c-1.1 0-2 .9-2 2v8c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 10H3V8h2v4h2V8h2v4h2V8h2v4h2V8h2v4h2V8h2v8z"/>
    </svg>'''

    _CUBE = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M21 16.5c0 .38-.21.71-.53.88l-7.9 4.44c-.16.12-.36.18-.57.18-.21 0-.41-.06-.57-.18l-7.9-4.44A.991.991 0 013 16.5v-9c0-.38.21-.71.53-.88l7.9-4.44c.16-.12.36-.18.57-.18.21 0 .41.06.57.18l7.9 4.44c.32.17.53.5.53.88v9zM12 4.15L6.04 7.5 12 10.85l5.96-3.35L12 4.15zM5 15.91l6 3.38v-6.71L5 9.21v6.7zm14 0v-6.7l-6 3.37v6.71l6-3.38z"/>
    </svg>'''

    _AIRCRAFT = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/>
    </svg>'''

    _TUNE = '''<svg viewBox="0 0 24 24" fill="{color}">
        <path d="M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z"/>
    </svg>'''

    @classmethod
    def _create_icon(cls, svg_template: str, color: str = None) -> QIcon:
        """Create a QIcon from SVG template."""
        if color is None:
            color = cls.COLOR

        svg_data = svg_template.format(color=color)
        svg_bytes = QByteArray(svg_data.encode('utf-8'))

        renderer = QSvgRenderer(svg_bytes)
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()

        return QIcon(pixmap)

    @classmethod
    def folder_open(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._FOLDER_OPEN, color)

    @classmethod
    def file(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._FILE, color)

    @classmethod
    def add(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._ADD, color)

    @classmethod
    def remove(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._REMOVE, color)

    @classmethod
    def delete(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._DELETE, color)

    @classmethod
    def refresh(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._REFRESH, color)

    @classmethod
    def settings(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._SETTINGS, color)

    @classmethod
    def chart(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._CHART, color)

    @classmethod
    def table(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._TABLE, color)

    @classmethod
    def save(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._SAVE, color)

    @classmethod
    def export(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._EXPORT, color)

    @classmethod
    def zoom_in(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._ZOOM_IN, color)

    @classmethod
    def zoom_out(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._ZOOM_OUT, color)

    @classmethod
    def home(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._HOME, color)

    @classmethod
    def visible(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._VISIBLE, color)

    @classmethod
    def hidden(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._HIDDEN, color)

    @classmethod
    def grid(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._GRID, color)

    @classmethod
    def calibration(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._CALIBRATION, color)

    @classmethod
    def play(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._PLAY, color)

    @classmethod
    def compare(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._COMPARE, color)

    @classmethod
    def filter(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._FILTER, color)

    @classmethod
    def ruler(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._RULER, color)

    @classmethod
    def cube(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._CUBE, color)

    @classmethod
    def aircraft(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._AIRCRAFT, color)

    @classmethod
    def tune(cls, color: str = None) -> QIcon:
        return cls._create_icon(cls._TUNE, color)
