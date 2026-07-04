"""Qt compatibility layer.

The project prefers PySide6, but falls back to PyQt5 when PySide6 is not
available. UI modules should import Qt classes from here instead of importing a
specific binding directly.
"""

from __future__ import annotations


QT_BINDING = ""

try:  # pragma: no cover - depends on local GUI dependency availability.
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSpacerItem,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"
except ImportError:  # pragma: no cover - active in current environment.
    from PyQt5.QtCore import Qt, pyqtSignal as Signal
    from PyQt5.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSpacerItem,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PyQt5"


__all__ = [
    "QApplication",
    "QFrame",
    "QHBoxLayout",
    "QLabel",
    "QLineEdit",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMessageBox",
    "QPushButton",
    "QSizePolicy",
    "QSpacerItem",
    "QStackedWidget",
    "QVBoxLayout",
    "QWidget",
    "QT_BINDING",
    "Qt",
    "Signal",
]

