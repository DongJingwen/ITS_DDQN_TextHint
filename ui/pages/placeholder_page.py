"""Placeholder pages for the staged desktop UI flow."""

from __future__ import annotations

from ui.qt_compat import QLabel, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    """Simple placeholder for pages implemented in later UI steps."""

    def __init__(self, title: str, description: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        description_label = QLabel(description)
        description_label.setObjectName("pageDescription")
        description_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addStretch()

