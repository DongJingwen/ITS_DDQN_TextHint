"""Login page for real-user desktop testing."""

from __future__ import annotations

from ui.qt_compat import (
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
    Signal,
)


class LoginPage(QWidget):
    """Collect student identity before starting the experiment flow."""

    login_submitted = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.student_id_input = QLineEdit()
        self.name_input = QLineEdit()
        self.start_button = QPushButton("Continue")
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(48, 48, 48, 48)
        root_layout.addItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))

        panel = QFrame()
        panel.setObjectName("loginPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(32, 28, 32, 28)
        panel_layout.setSpacing(14)

        title = QLabel("DDQN Text-Hint Tutoring System")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Student Login")
        subtitle.setObjectName("pageSubtitle")

        self.student_id_input.setPlaceholderText("Student ID")
        self.name_input.setPlaceholderText("Name (optional)")
        self.start_button.clicked.connect(self._submit)

        panel_layout.addWidget(title)
        panel_layout.addWidget(subtitle)
        panel_layout.addSpacing(8)
        panel_layout.addWidget(QLabel("Student ID"))
        panel_layout.addWidget(self.student_id_input)
        panel_layout.addWidget(QLabel("Name"))
        panel_layout.addWidget(self.name_input)
        panel_layout.addSpacing(10)
        panel_layout.addWidget(self.start_button)

        root_layout.addWidget(panel)
        root_layout.addItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def _submit(self) -> None:
        self.login_submitted.emit(
            self.student_id_input.text(),
            self.name_input.text(),
        )

