"""Login page for the desktop experiment flow."""

from __future__ import annotations

from ui.qt_compat import QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, Signal


class LoginPage(QWidget):
    """Collect student ID and optional name."""

    login_requested = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.student_id_input = QLineEdit()
        self.name_input = QLineEdit()
        self.message_label = QLabel("")
        self.start_button = QPushButton("Start")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)

        title = QLabel("Student Login")
        title.setObjectName("pageTitle")
        description = QLabel("Enter student information to begin the experiment.")
        description.setObjectName("pageDescription")
        description.setWordWrap(True)

        self.student_id_input.setPlaceholderText("Student ID")
        self.name_input.setPlaceholderText("Name (optional)")
        self.message_label.setObjectName("pageDescription")
        self.start_button.clicked.connect(self._emit_login)
        self.student_id_input.returnPressed.connect(self._emit_login)
        self.name_input.returnPressed.connect(self._emit_login)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Student ID"))
        layout.addWidget(self.student_id_input)
        layout.addWidget(QLabel("Name"))
        layout.addWidget(self.name_input)
        layout.addWidget(self.message_label)
        layout.addWidget(self.start_button)
        layout.addStretch()

    def _emit_login(self) -> None:
        self.clear_message()
        self.login_requested.emit(self.student_id_input.text(), self.name_input.text())

    def show_message(self, message: str) -> None:
        self.message_label.setText(message)

    def clear_message(self) -> None:
        self.message_label.setText("")

    def clear_inputs(self) -> None:
        self.student_id_input.clear()
        self.name_input.clear()
        self.clear_message()
