"""Main desktop GUI window with stacked-page navigation."""

from __future__ import annotations

from ui.controller import GUIController
from ui.learning_page import LearningPage
from ui.login_page import LoginPage
from ui.posttest_page import PosttestPage
from ui.pretest_page import PretestPage
from ui.progress_page import ProgressPage
from ui.qt_compat import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from ui.result_page import ResultPage
from ui.styles import APP_STYLESHEET


class MainWindow(QMainWindow):
    """Lightweight desktop shell for the real-user testing workflow."""

    APP_TITLE = "Intelligent Tutoring System for Quadratic Equations"

    PAGE_LABELS = [
        "Login",
        "Pre-test",
        "Learning",
        "Progress",
        "Post-test",
        "Results",
    ]

    def __init__(self, controller: GUIController | None = None) -> None:
        super().__init__()
        self.controller = controller or GUIController()
        self.sidebar = QListWidget()
        self.stack = QStackedWidget()
        self.status_label = QLabel("No student logged in")
        self.login_page = LoginPage()
        self.pretest_page = PretestPage()
        self.learning_page = LearningPage()
        self.progress_page = ProgressPage()
        self.posttest_page = PosttestPage()
        self.result_page = ResultPage()
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        self.setWindowTitle(self.APP_TITLE)
        self.resize(1120, 720)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(APP_STYLESHEET)

        central = QWidget()
        central.setObjectName("appRoot")
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar_container = QWidget()
        sidebar_container.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(20, 24, 20, 20)
        sidebar_layout.setSpacing(12)

        title = QLabel(self.APP_TITLE)
        title.setObjectName("sidebarTitle")
        title.setWordWrap(True)
        subtitle = QLabel("DDQN text-hint prototype for real-user testing")
        subtitle.setObjectName("sidebarSubtitle")
        subtitle.setWordWrap(True)
        self.status_label.setObjectName("statusPill")
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addWidget(self.status_label)
        sidebar_layout.addWidget(self.sidebar)
        sidebar_layout.addStretch()

        footer = QLabel("Quadratic Equations")
        footer.setObjectName("sidebarFooter")
        sidebar_layout.addWidget(footer)

        for label in self.PAGE_LABELS:
            self.sidebar.addItem(QListWidgetItem(label))

        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.pretest_page)
        self.stack.addWidget(self.learning_page)
        self.stack.addWidget(self.progress_page)
        self.stack.addWidget(self.posttest_page)
        self.stack.addWidget(self.result_page)

        self.sidebar.setCurrentRow(0)
        self.sidebar.setEnabled(False)

        root_layout.addWidget(sidebar_container, 1)
        root_layout.addWidget(self.stack, 4)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.login_page.login_requested.connect(self._handle_login)

    def _handle_login(self, student_id: str, student_name: str) -> None:
        try:
            session = self.controller.login_student(student_id, student_name)
        except ValueError as exc:
            self.login_page.show_message(str(exc))
            return

        display_name = session.student_name or session.student_id
        self.status_label.setText(f"Student: {display_name}")
        self.sidebar.setEnabled(True)
        self.sidebar.setCurrentRow(1)

    def reset_for_new_student(self) -> None:
        for page in (
            self.pretest_page,
            self.learning_page,
            self.progress_page,
            self.posttest_page,
            self.result_page,
        ):
            reset_page = getattr(page, "reset_page", None)
            if reset_page is not None:
                reset_page()

        self.login_page.clear_inputs()
        self.status_label.setText("No student logged in")
        self.sidebar.setCurrentRow(0)
        self.sidebar.setEnabled(False)
