"""Main desktop window for real-user ITS testing."""

from __future__ import annotations

from ui.pages.login_page import LoginPage
from ui.pages.placeholder_page import PlaceholderPage
from ui.qt_compat import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from ui.services.session_service import UISessionService


class MainWindow(QMainWindow):
    """Desktop shell with staged pages for the real-user workflow."""

    PAGE_NAMES = [
        "Login",
        "Pre-test",
        "Learning",
        "Progress",
        "Post-test",
        "Results",
    ]

    def __init__(self, session_service: UISessionService) -> None:
        super().__init__()
        self.session_service = session_service
        self.sidebar = QListWidget()
        self.stack = QStackedWidget()
        self.status_label = QLabel("Not logged in")
        self.login_page = LoginPage()
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        self.setWindowTitle("DDQN Text-Hint ITS")
        self.resize(1024, 680)

        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar_container = QWidget()
        sidebar_container.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(16, 18, 16, 18)
        sidebar_layout.setSpacing(12)

        app_title = QLabel("ITS Prototype")
        app_title.setObjectName("sidebarTitle")
        sidebar_layout.addWidget(app_title)
        sidebar_layout.addWidget(self.status_label)
        sidebar_layout.addWidget(self.sidebar)

        for page_name in self.PAGE_NAMES:
            item = QListWidgetItem(page_name)
            self.sidebar.addItem(item)

        self.sidebar.setCurrentRow(0)
        self.sidebar.setEnabled(False)

        self.stack.addWidget(self.login_page)
        self.stack.addWidget(
            PlaceholderPage(
                "Pre-test",
                "This page will collect a pre-test score and estimate the student's initial level.",
            )
        )
        self.stack.addWidget(
            PlaceholderPage(
                "Learning Session",
                "This page will show quadratic equation questions, recommended text hints, answer submission, and re-attempts.",
            )
        )
        self.stack.addWidget(
            PlaceholderPage(
                "Progress",
                "This optional page will display learning progress, hint usage, attempts, and recent accuracy.",
            )
        )
        self.stack.addWidget(
            PlaceholderPage(
                "Post-test",
                "This page will collect a post-test score and calculate learning gain.",
            )
        )
        self.stack.addWidget(
            PlaceholderPage(
                "Results",
                "This page will summarize reward, correctness, response time, attempts, hint usage, and learning gain.",
            )
        )

        root_layout.addWidget(sidebar_container, 1)
        root_layout.addWidget(self.stack, 4)
        self.setCentralWidget(central)
        self.setStyleSheet(self._stylesheet())

    def _connect_signals(self) -> None:
        self.login_page.login_submitted.connect(self._handle_login)
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)

    def _handle_login(self, student_id: str, name: str) -> None:
        try:
            profile = self.session_service.login_student(student_id, name)
        except ValueError as exc:
            QMessageBox.warning(self, "Login Required", str(exc))
            return

        display_name = profile.name or profile.student_id
        self.status_label.setText(f"Student: {display_name}")
        self.sidebar.setEnabled(True)
        self.sidebar.setCurrentRow(1)
        QMessageBox.information(self, "Login Complete", "Student profile saved.")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        self.session_service.close()
        event.accept()

    @staticmethod
    def _stylesheet() -> str:
        return """
        QWidget {
            font-family: Arial;
            font-size: 14px;
            color: #1f2933;
            background: #f7f8fa;
        }
        #sidebar {
            background: #eef1f5;
            border-right: 1px solid #d7dce2;
        }
        #sidebarTitle {
            font-size: 18px;
            font-weight: 700;
        }
        QListWidget {
            border: 1px solid #d7dce2;
            background: #ffffff;
            padding: 4px;
        }
        QListWidget::item {
            padding: 10px;
            border-radius: 4px;
        }
        QListWidget::item:selected {
            background: #dce8f7;
        }
        #loginPanel {
            background: #ffffff;
            border: 1px solid #d7dce2;
            border-radius: 6px;
            max-width: 480px;
        }
        #pageTitle {
            font-size: 24px;
            font-weight: 700;
        }
        #pageSubtitle {
            font-size: 16px;
            color: #52606d;
        }
        #pageDescription {
            color: #52606d;
            line-height: 1.4;
        }
        QLineEdit {
            background: #ffffff;
            border: 1px solid #cbd2d9;
            border-radius: 4px;
            padding: 8px;
        }
        QPushButton {
            background: #2563eb;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            padding: 10px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #1d4ed8;
        }
        """

