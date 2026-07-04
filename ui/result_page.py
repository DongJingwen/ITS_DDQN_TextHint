"""Final result summary page for real-user testing."""

from __future__ import annotations

from ui.qt_compat import QLabel, QMessageBox, QPushButton, QHBoxLayout, QVBoxLayout, QWidget


class ResultPage(QWidget):
    """Show the student's post-session summary and export option."""

    LOGIN_PAGE_INDEX = 0

    def __init__(self) -> None:
        super().__init__()
        self.student_label = QLabel("Student ID: -")
        self.pretest_label = QLabel("Pre-test score: Not available")
        self.posttest_label = QLabel("Post-test score: Not available")
        self.gain_label = QLabel("Learning gain: Not available")
        self.hints_label = QLabel("Total hints used: 0")
        self.attempts_label = QLabel("Average attempts: 0.00")
        self.response_time_label = QLabel("Average response time: 0.00 seconds")
        self.correct_rate_label = QLabel("Correct rate: 0.0%")
        self.status_label = QLabel("")
        self.finish_button = QPushButton("Finish")
        self.export_button = QPushButton("Export Summary")
        self.return_login_button = QPushButton("Return to Login")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)

        title = QLabel("Results")
        title.setObjectName("pageTitle")
        description = QLabel("Final learning summary for this student session.")
        description.setObjectName("pageDescription")
        description.setWordWrap(True)

        for label in (
            self.student_label,
            self.pretest_label,
            self.posttest_label,
            self.gain_label,
            self.hints_label,
            self.attempts_label,
            self.response_time_label,
            self.correct_rate_label,
            self.status_label,
        ):
            label.setObjectName("pageDescription")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.finish_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.return_login_button)

        self.finish_button.clicked.connect(self.finish_session)
        self.export_button.clicked.connect(self.export_summary)
        self.return_login_button.clicked.connect(self.return_to_login)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(12)
        layout.addWidget(self.student_label)
        layout.addWidget(self.pretest_label)
        layout.addWidget(self.posttest_label)
        layout.addWidget(self.gain_label)
        layout.addSpacing(10)
        layout.addWidget(self.hints_label)
        layout.addWidget(self.attempts_label)
        layout.addWidget(self.response_time_label)
        layout.addWidget(self.correct_rate_label)
        layout.addSpacing(10)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        layout.addStretch()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        super().showEvent(event)
        self.refresh()

    def _controller(self):
        window = self.window()
        return getattr(window, "controller", None)

    def refresh(self) -> None:
        controller = self._controller()
        if controller is None:
            self.status_label.setText("Controller is not available.")
            return

        summary = controller.get_result_summary()
        self.student_label.setText(f"Student ID: {summary['student_id']}")
        self.pretest_label.setText(
            f"Pre-test score: {self._format_score(summary['pretest_score'])}"
        )
        self.posttest_label.setText(
            f"Post-test score: {self._format_score(summary['posttest_score'])}"
        )
        self.gain_label.setText(
            f"Learning gain: {self._format_score(summary['learning_gain'])}"
        )
        self.hints_label.setText(f"Total hints used: {summary['total_hints_used']}")
        self.attempts_label.setText(
            f"Average attempts: {summary['average_attempts']:.2f}"
        )
        self.response_time_label.setText(
            f"Average response time: {summary['average_response_time']:.2f} seconds"
        )
        self.correct_rate_label.setText(f"Correct rate: {summary['correct_rate']:.1%}")
        self.status_label.setText("Session summary is ready.")

    def export_summary(self) -> None:
        controller = self._controller()
        if controller is None:
            self.status_label.setText("Controller is not available.")
            return

        try:
            output_path = controller.export_result_summary()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        self.status_label.setText(f"Summary exported to {output_path}")

    def finish_session(self) -> None:
        QMessageBox.information(self, "Session Complete", "The session summary has been recorded.")

    def return_to_login(self) -> None:
        controller = self._controller()
        if controller is not None:
            controller.clear_current_session()

        window = self.window()
        reset_window = getattr(window, "reset_for_new_student", None)
        if reset_window is not None:
            reset_window()
            return

        status_label = getattr(window, "status_label", None)
        sidebar = getattr(window, "sidebar", None)
        stack = getattr(window, "stack", None)
        if status_label is not None:
            status_label.setText("No student logged in")
        if sidebar is not None:
            sidebar.setCurrentRow(self.LOGIN_PAGE_INDEX)
            sidebar.setEnabled(False)
        elif stack is not None:
            stack.setCurrentIndex(self.LOGIN_PAGE_INDEX)

    def reset_page(self) -> None:
        self.student_label.setText("Student ID: -")
        self.pretest_label.setText("Pre-test score: Not available")
        self.posttest_label.setText("Post-test score: Not available")
        self.gain_label.setText("Learning gain: Not available")
        self.hints_label.setText("Total hints used: 0")
        self.attempts_label.setText("Average attempts: 0.00")
        self.response_time_label.setText("Average response time: 0.00 seconds")
        self.correct_rate_label.setText("Correct rate: 0.0%")
        self.status_label.setText("")

    @staticmethod
    def _format_score(value) -> str:
        if value is None:
            return "Not available"
        return f"{float(value):.2f}"
