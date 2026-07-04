"""Read-only learning progress page."""

from __future__ import annotations

from ui.qt_compat import QLabel, QVBoxLayout, QWidget


class ProgressPage(QWidget):
    """Show current real-user session progress without changing backend state."""

    def __init__(self) -> None:
        super().__init__()
        self.questions_attempted_label = QLabel("Questions attempted: 0")
        self.correct_rate_label = QLabel("Correct rate: 0.0%")
        self.average_attempts_label = QLabel("Average attempts: 0.00")
        self.hints_used_label = QLabel("Hints used: 0")
        self.mastery_label = QLabel("Current estimated mastery: Not available")
        self.status_label = QLabel("")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(14)

        title = QLabel("Progress")
        title.setObjectName("pageTitle")
        description = QLabel("Current learning session summary.")
        description.setObjectName("pageDescription")
        description.setWordWrap(True)

        for label in (
            self.questions_attempted_label,
            self.correct_rate_label,
            self.average_attempts_label,
            self.hints_used_label,
            self.mastery_label,
        ):
            label.setObjectName("pageDescription")

        self.status_label.setObjectName("pageDescription")

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(12)
        layout.addWidget(self.questions_attempted_label)
        layout.addWidget(self.correct_rate_label)
        layout.addWidget(self.average_attempts_label)
        layout.addWidget(self.hints_used_label)
        layout.addWidget(self.mastery_label)
        layout.addWidget(self.status_label)
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

        summary = controller.get_progress_summary()
        mastery_level = summary.get("current_mastery_level")
        mastery_text = (
            f"{mastery_level:.2f}"
            if isinstance(mastery_level, (float, int))
            else "Not available"
        )

        self.questions_attempted_label.setText(
            f"Questions attempted: {summary['questions_attempted']}"
        )
        self.correct_rate_label.setText(
            f"Correct rate: {summary['correct_rate']:.1%}"
        )
        self.average_attempts_label.setText(
            f"Average attempts: {summary['average_attempts']:.2f}"
        )
        self.hints_used_label.setText(f"Hints used: {summary['hints_used']}")
        self.mastery_label.setText(f"Current estimated mastery: {mastery_text}")
        self.status_label.setText("Progress updates when this page is opened.")
