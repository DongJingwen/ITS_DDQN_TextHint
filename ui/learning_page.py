"""Learning session page for real-user desktop testing."""

from __future__ import annotations

from ui.qt_compat import QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QWidget


class LearningPage(QWidget):
    """Display learning questions, submit answers, and show recommended hints."""

    POSTTEST_PAGE_INDEX = 4

    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.student_label = QLabel("Student: -")
        self.kp_label = QLabel("Knowledge point: -")
        self.question_label = QLabel("")
        self.answer_input = QLineEdit()
        self.hint_label = QLabel("Hint will appear here after an incorrect answer.")
        self.attempts_label = QLabel("Attempts: 0")
        self.status_label = QLabel("")
        self.submit_button = QPushButton("Submit Answer")
        self.next_button = QPushButton("Next Question")
        self.posttest_button = QPushButton("Go to Post-test")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)

        title = QLabel("Learning Session")
        title.setObjectName("pageTitle")
        self.question_label.setWordWrap(True)
        self.question_label.setMinimumHeight(90)
        self.question_label.setObjectName("questionText")
        self.hint_label.setWordWrap(True)
        self.hint_label.setMinimumHeight(70)
        self.hint_label.setObjectName("pageDescription")
        self.student_label.setObjectName("sectionLabel")
        self.kp_label.setObjectName("sectionLabel")
        self.attempts_label.setObjectName("sectionLabel")
        self.status_label.setObjectName("messageText")
        self.answer_input.setPlaceholderText("Type your answer here")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.submit_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.posttest_button)

        self.submit_button.clicked.connect(self.submit_answer)
        self.next_button.clicked.connect(self.load_next_question)
        self.posttest_button.clicked.connect(self.go_to_posttest)
        self.answer_input.returnPressed.connect(self.submit_answer)
        self.next_button.setEnabled(False)

        layout.addWidget(title)
        layout.addWidget(self.student_label)
        layout.addWidget(self.kp_label)
        layout.addWidget(self.question_label)
        layout.addWidget(QLabel("Answer"))
        layout.addWidget(self.answer_input)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.attempts_label)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        layout.addStretch()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        super().showEvent(event)
        if not self.started:
            self.start_learning()

    def _controller(self):
        window = self.window()
        return getattr(window, "controller", None)

    def start_learning(self) -> None:
        controller = self._controller()
        if controller is None:
            self.status_label.setText("Controller is not available.")
            return

        try:
            question = controller.start_learning_session()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        self.started = True
        self._display_question(question)

    def _display_question(self, question) -> None:
        controller = self._controller()
        session = controller.session_manager.current_student if controller else None
        student_id = session.student_id if session else "-"

        self.student_label.setText(f"Student: {student_id}")
        self.kp_label.setText(
            f"Knowledge point: {question.kp_id} | Difficulty: {question.difficulty}"
        )
        self.question_label.setText(question.prompt)
        self.answer_input.clear()
        self.hint_label.setText("Hint will appear here after an incorrect answer.")
        self.attempts_label.setText("Attempts: 0")
        self.status_label.setText("Enter an answer and submit.")
        self.next_button.setEnabled(False)

    def submit_answer(self) -> None:
        controller = self._controller()
        if controller is None:
            self.status_label.setText("Controller is not available.")
            return

        answer = self.answer_input.text().strip()
        if not answer:
            self.status_label.setText("Please enter an answer before submitting.")
            return

        try:
            result = controller.submit_learning_answer(answer)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        self.attempts_label.setText(f"Attempts: {result['attempts']}")
        self.status_label.setText(result["message"])

        if result["hint_text"]:
            self.hint_label.setText(
                f"Recommended hint ({result['hint_level']}): {result['hint_text']}"
            )

        if result["correct"]:
            self.next_button.setEnabled(True)
        else:
            self.answer_input.selectAll()
            self.answer_input.setFocus()

    def load_next_question(self) -> None:
        controller = self._controller()
        if controller is None:
            self.status_label.setText("Controller is not available.")
            return

        question = controller.move_to_next_learning_question()
        self._display_question(question)

    def go_to_posttest(self) -> None:
        window = self.window()
        stack = getattr(window, "stack", None)
        sidebar = getattr(window, "sidebar", None)

        if sidebar is not None:
            sidebar.setCurrentRow(self.POSTTEST_PAGE_INDEX)
        elif stack is not None:
            stack.setCurrentIndex(self.POSTTEST_PAGE_INDEX)

    def reset_page(self) -> None:
        self.started = False
        self.student_label.setText("Student: -")
        self.kp_label.setText("Knowledge point: -")
        self.question_label.setText("")
        self.answer_input.clear()
        self.hint_label.setText("Hint will appear here after an incorrect answer.")
        self.attempts_label.setText("Attempts: 0")
        self.status_label.setText("")
        self.next_button.setEnabled(False)
