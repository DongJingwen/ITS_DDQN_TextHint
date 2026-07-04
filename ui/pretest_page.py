"""Pre-test page for estimating a student's initial level."""

from __future__ import annotations

from ui.qt_compat import QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QWidget


class PretestPage(QWidget):
    """Display pre-test questions one at a time and submit a score."""

    LEARNING_PAGE_INDEX = 2

    def __init__(self) -> None:
        super().__init__()
        self.questions = []
        self.current_index = 0
        self.answers_by_question_id: dict[str, str] = {}

        self.progress_label = QLabel("")
        self.question_label = QLabel("")
        self.answer_input = QLineEdit()
        self.message_label = QLabel("")
        self.next_button = QPushButton("Next Question")
        self.submit_button = QPushButton("Submit Pre-test")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)

        title = QLabel("Pre-test")
        title.setObjectName("pageTitle")
        description = QLabel("Answer each question. The score estimates your initial learning level.")
        description.setObjectName("pageDescription")
        description.setWordWrap(True)

        self.question_label.setWordWrap(True)
        self.question_label.setMinimumHeight(90)
        self.question_label.setObjectName("questionText")
        self.answer_input.setPlaceholderText("Type your answer here")
        self.progress_label.setObjectName("sectionLabel")
        self.message_label.setObjectName("messageText")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.submit_button)

        self.next_button.clicked.connect(self.go_to_next_question)
        self.submit_button.clicked.connect(self.submit_pretest)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(10)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.question_label)
        layout.addWidget(QLabel("Answer"))
        layout.addWidget(self.answer_input)
        layout.addWidget(self.message_label)
        layout.addLayout(button_layout)
        layout.addStretch()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        super().showEvent(event)
        if not self.questions:
            self._load_questions()

    def _controller(self):
        window = self.window()
        return getattr(window, "controller", None)

    def _load_questions(self) -> None:
        controller = self._controller()
        if controller is None:
            self.message_label.setText("Controller is not available.")
            return

        self.questions = controller.get_pretest_questions()
        self.current_index = 0
        self.answers_by_question_id = {}
        self._display_current_question()

    def _display_current_question(self) -> None:
        if not self.questions:
            self.progress_label.setText("No pre-test questions available.")
            self.question_label.setText("")
            self.next_button.setEnabled(False)
            self.submit_button.setEnabled(False)
            return

        question = self.questions[self.current_index]
        self.progress_label.setText(
            f"Question {self.current_index + 1} of {len(self.questions)}"
        )
        self.question_label.setText(question.prompt)
        self.answer_input.setText(
            self.answers_by_question_id.get(question.question_id, "")
        )
        self.message_label.setText("")
        self.next_button.setEnabled(self.current_index < len(self.questions) - 1)
        self.submit_button.setEnabled(True)

    def _save_current_answer(self) -> None:
        if not self.questions:
            return
        question = self.questions[self.current_index]
        self.answers_by_question_id[question.question_id] = self.answer_input.text()

    def go_to_next_question(self) -> None:
        self._save_current_answer()
        if self.current_index < len(self.questions) - 1:
            self.current_index += 1
            self._display_current_question()

    def submit_pretest(self) -> None:
        self._save_current_answer()
        controller = self._controller()
        if controller is None:
            self.message_label.setText("Controller is not available.")
            return

        try:
            result = controller.submit_pretest(self.answers_by_question_id)
        except ValueError as exc:
            self.message_label.setText(str(exc))
            return

        self.message_label.setText(
            "Pre-test submitted. "
            f"Score: {result['score']:.2f}, "
            f"Initial difficulty: {result['initial_difficulty']}."
        )
        self._go_to_learning_page()

    def _go_to_learning_page(self) -> None:
        window = self.window()
        stack = getattr(window, "stack", None)
        sidebar = getattr(window, "sidebar", None)

        if sidebar is not None:
            sidebar.setCurrentRow(self.LEARNING_PAGE_INDEX)
        elif stack is not None:
            stack.setCurrentIndex(self.LEARNING_PAGE_INDEX)

    def reset_page(self) -> None:
        self.questions = []
        self.current_index = 0
        self.answers_by_question_id = {}
        self.progress_label.setText("")
        self.question_label.setText("")
        self.answer_input.clear()
        self.message_label.setText("")
        self.next_button.setEnabled(True)
        self.submit_button.setEnabled(True)
