CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT,
    group_type TEXT,
    created_at TEXT,
    initial_level REAL,
    current_level REAL
);

CREATE TABLE IF NOT EXISTS pretest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    score REAL,
    estimated_level REAL,
    initial_difficulty TEXT,
    timestamp TEXT,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS posttest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    score REAL,
    learning_gain REAL,
    timestamp TEXT,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS interaction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    episode INTEGER,
    step INTEGER,
    kp_id TEXT,
    question_id TEXT,
    difficulty TEXT,
    state TEXT,
    action INTEGER,
    hint_level TEXT,
    reward REAL,
    correct INTEGER,
    response_time REAL,
    attempts INTEGER,
    mastery_level REAL,
    epsilon REAL,
    timestamp TEXT,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS hint_usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    kp_id TEXT,
    question_id TEXT,
    action INTEGER,
    hint_level TEXT,
    used INTEGER,
    timestamp TEXT,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);
