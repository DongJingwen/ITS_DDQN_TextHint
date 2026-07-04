"""Shared stylesheet for the desktop GUI."""


APP_STYLESHEET = """
QWidget {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 14px;
    color: #22313f;
    background: #f5f7fa;
}

QWidget#appRoot {
    background: #eef2f6;
}

QWidget#sidebar {
    background: #ffffff;
    border-right: 1px solid #d8dee6;
}

QLabel#sidebarTitle {
    font-size: 20px;
    font-weight: 700;
    line-height: 1.25;
    color: #162235;
    background: transparent;
}

QLabel#sidebarSubtitle {
    color: #617083;
    font-size: 13px;
    background: transparent;
}

QLabel#sidebarFooter {
    color: #7b8794;
    font-size: 12px;
    background: transparent;
}

QLabel#statusPill {
    background: #eef5ff;
    border: 1px solid #c8ddf6;
    border-radius: 4px;
    color: #234f83;
    padding: 8px 10px;
}

QLabel#pageTitle {
    font-size: 26px;
    font-weight: 700;
    color: #162235;
    background: transparent;
}

QLabel#pageDescription {
    color: #526173;
    background: transparent;
    line-height: 1.35;
}

QLabel#sectionLabel {
    color: #334155;
    font-size: 13px;
    font-weight: 700;
    background: transparent;
}

QLabel#questionText {
    background: #ffffff;
    border: 1px solid #d8dee6;
    border-radius: 4px;
    color: #1f2937;
    font-size: 15px;
    padding: 14px;
}

QLabel#messageText {
    color: #44546a;
    background: transparent;
    padding-top: 4px;
}

QListWidget {
    background: #ffffff;
    border: 1px solid #d8dee6;
    border-radius: 4px;
    outline: none;
}

QListWidget::item {
    padding: 11px 10px;
    border-bottom: 1px solid #eef1f5;
}

QListWidget::item:selected {
    background: #e7f0fb;
    color: #173f6f;
    font-weight: 600;
}

QListWidget::item:disabled {
    color: #98a2b3;
}

QPushButton {
    background: #245ea8;
    color: #ffffff;
    border: 1px solid #245ea8;
    border-radius: 4px;
    padding: 9px 14px;
    min-height: 18px;
}

QPushButton {
    font-weight: 600;
}

QPushButton:hover {
    background: #1f528f;
    border-color: #1f528f;
}

QPushButton:pressed {
    background: #183f70;
    border-color: #183f70;
}

QPushButton:disabled {
    background: #d5dce5;
    border-color: #d5dce5;
    color: #7b8794;
}

QLineEdit {
    background: #ffffff;
    border: 1px solid #c8d1dc;
    border-radius: 4px;
    padding: 9px 10px;
    selection-background-color: #b8d4f6;
}

QLineEdit:focus {
    border: 1px solid #245ea8;
}

QStackedWidget {
    background: #f5f7fa;
}
"""
