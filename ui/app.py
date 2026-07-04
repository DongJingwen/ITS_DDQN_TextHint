"""Minimal runnable desktop GUI entry point."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.controller import GUIController
from ui.main_window import MainWindow
from ui.qt_compat import QApplication, QT_BINDING


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("DDQN Text-Hint ITS")
    window = MainWindow(controller=GUIController())
    window.show()
    print(f"Using Qt binding: {QT_BINDING}")
    exec_method = getattr(app, "exec", None) or getattr(app, "exec_", None)
    return exec_method()


if __name__ == "__main__":
    raise SystemExit(main())
