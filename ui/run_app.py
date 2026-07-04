"""Launch the desktop UI for the DDQN text-hint ITS prototype."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ui.main_window import MainWindow
from ui.qt_compat import QApplication, QT_BINDING
from ui.services.session_service import UISessionService


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("DDQN Text-Hint ITS")

    session_service = UISessionService()
    window = MainWindow(session_service=session_service)
    window.show()

    print(f"Using Qt binding: {QT_BINDING}")
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
