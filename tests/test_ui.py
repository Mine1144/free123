"""Native smoke tests: skipped if this Linux image has no Qt system libraries."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    pytest.skip("Qt platform libraries unavailable", allow_module_level=True)

from PySide6.QtCore import QRectF, QTimer
from screen_thai.models import Block, Result, Settings, Translation
from screen_thai.overlay import Overlay


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_overlay_paints_thai(app):
    overlay = Overlay()
    overlay.configure(app.primaryScreen(), QRectF(0, 0, 1, 1), Settings())
    overlay.display([Block(0, "Hello", (10, 10, 200, 50))], Result([Translation(0, "สวัสดีครับ")]), (800, 600))
    overlay.show()
    app.processEvents()
    assert not overlay.grab().isNull()
    overlay.clear()
    overlay.close()


def test_window_creation_and_graceful_shutdown(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from screen_thai.app import Window
    window = Window()
    assert window.tabs.count() == 4
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()
    assert not window.ocr_thread.isRunning()
    assert not window.ai_thread.isRunning()
