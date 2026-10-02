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


def test_new_pages_render(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from screen_thai.app import Window
    window = Window()
    window.research_page.override_edit.setPlainText("ยืนยัน: ไรซ่าใช้คำว่า พี่สาว")
    window.research_save_override()
    window.characters_page.refresh(window.memory, [], None)
    assert "ไรซ่า" in window.brief.get("user_notes", "")
    assert window.research_page.values()["provider"] in ("off", "searxng", "brave", "serper",
                                                         "tavily", "custom")
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_window_creation_and_graceful_shutdown(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from screen_thai.app import Window
    window = Window()
    assert window.tabs.count() == 6
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()
    assert not window.ocr_thread.isRunning()
    assert not window.ai_thread.isRunning()


def test_quality_settings_round_trip(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from screen_thai.app import Window
    window = Window()
    window.merge_lines.setChecked(False)
    window.translation_memory.setChecked(False)
    window.glossary_mine.setChecked(True)
    window.review_consistency.setChecked(True)
    window.review_max_lines.setValue(7)
    settings = window.read_settings()
    assert (settings.merge_lines, settings.translation_memory, settings.glossary_mine,
            settings.review_consistency, settings.review_max_lines) == (False, False, True,
                                                                       True, 7)
    # The values survive a save/load cycle through the store.
    window.store.save_settings(settings)
    reloaded = window.store.load_settings()
    assert reloaded.review_consistency is True and reloaded.review_max_lines == 7
    assert reloaded.translation_memory is False and reloaded.merge_lines is False
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_selfcheck_panel_reports_without_network(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from screen_thai.app import Window
    window = Window()
    window.run_selfcheck(False)          # no probe: must never touch the network
    report = window.selfcheck_view.toPlainText()
    assert "โฟลเดอร์ข้อมูล" in report and "รวม" in report
    assert "ยังไม่ได้ตรวจ" in report      # the AI connection was not probed
    assert "✔" in report or "!" in report or "✘" in report
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_character_suggestions_list_stays_inert_until_accepted(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from screen_thai.app import Window
    window = Window()
    window.suggestions = [{"source": "Ryza", "thai": "ไรซ่า", "count": 2, "confidence": 0.6,
                           "evidence": ["Ryza is late. → ไรซ่าสายแล้ว"]}]
    window.refresh_pages()
    assert window.characters_page.suggest_list.count() == 1
    text = window.characters_page.suggest_list.item(0).text()
    assert "Ryza" in text and "ไรซ่า" in text and "2 ครั้ง" in text
    assert "ยังไม่ถูกใช้" in window.characters_page.suggest_list.parent().title()
    assert window.glossary.toPlainText() == ""      # nothing is used yet
    window.accept_suggestions([0])
    assert "Ryza = ไรซ่า" in window.glossary.toPlainText()
    assert window.characters_page.suggest_list.count() == 0
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()
