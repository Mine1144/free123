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


def wait_until(app, predicate, timeout: float = 8.0) -> bool:
    """Process Qt events with a real sleep so other threads get scheduled too."""
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        _time.sleep(0.02)
    return False


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
    assert window.tabs.count() == 7
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()
    # Shutdown is intentionally asynchronous (workers finish their current item first).
    assert wait_until(app, lambda: not window.ocr_thread.isRunning(), timeout=15)
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


def test_local_ai_page_shows_catalog_and_never_starts_anything(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from screen_thai.app import Window
    window = Window()
    page = window.local_page
    assert page.catalog_combo.count() >= 3
    assert page.asset_combo.count() >= 1
    assert page.model_combo.count() >= 1            # the empty-state row
    assert "ยังไม่ทำงาน" in page.status_label.text()
    assert window.llama.state.running is False      # opening the tab must not launch a process
    # Settings round-trip for the llama fields.
    page.port.setValue(8099)
    page.ctx.setValue(16384)
    page.gpu.setValue(8)
    page.binary_edit.setText("/tmp/llama-server")
    settings = window.read_settings()
    assert (settings.llama_port, settings.llama_ctx, settings.llama_gpu_layers) == (8099, 16384, 8)
    assert settings.llama_binary == "/tmp/llama-server"
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_download_requires_consent_and_never_touches_the_network(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from PySide6.QtWidgets import QMessageBox
    from screen_thai.app import Window

    asked = {}

    def refuse(*args, **kwargs):
        asked["called"] = True
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(refuse))
    window = Window()
    monkeypatch.setattr(window, "_start_download",
                        lambda label, url, target: asked.__setitem__("started", label))
    window.pending_llama_files = [{"name": "M-Q4_K_M.gguf", "size": 10, "mmproj": False}]
    window.local_page.repo_edit.setText("owner/repo")
    window.download_llama_selection(0)
    assert asked.get("called") is True
    assert "started" not in asked, "a refused dialog must not start a download"
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_download_progress_then_success_updates_the_page(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    # The worker really runs in its own thread: stub the transfer so no test ever hits the net.
    from screen_thai import llama

    def fake_fetch(self, url, target, progress=None, cancel=None, sha256="", extra_gb=0.0):
        target = llama.Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        if progress:
            progress(5, 5)
        return llama.DownloadResult(target, 5, True)

    monkeypatch.setattr(llama.Downloader, "fetch", fake_fetch)
    from screen_thai.app import Window
    window = Window()
    window.pending_llama_files = [{"name": "M-Q4_K_M.gguf", "size": 1, "mmproj": False}]
    window.local_page.show_files(window.pending_llama_files, "owner/repo")
    window._download_progress("M-Q4_K_M.gguf", 500, 1000)
    assert "50%" in window.local_page.download_note.text()

    target = tmp_path / "M-Q4_K_M.gguf"
    window._start_download("M-Q4_K_M.gguf",
                           "https://huggingface.co/o/r/resolve/main/M-Q4_K_M.gguf", target)
    # The worker runs in its own thread: give it real time (not a tight processEvents loop).
    assert wait_until(app, lambda: not window.downloads, timeout=10), \
        "the stubbed worker never reported back"
    assert target.exists()
    assert window.local_page.file_table.item(0, 3).text() == "เสร็จแล้ว"
    assert "เสร็จแล้ว" in window.local_page.download_note.text()
    assert window.local_page.cancel_button.isEnabled() is False

    # A failure is surfaced with the reason instead of a silent stop.
    window._download_done("other.gguf", "", "โฮสต์ไม่ได้รับอนุญาต")
    assert "ไม่สำเร็จ" in window.status.text()
    assert window.download_failures
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_auto_start_only_applies_to_the_loopback_endpoint(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from dataclasses import replace
    from screen_thai.app import Window
    window = Window()
    local = dict(provider="openai", endpoint="http://127.0.0.1:8081/v1",
                 llama_binary="/opt/llama-server", llama_model="/m/model.gguf", llama_auto_start=True)
    window.settings = replace(window.settings, **local)
    assert window.llama_auto_start_allowed() is True
    for key, value in (("provider", "gemini"), ("endpoint", "https://api.example.com/v1"),
                       ("llama_auto_start", False), ("llama_model", ""), ("llama_binary", "")):
        window.settings = replace(window.settings, **{key: value})
        assert window.llama_auto_start_allowed() is False, key
        window.settings = replace(window.settings, **local)
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_use_llama_endpoint_refuses_while_stopped_and_saves_when_running(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from PySide6.QtWidgets import QMessageBox
    from screen_thai.app import Window
    window = Window()
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args, **kwargs: warned.append(args[1])))
    assert window.use_llama_endpoint() is False       # not running yet
    assert warned and "ยังไม่ทำงาน" in warned[-1]

    window.llama.state.running = True                 # pretend the server answered /health
    window.llama.base_url = "http://127.0.0.1:8099/v1"
    monkeypatch.setattr(window.llama, "models", lambda: ["local-Q4_K_M.gguf"])
    assert window.use_llama_endpoint() is True
    assert window.endpoint.text() == "http://127.0.0.1:8099/v1"
    assert window.model.text() == "local-Q4_K_M.gguf"
    assert window.provider.currentData() == "openai"
    saved = window.read_settings()
    assert (saved.provider, saved.endpoint, saved.model) == (
        "openai", "http://127.0.0.1:8099/v1", "local-Q4_K_M.gguf")
    assert "ในเครื่อง" in window.status.text()
    window.llama.state.running = False
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_github_listing_needs_consent_and_refresh_is_offline(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from PySide6.QtWidgets import QMessageBox
    from screen_thai import llama
    from screen_thai.app import Window
    window = Window()
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.No))
    monkeypatch.setattr(llama, "llama_assets",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    window.fetch_llama_assets()
    assert window.pending_assets == []
    # The installed-model view reads the local folder only.
    assert window.refresh_llama_models() == []
    assert "ยังไม่มีโมเดล" in window.status.text()
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_llama_settings_survive_a_restart(app, tmp_path, monkeypatch):
    """save → new window must keep binary/model/port (no silent wipe of llama_* fields)."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from screen_thai.app import Window
    window = Window()
    window.local_page.binary_edit.setText("/opt/llama-server")
    window.local_page.port.setValue(8099)
    window.store.save_settings(window.read_settings())
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()

    second = Window()
    assert second.settings.llama_binary == "/opt/llama-server"
    assert second.settings.llama_port == 8099
    second.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()


def test_starting_server_shows_loading_not_a_fake_error(app, tmp_path, monkeypatch):
    """The first health probe happens while the model is still loading: show กำลังโหลด, not มีปัญหา."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    import httpx
    from screen_thai import llama
    from screen_thai.app import Window

    class StubProcess:
        def __init__(self, args, **kwargs):
            self.args = args
            self.returncode = None
            self.stdout = iter(["loading model\n"])

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    monkeypatch.setattr(llama.subprocess, "Popen", lambda args, **kw: StubProcess(args, **kw))
    window = Window()
    mdir = llama.models_dir(window.store.root)
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "Model-Q4_K_M.gguf").write_bytes(b"m" * 128)
    window.refresh_llama_models()
    window.local_page.binary_edit.setText(str(binary))
    window.local_page.model_combo.setCurrentIndex(0)
    window.llama.transport = httpx.MockTransport(
        lambda request: httpx.Response(503, json={"status": "loading"}))
    assert window.start_llama() is True
    window._llama_poll(0)
    label = window.local_page.status_label.text()
    assert "กำลังโหลด" in label and "มีปัญหา" not in label, label
    assert "กำลังโหลดโมเดลเข้า llama-server" in window.status.text()
    window._llama_poll(1)                    # odd attempts must not block on the network
    assert window.llama.state.error == ""
    assert window.llama.state.error == ""
    assert window.llama.state.starting is True
    assert "กำลังโหลด" in window.status.text()
    window.llama.stop()
    window.close()
    QTimer.singleShot(5000, app.quit)
    app.exec()
