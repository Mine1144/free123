"""End-to-end session flow with a mocked AI transport: capture -> layout -> faces -> translate.

This exercises the real window code path (workers, state, storage, dialogue memory, QC) while
never touching a network service or a game.
"""
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import httpx
import pytest
from PIL import Image

try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    pytest.skip("Qt platform libraries unavailable", allow_module_level=True)

from dataclasses import replace

from screen_thai.models import Block
from screen_thai.vision import Appearance, ExpressionCues, FaceBox, FaceObservation

AI_REPLY = {
    "translations": [{"id": 1, "thai": "ฉันจะปกป้องเธอเอง", "speaker": "Alyssa",
                      "emotion": "กังวล", "delivery": "พูดช้า ๆ หนักแน่น", "particles": "ค่ะ"}],
    "scene": {"summary": "Alyssa สัญญาว่าจะปกป้อง Grace",
              "characters": [{"name": "Grace", "role": "ลูกสาว", "evidence": "Alyssa: Grace, listen",
                              "confidence": 0.9, "gender": "female"}],
              "faces": [{"track": "ใบหน้าที่ 1", "character": "Grace",
                         "evidence": "Alyssa: Grace, listen", "confidence": 0.85}]},
}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_full_frame_flow_with_mocked_ai(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured["payload"] = json.loads(body["messages"][-1]["content"])
        return httpx.Response(200, json={"message": {"content": json.dumps(AI_REPLY)}})

    from screen_thai import providers
    original_init = providers.AIClient.__init__

    def patched_init(self, settings, key="", transport=None):
        original_init(self, settings, key or "test-key", httpx.MockTransport(handler))

    monkeypatch.setattr(providers.AIClient, "__init__", patched_init)

    from screen_thai.app import Window
    window = Window()
    window.current_profile = "เกมทดสอบ"
    window.profile.setText("เกมทดสอบ")
    window.memory.ensure("Alyssa", confirmed=True)
    window.memory.set_voice("Alyssa", self="ฉัน", address="เธอ", particles="ค่ะ", locked=True)
    window.store.save_memory("เกมทดสอบ", window.memory)

    blocks = [Block(0, "Alyssa", (120, 480, 160, 34)),
              Block(1, "I will protect you, Grace.", (100, 520, 1080, 150))]
    image = Image.new("RGB", (1280, 720), (18, 18, 24))

    window.active, window.single = True, False
    window.epoch = 7
    window.settings = replace(window.read_settings(), profile="เกมทดสอบ",
                              face_analysis=True, layout_hints=True, speech_plan=True,
                              show_speaker_label=True)
    observation = FaceObservation(1, FaceBox(700, 120, 160, 170),
                                  ExpressionCues("ยิ้ม", 0.5, 0.1, 0.2, 0.6, 0.2, 2.0, 0.1),
                                  Appearance("#332211", "#554433", "#223344", ("#332211",), 0.5,
                                             0.3, 0.1))
    observation.character = "Alyssa"
    observation.match = 0.91

    window.latest_image = image
    window.overlay.settings = window.settings      # what start() would configure
    window.state.update(blocks)
    # Claim the vision slot so _ocr_done does not start the real worker (the model would have to
    # be downloaded); this test injects the observation it wants to exercise instead.
    window.faces = [observation]
    window.vision_busy = True
    window.last_vision_at = time.monotonic()
    window._ocr_done(7, image, blocks, "")

    # At the same time the vision worker returns its frame: this must be recorded for the trend
    # without starting a second AI request while one is already in flight.
    window._vision_done(7, window.job_id, [observation], "")
    assert window.ai_busy is True and window.job_id == 1, "vision must not queue a second job"

    arrived = wait_until(app, lambda: window.table.rowCount() > 0)
    assert arrived, (f"AI result never arrived • status={window.status.text()!r} "
                     f"ai_busy={window.ai_busy} job={window.job_id} "
                     f"blocks={[b.id for b in window.job_blocks]} "
                     f"tokens={window.job_tokens} resolved={window.state.resolved}")

    payload = captured["payload"]
    assert payload["visible_name_tags"]["1"] == "Alyssa"
    assert payload["speech_plans"]["1"]["must_follow"] is True
    assert payload["faces"][0]["identified_as"] == "Alyssa"

    row = [window.table.item(0, column).text() for column in range(5)]
    assert row[1] == "Alyssa"
    assert row[3] == "ฉันจะปกป้องเธอเอง"
    assert "กังวล" in row[4]

    assert window.dialogue.recent(1)[0].speaker == "Alyssa"
    profile = window.store.load_profile("เกมทดสอบ")
    assert profile["summary"].startswith("Alyssa")
    assert "Grace" in window.memory.names()
    assert window.memory.card("Alyssa")["voice"]["self"] == "ฉัน"
    assert window.overlay.items and window.overlay.items[0][1] == "ฉันจะปกป้องเธอเอง"
    assert window.overlay.items[0][2] == "Alyssa"

    window.active = False
    window.close()
    assert wait_until(app, lambda: not window._workers_running(), timeout=10)
    app.processEvents()


def test_repeated_line_is_reused_without_calling_the_ai(app, tmp_path, monkeypatch):
    """The translation memory must actually stop a second request for the same line."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        asked = [block["id"] for block in json.loads(body["messages"][-1]["content"])["blocks"]]
        thai = {1: "อรุณสวัสดิ์", 2: "คิโรอยู่ที่นี่"}
        # The model decides what to translate: the name tag (id 0) is intentionally skipped.
        reply = {"translations": [{"id": block_id, "thai": thai[block_id], "speaker": "Alyssa"}
                                  for block_id in asked if block_id in thai]}
        return httpx.Response(200, json={"message": {"content": json.dumps(reply)}})

    from screen_thai import providers
    original_init = providers.AIClient.__init__
    monkeypatch.setattr(providers.AIClient, "__init__",
                        lambda self, settings, key="", transport=None: original_init(
                            self, settings, key or "k", httpx.MockTransport(handler)))

    from screen_thai.app import Window
    window = Window()
    window.current_profile = "เกมทดสอบ"
    window.profile.setText("เกมทดสอบ")
    window.active, window.single = True, False
    window.epoch = 3
    window.settings = replace(window.read_settings(), profile="เกมทดสอบ", layout_hints=True,
                              speech_plan=True, face_analysis=False, merge_lines=False)
    image = Image.new("RGB", (1280, 900), (12, 12, 16))
    blocks = [Block(0, "Alyssa", (120, 500, 160, 34)),
              Block(1, "Good morning, Kiro.", (100, 545, 1080, 60))]

    window.latest_image = image
    window.overlay.settings = window.settings
    window._ocr_done(3, image, blocks, "")
    assert wait_until(app, lambda: window.table.rowCount() == 1), "first translation missing"
    assert len(calls) == 1

    # Same frame again: the accepted line must come back from memory, not from the model.
    window._ocr_done(3, image, blocks, "")
    time.sleep(0.3)
    app.processEvents()
    assert len(calls) == 1, "translation memory did not prevent a second AI request"
    assert window.table.rowCount() == 1
    assert any(item.thai == "อรุณสวัสดิ์" for item in window.state.rendered().translations)

    # A new line appears next to the cached one: only the new line may be sent, and the answer
    # must not wipe the cached rendering (its occurrence was never part of the request).
    blocks2 = blocks + [Block(2, "Kiro is here.", (100, 620, 1080, 60))]
    window._ocr_done(3, image, blocks2, "")
    assert wait_until(app, lambda: len(calls) == 2), "the new line was never sent"
    payload = json.loads(calls[1]["messages"][-1]["content"])
    assert [block["id"] for block in payload["blocks"]] == [2], "cached line was sent again"
    rendered = {item.id: item.thai for item in window.state.rendered().translations}
    assert rendered.get(1) == "อรุณสวัสดิ์" and rendered.get(2) is not None
    # Both accepted lines are remembered for the next session.
    assert window.store.load_memory_of_translation("เกมทดสอบ").stats()["entries"] == 2

    window.active = False
    window.close()
    assert wait_until(app, lambda: not window._workers_running(), timeout=10)
    app.processEvents()


def test_glossary_suggestion_is_only_used_after_the_user_accepts(app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from screen_thai.app import Window
    window = Window()
    window.current_profile = "เกมทดสอบ"
    window.profile.setText("เกมทดสอบ")
    window.pending_pairs = [("Ryza is late.", "ไรซ่าสายแล้ว"),
                            ("Ryza is here.", "ไรซ่าอยู่ที่นี่"),
                            ("Kiro waits.", "คิโรรออยู่")]
    window.status = window.status  # keep the real status bar
    added = window.mine_glossary()
    assert added >= 1
    assert window.glossary.toPlainText().strip() == "", "the miner must not edit the glossary"
    sources = [item["source"] for item in window.suggestions]
    assert "Ryza" in sources
    assert window.suggestions[0]["thai"] == "ไรซ่า"

    accepted = window.accept_suggestions([sources.index("Ryza")])
    assert accepted == 1
    assert "Ryza = ไรซ่า" in window.glossary.toPlainText()
    assert "Ryza" not in [item["source"] for item in window.suggestions]
    # The saved profile keeps the user's glossary, so the next session starts from it.
    profile = window.store.load_profile("เกมทดสอบ")
    assert "Ryza = ไรซ่า" in profile["glossary"]

    window.close()
    assert wait_until(app, lambda: not window._workers_running(), timeout=10)
    app.processEvents()
