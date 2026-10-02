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
    window.latest_image = image
    window.overlay.settings = window.settings      # what start() would configure
    window.state.update(blocks)
    window._ocr_done(7, image, blocks, "")

    observation = FaceObservation(1, FaceBox(700, 120, 160, 170),
                                  ExpressionCues("ยิ้ม", 0.5, 0.1, 0.2, 0.6, 0.2, 2.0, 0.1),
                                  Appearance("#332211", "#554433", "#223344", ("#332211",), 0.5,
                                             0.3, 0.1))
    observation.character = "Alyssa"
    observation.match = 0.91

    # A busy vision worker is skipped rather than queued; inject the frame it would return.
    window.vision_busy = True
    window._vision_done(7, window.job_id, [observation], "")

    assert wait_until(app, lambda: window.table.rowCount() > 0), "AI result never arrived"

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
