import json

import pytest

from screen_thai.models import Block, Result, Settings, Translation, parse_result
from screen_thai.state import FrameState
from screen_thai.storage import Store


def block(i=0, text="Hello", x=10, y=10):
    return Block(i, text, (x, y, 100, 30))


def test_untrusted_model_output():
    output = {"translations": [{"id": 0, "thai": "สวัสดี"}, {"id": 0, "thai": "ซ้ำ"},
                              {"id": 900, "thai": "ปลอม"}, {"id": False, "thai": "ไม่ใช่เลข"}],
              "scene": {"relations": [{"from": "A", "to": "B", "relation": "แม่", "confidence": 0.95},
                                      {"from": "A", "to": "B", "relation": "แม่", "evidence": "Mom",
                                       "confidence": 0.9}], "people": "not a list"}}
    result = parse_result("```json\n" + json.dumps(output) + "\n```", [block()])
    assert result.translations == [Translation(0, "สวัสดี")]
    assert len(result.scene.relations) == 1
    assert result.scene.people == []


@pytest.mark.parametrize("raw", ['[]', '{}', 'not json', '{"translations":42}', 'x' * 100001])
def test_bad_json(raw):
    with pytest.raises(ValueError):
        parse_result(raw, [block()])


def test_motion_and_id_reorder_keep_translation():
    state = FrameState()
    state.update([block()])
    submitted = state.snapshot()
    state.update([block(5, x=13)])
    assert state.accept(submitted, Result([Translation(0, "สวัสดี")]))
    assert state.rendered().translations == [Translation(5, "สวัสดี")]
    assert not state.needs_translation()


def test_disappeared_and_returned_phrase_rejects_old_answer():
    state = FrameState()
    state.update([block()])
    submitted = state.snapshot()
    state.update([])
    state.update([block()])
    assert not state.accept(submitted, Result([Translation(0, "สวัสดี")]))
    assert state.needs_translation()
    assert state.rendered().translations == []


def test_changing_hud_cannot_starve_subtitle():
    state = FrameState()
    state.update([block(), block(1, "Quest 1", y=80)])
    submitted = state.snapshot()
    for i in range(10):
        state.update([block(), block(1, f"Quest {i+2}", y=80)])
    assert state.accept(submitted, Result([Translation(0, "สวัสดี"), Translation(1, "ภารกิจเก่า")]))
    assert state.rendered().translations == [Translation(0, "สวัสดี")]
    assert state.needs_translation()


def test_same_phrase_different_speakers_not_conflated():
    state = FrameState()
    state.update([block(), block(1, y=90)])
    state.accept(state.snapshot(), Result([Translation(0, "ครับ"), Translation(1, "ค่ะ")]))
    assert [t.thai for t in state.rendered().translations] == ["ครับ", "ค่ะ"]


def test_skipped_blocks_not_repeated_forever():
    state = FrameState()
    state.update([block()])
    state.accept(state.snapshot(), Result())
    assert not state.needs_translation()
    state.update([block(text="Goodbye")])
    assert state.needs_translation()


def test_profile_isolation_and_path_safety(tmp_path):
    store = Store(tmp_path)
    profile = store.load_profile("../../outside")
    profile["notes"] = "แม่กับลูก"
    store.save_profile("../../outside", profile)
    assert store.profile_path("../../outside").parent == tmp_path / "profiles"
    assert store.load_profile("../../outside")["notes"] == "แม่กับลูก"
    assert store.load_profile("other")["notes"] == ""
    (tmp_path / "settings.json").write_text("broken")
    assert store.load_settings() == Settings()


def test_settings_validation_and_no_key_storage(tmp_path):
    settings = Settings.from_dict({"interval_ms": -999, "opacity": 9999, "max_blocks": 0,
                                   "vision": "false", "key": "secret", "provider": "invalid"})
    assert settings.interval_ms == 500 and settings.opacity == 255 and settings.max_blocks == 1
    assert settings.vision is True and settings.provider == "ollama"
    store = Store(tmp_path)
    store.save_settings(settings)
    assert "secret" not in (tmp_path / "settings.json").read_text()


def test_learning_keeps_manual_notes(tmp_path):
    store = Store(tmp_path)
    p = store.load_profile("game")
    p["notes"] = "Use หนู/แม่"
    store.save_profile("game", p)
    result = parse_result(json.dumps({"translations": [], "scene": {"summary": "สรุป"}}), [])
    store.learn("game", result)
    assert store.load_profile("game")["notes"] == "Use หนู/แม่"
    assert store.load_profile("game")["summary"] == "สรุป"
