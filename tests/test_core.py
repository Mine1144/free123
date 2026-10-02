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


def test_new_translation_fields_are_parsed_and_clamped():
    output = {"translations": [{"id": 0, "thai": "สวัสดี", "speaker": "Grace", "listener": "Kiro",
                               "emotion": "เศร้า", "delivery": "พูดช้า ๆ", "self_pronoun": "หนู",
                               "address_pronoun": "พี่", "particles": "ค่ะ", "confidence": 0.8,
                               "emotion_typo": "โกรธ"},
                              {"id": 0, "thai": "ซ้ำ", "emotion": "ไม่ใช่ค่าที่อนุญาต"}]}
    result = parse_result(json.dumps(output), [block()])
    line = result.translations[0]
    assert (line.speaker, line.listener, line.emotion) == ("Grace", "Kiro", "เศร้า")
    assert line.particles == "ค่ะ" and round(line.confidence, 2) == 0.8
    assert len(result.translations) == 1


def test_character_updates_need_evidence_and_never_claim_gender_without_it():
    output = {"translations": [], "scene": {"characters": [
        {"name": "Kiro", "role": "นักดาบ", "evidence": "Kiro: I will protect you", "confidence": 0.9,
         "gender": "male"},
        {"name": "ไม่มีหลักฐาน", "role": "?", "evidence": "", "confidence": 0.95},
        {"name": "เดาเพศ", "evidence": "หน้าตาเหมือนผู้หญิง", "confidence": 0.9, "gender": "female"}]}}
    result = parse_result(json.dumps(output), [])
    names = [card["name"] for card in result.scene.characters]
    assert names == ["Kiro", "เดาเพศ"]
    assert result.scene.characters[0]["gender"] == "male"


def test_face_links_require_evidence_and_a_known_or_visible_name():
    blocks = [Block(0, "Grace", (0, 0, 10, 10)), Block(1, "Hello", (0, 20, 10, 10))]
    output = {"translations": [], "scene": {"faces": [
        {"track": "ใบหน้าที่ 1", "character": "Grace", "evidence": "name tag above her", "confidence": 0.9},
        {"track": "ใบหน้าที่ 2", "character": "นักฆ่า", "evidence": "", "confidence": 0.9},
        {"track": "ใบหน้าที่ 3", "character": "ไม่รู้จัก", "evidence": "maybe", "confidence": 0.99},
        {"track": "ใบหน้าที่ 1", "character": "Mio", "evidence": "Mio: hey", "confidence": 0.5}]}}
    result = parse_result(json.dumps(output), blocks, characters=("Mio",))
    assert [link["character"] for link in result.scene.faces] == ["Grace"]


def test_expressions_need_cues_or_evidence_and_are_capped():
    output = {"translations": [], "scene": {"expressions": [
        {"character": "Grace", "expression": "ยิ้ม", "cues": "มุมปากโค้งขึ้น 0.2", "confidence": 0.6},
        {"character": "Grace", "expression": "ยิ้ม", "confidence": 0.6},
        {"character": "Kiro", "expression": "ไม่ใช่ค่าที่อนุญาต", "cues": "x", "confidence": 0.9},
        {"character": "Kiro", "expression": "โกรธ", "cues": "คิ้วขมวด", "confidence": 0.1}]}}
    result = parse_result(json.dumps(output), [])
    assert [item["expression"] for item in result.scene.expressions] == ["ยิ้ม"]


def test_settings_new_fields_are_validated():
    settings = Settings.from_dict({"face_backend": "ไม่รู้จัก", "face_sample_ms": 10,
                                   "max_faces": 99, "research_search": "google",
                                   "research_max_sources": 50, "show_speaker_label": True})
    assert settings.face_backend == "auto"
    assert settings.face_sample_ms == 400 and settings.max_faces == 8
    assert settings.research_search == "off" and settings.research_max_sources == 12
    assert settings.show_speaker_label is True
    assert settings.to_dict()["research_auto"] is True


def test_translation_quality_settings_are_validated():
    settings = Settings.from_dict({"review_max_lines": 9999, "merge_lines": "yes",
                                   "translation_memory": False, "glossary_mine": True,
                                   "review_consistency": True})
    assert settings.review_max_lines == 40          # clamped, not accepted blindly
    assert settings.merge_lines is True             # wrong type falls back to the default
    assert settings.translation_memory is False and settings.review_consistency is True
    assert settings.glossary_mine is True


def test_translation_memory_is_bounded_on_the_way_in():
    from screen_thai.memory import TranslationMemory
    raw = {"entries": {f"line {index}": {"thai": "ท" * (index + 1)} for index in range(900)}}
    tm = TranslationMemory(raw)
    assert len(tm.entries) <= 800
    assert tm.to_dict()["entries"]


def test_translation_memory_survives_a_profile_round_trip(tmp_path):
    from screen_thai.faces import FaceMemory
    from screen_thai.memory import TranslationMemory
    from screen_thai.storage import Store
    store = Store(tmp_path)
    tm = TranslationMemory()
    tm.learn("Wait here", "รอที่นี่")
    store.save_translation_memory("เกม", tm)
    store.save_memory("เกม", FaceMemory())
    again = store.load_memory_of_translation("เกม")
    assert again.get("Wait here")["thai"] == "รอที่นี่"
    store.save_suggestions("เกม", [{"source": "Ryza", "thai": "ไรซ่า"}])
    assert store.load_suggestions("เกม")[0]["source"] == "Ryza"
