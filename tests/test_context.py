import json

from screen_thai import context
from screen_thai.dialogue import DialogueMemory
from screen_thai.faces import FaceMemory, FaceTracker
from screen_thai.models import Block, Settings
from screen_thai.vision import Appearance, ExpressionCues, FaceBox, FaceObservation
from screen_thai.layout import classify


def blocks():
    return [Block(0, "Alyssa", (120, 480, 160, 34)),
            Block(1, "あたしが守るわ。", (100, 520, 1080, 150))]


def memory_with_alyssa():
    memory = FaceMemory()
    memory.ensure("Alyssa", role="แม่", relation="พ่อแม่ลูก", confirmed=True)
    memory.set_gender("Alyssa", "female", "user")
    memory.set_voice("Alyssa", self="แม่", address="ลูก", particles="ค่ะ", register="กันเอง",
                     locked=True, source="user")
    return memory


def face_observation(track: int = 1, character: str = "", possible: str = "") -> FaceObservation:
    observation = FaceObservation(
        track, FaceBox(100, 100, 120, 120),
        ExpressionCues("ยิ้ม", 0.4, 0.2, 0.15, 0.5, 0.2, 3.0, 0.1, "landmarks"),
        Appearance("#221111", "#553322", "#334455", ("#221111",), 0.5, 0.3, 0.2))
    observation.character = character
    observation.match = 0.9
    observation.possible = possible
    observation.possible_match = 0.8
    return observation


def test_profile_context_labels_trust_levels():
    settings = Settings()
    profile = {"notes": "โน้ตผู้ใช้", "glossary": "Ryza = ไรซ่า", "summary": "สรุปเดิม",
               "relations": [{"from": "A", "to": "B", "relation": "พี่น้อง", "evidence": "เรียกพี่",
                              "confidence": 0.9}],
               "research": {"title": "Atelier", "summary": "เด็กสาวนักเล่นแร่แปรธาตุ"}}
    data = context.profile_context(settings, profile, memory_with_alyssa())
    assert data["manual_notes"] == "โน้ตผู้ใช้"
    assert "Alyssa" in data["characters_confirmed_by_user"]
    assert data["characters_confirmed_by_user"]["Alyssa"]["gender_confirmed"] == "female"
    assert data["characters_confirmed_by_user"]["Alyssa"]["locked_voice"]["self"] == "แม่"
    assert "ยังไม่ยืนยัน" in data["game_research_unverified"]


def test_frame_context_includes_locked_plan_and_layout():
    settings = Settings()
    memory = memory_with_alyssa()
    frame_hints = classify(blocks(), (1280, 720))
    dialogue = DialogueMemory()
    dialogue.add("Alyssa", "เจ้ากลับมาแล้วสินะ", speaker_source="name_tag")
    frame = context.frame_context(settings, blocks(), frame_hints, [], memory, dialogue)
    assert frame["visible_name_tags"]["1"] == "Alyssa"
    plan = frame["speech_plans"]["1"]
    assert plan["must_follow"] is True
    assert "ล็อกโดยผู้ใช้" in plan["plan"]
    assert plan["source_evidence"]["gender_hint_in_source"] == "female"
    assert frame["recent_dialogue"][0]["speaker"] == "Alyssa"
    assert "Alyssa" in frame["continuity"]


def test_unconfirmed_face_suggestion_is_never_asserted():
    settings = Settings()
    frame = context.frame_context(settings, blocks(), classify(blocks(), (1280, 720)),
                                  [face_observation(possible="Mio")], FaceMemory())
    entry = frame["faces"][0]
    assert "possible_character" in entry and "identified_as" not in entry
    assert "ห้ามใส่ชื่อนี้ในบทแปล" in entry["must_not_assert"]
    assert entry["expression_cues"]["note"].startswith("ค่าที่วัดได้")


def test_confirmed_face_is_reported_as_user_confirmed():
    settings = Settings()
    frame = context.frame_context(settings, blocks(), classify(blocks(), (1280, 720)),
                                  [face_observation(character="Alyssa")], memory_with_alyssa())
    entry = frame["faces"][0]
    assert entry["identified_as"] == "Alyssa"
    assert entry["identified_source"] == "ผู้ใช้ยืนยัน"


def test_expressions_are_marked_as_weak_evidence():
    settings = Settings()
    frame = context.frame_context(settings, blocks(), classify(blocks(), (1280, 720)),
                                  [face_observation()], FaceMemory())
    assert "ไม่ใช่ผลตรวจอารมณ์" in frame["faces"][0]["expression_cues"]["note"]


def test_trim_reduces_payload_to_limit():
    big = {"characters_confirmed_by_user": {f"C{index}": {"role": "x" * 400} for index in range(40)},
           "recent_dialogue": [{"speaker": "A", "thai": "ย" * 200} for _ in range(40)],
           "manual_notes": "ก" * 5000, "layout": [{"id": i} for i in range(50)]}
    trimmed = context.trim(big, limit=2000)
    assert len(json.dumps(trimmed, ensure_ascii=False)) <= 2000


def test_no_face_warning_prevents_guessing_speaker_from_face():
    settings = Settings()
    no_tag = [Block(1, "あたしが守るわ。", (100, 520, 1080, 150))]
    frame = context.frame_context(settings, no_tag, classify(no_tag, (1280, 720)),
                                  [face_observation()], FaceMemory())
    assert "faces_note" in frame
    assert "ห้ามเดาชื่อ" in frame["faces_note"]


def test_plan_for_uses_memory_cards():
    memory = memory_with_alyssa()
    from screen_thai import speech
    plan = context.plan_for("Alyssa", "", speech.analyse_source("私が守るわ。"), "กังวล", memory)
    assert plan.self_pronoun == "แม่"
    assert plan.locked


def test_tracker_is_used_for_stable_trends():
    from screen_thai.vision import ExpressionCues
    tracker = FaceTracker()
    track = tracker.assign([(10, 10, 50, 50)])[0]
    memory = FaceMemory()
    for mouth in (0.1, 0.12, 0.5, 0.15, 0.55):
        memory.note_cues(track, ExpressionCues("ไม่ทราบ", mouth, mouth, 0.0, 0.5, 0.1, 0.0, 0.0))
    trend = memory.trend(track)
    assert "ปากขยับ" in trend or "ปากเปิดมากกว่าปกติ" in trend
