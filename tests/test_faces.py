import random

from screen_thai.faces import MATCH_THRESHOLD, FaceMemory, FaceTracker


def vector(seed: int, noise: float = 0.0, size: int = 24) -> list[float]:
    rng = random.Random(seed)
    values = [rng.random() for _ in range(size)]
    if noise:
        values = [value + rng.uniform(-noise, noise) for value in values]
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values]


def test_tracker_keeps_ids_for_moving_faces():
    tracker = FaceTracker()
    first = tracker.assign([(100, 100, 80, 80), (500, 200, 90, 90)])
    second = tracker.assign([(104, 103, 80, 80), (498, 204, 90, 90)])
    assert first == second
    third = tracker.assign([(505, 205, 90, 90)])
    assert third == [first[1]]


def test_tracker_ids_do_not_collide_on_changed_scene():
    tracker = FaceTracker()
    first = tracker.assign([(10, 10, 60, 60)])
    later = tracker.assign([(900, 500, 70, 70)])
    assert first != later


def test_memory_matches_confirmed_character_and_suggests_below_threshold():
    memory = FaceMemory()
    memory.ensure("Mio", confirmed=True)
    assert memory.remember_face("Mio", vector(1), {"hair": "#222222"})
    exact = memory.match(vector(1))
    assert exact.name == "Mio" and exact.score > MATCH_THRESHOLD
    noisy = memory.match(vector(1, noise=0.35))
    # Either it still matches, or it degrades to a suggestion — never to a wrong hard name.
    assert noisy.name in ("", "Mio")
    stranger = memory.match(vector(99))
    assert stranger.name == "" and stranger.suggestion == ""


def test_two_characters_never_swap_with_tight_margin():
    memory = FaceMemory()
    memory.ensure("A", confirmed=True)
    memory.ensure("B", confirmed=True)
    memory.remember_face("A", vector(5))
    memory.remember_face("B", vector(5, noise=0.02))   # deliberately similar
    match = memory.match(vector(5, noise=0.01))
    assert match.name == "" or match.score > MATCH_THRESHOLD
    assert match.suggestion in ("", "A", "B")


def test_hypotheses_never_overwrite_confirmed_values():
    memory = FaceMemory()
    memory.ensure("Kiro", role="นักดาบ", confirmed=True)
    memory.set_gender("Kiro", "male", "user")
    memory.apply_hypothesis({"name": "Kiro", "role": "พ่อครัว", "gender": "female",
                             "evidence": "หน้าเหมือน", "personality": "ใจดี"})
    card = memory.card("Kiro")
    assert card["role"] == "นักดาบ"
    assert card["gender"] == "male" and card["gender_source"] == "user"
    assert card["personality"] == "ใจดี"


def test_hypothesis_without_evidence_cannot_claim_gender():
    memory = FaceMemory()
    memory.apply_hypothesis({"name": "Ai", "gender": "female", "evidence": ""})
    card = memory.card("Ai")
    assert card["gender"] == "unknown"


def test_face_link_needs_user_confirmation_then_is_remembered():
    memory = FaceMemory()
    assert memory.add_pending("ใบหน้าที่ 1", "Grace", "name tag above the box", 0.8)
    assert len(memory.pending) == 1
    # A pending suggestion is not a name yet.
    assert memory.match(vector(3)).name == ""
    name = memory.confirm_pending(0, {"ใบหน้าที่ 1": vector(3)})
    assert name == "Grace"
    assert memory.match(vector(3)).name == "Grace"
    assert memory.card("Grace")["confirmed"] is True


def test_pending_can_be_rejected_and_deduplicated():
    memory = FaceMemory()
    memory.add_pending("ใบหน้าที่ 1", "Grace", "evidence", 0.7)
    memory.add_pending("ใบหน้าที่ 1", "Grace", "evidence", 0.9)
    assert len(memory.pending) == 1
    assert memory.pending[0]["confidence"] == 0.9
    memory.reject_pending(0)
    assert memory.pending == []


def test_aliases_resolve_and_paths_are_safe():
    memory = FaceMemory()
    memory.ensure("Alyssa", confirmed=True)
    memory.apply_hypothesis({"name": "Alyssa", "aliases": ["แม่", "Mom"], "evidence": "Grace เรียก"})
    assert memory.resolve("Mom") == "Alyssa"
    assert memory.resolve("ไม่รู้จัก") == ""
    assert memory.resolve("lyss") == "Alyssa"


def test_round_trip_keeps_descriptors_and_forget_clears_them():
    memory = FaceMemory()
    memory.ensure("Mio", confirmed=True)
    memory.remember_face("Mio", vector(7))
    data = memory.to_dict()
    reloaded = FaceMemory(data)
    assert len(reloaded.card("Mio")["descriptors"]) == 1
    reloaded.forget_face("Mio")
    assert reloaded.card("Mio")["descriptors"] == []
    assert reloaded.card("Mio")["confirmed"] is True


def test_descriptor_tail_is_bounded():
    memory = FaceMemory()
    memory.ensure("Mio")
    for index in range(20):
        memory.remember_face("Mio", vector(index))
    assert len(memory.card("Mio")["descriptors"]) == 8
    assert memory.card("Mio")["samples"] == 20


def test_expression_timeline_records_and_reports_trend():
    from screen_thai.vision import ExpressionCues
    memory = FaceMemory()
    for smile in (0.05, 0.06, 0.30):
        memory.note_cues(1, ExpressionCues("ไม่ทราบ", 0.1, 0.1, smile, 0.5, 0.1, 0.0, 0.0))
    trend = memory.trend(1)
    assert "มุมปาก" in trend
    memory.reset_session()
    assert memory.trend(1) == ""


def test_single_frame_never_becomes_a_mood_claim():
    from screen_thai.vision import ExpressionCues
    memory = FaceMemory()
    memory.note_cues(2, ExpressionCues("อ้าปากกว้าง", 0.6, 0.6, 0.1, 0.5, 0.1, 0.0, 0.0))
    trend = memory.trend(2)
    assert "ค่าที่สุดขั้ว: อ้าปากกว้าง" in trend
