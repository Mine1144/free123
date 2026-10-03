from screen_thai.layout import apply_inline_speakers, choices, classify, describe, speakers
from screen_thai.models import Block

SIZE = (1280, 720)


def block(i, text, x, y, w, h):
    return Block(i, text, (x, y, w, h))


def vn_frame():
    return [
        block(0, "HP 250/250", 20, 20, 120, 24),
        block(1, "Alyssa", 120, 480, 160, 34),
        block(2, "I will protect you, no matter what happens next.", 100, 520, 1080, 150),
        block(3, "Quest: Find the key", 20, 660, 260, 26),
    ]


def test_vn_dialogue_name_tag_and_hud():
    hints = classify(vn_frame(), SIZE)
    roles = {hint.id: hint.role for hint in hints}
    assert roles[2] == "dialogue"
    assert roles[1] == "name_tag"
    assert roles[0] == "hud" and roles[3] == "hud"
    assert speakers(vn_frame(), hints) == {2: "Alyssa"}


def test_name_inside_dialogue_box_head():
    blocks = [block(0, "Kiro", 130, 530, 120, 30),
              block(1, "Then we go together.", 100, 520, 900, 150)]
    hints = classify(blocks, SIZE)
    assert {hint.id: hint.role for hint in hints}[0] == "name_tag"
    assert speakers(blocks, hints) == {1: "Kiro"}


def test_choice_menu_detected_as_group():
    blocks = [block(0, "Yes, I will go", 500, 300, 260, 40),
              block(1, "No, not yet", 500, 350, 260, 40),
              block(2, "Ask about mother", 500, 400, 260, 40)]
    hints = classify(blocks, SIZE)
    assert [hint.role for hint in hints] == ["choice", "choice", "choice"]
    assert [text for _id, text in choices(blocks, hints)] == [
        "Yes, I will go", "No, not yet", "Ask about mother"]


def test_inline_speaker_pattern():
    blocks = [block(0, "Grace: Do you remember me?", 200, 540, 800, 120)]
    hints = apply_inline_speakers(blocks, classify(blocks, SIZE))
    assert hints[0].speaker == "Grace"
    assert speakers(blocks, hints) == {0: "Grace"}


def test_reading_order_and_describe_is_compact():
    blocks = vn_frame()
    hints = classify(blocks, SIZE)
    order = [hint.id for hint in hints if hint.role == "dialogue"]
    assert order == [2]
    described = describe(blocks, hints)
    assert {entry["id"]: entry["role"] for entry in described}[2] == "dialogue"
    assert all(len(entry) <= 3 for entry in described)
    assert any(entry.get("visible_name_tag") == "Alyssa" for entry in described)


def test_thai_text_is_not_mistaken_for_a_name_tag():
    blocks = [block(0, "ผมจะปกป้องเธอเอง", 120, 480, 400, 40),
              block(1, "แล้วเราจะไปด้วยกัน", 100, 540, 900, 140)]
    hints = classify(blocks, SIZE)
    # A long Thai line above the box must not become a speaker name.
    assert speakers(blocks, hints) == {}
