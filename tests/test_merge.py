"""Subtitle line merging: join only what is clearly the same sentence by the same speaker."""
from screen_thai.layout import classify
from screen_thai.merge import merge_lines, restore_ids
from screen_thai.models import Block, Result, Translation


def wrap(text_a, text_b, gap=6):
    """A name tag above two low, wide lines: exactly the wrapped-subtitle shape."""
    blocks = [Block(0, "Alyssa", (120, 500, 160, 34)),
              Block(1, text_a, (100, 545, 1080, 60)),
              Block(2, text_b, (100, 545 + 60 + gap, 1080, 60))]
    return blocks, classify(blocks, (1280, 900))


def test_a_menu_column_is_never_joined():
    blocks = [Block(0, "▶ Attack", (150, 400, 300, 40)),
              Block(1, "▶ Defend", (150, 450, 300, 40))]
    merged, swallowed, _ = merge_lines(blocks, classify(blocks, (1280, 720)))
    assert [block.id for block in merged] == [0, 1] and swallowed == {}


def test_a_second_name_tag_blocks_the_join():
    blocks = [Block(0, "Alyssa", (120, 500, 160, 34)),
              Block(1, "The ancient seal is breaking,", (100, 545, 1080, 60)),
              Block(2, "Kiro", (120, 615, 160, 34)),
              Block(3, "so we must hurry.", (100, 660, 1080, 60))]
    merged, swallowed, _ = merge_lines(blocks, classify(blocks, (1280, 900)))
    assert swallowed == {}


def test_two_wrapped_lines_become_one_block():
    blocks, hints = wrap("The ancient seal is breaking,", "so we must hurry.")
    merged, swallowed, groups = merge_lines(blocks, hints)
    ids = [block.id for block in merged]
    assert ids == [0, 1] and swallowed == {2: 1} and groups[1] == [1, 2]
    joined = next(block for block in merged if block.id == 1)
    assert joined.text == "The ancient seal is breaking, so we must hurry."
    # The union box still covers both original lines so the overlay stays in place.
    assert joined.box[3] >= 60 + 6 + 60


def test_finished_sentence_is_not_joined():
    blocks, hints = wrap("We must hurry.", "Kiro is waiting.")
    merged, swallowed, _ = merge_lines(blocks, hints)
    assert [block.id for block in merged] == [0, 1, 2] and swallowed == {}


def test_lines_far_apart_are_not_joined():
    blocks, hints = wrap("The ancient seal is breaking,", "so we must hurry.", gap=200)
    merged, swallowed, _ = merge_lines(blocks, hints)
    assert swallowed == {}


def test_different_roles_are_not_joined():
    blocks = [Block(0, "The ancient seal is breaking,", (100, 340, 1080, 60)),
              Block(1, "so we must hurry.", (100, 404, 200, 30))]
    hints = classify(blocks, (1280, 720))
    # Force different roles to prove the guard, whatever the geometry produced.
    from dataclasses import replace
    hints = [replace(hints[0], role="subtitle"), replace(hints[1], role="hud")]
    merged, swallowed, _ = merge_lines(blocks, hints)
    assert swallowed == {} and len(merged) == 2


def test_at_most_three_lines_per_group():
    blocks = [Block(0, "One part of a very long sentence", (100, 545, 1080, 60)),
              Block(1, "that continues on this line", (100, 611, 1080, 60)),
              Block(2, "and even further down here", (100, 677, 1080, 60)),
              Block(3, "until it finally stops", (100, 743, 1080, 60))]
    hints = classify(blocks, (1280, 900))
    merged, swallowed, groups = merge_lines(blocks, hints)
    # MAX_LINES caps the group at three: the fourth line stays a block of its own.
    assert groups[0] == [0, 1, 2] and swallowed == {1: 0, 2: 0}
    assert [block.id for block in merged] == [0, 3]


def test_restore_ids_drops_swallowed_translations():
    result = Result([Translation(1, "ก", ""), Translation(2, "ข", "")])
    kept = restore_ids(result, {1: [1, 2]}, keep={1, 5})
    assert [translation.id for translation in kept.translations] == [1]
