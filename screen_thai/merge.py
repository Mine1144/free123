"""Merge wrapped subtitle lines into one sentence before translating.

JRPG/VN text boxes wrap a single sentence over two or three lines. Translating each line alone
produces broken Thai and wastes requests, so adjacent lines of the same role and speaker are
joined into one block. The merge is deliberately conservative — joining the wrong boxes (a HUD
timer under a subtitle, two different speakers) is worse than leaving them apart.

The joined block keeps the **first** line's id and a union box, so the overlay still covers the
whole original text; the swallowed ids are reported so the caller can ignore them.
"""
from __future__ import annotations

from dataclasses import replace

_FINAL = tuple(".!?…。！？")
_CLOSERS = tuple("\"'’”»)]}」』")
# Menus are never merged: an option list looks geometrically identical to wrapped dialogue,
# and joining two options would change their meaning.
MERGEABLE_ROLES = ("dialogue", "subtitle")
MAX_LINES = 3
MAX_CHARS = 400


def _ends_sentence(text: str) -> bool:
    return text.rstrip().rstrip("".join(_CLOSERS)).endswith(_FINAL)


def _union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _h_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    right = min(a[0] + a[2], b[0] + b[2])
    overlap = max(0, right - left)
    return overlap / max(1, min(a[2], b[2]))


def can_join(previous, current, previous_role: str, current_role: str,
             previous_speaker: str, current_speaker: str, lines: int) -> bool:
    if previous_role != current_role or previous_role not in MERGEABLE_ROLES:
        return False
    # A wrapped line usually has no name tag of its own: the tag above the first line speaks for
    # the whole group. A line that names a *different* speaker must never be joined.
    if current_speaker and (not previous_speaker or current_speaker != previous_speaker):
        return False
    if lines >= MAX_LINES:
        return False
    text = previous.text.strip()
    if not text or _ends_sentence(text):
        return False
    if len(text) + len(current.text) > MAX_CHARS:
        return False
    if text.startswith(("-", "—", "•")) and not current.text.strip():
        return False
    px, py, pw, ph = previous.box
    cx, cy, cw, ch = current.box
    if cy + ch <= py or py + ph <= cy:
        gap = max(py, cy) - min(py + ph, cy + ch)
    else:
        gap = 0
    if gap > 0.9 * max(ph, ch):
        return False
    return _h_overlap(previous.box, current.box) >= 0.35


def merge_lines(blocks, hints) -> tuple[list, list, dict]:
    """Return ``(merged_blocks, swallowed, groups)``.

    ``swallowed`` maps a removed block id to the id that now carries its text so callers can drop
    stale state; ``groups`` maps the surviving id to every id it represents.
    """
    hint_by_id = {hint.id: hint for hint in (hints or [])}
    ordered = sorted(blocks, key=lambda block: (block.box[1], block.box[0]))
    merged: list = []
    swallowed: dict[int, int] = {}
    groups: dict[int, list[int]] = {}

    for block in ordered:
        role = hint_by_id[block.id].role if block.id in hint_by_id else "unknown"
        speaker = hint_by_id[block.id].speaker if block.id in hint_by_id else ""
        if merged:
            last = merged[-1]
            last_role = hint_by_id[last.id].role if last.id in hint_by_id else "unknown"
            last_speaker = hint_by_id[last.id].speaker if last.id in hint_by_id else ""
            if can_join(last, block, last_role, role, last_speaker, speaker,
                        len(groups.get(last.id, [last.id]))):
                merged[-1] = replace(last, text=f"{last.text.rstrip()} {block.text.strip()}",
                                     box=_union(last.box, block.box))
                swallowed[block.id] = last.id
                groups.setdefault(last.id, [last.id]).append(block.id)
                continue
        merged.append(block)
        groups.setdefault(block.id, [block.id])
    return merged, swallowed, groups


def restore_ids(result, groups: dict, keep: set[int]):
    """Drop translations for swallowed ids and keep only blocks that still exist."""
    kept = [translation for translation in result.translations if translation.id in keep]
    return replace(result, translations=kept)
