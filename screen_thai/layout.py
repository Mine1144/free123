"""Deterministic layout reading of one captured frame.

The model must never be asked to guess where a name tag is: geometry is cheap, local and
repeatable. These hints tell the AI which block is dialogue, who the visible name tag says
is speaking, and which blocks form a choice menu.
"""
from __future__ import annotations

import re

from .models import Block, BlockHint

ROLES = ("dialogue", "subtitle", "name_tag", "choice", "menu", "item", "tooltip", "hud", "unknown")
ROLE_ORDER = {role: index for index, role in enumerate(ROLES)}

INLINE_SPEAKER = re.compile(r"^([^\s:：]{1,20})\s*[:：]\s*(\S.*)$", re.S)
NUMBERISH = re.compile(r"^(?=.*\d)[\d\s%./:+\-*×x,()]*$")
SENTENCE_END = ("。", "！", "？", "…", ".", "!", "?", "!", "?")
# Menus mark their options; wrapped dialogue usually starts with plain text.
CHOICE_MARKERS = ("▶", "►", "▸", ">", "»", "・", "·", "*", "•", "●", "○", "◆", "◇", "-", "–")


def _inside(box, other, pad: float = 0.0):
    x, y, w, h = box
    xx, yy, ww, hh = other
    return (x >= xx - pad and y >= yy - pad and x + w <= xx + ww + pad and y + h <= yy + hh + pad)


def _overlap_x(a, b) -> float:
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    return overlap / max(1, min(aw, bw))


def classify(blocks: list[Block], size: tuple[int, int]) -> list[BlockHint]:
    """Return one hint per block, ordered by reading order."""
    width, height = max(1, size[0]), max(1, size[1])
    rows: dict[int, dict] = {}
    for block in blocks:
        x, y, w, h = block.box
        rows[block.id] = {
            "block": block, "text": block.text.strip(), "x": x, "y": y, "w": w, "h": h,
            "rw": w / width, "rh": (y + h) / height, "top": y / height, "rx": x / width,
            "ar": w / max(1, h),
            "digits": bool(NUMBERISH.match(block.text.strip())),
            "len": len(block.text.strip()),
        }

    for row in rows.values():
        text, rw, rh, ar = row["text"], row["rw"], row["rh"], row["ar"]
        if len(text) < 2 or row["digits"]:
            continue
        if rw >= 0.42 and rh >= 0.68 and ar >= 3.2:
            row["role"] = "dialogue"
        elif (0.2 <= rw <= 0.85 and rh >= 0.86 and ar >= 2.5
              and abs(row["rx"] + rw / 2 - 0.5) <= 0.3):
            row["role"] = "subtitle"
        elif rw >= 0.3 and 0.55 <= rh <= 0.9 and row["len"] <= 60 and ar >= 3:
            row["role"] = "dialogue"

    _mark_name_tags(rows)
    _mark_choices(rows)
    _mark_hud(rows)

    hints = []
    for row in rows.values():
        role = row.get("role", "unknown")
        hints.append(BlockHint(row["block"].id, role, row.get("speaker", ""), 0))
    hints.sort(key=_reading_key)
    return [BlockHint(hint.id, hint.role, hint.speaker, index)
            for index, hint in enumerate(hints)]


def _mark_name_tags(rows: dict[int, dict]):
    dialogues = [row for row in rows.values() if row.get("role") in ("dialogue", "subtitle")]
    for row in rows.values():
        if row.get("role") or row["len"] > 24 or row["top"] < 0.4 or row["rh"] < 0.4:
            continue
        if row["h"] / max(1, row["w"]) > 0.5 or row["text"].endswith(SENTENCE_END):
            continue
        for dialogue in dialogues:
            box, target = (row["x"], row["y"], row["w"], row["h"]), (dialogue["x"], dialogue["y"],
                                                                     dialogue["w"], dialogue["h"])
            gap = dialogue["y"] - (row["y"] + row["h"])
            above = -int(0.03 * dialogue["h"]) <= gap <= int(0.6 * dialogue["h"])
            head_of_box = _inside(box, target, pad=int(0.05 * dialogue["w"])) and \
                row["y"] + row["h"] <= dialogue["y"] + 0.75 * dialogue["h"]
            if (above or head_of_box) and _overlap_x(box, target) >= 0.25:
                row["role"] = "name_tag"
                dialogue["speaker"] = dialogue.get("speaker") or row["text"]
                break


def _wrapped_continuation(previous: str, current: str) -> bool:
    """True when two stacked lines read as one wrapped sentence rather than two menu options.

    Menu options are short and/or carry a marker; a continuation of the same sentence usually
    starts with a plain lowercase word, a comma, or a long unbroken run of text.
    """
    previous, current = previous.strip(), current.strip()
    if not previous or not current:
        return False
    if current.startswith(CHOICE_MARKERS):
        return False
    if previous.endswith(SENTENCE_END):
        return False
    if current[:1].isupper() and len(current) < 25:
        return False
    return len(current) >= 12 or len(previous) >= 25


def _name_tag_above(rows: dict[int, dict], group: list[dict]) -> bool:
    """A name tag sitting right above a stacked group means it is dialogue, not a menu."""
    first = group[0]
    for row in rows.values():
        if row.get("role") != "name_tag":
            continue
        gap = first["y"] - (row["y"] + row["h"])
        if -int(0.05 * first["h"]) <= gap <= int(0.7 * first["h"]) and \
                _overlap_x((row["x"], row["y"], row["w"], row["h"]),
                           (first["x"], first["y"], first["w"], first["h"])) >= 0.25:
            return True
    return False


def _mark_choices(rows: dict[int, dict]):
    stacked = [row for row in rows.values() if not row.get("role")
               or row.get("role") in ("dialogue", "unknown")]
    stacked = [row for row in stacked if 0.15 <= row["top"] <= 0.85 and row["len"] <= 90]
    columns: dict[int, list[dict]] = {}
    for row in stacked:
        columns.setdefault(round(row["x"] / 40), []).append(row)
    for group in columns.values():
        if not 2 <= len(group) <= 6:
            continue
        group.sort(key=lambda item: item["y"])
        gaps = [group[i + 1]["y"] - (group[i]["y"] + group[i]["h"]) for i in range(len(group) - 1)]
        heights = [item["h"] for item in group]
        even = max(heights) <= 2.6 * max(1, min(heights))
        if not (even and all(-0.02 * heights[0] <= gap <= 3.0 * heights[0] for gap in gaps)):
            continue
        if _name_tag_above(rows, group):
            continue
        if any(_wrapped_continuation(group[index]["text"], group[index + 1]["text"])
               for index in range(len(group) - 1)):
            continue
        for row in group:
            row["role"] = "choice"


def _mark_hud(rows: dict[int, dict]):
    for row in rows.values():
        if row.get("role"):
            continue
        left = row["rx"]
        right = row["rx"] + row["rw"]
        if row["digits"] or row["top"] <= 0.14:
            row["role"] = "hud"
        elif (left <= 0.06 or right >= 0.94) and row["len"] <= 30:
            row["role"] = "hud"
        elif row["rh"] >= 0.94 and row["rw"] <= 0.3:
            row["role"] = "hud"
        elif row["rw"] <= 0.06 and row["len"] <= 8:
            row["role"] = "item"


def _reading_key(hint: BlockHint):
    return (ROLE_ORDER.get(hint.role, 9), hint.id)


def apply_inline_speakers(blocks: list[Block], hints: list[BlockHint]) -> list[BlockHint]:
    """Detect "Name: text" inside a dialogue block and re-read it as name tag + line."""
    lookup = {hint.id: hint for hint in hints}
    index = {block.id: block for block in blocks}
    output = []
    for hint in hints:
        block = index.get(hint.id)
        if hint.role not in ("dialogue", "unknown") or block is None:
            continue
        match = INLINE_SPEAKER.match(block.text.strip())
        if match and len(match.group(1)) <= 20 and not NUMBERISH.match(match.group(1)):
            output.append(BlockHint(hint.id, "dialogue", match.group(1).strip(), hint.order))
            lookup[hint.id] = output[-1]
    return [lookup.get(hint.id, hint) for hint in hints]


def speakers(blocks: list[Block], hints: list[BlockHint]) -> dict[int, str]:
    """Map dialogue block id -> speaker name taken from a visible name tag only."""
    text = {block.id: block.text.strip() for block in blocks}
    found = {}
    for hint in hints:
        if hint.role in ("dialogue", "subtitle") and hint.speaker:
            found[hint.id] = hint.speaker
    return {bid: name for bid, name in found.items() if name and name != text.get(bid, "")}


def choices(blocks: list[Block], hints: list[BlockHint]) -> list[tuple[int, str]]:
    text = {block.id: block.text for block in blocks}
    return [(hint.id, text[hint.id]) for hint in hints if hint.role == "choice" and hint.id in text]


def describe(blocks: list[Block], hints: list[BlockHint], limit: int = 40) -> list[dict]:
    """Compact per-block layout summary for the model prompt (no duplicated OCR text)."""
    described = []
    for hint in hints[:limit]:
        entry: dict = {"id": hint.id, "role": hint.role}
        if hint.speaker:
            entry["visible_name_tag"] = hint.speaker[:40]
        described.append(entry)
    return described
