"""Assemble the context that goes into one AI request.

Two levels are kept apart on purpose:

* **profile context** — research brief, character cards, glossary and user notes for the game.
* **frame context** — what is on screen in this frame: OCR blocks with their layout role,
  the visible name tags, face observations, speech plans and the last dialogue lines.

Every entry says where it came from (user-confirmed / research-unverified / measured), and the
trim step keeps the whole payload inside a fixed budget so a chatty scene cannot grow the prompt
without bound. The AI is told to trust the user first and to treat the rest as evidence only.
"""
from __future__ import annotations

import json

from .faces import FaceMemory
from .research import brief_text
from .speech import SourceCues, analyse_source, plan_speech, relation_type

MAX_CONTEXT_CHARS = 12_000
MAX_DIALOGUE_LINES = 8

# Least critical first: the trimmer drops from the top until the payload fits.
DROP_ORDER = (
    "recent_dialogue",
    "tentative_relationships",
    "characters_ai_hypothesis",
    "layout",
    "visible_name_tags",
    "faces",
    "speech_plans",
    "characters_confirmed_by_user",
    "game_research_unverified",
)

FACE_RULES = ("ห้ามใส่ชื่อนี้ในบทแปล", "ห้ามแต่งชื่อหรือเพศจากใบหน้า",
              "ถ้าจะกล่าวถึงให้เขียนว่า ‘ใบหน้าที่ …’ หรือ ‘คนที่อาจเป็น …’ เท่านั้น")


def plan_for(name: str, listener: str, cues: SourceCues, emotion: str = "",
             memory: FaceMemory | None = None):
    """Speech plan for one line. Values the user locked always win (see speech.plan_speech)."""
    speaker_card = memory.card(name) if memory is not None and name else None
    listener_card = memory.card(listener) if memory is not None and listener else None
    return plan_speech(speaker_card, listener_card, cues, emotion)


def _cues_payload(cues: SourceCues) -> dict:
    return {
        "language": cues.language,
        "politeness": cues.politeness,
        "relative_age": cues.relative_age,
        "gender_hint_in_source": cues.gender_hint,
        "honorifics": list(cues.honorifics)[:6],
        "kinship": list(cues.kinship)[:6],
        "pronouns": list(cues.pronouns)[:6],
        "names_called": list(cues.names_called)[:6],
        "notes": [str(note)[:120] for note in cues.notes[:6]],
    }


def _face_entry(observation, match, trend: str = "") -> dict:
    box = getattr(observation, "box", None)
    cues = getattr(observation, "cues", None)
    entry: dict = {
        "track": getattr(observation, "track_id", 0),
        "box": [box.x, box.y, box.w, box.h] if box is not None else [],
        "expression_cues": {
            "extreme_value": cues.label if cues else "",
            "measure": cues.summary() if cues else "",
            "quality": cues.quality if cues else "none",
            "note": "ค่าที่วัดได้จากภาพนิ่ง ไม่ใช่ผลตรวจอารมณ์: ใช้เป็นหลักฐานประกอบ "
                    "และห้ามสรุปอารมณ์ราวกับตรวจวัดได้",
        },
        "appearance": observation.appearance.summary()
                      if getattr(observation, "appearance", None) else "",
    }
    name = getattr(observation, "character", "") or (match.name if match else "")
    possible = getattr(observation, "possible", "") or (match.suggestion if match else "")
    if trend:
        entry["expression_trend"] = trend
    if name:
        entry["identified_as"] = name
        entry["identified_source"] = "ผู้ใช้ยืนยัน"
    elif possible:
        entry["possible_character"] = possible
        entry["identified_source"] = "คำแนะนำ ยังไม่ยืนยัน"
        entry["must_not_assert"] = FACE_RULES[0]
    else:
        entry["identified_source"] = "ไม่ทราบ"
        entry["must_not_assert"] = FACE_RULES[1]
    return entry


def frame_context(settings, blocks, hints, faces, memory: FaceMemory | None,
                  dialogue=None, source: dict | None = None,
                  limit: int = MAX_CONTEXT_CHARS) -> dict:
    """The per-frame half of the prompt. ``source`` maps block id -> original text."""
    hint_map = {hint.id: hint for hint in (hints or [])}
    text_of = dict(source or {})
    for block in blocks:
        text_of.setdefault(block.id, block.text)

    visible_tags: dict[str, str] = {}
    for block in blocks:
        hint = hint_map.get(block.id)
        if hint is not None and hint.role == "name_tag":
            visible_tags[str(block.id)] = block.text
        elif hint is not None and hint.speaker:
            visible_tags[str(block.id)] = hint.speaker

    face_entries = []
    for observation in faces or []:
        match = None
        if memory is not None and getattr(observation, "descriptor", None):
            match = memory.match(observation.descriptor)
        trend = memory.trend(observation.track_id) if memory is not None else ""
        face_entries.append(_face_entry(observation, match, trend))

    plans: dict[str, dict] = {}
    for block in blocks:
        hint = hint_map.get(block.id)
        if hint is None or hint.role not in ("dialogue", "subtitle", "choice"):
            continue
        raw_speaker = hint.speaker or ""
        if not raw_speaker:
            continue
        speaker = memory.resolve(raw_speaker) if memory is not None else raw_speaker
        cues = analyse_source(text_of.get(block.id, block.text))
        plan = plan_for(speaker, dialogue.listener_of(speaker) if dialogue else "", cues, "", memory)
        plans[str(block.id)] = {
            "character": speaker,
            "must_follow": bool(plan.locked),
            "locked": list(plan.locked),
            "plan": plan.describe(),
            "relation": relation_type(memory.card(speaker) if memory else {} or {}, {}),
            "source_evidence": _cues_payload(cues),
        }

    recent = dialogue.recent(MAX_DIALOGUE_LINES) if dialogue is not None else []
    frame: dict = {
        "layout": [{"id": block.id, "role": hint_map[block.id].role,
                    "speaker": hint_map[block.id].speaker} for block in blocks
                   if block.id in hint_map],
        "visible_name_tags": visible_tags,
        "faces": face_entries,
        "speech_plans": plans,
        "recent_dialogue": [{"speaker": line.speaker or "ไม่ทราบ",
                             "speaker_source": line.speaker_source,
                             "thai": str(line.thai)[:200],
                             "emotion": line.emotion,
                             "delivery": str(line.delivery)[:120],
                             "to": line.listener} for line in recent],
        "continuity": dialogue.continuity_note() if dialogue is not None else "",
        "rules": [
            "visible_name_tags เป็นข้อความที่เห็นบนจอ ไม่ใช่คำยืนยันตัวตน",
            "faces ที่ไม่มี identified_as ห้ามใช้ชื่อ • possible_character เป็นคำแนะนำเท่านั้น",
            "expression_cues เป็นการวัด ไม่ใช่อารมณ์ที่ยืนยัน ให้พูดแบบประมาณและอ้างหลักฐาน",
            "speech_plans.must_follow ต้องใช้ตามนั้น และรักษาสรรพนามให้คงเส้นคงวา",
        ],
    }
    if face_entries and not visible_tags:
        frame["faces_note"] = (f"เห็นใบหน้า {len(face_entries)} หน้าแต่ไม่พบป้ายชื่อบนจอ: "
                               "ห้ามเดาชื่อผู้พูดจากใบหน้า ให้ใช้สรรพนามกลางหรือรอหลักฐานจากบทพูดก่อน")
    return trim(frame, limit)


def profile_context(settings, profile: dict | None, memory: FaceMemory | None,
                    limit: int = MAX_CONTEXT_CHARS) -> dict:
    """The persistent half: what we already know about this game before looking at the frame."""
    profile = profile or {}
    cards: dict[str, dict] = {}
    if memory is not None:
        for name in memory.confirmed_names():
            card = memory.card(name)
            voice = card.get("voice") if isinstance(card.get("voice"), dict) else {}
            entry: dict = {"confirmed": True}
            for key in ("role", "relation", "personality", "age_gap", "aliases"):
                value = card.get(key)
                if value:
                    entry[key] = str(value)[:160]
            if card.get("gender") and card.get("gender") != "unknown":
                entry["gender_confirmed"] = card["gender"]
                if card.get("gender_source"):
                    entry["gender_source"] = card["gender_source"]
            if voice and voice.get("locked"):
                entry["locked_voice"] = {key: str(value)[:80] for key, value in voice.items()
                                         if value not in ("", False, None)}
            facts = card.get("facts") or []
            if facts:
                entry["facts_confirmed_by_user"] = [str(fact)[:160] for fact in facts[:8]]
            cards[name] = entry
    glossary = profile.get("glossary")
    if not isinstance(glossary, (str, dict)):
        glossary = str(glossary or "")
    research = profile.get("research") or {}
    research_text = brief_text(research, limit=1500) if isinstance(research, dict) else ""
    if not research_text:
        research_text = "ยังไม่มีข้อมูลวิจัยจากอินเทอร์เน็ต (ยังไม่ยืนยัน) ให้ยึดข้อความบนจอเป็นหลัก"
    payload = {
        "profile_name": str(profile.get("name") or "")[:80],
        "manual_notes": str(profile.get("notes") or "")[:6000],
        "glossary": glossary if isinstance(glossary, dict) else glossary[:4000],
        "previous_summary": str(profile.get("summary") or "")[:1000],
        "tentative_relationships": [dict(rel) for rel in (profile.get("relations") or [])][-40:],
        "characters_confirmed_by_user": cards,
        "game_research_unverified": research_text,
        "rules_for_trust": [
            "manual_notes/glossary/characters_confirmed_by_user คือคำยืนยันของผู้ใช้ ใช้ได้เต็มที่",
            "tentative_relationships ต้องมีหลักฐานประกอบและเรียกเป็น ‘ข้อสันนิษฐาน’",
            "game_research_unverified ห้ามอ้างเป็นข้อเท็จจริง และห้ามเพิ่มข้อมูลที่ไม่มีในนั้น",
        ],
    }
    return trim(payload, limit)


def trim(payload: dict, limit: int = MAX_CONTEXT_CHARS) -> dict:
    """Shrink an over-long context payload, dropping the least critical parts first."""
    payload = dict(payload or {})

    def size() -> int:
        return len(json.dumps(payload, ensure_ascii=False))

    for key in DROP_ORDER:
        if size() <= limit:
            break
        payload.pop(key, None)
    for key, value in list(payload.items()):
        if isinstance(value, str) and len(value) > 600:
            payload[key] = value[:600]
    for key, value in list(payload.items()):
        if isinstance(value, list) and len(value) > 5:
            payload[key] = value[:5]
        elif isinstance(value, dict) and len(value) > 5:
            payload[key] = dict(list(value.items())[:5])
    if size() > limit:
        for key, value in list(payload.items()):
            if isinstance(value, str):
                payload[key] = value[: max(60, limit // 10)]
    if size() > limit:
        payload = {"note": json.dumps(payload, ensure_ascii=False)[: max(200, limit - 60)]}
    return payload
