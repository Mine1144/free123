from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any


@dataclass(frozen=True)
class Block:
    id: int
    text: str
    # Coordinates in physical pixels, relative to the captured region.
    box: tuple[int, int, int, int]
    confidence: float = 1.0


@dataclass(frozen=True)
class BlockHint:
    """Local layout interpretation for one OCR block. Never produced by the model."""

    id: int
    role: str = "unknown"  # name_tag | dialogue | subtitle | choice | hud | menu | item | tooltip
    speaker: str = ""      # suggested speaker from a nearby name tag (text evidence only)
    order: int = 0


@dataclass(frozen=True)
class Translation:
    id: int
    thai: str
    speaker: str = ""
    listener: str = ""
    emotion: str = ""
    delivery: str = ""
    self_pronoun: str = ""
    address_pronoun: str = ""
    particles: str = ""
    confidence: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass
class Scene:
    summary: str = ""
    people: list[str] = field(default_factory=list)
    place: str = ""
    action: str = ""
    relations: list[dict] = field(default_factory=list)
    characters: list[dict] = field(default_factory=list)
    expressions: list[dict] = field(default_factory=list)
    faces: list[dict] = field(default_factory=list)


@dataclass
class Result:
    translations: list[Translation] = field(default_factory=list)
    scene: Scene = field(default_factory=Scene)


@dataclass
class Settings:
    provider: str = "ollama"
    endpoint: str = "http://localhost:11434"
    model: str = "qwen2.5vl:7b"
    vision: bool = True
    scope: str = "all"
    interval_ms: int = 1500
    timeout_s: int = 90
    font_size: int = 18
    opacity: int = 220
    max_blocks: int = 48
    min_confidence: float = 0.55
    profile: str = "ทั่วไป"
    capture_mode: str = "safe"
    # Face / expression analysis (runs locally on the captured still).
    face_analysis: bool = True
    face_backend: str = "auto"
    face_model: str = ""              # optional ONNX path for YuNet face detection
    face_sample_ms: int = 1200
    max_faces: int = 4
    expression_cues: bool = True
    # Layout interpretation and per-character speech planning.
    layout_hints: bool = True
    speech_plan: bool = True
    show_speaker_label: bool = False
    # Research step (all network work is opt-in and never includes screenshots).
    translation_memory: bool = True      # reuse your own accepted wording for repeated lines
    glossary_mine: bool = True           # suggest glossary terms from repeated names
    merge_lines: bool = True             # join wrapped subtitle lines before translating
    review_consistency: bool = False     # one extra AI pass over flagged lines only
    review_max_lines: int = 12

    # Local AI through llama.cpp, installed and downloaded from inside the app.
    llama_binary: str = ""             # path to llama-server(.exe); empty = look in the app folder
    llama_model: str = ""              # path to the .gguf to load
    llama_mmproj: str = ""             # optional vision projector
    llama_port: int = 8081
    llama_ctx: int = 8192
    llama_gpu_layers: int = 0
    llama_threads: int = 0
    llama_auto_start: bool = True      # start the server when translation begins
    llama_extra_args: str = ""

    research_search: str = "off"       # off | searxng | brave | serper | tavily | custom
    research_endpoint: str = ""
    research_max_sources: int = 6
    research_title: str = ""
    research_aliases: str = ""
    research_auto: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Settings:
        default = cls()
        for key, value in raw.items():
            if hasattr(default, key) and type(value) is type(getattr(default, key)):
                setattr(default, key, value)
        if default.provider not in ("ollama", "openai"):
            default.provider = "ollama"
        if default.scope not in ("all", "dialogue", "smart"):
            default.scope = "all"
        if default.capture_mode not in ("safe", "excluded"):
            default.capture_mode = "safe"
        if default.face_backend not in ("auto", "mediapipe", "opencv", "off"):
            default.face_backend = "auto"
        if default.research_search not in ("off", "searxng", "brave", "serper", "tavily", "custom"):
            default.research_search = "off"
        default.interval_ms = max(500, min(10000, default.interval_ms))
        default.timeout_s = max(5, min(180, default.timeout_s))
        default.font_size = max(10, min(48, default.font_size))
        default.opacity = max(80, min(255, default.opacity))
        default.max_blocks = max(1, min(80, default.max_blocks))
        default.min_confidence = max(0.0, min(1.0, default.min_confidence))
        default.face_sample_ms = max(400, min(10000, default.face_sample_ms))
        default.max_faces = max(1, min(8, default.max_faces))
        default.research_max_sources = max(1, min(12, default.research_max_sources))
        default.review_max_lines = max(1, min(40, default.review_max_lines))
        default.llama_port = max(1024, min(65535, default.llama_port))
        default.llama_ctx = max(2048, min(131072, default.llama_ctx))
        default.llama_gpu_layers = max(0, min(999, default.llama_gpu_layers))
        default.llama_threads = max(0, min(64, default.llama_threads))
        default.llama_binary = default.llama_binary[:400]
        default.llama_model = default.llama_model[:400]
        default.llama_mmproj = default.llama_mmproj[:400]
        default.llama_extra_args = default.llama_extra_args[:200]
        default.research_title = default.research_title[:120]
        default.research_aliases = default.research_aliases[:200]
        return default


def short(value: Any, limit: int = 1000) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _number(value: Any, low: float = 0.0, high: float = 1.0) -> float | None:
    if type(value) not in (int, float):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return max(low, min(high, value))


def _string_list(value: Any, limit: int, item_limit: int = 120) -> list[str]:
    if not isinstance(value, list):
        return []
    return [short(item, item_limit).strip() for item in value[:limit]
            if isinstance(item, str) and item.strip()]


def _one_of(value: Any, allowed: tuple[str, ...], fallback: str = "") -> str:
    text = short(value, 60).strip()
    return text if text in allowed else fallback


def _unicode_hint(text: str) -> str:
    """Local, cheap language hint so the prompt can warn about honorific systems."""
    for chars, name in (("\u3040\u30ff\u4e00\u9fff", "ja"), ("\uac00\ud7af", "ko"),
                        ("\u4e00\u9fff", "zh"), ("\u0e00\u0e7f", "th"), ("\u0400\u04ff", "ru")):
        if any(chars[0] <= ch <= chars[1] for ch in text):
            return name
    return ""


def parse_frame_line(entry: Any, known: set[int], seen: set[int]) -> Translation | None:
    if not isinstance(entry, dict):
        return None
    bid = entry.get("id")
    thai = short(entry.get("thai"), 1500).strip()
    if type(bid) is not int or bid not in known or bid in seen or not thai:
        return None
    seen.add(bid)
    return Translation(
        bid,
        thai,
        speaker=short(entry.get("speaker"), 80).strip(),
        listener=short(entry.get("listener"), 80).strip(),
        emotion=_one_of(entry.get("emotion"), (
            "โกรธ", "ดีใจ", "เศร้า", "กังวล", "กลัว", "ตกใจ", "ฉงน", "เขิน", "เหนื่อย", "เจ็บปวด",
            "เป็นกลาง", "อื่น ๆ", "อื่นๆ"), ""),
        delivery=short(entry.get("delivery"), 240).strip(),
        self_pronoun=short(entry.get("self_pronoun"), 40).strip(),
        address_pronoun=short(entry.get("address_pronoun"), 40).strip(),
        particles=short(entry.get("particles"), 60).strip(),
        confidence=_number(entry.get("confidence")) or 0.0,
    )


def parse_characters(raw: Any, limit: int = 12) -> list[dict]:
    out = []
    for item in raw[:limit] if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = short(item.get("name"), 80).strip()
        evidence = short(item.get("evidence"), 240).strip()
        confidence = _number(item.get("confidence"))
        # An identity claim without visible text evidence is not knowledge.
        if not name or not evidence or confidence is None or confidence < 0.6:
            continue
        gender = _one_of(item.get("gender"), ("male", "female", "unknown"), "unknown")
        out.append({
            "name": name,
            "aliases": _string_list(item.get("aliases"), 6, 60),
            "role": short(item.get("role"), 120).strip(),
            "personality": short(item.get("personality"), 300).strip(),
            "speech_style": short(item.get("speech_style"), 300).strip(),
            "gender": gender,
            "evidence": evidence,
            "confidence": confidence,
        })
    return out


def parse_expressions(raw: Any, limit: int = 12) -> list[dict]:
    allowed = ("ยิ้ม", "หัวเราะ", "โกรธ", "เศร้า", "ร้องไห้", "กังวล", "กลัว", "ตกใจ", "ฉงน",
               "เขิน", "เหนื่อย", "เจ็บปวด", "เฉยชา", "อึ้ง", "ไม่ทราบ")
    out = []
    for item in raw[:limit] if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        confidence = _number(item.get("confidence"))
        cues = short(item.get("cues"), 240).strip()
        evidence = short(item.get("evidence"), 240).strip()
        expression = _one_of(item.get("expression"), allowed, "")
        # Expression reading is a weak hypothesis: it needs a cue or text evidence.
        if not expression or confidence is None or confidence < 0.35 or not (cues or evidence):
            continue
        out.append({
            "character": short(item.get("character"), 80).strip(),
            "expression": expression,
            "cues": cues,
            "evidence": evidence,
            "confidence": confidence,
        })
    return out


def parse_face_links(raw: Any, limit: int = 8) -> list[dict]:
    out = []
    for item in raw[:limit] if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        character = short(item.get("character"), 80).strip()
        track = short(item.get("track"), 80).strip()
        evidence = short(item.get("evidence"), 240).strip()
        confidence = _number(item.get("confidence"))
        # Never let the model assert "this face = this person" without evidence.
        if not character or not track or not evidence or confidence is None or confidence < 0.6:
            continue
        out.append({"track": track, "character": character, "evidence": evidence,
                    "confidence": confidence})
    return out


def parse_result(content: str, blocks: list[Block], characters: tuple[str, ...] = ()) -> Result:
    """Model output is untrusted: reject unknown IDs, duplicates and arbitrary geometry.

    `characters` are names the profile already knows. Character updates and face links are
    kept strictly as evidence-backed hypotheses; the parser never promotes them to facts.
    """
    if len(content) > 200_000:
        raise ValueError("คำตอบ AI ใหญ่เกินกำหนด")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("translations"), list):
        raise ValueError("AI ต้องตอบ JSON ที่มีรายการ translations")
    known = {b.id for b in blocks}
    seen: set[int] = set()
    translations = [line for entry in data["translations"][:120]
                    if (line := parse_frame_line(entry, known, seen)) is not None]
    raw = data.get("scene", {})
    if not isinstance(raw, dict):
        raw = {}
    scene = Scene(
        summary=short(raw.get("summary")),
        people=_string_list(raw.get("people"), 12),
        place=short(raw.get("place"), 200),
        action=short(raw.get("action"), 300),
    )
    relations = raw.get("relations", [])
    for rel in relations[:24] if isinstance(relations, list) else []:
        if not isinstance(rel, dict):
            continue
        confidence = rel.get("confidence", 0)
        if type(confidence) not in (int, float) or not math.isfinite(confidence):
            continue
        clean = {key: short(rel.get(key), 160) for key in ("from", "to", "relation", "evidence")}
        # Do not promote guesses into durable knowledge.
        if confidence >= 0.8 and all(clean.values()):
            clean["confidence"] = min(1.0, confidence)
            scene.relations.append(clean)
    scene.characters = parse_characters(raw.get("characters"))
    scene.expressions = parse_expressions(raw.get("expressions"))
    # A face link may only name a character the profile already knows, a name the model
    # listed as visible in this scene, or a name written in the frame / its own evidence.
    allowed = {name.strip().casefold() for name in characters if name.strip()}
    allowed |= {name.casefold() for name in scene.people}
    allowed |= {c["name"].casefold() for c in scene.characters}
    frame_text = " ".join(b.text for b in blocks).casefold()
    scene.faces = [link for link in parse_face_links(raw.get("faces"))
                   if link["character"].casefold() in allowed
                   or link["character"].casefold() in frame_text
                   or link["character"].casefold() in link["evidence"].casefold()]
    return Result(translations, scene)
