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
class Translation:
    id: int
    thai: str
    speaker: str = ""


@dataclass
class Scene:
    summary: str = ""
    people: list[str] = field(default_factory=list)
    place: str = ""
    action: str = ""
    relations: list[dict] = field(default_factory=list)


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
        default.interval_ms = max(500, min(10000, default.interval_ms))
        default.timeout_s = max(5, min(180, default.timeout_s))
        default.font_size = max(10, min(48, default.font_size))
        default.opacity = max(80, min(255, default.opacity))
        default.max_blocks = max(1, min(80, default.max_blocks))
        default.min_confidence = max(0.0, min(1.0, default.min_confidence))
        return default


def short(value: Any, limit: int = 1000) -> str:
    return value[:limit] if isinstance(value, str) else ""


def parse_result(content: str, blocks: list[Block]) -> Result:
    """Model output is untrusted: reject unknown IDs, duplicates and arbitrary geometry."""
    if len(content) > 100_000:
        raise ValueError("คำตอบ AI ใหญ่เกินกำหนด")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("translations"), list):
        raise ValueError("AI ต้องตอบ JSON ที่มีรายการ translations")
    known = {b.id for b in blocks}
    seen = set()
    translations = []
    for entry in data["translations"][:100]:
        if not isinstance(entry, dict):
            continue
        bid = entry.get("id")
        thai = short(entry.get("thai"), 1500).strip()
        if type(bid) is int and bid in known and bid not in seen and thai:
            seen.add(bid)
            translations.append(Translation(bid, thai, short(entry.get("speaker"), 80)))
    raw = data.get("scene", {})
    if not isinstance(raw, dict):
        raw = {}
    people = raw.get("people", [])
    scene = Scene(
        summary=short(raw.get("summary")),
        people=[short(p, 120) for p in people[:12] if isinstance(p, str)]
        if isinstance(people, list) else [],
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
    return Result(translations, scene)
