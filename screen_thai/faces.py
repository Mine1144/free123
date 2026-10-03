"""Stable, honest character memory built from what the user confirms plus visible evidence.

Rules that make this module trustworthy:

* A name is only ever attached to a face by the user, or suggested by the AI *with* textual
  evidence. Suggestions stay suggestions until the user confirms them.
* Only descriptors (small float vectors) and colour summaries are stored. No face crops, no
  screenshots, ever.
* Below the match threshold a face stays an anonymous "ใบหน้าที่ N" — the app never invents
  a name or a gender to look smart.
* Everything here is local. Nothing in this module talks to the network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

MATCH_THRESHOLD = 0.86        # ใช้ชื่อที่ผู้ใช้ยืนยันแล้ว
SUGGEST_THRESHOLD = 0.78      # เสนอเป็น "อาจเป็น" เท่านั้น ไม่ใส่ชื่อลงบทแปล
MARGIN = 0.02                 # ต้องชนะอันดับสองพอสมควร ไม่งั้นไม่ฟันธง
MAX_CHARACTERS = 60
MAX_DESCRIPTORS = 8           # ต่อตัวละคร


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * \
        max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


class FaceTracker:
    """Keeps a stable id per visible face across frames, purely from overlap and motion."""

    def __init__(self, iou_threshold: float = 0.25, ttl: int = 6):
        self.iou_threshold = iou_threshold
        self.ttl = ttl
        self._tracks: dict[int, dict] = {}
        self._next = 0

    def assign(self, boxes: list[tuple[int, int, int, int]]) -> list[int]:
        for track in self._tracks.values():
            track["age"] += 1
        used: set[int] = set()
        ids: list[int] = []
        for box in boxes:
            best, best_score = None, 0.0
            for track_id, track in self._tracks.items():
                if track_id in used:
                    continue
                score = _iou(box, track["box"])
                if score > best_score:
                    best, best_score = track_id, score
            if best is None or best_score < self.iou_threshold:
                self._next += 1
                best = self._next
                self._tracks[best] = {"box": box, "age": 0, "seen": 0}
            used.add(best)
            self._tracks[best].update(box=box, age=0, seen=self._tracks[best]["seen"] + 1)
            ids.append(best)
        for track_id in [key for key, track in self._tracks.items() if track["age"] > self.ttl]:
            del self._tracks[track_id]
        return ids


@dataclass
class Match:
    name: str = ""
    score: float = 0.0
    suggestion: str = ""
    suggestion_score: float = 0.0


@dataclass
class Observation:
    """Rolling per-track measurements inside one session (RAM only).

    Because a single still cannot name a mood, the value here is change over time: each new
    frame is compared with this face's own recent baseline.
    """

    track_id: int
    label: str = ""
    samples: int = 0
    measures: list[tuple[float, dict]] = field(default_factory=list)

    FIELDS = ("mouth_open", "smile", "eye_open", "brow_raise", "tilt_deg", "gaze_x")

    def add(self, cues, now: float | None = None):
        if cues is None:
            return
        now = time.monotonic() if now is None else now
        values = {name: float(getattr(cues, name, 0.0)) for name in self.FIELDS}
        self.measures.append((now, values))
        del self.measures[:-10]
        self.samples += 1
        if getattr(cues, "label", "") and cues.label != "ไม่ทราบ":
            self.label = cues.label

    def recent(self, window: float = 8.0, now: float | None = None) -> list[dict]:
        now = time.monotonic() if now is None else now
        return [values for stamp, values in self.measures if now - stamp <= window]

    def deviation(self, now: float | None = None) -> dict:
        """current vs own median baseline, per measure."""
        recent = self.recent(now=now)
        if len(recent) < 3:
            return {}
        baseline = {}
        for name in self.FIELDS:
            ordered = sorted(item[name] for item in recent)
            baseline[name] = ordered[len(ordered) // 2]
        current = recent[-1]
        deltas = {name: round(current[name] - baseline[name], 3) for name in self.FIELDS
                  if abs(current[name] - baseline[name]) > 0.05}
        return deltas

    def motion(self, now: float | None = None) -> str:
        """Detect a mouth that keeps opening and closing (likely talking) without a classifier."""
        recent = self.recent(now=now)
        if len(recent) < 4:
            return ""
        values = [item["mouth_open"] for item in recent]
        swings = sum(1 for index in range(1, len(values))
                     if (values[index] - values[index - 1]) * (values[index - 1] - values[index - 2]) < 0
                     and abs(values[index] - values[index - 1]) > 0.12)
        return "ปากขยับไปมาหลายจังหวะ (น่าจะกำลังพูด)" if swings >= 2 else ""

    def trend(self, now: float | None = None) -> str:
        notes = []
        motion = self.motion(now=now)
        if motion:
            notes.append(motion)
        deltas = self.deviation(now=now)
        wording = {"mouth_open": "ปากเปิดมากกว่าปกติของตัวเอง", "smile": "มุมปากเปลี่ยนไป",
                   "eye_open": "ระดับการเปิดตเปลี่ยนไป", "brow_raise": "คิ้วเปลี่ยนระดับ",
                   "tilt_deg": "ศีรษะเอียงเปลี่ยนไป", "gaze_x": "สายตาเปลี่ยนทิศ"}
        for name, delta in list(deltas.items())[:3]:
            notes.append(f"{wording.get(name, name)} ({delta:+.2f})")
        if self.label:
            notes.append("ค่าที่สุดขั้ว: " + self.label)
        return " • ".join(notes)


class FaceMemory:
    """Character cards + face descriptors for one game profile."""

    def __init__(self, data: dict | None = None):
        data = data if isinstance(data, dict) else {}
        self.characters: dict[str, dict] = {}
        for name, card in (data.get("characters") or {}).items():
            if isinstance(card, dict) and isinstance(name, str) and name.strip():
                self.characters[name[:60]] = self._clean_card(card)
        self.observations: dict[int, Observation] = {}
        self.pending: list[dict] = []          # AI face-link suggestions awaiting confirmation

    # --- serialisation -----------------------------------------------------------------

    @staticmethod
    def _clean_card(card: dict) -> dict:
        out = {
            "aliases": [str(a)[:60] for a in card.get("aliases", [])[:8] if isinstance(a, str)],
            "role": str(card.get("role", ""))[:120],
            "relation": str(card.get("relation", ""))[:120],
            "personality": str(card.get("personality", ""))[:300],
            "speech_style": str(card.get("speech_style", ""))[:300],
            "gender": card.get("gender", "unknown") if card.get("gender") in ("male", "female", "unknown")
            else "unknown",
            "gender_source": card.get("gender_source", "") if card.get("gender_source") in ("user", "ai", "source")
            else "",
            "age_gap": card.get("age_gap", "") if card.get("age_gap") in ("older", "younger", "same", "")
            else "",
            "confirmed": bool(card.get("confirmed")),
            "notes": str(card.get("notes", ""))[:600],
            "evidence": str(card.get("evidence", ""))[:400],
            "sources": [str(s)[:300] for s in card.get("sources", [])[:8] if isinstance(s, str)],
            "descriptors": [d for d in card.get("descriptors", [])[:MAX_DESCRIPTORS]
                            if isinstance(d, list) and all(isinstance(x, float) for x in d)],
            "appearance": card.get("appearance") if isinstance(card.get("appearance"), dict) else {},
            "voice": FaceMemory._clean_voice(card.get("voice")),
            "samples": int(card.get("samples", 0)) if isinstance(card.get("samples"), int) else 0,
            "first_seen": str(card.get("first_seen", ""))[:32],
            "last_seen": str(card.get("last_seen", ""))[:32],
        }
        return out

    @staticmethod
    def _clean_voice(voice) -> dict:
        voice = voice if isinstance(voice, dict) else {}
        return {
            "self": str(voice.get("self", ""))[:40],
            "address": str(voice.get("address", ""))[:40],
            "particles": str(voice.get("particles", ""))[:60],
            "register": str(voice.get("register", ""))[:40],
            "quirks": str(voice.get("quirks", ""))[:300],
            "locked": bool(voice.get("locked")),
            "source": str(voice.get("source", ""))[:40],
        }

    def to_dict(self) -> dict:
        return {"characters": self.characters}

    # --- reading -----------------------------------------------------------------------

    def card(self, name: str) -> dict:
        card = self.characters.get(name)
        if card is None:
            return {}
        merged = dict(card)
        merged["name"] = name
        return merged

    def names(self) -> list[str]:
        return list(self.characters)

    def confirmed_names(self) -> list[str]:
        return [name for name, card in self.characters.items() if card.get("confirmed")]

    def resolve(self, label: str) -> str:
        """Map an alias / partial label written on screen to a stored character name."""
        needle = label.strip().casefold()
        if not needle:
            return ""
        for name, card in self.characters.items():
            if name.casefold() == needle or needle in (a.casefold() for a in card["aliases"]):
                return name
        candidates = [name for name in self.characters
                      if needle and (needle in name.casefold() or name.casefold() in needle)]
        return candidates[0] if len(candidates) == 1 else ""

    def match(self, vector: list[float]) -> Match:
        scores = []
        for name, card in self.characters.items():
            for stored in card["descriptors"]:
                scores.append((self.cosine(vector, stored), name))
        if not scores:
            return Match()
        scores.sort(reverse=True)
        best, best_name = scores[0]
        second = next((score for score, name in scores if name != best_name), 0.0)
        if best >= MATCH_THRESHOLD and best - second >= MARGIN:
            return Match(best_name, best)
        if best >= SUGGEST_THRESHOLD:
            return Match("", 0.0, best_name, best)
        return Match()

    @staticmethod
    def cosine(a, b) -> float:
        from .vision import cosine as _cosine
        return _cosine(a, b)

    # --- writing -----------------------------------------------------------------------

    def ensure(self, name: str, *, role: str = "", relation: str = "", personality: str = "",
               speech_style: str = "", evidence: str = "", sources: list[str] | None = None,
               confirmed: bool = False, gender: str = "unknown", gender_source: str = "") -> dict:
        name = name.strip()[:60]
        if not name:
            return {}
        card = self.characters.get(name)
        if card is None:
            if len(self.characters) >= MAX_CHARACTERS:
                raise ValueError("จำนวนตัวละครถึงเพดานแล้ว กรุณาลบตัวละครที่ไม่ใช้")
            card = self._clean_card({"confirmed": confirmed, "first_seen": _stamp()})
            self.characters[name] = card
        if role:
            card["role"] = role[:120]
        if relation:
            card["relation"] = relation[:120]
        if personality:
            card["personality"] = personality[:300]
        if speech_style:
            card["speech_style"] = speech_style[:300]
        if evidence:
            card["evidence"] = evidence[:400]
        if sources:
            card["sources"] = list(dict.fromkeys(card["sources"] + sources))[:8]
        if gender in ("male", "female", "unknown"):
            card["gender"] = gender
            card["gender_source"] = gender_source or card["gender_source"]
        card["confirmed"] = card["confirmed"] or confirmed
        card["last_seen"] = _stamp()
        return self.card(name)

    def remember_face(self, name: str, vector: list[float], appearance: dict | None = None,
                      confirmed: bool = False) -> bool:
        name = name.strip()[:60]
        if not name or not vector:
            return False
        card = self.characters.get(name)
        if card is None:
            card = self.ensure(name, confirmed=confirmed)
            card = self.characters.get(name)
        if card is None:
            return False
        if len(card["descriptors"]) >= MAX_DESCRIPTORS:
            card["descriptors"].pop(0)
        card["descriptors"].append([round(float(x), 5) for x in vector])
        card["samples"] += 1
        if appearance:
            card["appearance"] = {key: value for key, value in appearance.items()
                                  if key in ("hair", "eyes", "outfit", "palette")}
        card["last_seen"] = _stamp()
        return True

    def set_voice(memory, name: str, **fields) -> dict:
        # The receiver is named `memory` on purpose: one of the fields is literally `self`.
        card = memory.characters.get(name)
        if card is None:
            card = memory.characters.get(memory.ensure(name).get("name", ""))
        if card is None:
            return {}
        voice = card["voice"]
        for key in ("self", "address", "particles", "register", "quirks"):
            if key in fields and fields[key] is not None:
                voice[key] = str(fields[key])[:60 if key != "quirks" else 300]
        if "locked" in fields and fields["locked"] is not None:
            voice["locked"] = bool(fields["locked"])
        voice["source"] = str(fields.get("source", voice.get("source") or "user"))[:40]
        return memory.card(name)

    def set_gender(self, name: str, gender: str, source: str = "user") -> dict:
        card = self.characters.get(name)
        if card is None:
            return {}
        if gender in ("male", "female", "unknown"):
            card["gender"] = gender
            card["gender_source"] = source if gender != "unknown" else ""
        return self.card(name)

    def set_relation(self, name: str, other: str, relation: str) -> dict:
        card = self.characters.get(name)
        if card is None:
            return {}
        card["relation"] = f"{relation} ({other})"[:120] if other else relation[:120]
        return self.card(name)

    def apply_hypothesis(self, update: dict, confirmed_by: str = "ai") -> str:
        """Store an AI hypothesis without ever overwriting user-confirmed values."""
        name = self.resolve(str(update.get("name", ""))) or str(update.get("name", "")).strip()[:60]
        if not name:
            return ""
        card = self.characters.get(name)
        if card is None:
            card = self._clean_card({"first_seen": _stamp()})
            self.characters[name] = card
        for key in ("role", "personality", "speech_style"):
            if update.get(key) and not card.get(key):
                card[key] = str(update[key])[:300]
        if update.get("evidence"):
            card["evidence"] = str(update["evidence"])[:400]
        if update.get("aliases"):
            card["aliases"] = list(dict.fromkeys(card["aliases"] +
                                                [str(a)[:60] for a in update["aliases"][:6]]))[:8]
        gender = update.get("gender", "unknown")
        # Gender is only stored as a hypothesis, and only when the model showed evidence.
        if gender in ("male", "female") and update.get("evidence") and card["gender_source"] != "user":
            card["gender"] = gender
            card["gender_source"] = "ai"
        return name

    def add_pending(self, track_label: str, character: str, evidence: str, confidence: float) -> bool:
        character = self.resolve(character) or character.strip()[:60]
        if not character or not evidence:
            return False
        for item in self.pending:
            if item["track"] == track_label and item["character"] == character:
                item["confidence"] = max(item["confidence"], float(confidence))
                return False
        self.pending.append({"track": track_label[:60], "character": character,
                             "evidence": evidence[:240], "confidence": float(confidence)})
        del self.pending[:-12]
        return True

    def confirm_pending(self, index: int, vector_by_track: dict[str, list[float]]) -> str:
        if not 0 <= index < len(self.pending):
            return ""
        item = self.pending.pop(index)
        vector = vector_by_track.get(item["track"], [])
        self.ensure(item["character"], evidence=item["evidence"], confirmed=True)
        if vector:
            self.remember_face(item["character"], vector, confirmed=True)
        return item["character"]

    def reject_pending(self, index: int):
        if 0 <= index < len(self.pending):
            self.pending.pop(index)

    def forget_face(self, name: str):
        card = self.characters.get(name)
        if card is not None:
            card["descriptors"] = []
            card["samples"] = 0

    def remove(self, name: str):
        self.characters.pop(name, None)

    def prune(self):
        """Keep the durable profile small: drop the weakest, unconfirmed, long-unseen cards."""
        if len(self.characters) <= MAX_CHARACTERS:
            return
        ranked = sorted(self.characters.items(),
                        key=lambda item: (item[1]["confirmed"], item[1]["samples"],
                                          item[1]["last_seen"]))
        for name, _card in ranked[:len(self.characters) - MAX_CHARACTERS]:
            del self.characters[name]

    # --- session observations -----------------------------------------------------------

    def note_cues(self, track_id: int, cues):
        observation = self.observations.setdefault(track_id, Observation(track_id))
        observation.add(cues)

    def trend(self, track_id: int) -> str:
        observation = self.observations.get(track_id)
        return observation.trend() if observation else ""

    def reset_session(self):
        self.observations.clear()
        self.pending.clear()


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M")
