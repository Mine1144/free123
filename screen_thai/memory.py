"""Translation memory, glossary mining and terminology QC.

Why this exists: games repeat the same lines constantly (menus, battle cries, greetings) and
repeatedly asking a model for them costs money, adds latency and drifts in wording. ScreenThai
therefore keeps a small per-profile memory of *your* confirmed wordings and reuses it.

Rules that keep the memory honest:

* Only lines that passed local QC with no warnings are reused automatically. A warned line is
  still stored, but marked "needs review" and never replayed without the model.
* A line whose speaker now has a different locked voice card is *not* replayed: the user changed
  the intended wording, so the model must see it again.
* Anything the model changes later overwrites the entry, so drift cannot accumulate.
* The memory stays in the game's profile JSON (bounded), never in the log or the AI payload.
"""
from __future__ import annotations

import hashlib
import re
import time

from .text import normalized

MAX_ENTRIES = 800
MAX_THAI_CHARS = 600
MAX_SOURCE_CHARS = 400
MIN_REUSE_CHARS = 2

# Sentence-final punctuation means "the line is complete, do not reuse it as a prefix".
_END_OK = tuple(".!?…。！？:;\"'’”»)]}")

_LATIN_TERM = re.compile(r"\b[A-Z][A-Za-z'’\-]{2,}\b")
_CJK_TERM = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]{2,}")
_THAI_WORD = re.compile(r"[\u0e00-\u0e7f]{3,}")

# Words that carry no terminology value; a suggestion built only from these is noise.
THAI_STOPWORDS = {
    "ครับ", "ค่ะ", "คะ", "นะ", "นะคะ", "เลย", "แล้ว", "ไม่", "ได้", "เป็น", "ที่", "ของ", "ให้",
    "กับ", "ก็", "จะ", "ว่า", "แต่", "หรือ", "มาก", "น้อย", "อีก", "ต้อง", "กำลัง", "เพราะ",
    "เพราะว่า", "ทำไม", "อะไร", "ใคร", "ตรงนี้", "ตรงนั้น", "ตอนนี้", "เดี๋ยว", "ช่วย", "อยาก",
    "รู้", "ไป", "มา", "อยู่", "มี", "ไม่มี", "คน", "วัน", "เวลา", "เรื่อง", "เหมือน", "เท่า",
    "สวัสดี", "ขอบคุณ", "ขอโทษ", "ยินดี", "ตกลง", "เข้าใจ", "เอา", "ไหม", "มั้ย", "กัน", "เรา",
    "ฉัน", "ผม", "หนู", "คุณ", "เธอ", "เขา", "มัน", "ท่าน", "เจ้า", "พี่", "น้อง", "ตัวเอง",
}


def _safe_int(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def voice_signature(card: dict | None) -> str:
    """Short hash of the parts of a voice card that change the wording."""
    if not isinstance(card, dict):
        return ""
    voice = card.get("voice") if isinstance(card.get("voice"), dict) else {}
    parts = [str(voice.get(key, "")) for key in ("self", "address", "particles", "register")]
    parts.append("locked" if voice.get("locked") else "")
    text = "|".join(parts)
    return hashlib.sha256(text.encode()).hexdigest()[:12] if text.strip("|") else ""


def key_for(source: str) -> str:
    text = normalized(source)
    return text[:MAX_SOURCE_CHARS]


def is_complete(text: str) -> bool:
    text = text.rstrip()
    return not text or text.endswith(_END_OK)


class TranslationMemory:
    """Source line → Thai wording, learned from accepted translations."""

    def __init__(self, data: dict | None = None):
        self.entries: dict[str, dict] = {}
        if isinstance(data, dict):
            self.from_dict(data)

    # --- storage ---------------------------------------------------------------------
    def from_dict(self, data: dict):
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, dict):
            return
        for key, value in list(entries.items())[:MAX_ENTRIES]:
            if not isinstance(value, dict):
                continue
            thai = str(value.get("thai", "")).strip()
            if not key or not thai:
                continue
            self.entries[str(key)[:MAX_SOURCE_CHARS]] = {
                "thai": thai[:MAX_THAI_CHARS],
                "speaker": str(value.get("speaker", ""))[:80],
                "voice": str(value.get("voice", ""))[:16],
                "hits": _safe_int(value.get("hits")),
                "warned": bool(value.get("warned")),
                "stamp": _safe_float(value.get("stamp")),
            }

    def to_dict(self) -> dict:
        return {"entries": dict(list(self.entries.items())[-MAX_ENTRIES:])}

    def prune(self):
        if len(self.entries) <= MAX_ENTRIES:
            return
        ranked = sorted(self.entries.items(), key=lambda item: (item[1]["hits"], item[1]["stamp"]))
        for key, _ in ranked[: len(self.entries) - MAX_ENTRIES]:
            self.entries.pop(key, None)

    # --- learning / reuse ------------------------------------------------------------
    def learn(self, source: str, thai: str, speaker: str = "", voice: str = "",
              warned: bool = False, stamp: float | None = None) -> bool:
        key = key_for(source)
        thai = str(thai).strip()
        if len(key) < MIN_REUSE_CHARS or not thai or len(thai) > MAX_THAI_CHARS:
            return False
        entry = self.entries.get(key)
        if entry is None:
            self.entries[key] = {"thai": thai, "speaker": speaker[:80], "voice": voice[:16],
                                 "hits": 1, "warned": bool(warned),
                                 "stamp": time.monotonic() if stamp is None else stamp}
        else:
            if entry["thai"] != thai:
                entry["thai"] = thai
                entry["hits"] = 1
            else:
                entry["hits"] += 1
            entry["warned"] = bool(warned)
            entry["speaker"] = speaker[:80] or entry["speaker"]
            entry["voice"] = voice[:16] or entry["voice"]
            entry["stamp"] = time.monotonic() if stamp is None else stamp
        self.prune()
        return True

    def get(self, source: str, voice: str = "") -> dict | None:
        """The stored wording, or None when it may no longer be valid.

        The signature must match exactly: a line translated under one locked voice card must not
        be replayed for a different card, and a line stored without a card must not be replayed
        once the player locks one (the model has to respect the new card).
        """
        entry = self.entries.get(key_for(source))
        if not entry or entry["warned"]:
            return None
        if entry.get("voice", "") != voice:
            return None
        return entry

    def lookup(self, blocks, hints, memory=None) -> dict[int, dict]:
        """Blocks that can be rendered straight from memory, with the wording to use.

        Speaker resolution and a voice-signature check happen here so a reused line can never
        contradict a locked character card.
        """
        from .speech import plan_speech       # local import keeps module import light

        hint_by_id = {hint.id: hint for hint in (hints or [])}
        found: dict[int, dict] = {}
        for block in blocks:
            hint = hint_by_id.get(block.id)
            if hint is not None and hint.role in ("name_tag", "hud", "menu", "item", "tooltip"):
                continue
            speaker = (hint.speaker if hint else "") or ""
            if memory is not None and speaker:
                speaker = memory.resolve(speaker) or speaker
            card = memory.card(speaker) if memory is not None and speaker else None
            signature = voice_signature(card)
            entry = self.get(block.text, signature)
            if entry is None:
                continue
            if card and plan_speech(card, None, _empty_cues()).locked and not signature:
                continue
            found[block.id] = {"thai": entry["thai"], "speaker": speaker or entry["speaker"],
                               "hits": entry["hits"]}
        return found

    def stats(self) -> dict:
        total = len(self.entries)
        warned = sum(1 for entry in self.entries.values() if entry["warned"])
        hits = sum(entry["hits"] for entry in self.entries.values())
        return {"entries": total, "needs_review": warned, "reuses": max(0, hits - total)}


def _empty_cues():
    from .speech import SourceCues
    return SourceCues()


# --- glossary helpers ------------------------------------------------------------------

_PAIR_SPLIT = re.compile(r"\s*(?:=|→|->|=>|:)\s*")


def parse_glossary(glossary) -> list[tuple[str, str]]:
    """Accept the textarea format ("Ryza = ไรซ่า" per line or comma separated) or a dict."""
    pairs: list[tuple[str, str]] = []
    if isinstance(glossary, dict):
        for source, thai in glossary.items():
            pairs.append((str(source).strip(), str(thai).strip()))
        return [(s, t) for s, t in pairs if s and t]
    text = str(glossary or "")
    for raw_line in re.split(r"[\n;]+", text):
        for chunk in raw_line.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = _PAIR_SPLIT.split(chunk, maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                pairs.append((parts[0].strip()[:80], parts[1].strip()[:120]))
    return pairs


def glossary_text(pairs) -> str:
    return "\n".join(f"{source} = {thai}" for source, thai in pairs if source and thai)


def check_terms(source: str, thai: str, glossary) -> list[str]:
    """Warn when a line contains a glossary term whose Thai rendering was not used."""
    warnings = []
    for term, expected in parse_glossary(glossary)[:60]:
        if not expected or term.lower() not in source.lower():
            continue
        if expected in thai:
            continue
        warnings.append(f"คำศัพท์: “{term}” ควรใช้ “{expected}” ตาม glossary")
    return warnings[:3]


def _tokens(text: str) -> list[str]:
    tokens = _LATIN_TERM.findall(text) + _CJK_TERM.findall(text)
    # A capitalized word that only ever starts a sentence is usually not a proper noun.
    return [token for token in tokens if len(token) >= 3 or not token.isascii()]


def _thai_substrings(text: str, low: int = 3, high: int = 12) -> set[str]:
    """Every Thai chunk of 3..12 characters — Thai script has no word separators to rely on."""
    chunks = set()
    for start in range(len(text)):
        for size in range(low, high + 1):
            piece = text[start:start + size]
            if len(piece) < low or not all("\u0e00" <= char <= "\u0e7f" for char in piece):
                continue
            chunks.add(piece)
    return chunks


def mine_candidates(pairs, existing=None, min_count: int = 2,
                    limit: int = 12, negative: list[str] | None = None) -> list[dict]:
    """Find repeated source terms and suggest the Thai chunk their translations share.

    ``pairs`` is a list of ``(source, thai)`` from accepted lines. ``negative`` holds Thai lines
    where the term did *not* appear: a chunk that also shows up there is not evidence, so it is
    rejected instead of guessed. A term is only suggested when it appears in at least
    ``min_count`` different source lines and the shared chunk appears in at least that many Thai
    translations of those lines.
    """
    known = {term.lower() for term, _ in parse_glossary(existing or "")}
    negative_chunks: set[str] = set()
    for line in negative or []:
        negative_chunks |= _thai_substrings(str(line))

    by_term: dict[str, list[tuple[str, str]]] = {}
    for source, thai in pairs:
        source, thai = str(source or ""), str(thai or "")
        if not source or not thai:
            continue
        for term in set(_tokens(source)):
            if term.lower() in known:
                continue
            by_term.setdefault(term, []).append((source, thai))

    candidates = []
    for term, samples in by_term.items():
        distinct = {source for source, _ in samples}
        if len(distinct) < min_count:
            continue
        votes: dict[str, int] = {}
        for _, thai in samples:
            for chunk in _thai_substrings(thai):
                votes[chunk] = votes.get(chunk, 0) + 1
        scored = []
        for chunk, count in votes.items():
            if count < min_count or chunk in THAI_STOPWORDS or chunk in negative_chunks:
                continue
            scored.append((count, len(chunk), chunk))
        if not scored:
            continue
        count, _, best = max(scored)
        confidence = min(0.9, 0.4 + 0.1 * count + 0.05 * len(distinct))
        candidates.append({
            "source": term,
            "thai": best,
            "count": len(distinct),
            "confidence": round(confidence, 2),
            "evidence": [f"{source[:60]} → {thai[:60]}" for source, thai in samples[:2]],
        })
    candidates.sort(key=lambda item: (-item["count"], -item["confidence"], item["source"]))
    return candidates[:limit]


def merge_candidates(existing: list, new: list, limit: int = 40) -> list[dict]:
    """Keep the strongest evidence per source term; never duplicate an entry."""
    merged: dict[str, dict] = {}
    for item in list(existing or []) + list(new or []):
        if not isinstance(item, dict) or not item.get("source"):
            continue
        source = str(item["source"])[:80]
        item = {"source": source, "thai": str(item.get("thai", ""))[:120],
                "count": int(item.get("count", 1) or 1),
                "confidence": float(item.get("confidence", 0.4) or 0.4),
                "evidence": [str(line)[:120] for line in (item.get("evidence") or [])][:3]}
        if source in merged:
            merged[source]["count"] = max(merged[source]["count"], item["count"])
            merged[source]["confidence"] = max(merged[source]["confidence"], item["confidence"])
            merged[source]["evidence"] = (merged[source]["evidence"] + item["evidence"])[:3]
        else:
            merged[source] = item
    return sorted(merged.values(), key=lambda item: (-item["count"], item["source"]))[:limit]
