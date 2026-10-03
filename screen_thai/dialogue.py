"""Session dialogue memory: who spoke, to whom, with what tone.

Kept in RAM on purpose — the durable profile only stores the story summary and confirmed
character facts, never a transcript of the player's screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MAX_LINES = 60
TRUSTED = ("name_tag", "user", "confirmed")


@dataclass
class DialogueLine:
    speaker: str
    thai: str
    source: str = ""
    listener: str = ""
    emotion: str = ""
    delivery: str = ""
    speaker_source: str = ""      # name_tag | ai | user
    scene: str = ""
    stamp: float = 0.0

    def compact(self) -> str:
        who = self.speaker or "ไม่ทราบผู้พูด"
        bits = [f"{who}: {self.thai}"]
        if self.emotion:
            bits.append(f"[อารมณ์ {self.emotion}]")
        if self.delivery:
            bits.append(f"[วิธีพูด {self.delivery[:60]}]")
        return " ".join(bits)


@dataclass
class DialogueMemory:
    lines: list[DialogueLine] = field(default_factory=list)

    def add(self, speaker: str, thai: str, source: str = "", listener: str = "",
            emotion: str = "", delivery: str = "", speaker_source: str = "",
            scene: str = "", stamp: float = 0.0) -> None:
        if not thai.strip():
            return
        self.lines.append(DialogueLine(speaker[:80], thai[:600], source[:600], listener[:80],
                                       emotion[:40], delivery[:240], speaker_source[:20],
                                       scene[:200], stamp))
        del self.lines[:-MAX_LINES]

    def clear(self) -> None:
        self.lines.clear()

    def recent(self, count: int = 8) -> list[DialogueLine]:
        return self.lines[-count:]

    def last_speaker(self) -> str:
        for line in reversed(self.lines):
            if line.speaker:
                return line.speaker
        return ""

    def turns(self, speaker: str, window: int = 12) -> int:
        return sum(1 for line in self.lines[-window:] if line.speaker == speaker)

    def speakers(self, window: int = 20) -> list[str]:
        return list(dict.fromkeys(line.speaker for line in self.lines[-window:] if line.speaker))

    def listener_of(self, speaker: str, window: int = 6) -> str:
        for line in reversed(self.lines[-window:]):
            if line.speaker == speaker and line.listener:
                return line.listener
        return ""

    def describe(self, count: int = 8) -> list[dict]:
        return [{"speaker": line.speaker or "ไม่ทราบ",
                 "speaker_source": line.speaker_source,
                 "thai": line.thai,
                 "emotion": line.emotion,
                 "delivery": line.delivery,
                 "to": line.listener}
                for line in self.recent(count)]

    def continuity_note(self) -> str:
        """Short Thai hint that helps the model keep voice consistent between lines."""
        if not self.lines:
            return ""
        speakers = self.speakers()
        if not speakers:
            return ""
        last = self.last_speaker()
        notes = [f"ผู้พูดล่าสุด: {last}"] if last else []
        for name in speakers[:4]:
            turns = self.turns(name)
            last = next((line for line in reversed(self.lines) if line.speaker == name), None)
            if last and last.delivery:
                notes.append(f"{name} ({turns} บรรทัดล่าสุด): วิธีพูดที่ใช้ไปคือ {last.delivery[:80]}")
            elif turns >= 2:
                notes.append(f"{name} พูดไปแล้ว {turns} บรรทัด ให้คงสรรพนามและโทนเดิม")
        return " • ".join(notes)
