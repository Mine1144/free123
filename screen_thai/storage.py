from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import tempfile

from .faces import FaceMemory
from .models import Result, Settings


def data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    folder = base / "ScreenThai"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


DEFAULT_PROFILE = {
    "name": "ทั่วไป", "notes": "", "glossary": "", "summary": "", "relations": [],
    "characters": {}, "research": {}, "research_sources": [],
}

# Sections the UI is allowed to write back; anything else stays read-only for safety.
EDITABLE_KEYS = ("notes", "glossary", "name", "research")


class Store:
    def __init__(self, root: Path | None = None):
        self.root = root or data_dir()

    def load_settings(self) -> Settings:
        raw = read_json(self.root / "settings.json", {})
        return Settings.from_dict(raw if isinstance(raw, dict) else {})

    def save_settings(self, settings: Settings):
        write_json(self.root / "settings.json", settings.to_dict())

    def profile_path(self, name: str) -> Path:
        # User profile names can never become arbitrary filesystem paths.
        digest = hashlib.sha256(name.encode()).hexdigest()[:24]
        return self.root / "profiles" / f"{digest}.json"

    def load_profile(self, name: str) -> dict:
        defaults = dict(DEFAULT_PROFILE)
        defaults["name"] = name
        defaults["characters"] = {}
        defaults["research"] = {}
        defaults["research_sources"] = []
        raw = read_json(self.profile_path(name), {})
        if isinstance(raw, dict):
            for key, value in defaults.items():
                if key in raw and type(raw[key]) is type(value):
                    defaults[key] = raw[key]
        return defaults

    def save_profile(self, name: str, profile: dict):
        write_json(self.profile_path(name), profile)

    def load_memory(self, name: str) -> FaceMemory:
        """Face/character memory for one profile (descriptors only, never images)."""
        profile = self.load_profile(name)
        return FaceMemory({"characters": profile.get("characters", {})})

    def save_memory(self, name: str, memory: FaceMemory):
        profile = self.load_profile(name)
        memory.prune()
        profile["characters"] = memory.to_dict()["characters"]
        self.save_profile(name, profile)

    def save_research(self, name: str, brief: dict, sources: list | None = None):
        profile = self.load_profile(name)
        profile["research"] = brief if isinstance(brief, dict) else {}
        if sources is not None:
            profile["research_sources"] = sources[:12]
        self.save_profile(name, profile)

    def learn(self, name: str, result: Result, memory: FaceMemory | None = None):
        profile = self.load_profile(name)
        if result.scene.summary:
            profile["summary"] = result.scene.summary
        # Upsert bounded, tentative relationships. Manual notes take precedence in the prompt.
        relations = {}
        for rel in profile["relations"] + result.scene.relations:
            if isinstance(rel, dict) and rel.get("from") and rel.get("to"):
                relations[(str(rel["from"]), str(rel["to"]))] = rel
        profile["relations"] = list(relations.values())[-40:]
        self.save_profile(name, profile)
        if memory is None:
            memory = self.load_memory(name)
        changed = False
        for update in result.scene.characters:
            if memory.apply_hypothesis(update):
                changed = True
        for link in getattr(result.scene, "faces", []) or []:
            memory.add_pending(link.get("track", ""), link.get("character", ""),
                               link.get("evidence", ""), float(link.get("confidence", 0)))
        if changed or (getattr(result.scene, "faces", None) or []):
            self.save_memory(name, memory)
        return memory


class Secrets:
    """Never persist a key in settings. On Windows keyring uses Credential Manager."""
    SERVICE = "ScreenThai"

    @staticmethod
    def account(endpoint: str) -> str:
        return hashlib.sha256(endpoint.rstrip("/").encode()).hexdigest()

    def get(self, endpoint: str) -> str:
        import keyring
        return keyring.get_password(self.SERVICE, self.account(endpoint)) or ""

    def set(self, endpoint: str, value: str):
        import keyring
        if value:
            keyring.set_password(self.SERVICE, self.account(endpoint), value)
        else:
            try:
                keyring.delete_password(self.SERVICE, self.account(endpoint))
            except keyring.errors.PasswordDeleteError:
                pass
