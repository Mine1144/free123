"""Thai speech planning: how a specific character must talk in a specific scene.

This is deliberately a rule engine, not a model guess. It reads (a) what the source text says
about honorifics/pronouns/politeness and (b) what the user confirmed about the characters,
then produces a pronoun plan the AI must follow. Confirmed values are locked: the AI may not
replace them, and the finished Thai line is checked against the plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- Source-language evidence -----------------------------------------------------------------

HONORIFICS = {
    "\u3055\u3093": "คุยทั่วไปแบบสุภาพ",          # さん
    "\u3061\u3083\u3093": "สนิท/เอ็นดู",            # ちゃん
    "\u304f\u3093": "สนิท ผู้พูดมักอาวุโสกว่า",      # くん
    "\u3055\u307e": "สุภาพมาก/เจ้านาย",             # 様
    "\u6bbf": "สุภาพมาก/เจ้านาย",                   # 殿
    "\u5148\u8f29": "รุ่นพี่",                      # 先輩
    "\u5148\u751f": "ครู/หมอ",                      # 先生
    "\u90e8\u9577": "หัวหน้าแผนก",                   # 部長
    "\u793e\u9577": "ประธานบริษัท",                  # 社長
    "\u54e5\u54e5": "พี่ชาย",                       # 哥哥
    "\u59d0\u59d0": "พี่สาว",                       # 姐姐
    "\u5f1f\u5f1f": "น้องชาย",                      # 弟弟
    "\u59b9\u59b9": "น้องสาว",                      # 妹妹
    "\u5c0f\u59d0": "คุณผู้หญิง",                    # 小姐
    "\ub2d8": "คุณ (สุภาพ)",                        # 님
    "\uc528": "คุณ (กันเอง)",                        # 씨
    "\uc624\ube60": "พี่ชาย (หญิงพูด)",              # 오빠
    "\uc5b8\ub2c8": "พี่สาว (หญิงพูด)",              # 언니
    "\ud615": "พี่ชาย (ชายพูด)",                     # 형
    "\ub204\ub098": "พี่สาว (ชายพูด)",               # 누나
    "\ub3d9\uc0dd": "น้อง",                          # 동생
    "mr.": "คุณ (ชาย)",
    "mrs.": "คุณ (หญิง แต่งงาน)",
    "ms.": "คุณ (หญิง)",
    "miss": "คุณ (หญิง)",
    "sir": "ท่าน/คุณ (ชาย)",
    "ma'am": "ท่าน/คุณ (หญิง)",
    "madam": "ท่าน/คุณ (หญิง)",
    "bro": "พี่/น้องชาย (กันเอง)",
    "sis": "พี่/น้องสาว (กันเอง)",
    "brother": "พี่/น้องชาย",
    "sister": "พี่/น้องสาว",
    "big bro": "พี่ชาย",
    "big sis": "พี่สาว",
}

KINSHIP_OLDER = ("\u5148\u8f29", "\u5144", "\u59ca", "\u54e5\u54e5", "\u59d0\u59d0",
                 "\uc624\ube60", "\uc5b8\ub2c8", "\ud615", "\ub204\ub098", "big bro", "big sis",
                 "older brother", "older sister")
KINSHIP_YOUNGER = ("\u5f1f", "\u59b9", "\u5f1f\u5f1f", "\u59b9\u59b9", "\ub3d9\uc0dd",
                   "little brother", "little sister", "younger brother", "younger sister")

FIRST_PERSON = {
    "ja": (("\u79c1", "ทางการ/กลาง"), ("\u50d5", "ผู้ชาย สุภาพ"), ("\u4ffa", "ผู้ชาย หยาบ/กันเอง"),
           ("\u3042\u305f\u3057", "ผู้หญิง กันเอง"), ("\u308f\u305f\u304f\u3057", "ทางการมาก"),
           ("\u4ffa\u305f\u3061", "กลุ่ม กันเอง"), ("\u79c1\u305f\u3061", "กลุ่ม สุภาพ")),
    "zh": (("\u6211", "กลาง"), ("\u6211\u4eec", "กลุ่ม"), ("\u672c\u5bab", "ทางการ (เก่า)")),
    "ko": (("\uc800", "ถ่อมตัว/สุภาพ"), ("\ub098", "กันเอง"), ("\uc6b0\ub9ac", "กลุ่ม")),
}
SECOND_PERSON = {
    "ja": (("\u3042\u306a\u305f", "กลาง/สุภาพ"), ("\u541b", "สนิทหรือสูงกว่า"),
           ("\u304a\u524d", "หยาบ/สนิทมาก"), ("\u3042\u3093\u305f", "กันเอง")),
    "zh": (("\u60a8", "สุภาพ"), ("\u4f60", "กลาง")),
    "ko": (("\ub2f9\uc2e0", "กลาง"), ("\ub108", "กันเอง")),
    "en": (("you", "กลาง"),),
}

MALE_HINT_PRONOUNS = ("\u50d5", "\u4ffa")                    # 僕 / 俺
FEMALE_HINT_PRONOUNS = ("\u3042\u305f\u3057", "\u308f\u305f\u304f\u3057")  # あたし / わたくし

POLITE_MARKERS = ("\u3067\u3059", "\u307e\u3059", "\u304f\u3060\u3055\u3044", "\u3054\u3056\u3044\u307e\u3059",
                  "\u60a8", "\u8bf7", "\uc2b5\ub2c8\ub2e4", "\uc694", "please", "could you", "would you",
                  "sir", "ma'am")
CASUAL_MARKERS = ("\u3060\u305c", "\u3060\u305e", "\u3058\u3083\u306d\u3047", "\u304b\u3088", "\u305c", "\u305e",
                  "gonna", "wanna", "yeah", "hey", "dude", "buddy", "\uc57c", "\ub2e4")
ROUGH_MARKERS = ("\u3066\u3081\u3047", "\u304d\u3055\u307e", "\u3076\u3063\u3053\u308d\u3059",
                 "\u6df7\u86cb", "\u6eda", "\uc9c4\uc9dc", "\ub2f9\uc7a5", "damn", "hell", "bastard", "shut up")

# กู มึง วะ โว้ย ไอ้ หาย(?) เห้ย
THAI_ROUGH = ("\u0e01\u0e39", "\u0e21\u0e36\u0e07", "\u0e27\u0e30", "\u0e42\u0e27\u0e49\u0e22",
              "\u0e44\u0e2d\u0e49", "\u0e40\u0e2b\u0e49\u0e22")
# ครับ ค่ะ ค่า เคร๊ะ พะคะ
THAI_POLITE_PARTICLES = ("\u0e04\u0e23\u0e31\u0e1a", "\u0e04\u0e48\u0e30", "\u0e04\u0e48\u0e32",
                         "\u0e40\u0e04\u0e23\u0e4a\u0e30", "\u0e1e\u0e30\u0e04\u0e30")
MALE_PARTICLES = ("\u0e04\u0e23\u0e31\u0e1a",)               # ครับ
FEMALE_PARTICLES = ("\u0e04\u0e48\u0e30", "\u0e04\u0e48\u0e32", "\u0e14\u0e34\u0e09\u0e31\u0e19")  # ค่ะ ค่า ดิฉัน

# กู ข้า ข้าพเจ้า ฉัน ดิฉัน ผม หนู เรา เธอ คุณ แก มึง ท่าน พี่ น้อง
THAI_PRONOUNS = ("\u0e01\u0e39", "\u0e02\u0e49\u0e32", "\u0e02\u0e49\u0e32\u0e1e\u0e40\u0e08\u0e49\u0e32",
                 "\u0e09\u0e31\u0e19", "\u0e14\u0e34\u0e09\u0e31\u0e19", "\u0e1c\u0e21", "\u0e2b\u0e19\u0e39",
                 "\u0e40\u0e23\u0e32", "\u0e40\u0e18\u0e2d", "\u0e04\u0e38\u0e13", "\u0e41\u0e01",
                 "\u0e21\u0e36\u0e07", "\u0e17\u0e48\u0e32\u0e19", "\u0e1e\u0e35\u0e48", "\u0e19\u0e49\u0e2d\u0e07")

RELATION_PRESETS = {
    "พี่น้อง": {
        "older": {"self": "พี่", "address": "น้อง", "register": "กันเอง", "particles": "นะ"},
        "younger": {"self": "หนู", "address": "พี่", "register": "สุภาพ", "particles": "ครับ/ค่ะ"},
        "any": {"self": "เรา", "address": "พี่/น้อง", "register": "กันเอง", "particles": "นะ"},
    },
    "พ่อแม่ลูก": {
        "parent": {"self": "พ่อ/แม่", "address": "ลูก", "register": "กันเอง", "particles": "นะ"},
        "child": {"self": "หนู", "address": "แม่/พ่อ", "register": "สุภาพ", "particles": "ครับ/ค่ะ"},
    },
    "ปู่ย่าตายาย": {
        "elder": {"self": "ปู่/ย่า/ตา/ยาย", "address": "หลาน", "register": "กันเอง", "particles": "นะ"},
        "younger": {"self": "หนู", "address": "ปู่/ย่า/ตา/ยาย", "register": "สุภาพ", "particles": "ครับ/ค่ะ"},
    },
    "คู่รัก": {"any": {"self": "เรา/ฉัน", "address": "เธอ", "register": "กันเอง", "particles": "นะ"}},
    "เพื่อน": {"any": {"self": "เรา/ฉัน", "address": "เธอ/แก", "register": "กันเอง", "particles": "นะ"}},
    "เพื่อนสนิท": {"any": {"self": "เรา", "address": "แก", "register": "กันเอง", "particles": "นะ/เหอะ"}},
    "รุ่นพี่รุ่นน้อง": {
        "older": {"self": "พี่", "address": "น้อง", "register": "กันเอง", "particles": "นะ"},
        "younger": {"self": "เรา/ผม", "address": "พี่", "register": "สุภาพ", "particles": "ครับ/ค่ะ"},
        "any": {"self": "เรา", "address": "พี่/น้อง", "register": "กันเอง", "particles": "นะ"},
    },
    "ครูนักเรียน": {
        "teacher": {"self": "ครู", "address": "นักเรียน/หนู", "register": "กึ่งทางการ", "particles": "นะ"},
        "student": {"self": "หนู/ผม", "address": "ครู/อาจารย์", "register": "สุภาพ", "particles": "ครับ/ค่ะ"},
    },
    "เจ้านายลูกน้อง": {
        "boss": {"self": "ผม/ฉัน", "address": "คุณ", "register": "กึ่งทางการ", "particles": "ครับ/ค่ะ"},
        "staff": {"self": "ผม/หนู", "address": "ท่าน/หัวหน้า/คุณ", "register": "สุภาพ", "particles": "ครับ/ค่ะ"},
    },
    "เจ้าเมืองไพร่": {
        "lord": {"self": "ข้า", "address": "เจ้า/ท่าน", "register": "โบราณ", "particles": "ดอก"},
        "servant": {"self": "ข้าพระพุทธเจ้า/หม่อมฉัน", "address": "ฝ่าพระบาท/ท่าน",
                    "register": "โบราณ", "particles": "พระพุทธเจ้าข้า/เพคะ"},
    },
    "หมอคนไข้": {
        "doctor": {"self": "หมอ/ผม", "address": "คุณ", "register": "กึ่งทางการ", "particles": "นะ/ครับ"},
        "patient": {"self": "ผม/หนู", "address": "คุณหมอ", "register": "สุภาพ", "particles": "ครับ/ค่ะ"},
    },
    "ศัตรู": {"any": {"self": "ข้า", "address": "เจ้า", "register": "สนิทหยาบ", "particles": "วะ"}},
    "คนแปลกหน้า": {"any": {"self": "ผม/ฉัน", "address": "คุณ", "register": "สุภาพ", "particles": "ครับ/ค่ะ"}},
}

RELATION_KEYWORDS = (
    ("พี่น้อง", ("พี่น้อง", "พี่ชาย", "พี่สาว", "น้องชาย", "น้องสาว", "brother", "sister",
                "\u5144", "\u59ca", "\u5f1f", "\u59b9", "\uc624\ube60", "\uc5b8\ub2c8", "\ud615",
                "\ub204\ub098", "\ub3d9\uc0dd")),
    ("พ่อแม่ลูก", ("พ่อ", "แม่", "ลูกชาย", "ลูกสาว", "father", "mother", "dad", "mom",
                  "\u7236", "\u6bcd", "\uc544\ubc84\uc9c0", "\uc5b4\uba38\ub2c8")),
    ("ปู่ย่าตายาย", ("ปู่", "ย่า", "ตา", "ยาย", "grandfather", "grandmother", "\ud560\uc544\ubc84\uc9c0")),
    ("คู่รัก", ("คู่รัก", "คนรัก", "แฟน", "สามี", "ภรรยา", "lover", "wife", "husband",
               "girlfriend", "boyfriend", "\uc57c\uc778")),
    ("เพื่อนสนิท", ("เพื่อนสนิท", "best friend", "\u4eb2\u53cb")),
    ("เพื่อน", ("เพื่อน", "friend", "\u670b\u53cb")),
    ("รุ่นพี่รุ่นน้อง", ("รุ่นพี่", "รุ่นน้อง", "senior", "junior", "\u5148\u8f29")),
    ("ครูนักเรียน", ("ครู", "อาจารย์", "นักเรียน", "นักศึกษา", "teacher", "student", "\uc120\uc0dd")),
    ("เจ้านายลูกน้อง", ("เจ้านาย", "หัวหน้า", "ลูกน้อง", "boss", "employee")),
    ("เจ้าเมืองไพร่", ("กษัตริย์", "ราชา", "องค์หญิง", "องค์ชาย", "ขุนนาง", "ทาส", "king", "queen",
                     "princess", "prince")),
    ("หมอคนไข้", ("หมอ", "พยาบาล", "คนไข้", "doctor", "nurse", "patient", "\uc758\uc0ac")),
    ("ศัตรู", ("ศัตรู", "คู่อริ", "enemy", "rival", "villain")),
)

ROLE_WORDS = {
    "parent": ("พ่อ", "แม่", "dad", "mom", "father", "mother"),
    "child": ("ลูก", "son", "daughter"),
    "elder": ("ปู่", "ย่า", "ตา", "ยาย", "grand"),
    "teacher": ("ครู", "อาจารย์", "teacher", "professor"),
    "student": ("นักเรียน", "นักศึกษา", "student"),
    "boss": ("เจ้านาย", "หัวหน้า", "boss", "manager"),
    "staff": ("ลูกน้อง", "พนักงาน", "employee"),
    "doctor": ("หมอ", "แพทย์", "พยาบาล", "doctor", "nurse"),
    "patient": ("คนไข้", "patient"),
    "lord": ("กษัตริย์", "ราชา", "เจ้า", "องค์", "king", "queen", "prince", "princess", "lord"),
    "servant": ("ทาส", "ข้ารับใช้", "servant"),
}

NAME_PATTERN = re.compile(r"[\u3040-\u30ff]{2,5}\u3055\u3093|\b[A-Z][a-z]{2,12}\b")


@dataclass(frozen=True)
class SourceCues:
    language: str = "unknown"
    honorifics: tuple[str, ...] = ()
    kinship: tuple[str, ...] = ()
    pronouns: tuple[str, ...] = ()
    politeness: str = "unknown"        # formal | polite | casual | rough | neutral
    relative_age: str = "unknown"      # speaker_older | speaker_younger | unknown
    gender_hint: str = "unknown"       # male | female | unknown (source wording only)
    names_called: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PronounPlan:
    self_pronoun: str = ""
    address: str = ""
    particles: str = ""
    register: str = ""
    relation: str = ""
    reasons: tuple[str, ...] = ()
    locked: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = []
        if self.self_pronoun:
            parts.append(f"แทนตัวเอง: {self.self_pronoun}")
        if self.address:
            parts.append(f"เรียกอีกฝ่าย: {self.address}")
        if self.particles:
            parts.append(f"คำลงท้าย: {self.particles}")
        if self.register:
            parts.append(f"ระดับภาษา: {self.register}")
        text = " • ".join(parts)
        if self.locked:
            text = "ล็อกโดยผู้ใช้ (" + ", ".join(self.locked) + ") • " + text
        return text


def detect_language(text: str) -> str:
    for low, high, name in (("\u3040", "\u30ff", "ja"), ("\uac00", "\ud7af", "ko"),
                            ("\u4e00", "\u9fff", "zh"), ("\u0e00", "\u0e7f", "th"),
                            ("\u0400", "\u04ff", "ru")):
        if any(low <= ch <= high for ch in text):
            return name
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"


def analyse_source(text: str) -> SourceCues:
    """Cheap, offline evidence extraction from the OCR text of one line."""
    lowered = text.casefold()
    language = detect_language(text)
    honorifics = tuple(name for name in HONORIFICS if name in text or name in lowered)
    older = [k for k in KINSHIP_OLDER if k in text or k in lowered]
    younger = [k for k in KINSHIP_YOUNGER if k in text or k in lowered]
    pronouns = []
    for _lang, table in FIRST_PERSON.items():
        for token, style in table:
            if token in text or token in lowered:
                pronouns.append(f"ผู้พูด: {token} ({style})")
    for _lang, table in SECOND_PERSON.items():
        for token, style in table:
            if token in text or token in lowered:
                pronouns.append(f"ผู้ฟัง: {token} ({style})")
    if any(marker in lowered for marker in ("please", "could you", "would you", "\u3067\u3059",
                                            "\u307e\u3059", "\u304f\u3060\u3055\u3044", "\u60a8",
                                            "\uc2b5\ub2c8\ub2e4")):
        politeness = "formal"
    elif any(marker in lowered for marker in POLITE_MARKERS):
        politeness = "polite"
    elif any(marker in lowered for marker in ROUGH_MARKERS):
        politeness = "rough"
    elif any(marker in lowered for marker in CASUAL_MARKERS):
        politeness = "casual"
    else:
        politeness = "neutral"
    if older and not younger:
        relative_age = "speaker_older"
    elif younger and not older:
        relative_age = "speaker_younger"
    else:
        relative_age = "unknown"
    if any(token in text for token in MALE_HINT_PRONOUNS):
        gender_hint = "male"
    elif any(token in text for token in FEMALE_HINT_PRONOUNS):
        gender_hint = "female"
    else:
        gender_hint = "unknown"
    names = tuple(dict.fromkeys(NAME_PATTERN.findall(text)))[:4]
    notes = []
    if honorifics:
        meanings = ", ".join(f"{name} = {HONORIFICS[name]}" for name in honorifics[:4])
        notes.append("คำเรียก/คำสุภาพในต้นฉบับ: " + meanings +
                     " • ต้องรักษาความสัมพันธ์ให้ตรง ไม่ทำให้สนิทหรือห่างเกิน")
    if relative_age == "speaker_older":
        notes.append("ต้นฉบับบอกว่าผู้พูดอาวุโสกว่า")
    elif relative_age == "speaker_younger":
        notes.append("ต้นฉบับบอกว่าผู้พูดอ่อนกว่า")
    if gender_hint != "unknown":
        notes.append("สรรพนามต้นฉบับชี้เพศผู้พูดได้ (ยังไม่ยืนยัน)")
    return SourceCues(language, honorifics, tuple(older + younger), tuple(pronouns), politeness,
                      relative_age, gender_hint, names, tuple(notes))


def relation_type(speaker_card: dict, listener_card: dict) -> str:
    """Classify a relationship only from words the user or the research step already stated."""
    haystack = " ".join(str((card or {}).get(key, "")) for card in (speaker_card, listener_card)
                        for key in ("relation", "role", "personality", "speech_style"))
    for relation, keywords in RELATION_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return relation
    return ""


def _pick_side(table: dict, speaker_card: dict, cues: SourceCues) -> str:
    if set(table) <= {"any"}:
        return "any"
    if "older" in table or "younger" in table:
        gap = str(speaker_card.get("age_gap", "")).strip() or cues.relative_age
        if gap in ("older", "speaker_older"):
            return "older"
        if gap in ("younger", "speaker_younger"):
            return "younger"
        return "any" if "any" in table else "older"
    role = " ".join(str(speaker_card.get(key, ""))
                    for key in ("role", "relation", "notes", "personality"))
    for key, words in ROLE_WORDS.items():
        if key in table and any(word in role for word in words):
            return key
    return "any" if "any" in table else next(iter(table))


def plan_speech(speaker_card: dict | None, listener_card: dict | None, cues: SourceCues,
                emotion: str = "") -> PronounPlan:
    """Build the plan. Values the user confirmed always win and are reported as locked."""
    speaker_card = speaker_card or {}
    listener_card = listener_card or {}
    voice = speaker_card.get("voice") if isinstance(speaker_card.get("voice"), dict) else {}
    voice = voice or {}
    locked_by_user = bool(voice.get("locked"))
    reasons: list[str] = []
    locked: list[str] = []

    def locked_value(label: str, key: str) -> str:
        value = str(voice.get(key, "")).strip()
        if value and locked_by_user:
            locked.append(label)
            reasons.append(f"{label}: ใช้ค่าที่ผู้ใช้ยืนยันแล้ว ({value}) และห้าม AI เปลี่ยน")
            return value
        return ""

    self_pronoun = locked_value("สรรพนามแทนตัวเอง", "self")
    address = locked_value("สรรพนามเรียกอีกฝ่าย", "address")
    particles = locked_value("คำลงท้าย", "particles")
    register = locked_value("ระดับภาษา", "register")

    relation = relation_type(speaker_card, listener_card)
    preset: dict = {}
    if relation:
        table = RELATION_PRESETS.get(relation, {})
        side = _pick_side(table, speaker_card, cues) if table else "any"
        preset = dict(table.get(side) or table.get("any") or {})
        reasons.append(f"ความสัมพันธ์จากข้อมูลที่บันทึกไว้: {relation} (บทบาท {side})")
        if side == "any" and ("older" in table or "younger" in table):
            reasons.append("ยังไม่ยืนยันว่าใครอาวุโสกว่า จึงเลี่ยงสรรพนามที่ฟันธงอาวุโส")
    else:
        reasons.append("ยังไม่รู้ความสัมพันธ์ของคู่นี้ ใช้สรรพนามกลางและไม่เดาเพศ")

    self_pronoun = self_pronoun or preset.get("self", "")
    address = address or preset.get("address", "")
    particles = particles or preset.get("particles", "")
    register = register or preset.get("register", "")

    gender = str(speaker_card.get("gender", "unknown")).strip()
    gender_source = str(speaker_card.get("gender_source", "")).strip()
    if gender in ("male", "female") and gender_source == "user":
        if gender == "male":
            self_pronoun = self_pronoun or "ผม"
            particles = particles or "ครับ"
        else:
            self_pronoun = self_pronoun or "ฉัน"
            particles = particles or "ค่ะ"
        reasons.append("เพศผู้พูดยืนยันโดยผู้ใช้ จึงเลือกคำลงท้ายได้เหมาะสม")
    elif gender in ("male", "female"):
        reasons.append("มีเพศจาก AI/แหล่งข้อมูลแต่ผู้ใช้ยังไม่ยืนยัน จึงไม่ผูก ครับ/ค่ะ ให้เดา")
    if cues.gender_hint != "unknown":
        reasons.append(f"ต้นฉบับชี้แนวโน้มเพศผู้พูด: {cues.gender_hint} (ยังไม่ยืนยัน)")
    if cues.relative_age == "speaker_older":
        reasons.append("ต้นฉบับบอกว่าผู้พูดอาวุโสกว่า จึงใช้สรรพนามแบบพี่ได้")
    elif cues.relative_age == "speaker_younger":
        reasons.append("ต้นฉบับบอกว่าผู้พูดอ่อนกว่า จึงใช้สรรพนามแบบน้อง/ผู้น้อย")

    if not self_pronoun:
        self_pronoun = "ฉัน" if gender == "female" else "ผม" if gender == "male" else "เรา"
    if not address:
        address = "คุณ"
    if cues.politeness == "rough":
        if not locked_by_user:
            particles = particles or "วะ"
            register = register or "สนิทหยาบ"
        reasons.append("ต้นฉบับมีคำหยาบ จึงคงความดิบไว้ ไม่ทำให้สุภาพเกิน")
    elif cues.politeness in ("formal", "polite") and not register:
        register = "สุภาพ"
        reasons.append("ต้นฉบับสุภาพ จึงคงความสุภาพและคำลงท้ายให้เหมาะกับคู่สนทนา")
    elif not register:
        register = "ทั่วไป"
    if emotion:
        reasons.append(f"อารมณ์ในฉาก: {emotion} — ให้เลือกคำลงท้าย/จังหวะให้ตรงอารมณ์นี้")
    return PronounPlan(self_pronoun, address, particles, register, relation, tuple(reasons),
                       tuple(dict.fromkeys(locked)))


def check_translation(thai: str, plan: PronounPlan, gender_source: str = "") -> tuple[str, ...]:
    """Post-translation QA against the locked plan. Returns Thai warning strings."""
    warnings: list[str] = []
    if plan.locked:
        locked_pronoun = plan.self_pronoun.split("/")[0].strip()
        if locked_pronoun and locked_pronoun not in thai:
            others = [word for word in THAI_PRONOUNS
                      if word not in plan.self_pronoun and word in thai]
            if others:
                warnings.append("สรรพนามอาจไม่ตรงกับที่ล็อกไว้: " + ", ".join(others[:3]))
    if plan.register in ("สุภาพ", "กึ่งทางการ", "ทางการ", "โบราณ"):
        rough = [word for word in THAI_ROUGH if word in thai]
        if rough:
            warnings.append("โทนที่ล็อกไว้สุภาพ แต่พบคำหยาบ: " + ", ".join(rough[:3]))
    if plan.particles:
        expected = plan.particles.split("/")[0].strip()
        if expected and expected not in thai and any(p in thai for p in THAI_POLITE_PARTICLES):
            warnings.append("คำลงท้ายไม่ตรงกับแผน (คาดว่า " + plan.particles + ")")
    if gender_source == "user":
        if gender_source and (plan.self_pronoun in ("ผม",) or "ครับ" in plan.particles):
            wrong = [word for word in FEMALE_PARTICLES if word in thai]
            if wrong:
                warnings.append("ผู้พูดถูกยืนยันว่าเป็นชาย แต่พบคำลงท้ายหญิง: " + ", ".join(wrong[:2]))
        if gender_source and (plan.self_pronoun in ("ฉัน", "ดิฉัน", "หนู") or "ค่ะ" in plan.particles):
            if "ครับ" in thai:
                warnings.append("ผู้พูดถูกยืนยันว่าเป็นหญิง แต่พบคำลงท้ายชาย: ครับ")
    return tuple(dict.fromkeys(warnings))
