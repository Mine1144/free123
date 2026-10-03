"""Self-check: prove which parts of the app really work on this machine.

ScreenThai cannot be fully tested on the machine it was built on, so the app checks itself and
reports honestly. Every result is one of:

* ``ok``   — verified working right now
* ``warn`` — usable but limited, with the limitation stated
* ``fail`` — broken, with a concrete first step
* ``skip`` — could not be checked (and why), never silently treated as fine
"""
from __future__ import annotations

import os
import tempfile
import time

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
ICON = {OK: "✔", WARN: "!", FAIL: "✘", SKIP: "–"}
LABEL = {OK: "ผ่าน", WARN: "มีข้อจำกัด", FAIL: "ไม่ผ่าน", SKIP: "ข้าม"}


def _check(name: str, status: str, detail: str, hint: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "hint": hint}


def check_data_dir(store) -> dict:
    try:
        store.root.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=store.root, suffix=".selftest")
        os.close(fd)
        os.unlink(path)
    except OSError as exc:
        return _check("โฟลเดอร์ข้อมูล", FAIL, f"เขียนไม่ได้: {exc.__class__.__name__}",
                      "ตรวจสิทธิ์ของ %LOCALAPPDATA%\\ScreenThai หรือย้ายที่เก็บด้วย LOCALAPPDATA")
    return _check("โฟลเดอร์ข้อมูล", OK, str(store.root))


def check_settings(settings) -> dict:
    if not settings.model.strip():
        return _check("ตั้งค่าโมเดล", WARN, "ยังไม่ได้ระบุชื่อโมเดล",
                      "กรอกชื่อโมเดลในแท็บตั้งค่า เช่น qwen2.5vl:7b")
    detail = (f"{settings.provider} · {settings.model} · "
              f"scope={settings.scope} · ภาพ={'เปิด' if settings.vision else 'ปิด'}")
    return _check("ตั้งค่าโมเดล", OK, detail)


def check_endpoint(settings) -> dict:
    from .providers import ProviderError, validate_endpoint
    try:
        url = validate_endpoint(settings.endpoint)
    except (ProviderError, ValueError) as exc:
        return _check("ปลายทาง AI", FAIL, str(exc),
                      "ใช้ http://localhost:11434 สำหรับ Ollama หรือ https://… สำหรับบริการภายนอก")
    loopback = any(host in url for host in ("localhost", "127.0.0.1", "[::1]"))
    status = OK if loopback or url.startswith("https://") else WARN
    return _check("ปลายทาง AI", status, url,
                  "" if status == OK else "ปลายทางภายนอกต้องเป็น HTTPS")


def check_credentials() -> dict:
    try:
        import keyring
    except ImportError:
        return _check("ที่เก็บคีย์", WARN, "ไม่มีไลบรารี keyring",
                      "ติดตั้งด้วย pip install keyring หรือใช้คีย์ในหน่วยความจำรอบเดียว")
    backend = getattr(keyring, "get_keyring", lambda: None)()
    name = backend.__class__.__name__ if backend is not None else "unknown"
    if "fail" in name.lower() or "null" in name.lower():
        return _check("ที่เก็บคีย์", WARN, f"backend ไม่พร้อม ({name})",
                      "บน Windows ติดตั้ง keyring เพื่อใช้ Credential Manager")
    return _check("ที่เก็บคีย์", OK, f"พร้อมใช้งาน ({name})")


def check_ocr() -> dict:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return _check("OCR ในเครื่อง", FAIL, "นำเข้า rapidocr_onnxruntime ไม่ได้",
                      "ติดตั้ง dependency ใหม่: pip install -e . (หรือรัน Start-ScreenThai.cmd)")
    try:
        import onnxruntime
    except ImportError:
        return _check("OCR ในเครื่อง", FAIL, "ไม่มี onnxruntime",
                      "ติดตั้ง dependency ใหม่: pip install -e .")
    providers = "; ".join(onnxruntime.get_available_providers())
    return _check("OCR ในเครื่อง", OK, f"โมดูลพร้อม (onnxruntime {onnxruntime.__version__}: {providers})",
                  "โมเดลจะถูกโหลดจริงตอนเริ่มแปลครั้งแรก")


def check_faces(settings) -> dict:
    if not settings.face_analysis or settings.face_backend == "off":
        return _check("ใบหน้า/สีหน้า", SKIP, "ปิดอยู่ตามการตั้งค่า")
    from .vision import build_detector
    detector = build_detector(settings.face_backend, settings.face_model, settings.max_faces)
    if not detector.available:
        return _check("ใบหน้า/สีหน้า", WARN, f"{detector.name}: {detector.reason}",
                      "ติดตั้ง mediapipe (pip install -e \".[faces]\") หรือใช้ backend OpenCV")
    detail = f"{detector.name} ({detector.quality})"
    if detector.name.startswith("haar"):
        return _check("ใบหน้า/สีหน้า", WARN, detail + " · วัดได้เฉพาะบริเวณปากแบบหยาบ",
                      "ติดตั้ง mediapipe เพื่อวัดตา/คิ้ว/เอียงศีรษะได้ละเอียดขึ้น")
    return _check("ใบหน้า/สีหน้า", OK, detail)


def check_layout() -> dict:
    try:
        from .layout import classify
        from .models import Block
        blocks = [Block(0, "Alyssa", (120, 480, 160, 34)),
                  Block(1, "Hello there.", (100, 520, 600, 60))]
        hints = classify(blocks, (1280, 720))
    except Exception as exc:                     # pragma: no cover - defensive
        return _check("อ่านผังหน้าจอ", FAIL, exc.__class__.__name__, "ส่ง log ให้ผู้พัฒนา")
    roles = {hint.role for hint in hints}
    if "name_tag" in roles:
        return _check("อ่านผังหน้าจอ", OK, "แยกป้ายชื่อ/บทสนทนาได้")
    return _check("อ่านผังหน้าจอ", WARN, f"แยกได้เฉพาะ {', '.join(sorted(roles)) or 'ไม่ทราบ'}",
                  "ตรวจว่าป้ายชื่ออยู่ในกรอบข้อความและอยู่ใกล้บรรทัดบทสนทนา")


def check_speech(settings) -> dict:
    if not settings.speech_plan:
        return _check("แผนการพูด", SKIP, "ปิดอยู่ตามการตั้งค่า")
    try:
        from .speech import analyse_source, plan_speech
        cues = analyse_source("先輩、ありがとうございます。")
        plan = plan_speech({"role": "รุ่นพี่"}, None, cues)
    except Exception as exc:                     # pragma: no cover - defensive
        return _check("แผนการพูด", FAIL, exc.__class__.__name__, "ส่ง log ให้ผู้พัฒนา")
    if not plan.describe():
        return _check("แผนการพูด", WARN, "วิเคราะห์ตัวอย่างแล้วไม่ได้แผน",
                      "เพิ่มการ์ดตัวละครหรือความสัมพันธ์ในแท็บตัวละคร")
    return _check("แผนการพูด", OK, plan.describe()[:160])


def check_memory(store, profile_name: str, memory=None, tm=None, brief=None) -> dict:
    parts = []
    try:
        loaded = store.load_profile(profile_name)
    except (OSError, AttributeError):
        loaded = {}
    brief = brief if brief is not None else (loaded.get("research") or {})
    parts.append(f"โปรไฟล์ “{profile_name}”")
    parts.append(f"brief={'มี' if brief else 'ยังไม่มี'}")
    if tm is not None:
        stats = tm.stats()
        parts.append(f"หน่วยความจำคำแปล {stats['entries']} บรรทัด (ใช้ซ้ำ {stats['reuses']})")
    if memory is not None:
        try:
            parts.append(f"การ์ดยืนยัน {len(memory.confirmed_names())} ตัว")
        except AttributeError:
            pass
    status = OK if brief else WARN
    hint = "" if brief else "กด ‘เริ่มวิจัยเกม’ ในแท็บวิจัยเกมก่อนแปล เพื่อให้บริบทครบ"
    return _check("ข้อมูลเกมที่เก็บไว้", status, " · ".join(parts), hint)


def check_thai_font() -> dict:
    """Only runs inside a live Qt application: QFontDatabase aborts the process without one."""
    try:
        from PySide6.QtGui import QFontDatabase
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return _check("ฟอนต์ไทย", SKIP, "ไม่มี Qt ในสภาพแวดล้อมนี้")
    if QApplication.instance() is None:
        return _check("ฟอนต์ไทย", SKIP, "ต้องมีหน้าต่างแอปเปิดอยู่จึงตรวจฟอนต์ได้",
                      "เปิดแอปแล้วกดตรวจอีกครั้ง")
    families = QFontDatabase.families()
    if not families:
        return _check("ฟอนต์ไทย", SKIP, "ยังโหลดรายการฟอนต์ไม่ได้ (offscreen)")
    thai_ok = any(any(key in family for key in ("Thai", "Leelawadee", "Tahoma", "Sarabun",
                                                "Noto Sans Thai", "Angsana", "Cordia"))
                  for family in families)
    if thai_ok:
        return _check("ฟอนต์ไทย", OK, "พบฟอนต์ที่รองรับภาษาไทย")
    return _check("ฟอนต์ไทย", WARN, f"ไม่พบฟอนต์ไทยจาก {len(families)} ตระกูล",
                  "ติดตั้งฟอนต์ไทย (เช่น Leelawadee UI / Sarabun) มิฉะนั้นคำแปลอาจเป็นกล่อง")


def check_ai_reachable(settings, transport=None, key: str = "") -> dict:
    from .providers import ProviderError, list_models, vision_model_hint
    try:
        models = list_models(settings, transport=transport, key=key)
    except ProviderError as exc:
        return _check("เชื่อมต่อบริการ AI", WARN, str(exc),
                      "เปิด Ollama หรือตรวจ URL/คีย์ แล้วกดตรวจอีกครั้ง")
    except Exception as exc:                     # pragma: no cover - network paths
        return _check("เชื่อมต่อบริการ AI", WARN, f"ตรวจไม่ได้: {exc.__class__.__name__}")
    if not models:
        return _check("เชื่อมต่อบริการ AI", WARN, "เชื่อมต่อได้แต่ไม่พบโมเดล",
                      "ติดตั้งโมเดลก่อน เช่น ollama pull qwen2.5vl:7b")
    installed = settings.model in models
    vision = [name for name in models if vision_model_hint(name)]
    detail = f"พบ {len(models)} โมเดล · โมเดลที่ตั้งไว้ {'มี' if installed else 'ไม่พบ'}"
    if vision:
        detail += f" · โมเดลภาพ {len(vision)}: " + ", ".join(vision[:3])
    if not installed:
        return _check("เชื่อมต่อบริการ AI", WARN, detail,
                      f"ติดตั้ง {settings.model} หรือเปลี่ยนชื่อโมเดลในแท็บตั้งค่า")
    if settings.vision and not vision_model_hint(settings.model):
        return _check("เชื่อมต่อบริการ AI", WARN, detail,
                      "โมเดลนี้อาจไม่รับภาพ: ปิด ‘ให้ AI ดูภาพ’ หรือเปลี่ยนเป็นโมเดล vision")
    return _check("เชื่อมต่อบริการ AI", OK, detail)


def run_checks(settings, store, *, profile_name: str = "", memory=None, tm=None, brief=None,
               probe_ai: bool = False, transport=None, key: str = "") -> list[dict]:
    """Run every cheap check; the network probe only when the user asked for it."""
    started = time.monotonic()
    checks = [
        check_data_dir(store),
        check_settings(settings),
        check_endpoint(settings),
        check_credentials(),
        check_ocr(),
        check_faces(settings),
        check_layout(),
        check_speech(settings),
        check_memory(store, profile_name or settings.profile, memory, tm, brief),
        check_thai_font(),
    ]
    if probe_ai:
        checks.append(check_ai_reachable(settings, transport=transport, key=key))
    else:
        checks.append(_check("เชื่อมต่อบริการ AI", SKIP, "ยังไม่ได้ตรวจ (กด ‘ตรวจการเชื่อมต่อ’)",
                             "การตรวจจะเรียกดูรายชื่อโมเดลเท่านั้น ไม่ส่งข้อความหรือภาพ"))
    checks.append(_check("เวลาในการตรวจ", OK, f"{time.monotonic() - started:.2f} วินาที"))
    return checks


def summary(checks) -> dict:
    counts = {status: 0 for status in (OK, WARN, FAIL, SKIP)}
    for item in checks:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    counts["total"] = len(checks)
    return counts


def report_text(checks) -> str:
    lines = []
    for item in checks:
        lines.append(f"{ICON.get(item['status'], '?')} {item['name']}: {item['detail']}")
        if item.get("hint"):
            lines.append(f"   → {item['hint']}")
    counts = summary(checks)
    lines.append(f"\nรวม {counts['total']} รายการ · ผ่าน {counts[OK]} · "
                 f"มีข้อจำกัด {counts[WARN]} · ไม่ผ่าน {counts[FAIL]} · ข้าม {counts[SKIP]}")
    return "\n".join(lines)
