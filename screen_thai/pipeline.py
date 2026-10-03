from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal, Slot

from .faces import FaceMemory
from .models import Block, Settings
from .providers import AIClient


def extract_blocks(raw, width: int, height: int, confidence: float, limit: int) -> list[Block]:
    entries = []
    for points, text, score in raw or []:
        text = str(text).strip()
        if float(score) < confidence or not text or not any(ch.isalpha() for ch in text):
            continue
        if all(not ch.isalpha() or "\u0e00" <= ch <= "\u0e7f" for ch in text):
            continue
        x = max(0, min(width - 1, int(min(p[0] for p in points))))
        y = max(0, min(height - 1, int(min(p[1] for p in points))))
        right = max(x + 1, min(width, int(max(p[0] for p in points))))
        bottom = max(y + 1, min(height, int(max(p[1] for p in points))))
        entries.append((text[:1500], (x, y, right-x, bottom-y), float(score)))
    entries.sort(key=lambda e: (e[1][1], e[1][0]))
    return [Block(i, text, box, score) for i, (text, box, score) in enumerate(entries[:limit])]


class OCRWorker(QObject):
    done = Signal(int, object, object, str)

    def __init__(self):
        super().__init__()
        self.engine = None

    @Slot(int, object, object)
    def process(self, epoch: int, image: Image.Image, settings: Settings):
        try:
            if self.engine is None:
                from rapidocr_onnxruntime import RapidOCR
                self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
            raw, _ = self.engine(np.asarray(image.convert("RGB"))[:, :, ::-1].copy())
            blocks = extract_blocks(raw, image.width, image.height,
                                    settings.min_confidence, settings.max_blocks)
            self.done.emit(epoch, image, blocks, "")
        except Exception:
            self.done.emit(epoch, image, [], "OCR ทำงานไม่สำเร็จ ตรวจไฟล์โมเดลและ Visual C++ Runtime")


class VisionWorker(QObject):
    """Face detection, expression cues and appearance sampling — always local, never sent as an image.

    The worker receives a snapshot of the profile's character cards (with descriptors) and
    returns observations that already carry confirmed names / unconfirmed suggestions.
    """

    done = Signal(int, int, object, str)

    def __init__(self):
        super().__init__()
        self._engine = None
        self._key = None

    def _engine_for(self, settings: Settings):
        key = (settings.face_backend, settings.face_model, settings.max_faces,
               settings.face_analysis, settings.expression_cues)
        if self._engine is None or self._key != key:
            from .vision import VisionEngine
            if self._engine is not None:
                self._engine.close()
            self._engine = VisionEngine(settings)
            self._key = key
        return self._engine

    @Slot(int, int, object, object, object)
    def process(self, epoch: int, revision: int, image: Image.Image, settings: Settings,
                characters: object):
        engine = None
        try:
            engine = self._engine_for(settings)
            memory = FaceMemory({"characters": characters if isinstance(characters, dict) else {}})
            observations = engine.analyze(image, memory)
            self.done.emit(epoch, revision, observations, "")
        except Exception as exc:
            message = str(exc) if isinstance(exc, ValueError) \
                else "วิเคราะห์ใบหน้าไม่สำเร็จ (ข้ามเฉพาะฟีเจอร์นี้)"
            self.done.emit(epoch, revision, [], message)

    def shutdown(self):
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None


class AIWorker(QObject):
    done = Signal(int, int, object, str, object)

    @Slot(int, int, object, object, object, str, object, object)
    def process(self, epoch, revision, blocks, image, settings, key, profile, frame):
        try:
            memory = FaceMemory({"characters": profile.get("characters", {})}) \
                if isinstance(profile, dict) else FaceMemory({})
            result = AIClient(settings, key).translate(blocks, image, profile,
                                                       frame if isinstance(frame, dict) else None,
                                                       memory)
            self.done.emit(epoch, revision, result, "", memory)
        except Exception as exc:
            from .providers import ProviderError
            message = str(exc) if isinstance(exc, (ProviderError, ValueError)) \
                else "เกิดข้อผิดพลาดภายใน AI client"
            self.done.emit(epoch, revision, None, message, None)


class DownloadWorker(QObject):
    """Download one URL into one file. Runs in its own thread so the UI stays responsive.

    Progress is throttled to ~4 updates/second (a 6 GB file would otherwise flood the event
    queue), cancellation is cooperative, and a partial file is kept so the next attempt resumes.
    """

    progress = Signal(str, int, int)      # label, bytes done, bytes total
    done = Signal(str, str, str)          # label, path ("" on failure), error

    def __init__(self):
        super().__init__()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @Slot(str, str, str)
    def process(self, label: str, url: str, target: str):
        from .llama import Downloader, LlamaError
        self._cancel = False
        last = [0.0]

        def report(written: int, total: int):
            now = time.monotonic()
            if now - last[0] >= 0.25 or written == total:
                last[0] = now
                self.progress.emit(label, written, total)

        try:
            result = Downloader().fetch(url, Path(target), progress=report,
                                        cancel=lambda: self._cancel)
            self.done.emit(label, str(result.path), "")
        except LlamaError as exc:
            self.done.emit(label, "", str(exc))
        except Exception as exc:                      # pragma: no cover - defensive
            self.done.emit(label, "", f"ดาวน์โหลดล้มเหลว: {exc.__class__.__name__}")


class ResearchWorker(QObject):
    """Search + fetch + summarise. Runs only when the player asks for it."""

    progress = Signal(str)
    done = Signal(object, object, str)

    @Slot(str, object, str, str, str, object)
    def process(self, game: str, settings: Settings, key: str, query: str, extra_text: str,
                urls: object):
        from . import research
        try:
            docs = []
            hits = []
            if str(extra_text).strip():
                docs.append(research.SourceDoc("ข้อความที่ผู้ใช้วาง", "paste://local",
                                               str(extra_text).strip()[:research.MAX_SOURCE_CHARS]))
            for url in list(urls or [])[:6]:
                self.progress.emit(f"กำลังเปิด {str(url)[:80]}")
                try:
                    docs.append(research.fetch_page(str(url)))
                except research.ResearchError as exc:
                    self.progress.emit(f"ข้าม {str(url)[:60]}: {exc}")
            if settings.research_search not in ("off", "") and str(query).strip():
                self.progress.emit("กำลังค้นหาเว็บ")
                try:
                    hits = research.search_web(str(query), settings.research_search,
                                               settings.research_endpoint, key,
                                               settings.research_max_sources)
                except research.ResearchError as exc:
                    self.progress.emit(f"ค้นเว็บไม่ได้: {exc}")
            if not docs and not hits:
                self.done.emit(None, [], "ไม่มีแหล่งข้อมูล: วางข้อความหรือใส่ลิงก์ก่อน "
                                         "หรือเปิดใช้การค้นเว็บในแท็บวิจัยเกม")
                return
            self.progress.emit("AI กำลังสรุปข้อมูลเกมเป็นภาษาไทย")
            prompt = research.build_brief_prompt(game, docs, hits)
            content = AIClient(settings, key).research(prompt, research.RESEARCH_SYSTEM)
            brief = research.parse_brief(content)
            source_list = [{"title": doc.title, "url": doc.url, "chars": len(doc.text)}
                           for doc in docs]
            source_list += [{"title": hit.title, "url": hit.url, "chars": 0} for hit in hits]
            self.done.emit(brief, source_list, "")
        except Exception as exc:
            from .providers import ProviderError
            message = str(exc) if isinstance(exc, (ProviderError, research.ResearchError, ValueError)) \
                else "ขั้นตอนวิจัยล้มเหลวโดยไม่คาดคิด"
            self.done.emit(None, [], message)
