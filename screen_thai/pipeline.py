from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal, Slot

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


class AIWorker(QObject):
    done = Signal(int, int, object, str)

    @Slot(int, int, object, object, object, str, object)
    def process(self, epoch, revision, blocks, image, settings, key, profile):
        try:
            result = AIClient(settings, key).translate(blocks, image, profile)
            self.done.emit(epoch, revision, result, "")
        except Exception as exc:
            from .providers import ProviderError
            message = str(exc) if isinstance(exc, (ProviderError, ValueError)) else "เกิดข้อผิดพลาดภายใน AI client"
            self.done.emit(epoch, revision, None, message)
