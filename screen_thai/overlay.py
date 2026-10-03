from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QTextLayout, QTextOption
from PySide6.QtWidgets import QWidget

from .models import Block, Result, Settings


def exclude_from_capture(widget: QWidget) -> bool:
    if sys.platform != "win32":
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    fn = user32.SetWindowDisplayAffinity
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    fn.restype = ctypes.c_bool
    # Exclusion (not black fill) needs Windows 10 2004+.
    if sys.getwindowsversion().build < 19041:
        return False
    return bool(fn(int(widget.winId()), 0x11))


class Overlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowTransparentForInput
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.items = []
        self.image_size = (1, 1)
        self.settings = Settings()
        self.excluded = False

    def configure(self, screen, region: QRectF, settings: Settings):
        geometry = screen.geometry()
        self.setGeometry(
            geometry.x() + round(region.x() * geometry.width()),
            geometry.y() + round(region.y() * geometry.height()),
            max(1, round(region.width() * geometry.width())),
            max(1, round(region.height() * geometry.height())),
        )
        self.settings = settings
        self.excluded = exclude_from_capture(self)

    def display(self, blocks: list[Block], result: Result, size):
        lookup = {b.id: b for b in blocks}
        self.items = [(lookup[t.id], t.thai, t.speaker if self.settings.show_speaker_label else "")
                      for t in result.translations if t.id in lookup]
        self.image_size = size
        self.update()

    def clear(self):
        self.items = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        sx, sy = self.width() / self.image_size[0], self.height() / self.image_size[1]
        for block, thai, speaker in self.items:
            x, y, w, h = block.box
            x, y = max(0, x*sx-4), max(0, y*sy-3)
            width = min(self.width()-x, max(w*sx+8, 70))
            if width < 8:
                continue
            # Qt text layout uses Thai shaping and word breaking; never paint text as HTML.
            font = QFont("Leelawadee UI", self.settings.font_size)
            body = f"{speaker}: {thai}" if speaker and speaker not in thai else thai
            layout = QTextLayout(body, font)
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            layout.setTextOption(option)
            layout.beginLayout()
            height = 0.0
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(max(1, width-12))
                from PySide6.QtCore import QPointF
                line.setPosition(QPointF(6, height+4))
                height += line.height()
            layout.endLayout()
            height = min(self.height(), max(h*sy+6, height+8))
            y = max(0, min(y, self.height()-height))
            rect = QRectF(x, y, width, height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(12, 18, 25, self.settings.opacity))
            painter.drawRoundedRect(rect, 5, 5)
            painter.save()
            painter.setClipRect(rect)
            painter.setPen(QColor("#f4f7fb"))
            from PySide6.QtCore import QPointF
            layout.draw(painter, QPointF(x, y))
            painter.restore()


class RegionPicker(QWidget):
    chosen = Signal(object)

    def __init__(self, screen):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.start = None
        self.end = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = event.position().toPoint()
            self.end = self.start

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start is None or event.button() != Qt.MouseButton.LeftButton:
            return
        rect = QRect(self.start, event.position().toPoint()).normalized().intersected(self.rect())
        if rect.width() >= 30 and rect.height() >= 20:
            self.chosen.emit(QRectF(rect.x()/self.width(), rect.y()/self.height(),
                                    rect.width()/self.width(), rect.height()/self.height()))
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(5, 12, 22, 135))
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Leelawadee UI", 18))
        painter.drawText(self.rect().adjusted(30, 25, -30, -25), Qt.AlignmentFlag.AlignTop,
                         "ลากกรอบพื้นที่ที่ต้องการแปล • Esc ยกเลิก")
        if self.start is not None and self.end is not None:
            painter.setPen(QPen(QColor("#64e1b5"), 2))
            painter.setBrush(QColor(100, 225, 181, 25))
            painter.drawRect(QRect(self.start, self.end).normalized())
