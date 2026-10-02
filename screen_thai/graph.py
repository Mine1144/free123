"""Relationship graph of textual hypotheses, not face or speaker recognition."""
import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush, QFont, QPen
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView


class RelationshipGraph(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph = QGraphicsScene(self)
        self.setScene(self.graph)
        self.setMinimumHeight(160)
        self.setBackgroundBrush(QColor("#111927"))
        self.setStyleSheet("border: 1px solid #253143; border-radius: 6px;")
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        from PySide6.QtGui import QPainter
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

    def update_relations(self, relations):
        self.graph.clear()
        valid = [r for r in relations if isinstance(r, dict) and r.get("from") and r.get("to")]
        names = list(dict.fromkeys(str(r[key]) for r in valid for key in ("from", "to")))[:12]
        if not names:
            label = self.graph.addText("กราฟความสัมพันธ์จะปรากฏเมื่อมีหลักฐานจากบทสนทนา")
            label.setDefaultTextColor(QColor("#93a1b5"))
            self._fit()
            return
        points = {name: (240*math.cos(i*2*math.pi/len(names)),
                         170*math.sin(i*2*math.pi/len(names))) for i, name in enumerate(names)}
        for rel in valid:
            if str(rel["from"]) not in points or str(rel["to"]) not in points:
                continue
            x, y = points[str(rel["from"])]
            xx, yy = points[str(rel["to"])]
            line = self.graph.addLine(x, y, xx, yy, QPen(QColor("#467765"), 2, Qt.PenStyle.DashLine))
            tooltip = f'{rel["from"]} → {rel["to"]}: {rel.get("relation", "")}\nหลักฐาน: {rel.get("evidence", "")}\nข้อสันนิษฐาน AI ไม่ใช่ข้อมูลยืนยัน'
            line.setToolTip(tooltip)
            label = self.graph.addText(str(rel.get("relation", ""))[:30], QFont("Leelawadee UI", 10))
            label.setDefaultTextColor(QColor("#a4b9ad"))
            label.setPos((x+xx)/2, (y+yy)/2)
            label.setToolTip(tooltip)
        for name, (x, y) in points.items():
            node = self.graph.addEllipse(x-58, y-25, 116, 50, QPen(QColor("#75e3b9")), QBrush(QColor("#19382f")))
            node.setToolTip(name + "\nชื่อตามข้อความ ไม่ได้ระบุตัวจากใบหน้า")
            label = self.graph.addText(name[:18], QFont("Leelawadee UI", 11))
            label.setDefaultTextColor(QColor("#e3fff4"))
            label.setPos(x-label.boundingRect().width()/2, y-label.boundingRect().height()/2)
        self._fit()

    def _fit(self):
        self.graph.setSceneRect(self.graph.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        self.fitInView(self.graph.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()
