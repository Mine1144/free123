from __future__ import annotations

import ctypes
from dataclasses import replace
import json
import sys
import time

import mss
from PIL import Image
from PySide6.QtCore import QRectF, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QStyle, QSystemTrayIcon, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .overlay import Overlay, RegionPicker
from .pipeline import AIWorker, OCRWorker
from .providers import validate_endpoint
from .state import FrameState
from .storage import Secrets, Store
from .theme import STYLE
from .windows import Hotkeys


class Window(QMainWindow):
    ocr_requested = Signal(int, object, object)
    ai_requested = Signal(int, int, object, object, object, str, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScreenThai · แปลหน้าจอด้วย AI")
        self.resize(1050, 820)
        self.store, self.secrets = Store(), Secrets()
        self.settings = self.store.load_settings()
        self.region = QRectF(0, 0, 1, 1)
        self.overlay = Overlay()
        self.epoch, self.job_id = 0, 0
        self.active = self.single = self.ocr_busy = self.ai_busy = self.closing = False
        self.overlay_visible = True
        self.state = FrameState()
        self.latest_image = None
        self.job_blocks = []
        self.history = []
        self.last_frame = 0.0
        self.capture_pending = False
        self.picker = None
        self.current_profile = self.settings.profile
        self._build_ui()
        self._load_settings()
        self.ocr_thread = QThread(self)
        self.ocr_worker = OCRWorker()
        self.ocr_worker.moveToThread(self.ocr_thread)
        self.ocr_requested.connect(self.ocr_worker.process)
        self.ocr_worker.done.connect(self._ocr_done)
        self.ocr_thread.finished.connect(self.ocr_worker.deleteLater)
        self.ocr_thread.finished.connect(self._shutdown_ready)
        self.ocr_thread.start()
        self.ai_thread = QThread(self)
        self.ai_worker = AIWorker()
        self.ai_worker.moveToThread(self.ai_thread)
        self.ai_requested.connect(self.ai_worker.process)
        self.ai_worker.done.connect(self._ai_done)
        self.ai_thread.finished.connect(self.ai_worker.deleteLater)
        self.ai_thread.finished.connect(self._shutdown_ready)
        self.ai_thread.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture)
        self.expiry = QTimer(self)
        self.expiry.setInterval(500)
        self.expiry.timeout.connect(self._expire)
        self.expiry.start()
        self.hotkeys = Hotkeys(self.toggle, self.toggle_overlay, self.translate_once)
        QApplication.instance().installNativeEventFilter(self.hotkeys)
        if sys.platform == "win32" and len(self.hotkeys.registered) < 3:
            self.status.setText("ปุ่มลัดบางปุ่มถูกใช้อยู่ • ใช้ปุ่มในหน้าต่างแทนได้")
        self.tray = QSystemTrayIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon), self)
        self.tray.setToolTip("ScreenThai")
        menu = QMenu(self)
        for label, callback in [("เปิดหน้าต่าง", self.restore), ("เริ่ม / หยุด", self.toggle),
                                ("แสดง / ซ่อนคำแปล", self.toggle_overlay), ("ออก", self.close)]:
            action = QAction(label, self)
            action.triggered.connect(callback)
            menu.addAction(action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self.restore()
                                   if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()
        QApplication.instance().screenRemoved.connect(lambda _: self.stop())
        QApplication.instance().screenAdded.connect(lambda _: self.stop())
        for screen in QApplication.screens():
            screen.geometryChanged.connect(lambda _: self.stop())
            screen.logicalDotsPerInchChanged.connect(lambda _: self.stop())

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 22, 28, 20)
        eyebrow = QLabel("LOCAL FIRST  /  WINDOWS OVERLAY")
        eyebrow.setObjectName("eyebrow")
        layout.addWidget(eyebrow)
        heading = QHBoxLayout()
        brand = QLabel("ScreenThai")
        brand.setObjectName("brand")
        heading.addWidget(brand)
        heading.addStretch()
        tag = QLabel("ไม่ผ่าน OBS  •  ไม่ inject เข้าเกม")
        tag.setObjectName("muted")
        heading.addWidget(tag)
        layout.addLayout(heading)
        description = QLabel("เข้าใจทุกเรื่องราว บนหน้าจอของคุณ — แปลไทยตรงตำแหน่งข้อความด้วย AI")
        description.setObjectName("muted")
        layout.addWidget(description)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        control = QWidget()
        control_layout = QVBoxLayout(control)
        control_layout.setContentsMargins(18, 16, 18, 16)
        self.capture_group = QGroupBox("01  พื้นที่และข้อความที่ต้องการแปล")
        form = QFormLayout(self.capture_group)
        self.monitor = QComboBox()
        for screen in QApplication.screens():
            g = screen.geometry()
            self.monitor.addItem(f"{screen.name()} · {g.width()} × {g.height()} (logical)", screen)
        self.monitor.currentIndexChanged.connect(self.full_region)
        form.addRow("จอภาพ", self.monitor)
        row = QHBoxLayout()
        select = QPushButton("ลากเลือกพื้นที่")
        select.clicked.connect(self.select_region)
        full = QPushButton("ทั้งจอ")
        full.clicked.connect(self.full_region)
        self.region_label = QLabel("ทั้งจอ")
        row.addWidget(select)
        row.addWidget(full)
        row.addWidget(self.region_label, 1)
        form.addRow("ขอบเขต", row)
        self.scope = QComboBox()
        for text, data in [("ข้อความทั้งหมด · เมนู / UI / บทสนทนา", "all"),
                           ("เฉพาะบทสนทนาและซับไตเติล", "dialogue"),
                           ("AI เลือกข้อความสำคัญตามบริบท", "smart")]:
            self.scope.addItem(text, data)
        form.addRow("สิ่งที่จะแปล", self.scope)
        control_layout.addWidget(self.capture_group)
        self.provider_group = QGroupBox("02  เครื่องมือแปลภาษา")
        form = QFormLayout(self.provider_group)
        self.provider = QComboBox()
        self.provider.addItem("Local AI · Ollama", "ollama")
        self.provider.addItem("API · OpenAI-compatible / LM Studio", "openai")
        self.provider.currentIndexChanged.connect(self.provider_changed)
        form.addRow("ผู้ให้บริการ", self.provider)
        self.endpoint = QLineEdit()
        self.endpoint.setPlaceholderText("http://localhost:11434")
        self.endpoint.editingFinished.connect(self.load_key)
        form.addRow("Base URL", self.endpoint)
        self.model = QLineEdit()
        form.addRow("ชื่อโมเดล", self.model)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("ไม่จำเป็นสำหรับ Ollama / local server")
        form.addRow("API key", self.key)
        self.remember = QCheckBox("จำ key ใน Windows Credential Manager (ไม่เก็บในไฟล์ตั้งค่า)")
        form.addRow("", self.remember)
        self.vision = QCheckBox("ให้ AI ดูภาพหน้าจอเพื่อเข้าใจฉาก • ต้องใช้โมเดล Vision")
        form.addRow("การมองเห็น", self.vision)
        privacy = QLabel("ปิดดูภาพ = ส่งเฉพาะข้อความ OCR และบริบท • API ภายนอกอาจคิดค่าบริการ\n"
                         "Ollama ทำงานในเครื่องหลังดาวน์โหลดโมเดล • OCR ทำงานในเครื่องเสมอ")
        privacy.setObjectName("muted")
        privacy.setWordWrap(True)
        form.addRow(privacy)
        control_layout.addWidget(self.provider_group)
        control_layout.addStretch()
        self.tabs.addTab(control, "แปลหน้าจอ")
        memory = QWidget()
        ml = QVBoxLayout(memory)
        ml.setContentsMargins(20, 20, 20, 20)
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("โปรไฟล์เกม / โปรแกรม"))
        self.profile = QLineEdit()
        profile_row.addWidget(self.profile, 1)
        self.load_profile_button = QPushButton("เปิด / สร้าง")
        self.load_profile_button.clicked.connect(self.switch_profile)
        profile_row.addWidget(self.load_profile_button)
        ml.addLayout(profile_row)
        ml.addWidget(QLabel("ข้อมูลที่ยืนยันเอง • ตัวละคร ความสัมพันธ์ และสรรพนาม (สำคัญกว่าข้อสันนิษฐาน AI)"))
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("เช่น Grace เป็นลูกของ Alyssa ใช้คำว่า หนู / แม่ เมื่อคุยกัน")
        ml.addWidget(self.notes, 1)
        ml.addWidget(QLabel("คำศัพท์เฉพาะ • หนึ่งคำต่อบรรทัด"))
        self.glossary = QPlainTextEdit()
        self.glossary.setPlaceholderText("Grace = เกรซ\nRaccoon City = แร็กคูนซิตี")
        self.glossary.setMaximumHeight(100)
        ml.addWidget(self.glossary)
        row = QHBoxLayout()
        self.save_memory_button = QPushButton("บันทึกบริบท")
        self.save_memory_button.clicked.connect(self.save_memory)
        self.clear_memory_button = QPushButton("ล้างสิ่งที่ AI เรียนรู้")
        self.clear_memory_button.clicked.connect(self.clear_memory)
        row.addWidget(self.save_memory_button)
        row.addWidget(self.clear_memory_button)
        row.addStretch()
        ml.addLayout(row)
        ml.addWidget(QLabel("สิ่งที่ AI สังเกตล่าสุด / ความสัมพันธ์ที่ยังต้องตรวจสอบ"))
        self.scene = QPlainTextEdit()
        self.scene.setReadOnly(True)
        self.scene.setMaximumHeight(125)
        ml.addWidget(self.scene, 1)
        from .graph import RelationshipGraph
        self.graph = RelationshipGraph()
        ml.addWidget(self.graph, 1)
        self.tabs.addTab(memory, "บริบทและตัวละคร")
        history_page = QWidget()
        hl = QVBoxLayout(history_page)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["เวลา", "ต้นฉบับ", "คำแปลไทย"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        hl.addWidget(self.table)
        row = QHBoxLayout()
        export = QPushButton("ส่งออกประวัติ JSON")
        export.clicked.connect(self.export_history)
        clear = QPushButton("ล้างประวัติ")
        clear.clicked.connect(self.clear_history)
        row.addWidget(export)
        row.addWidget(clear)
        row.addWidget(QLabel("เก็บ 300 รายการในหน่วยความจำ • ไม่บันทึกภาพหน้าจอ"), 1)
        hl.addLayout(row)
        self.tabs.addTab(history_page, "ประวัติคำแปล")
        advanced = QWidget()
        al = QVBoxLayout(advanced)
        self.advanced_group = QGroupBox("ปรับการทำงาน")
        form = QFormLayout(self.advanced_group)
        self.interval = QSpinBox()
        self.interval.setRange(500, 10000)
        self.interval.setSingleStep(500)
        self.interval.setSuffix(" ms")
        form.addRow("ช่วงจับภาพ (ไม่ใช่ความเร็ว AI)", self.interval)
        self.timeout = QSpinBox()
        self.timeout.setRange(5, 180)
        self.timeout.setSuffix(" วินาที")
        form.addRow("รอคำตอบ AI สูงสุด", self.timeout)
        self.font = QSpinBox()
        self.font.setRange(10, 48)
        form.addRow("ขนาดตัวอักษรคำแปล (pt)", self.font)
        self.opacity = QSpinBox()
        self.opacity.setRange(80, 255)
        form.addRow("ความทึบพื้นหลังคำแปล (255 = ทึบ)", self.opacity)
        self.max_blocks = QSpinBox()
        self.max_blocks.setRange(1, 80)
        form.addRow("จำนวนกล่องข้อความสูงสุด / เฟรม", self.max_blocks)
        self.capture_mode = QComboBox()
        self.capture_mode.addItem("ปลอดภัย · ซ่อน overlay ชั่วครู่ก่อนจับภาพ", "safe")
        self.capture_mode.addItem("ทดลอง · Windows capture exclusion (ลดการกะพริบ)", "excluded")
        form.addRow("ป้องกันจับคำแปลซ้ำ", self.capture_mode)
        al.addWidget(self.advanced_group)
        help_text = QLabel(
            "เริ่มต้นใช้งาน\n"
            "1. ติดตั้ง Ollama และดาวน์โหลดโมเดล เช่น ollama pull qwen2.5vl:7b\n"
            "2. เปิดเกมแบบ Windowed / Borderless แล้วเลือกจอหรือพื้นที่\n"
            "3. กดเริ่มแปล หน้าต่างนี้จะย่อเพื่อไม่บังเกม\n\n"
            "Ctrl + Alt + T   เริ่ม / หยุด     •     Ctrl + Alt + H   ซ่อน / แสดงคำแปล\n"
            "Ctrl + Alt + S   แปลหนึ่งครั้ง     •     เปิดหน้าต่างกลับจาก system tray\n\n"
            "ข้อจำกัด: OCR รุ่นเริ่มต้นเน้นอังกฤษ/จีน ไม่รับประกันภาษาอื่นหรือฟอนต์เกม\n"
            "ไม่ได้ฟังเสียง จำใบหน้า หรืออ่านปาก • ความสัมพันธ์จาก AI อาจผิด\n"
            "Exclusive fullscreen / DRM / anti-cheat บางระบบไม่รองรับ overlay\n"
            "โหมดจับภาพทดลองอาจเป็นสีดำหรือจับ overlay ซ้ำ ให้กลับไปใช้โหมดปลอดภัย\n"
            "คำแปลแทนข้อความด้วยแผ่นพื้นหลัง ไม่ใช่การลบตัวอักษรด้วย inpainting"
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("muted")
        al.addWidget(help_text)
        al.addStretch()
        self.tabs.addTab(advanced, "ตั้งค่าและวิธีใช้")
        self.status = QLabel("พร้อมเริ่ม • เลือก AI แล้วกดเริ่มแปลหน้าจอ")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.start_button = QPushButton("▶  เริ่มแปลหน้าจอ")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.toggle)
        self.once_button = QPushButton("แปลหนึ่งครั้ง")
        self.once_button.clicked.connect(self.translate_once)
        hide_button = QPushButton("ซ่อน / แสดงคำแปล")
        hide_button.clicked.connect(self.toggle_overlay)
        row.addWidget(self.start_button, 2)
        row.addWidget(self.once_button, 1)
        row.addWidget(hide_button, 1)
        layout.addLayout(row)

    def _load_settings(self):
        s = self.settings
        self.provider.blockSignals(True)
        self.provider.setCurrentIndex(self.provider.findData(s.provider))
        self.provider.blockSignals(False)
        self.endpoint.setText(s.endpoint)
        self.model.setText(s.model)
        self.vision.setChecked(s.vision)
        self.scope.setCurrentIndex(self.scope.findData(s.scope))
        self.interval.setValue(s.interval_ms)
        self.timeout.setValue(s.timeout_s)
        self.font.setValue(s.font_size)
        self.opacity.setValue(s.opacity)
        self.max_blocks.setValue(s.max_blocks)
        self.capture_mode.setCurrentIndex(self.capture_mode.findData(s.capture_mode))
        self.profile.setText(s.profile)
        self.load_key()
        self.load_memory()

    def provider_changed(self):
        local = self.provider.currentData() == "ollama"
        self.endpoint.setText("http://localhost:11434" if local else "https://api.openai.com/v1")
        self.model.setText("qwen2.5vl:7b" if local else "")
        self.model.setPlaceholderText("ชื่อโมเดล Vision ตามผู้ให้บริการ")
        self.load_key()

    def load_key(self):
        self.key.clear()
        try:
            value = self.secrets.get(self.endpoint.text().strip())
            self.key.setText(value)
            self.remember.setChecked(bool(value))
        except Exception:
            self.remember.setChecked(False)

    def load_memory(self):
        profile = self.store.load_profile(self.current_profile)
        self.graph.update_relations(profile["relations"])
        self.notes.setPlainText(profile["notes"])
        self.glossary.setPlainText(profile["glossary"])
        self.scene.setPlainText(profile["summary"] + "\n\n" + json.dumps(
            profile["relations"], ensure_ascii=False, indent=2))

    def save_memory(self):
        try:
            profile = self.store.load_profile(self.current_profile)
            profile.update(notes=self.notes.toPlainText()[:6000], glossary=self.glossary.toPlainText()[:4000])
            self.store.save_profile(self.current_profile, profile)
            return True
        except OSError:
            QMessageBox.warning(self, "บันทึกไม่ได้", "ตรวจสิทธิ์เขียนโฟลเดอร์ข้อมูล ScreenThai")
            return False

    def switch_profile(self):
        if self.active:
            return
        name = self.profile.text().strip()[:100] or "ทั่วไป"
        if self.save_memory():
            self.current_profile = name
            self.profile.setText(name)
            self.load_memory()

    def clear_memory(self):
        if self.active:
            return
        if QMessageBox.question(self, "ล้างความจำ", "ล้างสรุปเรื่องและความสัมพันธ์ AI? ข้อมูลที่กรอกเองจะคงอยู่") != QMessageBox.StandardButton.Yes:
            return
        try:
            profile = self.store.load_profile(self.current_profile)
            profile.update(summary="", relations=[])
            self.store.save_profile(self.current_profile, profile)
            self.scene.clear()
            self.graph.update_relations([])
        except OSError:
            QMessageBox.warning(self, "ล้างไม่ได้", "ตรวจสิทธิ์เขียนโฟลเดอร์ข้อมูล")

    def full_region(self, *_):
        self.region = QRectF(0, 0, 1, 1)
        self.region_label.setText("ทั้งจอ")

    def select_region(self):
        if self.active or self.picker is not None:
            return
        self.picker = RegionPicker(self.monitor.currentData())
        self.picker.chosen.connect(self._region_chosen)
        self.picker.destroyed.connect(self._picker_closed)
        self.showMinimized()
        self.picker.show()
        self.picker.activateWindow()

    def _region_chosen(self, region):
        self.region = region
        self.region_label.setText(f"พื้นที่ {region.width():.0%} × {region.height():.0%}")

    def _picker_closed(self):
        self.picker = None
        self.restore()

    def read_settings(self):
        endpoint = validate_endpoint(self.endpoint.text())
        if not self.model.text().strip():
            raise ValueError("กรุณาระบุชื่อโมเดล")
        if self.profile.text().strip() != self.current_profile:
            raise ValueError("กด เปิด / สร้าง ก่อนใช้งานโปรไฟล์ที่เปลี่ยนชื่อ")
        return replace(self.settings, provider=self.provider.currentData(), endpoint=endpoint,
                       model=self.model.text().strip(), vision=self.vision.isChecked(),
                       scope=self.scope.currentData(), interval_ms=self.interval.value(),
                       timeout_s=self.timeout.value(), font_size=self.font.value(),
                       opacity=self.opacity.value(), max_blocks=self.max_blocks.value(),
                       capture_mode=self.capture_mode.currentData(), profile=self.current_profile)

    def toggle(self):
        if self.closing:
            return
        if self.active:
            self.stop()
        else:
            self.start(False)

    def translate_once(self):
        if not self.active and not self.closing:
            self.start(True)

    def _lock(self, locked):
        for widget in (self.capture_group, self.provider_group, self.advanced_group, self.profile,
                       self.notes, self.glossary, self.load_profile_button, self.save_memory_button,
                       self.clear_memory_button, self.once_button):
            widget.setEnabled(not locked)

    def start(self, single):
        if self.ocr_busy or self.ai_busy or self.capture_pending:
            self.status.setText("รอคำขอก่อนหน้าจบก่อนเริ่มใหม่ • ไม่มีการสร้างคิวคำขอซ้ำ")
            return
        try:
            settings = self.read_settings()
            from urllib.parse import urlsplit
            host = urlsplit(settings.endpoint).hostname
            if host not in ("localhost", "127.0.0.1", "::1"):
                kind = "ภาพหน้าจอ + ข้อความ OCR" if settings.vision else "ข้อความ OCR (ไม่ส่งภาพ)"
                answer = QMessageBox.question(self, "อนุญาตส่งข้อมูลออกจากเครื่อง?",
                    f"จะส่ง {kind} และบริบทตัวละครไปที่\n{settings.endpoint}\n\n"
                    "อาจมีข้อมูลส่วนตัวบนจอ และอาจมีค่าบริการต่อคำขอ\nอนุญาตสำหรับการแปลรอบนี้หรือไม่?")
                if answer != QMessageBox.StandardButton.Yes:
                    return
            if not self.save_memory():
                return
            self.store.save_settings(settings)
            if self.remember.isChecked():
                try:
                    self.secrets.set(settings.endpoint, self.key.text().strip())
                except Exception:
                    QMessageBox.warning(self, "จำ key ไม่สำเร็จ", "Credential Manager ไม่พร้อมใช้งาน จะใช้ key ในหน่วยความจำรอบนี้เท่านั้น")
            else:
                try:
                    self.secrets.set(settings.endpoint, "")
                except Exception:
                    QMessageBox.warning(self, "ลบ key ที่จำไว้ไม่สำเร็จ",
                                        "หากเคยจำ key ไว้ กรุณาลบรายการ ScreenThai ใน Windows Credential Manager เอง")
            self.settings = settings
            self.session_key = self.key.text().strip()
            self._map_monitor()
            self.overlay.configure(self.monitor.currentData(), self.region, settings)
        except (ValueError, OSError, RuntimeError) as exc:
            QMessageBox.warning(self, "ยังเริ่มไม่ได้", str(exc))
            return
        self.epoch += 1
        self.active, self.single = True, single
        self.state = FrameState()
        self.latest_image = None
        self.overlay.clear()
        self.overlay_visible = True
        self.overlay.show()
        # Re-apply after native window creation.
        from .overlay import exclude_from_capture
        self.overlay.excluded = exclude_from_capture(self.overlay)
        if settings.capture_mode == "excluded" and not self.overlay.excluded:
            self.settings = replace(settings, capture_mode="safe")
        self._lock(True)
        self.start_button.setText("■  หยุดแปลหน้าจอ")
        self.status.setText("กำลังอ่านหน้าจอ • ครั้งแรกอาจใช้เวลาโหลด OCR / โมเดล AI")
        self.showMinimized()
        if not single:
            self.timer.start(settings.interval_ms)
        QTimer.singleShot(350, self.capture)

    def _map_monitor(self):
        screen = self.monitor.currentData()
        if screen is None or screen not in QApplication.screens():
            raise ValueError("จอภาพเปลี่ยนไป กรุณาปิดและเปิดโปรแกรมใหม่")
        # MSS Windows monitor order may differ from Qt. Match Windows display device names.
        from .capture import monitor_for_screen, region_bounds
        with mss.mss() as capture:
            monitor = monitor_for_screen(screen, capture.monitors[1:])
        self.capture_bounds = region_bounds(monitor, self.region)

    def stop(self, keep_overlay=False):
        self.active = False
        self.epoch += 1
        self.timer.stop()
        self.latest_image = None
        self.session_key = ""
        if not keep_overlay:
            self.overlay.clear()
            self.overlay.hide()
        self._lock(False)
        self.start_button.setText("▶  เริ่มแปลหน้าจอ")
        self.status.setText("หยุดแล้ว • คำขอที่ส่งไปแล้วอาจยังทำงานจนจบหรือ timeout")

    def capture(self):
        if not self.active or self.ocr_busy or self.capture_pending:
            return
        self.capture_pending = True
        epoch = self.epoch
        if self.settings.capture_mode == "safe":
            self.overlay.hide()
            QTimer.singleShot(60, lambda: self._grab(epoch))
        else:
            self._grab(epoch)

    def _grab(self, epoch):
        self.capture_pending = False
        if not self.active or epoch != self.epoch:
            return
        try:
            if sys.platform == "win32" and self.settings.capture_mode == "safe":
                ctypes.windll.dwmapi.DwmFlush()
            with mss.mss() as capture:
                shot = capture.grab(self.capture_bounds)
            image = Image.frombytes("RGB", shot.size, shot.rgb)
            self.ocr_busy = True
            self.ocr_requested.emit(epoch, image, replace(self.settings))
        except Exception:
            self.stop()
            self.status.setText("จับภาพไม่ได้ • ตรวจการเชื่อมต่อจอ / โหมดเกม / การป้องกัน DRM")
        finally:
            if self.active and self.overlay_visible:
                self.overlay.show()

    def _ocr_done(self, epoch, image, blocks, error):
        self.ocr_busy = False
        if self.closing or not self.active or epoch != self.epoch:
            return
        if error:
            self.stop()
            self.status.setText(error)
            return
        self.last_frame = time.monotonic()
        self.latest_image = image
        self.state.update(blocks)
        self._render()
        if not blocks:
            self.status.setText("ไม่พบข้อความที่ OCR อ่านได้ • คำแปลเดิมถูกล้างแล้ว")
            if self.single:
                self.stop()
            return
        self._dispatch()

    def _render(self):
        if self.latest_image is not None:
            self.overlay.display(self.state.blocks, self.state.rendered(), self.latest_image.size)

    def _dispatch(self):
        if (not self.active or self.ai_busy or not self.state.needs_translation()
                or self.latest_image is None):
            return
        self.ai_busy = True
        self.job_id += 1
        self.job_tokens = self.state.snapshot()
        self.job_blocks = list(self.state.blocks)
        self.job_started = time.monotonic()
        self.status.setText(f"AI กำลังพิจารณา {len(self.job_blocks)} กล่องข้อความ • ไม่มีคิวภาพสะสม")
        self.ai_requested.emit(self.epoch, self.job_id, self.job_blocks, self.latest_image,
                               replace(self.settings), self.session_key,
                               self.store.load_profile(self.current_profile))

    def _ai_done(self, epoch, job_id, result, error):
        self.ai_busy = False
        if self.closing or not self.active or epoch != self.epoch or job_id != self.job_id:
            return
        if error:
            # Pause on provider errors rather than repeatedly spending quota.
            self.stop()
            self.status.setText(error + " • หยุดอัตโนมัติแล้ว")
            self.restore()
            return
        if not self.state.accept(self.job_tokens, result):
            self.status.setText("ข้อความเปลี่ยนแล้ว • ข้ามผลแปลเก่าและพิจารณาเฟรมล่าสุด")
            self._dispatch()
            return
        self._render()
        elapsed = time.monotonic() - self.job_started
        self.status.setText(f"แปลแล้ว {len(result.translations)} ข้อความ • AI {elapsed:.1f} วินาที • รอข้อความใหม่")
        if result.scene.summary or result.scene.relations:
            try:
                self.store.learn(self.current_profile, result)
            except OSError:
                self.status.setText("แปลสำเร็จ แต่บันทึกบริบทลงดิสก์ไม่ได้")
        self.scene.setPlainText(
            f"People: {', '.join(result.scene.people)}\nPlace: {result.scene.place}\n"
            f"Action: {result.scene.action}\n\n{result.scene.summary}\n\n"
            "ความสัมพันธ์ที่ AI เสนอ (ไม่ใช่ข้อเท็จจริงที่ยืนยัน):\n" + json.dumps(
                self.store.load_profile(self.current_profile)["relations"], ensure_ascii=False, indent=2))
        self.graph.update_relations(self.store.load_profile(self.current_profile)["relations"])
        source = {b.id: b.text for b in self.job_blocks}
        for translation in result.translations:
            record = {"time": time.strftime("%H:%M:%S"), "source": source.get(translation.id, ""),
                      "thai": translation.thai}
            self.history.append(record)
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, name in enumerate(("time", "source", "thai")):
                self.table.setItem(row, col, QTableWidgetItem(record[name]))
        while len(self.history) > 300:
            self.history.pop(0)
            self.table.removeRow(0)
        self.table.scrollToBottom()
        if self.single:
            self.stop(keep_overlay=True)
            self.status.setText(f"แปลหนึ่งครั้งเสร็จ • {elapsed:.1f} วินาที • ซ่อนคำแปลด้วย Ctrl+Alt+H")
        else:
            self._dispatch()

    def _expire(self):
        if self.active and self.last_frame and time.monotonic() - self.last_frame > max(4, self.settings.interval_ms/500):
            self.overlay.clear()

    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        if self.overlay_visible and not self.capture_pending:
            self.overlay.show()
        else:
            self.overlay.hide()

    def restore(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def clear_history(self):
        self.history.clear()
        self.table.setRowCount(0)

    def export_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "ส่งออกประวัติ", "screen-thai-history.json", "JSON (*.json)")
        if path:
            try:
                from pathlib import Path
                Path(path).write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                QMessageBox.warning(self, "ส่งออกไม่ได้", "ตรวจสิทธิ์เขียนไฟล์ปลายทาง")

    def closeEvent(self, event: QCloseEvent):
        if self.closing and not self.ai_thread.isRunning() and not self.ocr_thread.isRunning():
            event.accept()
            return
        event.ignore()
        if not self.closing:
            self.closing = True
            self.stop()
            self.expiry.stop()
            self.hotkeys.close()
            self.tray.hide()
            self.overlay.close()
            self.centralWidget().setEnabled(False)
            self.status.setText("กำลังปิด • รอ OCR / AI ที่ส่งไปแล้วจบหรือ timeout")
            self.ocr_thread.quit()
            self.ai_thread.quit()
            QTimer.singleShot(0, self._shutdown_ready)

    def _shutdown_ready(self):
        if self.closing and not self.ai_thread.isRunning() and not self.ocr_thread.isRunning():
            self.close()
            QApplication.instance().quit()


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            pass
    app = QApplication(sys.argv)
    app.setApplicationName("ScreenThai")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    from .overlay import exclude_from_capture
    exclude_from_capture(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
