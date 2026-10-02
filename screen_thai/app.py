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

from . import context as context_module
from . import layout, speech
from .dialogue import DialogueMemory
from .faces import FaceMemory
from .overlay import Overlay, RegionPicker
from .pages import CharactersPage, ResearchPage, ask_name, warning
from .pipeline import AIWorker, OCRWorker, ResearchWorker, VisionWorker
from .providers import validate_endpoint
from .state import FrameState
from .storage import Secrets, Store
from .theme import STYLE
from .windows import Hotkeys

HISTORY_COLUMNS = ("เวลา", "ผู้พูด", "ต้นฉบับ", "คำแปลไทย", "โทน/ข้อสังเกต")


class Window(QMainWindow):
    ocr_requested = Signal(int, object, object)
    ai_requested = Signal(int, int, object, object, object, str, object, object)
    vision_requested = Signal(int, int, object, object, object)
    research_requested = Signal(str, object, str, str, str, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScreenThai · แปลหน้าจอด้วย AI")
        self.resize(1150, 880)
        self.store, self.secrets = Store(), Secrets()
        self.settings = self.store.load_settings()
        self.region = QRectF(0, 0, 1, 1)
        self.overlay = Overlay()
        self.epoch, self.job_id = 0, 0
        self.active = self.single = self.ocr_busy = self.ai_busy = self.closing = False
        self.vision_busy = self.research_busy = False
        self.overlay_visible = True
        self.state = FrameState()
        self.session_key = ""
        self.job_hints = []
        self.job_tokens = {}
        self.dialogue = DialogueMemory()
        self.latest_image = None
        self.job_blocks = []
        self.hints = []
        self.faces = []
        self.last_scan_image = None
        self.last_scan_faces = []
        self.last_vision_at = 0.0
        self.research_started_for = ""
        self.history = []
        self.last_frame = 0.0
        self.capture_pending = False
        self.picker = None
        self.current_profile = self.settings.profile
        self.memory = self.store.load_memory(self.current_profile)
        self.profile_data = self.store.load_profile(self.current_profile)
        self.brief = self.profile_data.get("research", {})
        self._build_ui()
        self._load_settings()
        self._start_workers()
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

    # --- workers -----------------------------------------------------------------------

    def _start_workers(self):
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
        self.vision_thread = QThread(self)
        self.vision_worker = VisionWorker()
        self.vision_worker.moveToThread(self.vision_thread)
        self.vision_requested.connect(self.vision_worker.process)
        self.vision_worker.done.connect(self._vision_done)
        self.vision_thread.finished.connect(self.vision_worker.shutdown)
        self.vision_thread.finished.connect(self.vision_worker.deleteLater)
        self.vision_thread.finished.connect(self._shutdown_ready)
        self.vision_thread.start()
        self.research_thread = QThread(self)
        self.research_worker = ResearchWorker()
        self.research_worker.moveToThread(self.research_thread)
        self.research_requested.connect(self.research_worker.process)
        self.research_worker.progress.connect(self._research_progress)
        self.research_worker.done.connect(self._research_done)
        self.research_thread.finished.connect(self.research_worker.deleteLater)
        self.research_thread.finished.connect(self._shutdown_ready)
        self.research_thread.start()

    # --- UI ----------------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout_ = QVBoxLayout(root)
        layout_.setContentsMargins(28, 22, 28, 20)
        eyebrow = QLabel("LOCAL FIRST  /  WINDOWS OVERLAY")
        eyebrow.setObjectName("eyebrow")
        layout_.addWidget(eyebrow)
        heading = QHBoxLayout()
        brand = QLabel("ScreenThai")
        brand.setObjectName("brand")
        heading.addWidget(brand)
        heading.addStretch()
        tag = QLabel("ไม่ผ่าน OBS  •  ไม่ inject เข้าเกม")
        tag.setObjectName("muted")
        heading.addWidget(tag)
        layout_.addLayout(heading)
        description = QLabel("เข้าใจทุกเรื่องราว บนหน้าจอของคุณ — แปลไทยตรงตำแหน่งข้อความด้วย AI")
        description.setObjectName("muted")
        layout_.addWidget(description)
        self.tabs = QTabWidget()
        layout_.addWidget(self.tabs, 1)
        self._build_translate_tab()
        self._build_memory_tab()
        self.research_page = ResearchPage(self)
        self.tabs.addTab(self.research_page, "วิจัยเกมก่อนแปล")
        self.characters_page = CharactersPage(self)
        self.tabs.addTab(self.characters_page, "ตัวละคร · ใบหน้า · วิธีพูด")
        self._build_history_tab()
        self._build_advanced_tab()
        self.status = QLabel("พร้อมเริ่ม • เลือก AI แล้วกดเริ่มแปลหน้าจอ")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout_.addWidget(self.status)
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
        layout_.addLayout(row)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture)
        self.expiry = QTimer(self)
        self.expiry.setInterval(500)
        self.expiry.timeout.connect(self._expire)
        self.expiry.start()

    def _build_translate_tab(self):
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
                         "Ollama ทำงานในเครื่องหลังดาวน์โหลดโมเดล • OCR และการอ่านสีหน้าทำในเครื่องเสมอ")
        privacy.setObjectName("muted")
        privacy.setWordWrap(True)
        form.addRow(privacy)
        control_layout.addWidget(self.provider_group)
        control_layout.addStretch()
        self.tabs.addTab(control, "แปลหน้าจอ")

    def _build_memory_tab(self):
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

    def _build_history_tab(self):
        history_page = QWidget()
        hl = QVBoxLayout(history_page)
        self.table = QTableWidget(0, len(HISTORY_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(HISTORY_COLUMNS))
        for column in (2, 3, 4):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        hl.addWidget(self.table)
        row = QHBoxLayout()
        export = QPushButton("ส่งออกประวัติ JSON")
        export.clicked.connect(self.export_history)
        clear = QPushButton("ล้างประวัติ")
        clear.clicked.connect(self.clear_history)
        row.addWidget(export)
        row.addWidget(clear)
        row.addWidget(QLabel("เก็บ 300 รายการในหน่วยความจำ • ประวัติบทสนทนาไม่ถูกบันทึกอัตโนมัติ • ไม่บันทึกภาพหน้าจอ"), 1)
        hl.addLayout(row)
        self.tabs.addTab(history_page, "ประวัติคำแปล")

    def _build_advanced_tab(self):
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

        self.face_group = QGroupBox("อ่านสีหน้า · จำใบหน้า · วางแผนการพูด (ทำงานในเครื่อง)")
        face_form = QFormLayout(self.face_group)
        self.face_analysis = QCheckBox("วิเคราะห์ใบหน้าและสีหน้าจากเฟรมที่จับได้")
        self.expression_cues = QCheckBox("วัดค่าสีหน้า (ปาก/ตา/คิ้ว/เอียงศีรษะ) เป็นหลักฐานประกอบ")
        self.layout_hints = QCheckBox("อ่านผังหน้าจอ: ป้ายชื่อ บทสนทนา เมนูตัวเลือก HUD")
        self.speech_plan = QCheckBox("วางแผนสรรพนาม/คำลงท้ายตามความสัมพันธ์และคำสุภาพในต้นฉบับ")
        self.show_speaker_label = QCheckBox("แสดงชื่อผู้พูดนำหน้าคำแปลบน overlay")
        for widget in (self.face_analysis, self.expression_cues, self.layout_hints,
                       self.speech_plan, self.show_speaker_label):
            face_form.addRow("", widget)
        self.face_backend = QComboBox()
        for label, data in (("อัตโนมัติ · MediaPipe ถ้ามี ไม่งั้น OpenCV", "auto"),
                            ("MediaPipe FaceMesh (ต้องติดตั้ง mediapipe)", "mediapipe"),
                            ("OpenCV เท่านั้น (ใช้ได้ทันที)", "opencv"),
                            ("ปิด", "off")):
            self.face_backend.addItem(label, data)
        face_form.addRow("ตัวตรวจจับใบหน้า", self.face_backend)
        model_row = QHBoxLayout()
        self.face_model = QLineEdit()
        self.face_model.setPlaceholderText("ไม่บังคับ: ไฟล์โมเดล YuNet .onnx สำหรับ OpenCV")
        pick_model = QPushButton("เลือกไฟล์")
        pick_model.clicked.connect(self.pick_face_model)
        model_row.addWidget(self.face_model, 1)
        model_row.addWidget(pick_model)
        face_form.addRow("โมเดลใบหน้า (ไม่บังคับ)", model_row)
        self.face_sample = QSpinBox()
        self.face_sample.setRange(400, 10000)
        self.face_sample.setSingleStep(200)
        self.face_sample.setSuffix(" ms")
        face_form.addRow("วิเคราะห์ใบหน้าทุก ๆ", self.face_sample)
        self.max_faces = QSpinBox()
        self.max_faces.setRange(1, 8)
        face_form.addRow("จำนวนใบหน้าสูงสุดต่อเฟรม", self.max_faces)
        face_note = QLabel("ค่าเหล่านี้เป็นการวัดจากภาพนิ่ง ไม่ใช่การตรวจจับอารมณ์ที่แม่นยำ • "
                           "ชื่อจะถูกผูกกับใบหน้าก็ต่อเมื่อคุณยืนยันในแท็บตัวละคร • "
                           "ไม่มีการบันทึกภาพใบหน้าลงดิสก์ เก็บเฉพาะค่าตัวเลข")
        face_note.setObjectName("muted")
        face_note.setWordWrap(True)
        face_form.addRow(face_note)
        al.addWidget(self.face_group)
        help_text = QLabel(
            "ลำดับการทำงาน\n"
            "1. แท็บ ‘วิจัยเกมก่อนแปล’ — ใส่ชื่อเกม/ลิงก์/ข้อความ หรือเปิดค้นเว็บ แล้วกด เริ่มวิจัยเกม\n"
            "2. แท็บ ‘ตัวละคร · ใบหน้า · วิธีพูด’ — สแกนใบหน้า ผูกชื่อ และล็อกสรรพนามของแต่ละตัว\n"
            "3. เปิดเกมแบบ Windowed / Borderless เลือกพื้นที่ แล้วกดเริ่มแปล\n\n"
            "Ctrl + Alt + T เริ่ม/หยุด • Ctrl + Alt + H ซ่อน/แสดงคำแปล • Ctrl + Alt + S แปลหนึ่งครั้ง\n\n"
            "ข้อจำกัด: OCR รุ่นเริ่มต้นเน้นอังกฤษ/จีน • สีหน้าเป็นค่าที่วัดได้จากการเคลื่อนไหวปาก/ตา/คิ้ว "
            "ไม่ใช่การอ่านใจ • ใบหน้าที่คล้ายกันอาจสับสนได้ ต้องยืนยันเอง • "
            "ไม่ได้ฟังเสียงหรืออ่านปาก • ความสัมพันธ์และตัวตนจาก AI อาจผิด"
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("muted")
        al.addWidget(help_text)
        al.addStretch()
        self.tabs.addTab(advanced, "ตั้งค่าและวิธีใช้")

    # --- settings ----------------------------------------------------------------------

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
        self.face_analysis.setChecked(s.face_analysis)
        self.expression_cues.setChecked(s.expression_cues)
        self.layout_hints.setChecked(s.layout_hints)
        self.speech_plan.setChecked(s.speech_plan)
        self.show_speaker_label.setChecked(s.show_speaker_label)
        self.face_backend.setCurrentIndex(max(0, self.face_backend.findData(s.face_backend)))
        self.face_model.setText(s.face_model)
        self.face_sample.setValue(s.face_sample_ms)
        self.max_faces.setValue(s.max_faces)
        self.profile.setText(s.profile)
        self.load_key()
        self.load_memory()
        self.refresh_pages()

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
        self.profile_data = self.store.load_profile(self.current_profile)
        self.memory = FaceMemory({"characters": self.profile_data.get("characters", {})})
        self.brief = self.profile_data.get("research", {})
        self.dialogue.clear()
        profile = self.profile_data
        self.graph.update_relations(profile["relations"])
        self.notes.setPlainText(profile["notes"])
        self.glossary.setPlainText(profile["glossary"])
        self.scene.setPlainText(profile["summary"] + "\n\n" + json.dumps(
            profile["relations"], ensure_ascii=False, indent=2))

    def save_memory(self) -> bool:
        try:
            profile = self.store.load_profile(self.current_profile)
            profile.update(notes=self.notes.toPlainText()[:6000],
                           glossary=self.glossary.toPlainText()[:4000])
            self.store.save_profile(self.current_profile, profile)
            self.store.save_memory(self.current_profile, self.memory)
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
            self.refresh_pages()

    def clear_memory(self):
        if self.active:
            return
        if QMessageBox.question(self, "ล้างความจำ", "ล้างสรุปเรื่องและความสัมพันธ์ AI? "
                               "ข้อมูลที่กรอกเองจะคงอยู่") != QMessageBox.StandardButton.Yes:
            return
        try:
            profile = self.store.load_profile(self.current_profile)
            profile.update(summary="", relations=[])
            self.store.save_profile(self.current_profile, profile)
            self.scene.clear()
            self.graph.update_relations([])
        except OSError:
            QMessageBox.warning(self, "ล้างไม่ได้", "ตรวจสิทธิ์เขียนโฟลเดอร์ข้อมูล")

    def read_settings(self):
        endpoint = validate_endpoint(self.endpoint.text())
        if not self.model.text().strip():
            raise ValueError("กรุณาระบุชื่อโมเดล")
        if self.profile.text().strip() != self.current_profile:
            raise ValueError("กด เปิด / สร้าง ก่อนใช้งานโปรไฟล์ที่เปลี่ยนชื่อ")
        research = self.research_page.values()
        return replace(self.settings, provider=self.provider.currentData(), endpoint=endpoint,
                       model=self.model.text().strip(), vision=self.vision.isChecked(),
                       scope=self.scope.currentData(), interval_ms=self.interval.value(),
                       timeout_s=self.timeout.value(), font_size=self.font.value(),
                       opacity=self.opacity.value(), max_blocks=self.max_blocks.value(),
                       capture_mode=self.capture_mode.currentData(), profile=self.current_profile,
                       face_analysis=self.face_analysis.isChecked(),
                       face_backend=self.face_backend.currentData(),
                       face_model=self.face_model.text().strip(),
                       face_sample_ms=self.face_sample.value(), max_faces=self.max_faces.value(),
                       expression_cues=self.expression_cues.isChecked(),
                       layout_hints=self.layout_hints.isChecked(),
                       speech_plan=self.speech_plan.isChecked(),
                       show_speaker_label=self.show_speaker_label.isChecked(),
                       research_search=research["provider"],
                       research_endpoint=research["endpoint"],
                       research_max_sources=research["max_sources"],
                       research_title=research["title"], research_aliases=research["aliases"],
                       research_auto=research["auto"])

    # --- pages -------------------------------------------------------------------------

    def refresh_pages(self):
        self.research_page.load(self.settings, self.brief,
                                self.profile_data.get("research_sources", []),
                                str(self.brief.get("user_notes", "")),
                                self.settings.research_title or self.current_profile)
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)

    def face_backend_note(self) -> str:
        from .vision import build_detector
        detector = build_detector(self.settings.face_backend, self.settings.face_model,
                                  self.settings.max_faces)
        if not detector.available:
            return f"ยังใช้งานไม่ได้ ({getattr(detector, 'reason', 'ไม่พบ backend')}) — " \
                   "ติดตั้ง mediapipe หรือใช้ OpenCV"
        return f"backend: {detector.name} (ทำงานในเครื่องเท่านั้น)"

    # --- research ----------------------------------------------------------------------

    def research_start(self):
        if self.research_busy:
            self.status.setText("กำลังวิจัยเกมอยู่ • รอให้จบก่อน")
            return
        try:
            settings = self.read_settings()
        except ValueError as exc:
            QMessageBox.warning(self, "ยังเริ่มวิจัยไม่ได้", str(exc))
            return
        values = self.research_page.values()
        if not values["title"]:
            QMessageBox.warning(self, "ต้องมีชื่อเกม", "กรอกชื่อเกมก่อน เพื่อให้ค้นหาได้ตรงเรื่อง")
            return
        if not values["urls"] and not values["paste"].strip() and values["provider"] == "off":
            QMessageBox.warning(self, "ยังไม่มีแหล่งข้อมูล",
                                "วางลิงก์/ข้อความ หรือเลือกผู้ให้บริการค้นเว็บในแท็บวิจัยก่อน")
            return
        host = ""
        if values["provider"] != "off":
            from urllib.parse import urlsplit
            endpoint = values["endpoint"] or ""
            host = urlsplit(endpoint).hostname or ""
            if host not in ("localhost", "127.0.0.1", "::1"):
                answer = QMessageBox.question(
                    self, "อนุญาตค้นเว็บ?",
                    f"จะส่งคำค้น (ชื่อเกม) ไปที่\n{values['endpoint']}\n\n"
                    "ไม่มีการส่งภาพหน้าจอหรือข้อความในเกมในการค้นหา • อาจมีค่าบริการตามบัญชีของคุณ\n"
                    "อนุญาตหรือไม่?")
                if answer != QMessageBox.StandardButton.Yes:
                    return
        self.store.save_settings(settings)
        if values["provider"] != "off" and values["key"] and values["remember_key"]:
            try:
                self.secrets.set("research:" + values["endpoint"], values["key"])
            except Exception:
                QMessageBox.warning(self, "จำ key ไม่สำเร็จ", "จะใช้ key ในรอบนี้เท่านั้น")
        self.settings = settings
        self.research_busy = True
        self.research_page.run_button.setEnabled(False)
        self.research_page.progress.setText("เริ่มวิจัย…")
        query = values["title"] + (" " + values["aliases"] if values["aliases"] else "")
        self.research_requested.emit(values["title"], replace(settings), values["key"], query,
                                     values["paste"], values["urls"])

    def _research_progress(self, message: str):
        self.research_page.progress.setText(message)
        self.status.setText("วิจัยเกม: " + message)

    def _research_done(self, brief, sources, error):
        self.research_busy = False
        self.research_page.run_button.setEnabled(True)
        if error:
            self.research_page.progress.setText("วิจัยไม่สำเร็จ: " + error)
            self.status.setText("วิจัยเกมไม่สำเร็จ: " + error)
            return
        from .research import merge_brief
        self.brief = merge_brief(self.brief, brief or {})
        try:
            self.store.save_research(self.current_profile, self.brief, sources)
        except OSError:
            self.status.setText("วิจัยสำเร็จ แต่บันทึกข้อมูลลงดิสก์ไม่ได้")
        self.profile_data = self.store.load_profile(self.current_profile)
        self.refresh_pages()
        self.research_page.progress.setText(
            "วิจัยเสร็จแล้ว • ข้อมูลที่ได้เป็น ‘ยังไม่ยืนยัน’ จนกว่าคุณจะแก้ในช่องขวา")
        self.status.setText("วิจัยเกมเสร็จแล้ว • ข้อมูลจะถูกใช้เป็นบริบทก่อนแปล")

    def research_save(self):
        try:
            settings = self.read_settings()
            self.store.save_settings(settings)
            self.settings = settings
        except ValueError as exc:
            QMessageBox.warning(self, "บันทึกไม่ได้", str(exc))
            return
        values = self.research_page.values()
        if values["provider"] != "off" and values["key"] and values["remember_key"]:
            try:
                self.secrets.set("research:" + values["endpoint"], values["key"])
            except Exception:
                pass
        self.status.setText("บันทึกการตั้งค่าและชื่อเกมแล้ว")

    def research_save_override(self):
        text = self.research_page.override_edit.toPlainText()[:6000]
        self.brief = dict(self.brief or {})
        self.brief["user_notes"] = text
        try:
            self.store.save_research(self.current_profile, self.brief,
                                     self.profile_data.get("research_sources", []))
            self.profile_data = self.store.load_profile(self.current_profile)
        except OSError:
            QMessageBox.warning(self, "บันทึกไม่ได้", "ตรวจสิทธิ์เขียนโฟลเดอร์ข้อมูล")
            return
        self.status.setText("บันทึกโน้ตที่คุณยืนยันแล้ว • โน้ตนี้จะถูกใช้ก่อนข้อมูลวิจัยอื่น")

    def research_clear(self):
        if QMessageBox.question(self, "ล้างข้อมูลวิจัย", "ล้างสรุปวิจัยและแหล่งข้อมูลของโปรไฟล์นี้?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.brief = {}
        try:
            self.store.save_research(self.current_profile, {}, [])
        except OSError:
            QMessageBox.warning(self, "ล้างไม่ได้", "ตรวจสิทธิ์เขียนโฟลเดอร์ข้อมูล")
            return
        self.profile_data = self.store.load_profile(self.current_profile)
        self.refresh_pages()

    def maybe_auto_research(self, settings) -> bool:
        """One research pass per game before the first translation, if the user allowed it.

        Runs in parallel with translating, never blocks it, and always asks for consent before
        any query leaves the machine.
        """
        if not settings.research_auto or self.research_busy:
            return False
        title = settings.research_title or self.current_profile
        if not title or title == "ทั่วไป":
            return False
        if self.brief or self.research_started_for == title:
            return False
        if settings.research_search in ("off", ""):
            return False
        values = self.research_page.values()
        if not values["urls"] and not values["paste"].strip() and settings.research_search in ("off", ""):
            return False
        self.research_started_for = title
        self.research_page.title_edit.setText(title)
        self.research_start()
        return self.research_busy

    # --- character / face studio -------------------------------------------------------

    def load_character_card(self, name: str):
        card = self.memory.card(name) if name else {}
        self.characters_page.set_card(name, card)

    def save_character(self, name: str):
        if not name:
            warning(self, "ยังไม่ได้เลือก", "เลือกตัวละครในตารางก่อน")
            return
        values = self.characters_page.card_values()
        self.memory.ensure(name, relation=values["relation"] or "", confirmed=True)
        self.memory.set_gender(name, values["gender"], "user" if values["gender"] != "unknown" else "")
        card = self.memory.characters.get(name)
        if card is not None and values["age_gap"]:
            card["age_gap"] = values["age_gap"]
        self.memory.set_voice(name, self=values["self"], address=values["address"],
                              particles=values["particles"], register=values["register"],
                              quirks=values["quirks"], locked=values["locked"], source="user")
        try:
            self.store.save_memory(self.current_profile, self.memory)
        except OSError:
            warning(self, "บันทึกไม่ได้", "ตรวจสิทธิ์เขียนโฟลเดอร์ข้อมูล")
            return
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)
        self.status.setText(f"บันทึกการ์ดของ {name} แล้ว • AI จะใช้ค่านี้กับทุกบทพูดถ้าล็อกไว้")

    def character_add(self):
        name = ask_name(self, "เพิ่มตัวละคร", "ชื่อตัวละคร (ตามที่ผู้ใช้ยืนยัน)")
        if not name:
            return
        self.memory.ensure(name, confirmed=True)
        try:
            self.store.save_memory(self.current_profile, self.memory)
        except OSError:
            warning(self, "บันทึกไม่ได้", "ตรวจสิทธิ์เขียนโฟลเดอร์ข้อมูล")
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)

    def character_confirm(self, name: str):
        if not name:
            warning(self, "ยังไม่ได้เลือก", "เลือกตัวละครในตารางก่อน")
            return
        self.memory.ensure(name, confirmed=True)
        try:
            self.store.save_memory(self.current_profile, self.memory)
        except OSError:
            warning(self, "บันทึกไม่ได้", "ตรวจสิทธิ์เขียนโฟลเดอร์ข้อมูล")
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)

    def character_remove(self, name: str):
        if not name:
            return
        if QMessageBox.question(self, "ลบตัวละคร", f"ลบ {name} และใบหน้าที่จำไว้ทั้งหมด?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.memory.remove(name)
        try:
            self.store.save_memory(self.current_profile, self.memory)
        except OSError:
            pass
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)

    def character_forget_face(self, name: str):
        if not name:
            return
        self.memory.forget_face(name)
        try:
            self.store.save_memory(self.current_profile, self.memory)
        except OSError:
            pass
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)
        self.status.setText(f"ลบตัวอย่างใบหน้าของ {name} แล้ว (ชื่อและการ์ดยังอยู่)")

    def scan_faces(self):
        if self.vision_busy:
            self.status.setText("กำลังสแกนใบหน้าอยู่ รอสักครู่")
            return
        try:
            self._map_monitor()
        except ValueError as exc:
            QMessageBox.warning(self, "สแกนไม่ได้", str(exc))
            return
        if not self.face_analysis.isChecked():
            QMessageBox.warning(self, "ปิดการวิเคราะห์ใบหน้าไว้",
                                "เปิด ‘วิเคราะห์ใบหน้าและสีหน้า’ ในแท็บตั้งค่าและวิธีใช้ก่อน")
            return
        self.characters_page.scan_note.setText("กำลังจับภาพและสแกน…")
        if self.overlay.isVisible():
            self.overlay.hide()
            QTimer.singleShot(80, self._scan_grab)
        else:
            self._scan_grab()

    def _scan_grab(self):
        try:
            if sys.platform == "win32" and self.settings.capture_mode == "safe":
                ctypes.windll.dwmapi.DwmFlush()
            with mss.mss() as capture:
                shot = capture.grab(self.capture_bounds)
            image = Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception:
            self.characters_page.scan_note.setText("จับภาพไม่ได้ • ตรวจโหมดเกม/DRM")
            return
        finally:
            if self.overlay_visible:
                self.overlay.show()
        settings = replace(self.settings, face_analysis=True)
        self.vision_busy = True
        self.last_scan_image = image
        self.vision_requested.emit(self.epoch, -1, image, settings,
                                   self.memory.to_dict()["characters"])

    def bind_face(self):
        index = self.characters_page.selected_face_index()
        name = self.characters_page.bind_target_name()
        if not name:
            warning(self, "ต้องมีชื่อ", "เลือกหรือพิมพ์ชื่อตัวละครก่อนผูกกับใบหน้า")
            return
        if not 0 <= index < len(self.last_scan_faces):
            warning(self, "ยังไม่ได้เลือกใบหน้า", "กด ‘สแกนใบหน้า’ แล้วเลือกใบหน้าที่ต้องการก่อน")
            return
        observation = self.last_scan_faces[index]
        self.memory.ensure(name, confirmed=True)
        self.memory.remember_face(name, observation.descriptor,
                                  {"hair": observation.appearance.hair,
                                   "eyes": observation.appearance.eyes,
                                   "outfit": observation.appearance.outfit,
                                   "palette": list(observation.appearance.palette)}, confirmed=True)
        try:
            self.store.save_memory(self.current_profile, self.memory)
        except OSError:
            warning(self, "บันทึกไม่ได้", "ตรวจสิทธิ์เขียนโฟลเดอร์ข้อมูล")
            return
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)
        self.load_character_card(name)
        self.status.setText(f"ผูกใบหน้ากับ {name} แล้ว (คุณยืนยันเอง) • "
                            "ครั้งต่อไปที่ใบหน้าเดิมปรากฏจะใช้ชื่อนี้")

    def confirm_pending(self, index: int):
        vectors = {}
        for observation in self.last_scan_faces:
            vectors[f"ใบหน้าที่ {observation.track_id}"] = observation.descriptor
        name = self.memory.confirm_pending(index, vectors)
        if not name:
            return
        try:
            self.store.save_memory(self.current_profile, self.memory)
        except OSError:
            pass
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)
        self.status.setText(f"ยืนยันข้อเสนอ: {name}")

    def reject_pending(self, index: int):
        self.memory.reject_pending(index)
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)

    def pick_face_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกโมเดลใบหน้า", "", "ONNX (*.onnx)")
        if path:
            self.face_model.setText(path)

    # --- session -----------------------------------------------------------------------

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
        for widget in (self.capture_group, self.provider_group, self.advanced_group, self.face_group,
                       self.profile, self.notes, self.glossary, self.load_profile_button,
                       self.save_memory_button, self.clear_memory_button, self.once_button):
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
                    QMessageBox.warning(self, "จำ key ไม่สำเร็จ",
                                        "Credential Manager ไม่พร้อมใช้งาน จะใช้ key ในหน่วยความจำรอบนี้เท่านั้น")
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
        self.dialogue.clear()
        self.memory.reset_session()
        self.latest_image = None
        self.faces = []
        self.hints = []
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
        if self.maybe_auto_research(settings):
            self.status.setText("เริ่มค้นคว้าข้อมูลเกมไปพร้อมกับการแปล • ข้อมูลจะถูกใช้ทันทีที่พร้อม")
        elif settings.research_auto and not self.brief:
            self.status.setText("คำแนะนำ: ยังไม่มีข้อมูลวิจัยของเกมนี้ • กดเริ่มวิจัยในแท็บ "
                                "‘วิจัยเกมก่อนแปล’ เพื่อให้คำแปลตรงกับเนื้อเรื่องมากขึ้น")
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
        self.faces = []
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
        self.hints = layout.classify(blocks, image.size) if self.settings.layout_hints else []
        self._render()
        if not blocks:
            self.status.setText("ไม่พบข้อความที่ OCR อ่านได้ • คำแปลเดิมถูกล้างแล้ว")
            if self.single:
                self.stop()
            return
        due = time.monotonic() - self.last_vision_at >= self.settings.face_sample_ms / 1000
        if (self.settings.face_analysis and self.settings.face_backend != "off" and due
                and not self.vision_busy):
            self.vision_busy = True
            self.last_vision_at = time.monotonic()
            self.vision_requested.emit(epoch, self.job_id, image, replace(self.settings),
                                       self.memory.to_dict()["characters"])
            return
        self._dispatch()

    def _vision_done(self, epoch, revision, observations, error):
        if revision == -1:
            self.vision_busy = False
            if error:
                self.characters_page.scan_note.setText(error)
                return
            self.last_scan_faces = observations
            self.characters_page.scan_note.setText(
                f"พบ {len(observations)} ใบหน้า • เลือกใบหน้าแล้วผูกชื่อได้เลย" if observations
                else "ไม่พบใบหน้าในเฟรมนี้ • ลองขยับกล้อง/ฉาก หรือเปลี่ยน backend")
            self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)
            return
        self.vision_busy = False
        if self.closing or not self.active or epoch != self.epoch:
            return
        if error:
            self.status.setText(error)
        self.faces = observations
        for observation in observations:
            # Record the raw measurements; the app never turns one still into a mood claim.
            self.memory.note_cues(observation.track_id, observation.cues)
        self._dispatch()

    def _render(self):
        if self.latest_image is not None:
            self.overlay.display(self.state.blocks, self.state.rendered(), self.latest_image.size)

    def _frame_context(self, blocks):
        source = {b.id: b.text for b in blocks}
        return context_module.frame_context(self.settings, blocks, self.hints, self.faces,
                                            self.memory, self.dialogue, source)

    def _dispatch(self):
        if (not self.active or self.ai_busy or not self.state.needs_translation()
                or self.latest_image is None):
            return
        self.ai_busy = True
        self.job_id += 1
        self.job_tokens = self.state.snapshot()
        self.job_blocks = list(self.state.blocks)
        self.job_hints = list(self.hints)
        self.job_started = time.monotonic()
        frame = self._frame_context(self.job_blocks)
        faces_note = f" • ใบหน้า {len(self.faces)}" if self.settings.face_analysis and self.faces else ""
        self.status.setText(f"AI กำลังพิจารณา {len(self.job_blocks)} กล่องข้อความ{faces_note} • "
                            "ไม่มีคิวภาพสะสม")
        self.ai_requested.emit(self.epoch, self.job_id, self.job_blocks, self.latest_image,
                               replace(self.settings), self.session_key,
                               self.store.load_profile(self.current_profile), frame)

    def _apply_result(self, result):
        """Local QA: compare the Thai line with the locked plan and record dialogue memory."""
        hint_by_id = {hint.id: hint for hint in self.job_hints}
        source = {b.id: b.text for b in self.job_blocks}
        rendered = []
        for translation in result.translations:
            hint = hint_by_id.get(translation.id)
            speaker = translation.speaker or (hint.speaker if hint else "")
            resolved = self.memory.resolve(speaker) or speaker
            plan = context_module.plan_for(resolved, translation.listener,
                                           speech.analyse_source(source.get(translation.id, "")),
                                           translation.emotion, self.memory)
            gender_source = (self.memory.card(resolved).get("gender_source", "")
                             if resolved else "")
            warnings = list(translation.warnings)
            if self.settings.speech_plan:
                warnings += list(speech.check_translation(translation.thai, plan, gender_source))
            rendered.append(replace(translation, speaker=resolved, warnings=tuple(warnings)))
            self.dialogue.add(resolved, translation.thai, source.get(translation.id, ""),
                              translation.listener, translation.emotion, translation.delivery,
                              "ai", "", time.monotonic())
        result = replace(result, translations=rendered)
        return result

    def _ai_done(self, epoch, job_id, result, error, memory):
        self.ai_busy = False
        if self.closing or not self.active or epoch != self.epoch or job_id != self.job_id:
            return
        if error:
            # Pause on provider errors rather than repeatedly spending quota.
            self.stop()
            self.status.setText(error + " • หยุดอัตโนมัติแล้ว")
            self.restore()
            return
        if memory is not None:
            self.memory = memory
        result = self._apply_result(result)
        if not self.state.accept(self.job_tokens, result):
            self.status.setText("ข้อความเปลี่ยนแล้ว • ข้ามผลแปลเก่าและพิจารณาเฟรมล่าสุด")
            self._dispatch()
            return
        self._render()
        elapsed = time.monotonic() - self.job_started
        self.status.setText(f"แปลแล้ว {len(result.translations)} ข้อความ • AI {elapsed:.1f} วินาที • "
                            "รอข้อความใหม่")
        if result.scene.summary or result.scene.relations or result.scene.characters:
            try:
                # learn() applies evidence-backed hypotheses to the same memory object,
                # so the session's expression timeline is never dropped.
                self.store.learn(self.current_profile, result, self.memory)
            except OSError:
                self.status.setText("แปลสำเร็จ แต่บันทึกบริบทลงดิสก์ไม่ได้")
        self.profile_data = self.store.load_profile(self.current_profile)
        self.scene.setPlainText(
            f"People: {', '.join(result.scene.people)}\nPlace: {result.scene.place}\n"
            f"Action: {result.scene.action}\n\n{result.scene.summary}\n\n"
            "ข้อสันนิษฐานจาก AI (ไม่ใช่ข้อเท็จจริงที่ยืนยัน):\n" + json.dumps(
                self.profile_data["relations"], ensure_ascii=False, indent=2))
        self.graph.update_relations(self.profile_data["relations"])
        if result.scene.characters or result.scene.faces:
            self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)
        source = {b.id: b.text for b in self.job_blocks}
        for translation in result.translations:
            record = {
                "time": time.strftime("%H:%M:%S"),
                "speaker": translation.speaker,
                "source": source.get(translation.id, ""),
                "thai": translation.thai,
                "note": " · ".join(part for part in (
                    translation.emotion, translation.delivery[:60],
                    "; ".join(translation.warnings)) if part),
            }
            self.history.append(record)
            row = self.table.rowCount()
            self.table.insertRow(row)
            for column, name in enumerate(("time", "speaker", "source", "thai", "note")):
                self.table.setItem(row, column, QTableWidgetItem(record[name]))
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
        if self.active and self.last_frame and \
                time.monotonic() - self.last_frame > max(4, self.settings.interval_ms/500):
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
        path, _ = QFileDialog.getSaveFileName(self, "ส่งออกประวัติ", "screen-thai-history.json",
                                              "JSON (*.json)")
        if path:
            try:
                from pathlib import Path
                Path(path).write_text(json.dumps(self.history, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
            except OSError:
                QMessageBox.warning(self, "ส่งออกไม่ได้", "ตรวจสิทธิ์เขียนไฟล์ปลายทาง")

    def closeEvent(self, event: QCloseEvent):
        if self.closing and not self._workers_running():
            event.accept()
            return
        event.ignore()
        if not self.closing:
            self.closing = True
            self.closing_at = time.monotonic()
            self.stop()
            self.expiry.stop()
            self.hotkeys.close()
            self.tray.hide()
            self.overlay.close()
            self.centralWidget().setEnabled(False)
            self.status.setText("กำลังปิด • รอ OCR / AI / ใบหน้า ที่ส่งไปแล้วจบหรือ timeout")
            for thread in (self.ocr_thread, self.ai_thread, self.vision_thread, self.research_thread):
                thread.quit()
            QTimer.singleShot(0, self._shutdown_ready)

    def _workers_running(self) -> bool:
        return any(thread.isRunning() for thread in (self.ocr_thread, self.ai_thread,
                                                     self.vision_thread, self.research_thread))

    def _shutdown_ready(self):
        """Poll until the workers have really stopped, then quit.

        A worker that is inside an OCR / vision / network call needs time to reach the end of the
        slot after quit(); closing on the first check would leave a QThread running behind a
        closed window.
        """
        if not self.closing:
            return
        if self._workers_running():
            if time.monotonic() - getattr(self, "closing_at", time.monotonic()) < 60.0:
                QTimer.singleShot(150, self._shutdown_ready)
            else:
                self.close()
                QApplication.instance().quit()
            return
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
