from __future__ import annotations

import ctypes
from dataclasses import replace
import json
import sys
import time
from urllib.parse import urlsplit

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
from . import diagnostics, layout, speech
from .dialogue import DialogueMemory
from .faces import FaceMemory
from pathlib import Path

from .llama import LlamaError, LlamaRuntime, bin_dir, extract_binary, find_binary, hf_model_files
from .llama import installed_models, models_dir, pick_quant_files, settings_guess
from .memory import TranslationMemory, mine_candidates, merge_candidates, voice_signature
from .merge import merge_lines
from .models import Result, Translation
from .overlay import Overlay, RegionPicker
from .pages import CharactersPage, LocalAIPage, ResearchPage, ask_name, warning
from .pipeline import AIWorker, DownloadWorker, OCRWorker, ResearchWorker, VisionWorker
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
    download_requested = Signal(str, str, str)

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
        self.tm = TranslationMemory()
        self.llama = LlamaRuntime()
        self.pending_pairs = []          # (source, thai) waiting to be mined for glossary terms
        self.negative_lines = []         # Thai lines used as negative evidence by the miner
        self.mine_at = 0
        self.suggestions = []
        self.swallowed = {}              # merged-away block ids -> the id that keeps the text
        self.cache_hits = 0
        self.latest_image = None
        self.job_blocks = []
        self.hints = []
        self.faces = []
        self.last_scan_image = None
        self.last_scan_faces = []
        self.last_vision_at = 0.0
        self.research_started_for = ""
        self.downloads = {}                 # label -> {"total": int, "done": int, "name": str}
        self.download_failures = []
        self.pending_assets = []
        self.pending_llama_files = []
        self.history = []
        self.last_frame = 0.0
        self.capture_pending = False
        self.picker = None
        self.current_profile = self.settings.profile
        self.memory = self.store.load_memory(self.current_profile)
        self.tm = self.store.load_memory_of_translation(self.current_profile)
        self.suggestions = self.store.load_suggestions(self.current_profile)
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
        self.download_thread = QThread(self)
        self.download_worker = DownloadWorker()
        self.download_worker.moveToThread(self.download_thread)
        self.download_requested.connect(self.download_worker.process)
        self.download_worker.progress.connect(self._download_progress)
        self.download_worker.done.connect(self._download_done)
        self.download_thread.finished.connect(self.download_worker.deleteLater)
        self.download_thread.finished.connect(self._shutdown_ready)
        self.download_thread.start()
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
        self.local_page = LocalAIPage(self)
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

        self.quality_group = QGroupBox("คุณภาพการแปล · ประหยัดเวลาและค่าใช้จ่าย")
        quality_form = QFormLayout(self.quality_group)
        self.merge_lines = QCheckBox("ผสานบรรทัดซับที่ถูกตัดขึ้นบรรทัดใหม่ให้เป็นประโยคเดียวก่อนแปล")
        self.translation_memory = QCheckBox("ใช้คำแปลเดิมที่คุณเคยยืนยันแล้วกับข้อความที่ซ้ำ "
                                            "(ไม่เรียก AI ซ้ำ)")
        self.glossary_mine = QCheckBox("ขุดคำศัพท์/ชื่อที่พบบ่อยขึ้นมาเป็นข้อเสนอ glossary "
                                       "(คุณกดรับก่อนจึงจะถูกใช้)")
        self.review_consistency = QCheckBox("รอบตรวจความสม่ำเสมอ: ให้ AI แก้เฉพาะบรรทัดที่ผิด "
                                            "glossary หรือการ์ดที่ล็อกไว้ (เพิ่ม 1 คำขอ)")
        self.review_max_lines = QSpinBox()
        self.review_max_lines.setRange(1, 40)
        self.review_max_lines.setSuffix(" บรรทัด/รอบ")
        for widget in (self.merge_lines, self.translation_memory, self.glossary_mine,
                       self.review_consistency):
            quality_form.addRow("", widget)
        quality_form.addRow("เพดานบรรทัดในรอบตรวจ", self.review_max_lines)
        quality_note = QLabel("หน่วยความจำคำแปลเก็บในโปรไฟล์เกมนี้เท่านั้น และบรรทัดที่ขึ้นเตือน "
                              "จะไม่ถูกใช้ซ้ำอัตโนมัติ • ข้อเสนอ glossary ไม่ถูกใช้จนคุณกดรับ")
        quality_note.setObjectName("muted")
        quality_note.setWordWrap(True)
        quality_form.addRow(quality_note)
        al.addWidget(self.quality_group)

        self.selfcheck_group = QGroupBox("ตรวจสุขภาพระบบ (ทำงานในเครื่อง ไม่ส่งข้อมูล)")
        check_layout = QVBoxLayout(self.selfcheck_group)
        check_row = QHBoxLayout()
        run_check = QPushButton("ตรวจตอนนี้")
        run_check.clicked.connect(lambda: self.run_selfcheck(False))
        run_check_ai = QPushButton("ตรวจการเชื่อมต่อ AI ด้วย")
        run_check_ai.clicked.connect(lambda: self.run_selfcheck(True))
        self.model_picker = QComboBox()
        self.model_picker.setMinimumWidth(220)
        pick = QPushButton("ใช้โมเดลนี้")
        pick.clicked.connect(self.apply_picked_model)
        check_row.addWidget(run_check)
        check_row.addWidget(run_check_ai)
        check_row.addWidget(self.model_picker, 1)
        check_row.addWidget(pick)
        check_layout.addLayout(check_row)
        self.selfcheck_view = QPlainTextEdit()
        self.selfcheck_view.setReadOnly(True)
        self.selfcheck_view.setMaximumHeight(190)
        self.selfcheck_view.setPlaceholderText(
            "กด ‘ตรวจตอนนี้’ เพื่อตรวจว่า OCR, ใบหน้า, แผนการพูด, โฟลเดอร์ข้อมูล, ฟอนต์ไทย "
            "และหน่วยความจำคำแปล พร้อมใช้งานจริงบนเครื่องนี้หรือไม่")
        check_layout.addWidget(self.selfcheck_view)
        al.addWidget(self.selfcheck_group)

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
        self.tabs.addTab(self.local_page, "Local AI (llama.cpp)")
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
        self.merge_lines.setChecked(s.merge_lines)
        self.translation_memory.setChecked(s.translation_memory)
        self.glossary_mine.setChecked(s.glossary_mine)
        self.review_consistency.setChecked(s.review_consistency)
        self.review_max_lines.setValue(s.review_max_lines)
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
        self.tm = self.store.load_memory_of_translation(self.current_profile)
        self.suggestions = self.store.load_suggestions(self.current_profile)
        self.pending_pairs = []
        self.negative_lines = []
        self.cache_hits = 0
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
            self.store.save_translation_memory(self.current_profile, self.tm)
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
                       merge_lines=self.merge_lines.isChecked(),
                       translation_memory=self.translation_memory.isChecked(),
                       glossary_mine=self.glossary_mine.isChecked(),
                       review_consistency=self.review_consistency.isChecked(),
                       review_max_lines=self.review_max_lines.value(),
                       research_search=research["provider"],
                       research_endpoint=research["endpoint"],
                       research_max_sources=research["max_sources"],
                       research_title=research["title"], research_aliases=research["aliases"],
                       research_auto=research["auto"],
                       llama_binary=self.local_page.values()["binary"],
                       llama_port=self.local_page.values()["port"],
                       llama_ctx=self.local_page.values()["ctx"],
                       llama_gpu_layers=self.local_page.values()["gpu"])

    # --- pages -------------------------------------------------------------------------

    def refresh_pages(self):
        self.research_page.load(self.settings, self.brief,
                                self.profile_data.get("research_sources", []),
                                str(self.brief.get("user_notes", "")),
                                self.settings.research_title or self.current_profile)
        self.characters_page.refresh(self.memory, self.last_scan_faces, self.last_scan_image)
        self.characters_page.refresh_suggestions(self.suggestions)
        self.local_page.load(self.settings)
        self.refresh_llama_models()

    # --- local AI (llama.cpp) ------------------------------------------------------------

    def llama_note(self, message: str):
        self.local_page.download_note.setText(message)

    def pick_llama_binary(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "เลือกไฟล์ llama-server", "", "llama-server (llama-server*);;ทุกไฟล์ (*)")
        if path:
            self.local_page.binary_edit.setText(path)
            self.status.setText("เลือกไฟล์ llama-server แล้ว • กด ‘เริ่ม llama-server’ เพื่อทดสอบ")

    def fetch_llama_assets(self):
        """Ask GitHub for the published llama.cpp builds (metadata only, no download yet)."""
        answer = QMessageBox.question(
            self, "ดึงรายการรุ่นจาก GitHub?",
            "จะเชื่อมต่อ api.github.com เพื่อดูรายชื่อไฟล์รุ่นล่าสุดของ llama.cpp\n\n"
            "ยังไม่มีการดาวน์โหลดไฟล์ใด ๆ ในขั้นนี้ • ไม่มีการส่งข้อมูลหน้าจอหรือโปรไฟล์เกม\n"
            "อนุญาตหรือไม่?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            from .llama import llama_assets
            assets = llama_assets()
        except LlamaError as exc:
            QMessageBox.warning(self, "ดึงรายการไม่ได้", str(exc))
            return
        self.pending_assets = assets
        self.local_page.asset_combo.clear()
        if not assets:
            self.local_page.asset_combo.addItem("ไม่พบไฟล์ zip ที่ใช้กับระบบนี้", "")
            return
        for entry in assets:
            size = entry["size"] / (1 << 30)
            self.local_page.asset_combo.addItem(
                f"{entry['tag']} · {entry['name']} ({size:.2f} GB)", entry)
        self.status.setText(f"พบ {len(assets)} รุ่น • เลือกรุ่นที่ตรงกับการ์ดจอของคุณ "
                            "(cuda = NVIDIA, vulkan = ทั่วไป, cpu = ไม่มีการ์ดจอ)")

    def install_llama_asset(self, asset):
        if not isinstance(asset, dict) or not asset.get("url"):
            QMessageBox.warning(self, "ยังไม่ได้เลือกรุ่น", "กด ‘ดึงรายการจาก GitHub’ แล้วเลือกรุ่นก่อน")
            return
        if not self._confirm_download([(asset["name"], asset.get("size", 0))],
                                      "ไฟล์รัน llama-server (zip จาก GitHub ทางการของ llama.cpp)"):
            return
        target = self.store.root / "llama" / "downloads" / str(asset["name"])
        self._start_download("llama-server", asset["url"], target)

    def list_llama_files(self, repo: str):
        if not repo:
            QMessageBox.warning(self, "ยังไม่มี repo", "เลือกโมเดลแนะนำหรือพิมพ์ repo ก่อน")
            return
        answer = QMessageBox.question(
            self, "ดูรายการไฟล์บน Hugging Face?",
            f"จะอ่านรายชื่อไฟล์ของ {repo} ผ่าน huggingface.co\n\n"
            "เป็นการอ่านข้อมูลสาธารณะ ไม่ได้ส่งข้อมูลหน้าจอหรือโปรไฟล์เกม\nอนุญาตหรือไม่?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            files = hf_model_files(repo)
        except LlamaError as exc:
            QMessageBox.warning(self, "อ่านรายการไฟล์ไม่ได้", str(exc))
            return
        if not files:
            QMessageBox.information(self, "ไม่พบไฟล์ GGUF",
                                    "repo นี้ไม่มีไฟล์ .gguf • ลองชื่อ repo อื่น")
            return
        self.pending_llama_files = files
        self.local_page.repo_edit.setText(repo)
        self.local_page.show_files(files, repo)
        self.status.setText(f"พบ {len(files)} ไฟล์ GGUF • เลือกแถวแล้วกดดาวน์โหลด")

    def _confirm_download(self, entries, what: str) -> bool:
        """One consent dialog per batch, naming every host, size and where the file will land."""
        from .llama import ALLOWED_HOSTS
        total = sum(size for _, size in entries) / (1 << 30)
        names = "\n".join(f"• {name} ({size / (1 << 30):.2f} GB)" if size else f"• {name}"
                           for name, size in entries[:6])
        answer = QMessageBox.question(
            self, "ยืนยันการดาวน์โหลด?",
            f"{what}\n\n{names}\n\nรวมประมาณ {total:.2f} GB • "
            f"เก็บไว้ที่ {models_dir(self.store.root).parent}\n"
            f"อนุญาตให้เชื่อมต่อเฉพาะ: {', '.join(ALLOWED_HOSTS[:2])} …\n"
            "ไฟล์เป็นข้อมูลจากอินเทอร์เน็ต — ตรวจพื้นที่ว่างและแพ็กเกจอินเทอร์เน็ตของคุณด้วย\n"
            "เริ่มดาวน์โหลดหรือไม่?")
        return answer == QMessageBox.StandardButton.Yes

    def download_llama_selection(self, row: int):
        if row < 0 or row >= len(self.pending_llama_files):
            QMessageBox.warning(self, "ยังไม่ได้เลือกไฟล์", "เลือกรายการในตารางก่อน")
            return
        entry = self.pending_llama_files[row]
        repo = self.local_page.repo_edit.text().strip()
        self._download_entries(repo, [entry])

    def download_llama_recommended(self):
        """Download the suggested quant plus the matching mmproj, using the HF API list."""
        repo = self.local_page.repo_edit.text().strip() or self.local_page._catalog_field("repo")
        if not self.pending_llama_files or self.local_page.repo_edit.text().strip() != repo:
            self.list_llama_files(repo)
            if not self.pending_llama_files:
                return
        quant = self.local_page.quant_combo.currentText()
        model, projector = pick_quant_files(self.pending_llama_files, quant)
        entries = [entry for entry in (model, projector) if entry]
        if not entries:
            QMessageBox.warning(self, "เลือกไฟล์ไม่ได้", "ไม่พบไฟล์ที่ตรงกับ quant นี้")
            return
        self._download_entries(repo, entries)

    def _download_entries(self, repo: str, entries):
        unknown = [entry for entry in entries if not entry.get("size")]
        if unknown and not self._confirm_download([(entry["name"], 0) for entry in unknown],
                                                 "ไฟล์ที่ไม่ทราบขนาดล่วงหน้า"):
            return
        if not self._confirm_download([(entry["name"], entry.get("size", 0)) for entry in entries],
                                      f"โมเดลจาก Hugging Face ({repo})"):
            return
        for entry in entries:
            from .llama import hf_download_url
            try:
                url = hf_download_url(repo, entry["name"])
            except LlamaError as exc:
                QMessageBox.warning(self, "ลิงก์ไม่ถูกต้อง", str(exc))
                continue
            target = models_dir(self.store.root) / entry["name"]
            self._start_download(entry["name"], url, target)

    def _start_download(self, label: str, url: str, target: Path):
        if label in self.downloads:
            self.status.setText(f"กำลังดาวน์โหลด {label} อยู่แล้ว")
            return
        self.downloads[label] = {"done": 0, "total": 0, "name": label}
        self.local_page.cancel_button.setEnabled(True)
        self.download_requested.emit(label, url, str(target))
        self.status.setText(f"เริ่มดาวน์โหลด {label} … (เรียนต่อได้ถ้าหยุดกลางคัน)")

    def _download_progress(self, label: str, done: int, total: int):
        # Only an *active* download is tracked here: a stray progress signal must never make the
        # next attempt think a transfer is already running (that would block retries).
        entry = self.downloads.get(label)
        if entry is not None:
            entry["done"], entry["total"] = done, total
        percent = (done / total * 100) if total else 0.0
        self.local_page.download_note.setText(
            f"{label}: {done / (1 << 30):.2f}/{total / (1 << 30):.2f} GB ({percent:.0f}%)")
        self.status.setText(f"ดาวน์โหลด {label} • {percent:.0f}%")

    def _download_done(self, label: str, path: str, error: str):
        entry = self.downloads.pop(label, None)
        if not self.downloads:
            self.local_page.cancel_button.setEnabled(False)
        name = (entry or {}).get("name", label)
        self.local_page.mark_file(name, "เสร็จแล้ว" if path else f"ล้มเหลว: {error[:40]}")
        if error:
            self.download_failures.append((name, error))
            self.local_page.download_note.setText(f"{name}: {error}")
            self.status.setText(f"ดาวน์โหลด {name} ไม่สำเร็จ: {error}")
            return
        size_gb = Path(path).stat().st_size / (1 << 30)
        self.local_page.download_note.setText(f"{name}: เสร็จแล้ว ({size_gb:.2f} GB) → {path}")
        self.status.setText(f"ดาวน์โหลด {name} เสร็จ • ไปที่ขั้นที่ 3 เพื่อเริ่มใช้งาน")
        if name.lower().endswith(".zip"):
            self._install_binary_zip(Path(path))
            return
        self.refresh_llama_models()

    def _install_binary_zip(self, zip_path: Path):
        try:
            written = extract_binary(zip_path, bin_dir(self.store.root))
        except (LlamaError, OSError) as exc:
            QMessageBox.warning(self, "แตกไฟล์ไม่ได้", str(exc))
            return
        binary = next((path for path in written
                       if path.name.lower().startswith("llama-server")), None)
        if binary is None:
            QMessageBox.warning(self, "ไม่พบ llama-server", "แตกไฟล์ได้แต่ไม่พบโปรแกรม")
            return
        self.local_page.binary_edit.setText(str(binary))
        self.settings = replace(self.settings, llama_binary=str(binary))
        try:
            self.store.save_settings(self.settings)
        except OSError:
            pass
        zip_path.unlink(missing_ok=True)
        QMessageBox.information(
            self, "ติดตั้งแล้ว",
            f"ติดตั้ง llama-server ที่\n{binary}\n\nไฟล์ zip ถูกลบหลังแตกแล้ว • "
            "ต่อไปเลือกรุ่นโมเดลในขั้นที่ 2 และกด ‘เริ่ม llama-server’")

    def cancel_downloads(self):
        self.download_worker.cancel()
        self.status.setText("กำลังยกเลิกการดาวน์โหลด (ไฟล์บางส่วนถูกเก็บไว้ให้เรียนต่อ)")
        self.local_page.cancel_button.setEnabled(False)

    def refresh_llama_models(self):
        models = installed_models(self.store.root)
        self.local_page.show_models(models)
        count = len(models)
        self.status.setText(f"พบโมเดลในเครื่อง {count} ไฟล์" if count
                            else "ยังไม่มีโมเดลในเครื่อง • ดาวน์โหลดในขั้นที่ 2")
        return models

    def start_llama(self) -> bool:
        values = self.local_page.values()
        entry = self.local_page.model_combo.currentData() or {}
        model = entry.get("path") or self.settings.llama_model
        binary = values["binary"] or self.settings.llama_binary
        if not binary:
            binary = ""
        if not model:
            QMessageBox.warning(self, "ยังไม่ได้เลือกโมเดล", "ดาวน์โหลดหรือเลือกไฟล์ .gguf ก่อน")
            return False
        guess = settings_guess(installed_models(self.store.root))
        try:
            self.llama.binary = find_binary(binary, bin_dir(self.store.root)) or \
                (Path(binary) if binary else None)
            url = self.llama.start(Path(model), port=values["port"], ctx=values["ctx"],
                                   gpu_layers=values["gpu"],
                                   mmproj=Path(entry["projector"]) if entry.get("projector") else None)
        except LlamaError as exc:
            QMessageBox.warning(self, "เริ่มไม่ได้", str(exc))
            return False
        self.local_page.status_label.setText(
            f"กำลังเริ่ม… {Path(model).name} • {guess['note']}")
        self.settings = replace(self.settings, llama_binary=str(self.llama.binary or binary),
                                llama_model=str(model), llama_port=values["port"],
                                llama_ctx=values["ctx"], llama_gpu_layers=values["gpu"])
        self.local_page.port.setValue(self.settings.llama_port)
        self.local_page.ctx.setValue(self.settings.llama_ctx)
        self.local_page.gpu.setValue(self.settings.llama_gpu_layers)
        self._llama_poll(attempts=0)
        self._llama_log_tick()
        self.status.setText(f"กำลังโหลดโมเดลเข้า llama-server ที่ {url} …")
        return True

    def _llama_poll(self, attempts: int):
        """Poll /health without blocking the UI; report the real reason when it fails."""
        if self.llama.state.running:
            self.local_page.status_label.setText(self.llama.status_text())
            self.status.setText("llama-server พร้อมใช้งาน • กด ‘ตั้งเป็น AI ของแอป’ เพื่อใช้")
            models = self.llama.models()
            if models:
                self.local_page.status_label.setText(
                    self.llama.status_text() + f" • โมเดลที่โหลด: {models[0][:60]}")
            return
        if attempts > 300:                     # ~150 วินาที จึงยอมสรุปว่าโหลดไม่สำเร็จ
            self.llama.timed_out()
            self.local_page.status_label.setText(self.llama.status_text())
            self.status.setText("llama-server ยังไม่พร้อม • ดู log ในแท็บ Local AI "
                                "(ลอง quant เล็กลงหรือลด context)")
            return
        if attempts % 2 == 0:
            # Short, non-terminal probe: 503/connection-refused during model load is normal and
            # must not be reported as a failure (and must not block the UI for long).
            self.llama.probe(timeout=0.7)
            if not self.llama.state.running and self.llama.state.starting:
                self.local_page.status_label.setText(self.llama.status_text())
        if self.llama.state.error and not self.llama.state.starting:
            self.local_page.status_label.setText("มีปัญหา: " + self.llama.state.error)
            self.status.setText("llama-server มีปัญหา • ดู log ในแท็บ Local AI")
            return
        QTimer.singleShot(500, lambda: self._llama_poll(attempts + 1))

    def _llama_log_tick(self):
        log = self.llama.state.log[-40:]
        if log:
            text = "\n".join(log)
            if text != self.local_page.log_view.toPlainText()[-4000:]:
                self.local_page.log_view.setPlainText(text)
        if self.llama.state.starting or self.llama.state.running:
            QTimer.singleShot(1000, self._llama_log_tick)

    def stop_llama(self) -> bool:
        stopped = self.llama.stop()
        self.local_page.status_label.setText(self.llama.status_text())
        self.status.setText("หยุด llama-server แล้ว" if stopped else "หยุดไม่สำเร็จ")
        return stopped

    def use_llama_endpoint(self) -> bool:
        """Point the app at the local server without touching anything else."""
        if not self.llama.state.running:
            QMessageBox.warning(self, "ยังไม่ทำงาน",
                                "กด ‘เริ่ม llama-server’ และรอสถานะเป็น ‘ทำงานอยู่’ ก่อน")
            return False
        endpoint = self.llama.base_url
        self.provider.setCurrentIndex(max(0, self.provider.findData("openai")))
        self.endpoint.setText(endpoint)
        models = self.llama.models()
        if models:
            self.model.setText(models[0])
        elif not self.model.text().strip():
            self.model.setText(Path(self.settings.llama_model).name or "local-model")
        self.settings = replace(self.settings, provider="openai", endpoint=endpoint,
                                model=self.model.text().strip())
        try:
            self.store.save_settings(self.read_settings())
        except (OSError, ValueError):
            pass
        self.status.setText(f"ตั้งเป็น AI ของแอปแล้ว • {endpoint} • "
                            "โหมดนี้อยู่ในเครื่อง จึงไม่ต้องยืนยันส่งข้อมูลออกนอกเครื่อง")
        self.local_page.status_label.setText(self.llama.status_text() + " • ใช้เป็น AI ของแอปแล้ว")
        return True

    def llama_auto_start_allowed(self) -> bool:
        """Auto-start only when the app already points at the loopback server.

        A remote endpoint or a non-openai provider must never launch a local process behind the
        user's back; a missing binary/model is likewise a quiet "no".
        """
        s = self.settings
        return bool(s.llama_auto_start and s.llama_binary and s.llama_model
                    and s.provider == "openai"
                    and s.endpoint.startswith(("http://127.0.0.1", "http://localhost")))

    def ensure_llama_running(self) -> bool:
        """Start the local server when it is configured and wanted (never silently downloads)."""
        if self.llama.state.running:
            return True
        if not self.llama_auto_start_allowed():
            return False
        return self.start_llama()

    # --- self-check ---------------------------------------------------------------------

    def run_selfcheck(self, probe_ai: bool = False):
        """Report honestly what works on this machine; never claim an untested part is fine."""
        try:
            settings = self.read_settings()
        except ValueError:
            settings = self.settings
        key = ""
        if probe_ai:
            try:
                parts = urlsplit(settings.endpoint)
                if parts.hostname not in ("localhost", "127.0.0.1", "::1"):
                    answer = QMessageBox.question(
                        self, "ตรวจการเชื่อมต่อ AI?",
                        f"จะเรียกดู ‘รายชื่อโมเดล’ จาก\n{settings.endpoint}\n\n"
                        "ไม่มีการส่งภาพหน้าจอ ข้อความ OCR หรือ notes ในการตรวจครั้งนี้\nอนุญาตหรือไม่?")
                    if answer != QMessageBox.StandardButton.Yes:
                        probe_ai = False
                if probe_ai:
                    key = self.key.text().strip() or self.secrets.get(settings.endpoint)
            except Exception:
                key = ""
        self.status.setText("กำลังตรวจสุขภาพระบบ…")
        QApplication.processEvents()
        checks = diagnostics.run_checks(settings, self.store, profile_name=self.current_profile,
                                        memory=self.memory, tm=self.tm, brief=self.brief,
                                        probe_ai=probe_ai, key=key)
        self.selfcheck_view.setPlainText(diagnostics.report_text(checks))
        if probe_ai:
            try:
                from .providers import list_models
                models = list_models(settings, key=key)
            except Exception:
                models = []
            if models:
                current = self.model.text().strip()
                self.model_picker.clear()
                self.model_picker.addItems(models)
                if current in models:
                    self.model_picker.setCurrentText(current)
        counts = diagnostics.summary(checks)
        self.status.setText(f"ตรวจแล้ว: ผ่าน {counts[diagnostics.OK]} · "
                            f"มีข้อจำกัด {counts[diagnostics.WARN]} · "
                            f"ไม่ผ่าน {counts[diagnostics.FAIL]} · ข้าม {counts[diagnostics.SKIP]}")
        return checks

    def apply_picked_model(self):
        name = self.model_picker.currentText().strip()
        if not name:
            return
        self.model.setText(name)
        self.status.setText(f"ตั้งโมเดลเป็น {name} แล้ว • กดบันทึกการตั้งค่าหรือเริ่มแปลเพื่อใช้")
        self.run_selfcheck(False)

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
            if self.llama_auto_start_allowed() and not self.llama.state.running:
                self.ensure_llama_running()
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
        self.hints = layout.classify(blocks, image.size) if self.settings.layout_hints else []
        blocks, self.swallowed, _ = merge_lines(blocks, self.hints) \
            if self.settings.merge_lines and self.settings.layout_hints else (blocks, {}, {})
        if self.swallowed:
            keep = {block.id for block in blocks}
            self.hints = [hint for hint in self.hints if hint.id in keep]
        self.state.update(blocks)
        self._apply_memory_hits()
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

    def _apply_memory_hits(self):
        """Render repeated lines straight from the player's own translation memory.

        Only lines that were accepted without QC warnings come back, and a line whose character
        card changed since it was stored is left for the model again.
        """
        if not self.settings.translation_memory:
            return
        hits = self.tm.lookup(self.state.blocks, self.hints, self.memory)
        if not hits:
            return
        cached = Result([Translation(block_id, hit["thai"], hit["speaker"])
                         for block_id, hit in sorted(hits.items())])
        tokens = {block_id: self.state.tokens[block_id] for block_id in hits
                  if block_id in self.state.tokens}
        if self.state.accept(tokens, cached) and cached.translations:
            self.cache_hits += len(cached.translations)
            self.status.setText(f"ใช้คำแปลที่คุณเคยยืนยันไว้ {len(cached.translations)} บรรทัด "
                                f"(ไม่เรียก AI) • ประหยัดเวลา/ค่าใช้จ่าย")

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
        # Only blocks whose occurrence has no answer yet: lines already served from the
        # translation memory are never sent to the model again.
        self.job_blocks = [block for block in self.state.blocks
                           if self.state.tokens.get(block.id) not in self.state.resolved]
        if not self.job_blocks:
            self.ai_busy = False
            self.status.setText("ทุกบรรทัดมาจากหน่วยความจำคำแปลแล้ว • รอข้อความใหม่")
            return
        # The answer may only resolve the occurrences we are asking about; a cached line keeps
        # its rendering even though this request never mentioned it.
        self.job_tokens = {block.id: self.state.tokens[block.id] for block in self.job_blocks}
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
            self._learn_line(source.get(translation.id, ""), translation.thai, resolved,
                             warnings)
        result = replace(result, translations=rendered)
        return result

    def _learn_line(self, source: str, thai: str, speaker: str, warnings):
        """Feed the translation memory and the glossary miner.

        A line that raised a QC warning is stored but marked for review, so it is never replayed
        blindly. The miner only gets clean pairs, so a bad draft cannot teach a bad term.
        """
        if not source or not thai:
            return
        card = self.memory.card(speaker) if speaker else None
        voice = voice_signature(card) if card else ""
        if self.settings.translation_memory:
            self.tm.learn(source, thai, speaker, voice, warned=bool(warnings))
        if warnings or not self.settings.glossary_mine:
            return
        self.pending_pairs.append((source, thai))
        del self.pending_pairs[:-400]
        # Thai lines from another frame are negative evidence for the miner: a chunk that also
        # shows up in a line without the term is not that term's rendering.
        self.negative_lines.append(thai)
        del self.negative_lines[:-60]
        if len(self.pending_pairs) - self.mine_at >= 15:
            self.mine_glossary()

    def mine_glossary(self) -> int:
        """Suggest glossary terms from repeated names/kanji and the Thai wording already used."""
        if not self.pending_pairs:
            return 0
        self.mine_at = len(self.pending_pairs)
        found = mine_candidates(self.pending_pairs, self.glossary.toPlainText(), min_count=2,
                                negative=self.negative_lines[-40:])
        before = len(self.suggestions)
        self.suggestions = merge_candidates(self.suggestions, found)
        added = max(0, len(self.suggestions) - before)
        try:
            self.store.save_suggestions(self.current_profile, self.suggestions)
        except OSError:
            self.status.setText("ขุดคำศัพท์ได้ แต่บันทึกข้อเสนอลงดิสก์ไม่ได้")
        if added:
            self.status.setText(f"พบคำศัพท์ที่น่าเพิ่มใน glossary {added} คำ "
                                f"(ดูได้ในแท็บตัวละคร · ยังไม่ถูกใช้จนคุณกดเพิ่ม)")
        self.refresh_pages()
        return added

    def dismiss_suggestions(self, indexes) -> int:
        """Hide suggestions the player does not want; they can be mined again later."""
        indexes = set(indexes or [])
        if not indexes:
            return 0
        self.suggestions = [item for index, item in enumerate(self.suggestions)
                            if index not in indexes]
        try:
            self.store.save_suggestions(self.current_profile, self.suggestions)
        except OSError:
            pass
        self.refresh_pages()
        return len(indexes)

    def accept_suggestions(self, indexes) -> int:
        """Move chosen suggestions into the user's glossary — the only place terms are used from."""
        current = self.glossary.toPlainText()
        pairs = list(self.memory_module_pairs(current))
        accepted = 0
        keep = []
        for index, item in enumerate(self.suggestions):
            if index in set(indexes):
                pairs.append((item["source"], item["thai"]))
                accepted += 1
            else:
                keep.append(item)
        if not accepted:
            return 0
        seen = []
        for source, thai in pairs:
            if (source, thai) not in seen:
                seen.append((source, thai))
        from .memory import glossary_text
        self.glossary.setPlainText(glossary_text(seen)[:4000])
        self.suggestions = keep
        self.store.save_suggestions(self.current_profile, keep)
        self.save_memory()
        self.refresh_pages()
        self.status.setText(f"เพิ่ม {accepted} คำใน glossary แล้ว • จะถูกใช้กับการแปลครั้งถัดไป")
        return accepted

    @staticmethod
    def memory_module_pairs(text: str):
        from .memory import parse_glossary
        return parse_glossary(text)

    def _ai_done(self, epoch, job_id, result, error, memory):
        if job_id == self.job_id:
            # Only the response of the current job may clear the flag: a late answer from an
            # older request must not make the app think the newer request finished.
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
        try:
            if self.settings.translation_memory:
                self.store.save_translation_memory(self.current_profile, self.tm)
        except OSError:
            self.status.setText("แปลสำเร็จ แต่บันทึกหน่วยความจำคำแปลไม่ได้")
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
            try:
                self.llama.stop(grace=4.0)      # never leave a llama-server child behind
            except Exception:
                pass
            self.expiry.stop()
            self.hotkeys.close()
            self.tray.hide()
            self.overlay.close()
            self.centralWidget().setEnabled(False)
            self.status.setText("กำลังปิด • รอ OCR / AI / ใบหน้า ที่ส่งไปแล้วจบหรือ timeout")
            for thread in (self.ocr_thread, self.ai_thread, self.vision_thread,
                           self.research_thread, self.download_thread):
                thread.quit()
            QTimer.singleShot(0, self._shutdown_ready)

    def _workers_running(self) -> bool:
        return any(thread.isRunning() for thread in (self.ocr_thread, self.ai_thread,
                                                     self.vision_thread, self.research_thread,
                                                     self.download_thread))

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
