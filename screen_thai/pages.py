"""Extra UI pages: pre-translation research and the character/face/voice studio."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .llama import CATALOG, QUANTS, TIP
from .research import DEFAULT_ENDPOINTS, SEARCH_PROVIDERS

SEARCH_LABELS = [("ปิด · ไม่ค้นเว็บ", "off"), ("SearXNG (ของตัวเอง/ในเครื่อง)", "searxng"),
                 ("Brave Search API", "brave"), ("Serper (Google)", "serper"),
                 ("Tavily", "tavily"), ("กำหนดเอง (JSON)", "custom")]


def pil_to_pixmap(image, box=None, size: int = 108) -> QPixmap:
    if image is None:
        return QPixmap()
    crop = image
    if box is not None:
        x, y, w, h = box
        pad = int(0.25 * max(w, h))
        crop = image.crop((max(0, x - pad), max(0, y - pad),
                           min(image.width, x + w + pad), min(image.height, y + h + pad)))
    crop = crop.convert("RGB")
    crop.thumbnail((size, size))
    data = crop.tobytes("raw", "RGB")
    qimage = QImage(data, crop.width, crop.height, crop.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


def render_brief(brief: dict) -> str:
    """Human-readable brief for the UI (the prompt version is built in research.brief_text)."""
    if not isinstance(brief, dict) or not brief:
        return "ยังไม่มีข้อมูลวิจัย — วางข้อความหรือลิงก์ แล้วกด เริ่มวิจัยเกม"
    lines = ["ข้อมูลนี้มาจากการค้นคว้า/ผู้ใช้ ยังไม่ยืนยัน — ถ้าขัดกับหน้าจอ ให้เชื่อหน้าจอ", ""]
    for key, label in (("title", "ชื่อเกม"), ("genre", "แนว"), ("setting", "ฉาก/สถานที่"),
                       ("era", "ยุคสมัย"), ("region", "ภูมิภาค"), ("developer", "ผู้พัฒนา"),
                       ("publisher", "ผู้จัดจำหน่าย"), ("tone", "โทนเรื่อง"),
                       ("summary", "เรื่องย่อ"), ("localization_style", "สไตล์การแปล"),
                       ("honorific_policy", "นโยบายคำเรียก/ความสุภาพ"),
                       ("pronoun_notes", "โน้ตสรรพนาม"), ("spoiler_notes", "ข้อควรระวัง")):
        value = str(brief.get(key, "")).strip()
        if value:
            lines.append(f"{label}: {value}")
    for key, label in (("aliases", "ชื่ออื่น"), ("languages", "ภาษา")):
        values = brief.get(key)
        if isinstance(values, list) and values:
            lines.append(f"{label}: " + ", ".join(str(v) for v in values[:12]))
    characters = brief.get("characters")
    if isinstance(characters, list) and characters:
        lines.append("\nตัวละคร:")
        for card in characters[:20]:
            if isinstance(card, dict):
                lines.append(f"  • {card.get('name', '')} — {card.get('role', '')} "
                             f"{card.get('personality', '')} | พูด: {card.get('speech_style', '')}"
                             f" | หลักฐาน: {card.get('evidence', '')}")
    for key, label in (("names", "ชื่อเฉพาะ"), ("terms", "คำศัพท์")):
        entries = brief.get(key)
        if isinstance(entries, list) and entries:
            lines.append(f"\n{label}:")
            for item in entries[:24]:
                if isinstance(item, dict):
                    lines.append(f"  • {item.get('source', '')} → {item.get('thai', '')} "
                                 f"({item.get('reason', '')})")
    uncertain = brief.get("uncertain")
    if isinstance(uncertain, list) and uncertain:
        lines.append("\nยังไม่ยืนยัน: " + "; ".join(str(item) for item in uncertain[:10]))
    sources = brief.get("sources")
    if isinstance(sources, list) and sources:
        lines.append("\nแหล่งอ้างอิง:")
        for item in sources[:10]:
            if isinstance(item, dict):
                lines.append(f"  • {item.get('title', '')} {item.get('url', '')}")
    return "\n".join(lines)


class ResearchPage(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        title = QLabel("ขั้นที่ 1 · ให้ AI รู้จักเกมก่อนแปล (ข้อมูลไม่ยืนยันจนกว่าคุณจะแก้)")
        title.setObjectName("muted")
        root.addWidget(title)

        game_box = QGroupBox("เกมที่จะแปล")
        form = QFormLayout(game_box)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("เช่น Atelier Ryza, Trails of Cold Steel")
        self.aliases_edit = QLineEdit()
        self.aliases_edit.setPlaceholderText("ชื่ออื่น/ชื่อญี่ปุ่น คั่นด้วยจุลภาค")
        self.auto_check = QCheckBox("ค้นคว้าอัตโนมัติตอนเริ่มแปล ถ้ายังไม่มีข้อมูลเกมนี้ "
                                    "(ส่งออกแค่ชื่อเกม ไม่ส่งภาพหรือข้อความบนจอ)")
        form.addRow("ชื่อเกม", self.title_edit)
        form.addRow("ชื่ออื่น", self.aliases_edit)
        form.addRow("", self.auto_check)
        root.addWidget(game_box)

        source_box = QGroupBox("แหล่งข้อมูลที่คุณเลือกเอง")
        source_layout = QVBoxLayout(source_box)
        source_layout.addWidget(QLabel("ลิงก์หน้าเว็บ (บรรทัดละ 1 ลิงก์ · HTTPS เท่านั้น)"))
        self.urls_edit = QPlainTextEdit()
        self.urls_edit.setMaximumHeight(72)
        self.urls_edit.setPlaceholderText("https://en.wikipedia.org/wiki/...")
        source_layout.addWidget(self.urls_edit)
        source_layout.addWidget(QLabel("หรือวางข้อความที่คัดลอกมาเอง (บทสรุปตัวละคร เนื้อเรื่อง ฯลฯ)"))
        self.paste_edit = QPlainTextEdit()
        self.paste_edit.setMaximumHeight(110)
        source_layout.addWidget(self.paste_edit)
        root.addWidget(source_box)

        search_box = QGroupBox("ค้นเว็บ (ใช้ key ของคุณ · เก็บใน Windows Credential Manager เท่านั้น)")
        search_form = QFormLayout(search_box)
        self.provider = QComboBox()
        for label, data in SEARCH_LABELS:
            self.provider.addItem(label, data)
        self.provider.currentIndexChanged.connect(self._provider_changed)
        self.endpoint = QLineEdit()
        self.search_key = QLineEdit()
        self.search_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.remember_key = QCheckBox("จำ key ของระบบค้นหา")
        self.max_sources = QSpinBox()
        self.max_sources.setRange(1, 10)
        search_form.addRow("ผู้ให้บริการ", self.provider)
        search_form.addRow("ปลายทาง", self.endpoint)
        search_form.addRow("API key", self.search_key)
        search_form.addRow("", self.remember_key)
        search_form.addRow("จำนวนผลลัพธ์สูงสุด", self.max_sources)
        root.addWidget(search_box)

        row = QHBoxLayout()
        self.run_button = QPushButton("เริ่มวิจัยเกม")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(lambda: self.session.research_start())
        self.save_button = QPushButton("บันทึกชื่อเกม/การตั้งค่า")
        self.save_button.clicked.connect(lambda: self.session.research_save())
        self.clear_button = QPushButton("ล้างข้อมูลวิจัย")
        self.clear_button.clicked.connect(lambda: self.session.research_clear())
        row.addWidget(self.run_button, 2)
        row.addWidget(self.save_button, 1)
        row.addWidget(self.clear_button, 1)
        root.addLayout(row)
        self.progress = QLabel("พร้อม — ยังไม่ได้ค้นคว้า")
        self.progress.setObjectName("status")
        self.progress.setWordWrap(True)
        root.addWidget(self.progress)

        split = QSplitter(Qt.Orientation.Horizontal)
        brief_box = QGroupBox("สรุปที่ AI รวบรวมได้")
        brief_layout = QVBoxLayout(brief_box)
        self.brief_view = QPlainTextEdit()
        self.brief_view.setReadOnly(True)
        brief_layout.addWidget(self.brief_view)
        split.addWidget(brief_box)
        override_box = QGroupBox("สิ่งที่คุณยืนยันเอง (สำคัญกว่าข้อมูลวิจัยและข้อสันนิษฐาน AI)")
        override_layout = QVBoxLayout(override_box)
        self.override_edit = QPlainTextEdit()
        self.override_edit.setPlaceholderText(
            "เช่น เกมนี้ใช้คำว่า 'เซน' ไม่ใช่ 'เซ็น' • เรย์ซ่าอายุ 17 เป็นลูกพี่ลูกน้องกับคิโร")
        override_layout.addWidget(self.override_edit)
        save_override = QPushButton("บันทึกโน้ตยืนยัน")
        save_override.clicked.connect(lambda: self.session.research_save_override())
        override_layout.addWidget(save_override)
        split.addWidget(override_box)
        split.setSizes([620, 420])
        root.addWidget(split, 1)

    def _provider_changed(self):
        provider = self.provider.currentData()
        self.endpoint.setText(DEFAULT_ENDPOINTS.get(provider, ""))
        self.endpoint.setEnabled(provider not in ("off",))
        self.search_key.setEnabled(provider not in ("off",))
        self.remember_key.setEnabled(provider not in ("off",))
        if provider == "off":
            self.endpoint.clear()

    def load(self, settings, brief: dict, sources: list, research_notes: str, game: str):
        self.title_edit.setText(settings.research_title or game)
        self.aliases_edit.setText(settings.research_aliases)
        self.auto_check.setChecked(settings.research_auto)
        self.provider.blockSignals(True)
        self.provider.setCurrentIndex(max(0, self.provider.findData(settings.research_search)))
        self.provider.blockSignals(False)
        self.endpoint.setText(settings.research_endpoint or DEFAULT_ENDPOINTS.get(settings.research_search, ""))
        self.endpoint.setEnabled(settings.research_search not in ("off",))
        self.max_sources.setValue(settings.research_max_sources)
        self.brief_view.setPlainText(render_brief(brief))
        self.override_edit.setPlainText(research_notes)
        if sources:
            self.progress.setText("แหล่งข้อมูลที่เคยดึง: " + ", ".join(
                str(item.get("title", item.get("url", "")))[:40] for item in sources[:6]
                if isinstance(item, dict)))

    def values(self) -> dict:
        urls = [line.strip() for line in self.urls_edit.toPlainText().splitlines() if line.strip()]
        return {
            "title": self.title_edit.text().strip()[:120],
            "aliases": self.aliases_edit.text().strip()[:200],
            "auto": self.auto_check.isChecked(),
            "urls": urls[:6],
            "paste": self.paste_edit.toPlainText()[:20000],
            "provider": self.provider.currentData(),
            "endpoint": self.endpoint.text().strip()[:300],
            "key": self.search_key.text().strip(),
            "remember_key": self.remember_key.isChecked(),
            "max_sources": self.max_sources.value(),
        }


class CharactersPage(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.selected_name = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.addWidget(self._build_header())
        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ชื่อ", "สถานะ", "บทบาท", "เพศ", "ใบหน้า", "เห็นล่าสุด"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._row_selected)
        left_layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        for label, callback in (("เพิ่มตัวละครเอง", lambda: self.session.character_add()),
                                ("ยืนยันตัวละครนี้", lambda: self.session.character_confirm(self.selected_name)),
                                ("ลบ", lambda: self.session.character_remove(self.selected_name)),
                                ("ล้างใบหน้า", lambda: self.session.character_forget_face(self.selected_name))):
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        left_layout.addLayout(buttons)

        face_box = QGroupBox("ใบหน้าในการสแกนล่าสุด (ภาพอยู่ใน RAM เท่านั้น ไม่บันทึกลงดิสก์)")
        face_layout = QVBoxLayout(face_box)
        scan_row = QHBoxLayout()
        scan = QPushButton("สแกนใบหน้าจากพื้นที่ที่เลือก")
        scan.clicked.connect(lambda: self.session.scan_faces())
        self.scan_note = QLabel("ตรวจด้วย " + self.session.face_backend_note())
        self.scan_note.setObjectName("muted")
        self.scan_note.setWordWrap(True)
        scan_row.addWidget(scan)
        scan_row.addWidget(self.scan_note, 1)
        face_layout.addLayout(scan_row)
        self.faces_list = QListWidget()
        self.faces_list.setIconSize(self.faces_list.iconSize() or Qt.QSize(108, 108))
        self.faces_list.setMinimumHeight(150)
        self.faces_list.currentRowChanged.connect(self._face_selected)
        face_layout.addWidget(self.faces_list, 1)
        bind_row = QHBoxLayout()
        self.bind_name = QComboBox()
        self.bind_name.setEditable(True)
        self.bind_name.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        bind = QPushButton("ผูกชื่อนี้กับใบหน้าที่เลือก (ยืนยันโดยผู้ใช้)")
        bind.clicked.connect(lambda: self.session.bind_face())
        bind_row.addWidget(QLabel("ชื่อ"))
        bind_row.addWidget(self.bind_name, 1)
        bind_row.addWidget(bind, 2)
        face_layout.addLayout(bind_row)
        left_layout.addWidget(face_box, 2)
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.card_box = QGroupBox("การ์ดการพูดของตัวละครที่เลือก")
        card_form = QFormLayout(self.card_box)
        self.card_name = QLabel("—")
        self.gender = QComboBox()
        for label, data in (("ไม่ทราบ (ห้ามเดา ครับ/ค่ะ)", "unknown"), ("ชาย (ยืนยัน)", "male"),
                            ("หญิง (ยืนยัน)", "female")):
            self.gender.addItem(label, data)
        self.age_gap = QComboBox()
        for label, data in (("ไม่ทราบ (ไม่ฟันธงอาวุโส)", ""), ("อาวุโสกว่าคู่สนทนา", "older"),
                            ("อ่อนกว่าคู่สนทนา", "younger"), ("รุ่นเดียวกัน", "same")):
            self.age_gap.addItem(label, data)
        self.relation = QLineEdit()
        self.relation.setPlaceholderText("เช่น พี่น้อง / ครูกับนักเรียน / คู่รัก / ศัตรู")
        self.voice_self = QLineEdit()
        self.voice_address = QLineEdit()
        self.voice_particles = QLineEdit()
        self.voice_register = QLineEdit()
        self.voice_quirks = QLineEdit()
        self.voice_lock = QCheckBox("ล็อกการ์ดนี้: AI ต้องใช้ค่าข้างบนนี้และห้ามเปลี่ยน")
        for label, widget in (("ชื่อ", self.card_name), ("เพศ", self.gender),
                              ("อาวุโสเทียบคู่สนทนา", self.age_gap), ("ความสัมพันธ์", self.relation),
                              ("สรรพนามแทนตัวเอง (ผม/ฉัน/หนู/พี่/ข้า)", self.voice_self),
                              ("สรรพนามเรียกอีกฝ่าย (คุณ/เธอ/พี่/น้อง/ท่าน)", self.voice_address),
                              ("คำลงท้าย (ครับ/ค่ะ/นะ/สิ/เหอะ/วะ)", self.voice_particles),
                              ("ระดับภาษา", self.voice_register),
                              ("เอกลักษณ์การพูด", self.voice_quirks)):
            card_form.addRow(label, widget)
        card_form.addRow("", self.voice_lock)
        save_card = QPushButton("บันทึกการ์ดการพูด")
        save_card.clicked.connect(lambda: self.session.save_character(self.selected_name))
        card_form.addRow(save_card)
        right_layout.addWidget(self.card_box)

        pending_box = QGroupBox("ข้อเสนอผูกชื่อจาก AI · ต้องได้รับการยืนยันจากคุณก่อนจึงจะถือว่าเป็นชื่อจริง")
        pending_layout = QVBoxLayout(pending_box)
        self.pending_list = QListWidget()
        self.pending_list.setMinimumHeight(110)
        pending_layout.addWidget(self.pending_list)
        pending_row = QHBoxLayout()
        confirm = QPushButton("ยืนยันข้อเสนอที่เลือก")
        confirm.clicked.connect(lambda: self.session.confirm_pending(self.pending_list.currentRow()))
        reject = QPushButton("ปฏิเสธ")
        reject.clicked.connect(lambda: self.session.reject_pending(self.pending_list.currentRow()))
        pending_row.addWidget(confirm)
        pending_row.addWidget(reject)
        pending_row.addStretch()
        pending_layout.addLayout(pending_row)
        right_layout.addWidget(pending_box, 1)
        suggest_box = QGroupBox("คำศัพท์/ชื่อที่ AI เสนอจากข้อความที่ซ้ำ · ยังไม่ถูกใช้จนคุณกดรับ")
        suggest_layout = QVBoxLayout(suggest_box)
        self.suggest_list = QListWidget()
        self.suggest_list.setMinimumHeight(96)
        suggest_layout.addWidget(self.suggest_list)
        suggest_row = QHBoxLayout()
        accept = QPushButton("เพิ่มที่เลือกเข้า glossary")
        accept.clicked.connect(lambda: self.session.accept_suggestions(self._selected_suggestions()))
        dismiss = QPushButton("ไม่ใช้ที่เลือก")
        dismiss.clicked.connect(lambda: self.session.dismiss_suggestions(self._selected_suggestions()))
        suggest_row.addWidget(accept)
        suggest_row.addWidget(dismiss)
        suggest_row.addStretch()
        suggest_layout.addLayout(suggest_row)
        right_layout.addWidget(suggest_box)

        hint = QLabel("ระบบนี้ไม่สอนว่า 'เห็นหน้า = กำลังพูด' และไม่เดาเพศจากหน้าตา: "
                      "ชื่อจะผูกกับใบหน้าก็ต่อเมื่อผู้ใช้ยืนยัน และสีหน้าเป็นค่าที่วัดจากภาพนิ่ง "
                      "ไม่ใช่ผลตรวจอารมณ์ที่แม่นยำ")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        right_layout.addWidget(hint)
        split.addWidget(right)
        split.setSizes([520, 520])
        root.addWidget(split, 1)
        self._scan_rows = []

    def _build_header(self):
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("ตัวละคร · ใบหน้า · วิธีพูด")
        layout.addWidget(label)
        layout.addStretch()
        self.summary = QLabel("")
        self.summary.setObjectName("muted")
        layout.addWidget(self.summary)
        return header

    # --- table -------------------------------------------------------------------------

    def refresh(self, memory, faces, scan_image):
        self._scan_rows = list(faces or [])
        characters = memory.characters if memory else {}
        names = list(characters)
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            card = characters[name]
            face_count = len(card.get("descriptors", []))
            items = [name,
                     "ยืนยันโดยผู้ใช้" if card.get("confirmed") else "ข้อสันนิษฐาน AI",
                     card.get("role", "") or card.get("relation", ""),
                     {"male": "ชาย (ยืนยัน)", "female": "หญิง (ยืนยัน)"}.get(
                         card.get("gender", "unknown"),
                         "ยังไม่ยืนยัน" if card.get("gender", "unknown") != "unknown" else "ไม่ทราบ"),
                     f"{face_count} ตัวอย่าง",
                     card.get("last_seen", "")]
            for column, text in enumerate(items):
                cell = QTableWidgetItem(str(text))
                if column == 0 and card.get("confirmed"):
                    cell.setToolTip("ผู้ใช้ยืนยันแล้ว: AI ห้ามเปลี่ยนชื่อนี้")
                self.table.setItem(row, column, cell)
        self.summary.setText(f"{len(names)} ตัวละคร · "
                             f"{sum(1 for c in characters.values() if c.get('confirmed'))} ยืนยันแล้ว · "
                             f"{sum(len(c.get('descriptors', [])) for c in characters.values())} ตัวอย่างใบหน้า")

        self.bind_name.clear()
        self.bind_name.addItems(names)

        self.pending_list.clear()
        for item in memory.pending if memory else []:
            self.pending_list.addItem(
                f"{item['track']} → {item['character']} · หลักฐาน: {item['evidence'][:80]} "
                f"({item['confidence']:.2f})")

        self.faces_list.clear()
        for observation in self._scan_rows:
            pixmap = pil_to_pixmap(scan_image, (observation.box.x, observation.box.y,
                                                observation.box.w, observation.box.h))
            name = observation.character or observation.possible or f"ใบหน้าที่ {observation.track_id}"
            if observation.possible and not observation.character:
                name += " (ยังไม่ยืนยัน)"
            text = f"{name} · {observation.box.w}×{observation.box.h}px\n" \
                   f"สีหน้า: {observation.cues.label} [{observation.cues.summary() or 'วัดไม่ได้'}]\n" \
                   f"{observation.appearance.summary()}"
            item = QListWidgetItem(text)
            item.setIcon(pixmap)
            self.faces_list.addItem(item)

    def refresh_suggestions(self, suggestions):
        """Show mined terms with their evidence. Nothing here is used until the user accepts."""
        self.suggest_list.clear()
        for item in suggestions or []:
            if not isinstance(item, dict) or not item.get("source"):
                continue
            evidence = (item.get("evidence") or [""])[0]
            self.suggest_list.addItem(
                f"{item['source']} → {item.get('thai', '')} "
                f"(พบ {item.get('count', 1)} ครั้ง · {item.get('confidence', 0):.2f}) · {evidence[:70]}")

    def _selected_suggestions(self):
        return [index.row() for index in self.suggest_list.selectedIndexes()] or (
            [self.suggest_list.currentRow()] if self.suggest_list.currentRow() >= 0 else [])

    def _row_selected(self):
        row = self.table.currentRow()
        if row < 0 or row >= self.table.rowCount():
            self.selected_name = ""
            self.card_name.setText("—")
            return
        self.selected_name = self.table.item(row, 0).text()
        self.session.load_character_card(self.selected_name)

    def _face_selected(self, row: int):
        if 0 <= row < len(self._scan_rows):
            observation = self._scan_rows[row]
            if observation.possible and not observation.character:
                self.bind_name.setCurrentText(observation.possible)

    def set_card(self, name: str, card: dict):
        self.card_name.setText(name or "—")
        voice = card.get("voice") or {}
        self.gender.setCurrentIndex(max(0, self.gender.findData(card.get("gender", "unknown"))))
        self.age_gap.setCurrentIndex(max(0, self.age_gap.findData(card.get("age_gap", ""))))
        self.relation.setText(card.get("relation", ""))
        self.voice_self.setText(voice.get("self", ""))
        self.voice_address.setText(voice.get("address", ""))
        self.voice_particles.setText(voice.get("particles", ""))
        self.voice_register.setText(voice.get("register", ""))
        self.voice_quirks.setText(voice.get("quirks", ""))
        self.voice_lock.setChecked(bool(voice.get("locked")))

    def card_values(self) -> dict:
        return {
            "gender": self.gender.currentData(),
            "age_gap": self.age_gap.currentData(),
            "relation": self.relation.text().strip(),
            "self": self.voice_self.text().strip(),
            "address": self.voice_address.text().strip(),
            "particles": self.voice_particles.text().strip(),
            "register": self.voice_register.text().strip(),
            "quirks": self.voice_quirks.text().strip(),
            "locked": self.voice_lock.isChecked(),
        }

    def selected_face_index(self) -> int:
        return self.faces_list.currentRow()

    def bind_target_name(self) -> str:
        return self.bind_name.currentText().strip()


class LocalAIPage(QWidget):
    """Install llama.cpp, download GGUF models and run them — all without leaving the app."""

    def __init__(self, session):
        super().__init__()
        self.session = session
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        title = QLabel("Local AI ในเครื่อง · llama.cpp (ดาวน์โหลด/รันได้จากที่นี่ ไม่ต้องติดตั้ง Ollama)")
        title.setObjectName("muted")
        root.addWidget(title)

        # --- step 1: the server binary
        binary_box = QGroupBox("ขั้นที่ 1 · ตัวรัน llama-server")
        binary_layout = QFormLayout(binary_box)
        row = QHBoxLayout()
        self.binary_edit = QLineEdit()
        self.binary_edit.setPlaceholderText("ยังไม่มีไฟล์ — กด ‘เลือกไฟล์’ หรือ ‘ดึงรายการจาก GitHub’")
        browse = QPushButton("เลือกไฟล์…")
        browse.clicked.connect(lambda: self.session.pick_llama_binary())
        row.addWidget(self.binary_edit, 1)
        row.addWidget(browse)
        binary_layout.addRow("ไฟล์ llama-server", row)
        fetch_row = QHBoxLayout()
        self.asset_combo = QComboBox()
        self.asset_combo.setMinimumWidth(320)
        self.asset_combo.addItem("— กดปุ่มขวาเพื่อดึงรายการจาก GitHub releases —", "")
        self.fetch_assets = QPushButton("ดึงรายการจาก GitHub")
        self.fetch_assets.clicked.connect(lambda: self.session.fetch_llama_assets())
        self.install_binary = QPushButton("ดาวน์โหลด + ติดตั้งที่เลือก")
        self.install_binary.clicked.connect(lambda: self.session.install_llama_asset(
            self.asset_combo.currentData()))
        fetch_row.addWidget(self.asset_combo, 1)
        fetch_row.addWidget(self.fetch_assets)
        fetch_row.addWidget(self.install_binary)
        binary_layout.addRow("รุ่นที่พบ", fetch_row)
        root.addWidget(binary_box)

        # --- step 2: the model
        model_box = QGroupBox("ขั้นที่ 2 · โมเดล GGUF (จาก Hugging Face)")
        model_layout = QVBoxLayout(model_box)
        model_layout.addWidget(QLabel(TIP))
        pick = QHBoxLayout()
        self.catalog_combo = QComboBox()
        for entry in CATALOG:
            self.catalog_combo.addItem(entry["label"], entry["id"])
        self.catalog_combo.currentIndexChanged.connect(self._catalog_changed)
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(QUANTS)
        self.repo_edit = QLineEdit()
        self.repo_edit.setPlaceholderText("หรือพิมพ์ repo เอง เช่น bartowski/Qwen2.5-7B-Instruct-GGUF")
        self.list_button = QPushButton("ดูไฟล์ใน repo")
        self.list_button.clicked.connect(lambda: self.session.list_llama_files(
            self.repo_edit.text().strip() or self._catalog_field("repo")))
        pick.addWidget(QLabel("โมเดลแนะนำ"))
        pick.addWidget(self.catalog_combo, 2)
        pick.addWidget(QLabel("quant"))
        pick.addWidget(self.quant_combo, 1)
        model_layout.addLayout(pick)
        repo_row = QHBoxLayout()
        repo_row.addWidget(self.repo_edit, 1)
        repo_row.addWidget(self.list_button)
        model_layout.addLayout(repo_row)
        self.file_table = QTableWidget(0, 4)
        self.file_table.setHorizontalHeaderLabels(["ไฟล์", "ขนาด (GB)", "ชนิด", "สถานะ"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setMinimumHeight(150)
        model_layout.addWidget(self.file_table)
        download_row = QHBoxLayout()
        self.download_button = QPushButton("ดาวน์โหลดไฟล์ที่เลือก")
        self.download_button.clicked.connect(lambda: self.session.download_llama_selection(
            self.file_table.currentRow()))
        self.download_recommended = QPushButton("ดาวน์โหลดชุดที่แนะนำ (โมเดล + mmproj)")
        self.download_recommended.clicked.connect(lambda: self.session.download_llama_recommended())
        self.cancel_button = QPushButton("ยกเลิกการดาวน์โหลด")
        self.cancel_button.clicked.connect(lambda: self.session.cancel_downloads())
        self.cancel_button.setEnabled(False)
        download_row.addWidget(self.download_button, 1)
        download_row.addWidget(self.download_recommended, 2)
        download_row.addWidget(self.cancel_button, 1)
        model_layout.addLayout(download_row)
        self.download_note = QLabel("ยังไม่เริ่มดาวน์โหลด • ทุกอย่างเป็น HTTPS และหยุด/เรียนต่อได้")
        self.download_note.setObjectName("muted")
        self.download_note.setWordWrap(True)
        model_layout.addWidget(self.download_note)
        root.addWidget(model_box)

        # --- step 3: run it
        run_box = QGroupBox("ขั้นที่ 3 · เปิดใช้เป็น AI ของแอป")
        run_layout = QVBoxLayout(run_box)
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        reload_models = QPushButton("สแกนไฟล์ในเครื่อง")
        reload_models.clicked.connect(lambda: self.session.refresh_llama_models())
        model_row.addWidget(QLabel("โมเดลในเครื่อง"))
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(reload_models)
        run_layout.addLayout(model_row)
        settings_row = QHBoxLayout()
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.ctx = QSpinBox()
        self.ctx.setRange(2048, 131072)
        self.ctx.setSingleStep(2048)
        self.gpu = QSpinBox()
        self.gpu.setRange(0, 999)
        self.gpu.setToolTip("0 = ใช้ CPU ล้วน • เพิ่มตาม VRAM ที่เหลือ")
        for label, widget in (("พอร์ต", self.port), ("ctx", self.ctx), ("n-gpu-layers", self.gpu)):
            settings_row.addWidget(QLabel(label))
            settings_row.addWidget(widget)
        run_layout.addLayout(settings_row)
        control_row = QHBoxLayout()
        self.start_button = QPushButton("เริ่ม llama-server")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(lambda: self.session.start_llama())
        self.stop_button = QPushButton("หยุด")
        self.stop_button.clicked.connect(lambda: self.session.stop_llama())
        self.use_button = QPushButton("ตั้งเป็น AI ของแอป (OpenAI-compatible)")
        self.use_button.clicked.connect(lambda: self.session.use_llama_endpoint())
        control_row.addWidget(self.start_button, 2)
        control_row.addWidget(self.stop_button, 1)
        control_row.addWidget(self.use_button, 2)
        run_layout.addLayout(control_row)
        self.status_label = QLabel("ยังไม่ทำงาน")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        run_layout.addWidget(self.status_label)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.setPlaceholderText("log ของ llama-server จะแสดงที่นี่")
        run_layout.addWidget(self.log_view)
        note = QLabel("เซิร์ฟเวอร์ผูกกับ 127.0.0.1 เท่านั้น (ไม่เปิดสู่วงนอก) และปิดเองเมื่อปิดแอป • "
                      "โมเดลที่โหลดแล้วเป็นไฟล์ในเครื่องคุณ ใช้ซ้ำได้ไม่ต้องโหลดใหม่ • "
                      "ถ้าไม่กด ‘ตั้งเป็น AI ของแอป’ การตั้งค่าเดิม (Ollama/ภายนอก) จะไม่เปลี่ยน")
        note.setObjectName("muted")
        note.setWordWrap(True)
        run_layout.addWidget(note)
        root.addWidget(run_box)
        root.addStretch()

    # --- helpers -----------------------------------------------------------------------
    def _catalog_field(self, field: str):
        for entry in CATALOG:
            if entry["id"] == self.catalog_combo.currentData():
                return entry.get(field, "")
        return ""

    def _catalog_changed(self):
        entry = next((item for item in CATALOG if item["id"] == self.catalog_combo.currentData()), {})
        if entry.get("quant"):
            self.quant_combo.setCurrentText(entry["quant"])
        self.repo_edit.setText(entry.get("repo", ""))

    def _model_changed(self):
        entry = self.model_combo.currentData() or {}
        if not entry:
            return
        rmem = entry.get("ram_gb")
        if rmem:
            self.session.llama_note(f"โมเดลนี้เหมาะกับ RAM ประมาณ {rmem} GB ขึ้นไป")

    def load(self, settings):
        self.binary_edit.setText(settings.llama_binary)
        self.port.setValue(settings.llama_port)
        self.ctx.setValue(settings.llama_ctx)
        self.gpu.setValue(settings.llama_gpu_layers)

    def values(self) -> dict:
        return {"binary": self.binary_edit.text().strip()[:400], "port": self.port.value(),
                "ctx": self.ctx.value(), "gpu": self.gpu.value()}

    def show_files(self, files, repo: str):
        self.file_table.setRowCount(0)
        for entry in files:
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            size = entry["size"] / (1 << 30)
            cells = (entry["name"], f"{size:.2f}" if entry["size"] else "?",
                     "vision projector" if entry["mmproj"] else "โมเดล", "ยังไม่มี")
            for column, text in enumerate(cells):
                self.file_table.setItem(row, column, QTableWidgetItem(text))
        self.download_note.setText(f"พบ {len(files)} ไฟล์ใน {repo} • เลือกแถวแล้วกดดาวน์โหลด "
                                   "(หรือใช้ชุดที่แนะนำ)")

    def mark_file(self, name: str, status: str):
        for row in range(self.file_table.rowCount()):
            item = self.file_table.item(row, 0)
            cell = self.file_table.item(row, 3)
            if item is not None and cell is not None and item.text() == name:
                cell.setText(status)

    def ready_file_names(self) -> list[str]:
        return [self.file_table.item(row, 0).text() for row in range(self.file_table.rowCount())
                if self.file_table.item(row, 3) and
                self.file_table.item(row, 3).text().startswith("เสร็จ")]

    def show_models(self, models):
        current = self.model_combo.currentData()
        self.model_combo.clear()
        if not models:
            self.model_combo.addItem("— ยังไม่มีโมเดลในเครื่อง กดดาวน์โหลดด้านบน —", {})
            return
        for entry in models:
            label = f"{entry['name']} ({entry['size'] / (1 << 30):.1f} GB)" + \
                (" · มี mmproj" if entry.get("projector") else "")
            self.model_combo.addItem(label, entry)
        if current:
            for index in range(self.model_combo.count()):
                if self.model_combo.itemData(index).get("path") == current.get("path"):
                    self.model_combo.setCurrentIndex(index)
                    break


def ask_name(parent, title: str, label: str, default: str = "") -> str:
    from PySide6.QtWidgets import QInputDialog
    text, ok = QInputDialog.getText(parent, title, label, text=default)
    return text.strip() if ok else ""


def warning(parent, title: str, message: str):
    QMessageBox.warning(parent, title, message)


def information(parent, title: str, message: str):
    QMessageBox.information(parent, title, message)


__all__ = ["ResearchPage", "CharactersPage", "LocalAIPage", "render_brief", "pil_to_pixmap", "ask_name",
           "warning", "information", "SEARCH_PROVIDERS"]
