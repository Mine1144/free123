# การตรวจสอบและข้อจำกัดการทดสอบ

## ผลที่รันจริงใน sandbox นี้

- Python 3.11.2 / Linux, PySide6 6.11.2, mediapipe 0.10.21, opencv-python-headless 4.11.0, onnxruntime 1.30.0
- `pytest -q`: **157 passed, 1 skipped** จาก 158 เทสต์ใน 14 ไฟล์ (`tests/test_*.py`)
  - เทสต์ที่ถูกข้ามคือ `tests/test_vision.py::test_real_photo_face_analysis_if_available` ซึ่งต้องตั้ง `SCREEN_THAI_FACE_FIXTURE=<path รูปใบหน้า>`; เมื่อตั้งค่าแล้วรันผ่าน 12/12 ในไฟล์นั้น
- `ruff check .`: ผ่าน
- `python -m compileall -q screen_thai`: ผ่าน
- `python scripts/smoke-ocr.py`: **ผ่าน** ด้วยโมเดล RapidOCR จริงบนภาพสังเคราะห์ "Hello world" (sandbox venv ใช้ `opencv-python-headless` เพื่อหลีกเลี่ยง libGL ที่ไม่มี; Windows ยังใช้ opencv ปกติ)
- ตรวจใบหน้าจริงด้วย mediapipe: ภาพบุคคล 2 ภาพ → ตรวจเจอใบหน้า, วัดค่าสีหน้า, สร้าง descriptor (172 มิติ)
  - เฟรมเดิมเลื่อน 6 พิกเซล: cosine 0.999 → `FaceMemory` คืนชื่อที่ผู้ใช้ยืนยันแล้ว
  - คนละภาพ: cosine 0.842 → ได้เพียงสถานะ "อาจเป็น" ระบบไม่ผูกชื่อให้เอง
- Provider tests ใช้ `httpx.MockTransport` ตรวจ payload/response/errors **ไม่ได้เรียกโมเดลหรือใช้ key จริง** และไม่ได้ทดสอบ search API จริง
- ยังไม่ได้รัน build `.exe`, Windows native UI, เกมจริง, mixed-DPI capture, Credential Manager จริง หรือ external/local inference end-to-end

### ฟีเจอร์ชุด "คุณภาพการแปล" ที่ตรวจแล้ว

- **หน่วยความจำคำแปล**: เรียน/ใช้ซ้ำ/นับครั้ง, ไม่ใช้บรรทัดที่ขึ้นเตือน, ไม่ใช้เมื่อการ์ดเสียงเปลี่ยน, ไม่ใช้เมื่อไม่รู้ว่าผู้พูดเป็นใคร, ข้ามป้ายชื่อ/HUD, เพดาน 800 รายการ, ทนข้อมูลเสียในโปรไฟล์, round-trip ผ่าน Store
- **end-to-end**: เทสต์จำลองรอบจริง — แปลครั้งแรกเรียก AI 1 ครั้ง, เฟรมเดิมครั้งถัดไป **ไม่เรียก AI เลย** และข้อความที่แคชไว้ไม่ถูกลบเมื่อมีบรรทัดใหม่ถูกส่งไปพร้อมกัน (ตรวจว่า payload ส่งเฉพาะ id ที่ยังไม่มีคำตอบ)
- **ผสานบรรทัดซับ**: รวม 2–3 บรรทัดของผู้พูดเดียวกัน, ไม่รวมเมื่อจบประโยค/ไกลกัน/ต่างบทบาท/มีป้ายชื่อใหม่คั่น, และเมนูตัวเลือกไม่ถูกรวมหลังปรับ `layout` ให้แยกบรรทัดที่ตัดคำออกจากเมนู
- **ขุดคำศัพท์**: ต้องซ้ำ ≥2 บรรทัดและคำไทยที่เสนอต้องไม่โผล่ในบรรทัดที่ไม่มีคำนั้น; ข้ามคำที่อยู่ใน glossary แล้ว; ผู้ใช้ต้องกดรับก่อนจึงถูกใช้ (มีเทสต์ยืนยันว่า glossary ว่างจนกว่าจะกด)
- **รอบตรวจความสม่ำเสมอ**: แก้เฉพาะ id ที่ตั้งธง, ปฏิเสธ id ปลอม, ใส่เหตุผลใน warnings, ข้ามเมื่อไม่มีอะไรผิด, และคำแปลเดิมรอดเมื่อรอบตรวจล้มเหลว
- **ตรวจสุขภาพระบบ**: ทุกเช็กกำหนดสถานะ ok/warn/fail/skip พร้อมเหตุผลถัดไป, URL ผิดต้อง fail ไม่ใช่ throw, โฟลเดอร์เขียนไม่ได้ต้อง fail, และการตรวจฟอนต์จะข้ามเองเมื่อไม่มี QApplication

### ข้อจำกัดของ sandbox ที่ต้องเข้าใจ

- โมดูล UI รันได้เพราะติดตั้ง PySide6 และใช้ shim library เฉพาะกิจใน sandbox (libGL/libEGL/libxcb) ซึ่ง**ไม่ได้อยู่ในรีโป**; บน Windows ที่ติดตั้ง `.[dev]` โมดูลนี้รันได้ตามปกติ
- ไม่สามารถติดตั้งฟอนต์ไทย (`fonts-thai-tlwg`) ใน sandbox ได้ (ไม่มี apt/curl ไป mirror) การทดสอบข้อความไทยจึงยืนยัน **ตรรกะ จำนวน ตัวเลข และการไม่ crash** ไม่ได้ยืนยันรูปร่าง glyph ที่เรนเดอร์จริง (อาจเป็น tofu ใน offscreen)
- ไม่มีการเชื่อมต่อเครือข่ายในเทสต์ ยกเว้นการติดตั้งแพ็กเกจ

## Automated coverage

- **models**: JSON ผิดรูปแบบ, ID ปลอม/ซ้ำ/boolean, ขนาดใหญ่, ความสัมพันธ์ไม่มีหลักฐาน, ค่า settings ผิด (face_backend/face_sample_ms/max_faces/research_search/research_max_sources/show_speaker_label) ถูก clamp/ปฏิเสธ
- **state**: occurrence matching — พิกัดสั่น, ID reorder, ข้อความหายแล้วกลับมา, HUD เปลี่ยนตลอด, กล่องเหมือนกันแต่คำแปลต่างกัน, ล้างข้อความเมื่อไม่มี OCR
- **vision**: คณิตศาสตร์การวัดสีหน้าจาก landmark สังเคราะห์ (ยิ้ม/อ้าปาก/กะพริบ → ค่าเปลี่ยนตามจริง), `regions` fallback, descriptor ปกติ/กล่องเล็ก, การสุ่มสี appearance, `build_detector` (off/fallback/mediapipe), NullDetector, และเส้นทาง "งานไม่ทับกัน" (busy) — รวมเทสต์ภาพจริงแบบ opt-in
- **faces**: tracker IoU/ttl, threshold ยืนยันชื่อ (0.86 + margin) vs คำแนะนำ (0.78), unknown ไม่ถูกตั้งชื่อ, pending confirm/reject, alias/fuzzy resolve, round-trip JSON + ขนาดคลังมีเพดาน, trend จากค่าที่วัด, การล้างใบหน้าโดยไม่ลบการ์ด
- **layout**: บทบาทจากเรขาคณิต (name tag / dialogue / subtitle / choice / HUD), ผู้พูดจากป้ายชื่อที่เห็นจริง, ลำดับอ่าน, บรรทัดไทยยาวไม่ถูกตีเป็นชื่อ, และ subtitle ต้องอยู่กลางจอตามเกณฑ์
- **speech**: คำสุภาพ/คำเรียกญี่ปุ่น (先輩 → รุ่นพี่), ร่องรอยเพศจากถ้อยคำ, register หยาบ (วะ), พี่–น้องโดยไม่ยืนยันอาวุโส, locked voice ชนะ preset, คำเตือน QC, relation_type, detect_language
- **research**: html→text, URL ที่ไม่ปลอดภัย 7 แบบถูกปฏิเสธ, redirect/content-type, การใช้ key ของ SearXNG/Brave/Serper/Tavily, sanitize query, parse/merge brief, พรอมป์ต์เป็น text-only (ไม่ส่งภาพ)
- **context**: ป้ายระดับความเชื่อถือ (ผู้ใช้ยืนยัน/ยังไม่ยืนยัน/วัดได้), locked plan + layout/name tag, ใบหน้าที่ไม่ยืนยันห้ามถูกกล่าวชื่อ, เตือนเมื่อมีใบหน้าแต่ไม่มีป้ายชื่อ, `trim` บังคับเพดานขนาดจริง
- **session flow**: หนึ่งรอบเต็มแบบ end-to-end (OCR → vision → AI → overlay/ตาราง/โปรไฟล์) ด้วย `httpx.MockTransport` ตรวจว่า payload มี `faces`/`speech_plans`/ป้ายชื่อ และผลลัพธ์ไหลถึงตาราง, dialogue, store และ overlay จริง
- **UI smoke**: สร้างหน้าต่าง 6 แท็บ (วิจัยเกม / ตัวละคร·ใบหน้า·วิธีพูด / แปลหน้าจอ / บริบทและตัวละคร / ประวัติ / ตั้งค่า), render, และปิด worker threads อย่างปลอดภัย (รวมกรณีปิดระหว่าง worker ยังทำงาน)

## Windows acceptance checklist ก่อนเรียกว่า production-ready

### การติดตั้ง
- [ ] Windows 10 22H2 / Windows 11 x64 สะอาด รัน launcher จาก path ภาษาไทย/มีช่องว่าง
- [ ] `pip install -e ".[dev,faces]"` แล้ว `mediapipe` import ได้; ถอน `faces` ออกแล้วแอปยังทำงานโดยลดระดับเป็น OpenCV
- [ ] ทดสอบ portable folder จาก artifact บนเครื่อง **ไม่มี Python** และเปิด OCR/ใบหน้าได้
- [ ] Credential Manager save/load/delete; ไม่พบ key (ทั้ง key แปลและ key ค้นเว็บ) ใน settings, log หรือ error
- [ ] โมเดล OCR/face mesh load จาก packaged `_internal` ได้โดยไม่พึ่ง source checkout

### ใบหน้า / สีหน้า / ตัวละคร
- [ ] สแกนใบหน้าในเกมการ์ตูน 3D, เซลเฉด, และภาพถ่ายสมจริง — วัดค่าได้โดยไม่ crash และไม่ตั้งชื่อผิดโดยไม่ยืนยัน
- [ ] ผูกชื่อ → ปิด/เปิดแอป → descriptor ยังอยู่และ match ได้; *ล้างใบหน้า* แล้วไม่ match อีก
- [ ] ข้อเสนอผูกชื่อต้องไม่กลายเป็นชื่อจริงจนกดยืนยัน; ปฏิเสธแล้วไม่กลับมาซ้ำ
- [ ] Trend/การขยับปาก: ตัวละครที่กำลังพูดถูกบันทึกว่า "ปากขยับ" โดยไม่ทำให้ใบหน้านิ่งขึ้นป้าย
- [ ] การ์ดที่ล็อกไว้: แปล 10 บรรทัดติดกันแล้วสรรพนาม/คำลงท้ายคงเดิม

### วิจัยเกม / ความเป็นส่วนตัว
- [ ] ลิงก์ HTTPS ที่ให้เอง, ข้อความวางเอง, และ provider จริง (SearXNG ในเครื่อง + เจ้าอื่นอย่างน้อยหนึ่งราย)
- [ ] กล่องยินยอมแสดงปลายทางและระบุว่าไม่ส่งภาพ/ข้อความบนจอ และCancel ไม่ทำให้เกิดคำขอ
- [ ] สลับโปรไฟล์เกมแล้ว brief/การ์ด/ใบหน้าไม่ข้ามกัน; brief ติดป้าย "ยังไม่ยืนยัน" จนกว่าจะแก้
- [ ] ตรวจว่า URL ภายใน/`file://`/redirect ไป localhost ถูกปฏิเสธ (และไม่มีการ log เนื้อหาเว็บ)

### Overlay / Capture / AI
- [ ] Borderless เกมและหน้าต่างทั่วไป: คำแปลตรงตำแหน่ง คลิก/เลื่อนเมาส์ทะลุ
- [ ] DPI 100%, 125%, 150%, 200%; สองจอ scaling ต่างกันและจอซ้าย origin ติดลบ
- [ ] Safe capture ไม่อ่านคำแปลตัวเองซ้ำ; exclusion mode ให้ภาพไม่มี overlay และไม่เป็นแผ่นดำ
- [ ] ภาษาไทยยาว สระ/วรรณยุกต์ ไม่ถูกตัดผิด และตรวจกล่องชนกัน
- [ ] Ollama Vision / text-only; ปิด vision แล้ว request ต้องไม่มีภาพ; ภายนอกต้องถามยินยอมก่อนส่งจริง
- [ ] Invalid key, rate-limit, wrong model, malformed JSON, offline, timeout → ข้อความแก้ปัญหาและหยุด (ไม่มี retry อัตโนมัติ)
- [ ] ปิดแอปกลาง request/กลางสแกนใบหน้า → รอ worker อย่างปลอดภัย ไม่มี QThread destroyed crash
- [ ] วัด OCR/inference latency และ VRAM ขณะเล่นเกมบนเครื่องเป้าหมาย ไม่รายงาน FPS จาก timer เป็น FPS การแปล
