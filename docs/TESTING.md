# การตรวจสอบและข้อจำกัดการทดสอบ

## ผลที่รันจริงใน sandbox นี้

- Python 3.11.2 / Linux, PySide6 6.11.2, mediapipe 0.10.21, opencv-python-headless 4.11.0, onnxruntime 1.30.0
- `pytest -q`: **199 passed, 1 skipped** จาก 200 เทสต์ใน 15 ไฟล์ (`tests/test_*.py`)
  - เทสต์ที่ถูกข้ามคือ `tests/test_vision.py::test_real_photo_face_analysis_if_available` ซึ่งต้องตั้ง `SCREEN_THAI_FACE_FIXTURE=<path รูปใบหน้า>`; เมื่อตั้งค่าแล้วรันผ่าน 12/12 ในไฟล์นั้น
- `ruff check .`: ผ่าน
- `python -m compileall -q screen_thai`: ผ่าน
- `python scripts/smoke-ocr.py`: **ผ่าน** ด้วยโมเดล RapidOCR จริงบนภาพสังเคราะห์ "Hello world" (sandbox venv ใช้ `opencv-python-headless` เพื่อหลีกเลี่ยง libGL ที่ไม่มี; Windows ยังใช้ opencv ปกติ)
- ตรวจใบหน้าจริงด้วย mediapipe: ภาพบุคคล 2 ภาพ → ตรวจเจอใบหน้า, วัดค่าสีหน้า, สร้าง descriptor (172 มิติ)
  - เฟรมเดิมเลื่อน 6 พิกเซล: cosine 0.999 → `FaceMemory` คืนชื่อที่ผู้ใช้ยืนยันแล้ว
  - คนละภาพ: cosine 0.842 → ได้เพียงสถานะ "อาจเป็น" ระบบไม่ผูกชื่อให้เอง
- Provider tests ใช้ `httpx.MockTransport` ตรวจ payload/response/errors **ไม่ได้เรียกโมเดลหรือใช้ key จริง** และไม่ได้ทดสอบ search API จริง
- ยังไม่ได้รัน build `.exe`, Windows native UI, เกมจริง, mixed-DPI capture, Credential Manager จริง หรือ external/local inference end-to-end

### ฟีเจอร์ชุด "Local AI แบบ llama.cpp" ที่ตรวจแล้ว (37 เทสต์)

- **ความปลอดภัยของ URL/ชื่อไฟล์**: อนุญาตเฉพาะ HTTPS บนโฮสต์ที่กำหนด; ปฏิเสธ `http://`, โฮสต์อื่น, userinfo (`https://huggingface.co@evil.com`), query/fragment; `validate_repo`/`validate_filename` กัน `..`, path สัมบูรณ์ และนามสกุลที่ไม่ใช่ `.gguf`; zip ที่มี path อันตรายถูกข้าม
- **Hugging Face**: อ่านรายชื่อไฟล์ + ขนาด, รู้จักไฟล์ `mmproj`, เลือก quant ที่ขอพร้อม fallback, 404/เน็ตล่มคืนข้อความไทย (ไม่ echo body ของเซิร์ฟเวอร์)
- **ตัวดาวน์โหลด**: เขียนไฟล์ปกติ, ต่อจาก `.part` ด้วย `Range` (206), เซิร์ฟเวอร์ที่ไม่รองรับ Range → เริ่มใหม่ไม่ทำให้ไฟล์เสียหาย, ไฟล์สั้นกว่า `Content-Length` → เก็บ `.part` ไว้, SHA-256 ผิด → ลบ, ไฟล์ครบอยู่แล้ว → ไม่แตะเน็ต, ยกเลิกกลางทาง → เก็บ `.part`, ไฟล์ใหญ่เกินเพดาน → หยุด, โฮสต์ต้องห้าม → ไม่มีการเชื่อมต่อเลย
- **ตัวรัน**: `build_args` บังคับ `--host 127.0.0.1` และ clamp ctx/gpu/threads, ตัด `-h`/`--help` ออกจากอาร์กิวเมนต์ผู้ใช้, ตรวจพอร์ตชน, `/health` ตอบ → `running`, โปรเซสตาย → รายงาน exit code + log ท้าย, หยุดแล้วโปรเซสไม่ค้าง, ไม่พบไบนารี/โมเดล → ข้อความชัดเจน
- **ไฟล์ในเครื่อง**: จับคู่ `mmproj` กับโมเดล, `settings_guess` ปรับตามขนาด, โครงสร้าง `llama/{bin,models,downloads}`
- **การรอโหลดแบบไม่หลอกผู้ใช้**: `probe()` ระหว่างที่โมเดลยังโหลด (503/ยังไม่เปิดพอร์ต) **ไม่** ถูกบันทึกเป็น error, โปรเซสตาย → บันทึก exit code + log, ครบเวลาจริง → ข้อความ "นานเกินกำหนด" พร้อมคำแนะนำ; UI poll ผ่าน QTimer ไม่บล็อกเธรดหลัก (บั๊กนี้ถูกพบโดยสคริปต์ smoke ด้านล่าง)

### สคริปต์ smoke แบบออฟไลน์

`python scripts/smoke-llama.py` เดินครบเส้นทางในแอปจริงโดยไม่มีเน็ตและไม่มีเซิร์ฟเวอร์จริง
(Hugging Face = `httpx.MockTransport`, ดาวน์โหลด = ไฟล์ปลอม, `llama-server` = โปรเซสปลอม):
ยินยอม → อ่านรายชื่อไฟล์ → ดาวน์โหลดคู่แนะนำ → เห็นไฟล์ใน `llama/models` → เริ่มเซิร์ฟเวอร์ (ตรวจ `--host 127.0.0.1`, `--no-webui`, `--mmproj`) → `/health` → "ตั้งเป็น AI ของแอป" → บันทึก settings → ปิดแอปแล้วโปรเซสถูกสั่งหยุด
ผลรันล่าสุด: **23/23 checks passed**

> หมายเหตุสภาพแวดล้อม: การรันบน sandbox headless (offscreen Qt + TFLite + stub libs) พบ hang/crash ระดับ *teardown ของโปรเซส*
> เป็นครั้งคราว **ซึ่งมีอยู่ก่อนรอบนี้** (ทดสอบย้อนกับ commit `097dc15` แล้วพบอาการเดียวกัน 1/12 รอบ) โดยผลเทสต์ยังเป็น 199 passed/1 skipped
> ทุกครั้งที่รันจบ — ไม่ใช่ความล้มเหลวของเทสต์; ไม่มีการทดสอบดาวน์โหลดไฟล์หลาย GB จากอินเทอร์เน็ตจริงในรีโปนี้

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
- **UI smoke**: สร้างหน้าต่าง 7 แท็บ (วิจัยเกม / ตัวละคร·ใบหน้า·วิธีพูด / แปลหน้าจอ / Local AI (llama.cpp) / บริบทและตัวละคร / ประวัติ / ตั้งค่า), render, และปิด worker threads อย่างปลอดภัย (รวมกรณีปิดระหว่าง worker ยังทำงาน); รอบ llama เพิ่มเคส consent (กด Cancel แล้วไม่เกิดคำขอ), แถบความคืบหน้า/สถานะไฟล์, auto-start เฉพาะปลายทาง loopback, และการบันทึกค่า `llama_*` ข้ามการเปิดแอปใหม่

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

### Local AI (llama.cpp) — ต้องทดสอบบนเครื่องจริง
- [ ] ดาวน์โหลด `llama-server` รุ่น win-x64 จาก GitHub ผ่านกล่องยินยอม → แตก zip → ไฟล์อยู่ใน `llama\bin` และ zip ถูกลบ
- [ ] ดาวน์โหลด GGUF ขนาดจริง: ระหว่างโหลดกด **ยกเลิก** → มี `.part`; เปิดใหม่ → ต่อจากเดิม (หรือเริ่มใหม่ถ้าเซิร์ฟเวอร์ไม่รองรับ Range) โดยไฟล์ไม่เสีย
- [ ] ไฟล์เสีย/ตัดการเชื่อมต่อกลางทาง → ไม่มีไฟล์ `.gguf` ปลอมโผล่ (ยังเป็น `.part`) และมีข้อความบอกเหตุ
- [ ] พอร์ตชน (เช่น Ollama/LM Studio ยึดพอร์ต) → ข้อความบอกให้เปลี่ยนพอร์ต ไม่สตาร์ตซ้อน
- [ ] เริ่มเซิร์ฟเวอร์ → `/health` ตอบ → ตั้งเป็น AI ของแอป → แปลได้จริง; ปิดแอปกลางทาง → ไม่มี `llama-server` ค้างใน Task Manager
- [ ] เปิด `llama_auto_start` แล้วตั้ง Base URL เป็นบริการภายนอก → **ต้องไม่** เริ่มเซิร์ฟเวอร์ในเครื่อง
- [ ] ตรวจว่าไฟล์/ข้อความบนหน้าจอไม่ถูกส่งไปที่ GitHub/Hugging Face (มีแค่คำขอรายชื่อไฟล์/ดาวน์โหลด)

### Overlay / Capture / AI
- [ ] Borderless เกมและหน้าต่างทั่วไป: คำแปลตรงตำแหน่ง คลิก/เลื่อนเมาส์ทะลุ
- [ ] DPI 100%, 125%, 150%, 200%; สองจอ scaling ต่างกันและจอซ้าย origin ติดลบ
- [ ] Safe capture ไม่อ่านคำแปลตัวเองซ้ำ; exclusion mode ให้ภาพไม่มี overlay และไม่เป็นแผ่นดำ
- [ ] ภาษาไทยยาว สระ/วรรณยุกต์ ไม่ถูกตัดผิด และตรวจกล่องชนกัน
- [ ] Ollama Vision / text-only; ปิด vision แล้ว request ต้องไม่มีภาพ; ภายนอกต้องถามยินยอมก่อนส่งจริง
- [ ] Invalid key, rate-limit, wrong model, malformed JSON, offline, timeout → ข้อความแก้ปัญหาและหยุด (ไม่มี retry อัตโนมัติ)
- [ ] ปิดแอปกลาง request/กลางสแกนใบหน้า → รอ worker อย่างปลอดภัย ไม่มี QThread destroyed crash
- [ ] วัด OCR/inference latency และ VRAM ขณะเล่นเกมบนเครื่องเป้าหมาย ไม่รายงาน FPS จาก timer เป็น FPS การแปล

## รันเทสต์ UI บน Linux/คอนเทนเนอร์ที่ไม่มีจอ (ใช้ใน sandbox นี้)

```bash
python3 -m venv /tmp/stv && /tmp/stv/bin/pip install PySide6 ruff pytest mss keyring httpx Pillow \
    "numpy<2" mediapipe opencv-python-headless rapidocr_onnxruntime
/tmp/stv/bin/python scripts/headless-qt-stubs.py /tmp/qtstubs      # สร้าง libGL/libEGL/... ปลอมจาก undefined symbols จริง
QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH=/tmp/qtstubs /tmp/stv/bin/python -m pytest -q
QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH=/tmp/qtstubs /tmp/stv/bin/python scripts/smoke-llama.py
```

`scripts/headless-qt-stubs.py` เป็นตัวช่วยเฉพาะ sandbox (ไม่เกี่ยวกับแอป Windows) — อธิบายเหตุผลของ
versioned/unversioned stub และการไล่ dependency ไว้ใน docstring ของไฟล์
