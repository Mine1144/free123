# การตรวจสอบและข้อจำกัดการทดสอบ

## ผลที่รันจริงใน sandbox นี้

- Python 3.11.2 / Linux
- `pytest -q`: **31 passed, 1 skipped** (native Qt UI test module ถูกข้าม เพราะไม่มี system libraries เช่น libGL/libEGL)
- `ruff check .`: ผ่าน
- `python -m compileall -q screen_thai`: ผ่าน
- `python scripts/smoke-ocr.py`: ผ่าน ด้วยโมเดล RapidOCR จริงบนภาพสังเคราะห์ “Hello world” ใช้ `opencv-python-headless` ใน virtualenv ของ sandbox เพื่อหลีกเลี่ยง libGL ที่ไม่มี ไม่ได้เปลี่ยน Windows dependency เป็น headless
- Provider tests ใช้ `httpx.MockTransport` ตรวจ payload/response/errors **ไม่ได้เรียกโมเดลหรือใช้ key จริง**
- ยังไม่ได้รัน build `.exe`, Windows native UI, เกมจริง, mixed-DPI capture, Credential Manager จริง หรือ external/local inference end-to-end
- GitHub Actions เตรียมรัน Windows UI smoke/OCR smoke/tests/build แต่ผลไม่ใช่การทดสอบเกมจริง และ workflow ยังไม่ได้รันในรอบพัฒนานี้

## Automated coverage

- Model JSON ผิดรูปแบบ, ID ปลอม/ซ้ำ/boolean, ขนาดใหญ่, relationship ไม่มีหลักฐาน
- Text occurrence matching: พิกัดสั่น, ID reorder, ข้อความหายแล้วกลับมา, HUD เปลี่ยนตลอด, สองกล่องข้อความเหมือนกันแต่คำแปลต่างกัน
- ข้ามข้อความโดย AI แล้วไม่ยิงคำขอเดิมวน, ล้างข้อความเมื่อไม่มี OCR
- Profile isolation/path traversal, settings ผิดรูปแบบ, ไม่เก็บ key ใน JSON
- HTTP/HTTPS endpoint policy, local/external payload, ปิดภาพแล้วไม่ส่ง image field, timeout และ HTTP error ไม่รั่ว response body
- แปลงพิกัดพื้นที่บนจอที่ origin ติดลบ, OCR filtering/clipping/limit
- UI smoke (เมื่อมี Qt runtime): สร้างหน้าต่าง, render ภาษาไทย, ปิด threads อย่างปลอดภัย

## Windows acceptance checklist ก่อนเรียกว่า production-ready

### การติดตั้ง
- [ ] Windows 10 22H2 / Windows 11 x64 สะอาด รัน launcher จาก path ภาษาไทย/มีช่องว่าง
- [ ] ทดสอบ portable folder จาก artifact บนเครื่อง **ไม่มี Python** และเปิด OCR ได้
- [ ] Credential Manager save/load/delete; ไม่พบ key ใน settings, log หรือ error
- [ ] โมเดล OCR load จาก packaged `_internal` ได้โดยไม่พึ่ง source checkout

### Overlay / Capture
- [ ] Borderless เกมและหน้าต่างทั่วไป: คำแปลตรงตำแหน่งและคลิก/เลื่อนเมาส์ทะลุ
- [ ] DPI 100%, 125%, 150%, 200%; สองจอ scaling ต่างกันและจอซ้าย origin ติดลบ
- [ ] เลือกพื้นที่ข้ามขอบใกล้สุดของจอ แล้ว overlay ไม่เลื่อนตำแหน่ง
- [ ] เปลี่ยน resolution / ถอดจอระหว่างทำงานแล้วหยุด ไม่จับผิดจอ
- [ ] Safe capture ไม่อ่านคำแปลตัวเองซ้ำ ทดสอบ DWM latency หลาย GPU
- [ ] ทดลอง exclusion: ภาพที่ได้ไม่มี overlay และไม่เป็นแผ่นดำ ถ้ามีปัญหากลับ safe
- [ ] ซับหาย → คำแปลหายหลัง OCR รอบถัดไป; สลับฉากเร็ว → ไม่วาดผลเก่าคนละ occurrence
- [ ] ภาษาไทยยาว สระ/วรรณยุกต์ ไม่ถูกตัดผิด และตรวจกล่องชนกัน

### AI / Session
- [ ] Ollama Vision / text-only; บริการ Chat Completions ภายนอกที่รองรับภาพ
- [ ] ปิด vision แล้วตรวจ request ไม่มีภาพ; ภายนอกต้องถามยินยอมก่อนส่งจริง
- [ ] Invalid key, rate-limit, wrong model, malformed JSON, offline, timeout → ข้อความแก้ปัญหาและหยุด
- [ ] เกมเปลี่ยนข้อความเร็ว → ไม่มี inference คิวสะสม; หยุด/เริ่มใหม่ไม่รับผล session เก่า
- [ ] ปิดแอปกลาง request → รอ worker อย่างปลอดภัย ไม่มี QThread destroyed crash
- [ ] สองโปรไฟล์: glossary และบริบทไม่ข้ามกัน; manual notes สำคัญกว่า AI hypothesis
- [ ] Hotkeys ขณะเกมมี focus, conflict ของ hotkey, restore ผ่าน tray
- [ ] วัด OCR/inference latency และ VRAM ขณะเล่นเกมบนเครื่องเป้าหมาย ไม่รายงาน FPS จาก timer เป็น FPS การแปล
