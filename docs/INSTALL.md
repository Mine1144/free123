# วิธีติดตั้ง ScreenThai (Windows) และติดตั้ง Local AI ให้ครบในตัวแอป

คู่มือนี้ครอบคลุมทุกเส้นทาง: **ติดตั้งเร็ว** (ไฟล์ `.cmd` ตัวเดียว), **ติดตั้งเองทีละคำสั่ง**,
**สร้าง `.exe` แบบ portable**, และ **ติดตั้ง local AI (llama.cpp) ผ่านตัวโปรแกรม** ซึ่งไม่ต้องติดตั้ง
Ollama หรือเครื่องมืออื่นเลย

> เป้าหมายที่ทดสอบ: **Windows 10/11 x64** เท่านั้น (ARM/32-bit ไม่อยู่ในขอบเขต)
> โปรเจกต์นี้ยังไม่มี `.exe` ที่ build และตรวจสอบแล้วแนบมา — วิธีที่ 3 อธิบายการ build เอง

## 0. สรุปสั้นที่สุด (3 ขั้น)

```powershell
# 1) ติดตั้ง Python 3.11 x64 จาก python.org (ติ๊ก "Add python.exe to PATH" และ Python Launcher)
# 2) แตกไฟล์โปรเจกต์ แล้วดับเบิลคลิก
Start-ScreenThai.cmd
# 3) ในแอป: แท็บ "Local AI (llama.cpp)" → ดาวน์โหลด llama-server → ดาวน์โหลดโมเดล → เริ่ม + ตั้งเป็น AI ของแอป
```

## 1. สิ่งที่ต้องมีก่อนติดตั้ง

| รายการ | อย่างต่ำ | แนะนำ | หมายเหตุ |
|---|---|---|---|
| ระบบปฏิบัติการ | Windows 10 x64 (21H1+) | Windows 11 x64 | ARM/32-bit ไม่รองรับ |
| Python | 3.11 x64 (มี `py` launcher) | 3.11 หรือ 3.12 | โปรเจกต์กำหนด `>=3.11,<3.14` |
| พื้นที่ดิสก์ (ตัวแอป) | ~1.5 GB | 3 GB | ตัว env + OCR model |
| พื้นที่ดิสก์ (Local AI) | +3 GB | +12 GB | ไฟล์ `.gguf` ต่อโมเดล 2–6 GB + mmproj ~1 GB |
| RAM | 8 GB | 16 GB ขึ้นไป | โมเดล 7B Q4_K_M ใช้ราว 6–8 GB ระหว่างโหลด |
| การ์ดจอ | ไม่ต้องมี | NVIDIA/AMD/Intel ที่ llama.cpp รองรับ | ไม่มีก็รัน CPU ล้วน (`n-gpu-layers = 0`) |
| อินเทอร์เน็ต | ต้องมีตอนติดตั้ง env | — | ใช้ดาวน์โหลด dependency และ (ถ้าต้องการ) ไฟล์ llama.cpp/โมเดล |
| Visual C++ Redistributable x64 | — | ติดตั้งไว้ | จำเป็นบางกรณี โดยเฉพาะเมื่อใช้ไฟล์ `.exe` |

แอป **ไม่ต้องใช้สิทธิ์ Administrator**, ไม่ inject เข้าเกม และไม่ติดตั้ง service ใด ๆ —
ทุกอย่างอยู่ในโฟลเดอร์โปรเจกต์และโฟลเดอร์ข้อมูลผู้ใช้

## 2. วิธีที่ 1 — ติดตั้งอัตโนมัติ (แนะนำสำหรับผู้ใช้ทั่วไป)

1. ติดตั้ง **Python 3.11 x64** จาก [python.org](https://www.python.org/downloads/windows/)
   - ในตัวติดตั้ง ติ๊ก **Add python.exe to PATH** และ **py launcher** (ค่าเริ่มต้นเปิดอยู่)
   - ตรวจว่าติดตั้งแล้ว: เปิด PowerShell แล้วพิมพ์
     ```powershell
     py -3.11 --version
     ```
     ต้องขึ้น `Python 3.11.x`
2. ดาวน์โหลดโปรเจกต์นี้ (ปุ่ม Code → Download ZIP) แล้ว **แตกไฟล์ทั้งโฟลเดอร์**
   (อยู่วางใน path ที่มีช่องว่างได้ เช่น `C:\Games\ScreenThai`)
3. ดับเบิลคลิก **`Start-ScreenThai.cmd`**
   - ครั้งแรกจะสร้าง `.venv` และติดตั้ง dependency จากอินเทอร์เน็ต (ใช้เวลาสักครู่)
   - ครั้งต่อไปจะเปิดแอปทันที ไม่ติดตั้งซ้ำ
4. ถ้าหน้าต่าง CMD ขึ้น error ให้อ่านข้อความนั้นตรง ๆ (สคริปต์ไม่ซ่อน error) แล้วดูหัวข้อ [9. แก้ปัญหา](#9-แก้ปัญหาการติดตั้งที่พบบ่อย)

## 3. วิธีที่ 2 — ติดตั้งเองทีละคำสั่ง (ควบคุมได้ทุกอย่าง)

```powershell
cd C:\Games\ScreenThai              # โฟลเดอร์โปรเจกต์

py -3.11 -m venv .venv              # สร้าง environment
.venv\Scripts\python -m pip install --upgrade pip

# ติดตั้งแบบพื้นฐาน (พอสำหรับแปล + Local AI llama.cpp)
.venv\Scripts\python -m pip install -e .

# เพิ่มความสามารถใบหน้า/สีหน้า (mediapipe — ดึง numpy<2 ให้อัตโนมัติ)
.venv\Scripts\python -m pip install -e ".[faces]"

# เพิ่มเครื่องมือพัฒนา (pytest/ruff/pyinstaller) — ใช้เมื่อจะรันเทสต์หรือ build exe
.venv\Scripts\python -m pip install -e ".[dev,faces]"
```

เปิดแอปด้วยหนึ่งในนี้

```powershell
.venv\Scripts\python -m screen_thai.app
# หรือ
.venv\Scripts\screen-thai.exe
```

หมายเหตุสำคัญ

- **ติดตั้งใบหน้าไม่จำเป็นต่อการแปล** ถ้าไม่มี `mediapipe` แอปจะลดระดับเป็น OpenCV/LiDAR-free
  (OpenCV Haar) หรือปิดฟีเจอร์พร้อมบอกเหตุผล และการแปลยังทำงานปกติ
- `mediapipe` บังคับ `numpy<2` — ถ้าคุณติดตั้ง `numpy 2.x` เองไว้ pip จะปรับให้เอง
  ถ้าเคย `pip install numpy --upgrade` ทีหลัง ให้ติดตั้ง `.[faces]` ใหม่
- ถ้า pip ช้า/ค้าง ให้เพิ่ม `--timeout 120` หรือใช้ mirror ที่เชื่อถือได้ (บริษัท/มหาวิทยาลัย)

## 4. วิธีที่ 3 — สร้างแอป `.exe` แบบ portable

### 4.1 build บนเครื่องคุณ

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-windows.ps1
```

สคริปต์จะ: สร้าง `.venv` → ติดตั้ง `.[dev,faces]` → **รัน `pytest -q` (ถ้าเทสต์ไม่ผ่านจะไม่ build ต่อ)**
→ เก็บไฟล์ด้วย PyInstaller (รวม OCR/onnxruntime/mediapipe/keyring) → คัดลอก `README.md` + `docs`
→ ได้ **`dist\ScreenThai-Windows-x64.zip`**

วิธีใช้: แตก **ทั้งโฟลเดอร์** แล้วเปิด `ScreenThai.exe`
**อย่าย้ายเฉพาะไฟล์ exe ออกจากโฟลเดอร์ `_internal`** (จะเปิดไม่ขึ้น)

### 4.2 build ผ่าน GitHub Actions

บน GitHub → แท็บ **Actions** → **Windows build and tests** → **Run workflow**
แล้วดาวน์โหลด artifact `ScreenThai-Windows-x64` เมื่อ build ผ่าน

ข้อควรรู้: ตัวแอปยัง **ไม่มี code signing** Windows SmartScreen อาจเตือน —
กด *More info → Run anyway* เฉพาะเมื่อคุณได้ไฟล์จากแหล่งที่คุณไว้ใจ และอย่าลืมติดตั้ง
Microsoft Visual C++ Redistributable x64 บนเครื่องปลายทางหากเปิดแล้ว error เรื่อง DLL

## 5. ติดตั้ง Local AI (llama.cpp) ผ่านตัวโปรแกรม — ไม่ต้องติดตั้งอะไรเพิ่ม

นี่คือหัวใจของรอบล่าสุด: **ตัวแอปดาวน์โหลดและจัดการให้เองทั้งหมด** ไม่ต้องมี Ollama, ไม่ต้องใช้คำสั่ง `ollama pull`,
ไม่ต้องโหลด GGUF ด้วยมือ และไม่มีอะไรถูกติดตั้งลง PATH หรือเป็น service ของระบบ

### ขั้นที่ 1 — ไฟล์ `llama-server`

1. เปิดแอป → แท็บ **Local AI (llama.cpp)**
2. ในช่อง "ไฟล์ llama-server" กด **ดาวน์โหลดจาก GitHub**
   - แอปจะถามยินยอมก่อนเชื่อมต่อ `api.github.com` (§6) → กด Yes
   - เลือกรุ่นที่ตรงกับเครื่องคุณ (`llama-*-bin-win-*` สำหรับ Windows x64; ถ้ามี NVIDIA เลือกตัว `-cuda`, AMD/Intel เลือก `-vulkan` ได้)
   - กดดาวน์โหลด → zip จะถูกแตกอย่างปลอดภัย → **ลบ zip ทิ้ง** → พาธถูกจำไว้ให้
3. ทางเลือกถ้ามีไฟล์อยู่แล้ว: **เลือกไฟล์เอง** ชี้ไปที่ `llama-server.exe`
   หรือ **ติดตั้งจาก zip** ที่คุณโหลดมา

ไฟล์จะอยู่ใน `%LOCALAPPDATA%\ScreenThai\llama\bin` (ขนาด zip หลักสิบ–ร้อย MB ตามรุ่น/แบ็กเอนด์)

### ขั้นที่ 2 — โมเดล GGUF

1. เลือกจาก **แคตตาล็อก** หรือพิมพ์ `owner/repo` เอง แล้วกด **ดูรายการไฟล์**

| แคตตาล็อกในแอป | เหมาะกับ | RAM ที่แนะนำ | ขนาดโดยประมาณ |
|---|---|---|---|
| Qwen2.5-VL 7B (ดูภาพได้) | แปลหน้าจอเต็มรูปแบบ (แนะนำ) | 8 GB+ | ~5–6 GB + mmproj |
| Qwen2.5 7B (ข้อความล้วน) | เครื่องกลาง ๆ ไม่เปิดให้ AI ดูภาพ | 6 GB+ | ~4–5 GB |
| Gemma 3 4B (ดูภาพได้) | เครื่อง RAM 8 GB | 5 GB+ | ~3 GB + mmproj |
| Qwen2.5 3B (เบามาก) | เครื่องเก่า/ไม่มีการ์ดจอ | 4 GB+ | ~2 GB |

2. เลือก **quant** (ค่าเริ่มต้น `Q4_K_M` คือสมดุลที่ดีที่สุดของขนาดกับคุณภาพ;
   `Q5_K_M`/`Q6_K`/`Q8_0` คุณภาพสูงขึ้นแต่ไฟล์ใหญ่ขึ้น)
3. กด **ดาวน์โหลดคู่ที่แนะนำ** (โมเดล + ไฟล์ `mmproj` สำหรับดูภาพ) หรือกด **ดาวน์โหลดแถวที่เลือก**
   - ยินยอมก่อนทุกครั้ง โดยบอกปลายทาง (huggingface.co) ขนาดรวม และโฟลเดอร์ปลายทาง
   - ระหว่างโหลดดูเปอร์เซ็นต์ได้ และ **กดยกเลิกได้ตลอด** — ไฟล์ `.part` ถูกเก็บไว้ ครั้งถัดไปจะโหลดต่อจากเดิม (ถ้าเซิร์ฟเวอร์รองรับ Range)
   - ถ้ามีการแจ้ง SHA-256 แอปจะตรวจให้และลบไฟล์ทิ้งถ้าไม่ตรง; ถ้าไฟล์ไม่ครบจะ **ไม่** เปลี่ยนชื่อเป็นไฟล์จริง
4. ไฟล์จะอยู่ใน `%LOCALAPPDATA%\ScreenThai\llama\models`
   ต้องมีพื้นที่ว่างเหลือ ≥ 2 GB และไฟล์เดียวไม่เกินเพดาน 60 GB

### ขั้นที่ 3 — เริ่มเซิร์ฟเวอร์และตั้งเป็น AI ของแอป

1. เลือกโมเดลในดรอปดาวน์ (ถ้าเพิ่งโหลดเสร็จ กดปุ่มโหลดรายการใหม่/สลับแท็บให้รีเฟรช)
2. ตั้งค่า: พอร์ต (ค่าเริ่มต้น **8081**), context (**8192**), `n-gpu-layers` (**0 = CPU ล้วน**;
   มีการ์ดจอให้เพิ่มเป็น 20–32), threads (0 = อัตโนมัติ) แล้วกด **เริ่ม llama-server**
3. รอสถานะไล่ `กำลังโหลด…` → `ทำงานอยู่` (โมเดลใหญ่ใช้เวลาหลายสิบวินาทีถึง ~2 นาที)
   ระหว่างนี้ดู log ท้ายหน้าจอได้ ถ้าโปรเซสตายจะมี exit code + log บอก
4. กด **ตั้งเป็น AI ของแอป** → แอปจะตั้งผู้ให้บริการเป็น `API · OpenAI-compatible`,
   Base URL `http://127.0.0.1:8081/v1` และชื่อโมเดลให้เอง
5. กลับไปแท็บ **แปลหน้าจอ** ตั้งค่าจอ/พื้นที่ แล้วกด **เริ่มแปลหน้าจอ** — ต่อจากนี้คำแปลไม่ต้องออกอินเทอร์เน็ตเลย

ความปลอดภัยที่ฝังไว้แล้ว: เซิร์ฟเวอร์ผูกกับ `127.0.0.1` **เท่านั้น** (เครื่องอื่นใน LAN เข้าไม่ได้)
และเปิดด้วย `--no-webui` (ไม่มีหน้าเว็บของ llama.cpp) เมื่อปิดหน้าต่างแอป แอปจะสั่งหยุดเซิร์ฟเวอร์ให้เสมอ

### 5.1 ตั้งค่า llama.cpp ที่บันทึกไว้

อยู่ใน `%LOCALAPPDATA%\ScreenThai\settings.json` (อ่านได้ ไม่เข้ารหัส):

| คีย์ | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `llama_binary` | ว่าง | พาธ `llama-server.exe` |
| `llama_model` | ว่าง | ไฟล์ `.gguf` ที่ใช้ล่าสุด |
| `llama_mmproj` | ว่าง | ไฟล์ projector (สำหรับการดูภาพ) |
| `llama_port` | 8081 | พอร์ต (1024–65535) |
| `llama_ctx` | 8192 | context (2048–131072) |
| `llama_gpu_layers` | 0 | จำนวน layer ที่ส่งให้ GPU |
| `llama_threads` | 0 | 0 = อัตโนมัติ |
| `llama_auto_start` | true | เริ่มเซิร์ฟเวอร์ให้เมื่อกดเริ่มแปล **ถ้าปลายทางที่ตั้งไว้เป็น `127.0.0.1`/`localhost` อยู่แล้วเท่านั้น** |
| `llama_extra_args` | ว่าง | อาร์กิวเมนต์เพิ่มเติม เช่น `--flash-attn` (≤20 รายการ) |

### 5.2 ลบโมเดล/คืนพื้นที่

ยังไม่มีปุ่มลบในแอป — ปิดแอปก่อน แล้วลบโฟลเดอร์ `%LOCALAPPDATA%\ScreenThai\llama\models`
(หรือลบทั้ง `llama` ถ้าต้องการเอาไฟล์เซิร์ฟเวอร์ออกด้วย) รายการในดรอปดาวน์จะว่างหลังรีเฟรช

## 6. กล่องยินยอมและข้อมูลที่ออกอินเทอร์เน็ต

| การกระทำ | ปลายทาง | ขอความยินยอม |
|---|---|---|
| ดูรายชื่อรุ่น llama.cpp | `api.github.com`, `github.com`, `objects.githubusercontent.com` | ใช่ (ทุกครั้ง) |
| อ่านรายชื่อไฟล์โมเดล | `huggingface.co` | ใช่ (ทุกครั้ง) |
| ดาวน์โหลดไฟล์ | `huggingface.co`, `cdn-lfs*.huggingface.co`, `hf.co` | ใช่ (ทุกรอบดาวน์โหลด) |
| รัน/แปลด้วยเซิร์ฟเวอร์ในเครื่อง | `127.0.0.1` — ไม่ออกนอกเครื่อง | ไม่ต้อง (ไม่มีอะไรออกไปข้างนอก) |
| แปลด้วยบริการภายนอก (OpenAI/LM Studio ฯลฯ) | ตามที่คุณตั้ง | ใช่ ก่อนส่งจริงครั้งแรก |

อนุญาตเฉพาะ **HTTPS** และเฉพาะโฮสต์ในรายการ — URL ที่มี userinfo (`https://huggingface.co@evil.com`),
query/fragment หรือโฮสต์อื่นจะถูกปฏิเสธก่อนเปิดการเชื่อมต่อ

**ลิขสิทธิ์**: ตัว llama.cpp เป็น MIT แต่ **น้ำหนักโมเดล GGUF มีใบอนุญาตต่างกัน** (Qwen, Gemma ฯลฯ)
ผู้ใช้ต้องตรวจใบอนุญาตของ repo ที่เลือกเองก่อนใช้ในงานจริง

## 7. ทางเลือก AI อื่น ๆ (ถ้าไม่ใช้ llama.cpp)

| ทางเลือก | ต้องติดตั้งอะไร | ตั้งค่าในแอป |
|---|---|---|
| **Local AI ในแอป (llama.cpp)** | ไม่ต้องติดตั้งอะไรเลย | ทำตาม §5 |
| **Ollama** | ติดตั้ง [Ollama for Windows](https://ollama.com/download/windows) แล้ว `ollama pull qwen2.5vl:7b` | เลือก `Local AI · Ollama`, Base URL `http://localhost:11434` (**ไม่เติม** `/api` หรือ `/v1`), โมเดล `qwen2.5vl:7b`, ไม่ต้องใส่ key |
| **LM Studio / OpenAI-compatible** | ติดตั้ง LM Studio แล้วเปิด local server | เลือก `API · OpenAI-compatible` แล้วกรอก Base URL ของเซิร์ฟเวอร์นั้น |
| **OpenAI / บริการคลาวด์** | ไม่ต้องติดตั้ง | เลือกบริการ, ใส่ Base URL + โมเดล + **API key** (เก็บใน Windows Credential Manager) |

API key ไม่ถูกเก็บในไฟล์ `settings.json` — เก็บใน Windows Credential Manager หรืออยู่ใน RAM เฉพาะรอบ
ล้างได้ที่ *Credential Manager → Windows Credentials → `ScreenThai`*

## 8. ตรวจว่าติดตั้งสำเร็จ

1. **ในแอป**: แท็บ **ตั้งค่าและวิธีใช้** → กด **ตรวจสุขภาพระบบ**
   จะได้รายงานจริงทีละรายการ เช่น `✔ OCR`, `✔ ใบหน้า (mediapipe)`, `✔ ผังหน้าจอ`, `✔ แผนการพูด`,
   `! ฟอนต์ไทย`, `! ยังไม่มีข้อมูลวิจัยเกม` — เครื่องหมาย `!`/`✘` พร้อมเหตุผลคือค่าที่ควรดู
2. **Local AI**: แท็บ **Local AI (llama.cpp)** ต้องขึ้น `ทำงานอยู่ • <ชื่อโมเดล> • http://127.0.0.1:8081/v1`
3. **สำหรับผู้พัฒนา** (ต้องติดตั้ง `.[dev]`):

```powershell
.venv\Scripts\python -m pytest -q            # ต้องได้ 199 passed, 1 skipped
.venv\Scripts\python -m ruff check .         # ต้องผ่าน
.venv\Scripts\python scripts/smoke-ocr.py    # ทดสอบ OCR จริงด้วยภาพสังเคราะห์
.venv\Scripts\python scripts/smoke-llama.py  # เดินเส้นทาง Local AI ครบโดยไม่ต้องมีเน็ต/เซิร์ฟเวอร์จริง (23/23)
```

### โครงสร้างโฟลเดอร์ข้อมูลผู้ใช้

```
%LOCALAPPDATA%\ScreenThai\
├── settings.json          # ค่าตั้งแอป (ไม่เก็บ API key)
├── profiles\              # โปรไฟล์แยกเกม (ไฟล์ JSON ต่อเกม): บริบท, glossary, ใบหน้า,
│                          #   การ์ดวิธีพูด, หน่วยความจำคำแปล และ brief วิจัยเกม
└── llama\
    ├── bin\               # llama-server.exe + DLL ที่แตกจาก zip
    ├── models\            # ไฟล์ .gguf (โมเดล + mmproj)
    └── downloads\         # ไฟล์ .part ที่ยังโหลดไม่จบ
```

(path ของข้อมูลจริงดูได้จากแท็บ *ตั้งค่าและวิธีใช้* → ตรวจสุขภาพระบบ ซึ่งรายงานโฟลเดอร์ข้อมูลที่ใช้อยู่)

## 9. แก้ปัญหาการติดตั้งที่พบบ่อย

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `py` is not recognized | ยังไม่ติดตั้ง Python/ไม่ได้เปิด Python Launcher | ติดตั้ง Python 3.11 x64 ใหม่ แล้วติ๊ก *py launcher* |
| `Start-ScreenThai.cmd` ปิดทันที | สร้าง `.venv` ไม่สำเร็จ | เปิด CMD ในโฟลเดอร์นั้นแล้วรัน `py -3.11 -m venv .venv` ดู error จริง |
| `pip` ค้าง/ช้า | เครือข่าย/ไฟร์วอลล์ | `pip install -e . --timeout 120` แล้วลองใหม่; ถ้าอยู่หลัง proxy ให้ตั้ง `HTTPS_PROXY` |
| ติดตั้ง `mediapipe` ไม่ได้ | numpy ชนกัน/มี numpy 2.x | `pip install -e ".[faces]"` ใหม่ (จะดึง `numpy<2`) หรือข้าม `faces` ไปก่อน แอปยังแปลได้ |
| เปิดแอปแล้วขึ้น error เรื่อง Qt platform plugin | ขาด DLL ของ Qt/ย้ายไฟล์ผิด | ใช้วิธีที่ 1 หรือ 2; ถ้าใช้ `.exe` **อย่าย้าย exe ออกจาก `_internal`** และติดตั้ง VC++ Redistributable x64 |
| SmartScreen เตือน | ยังไม่มี code signing | ตรวจแหล่งที่มาให้แน่ใจก่อนกด *More info → Run anyway* |
| ข้อความไทยเป็นสี่เหลี่ยม | เครื่องไม่มีฟอนต์ไทย | Windows 10/11 มี *Leelawadee UI* มาให้อยู่แล้ว; ถ้าถูกถอดออก ให้ติดตั้งฟอนต์ไทยของ Windows เพิ่ม |
| ใบหน้าตรวจไม่ได้ แต่แปลได้ | ไม่ได้ติดตั้ง `.[faces]` | `pip install -e ".[faces]"` (หรือปล่อยไว้แล้วใช้ OpenCV) |
| llama.cpp: "พอร์ต 8081 ถูกใช้อยู่" | มีโปรแกรมอื่นยึดพอร์ต | เปลี่ยนพอร์ตในขั้นที่ 3 (เช่น 8082) |
| llama.cpp: "รอ llama-server โหลดโมเดลนานเกินกำหนด" | โมเดลใหญ่เกิน RAM/context สูงเกิน | เลือก quant เล็กลง (`Q4_K_M`), ลด context (4096), ปิดโปรแกรมอื่น |
| ดาวน์โหลดโมเดลค้าง/ขาด | เน็ตสะดุด | กดยกเลิกได้เลย ไฟล์ `.part` อยู่ — ครั้งถัดไปกดใหม่จะโหลดต่อ (ไม่เริ่มจากศูนย์ถ้าเซิร์ฟเวอร์รองรับ Range) |
| แปลแล้วไม่มีอะไรขึ้น | ยังไม่ได้ตั้ง AI หรือผู้ให้บริการยังไม่พร้อม | กด *ตรวจสุขภาพระบบ* และดูข้อความ error ในแถบสถานะ (แอปไม่ retry เองโดยไม่บอก) |

## 10. ถอนการติดตั้ง

1. ปิดแอป
2. ลบโฟลเดอร์โปรเจกต์ (รวม `.venv`) หรือโฟลเดอร์ที่แตกจาก `dist\ScreenThai-Windows-x64.zip`
3. ลบข้อมูลผู้ใช้: `%LOCALAPPDATA%\ScreenThai` (โปรไฟล์, โมเดล GGUF, ไฟล์ `.part`, settings)
4. ลบ API key: *Credential Manager → Windows Credentials → `ScreenThai`*
5. ตัวแอปไม่สร้าง service, ไม่แตะ PATH และไม่เพิ่มรายการใน Startup — ไม่มีอะไรต้องลบนอกเหนือจากข้อ 2–4

## 11. คำสั่งสรุป (cheat sheet)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,faces]"
.venv\Scripts\python -m screen_thai.app                 # เปิดแอป
.venv\Scripts\python -m pytest -q                       # เทสต์
.venv\Scripts\python scripts/smoke-llama.py             # smoke ของ Local AI
powershell -ExecutionPolicy Bypass -File scripts/build-windows.ps1   # สร้าง .exe portable
```
