"""Offline end-to-end smoke for the in-app llama.cpp features (no internet, no real server).

Run:  QT_QPA_PLATFORM=offscreen python scripts/smoke-llama.py

Everything is faked: Hugging Face is an ``httpx.MockTransport``, the transfer writes a
small fake file, and ``llama-server`` is a stub process. The point is to prove the *wiring*
inside the app (consent -> list -> download -> start -> use as app AI -> stop on close)
works together — not to download gigabytes.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
TMP = tempfile.mkdtemp(prefix="smoke-llama-")
os.environ["LOCALAPPDATA"] = TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import screen_thai.app as app_module  # noqa: E402
from screen_thai import llama  # noqa: E402

BINARY = Path(TMP) / "llama-server"
MODEL_NAME = "Model-Q4_K_M.gguf"
PROJECTOR_NAME = "mmproj-Model-f16.gguf"
results = []


def check(name: str, condition: bool, detail: str = ""):
    results.append((name, bool(condition), detail))
    print(("✔ " if condition else "✘ ") + name + (f" — {detail}" if detail else ""), flush=True)


def server_process():
    """The stub process that is really llama-server (other libraries shell out to ldconfig)."""
    for process in StubProcess.instances:
        if "llama-server" in str(process.args[0]):
            return process
    return None


def wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --- fakes ------------------------------------------------------------------------------------
HF_FILES = {
    "siblings": [
        {"rfilename": MODEL_NAME, "size": 3_500_000_000},
        {"rfilename": PROJECTOR_NAME, "size": 900_000_000},
        {"rfilename": "README.md"},
    ]
}
REPO = "smoke/Model-GGUF"


def hf_handler(request: httpx.Request) -> httpx.Response:
    assert request.url.host in llama.ALLOWED_HOSTS, request.url
    return httpx.Response(200, json=HF_FILES)


def fake_hf_files(repo, transport=None):
    return llama.hf_model_files(repo, transport=httpx.MockTransport(hf_handler))


def fake_fetch(self, url, target, progress=None, cancel=None, sha256="", extra_gb=0.0):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = b"G" * 4096
    if progress:
        progress(len(body), len(body))
    target.write_bytes(body)
    return llama.DownloadResult(target, len(body), True)


class StubProcess:
    instances: list["StubProcess"] = []

    def __init__(self, args, **kwargs):
        self.args = args
        self.returncode = None
        self.stdout = iter(["llama_model_loader: loaded model\n", "server is listening on 127.0.0.1\n"])
        self.terminated = False
        StubProcess.instances.append(self)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            time.sleep(min(timeout or 0.05, 0.05))
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.terminated = True
        self.returncode = -9


def health_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok"})


# --- run --------------------------------------------------------------------------------------
app = QApplication([])
prompts = []
QMessageBox.question = staticmethod(
    lambda *args, **kwargs: (prompts.append(args[1]), QMessageBox.StandardButton.Yes)[1])
QMessageBox.warning = staticmethod(lambda *args, **kwargs: prompts.append("WARN: " + str(args[1])))
QMessageBox.information = staticmethod(lambda *args, **kwargs: None)

app_module.hf_model_files = fake_hf_files
llama.Downloader.fetch = fake_fetch
llama.subprocess.Popen = StubProcess

window = app_module.Window()
check("หน้าต่างมีแท็บ Local AI", window.tabs.count() == 7 and
      "Local AI" in window.tabs.tabText(window.tabs.count() - 2), window.tabs.tabText(4))
check("ยังไม่เริ่มเซิร์ฟเวอร์เองตอนเปิดแอป", not window.llama.state.running)

# 1) list files (consent dialog first)
window.list_llama_files(REPO)
check("ขอยินยอมก่อนอ่านรายชื่อไฟล์",
      any("hugging face" in str(p).lower() or "huggingface.co" in str(p) for p in prompts))
check("อ่านรายชื่อไฟล์ GGUF ได้", len(window.pending_llama_files) == 2,
      str([f["name"] for f in window.pending_llama_files]))
check("ไฟล์ mmproj ถูกทำเครื่องหมาย", window.pending_llama_files[-1]["mmproj"] is True)

# 2) download the recommended pair
window.download_llama_recommended()
check("ขอยินยอมก่อนดาวน์โหลด", sum(1 for p in prompts if "ยืนยันการดาวน์โหลด" in str(p)) == 1)
check("ดาวน์โหลดจบภายในเวลาที่รอ", wait_until(lambda: not window.downloads), "คิวว่าง")
models = window.refresh_llama_models()
on_disk = sorted(path.name for path in llama.models_dir(window.store.root).glob("*.gguf"))
check("ไฟล์โมเดล + mmproj อยู่ในโฟลเดอร์ llama/models",
      on_disk == sorted([MODEL_NAME, PROJECTOR_NAME]), str(on_disk))
check("installed_models ไม่นับ mmproj เป็นโมเดลแยก", [m["name"] for m in models] == [MODEL_NAME],
      str([m["name"] for m in models]))
check("mmproj ถูกจับคู่กับโมเดล", any(m["name"] == MODEL_NAME and m["projector"].endswith(PROJECTOR_NAME)
                                    for m in models))

# 3) start the (stub) server and use it
BINARY.write_bytes(b"stub")
window.local_page.binary_edit.setText(str(BINARY))
window.llama.transport = httpx.MockTransport(health_handler)
window.refresh_llama_models()
window.local_page.model_combo.setCurrentIndex(0)
entry = window.local_page.model_combo.currentData() or {}
window.local_page.port.setValue(window.local_page.values()["port"])
check("เริ่ม llama-server ได้", window.start_llama() is True, entry.get("name", ""))
window.llama.poll(timeout=3)          # blocking poll is fine in a smoke script
check("health ตอบแล้วสถานะเป็นทำงานอยู่", window.llama.state.running, window.llama.status_text())
server = server_process()
check("มีโปรเซส llama-server ถูกเรียกจริง", server is not None)
started_args = list(server.args) if server else []
check("ผูกที่อยู่กับ 127.0.0.1 เท่านั้น", "--host" in started_args and
      started_args[started_args.index("--host") + 1] == "127.0.0.1", " ".join(started_args[1:]))
check("เปิดเองโดยไม่มี --webui", "--no-webui" in started_args)
check("ส่ง --mmproj ให้โมเดลที่ดูภาพได้", "--mmproj" in started_args,
      str(entry.get("projector")))

window.llama.models = lambda: [MODEL_NAME]     # avoid a real HTTP call to the stub server
window._llama_poll(1)
check("หน้า Local AI รายงานว่าใช้เป็น AI ของแอปได้", "พร้อมใช้งาน" in window.status.text())
check("ตั้งเป็น AI ของแอปได้", window.use_llama_endpoint() is True)
check("ปลายทางชี้ไปที่เซิร์ฟเวอร์ในเครื่อง",
      window.endpoint.text().startswith("http://127.0.0.1:"), window.endpoint.text())
check("ผู้ให้บริการสลับเป็น OpenAI-compatible", window.provider.currentData() == "openai")
saved = window.store.load_settings()
check("บันทึกค่า llama_* ลง settings.json",
      saved.llama_binary == str(BINARY) and saved.endpoint == window.endpoint.text())

# 4) close the app: the child process must not be left behind
window.close()
check("ปิดแอปแล้วสั่งหยุด llama-server", bool(server and server.terminated))
check("ไม่เหลือ worker thread", wait_until(lambda: not window._workers_running(), 15))

failed = [name for name, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED: " + "; ".join(failed))
sys.exit(1 if failed else 0)
