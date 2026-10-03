"""llama.cpp runtime, model catalogue and resumable downloads — driven from inside ScreenThai.

The idea: the player should not have to install Ollama or hunt for GGUF files. The app can fetch
a llama.cpp server build, download a model (and its vision projector) from Hugging Face, start the
server on 127.0.0.1 and point the existing OpenAI-compatible client at it.

Rules that keep this safe:

* Downloads are **HTTPS only** and only from an allowlist of hosts (Hugging Face and GitHub
  release storage). A repo id or file name is validated before it becomes a URL or a path.
* Zips are extracted strictly: no absolute paths, no `..`, only the files we asked for.
* Nothing is ever started automatically: the user presses Start, and the server binds to
  loopback only.
* Sizes are checked against the server's own Content-Length; a short download stays a `.part`
  file and can be resumed. A SHA-256 is verified when the catalogue provides one.
* Every action is cancellable and the partial file is kept, so a slow connection is not punished.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

from .providers import ProviderError

ALLOWED_HOSTS = (
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "hf.co",
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
)

HF_API = "https://huggingface.co/api/models/{repo}"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{name}"
GH_RELEASES = "https://api.github.com/repos/{repo}/releases"

REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+ ,()\[\]-]*\.gguf$", re.IGNORECASE)
MAX_DOWNLOAD_GB = 60.0
CHUNK = 1 << 20
MIN_FREE_GB = 2.0

QUANTS = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q3_K_M", "Q4_K_S", "Q5_K_S", "F16")

# Suggestions only: the file list always comes from the Hugging Face API, so a repo that changes
# its quant names still works. `vision=True` means the repo ships an mmproj projector.
CATALOG = (
    {"id": "qwen2.5-vl-7b",
     "label": "Qwen2.5-VL 7B · ดูภาพได้ (แนะนำสำหรับแปลหน้าจอ)",
     "repo": "bartowski/Qwen2.5-VL-7B-Instruct-GGUF", "vision": True, "quant": "Q4_K_M",
     "ram_gb": 8, "note": "อ่านภาพหน้าจอได้เองทั้งภาพและข้อความ • คุณภาพไทยดี • ไฟล์ประมาณ 5–6 GB"},
    {"id": "qwen2.5-7b",
     "label": "Qwen2.5 7B · ข้อความล้วน (เบาและเร็ว)",
     "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF", "vision": False, "quant": "Q4_K_M",
     "ram_gb": 6, "note": "ใช้เมื่อเปิด ‘ให้ AI ดูภาพ’ ออก หรือเครื่องไม่พอ • เร็วขึ้นชัดเจน"},
    {"id": "gemma-3-4b",
     "label": "Gemma 3 4B · ดูภาพได้ (ไฟล์เล็ก)",
     "repo": "ggml-org/gemma-3-4b-it-GGUF", "vision": True, "quant": "Q4_K_M",
     "ram_gb": 5, "note": "เหมาะกับเครื่อง RAM 8 GB • ต้องโหลด mmproj เพิ่มเพื่อดูภาพ"},
    {"id": "qwen2.5-3b",
     "label": "Qwen2.5 3B · เบามาก",
     "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF", "vision": False, "quant": "Q4_K_M",
     "ram_gb": 4, "note": "เครื่องเก่า/ไม่มีการ์ดจอ • คุณภาพต่ำลงแต่ยังใช้บริบทและ glossary ได้"},
)

TIP = ("โหลดครั้งเดียวต่อโมเดล • ไฟล์เก็บในเครื่องคุณ • ไม่มีการส่งข้อมูลออกไปหลังดาวน์โหลดเสร็จ "
       "• ระหว่างเล่นเกมแนะนำปิดโปรแกรมอื่นเพื่อเก็บ RAM/VRAM")


class LlamaError(Exception):
    pass


def check_url(url: str) -> str:
    """HTTPS + allowlisted host, no userinfo/query/fragment. Raises LlamaError."""
    url = str(url or "").strip()
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise LlamaError("ต้องเป็นลิงก์ HTTPS เท่านั้น")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise LlamaError("ลิงก์ต้องไม่มี userinfo/query/#")
    host = (parts.hostname or "").lower()
    if not any(host == allowed or host.endswith("." + allowed) for allowed in ALLOWED_HOSTS):
        raise LlamaError(f"โฮสต์ {host or 'ไม่ระบุ'} ไม่อยู่ในรายการที่อนุญาต "
                         f"({', '.join(ALLOWED_HOSTS[:3])} …)")
    return url


def validate_repo(repo: str) -> str:
    repo = str(repo or "").strip().strip("/")
    if not REPO_RE.match(repo) or ".." in repo:
        raise LlamaError("ชื่อ repo ต้องเป็นรูปแบบ owner/name (ตัวอักษร ตัวเลข . _ -)")
    return repo


def validate_filename(name: str) -> str:
    name = str(name or "").strip()
    if not FILE_RE.match(name) or "/" in name or "\\" in name or ".." in name:
        raise LlamaError("ชื่อไฟล์ไม่ถูกต้อง (ต้องเป็น .gguf และไม่มีเส้นทาง)")
    return name


def hf_api_url(repo: str) -> str:
    return HF_API.format(repo=validate_repo(repo))


def hf_download_url(repo: str, name: str) -> str:
    return check_url(HF_FILE.format(repo=validate_repo(repo), name=quote(validate_filename(name))))


def _client(transport=None, timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(transport=transport, timeout=httpx.Timeout(timeout, connect=10),
                        follow_redirects=False, trust_env=False)


def hf_model_files(repo: str, transport=None) -> list[dict]:
    """List the GGUF files of a repo (name + size) straight from the Hugging Face API."""
    url = check_url(hf_api_url(repo))
    try:
        with _client(transport, timeout=20) as client:
            response = client.get(url)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise LlamaError("ไม่พบ repo นี้บน Hugging Face") from None
        raise LlamaError(f"Hugging Face ตอบ HTTP {exc.response.status_code}") from None
    except httpx.RequestError:
        raise LlamaError("เชื่อมต่อ Hugging Face ไม่ได้ ตรวจอินเทอร์เน็ต/ไฟร์วอลล์") from None
    except ValueError:
        raise LlamaError("Hugging Face ตอบข้อมูลที่อ่านไม่ได้") from None
    files = []
    for item in data.get("siblings") or []:
        name = str(item.get("rfilename", "")) if isinstance(item, dict) else ""
        if not name.lower().endswith(".gguf"):
            continue
        try:
            name = validate_filename(name)
        except LlamaError:
            continue
        files.append({"name": name, "size": int(item.get("size") or 0),
                      "mmproj": "mmproj" in name.lower()})
    files.sort(key=lambda entry: (entry["mmproj"], entry["name"]))
    return files


def pick_quant_files(files: list[dict], quant: str) -> tuple[dict | None, dict | None]:
    """Choose the model file for a quant plus the matching mmproj (for vision models)."""
    quant = (quant or "").upper()
    models = [entry for entry in files if not entry["mmproj"]]
    chosen = next((entry for entry in models if quant and quant in entry["name"].upper()), None)
    if chosen is None:
        chosen = next((entry for entry in models
                       if "mmproj" not in entry["name"] and "Q4_K_M" in entry["name"].upper()),
                      None) or (models[0] if models else None)
    projectors = [entry for entry in files if entry["mmproj"]]
    mmproj = next((entry for entry in projectors if "f16" in entry["name"].lower()), None) or \
        (projectors[0] if projectors else None)
    return chosen, mmproj


def disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(str(Path(path).anchor or "."))
    except OSError:
        return 0.0
    return usage.free / (1 << 30)


def ensure_space(target: Path, needed_bytes: int, extra_gb: float = MIN_FREE_GB) -> None:
    needed = needed_bytes / (1 << 30)
    free = disk_free_gb(target)
    if free and free < needed + extra_gb:
        raise LlamaError(f"พื้นที่ไม่พอ: ต้องใช้ประมาณ {needed:.1f} GB "
                         f"และเหลือ {free:.1f} GB (สงวน {extra_gb:.0f} GB ให้ระบบ)")


# --- downloader -------------------------------------------------------------------------


@dataclass
class DownloadResult:
    path: Path
    bytes_written: int
    complete: bool
    sha256: str = ""


class Downloader:
    """Streaming HTTPS download with resume, progress and honest size verification."""

    def __init__(self, transport=None, chunk: int = CHUNK, max_gb: float = MAX_DOWNLOAD_GB):
        self.transport = transport
        self.chunk = chunk
        self.max_bytes = int(max_gb * (1 << 30))

    def fetch(self, url: str, target: Path, progress=None, cancel=None,
              sha256: str = "", extra_gb: float = MIN_FREE_GB) -> DownloadResult:
        url = check_url(url)
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        if target.exists():
            digest = sha256_file(target) if sha256 else ""
            if not sha256 or digest == sha256:
                return DownloadResult(target, target.stat().st_size, True, digest)
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Accept": "application/octet-stream"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with _client(self.transport, timeout=60) as client:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 416:
                        # The remote is not larger than what we already have: restart cleanly.
                        offset = 0
                        part.unlink(missing_ok=True)
                        raise LlamaError("ไฟล์บางส่วนไม่ตรงกับต้นทาง กรุณากดดาวน์โหลดอีกครั้ง")
                    if response.status_code not in (200, 206):
                        raise LlamaError(f"ดาวน์โหลดไม่ได้: HTTP {response.status_code}")
                    if response.status_code == 200 and offset:
                        offset = 0            # server ignored Range: start over
                    length = int(response.headers.get("content-length") or 0)
                    total = length + offset
                    if total > self.max_bytes:
                        raise LlamaError(f"ไฟล์ใหญ่เกินเพดาน {self.max_bytes / (1 << 30):.0f} GB "
                                         "(แก้ได้ที่ค่าเพดานในโค้ด/ตั้งค่า)")
                    ensure_space(target, max(0, total - offset), extra_gb)
                    written = offset
                    mode = "ab" if offset else "wb"
                    with open(part, mode) as handle:
                        for block in response.iter_bytes(self.chunk):
                            if cancel is not None and cancel():
                                raise LlamaError("ยกเลิกการดาวน์โหลดแล้ว (ไฟล์บางส่วนถูกเก็บไว้)")
                            handle.write(block)
                            written += len(block)
                            if progress is not None:
                                progress(written, total)
                    if length and written != total:
                        raise LlamaError(f"ไฟล์ไม่ครบ ({written}/{total} ไบต์) "
                                         "กดดาวน์โหลดอีกครั้งเพื่อเรียนต่อจากเดิม")
        except httpx.HTTPError as exc:
            raise LlamaError(f"เชื่อมต่อต้นทางไม่ได้: {exc.__class__.__name__}") from None
        digest = sha256_file(part) if sha256 else ""
        if sha256 and digest != sha256:
            part.unlink(missing_ok=True)
            raise LlamaError("ค่า SHA-256 ไม่ตรงกับที่ประกาศไว้: ลบไฟล์และลองใหม่")
        part.replace(target)
        return DownloadResult(target, target.stat().st_size, True, digest)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# --- llama.cpp server binary ----------------------------------------------------------------


def llama_assets(transport=None) -> list[dict]:
    """Windows/Linux builds published by the llama.cpp release page (via the GitHub API)."""
    url = check_url(GH_RELEASES.format(repo="ggml-org/llama.cpp"))
    try:
        with _client(transport, timeout=20) as client:
            response = client.get(url, headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        releases = response.json()
    except httpx.HTTPStatusError as exc:
        raise LlamaError(f"GitHub ตอบ HTTP {exc.response.status_code} "
                         "(อาจติด rate limit ให้ลองใหม่ภายหลัง)") from None
    except httpx.RequestError:
        raise LlamaError("เชื่อมต่อ GitHub ไม่ได้ ตรวจอินเทอร์เน็ต/ไฟร์วอลล์") from None
    except ValueError:
        raise LlamaError("GitHub ตอบข้อมูลที่อ่านไม่ได้") from None
    assets = []
    for release in releases[:3] if isinstance(releases, list) else []:
        for asset in release.get("assets") or []:
            name = str(asset.get("name", ""))
            if not name.endswith(".zip"):
                continue
            lower = name.lower()
            if sys.platform == "win32" and "win" not in lower:
                continue
            if sys.platform != "win32" and "win" in lower:
                continue
            assets.append({"name": name, "url": str(asset.get("browser_download_url", "")),
                           "size": int(asset.get("size") or 0),
                           "tag": str(release.get("tag_name", ""))})
    assets.sort(key=lambda item: (("cuda" in item["name"].lower()), item["name"]))
    return assets[:12]


def extract_binary(zip_path: Path, destination: Path, keep: tuple[str, ...] = ()) -> list[Path]:
    """Extract llama-server (plus its DLLs) with strict path checks. Returns what was written."""
    import zipfile

    destination.mkdir(parents=True, exist_ok=True)
    wanted = (".dll", ".so", ".dylib", *keep)
    written: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if name.endswith("/") or name.startswith("/") or ".." in name.split("/"):
                continue
            base = name.rsplit("/", 1)[-1]
            lower = base.lower()
            if not (lower.startswith("llama-server") or lower.endswith(wanted)):
                continue
            target = destination / base
            with archive.open(info) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
            written.append(target)
    if not written:
        raise LlamaError("ไม่พบ llama-server ในไฟล์ zip ที่ดาวน์โหลดมา")
    if os.name != "nt":
        for path in written:
            if path.name.lower().startswith("llama-server"):
                path.chmod(0o755)
    return written


def find_binary(*candidates: str | Path) -> Path | None:
    """First existing llama-server among the given paths and the app's own folder."""
    names = ("llama-server.exe", "llama-server") if os.name == "nt" else ("llama-server",)
    for candidate in list(candidates) + [Path(p) for p in (os.environ.get("PATH", "") or "").split(os.pathsep) if p]:
        path = Path(str(candidate))
        if path.is_dir():
            for name in names:
                if (path / name).exists():
                    return path / name
        elif path.exists() and path.name.lower().startswith(("llama-server", "llama_server")):
            return path
    return None


# --- runtime --------------------------------------------------------------------------------


@dataclass
class ServerState:
    running: bool = False
    starting: bool = False
    port: int = 8081
    model: str = ""
    error: str = ""
    started_at: float = 0.0
    log: list[str] = field(default_factory=list)


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def build_args(binary: Path, model: Path, *, port: int = 8081, ctx: int = 8192,
               gpu_layers: int = 0, threads: int = 0, mmproj: Path | None = None,
               extra: str = "") -> list[str]:
    """Command line for llama-server. Loopback only; no shell, no wildcards."""
    args = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(max(2048, min(131072, ctx))), "--n-gpu-layers",
            str(max(0, min(999, gpu_layers))), "--no-webui"]
    if threads:
        args += ["--threads", str(max(1, min(64, threads)))]
    if mmproj is not None:
        args += ["--mmproj", str(mmproj)]
    if extra.strip():
        args += [part for part in extra.strip().split() if not part.startswith(("-h", "--help"))][:20]
    return args


class LlamaRuntime:
    """Owns the llama-server child process: start, poll, stop, and keep its log in RAM."""

    def __init__(self, binary: Path | None = None, base_url: str = "http://127.0.0.1:8081/v1",
                 transport=None):
        self.binary = Path(binary) if binary else None
        self.base_url = base_url
        self.transport = transport
        self.state = ServerState()
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------------------
    def start(self, model: Path, *, port: int = 8081, ctx: int = 8192, gpu_layers: int = 0,
              threads: int = 0, mmproj: Path | None = None, extra: str = "") -> str:
        if self.state.running or (self._process and self._process.poll() is None):
            return self.base_url
        if self.binary is None or not Path(self.binary).exists():
            raise LlamaError("ยังไม่พบไฟล์ llama-server: ดาวน์โหลดหรือเลือกไฟล์ก่อน")
        if not Path(model).exists():
            raise LlamaError("ไม่พบไฟล์โมเดล: ดาวน์โหลดให้เสร็จก่อน")
        if not port_free(port):
            raise LlamaError(f"พอร์ต {port} ถูกใช้อยู่ เลือกพอร์ตอื่น")
        args = build_args(self.binary, Path(model), port=port, ctx=ctx, gpu_layers=gpu_layers,
                          threads=threads, mmproj=mmproj, extra=extra)
        creation = 0
        if os.name == "nt":
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding="utf-8", errors="replace",
                                             bufsize=1, creationflags=creation)
        except OSError as exc:
            raise LlamaError(f"เริ่ม llama-server ไม่ได้: {exc.__class__.__name__}") from None
        self.state = ServerState(running=False, starting=True, port=port, model=Path(model).name,
                                 started_at=time.monotonic())
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self._reader = threading.Thread(target=self._pump_log, daemon=True)
        self._reader.start()
        return self.base_url

    def _pump_log(self):
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            with self._lock:
                self.state.log.append(line.rstrip()[:400])
                del self.state.log[:-300]

    def probe(self, timeout: float = 0.7) -> bool:
        """One *non-terminal* health check, safe to call from the UI thread.

        While the model is still loading llama.cpp answers 503 (or nothing at all): that is
        normal, so a miss here must never be recorded as an error. Returns True once healthy.
        """
        if not self.check_process():           # the child died → that *is* a real error
            return False
        try:
            with _client(self.transport, timeout=timeout) as client:
                response = client.get(self.base_url.replace("/v1", "") + "/health")
        except httpx.HTTPError:
            return False
        if response.status_code < 500:
            self.state.running, self.state.starting, self.state.error = True, False, ""
            return True
        return False                       # 503 = still loading, not a failure

    def check_process(self) -> bool:
        """True while the child may still work; False records *why* it stopped."""
        process = self._process
        if process is None or process.poll() is None:
            return True
        self.state.starting = self.state.running = False
        tail = " ".join(self.state.log[-3:])[:300]
        self.state.error = f"llama-server ปิดตัวเอง (exit {process.returncode}) {tail}"
        return False

    def timed_out(self):
        """Called by the UI once the generous startup window has really passed."""
        if self.state.running:
            return
        self.state.starting = False
        self.state.error = ("รอ llama-server โหลดโมเดลนานเกินกำหนด — โมเดลอาจใหญ่เกินแรม "
                            "ลอง quant เล็กลงหรือลด context (ดู log ด้านล่าง)")

    def poll(self, timeout: float = 60.0) -> bool:
        """Blocking wait until /health says the model is loaded (or report why it is not).

        The app itself never calls this from the UI thread — ``probe`` + timers do that.
        """
        deadline = time.monotonic() + max(1.0, timeout)
        while time.monotonic() < deadline:
            if self.probe(timeout=min(4.0, max(0.5, deadline - time.monotonic()))):
                return True
            if self.state.error:               # the process died: nothing to wait for
                return False
            time.sleep(0.5)
        self.timed_out()
        return False

    def stop(self, grace: float = 6.0) -> bool:
        process = self._process
        if process is None or process.poll() is not None:
            self.state.running = self.state.starting = False
            self._process = None
            return True
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        self._process = None
        self.state.running = self.state.starting = False
        with self._lock:
            self.state.log.append("— หยุด llama-server แล้ว —")
        return True

    def models(self) -> list[str]:
        from .providers import list_models
        from .models import Settings
        settings = Settings(provider="openai", endpoint=self.base_url.replace("/v1", "") + "/v1",
                            model="local", vision=False)
        try:
            return list_models(settings, transport=self.transport)
        except (ProviderError, ValueError):
            return []

    def status_text(self) -> str:
        if self.state.running:
            return (f"ทำงานอยู่ • {self.state.model} • {self.base_url} • "
                    f"{int(time.monotonic() - self.state.started_at)} วินาที")
        if self.state.starting:
            return f"กำลังโหลดโมเดล… ({self.state.model})"
        if self.state.error:
            return "มีปัญหา: " + self.state.error
        return "ยังไม่ทำงาน"


def models_dir(store_root: Path) -> Path:
    return Path(store_root) / "llama" / "models"


def bin_dir(store_root: Path) -> Path:
    return Path(store_root) / "llama" / "bin"


def installed_models(store_root: Path) -> list[dict]:
    """GGUF files already on disk, with the projector that matches each one if present."""
    folder = models_dir(store_root)
    if not folder.is_dir():
        return []
    entries = []
    for path in sorted(folder.glob("*.gguf")):
        entry = {"path": str(path), "name": path.name, "size": path.stat().st_size,
                 "mmproj": "mmproj" in path.name.lower()}
        entries.append(entry)
    models = [entry for entry in entries if not entry["mmproj"]]
    for entry in models:
        entry["projector"] = next((other["path"] for other in entries
                                   if other["mmproj"]), "")
    return models


def settings_guess(entries: list[dict], repo: str = "") -> dict:
    """Sensible llama-server flags for the downloaded files (honest about being a guess)."""
    total_gb = sum(entry["size"] for entry in entries) / (1 << 30)
    if total_gb >= 20:
        ctx, gpu = 4096, 20
    elif total_gb >= 8:
        ctx, gpu = 8192, 32
    else:
        ctx, gpu = 8192, 0
    return {"ctx": ctx, "gpu_layers": gpu, "note": "ค่าตั้งต้นจากการประมาณขนาดไฟล์ "
                                                   "ปรับได้ตาม RAM/VRAM ของคุณ"}


def describe_json(payload: str) -> str:
    """Small helper for the log view: pretty JSON or the raw text, never a crash."""
    try:
        return json.dumps(json.loads(payload), ensure_ascii=False, indent=2)[:4000]
    except ValueError:
        return str(payload)[:4000]
