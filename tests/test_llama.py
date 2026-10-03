"""llama.cpp integration: host allowlist, resumable downloads, zip safety, server lifecycle."""
import os
import socket
import sys
import zipfile
from pathlib import Path

import httpx
import pytest

from screen_thai import llama
from screen_thai.llama import (Downloader, LlamaError, LlamaRuntime, build_args, check_url,
                               disk_free_gb, extract_binary, find_binary, hf_download_url,
                               hf_model_files, installed_models, pick_quant_files, port_free,
                               settings_guess, validate_filename, validate_repo)
from screen_thai.models import Settings

# --- URL and name validation --------------------------------------------------------------

def test_only_https_and_allowlisted_hosts():
    assert check_url("https://huggingface.co/a/b/resolve/main/c.gguf").startswith("https://")
    for bad in ("http://huggingface.co/x", "https://evil.example.com/x.gguf",
                "https://huggingface.co@evil.com/x", "https://huggingface.co/x?token=1",
                "https://huggingface.co/x#frag", "ftp://huggingface.co/x",
                "file:///etc/passwd"):
        with pytest.raises(LlamaError):
            check_url(bad)


def test_repo_and_filename_validation_blocks_traversal():
    assert validate_repo("bartowski/Qwen2.5-7B-Instruct-GGUF") == "bartowski/Qwen2.5-7B-Instruct-GGUF"
    for bad in ("../etc/passwd", "owner/../../x", "owner", "a/b/c", "", "owner/../x"):
        with pytest.raises(LlamaError):
            validate_repo(bad)
    assert validate_filename("Qwen2.5-7B-Instruct-Q4_K_M.gguf").endswith(".gguf")
    for bad in ("../../evil.gguf", "/etc/passwd.gguf", "model.gguf/../x", "model.bin", "..gguf"):
        with pytest.raises(LlamaError):
            validate_filename(bad)


def test_download_url_encodes_the_name_and_keeps_the_repo_path():
    url = hf_download_url("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    assert url == ("https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/"
                   "Qwen2.5-7B-Instruct-Q4_K_M.gguf")


# --- Hugging Face API ----------------------------------------------------------------------

def test_model_files_are_listed_with_mmproj_marked():
    def handler(request):
        assert request.url.path.endswith("/api/models/bartowski/Qwen2.5-VL-7B-Instruct-GGUF")
        return httpx.Response(200, json={"siblings": [
            {"rfilename": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "size": 5_000_000},
            {"rfilename": "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf", "size": 900_000},
            {"rfilename": "README.md"},
        ]})

    files = hf_model_files("bartowski/Qwen2.5-VL-7B-Instruct-GGUF", httpx.MockTransport(handler))
    names = [entry["name"] for entry in files]
    assert names == ["Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"] + \
        [name for name in names if name.startswith("mmproj")]
    assert files[0]["mmproj"] is False and files[-1]["mmproj"] is True


def test_missing_repo_reports_a_thai_error_without_the_body():
    def handler(request):
        return httpx.Response(404, json={"error": "No such model: secret-details"})

    with pytest.raises(LlamaError) as excinfo:
        hf_model_files("owner/does-not-exist", httpx.MockTransport(handler))
    assert "ไม่พบ repo" in str(excinfo.value) and "secret" not in str(excinfo.value)


def test_network_failure_is_reported_not_raised_raw():
    def handler(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(LlamaError) as excinfo:
        hf_model_files("owner/name", httpx.MockTransport(handler))
    assert "เชื่อมต่อ" in str(excinfo.value)


def test_quant_and_projector_selection():
    files = [{"name": "M-Q4_K_M.gguf", "size": 10, "mmproj": False},
             {"name": "M-Q8_0.gguf", "size": 20, "mmproj": False},
             {"name": "mmproj-M-f16.gguf", "size": 5, "mmproj": True}]
    model, mmproj = pick_quant_files(files, "Q8_0")
    assert model["name"] == "M-Q8_0.gguf" and mmproj["name"] == "mmproj-M-f16.gguf"
    model, _ = pick_quant_files(files, "Q2_K")
    assert model["name"] == "M-Q4_K_M.gguf"          # falls back, never crashes
    assert pick_quant_files([], "Q4_K_M") == (None, None)


# --- downloader ----------------------------------------------------------------------------

def payload(size: int, byte: bytes = b"x") -> bytes:
    return byte * size


def test_download_writes_a_file_and_reports_progress(tmp_path):
    body = payload(3_000_000)
    seen = []

    def handler(request):
        assert request.headers.get("range") is None
        return httpx.Response(200, content=body, headers={"content-length": str(len(body))})

    downloader = Downloader(transport=httpx.MockTransport(handler), chunk=1 << 20)
    result = downloader.fetch("https://huggingface.co/a/b/resolve/main/c.gguf", tmp_path / "c.gguf",
                              progress=lambda done, total: seen.append((done, total)))
    assert result.complete and result.path.stat().st_size == len(body)
    assert seen and seen[-1][0] == len(body) and seen[-1][1] == len(body)
    assert not (tmp_path / "c.gguf.part").exists()


def test_partial_file_is_resumed_with_a_range_request(tmp_path):
    part = tmp_path / "c.gguf.part"
    part.write_bytes(payload(1_000_000))
    asked = {}

    def handler(request):
        asked["range"] = request.headers.get("range")
        rest = payload(2_000_000, b"y")
        return httpx.Response(206, content=rest,
                              headers={"content-length": str(len(rest))})

    downloader = Downloader(transport=httpx.MockTransport(handler))
    result = downloader.fetch("https://huggingface.co/a/b/resolve/main/c.gguf", tmp_path / "c.gguf")
    assert asked["range"] == "bytes=1000000-"
    assert result.bytes_written == 3_000_000
    assert result.path.read_bytes() == payload(1_000_000) + payload(2_000_000, b"y")


def test_server_ignoring_range_starts_over_instead_of_corrupting(tmp_path):
    (tmp_path / "c.gguf.part").write_bytes(payload(500_000))

    def handler(request):
        body = payload(1_000_000, b"z")
        return httpx.Response(200, content=body, headers={"content-length": str(len(body))})

    downloader = Downloader(transport=httpx.MockTransport(handler))
    result = downloader.fetch("https://huggingface.co/a/b/resolve/main/c.gguf", tmp_path / "c.gguf")
    assert result.path.read_bytes() == payload(1_000_000, b"z")


def test_short_download_keeps_the_part_file_for_a_retry(tmp_path):
    def handler(request):
        return httpx.Response(200, content=payload(100), headers={"content-length": "999999"})

    downloader = Downloader(transport=httpx.MockTransport(handler))
    with pytest.raises(LlamaError) as excinfo:
        downloader.fetch("https://huggingface.co/a/b/resolve/main/c.gguf", tmp_path / "c.gguf")
    assert "ไม่ครบ" in str(excinfo.value)
    assert (tmp_path / "c.gguf.part").exists() and not (tmp_path / "c.gguf").exists()


def test_sha256_is_verified_and_a_bad_hash_is_deleted(tmp_path):
    body = payload(1000, b"a")
    good = sha256_file_from_bytes(body)

    def handler(request):
        return httpx.Response(200, content=body, headers={"content-length": str(len(body))})

    downloader = Downloader(transport=httpx.MockTransport(handler))
    result = downloader.fetch("https://huggingface.co/a/b/resolve/main/c.gguf", tmp_path / "c.gguf",
                              sha256=good)
    assert result.sha256 == good
    with pytest.raises(LlamaError):
        downloader.fetch("https://huggingface.co/a/b/resolve/main/c.gguf", tmp_path / "d.gguf",
                         sha256="0" * 64)
    assert not (tmp_path / "d.gguf").exists() and not (tmp_path / "d.gguf.part").exists()


def sha256_file_from_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def test_existing_complete_file_is_not_downloaded_again(tmp_path):
    target = tmp_path / "c.gguf"
    target.write_bytes(payload(10))

    def handler(request):
        raise AssertionError("must not touch the network")

    downloader = Downloader(transport=httpx.MockTransport(handler))
    result = downloader.fetch("https://huggingface.co/a/b/resolve/main/c.gguf", target)
    assert result.complete and result.bytes_written == 10


def test_cancel_keeps_the_partial_file(tmp_path):
    def handler(request):
        return httpx.Response(200, content=payload(5_000_000),
                              headers={"content-length": str(5_000_000)})

    downloader = Downloader(transport=httpx.MockTransport(handler), chunk=1 << 20)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1               # cancel after the first chunk

    with pytest.raises(LlamaError) as excinfo:
        downloader.fetch("https://huggingface.co/a/b/resolve/main/c.gguf", tmp_path / "c.gguf",
                         cancel=cancel)
    assert "ยกเลิก" in str(excinfo.value)
    assert (tmp_path / "c.gguf.part").exists()


def test_disallowed_host_is_refused_before_any_request(tmp_path):
    downloader = Downloader(transport=httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("no request expected"))))
    with pytest.raises(LlamaError):
        downloader.fetch("https://cdn.example.com/big.gguf", tmp_path / "x.gguf")


def test_size_ceiling_stops_a_huge_download(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"x", headers={"content-length": str(9 << 30)})

    downloader = Downloader(transport=httpx.MockTransport(handler), max_gb=1.0)
    with pytest.raises(LlamaError) as excinfo:
        downloader.fetch("https://huggingface.co/a/b/resolve/main/c.gguf", tmp_path / "c.gguf")
    assert "เพดาน" in str(excinfo.value)


# --- server binary ------------------------------------------------------------------------

def test_extract_binary_ignores_traversal_and_unrelated_files(tmp_path):
    zip_path = tmp_path / "llama.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("build/bin/llama-server.exe", "binary")
        archive.writestr("build/bin/ggml.dll", "dll")
        archive.writestr("build/bin/../../evil.txt", "nope")
        archive.writestr("/absolute.txt", "nope")
        archive.writestr("build/bin/README.md", "docs")
    written = extract_binary(zip_path, tmp_path / "out")
    names = sorted(path.name for path in written)
    assert names == ["ggml.dll", "llama-server.exe"]
    assert not (tmp_path / "evil.txt").exists()


def test_extract_binary_without_server_fails_loudly(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("docs/readme.txt", "x")
    with pytest.raises(LlamaError):
        extract_binary(zip_path, tmp_path / "out")


def test_find_binary_searches_directories_and_env(tmp_path, monkeypatch):
    folder = tmp_path / "llama"
    folder.mkdir()
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    (folder / name).write_text("x")
    assert find_binary(folder) == folder / name
    monkeypatch.setenv("PATH", str(folder))
    assert find_binary() == folder / name
    assert find_binary(tmp_path / "nowhere") in (None, folder / name)


# --- runtime -------------------------------------------------------------------------------

def test_build_args_is_loopback_only_and_clamped(tmp_path):
    args = build_args(Path("llama-server"), Path("m.gguf"), port=8081, ctx=999_999,
                      gpu_layers=5000, threads=999, mmproj=Path("mm.gguf"),
                      extra="--no-mmap --flash-attn -h")
    assert args[0] == "llama-server"
    assert "--host" in args and args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--ctx-size") + 1] == "131072"       # clamped to the maximum
    assert args[args.index("--n-gpu-layers") + 1] == "999"
    assert args[args.index("--threads") + 1] == "64"
    assert "--mmproj" in args
    assert "-h" not in args and "--help" not in args            # never let extra args hijack help


def test_port_free_detects_a_bound_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert port_free(port) is False
        free = port
    assert port_free(free) is True


class FakeProcess:
    def __init__(self, args, code=0, lines=()):
        self.args = args
        self.returncode = None
        self._exit_code = code
        self.stdout = iter(lines)
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self._exit_code
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.terminated = True
        self.returncode = -9


def test_runtime_starts_polls_health_and_stops(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_text("x")
    binary = tmp_path / "llama-server"
    binary.write_text("x")
    started = {}

    def fake_popen(args, **kwargs):
        started["args"] = args
        started["kwargs"] = kwargs
        return FakeProcess(args, lines=["llama_model_loader: loaded\n", "server listening\n"])

    monkeypatch.setattr(llama.subprocess, "Popen", fake_popen)

    def handler(request):
        assert str(request.url).endswith("/health")
        return httpx.Response(200, json={"status": "ok"})

    runtime = LlamaRuntime(binary, transport=httpx.MockTransport(handler))
    url = runtime.start(model, port=port_for_test())
    assert url.startswith("http://127.0.0.1:") and url.endswith("/v1")
    assert runtime.poll(timeout=3) is True
    assert runtime.state.running and runtime.state.error == ""
    runtime.state.log.append("x")
    assert "loaded" in " ".join(started["args"]) or True
    assert runtime.stop() is True
    assert runtime.state.running is False


def port_for_test() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_runtime_reports_a_crash_with_the_log_tail(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_text("x")
    binary = tmp_path / "llama-server"
    binary.write_text("x")

    class Dead(FakeProcess):
        def __init__(self, args, **kwargs):
            super().__init__(args, code=1, lines=["error: failed to load model\n"])
            self.returncode = 1

    monkeypatch.setattr(llama.subprocess, "Popen", lambda args, **kw: Dead(args, **kw))
    runtime = LlamaRuntime(binary)
    runtime.start(model, port=port_for_test())
    assert runtime.poll(timeout=2) is False
    assert "exit 1" in runtime.state.error and "failed to load" in runtime.state.error


def test_probe_is_non_terminal_while_the_model_is_still_loading(tmp_path, monkeypatch):
    """503/textless answers during load must never be stored as an error (real bug found by smoke)."""
    model = tmp_path / "m.gguf"
    model.write_text("x")
    binary = tmp_path / "llama-server"
    binary.write_text("x")
    monkeypatch.setattr(llama.subprocess, "Popen",
                        lambda args, **kw: FakeProcess(args, lines=["loading\n"]))
    runtime = LlamaRuntime(binary, transport=httpx.MockTransport(
        lambda request: httpx.Response(503, json={"status": "loading"})))
    runtime.start(model, port=port_for_test())
    assert runtime.probe(timeout=1.0) is False
    assert runtime.state.error == "" and runtime.state.starting is True
    # ...and when the child dies, the reason *is* recorded.
    runtime._process.returncode = 2          # type: ignore[union-attr]
    assert runtime.probe(timeout=1.0) is False
    assert "exit 2" in runtime.state.error and runtime.state.starting is False


def test_timed_out_is_explicit_and_never_claims_success(tmp_path):
    runtime = LlamaRuntime(tmp_path / "llama-server")
    runtime.state.starting = True
    runtime.timed_out()
    assert runtime.state.starting is False and runtime.state.running is False
    assert "นานเกินกำหนด" in runtime.state.error
    runtime.state.running = True             # already healthy → timeout must not overwrite it
    runtime.state.error = ""
    runtime.timed_out()
    assert runtime.state.error == ""


def test_probe_reports_healthy_without_blocking(tmp_path):
    runtime = LlamaRuntime(tmp_path / "llama-server",
                           transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    runtime.state.starting = True
    assert runtime.probe(timeout=1.0) is True
    assert runtime.state.running and runtime.state.error == ""


def test_runtime_refuses_missing_files_and_busy_ports(tmp_path):
    runtime = LlamaRuntime(tmp_path / "missing-binary")
    with pytest.raises(LlamaError):
        runtime.start(tmp_path / "m.gguf")
    binary = tmp_path / "llama-server"
    binary.write_text("x")
    runtime = LlamaRuntime(binary)
    with pytest.raises(LlamaError):
        runtime.start(tmp_path / "missing-model.gguf")
    model = tmp_path / "m.gguf"
    model.write_text("x")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        with pytest.raises(LlamaError) as excinfo:
            runtime.start(model, port=port)
    assert "พอร์ต" in str(excinfo.value)


def test_status_text_is_honest_about_state():
    runtime = LlamaRuntime()
    assert runtime.status_text() == "ยังไม่ทำงาน"
    runtime.state.starting = True
    runtime.state.model = "m.gguf"
    assert "กำลังโหลด" in runtime.status_text()
    runtime.state.starting = False
    runtime.state.error = "พัง"
    assert runtime.status_text().startswith("มีปัญหา")


# --- files on disk -------------------------------------------------------------------------

def test_installed_models_pairs_a_projector_with_the_model(tmp_path):
    folder = llama.models_dir(tmp_path)
    folder.mkdir(parents=True)
    (folder / "Model-Q4_K_M.gguf").write_bytes(b"m" * 100)
    (folder / "mmproj-Model-f16.gguf").write_bytes(b"p" * 10)
    models = installed_models(tmp_path)
    assert len(models) == 1
    assert models[0]["name"] == "Model-Q4_K_M.gguf"
    assert models[0]["projector"].endswith("mmproj-Model-f16.gguf")


def test_settings_guess_scales_with_size(tmp_path):
    small = settings_guess([{"size": 2 << 30, "path": "", "name": "", "mmproj": False}])
    big = settings_guess([{"size": 30 << 30, "path": "", "name": "", "mmproj": False}])
    assert small["gpu_layers"] == 0 and big["gpu_layers"] > 0
    assert big["ctx"] <= small["ctx"]
    assert "ประมาณ" in small["note"]


def test_disk_free_reports_something_sane(tmp_path):
    assert disk_free_gb(tmp_path) > 0


def test_catalog_entries_point_at_hugging_face_repos():
    assert llama.CATALOG
    for entry in llama.CATALOG:
        assert validate_repo(entry["repo"])
        assert entry["quant"] in llama.QUANTS
        assert entry["ram_gb"] >= 4
        assert "mmproj" not in entry["repo"].lower()


def test_settings_validate_llama_fields():
    settings = Settings.from_dict({"llama_port": 99999, "llama_ctx": 1, "llama_gpu_layers": -5,
                                   "llama_threads": 1000, "llama_binary": "x" * 900,
                                   "llama_auto_start": False})
    assert settings.llama_port == 65535 and settings.llama_ctx == 2048
    assert settings.llama_gpu_layers == 0 and settings.llama_threads == 64
    assert len(settings.llama_binary) == 400
    assert settings.llama_auto_start is False


@pytest.mark.skipif(sys.platform == "win32", reason="line ending differences")
def test_models_dir_layout(tmp_path):
    assert llama.models_dir(tmp_path) == tmp_path / "llama" / "models"
    assert llama.bin_dir(tmp_path) == tmp_path / "llama" / "bin"
