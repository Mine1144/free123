"""Self-check: every result must state what was verified and what was not."""
from pathlib import Path

import httpx

from screen_thai import diagnostics
from screen_thai.models import Settings
from screen_thai.providers import ProviderError, list_models


def fake_store(tmp_path: Path):
    class Store:
        root = tmp_path / "ScreenThai"

        def load_profile(self, name):
            return {"name": name, "research": {"title": "Atelier"}}

        def load_memory(self, name):
            from screen_thai.faces import FaceMemory
            return FaceMemory()

    return Store()


def test_checks_report_ok_for_a_healthy_local_setup(tmp_path):
    settings = Settings(provider="ollama", endpoint="http://localhost:11434",
                        model="qwen2.5vl:7b", face_backend="off")
    checks = diagnostics.run_checks(settings, fake_store(tmp_path), profile_name="Atelier")
    by_name = {item["name"]: item for item in checks}
    assert by_name["โฟลเดอร์ข้อมูล"]["status"] == diagnostics.OK
    assert by_name["ปลายทาง AI"]["status"] == diagnostics.OK
    assert by_name["ใบหน้า/สีหน้า"]["status"] == diagnostics.SKIP
    assert by_name["เชื่อมต่อบริการ AI"]["status"] == diagnostics.SKIP
    assert by_name["เชื่อมต่อบริการ AI"]["hint"]
    assert diagnostics.summary(checks)["total"] == len(checks)


def test_external_http_endpoint_is_flagged_not_hidden(tmp_path):
    settings = Settings(provider="openai", endpoint="https://api.openai.com/v1", model="gpt-4o")
    checks = diagnostics.run_checks(settings, fake_store(tmp_path), profile_name="Game")
    by_name = {item["name"]: item for item in checks}
    assert by_name["ปลายทาง AI"]["status"] == diagnostics.OK


def test_missing_model_is_a_warning_with_a_next_step(tmp_path):
    checks = diagnostics.run_checks(Settings(model=""), fake_store(tmp_path), profile_name="Game")
    item = next(entry for entry in checks if entry["name"] == "ตั้งค่าโมเดล")
    assert item["status"] == diagnostics.WARN and "ชื่อโมเดล" in item["hint"]


def test_report_text_lists_every_check_with_an_icon(tmp_path):
    checks = diagnostics.run_checks(Settings(), fake_store(tmp_path), profile_name="Game")
    text = diagnostics.report_text(checks)
    for item in checks:
        assert item["name"] in text
    assert "รวม" in text and "ผ่าน" in text


def test_bad_url_fails_with_an_explanation(tmp_path):
    settings = Settings(endpoint="http://example.com/v1", model="m")
    checks = diagnostics.run_checks(settings, fake_store(tmp_path), profile_name="Game")
    item = next(entry for entry in checks if entry["name"] == "ปลายทาง AI")
    assert item["status"] == diagnostics.FAIL and "HTTPS" in item["detail"]


def test_unwritable_data_dir_fails_instead_of_crashing(tmp_path, monkeypatch):
    class Broken:
        root = Path("/proc/definitely/not/writable")

    checks = diagnostics.run_checks(Settings(model="m"), Broken(), profile_name="Game")
    item = next(entry for entry in checks if entry["name"] == "โฟลเดอร์ข้อมูล")
    assert item["status"] == diagnostics.FAIL and item["hint"]


# --- model discovery ------------------------------------------------------------------

def test_ollama_model_list_uses_tags_endpoint():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"models": [{"name": "qwen2.5vl:7b"}, {"name": "llama3:8b"}]})

    settings = Settings(provider="ollama", endpoint="http://localhost:11434")
    models = list_models(settings, transport=httpx.MockTransport(handler))
    assert models == ["llama3:8b", "qwen2.5vl:7b"]
    assert seen["url"].endswith("/api/tags")


def test_openai_model_list_uses_models_endpoint_and_key():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})

    settings = Settings(provider="openai", endpoint="https://api.example.com/v1")
    assert list_models(settings, transport=httpx.MockTransport(handler), key="k-123") == ["gpt-4o"]
    assert seen["auth"] == "Bearer k-123"


def test_model_list_errors_are_thai_and_do_not_leak_the_body():
    def handler(request):
        return httpx.Response(500, json={"error": "secret token abc"})

    settings = Settings(provider="openai", endpoint="https://api.example.com/v1")
    try:
        list_models(settings, transport=httpx.MockTransport(handler))
    except ProviderError as exc:
        assert "secret" not in str(exc) and "HTTP 500" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_healthy_probe_reports_installed_and_vision_models(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "qwen2.5vl:7b"}]})

    settings = Settings(provider="ollama", endpoint="http://localhost:11434",
                        model="qwen2.5vl:7b", vision=True)
    item = diagnostics.check_ai_reachable(settings, transport=httpx.MockTransport(handler))
    assert item["status"] == diagnostics.OK and "โมเดลภาพ" in item["detail"]


def test_probe_warns_when_the_configured_model_is_missing(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    settings = Settings(provider="ollama", endpoint="http://localhost:11434", model="qwen2.5vl:7b")
    item = diagnostics.check_ai_reachable(settings, transport=httpx.MockTransport(handler))
    assert item["status"] == diagnostics.WARN and "ไม่พบ" in item["detail"]
