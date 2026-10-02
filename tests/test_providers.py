import json

import httpx
import pytest
from PIL import Image

from screen_thai.models import Block, Settings
from screen_thai.providers import AIClient, ProviderError, validate_endpoint


@pytest.mark.parametrize("endpoint", ["http://evil.example/v1", "file:///tmp/key", "https://user:pass@api.example",
                                     "https://example.com?key=secret", "ftp://example.com"])
def test_reject_insecure_endpoint(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


@pytest.mark.parametrize("endpoint", ["http://localhost:11434", "http://127.0.0.1:1234/v1",
                                     "http://[::1]:11434", "https://api.example/v1"])
def test_accept_secure_endpoint(endpoint):
    assert validate_endpoint(endpoint + "/") == endpoint


@pytest.mark.parametrize("provider", ["ollama", "openai"])
@pytest.mark.parametrize("vision", [True, False])
def test_payload_and_parse(provider, vision):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "vision-model"
        message = payload["messages"][-1]
        if provider == "ollama":
            assert request.url.path == "/api/chat"
            assert bool(message.get("images")) is vision
        else:
            assert request.url.path == "/v1/chat/completions"
            assert any(item["type"] == "image_url" for item in message["content"]) is vision
        assert request.headers["Authorization"] == "Bearer secret-test"
        content = json.dumps({"translations": [{"id": 0, "thai": "สวัสดี"}]})
        return httpx.Response(200, json={"message": {"content": content}} if provider == "ollama"
                              else {"choices": [{"message": {"content": content}}]})
    settings = Settings(provider=provider, endpoint="http://localhost:11434" + ("/v1" if provider == "openai" else ""),
                        model="vision-model", vision=vision)
    client = AIClient(settings, "secret-test", transport=httpx.MockTransport(handler))
    result = client.translate([Block(0, "Hello", (0, 0, 30, 20))], Image.new("RGB", (100, 100)), {})
    assert result.translations[0].thai == "สวัสดี"


def test_http_error_does_not_expose_body_or_key():
    def handler(request):
        return httpx.Response(401, text="your secret-key and private screen data")
    client = AIClient(Settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as error:
        client.translate([Block(0, "Hello", (0, 0, 20, 20))], Image.new("RGB", (20, 20)), {})
    assert "401" in str(error.value)
    assert "secret" not in str(error.value)


def test_timeout_has_actionable_message():
    def handler(request):
        raise httpx.ReadTimeout("secret-test", request=request)
    client = AIClient(Settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="timeout"):
        client.translate([Block(0, "Hello", (0, 0, 20, 20))], Image.new("RGB", (20, 20)), {})


def test_translate_sends_layout_faces_and_locked_voice():
    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured["payload"] = json.loads(body["messages"][-1]["content"])
        content = json.dumps({"translations": [{"id": 0, "thai": "สวัสดี", "speaker": "Alyssa"}]})
        return httpx.Response(200, json={"message": {"content": content}})

    from screen_thai.faces import FaceMemory
    from screen_thai.models import Block
    from screen_thai.vision import Appearance, ExpressionCues, FaceBox, FaceObservation
    memory = FaceMemory()
    memory.ensure("Alyssa", confirmed=True)
    memory.set_voice("Alyssa", self="แม่", address="ลูก", particles="ค่ะ", locked=True)
    blocks = [Block(0, "Alyssa", (120, 480, 160, 34)), Block(1, "私が守るわ。", (100, 520, 900, 150))]
    observation = FaceObservation(1, FaceBox(100, 300, 120, 120),
                                  ExpressionCues("ยิ้ม", 0.3, 0.1, 0.2, 0.6, 0.2, 2.0, 0.0, "landmarks"),
                                  Appearance("#111111", "#222222", "#333333", ("#111111",), 0.4,
                                             0.2, 0.1))
    observation.character = "Alyssa"
    frame = {"layout": [{"id": 1, "role": "dialogue"}], "faces": [
        {"track": "ใบหน้าที่ 1", "identified_as": "Alyssa", "identified_source": "ผู้ใช้ยืนยัน"}]}
    client = AIClient(Settings(), "k", transport=httpx.MockTransport(handler))
    result = client.translate(blocks, Image.new("RGB", (100, 100)), {"characters": {}}, frame, memory)
    assert result.translations[0].speaker == "Alyssa"
    payload = captured["payload"]
    assert payload["layout"][0]["role"] == "dialogue"
    assert payload["faces"][0]["identified_as"] == "Alyssa"
    card = payload["characters_confirmed_by_user"]["Alyssa"]
    assert card["locked_voice"]["self"] == "แม่" and card["locked_voice"]["particles"] == "ค่ะ"


def test_research_call_never_attaches_an_image():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": '{"title": "X"}'}})

    client = AIClient(Settings(), "k", transport=httpx.MockTransport(handler))
    text = client.research("prompt", "system")
    assert text.startswith("{")
    message = captured["payload"]["messages"][-1]
    assert "images" not in message
    assert captured["payload"]["messages"][0]["role"] == "system"


def test_invalid_model_json_becomes_actionable_error():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "not json at all"}})
    client = AIClient(Settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="JSON"):
        client.translate([Block(0, "Hello", (0, 0, 20, 20))], Image.new("RGB", (20, 20)), {})
