import json

import httpx
import pytest

from screen_thai.research import (ResearchError, build_brief_prompt, brief_text, fetch_page,
                                  html_to_text, merge_brief, parse_brief, sanitize_query,
                                  search_web, validate_source_url)


def test_html_to_text_strips_scripts_and_keeps_content():
    html = ("<html><head><title>Game</title><style>body{}</style></head>"
            "<body><nav>menu</nav><h1>Atelier</h1><p>Ryza is an alchemist.</p>"
            "<script>alert('x')</script></body></html>")
    text = html_to_text(html)
    assert "Ryra" in text or "Ryza is an alchemist." in text
    assert "alert" not in text and "menu" not in text and "body{}" not in text


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "http://127.0.0.1/x",
                                 "http://localhost:8000/x", "http://example.com/x",
                                 "https://user:pass@example.com/x", "https://10.0.0.5/x"])
def test_reject_unsafe_source_urls(url):
    with pytest.raises(ResearchError):
        validate_source_url(url)


def test_https_public_url_is_accepted():
    assert validate_source_url("https://en.wikipedia.org/wiki/Atelier_Ryza") == \
        "https://en.wikipedia.org/wiki/Atelier_Ryza"


def test_fetch_page_follows_redirect_and_requires_html():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://example.com/article"})
        if request.url.path == "/article":
            return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                                  text="<html><title>Ryza</title><p>Alchemy student.</p></html>")
        return httpx.Response(200, headers={"content-type": "application/pdf"}, text="%PDF")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        doc = fetch_page("https://example.com/start", client)
        assert doc.title == "Ryza" and "Alchemy student." in doc.text
        with pytest.raises(ResearchError, match="HTML"):
            fetch_page("https://example.com/file", client)


def test_fetch_page_never_reveals_body_on_error():
    def handler(request):
        return httpx.Response(500, text="secret internal detail")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ResearchError) as error:
            fetch_page("https://example.com/x", client)
    assert "secret" not in str(error.value)


def test_searxng_search_sends_key_free_query():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"results": [
            {"title": "Ryza", "url": "https://example.com/ryza", "content": "alchemist"},
            {"title": "", "url": "not-a-url", "content": ""}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        hits = search_web("Atelier Ryza", "searxng", "http://localhost:8080", "", 5, client)
    assert len(hits) == 1 and hits[0].title == "Ryza"
    assert "format=json" in seen["url"] and "secret" not in seen["url"]


def test_brave_search_uses_header_not_url():
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"web": {"results": [
            {"title": "T", "url": "https://example.com/a", "description": "d"}]}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        hits = search_web("Ryza", "brave", "https://api.search.brave.com", "k-123", 3, client)
    assert hits and seen["headers"].get("x-subscription-token") == "k-123"


def test_search_rejects_key_inside_endpoint_and_plain_http():
    with pytest.raises(ResearchError):
        search_web("x", "custom", "https://example.com/search?key=abc")
    with pytest.raises(ResearchError):
        search_web("x", "searxng", "http://example.com:8080")


def test_sanitize_query_caps_and_strips_control_characters():
    query = "Atelier\nRyza\x00  2"
    assert sanitize_query(query) == "Atelier Ryza 2"
    with pytest.raises(ResearchError):
        sanitize_query("   ")


def test_parse_brief_is_strict_about_shape():
    content = json.dumps({
        "title": "Atelier Ryza", "characters": [{"name": "Ryza", "role": "alchemist",
                                                 "speech_style": "กันเอง", "evidence": "official page"},
                                                "not a dict", {"name": ""}],
        "names": [{"source": "ライザ", "thai": "ไรซ่า", "reason": "เสียงอ่าน"}],
        "uncertain": ["ยังไม่ยืนยันวันเกิด"], "unknown_extra": 42})
    brief = parse_brief(content)
    assert brief["title"] == "Atelier Ryza"
    assert len(brief["characters"]) == 2
    assert brief["verified"] is False
    assert "unknown_extra" not in brief
    with pytest.raises(ResearchError):
        parse_brief("not json")


def test_merge_brief_keeps_user_values_and_adds_sources():
    existing = {"title": "ชื่อที่ผู้ใช้แก้", "verified": True,
                "sources": [{"title": "old", "url": "https://a"}], "user_notes": "ห้ามแก้"}
    merged = merge_brief(existing, {"title": "ชื่อใหม่จาก AI", "genre": "RPG",
                                    "sources": [{"title": "new", "url": "https://b"}]})
    assert merged["title"] == "ชื่อที่ผู้ใช้แก้"
    assert merged["genre"] == "RPG"
    assert merged["user_notes"] == "ห้ามแก้"
    assert {source["url"] for source in merged["sources"]} == {"https://a", "https://b"}


def test_brief_text_is_bounded_and_marks_data_unverified():
    brief = {"title": "X", "characters": [{"name": "A", "role": "hero", "speech_style": "ห้วน",
                                           "evidence": "page"}],
             "summary": "ย" * 5000, "sources": [{"title": "t", "url": "https://u"}]}
    text = brief_text(brief, limit=800)
    assert len(text) <= 800
    assert "ยังไม่ยืนยัน" in text and "A" in text


def test_build_brief_prompt_includes_only_text_sources():
    from screen_thai.research import SourceDoc
    prompt = build_brief_prompt("Ryza", [SourceDoc("t", "https://u", "เนื้อเรื่อง")], [])
    payload = json.loads(prompt)
    assert payload["game_title"] == "Ryza"
    assert payload["sources"][0]["text"] == "เนื้อเรื่อง"
    assert "image" not in prompt and "base64" not in prompt
