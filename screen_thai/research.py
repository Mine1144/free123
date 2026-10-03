"""Pre-translation research: learn what the game is before translating a single line.

Everything here is opt-in and auditable:

* Sources are pages the user pasted, files the user opened, or search results from a provider
  the user configured with their own key. The app never scrapes silently.
* Only the query text (game title / alias) leaves the machine. Never a screenshot, never OCR
  text from the screen.
* Pages are fetched once, converted to plain text, capped, and shown in the UI. Every fact in
  the resulting brief carries the source it came from; nothing is stored as verified truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import ipaddress
import json
import re
import socket
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

MAX_SOURCE_CHARS = 20_000
MAX_QUERY_CHARS = 200
USER_AGENT = "ScreenThai/0.2 (research; contact: local user)"

SEARCH_PROVIDERS = {
    "searxng": {"method": "GET", "url": "{endpoint}/search?q={query}&format=json", "auth": "none"},
    "brave": {"method": "GET", "url": "{endpoint}/res/v1/web/search?q={query}&count={limit}",
              "auth": "X-Subscription-Token"},
    "serper": {"method": "POST", "url": "{endpoint}/search", "auth": "X-API-KEY"},
    "tavily": {"method": "POST", "url": "{endpoint}/search", "auth": "Bearer"},
    "custom": {"method": "GET", "url": "{endpoint}?q={query}", "auth": "Bearer"},
}
DEFAULT_ENDPOINTS = {
    "searxng": "http://localhost:8080",
    "brave": "https://api.search.brave.com",
    "serper": "https://google.serper.dev",
    "tavily": "https://api.tavily.com",
    "custom": "",
}


class ResearchError(Exception):
    pass


@dataclass(frozen=True)
class SourceDoc:
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "iframe"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth:
            text = data.strip()
            if text:
                self.parts.append(text)


def html_to_text(html: str, limit: int = MAX_SOURCE_CHARS) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html[:400_000])
    except Exception:
        pass
    text = re.sub(r"[ \t\u00a0]+", " ", "\n".join(parser.parts))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:limit]


def _host_is_public(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in ("localhost",) or host.endswith((".local", ".internal", ".home.arpa")):
        return False
    try:
        address = ipaddress.ip_address(host)
        return address.is_global
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        raise ResearchError("แปลงชื่อเว็บไม่ได้ ตรวจการเชื่อมต่ออินเทอร์เน็ต") from None
    for info in infos:
        try:
            if not ipaddress.ip_address(info[4][0]).is_global:
                return False
        except ValueError:
            return False
    return True


def validate_source_url(url: str) -> str:
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ResearchError("ต้องเป็นลิงก์ http/https เท่านั้น")
    if parts.username or parts.password:
        raise ResearchError("ห้ามใส่ชื่อผู้ใช้หรือรหัสผ่านในลิงก์")
    if parts.scheme == "http" and not _is_loopback(parts.hostname):
        raise ResearchError("ลิงก์ภายนอกต้องเป็น HTTPS")
    if not _host_is_public(parts.hostname):
        raise ResearchError("ไม่อนุญาตที่อยู่ภายในเครื่อง/เครือข่ายส่วนตัว")
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(20, connect=8), follow_redirects=False,
                        trust_env=False, headers={"User-Agent": USER_AGENT})


def fetch_page(url: str, client: httpx.Client | None = None) -> SourceDoc:
    safe = validate_source_url(url)
    owned = client is None
    client = client or _client()
    try:
        target = safe
        for _hop in range(4):
            response = client.get(target)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location", "")
                if not location:
                    break
                target = validate_source_url(str(httpx.URL(target).join(location)))
                continue
            response.raise_for_status()
            break
        else:
            raise ResearchError("ลิงก์เปลี่ยนเส้นทางหลายครั้งเกินไป")
        if "text/html" not in response.headers.get("content-type", "text/html"):
            raise ResearchError("ลิงก์นี้ไม่ใช่หน้าเว็บ HTML")
        title = ""
        match = re.search(r"<title[^>]*>(.*?)</title>", response.text[:20_000], re.S | re.I)
        if match:
            title = re.sub(r"\s+", " ", match.group(1)).strip()[:200]
        return SourceDoc(title or target, target, html_to_text(response.text))
    except httpx.HTTPStatusError as exc:
        raise ResearchError(f"เว็บตอบกลับ HTTP {exc.response.status_code}") from None
    except httpx.RequestError:
        raise ResearchError("เชื่อมต่อเว็บไม่ได้ ตรวจอินเทอร์เน็ตหรือที่อยู่") from None
    finally:
        if owned:
            client.close()


def sanitize_query(query: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", query)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ResearchError("ต้องมีชื่อเกมก่อนค้นเว็บ")
    return text[:MAX_QUERY_CHARS]


def _search_headers(provider: str, key: str) -> dict:
    spec = SEARCH_PROVIDERS.get(provider, {})
    if spec.get("auth") == "Bearer":
        return {"Authorization": f"Bearer {key}"} if key else {}
    if spec.get("auth") and spec["auth"] != "none":
        return {spec["auth"]: key} if key else {}
    return {}


def _hits_from_payload(payload) -> list[SearchHit]:
    if isinstance(payload, dict):
        for key in ("results", "organic", "data", "items", "web"):
            value = payload.get(key)
            if isinstance(value, dict):
                value = value.get("results")
            if isinstance(value, list):
                payload = value
                break
    if not isinstance(payload, list):
        return []
    hits = []
    for item in payload[:12]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or item.get("href") or "")
        title = str(item.get("title") or item.get("name") or url)[:200]
        snippet = str(item.get("content") or item.get("snippet") or item.get("description") or "")
        if url.startswith("http"):
            hits.append(SearchHit(title.strip(), url.strip(), re.sub(r"\s+", " ", snippet)[:400]))
    return hits


def search_web(query: str, provider: str, endpoint: str = "", key: str = "",
               limit: int = 5, client: httpx.Client | None = None) -> list[SearchHit]:
    spec = SEARCH_PROVIDERS.get(provider)
    if not spec or provider == "off":
        raise ResearchError("ยังไม่ได้เลือกผู้ให้บริการค้นเว็บ")
    query = sanitize_query(query)
    base = (endpoint or DEFAULT_ENDPOINTS.get(provider) or "").rstrip("/")
    if not base:
        raise ResearchError("ต้องระบุปลายทางของระบบค้นเว็บ")
    if not base.startswith(("http://", "https://")):
        raise ResearchError("ปลายทางต้องเริ่มด้วย http:// หรือ https://")
    if base.startswith("http://") and not _is_loopback(urlsplit(base).hostname or ""):
        raise ResearchError("ปลายทางภายนอกต้องเป็น HTTPS")
    if any(token in base for token in ("{query}", "{key}")):
        raise ResearchError("ห้ามใส่ query หรือ key ลงใน URL ให้แอปส่งให้เอง")
    url = spec["url"].format(endpoint=base, query=quote(query, safe=""), limit=max(1, min(10, limit)))
    body = None
    if spec["method"] == "POST":
        url = base + ("/search" if not base.endswith("/search") else "")
        if provider == "serper":
            body = {"q": query, "num": max(1, min(10, limit))}
        else:
            body = {"query": query, "max_results": max(1, min(10, limit))}
    owned = client is None
    client = client or _client()
    try:
        headers = _search_headers(provider, key)
        response = (client.post(url, json=body, headers=headers) if body is not None
                    else client.get(url, headers=headers))
        response.raise_for_status()
        hits = _hits_from_payload(response.json())
        if not hits:
            raise ResearchError("ค้นเจอแต่ไม่พบผลลัพธ์ที่อ่านได้ ลองเปลี่ยนคำค้นหรือผู้ให้บริการ")
        return hits[:limit]
    except httpx.HTTPStatusError as exc:
        hint = {401: "ตรวจ API key ของระบบค้นหา", 403: "บัญชีไม่มีสิทธิ์ค้น",
                429: "โควตาค้นหาเต็ม"}.get(exc.response.status_code, "ตรวจปลายทางและ key")
        raise ResearchError(f"ระบบค้นหาตอบ HTTP {exc.response.status_code}: {hint}") from None
    except httpx.RequestError:
        raise ResearchError("เชื่อมต่อระบบค้นหาไม่ได้") from None
    except ValueError:
        raise ResearchError("ระบบค้นหาตอบไม่ใช่ JSON ที่อ่านได้") from None
    finally:
        if owned:
            client.close()


RESEARCH_SYSTEM = """You are a game-localisation researcher. You receive a game title, aliases
and plain-text source pages. Produce a structured brief in Thai that a translator will use to
keep names and speech styles consistent.

Rules:
- Use ONLY the supplied source text. If something is not in the sources, do not invent it.
- Mark anything uncertain in "uncertain" instead of guessing.
- Never copy long passages; summarise in your own words.
- For every character, record name (as written in the source), role, personality, how they
  speak, and the exact short quote or heading that supports it.
- Name localisation: give the source spelling and a Thai transliteration with a reason.
- Honorific policy: explain how the source language marks politeness/relative age and how Thai
  should mirror it (พี่/น้อง, ครับ/ค่ะ, คำเรียก).
- Reply ONLY with valid JSON, no markdown:
{"title":"","aliases":[],"genre":"","setting":"","era":"","region":"","developer":"","publisher":"",
 "languages":[],"summary":"","tone":"","localization_style":"",
 "characters":[{"name":"","role":"","personality":"","speech_style":"","evidence":""}],
 "terms":[{"source":"","thai":"","reason":""}],
 "names":[{"source":"","thai":"","reason":""}],
 "honorific_policy":"","pronoun_notes":"","spoiler_notes":"","uncertain":[],
 "sources":[{"title":"","url":"","supports":""}]}"""

_BRIEF_LISTS = ("aliases", "languages", "characters", "terms", "names", "uncertain", "sources")
_BRIEF_TEXT = ("title", "genre", "setting", "era", "region", "developer", "publisher", "summary",
               "tone", "localization_style", "honorific_policy", "pronoun_notes", "spoiler_notes")


def parse_brief(content: str) -> dict:
    if len(content) > 200_000:
        raise ResearchError("คำตอบวิจัยยาวเกินกำหนด")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except ValueError:
        raise ResearchError("AI ตอบวิจัยไม่ใช่ JSON ตามรูปแบบ") from None
    if not isinstance(data, dict):
        raise ResearchError("AI ตอบวิจัยไม่ใช่ JSON object")
    brief: dict = {"verified": False}
    for key in _BRIEF_TEXT:
        value = data.get(key)
        if isinstance(value, str):
            brief[key] = value.strip()[:3000]
    for key in _BRIEF_LISTS:
        value = data.get(key)
        if isinstance(value, list):
            if key in ("characters", "terms", "names", "sources"):
                entries = []
                for item in value[:40]:
                    if not isinstance(item, dict):
                        continue
                    entries.append({k: str(v)[:600] for k, v in item.items()
                                    if k in ("name", "role", "personality", "speech_style", "evidence",
                                             "source", "thai", "reason", "title", "url", "supports")})
                brief[key] = entries
            else:
                brief[key] = [str(item)[:300] for item in value[:30]]
    return brief


def build_brief_prompt(game: str, docs: list[SourceDoc], hits: list[SearchHit] | None = None) -> str:
    payload = {
        "game_title": game[:200],
        "search_results": [{"title": hit.title, "url": hit.url, "snippet": hit.snippet}
                           for hit in (hits or [])][:10],
        "sources": [{"title": doc.title, "url": doc.url, "text": doc.text[:MAX_SOURCE_CHARS]}
                    for doc in docs][:6],
    }
    return json.dumps(payload, ensure_ascii=False)


def merge_brief(existing: dict | None, new: dict) -> dict:
    """Keep the user's edits; add anything new the research step found."""
    merged = dict(existing or {})
    for key, value in new.items():
        if key == "verified":
            continue
        if key == "sources":
            current = merged.get(key) if isinstance(merged.get(key), list) else []
            seen = {str(item.get("url", "")) for item in current if isinstance(item, dict)}
            merged[key] = current + [item for item in value
                                     if isinstance(item, dict) and str(item.get("url", "")) not in seen]
        elif key not in merged or not merged[key]:
            merged[key] = value
        elif isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = value + [item for item in merged[key] if item not in value]
    merged["verified"] = False if not existing else bool(existing.get("verified"))
    return merged


def brief_text(brief: dict | None, limit: int = 4000) -> str:
    """Compact Thai text handed to the translation prompt (bounded, disclaimer first)."""
    if not isinstance(brief, dict) or not brief:
        return ""
    lines = ["ข้อมูลนี้มาจากการค้นคว้า ยังไม่ยืนยัน: ถ้าขัดกับข้อความบนจอ ให้เชื่อข้อความบนจอ "
             "และห้ามแต่งข้อมูลที่ไม่มีในนี้"]
    user_notes = str(brief.get("user_notes", "")).strip()
    if user_notes:
        lines.append("ผู้ใช้ยืนยันเอง (สำคัญที่สุด): " + user_notes[:1500])
    for key, label, cap in (("title", "ชื่อเกม", 120), ("genre", "แนว", 120),
                            ("setting", "ฉาก/สถานที่", 200), ("era", "ยุคสมัย", 120),
                            ("region", "ภูมิภาค/วัฒนธรรม", 200), ("tone", "โทนเรื่อง", 200),
                            ("localization_style", "สไตล์การแปล", 300),
                            ("honorific_policy", "นโยบายคำเรียก/ความสุภาพ", 600),
                            ("pronoun_notes", "โน้ตสรรพนาม", 400)):
        value = str(brief.get(key, "")).strip()
        if value:
            lines.append(f"{label}: {value[:cap]}")
    characters = brief.get("characters")
    if isinstance(characters, list) and characters:
        entries = []
        for card in characters[:12]:
            if not isinstance(card, dict):
                continue
            bit = f"{card.get('name', '')} — {card.get('role', '')} {card.get('personality', '')}".strip()
            if card.get("speech_style"):
                bit += f" | วิธีพูด: {card['speech_style'][:120]}"
            if card.get("evidence"):
                bit += f" (หลักฐาน: {card['evidence'][:80]})"
            entries.append("• " + " ".join(bit.split())[:300])
        if entries:
            lines.append("ตัวละครจากแหล่งข้อมูล:\n" + "\n".join(entries))
    for key, label in (("names", "ชื่อเฉพาะที่เสนอ"), ("terms", "คำศัพท์ที่เสนอ")):
        entries = brief.get(key)
        if isinstance(entries, list) and entries:
            pairs = [f"{item.get('source', '')} → {item.get('thai', '')}"
                     for item in entries[:20] if isinstance(item, dict) and item.get("source")]
            if pairs:
                lines.append(f"{label}: " + "; ".join(pairs)[:600])
    summary = str(brief.get("summary", "")).strip()
    if summary:
        lines.append("เรื่องย่อ: " + summary[:800])
    spoiler = str(brief.get("spoiler_notes", "")).strip()
    if spoiler:
        lines.append("ข้อควรระวังสปอยล์: " + spoiler[:300])
    uncertain = brief.get("uncertain")
    if isinstance(uncertain, list) and uncertain:
        lines.append("ยังไม่ยืนยัน: " + "; ".join(str(item) for item in uncertain[:8])[:400])
    sources = brief.get("sources")
    if isinstance(sources, list) and sources:
        lines.append("แหล่งอ้างอิง: " + "; ".join(
            f"{item.get('title', '')} ({item.get('url', '')})" for item in sources[:6]
            if isinstance(item, dict))[:400])
    return "\n".join(lines)[:limit]
