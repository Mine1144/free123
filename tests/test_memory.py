"""Translation memory: what may be reused, what must go back to the model, and glossary mining."""
from screen_thai.memory import (TranslationMemory, check_terms, glossary_text, key_for,
                                mine_candidates, parse_glossary, voice_signature)
from screen_thai.models import Block


def test_learning_and_reuse_requires_clean_lines():
    tm = TranslationMemory()
    assert tm.learn("Potion", "ยาเพิ่มพลัง") is True
    assert tm.get("Potion")["thai"] == "ยาเพิ่มพลัง"
    # A warned line is stored but never replayed automatically.
    tm.learn("Careful!", "ระวัง!", warned=True)
    assert tm.get("Careful!") is None
    assert tm.stats()["needs_review"] == 1


def test_short_or_empty_lines_are_not_memorised():
    tm = TranslationMemory()
    assert tm.learn("", "อะไร") is False
    assert tm.learn("a", "ก") is False
    assert tm.learn("Hello", "") is False


def test_repeated_line_keeps_single_entry_and_counts():
    tm = TranslationMemory()
    for _ in range(3):
        tm.learn("Good morning", "อรุณสวัสดิ์")
    assert len(tm.entries) == 1
    entry = tm.get("Good morning")
    assert entry["thai"] == "อรุณสวัสดิ์" and entry["hits"] == 3
    assert tm.stats()["reuses"] == 2


def test_changed_wording_overwrites_and_resets_hits():
    tm = TranslationMemory()
    tm.learn("Wait here", "รอที่นี่")
    tm.learn("Wait here", "รอตรงนี้ก่อน")
    assert tm.get("Wait here")["thai"] == "รอตรงนี้ก่อน"
    assert tm.stats()["reuses"] == 0


def test_locked_voice_change_sends_the_line_back_to_the_model():
    from screen_thai.faces import FaceMemory
    from screen_thai.layout import classify
    memory = FaceMemory()
    memory.ensure("Alyssa", confirmed=True)
    memory.set_voice("Alyssa", self="ฉัน", address="เธอ", particles="ค่ะ", locked=True)
    signature = voice_signature(memory.card("Alyssa"))
    tm = TranslationMemory()
    tm.learn("I will go.", "ฉันจะไป", speaker="Alyssa", voice=signature)
    blocks = [Block(0, "Alyssa", (120, 480, 160, 34)),
              Block(1, "I will go.", (100, 520, 600, 60))]
    hints = classify(blocks, (1280, 720))
    assert tm.lookup(blocks, hints, memory)[1]["speaker"] == "Alyssa"
    # The player edits the card: the stored wording may now be wrong, so it must not be replayed.
    memory.set_voice("Alyssa", self="ข้า", locked=True)
    assert tm.lookup(blocks, hints, memory) == {}
    # Without knowing who is speaking we cannot confirm the voice, so the model must answer again.
    assert tm.lookup([Block(1, "I will go.", (0, 0, 10, 10))], [], memory) == {}


def test_lookup_skips_name_tags_and_hud():
    from screen_thai.layout import classify
    tm = TranslationMemory()
    tm.learn("Alyssa", "อลิสซ่า")                  # the name tag itself
    tm.learn("Alyssa is here.", "อลิสซ่าอยู่ที่นี่")
    blocks = [Block(0, "Alyssa", (120, 480, 160, 34)),
              Block(1, "Alyssa is here.", (100, 520, 600, 60))]
    hints = classify(blocks, (1280, 720))
    hits = tm.lookup(blocks, hints)
    assert 0 not in hits and hits[1]["thai"] == "อลิสซ่าอยู่ที่นี่"


def test_prune_keeps_the_bounded_maximum():
    tm = TranslationMemory()
    for index in range(1200):
        tm.learn(f"Line number {index}", f"บรรทัด {index}")
    assert len(tm.entries) <= 800
    assert tm.to_dict()["entries"] == tm.entries


def test_round_trip_from_dict_ignores_junk():
    tm = TranslationMemory({"entries": {"OK": {"thai": "โอเค", "hits": "x"},
                                        "": {"thai": "ว่าง"}, "Bad": "not a dict"}})
    assert list(tm.entries) == ["OK"]
    assert tm.entries["OK"]["hits"] == 0


def test_source_key_is_normalised():
    assert key_for("  Hello   world ") == "Hello world"
    tm = TranslationMemory()
    tm.learn("Hello world", "สวัสดีชาวโลก")
    assert tm.get("Hello   world")["thai"] == "สวัสดีชาวโลก"


# --- glossary -------------------------------------------------------------------------

def test_parse_glossary_accepts_lines_commas_and_dict():
    assert parse_glossary("Ryza = ไรซ่า\nKiro: คิโร") == [("Ryza", "ไรซ่า"), ("Kiro", "คิโร")]
    assert parse_glossary("Ryza = ไรซ่า, Kiro = คิโร")[1] == ("Kiro", "คิโร")
    assert parse_glossary({"Ryza": "ไรซ่า"}) == [("Ryza", "ไรซ่า")]
    assert glossary_text([("A", "ก")]) == "A = ก"


def test_check_terms_warns_only_when_rendering_is_missing():
    glossary = "Ryza = ไรซ่า\nKiro = คิโร"
    assert check_terms("Ryza is here", "ไรซ่าอยู่ที่นี่", glossary) == []
    warnings = check_terms("Ryza is here", "เด็กสาวอยู่ที่นี่", glossary)
    assert warnings and "ไรซ่า" in warnings[0]
    # A term that is not in the source line must never be flagged.
    assert check_terms("He waits", "ไรซ่ารออยู่", glossary) == []
    assert check_terms("Kiro waits", "เขารออยู่", glossary) == [
        "คำศัพท์: “Kiro” ควรใช้ “คิโร” ตาม glossary"]


def test_mining_requires_repetition_and_agreement():
    pairs = [("Ryza is late.", "ไรซ่าสายแล้ว"),
             ("Ryza is here.", "ไรซ่าอยู่ที่นี่"),
             ("Kiro waits.", "คิโรรออยู่")]
    found = {item["source"]: item["thai"] for item in mine_candidates(pairs, min_count=2)}
    assert found.get("Ryza") == "ไรซ่า"
    assert "Kiro" not in found          # seen once: not a candidate yet


def test_mining_skips_known_terms_and_stopwords_only():
    pairs = [("Potion ready", "พร้อมแล้ว"), ("Potion done", "เสร็จแล้ว")]
    assert mine_candidates(pairs, "Potion = ยาเพิ่มพลัง", min_count=2) == []


def test_mining_confidence_grows_with_evidence():
    one = mine_candidates([("Ryza", "ไรซ่า")], min_count=1)[0]
    more = mine_candidates([("Ryza A", "ไรซ่า ก"), ("Ryza B", "ไรซ่า ข")], min_count=2)[0]
    assert more["confidence"] >= one["confidence"]
    assert more["count"] == 2 and more["evidence"]


def test_mining_rejects_a_chunk_that_appears_without_the_term():
    pairs = [("Ryza is late.", "ไรซ่าสายแล้ว"), ("Ryza is here.", "ไรซ่าอยู่ที่นี่")]
    # "แล้ว" also appears in an unrelated line, so it is not evidence for the term's rendering.
    found = mine_candidates(pairs, min_count=2, negative=["Kiro เสร็จแล้ว"])
    assert {item["source"]: item["thai"] for item in found}.get("Ryza") == "ไรซ่า"


def test_mining_can_be_disabled_by_known_term_only():
    pairs = [("Alyssa speaks", "อลิสซ่าพูด"), ("Alyssa smiles", "อลิสซ่ายิ้ม")]
    assert mine_candidates(pairs, "Alyssa = อลิสซ่า", min_count=2) == []
