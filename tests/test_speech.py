from screen_thai.speech import (analyse_source, check_translation, detect_language, plan_speech,
                                relation_type)


def test_detects_japanese_honorific_and_politeness():
    cues = analyse_source("先輩、これは私の本です。")
    assert cues.language == "ja"
    assert "先輩" in cues.honorifics
    assert cues.politeness in ("polite", "neutral", "formal")
    assert "先輩" in cues.honorifics
    assert any("รุ่นพี่" in note for note in cues.notes)
    assert any("ผู้พูด" in pronoun for pronoun in cues.pronouns)


def test_detects_female_and_male_pronoun_hints():
    assert analyse_source("あたし、行くね").gender_hint == "female"
    assert analyse_source("俺が守る").gender_hint == "male"
    assert analyse_source("I will go").gender_hint == "unknown"


def test_rough_speech_is_kept_rough():
    cues = analyse_source("Shut up, you bastard!")
    assert cues.politeness == "rough"
    plan = plan_speech({}, {}, cues)
    assert plan.particles == "วะ"
    assert plan.register == "สนิทหยาบ"


def test_sibling_plan_without_confirmed_order_avoids_asserting_seniority():
    speaker = {"relation": "พี่น้อง"}
    plan = plan_speech(speaker, {}, analyse_source("お兄ちゃん、待って"))
    assert plan.relation == "พี่น้อง"
    assert any("อาวุโส" in reason for reason in plan.reasons)
    older = plan_speech({**speaker, "age_gap": "older"}, {}, analyse_source("お兄ちゃん"))
    assert older.self_pronoun == "พี่" and older.address == "น้อง"
    younger = plan_speech({**speaker, "age_gap": "younger"}, {}, analyse_source("ごめんね"))
    assert younger.self_pronoun == "หนู" and younger.address == "พี่"


def test_locked_voice_card_wins_over_ai_and_source():
    speaker = {"voice": {"self": "ข้า", "address": "เจ้า", "particles": "วะ", "register": "สนิทหยาบ",
                         "locked": True}, "confirmed": True}
    plan = plan_speech(speaker, {}, analyse_source("I would be honoured, sir."))
    assert plan.self_pronoun == "ข้า" and plan.address == "เจ้า"
    assert plan.particles == "วะ"
    assert set(plan.locked) >= {"สรรพนามแทนตัวเอง", "สรรพนามเรียกอีกฝ่าย"}
    assert "ล็อกโดยผู้ใช้" in plan.describe()


def test_unconfirmed_gender_never_forces_thai_particles():
    plan = plan_speech({"gender": "male", "gender_source": "ai"}, {}, analyse_source("hello"))
    assert "ครับ" not in plan.particles
    assert any("ยังไม่ยืนยัน" in reason for reason in plan.reasons)
    confirmed = plan_speech({"gender": "female", "gender_source": "user"}, {}, analyse_source("hi"))
    assert confirmed.particles == "ค่ะ"


def test_check_translation_flags_rough_vs_locked_polite_and_gender_mismatch():
    plan = plan_speech({"gender": "male", "gender_source": "user"}, {}, analyse_source("Yes."))
    warnings = check_translation("ค่ะ ฉันจะไปเดี๋ยวนี้", plan, "user")
    assert any("หญิง" in warning for warning in warnings)
    polite = plan_speech({"voice": {"register": "สุภาพ", "locked": True}}, {},
                         analyse_source("please"))
    assert any("คำหยาบ" in warning for warning in check_translation("เออ มึงรีบไปเลย", polite))


def test_relation_type_reads_only_recorded_words():
    assert relation_type({"relation": "พี่น้องกับ Kiro"}, {}) == "พี่น้อง"
    assert relation_type({"role": "คุณครู"}, {"role": "นักเรียน"}) == "ครูนักเรียน"
    assert relation_type({}, {}) == ""


def test_language_detection():
    assert detect_language("こんにちは") == "ja"
    assert detect_language("안녕하세요") == "ko"
    assert detect_language("你好") == "zh"
    assert detect_language("Hello there") == "en"
    assert detect_language("สวัสดี") == "th"
