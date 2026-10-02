import os

import numpy as np
import pytest
from PIL import Image

from screen_thai.faces import FaceMemory
from screen_thai.models import Settings
from screen_thai.vision import (Appearance, ExpressionCues, FaceBox, NullDetector, VisionEngine,
                                build_detector, cosine, cues_from_landmarks, descriptor,
                                sample_appearance)

FACE_FIXTURE = os.environ.get("SCREEN_THAI_FACE_FIXTURE", "")


def synthetic_landmarks(scale: float = 1.0) -> list[tuple[float, float]]:
    """Neutral face: left/right eye corners, lids, brows, mouth corners and lips."""
    points = [(0.0, 0.0)] * 480
    points[33] = (100, 120)          # left eye outer
    points[133] = (140, 120)         # left eye inner
    points[159] = (120, 116)         # left lid
    points[145] = (120, 116 + 8 * scale)
    points[362] = (200, 120)
    points[263] = (240, 120)
    points[386] = (220, 116)
    points[374] = (220, 116 + 8 * scale)
    points[105] = (110, 104)         # brows
    points[334] = (210, 104)
    points[66] = (140, 104)
    points[296] = (240, 104)
    points[61] = (150, 170)          # left mouth corner
    points[291] = (240, 170)         # right mouth corner
    points[13] = (196, 168)          # upper inner lip
    points[14] = (196, 172)          # lower inner lip
    points[78] = (170, 170)
    points[308] = (222, 170)
    points[17] = (196, 186)
    points[10] = (180, 60)
    points[152] = (185, 240)
    points[1] = (185, 150)
    points[468] = (120, 120)
    points[473] = (220, 120)
    return points


def test_neutral_face_has_no_strong_expression():
    cues = cues_from_landmarks(synthetic_landmarks())
    assert cues.quality == "landmarks"
    assert cues.label in ("ไม่ทราบ", "กำลังพูดหรือตกใจ", "ตาโต/ประหลาดใจ")
    assert abs(cues.smile) < 0.12


def test_smile_and_open_mouth_are_measured():
    neutral = cues_from_landmarks(synthetic_landmarks())
    landmarks = synthetic_landmarks()
    landmarks[61] = (150, 160)       # corners pulled up
    landmarks[291] = (240, 160)
    landmarks[14] = (196, 190)       # mouth open
    cues = cues_from_landmarks(landmarks)
    assert cues.smile > neutral.smile
    assert cues.mouth_open > neutral.mouth_open
    # A single still frame never claims a mood: only physical extremes get a label.
    assert cues.label in ("ไม่ทราบ", "อ้าปากกว้าง")


def test_blink_and_wide_mouth_are_labelled_as_physical_facts():
    landmarks = synthetic_landmarks()
    landmarks[145] = (120, 117)
    landmarks[374] = (220, 117)
    assert cues_from_landmarks(landmarks).label == "หลับตาหรือกะพริบ"
    landmarks = synthetic_landmarks()
    landmarks[14] = (196, 210)
    assert cues_from_landmarks(landmarks).label == "อ้าปากกว้าง"


def test_frown_and_blink_are_measured():
    landmarks = synthetic_landmarks()
    landmarks[61] = (150, 182)
    landmarks[291] = (240, 182)
    landmarks[145] = (120, 118)
    landmarks[374] = (220, 118)
    cues = cues_from_landmarks(landmarks)
    assert cues.smile < -0.05
    assert cues.eye_open < 0.4
    assert cues.label in ("หลับตาหรือกะพริบ", "ไม่ทราบ")


def test_region_cues_work_without_landmarks():
    from screen_thai.vision import cues_from_regions
    image = np.full((200, 200, 3), 120, dtype="uint8")
    image[140:170, 60:140] = 10      # dark "open mouth" band
    cues = cues_from_regions(image, FaceBox(20, 20, 160, 160))
    assert cues.quality == "regions"
    assert cues.mouth_open > 0.0
    assert cues.label != ""


def test_descriptor_is_normalised_and_similar_for_similar_crops():
    image = np.zeros((240, 240, 3), dtype="uint8")
    image[40:200, 60:180] = 180
    image[80:120, 90:150] = 40
    image[140:170, 100:150] = 220
    first = descriptor(image, FaceBox(60, 40, 120, 160))
    second = descriptor(image, FaceBox(62, 42, 120, 160))
    other = descriptor(image, FaceBox(40, 60, 80, 100))
    assert len(first) > 60
    assert abs(sum(value * value for value in first) - 1.0) < 1e-3
    assert cosine(first, second) > cosine(first, other)


def test_descriptor_is_empty_for_tiny_boxes():
    image = np.zeros((50, 50, 3), dtype="uint8")
    assert descriptor(image, FaceBox(10, 10, 4, 4)) == []


def test_appearance_sampling_reports_colours_without_background():
    image = np.full((260, 260, 3), 30, dtype="uint8")        # dark background
    image[60:200, 80:180] = (220, 180, 160)                  # face
    image[30:70, 80:180] = (60, 40, 30)                      # hair band
    image[200:260, 70:190] = (40, 90, 200)                   # outfit band
    appearance = sample_appearance(image, FaceBox(80, 60, 100, 140))
    assert appearance.hair and appearance.outfit
    assert appearance.outfit != appearance.hair
    assert appearance.brightness > 0.0
    assert appearance.summary()


def test_build_detector_off_and_unavailable_backends_degrade_gracefully():
    detector = build_detector("off")
    assert isinstance(detector, NullDetector) and not detector.available
    fallback = build_detector("mediapipe", max_faces=2)
    assert hasattr(fallback, "available")
    assert fallback.available or getattr(fallback, "reason", "")


def test_engine_with_null_backend_never_crashes_and_returns_nothing():
    settings = Settings(face_backend="off", face_analysis=True)
    engine = VisionEngine(settings)
    assert engine.available is False
    assert engine.analyze(Image.new("RGB", (64, 64))) == []


def test_engine_records_busy_state_instead_of_queueing():
    settings = Settings(face_backend="off")
    engine = VisionEngine(settings)
    engine._running = True                      # simulate an in-flight frame
    assert engine.analyze(Image.new("RGB", (32, 32))) == []
    assert engine.skipped is True


@pytest.mark.skipif(not FACE_FIXTURE, reason="ต้องตั้ง SCREEN_THAI_FACE_FIXTURE เป็นภาพใบหน้าจริง")
def test_real_photo_face_analysis_if_available():
    """Local integration check with a real photograph; never part of the committed fixture set."""
    image = Image.open(FACE_FIXTURE)
    settings = Settings(face_backend="auto", face_analysis=True, max_faces=2)
    engine = VisionEngine(settings)
    assert engine.available, getattr(engine, "error", "")
    observations = engine.analyze(image, FaceMemory())
    assert observations, "ไม่พบใบหน้าในภาพทดสอบ"
    first = observations[0]
    assert isinstance(first.cues, ExpressionCues)
    assert isinstance(first.appearance, Appearance)
    assert first.box.w > 40 and first.descriptor
