"""Local face geometry: detection, expression cues and appearance sampling.

Two real backends, no invented confidence:

* MediaPipe FaceMesh (optional install, 468+ landmarks) gives geometric expression cues:
  mouth opening, mouth-corner curvature (smile/frown), eye openness, brow raise, head tilt
  and iris gaze offset. These are measurements, not a mood classifier.
* OpenCV (already required by RapidOCR) gives face boxes via YuNet (ONNX model file) or the
  Haar cascade, and only coarse region statistics as cues.

The AI layer only ever receives these numbers plus the still image; it is explicitly told
that cues are weak evidence and that it cannot hear the character.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .faces import FaceTracker

# --- MediaPipe FaceMesh landmark indices ------------------------------------------------------

LEFT_EYE = (33, 133, 159, 145)      # outer corner, inner corner, upper lid, lower lid
RIGHT_EYE = (362, 263, 386, 374)
BROWS = (105, 334, 66, 296)
MOUTH_CORNERS = (61, 291)
MOUTH_INNER = (13, 14, 78, 308)
MOUTH_OUTER_BOTTOM = 17
IRIS_LEFT, IRIS_RIGHT = 468, 473
FOREHEAD, CHIN, NOSE_TIP = 10, 152, 1

CUE_LABELS = {
    "smile": "ยิ้ม",
    "talking": "กำลังพูด/ปากขยับ",
    "wide_eyes": "ตาโต/ตกใจ",
    "blink": "หลับตา/กะพริบ",
    "brow_raise": "ยกคิ้ว สงสัยหรือประหลาดใจ",
    "frown": "มุมปากตก เครียดหรือไม่พอใจ",
    "head_tilt": "เอียงศีรษะ",
    "look_away": "มองไปด้านข้าง ไม่ได้มองคู่สนทนา",
}

MIN_FACE_PX = 40          # ignore tiny detections: they cannot carry expression information
GRID = 4                  # descriptor spatial cells per axis


def _distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0


@dataclass
class ExpressionCues:
    """All values are measurements; `label`/`strength` are the honest summary of them."""

    label: str = "ไม่ทราบ"
    strength: float = 0.0
    mouth_open: float = 0.0
    smile: float = 0.0
    eye_open: float = 0.0
    brow_raise: float = 0.0
    tilt_deg: float = 0.0
    gaze_x: float = 0.0
    quality: str = "landmarks"      # landmarks | regions | none

    def summary(self) -> str:
        checks = (
            ("ปากเปิด", self.mouth_open, 0.12, "%.2f"),
            ("มุมปากยกขึ้น", self.smile, 0.06, "%.2f"),
            ("มุมปากต่ำลง", -self.smile, 0.06, "%.2f"),
            ("ตากว้าง", self.eye_open, 0.45, "%.2f"),
            ("ตาหรี่", 0.24 - self.eye_open, 0.06, "%.2f"),
            ("คิ้วสูง", self.brow_raise, 0.55, "%.2f"),
            ("มองด้านข้าง", abs(self.gaze_x), 0.35, "%.2f"),
            ("ศีรษะเอียง", abs(self.tilt_deg), 8.0, "%.0f องศา"),
        )
        bits = []
        for name, value, low, fmt in checks:
            if value >= low:
                bits.append(name + " " + (fmt % value))
        return ", ".join(bits[:4])


@dataclass
class Appearance:
    hair: str = ""
    eyes: str = ""
    outfit: str = ""
    palette: tuple[str, ...] = ()
    brightness: float = 0.0
    contrast: float = 0.0
    warmth: float = 0.0        # + อบอุ่น (แดง/ส้ม)  - เย็น (ฟ้า/เขียว)

    def summary(self) -> str:
        bits = []
        if self.hair:
            bits.append(f"ผม {self.hair}")
        if self.eyes:
            bits.append(f"ตา {self.eyes}")
        if self.outfit:
            bits.append(f"ชุด {self.outfit}")
        if self.palette:
            bits.append("โทนสี " + "/".join(self.palette[:3]))
        return " • ".join(bits)


@dataclass
class FaceObservation:
    track_id: int
    box: FaceBox
    cues: ExpressionCues
    appearance: Appearance
    descriptor: list[float] = field(default_factory=list)
    character: str = ""
    match: float = 0.0
    possible: str = ""
    possible_match: float = 0.0

    def summary(self) -> str:
        name = self.character or self.possible or f"ใบหน้าที่ {self.track_id}"
        parts = [f"{name} ({self.box.w}×{self.box.h}px)"]
        if self.cues.quality != "none":
            parts.append(f"สีหน้า: {self.cues.label} [{self.cues.summary()}]")
        if self.appearance.summary():
            parts.append(self.appearance.summary())
        if self.possible and not self.character:
            parts.append(f"อาจเป็น {self.possible} ({self.possible_match:.2f}) แต่ยังไม่ยืนยัน")
        return " • ".join(parts)


# --- Detector backends -------------------------------------------------------------------------


class Detector:
    name = "none"
    quality = "none"

    @property
    def available(self) -> bool:
        return False

    def detect(self, image) -> list[tuple[FaceBox, Sequence[tuple[float, float]] | None]]:
        return []


class MediaPipeDetector(Detector):
    name = "mediapipe"
    quality = "landmarks"

    def __init__(self, max_faces: int = 4, refine: bool = True):
        import mediapipe as mp  # imported lazily: optional dependency
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=max(1, min(8, max_faces)), refine_landmarks=refine,
            min_detection_confidence=0.4, min_tracking_confidence=0.4)
        self._refine = refine
        self.name = "mediapipe" + ("+iris" if refine else "")

    @property
    def available(self) -> bool:
        return True

    def detect(self, image):
        height, width = image.shape[:2]
        result = self._mesh.process(image)
        found = []
        for face in result.multi_face_landmarks or []:
            points = face.landmark
            xs = [_clamp(p.x, -0.05, 1.05) * width for p in points]
            ys = [_clamp(p.y, -0.05, 1.05) * height for p in points]
            left, right = max(0, min(xs)), min(width, max(xs))
            top, bottom = max(0, min(ys)), min(height, max(ys))
            if right - left < MIN_FACE_PX or bottom - top < MIN_FACE_PX:
                continue
            landmarks = [(p.x * width, p.y * height) for p in points]
            found.append((FaceBox(int(left), int(top), int(right - left), int(bottom - top), 1.0),
                          landmarks))
        return found


class OpenCVDetector(Detector):
    name = "opencv"
    quality = "regions"

    def __init__(self, model_path: str = "") -> None:
        import cv2
        self._cv2 = cv2
        self._yunet = None
        self._cascade = None
        if model_path and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.6, 0.3, 5000)
                self.name = "opencv-yunet"
                return
            except Exception:
                self._yunet = None
        data = getattr(getattr(cv2, "data", None), "haarcascades", "")
        if data and hasattr(cv2, "CascadeClassifier"):
            cascade = cv2.CascadeClassifier(data + "haarcascade_frontalface_default.xml")
            if not cascade.empty():
                self._cascade = cascade
                self.name = "opencv-haar"
                if hasattr(cv2, "data") and hasattr(cv2, "CascadeClassifier"):
                    self._smile = cv2.CascadeClassifier(data + "haarcascade_smile.xml")
                    self._eye = cv2.CascadeClassifier(data + "haarcascade_eye.xml")

    @property
    def available(self) -> bool:
        return self._yunet is not None or self._cascade is not None

    def detect(self, image):
        import numpy as np
        if self._yunet is not None:
            self._yunet.setInputSize((image.shape[1], image.shape[0]))
            _retval, faces = self._yunet.detect(image)
            found = []
            for face in [] if faces is None else faces:
                x, y, w, h = (int(v) for v in face[:4])
                if w < MIN_FACE_PX or h < MIN_FACE_PX:
                    continue
                landmarks = [(float(face[4 + i * 2]), float(face[5 + i * 2])) for i in range(5)]
                found.append((FaceBox(x, y, w, h, float(face[-1])), landmarks))
            return found
        grey = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2GRAY)
        boxes = self._cascade.detectMultiScale(grey, 1.15, 5, minSize=(MIN_FACE_PX, MIN_FACE_PX))
        return [(FaceBox(int(x), int(y), int(w), int(h)), None)
                for x, y, w, h in np.array(boxes).reshape(-1, 4)] if len(boxes) else []


class NullDetector(Detector):
    name = "off"
    quality = "none"

    def __init__(self, reason: str = "ปิดการวิเคราะห์ใบหน้า"):
        self.reason = reason


def build_detector(backend: str, model_path: str = "", max_faces: int = 4,
                   refine: bool = True) -> Detector:
    """Choose a backend without ever failing the whole session over an optional feature."""
    order: list[str] = []
    if backend == "off":
        return NullDetector()
    if backend in ("auto", "mediapipe"):
        order.append("mediapipe")
    if backend in ("auto", "opencv"):
        order.extend(["opencv-yunet", "opencv-haar"])
    errors = []
    for candidate in order:
        try:
            if candidate == "mediapipe":
                return MediaPipeDetector(max_faces, refine)
            detector = OpenCVDetector(model_path if candidate == "opencv-yunet" else "")
            if detector.available:
                return detector
            errors.append(f"{candidate}: ใช้ไม่ได้")
        except ImportError as exc:
            errors.append(f"{candidate}: ยังไม่ได้ติดตั้ง ({exc.name})")
        except Exception as exc:  # pragma: no cover - defensive, backend specific
            errors.append(f"{candidate}: {type(exc).__name__}")
    return NullDetector("ไม่พบ backend ใบหน้า: " + "; ".join(errors[:3]))


# --- Cue measurement ---------------------------------------------------------------------------


def cues_from_landmarks(landmarks: Sequence[tuple[float, float]]) -> ExpressionCues:
    """Geometric measurements from a still frame.

    These numbers are honest: lip gap, mouth-corner height, eye aspect ratio, brow distance,
    head tilt and iris offset. They are NOT an emotion classifier, so the label only states
    physical facts (blink, wide open mouth). Mood is inferred by the model from the image plus
    these measurements, and always compared against the character's own baseline over time.
    """
    if len(landmarks) < 400:
        return ExpressionCues(quality="regions")
    left_corner, right_corner = landmarks[MOUTH_CORNERS[0]], landmarks[MOUTH_CORNERS[1]]
    mouth_width = max(1e-3, _distance(left_corner, right_corner))
    eye_scale = max(1e-3, _distance(landmarks[LEFT_EYE[0]], landmarks[RIGHT_EYE[1]]))
    upper_lip, lower_lip = landmarks[MOUTH_INNER[0]], landmarks[MOUTH_INNER[1]]
    lip_gap = _distance(upper_lip, lower_lip) / mouth_width
    mouth_open = _clamp(lip_gap / 0.5)
    outer_center = (landmarks[13][1] + landmarks[MOUTH_OUTER_BOTTOM][1]) / 2
    corners_y = (left_corner[1] + right_corner[1]) / 2
    smile = (outer_center - corners_y) / mouth_width        # บวก = ยกมุมปากขึ้น
    ears = [_distance(landmarks[upper], landmarks[lower]) /
            max(1e-3, _distance(landmarks[outer], landmarks[inner]))
            for outer, inner, upper, lower in (LEFT_EYE, RIGHT_EYE)]
    eye_open = _clamp((sum(ears) / len(ears)) / 0.30)
    brow_distance = (_distance(landmarks[BROWS[0]], landmarks[LEFT_EYE[2]]) +
                     _distance(landmarks[BROWS[1]], landmarks[RIGHT_EYE[2]])) / 2
    brow_raise = _clamp((brow_distance / eye_scale - 0.15) / 0.20)
    dx = landmarks[RIGHT_EYE[1]][0] - landmarks[LEFT_EYE[0]][0]
    dy = landmarks[RIGHT_EYE[1]][1] - landmarks[LEFT_EYE[0]][1]
    tilt = math.degrees(math.atan2(dy, dx))
    gaze_x = 0.0
    if len(landmarks) > IRIS_RIGHT:
        eye_left_x = (landmarks[LEFT_EYE[0]][0] + landmarks[LEFT_EYE[1]][0]) / 2
        eye_right_x = (landmarks[RIGHT_EYE[0]][0] + landmarks[RIGHT_EYE[1]][0]) / 2
        gaze_x = _clamp(((landmarks[IRIS_LEFT][0] - eye_left_x) +
                         (landmarks[IRIS_RIGHT][0] - eye_right_x)) / 2 / eye_scale, -1.0, 1.0)
    label = "ไม่ทราบ"
    if min(ears) < 0.15:
        label = "หลับตาหรือกะพริบ"
    elif lip_gap > 0.45:
        label = "อ้าปากกว้าง"
    strength = max(mouth_open, smile, eye_open, brow_raise, abs(gaze_x), min(1.0, abs(tilt) / 25.0))
    return ExpressionCues(label, min(1.0, strength), mouth_open, smile, eye_open, brow_raise,
                          tilt, gaze_x, "landmarks")


def cues_from_regions(image, box: FaceBox) -> ExpressionCues:
    """Coarse fallback when only a face box is available (no landmarks)."""
    height, width = image.shape[:2]
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(width, box.x + box.w), min(height, box.y + box.h)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return ExpressionCues(quality="none")
    face = image[y0:y1, x0:x1].astype("float32")
    grey = face.mean(axis=2)
    mouth_band = grey[int(0.62 * grey.shape[0]):int(0.9 * grey.shape[0]), :]
    mouth_dark = float((mouth_band < grey.mean() - 25).mean()) if mouth_band.size else 0.0
    rows = [grey[int(frac * grey.shape[0]):int((frac + 0.15) * grey.shape[0]), :].mean()
            for frac in (0.2, 0.35, 0.5, 0.65, 0.8)]
    smile = 0.0
    if len(rows) >= 4:
        smile = _clamp((rows[1] - rows[-2]) / 60.0, -1.0, 1.0)
    mouth_open = _clamp(mouth_dark / 0.3)
    label = "ไม่ทราบ"
    if mouth_open > 0.6:
        label = "บริเวณปากมืดกว่าปกติชัดเจน (ค่าหยาบ)"
    return ExpressionCues(label, mouth_open, mouth_open, smile, 0.0, 0.0, 0.0, 0.0, "regions")


def expression_label(cues: ExpressionCues) -> str:
    return cues.label


# --- Appearance + identity descriptor ----------------------------------------------------------


def _hex(pixel) -> str:
    return "#%02x%02x%02x" % (int(pixel[0]), int(pixel[1]), int(pixel[2]))


def _median_color(region) -> str:
    import numpy as np
    if region.size == 0:
        return ""
    flat = region.reshape(-1, region.shape[-1]).astype("float32")
    return _hex(np.median(flat, axis=0)[:3])


def background_color(image):
    import numpy as np
    height, width = image.shape[:2]
    step_x, step_y = max(1, width // 40), max(1, height // 40)
    corners = [image[:step_y, :step_x], image[:step_y, -step_x:],
               image[-step_y:, :step_x], image[-step_y:, -step_x:]]
    sample = np.concatenate([c.reshape(-1, image.shape[-1]) for c in corners if c.size])
    return np.median(sample, axis=0)[:3].astype("float32")


def sample_appearance(image, box: FaceBox, landmarks=None) -> Appearance:
    """Sample hair / eye / outfit colours and a small palette. Pure pixels, no claims."""
    import numpy as np
    height, width = image.shape[:2]
    background = background_color(image)
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(width, box.x + box.w), min(height, box.y + box.h)

    def colour_band(y_start: int, y_end: int, x_start: int, x_end: int) -> str:
        y_start, y_end = max(0, y_start), min(height, y_end)
        x_start, x_end = max(0, x_start), min(width, x_end)
        if y_end - y_start < 2 or x_end - x_start < 2:
            return ""
        band = image[y_start:y_end, x_start:x_end].reshape(-1, image.shape[-1]).astype("float32")
        if band.size == 0:
            return ""
        distance = np.linalg.norm(band[:, :3] - background, axis=1)
        keep = band[distance > 35]
        if keep.shape[0] < 12:                     # almost everything looks like background
            keep = band
        quantised = (keep[:, :3] // 24).astype("int32")
        keys, counts = np.unique(quantised, axis=0, return_counts=True)
        dominant = keys[int(np.argmax(counts))]
        mask = np.all(quantised == dominant, axis=1)
        return _hex(np.median(keep[mask][:, :3], axis=0))

    top = landmarks[FOREHEAD][1] if landmarks else y0 + 0.18 * (y1 - y0)
    hair = colour_band(int(y0 - 0.45 * (y1 - y0)), int(top + 0.12 * (y1 - y0)), x0, x1)
    chin = landmarks[CHIN][1] if landmarks else y1
    outfit = colour_band(int(chin + 0.1 * (y1 - y0)), int(chin + 0.9 * (y1 - y0)),
                         int(x0 - 0.15 * (x1 - x0)), int(x1 + 0.15 * (x1 - x0)))
    eye_x = int(landmarks[IRIS_LEFT][0]) if landmarks and len(landmarks) > IRIS_LEFT else (x0 + x1) // 2
    eye_y = int(landmarks[IRIS_LEFT][1]) if landmarks and len(landmarks) > IRIS_LEFT else y0 + int(0.35 * (y1 - y0))
    patch = image[max(0, eye_y - 3):eye_y + 4, max(0, eye_x - 3):eye_x + 4]
    eyes = _median_color(patch) if patch.size else ""
    region = image[y0:min(height, y1 + int(0.5 * (y1 - y0))), x0:x1]
    palette: tuple[str, ...] = ()
    if region.size:
        flat = region.reshape(-1, region.shape[-1])[:, :3].astype("float32")
        quantised = (flat // 48).astype("int32")
        keys, counts = np.unique(quantised, axis=0, return_counts=True)
        order = np.argsort(-counts)[:4]
        palette = tuple(_hex(np.median(flat[np.all(quantised == keys[i], axis=1)], axis=0))
                        for i in order)
    face = image[y0:y1, x0:x1].astype("float32") if y1 > y0 and x1 > x0 else None
    brightness = float(face.mean() / 255.0) if face is not None and face.size else 0.0
    contrast = float(np.clip(face.mean(axis=2).std() / 64.0, 0, 1)) if face is not None and face.size else 0.0
    warmth = float(np.clip((face[..., 0].mean() - face[..., 2].mean()) / 64.0, -1, 1)) \
        if face is not None and face.size else 0.0
    return Appearance(hair, eyes, outfit, palette, brightness, contrast, warmth)


def descriptor(image, box: FaceBox, landmarks=None) -> list[float]:
    """Small, honest identity signature: aligned grey gradients + colour statistics.

    It is not a neural face embedding. It separates characters that differ in face shape,
    hair framing and palette, and it is meant to be confirmed by the user before it is trusted.
    """
    import numpy as np
    height, width = image.shape[:2]
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(width, box.x + box.w), min(height, box.y + box.h)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return []
    if landmarks and len(landmarks) > RIGHT_EYE[1]:
        left_eye, right_eye = landmarks[LEFT_EYE[0]], landmarks[RIGHT_EYE[1]]
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        if abs(dx) + abs(dy) > 1e-3:
            angle = math.degrees(math.atan2(dy, dx))
            centre = ((x0 + x1) / 2, (y0 + y1) / 2)
            matrix = cv2_rotation(angle, centre)
            image = warp_affine(image, matrix, width, height)
    patch = image[y0:y1, x0:x1].astype("float32")
    grey = patch.mean(axis=2)
    gy, gx = np.gradient(grey)
    magnitude = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + math.pi) % math.pi
    bins = np.minimum(7, (angle / (math.pi / 8)).astype("int32"))
    cells = []
    for row in range(GRID):
        for column in range(GRID):
            ys = slice(row * grey.shape[0] // GRID, (row + 1) * grey.shape[0] // GRID)
            xs = slice(column * grey.shape[1] // GRID, (column + 1) * grey.shape[1] // GRID)
            cell_bins, cell_magnitude = bins[ys, xs], magnitude[ys, xs]
            histogram = np.bincount(cell_bins.ravel(), weights=cell_magnitude.ravel(), minlength=8)
            vector = histogram / (np.linalg.norm(histogram) + 1e-6)
            moments = np.array([grey[ys, xs].mean() / 255.0, grey[ys, xs].std() / 255.0])
            cells.extend(vector.tolist())
            cells.extend(moments.tolist())
    flat = patch.reshape(-1, patch.shape[-1])[:, :3]
    colour = []
    for channel in range(3):
        hist, _ = np.histogram(flat[:, channel], bins=4, range=(0, 256))
        histogram = hist / (hist.sum() + 1e-6)
        colour.extend(histogram.tolist())
    vector = np.array(cells + colour, dtype="float32")
    norm = float(np.linalg.norm(vector))
    return (vector / norm).tolist() if norm else []


def cv2_rotation(angle: float, centre):
    import cv2
    import numpy as np
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return matrix if isinstance(matrix, np.ndarray) else np.array(matrix, dtype="float32")


def warp_affine(image, matrix, width: int, height: int):
    import cv2
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot))


class VisionEngine:
    """Face analysis for one captured frame. Create one per worker thread."""

    def __init__(self, settings, embedder=None):
        self.settings = settings
        self._detector: Detector | None = None
        self.error = ""
        self.embedder = embedder
        self._running = False
        self.skipped = False
        self.tracker = FaceTracker()

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = build_detector(self.settings.face_backend, self.settings.face_model,
                                            self.settings.max_faces)
            if not self._detector.available:
                self.error = getattr(self._detector, "reason", "ไม่พบ backend สำหรับตรวจจับใบหน้า")
        return self._detector

    @property
    def available(self) -> bool:
        return self.detector.available

    def close(self):
        mesh = getattr(self._detector, "_mesh", None)
        if mesh is not None and hasattr(mesh, "close"):
            try:
                mesh.close()
            except Exception:  # pragma: no cover - backend cleanup is best effort
                pass

    def analyze(self, image, memory=None) -> list[FaceObservation]:
        """Detect faces in a PIL image and build observations, optionally matching memory."""
        import numpy as np
        if not self.settings.face_analysis:
            return []
        if self._running:                      # never queue frames: skip while busy
            self.skipped = True
            return []
        detector = self.detector
        if not detector.available:
            return []
        self._running = True
        try:
            array = np.asarray(image.convert("RGB"))
            raw = detector.detect(array)
            observations = []
            for index, (box, landmarks) in enumerate(raw[:self.settings.max_faces]):
                if box.w < MIN_FACE_PX or box.h < MIN_FACE_PX:
                    continue
                cues = (cues_from_landmarks(landmarks) if landmarks and detector.quality == "landmarks"
                        else cues_from_regions(array, box))
                if not self.settings.expression_cues:
                    cues = ExpressionCues(quality="none")
                appearance = sample_appearance(
                    array, box, landmarks if detector.quality == "landmarks" else None)
                vector = self.embedder.embed(array, box, landmarks) if self.embedder \
                    else descriptor(array, box, landmarks)
                observation = FaceObservation(index + 1, box, cues, appearance, vector)
                if memory is not None and vector:
                    match = memory.match(vector)
                    observation.character = match.name
                    observation.match = match.score
                    observation.possible = match.suggestion
                    observation.possible_match = match.suggestion_score
                observations.append(observation)
            return observations
        finally:
            self._running = False
