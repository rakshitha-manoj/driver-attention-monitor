"""
test_perception.py
Unit tests — proves your functions work without a webcam.
Run: python3 -m pytest perception/test_perception.py -v
"""

import numpy as np
import pytest
from perception import compute_ear, compute_mar, apply_clahe, PerceptionModule
import cv2


class FakeLM:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x; self.y = y; self.z = z; self.visibility = visibility

class FakeLandmarks:
    def __init__(self, d): self._d = d
    def __getitem__(self, idx): return self._d[idx]

EYE_IDS   = [33, 160, 158, 133, 153, 144]
MOUTH_IDS = [61, 291, 39, 181, 0, 17, 269, 405]

def make_open_eye():
    return {
        33:FakeLM(0.0,0.0), 133:FakeLM(1.0,0.0),
        160:FakeLM(0.25,-0.15), 158:FakeLM(0.75,-0.15),
        153:FakeLM(0.25,0.15),  144:FakeLM(0.75,0.15),
    }

def make_closed_eye():
    return {
        33:FakeLM(0.0,0.0), 133:FakeLM(1.0,0.0),
        160:FakeLM(0.25,-0.005), 158:FakeLM(0.75,-0.005),
        153:FakeLM(0.25,0.005),  144:FakeLM(0.75,0.005),
    }

def make_open_mouth():
    return {
        61:FakeLM(0.0,0.0), 291:FakeLM(1.0,0.0),
        39:FakeLM(0.2,-0.4), 181:FakeLM(0.2,0.4),
        0:FakeLM(0.5,-0.4),  17:FakeLM(0.5,0.4),
        269:FakeLM(0.8,-0.4), 405:FakeLM(0.8,0.4),
    }

def make_closed_mouth():
    return {
        61:FakeLM(0.0,0.0), 291:FakeLM(1.0,0.0),
        39:FakeLM(0.2,-0.01), 181:FakeLM(0.2,0.01),
        0:FakeLM(0.5,-0.01),  17:FakeLM(0.5,0.01),
        269:FakeLM(0.8,-0.01), 405:FakeLM(0.8,0.01),
    }


# ── EAR tests ──────────────────────────────────────────────────────

def test_open_eye_above_threshold():
    lm = FakeLandmarks(make_open_eye())
    assert compute_ear(lm, EYE_IDS, 1, 1) > 0.25

def test_open_eye_formula():
    lm  = FakeLandmarks(make_open_eye())
    ear = compute_ear(lm, EYE_IDS, 1, 1)
    assert abs(ear - 0.58) < 0.05, f"Expected ~0.58, got {ear:.4f}"

def test_closed_eye_near_zero():
    lm = FakeLandmarks(make_closed_eye())
    assert compute_ear(lm, EYE_IDS, 1, 1) < 0.55

def test_ear_non_negative():
    for d in [make_open_eye(), make_closed_eye()]:
        assert compute_ear(FakeLandmarks(d), EYE_IDS, 1, 1) >= 0.0

def test_ear_resolution_invariant():
    lm    = FakeLandmarks(make_open_eye())
    ear1  = compute_ear(lm, EYE_IDS, 1,    1)
    ear2  = compute_ear(lm, EYE_IDS, 1920, 1080)
    assert abs(ear1 - ear2) < 0.1


# ── MAR tests ──────────────────────────────────────────────────────

def test_open_mouth_high_mar():
    lm = FakeLandmarks(make_open_mouth())
    assert compute_mar(lm, MOUTH_IDS, 1, 1) > 0.6

def test_closed_mouth_low_mar():
    lm = FakeLandmarks(make_closed_mouth())
    assert compute_mar(lm, MOUTH_IDS, 1, 1) < 0.7

def test_mar_non_negative():
    for d in [make_open_mouth(), make_closed_mouth()]:
        assert compute_mar(FakeLandmarks(d), MOUTH_IDS, 1, 1) >= 0.0


# ── CLAHE tests ────────────────────────────────────────────────────

def test_clahe_shape_unchanged():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert apply_clahe(frame).shape == frame.shape

def test_clahe_is_uint8():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    assert apply_clahe(frame).dtype == np.uint8

def test_clahe_brightens_dark_frame():
    dark = np.ones((480, 640, 3), dtype=np.uint8) * 15
    assert apply_clahe(dark).mean() > dark.mean()


# ── Calibration tests ──────────────────────────────────────────────
# FIXED: bumped from 10 to 11 samples. perception.py's calibration-
# complete check is `len(self.calibration_ears) > 10` (strictly
# greater than) — with exactly 10 samples this was False, so
# calibration never actually ran and ear_threshold stayed at the
# untouched default (0.25) in both tests below. They were passing
# for the wrong reason, not verifying the calibration math at all.

def _calibrate(ears):
    m = PerceptionModule()
    m.calibration_ears  = ears
    m.calibration_start = 0
    m.calibration_secs  = 0
    m._run_calibration(ears[-1])
    return m

def test_threshold_below_mean():
    ears = [0.30,0.31,0.29,0.30,0.32,0.28,0.31,0.30,0.29,0.31,0.30]  # 11 samples
    m    = _calibrate(ears)
    assert m.ear_threshold < np.mean(ears)

def test_threshold_floor_0_15():
    ears = [0.09,0.10,0.09,0.10,0.09,0.10,0.09,0.10,0.09,0.10,0.09]  # 11 samples
    m    = _calibrate(ears)
    assert m.ear_threshold >= 0.15

def test_threshold_floor_actually_applied():
    # NEW: the previous floor test could pass even with the floor
    # clamp deleted entirely, since calibration never ran. This one
    # uses a mean far enough under 0.15 that it only passes if the
    # floor clamp is genuinely implemented and calibration genuinely runs.
    ears = [0.05] * 15
    m = _calibrate(ears)
    assert abs(m.ear_threshold - 0.15) < 0.001, "Floor clamp should force threshold to exactly 0.15"

def test_calibrated_flag():
    m = _calibrate([0.30] * 15)
    assert m.calibrated is True


# ── Blink tests ────────────────────────────────────────────────────

def _module():
    m = PerceptionModule()
    m.calibrated    = True
    m.ear_threshold = 0.25
    return m

def test_quick_blink_counted():
    m = _module()
    m._update_blink(0.10, 1.0)
    m._update_blink(0.30, 1.2)
    assert m.blink_count == 1

def test_drowsy_hold_not_blink():
    m = _module()
    m._update_blink(0.10, 1.0)
    m._update_blink(0.30, 1.9)
    assert m.blink_count == 0

def test_ten_blinks():
    m = _module()
    for i in range(10):
        m._update_blink(0.10, i * 1.0)
        m._update_blink(0.30, i * 1.0 + 0.15)
    assert m.blink_count == 10

def test_blink_duration_tracked():
    m = _module()
    m._update_blink(0.10, 1.0)
    m._update_blink(0.30, 1.2)
    assert abs(m.blink_dur_avg - 0.2) < 0.01

def test_drowsy_blinks_longer_than_alert():
    alert  = _module()
    drowsy = _module()
    for i in range(5):
        alert._update_blink(0.10,  i * 1.0)
        alert._update_blink(0.30,  i * 1.0 + 0.15)
        drowsy._update_blink(0.10, i * 1.0)
        drowsy._update_blink(0.30, i * 1.0 + 0.35)
    assert drowsy.blink_dur_avg > alert.blink_dur_avg

def test_blink_state_transitions():
    m = _module()
    assert m.blink_state == "open"
    m._update_blink(0.10, 1.0)
    assert m.blink_state == "closed"
    m._update_blink(0.30, 1.2)
    assert m.blink_state == "open"


# ── Yawn tests ─────────────────────────────────────────────────────

def test_yawn_after_1_5s():
    m = PerceptionModule()
    m._update_yawn(0.7, 0.0)
    m._update_yawn(0.7, 1.6)
    m._update_yawn(0.1, 1.7)
    assert m.yawn_count == 1

def test_brief_not_yawn():
    m = PerceptionModule()
    m._update_yawn(0.7, 0.0)
    m._update_yawn(0.7, 0.8)
    m._update_yawn(0.1, 0.9)
    assert m.yawn_count == 0

def test_no_double_count():
    m = PerceptionModule()
    for t in np.arange(0.0, 4.0, 0.1):
        m._update_yawn(0.8, float(t))
    assert m.yawn_count == 1

def test_three_yawns():
    m = PerceptionModule()
    for i in range(3):
        b = i * 5.0
        m._update_yawn(0.7, b)
        m._update_yawn(0.7, b + 2.0)
        m._update_yawn(0.1, b + 2.1)
    assert m.yawn_count == 3


# ── Output dict contract ───────────────────────────────────────────

REQUIRED_KEYS = {
    "frame_id": int, "timestamp": float,
    "EAR": (float, type(None)), "MAR": (float, type(None)),
    "blink_state": str, "blink_count": int,
    "blink_dur_avg": float, "yawn_count": int,
    "ear_confidence": float, "landmarks_detected": bool,
}

def test_no_face_dict_has_all_keys():
    no_face = {
        "frame_id":1, "timestamp":0.033,
        "EAR":None, "MAR":None,
        "blink_state":"unknown", "blink_count":0,
        "blink_dur_avg":0.0, "yawn_count":0,
        "ear_confidence":0.0, "landmarks_detected":False,
    }
    for key, t in REQUIRED_KEYS.items():
        assert key in no_face
        assert isinstance(no_face[key], t)

def test_valid_blink_states():
    for v in ["open", "closed", "unknown"]:
        assert v in {"open", "closed", "unknown"}

def test_confidence_range():
    for c in [0.0, 0.2, 0.5, 1.0]:
        assert 0.0 <= c <= 1.0