def _normalize(value, low, high):
    if high == low:
        return 0.0
    n = (value - low) / (high - low)
    return max(0.0, min(1.0, n))


def normalize_inputs(perclos, yawn_rate, blink_duration, nod_count,
                      yawn_rate_max=5.0, blink_duration_max=0.6,
                      nod_count_max=5.0):
    """
    Converts raw signals into 0-1 normalized components.
    perclos is already 0-1. The *_max args are the values at which
    a component saturates to 1.0 -- placeholders, tune in Week 5.

    blink_duration_max=0.6 assumes blink_duration is Hafsa's
    blink_dur_avg (average of completed blinks: ~0.15s alert,
    ~0.3s+ drowsy per her module). This was 2.0 when blink_duration
    meant "current live closure duration" -- recalibrate again if
    that input's source changes.
    """
    return {
        "perclos_normalized": max(0.0, min(1.0, perclos)),
        "yawn_rate_normalized": _normalize(yawn_rate, 0, yawn_rate_max),
        "blink_duration_normalized": _normalize(blink_duration, 0, blink_duration_max),
        "nod_count_normalized": _normalize(nod_count, 0, nod_count_max),
    }


# Starting-point weights from the spec (tune empirically in Week 5)
DEFAULT_WEIGHTS = {
    "perclos": 0.40,
    "yawn_rate": 0.25,
    "blink_duration": 0.20,
    "nod_count": 0.15,
}

# Used when Hafsa's ear_confidence is low (glasses, poor lighting):
# down-weight the eye-derived signals (perclos, blink duration),
# up-weight mouth (yawn) and head pose (nod) to compensate. Also a
# placeholder split -- worth validating against real low-confidence
# footage before Week 5.
FALLBACK_WEIGHTS = {
    "perclos": 0.20,
    "yawn_rate": 0.35,
    "blink_duration": 0.10,
    "nod_count": 0.35,
}


def compute_drowsiness_score(perclos, yawn_rate, blink_duration, nod_count,
                              ear_confidence=1.0, confidence_threshold=0.5):
    """
    Returns (score_0_100, weights_used_dict) so callers/logs can see
    which weight set fired.
    """
    norm = normalize_inputs(perclos, yawn_rate, blink_duration, nod_count)

    weights = (FALLBACK_WEIGHTS if ear_confidence < confidence_threshold
               else DEFAULT_WEIGHTS)

    score = (
        weights["perclos"] * norm["perclos_normalized"] +
        weights["yawn_rate"] * norm["yawn_rate_normalized"] +
        weights["blink_duration"] * norm["blink_duration_normalized"] +
        weights["nod_count"] * norm["nod_count_normalized"]
    ) * 100.0

    return round(score, 2), weights
