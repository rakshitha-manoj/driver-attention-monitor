def _normalize(value, low, high):
    if high == low:
        return 0.0
    n = (value - low) / (high - low)
    return max(0.0, min(1.0, n))


def normalize_inputs(perclos, yawn_rate, blink_duration, nod_count,
                      microsleep_seconds=0.0,
                      yawn_rate_max=5.0, blink_duration_max=0.6,
                      nod_count_max=5.0, microsleep_max=5.0):
    """
    Converts raw signals into 0-1 normalized components.
    perclos is already 0-1. The *_max args are the values at which
    a component saturates to 1.0.

    microsleep_seconds: duration of continuous eye closure right now.
        This is qualitatively different from PERCLOS (which averages
        over 60 seconds). A 3-second sustained closure is a microsleep
        event that should trigger WARNING/CRITICAL regardless of what
        the 60-second average looks like. Saturates at 5 seconds.
    """
    return {
        "perclos_normalized": max(0.0, min(1.0, perclos)),
        "yawn_rate_normalized": _normalize(yawn_rate, 0, yawn_rate_max),
        "blink_duration_normalized": _normalize(blink_duration, 0, blink_duration_max),
        "nod_count_normalized": _normalize(nod_count, 0, nod_count_max),
        "microsleep_normalized": _normalize(microsleep_seconds, 0, microsleep_max),
    }


# Weights WITH microsleep signal.
# Microsleep is the strongest single signal: if your eyes have been
# closed for 3+ seconds straight, that alone should drive the score
# past WARNING threshold. The other signals provide supporting evidence.
DEFAULT_WEIGHTS = {
    "perclos": 0.25,
    "yawn_rate": 0.15,
    "blink_duration": 0.10,
    "nod_count": 0.10,
    "microsleep": 0.40,
}

# Used when ear_confidence is low (glasses, poor lighting):
# down-weight eye-derived signals, up-weight mouth and head pose.
# Microsleep weight stays moderate since eye closure detection may
# still work (depends on the specific failure mode).
FALLBACK_WEIGHTS = {
    "perclos": 0.15,
    "yawn_rate": 0.25,
    "blink_duration": 0.05,
    "nod_count": 0.30,
    "microsleep": 0.25,
}


def compute_drowsiness_score(perclos, yawn_rate, blink_duration, nod_count,
                              microsleep_seconds=0.0,
                              ear_confidence=1.0, confidence_threshold=0.5):
    """
    Returns (score_0_100, weights_used_dict) so callers/logs can see
    which weight set fired.

    microsleep_seconds: how long the eyes have been continuously closed
        right now (0 if eyes are open). This is the key signal for
        catching sustained eye closure events that the 60-second
        PERCLOS window would dilute.
    """
    norm = normalize_inputs(perclos, yawn_rate, blink_duration, nod_count,
                            microsleep_seconds=microsleep_seconds)

    weights = (FALLBACK_WEIGHTS if ear_confidence < confidence_threshold
               else DEFAULT_WEIGHTS)

    score = (
        weights["perclos"] * norm["perclos_normalized"] +
        weights["yawn_rate"] * norm["yawn_rate_normalized"] +
        weights["blink_duration"] * norm["blink_duration_normalized"] +
        weights["nod_count"] * norm["nod_count_normalized"] +
        weights["microsleep"] * norm["microsleep_normalized"]
    ) * 100.0

    return round(score, 2), weights
