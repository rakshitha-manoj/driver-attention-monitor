"""
SYNTHETIC DATA MODULE
Everything here stands in for Sheethal's Decision module output, which
is not fully built yet (no PERCLOS, scoring, state machine, or head
pose fusion at time of writing). Every function is clearly flagged.
Replace calls to these once her real module is ready, do not leave
synthetic values silently feeding into final reported results.
"""
import numpy as np

SYNTHETIC_WARNING = "SYNTHETIC - replace with Sheethal's real output when ready"

def synthetic_head_pitch(n_samples, drowsy_labels):
    print(SYNTHETIC_WARNING, "- synthetic_head_pitch()")
    base = np.random.normal(0, 3, n_samples)
    drowsy_boost = np.array(drowsy_labels) * np.random.normal(15, 5, n_samples)
    return base + drowsy_boost

def synthetic_head_yaw(n_samples):
    print(SYNTHETIC_WARNING, "- synthetic_head_yaw()")
    return np.random.normal(0, 8, n_samples)

def synthetic_perclos(ear_series, threshold=0.25, window=60):
    print(SYNTHETIC_WARNING, "- synthetic_perclos() [approximation, not Sheethal's real logic]")
    ear_series = np.array(ear_series)
    perclos = []
    for i in range(len(ear_series)):
        start = max(0, i - window)
        segment = ear_series[start:i+1]
        below = np.sum(segment < threshold)
        perclos.append(below / len(segment))
    return np.array(perclos)

def synthetic_nod_count(n_samples, drowsy_labels):
    print(SYNTHETIC_WARNING, "- synthetic_nod_count()")
    return (np.array(drowsy_labels) * np.random.poisson(2, n_samples)).astype(int)

def synthetic_drowsiness_score(perclos, yawn_rate_norm, blink_dur_norm, nod_norm):
    print(SYNTHETIC_WARNING, "- inputs partially synthetic, formula itself is real")
    return (0.40 * perclos + 0.25 * yawn_rate_norm +
            0.20 * blink_dur_norm + 0.15 * nod_norm) * 100

def synthetic_alert_latency(n_events=20, mean_sec=3.2, std_sec=1.1):
    print(SYNTHETIC_WARNING, "- synthetic_alert_latency() - DO NOT report these numbers as real")
    return np.clip(np.random.normal(mean_sec, std_sec, n_events), 0.5, None)
