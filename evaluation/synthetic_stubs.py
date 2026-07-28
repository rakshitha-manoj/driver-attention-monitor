"""
SYNTHETIC DATA MODULE - trimmed.

Head pose, PERCLOS, nod count, and yawn rate are now REAL (Sheethal's
Decision module is built). The only thing left with no real
equivalent is detection latency, which needs the live alert system
actually firing timestamped alerts - not something batch dataset
extraction can produce. Kept here, clearly flagged, until the full
live app exists.
"""
import numpy as np

SYNTHETIC_WARNING = "SYNTHETIC - no real equivalent exists yet"

def synthetic_alert_latency(n_events=20, mean_sec=3.2, std_sec=1.1):
    print(SYNTHETIC_WARNING, "- synthetic_alert_latency() - DO NOT report these numbers as real")
    return np.clip(np.random.normal(mean_sec, std_sec, n_events), 0.5, None)
