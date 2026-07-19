"""
Real detection latency requires a working alert system, which doesn't
exist yet. This script builds and tests the measurement/reporting
pipeline now, using synthetic placeholder timings. DO NOT put these
numbers in the final report. Once the real app fires real alerts, swap
synthetic_alert_latency() for actual timestamped event logs.
"""
import numpy as np
import matplotlib.pyplot as plt
from synthetic_stubs import synthetic_alert_latency

def measure_latency(n_events=20):
    latencies = synthetic_alert_latency(n_events=n_events)

    print(f"Mean latency: {latencies.mean():.2f}s   Std: {latencies.std():.2f}s")
    print(f"Target: <5.0s   {'PASS (synthetic)' if latencies.mean() < 5 else 'FAIL (synthetic)'}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(latencies, bins=10, color="steelblue", edgecolor="black")
    ax.axvline(5.0, color="red", linestyle="--", label="5s target")
    ax.set_xlabel("Detection latency (seconds)")
    ax.set_title("SYNTHETIC latency data - placeholder only")
    ax.legend()
    plt.tight_layout()
    plt.savefig("latency_synthetic_PLACEHOLDER.png", dpi=150)
    print("\nSaved as 'latency_synthetic_PLACEHOLDER.png' - filename intentionally")
    print("flagged so it can't accidentally be mistaken for a real result.")

    return latencies

if __name__ == "__main__":
    measure_latency()
