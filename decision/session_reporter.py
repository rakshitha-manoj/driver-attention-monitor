"""
session_reporter.py — Decision / Reporting Layer
Generates an end-of-session driver trip report with performance metrics,
fatigue timeline visualizations, alert history, and a safety score.

Outputs:
  - Console ASCII summary box
  - Graphic timeline artifact: results/trip_summary.png
  - Summary JSON: results/trip_summary.json
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SessionReporter:
    """
    Analyzes session log data and creates visual & textual trip reports.
    """

    def __init__(self, log_filepath="output_log.csv", results_dir="results"):
        self.log_filepath = log_filepath
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def generate_report(self, duration_sec=None):
        """Parse the session CSV and compile summary statistics & charts."""
        if not os.path.exists(self.log_filepath):
            print(f"No log file found at {self.log_filepath} to generate report.")
            return None

        try:
            df = pd.read_csv(self.log_filepath)
        except Exception as e:
            print(f"Error reading session log: {e}")
            return None

        if len(df) < 5:
            print("Session too brief (< 5 frames) for a meaningful summary.")
            return None

        total_frames = len(df)
        session_duration = duration_sec if duration_sec else (
            float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if "timestamp" in df else total_frames / 30.0
        )
        avg_fps = float(total_frames / max(1.0, session_duration))

        # Face detection rate
        face_detect_pct = (
            float((df["landmarks_detected"] == True).sum() / total_frames * 100.0)
            if "landmarks_detected" in df else 100.0
        )

        # Drowsiness metrics
        mean_score = float(df["drowsiness_score"].mean()) if "drowsiness_score" in df else 0.0
        max_score = float(df["drowsiness_score"].max()) if "drowsiness_score" in df else 0.0
        mean_perclos = float(df["PERCLOS"].mean() * 100.0) if "PERCLOS" in df else 0.0
        max_perclos = float(df["PERCLOS"].max() * 100.0) if "PERCLOS" in df else 0.0

        # Event counts
        total_blinks = int(df["blink_count"].max()) if "blink_count" in df else 0
        total_yawns = int(df["yawn_count"].max()) if "yawn_count" in df else 0
        total_nods = int(df["nod_count"].max()) if "nod_count" in df else 0
        avg_blink_dur = (
            float(df["blink_dur_avg"].iloc[-1] * 1000.0) if "blink_dur_avg" in df and len(df["blink_dur_avg"]) > 0 else 130.0
        )

        # State percentages
        warning_frames = (df["system_state"] == "WARNING").sum() if "system_state" in df else 0
        critical_frames = (df["system_state"] == "CRITICAL").sum() if "system_state" in df else 0
        alert_frames = total_frames - warning_frames - critical_frames

        pct_alert = float(alert_frames / total_frames * 100.0)
        pct_warning = float(warning_frames / total_frames * 100.0)
        pct_critical = float(critical_frames / total_frames * 100.0)

        # Distractions & Hand presence
        distract_frames = int((df["distraction_flag"] == True).sum()) if "distraction_flag" in df else 0
        gaze_distract_frames = int((df["gaze_distraction"] == True).sum()) if "gaze_distraction" in df else 0
        hand_frames = int((df["hand_detected"] == True).sum()) if "hand_detected" in df else 0
        obj_frames = int((df["object_in_hand"] == True).sum()) if "object_in_hand" in df else 0

        # Heart rate
        # Compute Overall Driver Safety Grade (0–100 score)
        # Deduct for critical time, warning time, severe yawns/nods/distractions
        safety_penalty = (
            (pct_warning * 0.8) +
            (pct_critical * 2.5) +
            (min(30, total_yawns * 2.0)) +
            (min(30, total_nods * 5.0)) +
            (min(20, (distract_frames / total_frames) * 30.0))
        )
        safety_score = max(0, min(100, int(100 - safety_penalty)))

        if safety_score >= 85:
            safety_grade = "A (Alert & Focused)"
        elif safety_score >= 70:
            safety_grade = "B (Mild Fatigue Detected)"
        elif safety_score >= 50:
            safety_grade = "C (Moderate Drowsiness - Break Recommended)"
        else:
            safety_grade = "D (High Risk - Immediate Rest Required)"

        summary_data = {
            "session_duration_sec": round(session_duration, 1),
            "total_frames": total_frames,
            "average_fps": round(avg_fps, 1),
            "face_detection_rate": round(face_detect_pct, 1),
            "safety_score": safety_score,
            "safety_grade": safety_grade,
            "drowsiness_mean_score": round(mean_score, 1),
            "drowsiness_max_score": round(max_score, 1),
            "perclos_mean_pct": round(mean_perclos, 1),
            "perclos_max_pct": round(max_perclos, 1),
            "total_blinks": total_blinks,
            "avg_blink_duration_ms": round(avg_blink_dur, 1),
            "total_yawns": total_yawns,
            "total_nods": total_nods,
            "warning_time_pct": round(pct_warning, 1),
            "critical_time_pct": round(pct_critical, 1),
            "distraction_instances_pct": round(distract_frames / total_frames * 100.0, 1),
        }

        # Save JSON
        with open(os.path.join(self.results_dir, "trip_summary.json"), "w") as f:
            json.dump(summary_data, f, indent=2)

        # Plot graphic timeline
        self._plot_timeline(df, session_duration, summary_data)

        # Print Console ASCII Box
        self._print_console_summary(summary_data)

        return summary_data

    def _plot_timeline(self, df, duration, summary):
        """Generate trip summary timeline visualization graph."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        t = df["timestamp"] if "timestamp" in df else np.linspace(0, duration, len(df))

        # Subplot 1: Drowsiness Score & States
        axes[0].plot(t, df["drowsiness_score"], color="black", lw=1.5, label="Drowsiness Score (0-100)")
        axes[0].axhline(30, color="orange", linestyle="--", alpha=0.7, label="Warning (30)")
        axes[0].axhline(60, color="red", linestyle="--", alpha=0.7, label="Critical (60)")
        axes[0].fill_between(t, 0, df["drowsiness_score"], color="skyblue", alpha=0.3)
        axes[0].set_ylabel("Score (0–100)")
        axes[0].set_title(f"Driver Trip Summary — Safety Score: {summary['safety_score']}/100 [{summary['safety_grade']}]")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(alpha=0.3)

        # Subplot 2: PERCLOS & Gaze / Head Distraction
        if "PERCLOS" in df:
            axes[1].plot(t, df["PERCLOS"] * 100, color="blue", label="PERCLOS % (Eye Closure)")
            axes[1].axhline(15, color="purple", linestyle=":", label="Drowsy PERCLOS (15%)")
        if "distraction_flag" in df:
            distract_pts = np.where(df["distraction_flag"] == True, 20, np.nan)
            axes[1].scatter(t, distract_pts, color="orange", s=10, label="Head Distraction")
        if "gaze_distraction" in df:
            gaze_pts = np.where(df["gaze_distraction"] == True, 10, np.nan)
            axes[1].scatter(t, gaze_pts, color="green", s=10, label="Gaze Off-Road")
        axes[1].set_ylabel("Attention (%)")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(alpha=0.3)

        # Subplot 3: EAR & MAR Signals
        if "EAR" in df:
            axes[2].plot(t, df["EAR"], color="teal", lw=1.2, label="Eye Aspect Ratio (EAR)")
        if "MAR" in df:
            axes[2].plot(t, df["MAR"], color="magenta", lw=1.0, alpha=0.7, label="Mouth Aspect Ratio (MAR)")
        axes[2].set_ylabel("Aspect Ratio")
        axes[2].set_xlabel("Trip Time (seconds)")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, "trip_summary.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Trip summary visualization saved: {plot_path}")

    def _print_console_summary(self, s):
        """Print clean ASCII summary box."""
        mins = int(s["session_duration_sec"] // 60)
        secs = int(s["session_duration_sec"] % 60)
        print("\n" + "=" * 62)
        print(f"  {'DRIVER ATTENTION MONITOR -- TRIP SUMMARY':^58}")
        print("=" * 62)
        print(f"  Duration: {mins:02d}m {secs:02d}s ({s['total_frames']} frames @ {s['average_fps']} FPS)")
        print(f"  Face Detection Rate: {s['face_detection_rate']}%")
        print(f"  Overall Safety Score: {s['safety_score']}/100  [{s['safety_grade']}]")
        print("  " + "-" * 58)
        print(f"  Drowsiness Score (Mean / Max): {s['drowsiness_mean_score']:4.1f} / {s['drowsiness_max_score']:4.1f}")
        print(f"  PERCLOS (Mean / Max):          {s['perclos_mean_pct']:4.1f}% / {s['perclos_max_pct']:4.1f}%")
        print(f"  Blinks: {s['total_blinks']} (Avg: {s['avg_blink_duration_ms']:.0f}ms) | Yawns: {s['total_yawns']} | Nods: {s['total_nods']}")
        print(f"  Warning Time: {s['warning_time_pct']:4.1f}%  | Critical Time: {s['critical_time_pct']:4.1f}%")
        print("=" * 62 + "\n")
