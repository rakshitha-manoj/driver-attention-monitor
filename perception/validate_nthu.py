"""
validate_nthu.py
Tests your module across 4 NTHU conditions:
BareFace, Glasses, Sunglasses, Night

Tells you and Sheethal which conditions make EAR unreliable.
Run: python3 perception/validate_nthu.py
Dataset: data/nthu/Training_Evaluation_Dataset/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from perception import compute_ear, compute_mar, apply_clahe, compute_landmark_confidence

NTHU_ROOT        = "data/nthu/Training_Evaluation_Dataset"
RESULTS_DIR      = "results"
FRAMES_PER_VIDEO = 50

CONDITION_FOLDERS = {
    "BareFace"   : "BareFace",
    "Glasses"    : "Glasses",
    "Sunglasses" : "Sunglasses",
    "Night"      : "Night_Glasses",
}

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

os.makedirs(RESULTS_DIR, exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, min(FRAMES_PER_VIDEO, total), dtype=int)
    data    = {"ears":[], "confidences":[], "detected":[], "total": len(indices)}

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            data["detected"].append(False)
            continue
        enhanced  = apply_clahe(frame)
        rgb       = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        mp_result = face_mesh.process(rgb)
        if not mp_result.multi_face_landmarks:
            data["detected"].append(False)
            continue
        lm   = mp_result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        try:
            ear  = (compute_ear(lm, LEFT_EYE, w, h) + compute_ear(lm, RIGHT_EYE, w, h)) / 2
            conf = (compute_landmark_confidence(lm, LEFT_EYE) +
                    compute_landmark_confidence(lm, RIGHT_EYE)) / 2
            data["ears"].append(ear)
            data["confidences"].append(conf)
            data["detected"].append(True)
        except Exception:
            data["detected"].append(False)

    cap.release()
    return data


def run_condition(condition_name, folder_name, subject_dirs):
    all_ears, all_conf = [], []
    total_f, detected_f = 0, 0
    videos_done = 0
    print(f"\n  {condition_name}")

    for subj in subject_dirs:
        cpath = os.path.join(subj, folder_name)
        if not os.path.exists(cpath):
            continue
        vids = [f for f in os.listdir(cpath)
                if f.lower().endswith(('.avi','.mp4','.mov'))]
        for vf in vids[:3]:
            result = process_video(os.path.join(cpath, vf))
            if result is None:
                continue
            videos_done  += 1
            total_f      += result["total"]
            detected_f   += sum(result["detected"])
            all_ears.extend(result["ears"])
            all_conf.extend(result["confidences"])

    if not all_ears:
        print(f"    No data — check folder name '{folder_name}'")
        return None

    det_rate = detected_f / total_f * 100 if total_f > 0 else 0
    low_conf = sum(1 for c in all_conf if c < 0.5) / len(all_conf) * 100

    stats = {
        "condition"     : condition_name,
        "detection_rate": round(det_rate, 1),
        "ear_mean"      : round(float(np.mean(all_ears)), 4),
        "ear_std"       : round(float(np.std(all_ears)),  4),
        "conf_mean"     : round(float(np.mean(all_conf)), 3),
        "low_conf_pct"  : round(low_conf, 1),
        "all_ears"      : all_ears,
        "all_conf"      : all_conf,
    }
    print(f"    Detection:{det_rate:.1f}%  EAR mean:{stats['ear_mean']}  "
          f"Conf mean:{stats['conf_mean']}  Low conf:{low_conf:.1f}%")
    return stats


def run_validation():
    print("=" * 60)
    print("NTHU Robustness Test — Per-Condition Analysis")
    print("=" * 60)

    if not os.path.exists(NTHU_ROOT):
        print(f"ERROR: NTHU not found at {NTHU_ROOT}")
        print("Request access: http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/")
        return

    subject_dirs = sorted([
        os.path.join(NTHU_ROOT, n) for n in os.listdir(NTHU_ROOT)
        if os.path.isdir(os.path.join(NTHU_ROOT, n))
    ])
    print(f"Found {len(subject_dirs)} subjects.")

    all_stats = {}
    for cname, fname in CONDITION_FOLDERS.items():
        s = run_condition(cname, fname, subject_dirs)
        if s:
            all_stats[cname] = s

    print("\n" + "=" * 60)
    print(f"{'Condition':<15}{'Detection%':>11}{'EAR mean':>10}{'Conf mean':>10}{'Low conf%':>11}")
    print("-" * 60)
    for n, s in all_stats.items():
        print(f"{n:<15}{s['detection_rate']:>10}%{s['ear_mean']:>10}"
              f"{s['conf_mean']:>10}{s['low_conf_pct']:>10}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["blue","green","red","orange"]
    for (n, s), c in zip(all_stats.items(), colors):
        axes[0].hist(s["all_ears"], bins=40, alpha=0.5,
                     label=f"{n} (n={len(s['all_ears'])})", color=c, density=True)
    axes[0].axvline(0.25, color="black", linestyle="--", linewidth=2, label="Threshold 0.25")
    axes[0].set_xlabel("EAR")
    axes[0].set_ylabel("Density")
    axes[0].set_title("EAR by Condition (NTHU)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    names     = list(all_stats.keys())
    det_rates = [all_stats[n]["detection_rate"] for n in names]
    conf_pcts = [all_stats[n]["conf_mean"] * 100 for n in names]
    x, w      = np.arange(len(names)), 0.35
    axes[1].bar(x - w/2, det_rates, w, label="Detection rate (%)", color="steelblue")
    axes[1].bar(x + w/2, conf_pcts, w, label="Avg confidence ×100",  color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel("%")
    axes[1].set_title("Detection & Confidence by Condition")
    axes[1].legend()
    axes[1].set_ylim(0, 110)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "nthu_robustness.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {path}")
    plt.show()

    # Report for Sheethal
    with open(os.path.join(RESULTS_DIR, "nthu_robustness.txt"), "w") as f:
        f.write("NTHU Robustness Findings — for Sheethal (Decision layer)\n")
        f.write("=" * 50 + "\n\n")
        f.write("When ear_confidence < 0.5: reduce EAR weight in scoring,\n")
        f.write("increase MAR + head pose weight instead.\n\n")
        for n, s in all_stats.items():
            f.write(f"\n{n}:\n")
            f.write(f"  Detection rate: {s['detection_rate']}%\n")
            f.write(f"  EAR mean: {s['ear_mean']}  std: {s['ear_std']}\n")
            f.write(f"  Avg confidence: {s['conf_mean']}\n")
            f.write(f"  Frames with confidence < 0.5: {s['low_conf_pct']}%\n")
            if s["detection_rate"] < 80:
                f.write(f"  WARNING: Low detection rate — MediaPipe loses face frequently.\n")
            if s["low_conf_pct"] > 20:
                f.write(f"  WARNING: High proportion of low-confidence frames.\n")
            if s["detection_rate"] >= 80 and s["low_conf_pct"] <= 20:
                f.write(f"  OK: Good performance in this condition.\n")

    print("Report saved: perception/results/nthu_robustness.txt")
    print("Share this file with Sheethal for her fallback logic.")


if __name__ == "__main__":
    run_validation()