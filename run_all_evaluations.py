"""
run_all_evaluations.py — Run all dataset validations and generate report numbers.

Evaluates:
1. CEW dataset → EAR threshold accuracy, ROC AUC, optimal threshold
2. YawDD dataset → MAR threshold accuracy, yawn detection rate
3. NTHU dataset → Cross-condition robustness (bare face, glasses, sunglasses, night)

Run: python run_all_evaluations.py
Output: results/ directory with text reports and plots
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))
from perception import compute_ear, compute_mar, apply_clahe

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             precision_score, recall_score, f1_score)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

all_results = {}


# ══════════════════════════════════════════════════════════════
#  1. CEW Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_cew():
    print("\n" + "=" * 60)
    print("  CEW DATASET — EAR Threshold Validation")
    print("=" * 60)

    open_dir   = "data/cew/open"
    closed_dir = "data/cew/closed"
    if not os.path.exists(open_dir) or not os.path.exists(closed_dir):
        print("ERROR: CEW dataset not found")
        return

    def compute_ear_from_image(path):
        img = cv2.imread(path)
        if img is None:
            return None
        enhanced = apply_clahe(img)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]
        try:
            l = compute_ear(lm, LEFT_EYE, w, h)
            r = compute_ear(lm, RIGHT_EYE, w, h)
            return (l + r) / 2.0
        except:
            return None

    # Process open eyes (sample for speed — CEW has 14k+ open images)
    open_files = [f for f in os.listdir(open_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    closed_files = [f for f in os.listdir(closed_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Sample to keep runtime reasonable (~2000 from each class)
    max_per_class = 2000
    np.random.seed(42)
    if len(open_files) > max_per_class:
        open_files = list(np.random.choice(open_files, max_per_class, replace=False))
    if len(closed_files) > max_per_class:
        closed_files = list(np.random.choice(closed_files, max_per_class, replace=False))

    print(f"  Processing {len(open_files)} open + {len(closed_files)} closed images...")

    open_ears, closed_ears = [], []
    open_failed = closed_failed = 0

    for i, f in enumerate(open_files):
        if i % 200 == 0:
            print(f"    Open: {i}/{len(open_files)}...")
        ear = compute_ear_from_image(os.path.join(open_dir, f))
        if ear is not None:
            open_ears.append(ear)
        else:
            open_failed += 1

    for i, f in enumerate(closed_files):
        if i % 200 == 0:
            print(f"    Closed: {i}/{len(closed_files)}...")
        ear = compute_ear_from_image(os.path.join(closed_dir, f))
        if ear is not None:
            closed_ears.append(ear)
        else:
            closed_failed += 1

    print(f"  Open: {len(open_ears)} detected, {open_failed} failed")
    print(f"  Closed: {len(closed_ears)} detected, {closed_failed} failed")

    if not open_ears or not closed_ears:
        print("  ERROR: No EAR values computed")
        return

    # Evaluate at default threshold
    all_ears = open_ears + closed_ears
    all_labels = [0] * len(open_ears) + [1] * len(closed_ears)  # 0=open, 1=closed
    threshold = 0.25

    preds = [0 if e >= threshold else 1 for e in all_ears]
    acc = accuracy_score(all_labels, preds) * 100
    prec = precision_score(all_labels, preds) * 100
    rec = recall_score(all_labels, preds) * 100
    f1 = f1_score(all_labels, preds) * 100
    cm = confusion_matrix(all_labels, preds)

    # Find optimal threshold
    best_t, best_acc = threshold, acc
    for t in np.arange(0.05, 0.45, 0.005):
        p = [0 if e >= t else 1 for e in all_ears]
        a = accuracy_score(all_labels, p) * 100
        if a > best_acc:
            best_acc, best_t = a, t

    # ROC
    fpr, tpr, _ = roc_curve(all_labels, [-e for e in all_ears])
    roc_auc = auc(fpr, tpr)

    # Distribution stats
    open_mean = np.mean(open_ears)
    open_std = np.std(open_ears)
    closed_mean = np.mean(closed_ears)
    closed_std = np.std(closed_ears)

    report = classification_report(all_labels, preds, target_names=["Open", "Closed"])

    print(f"\n  ── CEW Results ──")
    print(f"  Total images evaluated: {len(all_ears)}")
    print(f"  Open eyes (mean±std):   {open_mean:.4f} ± {open_std:.4f}")
    print(f"  Closed eyes (mean±std): {closed_mean:.4f} ± {closed_std:.4f}")
    print(f"  Default threshold (0.25): Accuracy={acc:.2f}%  Precision={prec:.2f}%  Recall={rec:.2f}%  F1={f1:.2f}%")
    print(f"  Optimal threshold:        {best_t:.3f} (Accuracy={best_acc:.2f}%)")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix:\n  {cm}")
    print(report)

    # Save plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(open_ears, bins=50, alpha=0.6, color="green",
                 label=f"Open (n={len(open_ears)})", density=True)
    axes[0].hist(closed_ears, bins=50, alpha=0.6, color="red",
                 label=f"Closed (n={len(closed_ears)})", density=True)
    axes[0].axvline(threshold, color="blue", linestyle="--", lw=2, label=f"Default ({threshold})")
    axes[0].axvline(best_t, color="orange", linestyle="--", lw=2, label=f"Optimal ({best_t:.3f})")
    axes[0].set_xlabel("EAR"); axes[0].set_ylabel("Density")
    axes[0].set_title("EAR Distribution — CEW Dataset"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC={roc_auc:.3f}")
    axes[1].plot([0,1],[0,1], "navy", lw=1, ls="--")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve — EAR Eye State"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cew_evaluation.png"), dpi=150)
    plt.close()

    all_results["cew"] = {
        "total_images": len(all_ears),
        "open_count": len(open_ears), "closed_count": len(closed_ears),
        "open_mean": round(open_mean, 4), "open_std": round(open_std, 4),
        "closed_mean": round(closed_mean, 4), "closed_std": round(closed_std, 4),
        "default_threshold": threshold,
        "accuracy": round(acc, 2), "precision": round(prec, 2),
        "recall": round(rec, 2), "f1": round(f1, 2),
        "optimal_threshold": round(best_t, 3),
        "optimal_accuracy": round(best_acc, 2),
        "roc_auc": round(roc_auc, 4),
        "face_detection_rate": round(len(all_ears) / (len(all_ears) + open_failed + closed_failed) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════
#  2. YawDD Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_yawdd():
    print("\n" + "=" * 60)
    print("  YAWDD DATASET — MAR Threshold Validation")
    print("=" * 60)

    # Mirror directory has labeled filenames: *-Yawning.avi, *-Normal.avi, *-Talking.avi
    mirror_dirs = []
    for sub in ["Mirror/Mirror/Female_mirror", "Mirror/Mirror/Male_mirror Avi Videos"]:
        d = os.path.join("data/yawdd", sub)
        if os.path.exists(d):
            mirror_dirs.append(d)

    if not mirror_dirs:
        print("ERROR: YawDD Mirror directory not found")
        return

    def extract_mars_from_video(path, max_frames=30):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        mars = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            enhanced = apply_clahe(frame)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                continue
            lm = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            try:
                mars.append(compute_mar(lm, MOUTH, w, h))
            except:
                continue
        cap.release()
        return mars

    yawn_mars, nonyawn_mars = [], []
    yawn_videos = nonyawn_videos = 0

    for d in mirror_dirs:
        files = [f for f in os.listdir(d) if f.lower().endswith('.avi')]
        print(f"  Processing {len(files)} videos from {os.path.basename(d)}...")
        for i, f in enumerate(files):
            if i % 10 == 0:
                print(f"    {i}/{len(files)}: {f}")
            path = os.path.join(d, f)
            fname = f.lower()
            mars = extract_mars_from_video(path)
            if "yawning" in fname:
                yawn_mars.extend(mars)
                yawn_videos += 1
            elif "normal" in fname or "talking" in fname:
                nonyawn_mars.extend(mars)
                nonyawn_videos += 1

    print(f"  Yawn: {len(yawn_mars)} frames from {yawn_videos} videos")
    print(f"  Non-yawn: {len(nonyawn_mars)} frames from {nonyawn_videos} videos")

    if not yawn_mars or not nonyawn_mars:
        print("  ERROR: No frames extracted")
        return

    all_mars = yawn_mars + nonyawn_mars
    all_labels = [1] * len(yawn_mars) + [0] * len(nonyawn_mars)
    threshold = 0.6

    preds = [1 if m >= threshold else 0 for m in all_mars]
    acc = accuracy_score(all_labels, preds) * 100
    prec = precision_score(all_labels, preds, zero_division=0) * 100
    rec = recall_score(all_labels, preds) * 100
    f1 = f1_score(all_labels, preds) * 100
    cm = confusion_matrix(all_labels, preds)

    # Optimal threshold
    best_t, best_acc = threshold, acc
    for t in np.arange(0.2, 0.9, 0.01):
        p = [1 if m >= t else 0 for m in all_mars]
        a = accuracy_score(all_labels, p) * 100
        if a > best_acc:
            best_acc, best_t = a, t

    # ROC
    fpr, tpr, _ = roc_curve(all_labels, all_mars)
    roc_auc = auc(fpr, tpr)

    report = classification_report(all_labels, preds, target_names=["Non-yawn", "Yawn"])

    print(f"\n  ── YawDD Results ──")
    print(f"  Yawn MAR (mean±std):     {np.mean(yawn_mars):.4f} ± {np.std(yawn_mars):.4f}")
    print(f"  Non-yawn MAR (mean±std): {np.mean(nonyawn_mars):.4f} ± {np.std(nonyawn_mars):.4f}")
    print(f"  Default threshold (0.6): Accuracy={acc:.2f}%  Precision={prec:.2f}%  Recall={rec:.2f}%  F1={f1:.2f}%")
    print(f"  Optimal threshold:       {best_t:.2f} (Accuracy={best_acc:.2f}%)")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(report)

    # Save plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(nonyawn_mars, bins=50, alpha=0.6, color="blue",
                 label=f"Non-yawn (n={len(nonyawn_mars)})", density=True)
    axes[0].hist(yawn_mars, bins=50, alpha=0.6, color="orange",
                 label=f"Yawn (n={len(yawn_mars)})", density=True)
    axes[0].axvline(threshold, color="red", ls="--", lw=2, label=f"Default ({threshold})")
    axes[0].axvline(best_t, color="purple", ls="--", lw=2, label=f"Optimal ({best_t:.2f})")
    axes[0].set_xlabel("MAR"); axes[0].set_ylabel("Density")
    axes[0].set_title("MAR Distribution — YawDD Dataset"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC={roc_auc:.3f}")
    axes[1].plot([0,1],[0,1], "navy", lw=1, ls="--")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve — MAR Yawn Detection"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "yawdd_evaluation.png"), dpi=150)
    plt.close()

    all_results["yawdd"] = {
        "total_frames": len(all_mars),
        "yawn_frames": len(yawn_mars), "nonyawn_frames": len(nonyawn_mars),
        "yawn_videos": yawn_videos, "nonyawn_videos": nonyawn_videos,
        "yawn_mean": round(np.mean(yawn_mars), 4),
        "nonyawn_mean": round(np.mean(nonyawn_mars), 4),
        "default_threshold": threshold,
        "accuracy": round(acc, 2), "precision": round(prec, 2),
        "recall": round(rec, 2), "f1": round(f1, 2),
        "optimal_threshold": round(best_t, 2),
        "optimal_accuracy": round(best_acc, 2),
        "roc_auc": round(roc_auc, 4),
    }


# ══════════════════════════════════════════════════════════════
#  3. NTHU Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_nthu():
    print("\n" + "=" * 60)
    print("  NTHU-DDD DATASET — Cross-Condition Robustness")
    print("=" * 60)

    base = "data/nthu/Multi class/train"
    drowsy_dir = os.path.join(base, "drowsy")
    notdrowsy_dir = os.path.join(base, "notdrowsy")

    if not os.path.exists(drowsy_dir) or not os.path.exists(notdrowsy_dir):
        print("ERROR: NTHU dataset not found")
        return

    def get_condition(filename):
        """Extract condition from NTHU filename: glasses, noglasses, sunglasses, nightglasses."""
        fn = filename.lower()
        if "nightnoglasses" in fn or "night_noglasses" in fn:
            return "night_noglasses"
        elif "nightglasses" in fn or "night_glasses" in fn:
            return "night_glasses"
        elif "sunglasses" in fn:
            return "sunglasses"
        elif "noglasses" in fn:
            return "noglasses"
        elif "glasses" in fn:
            return "glasses"
        return "unknown"

    def compute_ear_from_image(path):
        img = cv2.imread(path)
        if img is None:
            return None, None
        enhanced = apply_clahe(img)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None, None
        lm = results.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]
        # Compute confidence from visibility scores
        eye_lms = LEFT_EYE + RIGHT_EYE
        vis = [lm[i].visibility for i in eye_lms if hasattr(lm[i], 'visibility')]
        confidence = np.mean(vis) if vis else 0.5
        try:
            l = compute_ear(lm, LEFT_EYE, w, h)
            r = compute_ear(lm, RIGHT_EYE, w, h)
            return (l + r) / 2.0, confidence
        except:
            return None, None

    # Collect all image files with conditions
    condition_stats = {}
    max_per_condition = 500  # Sample for speed

    for label_name, label_dir, label_val in [("drowsy", drowsy_dir, 1), ("notdrowsy", notdrowsy_dir, 0)]:
        files = []
        for root, dirs, fnames in os.walk(label_dir):
            for f in fnames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files.append(os.path.join(root, f))

        print(f"  Found {len(files)} {label_name} images")

        # Group by condition
        by_condition = {}
        for fp in files:
            cond = get_condition(os.path.basename(fp))
            by_condition.setdefault(cond, []).append(fp)

        for cond, cond_files in by_condition.items():
            if cond not in condition_stats:
                condition_stats[cond] = {
                    "ears": [], "labels": [], "confidences": [],
                    "detected": 0, "failed": 0, "total": 0
                }

            # Sample
            np.random.seed(42)
            sample = list(np.random.choice(cond_files, min(max_per_condition, len(cond_files)), replace=False))
            print(f"    {cond} ({label_name}): sampling {len(sample)}/{len(cond_files)}...")

            for i, fp in enumerate(sample):
                if i % 100 == 0 and i > 0:
                    print(f"      {i}/{len(sample)}...")
                condition_stats[cond]["total"] += 1
                ear, conf = compute_ear_from_image(fp)
                if ear is not None:
                    condition_stats[cond]["ears"].append(ear)
                    condition_stats[cond]["labels"].append(label_val)
                    condition_stats[cond]["confidences"].append(conf)
                    condition_stats[cond]["detected"] += 1
                else:
                    condition_stats[cond]["failed"] += 1

    # Per-condition evaluation
    print(f"\n  ── NTHU Results by Condition ──")
    print(f"  {'Condition':<20} {'Detect%':>8} {'EAR_mean':>8} {'EAR_std':>8} {'Conf':>8} {'LowConf%':>8} {'Acc@0.25':>8} {'N':>6}")
    print(f"  {'-'*88}")

    nthu_results = {}
    threshold = 0.25

    for cond in sorted(condition_stats.keys()):
        stats = condition_stats[cond]
        if not stats["ears"]:
            continue

        n = len(stats["ears"])
        detect_rate = stats["detected"] / max(stats["total"], 1) * 100
        ear_mean = np.mean(stats["ears"])
        ear_std = np.std(stats["ears"])
        conf_mean = np.mean(stats["confidences"])
        low_conf_pct = sum(1 for c in stats["confidences"] if c < 0.5) / n * 100

        # Accuracy: classify as drowsy (1) if EAR < threshold
        preds = [1 if e < threshold else 0 for e in stats["ears"]]
        acc = accuracy_score(stats["labels"], preds) * 100

        print(f"  {cond:<20} {detect_rate:>7.1f}% {ear_mean:>8.4f} {ear_std:>8.4f} {conf_mean:>8.3f} {low_conf_pct:>7.1f}% {acc:>7.1f}% {n:>6}")

        nthu_results[cond] = {
            "total_images": stats["total"],
            "detected": stats["detected"],
            "detection_rate": round(detect_rate, 1),
            "ear_mean": round(ear_mean, 4),
            "ear_std": round(ear_std, 4),
            "confidence_mean": round(conf_mean, 3),
            "low_confidence_pct": round(low_conf_pct, 1),
            "accuracy_at_025": round(acc, 1),
            "n_samples": n,
        }

    # Overall NTHU accuracy
    all_ears = []
    all_labels = []
    all_confs = []
    for stats in condition_stats.values():
        all_ears.extend(stats["ears"])
        all_labels.extend(stats["labels"])
        all_confs.extend(stats["confidences"])

    if all_ears:
        preds = [1 if e < threshold else 0 for e in all_ears]
        overall_acc = accuracy_score(all_labels, preds) * 100
        overall_prec = precision_score(all_labels, preds, zero_division=0) * 100
        overall_rec = recall_score(all_labels, preds) * 100
        overall_f1 = f1_score(all_labels, preds) * 100
        fpr, tpr, _ = roc_curve(all_labels, [-e for e in all_ears])
        roc_auc = auc(fpr, tpr)

        print(f"\n  Overall NTHU: Accuracy={overall_acc:.2f}%  Precision={overall_prec:.2f}%  "
              f"Recall={overall_rec:.2f}%  F1={overall_f1:.2f}%  AUC={roc_auc:.4f}")
        print(f"  Total samples: {len(all_ears)}")

        nthu_results["overall"] = {
            "accuracy": round(overall_acc, 2),
            "precision": round(overall_prec, 2),
            "recall": round(overall_rec, 2),
            "f1": round(overall_f1, 2),
            "roc_auc": round(roc_auc, 4),
            "total_samples": len(all_ears),
        }

    # Save condition comparison plot
    conds = sorted([c for c in condition_stats.keys() if condition_stats[c]["ears"]])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot of EAR by condition
    ear_data = [condition_stats[c]["ears"] for c in conds]
    try:
        axes[0].boxplot(ear_data, tick_labels=conds)
    except TypeError:
        axes[0].boxplot(ear_data, labels=conds)
    axes[0].axhline(threshold, color="red", ls="--", lw=1, label=f"Threshold ({threshold})")
    axes[0].set_ylabel("EAR")
    axes[0].set_title("EAR Distribution by Condition -- NTHU")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=30)

    # Detection rate bar chart
    det_rates = [condition_stats[c]["detected"] / max(condition_stats[c]["total"], 1) * 100
                 for c in conds]
    colors = ['green' if r > 90 else 'orange' if r > 70 else 'red' for r in det_rates]
    axes[1].bar(conds, det_rates, color=colors, alpha=0.7)
    axes[1].set_ylabel("Face Detection Rate (%)")
    axes[1].set_title("Face Detection Rate by Condition — NTHU")
    axes[1].tick_params(axis='x', rotation=30)
    for i, v in enumerate(det_rates):
        axes[1].text(i, v + 1, f"{v:.0f}%", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "nthu_evaluation.png"), dpi=150)
    plt.close()

    all_results["nthu"] = nthu_results


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start = time.time()
    print("=" * 50)
    print("  DRIVER ATTENTION MONITOR -- FULL EVALUATION  ")
    print("=" * 50)

    evaluate_cew()
    evaluate_yawdd()
    evaluate_nthu()

    # Save all results as JSON
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ALL EVALUATIONS COMPLETE -- {elapsed:.0f} seconds")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"{'='*60}")
