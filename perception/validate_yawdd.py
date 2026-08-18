"""
validate_yawdd.py
Validates your MAR threshold against the YawDD dataset.
Answers: does 0.6 actually separate yawning from non-yawning across many people?

Run: python3 perception/validate_yawdd.py
Dataset: data/yawdd/yawning/ and data/yawdd/non_yawning/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from perception import compute_mar, apply_clahe

YAWDD_YAWN_DIR    = "data/yawdd/yawning"
YAWDD_NONYAWN_DIR = "data/yawdd/non_yawning"
RESULTS_DIR       = "results"
MAR_THRESHOLD     = 0.6
FRAMES_PER_VIDEO  = 30

os.makedirs(RESULTS_DIR, exist_ok=True)

MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]
face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


def extract_mar_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(FRAMES_PER_VIDEO, total), dtype=int)
    mars = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        enhanced = apply_clahe(frame)
        rgb      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results  = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            continue
        lm   = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        try:
            mars.append(compute_mar(lm, MOUTH, w, h))
        except Exception:
            continue
    cap.release()
    return mars


def load_mars_from_folder(folder, label):
    mars, labels = [], []
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    print(f"Processing {len(files)} videos from {folder}...")
    for i, f in enumerate(files):
        if i % 5 == 0:
            print(f"  {i+1}/{len(files)}: {f}")
        vids = extract_mar_from_video(os.path.join(folder, f))
        mars.extend(vids)
        labels.extend([label] * len(vids))
    print(f"  Done. Frames extracted: {len(mars)}")
    return mars, labels


def find_optimal_mar_threshold(yawn_mars, nonyawn_mars):
    all_m = yawn_mars + nonyawn_mars
    all_l = [1] * len(yawn_mars) + [0] * len(nonyawn_mars)
    best_t, best_acc = MAR_THRESHOLD, 0.0
    for t in np.arange(0.2, 0.9, 0.01):
        preds = [1 if m >= t else 0 for m in all_m]
        acc   = accuracy_score(all_l, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_t, 2), round(best_acc * 100, 2)


def run_validation():
    print("=" * 60)
    print("YawDD Dataset Validation — MAR Threshold Check")
    print("=" * 60)

    if not os.path.exists(YAWDD_YAWN_DIR) or not os.path.exists(YAWDD_NONYAWN_DIR):
        print(f"ERROR: YawDD dataset not found.")
        print(f"Expected: {YAWDD_YAWN_DIR} and {YAWDD_NONYAWN_DIR}")
        print("Download: http://www.site.uottawa.ca/research/viva/projects/YawDD/")
        return

    yawn_m,    yawn_l    = load_mars_from_folder(YAWDD_YAWN_DIR,    label=1)
    nonyawn_m, nonyawn_l = load_mars_from_folder(YAWDD_NONYAWN_DIR, label=0)
    all_m = yawn_m + nonyawn_m
    all_l = yawn_l + nonyawn_l

    print(f"\nYawn frames:{len(yawn_m)}  Non-yawn:{len(nonyawn_m)}  Total:{len(all_m)}")
    if not all_m:
        print("No frames extracted. Check paths.")
        return

    preds_default = [1 if m >= MAR_THRESHOLD else 0 for m in all_m]
    acc_default   = accuracy_score(all_l, preds_default) * 100
    print(f"\nAt threshold {MAR_THRESHOLD}: accuracy = {acc_default:.2f}%")
    print(classification_report(all_l, preds_default, target_names=["Non-yawn","Yawn"]))

    opt_t, opt_acc = find_optimal_mar_threshold(yawn_m, nonyawn_m)
    print(f"Optimal MAR threshold: {opt_t}  accuracy: {opt_acc}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(nonyawn_m, bins=50, alpha=0.6, color="blue",
                 label=f"Non-yawn (n={len(nonyawn_m)})", density=True)
    axes[0].hist(yawn_m,   bins=50, alpha=0.6, color="orange",
                 label=f"Yawn (n={len(yawn_m)})",     density=True)
    axes[0].axvline(MAR_THRESHOLD, color="red",    linestyle="--",
                    linewidth=2, label=f"Default ({MAR_THRESHOLD})")
    axes[0].axvline(opt_t,         color="purple", linestyle="--",
                    linewidth=2, label=f"Optimal ({opt_t})")
    axes[0].set_xlabel("MAR")
    axes[0].set_ylabel("Density")
    axes[0].set_title("MAR Distribution — YawDD Dataset")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    thresholds = np.arange(0.2, 0.9, 0.01)
    accs       = [accuracy_score(all_l, [1 if m >= t else 0 for m in all_m]) * 100
                  for t in thresholds]
    axes[1].plot(thresholds, accs, color="green", linewidth=2)
    axes[1].axvline(MAR_THRESHOLD, color="red",    linestyle="--",
                    linewidth=2, label=f"Default ({MAR_THRESHOLD})")
    axes[1].axvline(opt_t,         color="purple", linestyle="--",
                    linewidth=2, label=f"Optimal ({opt_t})")
    axes[1].set_xlabel("MAR Threshold")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy vs MAR Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "yawdd_mar_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.show()

    with open(os.path.join(RESULTS_DIR, "yawdd_validation.txt"), "w") as f:
        f.write(f"YawDD Validation\n")
        f.write(f"Default ({MAR_THRESHOLD}) accuracy: {acc_default:.2f}%\n")
        f.write(f"Optimal ({opt_t}) accuracy: {opt_acc:.2f}%\n\n")
        f.write(classification_report(all_l, preds_default,
                                       target_names=["Non-yawn","Yawn"]))
    print("Report saved: perception/results/yawdd_validation.txt")


if __name__ == "__main__":
    run_validation()