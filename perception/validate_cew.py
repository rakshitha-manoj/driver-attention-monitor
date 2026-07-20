"""
validate_cew.py
Validates your EAR threshold against the CEW dataset (2,423 eye images).
Answers: does 0.25 actually separate open from closed eyes across many people?

Run: python3 perception/validate_cew.py
Needs: pip install scikit-learn matplotlib
Dataset: data/cew/openEyes/ and data/cew/closedEyes/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from perception import compute_ear, apply_clahe

CEW_OPEN_DIR   = "data/cew/openEyes"
CEW_CLOSED_DIR = "data/cew/closedEyes"
RESULTS_DIR    = "results"
EAR_THRESHOLD  = 0.25

os.makedirs(RESULTS_DIR, exist_ok=True)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


def compute_ear_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    enhanced = apply_clahe(img)
    rgb      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    results  = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w      = img.shape[:2]
    try:
        left_ear  = compute_ear(landmarks, LEFT_EYE,  w, h)
        right_ear = compute_ear(landmarks, RIGHT_EYE, w, h)
        return (left_ear + right_ear) / 2.0
    except Exception:
        return None


def load_dataset(folder, label):
    ears, labels, failed = [], [], 0
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Processing {len(files)} images from {folder}...")
    for i, f in enumerate(files):
        if i % 100 == 0:
            print(f"  {i}/{len(files)}...")
        ear = compute_ear_from_image(os.path.join(folder, f))
        if ear is not None:
            ears.append(ear)
            labels.append(label)
        else:
            failed += 1
    print(f"  Done. Detected:{len(ears)} Skipped:{failed}")
    return ears, labels


def find_optimal_threshold(open_ears, closed_ears):
    all_ears   = open_ears + closed_ears
    all_labels = [0] * len(open_ears) + [1] * len(closed_ears)
    best_t, best_acc = 0.25, 0.0
    for t in np.arange(0.05, 0.45, 0.005):
        preds = [0 if e >= t else 1 for e in all_ears]
        acc   = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_t, 3), round(best_acc * 100, 2)


def run_validation():
    print("=" * 60)
    print("CEW Dataset Validation — EAR Threshold Check")
    print("=" * 60)

    if not os.path.exists(CEW_OPEN_DIR) or not os.path.exists(CEW_CLOSED_DIR):
        print(f"ERROR: CEW dataset not found.")
        print(f"Expected: {CEW_OPEN_DIR} and {CEW_CLOSED_DIR}")
        print("Download from Kaggle: search 'CEW closed eyes in the wild'")
        return

    open_ears,   open_labels   = load_dataset(CEW_OPEN_DIR,   label=0)
    closed_ears, closed_labels = load_dataset(CEW_CLOSED_DIR, label=1)
    all_ears   = open_ears   + closed_ears
    all_labels = open_labels + closed_labels

    print(f"\nOpen:{len(open_ears)}  Closed:{len(closed_ears)}  Total:{len(all_ears)}")

    preds_default = [0 if e >= EAR_THRESHOLD else 1 for e in all_ears]
    acc_default   = accuracy_score(all_labels, preds_default) * 100
    cm            = confusion_matrix(all_labels, preds_default)

    print(f"\nAt threshold {EAR_THRESHOLD}: accuracy = {acc_default:.2f}%")
    print(classification_report(all_labels, preds_default, target_names=["Open","Closed"]))

    opt_t, opt_acc = find_optimal_threshold(open_ears, closed_ears)
    print(f"Optimal threshold: {opt_t}  accuracy: {opt_acc}%")
    if abs(opt_t - EAR_THRESHOLD) > 0.03:
        print(f"NOTE: Consider updating threshold from {EAR_THRESHOLD} to {opt_t}")
    else:
        print(f"GOOD: Default threshold {EAR_THRESHOLD} is close to optimal.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(open_ears,   bins=50, alpha=0.6, color="green",
                 label=f"Open (n={len(open_ears)})",   density=True)
    axes[0].hist(closed_ears, bins=50, alpha=0.6, color="red",
                 label=f"Closed (n={len(closed_ears)})", density=True)
    axes[0].axvline(EAR_THRESHOLD, color="blue",   linestyle="--",
                    linewidth=2, label=f"Default ({EAR_THRESHOLD})")
    axes[0].axvline(opt_t,         color="orange", linestyle="--",
                    linewidth=2, label=f"Optimal ({opt_t})")
    axes[0].set_xlabel("EAR")
    axes[0].set_ylabel("Density")
    axes[0].set_title("EAR Distribution — CEW Dataset")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    fpr, tpr, _ = roc_curve(all_labels, [-e for e in all_ears])
    roc_auc     = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC={roc_auc:.3f}")
    axes[1].plot([0,1],[0,1], color="navy", lw=1, linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve — EAR Eye State")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "cew_ear_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.show()

    with open(os.path.join(RESULTS_DIR, "cew_validation.txt"), "w") as f:
        f.write(f"CEW Validation\nTotal:{len(all_ears)}\n")
        f.write(f"Default threshold ({EAR_THRESHOLD}) accuracy: {acc_default:.2f}%\n")
        f.write(f"Optimal threshold ({opt_t}) accuracy: {opt_acc:.2f}%\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        f.write(classification_report(all_labels, preds_default,
                                       target_names=["Open","Closed"]))
    print("Report saved: perception/results/cew_validation.txt")


if __name__ == "__main__":
    run_validation()