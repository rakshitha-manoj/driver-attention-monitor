import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import matplotlib.pyplot as plt

from mlflow_utils import setup_experiment
import mlflow
from sklearn.metrics import accuracy_score

IMG_SIZE = (64, 64)

def apply_blur(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_brightness_shift(img, delta):
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

def apply_occlusion(img, box_frac=0.3):
    h, w = img.shape[:2]
    bh, bw = int(h * box_frac), int(w * box_frac)
    y0 = np.random.randint(0, h - bh)
    x0 = np.random.randint(0, w - bw)
    out = img.copy()
    out[y0:y0+bh, x0:x0+bw] = 0
    return out

DEGRADATIONS = {
    "clean":            lambda img: img,
    "blur_light":       lambda img: apply_blur(img, 3),
    "blur_heavy":       lambda img: apply_blur(img, 9),
    "bright_up":        lambda img: apply_brightness_shift(img, 60),
    "bright_down":      lambda img: apply_brightness_shift(img, -60),
    "occlusion":        lambda img: apply_occlusion(img, 0.3),
}

def extract_hog(img_gray):
    img_gray = cv2.resize(img_gray, IMG_SIZE)
    return hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")

def run_degradation_test(model_path, test_images, test_labels):
    clf = joblib.load(model_path)

    setup_experiment("degradation_stress_test")
    results = {}

    with mlflow.start_run(run_name="hog_svm_degradation_sweep"):
        for name, transform in DEGRADATIONS.items():
            X = []
            for img in test_images:
                degraded = transform(img)
                X.append(extract_hog(degraded))
            X = np.array(X)
            y_pred = clf.predict(X)
            acc = accuracy_score(test_labels, y_pred)
            results[name] = acc
            mlflow.log_metric(f"acc_{name}", acc)
            print(f"{name:15s}: accuracy = {acc:.4f}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(results.keys(), results.values(), color="steelblue")
        ax.axhline(results["clean"], color="red", linestyle="--", label="clean baseline")
        ax.set_ylabel("Accuracy")
        ax.set_title("HOG+SVM robustness under simulated degradation")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        mlflow.log_figure(fig, "degradation_robustness.png")
        plt.savefig("degradation_robustness.png", dpi=150)
        plt.close(fig)

    return results

if __name__ == "__main__":
    from hog_svm_baseline import CEW_OPEN_DIR, CEW_CLOSED_DIR
    import glob
    test_imgs, test_labels = [], []
    for p in glob.glob(os.path.join(CEW_OPEN_DIR, "*.jpg"))[:50]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            test_imgs.append(img); test_labels.append(1)
    for p in glob.glob(os.path.join(CEW_CLOSED_DIR, "*.jpg"))[:50]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            test_imgs.append(img); test_labels.append(0)
    run_degradation_test("hog_svm_model.pkl", test_imgs, test_labels)
