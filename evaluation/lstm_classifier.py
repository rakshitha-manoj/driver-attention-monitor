"""
LSTM classifier on sliding windows of [EAR, MAR, head_pitch, head_yaw].
All four channels are now REAL (Sheethal's head pose is no longer
synthetic). Sized conservatively for a 4GB VRAM card.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Set the environment variable immediately before any MLflow initializations
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from mlflow_utils import setup_experiment
import mlflow

# --- ROBUST PATH FIX ---
# Gets the absolute directory where this classifier script sits (evaluation/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolves absolute paths perfectly relative to the script location
CSV_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "nthu_features.csv"))
MODEL_SAVE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "lstm_model.pt"))

SEQ_LEN = 30
HIDDEN_SIZE = 32
NUM_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class WindowDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

class DrowsinessLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.sigmoid(self.fc(last_hidden)).squeeze(-1)

def build_sliding_windows(csv_path=CSV_PATH):
    print(f"Loading features from: {csv_path}")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        raise FileNotFoundError(f"Error: {csv_path} is missing or empty. Please run nthu_extraction.py completely first.")

    df = pd.read_csv(csv_path)
    df["label_bin"] = (df["label"] == "drowsy").astype(int)

    feature_cols = ["EAR", "MAR", "head_pitch", "head_yaw"]  # all real channels
    windows, labels = [], []

    for subj, group in df.groupby("subject_id"):
        group = group.sort_values("frame_id")
        feats = group[feature_cols].values
        labs = group["label_bin"].values
        for i in range(len(feats) - SEQ_LEN):
            windows.append(feats[i:i+SEQ_LEN])
            labels.append(1 if labs[i:i+SEQ_LEN].mean() > 0.5 else 0)

    return np.array(windows), np.array(labels), feature_cols

def train():
    X, y, feature_cols = build_sliding_windows()
    print(f"Built {len(X)} windows of shape {X.shape[1:]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(WindowDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = DrowsinessLSTM(input_size=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    setup_experiment("lstm_nthu")
    with mlflow.start_run(run_name="lstm_v2_real_features") as run:
        mlflow.log_params({
            "seq_len": SEQ_LEN, "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "features": feature_cols
        })

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                preds = model(xb).cpu().numpy()
                all_preds.extend((preds > 0.5).astype(int))
                all_labels.extend(yb.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)
        print(f"\nTest accuracy: {acc:.4f}  F1: {f1:.4f}")

        # Save weights and log directly to active MLflow context session
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved model weights locally to {MODEL_SAVE_PATH}")
        mlflow.log_artifact(MODEL_SAVE_PATH)
        print("Model artifact logged to MLflow successfully.")

    return model

if __name__ == "__main__":
    train()