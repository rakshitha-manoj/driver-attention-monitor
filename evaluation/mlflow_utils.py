import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, accuracy_score
import seaborn as sns
import numpy as np

def setup_experiment(experiment_name):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

def log_classification_run(run_name, params, y_true, y_pred, y_scores=None):
    """Logs params, metrics, confusion matrix, and ROC curve (if scores given) to MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {run_name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            mlflow.log_metric("roc_auc", roc_auc)

            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title(f"ROC Curve - {run_name}")
            ax2.legend()
            mlflow.log_figure(fig2, "roc_curve.png")
            plt.close(fig2)

        print(f"[{run_name}] acc={acc:.4f}  f1={f1:.4f}")
        return {"accuracy": acc, "f1": f1}
