from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import mlflow
from pathlib import Path

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
    save_dir: Path | None = None,
) -> dict[str, float]:
    """Evaluate classifier performance and generate metrics/plots."""
    # 1. Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # 2. Compute metrics
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }

    # 3. Print metrics
    print("\nModel Evaluation Results:")
    print("=========================")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    print(f"ROC AUC:             {roc_auc:.4f}")
    print(f"F1 Score:            {f1:.4f} (at threshold {threshold})")
    print(f"Precision:           {prec:.4f}")
    print(f"Recall:              {rec:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]:<8} FP: {cm[0,1]:<8}")
    print(f"FN: {cm[1,0]:<8} TP: {cm[1,1]:<8}")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=12)
        plt.ylabel('Actuals', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        cm_path = save_dir / "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        
        # Plot Precision-Recall Curve
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        pr_curve_path = save_dir / "pr_curve.png"
        plt.savefig(pr_curve_path, bbox_inches='tight')
        plt.close()

        # Log to MLflow if active
        if mlflow.active_run():
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(cm_path), "plots")
            mlflow.log_artifact(str(pr_curve_path), "plots")

    return metrics
