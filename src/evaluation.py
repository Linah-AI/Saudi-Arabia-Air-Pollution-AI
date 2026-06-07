# =============================================================================
# evaluation.py
# Evaluates the trained model using Stratified 5-Fold Cross-Validation
# and generates a confusion matrix and classification report.
#
# Why Stratified 5-Fold CV?
#   With 500 rows, Stratified K-Fold provides a reliable generalisation
#   estimate while preserving class distribution in every fold. It is more
#   computationally efficient than LOOCV on this dataset size.
#
# Reference:
#   Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
# =============================================================================

import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Use Stratified 5-Fold CV instead of full LOOCV for speed on 500 rows;
# still robust and avoids class-imbalance issues.
CV_FOLDS    = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    """
    Computes weighted Accuracy, Precision, Recall, and F1-score.

    Parameters
    ----------
    y_true : array-like — ground truth labels
    y_pred : array-like — predicted labels

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1_score
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred,        average="weighted", zero_division=0), 4),
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """
    Plots and optionally saves a colour-coded confusion matrix.

    Parameters
    ----------
    y_true    : array-like
    y_pred    : array-like
    save_path : str or None — file path to save the PNG
    """
    labels = ["High", "Low", "Medium"]
    cm     = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, label="Count")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold")

    ax.set_title(f"Confusion Matrix – Stratified {CV_FOLDS}-Fold CV",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label",    fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[INFO] Confusion matrix saved → {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Feature importance bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, save_path: str = None):
    """
    Plots a horizontal bar chart of Decision Tree feature importances.

    Parameters
    ----------
    model        : fitted DecisionTreeClassifier
    feature_names: list of str
    save_path    : str or None
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    colors      = ["#1F3864" if imp > 0.1 else "#8B6914" for imp in importances[indices]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Decision Tree – Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[INFO] Feature importance chart saved → {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Save metrics to CSV
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, save_path: str = "outputs/evaluation_results.csv"):
    """
    Appends evaluation metrics with a timestamp to a CSV file.

    Parameters
    ----------
    metrics   : dict — output of compute_metrics()
    save_path : str
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_exists = os.path.isfile(save_path)

    with open(save_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["accuracy", "precision", "recall",
                                                "f1_score", "timestamp", "model", "dataset"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            **metrics,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model":     "Decision Tree",
            "dataset":   "Saudi Arabia Air Pollution – 13 Regions",
        })
    print(f"[INFO] Metrics saved → {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(model, X: pd.DataFrame, y: pd.Series,
             cm_save_path: str     = "outputs/cv_confusion_matrix.png",
             fi_save_path: str     = "outputs/feature_importance.png",
             csv_save_path: str    = "outputs/evaluation_results.csv"):
    """
    Runs Stratified K-Fold Cross-Validation, prints a full report,
    plots a confusion matrix and feature importance chart, and saves results.

    Parameters
    ----------
    model         : fitted DecisionTreeClassifier
    X             : pd.DataFrame — feature matrix
    y             : pd.Series   — true labels
    cm_save_path  : str         — path to save confusion matrix PNG
    fi_save_path  : str         — path to save feature importance PNG
    csv_save_path : str         — path to save metrics CSV
    """
    print("\n" + "=" * 60)
    print(f"EVALUATION – Stratified {CV_FOLDS}-Fold Cross-Validation")
    print("=" * 60)

    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_pred  = cross_val_predict(model, X, y, cv=cv)

    metrics = compute_metrics(y, y_pred)
    print(f"\n  Accuracy  : {metrics['accuracy']  * 100:.2f}%")
    print(f"  Precision : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall    : {metrics['recall']    * 100:.2f}%")
    print(f"  F1-Score  : {metrics['f1_score']  * 100:.2f}%")

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y, y_pred, digits=4, zero_division=0))

    plot_confusion_matrix(y, y_pred, save_path=cm_save_path)
    plot_feature_importance(model, list(X.columns), save_path=fi_save_path)
    save_metrics(metrics, save_path=csv_save_path)

    return metrics
