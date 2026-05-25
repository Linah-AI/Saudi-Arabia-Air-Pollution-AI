# =============================================================================
# main.py
# Entry point for the Saudi Arabia Air Pollution Risk Classification Pipeline.
#
# Pipeline steps:
#   1. Load & validate data          (preprocessing)
#   2. Handle missing values         (preprocessing)
#   3. Assign risk labels            (preprocessing)
#   4. Augment data to 500 rows      (preprocessing)
#   5. Train Decision Tree model     (training)
#   6. Visualise decision tree       (main)
#   7. Evaluate with Stratified CV   (evaluation)
#
# Usage:
#   python main.py
#
# Requirements:
#   pip install pandas openpyxl scikit-learn matplotlib seaborn numpy
# =============================================================================

import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from src import preprocessing, training, evaluation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH      = "data/saudi_pollution_13regions.xlsx"
TARGET_ROWS    = 500
TREE_VIZ_PATH  = "outputs/decision_tree.png"
CM_PATH        = "outputs/cv_confusion_matrix.png"
FI_PATH        = "outputs/feature_importance.png"
CSV_PATH       = "outputs/evaluation_results.csv"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Saudi Arabia Air Pollution Risk Classification Pipeline")
    print("=" * 60)

    # ── Step 1–4: Preprocessing ──────────────────────────────────────────
    print("\n[STEP 1] Loading and preparing data...")
    df = preprocessing.load_and_prepare(DATA_PATH, target_rows=TARGET_ROWS)
    print(f"\nSample of prepared dataset (first 5 rows):")
    print(df[["region", "year", *preprocessing.FEATURE_COLS, "total_score", "risk_label"]].head())

    # ── Step 5: Training ─────────────────────────────────────────────────
    print("\n[STEP 2] Training Decision Tree model...")
    model, X, y = training.train_model(df)

    # ── Step 6: Decision Tree visualisation ──────────────────────────────
    print("\n[STEP 3] Generating Decision Tree visualisation...")
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        feature_names = preprocessing.FEATURE_COLS,
        class_names   = model.classes_,
        filled        = True,
        rounded       = True,
        fontsize      = 9,
        impurity      = False,
    )
    plt.title("Decision Tree – Air Pollution Risk Classification",
              fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(TREE_VIZ_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[INFO] Decision tree saved → {TREE_VIZ_PATH}")
    plt.show()

    # ── Step 7: Evaluation ───────────────────────────────────────────────
    print("\n[STEP 4] Evaluating model...")
    metrics = evaluation.evaluate(
        model,
        X,
        y,
        cm_save_path  = CM_PATH,
        fi_save_path  = FI_PATH,
        csv_save_path = CSV_PATH,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Final Accuracy  : {metrics['accuracy']  * 100:.2f}%")
    print(f"  Final F1-Score  : {metrics['f1_score']  * 100:.2f}%")
    print(f"  Outputs saved in: outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()