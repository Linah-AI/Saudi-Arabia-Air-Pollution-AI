# =============================================================================
# training.py
# Trains a Decision Tree classifier on the prepared dataset.
#
# Algorithm choice: Decision Tree (scikit-learn DecisionTreeClassifier)
# Justification:
#   - Fully interpretable: every prediction has a human-readable rule path,
#     which is mandatory for a government Decision Support System (DSS).
#   - Handles small, structured numerical datasets without feature scaling.
#   - Open-source, no specialised hardware required.
#
# Reference:
#   Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras,
#   and TensorFlow (3rd ed.). O'Reilly Media.
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from src.preprocessing import FEATURE_COLS


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# max_depth limits tree complexity and reduces overfitting on small datasets.
# Value of 4 was chosen after Iteration 1 experiments (see evaluation.py).
MAX_DEPTH         = 4
MIN_SAMPLES_SPLIT = 5   # minimum samples required to split an internal node
MIN_SAMPLES_LEAF  = 2   # minimum samples required at a leaf node
CRITERION         = "gini"
RANDOM_SEED       = 42


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame):
    """
    Extracts features and target from the prepared dataframe,
    then trains a Decision Tree classifier on the full dataset.

    The model is trained on ALL rows (original + synthetic) so that
    Leave-One-Out Cross-Validation in evaluation.py can use the complete
    dataset for a fair assessment.

    Parameters
    ----------
    df : pd.DataFrame
        Output of preprocessing.load_and_prepare — must contain
        FEATURE_COLS and 'risk_label'.

    Returns
    -------
    model : DecisionTreeClassifier — fitted model
    X     : pd.DataFrame           — feature matrix
    y     : pd.Series              — target labels
    """
    X = df[FEATURE_COLS].copy()
    y = df["risk_label"].copy()

    model = DecisionTreeClassifier(
        criterion         = CRITERION,
        max_depth         = MAX_DEPTH,
        min_samples_split = MIN_SAMPLES_SPLIT,
        min_samples_leaf  = MIN_SAMPLES_LEAF,
        random_state      = RANDOM_SEED,
    )

    model.fit(X, y)

    print(f"[INFO] Decision Tree trained | depth={model.get_depth()} | leaves={model.get_n_leaves()}")
    print(f"[INFO] Classes: {list(model.classes_)}")
    print(f"[INFO] Feature importances:")
    for col, imp in zip(FEATURE_COLS, model.feature_importances_):
        print(f"         {col:<20} {imp:.4f}")

    return model, X, y