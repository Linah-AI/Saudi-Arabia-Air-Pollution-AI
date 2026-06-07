# =============================================================================
# preprocessing.py
# Handles data loading, validation, augmentation, labeling, and preparation.
#
# Data Source:
#   General Authority for Statistics (GASTAT), "Household Environment
#   Statistics," DataSaudi, Ministry of Economy and Planning, 2023-2024.
#   https://datasaudi.sa/en
#
# Augmentation Method:
#   Gaussian Noise Injection — approved by project supervisor.
#   Each original record generates synthetic neighbours by adding
#   N(0, sigma) noise where sigma = 5% of the value + 0.5.
#   Reference: Goodfellow et al., Deep Learning, MIT Press, 2016.
# =============================================================================

import sys
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Feature columns used for model training
FEATURE_COLS = ["dust_pct", "smoking_pct", "waste_pct", "combustion_pct"]

# Risk label thresholds based on TotalPollutionScore distribution# Computed from the 26-row original dataset.
LOW_THRESHOLD    = 28.0   # below P25  → Low
HIGH_THRESHOLD   = 45.0   # above P75  → High

TARGET_ROWS      = 500    # desired dataset size after augmentation
NOISE_SCALE      = 0.05   # 5 % of the feature value
NOISE_FLOOR      = 0.5    # minimum sigma to avoid zero noise on small values
RANDOM_SEED      = 42


# ---------------------------------------------------------------------------
# Step 1 – Load & validate
# ---------------------------------------------------------------------------

def load_and_validate(file_path: str) -> pd.DataFrame:
    """
    Loads the Excel dataset and performs basic integrity checks.
    Exits safely if the file is missing or malformed.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing the 26-row wide-format dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with expected columns confirmed.
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Could not read file: {exc}")
        sys.exit(1)

    required = {"region", "year"} | set(FEATURE_COLS)
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        sys.exit(1)

    # Flag out-of-range values (must be 0–100 %)
    for col in FEATURE_COLS:
        bad = df[(df[col] < 0) | (df[col] > 100)]
        if not bad.empty:
            print(f"[WARNING] {len(bad)} out-of-range value(s) in '{col}' — clipping to [0, 100].")
            df[col] = df[col].clip(0, 100)

    print(f"[INFO] Loaded {len(df)} original records from '{file_path}'.")
    return df


# ---------------------------------------------------------------------------
# Step 2 – Handle missing values
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing pollution percentages using the regional median.
    Missing values in non-numeric columns are dropped.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with no missing values in feature columns.
    """
    before = df[FEATURE_COLS].isna().sum().sum()
    if before > 0:
        for col in FEATURE_COLS:
            # Use regional median first; fall back to global median
            df[col] = df.groupby("region")[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())
        print(f"[INFO] Imputed {before} missing value(s) using regional median.")
    else:
        print("[INFO] No missing values detected.")
    return df


# ---------------------------------------------------------------------------
# Step 3 – Assign risk labels
# ---------------------------------------------------------------------------

def assign_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes TotalPollutionScore and assigns Low / Medium / High labels.

    Thresholds (selected based on TotalPollutionScore distribution across the 26-row original dataset):
        Low    : TotalPollutionScore < 28.0
        Medium : 28.0 <= score < 45.0
        High   : score >= 45.0

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with added 'total_score' and 'risk_label' columns.
    """
    df = df.copy()
    df["total_score"] = df[FEATURE_COLS].sum(axis=1).round(3)

    def _label(score):
        if score >= HIGH_THRESHOLD:
            return "High"
        elif score >= LOW_THRESHOLD:
            return "Medium"
        return "Low"

    df["risk_label"] = df["total_score"].apply(_label)

    counts = df["risk_label"].value_counts().to_dict()
    print(f"[INFO] Risk label distribution (original): {counts}")
    return df


# ---------------------------------------------------------------------------
# Step 4 – Data augmentation (Gaussian Noise Injection)
# ---------------------------------------------------------------------------

def augment_data(df: pd.DataFrame, target_rows: int = TARGET_ROWS) -> pd.DataFrame:
    """
    Expands the dataset to `target_rows` using Gaussian Noise Injection.

    For each original record, synthetic copies are generated by adding
    independent Gaussian noise to each feature:
        new_value = original_value + N(0, sigma)
        sigma     = NOISE_SCALE * original_value + NOISE_FLOOR

    All generated values are clipped to [0, 100] to remain valid percentages.
    Risk labels are re-derived from the augmented values (not copied).

    Parameters
    ----------
    df         : pd.DataFrame — original labeled dataframe (26 rows)
    target_rows: int          — desired total rows after augmentation

    Returns
    -------
    pd.DataFrame
        Combined original + synthetic dataframe, shuffled.
    """
    rng           = np.random.default_rng(RANDOM_SEED)
    n_original    = len(df)
    n_synthetic   = target_rows - n_original
    per_row       = n_synthetic // n_original
    remainder     = n_synthetic  % n_original

    synthetic_records = []

    for idx, row in df.iterrows():
        # Generate `per_row` (+ 1 extra for first `remainder` rows) copies
        copies = per_row + (1 if idx < remainder else 0)
        for _ in range(copies):
            new = {"region": row["region"], "year": row["year"], "source": "synthetic"}
            for col in FEATURE_COLS:
                sigma     = NOISE_SCALE * row[col] + NOISE_FLOOR
                new[col]  = float(np.clip(row[col] + rng.normal(0, sigma), 0, 100))
                new[col]  = round(new[col], 3)
            synthetic_records.append(new)

    df_synthetic = pd.DataFrame(synthetic_records)

    # Re-derive labels from augmented values
    df_synthetic["total_score"] = df_synthetic[FEATURE_COLS].sum(axis=1).round(3)
    df_synthetic["risk_label"]  = df_synthetic["total_score"].apply(
        lambda s: "High" if s >= HIGH_THRESHOLD else ("Medium" if s >= LOW_THRESHOLD else "Low")
    )

    # Mark originals
    df_original         = df.copy()
    df_original["source"] = "original"

    combined = pd.concat([df_original, df_synthetic], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    counts = combined["risk_label"].value_counts().to_dict()
    print(f"[INFO] Dataset after augmentation: {len(combined)} rows | Labels: {counts}")
    return combined


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def load_and_prepare(file_path: str, target_rows: int = TARGET_ROWS) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
        load → validate → impute → label → augment

    Parameters
    ----------
    file_path   : str — path to the Excel file
    target_rows : int — desired size after augmentation (default 500)

    Returns
    -------
    pd.DataFrame
        Clean, labeled, augmented dataframe ready for model training.
    """
    df = load_and_validate(file_path)
    df = handle_missing(df)
    df = assign_risk_labels(df)
    df = augment_data(df, target_rows=target_rows)
    return df
