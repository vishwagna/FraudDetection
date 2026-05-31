from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "creditcard_processed.csv"

REQUIRED_COLUMNS = ["Time", "Amount", "Class"]
ENGINEERED_FEATURES = [
    "transaction_hour",
    "txns_per_hour",
    "amount_zscore",
    "amount_to_mean_ratio",
    "log_amount",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw credit card fraud dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    """Fail early if the raw dataset is missing columns used downstream."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required column(s): {missing_columns}")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create fraud-detection-friendly features from Time and Amount."""
    validate_columns(df)

    features = df.copy()
    features["transaction_hour"] = (features["Time"] // 3600).astype(int) % 24
    features["txns_per_hour"] = (
        features.groupby("transaction_hour", dropna=False)["Amount"].transform("size")
    )

    amount_mean = features["Amount"].mean()
    amount_std = features["Amount"].std()
    safe_std = amount_std if amount_std and not pd.isna(amount_std) else 1
    safe_mean = amount_mean if amount_mean and not pd.isna(amount_mean) else np.nan

    features["amount_zscore"] = (features["Amount"] - amount_mean) / safe_std
    features["amount_to_mean_ratio"] = (
        features["Amount"] / safe_mean
    ).replace([np.inf, -np.inf], 0).fillna(0)
    features["log_amount"] = np.log1p(features["Amount"])

    return features


def save_processed_data(
    df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH
) -> Path:
    """Save the processed dataset for training and later pipeline steps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def process_data(
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Run the full data processing step."""
    raw_df = load_raw_data(raw_path)
    processed_df = add_engineered_features(raw_df)
    save_processed_data(processed_df, processed_path)
    return processed_df


def main() -> None:
    processed_df = process_data()
    print(f"Processed rows: {processed_df.shape[0]:,}")
    print(f"Processed columns: {processed_df.shape[1]:,}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")
    print(f"Engineered features: {', '.join(ENGINEERED_FEATURES)}")


if __name__ == "__main__":
    main()
