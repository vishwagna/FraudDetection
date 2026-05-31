from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transactions.parquet"
USER_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "user_features.parquet"

REQUIRED_COLUMNS = ["Time", "Amount", "Class"]
ENGINEERED_FEATURES = [
    "transaction_hour",
    "log_amount",
    "amount_to_mean_ratio",
    "amount_zscore_user",
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

def process_data(
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    user_features_path: Path = USER_FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full data processing step."""
    raw_df = load_raw_data(raw_path)
    validate_columns(raw_df)

    # 1. Generate deterministic user_id and transaction_id
    rng = np.random.default_rng(42)
    processed_df = raw_df.copy()
    processed_df["user_id"] = rng.integers(1000, 6000, size=len(processed_df)).astype(np.int64)
    processed_df["transaction_id"] = [f"tx_{i}" for i in range(len(processed_df))]

    # 2. Add event_timestamp by converting Time (seconds) to datetime
    start_time = pd.to_datetime("2026-01-01 00:00:00")
    processed_df["event_timestamp"] = start_time + pd.to_timedelta(processed_df["Time"], unit="s")

    # 3. Compute user-level aggregates
    user_stats = processed_df.groupby("user_id").agg(
        user_txn_count=("Amount", "size"),
        user_avg_amount=("Amount", "mean"),
        user_std_amount=("Amount", "std")
    ).reset_index()
    user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(0.0)
    user_stats["user_txn_count"] = user_stats["user_txn_count"].astype(np.int64)
    
    # Feast requires an event_timestamp for the user features (set to baseline start_time)
    user_stats["event_timestamp"] = start_time

    # Save user aggregates to Parquet
    user_features_path.parent.mkdir(parents=True, exist_ok=True)
    user_stats.to_parquet(user_features_path, index=False)

    # 4. Map user aggregates back to compute transaction-level features
    processed_df = processed_df.merge(user_stats[["user_id", "user_avg_amount", "user_std_amount"]], on="user_id", how="left")

    processed_df["transaction_hour"] = (processed_df["Time"] // 3600).astype(np.int32) % 24
    processed_df["log_amount"] = np.log1p(processed_df["Amount"]).astype(np.float32)

    global_mean_amount = processed_df["Amount"].mean()
    processed_df["amount_to_mean_ratio"] = (processed_df["Amount"] / (global_mean_amount if global_mean_amount else 1.0)).astype(np.float32)

    # Amount zscore per user
    safe_std = processed_df["user_std_amount"].copy()
    safe_std[safe_std == 0] = 1.0
    processed_df["amount_zscore_user"] = ((processed_df["Amount"] - processed_df["user_avg_amount"]) / safe_std).astype(np.float32)

    # Drop intermediate columns used to merge
    processed_df = processed_df.drop(columns=["user_avg_amount", "user_std_amount"])

    # Cast other columns for compatibility
    for col in [f"V{i}" for i in range(1, 29)] + ["Amount"]:
        processed_df[col] = processed_df[col].astype(np.float32)
    processed_df["Class"] = processed_df["Class"].astype(np.int32)

    # Save processed transactions to Parquet
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(processed_path, index=False)

    return processed_df, user_stats

def main() -> None:
    processed_df, user_stats = process_data()
    print("Data processing complete.")
    print(f"Transactions: {processed_df.shape[0]:,} rows, {processed_df.shape[1]:,} columns")
    print(f"User profiles: {user_stats.shape[0]:,} rows")
    print(f"Saved transactions to: {PROCESSED_DATA_PATH}")
    print(f"Saved user profiles to: {USER_FEATURES_PATH}")

if __name__ == "__main__":
    main()
