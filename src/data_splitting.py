from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from src.data_processing import PROCESSED_DATA_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"

TARGET_COLUMN = "Class"

def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load processed parquet data created by the data processing step."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run data processing first."
        )
    return pd.read_parquet(path)

def stratified_train_test_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data while preserving the target class distribution."""
    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    test_parts = []
    train_parts = []

    for _, group in df.groupby(target_column, dropna=False):
        shuffled_group = group.sample(frac=1, random_state=random_state)
        test_count = max(1, round(len(shuffled_group) * test_size))

        test_parts.append(shuffled_group.iloc[:test_count])
        train_parts.append(shuffled_group.iloc[test_count:])

    # Concatenate and reshuffle train and test datasets
    train_df = (
        pd.concat(train_parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat(test_parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    return train_df, test_df

def save_split_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
) -> tuple[Path, Path]:
    """Save train and test datasets as Parquet files."""
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    return train_path, test_path

def summarize_class_distribution(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> pd.DataFrame:
    """Return count and percentage for each target class."""
    counts = df[target_column].value_counts().sort_index()
    percentages = df[target_column].value_counts(normalize=True).sort_index() * 100
    return pd.DataFrame({"count": counts, "percentage": percentages.round(4)})

def split_processed_data(
    processed_path: Path = PROCESSED_DATA_PATH,
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full train/test split step."""
    processed_df = load_processed_data(processed_path)
    train_df, test_df = stratified_train_test_split(
        processed_df,
        test_size=test_size,
        random_state=random_state,
    )
    save_split_data(train_df, test_df, train_path, test_path)
    return train_df, test_df

def main() -> None:
    train_df, test_df = split_processed_data()
    print("Train/test split complete.")
    print(f"Train rows: {train_df.shape[0]:,}")
    print(f"Test rows: {test_df.shape[0]:,}")
    print(f"Saved train data to: {TRAIN_DATA_PATH}")
    print(f"Saved test data to: {TEST_DATA_PATH}")
    print("\nTrain class distribution:")
    print(summarize_class_distribution(train_df))
    print("\nTest class distribution:")
    print(summarize_class_distribution(test_df))

if __name__ == "__main__":
    main()
