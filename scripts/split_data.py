from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_splitting import (
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    split_processed_data,
    summarize_class_distribution,
)


def main() -> None:
    train_df, test_df = split_processed_data()

    print("Train/test split complete")
    print(f"Train rows: {train_df.shape[0]:,}")
    print(f"Test rows: {test_df.shape[0]:,}")
    print(f"Train path: {TRAIN_DATA_PATH}")
    print(f"Test path: {TEST_DATA_PATH}")
    print()
    print("Train class distribution:")
    print(summarize_class_distribution(train_df))
    print()
    print("Test class distribution:")
    print(summarize_class_distribution(test_df))


if __name__ == "__main__":
    main()
