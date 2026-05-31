from __future__ import annotations

from src.data_processing import ENGINEERED_FEATURES, PROCESSED_DATA_PATH, process_data


def main() -> None:
    processed_df = process_data()

    print("Fraud Detection pipeline")
    print("========================")
    print(f"Processed rows: {processed_df.shape[0]:,}")
    print(f"Processed columns: {processed_df.shape[1]:,}")
    print(f"Saved processed data to: {PROCESSED_DATA_PATH}")
    print(f"Engineered features: {', '.join(ENGINEERED_FEATURES)}")


if __name__ == "__main__":
    main()
