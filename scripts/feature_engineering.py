from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import ENGINEERED_FEATURES, process_data


def main() -> None:
    features = process_data()

    print("Engineered feature preview:")
    print(features[["Time", "Amount", "Class", *ENGINEERED_FEATURES]].head(10))
    print()
    print("Processed shape:")
    print(features.shape)


if __name__ == "__main__":
    main()
