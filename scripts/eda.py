from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("Dataset preview:")
    print(df.head())
    print()

    print("Shape:")
    print(df.shape)
    print()

    print("Class distribution:")
    counts = df["Class"].value_counts()
    percentages = df["Class"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"Count": counts, "Percentage": percentages}))
    print()

    print("Amount summary:")
    print(df["Amount"].describe())
    print()

    print("Amount skew:")
    print(df["Amount"].skew())
    print()

    print("Log amount summary:")
    print(np.log1p(df["Amount"]).describe())


if __name__ == "__main__":
    main()
