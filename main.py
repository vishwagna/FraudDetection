from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import process_data
from src.data_splitting import split_processed_data
from feature_store.materialize import main as materialize_features
from training.train import main as train_model

def main() -> None:
    print("==================================================")
    print("Starting Fraud Detection End-to-End ML Pipeline")
    print("==================================================\n")

    # Step 1: Data Processing
    print("--- Step 1: Data Processing & Feature Engineering ---")
    processed_df, user_stats = process_data()
    print(f"Transactions shape: {processed_df.shape}")
    print(f"User profiles count: {user_stats.shape[0]}")
    print("Data processing finished.\n")

    # Step 2: Split data
    print("--- Step 2: Stratified Train/Test Split ---")
    train_df, test_df = split_processed_data()
    print(f"Train split size: {train_df.shape[0]:,}")
    print(f"Test split size: {test_df.shape[0]:,}")
    print("Split finished.\n")

    # Step 3: Feast materialization
    print("--- Step 3: Feast Feature Store Registry & Materialization ---")
    materialize_features()
    print("Feature store update finished.\n")

    # Step 4: Model Training
    print("--- Step 4: XGBoost HPO & Training with SMOTE + MLflow ---")
    train_model()
    print("Training and evaluation finished.\n")

    print("==================================================")
    print("End-to-End ML Pipeline Executed Successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
