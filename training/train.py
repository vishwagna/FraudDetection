from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.xgboost

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_splitting import TRAIN_DATA_PATH, TEST_DATA_PATH
from training.hpo import run_hpo
from training.evaluate import evaluate_model
from feast import FeatureStore

def main() -> None:
    # 1. Initialize Feast Feature Store
    print("Connecting to Feast Feature Store...")
    repo_path = PROJECT_ROOT / "feature_store" / "feature_repo"
    store = FeatureStore(repo_path=str(repo_path))

    # 2. Load train/test entities
    print("Loading training and testing datasets...")
    train_entities = pd.read_parquet(TRAIN_DATA_PATH)
    test_entities = pd.read_parquet(TEST_DATA_PATH)

    # 3. Retrieve historical features from Feast
    feature_names = [
        "user_features:user_txn_count",
        "user_features:user_avg_amount",
        "user_features:user_std_amount",
        "transaction_features:transaction_hour",
        "transaction_features:log_amount",
        "transaction_features:amount_to_mean_ratio",
        "transaction_features:amount_zscore_user",
        "transaction_features:Amount",
    ] + [f"transaction_features:V{i}" for i in range(1, 29)]

    print("Retrieving historical features from Feast offline store (train)...")
    train_features = store.get_historical_features(
        entity_df=train_entities[["user_id", "event_timestamp", "Class"]],
        features=feature_names,
    ).to_df()

    print("Retrieving historical features from Feast offline store (test)...")
    test_features = store.get_historical_features(
        entity_df=test_entities[["user_id", "event_timestamp", "Class"]],
        features=feature_names,
    ).to_df()

    # 4. Prepare X and y
    # Drop entity and timestamp columns
    y_train = train_features["Class"]
    X_train = train_features.drop(columns=["user_id", "event_timestamp", "Class"])

    y_test = test_features["Class"]
    X_test = test_features.drop(columns=["user_id", "event_timestamp", "Class"])

    # Ensure consistent column ordering
    feature_cols = sorted(X_train.columns)
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    # 5. Set up MLflow
    mlflow.set_experiment("Fraud_Detection_XGBoost")

    # Start parent run
    with mlflow.start_run(run_name="xgboost_pipeline") as parent_run:
        print(f"Parent Run ID: {parent_run.info.run_id}")
        
        # 6. Run HPO
        # Let's run 5 trials for quick execution, can be extended
        best_params = run_hpo(X_train, y_train, n_trials=5, random_state=42)
        
        # Extract SMOTE ratio and separate from XGBoost params
        best_smote_ratio = best_params.pop("smote_ratio")
        print(f"\nTraining final model with best params and SMOTE ratio: {best_smote_ratio:.4f}")
        
        # Log final params
        mlflow.log_params(best_params)
        mlflow.log_param("best_smote_ratio", best_smote_ratio)

        # 7. Apply SMOTE to training set with best ratio
        print("Applying SMOTE to training features...")
        smote = SMOTE(sampling_strategy=best_smote_ratio, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # 8. Train final XGBoost model
        print("Training final XGBoost classifier...")
        model = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_train_res, y_train_res)

        # 9. Log final model
        print("Logging model to MLflow...")
        mlflow.xgboost.log_model(model, artifact_path="model")

        # 10. Evaluate on test set
        print("Evaluating on test set...")
        model_dir = PROJECT_ROOT / "models" / "xgb_v1"
        evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=0.5,
            save_dir=model_dir,
        )
        
        # Save model locally for serving / inference stage later
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_dir / "model.json"))
        print(f"Final model saved locally to: {model_dir}/model.json")

if __name__ == "__main__":
    main()
