from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_ROOT / "feature_store" / "feature_repo"

def main():
    # Path to feast executable in virtual environment
    feast_bin = str(PROJECT_ROOT / "mlEnv" / "bin" / "feast")
    
    # 1. Run feast apply to parse features.py and update registry.db
    print("Running 'feast apply'...")
    cmd_apply = [feast_bin, "apply"]
    subprocess.run(cmd_apply, cwd=str(REPO_DIR), check=True)
    
    # 2. Run feast materialize to load features from parquet into sqlite online store
    # Date range covers Jan 1st 2026 (our generated events start time)
    start_date = "2025-12-31T00:00:00"
    end_date = "2026-01-05T00:00:00"
    print(f"Running 'feast materialize' from {start_date} to {end_date}...")
    cmd_materialize = [feast_bin, "materialize", start_date, end_date]
    subprocess.run(cmd_materialize, cwd=str(REPO_DIR), check=True)
    print("Feast features successfully materialized to online store.")

if __name__ == "__main__":
    main()
