"""
Train Model Pipeline
=====================
End-to-end: generate dataset → train XGBoost → train Isolation Forest → save all.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import generate_btms_dataset, validate_dataset
from src.train_xgboost import run_training_pipeline
from src.anomaly_detection import train_anomaly_detector, save_anomaly_model
from src.preprocessing import load_dataset


def main():
    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "models"
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    data_path = data_dir / "btms_dataset.csv"

    # Step 1: Generate dataset if not exists
    if not data_path.exists():
        print("\n[INFO] Generating Synthetic BTMS Dataset...")
        df = generate_btms_dataset(n_samples=10000, seed=42)
        df.to_csv(data_path, index=False)
        validate_dataset(df)
    else:
        print(f"\n[OK] Dataset already exists at {data_path}")

    # Step 2: Train XGBoost
    print("\n" + "=" * 60)
    print("  TRAINING XGBOOST TEMPERATURE MODEL")
    print("=" * 60)
    result = run_training_pipeline(str(data_path), str(model_dir))

    # Step 3: Train Isolation Forest
    print("\n" + "=" * 60)
    print("  TRAINING ISOLATION FOREST ANOMALY DETECTOR")
    print("=" * 60)
    df = load_dataset(str(data_path))
    iso_model = train_anomaly_detector(df)
    iso_path = save_anomaly_model(iso_model, str(model_dir))
    print(f"  [OK] Isolation Forest saved to {iso_path}")

    print("\n" + "=" * 60)
    print("  [OK] ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n  Model directory: {model_dir}")
    print(f"  XGBoost metrics: {result['metrics']}")
    print()


if __name__ == "__main__":
    main()
