"""
Train Model Pipeline
=====================
End-to-end: train PyTorch Failure Classifier → train XGBoost Baseline → train Autoencoder → train VAE → train cGAN
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_script(script_path):
    print("\n" + "=" * 60)
    print(f"  RUNNING {script_path.name}")
    print("=" * 60)
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(PROJECT_ROOT))

def main():
    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    data_path = data_dir / "ev_battery_failure_dataset.csv"

    if not data_path.exists():
        print(f"\n[ERROR] Dataset not found at {data_path}. Please place it there.")
        sys.exit(1)

    print(f"\n[OK] Found authoritative dataset at {data_path}")

    # 1. Train PyTorch Failure Model
    run_script(PROJECT_ROOT / "src" / "train_failure_model.py")
    
    # 2. Train XGBoost Baseline
    from src.train_xgboost import run_training_pipeline
    print("\n" + "=" * 60)
    print("  TRAINING XGBOOST BASELINE CLASSIFIER")
    print("=" * 60)
    run_training_pipeline(str(data_path), str(model_dir))
    
    # 3. Train Autoencoder (Anomaly Detection)
    run_script(PROJECT_ROOT / "src" / "train_autoencoder.py")
    
    # 4. Train VAE
    run_script(PROJECT_ROOT / "src" / "train_vae.py")
    
    # 5. Train cGAN
    run_script(PROJECT_ROOT / "src" / "train_cgan.py")

    print("\n" + "=" * 60)
    print("  [OK] ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n  Model directory: {model_dir}\n")

if __name__ == "__main__":
    main()
