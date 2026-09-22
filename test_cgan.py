import sys
from pathlib import Path
import torch
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from src.models.cgan import Generator
    gen_path = PROJECT_ROOT / "models" / "cgan" / "generator.pt"
    scaler_path = PROJECT_ROOT / "models" / "cgan" / "scaler.pkl"
    
    print(f"gen_path exists: {gen_path.exists()} ({gen_path})")
    print(f"scaler_path exists: {scaler_path.exists()} ({scaler_path})")

    scaler = joblib.load(scaler_path)
    
    # Generator init: noise_dim=16, num_classes=2, feature_dim=14
    generator = Generator(16, 2, 14)
    generator.load_state_dict(torch.load(gen_path, weights_only=True))
    generator.eval()
    print("Success")
except Exception as e:
    print(f"Error loading cGAN details: {e}")
