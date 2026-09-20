"""
Generate Synthetic BTMS Dataset
================================
Run this script to generate the synthetic dataset at data/btms_dataset.csv.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import generate_btms_dataset, validate_dataset


def main():
    print("\n[INFO] Generating Synthetic / Simulation-Inspired BTMS Dataset...\n")

    df = generate_btms_dataset(n_samples=10000, seed=42)

    # Save
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "btms_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"  [OK] Dataset saved to {output_path}\n")

    # Validate
    validate_dataset(df)


if __name__ == "__main__":
    main()
