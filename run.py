"""
EV Battery Mission Control — Application Launcher
=================================================
Validates environment prerequisites, checks dataset connectivity,
and launches the Streamlit mission control dashboard.

Usage:
    python run.py
"""

import sys
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "ev_battery_failure_dataset.csv"


def check_environment() -> None:
    """Check Python version and installed packages."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65)
    print("EV Battery Mission Control -- System Initialization")
    print("=" * 65)
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Project Root   : {PROJECT_ROOT}")

    # Check dataset availability
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        size_mb = DATA_PATH.stat().st_size / (1024 * 1024)
        print(f"Dataset Status : CONNECTED ({size_mb:.1f} MB at {DATA_PATH})")
    else:
        print("Dataset Status : DEMO MODE (data/ev_battery_failure_dataset.csv not detected)")
        print("                 Dashboard is running in calibrated demonstration mode.")
    print("=" * 65)


def run_dashboard() -> None:
    """Launch Streamlit application."""
    app_main = PROJECT_ROOT / "app" / "main.py"
    if not app_main.exists():
        print(f"Error: {app_main} not found.")
        sys.exit(1)

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_main)]
    print(f"\nLaunching Dashboard: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nMission Control stopped by user.")


if __name__ == "__main__":
    check_environment()
    run_dashboard()
