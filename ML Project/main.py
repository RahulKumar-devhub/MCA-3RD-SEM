"""
RoadSense - Main Entrypoint
Single-entrypoint command center for RoadSense.
Usage:
  python main.py           # Launches Streamlit Web Dashboard
  python main.py --train   # Executes Terminal ML Training Pipeline
  python main.py --report  # Compiles Complete Academic PDF Report
"""

import os
import sys
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="RoadSense - AI-Powered Road Surface Condition Monitoring")
    parser.add_argument("--train", action="store_true", help="Run the terminal ML training pipeline")
    parser.add_argument("--report", action="store_true", help="Generate the academic PDF report")
    parser.add_argument("--app", action="store_true", help="Launch the Streamlit web dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port for the Streamlit dashboard")

    args = parser.parse_args()

    # 1. Execute Training
    if args.train:
        print("[RoadSense] Launching terminal ML training pipeline...")
        train_script = os.path.join(BASE_DIR, "src", "train.py")
        subprocess.run([sys.executable, train_script], check=True)
        return

    # 2. Generate PDF Report
    if args.report:
        print("[RoadSense] Generating academic PDF report...")
        report_script = os.path.join(BASE_DIR, "report", "generate_report.py")
        subprocess.run([sys.executable, report_script], check=True)
        return

    # 3. Default: Launch Streamlit Web Application
    app_script = os.path.join(BASE_DIR, "app", "app.py")
    models_dir = os.path.join(BASE_DIR, "models")

    # Check if models exist; if not, train first
    if not os.path.exists(os.path.join(models_dir, "random_forest.joblib")):
        print("[RoadSense] Models not found. Initializing training pipeline first...")
        train_script = os.path.join(BASE_DIR, "src", "train.py")
        subprocess.run([sys.executable, train_script], check=True)

    print(f"\n==================================================")
    print(f"ROADSENSE DASHBOARD LAUNCHER")
    print(f"==================================================")
    print(f"Local URL: http://localhost:{args.port}")
    print(f"Starting Streamlit server...\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_script,
        "--server.port", str(args.port),
        "--server.headless", "true"
    ])


if __name__ == '__main__':
    main()
