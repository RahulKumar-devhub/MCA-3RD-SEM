"""
RoadSense - Machine Learning Training Pipeline
Executes data loading, cleaning, windowed feature extraction, train/test splitting,
model training for all Unit-2 algorithms, rigorous evaluation, and artifact serialization.
"""

import os
import sys
import json
import time

# Ensure UTF-8 output encoding on Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_raw_dataset, REQUIRED_SENSOR_COLUMNS
from src.preprocessing import clean_sensor_data, validate_sensor_dataframe
from src.feature_engineering import extract_features_from_dataframe, FEATURE_NAMES
from src.rule_based import RuleBasedClassifier
from src.evaluate import (
    evaluate_model, plot_confusion_matrices,
    plot_model_comparison_bar_chart, plot_feature_importance
)


def train_pipeline():
    print("=" * 50)
    print("ROADSENSE ML TRAINING PIPELINE")
    print("=" * 50)

    # 1. Loading dataset
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    t0 = time.time()
    raw_df = load_raw_dataset(raw_dir, max_records_per_class=30000)
    print(f"[1] Loading dataset .................... ✓ ({len(raw_df):,} records)")

    # 2. Validating sensor columns
    is_valid, missing, std_df = validate_sensor_dataframe(raw_df)
    if not is_valid:
        print(f"[2] Validating sensor columns .......... ✗ Missing: {missing}")
        sys.exit(1)
    print("[2] Validating sensor columns .......... ✓ (6-Axis IMU + GPS)")

    # 3. Cleaning data
    cleaned_df = clean_sensor_data(std_df)
    sensor_nulls = cleaned_df[REQUIRED_SENSOR_COLUMNS].isna().sum().sum()
    print(f"[3] Cleaning data ...................... ✓ ({sensor_nulls} sensor nulls)")

    # 4. Feature extraction
    t_feat = time.time()
    X_df, y, y_names = extract_features_from_dataframe(
        cleaned_df, window_size=100, overlap=50, has_labels=True
    )
    print(f"[4] Feature extraction ................ ✓ ({len(X_df):,} windows, {len(FEATURE_NAMES)} features)")

    # Save processed features for rapid testing
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    X_df_saved = X_df.copy()
    X_df_saved['target_label'] = y
    X_df_saved.to_csv(os.path.join(processed_dir, "windowed_features.csv"), index=False)

    # 5. Train / Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.to_numpy(), y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    print(f"[5] Train/Test split .................. ✓ (Train: {len(X_train):,}, Test: {len(X_test):,})")

    # Save Scaler
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    # Define Models
    models_dict = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Naïve Bayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', C=5.0, probability=True, random_state=42),
        'ANN / MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, activation='relu', early_stopping=True, random_state=42),
        'Rule-Based': RuleBasedClassifier()
    }

    print("\nTraining Models")
    print("-" * 50)

    eval_results = []
    saved_paths = {}

    for name, model in models_dict.items():
        # Rule-based fits on raw features; ML models fit on scaled features
        if name == 'Rule-Based':
            model.fit(X_train_raw, y_train)
            res = evaluate_model(model, X_test_raw, y_test, model_name=name)
        else:
            model.fit(X_train, y_train)
            res = evaluate_model(model, X_test, y_test, model_name=name)

        eval_results.append(res)

        # Save model
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        model_path = os.path.join(models_dir, f"{safe_name}.joblib")
        joblib.dump(model, model_path)
        saved_paths[name] = model_path

        # Print checkmark
        dots = "." * max(2, 35 - len(name))
        print(f"{name} {dots} ✓")

    # Format Results Table
    results_df = pd.DataFrame(eval_results)
    results_df = results_df.sort_values(by='f1_macro', ascending=False).reset_index(drop=True)

    print("\nMODEL EVALUATION")
    print("-" * 50)
    print(f"{'Model':<24} {'Accuracy':>10} {'F1-Score':>10} {'Latency(ms)':>12}")
    print("-" * 58)
    for _, row in results_df.iterrows():
        acc_str = f"{row['accuracy']*100:.2f}%"
        f1_str = f"{row['f1_macro']*100:.2f}"
        lat_str = f"{row['latency_ms_per_window']:.3f}"
        print(f"{row['model_name']:<24} {acc_str:>10} {f1_str:>10} {lat_str:>12}")

    best_model_name = results_df.iloc[0]['model_name']
    best_acc = results_df.iloc[0]['accuracy'] * 100
    best_f1 = results_df.iloc[0]['f1_macro'] * 100
    print("-" * 58)
    print(f"Best Model: {best_model_name} (Acc: {best_acc:.2f}%, F1: {best_f1:.2f}%)\n")

    # Save outputs
    outputs_results_dir = os.path.join(BASE_DIR, "outputs", "results")
    os.makedirs(outputs_results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(outputs_results_dir, "model_comparison.csv"), index=False)
    with open(os.path.join(outputs_results_dir, "model_comparison.json"), 'w') as f:
        json.dump(eval_results, f, indent=4)

    # Save metadata
    metadata = {
        'trained_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_raw_records': len(raw_df),
        'total_windows': len(X_df),
        'classes': ['Normal Road', 'Bump', 'Pothole'],
        'feature_names': FEATURE_NAMES,
        'best_model': best_model_name,
        'best_accuracy': best_acc,
        'best_f1': best_f1,
        'models': {r['model_name']: {'accuracy': r['accuracy'], 'f1_macro': r['f1_macro']} for r in eval_results}
    }
    with open(os.path.join(models_dir, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    # Generate Figures
    figures_dir = os.path.join(BASE_DIR, "outputs", "figures")
    plot_confusion_matrices(eval_results, figures_dir)
    plot_model_comparison_bar_chart(results_df, os.path.join(figures_dir, "model_comparison.png"))

    rf_model = joblib.load(os.path.join(models_dir, "random_forest.joblib"))
    plot_feature_importance(rf_model, FEATURE_NAMES, os.path.join(figures_dir, "feature_importance.png"))

    # Also build authentic demonstration simulation tracks from real segments
    generate_simulation_tracks(cleaned_df, os.path.join(processed_dir, "simulation_tracks.json"))

    print("Models saved successfully ✓")
    print("=" * 50)


def generate_simulation_tracks(cleaned_df: pd.DataFrame, output_path: str):
    """
    Extracts authentic consecutive sensor segments from RoadSens-4M for the Live Simulation mode:
    Normal -> Bump -> Normal -> Pothole -> Normal.
    """
    normal_seg = cleaned_df[cleaned_df['label_name'] == 'Normal Road'].iloc[:150]
    bump_seg = cleaned_df[cleaned_df['label_name'] == 'Bump'].iloc[:100]
    normal_seg2 = cleaned_df[cleaned_df['label_name'] == 'Normal Road'].iloc[200:350]
    pothole_seg = cleaned_df[cleaned_df['label_name'] == 'Pothole'].iloc[:100]
    normal_seg3 = cleaned_df[cleaned_df['label_name'] == 'Normal Road'].iloc[400:550]

    # Combine into a coherent chronological simulation stream
    sequence = [
        ('Normal Road', normal_seg),
        ('Bump', bump_seg),
        ('Normal Road', normal_seg2),
        ('Pothole', pothole_seg),
        ('Normal Road', normal_seg3)
    ]

    track_points = []
    t_clock = 0.0

    for event_name, seg in sequence:
        for _, row in seg.iterrows():
            ax = float(row['accelerometer_x'])
            ay = float(row['accelerometer_y'])
            az = float(row['accelerometer_z'])
            gx = float(row['gyroscope_x'])
            gy = float(row['gyroscope_y'])
            gz = float(row['gyroscope_z'])
            amag = float(np.sqrt(ax**2 + ay**2 + az**2))
            gmag = float(np.sqrt(gx**2 + gy**2 + gz**2))
            lat = float(row.get('location_latitude', 49.2606))
            lon = float(row.get('location_longitude', -123.2460))

            track_points.append({
                'time': round(t_clock, 2),
                'ground_truth': event_name,
                'acc_x': ax, 'acc_y': ay, 'acc_z': az, 'acc_mag': amag,
                'gyro_x': gx, 'gyro_y': gy, 'gyro_z': gz, 'gyro_mag': gmag,
                'latitude': lat, 'longitude': lon
            })
            t_clock += 0.01

    with open(output_path, 'w') as f:
        json.dump(track_points, f)


if __name__ == '__main__':
    train_pipeline()
