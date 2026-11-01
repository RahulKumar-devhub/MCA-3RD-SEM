"""
RoadSense - Sample CSV and Academic Figures Generator
Creates sample upload files for evaluator testing and academic figures for the report.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_raw_dataset, REQUIRED_SENSOR_COLUMNS
from src.preprocessing import clean_sensor_data


def generate_samples_and_figures():
    print("Generating sample CSV files and academic figures...")
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    df = load_raw_dataset(raw_dir, max_records_per_class=10000)
    df = clean_sensor_data(df)

    sample_dir = os.path.join(BASE_DIR, "data", "sample_uploads")
    os.makedirs(sample_dir, exist_ok=True)
    figures_dir = os.path.join(BASE_DIR, "outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Create Sample CSV files (with standard columns and GPS)
    save_cols = REQUIRED_SENSOR_COLUMNS + ['location_latitude', 'location_longitude', 'label_name']

    normal_sample = df[df['label_name'] == 'Normal Road'].iloc[:300][save_cols].copy()
    bump_sample = df[df['label_name'] == 'Bump'].iloc[:300][save_cols].copy()
    pothole_sample = df[df['label_name'] == 'Pothole'].iloc[:300][save_cols].copy()

    # Mixed trip: Normal -> Bump -> Normal -> Pothole -> Normal
    mixed_sample = pd.concat([
        df[df['label_name'] == 'Normal Road'].iloc[:150],
        df[df['label_name'] == 'Bump'].iloc[:150],
        df[df['label_name'] == 'Normal Road'].iloc[150:300],
        df[df['label_name'] == 'Pothole'].iloc[:150],
        df[df['label_name'] == 'Normal Road'].iloc[300:450]
    ], ignore_index=True)[save_cols]

    # Reset seconds_elapsed so it starts from 0.00
    for s_df in [normal_sample, bump_sample, pothole_sample, mixed_sample]:
        s_df['seconds_elapsed'] = np.round(np.arange(len(s_df)) * 0.01, 3)

    normal_sample.to_csv(os.path.join(sample_dir, "sample_normal_road.csv"), index=False)
    bump_sample.to_csv(os.path.join(sample_dir, "sample_bump.csv"), index=False)
    pothole_sample.to_csv(os.path.join(sample_dir, "sample_pothole.csv"), index=False)
    mixed_sample.to_csv(os.path.join(sample_dir, "sample_mixed_trip.csv"), index=False)
    print(f"Sample CSV files written to {sample_dir}")

    # 2. Figure: Sensor Signatures Comparison (Normal vs Bump vs Pothole)
    plt.figure(figsize=(12, 7))
    t_axis = np.arange(100) * 0.01  # 1 second window

    # Normal window
    norm_w = normal_sample.iloc[50:150]
    plt.subplot(3, 1, 1)
    plt.plot(t_axis, norm_w['accelerometer_z'], label='Acc Z (Vertical)', color='#10b981', lw=1.8)
    plt.plot(t_axis, norm_w['accelerometer_x'], label='Acc X (Lateral)', color='#64748b', lw=1, alpha=0.7)
    plt.title('Condition 1: Normal Road (Smooth Transit - Low Baseline Vibrations)', fontsize=11, fontweight='bold')
    plt.ylabel('Acc (m/s²)', fontsize=10)
    plt.ylim(-6, 6)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=8)

    # Bump window
    bump_w = bump_sample.iloc[50:150]
    plt.subplot(3, 1, 2)
    plt.plot(t_axis, bump_w['accelerometer_z'], label='Acc Z (Vertical)', color='#f59e0b', lw=1.8)
    plt.plot(t_axis, bump_w['accelerometer_x'], label='Acc X (Lateral)', color='#64748b', lw=1, alpha=0.7)
    plt.title('Condition 2: Speed Bump (Suspension Compression - High Positive Peak)', fontsize=11, fontweight='bold')
    plt.ylabel('Acc (m/s²)', fontsize=10)
    plt.ylim(-6, 6)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=8)

    # Pothole window
    poth_w = pothole_sample.iloc[50:150]
    plt.subplot(3, 1, 3)
    plt.plot(t_axis, poth_w['accelerometer_z'], label='Acc Z (Vertical)', color='#ef4444', lw=1.8)
    plt.plot(t_axis, poth_w['accelerometer_x'], label='Acc X (Lateral)', color='#64748b', lw=1, alpha=0.7)
    plt.title('Condition 3: Pothole (Suspension Drop Followed by Sudden Impact Shock)', fontsize=11, fontweight='bold')
    plt.xlabel('Time Elapsed (seconds)', fontsize=11, fontweight='bold')
    plt.ylabel('Acc (m/s²)', fontsize=10)
    plt.ylim(-6, 6)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sensor_signatures.png"), dpi=300)
    plt.close()
    print("Generated sensor_signatures.png")

    # 3. Figure: Class Distribution
    plt.figure(figsize=(8, 5))
    class_counts = df['label_name'].value_counts()
    colors = ['#10b981', '#f59e0b', '#ef4444']
    bars = plt.bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', linewidth=0.8, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 150, f"{int(yval):,}", ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.title('RoadSense - Dataset Class Distribution (RoadSens-4M)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Road Surface Condition Class', fontsize=11, fontweight='bold')
    plt.ylabel('Sensor Record Count', fontsize=11, fontweight='bold')
    plt.ylim(0, max(class_counts.values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "class_distribution.png"), dpi=300)
    plt.close()
    print("Generated class_distribution.png")

    # 4. Figure: ANN Training Loss & Accuracy Convergence
    ann_model = joblib.load(os.path.join(BASE_DIR, "models", "ann___mlp.joblib"))
    loss_curve = ann_model.loss_curve_
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, color='#6366f1', lw=2.2, label='Training Cross-Entropy Loss')
    plt.title('Artificial Neural Network (ANN / MLP) - Convergence Profile', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Training Epochs', fontsize=10, fontweight='bold')
    plt.ylabel('Cross-Entropy Loss', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ann_loss_curve.png"), dpi=300)
    plt.close()
    print("Generated ann_loss_curve.png")

    # 5. Figure: SVM 2D PCA Hyperplane Projection
    from sklearn.decomposition import PCA
    X_feat_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "windowed_features.csv"))
    y_feat = X_feat_df.pop('target_label')
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))
    X_scaled = scaler.transform(X_feat_df.to_numpy())

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 6))
    colors_map = {0: '#10b981', 1: '#f59e0b', 2: '#ef4444'}
    names_map = {0: 'Normal Road', 1: 'Bump', 2: 'Pothole'}

    for cls_idx in [0, 1, 2]:
        mask = (y_feat == cls_idx)
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors_map[cls_idx], label=names_map[cls_idx],
            alpha=0.6, edgecolors='none', s=25
        )

    plt.title(f'SVM Feature Space - 2D Principal Component Projection\n(Explained Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%)',
              fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('First Principal Component (Vibration Energy & Magnitude)', fontsize=10, fontweight='bold')
    plt.ylabel('Second Principal Component (Spectral & Vertical Excursion)', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best', framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "svm_pca_projection.png"), dpi=300)
    plt.close()
    print("Generated svm_pca_projection.png")
    print("All figures and samples generated successfully!")


if __name__ == '__main__':
    generate_samples_and_figures()
