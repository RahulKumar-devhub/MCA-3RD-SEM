"""
RoadSense - Model Evaluation & Visualization
Computes standard ML classification metrics and generates academic figures.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, Any, List

CLASS_NAMES = ['Normal Road', 'Bump', 'Pothole']


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates a single model and returns a metrics dictionary.
    """
    import time
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(X_test))

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC-AUC if predict_proba is supported
    roc_auc = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr'))
        except Exception:
            roc_auc = None

    return {
        'model_name': model_name,
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc,
        'latency_ms_per_window': latency_ms,
        'confusion_matrix': cm
    }


def plot_confusion_matrices(
    results_list: List[Dict[str, Any]],
    output_dir: str
):
    """
    Generates academic confusion matrix heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="white")

    for res in results_list:
        name = res['model_name']
        cm = np.array(res['confusion_matrix'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"size": 12, "weight": "bold"}
        )
        plt.title(f"{name} - Confusion Matrix\n(Accuracy: {res['accuracy']*100:.2f}%, F1: {res['f1_macro']*100:.2f}%)", fontsize=12, pad=12)
        plt.xlabel("Predicted Condition", fontsize=11, fontweight="bold")
        plt.ylabel("Actual Condition", fontsize=11, fontweight="bold")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"cm_{safe_name}.png"), dpi=300)
        plt.close()


def plot_model_comparison_bar_chart(
    comparison_df: pd.DataFrame,
    output_path: str
):
    """
    Plots grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models.
    """
    plt.figure(figsize=(12, 6))
    df_plot = comparison_df.copy()

    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    x = np.arange(len(df_plot))
    width = 0.2

    colors = ['#2563eb', '#10b981', '#f59e0b', '#8b5cf6']

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [float(v) * 100 for v in df_plot[metric]]
        plt.bar(x + i * width, vals, width, label=label, color=color, alpha=0.9, edgecolor='black', linewidth=0.5)

    plt.xlabel('Machine Learning Algorithm', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.title('RoadSense - Unit-2 Supervised Algorithm Benchmark', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x + width * 1.5, df_plot['model_name'], rotation=25, ha='right', fontsize=10)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(loc='lower right', framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(
    rf_model,
    feature_names: List[str],
    output_path: str,
    top_n: int = 15
):
    """
    Plots the top N features ranked by Random Forest MDI (Mean Decrease in Impurity).
    """
    if not hasattr(rf_model, 'feature_importances_'):
        return

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(top_n), top_scores[::-1], color='#3b82f6', edgecolor='black', linewidth=0.5)
    plt.yticks(range(top_n), [f.replace('_', ' ').upper() for f in top_features[::-1]], fontsize=10)
    plt.xlabel('Relative Feature Importance (MDI)', fontsize=11, fontweight='bold')
    plt.title(f'RoadSense - Top {top_n} Sensor Features (Random Forest)', fontsize=13, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
