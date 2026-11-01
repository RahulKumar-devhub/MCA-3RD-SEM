"""
RoadSense - Unified Inference Pipeline
Provides real-time and batch prediction capabilities for raw sensor data or windowed vectors.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_sensor_data, validate_sensor_dataframe
from src.feature_engineering import compute_window_features, FEATURE_NAMES

CLASS_LABELS = {0: 'Normal Road', 1: 'Bump', 2: 'Pothole'}


class RoadSensePredictor:
    """
    Inference manager that loads serialized models, scales features, and yields predictions.
    """
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(BASE_DIR, "models")
        self.models_dir = models_dir

        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        with open(os.path.join(models_dir, "model_metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        self.best_model_name = self.metadata.get('best_model', 'Random Forest')
        self.loaded_models = {}

    def get_model(self, model_name: Optional[str] = None):
        if not model_name:
            model_name = self.best_model_name

        if model_name not in self.loaded_models:
            safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
            model_path = os.path.join(self.models_dir, f"{safe_name}.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file '{model_path}' not found.")
            self.loaded_models[model_name] = joblib.load(model_path)

        return self.loaded_models[model_name]

    def predict_window(
        self,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        gyro_x: np.ndarray,
        gyro_y: np.ndarray,
        gyro_z: np.ndarray,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs ML prediction on a single 100-sample sensor window.
        """
        import time
        t0 = time.perf_counter()

        # 1. Feature extraction
        feats_dict = compute_window_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        feats_arr = np.array([[feats_dict[f] for f in FEATURE_NAMES]])

        # 2. Model selection & Scaling
        model = self.get_model(model_name)
        active_model_name = model_name or self.best_model_name

        if active_model_name == 'Rule-Based':
            X_input = feats_arr
        else:
            X_input = self.scaler.transform(feats_arr)

        # 3. Prediction & Probabilities
        pred_label_id = int(model.predict(X_input)[0])
        pred_class = CLASS_LABELS.get(pred_label_id, "Unknown")

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_input)[0]
            confidence = float(np.max(probs))
            prob_dict = {CLASS_LABELS[i]: float(p) for i, p in enumerate(probs)}
        else:
            confidence = 1.0
            prob_dict = {CLASS_LABELS[i]: 1.0 if i == pred_label_id else 0.0 for i in range(3)}

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            'predicted_class': pred_class,
            'predicted_label_id': pred_label_id,
            'confidence': confidence,
            'probabilities': prob_dict,
            'latency_ms': latency_ms,
            'features': feats_dict,
            'model_used': active_model_name
        }

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        model_name: Optional[str] = None,
        window_size: int = 100,
        overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Processes an entire DataFrame of sensor readings across consecutive sliding windows.
        """
        is_valid, missing, df_std = validate_sensor_dataframe(df)
        if not is_valid:
            raise ValueError(f"Uploaded data is missing required sensor columns: {missing}")

        df_clean = clean_sensor_data(df_std)
        n = len(df_clean)

        if n < window_size:
            # If smaller than window_size, pad with edge values
            pad_len = window_size - n
            df_pad = pd.concat([df_clean, df_clean.iloc[[-1] * pad_len]], ignore_index=True)
            df_clean = df_pad
            n = len(df_clean)

        step = max(1, window_size - overlap)
        ax = df_clean['accelerometer_x'].to_numpy(dtype=float)
        ay = df_clean['accelerometer_y'].to_numpy(dtype=float)
        az = df_clean['accelerometer_z'].to_numpy(dtype=float)
        gx = df_clean['gyroscope_x'].to_numpy(dtype=float)
        gy = df_clean['gyroscope_y'].to_numpy(dtype=float)
        gz = df_clean['gyroscope_z'].to_numpy(dtype=float)
        times = df_clean['seconds_elapsed'].to_numpy(dtype=float)

        window_predictions = []
        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            res = self.predict_window(
                ax[start:end], ay[start:end], az[start:end],
                gx[start:end], gy[start:end], gz[start:end],
                model_name=model_name
            )
            res['window_start_sec'] = float(times[start])
            res['window_end_sec'] = float(times[end - 1])
            window_predictions.append(res)

        # Summary counts
        class_counts = {c: 0 for c in CLASS_LABELS.values()}
        for wp in window_predictions:
            class_counts[wp['predicted_class']] += 1

        overall_condition = max(class_counts, key=class_counts.get)

        return {
            'total_windows': len(window_predictions),
            'total_records': n,
            'overall_condition': overall_condition,
            'class_distribution': class_counts,
            'window_predictions': window_predictions,
            'cleaned_data_preview': df_clean.head(10).to_dict(orient='records')
        }
