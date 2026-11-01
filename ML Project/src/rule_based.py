"""
RoadSense - Rule-Based Baseline Classifier
Interpretable baseline classifier utilizing physical kinematic thresholds
(Vertical peak acceleration, peak-to-peak excursion, and RMS vibration).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class RuleBasedClassifier(BaseEstimator, ClassifierMixin):
    """
    A rule-based classifier for road surface condition assessment.
    Classes:
      0: Normal Road
      1: Bump
      2: Pothole
    """
    def __init__(
        self,
        bump_p2p_threshold: float = 3.5,
        pothole_kurt_threshold: float = 2.5,
        pothole_p2p_threshold: float = 4.5,
        normal_rms_threshold: float = 0.8
    ):
        self.bump_p2p_threshold = bump_p2p_threshold
        self.pothole_kurt_threshold = pothole_kurt_threshold
        self.pothole_p2p_threshold = pothole_p2p_threshold
        self.normal_rms_threshold = normal_rms_threshold
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y=None):
        """
        Calibrates thresholds from training data if provided.
        """
        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy()
            feature_names = list(X.columns)
        else:
            X_mat = np.array(X)
            feature_names = None

        # If feature names match, calibrate data-driven thresholds
        if feature_names and 'acc_z_p2p' in feature_names:
            p2p_idx = feature_names.index('acc_z_p2p')
            rms_idx = feature_names.index('acc_z_rms')
            kurt_idx = feature_names.index('acc_z_kurtosis') if 'acc_z_kurtosis' in feature_names else 0

            if y is not None:
                y_arr = np.array(y)
                normal_mask = (y_arr == 0)
                bump_mask = (y_arr == 1)
                pothole_mask = (y_arr == 2)

                if np.any(normal_mask):
                    self.normal_rms_threshold = float(np.percentile(X_mat[normal_mask, rms_idx], 85))
                if np.any(bump_mask):
                    self.bump_p2p_threshold = float(np.percentile(X_mat[bump_mask, p2p_idx], 35))
                if np.any(pothole_mask):
                    self.pothole_p2p_threshold = float(np.percentile(X_mat[pothole_mask, p2p_idx], 45))
                    self.pothole_kurt_threshold = float(np.percentile(X_mat[pothole_mask, kurt_idx], 40))

        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy()
            cols = list(X.columns)
            p2p_idx = cols.index('acc_z_p2p') if 'acc_z_p2p' in cols else 19
            rms_idx = cols.index('acc_z_rms') if 'acc_z_rms' in cols else 16
            kurt_idx = cols.index('acc_z_kurtosis') if 'acc_z_kurtosis' in cols else 21
        else:
            X_mat = np.array(X)
            # Default column indices matching FEATURE_NAMES
            p2p_idx = 19
            rms_idx = 16
            kurt_idx = 21

        n_samples = X_mat.shape[0]
        probs = np.zeros((n_samples, 3))

        for i in range(n_samples):
            p2p = X_mat[i, p2p_idx]
            rms = X_mat[i, rms_idx]
            kurt = X_mat[i, kurt_idx]

            # Rule hierarchy
            if p2p >= self.pothole_p2p_threshold and kurt >= self.pothole_kurt_threshold:
                # High shock and peakiness indicates pothole
                probs[i] = [0.05, 0.20, 0.75]
            elif p2p >= self.bump_p2p_threshold:
                # Moderate/high peak-to-peak without extreme kurtosis indicates bump
                probs[i] = [0.10, 0.75, 0.15]
            elif rms <= self.normal_rms_threshold:
                # Low vibration energy indicates normal road
                probs[i] = [0.85, 0.10, 0.05]
            else:
                # Intermediate ambiguous
                probs[i] = [0.40, 0.35, 0.25]

        return probs
