"""
RoadSense - Feature Engineering Pipeline
Extracts time-domain statistical, kinematic, and frequency-domain spectral features
from windowed smartphone accelerometer and gyroscope signals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict, Any


FEATURE_NAMES = [
    # Accel X
    'acc_x_mean', 'acc_x_std', 'acc_x_rms', 'acc_x_max', 'acc_x_min', 'acc_x_p2p', 'acc_x_energy',
    # Accel Y
    'acc_y_mean', 'acc_y_std', 'acc_y_rms', 'acc_y_max', 'acc_y_min', 'acc_y_p2p', 'acc_y_energy',
    # Accel Z (Vertical Dynamics)
    'acc_z_mean', 'acc_z_std', 'acc_z_rms', 'acc_z_max', 'acc_z_min', 'acc_z_p2p', 'acc_z_energy', 'acc_z_kurtosis', 'acc_z_skew',
    # Accel Magnitude
    'acc_mag_mean', 'acc_mag_std', 'acc_mag_rms', 'acc_mag_max', 'acc_mag_min', 'acc_mag_p2p', 'acc_mag_energy',
    # Gyro X
    'gyro_x_rms', 'gyro_x_std', 'gyro_x_p2p',
    # Gyro Y
    'gyro_y_rms', 'gyro_y_std', 'gyro_y_p2p',
    # Gyro Z
    'gyro_z_rms', 'gyro_z_std', 'gyro_z_p2p',
    # Gyro Magnitude
    'gyro_mag_mean', 'gyro_mag_rms', 'gyro_mag_max', 'gyro_mag_energy',
    # Spectral Dynamics
    'acc_z_spectral_energy', 'acc_z_dominant_freq', 'acc_mag_spectral_energy'
]


def compute_window_features(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    gyro_x: np.ndarray,
    gyro_y: np.ndarray,
    gyro_z: np.ndarray,
    sampling_rate: float = 100.0
) -> Dict[str, float]:
    """
    Computes statistical and spectral features for a single time window.
    """
    # Magnitudes
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

    # Basic stats helper
    def channel_stats(arr: np.ndarray):
        mean = float(np.mean(arr))
        std = float(np.std(arr) + 1e-8)
        rms = float(np.sqrt(np.mean(arr**2)))
        mx = float(np.max(arr))
        mn = float(np.min(arr))
        p2p = mx - mn
        energy = float(np.mean(arr**2))
        return mean, std, rms, mx, mn, p2p, energy

    # Accel X
    ax_m, ax_s, ax_rms, ax_mx, ax_mn, ax_p2p, ax_e = channel_stats(acc_x)
    # Accel Y
    ay_m, ay_s, ay_rms, ay_mx, ay_mn, ay_p2p, ay_e = channel_stats(acc_y)
    # Accel Z (Critical vertical axis for bumps/potholes)
    az_m, az_s, az_rms, az_mx, az_mn, az_p2p, az_e = channel_stats(acc_z)
    az_kurt = float(stats.kurtosis(acc_z))
    az_skew = float(stats.skew(acc_z))
    # Accel Mag
    am_m, am_s, am_rms, am_mx, am_mn, am_p2p, am_e = channel_stats(acc_mag)

    # Gyro channels
    gx_m, gx_s, gx_rms, gx_mx, gx_mn, gx_p2p, gx_e = channel_stats(gyro_x)
    gy_m, gy_s, gy_rms, gy_mx, gy_mn, gy_p2p, gy_e = channel_stats(gyro_y)
    gz_m, gz_s, gz_rms, gz_mx, gz_mn, gz_p2p, gz_e = channel_stats(gyro_z)
    gm_m, gm_s, gm_rms, gm_mx, gm_mn, gm_p2p, gm_e = channel_stats(gyro_mag)

    # Frequency-domain spectral features on Z and Magnitude
    n_samples = len(acc_z)
    fft_z = np.abs(np.fft.rfft(acc_z - az_m))
    fft_freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
    acc_z_spec_energy = float(np.sum(fft_z**2) / (n_samples + 1e-8))
    dom_idx = np.argmax(fft_z[1:]) + 1 if len(fft_z) > 1 else 0
    dom_freq = float(fft_freqs[dom_idx]) if len(fft_freqs) > dom_idx else 0.0

    fft_mag = np.abs(np.fft.rfft(acc_mag - am_m))
    acc_mag_spec_energy = float(np.sum(fft_mag**2) / (n_samples + 1e-8))

    features = {
        'acc_x_mean': ax_m, 'acc_x_std': ax_s, 'acc_x_rms': ax_rms, 'acc_x_max': ax_mx, 'acc_x_min': ax_mn, 'acc_x_p2p': ax_p2p, 'acc_x_energy': ax_e,
        'acc_y_mean': ay_m, 'acc_y_std': ay_s, 'acc_y_rms': ay_rms, 'acc_y_max': ay_mx, 'acc_y_min': ay_mn, 'acc_y_p2p': ay_p2p, 'acc_y_energy': ay_e,
        'acc_z_mean': az_m, 'acc_z_std': az_s, 'acc_z_rms': az_rms, 'acc_z_max': az_mx, 'acc_z_min': az_mn, 'acc_z_p2p': az_p2p, 'acc_z_energy': az_e,
        'acc_z_kurtosis': az_kurt, 'acc_z_skew': az_skew,
        'acc_mag_mean': am_m, 'acc_mag_std': am_s, 'acc_mag_rms': am_rms, 'acc_mag_max': am_mx, 'acc_mag_min': am_mn, 'acc_mag_p2p': am_p2p, 'acc_mag_energy': am_e,
        'gyro_x_rms': gx_rms, 'gyro_x_std': gx_s, 'gyro_x_p2p': gx_p2p,
        'gyro_y_rms': gy_rms, 'gyro_y_std': gy_s, 'gyro_y_p2p': gy_p2p,
        'gyro_z_rms': gz_rms, 'gyro_z_std': gz_s, 'gyro_z_p2p': gz_p2p,
        'gyro_mag_mean': gm_m, 'gyro_mag_rms': gm_rms, 'gyro_mag_max': gm_mx, 'gyro_mag_energy': gm_e,
        'acc_z_spectral_energy': acc_z_spec_energy, 'acc_z_dominant_freq': dom_freq, 'acc_mag_spectral_energy': acc_mag_spec_energy
    }

    return features


def extract_features_from_dataframe(
    df: pd.DataFrame,
    window_size: int = 100,
    overlap: int = 50,
    has_labels: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Splits DataFrame by session / source file and extracts sliding window features.
    """
    all_feature_rows = []
    labels = []
    label_names = []

    # Process per continuous segment or file
    group_col = 'source_file' if 'source_file' in df.columns else None
    groups = df.groupby(group_col) if group_col else [(None, df)]

    for _, group in groups:
        n = len(group)
        if n < window_size:
            continue

        step = max(1, window_size - overlap)
        ax = group['accelerometer_x'].to_numpy(dtype=float)
        ay = group['accelerometer_y'].to_numpy(dtype=float)
        az = group['accelerometer_z'].to_numpy(dtype=float)
        gx = group['gyroscope_x'].to_numpy(dtype=float)
        gy = group['gyroscope_y'].to_numpy(dtype=float)
        gz = group['gyroscope_z'].to_numpy(dtype=float)

        y_vals = group['label'].to_numpy() if has_labels and 'label' in group.columns else None
        y_names = group['label_name'].to_numpy() if has_labels and 'label_name' in group.columns else None

        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            w_feats = compute_window_features(
                ax[start:end], ay[start:end], az[start:end],
                gx[start:end], gy[start:end], gz[start:end]
            )
            all_feature_rows.append(w_feats)

            if has_labels and y_vals is not None:
                # Majority vote within the window using numeric labels
                w_labels = y_vals[start:end]
                vals, counts = np.unique(w_labels, return_counts=True)
                maj_label = int(vals[np.argmax(counts)])
                labels.append(maj_label)
                if y_names is not None:
                    w_names_arr = y_names[start:end]
                    n_vals, n_counts = np.unique(w_names_arr, return_counts=True)
                    label_names.append(str(n_vals[np.argmax(n_counts)]))

    features_df = pd.DataFrame(all_feature_rows, columns=FEATURE_NAMES)
    return features_df, np.array(labels), np.array(label_names)
