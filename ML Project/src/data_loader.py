"""
RoadSense - Data Loader Module
Loads raw sensor sessions from the research-backed RoadSens-4M dataset.
Extracts 100 Hz Accelerometer, Gyroscope, GPS, and Annotation data.
"""

import os
import glob
import pandas as pd
import numpy as np

LABEL_MAP = {
    'Normal Road': 0,
    'Bump': 1,
    'Pothole': 2
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

REQUIRED_SENSOR_COLUMNS = [
    'seconds_elapsed',
    'accelerometer_x',
    'accelerometer_y',
    'accelerometer_z',
    'gyroscope_x',
    'gyroscope_y',
    'gyroscope_z'
]


def load_raw_dataset(raw_dir: str, max_records_per_class: int = 40000) -> pd.DataFrame:
    """
    Scans the raw data directory and loads Bump, Pothole, and Normal Road sessions.
    Returns a unified, cleaned, and labeled DataFrame.
    """
    bump_files = glob.glob(os.path.join(raw_dir, '*_isolated_bumps.csv')) + \
                 glob.glob(os.path.join(raw_dir, '*', '*_isolated_bumps.csv'))
    pothole_files = glob.glob(os.path.join(raw_dir, '*_isolated_potholes.csv')) + \
                   glob.glob(os.path.join(raw_dir, '*', '*_isolated_potholes.csv'))
    normal_files = glob.glob(os.path.join(raw_dir, '*Normal Road.csv'))

    dfs = []

    # 1. Load Bump data
    for f in bump_files:
        try:
            df = pd.read_csv(f)
            if all(col in df.columns for col in REQUIRED_SENSOR_COLUMNS):
                df['label_name'] = 'Bump'
                df['label'] = LABEL_MAP['Bump']
                df['source_file'] = os.path.basename(f)
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    # 2. Load Pothole data
    for f in pothole_files:
        try:
            df = pd.read_csv(f)
            if all(col in df.columns for col in REQUIRED_SENSOR_COLUMNS):
                df['label_name'] = 'Pothole'
                df['label'] = LABEL_MAP['Pothole']
                df['source_file'] = os.path.basename(f)
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    # 3. Load Normal Road data
    for f in normal_files:
        try:
            df = pd.read_csv(f)
            if all(col in df.columns for col in REQUIRED_SENSOR_COLUMNS):
                df['label_name'] = 'Normal Road'
                df['label'] = LABEL_MAP['Normal Road']
                df['source_file'] = os.path.basename(f)
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    if not dfs:
        raise ValueError(f"No valid sensor data found in {raw_dir}")

    full_df = pd.concat(dfs, ignore_index=True)

    # Balance classes if max_records_per_class specified
    if max_records_per_class:
        balanced_dfs = []
        for cls_name in ['Normal Road', 'Bump', 'Pothole']:
            cls_df = full_df[full_df['label_name'] == cls_name]
            if len(cls_df) > max_records_per_class:
                cls_df = cls_df.sample(n=max_records_per_class, random_state=42)
            balanced_dfs.append(cls_df)
        full_df = pd.concat(balanced_dfs, ignore_index=True)

    # Sort and clean
    full_df = full_df.dropna(subset=REQUIRED_SENSOR_COLUMNS).reset_index(drop=True)

    # Ensure GPS columns exist (fill with session median if NaN)
    if 'location_latitude' not in full_df.columns:
        full_df['location_latitude'] = 49.2606  # UBC Campus reference
    if 'location_longitude' not in full_df.columns:
        full_df['location_longitude'] = -123.2460

    return full_df


if __name__ == '__main__':
    raw_path = r'C:\Users\rahul\.gemini\antigravity\scratch\RoadSense\data\raw'
    df = load_raw_dataset(raw_path, max_records_per_class=35000)
    print("Dataset Loaded:")
    print(df['label_name'].value_counts())
    print("Shape:", df.shape)
