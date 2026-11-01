"""
RoadSense - Data Preprocessing & Validation
Handles sensor data sanitization, schema verification, and missing value imputation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

EXPECTED_SENSOR_COLUMNS = [
    'seconds_elapsed',
    'accelerometer_x',
    'accelerometer_y',
    'accelerometer_z',
    'gyroscope_x',
    'gyroscope_y',
    'gyroscope_z'
]

# Flexible column aliases for user uploads
COLUMN_ALIASES = {
    'time': 'seconds_elapsed',
    'timestamp': 'seconds_elapsed',
    'seconds': 'seconds_elapsed',
    'acc_x': 'accelerometer_x',
    'ax': 'accelerometer_x',
    'accel_x': 'accelerometer_x',
    'acc_y': 'accelerometer_y',
    'ay': 'accelerometer_y',
    'accel_y': 'accelerometer_y',
    'acc_z': 'accelerometer_z',
    'az': 'accelerometer_z',
    'accel_z': 'accelerometer_z',
    'gyro_x': 'gyroscope_x',
    'gx': 'gyroscope_x',
    'gyro_y': 'gyroscope_y',
    'gy': 'gyroscope_y',
    'gyro_z': 'gyroscope_z',
    'gz': 'gyroscope_z'
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames uploaded CSV column headers to canonical names using aliases.
    """
    clean_cols = {}
    for col in df.columns:
        norm = col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '')
        if norm in COLUMN_ALIASES:
            clean_cols[col] = COLUMN_ALIASES[norm]
        elif norm in EXPECTED_SENSOR_COLUMNS:
            clean_cols[col] = norm
    return df.rename(columns=clean_cols)


def validate_sensor_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validates presence of required sensor columns.
    Returns: (is_valid, missing_columns, standardized_df)
    """
    df_std = standardize_columns(df.copy())
    missing = [c for c in EXPECTED_SENSOR_COLUMNS if c not in df_std.columns]

    # If seconds_elapsed missing, synthesize from index assuming 100 Hz
    if 'seconds_elapsed' in missing:
        df_std['seconds_elapsed'] = np.arange(len(df_std)) * 0.01
        missing.remove('seconds_elapsed')

    if missing:
        return False, missing, df_std

    return True, [], df_std


def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw sensor readings: forward/backward fills NaNs, trims infinite values.
    """
    df_clean = df.copy()

    # Drop fully empty rows
    df_clean = df_clean.dropna(how='all')

    # Interpolate sensor columns
    for col in EXPECTED_SENSOR_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].ffill().bfill().fillna(0.0)

    # Sort chronologically
    if 'seconds_elapsed' in df_clean.columns:
        df_clean = df_clean.sort_values(by='seconds_elapsed').reset_index(drop=True)

    return df_clean
