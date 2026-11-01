"""
RoadSense - Jupyter Notebook Generator
Creates an academic Jupyter notebook in notebooks/RoadSense_Analysis.ipynb
"""

import os
import nbformat as nbf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# RoadSense: Machine Learning-Based Road Surface Condition Classification
## Academic Dissertation Notebook — MCA Unit-2 Supervised Machine Learning
**Author:** MCA Candidate  
**Dataset:** Research-Backed RoadSens-4M (Smartphone Accelerometer & Gyroscope Telemetry @ 100 Hz)  
**Target Classes:** Normal Road (0), Bump / Speed Breaker (1), Pothole (2)
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, os.path.abspath('..'))

from src.data_loader import load_raw_dataset
from src.preprocessing import clean_sensor_data, validate_sensor_dataframe
from src.feature_engineering import extract_features_from_dataframe, FEATURE_NAMES
from src.predict import RoadSensePredictor

print("Libraries imported successfully!")
"""))

# Step 1: Load Data
cells.append(nbf.v4.new_markdown_cell("""### 1. Load Research Dataset (RoadSens-4M)
Loading 100 Hz tri-axial accelerometer and gyroscope data across Normal Road, Bump, and Pothole sessions.
"""))

cells.append(nbf.v4.new_code_cell("""raw_dir = os.path.join('..', 'data', 'raw')
raw_df = load_raw_dataset(raw_dir, max_records_per_class=10000)
cleaned_df = clean_sensor_data(raw_df)
print("Dataset Loaded:")
print(cleaned_df['label_name'].value_counts())
"""))

# Step 2: Visualizing Waveforms
cells.append(nbf.v4.new_markdown_cell("""### 2. Physical Sensor Signatures
Visualizing vertical acceleration ($a_z$) across Normal Road, Speed Bump, and Pothole events.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 6))

for i, cls_name in enumerate(['Normal Road', 'Bump', 'Pothole']):
    sample = cleaned_df[cleaned_df['label_name'] == cls_name].iloc[50:150]
    plt.subplot(3, 1, i+1)
    plt.plot(np.arange(len(sample))*0.01, sample['accelerometer_z'], label=f'{cls_name} (Acc Z)')
    plt.ylabel('m/s²')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

plt.xlabel('Time Elapsed (s)')
plt.tight_layout()
plt.show()
"""))

# Step 3: Feature Extraction & Results
cells.append(nbf.v4.new_markdown_cell("""### 3. Evaluated Model Benchmark
Loading the benchmark results across all 8 Unit-2 machine learning algorithms.
"""))

cells.append(nbf.v4.new_code_cell("""results_df = pd.read_csv(os.path.join('..', 'outputs', 'results', 'model_comparison.csv'))
display(results_df[['model_name', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'latency_ms_per_window']])
"""))

# Step 4: Inference Demonstration
cells.append(nbf.v4.new_markdown_cell("""### 4. Real-Time Inference Demo
Executing real model inference using the trained Random Forest model on sample upload files.
"""))

cells.append(nbf.v4.new_code_cell("""predictor = RoadSensePredictor(os.path.join('..', 'models'))
sample_bump = pd.read_csv(os.path.join('..', 'data', 'sample_uploads', 'sample_bump.csv'))
res = predictor.predict_dataframe(sample_bump, model_name='Random Forest')
print(f"Predicted Condition: {res['overall_condition']}")
print(f"Class Breakdown: {res['class_distribution']}")
"""))

nb['cells'] = cells

out_nb_path = os.path.join(BASE_DIR, "notebooks", "RoadSense_Analysis.ipynb")
with open(out_nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Jupyter notebook created at: {out_nb_path}")
