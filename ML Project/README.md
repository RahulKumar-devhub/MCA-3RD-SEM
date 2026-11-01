# RoadSense
## Machine Learning-Based Road Surface Condition Classification Using Smartphone Inertial Sensors

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Academic Level](https://img.shields.io/badge/Academic-MCA%20Dissertation-emerald.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-RoadSens--4M-purple.svg)](https://doi.org/10.6084/m9.figshare.30341143)

---

### Executive Overview

**RoadSense** is a pervasive, supervised Machine Learning project developed as an MCA final project centered on the **Unit-2 Machine Learning Syllabus**. The system classifies road surface conditions into three primary target categories:
1. **Normal Road:** Smooth asphalt surface characterized by low baseline chassis vibrations ($a_z \approx 0.1 - 0.4\text{ m/s}^2$).
2. **Bump (Speed Breaker):** Elevated traffic-calming structures inducing suspension compression, positive vertical acceleration spikes, and angular pitch rate disturbances.
3. **Pothole:** Severe pavement depressions producing sudden negative acceleration drops followed by high-amplitude, high-kurtosis impact shocks.

The project strictly rejects fabricated or artificially generated sensor values. Training and validation are grounded in the research-backed **RoadSens-4M** multimodal dataset (Amith Khandakar, David Michelson, et al., arXiv: `2510.25211`, Figshare DOI: `10.6084/m9.figshare.30341143`).

---

### Project Architecture & File Hierarchy

```
RoadSense/
├── data/
│   ├── raw/                  # 37 extracted RoadSens-4M sessions (239,438 records @ 100 Hz)
│   ├── processed/            # 1,749 segmented windows (46 features) & simulation_tracks.json
│   └── sample_uploads/       # Preloaded test CSVs (Normal Road, Bump, Pothole, Mixed Trip)
├── models/                   # Serialized ML models (*.joblib), StandardScaler, metadata.json
├── src/
│   ├── data_loader.py        # Dataset ingestion & session aggregation
│   ├── preprocessing.py      # Schema validation, missing value imputation, cleaning
│   ├── feature_engineering.py# 46-dimensional time & frequency-domain feature pipeline
│   ├── rule_based.py         # Interpretable physical threshold baseline classifier
│   ├── train.py              # Terminal training execution script with academic checkmarks
│   ├── evaluate.py           # Metrics calculation, confusion matrices & publication plots
│   └── predict.py            # Unified sliding-window inference manager
├── app/
│   ├── app.py                # Modern Streamlit telematic dashboard (9 pages)
│   ├── components/           # HTML5/Canvas 60 FPS road simulation & live oscilloscopes
│   └── styles.css            # Custom slate/dark telematic styling
├── outputs/
│   ├── figures/              # Publication-grade figures (Confusion Matrices, MDI, Waveforms)
│   └── results/              # Model comparison summary tables (CSV / JSON)
├── report/
│   ├── generate_report.py    # ReportLab automated PDF generator
│   └── RoadSense_Academic_Report.pdf # 14-page formal MCA Dissertation Report
├── notebooks/
│   └── RoadSense_Analysis.ipynb # Interactive exploration and training notebook
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation & demonstration walkthrough
└── main.py                   # Central CLI execution command runner
```

---

### Machine Learning Benchmark Results (Unit-2 Syllabus)

All models were evaluated on an independent 20% test partition (350 windows) using a standardized 46-dimensional feature vector.

| Machine Learning Algorithm | Accuracy | Macro F1-Score | Precision | Recall | Latency (ms / window) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Ensemble)** | **74.00%** | **73.48%** | **73.34%** | **74.00%** | **0.319 ms** |
| **Support Vector Machine (SVM)** | **70.86%** | **70.15%** | **70.21%** | **70.86%** | **0.359 ms** |
| **K-Nearest Neighbors (KNN)** | **70.86%** | **69.99%** | **70.10%** | **70.86%** | **13.841 ms** |
| **Artificial Neural Network (ANN / MLP)** | **62.57%** | **61.21%** | **61.85%** | **62.57%** | **0.009 ms** |
| **Logistic Regression (Multinomial)** | **62.00%** | **60.96%** | **61.22%** | **62.00%** | **0.002 ms** |
| **Decision Tree Classifier** | **58.86%** | **58.35%** | **58.50%** | **58.86%** | **0.003 ms** |
| **Gaussian Naïve Bayes** | **48.57%** | **43.90%** | **45.10%** | **48.57%** | **0.010 ms** |
| **Rule-Based Physical Baseline** | **32.00%** | **24.42%** | **28.10%** | **32.00%** | **0.003 ms** |

*Key Technical Insight:* The Rule-Based thresholding baseline achieves only 32.00% accuracy due to severe false-positive overlap between acceleration braking and road bumps. This empirically proves why supervised Machine Learning (particularly Random Forest and SVM) is essential for robust road condition monitoring.

---

### Quickstart Guide

#### 1. Clone / Navigate to Directory
```bash
cd C:\Users\rahul\.gemini\antigravity\scratch\RoadSense
```

#### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run the Terminal Training Pipeline
To inspect the clean, academic terminal training output with checkmarks and ASCII tables:
```bash
python main.py --train
# or: python src/train.py
```

#### 4. Launch the Streamlit Web Application
To start the localhost interactive dashboard:
```bash
python main.py
# or: streamlit run app/app.py
```
Open your browser at **`http://localhost:8501`**.

#### 5. Compile the Formal Academic PDF Report
```bash
python main.py --report
# Generates report/RoadSense_Academic_Report.pdf
```

---

### Step-by-Step Teacher / Evaluator Demonstration Walkthrough

1. **Launch Application:** Run `python main.py`. Open `http://localhost:8501`.
2. **Explore Dashboard:** Review the top metric cards (239,438 records, 8 models, Random Forest leaderboard) and class distribution chart.
3. **Live Road Monitor Demonstration:**
   - Select **🚗 Live Road Monitor** from the sidebar.
   - Click **`[▶ Start Simulation]`**.
   - Observe the 2-lane asphalt road scrolling at 60 FPS with white dashed dividers and curbs.
   - Watch the overhead sedan navigate the road.
   - Observe the **Dual Live Oscilloscopes**: Accelerometer (X, Y, Z, Mag) and Gyroscope (Roll, Pitch, Yaw) updating in real time.
   - **Event 1 (Speed Bump):** As the car crosses the yellow/black chevron bump, notice the positive vertical Z-axis surge, the amber HUD alert (`⚠ BUMP DETECTED`), and the Random Forest model predicting `Bump` with ~95% confidence.
   - **Event 2 (Normal Road):** Vehicle returns to smooth road; indicator smoothly reverts to `✓ NORMAL ROAD`.
   - **Event 3 (Deep Pothole):** As the car strikes the dark road crater, notice the sharp negative drop followed by high-kurtosis rebound shock, the red HUD alert (`🚧 POTHOLE DETECTED`), and instantaneous ML inference.
   - Test the playback controls: `[⏸ Pause]`, `[Speed: 0.5x / 1.0x / 2.0x]`, and `[↻ Restart]`.
4. **Sensor Data Analyzer Demonstration:**
   - Select **🔬 Sensor Data Analyzer** from the sidebar.
   - Click **`[📄 Load Speed Bump Sample]`** or **`[📄 Load Pothole Sample]`** (or upload your own CSV).
   - Verify that the pipeline checks columns, imputes missing timestamps, runs windowed feature extraction, and displays the predicted condition and interactive Plotly waveform.
5. **Model Comparison & Explainability:**
   - Inspect the **📈 Model Comparison** tab to review grouped bar charts and confusion matrices.
   - Inspect **🧠 Model Explainability** to see that vertical Z-axis RMS, Peak-to-Peak excursion, and Kurtosis dominate classification.
   - Review the **🕸 ANN** and **📐 SVM** tabs for theoretical deep-dives.
   - Review **🗺 Geospatial Map View** to see genuine GPS incident markers.

---

### Academic Honesty Statement: Real vs. Simulation Mode

- **Real Sensor Data Mode (`Sensor Data Analyzer`):** Operates strictly on authentic, pre-recorded or user-uploaded CSV sensor feeds. All predictions are generated dynamically by the trained Machine Learning models without hardcoding.
- **Simulation Demonstration Mode (`Live Road Monitor`):** Designed for evaluator demonstration. It animates a virtual roadway while synchronizing genuine RoadSens-4M session telemetry and feeding it through the trained Random Forest model. It is explicitly labeled **"RoadSense Simulation Mode"** to maintain 100% academic transparency.

---

### Academic Screenshot Checklist

| # | Screenshot Target | What Must Be Visible | Report Section |
| :---: | :--- | :--- | :---: |
| **1** | Terminal Training Output | Checkmarks `[1]` to `[5]`, model training checkmarks, and ASCII evaluation table | Section 6.1 |
| **2** | Main Dashboard | Top KPI cards (239K records, 3 classes, 8 models), leaderboard, class distribution pie | Section 7.1 |
| **3** | Live Road Monitor (Normal) | Vehicle on smooth asphalt, green HUD banner (`✓ NORMAL ROAD`), baseline IMU waveforms | Section 8.1 |
| **4** | Live Road Monitor (Bump) | Vehicle traversing chevron bump, amber alert (`⚠ BUMP DETECTED`), Z-axis peak | Section 8.2 |
| **5** | Live Road Monitor (Pothole) | Vehicle striking crater, red warning (`🚧 POTHOLE DETECTED`), shock dip & rebound | Section 8.3 |
| **6** | Dual Live Oscilloscopes | 60 FPS plotting of Accelerometer (X, Y, Z, Mag) and Gyroscope (Roll, Pitch, Yaw) | Section 8.4 |
| **7** | Sensor Data Analyzer | CSV upload portal, sample buttons, schema check (`0 sensor nulls`) | Section 9.1 |
| **8** | Real ML Prediction Result | Batch prediction banner, window breakdown, interactive Plotly waveform | Section 9.2 |
| **9** | Model Comparison Page | Interactive comparison table, metric bar chart, per-model confusion matrix | Section 10.1 |
| **10** | Model Explainability Page | Random Forest feature importance bar chart (Z-RMS, Peak Accel, Kurtosis) | Section 11.1 |
| **11** | ANN & SVM Deep-Dives | Neural network architecture table, cross-entropy loss curve, SVM 2D PCA space | Section 12.1 |
| **12** | Geospatial Route Map | Dark-matter map with green, amber, and red incident markers plotted on real GPS | Section 13.1 |

---

### Academic Citation

If utilizing this codebase or methodology in academic research, please cite:
```bibtex
@misc{khandakar2025roadsens4m,
  title={RoadSens-4M: A Multimodal Smartphone & Camera Dataset for Holistic Road-way Analysis},
  author={Khandakar, Amith and Michelson, David and Rabbani, Shaikh Golam and others},
  year={2025},
  eprint={2510.25211},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  doi={10.6084/m9.figshare.30341143}
}
```
