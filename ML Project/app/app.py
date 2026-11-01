"""
RoadSense - Main Streamlit Web Application
AI-Powered Road Surface Condition Monitoring Using Smartphone Inertial Sensors
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root and app directory are in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.preprocessing import validate_sensor_dataframe, clean_sensor_data, EXPECTED_SENSOR_COLUMNS
from src.feature_engineering import compute_window_features, FEATURE_NAMES
from src.predict import RoadSensePredictor, CLASS_LABELS

try:
    from app.components.simulation_component import render_simulation_html
except (ModuleNotFoundError, ImportError):
    try:
        from components.simulation_component import render_simulation_html
    except (ModuleNotFoundError, ImportError):
        from simulation_component import render_simulation_html


# Page configuration
st.set_page_config(
    page_title="RoadSense - AI Road Condition Monitoring",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = os.path.join(BASE_DIR, "app", "styles.css")
if os.path.exists(css_path):
    with open(css_path, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return RoadSensePredictor(os.path.join(BASE_DIR, "models"))


@st.cache_data
def load_metadata():
    with open(os.path.join(BASE_DIR, "models", "model_metadata.json"), 'r') as f:
        return json.load(f)


@st.cache_data
def load_comparison_data():
    csv_path = os.path.join(BASE_DIR, "outputs", "results", "model_comparison.csv")
    return pd.read_csv(csv_path)


predictor = load_predictor()
metadata = load_metadata()
comparison_df = load_comparison_data()


# Sidebar Navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 12px 0 20px 0;">
    <h2 style="color: #38bdf8; margin: 0; font-weight: 800; letter-spacing: -0.5px;">🚗 RoadSense</h2>
    <p style="color: #94a3b8; font-size: 11px; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px;">
        Smartphone IMU ML Classifier
    </p>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🚗 Live Road Monitor",
        "🔬 Sensor Data Analyzer",
        "📈 Model Comparison",
        "🧠 Model Explainability",
        "🕸 Neural Network (ANN)",
        "📐 Support Vector Machine",
        "🗺 Geospatial Map View",
        "🏛 System Architecture"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 12px; font-size: 12px;">
    <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Core Dataset</div>
    <div style="color: #f8fafc; font-weight: 700; margin-top: 2px;">RoadSens-4M (2025/26)</div>
    <div style="color: #64748b; font-size: 11px; margin-top: 2px;">Figshare DOI: 10.6084</div>
    <hr style="border: 0; border-top: 1px solid #1f2937; margin: 8px 0;">
    <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Top Model</div>
    <div style="color: #38bdf8; font-weight: 700; margin-top: 2px;">{metadata.get('best_model', 'Random Forest')}</div>
    <div style="color: #34d399; font-size: 11px; margin-top: 2px;">F1-Score: {metadata.get('best_f1', 73.48):.2f}%</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 1. DASHBOARD PAGE
# ==========================================
if nav_selection == "📊 Dashboard":
    st.markdown("""
    <div class="rs-header-container">
        <div>
            <div class="rs-title">🚗 Road<span>Sense</span></div>
            <div class="rs-subtitle">AI-Powered Road Surface Condition Monitoring Using Smartphone Inertial Sensors</div>
        </div>
        <div class="rs-badge">MCA ML Project • Unit-2 Syllabus</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Metrics Grid
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="rs-stat-card">
            <div class="rs-stat-label">Total Raw Records</div>
            <div class="rs-stat-val">239,438</div>
            <div class="rs-stat-sub">Across 37 Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="rs-stat-card">
            <div class="rs-stat-label">Road Surface Classes</div>
            <div class="rs-stat-val">3</div>
            <div class="rs-stat-sub">Normal • Bump • Pothole</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="rs-stat-card">
            <div class="rs-stat-label">Trained ML Models</div>
            <div class="rs-stat-val">{len(comparison_df)}</div>
            <div class="rs-stat-sub">Unit-2 Supervised Suite</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="rs-stat-card">
            <div class="rs-stat-label">Top Algorithm</div>
            <div class="rs-stat-val" style="color: #38bdf8; font-size: 20px;">{metadata.get('best_model', 'Random Forest')}</div>
            <div class="rs-stat-sub">Ensemble Bagging</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="rs-stat-card">
            <div class="rs-stat-label">Peak F1-Score</div>
            <div class="rs-stat-val" style="color: #34d399;">{metadata.get('best_f1', 73.48):.2f}%</div>
            <div class="rs-stat-sub">Accuracy: {metadata.get('best_accuracy', 74.00):.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("""
        <div class="rs-box">
            <div class="rs-box-title">📊 Algorithm Benchmark Leaderboard</div>
        </div>
        """, unsafe_allow_html=True)

        fig_bar = px.bar(
            comparison_df,
            x='model_name',
            y=['accuracy', 'f1_macro'],
            barmode='group',
            labels={'value': 'Score', 'variable': 'Metric', 'model_name': 'Algorithm'},
            color_discrete_map={'accuracy': '#38bdf8', 'f1_macro': '#34d399'},
            height=320
        )
        fig_bar.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("""
        <div class="rs-box">
            <div class="rs-box-title">📈 Research Dataset Class Proportions</div>
        </div>
        """, unsafe_allow_html=True)

        class_counts = pd.DataFrame({
            'Condition': ['Normal Road', 'Bump / Speed Breaker', 'Pothole'],
            'Records': [44046, 129049, 66343]
        })
        fig_pie = px.pie(
            class_counts,
            values='Records',
            names='Condition',
            color='Condition',
            color_discrete_map={
                'Normal Road': '#10b981',
                'Bump / Speed Breaker': '#f59e0b',
                'Pothole': '#ef4444'
            },
            hole=0.45,
            height=320
        )
        fig_pie.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Physical Sensor Signatures Overview
    st.markdown("""
    <div class="rs-box">
        <div class="rs-box-title">🔬 Physical Sensor Signatures (Accelerometer Z-Axis Kinematics)</div>
    </div>
    """, unsafe_allow_html=True)

    sig_img_path = os.path.join(BASE_DIR, "outputs", "figures", "sensor_signatures.png")
    if os.path.exists(sig_img_path):
        st.image(sig_img_path, caption="Comparative 100 Hz Accelerometer Waveforms for Normal Road, Speed Bump, and Pothole Events", use_container_width=True)


# ==========================================
# 2. LIVE ROAD MONITOR PAGE
# ==========================================
elif nav_selection == "🚗 Live Road Monitor":
    st.markdown("""
    <div class="rs-header-container">
        <div>
            <div class="rs-title">🚗 Live Road <span>Monitor</span></div>
            <div class="rs-subtitle">Interactive Vehicle Simulation with Synchronized Research IMU Playback & Real-Time ML Inference</div>
        </div>
        <div class="rs-badge" style="background: rgba(245, 158, 11, 0.15); border-color: #f59e0b; color: #fbbf24;">
            RoadSense Simulation Mode
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "💡 **Academic Demarcation:** This demonstration page runs in **Simulation Mode**, utilizing high-resolution sequential playback "
        "of actual RoadSens-4M vehicle runs (Normal Road → Speed Bump → Normal Road → Deep Pothole → Normal Road). "
        "At each frame, genuine sensor-derived features are computed and classified in real time by the trained **Random Forest** model."
    )

    track_path = os.path.join(BASE_DIR, "data", "processed", "simulation_tracks.json")
    if os.path.exists(track_path):
        sim_html = render_simulation_html(track_path)
        components.html(sim_html, height=880, scrolling=False)
    else:
        st.error("Simulation track data not found. Please run `python src/train.py` first.")


# ==========================================
# 3. SENSOR DATA ANALYZER PAGE
# ==========================================
elif nav_selection == "🔬 Sensor Data Analyzer":
    st.markdown("""
    <div class="rs-header-container">
        <div>
            <div class="rs-title">🔬 Sensor Data <span>Analyzer</span></div>
            <div class="rs-subtitle">Upload Real Smartphone IMU Telemetry (CSV) for Feature Extraction and Multi-Model Classification</div>
        </div>
        <div class="rs-badge" style="background: rgba(16, 185, 129, 0.15); border-color: #10b981; color: #34d399;">
            Real Sensor Data Mode
        </div>
    </div>
    """, unsafe_allow_html=True)

    sample_dir = os.path.join(BASE_DIR, "data", "sample_uploads")

    st.markdown("### 1. Select a Sample Test CSV or Upload Your Own")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    selected_sample_file = None
    with col_s1:
        if st.button("📄 Load Normal Road Sample", use_container_width=True):
            selected_sample_file = os.path.join(sample_dir, "sample_normal_road.csv")
    with col_s2:
        if st.button("📄 Load Speed Bump Sample", use_container_width=True):
            selected_sample_file = os.path.join(sample_dir, "sample_bump.csv")
    with col_s3:
        if st.button("📄 Load Pothole Sample", use_container_width=True):
            selected_sample_file = os.path.join(sample_dir, "sample_pothole.csv")
    with col_s4:
        if st.button("📄 Load Mixed Trip (500m)", use_container_width=True):
            selected_sample_file = os.path.join(sample_dir, "sample_mixed_trip.csv")

    uploaded_file = st.file_uploader(
        "Or upload a custom smartphone sensor CSV (Requires: seconds_elapsed, accelerometer_x/y/z, gyroscope_x/y/z)",
        type=["csv"]
    )

    df_to_analyze = None
    data_source_label = ""

    if uploaded_file is not None:
        try:
            df_to_analyze = pd.read_csv(uploaded_file)
            data_source_label = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Failed to parse uploaded CSV: {e}")
    elif selected_sample_file and os.path.exists(selected_sample_file):
        df_to_analyze = pd.read_csv(selected_sample_file)
        data_source_label = f"Preloaded Sample: {os.path.basename(selected_sample_file)}"
        st.session_state['active_sample'] = selected_sample_file
    elif 'active_sample' in st.session_state and os.path.exists(st.session_state['active_sample']):
        df_to_analyze = pd.read_csv(st.session_state['active_sample'])
        data_source_label = f"Preloaded Sample: {os.path.basename(st.session_state['active_sample'])}"

    if df_to_analyze is not None:
        st.success(f"Loaded {data_source_label} ({len(df_to_analyze):,} rows)")

        # Validate Schema
        is_valid, missing, df_std = validate_sensor_dataframe(df_to_analyze)
        if not is_valid:
            st.error(f"❌ Uploaded CSV is missing mandatory sensor columns: {missing}. Please ensure your CSV includes: {EXPECTED_SENSOR_COLUMNS}")
        else:
            # Model Selection
            chosen_model_name = st.selectbox(
                "Select Machine Learning Algorithm for Inference",
                options=[
                    "Random Forest", "SVM", "KNN", "ANN / MLP",
                    "Logistic Regression", "Decision Tree", "Naïve Bayes", "Rule-Based"
                ],
                index=0
            )

            # Run batch sliding-window inference
            with st.spinner("Extracting statistical/spectral features and executing ML inference..."):
                results = predictor.predict_dataframe(df_to_analyze, model_name=chosen_model_name)

            cond = results['overall_condition']
            badge_class = "rs-indicator-normal" if cond == "Normal Road" else ("rs-indicator-bump" if cond == "Bump" else "rs-indicator-pothole")
            badge_icon = "✓" if cond == "Normal Road" else ("⚠" if cond == "Bump" else "🚧")

            st.markdown(f"""
            <div class="rs-indicator {badge_class}">
                <span style="font-size: 22px;">{badge_icon}</span>
                <div>
                    <div style="font-size: 18px; font-weight: 800;">PREDICTED CONDITION: {cond.upper()}</div>
                    <div style="font-size: 12px; opacity: 0.85;">Model: {chosen_model_name} • Analyzed {results['total_windows']} Windows ({results['total_records']} raw 100 Hz samples)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Class Distribution among windows
            c_dist1, c_dist2, c_dist3 = st.columns(3)
            with c_dist1:
                st.metric("Normal Road Windows", f"{results['class_distribution']['Normal Road']} ({results['class_distribution']['Normal Road']/max(1, results['total_windows'])*100:.1f}%)")
            with c_dist2:
                st.metric("Speed Bump Windows", f"{results['class_distribution']['Bump']} ({results['class_distribution']['Bump']/max(1, results['total_windows'])*100:.1f}%)")
            with c_dist3:
                st.metric("Pothole Windows", f"{results['class_distribution']['Pothole']} ({results['class_distribution']['Pothole']/max(1, results['total_windows'])*100:.1f}%)")

            # Interactive Plotly Waveform
            st.markdown("#### Raw Inertial Waveform Visualizer")
            df_plot = df_std.copy()
            if 'seconds_elapsed' not in df_plot.columns:
                df_plot['seconds_elapsed'] = np.arange(len(df_plot)) * 0.01

            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(x=df_plot['seconds_elapsed'], y=df_plot['accelerometer_z'], mode='lines', name='Acc Z (Vertical)', line=dict(color='#f59e0b', width=1.8)))
            fig_wave.add_trace(go.Scatter(x=df_plot['seconds_elapsed'], y=df_plot['accelerometer_x'], mode='lines', name='Acc X (Lateral)', line=dict(color='#38bdf8', width=1.2, dash='dot')))
            fig_wave.add_trace(go.Scatter(x=df_plot['seconds_elapsed'], y=df_plot['accelerometer_y'], mode='lines', name='Acc Y (Longitudinal)', line=dict(color='#a855f7', width=1.2, dash='dot')))

            fig_wave.update_layout(
                template='plotly_dark',
                title="Tri-Axial Acceleration Dynamics (m/s²)",
                xaxis_title="Time Elapsed (seconds)",
                yaxis_title="Acceleration (m/s²)",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_wave, use_container_width=True)

            # Feature Inspection Table
            with st.expander("🔍 View Extracted Window Features Table"):
                if results['window_predictions']:
                    feat_rows = [wp['features'] for wp in results['window_predictions']]
                    st.dataframe(pd.DataFrame(feat_rows).round(4), use_container_width=True)
    else:
        st.info("👆 Click one of the sample buttons above or upload a custom CSV file to test real ML classification.")


# ==========================================
# 4. MODEL COMPARISON PAGE
# ==========================================
elif nav_selection == "📈 Model Comparison":
    st.markdown("""
    <div class="rs-header-container">
        <div>
            <div class="rs-title">📈 Model <span>Comparison</span></div>
            <div class="rs-subtitle">Comprehensive Benchmark Across Unit-2 Syllabus Supervised Learning Algorithms</div>
        </div>
        <div class="rs-badge">Empirical Results</div>
    </div>
    """, unsafe_allow_html=True)

    # Leaderboard Table
    display_df = comparison_df.copy()
    display_df['accuracy'] = (display_df['accuracy'] * 100).round(2).astype(str) + "%"
    display_df['precision_macro'] = (display_df['precision_macro'] * 100).round(2).astype(str) + "%"
    display_df['recall_macro'] = (display_df['recall_macro'] * 100).round(2).astype(str) + "%"
    display_df['f1_macro'] = (display_df['f1_macro'] * 100).round(2).astype(str) + "%"
    display_df['latency_ms_per_window'] = display_df['latency_ms_per_window'].round(3).astype(str) + " ms"

    st.dataframe(
        display_df[['model_name', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'latency_ms_per_window']].rename(
            columns={
                'model_name': 'Algorithm',
                'accuracy': 'Accuracy',
                'precision_macro': 'Precision',
                'recall_macro': 'Recall',
                'f1_macro': 'Macro F1',
                'latency_ms_per_window': 'Inference Latency'
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### Metric Comparison Chart")
        comp_img_path = os.path.join(BASE_DIR, "outputs", "figures", "model_comparison.png")
        if os.path.exists(comp_img_path):
            st.image(comp_img_path, use_container_width=True)

    with col_c2:
        st.markdown("#### Individual Confusion Matrix Inspector")
        inspect_model = st.selectbox(
            "Select Model to Inspect Confusion Matrix",
            options=comparison_df['model_name'].tolist(),
            index=0
        )
        safe_name = inspect_model.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        # Match file name
        cm_files = [f for f in os.listdir(os.path.join(BASE_DIR, "outputs", "figures")) if f.startswith("cm_") and safe_name[:6] in f]
        if cm_files:
            st.image(os.path.join(BASE_DIR, "outputs", "figures", cm_files[0]), use_container_width=True)
        else:
            st.warning("Confusion matrix figure not found.")


# ==========================================
# 5. MODEL EXPLAINABILITY
# ==========================================
elif nav_selection == "🧠 Model Explainability":
    st.markdown("""
    <div class="rs-header-container">
        <div>
            <div class="rs-title">🧠 Model <span>Explainability</span></div>
            <div class="rs-subtitle">Feature Importance and Interpretability for Road Surface Anomaly Classification</div>
        </div>
        <div class="rs-badge">Feature Attribution</div>
    </div>
    """, unsafe_allow_html=True)

    fi_img_path = os.path.join(BASE_DIR, "outputs", "figures", "feature_importance.png")
    if os.path.exists(fi_img_path):
        st.image(fi_img_path, caption="Random Forest Feature Importance (Mean Decrease in Impurity - MDI)", use_container_width=True)

    st.markdown("""
    <div class="rs-box">
        <div class="rs-box-title">🔬 Physical Explanation of Dominant Sensor Features</div>
        <ul style="color: #cbd5e1; font-size: 13px; line-height: 1.8;">
            <li><strong>ACC_Z_RMS & ACC_Z_P2P (Vertical Excursion):</strong> Vertical acceleration holds the highest predictive weight. Speed bumps cause a pronounced positive compression peak followed by rebound, whereas potholes cause sudden negative drops followed by violent shocks.</li>
            <li><strong>ACC_Z_KURTOSIS (Impulsiveness):</strong> Kurtosis measures the "tailedness" of the signal distribution. Sharp impacts (potholes) produce large kurtosis values (> 3.0), cleanly distinguishing them from smooth roads.</li>
            <li><strong>ACC_MAG_RMS & GYRO_MAG_RMS (Energy):</strong> Overall kinetic energy dissipates through the vehicle chassis. Gyroscopic roll and pitch rates detect the lateral tilting when only one wheel falls into a pothole.</li>
            <li><strong>Dominant Frequency (FFT):</strong> Speed bumps excite low-frequency suspension modes (1-3 Hz), while sharp asphalt cracks generate high-frequency vibrations (8-20 Hz).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 6. ARTIFICIAL NEURAL NETWORK (ANN)
# ==========================================
elif nav_selection == "🕸 Neural Network (ANN)":
    st.markdown("""
    <div class="rs-header-container">
        <div>
            <div class="rs-title">🕸 Artificial Neural Network <span>(ANN / MLP)</span></div>
            <div class="rs-subtitle">Multi-Layer Perceptron Architecture, Layer Formulations, and Convergence Profile</div>
        </div>
        <div class="rs-badge">Deep Learning Baseline</div>
    </div>
    """, unsafe_allow_html=True)

    c_ann1, c_ann2 = st.columns([1.1, 1])

    with c_ann1:
        st.markdown("""
        <div class="rs-box">
            <div class="rs-box-title">📐 Architecture Specification</div>
            <table style="width: 100%; color: #cbd5e1; font-size: 13px;">
                <tr style="border-bottom: 1px solid #1f2937;"><td><strong>Input Dimension:</strong></td><td>46 Normalized Features</td></tr>
                <tr style="border-bottom: 1px solid #1f2937;"><td><strong>Hidden Layer 1:</strong></td><td>64 Neurons (ReLU Activation)</td></tr>
                <tr style="border-bottom: 1px solid #1f2937;"><td><strong>Regularization:</strong></td><td>L2 Penalty (alpha=0.0001) + Early Stopping</td></tr>
                <tr style="border-bottom: 1px solid #1f2937;"><td><strong>Hidden Layer 2:</strong></td><td>32 Neurons (ReLU Activation)</td></tr>
                <tr style="border-bottom: 1px solid #1f2937;"><td><strong>Output Layer:</strong></td><td>3 Neurons (Softmax Probability Distribution)</td></tr>
                <tr style="border-bottom: 1px solid #1f2937;"><td><strong>Optimization:</strong></td><td>Adam Optimizer (lr=0.001)</td></tr>
                <tr><td><strong>Loss Function:</strong></td><td>Categorical Cross-Entropy</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with c_ann2:
        ann_loss_path = os.path.join(BASE_DIR, "outputs", "figures", "ann_loss_curve.png")
        if os.path.exists(ann_loss_path):
            st.image(ann_loss_path, caption="Cross-Entropy Training Loss Convergence Curve", use_container_width=True)


# ==========================================
# 7. SUPPORT VECTOR MACHINE (SVM)
# ==========================================
elif nav_selection == "📐 Support Vector Machine":
    st.markdown("""
    <div class="rs-header-container">
        <div>
            <div class="rs-title">📐 Support Vector <span>Machine (SVM)</span></div>
            <div class="rs-subtitle">Hyperplane Margin Maximization, RBF Kernel Formulation, and 2D Decision Space</div>
        </div>
        <div class="rs-badge">Kernel Methods</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rs-box">
        <div class="rs-box-title">Mathematical Formulation</div>
        <p style="color: #cbd5e1; font-size: 13px; line-height: 1.6;">
            SVM finds the optimal separating hyperplane that maximizes the geometric margin between classes in high-dimensional feature space.
            Because road anomaly features exhibit non-linear boundaries, we employ the <strong>Radial Basis Function (RBF) Kernel</strong>:
            <br><br>
            <code>K(x, x') = exp(-γ ||x - x'||²)</code>
            <br><br>
            where <code>C = 5.0</code> controls the penalty for misclassification and <code>γ = scale</code> sets the influence radius of individual support vectors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    pca_img_path = os.path.join(BASE_DIR, "outputs", "figures", "svm_pca_projection.png")
    if os.path.exists(pca_img_path):
        st.image(pca_img_path, caption="2D PCA Feature Space Projection showing Class Clusters and Separability", use_container_width=True)


# ==========================================
# 8. GEOSPATIAL MAP VIEW
# ==========================================
elif nav_selection == "🗺 Geospatial Map View":
    st.markdown("""
    <div class="rs-header-container">
        <div>
            <div class="rs-title">🗺 Geospatial <span>Map View</span></div>
            <div class="rs-subtitle">Real-World Route Trajectory and Anomaly Incident Pinpointing (RoadSens-4M GPS Stream)</div>
        </div>
        <div class="rs-badge">Geographic Mapping</div>
    </div>
    """, unsafe_allow_html=True)

    track_path = os.path.join(BASE_DIR, "data", "processed", "simulation_tracks.json")
    if os.path.exists(track_path):
        with open(track_path, 'r') as f:
            points = json.load(f)

        df_map = pd.DataFrame(points)
        df_map = df_map.dropna(subset=['latitude', 'longitude'])

        if not df_map.empty:
            color_map = {'Normal Road': '#10b981', 'Bump': '#f59e0b', 'Pothole': '#ef4444'}

            fig_map = px.scatter_mapbox(
                df_map,
                lat='latitude',
                lon='longitude',
                color='ground_truth',
                color_discrete_map=color_map,
                hover_data=['ground_truth', 'acc_mag', 'time'],
                zoom=14,
                height=550
            )
            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02)
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("🟢 Green: Normal Road • 🟡 Amber: Speed Breaker / Bump • 🔴 Red: Pothole Incident")
        else:
            st.warning("No GPS coordinates found in track data.")


# ==========================================
# 9. SYSTEM ARCHITECTURE
# ==========================================
elif nav_selection == "🏛 System Architecture":
    st.markdown("""
    <div class="rs-header-container">
        <div>
            <div class="rs-title">🏛 System <span>Architecture</span></div>
            <div class="rs-subtitle">End-to-End Supervised Machine Learning Pipeline for Smartphone Road Telemetry</div>
        </div>
        <div class="rs-badge">Pipeline Design</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rs-box">
        <div class="rs-box-title">Visual ML Workflow Diagram</div>
        <div style="background: #0b0f19; padding: 24px; border-radius: 8px; font-family: monospace; font-size: 13px; line-height: 1.8; color: #38bdf8;">
            Smartphone 6-Axis IMU Sensors (Accelerometer + Gyroscope @ 100 Hz)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼<br>
            Data Cleaning & Timestamp Synchronization (0 Nulls, Chronological Sorting)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼<br>
            Sliding Window Segmentation (Window = 1.0s / 100 Samples, Overlap = 50%)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼<br>
            46-Dimensional Feature Extraction (RMS, P2P, Kurtosis, Energy, Spectral FFT)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼<br>
            Feature Normalization (StandardScaler Fitted Strictly on Training Set)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼<br>
            Supervised Classifiers (Random Forest, SVM, KNN, ANN/MLP, Decision Tree)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼<br>
            Predicted Road Condition [Normal Road | Bump | Pothole] &amp; Alert HUD
        </div>
    </div>
    """, unsafe_allow_html=True)
