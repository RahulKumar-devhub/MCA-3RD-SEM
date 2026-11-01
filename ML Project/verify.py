import os
import joblib
import json
import pandas as pd

base = r'C:\Users\rahul\.gemini\antigravity\scratch\RoadSense'
models = ['random_forest.joblib', 'svm.joblib', 'knn.joblib', 'ann_mlp.joblib', 'logistic_regression.joblib', 'decision_tree.joblib', 'naive_bayes.joblib', 'rule-based.joblib', 'scaler.joblib']
for m in models:
    p = os.path.join(base, 'models', m)
    assert os.path.exists(p), f'Missing {m}'
    joblib.load(p)
print('[1] All 9 model/scaler artifacts loadable: OK')

meta = json.load(open(os.path.join(base, 'models', 'model_metadata.json')))
print(f'[2] Metadata: Best={meta.get("best_model")} Acc={meta.get("best_accuracy"):.2f}%: OK')

res_df = pd.read_csv(os.path.join(base, 'outputs', 'results', 'model_comparison.csv'))
assert len(res_df) == 8
print(f'[3] Comparison Table: {len(res_df)} algorithms: OK')

figures = ['model_comparison.png', 'feature_importance.png', 'sensor_signatures.png', 'class_distribution.png', 'ann_loss_curve.png', 'svm_pca_projection.png']
for f in figures:
    p = os.path.join(base, 'outputs', 'figures', f)
    assert os.path.exists(p), f'Missing {f}'
print(f'[4] All {len(figures)} figures verified on disk: OK')

samples = ['sample_normal_road.csv', 'sample_bump.csv', 'sample_pothole.csv', 'sample_mixed_trip.csv']
for s in samples:
    p = os.path.join(base, 'data', 'sample_uploads', s)
    assert os.path.exists(p), f'Missing {s}'
print(f'[5] All {len(samples)} sample CSVs present: OK')

tracks = json.load(open(os.path.join(base, 'data', 'processed', 'simulation_tracks.json')))
assert len(tracks) >= 650
print(f'[6] Simulation track: {len(tracks)} frames ready: OK')

pdf_path = os.path.join(base, 'report', 'RoadSense_Academic_Report.pdf')
assert os.path.exists(pdf_path)
print(f'[7] Academic PDF: {os.path.getsize(pdf_path)/(1024*1024):.2f} MB verified: OK')

nb_path = os.path.join(base, 'notebooks', 'RoadSense_Analysis.ipynb')
assert os.path.exists(nb_path)
print('[8] Academic Jupyter Notebook verified: OK')
print('=== ALL 8 SYSTEM CHECKS PASSED ===')
