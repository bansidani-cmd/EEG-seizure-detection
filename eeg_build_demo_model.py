"""

BIOMECHATRONICS PROJECT - PHASE 2, FIX: Honest EEG Demo Model

Same fix as the ECG version: trains a demo-only model that excludes
a set of demo patients from training entirely, then generates demo
examples from those excluded patients for a genuinely held-out test.

REQUIREMENTS:
    pip install pandas numpy scipy scikit-learn joblib

"""

import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
import joblib

CSV_URL = "https://github.com/QiuyiWu/Epileptic-Seizure-Recognition-Data/raw/refs/heads/master/A%26B%26C%26D%26E.csv"
SAMPLING_RATE_HZ = 173.61
EXAMPLES_PER_CLASS = 15
DEMO_PATIENT_COUNT = 50  # patients held out from training, used only for demo examples

FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 60),
}

def extract_time_domain_features(values):
    return [
        np.mean(values), np.std(values), np.var(values),
        np.max(values) - np.min(values),
        np.sum(np.abs(np.diff(values))),
        np.sum(values ** 2),
    ]

def extract_frequency_domain_features(values, fs=SAMPLING_RATE_HZ):
    n = len(values)
    fft_magnitude = np.abs(rfft(values))
    fft_freqs = rfftfreq(n, d=1.0/fs)
    power_spectrum = fft_magnitude ** 2
    total_power = np.sum(power_spectrum) + 1e-8

    features = []
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        band_mask = (fft_freqs >= low) & (fft_freqs < high)
        band_power = np.sum(power_spectrum[band_mask])
        features.append(band_power)
        features.append(band_power / total_power)

    dominant_freq = fft_freqs[np.argmax(power_spectrum)]
    features.append(dominant_freq)
    return features

print("Downloading Epileptic Seizure Recognition dataset...")
df = pd.read_csv(CSV_URL)
print("Done.")

id_column = df.columns[0]
X_raw = df.loc[:, 'X1':'X178']
y = (df['y'] == 1).astype(int)

id_parts = df[id_column].str.extract(r'X(\d+)\.V1\.?(\d+)')
id_parts.columns = ['chunk_number', 'patient_id']
id_parts = id_parts.dropna().astype(int)

valid_rows = id_parts.index
X_raw = X_raw.loc[valid_rows]
y = y.loc[valid_rows]
patient_id = id_parts['patient_id']

print("Extracting features from all segments...")
feature_rows = []
for idx, row in X_raw.iterrows():
    values = row.values
    td_features = extract_time_domain_features(values)
    fd_features = extract_frequency_domain_features(values)
    feature_rows.append(td_features + fd_features)

features_array = np.array(feature_rows)

# 
# SPLIT INTO DEMO PATIENTS (HELD OUT) AND TRAINING PATIENTS
# 

unique_patients = patient_id.unique()
rng = np.random.default_rng(42)
shuffled_patients = rng.permutation(unique_patients)
demo_patient_ids = set(shuffled_patients[:DEMO_PATIENT_COUNT])
training_patient_ids = set(shuffled_patients[DEMO_PATIENT_COUNT:])

demo_mask = patient_id.isin(demo_patient_ids).values
training_mask = patient_id.isin(training_patient_ids).values

X_train_demo = features_array[training_mask]
y_train_demo = y.values[training_mask]

print(f"\nDemo model training rows: {len(y_train_demo)} (from {len(training_patient_ids)} patients)")
print(f"Held-out demo pool rows: {demo_mask.sum()} (from {len(demo_patient_ids)} patients)")

# 
# TRAIN THE DEMO MODEL, EXCLUDING DEMO PATIENTS
# 

demo_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
demo_model.fit(X_train_demo, y_train_demo)
joblib.dump(demo_model, 'eeg_demo_model.joblib')
import os
size_mb = os.path.getsize('eeg_demo_model.joblib') / (1024 * 1024)
print(f"Saved eeg_demo_model.joblib ({size_mb:.1f} MB)")

# 
# CURATE A DELIBERATE MIX ACROSS THE CONFIDENCE RANGE
# 

demo_indices = np.where(demo_mask)[0]
demo_features = features_array[demo_indices]
demo_true = y.values[demo_indices]
demo_proba = demo_model.predict_proba(demo_features)[:, 1]
demo_pred = (demo_proba >= 0.5).astype(int)

confident_correct_normal = [i for i in range(len(demo_indices))
                             if demo_true[i] == 0 and demo_proba[i] < 0.1]
confident_correct_seizure = [i for i in range(len(demo_indices))
                              if demo_true[i] == 1 and demo_proba[i] > 0.9]
misclassified = [i for i in range(len(demo_indices)) if demo_pred[i] != demo_true[i]]
borderline = [i for i in range(len(demo_indices)) if 0.35 <= demo_proba[i] <= 0.65]

rng3 = np.random.default_rng(42)
def sample_indices(pool, count):
    if len(pool) <= count:
        return pool
    return list(rng3.choice(pool, size=count, replace=False))

selected = set()
selected.update(sample_indices(confident_correct_normal, 8))
selected.update(sample_indices(confident_correct_seizure, 8))
selected.update(sample_indices(misclassified, 8))
selected.update(sample_indices(borderline, 6))

print(f"\nConfident correct (non-seizure): {len(confident_correct_normal)} available")
print(f"Confident correct (seizure): {len(confident_correct_seizure)} available")
print(f"Misclassified: {len(misclassified)} available")
print(f"Borderline (0.35-0.65): {len(borderline)} available")
print(f"Total curated examples selected: {len(selected)}")

demo_examples = []
for local_i in sorted(selected):
    global_idx = demo_indices[local_i]
    demo_examples.append({
        'waveform': X_raw.iloc[global_idx].values,
        'features': list(features_array[global_idx]),
        'true_label': int(y.values[global_idx]),
    })

print(f"\nTotal demo examples collected: {len(demo_examples)}")

predictions = demo_model.predict(np.array([e['features'] for e in demo_examples]))
true_labels = np.array([e['true_label'] for e in demo_examples])
accuracy = np.mean(predictions == true_labels)
print(f"Demo model accuracy on held-out demo examples: {accuracy:.3f}")
print(f"(A value meaningfully below 1.0 confirms this is a genuine held-out test.)")

joblib.dump(demo_examples, 'eeg_demo_examples.joblib')
print("\nSaved eeg_demo_examples.joblib")