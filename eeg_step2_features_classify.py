"""

BIOMECHATRONICS PROJECT - PART 2B: EEG Seizure Detection
              STEP 2: Feature Extraction and Classification

Builds on Step 1. Extracts two categories of features per 1-second
EEG segment:

  TIME-DOMAIN features: describe the signal's shape and amplitude
  directly (mean, variance, peak-to-peak range, line length).

  FREQUENCY-DOMAIN features: describe how much signal energy falls
  into each of the standard EEG frequency bands (delta, theta, alpha,
  beta, gamma), computed via the Fast Fourier Transform (FFT). This
  targets the rhythmic, narrow-band oscillation pattern characteristic
  of seizure activity.

Evaluation uses a patient-level train/test split (no patient appears
in both sets), the same methodology applied throughout the ECG project.

REQUIREMENTS:
    pip install pandas numpy scipy scikit-learn matplotlib

"""

import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

CSV_URL = "https://github.com/QiuyiWu/Epileptic-Seizure-Recognition-Data/raw/refs/heads/master/A%26B%26C%26D%26E.csv"
SAMPLING_RATE_HZ = 173.61  # original Bonn dataset sampling rate

# Standard EEG frequency bands, in Hz
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 60),
}

# 
# STEP 1: LOAD DATA AND RECOVER PATIENT GROUPING
# 

print("Downloading Epileptic Seizure Recognition dataset...")
df = pd.read_csv(CSV_URL)
print("Done.")

id_column = df.columns[0]
X_raw = df.loc[:, 'X1':'X178']
y = (df['y'] == 1).astype(int)  # 1 = seizure, 0 = non-seizure

id_parts = df[id_column].str.extract(r'X(\d+)\.V1\.?(\d+)')
id_parts.columns = ['chunk_number', 'patient_id']
id_parts = id_parts.dropna().astype(int)

valid_rows = id_parts.index
X_raw = X_raw.loc[valid_rows]
y = y.loc[valid_rows]
patient_id = id_parts['patient_id']

print(f"Rows retained after excluding unparseable identifiers: {len(X_raw)}")
print(f"Unique patients: {patient_id.nunique()}")

# 
# STEP 2: TIME-DOMAIN FEATURES
# 

def extract_time_domain_features(row):
    values = row.values
    return {
        'td_mean': np.mean(values),
        'td_std': np.std(values),
        'td_variance': np.var(values),
        'td_peak_to_peak': np.max(values) - np.min(values),
        'td_line_length': np.sum(np.abs(np.diff(values))),  # a measure of signal complexity/roughness
        'td_energy': np.sum(values ** 2),
    }

# 
# STEP 3: FREQUENCY-DOMAIN FEATURES
# 
# The FFT converts a signal from "value over time" into "amount of
# energy at each frequency". Summing that energy within each standard
# EEG frequency band produces one feature per band, directly targeting
# the rhythmic, narrow-band pattern seen in seizure segments.

def extract_frequency_domain_features(row, fs=SAMPLING_RATE_HZ):
    values = row.values
    n = len(values)
    fft_magnitude = np.abs(rfft(values))
    fft_freqs = rfftfreq(n, d=1.0/fs)
    power_spectrum = fft_magnitude ** 2

    features = {}
    total_power = np.sum(power_spectrum) + 1e-8
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        band_mask = (fft_freqs >= low) & (fft_freqs < high)
        band_power = np.sum(power_spectrum[band_mask])
        features[f'fd_{band_name}_power'] = band_power
        features[f'fd_{band_name}_relative'] = band_power / total_power  # proportion of total energy in this band

    # Dominant frequency: the single frequency with the most energy,
    # a direct indicator of rhythmic, narrow-band activity.
    dominant_idx = np.argmax(power_spectrum)
    features['fd_dominant_frequency'] = fft_freqs[dominant_idx]

    return features

# 
# STEP 4: BUILD THE FULL FEATURE MATRIX
# 

print("\nExtracting features from all segments...")
feature_rows = []
for idx, row in X_raw.iterrows():
    td_features = extract_time_domain_features(row)
    fd_features = extract_frequency_domain_features(row)
    feature_rows.append({**td_features, **fd_features})

features_df = pd.DataFrame(feature_rows, index=X_raw.index)
print(f"Feature matrix shape: {features_df.shape}")
print(f"Feature columns: {list(features_df.columns)}")

# 
# STEP 5: PATIENT-LEVEL TRAIN/TEST SPLIT
# 
# Patients are split into two disjoint groups before any rows are
# assigned to train or test, ensuring no patient's segments appear in
# both sets.

unique_patients = patient_id.unique()
rng = np.random.default_rng(42)
shuffled_patients = rng.permutation(unique_patients)

split_point = int(0.8 * len(shuffled_patients))
train_patients = set(shuffled_patients[:split_point])
test_patients = set(shuffled_patients[split_point:])

train_mask = patient_id.isin(train_patients)
test_mask = patient_id.isin(test_patients)

overlap = train_patients & test_patients
print(f"\nPatient overlap between train and test sets: {len(overlap)}")
assert len(overlap) == 0, "Patient overlap detected. Split logic requires review."

X_train = features_df[train_mask]
y_train = y[train_mask]
X_test = features_df[test_mask]
y_test = y[test_mask]

print(f"\nTraining patients: {len(train_patients)}, training rows: {len(X_train)}")
print(f"Test patients: {len(test_patients)}, test rows: {len(X_test)}")
print(f"Training seizure rows: {y_train.sum()}, test seizure rows: {y_test.sum()}")

# 
# STEP 6: TRAIN
# 

clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 
# STEP 7: EVALUATE
# 

y_pred = clf.predict(X_test)

print("\n=== Inter-Patient Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Non-seizure', 'Seizure']))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-seizure', 'Seizure'])
disp.plot(cmap='Purples')
plt.title("Confusion Matrix - EEG Seizure Detection")
plt.tight_layout()
plt.savefig('eeg_confusion_matrix.png', dpi=120)
plt.close()
print("Saved eeg_confusion_matrix.png")

importances = clf.feature_importances_
feature_names = features_df.columns
importance_ranking = sorted(zip(feature_names, importances), key=lambda x: -x[1])

print("\n=== Feature Importances (top 10) ===")
for name, imp in importance_ranking[:10]:
    print(f"  {name:20s}: {imp:.4f}")

plt.figure(figsize=(10, 6))
top_features = importance_ranking[:10]
plt.barh([f[0] for f in top_features][::-1], [f[1] for f in top_features][::-1], color='mediumpurple')
plt.xlabel("Feature Importance")
plt.title("Top 10 Feature Importances - EEG Seizure Detection")
plt.tight_layout()
plt.savefig('eeg_feature_importance.png', dpi=120)
plt.close()
print("Saved eeg_feature_importance.png")

print("\nStep 2 complete.")