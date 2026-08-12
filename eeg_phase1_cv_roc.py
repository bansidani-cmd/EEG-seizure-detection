"""

BIOMECHATRONICS PROJECT - PART 2C: EEG Seizure Detection
        PHASE 1: Cross-Validation, ROC/PR Curves, Model Saving

Upgrades the single train/test split from Step 2 into proper
patient-grouped k-fold cross-validation, producing a mean performance
estimate with variability across folds rather than one single number.
Also adds ROC and precision-recall curves, and saves the final trained
model to disk for reuse in the combined monitoring application.

REQUIREMENTS:
    pip install pandas numpy scipy scikit-learn matplotlib joblib

"""

import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import joblib

CSV_URL = "https://github.com/QiuyiWu/Epileptic-Seizure-Recognition-Data/raw/refs/heads/master/A%26B%26C%26D%26E.csv"
SAMPLING_RATE_HZ = 173.61

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
y = (df['y'] == 1).astype(int)

id_parts = df[id_column].str.extract(r'X(\d+)\.V1\.?(\d+)')
id_parts.columns = ['chunk_number', 'patient_id']
id_parts = id_parts.dropna().astype(int)

valid_rows = id_parts.index
X_raw = X_raw.loc[valid_rows]
y = y.loc[valid_rows]
patient_id = id_parts['patient_id']

print(f"Rows retained: {len(X_raw)}, unique patients: {patient_id.nunique()}")

# 
# STEP 2: FEATURE EXTRACTION (same as Step 2)
# 

def extract_time_domain_features(row):
    values = row.values
    return {
        'td_mean': np.mean(values),
        'td_std': np.std(values),
        'td_variance': np.var(values),
        'td_peak_to_peak': np.max(values) - np.min(values),
        'td_line_length': np.sum(np.abs(np.diff(values))),
        'td_energy': np.sum(values ** 2),
    }

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
        features[f'fd_{band_name}_relative'] = band_power / total_power

    dominant_idx = np.argmax(power_spectrum)
    features['fd_dominant_frequency'] = fft_freqs[dominant_idx]

    return features

print("\nExtracting features from all segments...")
feature_rows = []
for idx, row in X_raw.iterrows():
    td_features = extract_time_domain_features(row)
    fd_features = extract_frequency_domain_features(row)
    feature_rows.append({**td_features, **fd_features})

features_df = pd.DataFrame(feature_rows, index=X_raw.index)
print(f"Feature matrix shape: {features_df.shape}")

# 
# STEP 3: PATIENT-GROUPED K-FOLD CROSS-VALIDATION
# 
# GroupKFold ensures every fold's test set contains only patients not
# present in that fold's training set, extending the single train/test
# split methodology to a full cross-validation procedure. This produces
# five separate performance estimates rather than one, from which a
# mean and standard deviation can be computed.

N_FOLDS = 5
group_kfold = GroupKFold(n_splits=N_FOLDS)

fold_results = []
all_fold_probabilities = np.zeros(len(y))  # collects out-of-fold predictions for ROC/PR curves

print(f"\nRunning {N_FOLDS}-fold patient-grouped cross-validation...")
for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(features_df, y, groups=patient_id)):
    X_train_fold = features_df.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_test_fold = features_df.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]

    fold_clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    fold_clf.fit(X_train_fold, y_train_fold)

    fold_pred = fold_clf.predict(X_test_fold)
    fold_proba = fold_clf.predict_proba(X_test_fold)[:, 1]
    all_fold_probabilities[test_idx] = fold_proba

    report = classification_report(y_test_fold, fold_pred, target_names=['Non-seizure', 'Seizure'], output_dict=True)
    fold_results.append({
        'fold': fold_idx + 1,
        'seizure_precision': report['Seizure']['precision'],
        'seizure_recall': report['Seizure']['recall'],
        'seizure_f1': report['Seizure']['f1-score'],
    })
    print(f"  Fold {fold_idx+1}: precision={report['Seizure']['precision']:.3f}, "
          f"recall={report['Seizure']['recall']:.3f}, f1={report['Seizure']['f1-score']:.3f}")

results_df = pd.DataFrame(fold_results)
print("\n=== Cross-Validation Summary (Seizure class) ===")
print(f"Precision: {results_df['seizure_precision'].mean():.3f} +/- {results_df['seizure_precision'].std():.3f}")
print(f"Recall:    {results_df['seizure_recall'].mean():.3f} +/- {results_df['seizure_recall'].std():.3f}")
print(f"F1:        {results_df['seizure_f1'].mean():.3f} +/- {results_df['seizure_f1'].std():.3f}")

# 
# STEP 4: ROC AND PRECISION-RECALL CURVES
# 
# These curves use out-of-fold predictions collected above, meaning
# every prediction was made by a model that never saw that patient
# during training, preserving the inter-patient evaluation standard
# across the full dataset rather than a single held-out slice.

fpr, tpr, _ = roc_curve(y, all_fold_probabilities)
roc_auc = roc_auc_score(y, all_fold_probabilities)

precision_curve, recall_curve, _ = precision_recall_curve(y, all_fold_probabilities)
avg_precision = average_precision_score(y, all_fold_probabilities)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve - EEG Seizure Detection')
axes[0].legend()

axes[1].plot(recall_curve, precision_curve, color='mediumpurple', label=f'PR curve (AP = {avg_precision:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve - EEG Seizure Detection')
axes[1].legend()

plt.tight_layout()
plt.savefig('eeg_roc_pr_curves.png', dpi=120)
plt.close()
print(f"\nROC AUC: {roc_auc:.3f}")
print(f"Average Precision: {avg_precision:.3f}")
print("Saved eeg_roc_pr_curves.png")

# 
# STEP 5: TRAIN FINAL MODEL ON ALL DATA AND SAVE
# 
# Cross-validation estimates expected performance on new patients.
# The final deployed model is trained on the full dataset, maximizing
# the data available to it, since no further held-out evaluation is
# needed once cross-validated performance has been established.

final_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
final_model.fit(features_df, y)

joblib.dump(final_model, 'eeg_seizure_model.joblib')
joblib.dump(list(features_df.columns), 'eeg_feature_columns.joblib')
print("\nSaved eeg_seizure_model.joblib and eeg_feature_columns.joblib")

print("\nPhase 1 complete.")