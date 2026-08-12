"""

BIOMECHATRONICS PROJECT - PHASE 2, PREP: Export EEG Demo Examples

Selects a small set of example EEG segments (raw waveform, computed
features, and true label) and saves them to disk for the combined
monitoring app to load directly.

REQUIREMENTS:
    pip install pandas numpy scipy joblib

"""

import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
import joblib

CSV_URL = "https://github.com/QiuyiWu/Epileptic-Seizure-Recognition-Data/raw/refs/heads/master/A%26B%26C%26D%26E.csv"
SAMPLING_RATE_HZ = 173.61
EXAMPLES_PER_CLASS = 15

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

X_raw = df.loc[:, 'X1':'X178']
y = (df['y'] == 1).astype(int)

normal_rows = X_raw[y == 0].sample(EXAMPLES_PER_CLASS, random_state=42)
seizure_rows = X_raw[y == 1].sample(EXAMPLES_PER_CLASS, random_state=42)

demo_examples = []
for label, rows in [(0, normal_rows), (1, seizure_rows)]:
    for idx, row in rows.iterrows():
        values = row.values
        td_features = extract_time_domain_features(values)
        fd_features = extract_frequency_domain_features(values)
        demo_examples.append({
            'waveform': values,
            'features': td_features + fd_features,
            'true_label': label,
        })

print(f"\nTotal demo examples collected: {len(demo_examples)}")
print(f"Non-seizure: {sum(1 for e in demo_examples if e['true_label']==0)}")
print(f"Seizure: {sum(1 for e in demo_examples if e['true_label']==1)}")

joblib.dump(demo_examples, 'eeg_demo_examples.joblib')
print("\nSaved eeg_demo_examples.joblib")