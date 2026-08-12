"""
BIOMECHATRONICS PROJECT - PART 2A: EEG Seizure Detection
                        STEP 1: Load and Visualize

Dataset: Epileptic Seizure Recognition (UCI Machine Learning Repository,
derived from the Bonn University EEG dataset)

Structure: 11,500 rows. Each row is one 1-second EEG segment, containing
178 raw voltage readings (columns) plus a class label (1-5).
  Class 1 = seizure activity
  Classes 2-5 = various non-seizure states (different brain regions/conditions)

Labels are binarized: 1 = seizure, 0 = everything else.

REQUIREMENTS:
    pip install pandas numpy matplotlib

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 
# STEP 1: DOWNLOAD THE DATASET
# 
# Pull the CSV directly from a GitHub mirror of the same official
# dataset, maintained by one of the dataset's original creators.

CSV_URL = "https://github.com/QiuyiWu/Epileptic-Seizure-Recognition-Data/raw/refs/heads/master/A%26B%26C%26D%26E.csv"

print("Downloading Epileptic Seizure Recognition dataset...")
df = pd.read_csv(CSV_URL)
print("Done.")

print(f"\nFull dataframe shape: {df.shape}")
print(f"Column names (first 5): {list(df.columns[:5])} ...")
print(f"Column names (last 3): {list(df.columns[-3:])}")

# The first column is a row identifier, e.g. 'X21.V1.791'.
# This will be parsed below to recover patient grouping for a later
# inter-patient train/test split.
id_column = df.columns[0]
print(f"\nFirst column name: '{id_column}'")
print(f"Example identifiers:\n{df[id_column].head(10).to_list()}")

# Columns X1...X178 are the raw EEG readings; 'y' is the class label.
X = df.loc[:, 'X1':'X178']
y = df['y']

print(f"\nX shape: {X.shape}  (rows = 1-second EEG segments, columns = raw voltage readings)")
print(f"y shape: {y.shape}")
print(f"\nClass distribution:\n{y.value_counts().sort_index()}")

# 
# STEP 2: BINARIZE THE LABELS
# 
# Classes 2-5 (different non-seizure brain states) collapse into a
# single "Normal" category, leaving a binary seizure vs. non-seizure task.

y_binary = (y == 1).astype(int)  # 1 = seizure, 0 = non-seizure
print(f"\nBinary class distribution:")
print(f"  Non-seizure (0): {(y_binary==0).sum()}")
print(f"  Seizure (1):     {(y_binary==1).sum()}")

# 
# STEP 2B: RECOVERING PATIENT GROUPING FROM THE ID COLUMN
# 
# The identifier contains two embedded numbers, e.g. 'X21.V1.791'.
# The dataset description states there were 500 original subjects,
# each contributing 23 one-second chunks. Splitting the identifier
# apart and counting unique values per component identifies which
# number corresponds to which quantity.

id_parts = df[id_column].str.extract(r'X(\d+)\.V1\.?(\d+)')
id_parts.columns = ['part_a', 'part_b']

unmatched = df[id_column][id_parts['part_a'].isna()]
if len(unmatched) > 0:
    print(f"\n{len(unmatched)} identifiers did not match the expected pattern. Examples:")
    print(unmatched.head(10).to_list())
    print("These rows will be excluded from patient grouping.")
else:
    print("\nAll identifiers matched the expected pattern.")

id_parts = id_parts.dropna().astype(int)

print(f"\n'part_a' (first number) - unique values: {id_parts['part_a'].nunique()}, range: {id_parts['part_a'].min()}-{id_parts['part_a'].max()}")
print(f"'part_b' (second number) - unique values: {id_parts['part_b'].nunique()}, range: {id_parts['part_b'].min()}-{id_parts['part_b'].max()}")

print("\npart_a (23 unique values) corresponds to chunk number within a recording.")
print("part_b (~495-500 unique values) corresponds to the original subject/patient identifier.")
print("part_b will be used as the patient grouping column for the inter-patient split.")

# Attached via pd.concat rather than two separate column assignments,
# which avoids DataFrame fragmentation warnings on large frames.
grouping_columns = pd.DataFrame({
    'patient_id': id_parts['part_b'],
    'chunk_number': id_parts['part_a'],
})
df = pd.concat([df, grouping_columns], axis=1)
print(f"\nRows successfully assigned a patient_id: {df['patient_id'].notna().sum()} / {len(df)}")

# 
# STEP 3: COMPARE SEIZURE VS NON-SEIZURE SIGNALS
# 
# A key sanity check before any modeling: does a seizure segment
# actually look different from a non-seizure one?

seizure_example = X[y_binary == 1].iloc[0].values
normal_example = X[y_binary == 0].iloc[0].values

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(normal_example, color='steelblue')
axes[0].set_title("Example: Non-seizure EEG segment (1 second)")
axes[0].set_ylabel("Voltage")

axes[1].plot(seizure_example, color='crimson')
axes[1].set_title("Example: Seizure EEG segment (1 second)")
axes[1].set_ylabel("Voltage")
axes[1].set_xlabel("Sample (178 samples = 1 second)")

plt.tight_layout()
plt.savefig('eeg_example_signals.png', dpi=120)
plt.close()
print("\nSaved eeg_example_signals.png")

# 
# STEP 4: MULTIPLE EXAMPLES PER CLASS
# 
# Checking variety within each class, not just a single example pair.

fig, axes = plt.subplots(4, 2, figsize=(14, 10))
normal_samples = X[y_binary == 0].sample(4, random_state=42).values
seizure_samples = X[y_binary == 1].sample(4, random_state=42).values

for i in range(4):
    axes[i, 0].plot(normal_samples[i], color='steelblue')
    axes[i, 0].set_title(f"Non-seizure example {i+1}")
    axes[i, 1].plot(seizure_samples[i], color='crimson')
    axes[i, 1].set_title(f"Seizure example {i+1}")

plt.tight_layout()
plt.savefig('eeg_multiple_examples.png', dpi=120)
plt.close()
print("Saved eeg_multiple_examples.png")

print("\nStep 1 complete. Review both PNG files before proceeding:")
print("  - Do seizure segments look visually distinct from non-seizure ones?")
print("  - Is there variety within each class, or do they all look similar?")
print("This informs which features are worth extracting next.")