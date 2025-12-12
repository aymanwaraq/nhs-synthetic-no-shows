# ──────────────────────────────────────────────────────────────
#  SUPER SIMPLE & FAST: Synthetic NHS-style GP Appointments (No SDV)
# ──────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load the original Kaggle data
raw_folder = Path("NHS_Synthetic_NoShow_Project/01_raw_data")
csv_file = list(raw_folder.glob("*.csv"))[0]
print(f"Loading: {csv_file.name}")

df = pd.read_csv(csv_file)

# 2. Basic cleaning & NHS-style mapping
df.columns = df.columns.str.lower()
df.rename(columns={
    'patientid': 'pseudo_nhs_number',
    'appointmentid': 'appointment_id',
    'no-show': 'dna',  # Did Not Attend
    'scheduledday': 'scheduled_day',
    'appointmentday': 'appointment_day',
    'neighbourhood': 'practice_area',
    'scholarship': 'on_benefits',
    'hipertension': 'hypertension',
    'handcap': 'handicap'
}, inplace=True)

df['scheduled_day'] = pd.to_datetime(df['scheduled_day']).dt.date
df['appointment_day'] = pd.to_datetime(df['appointment_day']).dt.date
df['wait_days'] = (pd.to_datetime(df['appointment_day']) - pd.to_datetime(df['scheduled_day'])).dt.days
df['wait_days'] = df['wait_days'].clip(0, 90)

df['gender'] = df['gender'].map({'F': 'Female', 'M': 'Male'})
df['dna'] = df['dna'].map({'Yes': 1, 'No': 0})
df['sms_reminder'] = df['sms_received']

# Add NHS-like columns
np.random.seed(42)
df['imd_quintile'] = np.random.choice([1,2,3,4,5], size=len(df), p=[0.28, 0.24, 0.20, 0.16, 0.12])
df['appointment_mode'] = np.random.choice(['Face-to-Face', 'Telephone', 'Video/Online'],
                                          size=len(df), p=[0.62, 0.32, 0.06])
df['hcp_type'] = np.random.choice(['GP', 'Nurse', 'Healthcare Assistant', 'Other'],
                                  size=len(df), p=[0.55, 0.30, 0.10, 0.05])

final_columns = ['pseudo_nhs_number', 'appointment_id', 'gender', 'age', 'imd_quintile',
                 'hypertension', 'diabetes', 'alcoholism', 'handicap',
                 'on_benefits', 'sms_reminder', 'wait_days',
                 'appointment_mode', 'hcp_type', 'scheduled_day',
                 'appointment_day', 'dna']

df = df[final_columns]

print(f"Prepared real data → {len(df):,} rows")

# 3. Generate 150,000 synthetic rows (bootstrapping + small noise)
print("\nGenerating 150,000 synthetic rows (fast & realistic)...")

# Bootstrap sample (preserves distributions & correlations)
synthetic_df = df.sample(n=150_000, replace=True, random_state=42)

# Add small realistic noise to numerical columns (age, wait_days)
synthetic_df['age'] = np.clip(synthetic_df['age'] + np.random.normal(0, 2, size=len(synthetic_df)), 0, 105).astype(int)
synthetic_df['wait_days'] = np.clip(synthetic_df['wait_days'] + np.random.normal(0, 3, size=len(synthetic_df)), 0, 60).astype(int)

# Randomise IDs to make fully synthetic
synthetic_df['pseudo_nhs_number'] = np.random.randint(100000000, 999999999, size=len(synthetic_df))  # 9-digit safe numbers
synthetic_df['appointment_id'] = np.arange(1, len(synthetic_df) + 1, dtype='int64')
# 4. Save
output_folder = Path("NHS_Synthetic_NoShow_Project/02_synthetic_data")
output_folder.mkdir(exist_ok=True)
output_file = output_folder / "synthetic_nhs_gp_appointments_150k_simple.csv"
synthetic_df.to_csv(output_file, index=False)

print("\nSUCCESS!")
print(f"Synthetic NHS-style data saved → {output_file}")
print(f"Total rows: {len(synthetic_df):,}")
print("Perfect for no-show/DNA prediction – realistic distributions preserved!")
synthetic_df['appointment_id'] = np.arange(1, len(synthetic_df) + 1, dtype='int64')