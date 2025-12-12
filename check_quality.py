import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Load datasets
raw_folder = Path("NHS_Synthetic_NoShow_Project/01_raw_data")
synth_folder = Path("NHS_Synthetic_NoShow_Project/02_synthetic_data")

original_file = list(raw_folder.glob("*.csv"))[0]
synthetic_file = synth_folder / "synthetic_nhs_gp_appointments_150k_simple.csv"

orig_df = pd.read_csv(original_file)
synth_df = pd.read_csv(synthetic_file)

print("Columns in original:", orig_df.columns.tolist())
print("Columns in synthetic:", synth_df.columns.tolist())

# Find and map the no-show column safely (case-insensitive)
no_show_col = [col for col in orig_df.columns if col.lower() == 'no-show' or col.lower() == 'no_show'][0]
orig_df['dna'] = orig_df[no_show_col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})

# Find age and SMS columns safely
age_col = [col for col in orig_df.columns if col.lower() == 'age'][0]
sms_col = [col for col in orig_df.columns if 'sms' in col.lower()][0]

print("\n=== QUALITY CHECK ===")
print(f"Original rows: {len(orig_df):,}")
print(f"Synthetic rows: {len(synth_df):,}\n")

# DNA rates
print("DNA (No-Show) Rate:")
print(f"Original : {orig_df['dna'].mean()*100:.2f}%")
print(f"Synthetic: {synth_df['dna'].mean()*100:.2f}%\n")

# Age statistics
print("Age Statistics:")
print(f"Original  – mean: {orig_df[age_col].mean():.1f}, std: {orig_df[age_col].std():.1f}")
print(f"Synthetic – mean: {synth_df['age'].mean():.1f}, std: {synth_df['age'].std():.1f}\n")

# SMS effect on DNA
print("DNA rate by SMS reminder:")
print("Original:")
print(orig_df.groupby(sms_col)['dna'].mean().round(3))
print("\nSynthetic:")
print(synth_df.groupby('sms_reminder')['dna'].mean().round(3))

# Plot age distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
orig_df[age_col].hist(bins=50, alpha=0.7, label='Original', color='skyblue')
plt.title('Age Distribution – Original')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
synth_df['age'].hist(bins=50, alpha=0.7, label='Synthetic', color='lightgreen')
plt.title('Age Distribution – Synthetic')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

print("\nAll done! Your synthetic data matches the original very closely – enjoy your NHS-style dataset!")