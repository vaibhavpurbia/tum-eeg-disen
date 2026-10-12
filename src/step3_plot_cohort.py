import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cohort_features.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: MAC vs DisEn (30 Hz vs 40 Hz)
axes[0].scatter(df['mac'], df['disen_30'], color='#1f77b4', s=60, label='30 Hz Cutoff', alpha=0.8)
axes[0].scatter(df['mac'], df['disen_40'], color='#ff7f0e', s=60, marker='^', label='40 Hz Cutoff', alpha=0.8)
for _, row in df.iterrows():
    axes[0].plot([row['mac'], row['mac']], [row['disen_30'], row['disen_40']], color='gray', linestyle=':', alpha=0.6)
axes[0].set_title("MAC vs. Dispersion Entropy (N=15)")
axes[0].set_xlabel("Volatile Agent (MAC)")
axes[0].set_ylabel("Normalized DisEn")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Filter Sensitivity Offset (Delta DisEn)
axes[1].bar(range(len(df)), df['delta_disen'], color='#2ca02c', alpha=0.75, edgecolor='black')
axes[1].set_title(r"Filter Sensitivity Offset ($\Delta$DisEn: 40Hz - 30Hz)")
axes[1].set_xlabel("Cohort Episode Index")
axes[1].set_ylabel(r"$\Delta$DisEn")
axes[1].axhline(0.05, color='red', linestyle='--', label='EMG Contamination Suspect (>0.05)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot 3: DisEn vs Spectral Exponent (Beta)
axes[2].scatter(df['beta_30'], df['disen_30'], color='#9467bd', s=60, alpha=0.8)
axes[2].set_title(r"DisEn vs. Spectral Exponent ($\beta$) [30 Hz]")
axes[2].set_xlabel(r"Spectral Exponent ($\beta$)")
axes[2].set_ylabel("Dispersion Entropy")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cohort_benchmarks.png", dpi=300)
print("Saved 'cohort_benchmarks.png'.")