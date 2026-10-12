import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def compute_pk(x, y):
    """
    Computes Smith's Prediction Probability (P_K).
    x: anesthetic depth indicator / drug concentration (e.g. MAC)
    y: evaluated metric (e.g. DisEn, BIS, Beta)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    
    concordant = 0
    discordant = 0
    tied_y_only = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Ignore exact ties in MAC (condition x)
            if x[i] == x[j]:
                continue
            
            # Since depth metrics should fall as MAC rises:
            # If x[i] > x[j], we expect y[i] < y[j]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            
            # Monotonic inverse relationship expected
            if (dx > 0 and dy < 0) or (dx < 0 and dy > 0):
                concordant += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                discordant += 1
            else:
                tied_y_only += 1
                
    total_pairs = concordant + discordant + tied_y_only
    if total_pairs == 0:
        return 0.5
    return (concordant + 0.5 * tied_y_only) / total_pairs

# Load the stratified cohort
df = pd.read_csv("stratified_cohort.csv")

# Exclude unrecorded/zero BIS artifacts for direct BIS comparison
df_bis_clean = df[df['bis'] > 5.0]

print("=" * 65)
print(f"BENCHMARK RESULTS (N = {len(df)} Stratified Epochs)")
print("=" * 65)

metrics = {
    'DisEn (0.5-30 Hz)': df['disen_30'],
    'DisEn (0.5-40 Hz)': df['disen_40'],
    'Spectral Exponent (Beta)': df['beta_30'],
    'Relative Delta Power': df['rel_delta_30'],
    'Relative Alpha Power': df['rel_alpha_30'],
}

summary = []

for name, series in metrics.items():
    pk = compute_pk(df['mac'], series)
    rho, pval = spearmanr(df['mac'], series)
    summary.append({
        'Feature': name,
        'P_K vs MAC': round(pk, 3),
        'Spearman Rho': round(rho, 3),
        'p-value': f"{pval:.4e}"
    })

# Compute BIS performance specifically on clean BIS epochs
pk_bis = compute_pk(df_bis_clean['mac'], df_bis_clean['bis'])
rho_bis, p_bis = spearmanr(df_bis_clean['mac'], df_bis_clean['bis'])
summary.append({
    'Feature': 'Commercial BIS (clean)',
    'P_K vs MAC': round(pk_bis, 3),
    'Spearman Rho': round(rho_bis, 3),
    'p-value': f"{p_bis:.4e}"
})

df_bench = pd.DataFrame(summary)
print(df_bench.to_string(index=False))
print("=" * 65)

# Visual Summary: P_K and Rho comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.barh(df_bench['Feature'], df_bench['P_K vs MAC'], color='#1f77b4', edgecolor='black')
plt.axvline(0.5, color='red', linestyle='--', label='Random Chance (0.5)')
plt.xlim(0.3, 1.0)
plt.xlabel("Prediction Probability (P_K)")
plt.title("Depth Tracking Accuracy (P_K vs. MAC)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
colors = ['#d62728' if r > 0 else '#2ca02c' for r in df_bench['Spearman Rho']]
plt.barh(df_bench['Feature'], df_bench['Spearman Rho'], color=colors, edgecolor='black')
plt.axvline(0.0, color='black', linewidth=1)
plt.xlabel("Spearman Rank Correlation (Rho)")
plt.title("Correlation with Volatile Concentration")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("benchmark_comparison.png", dpi=300)
print("Generated comparison figure: 'benchmark_comparison.png'")