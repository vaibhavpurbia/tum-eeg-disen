import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("stratified_cohort.csv")
df_bis = df[df['bis'] > 5.0]

fig = plt.figure(figsize=(14, 10))

# 1. MAC vs DisEn (30 Hz) with Trendline
ax1 = plt.subplot(2, 2, 1)
ax1.scatter(df['mac'], df['disen_30'], color='#1f77b4', edgecolors='k', s=70, alpha=0.85, label='DisEn (30 Hz)')
m, b = np.polyfit(df['mac'], df['disen_30'], 1)
ax1.plot(df['mac'], m*df['mac'] + b, color='#1f77b4', linestyle='--', label=f'Fit (rho = -0.496, p = 0.012)')
ax1.set_title("A. Volatile Dose vs. Dispersion Entropy", fontweight='bold')
ax1.set_xlabel("Volatile Concentration (MAC)")
ax1.set_ylabel("Normalized DisEn (30 Hz)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. BIS vs DisEn Correlation
ax2 = plt.subplot(2, 2, 2)
ax2.scatter(df_bis['bis'], df_bis['disen_30'], color='#2ca02c', edgecolors='k', s=70, alpha=0.85)
m_b, b_b = np.polyfit(df_bis['bis'], df_bis['disen_30'], 1)
ax2.plot(df_bis['bis'], m_b*df_bis['bis'] + b_b, color='#2ca02c', linestyle='--')
ax2.set_title("B. DisEn (30 Hz) vs. Commercial BIS Index", fontweight='bold')
ax2.set_xlabel("BIS Index (Clean Recordings)")
ax2.set_ylabel("Normalized DisEn (30 Hz)")
ax2.grid(True, alpha=0.3)

# 3. Filter Sensitivity Offset across Tiers
ax3 = plt.subplot(2, 2, 3)
tier_order = ['light', 'medium', 'deep']
offsets = [df[df['tier'] == t]['delta_disen'].values for t in tier_order]
bp = ax3.boxplot(offsets, patch_artist=True, tick_labels=['Light\n(0.3-0.6 MAC)', 'Medium\n(0.7-1.1 MAC)', 'Deep\n(>=1.2 MAC)'])
colors = ['#aec7e8', '#ffbb78', '#98df8a']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax3.axhline(0.05, color='red', linestyle=':', label='EMG Sensitivity Threshold (0.05)')
ax3.set_title(r"C. Filter Sensitivity Offset ($\Delta$DisEn: 40Hz - 30Hz)", fontweight='bold')
ax3.set_ylabel(r"$\Delta$DisEn")
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Metric Prediction Probability (P_K) Comparison
ax4 = plt.subplot(2, 2, 4)
features = ['DisEn (40 Hz)', 'DisEn (30 Hz)', 'Commercial BIS', 'Rel Alpha', 'Rel Delta', 'Beta']
pk_vals = [0.701, 0.697, 0.697, 0.669, 0.479, 0.398]
bar_colors = ['#1f77b4', '#1f77b4', '#2ca02c', '#ff7f0e', '#7f7f7f', '#9467bd']
ax4.barh(features[::-1], pk_vals[::-1], color=bar_colors[::-1], edgecolor='black', alpha=0.85)
ax4.axvline(0.5, color='red', linestyle='--', label='Chance Level (0.50)')
ax4.set_xlim(0.3, 0.8)
ax4.set_xlabel("Prediction Probability ($P_K$) vs. MAC")
ax4.set_title("D. Depth Discrimination Performance ($P_K$)", fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='lower right')

plt.tight_layout()
plt.savefig("final_project_benchmarks.png", dpi=300)
print("Saved final publication figure to 'final_project_benchmarks.png'.")