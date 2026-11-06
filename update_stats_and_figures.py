import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from benchmark_disen_vs_pe import (
    apply_bandpass, apply_notch, compute_dispen, compute_peen,
    compute_sampen, compute_sef95, bootstrap_pk
)

df = pd.read_csv("data/stratified_cohort_fully_enriched.csv")
manifest = pd.read_csv("data/raw_reproducible_epochs/manifest.csv")

epochs_30, epochs_40 = [], []
for fp in manifest['file_path']:
    raw = np.load(fp)
    epochs_30.append(apply_notch(apply_bandpass(raw, 128, 0.5, 30.0), 128))
    epochs_40.append(apply_notch(apply_bandpass(raw, 128, 0.5, 40.0), 128))

# Time execution speeds per 20s epoch
t0 = time.perf_counter()
for _ in range(100):
    _ = [compute_dispen(s, 3, 4, 1) for s in epochs_30]
ms_disp = ((time.perf_counter() - t0) / (100 * len(epochs_30))) * 1000.0

t0 = time.perf_counter()
for _ in range(100):
    _ = [compute_peen(s, 3, 1) for s in epochs_30]
ms_peen = ((time.perf_counter() - t0) / (100 * len(epochs_30))) * 1000.0

t0 = time.perf_counter()
samp_40 = np.array([compute_sampen(s, 2, 0.2) for s in epochs_40])
ms_samp = ((time.perf_counter() - t0) / len(epochs_40)) * 1000.0

sef_40 = np.array([compute_sef95(s, 128.0, 40.0) for s in epochs_40])
df['samp_40'] = samp_40
df['sef95_40'] = sef_40
df.to_csv("data/stratified_cohort_fully_enriched.csv", index=False)

mac = df['mac'].values
valid_bis = df['bis'].notna() & (df['bis'] > 0)
bis_v = df.loc[valid_bis, 'bis'].values

print("\n=== ADDITIONAL METRICS FOR MANUSCRIPT TABLE (N = 23, MAC <= 2.0) ===")
for name, col30, col40 in [
    ("DispEn (m=3,c=4,d=1)", "disp_30", "disen_40"),
    ("SampEn (m=2,r=0.2)",   "samp_30", "samp_40"),
    ("PeEn   (m=3,d=1)",     "peen_30", "peen_40"),
    ("SEF95  (0.5-30 Hz)",   "sef95_30", "sef95_40"),
    ("Spectral Exp (Beta)",  "beta_30", "beta_40"),
    ("Rel Delta (0.5-4Hz)",  "rel_delta_30", None),
    ("Rel Alpha (8-12Hz)",   "rel_alpha_30", None),
]:
    v30 = df[col30].values
    pk, cil, ciu = bootstrap_pk(v30, mac)
    rho_m, p_m = spearmanr(mac, v30)
    rho_b, p_b = spearmanr(bis_v, v30[valid_bis])
    emg_str = f"{np.mean(df[col40].values - v30):+.4f}" if col40 and col40 in df else "N/A"
    print(f"{name:<21} | P_K={pk:.3f} [{cil:.3f}-{ciu:.3f}] | r_MAC={rho_m:+.3f} (p={p_m:.4f}) | r_BIS={rho_b:+.3f} (p={p_b:.4f}) | EMG={emg_str}")

print(f"\nExecution Speeds per 20s epoch: DispEn = {ms_disp:.3f} ms | PeEn = {ms_peen:.3f} ms | SampEn = {ms_samp:.1f} ms ({ms_samp/ms_disp:.0f}x slower)")

# Generate Updated 3-Panel Figure for Overleaf
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
x_line = np.linspace(min(mac), max(mac), 100)

# Panel A: DispEn vs MAC
axes[0].scatter(mac, df['disp_30'], c='#1f77b4', s=55, alpha=0.85, edgecolors='k', label='DispEn (30 Hz)')
m1, b1 = np.polyfit(mac, df['disp_30'], 1)
axes[0].plot(x_line, m1 * x_line + b1, 'r--', lw=2, label='Fit (r=-0.457, p=0.0285)')
axes[0].set_title("A. Dispersion Entropy vs. MAC\n(P_K = 0.685 [0.520-0.842], p = 0.0285)", fontweight='bold')
axes[0].set_xlabel("Volatile Anesthetic Depth (MAC <= 2.0)")
axes[0].set_ylabel("Normalized DispEn (m=3, c=4, d=1)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Panel B: PeEn vs MAC
axes[1].scatter(mac, df['peen_30'], c='#ff7f0e', s=55, alpha=0.85, edgecolors='k', label='PeEn (30 Hz)')
m2, b2 = np.polyfit(mac, df['peen_30'], 1)
axes[1].plot(x_line, m2 * x_line + b2, 'k--', lw=2, label='Fit (r=-0.326, p=0.1291, NS)')
axes[1].set_title("B. Permutation Entropy vs. MAC\n(P_K = 0.633 [0.507-0.785], p = 0.1291)", fontweight='bold')
axes[1].set_xlabel("Volatile Anesthetic Depth (MAC <= 2.0)")
axes[1].set_ylabel("Normalized PeEn (m=3, d=1)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Panel C: P_K with 95% CI Error Bars across Metrics
labels = ['Commercial\nBIS', 'SampEn\n(m=2)', 'DispEn\n(m=3, c=4)', 'PeEn\n(m=3)']
pks = [0.720, 0.692, 0.685, 0.633]
yerr_low = [0.720 - 0.536, 0.692 - 0.515, 0.685 - 0.520, 0.633 - 0.507]
yerr_high = [0.877 - 0.720, 0.850 - 0.692, 0.842 - 0.685, 0.785 - 0.633]
colors = ['#7f7f7f', '#2ca02c', '#1f77b4', '#ff7f0e']

x_pos = np.arange(len(labels))
axes[2].bar(x_pos, pks, yerr=[yerr_low, yerr_high], capsize=5, width=0.55, color=colors, alpha=0.85, edgecolor='k')
axes[2].axhline(0.5, color='red', linestyle=':', lw=1.5, label='Chance (P_K = 0.50)')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(labels)
axes[2].set_ylim(0.4, 0.95)
axes[2].set_ylabel("Prediction Probability P_K (with 95% CI)")
axes[2].set_title("C. Diagnostic Performance (P_K & 95% CI)", fontweight='bold')
axes[2].grid(True, axis='y', alpha=0.3)
axes[2].legend(loc='lower right')

plt.tight_layout()
plt.savefig("figures/disen_vs_pe_comparison.png", dpi=300)
print("[*] Updated 'figures/disen_vs_pe_comparison.png' successfully.")
