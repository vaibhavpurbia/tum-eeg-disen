import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.special import ndtr
from scipy.stats import spearmanr, wilcoxon
from numpy.lib.stride_tricks import sliding_window_view
import vitaldb

# =====================================================================
# 1. IDENTICAL PREPROCESSING FROM STEP 4 (Bandpass + 50 Hz Notch)
# =====================================================================

def apply_bandpass(data, fs=128.0, lowcut=0.5, highcut=30.0, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def apply_notch(data, fs=128.0, notch_freq=50.0, Q=30.0):
    b, a = iirnotch(notch_freq, Q, fs)
    padlen = min(150, len(data) - 1)
    return filtfilt(b, a, data, padlen=padlen)

# =====================================================================
# 2. COMPLEXITY METRICS: DisEn (m=3, c=4, d=1) vs. PE (m=3, d=1)
# =====================================================================

def compute_disen(x: np.ndarray, m: int = 3, c: int = 4, d: int = 1) -> float:
    """
    Normalized Dispersion Entropy (Rostaghi & Azami, 2016).
    Matches step4_stratified_batch.py exactly using fast C-level ndtr.
    Time Complexity: O(N)
    """
    N = len(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return 0.0
    mu = np.mean(x)
    y = ndtr((x - mu) / sigma)
    z = np.clip(np.round(c * y + 0.5).astype(np.int32), 1, c)
    
    num_vectors = N - (m - 1) * d
    patterns = np.zeros(num_vectors, dtype=np.int32)
    for k in range(m):
        patterns += (z[k * d : k * d + num_vectors] - 1) * (c ** k)
        
    counts = np.bincount(patterns, minlength=c**m)
    p = counts[counts > 0] / num_vectors
    return float(-np.sum(p * np.log(p)) / np.log(c**m))


def compute_pe(x: np.ndarray, m: int = 3, d: int = 1) -> float:
    """
    Normalized Permutation Entropy (Bandt & Pompe, 2002).
    Time Complexity: O(N * m log m)
    """
    if np.std(x) < 1e-12:
        return 0.0
    
    windows = sliding_window_view(x, window_shape=(m - 1) * d + 1)[:, ::d]
    ranks = np.argsort(windows, axis=1, kind='mergesort')
    
    powers = m ** np.arange(m - 1, -1, -1, dtype=np.int32)
    pattern_ids = ranks @ powers
    
    counts = np.bincount(pattern_ids, minlength=m**m)
    p = counts[counts > 0] / len(pattern_ids)
    
    norm_factor = np.log(6.0) if m == 3 else np.sum(np.log(np.arange(1, m + 1)))
    return float(-np.sum(p * np.log(p)) / norm_factor)

# =====================================================================
# 3. SMITH'S PREDICTION PROBABILITY (P_K)
# =====================================================================

def compute_pk(metric_vals: np.ndarray, target_vals: np.ndarray) -> float:
    """Computes Smith's Prediction Probability (P_K)."""
    n = len(metric_vals)
    concordant, discordant, tied_m = 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if target_vals[i] == target_vals[j]:
                continue
            dm = metric_vals[i] - metric_vals[j]
            dt = target_vals[i] - target_vals[j]
            if dm == 0:
                tied_m += 1
            elif (dm > 0 and dt < 0) or (dm < 0 and dt > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant + tied_m
    if total == 0:
        return 0.5
    pk = (concordant + 0.5 * tied_m) / total
    return max(pk, 1.0 - pk)

# =====================================================================
# 4. VITALDB EPISODE EXTRACTION (FROM STEP 4)
# =====================================================================

def find_all_stable_episodes(mac_clean, min_samples=300, max_delta=0.05):
    episodes = []
    start = None
    for i in range(len(mac_clean)):
        val = mac_clean[i]
        if np.isnan(val) or val <= 0.2:
            if start is not None and (i - start) >= min_samples:
                seg = mac_clean[start:i]
                if (np.max(seg) - np.min(seg)) <= max_delta:
                    episodes.append((start, i, float(np.mean(seg))))
            start = None
            continue
        if start is None:
            start = i
        else:
            seg = mac_clean[start:i+1]
            if (np.max(seg) - np.min(seg)) > max_delta:
                if (i - start) >= min_samples:
                    episodes.append((start, i, float(np.mean(mac_clean[start:i]))))
                start = i
    if start is not None and (len(mac_clean) - start) >= min_samples:
        seg = mac_clean[start:]
        if (np.max(seg) - np.min(seg)) <= max_delta:
            episodes.append((start, len(mac_clean), float(np.mean(seg))))
    return episodes


def extract_case_episodes(caseid, fs=128):
    tracks = ['Primus/MAC', 'BIS/BIS']
    data = vitaldb.vital_recs(ipath=int(caseid), track_names=tracks, interval=1.0)
    if data is None or len(data) == 0:
        return []
    df_track = pd.DataFrame(data, columns=tracks).bfill().ffill()
    mac_clean = df_track['Primus/MAC'].values
    bis_clean = df_track['BIS/BIS'].values
    
    episodes = find_all_stable_episodes(mac_clean, min_samples=300, max_delta=0.05)
    if not episodes:
        return []
        
    raw_eeg = vitaldb.vital_recs(ipath=int(caseid), track_names=['BIS/EEG1_WAV'], interval=1/fs)
    if raw_eeg is None or len(raw_eeg) == 0:
        return []
        
    valid_extracted = []
    for (start, end, mean_mac) in episodes:
        center_sec = (start + end) // 2
        s_idx = int((center_sec - 10) * fs)
        e_idx = int((center_sec + 10) * fs)
        if e_idx > len(raw_eeg):
            continue
        epoch = raw_eeg[s_idx:e_idx].flatten()
        if np.isnan(epoch).any() or np.max(np.abs(epoch)) >= 100.0 or (np.sum(np.abs(epoch) < 5.0) / len(epoch)) >= 0.05:
            continue
        bis_val = bis_clean[center_sec] if not np.isnan(bis_clean[center_sec]) else None
        valid_extracted.append({'mac': mean_mac, 'bis': bis_val, 'epoch': epoch})
    return valid_extracted

# =====================================================================
# 5. MAIN HEAD-TO-HEAD BENCHMARK ON THE 25 STRATIFIED COHORT EPOCHS
# =====================================================================

if __name__ == '__main__':
    df_strat = pd.read_csv("data/stratified_cohort.csv")
    unique_cases = sorted(df_strat['caseid'].unique())
    print(f"[*] Loaded 'data/stratified_cohort.csv' ({len(df_strat)} plateaus across {len(unique_cases)} unique cases).")
    print(f"[*] Re-fetching exact 20s EEG epochs for cases: {unique_cases}...\n")

    epochs_30 = [None] * len(df_strat)
    epochs_40 = [None] * len(df_strat)

    for cid in unique_cases:
        eps = extract_case_episodes(cid, fs=128)
        sub_df = df_strat[df_strat['caseid'] == cid]
        
        for ep in eps:
            s30 = apply_notch(apply_bandpass(ep['epoch'], 128, 0.5, 30.0), 128)
            d30 = compute_disen(s30, m=3, c=4, d=1)
            
            # Match against exact row in stratified_cohort.csv
            for idx, row in sub_df.iterrows():
                if epochs_30[idx] is None and np.isclose(d30, row['disen_30'], atol=1e-5):
                    s40 = apply_notch(apply_bandpass(ep['epoch'], 128, 0.5, 40.0), 128)
                    epochs_30[idx] = s30
                    epochs_40[idx] = s40
                    print(f"  Matched Row {idx:2d} | Case {cid:4d} ({row['tier']:6s}) | MAC={row['mac']:.2f} | DisEn_30={d30:.4f}")
                    break

    # Verify all 25 epochs were matched
    assert all(x is not None for x in epochs_30), "Some epochs failed to match!"

    # Compute DisEn and PE (m=3, d=1) + time execution over 200 iterations for high precision
    n_repeats = 200
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        disen_30_vals = np.array([compute_disen(s, m=3, c=4, d=1) for s in epochs_30])
    disen_ms = ((time.perf_counter() - t0) / (n_repeats * len(epochs_30))) * 1000.0
    disen_40_vals = np.array([compute_disen(s, m=3, c=4, d=1) for s in epochs_40])

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        pe_30_vals = np.array([compute_pe(s, m=3, d=1) for s in epochs_30])
    pe_ms = ((time.perf_counter() - t0) / (n_repeats * len(epochs_30))) * 1000.0
    pe_40_vals = np.array([compute_pe(s, m=3, d=1) for s in epochs_40])

    # Save enriched CSV
    df_strat['pe_30'] = pe_30_vals
    df_strat['pe_40'] = pe_40_vals
    df_strat['delta_pe'] = pe_40_vals - pe_30_vals
    df_strat.to_csv("data/stratified_cohort_with_pe.csv", index=False)

    # Statistical Benchmarking
    mac = df_strat['mac'].values
    valid_bis_mask = (df_strat['bis'].notna()) & (df_strat['bis'] > 0)
    bis_valid = df_strat.loc[valid_bis_mask, 'bis'].values

    pk_disen_mac = compute_pk(disen_30_vals, mac)
    pk_pe_mac    = compute_pk(pe_30_vals, mac)
    pk_bis_mac   = compute_pk(bis_valid, mac[valid_bis_mask])

    rho_disen_mac, p_disen_mac = spearmanr(mac, disen_30_vals)
    rho_pe_mac,    p_pe_mac    = spearmanr(mac, pe_30_vals)

    rho_disen_bis, p_disen_bis = spearmanr(bis_valid, disen_30_vals[valid_bis_mask])
    rho_pe_bis,    p_pe_bis    = spearmanr(bis_valid, pe_30_vals[valid_bis_mask])

    emg_disen = disen_40_vals - disen_30_vals
    emg_pe    = pe_40_vals - pe_30_vals

    print("\n" + "=" * 95)
    print("HEAD-TO-HEAD BENCHMARK ON VITALDB STRATIFIED COHORT (N = 25 STEADY-STATE PLATEAUS, 20s @ 128Hz)")
    print("=" * 95)
    print(f"{'Metric (0.5-30 Hz)':<24} | {'P_K (MAC)':<10} | {'Spearman r (MAC)':<19} | {'Spearman r (BIS)':<19} | {'Speed (ms/ep)'}")
    print("-" * 95)
    print(f"{'DisEn (m=3, c=4, d=1)':<24} | {pk_disen_mac:<10.3f} | {rho_disen_mac:+.3f} (p={p_disen_mac:.4f}) | {rho_disen_bis:+.3f} (p={p_disen_bis:.4f}) | {disen_ms:.3f} ms")
    print(f"{'PE    (m=3, d=1)':<24} | {pk_pe_mac:<10.3f} | {rho_pe_mac:+.3f} (p={p_pe_mac:.4f}) | {rho_pe_bis:+.3f} (p={p_pe_bis:.4f}) | {pe_ms:.3f} ms")
    print(f"{'Commercial BIS ( Ground)':<24} | {pk_bis_mac:<10.3f} | {'--':<19} | {'1.000 (Reference)':<19} | {'Proprietary'}")
    print("=" * 95)
    print(f"Speedup Factor: DisEn is {pe_ms / disen_ms:.2f}x faster than PE per 20-second epoch.")
    print(f"Mean EMG Shift (40Hz - 30Hz): DisEn = {np.mean(emg_disen):+.4f} (+/-{np.std(emg_disen):.4f}) | PE = {np.mean(emg_pe):+.4f} (+/-{np.std(emg_pe):.4f})")
    print("=" * 95 + "\n")

    # Generate Publication-Ready 3-Panel Comparison Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: DisEn vs MAC
    axes[0].scatter(mac, disen_30_vals, c='#1f77b4', s=55, alpha=0.85, edgecolors='k', label='DisEn (30 Hz)')
    m_fit, b_fit = np.polyfit(mac, disen_30_vals, 1)
    x_line = np.linspace(min(mac), max(mac), 100)
    axes[0].plot(x_line, m_fit * x_line + b_fit, 'r--', lw=2, label=f'Fit (r={rho_disen_mac:.3f}, p={p_disen_mac:.4f})')
    axes[0].set_title(f"A. Dispersion Entropy vs. MAC\n(P_K = {pk_disen_mac:.3f}, p = {p_disen_mac:.4f})", fontweight='bold')
    axes[0].set_xlabel("Volatile Anesthetic Depth (MAC)")
    axes[0].set_ylabel("Normalized DisEn (m=3, c=4, d=1)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Panel B: PE vs MAC
    axes[1].scatter(mac, pe_30_vals, c='#ff7f0e', s=55, alpha=0.85, edgecolors='k', label='PE (30 Hz)')
    m_pe, b_pe = np.polyfit(mac, pe_30_vals, 1)
    axes[1].plot(x_line, m_pe * x_line + b_pe, 'k--', lw=2, label=f'Fit (r={rho_pe_mac:.3f}, p={p_pe_mac:.4f})')
    axes[1].set_title(f"B. Permutation Entropy vs. MAC\n(P_K = {pk_pe_mac:.3f}, p = {p_pe_mac:.4f})", fontweight='bold')
    axes[1].set_xlabel("Volatile Anesthetic Depth (MAC)")
    axes[1].set_ylabel("Normalized PE (m=3, d=1)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Panel C: Computational Latency & Prediction Accuracy Comparison
    x_pos = np.arange(2)
    ax_twin = axes[2].twinx()
    b1 = axes[2].bar(x_pos - 0.18, [pk_disen_mac, pk_pe_mac], width=0.32, color=['#1f77b4', '#ff7f0e'], alpha=0.85, edgecolor='k', label='Prediction Prob (P_K)')
    b2 = ax_twin.bar(x_pos + 0.18, [disen_ms, pe_ms], width=0.32, color=['#aec7e8', '#ffbb78'], alpha=0.85, edgecolor='k', hatch='//', label='Latency (ms/epoch)')
    axes[2].axhline(0.5, color='gray', linestyle=':', label='Chance (P_K = 0.50)')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(['DisEn\n(m=3, c=4)', 'PE\n(m=3)'])
    axes[2].set_ylim(0.4, 0.85)
    axes[2].set_ylabel("Prediction Probability (P_K)")
    ax_twin.set_ylabel("Execution Time per 20s Epoch (ms)")
    axes[2].set_title("C. Accuracy (P_K) vs. Computational Cost", fontweight='bold')
    axes[2].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/disen_vs_pe_comparison.png", dpi=300)
    print("[*] Saved figure to 'figures/disen_vs_pe_comparison.png' and data to 'data/stratified_cohort_with_pe.csv'.")