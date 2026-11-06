import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.special import ndtr
from scipy.stats import spearmanr
from numpy.lib.stride_tricks import sliding_window_view
import vitaldb

# =====================================================================
# 1. FILTERING & SIGNAL PROCESSING
# =====================================================================

def apply_bandpass(data, fs=128.0, lowcut=0.5, highcut=30.0, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def apply_notch(data, fs=128.0, notch_freq=50.0, Q=30.0):
    b, a = iirnotch(notch_freq, Q, fs)
    padlen = min(150, len(data) - 1)
    return filtfilt(b, a, data, padlen=padlen)

def compute_sef95(x, fs=128.0, f_max=30.0):
    """Computes the 95% Spectral Edge Frequency (SEF95)."""
    N = len(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.fft.rfft(x)
    psd = (np.abs(fft_vals) ** 2) / (fs * N)
    
    valid_mask = (freqs >= 0.5) & (freqs <= f_max)
    f_sub = freqs[valid_mask]
    p_sub = psd[valid_mask]
    
    cum_power = np.cumsum(p_sub)
    total_power = cum_power[-1]
    if total_power <= 0:
        return 0.0
    
    idx_95 = np.where(cum_power >= 0.95 * total_power)[0]
    return float(f_sub[idx_95[0]]) if len(idx_95) > 0 else 0.0

# =====================================================================
# 2. COMPLEXITY METRICS: DispEn, PeEn, SampEn
# =====================================================================

def compute_dispen(x: np.ndarray, m: int = 3, c: int = 4, d: int = 1) -> float:
    """Normalized Dispersion Entropy (Rostaghi & Azami, 2016)."""
    N = len(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return 0.0
    y = ndtr((x - np.mean(x)) / sigma)
    z = np.clip(np.round(c * y + 0.5).astype(np.int32), 1, c)
    
    num_vectors = N - (m - 1) * d
    patterns = np.zeros(num_vectors, dtype=np.int32)
    for k in range(m):
        patterns += (z[k * d : k * d + num_vectors] - 1) * (c ** k)
        
    counts = np.bincount(patterns, minlength=c**m)
    p = counts[counts > 0] / num_vectors
    return float(-np.sum(p * np.log(p)) / np.log(c**m))


def compute_peen(x: np.ndarray, m: int = 3, d: int = 1) -> float:
    """Normalized Permutation Entropy (Bandt & Pompe, 2002)."""
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


def compute_sampen(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample Entropy (Richman & Moorman, 2000)."""
    x = np.asarray(x)
    N = len(x)
    r = r_factor * np.std(x)
    if r == 0 or N <= m + 1:
        return 0.0

    def _max_dist(x_i, x_j, m_val):
        return max([abs(x_i[k] - x_j[k]) for k in range(m_val)])

    def _phi(m_val):
        embeddings = np.array([x[i:i + m_val] for i in range(N - m_val + 1)])
        count = 0
        num_vecs = len(embeddings)
        for i in range(num_vecs):
            # count matches excluding self-matches to avoid bias
            matches = sum(1 for j in range(num_vecs) if i != j and _max_dist(embeddings[i], embeddings[j], m_val) <= r)
            count += matches
        return count

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    if phi_m == 0 or phi_m1 == 0:
        return 0.0
    return float(-np.log(phi_m1 / phi_m))

# =====================================================================
# 3. STATISTICAL EVALUATION WITH BOOTSTRAPPED 95% CI FOR P_K
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


def bootstrap_pk(metric_vals: np.ndarray, target_vals: np.ndarray, n_boot: int = 1000) -> tuple:
    """Computes 95% confidence intervals for P_K via bootstrap resampling."""
    n = len(metric_vals)
    pk_boot = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pk_boot.append(compute_pk(metric_vals[idx], target_vals[idx]))
    return float(np.mean(pk_boot)), float(np.percentile(pk_boot, 2.5)), float(np.percentile(pk_boot, 97.5))

# =====================================================================
# 4. VITALDB EPISODE EXTRACTION
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
    tracks = ['Primus/MAC', 'BIS/BIS', 'BIS/SEF']
    data = vitaldb.vital_recs(ipath=int(caseid), track_names=tracks, interval=1.0)
    if data is None or len(data) == 0:
        return []
    df_track = pd.DataFrame(data, columns=tracks).bfill().ffill()
    mac_clean = df_track['Primus/MAC'].values
    bis_clean = df_track['BIS/BIS'].values
    sef_clean = df_track['BIS/SEF'].values if 'BIS/SEF' in df_track else np.full_like(bis_clean, np.nan)
    
    episodes = find_all_stable_episodes(mac_clean, min_samples=300, max_delta=0.05)
    if not episodes:
        return []
        
    raw_eeg = vitaldb.vital_recs(ipath=int(caseid), track_names=['BIS/EEG1_WAV'], interval=1/fs)
    if raw_eeg is None or len(raw_eeg) == 0:
        return []
        
    valid_extracted = []
    for (start, end, mean_mac) in episodes:
        if mean_mac > 2.0:  
            continue
        center_sec = (start + end) // 2
        s_idx = int((center_sec - 10) * fs)
        e_idx = int((center_sec + 10) * fs)
        if e_idx > len(raw_eeg):
            continue
        epoch = raw_eeg[s_idx:e_idx].flatten()
        if np.isnan(epoch).any() or np.max(np.abs(epoch)) >= 100.0 or (np.sum(np.abs(epoch) < 5.0) / len(epoch)) >= 0.05:
            continue
        bis_val = bis_clean[center_sec] if not np.isnan(bis_clean[center_sec]) else None
        sef_val = sef_clean[center_sec] if not np.isnan(sef_clean[center_sec]) else None
        valid_extracted.append({'mac': mean_mac, 'bis': bis_val, 'sef': sef_val, 'epoch': epoch})
    return valid_extracted

# =====================================================================
# 5. MAIN EXECUTION & REPOSITORY EXPORT
# =====================================================================

if __name__ == '__main__':
    df_strat = pd.read_csv("data/stratified_cohort.csv")
    # Filter out MAC > 2.0
    df_strat = df_strat[df_strat['mac'] <= 2.0].reset_index(drop=True)
    unique_cases = sorted(df_strat['caseid'].unique())
    
    print(f"[*] Loaded filtered cohort ({len(df_strat)} plateaus across {len(unique_cases)} cases, MAC <= 2.0).")
    print(f"[*] Re-fetching exact 20s EEG epochs for repository export...\n")

    epochs_30 = [None] * len(df_strat)
    epochs_40 = [None] * len(df_strat)
    raw_epochs_list = []

    os.makedirs("data/raw_reproducible_epochs", exist_ok=True)

    for cid in unique_cases:
        eps = extract_case_episodes(cid, fs=128)
        sub_df = df_strat[df_strat['caseid'] == cid]
        
        for ep in eps:
            s30 = apply_notch(apply_bandpass(ep['epoch'], 128, 0.5, 30.0), 128)
            d30 = compute_dispen(s30, m=3, c=4, d=1)
            
            for idx, row in sub_df.iterrows():
                if epochs_30[idx] is None and np.isclose(d30, row['disen_30'], atol=1e-5):
                    s40 = apply_notch(apply_bandpass(ep['epoch'], 128, 0.5, 40.0), 128)
                    epochs_30[idx] = s30
                    epochs_40[idx] = s40
                    
                    # Save individual reproducible numpy file for GitHub repository
                    epoch_filename = f"data/raw_reproducible_epochs/case_{cid}_plateau_{idx}.npy"
                    np.save(epoch_filename, ep['epoch'])
                    raw_epochs_list.append({
                        'caseid': cid,
                        'plateau_idx': idx,
                        'tier': row['tier'],
                        'mac': row['mac'],
                        'bis': row['bis'],
                        'file_path': epoch_filename
                    })
                    print(f"  Matched & Exported Row {idx:2d} | Case {cid:4d} ({row['tier']:6s}) | MAC={row['mac']:.2f}")
                    break

    assert all(x is not None for x in epochs_30), "Some epochs failed to match!"

    # Save repository manifest mapping metadata to raw arrays
    df_manifest = pd.DataFrame(raw_epochs_list)
    df_manifest.to_csv("data/raw_reproducible_epochs/manifest.csv", index=False)

    # Compute metrics across all valid epochs
    print("\n[*] Computing DispEn, PeEn, SampEn, and SEF95...")
    disp_30_vals = np.array([compute_dispen(s, m=3, c=4, d=1) for s in epochs_30])
    peen_30_vals = np.array([compute_peen(s, m=3, d=1) for s in epochs_30])
    samp_30_vals = np.array([compute_sampen(s, m=2, r_factor=0.2) for s in epochs_30])
    sef_30_vals  = np.array([compute_sef95(s, fs=128.0, f_max=30.0) for s in epochs_30])

    disp_40_vals = np.array([compute_dispen(s, m=3, c=4, d=1) for s in epochs_40])
    peen_40_vals = np.array([compute_peen(s, m=3, d=1) for s in epochs_40])

    # Enriched DataFrame export
    df_strat['disp_30'] = disp_30_vals
    df_strat['peen_30'] = peen_30_vals
    df_strat['samp_30'] = samp_30_vals
    df_strat['sef95_30'] = sef_30_vals
    df_strat.to_csv("data/stratified_cohort_fully_enriched.csv", index=False)

    # Statistical Evaluation & Bootstrapped CI
    mac = df_strat['mac'].values
    valid_bis_mask = df_strat['bis'].notna() & (df_strat['bis'] > 0)
    bis_valid = df_strat.loc[valid_bis_mask, 'bis'].values

    pk_disp, ci_disp_l, ci_disp_u = bootstrap_pk(disp_30_vals, mac)
    pk_peen, ci_peen_l, ci_peen_u = bootstrap_pk(peen_30_vals, mac)
    pk_samp, ci_samp_l, ci_samp_u = bootstrap_pk(samp_30_vals, mac)
    pk_bis,  ci_bis_l,  ci_bis_u  = bootstrap_pk(bis_valid, mac[valid_bis_mask])

    rho_disp_mac, p_disp_mac = spearmanr(mac, disp_30_vals)
    rho_peen_mac, p_peen_mac = spearmanr(mac, peen_30_vals)
    rho_samp_mac, p_samp_mac = spearmanr(mac, samp_30_vals)

    rho_disp_bis, p_disp_bis = spearmanr(bis_valid, disp_30_vals[valid_bis_mask])

    emg_disp = disp_40_vals - disp_30_vals
    emg_peen = peen_40_vals - peen_30_vals

    print("\n" + "=" * 105)
    print(f"UPDATED HEAD-TO-HEAD BENCHMARK (MAC <= 2.0, N = {len(df_strat)} PLATEAUS)")
    print("=" * 105)
    print(f"{'Metric (0.5-30 Hz)':<24} | {'P_K (MAC) [95% CI]':<22} | {'Spearman r (MAC)':<18} | {'Spearman r (BIS)':<18} | {'EMG Shift'}")
    print("-" * 105)
    print(f"{'DispEn (m=3, c=4, d=1)':<24} | {pk_disp:.3f} [{ci_disp_l:.3f}-{ci_disp_u:.3f}]  | {rho_disp_mac:+.3f} (p={p_disp_mac:.4f})  | {rho_disp_bis:+.3f} (p={p_disp_bis:.4f})  | {np.mean(emg_disp):+.4f}")
    print(f"{'PeEn   (m=3, d=1)':<24} | {pk_peen:.3f} [{ci_peen_l:.3f}-{ci_peen_u:.3f}]  | {rho_peen_mac:+.3f} (p={p_peen_mac:.4f})  | {'--':<18}  | {np.mean(emg_peen):+.4f}")
    print(f"{'SampEn (m=2, r=0.2)':<24} | {pk_samp:.3f} [{ci_samp_l:.3f}-{ci_samp_u:.3f}]  | {rho_samp_mac:+.3f} (p={p_samp_mac:.4f})  | {'--':<18}  | --")
    print(f"{'Commercial BIS':<24} | {pk_bis:.3f} [{ci_bis_l:.3f}-{ci_bis_u:.3f}]  | {'--':<18}  | 1.000 (Reference)    | Proprietary")
    print("=" * 105 + "\n")
  