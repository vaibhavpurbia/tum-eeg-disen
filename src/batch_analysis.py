import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch
from scipy.signal.windows import dpss
from scipy.stats import norm, spearmanr
import vitaldb

# --- Filtering Functions ---
def apply_bandpass(data, fs, lowcut=0.5, highcut=30.0, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def apply_notch(data, fs, notch_freq=50.0, Q=30.0):
    b, a = iirnotch(notch_freq, Q, fs)
    from scipy.signal import filtfilt
    padlen = min(150, len(data) - 1)
    return filtfilt(b, a, data, padlen=padlen)

# --- Feature Extraction Functions ---
def compute_disen(x, m=3, c=4, d=1):
    N = len(x)
    sigma = np.std(x)
    if sigma == 0:
        return 0.0
    mu = np.mean(x)
    y = norm.cdf(x, loc=mu, scale=sigma)
    z = np.clip(np.round(c * y + 0.5).astype(int), 1, c)
    num_vectors = N - (m - 1) * d
    patterns = np.zeros(num_vectors, dtype=int)
    for k in range(m):
        patterns += (z[k * d : k * d + num_vectors] - 1) * (c ** k)
    counts = np.bincount(patterns, minlength=c**m)
    p = counts[counts > 0] / num_vectors
    return -np.sum(p * np.log(p)) / np.log(c**m)

def compute_spectral_features(x, fs, f_max):
    N = len(x)
    tapers, _ = dpss(N, NW=3.0, Kmax=3, sym=False, return_ratios=True)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    psd_k = np.zeros((3, len(freqs)))
    for k in range(3):
        tapered = x * tapers[k]
        psd_k[k, :] = (np.abs(np.fft.rfft(tapered)) ** 2) / fs
    psd = np.mean(psd_k, axis=0)
    
    valid_mask = (freqs >= 0.5) & (freqs <= f_max)
    f_sub = freqs[valid_mask]
    p_sub = psd[valid_mask]
    total_power = np.sum(p_sub)
    
    rel_delta = np.sum(p_sub[(f_sub >= 0.5) & (f_sub <= 4.0)]) / total_power if total_power > 0 else 0.0
    rel_alpha = np.sum(p_sub[(f_sub >= 8.0) & (f_sub <= 12.0)]) / total_power if total_power > 0 else 0.0
    
    fit_mask = (f_sub >= 2.0) & (f_sub <= f_max)
    log_f = np.log10(f_sub[fit_mask])
    log_p = np.log10(p_sub[fit_mask] + 1e-12)
    slope, _ = np.polyfit(log_f, log_p, 1)
    
    return {'rel_delta': rel_delta, 'rel_alpha': rel_alpha, 'beta': -slope}

def extract_valid_epoch(caseid, fs=128):
    mac_data = vitaldb.vital_recs(ipath=caseid, track_names=['Primus/MAC'], interval=1.0)
    if mac_data is None or len(mac_data) == 0:
        return None
    mac_clean = pd.Series(mac_data.flatten()).bfill().ffill().values
    
    # Identify plateaus >= 5 min (300 samples)
    min_samples = 300
    start = None
    for i in range(len(mac_clean)):
        val = mac_clean[i]
        if np.isnan(val) or val <= 0.2:
            start = None
            continue
        if start is None:
            start = i
        else:
            seg = mac_clean[start:i+1]
            if (np.max(seg) - np.min(seg)) > 0.05:
                if (i - start) >= min_samples:
                    break
                start = i
                
    if start is None or (i - start) < min_samples:
        return None
        
    plateau_mac = float(np.mean(mac_clean[start:i]))
    center_sec = (start + i) // 2
    
    # Load central 20-second EEG window
    raw_eeg = vitaldb.vital_recs(ipath=caseid, track_names=['BIS/EEG1_WAV'], interval=1/fs)
    if raw_eeg is None or len(raw_eeg) == 0:
        return None
        
    s_idx = int((center_sec - 10) * fs)
    e_idx = int((center_sec + 10) * fs)
    if e_idx > len(raw_eeg):
        return None
        
    epoch = raw_eeg[s_idx:e_idx].flatten()
    if np.isnan(epoch).any() or np.max(np.abs(epoch)) >= 100.0 or (np.sum(np.abs(epoch) < 5.0) / len(epoch)) >= 0.05:
        return None
        
    return {'mac': plateau_mac, 'epoch': epoch}

# --- Batch Execution ---
if __name__ == '__main__':
    print("Loading track directory to get candidate case list...")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    eeg_cases = set(df_trks[df_trks['tname'] == 'BIS/EEG1_WAV']['caseid'])
    mac_cases = set(df_trks[df_trks['tname'] == 'Primus/MAC']['caseid'])
    candidate_cases = sorted(list(eeg_cases.intersection(mac_cases)))
    
    records = []
    target_count = 15
    print(f"Scanning for {target_count} clean episodes across candidates...\n")
    
    for cid in candidate_cases:
        res = extract_valid_epoch(cid)
        if res is not None:
            raw_eeg = res['epoch']
            mac = res['mac']
            
            # 30 Hz filter pipeline
            sig_30 = apply_notch(apply_bandpass(raw_eeg, 128, 0.5, 30.0), 128)
            disen_30 = compute_disen(sig_30)
            spec_30 = compute_spectral_features(sig_30, 128, 30.0)
            
            # 40 Hz filter pipeline
            sig_40 = apply_notch(apply_bandpass(raw_eeg, 128, 0.5, 40.0), 128)
            disen_40 = compute_disen(sig_40)
            spec_40 = compute_spectral_features(sig_40, 128, 40.0)
            
            records.append({
                'caseid': cid,
                'mac': mac,
                'disen_30': disen_30,
                'rel_delta_30': spec_30['rel_delta'],
                'rel_alpha_30': spec_30['rel_alpha'],
                'beta_30': spec_30['beta'],
                'disen_40': disen_40,
                'rel_delta_40': spec_40['rel_delta'],
                'rel_alpha_40': spec_40['rel_alpha'],
                'beta_40': spec_40['beta'],
                'delta_disen': disen_40 - disen_30
            })
            print(f"Captured Case {cid:4d} | MAC: {mac:.2f} | DisEn(30Hz): {disen_30:.4f} | DisEn(40Hz): {disen_40:.4f}")
            
        if len(records) >= target_count:
            break
            
    df_results = pd.DataFrame(records)
    df_results.to_csv("cohort_features.csv", index=False)
    print("\nSaved cohort results to 'cohort_features.csv'.")
    
    # Statistical Correlation against MAC
    rho_30, p_30 = spearmanr(df_results['mac'], df_results['disen_30'])
    rho_40, p_40 = spearmanr(df_results['mac'], df_results['disen_40'])
    print(f"\nSpearman Correlation (MAC vs. DisEn_30Hz): rho = {rho_30:.3f} (p = {p_30:.4f})")
    print(f"Spearman Correlation (MAC vs. DisEn_40Hz): rho = {rho_40:.3f} (p = {p_40:.4f})")