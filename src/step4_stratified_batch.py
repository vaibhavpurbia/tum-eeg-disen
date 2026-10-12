import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.signal.windows import dpss
from scipy.stats import norm, spearmanr
import vitaldb

def apply_bandpass(data, fs, lowcut=0.5, highcut=30.0, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def apply_notch(data, fs, notch_freq=50.0, Q=30.0):
    b, a = iirnotch(notch_freq, Q, fs)
    padlen = min(150, len(data) - 1)
    return filtfilt(b, a, data, padlen=padlen)

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
    data = vitaldb.vital_recs(ipath=caseid, track_names=tracks, interval=1.0)
    if data is None or len(data) == 0:
        return []
    
    df_track = pd.DataFrame(data, columns=tracks).bfill().ffill()
    mac_clean = df_track['Primus/MAC'].values
    bis_clean = df_track['BIS/BIS'].values
    
    episodes = find_all_stable_episodes(mac_clean, min_samples=300, max_delta=0.05)
    valid_extracted = []
    
    if not episodes:
        return []
        
    raw_eeg = vitaldb.vital_recs(ipath=caseid, track_names=['BIS/EEG1_WAV'], interval=1/fs)
    if raw_eeg is None or len(raw_eeg) == 0:
        return []
        
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

if __name__ == '__main__':
    print("Fetching candidate cases with EEG, MAC, and BIS tracks...")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    eeg_cases = set(df_trks[df_trks['tname'] == 'BIS/EEG1_WAV']['caseid'])
    mac_cases = set(df_trks[df_trks['tname'] == 'Primus/MAC']['caseid'])
    candidates = sorted(list(eeg_cases.intersection(mac_cases)))
    
    # Stratified quotas
    targets = {'light': 5, 'medium': 10, 'deep': 10}
    counts = {'light': 0, 'medium': 0, 'deep': 0}
    records = []
    
    print("Collecting stratified cohort across Light (0.3-0.6), Medium (0.7-1.1), and Deep (>=1.2) MAC...\n")
    
    for cid in candidates:
        if counts['light'] >= targets['light'] and counts['medium'] >= targets['medium'] and counts['deep'] >= targets['deep']:
            break
            
        episodes = extract_case_episodes(cid)
        for ep in episodes:
            mac = ep['mac']
            tier = None
            if 0.3 <= mac <= 0.6 and counts['light'] < targets['light']:
                tier = 'light'
            elif 0.7 <= mac <= 1.1 and counts['medium'] < targets['medium']:
                tier = 'medium'
            elif mac >= 1.2 and counts['deep'] < targets['deep']:
                tier = 'deep'
                
            if tier is not None:
                counts[tier] += 1
                raw_eeg = ep['epoch']
                
                sig_30 = apply_notch(apply_bandpass(raw_eeg, 128, 0.5, 30.0), 128)
                disen_30 = compute_disen(sig_30)
                spec_30 = compute_spectral_features(sig_30, 128, 30.0)
                
                sig_40 = apply_notch(apply_bandpass(raw_eeg, 128, 0.5, 40.0), 128)
                disen_40 = compute_disen(sig_40)
                spec_40 = compute_spectral_features(sig_40, 128, 40.0)
                
                records.append({
                    'caseid': cid,
                    'tier': tier,
                    'mac': mac,
                    'bis': ep['bis'],
                    'disen_30': disen_30,
                    'beta_30': spec_30['beta'],
                    'rel_delta_30': spec_30['rel_delta'],
                    'rel_alpha_30': spec_30['rel_alpha'],
                    'disen_40': disen_40,
                    'beta_40': spec_40['beta'],
                    'delta_disen': disen_40 - disen_30
                })
                print(f"[{tier.upper():6s}] Case {cid:4d} | MAC: {mac:.2f} | BIS: {str(ep['bis']):>4s} | DisEn(30Hz): {disen_30:.4f}")
                
    df_strat = pd.DataFrame(records)
    df_strat.to_csv("stratified_cohort.csv", index=False)
    print("\nSaved stratified cohort to 'stratified_cohort.csv'.")
    
    # Global Spearman Rank Correlation
    r_disen, p_disen = spearmanr(df_strat['mac'], df_strat['disen_30'])
    print(f"\nGlobal Spearman Correlation (MAC vs. DisEn_30Hz): rho = {r_disen:.3f} (p = {p_disen:.4e})")