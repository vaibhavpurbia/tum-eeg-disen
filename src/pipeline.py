import numpy as np
import pandas as pd
import vitaldb

def get_candidate_cases():
    """
    Finds all cases containing both the raw EEG waveform (BIS/EEG1_WAV) 
    and inhalational anesthetic MAC (Primus/MAC).
    """
    print("Querying VitalDB track catalog for BIS/EEG1_WAV and Primus/MAC...")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    
    eeg_cases = set(df_trks[df_trks['tname'] == 'BIS/EEG1_WAV']['caseid'])
    mac_cases = set(df_trks[df_trks['tname'] == 'Primus/MAC']['caseid'])
    
    candidates = sorted(list(eeg_cases.intersection(mac_cases)))
    print(f"Matched {len(candidates)} valid candidate cases.\n")
    return candidates


def find_stable_mac_episodes(mac_series, time_interval=1.0, min_duration_sec=300, max_delta=0.05):
    """
    Identifies contiguous intervals where MAC remains stable (delta <= 0.05) for >= 5 minutes.
    """
    stable_episodes = []
    min_samples = int(min_duration_sec / time_interval)
    mac_clean = pd.Series(mac_series).ffill().bfill().values
    
    start_idx = None
    for i in range(len(mac_clean)):
        val = mac_clean[i]
        
        # Exclude unrecorded or sub-clinical concentration
        if np.isnan(val) or val <= 0.2:
            if start_idx is not None and (i - start_idx) >= min_samples:
                seg = mac_clean[start_idx:i]
                if (np.max(seg) - np.min(seg)) <= max_delta:
                    stable_episodes.append({
                        'center_sec': ((start_idx + i) // 2) * time_interval,
                        'mean_mac': float(np.mean(seg)),
                        'duration_min': ((i - start_idx) * time_interval) / 60.0
                    })
            start_idx = None
            continue
            
        if start_idx is None:
            start_idx = i
        else:
            current_seg = mac_clean[start_idx : i + 1]
            if (np.max(current_seg) - np.min(current_seg)) > max_delta:
                if (i - start_idx) >= min_samples:
                    stable_episodes.append({
                        'center_sec': ((start_idx + i) // 2) * time_interval,
                        'mean_mac': float(np.mean(current_seg[:-1])),
                        'duration_min': ((i - start_idx) * time_interval) / 60.0
                    })
                start_idx = i

    if start_idx is not None and (len(mac_clean) - start_idx) >= min_samples:
        seg = mac_clean[start_idx:]
        if (np.max(seg) - np.min(seg)) <= max_delta:
            stable_episodes.append({
                'center_sec': ((start_idx + len(mac_clean)) // 2) * time_interval,
                'mean_mac': float(np.mean(seg)),
                'duration_min': ((len(mac_clean) - start_idx) * time_interval) / 60.0
            })
            
    return stable_episodes


def inspect_case(caseid):
    print(f"================ Checking Case {caseid} ================")
    
    # 1. Pull Primus/MAC at 1 Hz resolution
    mac_data = vitaldb.vital_recs(ipath=caseid, track_names=['Primus/MAC'], interval=1.0)
    if mac_data is None or len(mac_data) == 0:
        print("  No Primus/MAC recording.")
        return None
        
    mac_series = pd.Series(mac_data.flatten()).bfill().ffill().values
    episodes = find_stable_mac_episodes(mac_series, time_interval=1.0, min_duration_sec=300, max_delta=0.05)
    
    if not episodes:
        print("  No stable >= 5-minute plateau found.")
        return None
        
    fs = 128
    
    # 2. Iterate through plateaus to extract the 20-second central window
    for idx, ep in enumerate(episodes):
        center_sec = ep['center_sec']
        mean_mac = ep['mean_mac']
        dur = ep['duration_min']
        
        print(f"  Testing Plateau {idx+1}: Mean MAC = {mean_mac:.2f}, Duration = {dur:.1f} mins (Center: {center_sec:.0f}s)")
        
        # Load the 20-second center window: [center - 10s, center + 10s]
        raw_eeg = vitaldb.vital_recs(ipath=caseid, track_names=['BIS/EEG1_WAV'], interval=1/fs)
        if raw_eeg is None or len(raw_eeg) == 0:
            print("    BIS/EEG1_WAV empty.")
            continue
            
        start_samp = int((center_sec - 10) * fs)
        end_samp = int((center_sec + 10) * fs)
        
        if end_samp > len(raw_eeg):
            print("    Window extends beyond EEG duration.")
            continue
            
        epoch = raw_eeg[start_samp:end_samp].flatten()
        
        # Check for unrecorded NaNs
        nan_count = np.isnan(epoch).sum()
        if nan_count > 0:
            print(f"    Skipping: Contains {nan_count} NaNs.")
            continue
            
        # Check amplitude thresholds (artifacts and burst suppression)
        peak_amp = np.max(np.abs(epoch))
        clean_amp = peak_amp < 100.0
        supp_ratio = np.sum(np.abs(epoch) < 5.0) / len(epoch)
        clean_bs = supp_ratio < 0.05
        
        print(f"    Peak Amplitude: {peak_amp:.2f} µV (Passed <100 µV: {clean_amp})")
        print(f"    Suppression Ratio (<5 µV): {supp_ratio*100:.1f}% (Passed <5%: {clean_bs})")
        
        if clean_amp and clean_bs:
            print(f"  ===> [SUCCESS] Found valid epoch in Case {caseid}!")
            return {
                'caseid': caseid,
                'mean_mac': mean_mac,
                'epoch': epoch,
                'fs': fs,
                'center_sec': center_sec
            }
            
    return None


if __name__ == '__main__':
    candidates = get_candidate_cases()
    
    matched = None
    for cid in candidates:
        matched = inspect_case(cid)
        if matched is not None:
            break
            
    if matched:
        print("\n" + "="*50)
        print("FIRST VALID CLINICAL EPISODE EXTRACTED SUCCESSFULLY!")
        print(f"Case ID: {matched['caseid']}")
        print(f"Stable MAC: {matched['mean_mac']:.2f}")
        print(f"Epoch Length: {len(matched['epoch'])} samples ({len(matched['epoch'])/matched['fs']:.1f} s)")
        print("="*50)
        
        # Save array to disk
        np.save("sample_clean_epoch.npy", matched['epoch'])
        print("Saved clean epoch array to 'sample_clean_epoch.npy'.")