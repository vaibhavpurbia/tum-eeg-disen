import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch
from scipy.signal.windows import dpss
from scipy.stats import norm

# -------------------------------------------------------------
# 1. Filtering Regimes (Zero-phase Butterworth + Notch)
# -------------------------------------------------------------
def apply_bandpass(data, fs, lowcut=0.5, highcut=30.0, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def apply_notch(data, fs, notch_freq=50.0, Q=30.0):
    b, a = iirnotch(notch_freq, Q, fs)
    # Using zero-phase filtering
    padlen = min(150, len(data) - 1)
    from scipy.signal import filtfilt
    return filtfilt(b, a, data, padlen=padlen)

# -------------------------------------------------------------
# 2. Dispersion Entropy (DisEn)
# -------------------------------------------------------------
def compute_disen(x, m=3, c=4, d=1):
    """
    Computes normalized Dispersion Entropy using NCDF mapping.
    Parameters:
      m: embedding dimension (default 3)
      c: number of classes (default 4)
      d: time delay (default 1)
    """
    N = len(x)
    sigma = np.std(x)
    if sigma == 0:
        return 0.0
    
    # 1. Normal Cumulative Distribution Function mapping
    mu = np.mean(x)
    y = norm.cdf(x, loc=mu, scale=sigma)
    
    # 2. Class quantization into {1, 2, ..., c}
    z = np.clip(np.round(c * y + 0.5).astype(int), 1, c)
    
    # 3. Construct embedding pattern indices
    num_vectors = N - (m - 1) * d
    patterns = np.zeros(num_vectors, dtype=int)
    for k in range(m):
        patterns += (z[k * d : k * d + num_vectors] - 1) * (c ** k)
        
    # 4. Pattern distribution and Shannon entropy
    counts = np.bincount(patterns, minlength=c**m)
    p = counts[counts > 0] / num_vectors
    disen = -np.sum(p * np.log(p))
    
    # Normalized by theoretical maximum ln(c^m)
    return disen / np.log(c**m)

# -------------------------------------------------------------
# 3. Classical Spectral Features & Spectral Exponent
# -------------------------------------------------------------
def compute_spectral_features(x, fs, f_max):
    """
    Multitaper PSD (TW=3, K=3), band powers, and aperiodic spectral exponent (beta).
    """
    N = len(x)
    time_bandwidth = 3.0
    num_tapers = 3
    tapers, _ = dpss(N, NW=time_bandwidth, Kmax=num_tapers, sym=False, return_ratios=True)
    
    freqs = np.fft.rfftfreq(N, d=1/fs)
    psd_k = np.zeros((num_tapers, len(freqs)))
    
    for k in range(num_tapers):
        tapered = x * tapers[k]
        fft_v = np.fft.rfft(tapered)
        psd_k[k, :] = (np.abs(fft_v) ** 2) / fs
        
    psd = np.mean(psd_k, axis=0)
    
    # Frequency range up to the filter cutoff
    valid_mask = (freqs >= 0.5) & (freqs <= f_max)
    f_sub = freqs[valid_mask]
    p_sub = psd[valid_mask]
    total_power = np.sum(p_sub)
    
    # Absolute & Relative Band Powers
    delta_mask = (f_sub >= 0.5) & (f_sub <= 4.0)
    alpha_mask = (f_sub >= 8.0) & (f_sub <= 12.0)
    
    abs_delta = np.sum(p_sub[delta_mask])
    abs_alpha = np.sum(p_sub[alpha_mask])
    rel_delta = abs_delta / total_power if total_power > 0 else 0.0
    rel_alpha = abs_alpha / total_power if total_power > 0 else 0.0
    
    # Spectral Exponent (beta): fit linear slope on log-log power spectrum between 2 Hz and f_max
    fit_mask = (f_sub >= 2.0) & (f_sub <= f_max)
    log_f = np.log10(f_sub[fit_mask])
    log_p = np.log10(p_sub[fit_mask] + 1e-12)
    slope, intercept = np.polyfit(log_f, log_p, 1)
    beta = -slope  # 1/f^beta convention
    
    return {
        'abs_delta': abs_delta,
        'abs_alpha': abs_alpha,
        'rel_delta': rel_delta,
        'rel_alpha': rel_alpha,
        'beta': beta,
        'freqs': f_sub,
        'psd': p_sub
    }

# -------------------------------------------------------------
# Main Evaluation Run
# -------------------------------------------------------------
if __name__ == '__main__':
    # Load the clean epoch extracted in Step 1
    raw_epoch = np.load("sample_clean_epoch.npy")
    fs = 128
    
    print("Loaded 'sample_clean_epoch.npy' (2560 samples, 20.0s).")
    print("-" * 55)
    
    results = {}
    spectra = {}
    
    for cutoff in [30.0, 40.0]:
        # 1. Bandpass + Notch
        sig_bp = apply_bandpass(raw_epoch, fs=fs, lowcut=0.5, highcut=cutoff)
        sig_clean = apply_notch(sig_bp, fs=fs, notch_freq=50.0)
        
        # 2. Dispersion Entropy (m=3, c=4, d=1)
        disen = compute_disen(sig_clean, m=3, c=4, d=1)
        
        # 3. Spectral Features
        spec = compute_spectral_features(sig_clean, fs=fs, f_max=cutoff)
        spectra[cutoff] = spec
        
        results[cutoff] = {
            'DisEn': disen,
            'Rel Delta (0.5-4 Hz)': spec['rel_delta'],
            'Rel Alpha (8-12 Hz)': spec['rel_alpha'],
            'Spectral Exponent (Beta)': spec['beta']
        }
        
        print(f"Results for Filter Regime: 0.5 - {int(cutoff)} Hz")
        print(f"  Dispersion Entropy (DisEn) : {disen:.4f}")
        print(f"  Relative Delta Power       : {spec['rel_delta']:.4f}")
        print(f"  Relative Alpha Power       : {spec['rel_alpha']:.4f}")
        print(f"  Spectral Exponent (Beta)   : {spec['beta']:.4f}")
        print("-" * 55)
        
    delta_disen = results[40.0]['DisEn'] - results[30.0]['DisEn']
    print(f"Filter Sensitivity Offset (DisEn_40Hz - DisEn_30Hz): {delta_disen:+.4f}")
    print("-" * 55)

    # 4. Generate Diagnostic Visualization Plot
    time_axis = np.linspace(0, 20, len(raw_epoch))
    plt.figure(figsize=(12, 7))
    
    # Subplot 1: Filtered Time Series
    plt.subplot(2, 1, 1)
    sig_30 = apply_bandpass(raw_epoch, fs=fs, lowcut=0.5, highcut=30.0)
    sig_40 = apply_bandpass(raw_epoch, fs=fs, lowcut=0.5, highcut=40.0)
    plt.plot(time_axis, sig_30, label='0.5–30 Hz Filtered', color='#1f77b4', alpha=0.85)
    plt.plot(time_axis, sig_40, label='0.5–40 Hz Filtered', color='#ff7f0e', alpha=0.6, linestyle='--')
    plt.title("Case 1 Steady-State 20-Second Window (MAC = 2.20)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Voltage (µV)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Subplot 2: Power Spectral Density comparison
    plt.subplot(2, 1, 2)
    plt.plot(spectra[30.0]['freqs'], 10 * np.log10(spectra[30.0]['psd'] + 1e-12), 
             label=f"30 Hz Cutoff (Beta = {results[30.0]['Spectral Exponent (Beta)']:.2f})", color='#1f77b4')
    plt.plot(spectra[40.0]['freqs'], 10 * np.log10(spectra[40.0]['psd'] + 1e-12), 
             label=f"40 Hz Cutoff (Beta = {results[40.0]['Spectral Exponent (Beta)']:.2f})", color='#ff7f0e', linestyle='--')
    plt.title("Multitaper Power Spectral Density (PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB/Hz)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("case1_feature_benchmark.png", dpi=300)
    print("Saved diagnostic plot to 'case1_feature_benchmark.png'.")