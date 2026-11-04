# Benchmarking Normalized Dispersion Entropy for Intraoperative EEG Depth-of-Anesthesia Monitoring

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dataset: VitalDB](https://img.shields.io/badge/Dataset-VitalDB%20(N%3D5800%2B)-success.svg)](https://vitaldb.net/)
[![Affiliation: TUM](https://img.shields.io/badge/Research-TUM%20Anesthesiology-0065BD.svg)](https://www.tum.de/)

**Authors:** Vaibhav Purbia  
**Institution:** Department of Anesthesiology and Intensive Care Medicine, Technical University of Munich (TUM), Germany | Rajasthan Technical University, Kota, India  

---

## 📌 Overview
Commercial depth-of-anesthesia (DoA) monitors such as the **Bispectral Index (BIS)** rely on proprietary black-box algorithms that introduce multi-second processing delays, suffer from high-frequency electromyographic (EMG) interference ($30\text{--}100\text{ Hz}$), and occasionally drop out during impedance checks ($\text{BIS} = 0.0$).

This repository provides an automated, end-to-end pharmacokinetic steady-state mining and signal processing pipeline across the **VitalDB surgical repository ($N = 5,800+$ cases)**. It benchmarks **Normalized Dispersion Entropy ($\text{DisEn}$)** against classical Multitaper spectral features, the commercial BIS index, and **Permutation Entropy ($\text{PE}$)** during steady-state volatile anesthesia maintenance.

---

## 🚀 Key Empirical Findings ($N = 25$ Stratified Steady-State Plateaus)

Across $25$ artifact-free steady-state maintenance plateaus ($\ge 5\text{ min}$, $\Delta\text{MAC} \le 0.05$) spanning Light ($0.3\text{--}0.6\text{ MAC}$), Medium ($0.7\text{--}1.1\text{ MAC}$), and Deep ($\ge 1.2\text{ MAC}$) tiers:

| Complexity Metric ($0.5\text{--}30\text{ Hz}$) | Smith's $P_K$ (vs. MAC) | Spearman $\rho$ (vs. MAC) | Spearman $\rho$ (vs. BIS) | EMG Shift ($\Delta_{40-30\text{Hz}}$) | Latency (per $20\text{ s}$ epoch) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normalized DisEn ($m=3, c=4, d=1$)** | **$0.697$** | **$-0.496$ ($p = 0.0117$)** | **$+0.783$ ($p < 0.0001$)** | **$+0.0139 \pm 0.0202$** | **$0.226\text{ ms}$** |
| **Permutation Entropy ($m=3, d=1$)** | $0.535$ | $-0.100$ ($p = 0.6329$) | $-0.036$ ($p = 0.8720$) | $+0.0336 \pm 0.0206$ | $0.286\text{ ms}$ |
| **Commercial BIS Monitor (Ground Truth)** | **$0.697$** | — | $1.000$ (Reference) | Proprietary Filter | Multi-second lag |

### Core Highlights
1. **Exact $P_K$ Parity with Commercial BIS:** Under a strict $0.5\text{--}30\text{ Hz}$ passband, $\text{DisEn}$ matches commercial BIS prediction probability ($P_K = 0.697$) and tracks volatile anesthetic concentration monotonically ($\rho = -0.496, p = 0.0117$) without vendor smoothing latency.
2. **Strong Concordance with Ground-Truth BIS:** $\text{DisEn}$ correlates strongly with commercial BIS scores ($\rho = +0.783, p < 0.0001$) while maintaining continuous readout during BIS sensor dropouts.
3. **Why $\text{DisEn}$ Beats $\text{PE}$ ($P_K = 0.697$ vs. $0.535$):** At $f_s = 128\text{ Hz}$ with a $30\text{ Hz}$ low-pass filter, a 3-sample window ($15.6\text{ ms}$) is shorter than the minimum half-cycle of a $30\text{ Hz}$ wave ($16.7\text{ ms}$). Because $\text{PE}$ ignores metric amplitude and only sorts ordinal ranks, almost all windows collapse into strictly rising (`[0,1,2]`) or falling (`[2,1,0]`) motifs regardless of anesthetic depth. $\text{DisEn}$ preserves amplitude dynamics via Normal Cumulative Distribution Function (NCDF) class mapping ($c=4$, $64$ dispersion patterns).
4. **$2.42\times$ Higher EMG Immunity & $1.27\times$ Speedup:** $\text{DisEn}$ buffers micro-volt muscle ripples within NCDF class bins and runs in $\mathcal{O}(N)$ linear time ($0.226\text{ ms}$ vs. $0.286\text{ ms}$ per $20\text{ s}$ epoch).

---

## 📊 Visual Benchmarks

### Phase 1: Stratified Cohort Validation & Filter Sensitivity ($30\text{ Hz}$ vs. $40\text{ Hz}$)
![Phase 1 Cohort Benchmarks](figures/final_project_benchmarks.png)

### Phase 2: Head-to-Head Comparison ($\text{DisEn}$ vs. $\text{PE}$)
![Phase 2 DisEn vs PE Comparison](figures/disen_vs_pe_comparison.png)

---

## 📂 Repository Structure

```text
tum-eeg-disen/
├── data/
│   ├── cohort_features.csv             # Initial steady-state feature extraction
│   ├── stratified_cohort.csv           # Stratified 25-plateau dataset (Light, Medium, Deep MAC)
│   └── stratified_cohort_with_pe.csv   # Enriched dataset with DisEn and PE (30Hz & 40Hz)
├── figures/
│   ├── case1_feature_benchmark.png     # Single-case time series & Multitaper PSD diagnostic
│   ├── cohort_benchmarks.png           # Cohort correlation plots
│   ├── final_project_benchmarks.png    # Phase 1 multi-panel publication figure
│   └── disen_vs_pe_comparison.png      # Phase 2 DisEn vs. PE head-to-head benchmark
├── src/
│   ├── pipeline.py                     # Core VitalDB track ingestion & plateau detection
│   ├── step2_features.py               # NCDF Dispersion Entropy & DPSS Multitaper PSD
│   ├── step3_plot_cohort.py            # Cohort visualization scripts
│   ├── step4_stratified_batch.py       # Automated stratified cohort mining across VitalDB
│   ├── step5_benchmark_stats.py        # Smith's P_K and Spearman rank correlation analysis
│   └── generate_final_report.py        # Multi-panel figure generation
├── benchmark_disen_vs_pe.py            # Vectorized head-to-head DisEn vs. PE benchmark runner
├── find_tracks.py                      # VitalDB track discovery utility
└── requirements.txt                    # Python dependencies