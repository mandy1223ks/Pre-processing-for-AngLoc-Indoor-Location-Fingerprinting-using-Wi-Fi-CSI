# Pre-processing-for-AngLoc-Indoor-Location-Fingerprinting-using-Wi-Fi-CSI
AngLoc, an AOA-aware probabilistic indoor localization system using Wi-Fi CSI, can predict the indoor location. Before we construct the offline radio map, it is important to do CSI pre-processing, the techniques which serve as the precondition to attain superior localization performance.

## Data Setup
* end-to-end **MIMO-OFDM** wireless transceiver for IEEE 802.11 n/ac
* 56 subcarriers
* CFR (channel frequency response) of CSI:   $$H(f)=|H(f)|e^{j\angle H(f)}$$  
* CIF (channel implies response) of CSI: $$h(\tau)=\Sigma\alpha_ie^{-j\varphi_{i}\delta(\tau-\tau_i)}$$
* 4 reference points
* 2 TX antennas, and 2 RX antennas
* one position collect 3000 CSI packets

## System Design
### Noise Removal
* tap filtering
### Phase Calibration
* SFO removal
* STO removal
* CFO removal

## Result
We show the CFR and CIR figure, and there are four step figure: original, after tap filtering, after SFO removal and after STO removal.  
https://github.com/mandy1223ks/Pre-processing-for-AngLoc-Indoor-Location-Fingerprinting-using-Wi-Fi-CSI/blob/main/presentation.pdf
