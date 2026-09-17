"""
Experiment 9 - Analysis of ECG Signal

Corrupts a synthetic ECG signal with additive Gaussian noise, applies a
simple moving-average (boxcar) filter to denoise it, and compares the
original, noisy, and filtered signals in both the time domain and the
frequency domain (via FFT magnitude spectra).

Note: the original source lab exercise for this experiment analyzes a
recorded ECG dataset (ECG03.csv). No such dataset was provided for this
lab, so this page instead uses this app's existing synthetic ECG
generator (utils/signal_generators.py, also used in Experiments 1 and 3)
as an educational stand-in - the noise-corruption and moving-average
filtering method is unchanged from the source exercise.

UI logic lives here; the noise/filtering math lives in
utils/moving_average_utils.py and the FFT magnitude spectrum computation
is reused from utils/spectral_utils.py (Experiment 3), so the page stays
decoupled from the underlying computation.
"""

import matplotlib.pyplot as plt
import streamlit as st

from utils.moving_average_utils import add_gaussian_noise, moving_average_filter
from utils.signal_generators import SIGNAL_CATALOG, generate_signal
from utils.spectral_utils import compute_magnitude_spectrum

st.set_page_config(page_title="Exp 9 - ECG Signal Analysis", page_icon="🧪", layout="wide")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Experiment 9 — Analysis of ECG Signal")
st.markdown("---")

st.markdown("### Aim")
st.write(
    "To corrupt a synthetic ECG signal with additive Gaussian noise, "
    "denoise it using a moving-average filter, and analyze the original, "
    "noisy, and filtered signals in both the time and frequency domains."
)

st.markdown("### Learning Objectives")
st.markdown(
    """
    By the end of this experiment, students should be able to:

    - Model measurement/electrical noise as additive Gaussian noise on a biomedical signal.
    - Explain how a moving-average (boxcar) filter smooths a noisy signal.
    - Compare a signal's magnitude spectrum before and after noise corruption and filtering.
    - Evaluate the trade-off between noise reduction and waveform-detail loss as the averaging window widens.
    """
)

with st.expander("Theory (click to expand)"):
    st.markdown(
        """
        Biomedical signal acquisition (e.g. ECG electrodes) is affected
        by random measurement noise, often modeled as **additive white
        Gaussian noise**:
        """
    )
    st.latex(r"z[n] = x[n] + w[n], \qquad w[n] \sim \mathcal{N}(0, \sigma^2)")
    st.markdown(
        """
        A simple way to reduce this noise is a **moving-average (boxcar)
        filter**, which replaces each sample with the average of itself
        and the previous `L-1` samples:
        """
    )
    st.latex(r"y[n] = \frac{1}{L} \sum_{k=0}^{L-1} z[n-k]")
    st.markdown(
        """
        This is a simple FIR low-pass filter: it attenuates high-frequency
        content (including noise) but also smooths - and can blur - sharp
        features of the wanted signal (such as the ECG's R peak) if the
        window length `L` is too large. Comparing the FFT magnitude
        spectrum before and after filtering shows this effect directly:
        the noise's flat, broadband spectrum is suppressed at higher
        frequencies, while the ECG's own low-frequency structure is
        largely preserved.
        """
    )

st.markdown("### Pre-Lab Questions")
st.markdown(
    """
    1. What is meant by "additive white Gaussian noise"?
    2. Why does averaging consecutive samples reduce random noise?
    3. What kind of filter is a moving-average filter (low-pass, high-pass, etc.)?
    4. What happens to sharp features of a signal (like the ECG R peak) if the averaging window is too wide?
    5. How would you expect the FFT magnitude spectrum of noise to differ from the FFT magnitude spectrum of the ECG signal itself?
    """
)

st.markdown("---")
st.info("Use the controls in the sidebar to set the ECG and noise parameters, then click **Analyze Signal**.")

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("ECG and Noise Controls")

amplitude = st.sidebar.slider("Amplitude", 0.1, 5.0, 1.0, 0.1, key="exp9_amp")
heart_rate = st.sidebar.slider("Heart Rate (bpm)", 40.0, 180.0, 72.0, 1.0, key="exp9_hr")
duration = st.sidebar.slider("Duration (s)", 1.0, 10.0, 4.0, 0.5, key="exp9_dur")
sampling_freq = st.sidebar.slider("Sampling Frequency (Hz)", 100, 2000, 500, 50, key="exp9_fs")

st.sidebar.markdown("---")
noise_std = st.sidebar.slider("Noise Std. Deviation", 0.0, 1.0, 0.15, 0.01, key="exp9_noise_std")
window_length = st.sidebar.slider("Moving-Average Window Length (taps)", 1, 50, 8, 1, key="exp9_window")

analyze_clicked = st.sidebar.button("Analyze Signal", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Compute, store in session state
# --------------------------------------------------------------------------
if "exp9_result" not in st.session_state:
    st.session_state.exp9_result = None

if analyze_clicked:
    config = SIGNAL_CATALOG["Biomedical Signals"]["ECG (Electrocardiogram)"]
    t, ecg, _ = generate_signal(
        "Biomedical Signals", "ECG (Electrocardiogram)", amplitude=amplitude,
        duration=duration, fs=sampling_freq, frequency=None,
    )
    noisy, noise = add_gaussian_noise(ecg, noise_std=noise_std, seed=1)
    filtered = moving_average_filter(noisy, window_length=window_length)

    freqs_clean, mag_clean = compute_magnitude_spectrum(ecg, sampling_freq)
    freqs_noisy, mag_noisy = compute_magnitude_spectrum(noisy, sampling_freq)
    freqs_filtered, mag_filtered = compute_magnitude_spectrum(filtered, sampling_freq)

    st.session_state.exp9_result = {
        "t": t,
        "ecg": ecg,
        "noisy": noisy,
        "noise": noise,
        "filtered": filtered,
        "sampling_freq": sampling_freq,
        "window_length": window_length,
        "noise_std": noise_std,
        "freqs_clean": freqs_clean,
        "mag_clean": mag_clean,
        "freqs_noisy": freqs_noisy,
        "mag_noisy": mag_noisy,
        "freqs_filtered": freqs_filtered,
        "mag_filtered": mag_filtered,
    }

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.exp9_result

if result is not None:
    st.markdown("## Results")

    t = result["t"]

    # --- Time-domain: 3 stacked plots ---------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, result["ecg"], color="#1f6feb")
    axes[0].set_title("Original (Clean) Synthetic ECG")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(t, result["noisy"], color="#cf222e")
    axes[1].set_title(f"Noisy ECG (Gaussian noise, σ = {result['noise_std']:.2f})")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(t, result["filtered"], color="#1a7f37")
    axes[2].set_title(f"Filtered ECG (Moving Average, L = {result['window_length']})")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- Frequency-domain: 3 stacked plots -----------------------------
    st.markdown("#### FFT Magnitude Spectra")
    max_freq = min(100.0, result["sampling_freq"] / 2.0)

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, label, freqs, mag, color in [
        (axes2[0], "Clean ECG", result["freqs_clean"], result["mag_clean"], "#1f6feb"),
        (axes2[1], "Noisy ECG", result["freqs_noisy"], result["mag_noisy"], "#cf222e"),
        (axes2[2], "Filtered ECG", result["freqs_filtered"], result["mag_filtered"], "#1a7f37"),
    ]:
        mask = freqs <= max_freq
        ax.plot(freqs[mask], mag[mask], color=color)
        ax.set_title(label)
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes2[0].set_ylabel("Magnitude")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # --- Metrics ---------------------------------------------------------
    import numpy as np

    rms_noisy = float(np.sqrt(np.mean((result["noisy"] - result["ecg"]) ** 2)))
    rms_filtered = float(np.sqrt(np.mean((result["filtered"] - result["ecg"]) ** 2)))

    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.metric("RMS Error (Noisy vs. Clean)", f"{rms_noisy:.4f}")
    with metric_cols[1]:
        st.metric("RMS Error (Filtered vs. Clean)", f"{rms_filtered:.4f}")

    if rms_filtered < rms_noisy:
        st.success(
            "The moving-average filter reduced the RMS error relative to "
            "the clean ECG, demonstrating effective noise suppression."
        )
    else:
        st.warning(
            "The moving-average filter did not reduce the RMS error here "
            "- try a shorter/longer window length or a different noise "
            "level to see the trade-off between noise reduction and "
            "waveform blurring."
        )

    st.markdown("#### Observation")
    st.write(
        "Additive Gaussian noise spreads energy broadly across the "
        "frequency spectrum, visible as a raised noise floor in the "
        "'Noisy ECG' spectrum. The moving-average filter, being a simple "
        "low-pass filter, attenuates this high-frequency energy - "
        "visible as a lower noise floor in the 'Filtered ECG' spectrum - "
        "at the cost of also slightly smoothing (blurring) the ECG's own "
        "sharp QRS complex if the averaging window is made too wide."
    )

    st.markdown("#### Biomedical Relevance")
    st.write(
        "Real ECG acquisition hardware is subject to electrical and "
        "electrode-motion noise. A simple moving-average filter is a "
        "computationally cheap first step used in many wearable and "
        "point-of-care ECG devices to reduce this noise before further "
        "processing (such as QRS detection), though more sophisticated "
        "filters (as designed in Experiments 4, 5, 7, and 8) are used "
        "when sharper frequency selectivity is required."
    )

    st.markdown("---")

st.markdown("### Post-Lab Questions")
st.markdown(
    """
    1. What visibly changed in the FFT magnitude spectrum after adding Gaussian noise?
    2. What visibly changed in the FFT magnitude spectrum after moving-average filtering?
    3. What happens to the RMS error and the sharpness of the R peak as the window length increases?
    4. Why is a moving-average filter considered a type of low-pass filter?
    5. What are the limitations of using a moving-average filter to denoise an ECG signal?
    """
)

st.markdown("### Result")
st.success(
    "Thus, a synthetic ECG signal was corrupted with Gaussian noise, "
    "denoised using a moving-average filter, and the original, noisy, "
    "and filtered signals were compared and analyzed in both the time "
    "and frequency domains."
)
