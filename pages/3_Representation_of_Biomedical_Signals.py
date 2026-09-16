"""
Experiment 3 - Representation of Biomedical Signals

Lets students view a synthetic biomedical signal (reusing the generators
from Experiment 1) in three complementary representations: the time
domain, the frequency domain (magnitude spectrum via FFT), and the
time-frequency domain (spectrogram).

UI logic lives here; signal generation math lives in
utils/signal_generators.py and spectral analysis math lives in
utils/spectral_utils.py, so the page stays decoupled from the underlying
computations (mirrors the structure of Experiments 1 and 2).
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.signal_generators import SIGNAL_CATALOG, generate_signal
from utils.spectral_utils import (
    BAND_INFO,
    compute_magnitude_spectrum,
    compute_spectrogram,
    dominant_frequency,
    prepare_for_spectral_analysis,
)

st.set_page_config(page_title="Exp 3 - Representation of Biomedical Signals", page_icon="🧪", layout="wide")

BIOMEDICAL_SIGNALS = SIGNAL_CATALOG["Biomedical Signals"]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Experiment 3 — Representation of Biomedical Signals")
st.markdown("---")

st.markdown("### Aim")
st.write(
    "To represent biomedical signals in both the time domain and the "
    "frequency domain, and to understand what the frequency-domain view "
    "reveals that is not obvious from the waveform alone."
)

st.markdown("### Learning Objectives")
st.markdown(
    """
    By the end of this experiment, students should be able to:

    - Distinguish between a time-domain representation and a frequency-domain representation of a signal.
    - Compute and interpret the magnitude spectrum of a signal using the Fast Fourier Transform (FFT).
    - Identify the dominant frequency content of ECG, EEG, EMG, and EOG signals.
    - Interpret a spectrogram as a way of tracking how frequency content changes over time.
    - Explain why frequency-domain analysis is clinically useful for biomedical signals.
    """
)

with st.expander("Theory (click to expand)"):
    st.markdown(
        "A signal x(t) can be represented in the **time domain**, as "
        "amplitude varying with time, or in the **frequency domain**, as "
        "the strength of the different frequency components that make it "
        "up. The two are related by the **Fourier Transform**:"
    )
    st.latex(r"X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-j 2\pi f t}\, dt")
    st.markdown(
        """
        For a digitally sampled signal, this is computed using the
        **Discrete Fourier Transform (DFT)**, efficiently implemented as
        the **Fast Fourier Transform (FFT)**. Plotting the magnitude
        |X(f)| against frequency f gives the **magnitude spectrum**,
        showing which frequencies are present and how strong each one is.

        A single magnitude spectrum describes the *average* frequency
        content over the whole signal duration, but does not show *when*
        each frequency occurred. A **spectrogram** addresses this by
        computing the spectrum over many short, overlapping windows of the
        signal and displaying the result as frequency vs. time, with color
        representing power — a time-frequency view.

        Biomedical signals are especially suited to frequency-domain
        analysis because different physiological processes tend to
        produce energy in characteristic frequency ranges (EEG rhythms,
        for example, are classified into named frequency bands), and
        because pathological or fatigue-related changes often appear
        first as a shift in frequency content rather than a change in the
        raw waveform shape.
        """
    )

st.markdown("### Pre-Lab Questions")
st.markdown(
    """
    1. What is the difference between a time-domain and a frequency-domain representation of a signal?
    2. What does the Fast Fourier Transform (FFT) compute?
    3. What information does a spectrogram show that a single magnitude spectrum does not?
    4. Which EEG frequency band would you expect to dominate during deep relaxation?
    5. Why might frequency-domain analysis reveal muscle fatigue in an EMG signal before it becomes visible in the raw waveform?
    """
)

st.markdown("---")
st.info("Use the controls in the sidebar to choose a biomedical signal and click **Analyze Signal**.")

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Signal Controls")

signal_type = st.sidebar.selectbox("Biomedical Signal", list(BIOMEDICAL_SIGNALS.keys()))

amplitude = st.sidebar.slider("Amplitude (a.u.)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
duration = st.sidebar.slider("Duration (s)", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
fs = st.sidebar.slider("Sampling Frequency (Hz)", min_value=100, max_value=2000, value=500, step=50)

analyze_clicked = st.sidebar.button("Analyze Signal", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Generate + analyze, store in session state (persists until the button is
# pressed again)
# --------------------------------------------------------------------------
if "exp3_result" not in st.session_state:
    st.session_state.exp3_result = None

if analyze_clicked:
    t, x, cfg = generate_signal(
        category="Biomedical Signals",
        signal_type=signal_type,
        amplitude=amplitude,
        duration=duration,
        fs=fs,
    )
    x = prepare_for_spectral_analysis(signal_type, x, fs)
    freqs, magnitude = compute_magnitude_spectrum(x, fs)
    f_spec, t_spec, sxx = compute_spectrogram(x, fs)
    dom_freq = dominant_frequency(freqs, magnitude)

    st.session_state.exp3_result = {
        "t": t,
        "x": x,
        "cfg": cfg,
        "signal_type": signal_type,
        "amplitude": amplitude,
        "duration": duration,
        "fs": fs,
        "freqs": freqs,
        "magnitude": magnitude,
        "f_spec": f_spec,
        "t_spec": t_spec,
        "sxx": sxx,
        "dom_freq": dom_freq,
    }

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.exp3_result

if result is not None:
    st.markdown("## Results")

    r_t, r_x, r_cfg = result["t"], result["x"], result["cfg"]

    st.markdown(f"#### {result['signal_type']}")
    st.caption(f"Formula: {r_cfg.formula}")

    st.warning(
        "**Educational synthetic biomedical signal.** "
        "These signals are synthetic representations intended for "
        "educational visualization and are not clinical diagnostic data."
    )

    if result["signal_type"] == "EMG (Electromyogram)":
        st.caption(
            "Note: for this experiment, the EMG signal has been band-limited "
            "to approximately 20-450 Hz (the typical surface EMG range) "
            "before analysis, since unfiltered random noise would not "
            "meaningfully resemble real muscle activity in the frequency "
            "domain."
        )

    # --- Time domain plot --------------------------------------------------
    st.markdown("#### Time-Domain Representation")
    fig_t, ax_t = plt.subplots(figsize=(11, 3.5))
    ax_t.plot(r_t, r_x, color="#1f6feb", linewidth=1.2)
    ax_t.set_title(f"{result['signal_type']} — Time Domain")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Amplitude (a.u.)")
    ax_t.grid(True, linestyle="--", alpha=0.5)
    ax_t.axhline(0, color="black", linewidth=0.6)
    fig_t.tight_layout()
    st.pyplot(fig_t, use_container_width=True)
    plt.close(fig_t)

    # --- Frequency domain plot ----------------------------------------------
    st.markdown("#### Frequency-Domain Representation (Magnitude Spectrum)")
    freqs, magnitude = result["freqs"], result["magnitude"]
    # Limit the displayed range to where biomedical signal energy actually
    # lives, so the plot isn't dominated by empty high-frequency space.
    display_max_freq = min(float(freqs[-1]), 100.0) if len(freqs) else 1.0
    if display_max_freq <= 0:
        display_max_freq = 1.0

    fig_f, ax_f = plt.subplots(figsize=(11, 3.5))
    ax_f.plot(freqs, magnitude, color="#d1242f", linewidth=1.2)
    ax_f.set_title(f"{result['signal_type']} — Magnitude Spectrum")
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Magnitude (a.u.)")
    ax_f.set_xlim(0, display_max_freq)
    ax_f.grid(True, linestyle="--", alpha=0.5)
    fig_f.tight_layout()
    st.pyplot(fig_f, use_container_width=True)
    plt.close(fig_f)

    # --- Spectrogram ---------------------------------------------------------
    st.markdown("#### Time-Frequency Representation (Spectrogram)")
    f_spec, t_spec, sxx = result["f_spec"], result["t_spec"], result["sxx"]
    if sxx.size > 0:
        freq_mask = f_spec <= 100.0
        if not np.any(freq_mask):
            freq_mask = np.ones_like(f_spec, dtype=bool)

        fig_s, ax_s = plt.subplots(figsize=(11, 3.8))
        pcm = ax_s.pcolormesh(
            t_spec,
            f_spec[freq_mask],
            10 * np.log10(sxx[freq_mask] + 1e-12),
            shading="gouraud",
            cmap="viridis",
        )
        ax_s.set_title(f"{result['signal_type']} — Spectrogram")
        ax_s.set_xlabel("Time (s)")
        ax_s.set_ylabel("Frequency (Hz)")
        fig_s.colorbar(pcm, ax=ax_s, label="Power (dB)")
        fig_s.tight_layout()
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)
    else:
        st.info("Duration is too short to compute a spectrogram. Increase the duration slider and re-analyze.")

    # --- Signal information --------------------------------------------------
    st.markdown("#### Signal Information")
    info_cols = st.columns(3)
    with info_cols[0]:
        st.metric("Signal Type", result["signal_type"])
        st.metric("Amplitude", f"{result['amplitude']:.2f} a.u.")
    with info_cols[1]:
        st.metric("Duration", f"{result['duration']:.2f} s")
        st.metric("Sampling Frequency", f"{result['fs']} Hz")
    with info_cols[2]:
        st.metric("Number of Samples", f"{len(r_t)}")
        st.metric("Dominant Frequency", f"{result['dom_freq']:.2f} Hz")

    # --- Description -----------------------------------------------------
    st.markdown("#### Signal Description")
    st.write(r_cfg.description)

    # --- Biomedical relevance -----------------------------------------------
    st.markdown("#### Biomedical Relevance")
    st.write(BAND_INFO.get(result["signal_type"], ""))
    st.write(
        "Clinicians and researchers routinely use frequency-domain tools "
        "like these to summarize sleep-stage EEG activity by band power, "
        "detect muscle fatigue as a downward shift in EMG frequency "
        "content, and separate genuine ECG activity from noise and "
        "artifacts that occupy different frequency ranges than the heart's "
        "own electrical activity."
    )

    st.markdown("---")

st.markdown("### Post-Lab Questions")
st.markdown(
    """
    1. Which frequency dominated the magnitude spectrum of the signal you analyzed, and does that match its expected physiological frequency range?
    2. How did the spectrogram of the EMG signal differ from its magnitude spectrum, and what extra information did it reveal?
    3. If you increased the sampling frequency, what would you expect to happen to the frequency range covered by the magnitude spectrum?
    4. Why is a single "snapshot" magnitude spectrum less useful than a spectrogram for a signal whose frequency content changes over time?
    5. Name one clinical or research scenario where frequency-domain analysis of a biomedical signal would be more informative than looking at its raw waveform.
    """
)

st.markdown("### Result")
st.success(
    "Thus, biomedical signals were represented and analyzed successfully in "
    "the time domain, the frequency domain, and the time-frequency "
    "(spectrogram) domain."
)
