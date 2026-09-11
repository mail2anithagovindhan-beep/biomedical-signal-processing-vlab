"""
Experiment 1 - Signal Generation

Lets students generate and visualize basic mathematical test signals and
educational synthetic biomedical signals (ECG, EEG, EMG, EOG), adjusting
amplitude, frequency (where applicable), duration, and sampling frequency.

UI logic lives here; signal-generation math lives in
utils/signal_generators.py so the two stay decoupled.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.signal_generators import SIGNAL_CATALOG, generate_signal

st.set_page_config(page_title="Exp 1 - Signal Generation", page_icon="🧪", layout="wide")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Experiment 1 — Signal Generation")
st.markdown("---")

st.markdown("### Aim")
st.write(
    "To generate and visualize basic mathematical signals and introductory "
    "biomedical signals, and to understand how amplitude, frequency, "
    "duration, and sampling frequency affect their appearance."
)

st.markdown("### Learning Objectives")
st.markdown(
    """
    By the end of this experiment, students should be able to:

    - Define and distinguish between periodic and aperiodic signals.
    - Generate standard test signals (sine, cosine, square, triangle, sawtooth, step, impulse, ramp, exponential) from their mathematical definitions.
    - Explain the role of amplitude, frequency, duration, and sampling frequency in describing a signal.
    - Recognize the basic morphology of common biomedical signals (ECG, EEG, EMG, EOG).
    """
)

with st.expander("Theory (click to expand)"):
    st.markdown(
        """
        A **signal** is a function that conveys information about a
        phenomenon, typically expressed as a variation of some quantity
        over time, denoted x(t). A signal is **periodic** if it repeats
        itself after a fixed interval T (the period), so that
        x(t) = x(t + T) for all t; otherwise it is **aperiodic**.

        **Amplitude (A)** describes the strength or magnitude of a signal,
        while **frequency (f)**, measured in Hertz (Hz), describes how
        many cycles occur per second (f = 1/T for a periodic signal).

        When a continuous-time signal is processed digitally, it must be
        **sampled** at a sampling frequency (f_s) — the number of samples
        taken per second. The choice of f_s determines how finely the
        waveform is represented digitally (its importance for avoiding
        aliasing is explored in Experiment 2).

        **Biomedical signals** such as ECG, EEG, EMG, and EOG are
        electrical signals recorded from the body that reflect the
        activity of the heart, brain, muscles, and eyes respectively.
        They are typically more complex than simple mathematical
        waveforms, often combining multiple frequency components and
        characteristic morphological features.
        """
    )

st.markdown("### Pre-Lab Questions")
st.markdown(
    """
    1. What is a periodic signal?
    2. What is the difference between amplitude and frequency?
    3. What is the purpose of a sampling frequency?
    4. Which part of an ECG represents ventricular depolarization?
    5. Why are biomedical signals important in healthcare?
    """
)

st.markdown("---")
st.info("Use the controls in the sidebar to choose a signal and click **Generate Signal**.")

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Signal Controls")

category = st.sidebar.selectbox("Signal Category", list(SIGNAL_CATALOG.keys()))
signal_type = st.sidebar.selectbox("Signal Type", list(SIGNAL_CATALOG[category].keys()))
config = SIGNAL_CATALOG[category][signal_type]

amplitude = st.sidebar.slider("Amplitude (a.u.)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

frequency = None
if config.needs_frequency:
    frequency = st.sidebar.slider("Frequency (Hz)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
else:
    st.sidebar.caption("Frequency is not applicable for this signal and has been hidden.")

rate = None
if config.needs_rate:
    rate = st.sidebar.slider("Growth / Decay Rate  k  (1/s)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    st.sidebar.caption("k > 0: growth, k < 0: decay.")

duration = st.sidebar.slider("Duration (s)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
fs = st.sidebar.slider("Sampling Frequency (Hz)", min_value=10, max_value=2000, value=500, step=10)

generate_clicked = st.sidebar.button("Generate Signal", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Generate & store in session state (persists until the button is pressed again)
# --------------------------------------------------------------------------
if "exp1_result" not in st.session_state:
    st.session_state.exp1_result = None

if generate_clicked:
    t, x, cfg = generate_signal(
        category=category,
        signal_type=signal_type,
        amplitude=amplitude,
        duration=duration,
        fs=fs,
        frequency=frequency,
        rate=rate,
    )
    st.session_state.exp1_result = {
        "t": t,
        "x": x,
        "cfg": cfg,
        "category": category,
        "signal_type": signal_type,
        "amplitude": amplitude,
        "frequency": frequency,
        "rate": rate,
        "duration": duration,
        "fs": fs,
    }

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.exp1_result

if result is not None:
    st.markdown("## Results")

    r_t, r_x, r_cfg = result["t"], result["x"], result["cfg"]

    st.markdown(f"#### {result['signal_type']}")
    st.caption(f"Formula: {r_cfg.formula}")

    if r_cfg.is_biomedical:
        st.warning(
            "**Educational synthetic biomedical signal.** "
            "These signals are synthetic representations intended for "
            "educational visualization and are not clinical diagnostic data."
        )

    # --- Graph -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(r_t, r_x, color="#1f6feb", linewidth=1.4)
    ax.set_title(f"{result['signal_type']} — Time Domain Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- Signal information -------------------------------------------------
    st.markdown("#### Signal Information")
    info_cols = st.columns(3)
    with info_cols[0]:
        st.metric("Signal Type", result["signal_type"])
        st.metric("Amplitude", f"{result['amplitude']:.2f} a.u.")
    with info_cols[1]:
        freq_display = f"{result['frequency']:.2f} Hz" if result["frequency"] is not None else "N/A"
        st.metric("Frequency", freq_display)
        st.metric("Duration", f"{result['duration']:.2f} s")
    with info_cols[2]:
        st.metric("Sampling Frequency", f"{result['fs']} Hz")
        st.metric("Number of Samples", f"{len(r_t)}")

    # --- Description ---------------------------------------------------
    st.markdown("#### Signal Description")
    st.write(r_cfg.description)

    # --- Observation -----------------------------------------------------
    st.markdown("#### Observation")
    if r_cfg.needs_frequency and result["frequency"]:
        period = 1.0 / result["frequency"]
        st.write(
            f"This is a periodic signal that completes one full cycle every "
            f"T = 1/f = {period:.3f} s. Over the selected duration of "
            f"{result['duration']:.2f} s, approximately "
            f"{result['duration'] * result['frequency']:.1f} cycles are visible. "
            f"Increasing the frequency would pack more cycles into the same "
            f"duration; increasing the amplitude would stretch the waveform "
            f"further from the zero line without changing how often it repeats."
        )
    elif r_cfg.needs_rate:
        direction = "grows" if result["rate"] > 0 else ("decays" if result["rate"] < 0 else "remains constant")
        st.write(
            f"With k = {result['rate']:.2f} s⁻¹, the signal {direction} "
            f"exponentially with time. A larger |k| makes the growth or "
            f"decay happen more rapidly."
        )
    elif r_cfg.is_biomedical:
        st.write(
            "Notice how this waveform is more irregular and composite than "
            "the basic mathematical signals — real (and simulated) "
            "biomedical signals combine multiple underlying processes "
            "rather than a single clean frequency. Increasing the duration "
            "shows more repetitions of the underlying pattern; increasing "
            "the amplitude scales the overall signal strength."
        )
    else:
        st.write(
            "Notice the discontinuity/behaviour at t = 0, which is the "
            "defining feature of this signal. Changing the amplitude "
            "scales the signal's height without changing where this "
            "behaviour occurs."
        )

    st.markdown("---")

st.markdown("### Post-Lab Questions")
st.markdown(
    """
    1. How does increasing the frequency affect the number of cycles observed within a fixed duration?
    2. How does the amplitude parameter change the shape of a sine wave versus a unit step?
    3. Which synthetic biomedical signal did you generate, and what real physiological process does it represent?
    4. How did changing the sampling frequency affect the number of samples in your generated signal?
    5. Identify the P wave, QRS complex, and T wave in the synthetic ECG signal you generated.
    """
)

st.markdown("### Result")
st.success(
    "Thus, basic mathematical and synthetic biomedical signals were generated and visualized successfully."
)
