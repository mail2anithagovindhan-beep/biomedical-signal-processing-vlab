"""
Experiment 10 - Adaptive Noise Canceller (LMS Algorithm)

Simulates an Adaptive Noise Canceller (ANC): a chosen biomedical signal is
corrupted by a two-tone reference-correlated interference, and an
adaptive FIR filter (updated via the Least-Mean-Squares algorithm)
attempts to recover the original signal using only the noisy signal and
the reference interference.

Note: the original source lab exercise for this experiment analyzes a
recorded PPG dataset (PPG02.csv) as the primary signal. No such dataset
was provided for this lab, so this page instead uses this app's synthetic
PPG generator (utils/signal_generators.py) as an educational stand-in -
the LMS adaptive-filtering algorithm itself is unchanged from the source
exercise.

UI logic lives here; the LMS algorithm lives in utils/anc_utils.py, so
the page stays decoupled from the underlying computation (mirrors the
structure of the other experiments in this lab).
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.anc_utils import generate_reference_noise, lms_adaptive_filter, noise_reduction_db
from utils.signal_generators import SIGNAL_CATALOG, generate_signal

st.set_page_config(page_title="Exp 10 - Adaptive Noise Canceller", page_icon="🧪", layout="wide")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Experiment 10 — Adaptive Noise Canceller (LMS Algorithm)")
st.markdown("---")

st.markdown("### Aim")
st.write(
    "To simulate an Adaptive Noise Canceller using the Least-Mean-Squares "
    "(LMS) algorithm, and to study how well it recovers a wanted "
    "biomedical signal corrupted by reference-correlated interference."
)

st.markdown("### Learning Objectives")
st.markdown(
    """
    By the end of this experiment, students should be able to:

    - Explain the two-input (primary + reference) structure of an Adaptive Noise Canceller.
    - State the LMS weight-update rule and identify the role of the step size (mu).
    - Explain why the reference input must be correlated with the noise but not with the wanted signal.
    - Evaluate how filter order and step size affect convergence speed, stability, and cancellation quality.
    """
)

with st.expander("Theory (click to expand)"):
    st.markdown(
        """
        An **Adaptive Noise Canceller (ANC)** has two inputs: a
        **primary** input z(n) = s(n) + v(n), containing the wanted
        signal s(n) corrupted by interference v(n); and a **reference**
        input r(n), correlated with v(n) but not with s(n) (for example,
        picked up by a second sensor placed away from the physiological
        signal source).
        """
    )
    st.latex(r"z(n) = s(n) + v(n)")
    st.markdown(
        "An adaptive FIR filter processes r(n) to produce an output "
        "y(n), which is subtracted from the primary input:"
    )
    st.latex(r"e(n) = z(n) - y(n), \qquad y(n) = \mathbf{w}^T(n)\, \mathbf{r}(n)")
    st.markdown(
        "The **Least-Mean-Squares (LMS)** algorithm updates the filter "
        "weights on every sample to drive y(n) toward v(n), so that "
        "e(n) converges toward the wanted signal s(n):"
    )
    st.latex(r"\mathbf{w}(n+1) = \mathbf{w}(n) + 2\mu\, e(n)\, \mathbf{r}(n)")
    st.markdown(
        """
        where `mu` is the **step size** (learning rate): too small and
        the filter adapts slowly; too large and it can become unstable
        (diverge). The filter's **order** (number of taps) sets how much
        of the reference signal's recent history the filter can use.

        In this simulation, the reference and interference frequencies
        are close to the wanted signal's own natural frequency content
        (as is often the case for real physiological interference), so
        cancellation will typically be **partial rather than perfect** -
        this is an authentic limitation of adaptive noise cancellation
        in practice, not a bug in the simulation.
        """
    )

st.markdown("### Pre-Lab Questions")
st.markdown(
    """
    1. What are the two inputs to an Adaptive Noise Canceller, and what must be true of each relative to the wanted signal?
    2. Write the LMS weight-update equation and identify each term.
    3. What happens if the step size mu is set too large?
    4. Why can't perfect cancellation be guaranteed if the reference interference overlaps in frequency with the wanted signal?
    5. What is the "error signal" e(n) in an ANC, and why is it also the system's useful output?
    """
)

st.markdown("---")
st.info("Use the controls in the sidebar to set the signal, interference, and LMS parameters, then click **Run Adaptive Noise Canceller**.")

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Signal Controls")

bio_signals = list(SIGNAL_CATALOG["Biomedical Signals"].keys())
default_index = bio_signals.index("PPG (Photoplethysmogram)") if "PPG (Photoplethysmogram)" in bio_signals else 0
signal_type = st.sidebar.selectbox("Biomedical Signal", bio_signals, index=default_index, key="exp10_signal")

amplitude = st.sidebar.slider("Amplitude", 0.1, 5.0, 1.0, 0.1, key="exp10_amp")
duration = st.sidebar.slider("Duration (s)", 2.0, 30.0, 8.0, 1.0, key="exp10_dur")
sampling_freq = st.sidebar.slider("Sampling Frequency (Hz)", 100, 1000, 500, 50, key="exp10_fs")

st.sidebar.markdown("---")
st.sidebar.header("Reference Interference")
freq1 = st.sidebar.slider("Tone 1 Frequency (Hz)", 0.1, 5.0, 1.5, 0.1, key="exp10_freq1")
amp1 = st.sidebar.slider("Tone 1 Amplitude", 0.0, 1.0, 0.1, 0.05, key="exp10_amp1")
freq2 = st.sidebar.slider("Tone 2 Frequency (Hz)", 0.1, 5.0, 1.25, 0.1, key="exp10_freq2")
amp2 = st.sidebar.slider("Tone 2 Amplitude", 0.0, 1.0, 0.4, 0.05, key="exp10_amp2")

st.sidebar.markdown("---")
st.sidebar.header("LMS Filter Parameters")
mu = st.sidebar.slider("Step Size (mu)", 0.0005, 0.05, 0.005, 0.0005, format="%.4f", key="exp10_mu")
filter_order = st.sidebar.slider("Filter Order (taps)", 4, 150, 16, 1, key="exp10_order")

run_clicked = st.sidebar.button("Run Adaptive Noise Canceller", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Compute, store in session state
# --------------------------------------------------------------------------
if "exp10_result" not in st.session_state:
    st.session_state.exp10_result = None

if run_clicked:
    t, clean, _ = generate_signal(
        "Biomedical Signals", signal_type, amplitude=amplitude, duration=duration,
        fs=sampling_freq, frequency=None,
    )
    reference = generate_reference_noise(t, freq1=freq1, amp1=amp1, freq2=freq2, amp2=amp2)
    noisy = clean + reference

    output, error, weights = lms_adaptive_filter(noisy, reference, mu=mu, filter_order=filter_order)
    diverged = not np.all(np.isfinite(error))

    db_gain = None if diverged else noise_reduction_db(noisy, error, clean)

    st.session_state.exp10_result = {
        "t": t,
        "signal_type": signal_type,
        "clean": clean,
        "reference": reference,
        "noisy": noisy,
        "output": output,
        "error": error,
        "weights": weights,
        "mu": mu,
        "filter_order": filter_order,
        "diverged": diverged,
        "db_gain": db_gain,
    }

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.exp10_result

if result is not None:
    st.markdown("## Results")

    if result["diverged"]:
        st.error(
            "The adaptive filter diverged (its output grew without bound) "
            "with these settings. This is expected LMS behavior when the "
            "step size (mu) is too large for the given filter order and "
            "signal power - try reducing mu and running again."
        )
    else:
        t = result["t"]

        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(t, result["clean"], color="#1f6feb")
        axes[0].set_title(f"Clean {result['signal_type']} (Wanted Signal)")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(t, result["noisy"], color="#cf222e")
        axes[1].set_title("Noisy Primary Input z(n) = Clean Signal + Interference")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        axes[2].plot(t, result["reference"], color="#9a6700")
        axes[2].set_title("Reference Interference r(n)")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        axes[3].plot(t, result["error"], color="#1a7f37")
        axes[3].set_title("Filtered Output e(n) (Recovered Signal Estimate)")
        axes[3].set_xlabel("Time (s)")
        axes[3].grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        metric_cols = st.columns(3)
        rms_before = float(np.sqrt(np.mean((result["noisy"] - result["clean"]) ** 2)))
        rms_after = float(np.sqrt(np.mean((result["error"] - result["clean"]) ** 2)))
        with metric_cols[0]:
            st.metric("RMS Error Before Cancellation", f"{rms_before:.4f}")
        with metric_cols[1]:
            st.metric("RMS Error After Cancellation", f"{rms_after:.4f}")
        with metric_cols[2]:
            st.metric("Noise Reduction", f"{result['db_gain']:.2f} dB")

        if result["db_gain"] > 0.5:
            st.success(
                "The adaptive filter measurably reduced the error relative "
                "to the clean signal, demonstrating adaptive noise "
                "cancellation. Because the interference tones are close in "
                "frequency to the wanted signal's own natural rhythm, "
                "cancellation is partial rather than perfect - this is "
                "expected and matches real-world ANC behavior."
            )
        else:
            st.warning(
                "The adaptive filter did not meaningfully improve on the "
                "noisy signal with these settings. Try increasing the step "
                "size slightly, reducing the filter order, or choosing "
                "interference tones further from the signal's own natural "
                "frequency content."
            )

        st.markdown("#### Observation")
        st.write(
            "As the LMS algorithm processes more samples, its weights "
            "adapt to approximate the coupling between the reference "
            "interference and the noise present in the primary signal, "
            "so the filtered output e(n) should visibly track the shape "
            "of the clean signal more closely than the raw noisy input "
            "does - especially toward the later part of the recording, "
            "once the filter has had time to converge."
        )

        st.markdown("#### Biomedical Relevance")
        st.write(
            "Adaptive noise cancellation is used in biomedical monitoring "
            "wherever an interference source can be sensed separately "
            "from the wanted physiological signal - for example, removing "
            "maternal ECG interference from a fetal ECG recording, or "
            "removing motion-artifact interference from a wearable PPG "
            "(pulse oximetry) sensor using an accelerometer as the "
            "reference input."
        )

    st.markdown("---")

st.markdown("### Post-Lab Questions")
st.markdown(
    """
    1. What would happen if the reference input were identical to the wanted signal instead of the interference?
    2. How did increasing the filter order change the cancellation quality and computation cost?
    3. What happened when the step size mu was made very large? Why?
    4. Why is perfect cancellation not achieved when the interference frequencies are close to the wanted signal's own frequency content?
    5. Name one real biomedical application of adaptive noise cancellation other than the one used in this experiment.
    """
)

st.markdown("### Result")
st.success(
    "Thus, an Adaptive Noise Canceller was simulated using the LMS "
    "algorithm, and its ability to recover a wanted biomedical signal "
    "from a reference-correlated interference was studied for different "
    "step sizes and filter orders."
)
