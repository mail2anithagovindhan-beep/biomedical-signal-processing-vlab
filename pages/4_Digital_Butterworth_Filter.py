"""
Experiment 4 - Digital Butterworth Filter

Lets students design a digital IIR Butterworth filter (low-pass,
high-pass, band-pass, or band-stop), apply it to a signal (reusing the
Experiment 1 signal catalog), and observe both the filtered waveform and
the filter's frequency response.

UI logic lives here; signal generation math lives in
utils/signal_generators.py and filter design/analysis math lives in
utils/filter_utils.py, so the page stays decoupled from the underlying
computations (mirrors the structure of Experiments 1 and 3).
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.signal_generators import SIGNAL_CATALOG, generate_signal
from utils.filter_utils import (
    FILTER_RELEVANCE,
    apply_filter,
    compute_frequency_response,
    design_butterworth,
    measure_cutoff_frequencies,
)

st.set_page_config(page_title="Exp 4 - Digital Butterworth Filter", page_icon="🧪", layout="wide")

FILTER_TYPES = ["Low-Pass", "High-Pass", "Band-Pass", "Band-Stop"]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Experiment 4 — Digital Butterworth Filter")
st.markdown("---")

st.markdown("### Aim")
st.write(
    "To design a digital Butterworth filter (low-pass, high-pass, "
    "band-pass, or band-stop), apply it to a signal, and study how filter "
    "order and cutoff frequency shape the result."
)

st.markdown("### Learning Objectives")
st.markdown(
    """
    By the end of this experiment, students should be able to:

    - Explain the defining "maximally flat passband" property of a Butterworth filter.
    - Design a digital Butterworth filter of a given type, order, and cutoff frequency.
    - Interpret a filter's magnitude response plot, including passband, stopband, and roll-off.
    - Explain how increasing filter order affects roll-off steepness.
    - Identify practical biomedical uses for each filter type.
    """
)

with st.expander("Theory (click to expand)"):
    st.markdown(
        "A **Butterworth filter** is designed to have a **maximally flat "
        "magnitude response in the passband** — it has no ripple, at the "
        "cost of a less sharp transition (roll-off) compared to filters "
        "such as the Chebyshev filter studied in Experiment 5. Its "
        "continuous-time magnitude-squared response is:"
    )
    st.latex(r"|H(j\omega)|^2 = \frac{1}{1 + \left(\dfrac{\omega}{\omega_c}\right)^{2n}}")
    st.markdown(
        """
        where ω꜀ is the cutoff frequency and *n* is the filter **order**.
        As *n* increases, the response approaches an ideal "brick-wall"
        filter more closely: the passband stays flatter and the
        transition from passband to stopband becomes steeper, but higher
        orders also introduce more phase distortion and computational
        cost.

        To implement this digitally, SciPy's `butter` function designs the
        equivalent **digital IIR filter** directly for a given sampling
        frequency (internally using the **bilinear transform** with
        frequency pre-warping), returned here as **second-order sections
        (SOS)** for numerically stable filtering.

        Four filter types are available:

        - **Low-Pass** — passes frequencies below the cutoff, attenuates frequencies above it.
        - **High-Pass** — passes frequencies above the cutoff, attenuates frequencies below it.
        - **Band-Pass** — passes a range of frequencies between two cutoffs, attenuates everything else.
        - **Band-Stop (notch)** — attenuates a range of frequencies between two cutoffs, passes everything else.

        The filter in this experiment is applied **causally** (a single
        forward pass), the way a real-time digital filter processes data
        as it arrives — so, unlike a zero-phase filter, it introduces a
        small phase delay along with its magnitude effect.
        """
    )

st.markdown("### Pre-Lab Questions")
st.markdown(
    """
    1. What does "maximally flat passband" mean for a Butterworth filter?
    2. How does increasing the filter order affect the sharpness of the roll-off?
    3. What is the difference between a low-pass and a high-pass filter?
    4. Why would a band-stop (notch) filter be useful for removing power-line interference?
    5. What is the role of the sampling frequency in designing a digital filter?
    """
)

st.markdown("---")
st.info("Use the controls in the sidebar to choose a signal and filter, then click **Apply Filter**.")

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
    frequency = st.sidebar.slider("Signal Frequency (Hz)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
else:
    st.sidebar.caption("Frequency is not applicable for this signal and has been hidden.")

rate = None
if config.needs_rate:
    rate = st.sidebar.slider("Growth / Decay Rate  k  (1/s)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)

duration = st.sidebar.slider("Duration (s)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
fs = st.sidebar.slider("Sampling Frequency (Hz)", min_value=100, max_value=2000, value=500, step=10)
nyquist = fs / 2.0

st.sidebar.markdown("---")
st.sidebar.header("Filter Controls")

filter_type = st.sidebar.selectbox("Filter Type", FILTER_TYPES)
order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=4, step=1)

cutoff_max = round(nyquist * 0.95, 1)
if filter_type in ("Low-Pass", "High-Pass"):
    default_cutoff = min(20.0, cutoff_max * 0.5)
    cutoff = st.sidebar.slider(
        "Cutoff Frequency (Hz)", min_value=0.5, max_value=cutoff_max, value=default_cutoff, step=0.5
    )
else:
    default_low = max(0.5, cutoff_max * 0.2)
    default_high = max(default_low + 1.0, cutoff_max * 0.6)
    cutoff_range = st.sidebar.slider(
        "Cutoff Range (Hz)", min_value=0.5, max_value=cutoff_max, value=(default_low, default_high), step=0.5
    )
    cutoff = tuple(cutoff_range)

apply_clicked = st.sidebar.button("Apply Filter", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Generate + filter, store in session state (persists until the button is
# pressed again)
# --------------------------------------------------------------------------
if "exp4_result" not in st.session_state:
    st.session_state.exp4_result = None

if apply_clicked:
    t, x, cfg = generate_signal(
        category=category,
        signal_type=signal_type,
        amplitude=amplitude,
        duration=duration,
        fs=fs,
        frequency=frequency,
        rate=rate,
    )

    error_message = None
    x_filtered = freqs_resp = mag_db = None
    measured_cutoffs = []
    try:
        sos = design_butterworth(filter_type, order, cutoff, fs)
        x_filtered = apply_filter(sos, x)
        freqs_resp, mag_db = compute_frequency_response(sos, fs)
        measured_cutoffs = measure_cutoff_frequencies(freqs_resp, mag_db)
    except ValueError as exc:
        error_message = str(exc)

    st.session_state.exp4_result = {
        "t": t,
        "x": x,
        "cfg": cfg,
        "category": category,
        "signal_type": signal_type,
        "amplitude": amplitude,
        "frequency": frequency,
        "duration": duration,
        "fs": fs,
        "filter_type": filter_type,
        "order": order,
        "cutoff": cutoff,
        "x_filtered": x_filtered,
        "freqs_resp": freqs_resp,
        "mag_db": mag_db,
        "measured_cutoffs": measured_cutoffs,
        "error_message": error_message,
    }

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.exp4_result

if result is not None:
    st.markdown("## Results")

    if result["error_message"] is not None:
        st.error(
            f"Could not design this filter with the chosen settings: {result['error_message']} "
            "Try lowering the filter order or adjusting the cutoff frequency/frequencies."
        )
    else:
        r_t, r_x, r_cfg = result["t"], result["x"], result["cfg"]
        r_xf = result["x_filtered"]

        st.markdown(f"#### {result['signal_type']} — {result['filter_type']} Filter, Order {result['order']}")
        st.caption(f"Signal formula: {r_cfg.formula}")

        if r_cfg.is_biomedical:
            st.warning(
                "**Educational synthetic biomedical signal.** "
                "These signals are synthetic representations intended for "
                "educational visualization and are not clinical diagnostic data."
            )

        # --- Time domain: original vs filtered --------------------------------
        st.markdown("#### Time-Domain: Original vs. Filtered Signal")
        fig_t, ax_t = plt.subplots(figsize=(11, 4))
        ax_t.plot(r_t, r_x, color="#8b949e", linewidth=1.1, label="Original")
        ax_t.plot(r_t, r_xf, color="#1f6feb", linewidth=1.3, label="Filtered")
        ax_t.set_title(f"{result['signal_type']} — Before and After Filtering")
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel("Amplitude (a.u.)")
        ax_t.grid(True, linestyle="--", alpha=0.5)
        ax_t.axhline(0, color="black", linewidth=0.6)
        ax_t.legend(loc="upper right")
        fig_t.tight_layout()
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)

        # --- Frequency response -------------------------------------------------
        st.markdown("#### Filter Frequency Response")
        freqs_resp, mag_db = result["freqs_resp"], result["mag_db"]
        fig_r, ax_r = plt.subplots(figsize=(11, 4))
        ax_r.plot(freqs_resp, mag_db, color="#d1242f", linewidth=1.3)
        ax_r.axhline(-3.0, color="black", linewidth=0.8, linestyle="--", label="-3 dB")
        cutoff_val = result["cutoff"]
        cutoff_list = list(cutoff_val) if isinstance(cutoff_val, tuple) else [cutoff_val]
        for cf in cutoff_list:
            ax_r.axvline(cf, color="#1a7f37", linewidth=0.9, linestyle=":")
        ax_r.set_title(f"{result['filter_type']} Butterworth Filter — Magnitude Response")
        ax_r.set_xlabel("Frequency (Hz)")
        ax_r.set_ylabel("Magnitude (dB)")
        ax_r.set_ylim(bottom=max(-100, float(np.min(mag_db))))
        ax_r.grid(True, linestyle="--", alpha=0.5)
        ax_r.legend(loc="upper right")
        fig_r.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

        # --- Filter information --------------------------------------------------
        st.markdown("#### Filter Information")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Filter Type", result["filter_type"])
            st.metric("Filter Order", result["order"])
        with info_cols[1]:
            design_cutoff_str = (
                f"{cutoff_list[0]:.1f} Hz" if len(cutoff_list) == 1
                else f"{cutoff_list[0]:.1f} – {cutoff_list[1]:.1f} Hz"
            )
            st.metric("Design Cutoff", design_cutoff_str)
            st.metric("Sampling Frequency", f"{result['fs']} Hz")
        with info_cols[2]:
            measured = result["measured_cutoffs"]
            measured_str = ", ".join(f"{m:.1f} Hz" for m in measured) if measured else "N/A"
            st.metric("Measured -3 dB Point(s)", measured_str)
            st.metric("Number of Samples", f"{len(r_t)}")

        # --- Description ---------------------------------------------------
        st.markdown("#### Signal Description")
        st.write(r_cfg.description)

        # --- Observation -----------------------------------------------------
        st.markdown("#### Observation")
        roll_off_note = (
            f"With order {result['order']}, the roll-off is "
            f"{'gentle' if result['order'] <= 2 else 'moderately steep' if result['order'] <= 5 else 'steep'} "
            f"— higher orders push the response closer to an ideal brick-wall "
            f"cutoff, at the cost of a more complex filter."
        )
        st.write(
            f"Compare the grey (original) and blue (filtered) traces above: "
            f"the {result['filter_type'].lower()} filter has "
            f"{'smoothed out fast variations' if result['filter_type'] == 'Low-Pass' else 'removed the slow-moving trend' if result['filter_type'] == 'High-Pass' else 'kept only a band of frequencies' if result['filter_type'] == 'Band-Pass' else 'suppressed a specific band of frequencies'} "
            f"from the signal. {roll_off_note} The measured -3 dB point(s) "
            f"should line up closely with the dotted green cutoff marker(s) "
            f"on the frequency response plot."
        )

        # --- Biomedical relevance -----------------------------------------------
        st.markdown("#### Biomedical Relevance")
        st.write(FILTER_RELEVANCE.get(result["filter_type"], ""))

        st.markdown("---")

st.markdown("### Post-Lab Questions")
st.markdown(
    """
    1. How closely did the measured -3 dB point match your chosen design cutoff frequency?
    2. What happened to the filtered waveform when you increased the filter order significantly? Why?
    3. For the signal you chose, which filter type best removed unwanted content while preserving the signal's useful features?
    4. Why might a very high filter order be undesirable in a real-time biomedical monitoring system?
    5. Give an example of a biomedical scenario where a band-stop (notch) filter would be the right choice, and explain why.
    """
)

st.markdown("### Result")
st.success(
    "Thus, a digital Butterworth filter was designed and applied "
    "successfully, and its effect on the signal was analyzed in both the "
    "time domain and the frequency domain."
)
