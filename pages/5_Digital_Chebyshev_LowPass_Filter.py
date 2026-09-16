"""
Experiment 5 - Digital Chebyshev Type-I Low-Pass Filter

Lets students design a digital IIR Chebyshev Type-I low-pass filter,
apply it to a signal (reusing the Experiment 1 signal catalog), and
compare its passband ripple and roll-off characteristics against a
Butterworth low-pass filter of the same order and cutoff (Experiment 4).

UI logic lives here; signal generation math lives in
utils/signal_generators.py and filter design/analysis math lives in
utils/filter_utils.py, so the page stays decoupled from the underlying
computations (mirrors the structure of Experiments 1, 3, and 4).
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.signal_generators import SIGNAL_CATALOG, generate_signal
from utils.filter_utils import (
    apply_filter,
    compute_frequency_response,
    design_butterworth,
    design_chebyshev1_lowpass,
    measure_cutoff_frequencies,
)

st.set_page_config(page_title="Exp 5 - Digital Chebyshev Type-I Low-Pass Filter", page_icon="🧪", layout="wide")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Experiment 5 — Digital Chebyshev Type-I Low-Pass Filter")
st.markdown("---")

st.markdown("### Aim")
st.write(
    "To design a digital Chebyshev Type-I low-pass filter, apply it to a "
    "signal, and compare its passband ripple and roll-off sharpness "
    "against a Butterworth filter of the same order and cutoff."
)

st.markdown("### Learning Objectives")
st.markdown(
    """
    By the end of this experiment, students should be able to:

    - Explain the "equiripple passband" property of a Chebyshev Type-I filter.
    - Design a digital Chebyshev Type-I low-pass filter of a given order, ripple, and cutoff.
    - Compare a Chebyshev filter's frequency response against a Butterworth filter of the same order.
    - Explain the practical trade-off between passband ripple and roll-off sharpness.
    """
)

with st.expander("Theory (click to expand)"):
    st.markdown(
        "Unlike a Butterworth filter, which trades a flat passband for a "
        "gentler roll-off, a **Chebyshev Type-I filter** allows a small, "
        "controlled amount of **ripple in the passband** in exchange for a "
        "**sharper transition** to the stopband at the same filter order. "
        "Its magnitude-squared response is:"
    )
    st.latex(r"|H(j\omega)|^2 = \frac{1}{1 + \varepsilon^2\, T_n^2\!\left(\dfrac{\omega}{\omega_c}\right)}")
    st.markdown(
        """
        where *n* is the filter order, T꜀ₙ is the Chebyshev polynomial of
        order *n*, and ε controls the ripple amplitude (related to the
        passband ripple in decibels by ripple_dB = 10·log₁₀(1 + ε²)).

        Because of this ripple, a Chebyshev filter's passband edge is
        **not** defined by the -3 dB point used for Butterworth filters —
        instead, it is defined as the frequency where the response first
        drops to exactly **-ripple_dB below 0 dB gain**, the bottom of the
        equiripple band.

        For the **same filter order**, a Chebyshev Type-I filter achieves
        a steeper roll-off than a Butterworth filter — useful when a sharp
        cutoff is needed without increasing the filter order (and its
        associated phase distortion and computational cost) — at the cost
        of small, tolerable variations in gain within the passband itself.
        """
    )

st.markdown("### Pre-Lab Questions")
st.markdown(
    """
    1. What does "equiripple passband" mean for a Chebyshev Type-I filter?
    2. How is a Chebyshev filter's passband edge defined differently from a Butterworth filter's?
    3. For the same filter order, which filter type gives a sharper roll-off: Butterworth or Chebyshev Type-I?
    4. What is the practical cost of choosing a larger passband ripple value?
    5. Why might a designer prefer a filter with some ripple over one with none, at a given order?
    """
)

st.markdown("---")
st.info("Use the controls in the sidebar to choose a signal and filter settings, then click **Apply Filter**.")

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

order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=4, step=1)
ripple_db = st.sidebar.slider("Passband Ripple (dB)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

cutoff_max = round(nyquist * 0.95, 1)
cutoff = st.sidebar.slider(
    "Cutoff Frequency (Hz)", min_value=0.5, max_value=cutoff_max, value=min(20.0, cutoff_max * 0.5), step=0.5
)

compare_butterworth = st.sidebar.checkbox("Compare with Butterworth filter (same order & cutoff)", value=True)

apply_clicked = st.sidebar.button("Apply Filter", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Generate + filter, store in session state (persists until the button is
# pressed again)
# --------------------------------------------------------------------------
if "exp5_result" not in st.session_state:
    st.session_state.exp5_result = None

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
    x_filtered = freqs_cheby = mag_cheby = freqs_butter = mag_butter = None
    measured_edge = []
    try:
        sos_cheby = design_chebyshev1_lowpass(order, ripple_db, cutoff, fs)
        x_filtered = apply_filter(sos_cheby, x)
        freqs_cheby, mag_cheby = compute_frequency_response(sos_cheby, fs)
        measured_edge = measure_cutoff_frequencies(freqs_cheby, mag_cheby, threshold_db=-ripple_db)

        if compare_butterworth:
            sos_butter = design_butterworth("Low-Pass", order, cutoff, fs)
            freqs_butter, mag_butter = compute_frequency_response(sos_butter, fs)
    except ValueError as exc:
        error_message = str(exc)

    st.session_state.exp5_result = {
        "t": t,
        "x": x,
        "cfg": cfg,
        "signal_type": signal_type,
        "amplitude": amplitude,
        "duration": duration,
        "fs": fs,
        "order": order,
        "ripple_db": ripple_db,
        "cutoff": cutoff,
        "compare_butterworth": compare_butterworth,
        "x_filtered": x_filtered,
        "freqs_cheby": freqs_cheby,
        "mag_cheby": mag_cheby,
        "freqs_butter": freqs_butter,
        "mag_butter": mag_butter,
        "measured_edge": measured_edge,
        "error_message": error_message,
    }

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.exp5_result

if result is not None:
    st.markdown("## Results")

    if result["error_message"] is not None:
        st.error(
            f"Could not design this filter with the chosen settings: {result['error_message']} "
            "Try lowering the filter order or adjusting the cutoff frequency."
        )
    else:
        r_t, r_x, r_cfg = result["t"], result["x"], result["cfg"]
        r_xf = result["x_filtered"]

        st.markdown(f"#### {result['signal_type']} — Chebyshev Type-I Low-Pass, Order {result['order']}")
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
        ax_t.plot(r_t, r_xf, color="#8250df", linewidth=1.3, label="Chebyshev Type-I Filtered")
        ax_t.set_title(f"{result['signal_type']} — Before and After Filtering")
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel("Amplitude (a.u.)")
        ax_t.grid(True, linestyle="--", alpha=0.5)
        ax_t.axhline(0, color="black", linewidth=0.6)
        ax_t.legend(loc="upper right")
        fig_t.tight_layout()
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)

        # --- Frequency response: Chebyshev, optionally vs Butterworth ----------
        st.markdown("#### Filter Frequency Response")
        fig_r, ax_r = plt.subplots(figsize=(11, 4))
        ax_r.plot(result["freqs_cheby"], result["mag_cheby"], color="#8250df", linewidth=1.4,
                   label=f"Chebyshev Type-I (order {result['order']})")
        if result["compare_butterworth"] and result["freqs_butter"] is not None:
            ax_r.plot(result["freqs_butter"], result["mag_butter"], color="#d1242f", linewidth=1.2,
                       linestyle="--", label=f"Butterworth (order {result['order']})")
        ax_r.axhline(-result["ripple_db"], color="black", linewidth=0.8, linestyle=":",
                      label=f"-{result['ripple_db']:.1f} dB (Chebyshev passband edge)")
        ax_r.axvline(result["cutoff"], color="#1a7f37", linewidth=0.9, linestyle=":")
        ax_r.set_title("Chebyshev Type-I vs. Butterworth — Magnitude Response")
        ax_r.set_xlabel("Frequency (Hz)")
        ax_r.set_ylabel("Magnitude (dB)")
        ax_r.set_ylim(bottom=max(-100, float(np.min(result["mag_cheby"]))))
        ax_r.grid(True, linestyle="--", alpha=0.5)
        ax_r.legend(loc="upper right", fontsize=8)
        fig_r.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

        # --- Filter information --------------------------------------------------
        st.markdown("#### Filter Information")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Filter Type", "Chebyshev Type-I (Low-Pass)")
            st.metric("Filter Order", result["order"])
        with info_cols[1]:
            st.metric("Passband Ripple", f"{result['ripple_db']:.1f} dB")
            st.metric("Design Cutoff", f"{result['cutoff']:.1f} Hz")
        with info_cols[2]:
            measured = result["measured_edge"]
            # For a low-pass filter, the passband-to-stopband transition is
            # the highest-frequency crossing; even-order Chebyshev filters
            # can also show a crossing right at 0 Hz (their gain touches
            # exactly -ripple dB there by design), which isn't the edge of
            # interest here.
            measured_str = f"{measured[-1]:.1f} Hz" if measured else "N/A"
            st.metric("Measured Passband Edge", measured_str)
            st.metric("Sampling Frequency", f"{result['fs']} Hz")

        # --- Description -----------------------------------------------------
        st.markdown("#### Signal Description")
        st.write(r_cfg.description)

        # --- Observation -----------------------------------------------------
        st.markdown("#### Observation")
        st.write(
            f"Look closely at the passband (below {result['cutoff']:.1f} Hz) in the "
            f"frequency response plot: the Chebyshev curve ripples slightly "
            f"between 0 dB and -{result['ripple_db']:.1f} dB, while the "
            f"Butterworth curve (if shown) stays smooth and flat. Past the "
            f"cutoff, the Chebyshev curve drops off more sharply than the "
            f"Butterworth curve of the same order — this is the ripple-for-sharpness "
            f"trade-off in action. The measured passband edge should line up "
            f"closely with the dotted green cutoff marker, since it was "
            f"measured at exactly -{result['ripple_db']:.1f} dB, the "
            f"Chebyshev design convention."
        )

        # --- Biomedical relevance -----------------------------------------------
        st.markdown("#### Biomedical Relevance")
        st.write(
            "A sharper roll-off at a given filter order is valuable when a "
            "signal of interest and an unwanted component sit close "
            "together in frequency — for example, separating a biomedical "
            "signal's useful band from nearby noise without resorting to a "
            "much higher-order (and more computationally expensive) "
            "Butterworth filter. The trade-off is that the small ripple in "
            "the passband slightly distorts the relative amplitudes of "
            "frequency components within the signal's own band, which "
            "matters if precise amplitude fidelity is clinically important."
        )

        st.markdown("---")

st.markdown("### Post-Lab Questions")
st.markdown(
    """
    1. How did the passband of the Chebyshev filter differ visually from the Butterworth filter's passband?
    2. Which filter reached -20 dB attenuation closer to the cutoff frequency: Chebyshev or Butterworth, at the same order?
    3. What happened to the ripple pattern when you increased the passband ripple parameter?
    4. Why is the Chebyshev filter's passband edge measured at -ripple_dB rather than -3 dB?
    5. Describe a scenario where you would choose a Chebyshev Type-I filter over a Butterworth filter, and one where you would choose the opposite.
    """
)

st.markdown("### Result")
st.success(
    "Thus, a digital Chebyshev Type-I low-pass filter was designed and "
    "applied successfully, and its ripple and roll-off characteristics "
    "were compared against a Butterworth filter of the same order."
)
