"""
utils/fir_page_render.py
--------------------------
Shared page-rendering logic for Experiment 7 (FIR filter design using a
Hamming window) and Experiment 8 (FIR filter design using a Hanning
window).

The two experiments are structurally identical - both design a windowed
FIR filter using the same empirical order-estimation formula and apply it
to the same catalog of test/biomedical signals - and differ only in which
window function is used. Rather than duplicating ~200 lines of near
-identical Streamlit page code twice, the full page is rendered here once
and each page file (pages/07_FIR_Filter_Hamming_Window.py and
pages/08_FIR_Filter_Hanning_Window.py) simply calls
`render_fir_experiment(...)` with its own experiment number and window
name.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.fir_filter_utils import (
    BTYPE_MAP,
    FIR_RELEVANCE,
    apply_fir_filter,
    compute_frequency_response,
    design_fir_filter,
    estimate_fir_order,
)
from utils.filter_utils import measure_cutoff_frequencies
from utils.signal_generators import SIGNAL_CATALOG, generate_signal


def render_fir_experiment(experiment_number: int, window: str) -> None:
    """Render the full Experiment 7 or 8 page for the given window type.

    Parameters
    ----------
    experiment_number : int
        7 or 8, used in the page title and session-state key.
    window : str
        "Hamming" or "Hanning".
    """
    session_key = f"exp{experiment_number}_result"

    st.set_page_config(
        page_title=f"Exp {experiment_number} - FIR Filter ({window} Window)",
        page_icon="🧪",
        layout="wide",
    )

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    st.title(f"Experiment {experiment_number} — Design of Digital FIR Filter Using {window} Window")
    st.markdown("---")

    st.markdown("### Aim")
    st.write(
        f"To design a digital Finite Impulse Response (FIR) filter using "
        f"the windowing method with a **{window} window**, and to study "
        f"its magnitude response and effect on a test or biomedical signal."
    )

    st.markdown("### Learning Objectives")
    st.markdown(
        f"""
        By the end of this experiment, students should be able to:

        - Explain the windowing method of FIR filter design.
        - Estimate the required filter length from passband/stopband ripple and edge-frequency specifications.
        - Design Low-Pass, High-Pass, Band-Pass, and Band-Stop FIR filters using a {window} window.
        - Explain why windowed FIR filters have linear phase, and why that matters for biomedical signals.
        - Compare a windowed FIR filter's magnitude response against its design specification.
        """
    )

    with st.expander("Theory (click to expand)"):
        st.markdown(
            f"""
            An **FIR (Finite Impulse Response) filter** produces its output
            as a weighted sum of a finite number of past input samples:
            """
        )
        st.latex(r"y[n] = \sum_{k=0}^{M-1} h[k]\, x[n-k]")
        st.markdown(
            f"""
            where `h[k]` are the filter's *tap coefficients* and `M` is
            the filter length (number of taps).

            The **windowing method** designs `h[k]` by taking the
            (infinite, non-causal) ideal impulse response of the desired
            filter and multiplying it by a finite-length **window
            function** — here, the **{window} window** — to produce a
            practical, finite-length filter. The window's shape trades
            off the width of the transition band against the amount of
            ripple in the passband/stopband.

            A commonly used empirical formula estimates the required
            filter length `M` (forced to the nearest odd value for a
            Type I linear-phase filter) from the ripple specifications
            `rp` (passband ripple), `rs` (stopband ripple, both as linear
            fractions), and the edge frequencies `fp` (passband edge) and
            `fs` (stopband edge), relative to the sampling frequency `F`:
            """
        )
        st.latex(
            r"M = \left\lceil \dfrac{-20\log_{10}\!\sqrt{r_p r_s} - 13}"
            r"{14.6\, (f_{stop} - f_p)/F} \right\rceil \quad \text{(forced odd)}"
        )
        st.markdown(
            """
            Because an FIR filter designed this way is symmetric
            (Type I), it has an exactly **linear phase response** — every
            frequency component is delayed by the same amount of time.
            This is important for biomedical signals, where distorting
            the relative timing of waveform features (such as an ECG's
            QRS complex) can obscure clinically meaningful information -
            an IIR filter with a comparably sharp roll-off would not
            offer this guarantee.
            """
        )

    st.markdown("### Pre-Lab Questions")
    st.markdown(
        f"""
        1. What is the difference between an FIR filter and an IIR filter?
        2. Why does the windowing method multiply the ideal impulse response by a window function?
        3. What is the defining property of a Type I linear-phase FIR filter?
        4. What is the effect of a {window} window on the filter's transition width and stopband attenuation?
        5. Why must the FIR filter length be odd for a Type I linear-phase design?
        """
    )

    st.markdown("---")
    st.info("Use the controls in the sidebar to set the filter specifications, choose a signal, then click **Design and Apply Filter**.")

    # ------------------------------------------------------------------
    # Sidebar controls
    # ------------------------------------------------------------------
    st.sidebar.header("Signal Controls")
    category = st.sidebar.selectbox(
        "Signal Category", list(SIGNAL_CATALOG.keys()), key=f"exp{experiment_number}_category"
    )
    signal_type = st.sidebar.selectbox(
        "Signal Type", list(SIGNAL_CATALOG[category].keys()), key=f"exp{experiment_number}_signal"
    )
    config = SIGNAL_CATALOG[category][signal_type]

    amplitude = st.sidebar.slider(
        "Amplitude", 0.1, 5.0, 1.0, 0.1, key=f"exp{experiment_number}_amp"
    )
    frequency = None
    if config.needs_frequency:
        frequency = st.sidebar.slider(
            "Signal Frequency (Hz)", 0.5, 50.0, 5.0, 0.5, key=f"exp{experiment_number}_freq"
        )
    duration = st.sidebar.slider(
        "Duration (s)", 0.5, 10.0, 4.0, 0.5, key=f"exp{experiment_number}_dur"
    )
    sampling_freq = st.sidebar.slider(
        "Sampling Frequency F (Hz)", 200, 8000, 2000, 100, key=f"exp{experiment_number}_fs"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("FIR Filter Specifications")

    filter_type = st.sidebar.selectbox(
        "Filter Type", list(BTYPE_MAP.keys()), key=f"exp{experiment_number}_ftype"
    )
    rp = st.sidebar.number_input(
        "Passband Ripple, rp (linear)", 0.001, 0.5, 0.01, 0.001, format="%.3f",
        key=f"exp{experiment_number}_rp",
    )
    rs = st.sidebar.number_input(
        "Stopband Ripple, rs (linear)", 0.001, 0.5, 0.01, 0.001, format="%.3f",
        key=f"exp{experiment_number}_rs",
    )

    nyquist = sampling_freq / 2.0
    if filter_type in ("Low-Pass", "High-Pass"):
        fp = st.sidebar.slider(
            "Passband Edge fp (Hz)", 1.0, nyquist - 2.0, min(nyquist * 0.25, nyquist - 2.0),
            key=f"exp{experiment_number}_fp",
        )
        fstop = st.sidebar.slider(
            "Stopband Edge fs (Hz)", 1.0, nyquist - 1.0, min(nyquist * 0.4, nyquist - 1.0),
            key=f"exp{experiment_number}_fstop",
        )
        cutoff = (fp + fstop) / 2.0
        band_edges = (fp, fstop)
    else:
        f1 = st.sidebar.slider(
            "Lower Edge (Hz)", 1.0, nyquist - 3.0, min(nyquist * 0.2, nyquist - 3.0),
            key=f"exp{experiment_number}_f1",
        )
        f2 = st.sidebar.slider(
            "Upper Edge (Hz)", f1 + 1.0, nyquist - 1.0, min(nyquist * 0.5, nyquist - 1.0),
            key=f"exp{experiment_number}_f2",
        )
        fp, fstop = f1, f2
        cutoff = (f1, f2)
        band_edges = (f1, f2)

    apply_clicked = st.sidebar.button(
        "Design and Apply Filter", type="primary", use_container_width=True,
        key=f"exp{experiment_number}_apply",
    )

    # ------------------------------------------------------------------
    # Compute, store in session state
    # ------------------------------------------------------------------
    if session_key not in st.session_state:
        st.session_state[session_key] = None

    if apply_clicked:
        t, x, sig_config = generate_signal(
            category, signal_type, amplitude=amplitude, duration=duration,
            fs=sampling_freq, frequency=frequency,
        )

        error_message = None
        order = None
        taps = None
        x_filtered = None
        freqs = mag_db = None
        measured_edges = []

        try:
            order = estimate_fir_order(rp, rs, band_edges[0], band_edges[1], sampling_freq)
            taps = design_fir_filter(filter_type, window, order, cutoff, sampling_freq)
            x_filtered = apply_fir_filter(taps, x)
            freqs, mag_db = compute_frequency_response(taps, sampling_freq)
            measured_edges = measure_cutoff_frequencies(freqs, mag_db, threshold_db=-6.0)
        except ValueError as exc:
            error_message = str(exc)

        st.session_state[session_key] = {
            "category": category,
            "signal_type": signal_type,
            "config": sig_config if error_message is None else config,
            "t": t,
            "x": x,
            "sampling_freq": sampling_freq,
            "filter_type": filter_type,
            "rp": rp,
            "rs": rs,
            "band_edges": band_edges,
            "cutoff": cutoff,
            "order": order,
            "taps": taps,
            "x_filtered": x_filtered,
            "freqs": freqs,
            "mag_db": mag_db,
            "measured_edges": measured_edges,
            "error_message": error_message,
        }

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    result = st.session_state[session_key]

    if result is not None:
        st.markdown("## Results")

        if result["error_message"] is not None:
            st.error(f"Filter design failed: {result['error_message']}")
        else:
            t = result["t"]
            x = result["x"]
            x_filtered = result["x_filtered"]
            fs = result["sampling_freq"]

            st.markdown(
                f"#### {result['filter_type']} FIR Filter ({window} Window) "
                f"applied to **{result['signal_type']}**"
            )

            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("Estimated Filter Length (taps)", result["order"])
            with info_cols[1]:
                if result["filter_type"] in ("Low-Pass", "High-Pass"):
                    st.metric("Cutoff Frequency", f"{result['cutoff']:.1f} Hz")
                else:
                    lo, hi = result["cutoff"]
                    st.metric("Cutoff Frequencies", f"{lo:.1f}-{hi:.1f} Hz")
            with info_cols[2]:
                st.metric("Sampling Frequency", f"{fs:.0f} Hz")

            # --- Time-domain plot -----------------------------------
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t, x, label="Original Signal", color="#57606a", alpha=0.7, linewidth=1.2)
            ax.plot(t, x_filtered, label="FIR-Filtered Signal", color="#1f6feb", linewidth=1.4)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Time-Domain: Original vs. FIR-Filtered")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # --- Frequency response plot -----------------------------------
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(result["freqs"], result["mag_db"], color="#8250df", linewidth=1.5)
            ax2.axhline(-6.0, color="gray", linestyle=":", label="-6 dB reference")
            for edge in result["band_edges"]:
                ax2.axvline(edge, color="#cf222e", linestyle="--", alpha=0.6)
            ax2.set_ylim(bottom=max(-100, np.min(result["mag_db"]) - 5))
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Magnitude (dB)")
            ax2.set_title(f"FIR Filter Frequency Response ({window} Window)")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.5)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            measured_str = ", ".join(f"{m:.1f} Hz" for m in result["measured_edges"]) or "not found in range"
            st.caption(
                f"Design edge frequencies: {', '.join(f'{e:.1f} Hz' for e in result['band_edges'])}. "
                f"Measured -6 dB crossing(s) of the actual response: {measured_str}."
            )

            st.markdown("#### Observation")
            st.write(
                f"The {window} window shapes the FIR filter's transition band and "
                f"stopband attenuation. Increasing the filter length (more taps) "
                f"narrows the transition band and increases stopband attenuation, "
                f"at the cost of more computation and a longer group delay "
                f"(the filter's constant time delay, equal to half the filter "
                f"length in samples)."
            )

            st.markdown("#### Biomedical Relevance")
            st.write(FIR_RELEVANCE[result["filter_type"]])

    st.markdown("---")

    st.markdown("### Post-Lab Questions")
    st.markdown(
        f"""
        1. How does increasing the filter length affect the transition bandwidth and stopband attenuation?
        2. Why is the estimated filter length forced to be odd?
        3. What is group delay, and what is its value for the FIR filter designed here?
        4. Compare a {window}-window design with a differently-windowed design of the same length - which would you expect to have lower stopband ripple?
        5. Why might a biomedical engineer choose a linear-phase FIR filter over an IIR filter, even though the FIR filter usually needs more taps to achieve a similar roll-off?
        """
    )

    st.markdown("### Result")
    st.success(
        f"Thus, a digital FIR filter was designed using a {window} window "
        f"and applied to the selected signal, and its magnitude response "
        f"was verified against the design specification."
    )
