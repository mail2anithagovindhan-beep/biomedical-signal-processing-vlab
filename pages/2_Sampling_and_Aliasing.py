"""
Experiment 2 - Sampling and Aliasing

Lets students sample a sinusoidal signal x(t) = A sin(2*pi*f*t) at a
chosen sampling frequency Fs, see the sampled points superimposed on the
continuous waveform, and observe how the Nyquist criterion (Fs >= 2f)
governs whether the sampling is adequate or leads to aliasing.

UI logic lives here; sampling/Nyquist/aliasing math lives in
utils/sampling_utils.py so the two stay decoupled.
"""

import matplotlib.pyplot as plt
import streamlit as st

from utils.sampling_utils import (
    PRESET_CASES,
    continuous_signal,
    evaluate_sampling,
    sample_signal,
)

st.set_page_config(page_title="Exp 2 - Sampling and Aliasing", page_icon="🧪", layout="wide")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Experiment 2 — Sampling and Aliasing")
st.markdown("---")

st.markdown("### Aim")
st.write(
    "To study the effect of sampling frequency on a continuous sinusoidal "
    "signal, verify the Nyquist sampling criterion, and observe the "
    "phenomenon of aliasing when a signal is sampled below the Nyquist rate."
)

st.markdown("### Learning Objectives")
st.markdown(
    """
    By the end of this experiment, students should be able to:

    - Explain what a continuous-time signal and sampling are.
    - Define sampling frequency (Fs) and Nyquist frequency.
    - State the Nyquist sampling criterion.
    - Distinguish between adequate sampling, critical (Nyquist-limit) sampling, and under-sampling.
    - Describe what aliasing is and identify it in a sampled waveform.
    - Explain the effect of changing Fs on the sampled representation of a signal.
    """
)

with st.expander("Theory (click to expand)"):
    st.markdown(
        """
        **What is sampling?** Sampling is the process of converting a
        continuous-time signal x(t) into a discrete-time sequence by
        measuring its value at regular time intervals.

        **Why do biomedical signals need to be sampled?** Signals such as
        ECG, EEG, EMG, and EOG are continuous physiological phenomena. To
        store, process, or analyze them on a computer, they must first be
        converted to a sequence of discrete digital samples.

        **Sampling frequency (Fs)** is the number of samples taken per
        second, measured in Hz. The time between successive samples is
        1/Fs seconds.

        **Nyquist frequency** is the highest frequency component that can
        be correctly represented for a given sampling frequency; it is
        equal to Fs/2.

        **Nyquist sampling criterion:** to correctly represent a signal
        containing a maximum frequency component f_max, the sampling
        frequency must satisfy:
        """
    )
    st.latex(r"F_s \geq 2\,f_{max}")
    st.markdown(
        """
        - **When Fs > 2f:** the signal is adequately sampled — enough
          samples are taken per cycle to represent the waveform's frequency.
        - **When Fs = 2f (approximately):** sampling is at the critical
          Nyquist limit — the theoretical minimum, with essentially two
          samples per cycle.
        - **When Fs < 2f:** the signal is under-sampled. The samples are
          too sparse to represent the true frequency, and the resulting
          sample sequence can appear to represent a different, lower
          frequency than the original signal.

        **Aliasing** is this phenomenon: when Fs < 2f, a high-frequency
        signal "hides" behind (aliases as) a lower apparent frequency in
        the sampled data, so the sampled sequence no longer faithfully
        represents the original signal.
        """
    )

st.markdown("### Pre-Lab Questions")
st.markdown(
    """
    1. What is sampling?
    2. Define sampling frequency.
    3. State the Nyquist sampling criterion.
    4. What is aliasing?
    5. What happens when Fs < 2f_max?
    """
)

st.markdown("---")
st.info(
    "Use the controls in the sidebar to choose a preset case (or set your own "
    "values) and click **Generate / Update**."
)

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Sampling Controls")


def _apply_preset():
    choice = st.session_state.exp2_preset
    if choice in PRESET_CASES:
        vals = PRESET_CASES[choice]
        st.session_state.exp2_freq = vals["frequency"]
        st.session_state.exp2_fs = vals["fs"]
        st.session_state.exp2_just_presetted = True


st.sidebar.selectbox(
    "Preset Case",
    ["Custom"] + list(PRESET_CASES.keys()),
    key="exp2_preset",
    on_change=_apply_preset,
)

amplitude = st.sidebar.slider("Signal Amplitude A (a.u.)", 0.1, 5.0, 1.0, step=0.1, key="exp2_amp")
frequency = st.sidebar.slider("Signal Frequency f (Hz)", 0.5, 20.0, 2.0, step=0.5, key="exp2_freq")
fs = st.sidebar.slider("Sampling Frequency Fs (Hz)", 2.0, 100.0, 20.0, step=1.0, key="exp2_fs")
duration = st.sidebar.slider("Duration (s)", 0.5, 5.0, 1.0, step=0.1, key="exp2_dur")

generate_clicked = st.sidebar.button("Generate / Update", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Compute & store in session state (also refreshed automatically when a preset is chosen)
# --------------------------------------------------------------------------
def _compute_result():
    t_cont, x_cont = continuous_signal(amplitude, frequency, duration)
    t_samp, x_samp = sample_signal(amplitude, frequency, fs, duration)
    nyq_result = evaluate_sampling(frequency, fs)
    st.session_state.exp2_result = {
        "t_cont": t_cont,
        "x_cont": x_cont,
        "t_samp": t_samp,
        "x_samp": x_samp,
        "nyq": nyq_result,
        "amplitude": amplitude,
        "frequency": frequency,
        "fs": fs,
        "duration": duration,
    }


if "exp2_result" not in st.session_state:
    st.session_state.exp2_result = None

if generate_clicked or (st.session_state.exp2_result is None):
    _compute_result()

if st.session_state.get("exp2_just_presetted"):
    _compute_result()
    st.session_state.exp2_just_presetted = False

# --------------------------------------------------------------------------
# Part B & C - Results
# --------------------------------------------------------------------------
result = st.session_state.exp2_result

if result is not None:
    st.markdown("## Results")

    nyq = result["nyq"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(result["t_cont"], result["x_cont"], color="#1f6feb", linewidth=1.2,
            label="Continuous-time signal x(t)")
    ax.stem(
        result["t_samp"], result["x_samp"],
        linefmt="C1-", markerfmt="C1o", basefmt=" ",
    )
    ax.plot([], [], "C1o-", label="Sampled points")  # legend proxy for the stem plot
    ax.set_title("Continuous Signal with Sampled Points Superimposed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("#### Sampling Information")
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Signal Frequency (f)", f"{result['frequency']:.2f} Hz")
    with info_cols[1]:
        st.metric("Sampling Frequency (Fs)", f"{result['fs']:.2f} Hz")
    with info_cols[2]:
        st.metric("Nyquist Frequency (2f)", f"{nyq.nyquist_frequency:.2f} Hz")
    with info_cols[3]:
        st.metric("Number of Samples", f"{len(result['t_samp'])}")

    st.markdown("#### Nyquist Status")
    if nyq.status == "Adequately sampled":
        st.success(f"**{nyq.status}** — Fs is comfortably above 2f, so the sampled points closely follow the original waveform's shape.")
    elif nyq.status == "At Nyquist limit":
        st.warning(f"**{nyq.status}** — Fs is approximately equal to 2f, the theoretical minimum. Sampling is critical; a small reduction in Fs could cause aliasing.")
    else:
        st.error(f"**{nyq.status}** — Fs is below 2f. The samples are too sparse to represent the true frequency of the signal.")

    st.markdown("#### Observation")
    st.write(
        "Observe how reducing Fs changes the apparent waveform represented by the samples. "
        "When the sampling rate is insufficient, the sampled signal may appear as a different "
        "frequency. This phenomenon is called aliasing."
    )
    if nyq.status == "Aliasing likely":
        st.write(
            f"With f = {result['frequency']:.2f} Hz sampled at Fs = {result['fs']:.2f} Hz, the "
            f"sampled points are consistent with an apparent (aliased) frequency of approximately "
            f"{nyq.alias_frequency:.2f} Hz, rather than the true {result['frequency']:.2f} Hz. "
            "Note that this describes the apparent frequency suggested by the sample pattern — "
            "the original continuous waveform above is the true signal; the samples alone would "
            "not let you distinguish it from a genuine "
            f"{nyq.alias_frequency:.2f} Hz signal."
        )
    else:
        st.write(
            "At this sampling frequency, the sample points fall densely enough along the "
            "waveform that they clearly trace out its underlying shape."
        )

    st.markdown("---")

# --------------------------------------------------------------------------
# Part D - Desmos Activity
# --------------------------------------------------------------------------
st.markdown("## Interactive Desmos Activity — Visualizing Sampling and Aliasing")
st.write(
    "This companion Desmos activity lets students explore sampling and "
    "aliasing using sliders, alongside this Streamlit lab."
)

st.success("**Desmos activity is live.** Click the link below to open it.")
st.markdown("**Desmos link:** [Sampling and Aliasing — Desmos Activity](https://www.desmos.com/calculator/hvvn5khydg)")

st.markdown("#### Step-by-Step: Building This Activity in Desmos")
st.caption(
    "These six steps are done once by the instructor when creating the activity. "
    "Once built, students only need to move the A, f, and Fs sliders."
)

st.markdown("##### Step 1 — Define Sliders")
st.write(
    "Create three adjustable quantities: amplitude A, signal frequency f, "
    "and sampling frequency Fs."
)
st.code("A = 1\nf = 2\nFs = 20", language="text")
st.write(
    "Type each line into its own expression row in Desmos. After each one, "
    "a small slider toggle appears below the row — click it to turn that "
    "quantity into a draggable slider. Suggested ranges: A from 0.5 to 5; "
    "f from 0 to 20 in steps of 0.5; Fs from 1 to 50 in steps of 1."
)

st.markdown("##### Step 2 — Continuous Signal")
st.write("Plot the underlying continuous sinusoidal signal x(t) = A sin(2πft):")
st.code("y = A sin(2*pi*f*x)", language="text")
st.write("This draws the smooth reference curve that the samples will be compared against.")

st.markdown("##### Step 3 — Sample Indices")
st.write("Define a list of sample numbers 0 through 80 (enough points to see several cycles):")
st.code("n = [0...80]", language="text")

st.markdown("##### Step 4 — Sample Times")
st.write("Convert each sample number into an actual time in seconds, using the sampling frequency Fs:")
st.code("t = n/Fs", language="text")

st.markdown("##### Step 5 — Sampled Values")
st.write("Evaluate the same sine function, but only at the discrete sample times t:")
st.code("s = A sin(2*pi*f*t)", language="text")

st.markdown("##### Step 6 — Plot Sampled Points")
st.write("Plot the sample times against the sampled values as discrete points:")
st.code("(t, s)", language="text")
st.write(
    "Desmos will draw a row of dots on top of the continuous curve from Step 2 — "
    "these dots are what a digital acquisition system would actually record."
)

st.markdown("##### Publishing the Activity")
st.markdown(
    """
    1. Once all six steps are entered and the curve plus dots look correct, click **Share** in the top-right corner of Desmos.
    2. Desmos will generate a link to this graph (no account is required).
    3. Copy that link and give it to the instructor maintaining this Streamlit app, so the placeholder above can be replaced with the real, working link.

    Students should be able to change **f**, **Fs**, and **A** using the sliders, and directly see the continuous sinusoidal waveform, the discrete sampled points, the effect of changing Fs, and aliasing when Fs < 2f.
    """
)

st.markdown("#### What Students Should Observe")
st.markdown(
    """
    Have students set f = 2 Hz (keep it fixed) and try the following three
    ranges of Fs using the slider, comparing the dots to the continuous curve each time:
    """
)
obs_cols = st.columns(3)
with obs_cols[0]:
    st.success("**Fs >> 2f** (e.g. Fs = 20 Hz)")
    st.write(
        "Many samples fall within each cycle of the wave. The dots closely "
        "trace the true shape of the curve — this is adequate sampling."
    )
with obs_cols[1]:
    st.warning("**Fs = 2f** (e.g. Fs = 10 Hz)")
    st.write(
        "Only about two samples land in each cycle — the theoretical "
        "Nyquist minimum. The dots still roughly follow the wave, but the "
        "margin for error is gone; sampling here is critical."
    )
with obs_cols[2]:
    st.error("**Fs < 2f** (e.g. f = 8 Hz, Fs = 10 Hz)")
    st.write(
        "The dots are too sparse to trace the true fast oscillation. "
        "Instead, they line up in a pattern that looks like a slower, "
        "different-frequency wave — this apparent, incorrect wave is aliasing."
    )

st.markdown("---")

# --------------------------------------------------------------------------
# Part E - Desmos teaching flow
# --------------------------------------------------------------------------
st.markdown("## Suggested Student Activity (Desmos or this Streamlit Lab)")
st.markdown(
    """
    **Step 1.** Set f = 2 Hz and Fs = 20 Hz. Observe the sampled points —
    they should closely follow the continuous waveform.

    **Step 2.** Keep f = 2 Hz and gradually reduce Fs. Notice how the
    sampled points get sparser.

    **Step 3.** Set f = 5 Hz and Fs = 10 Hz. Observe the Nyquist limit —
    sampling is now critical.

    **Step 4.** Set f = 8 Hz and Fs = 10 Hz. Observe the apparent change
    in the sampled waveform — this is aliasing.

    **Step 5.** Increase Fs again and observe how the sampled
    representation improves.
    """
)

st.markdown("#### Observation Table")
st.markdown(
    """
    | Signal Frequency | Sampling Frequency | Nyquist Condition | Observation |
    |---|---|---|---|
    | 2 Hz | 20 Hz | Satisfied | Proper sampling |
    | 5 Hz | 10 Hz | At limit | Critical sampling |
    | 8 Hz | 10 Hz | Violated | Aliasing |
    """
)

st.markdown("---")

# --------------------------------------------------------------------------
# Part F - Biomedical Relevance
# --------------------------------------------------------------------------
st.markdown("## Biomedical Relevance")
st.write(
    "Biomedical signals such as ECG, EEG, EMG, and EOG are continuous "
    "physiological phenomena that are commonly acquired and represented "
    "digitally through sampling, using devices such as ECG machines, EEG "
    "amplifiers, or portable biosensors. If such a device samples below "
    "the Nyquist rate for the frequencies present in the physiological "
    "signal, aliasing can distort the digitized recording — for example, "
    "causing high-frequency components to appear as spurious "
    "low-frequency activity, which can affect subsequent visualization "
    "and analysis. This is why biomedical acquisition systems are "
    "designed with sampling frequencies well above the highest frequency "
    "of clinical interest for the signal being recorded."
)
st.caption(
    "This section describes general sampling principles and does not "
    "make any clinical or diagnostic claims."
)

st.markdown("---")

# --------------------------------------------------------------------------
# Part H - Post-Lab Questions
# --------------------------------------------------------------------------
st.markdown("### Post-Lab Questions")
st.markdown(
    """
    1. What happened when Fs was much greater than 2f?
    2. What happened when Fs approached 2f?
    3. What happened when Fs became less than 2f?
    4. How can aliasing be avoided?
    5. Why is proper sampling important for biomedical signals?
    """
)

# --------------------------------------------------------------------------
# Part I - Result
# --------------------------------------------------------------------------
st.markdown("### Result")
st.success(
    "Thus, the effect of sampling frequency on a sinusoidal signal was "
    "studied and the phenomenon of aliasing was demonstrated using "
    "interactive visualization."
)
