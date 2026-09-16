"""
Experiment 6 - DIT FFT (2-point, 4-point, and 8-point)

Lets students walk through the radix-2 Decimation-in-Time (DIT) Fast
Fourier Transform algorithm step by step for a small, student-entered
sequence (N = 2, 4, or 8): the bit-reversal reordering, every butterfly
stage, and the final spectrum - verified against the direct DFT.

UI logic lives here; the DIT FFT algorithm itself lives in
utils/fft_utils.py, so the page stays decoupled from the underlying
computation (mirrors the structure of Experiments 1, 3, 4, and 5).
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.fft_utils import (
    bit_reversal_indices,
    complex_array_to_rows,
    dit_fft_stages,
    max_error_vs_direct_dft,
    twiddle_factors_for_stage,
)

st.set_page_config(page_title="Exp 6 - DIT FFT", page_icon="🧪", layout="wide")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Experiment 6 — DIT FFT (2-point, 4-point, and 8-point)")
st.markdown("---")

st.markdown("### Aim")
st.write(
    "To implement and visualize the Decimation-in-Time (DIT) Fast Fourier "
    "Transform algorithm for 2-point, 4-point, and 8-point sequences, and "
    "to verify the result against the direct Discrete Fourier Transform (DFT)."
)

st.markdown("### Learning Objectives")
st.markdown(
    """
    By the end of this experiment, students should be able to:

    - State the direct DFT formula and its O(N²) computational cost.
    - Explain how Decimation-in-Time splits a DFT into smaller DFTs of even- and odd-indexed samples.
    - Perform the bit-reversal reordering of an input sequence.
    - Trace a signal through each butterfly stage of the DIT FFT algorithm.
    - Verify a computed FFT result against the direct DFT.
    """
)

with st.expander("Theory (click to expand)"):
    st.markdown("The **N-point Discrete Fourier Transform (DFT)** of a sequence x[n] is:")
    st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi k n / N}, \qquad k = 0, 1, \dots, N-1")
    st.markdown(
        """
        Computed directly, this takes O(N²) multiplications. The
        **Decimation-in-Time (DIT) FFT** reduces this to O(N log₂N) by
        recursively splitting the DFT into the DFTs of the even-indexed
        and odd-indexed samples:
        """
    )
    st.latex(r"X[k] = E[k] + W_N^{k}\, O[k], \qquad X[k + N/2] = E[k] - W_N^{k}\, O[k]")
    st.markdown(
        r"""
        where E[k] and O[k] are the (N/2)-point DFTs of the even- and
        odd-indexed samples, and $W_N^{k} = e^{-j 2\pi k / N}$ is a
        **twiddle factor**. Applying this split repeatedly, down to
        single-sample DFTs, produces log₂N **butterfly stages**.

        Because of how the even/odd splitting repeats at every level, the
        samples must first be reordered by **bit-reversal**: the sample
        originally at index *n* moves to the index formed by reversing
        the binary digits of *n*. After this reordering, each stage
        combines pairs of values using a **butterfly operation** — a sum
        and a difference, one of them scaled by a twiddle factor — and
        after log₂N such stages, the array holds the finished FFT result
        in natural order.

        This experiment implements that algorithm directly (not by simply
        calling a library FFT function), so every bit-reversal and
        butterfly stage can be inspected, and cross-checks the final
        result against NumPy's FFT to confirm correctness.
        """
    )

st.markdown("### Pre-Lab Questions")
st.markdown(
    """
    1. What is the computational complexity of the direct DFT, and of the FFT?
    2. What does "Decimation-in-Time" mean in the context of the FFT algorithm?
    3. What is a twiddle factor, and how is it defined?
    4. Why must the input sequence be reordered by bit-reversal before the butterfly stages?
    5. How many butterfly stages does an 8-point DIT FFT have?
    """
)

st.markdown("---")
st.info("Use the controls in the sidebar to choose N and enter an input sequence, then click **Compute FFT**.")

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("FFT Controls")

n_points = st.sidebar.selectbox("Number of Points (N)", [2, 4, 8], index=1)

default_sequences = {
    2: [1.0, 2.0],
    4: [1.0, 2.0, 3.0, 4.0],
    8: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
}
defaults = default_sequences[n_points]

st.sidebar.caption("Enter the real-valued input sequence x[n]:")
x_input = []
for i in range(n_points):
    val = st.sidebar.number_input(
        f"x[{i}]", value=defaults[i], step=1.0, key=f"fft_x_{i}_of_{n_points}"
    )
    x_input.append(val)

compute_clicked = st.sidebar.button("Compute FFT", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Compute, store in session state (persists until the button is pressed
# again)
# --------------------------------------------------------------------------
if "exp6_result" not in st.session_state:
    st.session_state.exp6_result = None

if compute_clicked:
    x = np.array(x_input, dtype=float)
    num_bits = int(round(np.log2(n_points)))
    rev_idx = bit_reversal_indices(num_bits)
    bit_reversed_input, stages, result = dit_fft_stages(x)
    error = max_error_vs_direct_dft(x, result)

    st.session_state.exp6_result = {
        "n_points": n_points,
        "num_bits": num_bits,
        "x": x,
        "rev_idx": rev_idx,
        "bit_reversed_input": bit_reversed_input,
        "stages": stages,
        "result": result,
        "error": error,
    }

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.exp6_result

if result is not None:
    st.markdown("## Results")

    n = result["n_points"]
    st.markdown(f"#### {n}-Point DIT FFT")

    # --- Input sequence and bit-reversal -----------------------------------
    st.markdown("#### Step 1 — Input Sequence and Bit-Reversal Reordering")
    input_rows = [
        {"n": i, "x[n]": float(result["x"][i]), "Bit-Reversed Index": result["rev_idx"][i]}
        for i in range(n)
    ]
    st.table(input_rows)
    st.caption(
        "The bit-reversed index shows where each original sample moves to "
        "before the butterfly stages begin."
    )

    st.markdown("Sequence after bit-reversal reordering:")
    st.table(complex_array_to_rows(result["bit_reversed_input"]))

    # --- Butterfly stages ----------------------------------------------------
    st.markdown("#### Step 2 — Butterfly Stages")
    stage_size = 2
    for stage_num, stage_values in enumerate(result["stages"], start=1):
        with st.expander(f"Stage {stage_num} (butterfly size {stage_size})", expanded=(stage_num == len(result["stages"]))):
            twiddles = twiddle_factors_for_stage(stage_size)
            twiddle_str = ", ".join(
                f"W_{stage_size}^{k} = {tw.real:.3f}{'+' if tw.imag >= 0 else ''}{tw.imag:.3f}j"
                for k, tw in enumerate(twiddles)
            )
            st.caption(f"Twiddle factors used in this stage: {twiddle_str}")
            st.table(complex_array_to_rows(stage_values))
        stage_size *= 2

    # --- Final result ----------------------------------------------------
    st.markdown("#### Step 3 — Final FFT Result")
    final_rows = complex_array_to_rows(result["result"])
    st.table(final_rows)

    fig, ax = plt.subplots(figsize=(9, 4))
    magnitudes = [row["Magnitude"] for row in final_rows]
    ax.bar(range(n), magnitudes, color="#1f6feb")
    ax.set_title(f"{n}-Point FFT — Magnitude Spectrum")
    ax.set_xlabel("Frequency Index k")
    ax.set_ylabel("|X[k]|")
    ax.set_xticks(range(n))
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- Verification -----------------------------------------------------
    st.markdown("#### Verification Against Direct DFT (NumPy FFT)")
    verify_cols = st.columns(2)
    with verify_cols[0]:
        st.metric("Butterfly Stages", int(result["num_bits"]))
    with verify_cols[1]:
        st.metric("Max Error vs. NumPy FFT", f"{result['error']:.2e}")

    if result["error"] < 1e-6:
        st.success(
            "The manually computed DIT FFT result matches NumPy's direct "
            "FFT to within floating-point precision — the algorithm is "
            "verified correct for this input."
        )
    else:
        st.warning(
            "The computed result differs from NumPy's FFT by more than "
            "expected — this would indicate a bug in the algorithm "
            "implementation (not expected for normal use of this page)."
        )

    # --- Biomedical relevance -----------------------------------------------
    st.markdown("#### Biomedical Relevance")
    st.write(
        "Real-time biomedical monitoring systems — for example, "
        "continuous EEG or ECG spectral analysis — need to repeatedly "
        "compute frequency spectra fast enough to keep up with incoming "
        "data. The direct DFT's O(N²) cost quickly becomes impractical as "
        "N grows, while the FFT's O(N log₂N) cost is what makes real-time "
        "frequency-domain analysis of biomedical signals (as used already "
        "in Experiment 3) computationally feasible."
    )

    st.markdown("---")

# --------------------------------------------------------------------------
# Companion Desmos activity (placeholder)
# --------------------------------------------------------------------------
st.markdown("### Interactive Desmos Activity — Visualizing the FFT Butterfly Structure")
st.info(
    "**Desmos activity:** _link to be added_. A companion Desmos graph "
    "illustrating the butterfly diagram and twiddle factors for the "
    "4-point or 8-point DIT FFT can be built and linked here later, the "
    "same way the Experiment 2 Desmos activity was added."
)

st.markdown("### Post-Lab Questions")
st.markdown(
    """
    1. Draw the bit-reversal permutation for N = 8 and explain the pattern.
    2. How many butterfly operations are performed in total for a 4-point DIT FFT? For an 8-point DIT FFT?
    3. What would happen to the FFT result if the bit-reversal step were skipped?
    4. How did the twiddle factors differ between the first stage and the last stage?
    5. Why is verifying against the direct DFT an important step when implementing a new FFT algorithm?
    """
)

st.markdown("### Result")
st.success(
    "Thus, the Decimation-in-Time FFT algorithm was implemented and "
    "verified successfully for 2-point, 4-point, and 8-point sequences."
)
