"""
utils/fft_utils.py
--------------------
Manual radix-2 Decimation-in-Time (DIT) FFT implementation for
Experiment 6 (DIT FFT, 2/4/8-point).

This computes the FFT the way the DIT algorithm actually works -
bit-reversal reordering followed by log2(N) butterfly stages - and
records every intermediate stage, rather than simply calling
numpy.fft.fft, so the algorithm's steps can be displayed on the page.

Kept separate from the Streamlit UI code in `pages/6_DIT_FFT.py` (mirrors
the utils/signal_generators.py, utils/spectral_utils.py, and
utils/filter_utils.py pattern used in the earlier experiments).
"""

import numpy as np


def bit_reversal_indices(num_bits: int):
    """Bit-reversal permutation indices for N = 2**num_bits points."""
    n = 1 << num_bits
    return [int(format(i, f"0{num_bits}b")[::-1], 2) for i in range(n)]


def dit_fft_stages(x):
    """Compute the radix-2 Decimation-in-Time FFT of x (length N = 2, 4,
    or 8), recording every intermediate butterfly stage.

    Parameters
    ----------
    x : array-like
        Real or complex input sequence. Length must be a power of 2.

    Returns
    -------
    bit_reversed_input : np.ndarray (complex)
        The input sequence reordered by bit-reversal, before the first
        butterfly stage.
    stages : list[np.ndarray] (complex)
        One entry per butterfly stage (log2(N) stages total), each the
        full array of values after that stage completes. stages[-1] is
        the final FFT result.
    result : np.ndarray (complex)
        The final FFT output (identical to stages[-1]).
    """
    x = np.asarray(x, dtype=complex)
    n = len(x)
    num_bits = int(round(np.log2(n))) if n > 0 else 0
    if n == 0 or 2 ** num_bits != n:
        raise ValueError("Input length must be a power of 2 (2, 4, or 8 for this experiment).")

    rev_idx = bit_reversal_indices(num_bits)
    a = x[rev_idx].copy()
    bit_reversed_input = a.copy()

    stages = []
    size = 2
    while size <= n:
        half = size // 2
        twiddles = np.exp(-2j * np.pi * np.arange(half) / size)
        new_a = a.copy()
        for start in range(0, n, size):
            for k in range(half):
                top = a[start + k]
                bottom = twiddles[k] * a[start + k + half]
                new_a[start + k] = top + bottom
                new_a[start + k + half] = top - bottom
        a = new_a
        stages.append(a.copy())
        size *= 2

    return bit_reversed_input, stages, a


def twiddle_factors_for_stage(stage_size: int):
    """The W_(stage_size)^k twiddle factors (k = 0 .. stage_size/2 - 1)
    used within one butterfly stage of the given size."""
    half = stage_size // 2
    return np.exp(-2j * np.pi * np.arange(half) / stage_size)


def max_error_vs_direct_dft(x, computed_result) -> float:
    """Largest absolute difference between the DIT FFT result and numpy's
    reference FFT, as a correctness check against the direct DFT."""
    reference = np.fft.fft(np.asarray(x, dtype=complex))
    return float(np.max(np.abs(np.asarray(computed_result) - reference)))


def complex_array_to_rows(values, index_offset: int = 0):
    """Format a complex array as a list of dict rows (Index, Real, Imag,
    Magnitude, Phase (deg)), suitable for display in a Streamlit table."""
    rows = []
    for i, v in enumerate(values):
        rows.append(
            {
                "Index": i + index_offset,
                "Real": round(float(np.real(v)), 4),
                "Imag": round(float(np.imag(v)), 4),
                "Magnitude": round(float(abs(v)), 4),
                "Phase (deg)": round(float(np.degrees(np.angle(v))), 2),
            }
        )
    return rows
