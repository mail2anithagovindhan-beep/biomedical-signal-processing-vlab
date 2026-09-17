"""
utils/moving_average_utils.py
-------------------------------
Moving-average (boxcar) filtering and noise-corruption helpers for
Experiment 9 (Analysis of ECG Signal).

This module is intentionally kept separate from the Streamlit UI code in
`pages/09_ECG_Signal_Analysis.py`, mirroring the utils/*.py pattern used
throughout this lab.
"""

import numpy as np


def add_gaussian_noise(x: np.ndarray, noise_std: float, seed: int = 0) -> np.ndarray:
    """Return `x` corrupted by additive white Gaussian noise with standard
    deviation `noise_std`, plus the noise itself (so it can be displayed
    or analyzed on its own)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=x.shape)
    return x + noise, noise


def moving_average_filter(x: np.ndarray, window_length: int = 8) -> np.ndarray:
    """Apply a simple moving-average (boxcar) filter via convolution:

        y[n] = (1/L) * sum_{k=0}^{L-1} x[n-k]

    where L = `window_length`. Implemented with `np.convolve` in "same"
    mode so the output has the same length as the input (this introduces
    a small boundary effect at the very start of the signal, since there
    aren't L-1 prior samples available there - this is expected and
    typical of a simple moving-average implementation).
    """
    if window_length < 1:
        raise ValueError("`window_length` must be at least 1.")
    kernel = np.ones(window_length) / window_length
    return np.convolve(x, kernel, mode="same")
