"""
utils/anc_utils.py
--------------------
Adaptive Noise Canceller (LMS algorithm) logic for Experiment 10.

This module is intentionally kept separate from the Streamlit UI code in
`pages/10_Adaptive_Noise_Canceller.py`, mirroring the utils/*.py pattern
used throughout this lab.

Theory (brief)
--------------
An Adaptive Noise Canceller has two inputs:

- A **primary** input z(n) = s(n) + v(n): a wanted signal s(n) (e.g. a
  biomedical waveform) corrupted by an interfering noise v(n).
- A **reference** input r(n): correlated with the noise v(n), but NOT
  with the wanted signal s(n).

An adaptive FIR filter processes r(n) to produce an output y(n), which is
subtracted from the primary input to form the error/output signal:

    e(n) = z(n) - y(n)

The Least-Mean-Squares (LMS) algorithm adjusts the filter's weights on
every sample so that y(n) converges toward the noise component v(n),
making e(n) converge toward the wanted signal s(n):

    y(n)       = w^T * buffer(n)
    e(n)       = z(n) - y(n)
    w(n+1)     = w(n) + 2 * mu * e(n) * buffer(n)

where `buffer(n)` holds the most recent `filter_order` samples of the
reference input, and `mu` is the step size (learning rate).
"""

import numpy as np


def generate_reference_noise(t, freq1=1.5, amp1=0.1, freq2=1.25, amp2=0.4):
    """Educational two-tone reference noise signal:

        r(t) = amp1 * sin(2*pi*freq1*t) + amp2 * sin(2*pi*freq2*t)

    This mirrors the reference-noise construction used in the source lab
    exercise. In a real ANC system the reference input would come from a
    second sensor placed to pick up the interference alone (e.g. a second
    electrode/photodiode away from the physiological signal source).
    """
    return amp1 * np.sin(2 * np.pi * freq1 * t) + amp2 * np.sin(2 * np.pi * freq2 * t)


def lms_adaptive_filter(primary: np.ndarray, reference: np.ndarray, mu: float = 0.001, filter_order: int = 150):
    """Adaptive noise cancellation using the Least-Mean-Squares (LMS)
    algorithm.

    Parameters
    ----------
    primary : np.ndarray
        The noisy primary signal z(n) = wanted_signal(n) + noise(n) - the
        signal that needs its noise removed.
    reference : np.ndarray
        A reference signal r(n), correlated with the noise component of
        `primary` but not with the wanted signal itself. Must be the same
        length as `primary`.
    mu : float
        Step size (learning rate) controlling adaptation speed vs.
        stability. Too large a value can make the filter diverge.
    filter_order : int
        Number of taps in the adaptive FIR filter.

    Returns
    -------
    output : np.ndarray
        The adaptive filter's noise-estimate output y(n), same length as
        the inputs.
    error : np.ndarray
        The error signal e(n) = primary(n) - y(n) - this is the
        noise-cancelled estimate of the wanted signal.
    weights : np.ndarray
        Final adaptive filter weights, length `filter_order`.
    """
    primary = np.asarray(primary, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(primary) != len(reference):
        raise ValueError("`primary` and `reference` must be the same length.")
    if filter_order < 1:
        raise ValueError("`filter_order` must be at least 1.")

    n = len(primary)
    weights = np.zeros(filter_order)
    buffer = np.zeros(filter_order)
    output = np.zeros(n)
    error = np.zeros(n)

    for i in range(n):
        buffer[1:] = buffer[:-1]
        buffer[0] = reference[i]

        y = float(np.dot(weights, buffer))
        e = primary[i] - y

        weights = weights + 2.0 * mu * e * buffer

        output[i] = y
        error[i] = e

    return output, error, weights


def noise_reduction_db(before: np.ndarray, after_error: np.ndarray, reference_free: np.ndarray) -> float:
    """A simple, educational measure of how much closer the cancelled
    signal `after_error` is to the true wanted signal `reference_free`
    compared to the original noisy signal `before`, expressed in dB of
    RMS-error reduction.

    A positive value means the ANC output is closer to the true wanted
    signal than the raw noisy input was; a value near/below 0 means the
    canceller did not help (e.g. mu too small, filter order too short,
    or too few samples for convergence).
    """
    before = np.asarray(before, dtype=float)
    after_error = np.asarray(after_error, dtype=float)
    reference_free = np.asarray(reference_free, dtype=float)

    rms_before = float(np.sqrt(np.mean((before - reference_free) ** 2)))
    rms_after = float(np.sqrt(np.mean((after_error - reference_free) ** 2)))

    if rms_after <= 1e-12:
        return 200.0  # effectively perfect cancellation
    if rms_before <= 1e-12:
        return 0.0
    return 20 * np.log10(rms_before / rms_after)
