"""
utils/filter_utils.py
----------------------
Digital IIR filter design and analysis helpers for Experiment 4 (Digital
Butterworth Filter).

This module is intentionally kept separate from the Streamlit UI code in
`pages/4_Digital_Butterworth_Filter.py`, so the filter design/analysis math
can be read, tested, and reused independently of the page layout (mirrors
the utils/signal_generators.py and utils/spectral_utils.py pattern used
for Experiments 1 and 3).
"""

from typing import Sequence, Union

import numpy as np
from scipy import signal as sp_signal

# Maps the UI-facing filter type label to the scipy.signal.butter btype.
BTYPE_MAP = {
    "Low-Pass": "lowpass",
    "High-Pass": "highpass",
    "Band-Pass": "bandpass",
    "Band-Stop": "bandstop",
}


def design_butterworth(filter_type: str, order: int, cutoff: Union[float, Sequence[float]], fs: float):
    """Design a digital Butterworth filter, returned as second-order
    sections (SOS) for numerically stable filtering.

    Parameters
    ----------
    filter_type : str
        One of "Low-Pass", "High-Pass", "Band-Pass", "Band-Stop".
    order : int
        Filter order. Higher orders give a steeper roll-off but a less
        flat phase response.
    cutoff : float or (float, float)
        Cutoff frequency in Hz for Low-Pass/High-Pass, or (low, high)
        cutoff frequencies in Hz for Band-Pass/Band-Stop.
    fs : float
        Sampling frequency in Hz.
    """
    btype = BTYPE_MAP[filter_type]
    sos = sp_signal.butter(order, cutoff, btype=btype, fs=fs, output="sos")
    return sos


def design_chebyshev1_lowpass(order: int, ripple_db: float, cutoff: float, fs: float):
    """Design a digital Chebyshev Type I low-pass filter, for Experiment 5.

    Unlike a Butterworth filter's -3 dB passband-edge convention, a
    Chebyshev Type I filter's passband edge is defined to sit exactly
    `ripple_db` below 0 dB gain - the filter is equiripple across the
    whole passband up to that point.

    Parameters
    ----------
    order : int
        Filter order.
    ripple_db : float
        Maximum allowed passband ripple, in dB.
    cutoff : float
        Passband-edge cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    """
    sos = sp_signal.cheby1(order, ripple_db, cutoff, btype="lowpass", fs=fs, output="sos")
    return sos


def apply_filter(sos, x: np.ndarray) -> np.ndarray:
    """Apply a digital filter (given as SOS) to a signal, causally
    (single-pass), the way a real-time digital filter would process data
    as it arrives."""
    if len(x) == 0:
        return x
    return sp_signal.sosfilt(sos, x)


def compute_frequency_response(sos, fs: float, n_points: int = 1024):
    """Magnitude response of a digital filter (given as SOS).

    Returns
    -------
    freqs : np.ndarray
        Frequency bins in Hz, from 0 to fs/2.
    magnitude_db : np.ndarray
        Magnitude response in dB (20 log10 |H(f)|), with unity (0 dB)
        representing no attenuation/gain.
    """
    w, h = sp_signal.sosfreqz(sos, worN=n_points, fs=fs)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    return w, magnitude_db


def measure_cutoff_frequencies(freqs: np.ndarray, magnitude_db: np.ndarray, threshold_db: float = -3.0):
    """Find the -3 dB crossing frequencies of a measured magnitude
    response, by linear interpolation between the two nearest frequency
    samples that bracket the threshold.

    This works for any filter shape (low-pass, high-pass, band-pass,
    band-stop) without assuming which side of the threshold is the
    passband, since it simply reports every crossing point.

    Returns
    -------
    list[float]
        Frequencies (Hz) where the response crosses `threshold_db`,
        in ascending order. A low-pass or high-pass filter typically
        produces one crossing; a band-pass or band-stop filter typically
        produces two.
    """
    if len(freqs) < 2:
        return []

    above = magnitude_db >= threshold_db
    crossings = []
    for i in range(1, len(freqs)):
        if above[i] != above[i - 1]:
            f0, f1 = freqs[i - 1], freqs[i]
            m0, m1 = magnitude_db[i - 1], magnitude_db[i]
            frac = 0.5 if m1 == m0 else (threshold_db - m0) / (m1 - m0)
            frac = min(max(frac, 0.0), 1.0)
            crossings.append(float(f0 + frac * (f1 - f0)))
    return crossings


# --------------------------------------------------------------------------
# Short, filter-type-specific notes on biomedical use cases, used in the
# "Biomedical Relevance" section of Experiment 4.
# --------------------------------------------------------------------------
FILTER_RELEVANCE = {
    "Low-Pass": (
        "Low-pass filtering is commonly used to remove high-frequency "
        "muscle (EMG) noise and mains/power-line-adjacent interference "
        "from an ECG or EEG recording, smoothing the signal while "
        "preserving its underlying low-frequency structure."
    ),
    "High-Pass": (
        "High-pass filtering is commonly used to remove slow baseline "
        "wander from an ECG (caused by breathing or electrode movement) "
        "or to remove DC offset drift from EEG and EMG recordings, "
        "without disturbing faster physiological features."
    ),
    "Band-Pass": (
        "Band-pass filtering is used to isolate a specific frequency band "
        "of interest — for example, extracting the QRS-dominant band from "
        "an ECG for heartbeat detection, or isolating a single EEG rhythm "
        "band (such as alpha, 8-13 Hz) for further analysis."
    ),
    "Band-Stop": (
        "Band-stop (notch) filtering is used to remove a narrow, known "
        "interference band — most commonly 50/60 Hz mains power-line "
        "interference — while leaving the rest of the signal's spectrum "
        "largely unaffected."
    ),
}
