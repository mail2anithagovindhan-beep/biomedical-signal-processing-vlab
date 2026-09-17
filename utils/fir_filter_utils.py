"""
utils/fir_filter_utils.py
---------------------------
Shared FIR filter design logic for Experiment 7 (FIR filter design using a
Hamming window) and Experiment 8 (FIR filter design using a Hanning
window).

Both experiments follow the exact same classical windowed-FIR design
procedure - only the window function itself differs - so the design,
filtering, and frequency-response math lives here once and is reused by
both page files (mirrors the utils/signal_generators.py /
utils/filter_utils.py pattern used for earlier experiments).
"""

from typing import Sequence, Union

import numpy as np
from scipy import signal as sp_signal

# Maps the UI-facing filter type label to a scipy.signal.firwin pass_zero
# setting (whether the response passes DC, i.e. 0 Hz).
BTYPE_MAP = {
    "Low-Pass": "lowpass",
    "High-Pass": "highpass",
    "Band-Pass": "bandpass",
    "Band-Stop": "bandstop",
}

_PASS_ZERO = {
    "lowpass": True,
    "highpass": False,
    "bandpass": False,
    "bandstop": True,
}

# Maps the UI-facing window label to the scipy.signal.firwin window name.
WINDOW_FUNCS = {
    "Hamming": "hamming",
    "Hanning": "hann",
}


def estimate_fir_order(rp: float, rs: float, fp: float, fstop: float, sampling_freq: float) -> int:
    """Estimate the required FIR filter length using the classic empirical
    formula for windowed FIR design:

        num = -20 * log10( sqrt(rp * rs) ) - 13
        den = 14.6 * (fstop - fp) / sampling_freq
        n   = ceil(num / den)

    Parameters
    ----------
    rp, rs : float
        Passband and stopband ripple, as LINEAR (not dB) fractions - e.g.
        rp = 0.01 means 1% passband ripple.
    fp, fstop : float
        Passband-edge and stopband-edge frequencies, in Hz.
    sampling_freq : float
        The overall sampling frequency, in Hz.

    Returns
    -------
    int
        The estimated filter length, forced to the next odd value so the
        result is a Type I (odd-length, linear-phase) FIR filter.
    """
    transition = abs(fstop - fp)
    if transition <= 0:
        raise ValueError("Passband and stopband edge frequencies must differ.")
    if rp <= 0 or rs <= 0:
        raise ValueError("Ripple values (rp, rs) must be positive.")

    num = -20 * np.log10(np.sqrt(rp * rs)) - 13
    den = 14.6 * transition / sampling_freq
    n = int(np.ceil(num / den))
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1
    return n


def design_fir_filter(
    filter_type: str,
    window: str,
    order: int,
    cutoff: Union[float, Sequence[float]],
    sampling_freq: float,
):
    """Design a windowed FIR filter with scipy.signal.firwin.

    Parameters
    ----------
    filter_type : str
        One of "Low-Pass", "High-Pass", "Band-Pass", "Band-Stop".
    window : str
        One of "Hamming", "Hanning".
    order : int
        Filter length (number of taps). Must be odd for Type I linear
        phase (this is enforced by `estimate_fir_order`, but is checked
        again here for filters where the order is entered directly).
    cutoff : float or (float, float)
        Cutoff frequency in Hz for Low-Pass/High-Pass, or (low, high)
        cutoff frequencies in Hz for Band-Pass/Band-Stop.
    sampling_freq : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        The FIR filter's tap (coefficient) array.
    """
    btype = BTYPE_MAP[filter_type]
    window_name = WINDOW_FUNCS[window]
    pass_zero = _PASS_ZERO[btype]

    n_taps = int(order)
    if n_taps % 2 == 0:
        n_taps += 1

    taps = sp_signal.firwin(
        n_taps, cutoff, window=window_name, pass_zero=pass_zero, fs=sampling_freq
    )
    return taps


def apply_fir_filter(taps: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply the FIR filter causally (single-pass), the way a real-time
    digital filter would process data as it arrives."""
    if len(x) == 0:
        return x
    return sp_signal.lfilter(taps, [1.0], x)


def compute_frequency_response(taps: np.ndarray, sampling_freq: float, n_points: int = 1024):
    """Magnitude response of an FIR filter.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins in Hz, from 0 to sampling_freq/2.
    magnitude_db : np.ndarray
        Magnitude response in dB (20 log10 |H(f)|), with unity (0 dB)
        representing no attenuation/gain.
    """
    w, h = sp_signal.freqz(taps, worN=n_points, fs=sampling_freq)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    return w, magnitude_db


# --------------------------------------------------------------------------
# Short, filter-type-specific notes on biomedical use cases, used in the
# "Biomedical Relevance" section of Experiments 7 and 8.
# --------------------------------------------------------------------------
FIR_RELEVANCE = {
    "Low-Pass": (
        "An FIR low-pass filter can smooth an ECG or EEG recording by "
        "removing high-frequency muscle (EMG) noise, while its guaranteed "
        "linear phase avoids distorting the relative timing of waveform "
        "features such as the QRS complex - something an IIR filter of "
        "comparable sharpness cannot guarantee."
    ),
    "High-Pass": (
        "An FIR high-pass filter can remove slow baseline wander from an "
        "ECG (caused by breathing or electrode movement) without the "
        "phase distortion an IIR filter would introduce, which is "
        "important when the exact timing of waveform onsets matters."
    ),
    "Band-Pass": (
        "An FIR band-pass filter can isolate a specific rhythm band of "
        "interest - for example, extracting alpha-band (8-13 Hz) activity "
        "from an EEG - with a linear phase response that preserves the "
        "relative shape of the extracted waveform."
    ),
    "Band-Stop": (
        "An FIR band-stop (notch) filter can remove narrowband "
        "interference, such as 50/60 Hz mains power-line noise, from a "
        "biomedical recording while keeping every other frequency "
        "component's phase relationship intact."
    ),
}
