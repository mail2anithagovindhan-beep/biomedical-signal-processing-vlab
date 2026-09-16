"""
utils/spectral_utils.py
------------------------
Frequency-domain analysis helpers for Experiment 3 (Representation of
Biomedical Signals).

This module is intentionally kept separate from the Streamlit UI code in
`pages/3_Representation_of_Biomedical_Signals.py`, so that the FFT and
spectrogram computations can be read, tested, and reused independently of
the page layout (mirrors the utils/signal_generators.py pattern used for
Experiment 1).
"""

import numpy as np
from scipy import signal as sp_signal


def compute_magnitude_spectrum(x: np.ndarray, fs: float):
    """Single-sided magnitude spectrum of a real signal, via FFT.

    Parameters
    ----------
    x : np.ndarray
        Real-valued time-domain signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs : np.ndarray
        Non-negative frequency bins in Hz (from 0 to fs/2).
    magnitude : np.ndarray
        Magnitude of the FFT at each frequency bin, normalized by the
        number of samples and doubled (except DC, and Nyquist when it
        exists) to account for the folded negative-frequency energy —
        the standard single-sided spectrum convention.
    """
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])

    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude = np.abs(spectrum) / n

    if n > 1:
        if n % 2 == 0:
            # Last bin is the Nyquist frequency - it should not be doubled.
            magnitude[1:-1] *= 2
        else:
            magnitude[1:] *= 2

    return freqs, magnitude


def dominant_frequency(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """Frequency (Hz) with the largest magnitude, ignoring the DC (0 Hz) bin."""
    if len(freqs) < 2:
        return 0.0
    idx_no_dc = int(np.argmax(magnitude[1:])) + 1
    return float(freqs[idx_no_dc])


def compute_spectrogram(x: np.ndarray, fs: float):
    """Time-frequency spectrogram via scipy.signal.spectrogram.

    Parameters
    ----------
    x : np.ndarray
        Real-valued time-domain signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    f : np.ndarray
        Frequency bins (Hz). Empty if the signal is too short.
    t_spec : np.ndarray
        Time bins (s), relative to the start of the signal.
    Sxx : np.ndarray
        Spectrogram power, shape (len(f), len(t_spec)).
    """
    n = len(x)
    if n < 16:
        # Too short for a meaningful windowed spectrogram.
        return np.array([]), np.array([]), np.zeros((0, 0))

    nperseg = min(256, n)
    noverlap = nperseg // 2
    f, t_spec, Sxx = sp_signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t_spec, Sxx


def prepare_for_spectral_analysis(signal_type: str, x: np.ndarray, fs: float) -> np.ndarray:
    """Band-limit certain signals to a physiologically realistic frequency
    range before frequency-domain analysis and display.

    The synthetic EMG generator (see utils/signal_generators.synthetic_emg)
    produces an enveloped burst of *unfiltered* Gaussian noise. That is a
    reasonable time-domain approximation, but it is spectrally flat
    (white) all the way to the Nyquist frequency, which is not realistic
    and would make a "dominant frequency" reading essentially random.
    Real surface EMG activity is concentrated in roughly the 20-450 Hz
    band, so for Experiment 3 - which specifically examines frequency
    content - the EMG signal is band-limited to that range before both
    analysis and display, so the time-domain and frequency-domain views
    shown together stay consistent. Other signal types are returned
    unchanged.
    """
    if signal_type != "EMG (Electromyogram)" or len(x) == 0:
        return x

    nyquist = fs / 2.0
    low = min(20.0, nyquist * 0.9)
    high = min(450.0, nyquist * 0.95)
    if high <= low:
        return x

    sos = sp_signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, x)


# --------------------------------------------------------------------------
# Short, signal-specific notes on typical frequency content, used in the
# "Biomedical Relevance" section of Experiment 3.
# --------------------------------------------------------------------------
BAND_INFO = {
    "ECG (Electrocardiogram)": (
        "Most ECG energy lies below about 40 Hz, dominated by a strong "
        "low-frequency component from the QRS complex (roughly 5-15 Hz) "
        "repeating at the heart-rate frequency."
    ),
    "EEG (Electroencephalogram)": (
        "EEG energy is concentrated below about 40 Hz and is conventionally "
        "split into named bands: delta (0.5-4 Hz), theta (4-8 Hz), "
        "alpha (8-13 Hz), and beta (13-30 Hz)."
    ),
    "EMG (Electromyogram)": (
        "EMG energy is broadband, typically spread across roughly "
        "20-450 Hz, reflecting the asynchronous firing of many motor "
        "units; the spectrum tends to shift toward lower frequencies as "
        "muscle fatigue develops."
    ),
    "EOG (Electrooculogram)": (
        "EOG energy is concentrated at very low frequencies (well below "
        "5 Hz), reflecting the slow, sweeping nature of eye movements, "
        "with brief higher-frequency content appearing during blinks."
    ),
}
