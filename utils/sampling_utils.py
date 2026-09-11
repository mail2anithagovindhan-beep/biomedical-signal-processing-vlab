"""
utils/sampling_utils.py
------------------------
Sampling / Nyquist / aliasing logic for Experiment 2 (Sampling and
Aliasing). Kept separate from the Streamlit UI in
pages/2_Sampling_and_Aliasing.py.

All signals here use a single sinusoid x(t) = A sin(2*pi*f*t), sampled at
a chosen sampling frequency Fs, so students can directly observe the
effect of Fs relative to the Nyquist requirement Fs >= 2f.
"""

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Continuous signal + sampling
# --------------------------------------------------------------------------
def continuous_signal(amplitude: float, frequency: float, duration: float, n_points: int = 2000):
    """A smooth, high-resolution version of x(t) = A sin(2 pi f t), used
    only for plotting the underlying continuous-time waveform (it is NOT
    the sampled signal)."""
    t = np.linspace(0.0, duration, max(n_points, 200))
    x = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, x


def sample_signal(amplitude: float, frequency: float, fs: float, duration: float):
    """Sample x(t) = A sin(2 pi f t) at sampling frequency `fs` over
    [0, duration]. Returns the sample times and sample values."""
    fs = max(fs, 1e-6)
    n_max = int(np.floor(duration * fs))
    n = np.arange(0, n_max + 1)
    t_samples = n / fs
    t_samples = t_samples[t_samples <= duration]
    x_samples = amplitude * np.sin(2 * np.pi * frequency * t_samples)
    return t_samples, x_samples


# --------------------------------------------------------------------------
# Nyquist criterion
# --------------------------------------------------------------------------
@dataclass
class NyquistResult:
    nyquist_frequency: float
    status: str            # "Adequately sampled" | "At Nyquist limit" | "Aliasing likely"
    alias_frequency: float  # apparent (folded) frequency, meaningful mainly when aliasing


def nyquist_frequency(frequency: float) -> float:
    """Minimum sampling frequency required by the Nyquist criterion, 2*f."""
    return 2.0 * frequency


def apparent_alias_frequency(frequency: float, fs: float) -> float:
    """Apparent ("folded") frequency that a real sinusoid of frequency
    `frequency`, sampled at `fs`, would appear to have once sampled -
    the classic aliasing fold-back formula. Only meaningful/relevant when
    the Nyquist criterion is violated; still well-defined otherwise.
    """
    if fs <= 0:
        return frequency
    k = round(frequency / fs)
    folded = abs(frequency - k * fs)
    # Fold into [0, fs/2]
    nyquist = fs / 2.0
    if folded > nyquist:
        folded = fs - folded
    return folded


def evaluate_sampling(frequency: float, fs: float, tolerance: float = 0.05) -> NyquistResult:
    """Classify the sampling scenario relative to the Nyquist criterion.

    - "Adequately sampled": Fs comfortably exceeds 2f (> 1.05 x 2f)
    - "At Nyquist limit":   Fs is within 5% of 2f
    - "Aliasing likely":    Fs is below 2f (by more than 5%)
    """
    nyq = nyquist_frequency(frequency)
    if nyq <= 0:
        status = "Adequately sampled"
    elif fs >= nyq * (1 + tolerance):
        status = "Adequately sampled"
    elif fs >= nyq * (1 - tolerance):
        status = "At Nyquist limit"
    else:
        status = "Aliasing likely"

    alias_f = apparent_alias_frequency(frequency, fs)
    return NyquistResult(nyquist_frequency=nyq, status=status, alias_frequency=alias_f)


# --------------------------------------------------------------------------
# Preset teaching cases (Part C / Part E of the experiment)
# --------------------------------------------------------------------------
PRESET_CASES = {
    "Case 1 - Proper Sampling (f = 2 Hz, Fs = 20 Hz)": {"frequency": 2.0, "fs": 20.0},
    "Case 2 - Near Nyquist (f = 5 Hz, Fs = 10 Hz)": {"frequency": 5.0, "fs": 10.0},
    "Case 3 - Aliasing (f = 8 Hz, Fs = 10 Hz)": {"frequency": 8.0, "fs": 10.0},
}
