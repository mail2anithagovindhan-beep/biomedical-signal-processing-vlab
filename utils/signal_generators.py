"""
utils/signal_generators.py
---------------------------
Signal-generation logic for Experiment 1 (Signal Generation).

This module is intentionally kept separate from the Streamlit UI code in
`pages/1_Signal_Generation.py`, so that the mathematical/synthetic-signal
definitions can be read, tested, and reused independently of the page
layout.

Two families of signals are provided:

1. Basic mathematical / test signals (sine, cosine, square, triangle,
   sawtooth, unit step, unit impulse, unit ramp, exponential).
2. Educational SYNTHETIC biomedical signals (ECG, EEG, EMG, EOG).

IMPORTANT: The biomedical signals generated here are simplified,
educational, synthetic approximations built from basic mathematical
building blocks (Gaussian pulses, sums of sinusoids, randomly placed
bursts). They are NOT derived from, or intended to resemble, real patient
recordings, and must not be presented as clinical or diagnostic data.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import signal as sp_signal


# --------------------------------------------------------------------------
# Time-axis helpers
# --------------------------------------------------------------------------
def make_time_axis(duration: float, fs: float, causal: bool = True) -> np.ndarray:
    """Build a time vector for a signal.

    Parameters
    ----------
    duration : float
        Signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    causal : bool
        If True, time runs from 0 to `duration` (used for periodic and
        biomedical signals). If False, time is centered on 0, running from
        -duration/2 to +duration/2 (used for Unit Step / Impulse / Ramp so
        the "before t=0" behaviour is visible).
    """
    n_samples = max(int(round(duration * fs)), 2)
    if causal:
        return np.linspace(0.0, duration, n_samples, endpoint=False)
    half = duration / 2.0
    return np.linspace(-half, half, n_samples, endpoint=False)


# --------------------------------------------------------------------------
# Category 1 - Basic mathematical signals
# --------------------------------------------------------------------------
def sine_wave(t, amplitude=1.0, frequency=1.0, **kwargs):
    """x(t) = A sin(2 pi f t)"""
    return amplitude * np.sin(2 * np.pi * frequency * t)


def cosine_wave(t, amplitude=1.0, frequency=1.0, **kwargs):
    """x(t) = A cos(2 pi f t)"""
    return amplitude * np.cos(2 * np.pi * frequency * t)


def square_wave(t, amplitude=1.0, frequency=1.0, **kwargs):
    """x(t) = A * sign(sin(2 pi f t)), via scipy.signal.square."""
    return amplitude * sp_signal.square(2 * np.pi * frequency * t)


def triangle_wave(t, amplitude=1.0, frequency=1.0, **kwargs):
    """Periodic triangular waveform (scipy.signal.sawtooth, width=0.5)."""
    return amplitude * sp_signal.sawtooth(2 * np.pi * frequency * t, width=0.5)


def sawtooth_wave(t, amplitude=1.0, frequency=1.0, **kwargs):
    """Periodic sawtooth waveform (scipy.signal.sawtooth, width=1)."""
    return amplitude * sp_signal.sawtooth(2 * np.pi * frequency * t, width=1.0)


def unit_step(t, amplitude=1.0, **kwargs):
    """x(t) = 0 for t < 0, 1 for t >= 0 (scaled by amplitude)."""
    return amplitude * np.where(t >= 0, 1.0, 0.0)


def unit_impulse(t, amplitude=1.0, **kwargs):
    """Discrete approximation of the unit impulse: a single non-zero sample
    at the time index closest to t = 0."""
    x = np.zeros_like(t)
    idx = int(np.argmin(np.abs(t)))
    x[idx] = amplitude
    return x


def unit_ramp(t, amplitude=1.0, **kwargs):
    """x(t) = t for t >= 0, 0 otherwise (scaled by amplitude)."""
    return amplitude * np.where(t >= 0, t, 0.0)


def exponential_signal(t, amplitude=1.0, rate=1.0, **kwargs):
    """x(t) = A * exp(k * t). `rate` (k) controls growth (k > 0) or decay
    (k < 0); this is a dedicated parameter, not treated as a frequency."""
    # Clip the exponent to avoid overflow warnings for large duration * |rate|
    exponent = np.clip(rate * t, -700, 700)
    return amplitude * np.exp(exponent)


# --------------------------------------------------------------------------
# Category 2 - Educational synthetic biomedical signals
# --------------------------------------------------------------------------
def _gaussian_pulse(t, mu, sigma, amp):
    return amp * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


def synthetic_ecg(t, amplitude=1.0, heart_rate=72.0, **kwargs):
    """Educational synthetic ECG built as a sum of Gaussian pulses
    (P, Q, R, S, T) repeated at a fixed heart rate. This is a simplified
    teaching model, not a physiological simulator.
    """
    rr = 60.0 / heart_rate  # seconds per beat
    x = np.zeros_like(t)

    # (fraction of RR interval for center, width as fraction of RR, relative amplitude)
    components = [
        (0.20, 0.045, 0.15),   # P wave
        (0.38, 0.010, -0.10),  # Q dip
        (0.40, 0.010, 1.00),   # R peak
        (0.42, 0.012, -0.20),  # S dip
        (0.60, 0.080, 0.30),   # T wave
    ]

    t_min, t_max = float(t.min()), float(t.max())
    first_beat = int(np.floor(t_min / rr)) - 1
    last_beat = int(np.ceil(t_max / rr)) + 1

    for beat in range(first_beat, last_beat + 1):
        beat_start = beat * rr
        for frac_center, frac_width, frac_amp in components:
            mu = beat_start + frac_center * rr
            sigma = max(frac_width * rr, 1e-4)
            x += _gaussian_pulse(t, mu, sigma, frac_amp * amplitude)

    return x


def synthetic_eeg(t, amplitude=1.0, **kwargs):
    """Educational synthetic EEG: a sum of sinusoids across classic EEG
    bands (delta, theta, alpha, beta) plus a small amount of noise."""
    rng = np.random.default_rng(42)
    bands = [(2.0, 0.30), (6.0, 0.25), (10.0, 0.30), (20.0, 0.15)]  # (Hz, relative weight)

    x = np.zeros_like(t)
    for freq, weight in bands:
        phase = rng.uniform(0, 2 * np.pi)
        x += weight * np.sin(2 * np.pi * freq * t + phase)

    noise = 0.05 * rng.standard_normal(t.shape)
    return amplitude * (x + noise)


def synthetic_emg(t, amplitude=1.0, **kwargs):
    """Educational synthetic EMG: bursts of high-frequency muscle-like
    activity with a randomly varying envelope."""
    rng = np.random.default_rng(7)
    duration = float(t.max() - t.min()) if t.size else 1.0
    n_bursts = max(1, int(duration / 0.5))
    burst_centers = rng.uniform(t.min(), t.max(), n_bursts)

    envelope = np.zeros_like(t)
    for center in burst_centers:
        width = rng.uniform(0.05, 0.15)
        envelope += np.exp(-((t - center) ** 2) / (2 * width ** 2))

    if envelope.max() > 0:
        envelope = envelope / envelope.max()

    raw_activity = rng.standard_normal(t.shape)
    return amplitude * envelope * raw_activity


def synthetic_eog(t, amplitude=1.0, **kwargs):
    """Educational synthetic EOG: a slow baseline drift representing eye
    movement, plus occasional sharper blink-like deflections."""
    rng = np.random.default_rng(3)
    slow_drift = 0.6 * np.sin(2 * np.pi * 0.2 * t)

    duration = float(t.max() - t.min()) if t.size else 1.0
    n_blinks = max(1, int(duration / 2.0))
    blink_times = rng.uniform(t.min(), t.max(), n_blinks)

    blinks = np.zeros_like(t)
    for blink_time in blink_times:
        width = 0.1
        blinks += 0.8 * np.exp(-((t - blink_time) ** 2) / (2 * width ** 2))

    return amplitude * (slow_drift + blinks)


# --------------------------------------------------------------------------
# Signal catalog: drives both the sidebar UI and the generation logic
# --------------------------------------------------------------------------
@dataclass
class SignalConfig:
    generator: Callable
    formula: str
    description: str
    needs_frequency: bool = True
    needs_rate: bool = False          # only True for Exponential
    causal_time: bool = True          # False => symmetric time axis about 0
    is_biomedical: bool = False


SIGNAL_CATALOG: dict[str, dict[str, SignalConfig]] = {
    "Basic Mathematical Signals": {
        "Sine Wave": SignalConfig(
            generator=sine_wave,
            formula="x(t) = A sin(2π f t)",
            description=(
                "A smooth periodic oscillation. Sine waves are the building "
                "blocks of Fourier analysis and are used to represent pure "
                "single-frequency components of a signal."
            ),
        ),
        "Cosine Wave": SignalConfig(
            generator=cosine_wave,
            formula="x(t) = A cos(2π f t)",
            description=(
                "Identical in shape to a sine wave but shifted by 90° "
                "(a quarter cycle) in phase."
            ),
        ),
        "Square Wave": SignalConfig(
            generator=square_wave,
            formula="x(t) = A · sign(sin(2π f t))",
            description=(
                "A periodic signal that alternates sharply between its "
                "maximum and minimum values. Square waves contain many odd "
                "harmonics of the fundamental frequency."
            ),
        ),
        "Triangle Wave": SignalConfig(
            generator=triangle_wave,
            formula="Periodic triangular waveform (linear rise and fall)",
            description=(
                "A periodic signal that rises and falls linearly between "
                "its extreme values, forming a triangular shape each cycle."
            ),
        ),
        "Sawtooth Wave": SignalConfig(
            generator=sawtooth_wave,
            formula="Periodic sawtooth waveform (linear ramp with sharp reset)",
            description=(
                "A periodic signal that ramps up (or down) linearly and "
                "then resets sharply, repeating every cycle."
            ),
        ),
        "Unit Step": SignalConfig(
            generator=unit_step,
            formula="x(t) = 0, t < 0;  x(t) = 1, t ≥ 0",
            description=(
                "A fundamental test signal that switches instantaneously "
                "from 0 to a constant value at t = 0. Used to study system "
                "response characteristics."
            ),
            needs_frequency=False,
            causal_time=False,
        ),
        "Unit Impulse": SignalConfig(
            generator=unit_impulse,
            formula="δ(t): 1 at t = 0, 0 elsewhere (discrete approximation)",
            description=(
                "An idealized signal that is zero everywhere except at "
                "t = 0. Here it is shown as a discrete approximation: a "
                "single non-zero sample at the time index nearest t = 0."
            ),
            needs_frequency=False,
            causal_time=False,
        ),
        "Unit Ramp": SignalConfig(
            generator=unit_ramp,
            formula="x(t) = t, t ≥ 0;  x(t) = 0, t < 0",
            description=(
                "A signal that increases linearly with time after t = 0. "
                "It is the running integral of the unit step."
            ),
            needs_frequency=False,
            causal_time=False,
        ),
        "Exponential Signal": SignalConfig(
            generator=exponential_signal,
            formula="x(t) = A e^{k t}",
            description=(
                "A signal that grows (k > 0) or decays (k < 0) "
                "exponentially with time. Exponential signals commonly "
                "describe charging/discharging and natural growth/decay "
                "processes."
            ),
            needs_frequency=False,
            needs_rate=True,
        ),
    },
    "Biomedical Signals": {
        "ECG (Electrocardiogram)": SignalConfig(
            generator=synthetic_ecg,
            formula="Sum of Gaussian pulses representing P, QRS, and T components",
            description=(
                "Educational synthetic biomedical signal. Represents the "
                "electrical activity of the heart over successive beats, "
                "including a P wave (atrial depolarization), a QRS complex "
                "(ventricular depolarization), and a T wave (ventricular "
                "repolarization)."
            ),
            needs_frequency=False,
            is_biomedical=True,
        ),
        "EEG (Electroencephalogram)": SignalConfig(
            generator=synthetic_eeg,
            formula="Sum of oscillations across delta, theta, alpha, and beta bands",
            description=(
                "Educational synthetic biomedical signal. Represents "
                "electrical brain activity as a mixture of oscillatory "
                "components across typical EEG frequency bands, with a "
                "small amount of background noise."
            ),
            needs_frequency=False,
            is_biomedical=True,
        ),
        "EMG (Electromyogram)": SignalConfig(
            generator=synthetic_emg,
            formula="Randomly-timed high-frequency bursts with varying envelope",
            description=(
                "Educational synthetic biomedical signal. Represents the "
                "electrical activity of skeletal muscle as bursts of "
                "high-frequency activity with varying amplitude, "
                "corresponding to periods of muscle contraction."
            ),
            needs_frequency=False,
            is_biomedical=True,
        ),
        "EOG (Electrooculogram)": SignalConfig(
            generator=synthetic_eog,
            formula="Slow baseline drift with occasional blink-like deflections",
            description=(
                "Educational synthetic biomedical signal. Represents eye "
                "movement activity as a slow-changing baseline with "
                "occasional sharper deflections representing blink-like "
                "events."
            ),
            needs_frequency=False,
            is_biomedical=True,
        ),
    },
}


def generate_signal(
    category: str,
    signal_type: str,
    amplitude: float,
    duration: float,
    fs: float,
    frequency: Optional[float] = None,
    rate: Optional[float] = None,
):
    """Look up the signal configuration and generate (t, x).

    Returns
    -------
    t : np.ndarray
        Time axis in seconds.
    x : np.ndarray
        Generated signal values.
    config : SignalConfig
        The catalog entry used, for description/formula/etc.
    """
    config = SIGNAL_CATALOG[category][signal_type]
    t = make_time_axis(duration, fs, causal=config.causal_time)
    x = config.generator(t, amplitude=amplitude, frequency=frequency, rate=rate)
    return t, x, config
