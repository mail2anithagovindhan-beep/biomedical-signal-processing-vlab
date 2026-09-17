"""
Experiment 8 - Design of Digital FIR Filter Using Hanning Window

Designs a windowed FIR filter (Low-Pass, High-Pass, Band-Pass, or
Band-Stop) using a Hanning (Hann) window and applies it to a chosen test
or biomedical signal.

All of the actual UI logic is shared with Experiment 7 (Hamming window),
since the two experiments are identical except for the window function;
see utils/fir_page_render.py for the shared implementation, and
utils/fir_filter_utils.py for the underlying FIR design math.
"""

from utils.fir_page_render import render_fir_experiment

render_fir_experiment(experiment_number=8, window="Hanning")
