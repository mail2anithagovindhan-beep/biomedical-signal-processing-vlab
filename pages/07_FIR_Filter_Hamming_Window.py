"""
Experiment 7 - Design of Digital FIR Filter Using Hamming Window

Designs a windowed FIR filter (Low-Pass, High-Pass, Band-Pass, or
Band-Stop) using a Hamming window and applies it to a chosen test or
biomedical signal.

All of the actual UI logic is shared with Experiment 8 (Hanning window),
since the two experiments are identical except for the window function;
see utils/fir_page_render.py for the shared implementation, and
utils/fir_filter_utils.py for the underlying FIR design math.
"""

from utils.fir_page_render import render_fir_experiment

render_fir_experiment(experiment_number=7, window="Hamming")
