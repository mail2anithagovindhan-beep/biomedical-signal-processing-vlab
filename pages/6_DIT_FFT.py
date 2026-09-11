"""
Experiment 6 - DIT FFT (2-point, 4-point and 8-point) (placeholder page).
Real FFT algorithm logic is not implemented yet (Phase 1 scope).
"""

import streamlit as st
from utils.placeholder import render_placeholder

st.set_page_config(page_title="Exp 6 - DIT FFT", page_icon="🧪", layout="wide")

render_placeholder(
    experiment_number=6,
    title="DIT FFT (2-point, 4-point and 8-point)",
    description=(
        "This experiment will walk through the Decimation-in-Time Fast "
        "Fourier Transform algorithm for 2-point, 4-point, and 8-point "
        "cases, showing the butterfly structure and computed spectra."
    ),
    desmos=True,
)
