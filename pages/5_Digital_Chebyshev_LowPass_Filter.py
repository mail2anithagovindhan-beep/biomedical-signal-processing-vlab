"""
Experiment 5 - Digital Low-Pass Chebyshev Type-I IIR Filter (placeholder page).
Real filter design/implementation logic is not implemented yet (Phase 1 scope).
"""

import streamlit as st
from utils.placeholder import render_placeholder

st.set_page_config(page_title="Exp 5 - Digital Chebyshev Type-I Low-Pass Filter", page_icon="🧪", layout="wide")

render_placeholder(
    experiment_number=5,
    title="Digital Chebyshev Type-I Low-Pass Filter",
    description=(
        "This experiment will let students design a digital IIR Chebyshev "
        "Type-I low-pass filter, compare its passband ripple and roll-off "
        "characteristics against the Butterworth filter, and apply it to "
        "a test or biomedical signal."
    ),
)
