"""
Experiment 4 - Digital Butterworth Filter (placeholder page).
Real filter design/implementation logic is not implemented yet (Phase 1 scope).
"""

import streamlit as st
from utils.placeholder import render_placeholder

st.set_page_config(page_title="Exp 4 - Digital Butterworth Filter", page_icon="🧪", layout="wide")

render_placeholder(
    experiment_number=4,
    title="Digital Butterworth Filter",
    description=(
        "This experiment will let students design a digital IIR "
        "Butterworth filter (low-pass/high-pass/band-pass) and observe its "
        "effect on a test or biomedical signal, along with its frequency "
        "response."
    ),
)
