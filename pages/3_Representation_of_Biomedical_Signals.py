"""
Experiment 3 - Representation of Biomedical Signals (placeholder page).
Real representation/visualization logic is not implemented yet (Phase 1 scope).
"""

import streamlit as st
from utils.placeholder import render_placeholder

st.set_page_config(page_title="Exp 3 - Representation of Biomedical Signals", page_icon="🧪", layout="wide")

render_placeholder(
    experiment_number=3,
    title="Representation of Biomedical Signals",
    description=(
        "This experiment will present time-domain and frequency-domain "
        "representations of biomedical signals (e.g. ECG, EMG, EEG), "
        "using sample datasets to illustrate key waveform features."
    ),
    colab=True,
)
