"""
Home.py
--------
Main entry point for the Biomedical Signal Processing Virtual Lab.

This Streamlit app is organized as a multipage application. Streamlit
automatically builds the sidebar navigation from the files placed in the
`pages/` folder (each numbered file becomes one experiment page).

Phase 1 scope: application framework only (this file + placeholder pages).
No experiment logic, datasets, or signal-processing algorithms are
implemented yet.
"""

import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Biomedical Signal Processing Virtual Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🧪 BSP Virtual Lab")
    st.caption("Biomedical Signal Processing")
    st.markdown("---")
    st.markdown(
        "Use the navigation above to open the **Home** page or any of the "
        "six experiment pages."
    )
    st.markdown("---")
    st.markdown("**External Resources**")
    st.caption("Desmos, Google Colab and Google Sites links will appear here.")

# --------------------------------------------------------------------------
# Hero banner (original decorative graphic - waveform, spectrogram, system
# block diagram, ECG trace)
# --------------------------------------------------------------------------
st.image("assets/hero_banner.png", use_container_width=True)

# --------------------------------------------------------------------------
# Institution header (logos)
# --------------------------------------------------------------------------
logo_left, logo_center, logo_right = st.columns([1, 4, 1])
with logo_left:
    st.image("assets/srm_logo.png", width=100)
with logo_right:
    st.image("assets/dept_logo.png", width=100)
with logo_center:
    st.markdown(
        "<h4 style='text-align:center; margin-bottom:0;'>SRM Institute of Science and Technology</h4>"
        "<p style='text-align:center; margin-top:0;'>Department of Biomedical Engineering</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------
st.title("Biomedical Signal Processing Virtual Lab")
st.subheader("An interactive educational platform for digital signal processing concepts in biomedical engineering")

st.markdown("---")

st.markdown(
    """
    ### About this Virtual Lab

    This Virtual Lab is designed to help students explore core concepts in
    **Biomedical Signal Processing** through interactive, hands-on
    experiments. Each experiment page in the sidebar corresponds to a
    specific topic, allowing you to visualize signals, understand sampling
    behavior, and study digital filter and transform techniques used in
    biomedical engineering applications such as ECG, EMG, and EEG analysis.

    ### Experiments in this Lab

    1. **Signal Generation** — Generate and visualize standard biomedical and test signals.
    2. **Sampling and Aliasing** — Explore the effects of sampling rate on signal fidelity.
    3. **Representation of Biomedical Signals** — Examine time-domain and frequency-domain representations.
    4. **Digital Butterworth Filter** — Design and apply IIR Butterworth filters.
    5. **Digital Chebyshev Type-I Low-Pass Filter** — Design and apply IIR Chebyshev Type-I filters.
    6. **DIT FFT (2/4/8-point)** — Study the Decimation-in-Time Fast Fourier Transform.

    ### Companion Tools

    Alongside this Streamlit application, the lab uses a few companion
    tools for specific purposes:

    - **Desmos** — interactive graphical demonstrations for sampling/aliasing and DIT FFT (2, 4, 8-point).
    - **Google Colab** — dataset-based experiments and notebooks where appropriate.
    - **Google Sites** — the student-facing portal for the overall course.

    Links to these resources will be added as they become available.
    """
)

st.markdown("---")

# --------------------------------------------------------------------------
# Faculty Profile
# --------------------------------------------------------------------------
st.markdown("## Faculty Profile")

profile_col, details_col = st.columns([1, 2])

with profile_col:
    st.image("assets/faculty_photo.jpg", width=180)
    st.markdown(
        "**Dr. G. Anitha**  \n"
        "Assistant Professor  \n"
        "Department of Biomedical Engineering  \n"
        "School of Bioengineering  \n"
        "SRM Institute of Science and Technology"
    )
    st.markdown("**Qualification**  \nPh.D. Biomedical Engineering")
    st.markdown(
        "**Contact**  \n"
        "Email: anitha.g@srmist.edu.in  \n"
        "Department of Biomedical Engineering, SRMIST"
    )

with details_col:
    st.markdown(
        "**Research Interests**\n\n"
        "- Biomedical Signal Processing\n"
        "- Wearable Sensors\n"
        "- Medical Robotics\n"
        "- Healthcare IoT\n"
        "- Rehabilitation Engineering\n"
        "- AI in Healthcare"
    )
    st.markdown(
        "**Research Areas**  \n"
        "Biomedical Signals · Biosensors · Medical Devices · Machine Learning in Healthcare · Digital Health"
    )
    st.markdown(
        "**Developed Laboratory**  \n"
        "Signals and Systems Virtual Laboratory — Simulation-Based Interactive Learning Platform\n\n"
        "✔ 10 Interactive Experiments  \n"
        "✔ Desmos Simulations  \n"
        "✔ Viva Questions  \n"
        "✔ Outcome-Based Learning  \n"
        "✔ Self-paced Learning"
    )

st.caption("Dr. G. Anitha / Department of Biomedical Engineering / SRM Institute of Science and Technology")

st.markdown("---")
st.caption("Status: Application framework (Phase 1). Experiment content is under development.")
