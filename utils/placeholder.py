"""
utils/placeholder.py
---------------------
Shared helper used by the experiment pages during Phase 1 (framework-only
stage). Each experiment page calls `render_placeholder(...)` so that all
pages look and behave consistently until their real logic is implemented.

No signal-processing logic lives here — this is presentation only.
"""

import streamlit as st


def render_placeholder(
    experiment_number: int,
    title: str,
    description: str,
    desmos: bool = False,
    colab: bool = False,
) -> None:
    """Render a consistent placeholder layout for an experiment page.

    Parameters
    ----------
    experiment_number : int
        The experiment number (1-6) shown in the page header.
    title : str
        The experiment title.
    description : str
        A short description of what this experiment will eventually cover.
    desmos : bool
        Whether this experiment will have a companion Desmos activity.
    colab : bool
        Whether this experiment will have a companion Google Colab notebook.
    """
    st.title(f"Experiment {experiment_number} — {title}")
    st.markdown("---")

    st.info("This experiment page is a placeholder. Implementation is planned for a later phase.")

    st.markdown("### Overview")
    st.write(description)

    if desmos or colab:
        st.markdown("### External Resources")
        if desmos:
            st.markdown("- **Desmos activity:** _link to be added_")
        if colab:
            st.markdown("- **Google Colab notebook:** _link to be added_")

    st.markdown("---")
    st.caption("Status: Not yet implemented (Phase 1 — framework only).")
