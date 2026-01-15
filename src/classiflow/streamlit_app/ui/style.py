"""Streamlit styling utilities."""

import streamlit as st


def use_theme():
    """Apply custom CSS theme."""
    st.markdown(
        """
        <style>
        /* Make images crisp */
        img { image-rendering: -webkit-optimize-contrast; }
        /* Tables: smaller line-height */
        .stDataFrame { font-size: 0.9rem; }
        .small { font-size: 0.85rem; color: #666; }
        </style>
        """,
        unsafe_allow_html=True,
    )
