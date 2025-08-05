# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from main import run_simulation  # Ensure main.py is in the same folder or in PYTHONPATH

# === Sidebar Controls ===
st.sidebar.title("Simulation Settings")

feedback_mode = st.sidebar.selectbox(
    "Feedback Mode",
    options=["exp", "linear", "sigmoid", "step", "inverse"],
    index=1
)

feedback_sensitivity = st.sidebar.slider(
    "Feedback Sensitivity", min_value=0.1, max_value=5.0, value=1.0, step=0.1
)

max_switch_rate = st.sidebar.slider(
    "Max Phenotypic Switch Rate", min_value=0.0, max_value=0.2, value=0.1, step=0.01
)

use_control_mode = st.sidebar.checkbox("Control Mode (No Feedback)", value=False)

run_button = st.sidebar.button("Run Simulation")


# === Main App ===
st.title("🧬 Evolutionary Learning via GFP Feedback")

if run_button:
    with st.spinner("Running simulation..."):
        df, div_times, temp_hist, temp_times, gfp_hist, gen_hist, time_hist = run_simulation(
            feedback_mode=feedback_mode,
            feedback_sensitivity=feedback_sensitivity,
            max_switch_rate=max_switch_rate,
            control_mode=use_control_mode
        )

    st.success("Simulation complete!")
    st.markdown(f"**Total Divisions:** {len(div_times) - 1}")

    # GFP over time
    fig_gfp = go.Figure()
    fig_gfp.add_trace(go.Scatter(x=time_hist, y=gfp_hist, mode='lines', name='Mean GFP'))
    fig_gfp.update_layout(title="Mean GFP Over Time", xaxis_title="Time (min)", yaxis_title="Mean GFP")
    st.plotly_chart(fig_gfp)

    # Temperature over time
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=temp_times, y=temp_hist, mode='lines', name='Temperature'))
    fig_temp.update_layout(title="Temperature Over Time", xaxis_title="Time (min)", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig_temp)

    # Generation time over time
    fig_gen = go.Figure()
    fig_gen.add_trace(go.Scatter(x=time_hist, y=gen_hist, mode='lines', name='Generation Time'))
    fig_gen.update_layout(title="Generation Time Over Time", xaxis_title="Time (min)", yaxis_title="Generation Time (min)")
    st.plotly_chart(fig_gen)

    # GFP histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df['GFP'], nbinsx=50, name='GFP'))
    fig_hist.update_layout(title="GFP Expression Distribution", xaxis_title="GFP Level", yaxis_title="Count")
    st.plotly_chart(fig_hist)

    # Optional data table
    if st.checkbox("Show Raw Division Data"):
        st.dataframe(df)

else:
    st.info("Adjust parameters on the left and click **Run Simulation**.")

