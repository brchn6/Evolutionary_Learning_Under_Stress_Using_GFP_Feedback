# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from main import run_simulation, run_simulation_with_external_temperature

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

num_passive_wells = st.sidebar.slider(
    "Number of Passive Wells", min_value=1, max_value=100, value=10, step=1
)

use_gaussian_inheritance = st.sidebar.checkbox("Use Gaussian Inheritance (Continuous GFP)", value=False)

run_button = st.sidebar.button("Run Simulation")

# === Main App ===
st.title("🧬 Evolutionary Learning via GFP Feedback")
st.markdown("Simulating **1 driver well** (coupled to feedback), "
            "**N passive wells** (sharing the environment), and "
            "a **control well** (constant stress at 39°C).")

# === Run Simulation ===
if run_button or 'df_driver' not in st.session_state:
    with st.spinner("Running full simulation (driver + passive + control)..."):
        # DRIVER
        df_driver, div_driver, temp_driver, temp_t_driver, gfp_driver, gen_driver, time_driver = run_simulation(
            feedback_mode=feedback_mode,
            feedback_sensitivity=feedback_sensitivity,
            max_switch_rate=max_switch_rate,
            control_mode=False,
            use_gaussian_inheritance=use_gaussian_inheritance
        )

        # PASSIVE
        passive_results = []
        for i in range(num_passive_wells):
            df_passive, div_passive, gfp_passive, gen_passive, time_passive = run_simulation_with_external_temperature(
                external_temp_times=temp_t_driver,
                external_temp_values=temp_driver,
                max_switch_rate=max_switch_rate,
                use_gaussian_inheritance=use_gaussian_inheritance
            )
            passive_results.append({
                'df': df_passive,
                'div': div_passive,
                'gfp': gfp_passive,
                'gen': gen_passive,
                'time': time_passive
            })

        avg_gfp_passive = np.mean([res['gfp'] for res in passive_results], axis=0)
        avg_gen_passive = np.mean([res['gen'] for res in passive_results], axis=0)
        avg_div_passive = np.mean([len(res['div']) for res in passive_results])
        combined_df_passive = pd.concat([res['df'] for res in passive_results])

        # CONTROL
        df_control, div_control, temp_control, temp_t_control, gfp_control, gen_control, time_control = run_simulation(
            max_switch_rate=max_switch_rate,
            control_mode=True,
            use_gaussian_inheritance=use_gaussian_inheritance
        )

        # Store in session_state
        st.session_state.df_driver = df_driver
        st.session_state.combined_df_passive = combined_df_passive
        st.session_state.df_control = df_control
        st.session_state.gfp_driver = gfp_driver
        st.session_state.avg_gfp_passive = avg_gfp_passive
        st.session_state.gfp_control = gfp_control
        st.session_state.gen_driver = gen_driver
        st.session_state.avg_gen_passive = avg_gen_passive
        st.session_state.gen_control = gen_control
        st.session_state.time_driver = time_driver
        st.session_state.time_control = time_control
        st.session_state.temp_driver = temp_driver
        st.session_state.temp_t_driver = temp_t_driver
        st.session_state.num_passive_wells = num_passive_wells
        st.session_state.div_driver = div_driver
        st.session_state.div_control = div_control
        st.session_state.avg_div_passive = avg_div_passive

# === Display Results ===
if 'df_driver' in st.session_state:
    st.success("Simulation complete!")
    st.markdown(f"**Driver divisions:** {len(st.session_state.div_driver) - 1}")
    st.markdown(f"**Avg Passive divisions ({st.session_state.num_passive_wells} wells):** {int(st.session_state.avg_div_passive)}")
    st.markdown(f"**Control divisions:** {len(st.session_state.div_control) - 1}")

    # Terminal log
    print("=== Simulation Summary ===")
    print(f"Feedback Mode: {feedback_mode}")
    print(f"Feedback Sensitivity: {feedback_sensitivity}")
    print(f"Max Switch Rate: {max_switch_rate}")
    print(f"Passive Wells: {num_passive_wells}")
    print(f"Gaussian Inheritance: {use_gaussian_inheritance}")
    print(f"Driver Final GFP: {np.round(st.session_state.gfp_driver[-1], 3)}")
    print(f"Passive Final GFP (avg): {np.round(st.session_state.avg_gfp_passive[-1], 3)}")
    print(f"Control Final GFP: {np.round(st.session_state.gfp_control[-1], 3)}")
    print("================================")

    # Mean GFP
    fig_gfp = go.Figure()
    fig_gfp.add_trace(go.Scatter(x=st.session_state.time_driver, y=st.session_state.gfp_driver, mode='lines', name='Driver'))
    fig_gfp.add_trace(go.Scatter(x=st.session_state.time_driver, y=st.session_state.avg_gfp_passive, mode='lines', name='Passive (avg)'))
    fig_gfp.add_trace(go.Scatter(x=st.session_state.time_control, y=st.session_state.gfp_control, mode='lines', name='Control'))
    fig_gfp.update_layout(title="Mean GFP Expression Over Time", xaxis_title="Time (min)", yaxis_title="Mean GFP")
    st.plotly_chart(fig_gfp)

    # Temperature
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=st.session_state.temp_t_driver, y=st.session_state.temp_driver, mode='lines', name='Temperature (Driver)'))
    fig_temp.update_layout(title="Temperature Over Time (Driver & Passive)", xaxis_title="Time (min)", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig_temp)

    # Generation Time
    fig_gen = go.Figure()
    fig_gen.add_trace(go.Scatter(x=st.session_state.time_driver, y=st.session_state.gen_driver, mode='lines', name='Driver'))
    fig_gen.add_trace(go.Scatter(x=st.session_state.time_driver, y=st.session_state.avg_gen_passive, mode='lines', name='Passive (avg)'))
    fig_gen.add_trace(go.Scatter(x=st.session_state.time_control, y=st.session_state.gen_control, mode='lines', name='Control'))
    fig_gen.update_layout(title="Mean Generation Time Over Time", xaxis_title="Time (min)", yaxis_title="Generation Time (min)")
    st.plotly_chart(fig_gen)

    # Final GFP Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=st.session_state.df_driver['GFP'], nbinsx=50, name='Driver', opacity=0.7))
    fig_hist.add_trace(go.Histogram(x=st.session_state.combined_df_passive['GFP'], nbinsx=50, name='Passive (all)', opacity=0.5))
    fig_hist.add_trace(go.Histogram(x=st.session_state.df_control['GFP'], nbinsx=50, name='Control', opacity=0.5))
    fig_hist.update_layout(barmode='overlay', title="Final GFP Expression Distribution", xaxis_title="GFP Level", yaxis_title="Count")
    st.plotly_chart(fig_hist)

    # Raw Data Toggles
    if st.checkbox("Show Raw Driver Division Data"):
        st.dataframe(st.session_state.df_driver)

    if st.checkbox("Show Raw Passive Data (Combined)"):
        st.dataframe(st.session_state.combined_df_passive)

    if st.checkbox("Show Raw Control Data"):
        st.dataframe(st.session_state.df_control)

else:
    st.info("Adjust parameters and click **Run Simulation** to begin.")
