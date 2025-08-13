import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from main import (
    run_simulation,
    run_simulation_with_external_temperature,
    feedback_temperature,
)

st.set_page_config(page_title="Evolutionary Learning (Moran-style)", layout="wide")

# ===============================
# Sidebar controls
# ===============================
st.sidebar.title("Simulation Parameters")

with st.sidebar.expander("Core horizon & population", expanded=True):
    total_time = st.number_input("Total time (min)", 100, 20000, 1000, step=100)
    max_population = st.number_input("Population cap (N)", 10, 20000, 1000, step=10)
    moran_after_cap = st.checkbox("Use Moran birth–death after reaching N", value=True)
    num_passive_wells = st.slider("Passive wells (share driver temp)", 1, 30, 5, 1)

with st.sidebar.expander("Feedback (driver)", expanded=True):
    feedback_mode = st.selectbox("Feedback mode", ["linear", "exp", "sigmoid", "step", "inverse"], index=0)
    feedback_sensitivity = st.slider("Feedback sensitivity  (lower values => more sensitive => temperature changes faster)" , 0.1, 5.0, 1.0, 0.1)

with st.sidebar.expander("Trait dynamics", expanded=True):
    inheritance_noise = st.slider("Inheritance noise (GFP SD)", 0.5, 10.0, 2.0, 0.5)
    base_boost_prob = st.slider("Max stress boost prob @39°C (per min)", 0.0, 0.02, 0.002, 0.001)

with st.sidebar.expander("Events & stability", expanded=True):
    high_gfp_threshold = st.slider("High-GFP threshold (event)", 0.0, 100.0, 20.0, 1.0)
    stationarity_window = st.slider("Stationarity window (min)", 10, 300, 60, 10)
    stationarity_tol = st.slider("Stationarity tolerance (°C)", 0.01, 1.0, 0.1, 0.01)

with st.sidebar.expander("Reproducibility", expanded=False):
    seed_on = st.checkbox("Set random seed", value=False)
    seed_val = st.number_input("Seed value", 0, 10_000_000, 42, step=1)

run_button = st.sidebar.button("Run simulation")

# ===============================
# Title
# ===============================
st.title("🧬 Evolutionary Learning via GFP Feedback (Continuous trait, Moran-style)")

st.caption(
    "Driver well: temperature updates every minute based on mean GFP. "
    "Passives follow the driver’s temperature but do not influence it. "
    "Controls are fixed at 39°C and 30°C. Once population reaches N, births replace random individuals (Moran)."
)


# ===============================
# Run simulations
# ===============================
if run_button or ("driver" not in st.session_state):
    if seed_on:
        np.random.seed(int(seed_val))

    with st.spinner("Running driver, passives, and controls..."):
        # DRIVER (feedback)
        driver = run_simulation(
            total_time=total_time,
            max_population=max_population,
            moran_after_cap=moran_after_cap,
            inheritance_noise=inheritance_noise,
            base_boost_prob=base_boost_prob,
            feedback_mode=feedback_mode,
            feedback_sensitivity=feedback_sensitivity,
            control_mode=None,
            start_temp=39,
            high_gfp_threshold=high_gfp_threshold,
            stationarity_window=stationarity_window,
            stationarity_tol=stationarity_tol,
        )

        # PASSIVES (follow driver temperature)
        passives = []
        # driver tuple indices for temp & time: (df, temp_hist, temp_times, mean_gfp_hist, mean_gen_hist, times, ...)
        _, temp_driver, temp_t_driver, _, _, _, _, _, _, _ = driver
        for _ in range(num_passive_wells):
            res = run_simulation_with_external_temperature(
                external_temp_times=temp_t_driver,
                external_temp_values=temp_driver,
                total_time=total_time,
                max_population=max_population,
                moran_after_cap=moran_after_cap,
                inheritance_noise=inheritance_noise,
                base_boost_prob=base_boost_prob,
                high_gfp_threshold=high_gfp_threshold,
                stationarity_window=stationarity_window,
                stationarity_tol=stationarity_tol,
            )
            passives.append(res)

        # CONTROLS
        ctrl39 = run_simulation(
            total_time=total_time,
            max_population=max_population,
            moran_after_cap=moran_after_cap,
            inheritance_noise=inheritance_noise,
            base_boost_prob=base_boost_prob,
            feedback_mode=feedback_mode,
            feedback_sensitivity=feedback_sensitivity,
            control_mode="fixed39",
            start_temp=39,
            high_gfp_threshold=high_gfp_threshold,
            stationarity_window=stationarity_window,
            stationarity_tol=stationarity_tol,
        )

        ctrl30 = run_simulation(
            total_time=total_time,
            max_population=max_population,
            moran_after_cap=moran_after_cap,
            inheritance_noise=inheritance_noise,
            base_boost_prob=base_boost_prob,
            feedback_mode=feedback_mode,
            feedback_sensitivity=feedback_sensitivity,
            control_mode="fixed30",
            start_temp=30,
            high_gfp_threshold=high_gfp_threshold,
            stationarity_window=stationarity_window,
            stationarity_tol=stationarity_tol,
        )

        st.session_state.driver = driver
        st.session_state.passives = passives
        st.session_state.ctrl39 = ctrl39
        st.session_state.ctrl30 = ctrl30


# ===============================
# Unpack state
# ===============================
if "driver" in st.session_state:
    (
        df_driver,
        temp_driver, temp_t_driver,
        gfp_driver, gen_driver, time_driver,
        pop_driver,
        t_first_high_driver,
        t_stationary_driver,
        final_gfps_driver,
    ) = st.session_state.driver

    passives = st.session_state.passives
    (
        df_c39, temp_c39, t_c39, gfp_c39, gen_c39, time_c39, pop_c39,
        t_first_high_c39, t_stat_c39, final_gfps_c39,
    ) = st.session_state.ctrl39

    (
        df_c30, temp_c30, t_c30, gfp_c30, gen_c30, time_c30, pop_c30,
        t_first_high_c30, t_stat_c30, final_gfps_c30,
    ) = st.session_state.ctrl30

    # ===============================
    # TOP METRICS
    # ===============================
    st.subheader("🔎 Key Events / Endpoints")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Driver: 1st high-GFP (min)", t_first_high_driver if t_first_high_driver is not None else "–")
        st.metric("Driver: Stationarity (min)", t_stationary_driver if t_stationary_driver is not None else "–")
    with c2:
        st.metric("Control 39°C: 1st high-GFP", t_first_high_c39 if t_first_high_c39 is not None else "–")
        st.metric("Control 39°C: final mean GFP", round(gfp_c39[-1], 2))
    with c3:
        st.metric("Control 30°C: 1st high-GFP", t_first_high_c30 if t_first_high_c30 is not None else "–")
        st.metric("Control 30°C: final mean GFP", round(gfp_c30[-1], 2))

    st.caption(
        "Learning pattern: rare high-GFP event → immediate cooling → shorter generation times → faster reproduction "
        "→ heritable maintenance → temperature stationarity."
    )

    # ===============================
    # TEMPERATURE (Driver)
    # ===============================
    st.markdown("### 🌡 Driver Temperature Over Time")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=temp_t_driver, y=temp_driver, mode="lines", name="Driver temp"))
    if t_first_high_driver is not None:
        fig_temp.add_vline(x=t_first_high_driver, line=dict(color="green", dash="dot"))
        fig_temp.add_annotation(x=t_first_high_driver, y=temp_driver[min(len(temp_driver)-1, t_first_high_driver)],
                                text="1st high-GFP", showarrow=True, yshift=20)
    if t_stationary_driver is not None:
        fig_temp.add_vline(x=t_stationary_driver, line=dict(color="orange", dash="dash"))
        fig_temp.add_annotation(x=t_stationary_driver, y=temp_driver[min(len(temp_driver)-1, t_stationary_driver)],
                                text="Stationary", showarrow=True, yshift=20)
    fig_temp.update_layout(xaxis_title="Time (min)", yaxis_title="Temperature (°C)", template="plotly_white")
    st.plotly_chart(fig_temp, key="temp_chart", use_container_width=True)

    # ===============================
    # MEAN GFP (Driver vs Controls + Passive avg)
    # ===============================
    st.markdown("### 🟢 Mean GFP (Driver, Controls, Passive Avg)")
    fig_gfp = go.Figure()
    fig_gfp.add_trace(go.Scatter(x=time_driver, y=gfp_driver, name="Driver", line=dict(width=3)))
    fig_gfp.add_trace(go.Scatter(x=time_c39, y=gfp_c39, name="Control 39°C", line=dict(dash="dash")))
    fig_gfp.add_trace(go.Scatter(x=time_c30, y=gfp_c30, name="Control 30°C", line=dict(dash="dot")))
    if passives:
        gfp_passive_avg = np.mean([res[3] for res in passives], axis=0)
        fig_gfp.add_trace(go.Scatter(x=passives[0][5], y=gfp_passive_avg, name=f"Passive (avg of {len(passives)})"))
    fig_gfp.update_layout(xaxis_title="Time (min)", yaxis_title="Mean GFP (0–100)", template="plotly_white")
    st.plotly_chart(fig_gfp, key="mean_gfp_chart", use_container_width=True)

    # ===============================
    # ALL WELLS (UNAGGREGATED GFP)
    # ===============================
    st.markdown("### 🌊 GFP — All Wells (Unaggregated)")
    fig_all = go.Figure()
    # Driver
    fig_all.add_trace(go.Scatter(x=time_driver, y=gfp_driver, name="Driver", line=dict(width=3)))
    # Passives
    for i, res in enumerate(passives):
        _, _, _, gfp_p, _, time_p, _, _, _, _ = res
        fig_all.add_trace(go.Scatter(x=time_p, y=gfp_p, name=f"Passive {i+1}", opacity=0.4, line=dict(width=1)))
    # Controls
    fig_all.add_trace(go.Scatter(x=time_c39, y=gfp_c39, name="Control 39°C", line=dict(dash="dash")))
    fig_all.add_trace(go.Scatter(x=time_c30, y=gfp_c30, name="Control 30°C", line=dict(dash="dot")))
    fig_all.update_layout(xaxis_title="Time (min)", yaxis_title="Mean GFP (0–100)", template="plotly_white", showlegend=True)
    st.plotly_chart(fig_all, key="all_gfp_chart", use_container_width=True)

    # ===============================
    # POPULATION SIZE
    # ===============================
    st.markdown("### 👥 Population Size")
    fig_pop = go.Figure()
    fig_pop.add_trace(go.Scatter(x=time_driver, y=pop_driver, name="Driver"))
    for i, res in enumerate(passives):
        fig_pop.add_trace(go.Scatter(x=res[5], y=res[6], name=f"Passive {i+1}", opacity=0.35))
    fig_pop.add_trace(go.Scatter(x=time_c39, y=pop_c39, name="Control 39°C", line=dict(dash="dash")))
    fig_pop.add_trace(go.Scatter(x=time_c30, y=pop_c30, name="Control 30°C", line=dict(dash="dot")))
    fig_pop.update_layout(xaxis_title="Time (min)", yaxis_title="# Cells", template="plotly_white")
    st.plotly_chart(fig_pop, key="pop_chart", use_container_width=True)

    # ===============================
    # PHASE PLOT (Driver): GFP -> Temperature
    # ===============================
    st.markdown("### 🔁 Phase Plot (Driver): Mean GFP → Temperature")
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=gfp_driver, y=temp_driver, mode="lines+markers", name="Trajectory"))
    fig_phase.update_layout(xaxis_title="Mean GFP (0–100)", yaxis_title="Temperature (°C)", template="plotly_white")
    st.plotly_chart(fig_phase, key="phase_chart", use_container_width=True)
    st.caption("Interpretation: as mean GFP rises, temperature drops immediately. A stable loop appears as the trajectory settles.")

    # ===============================
    # FINAL GFP DISTRIBUTIONS
    # ===============================
    st.markdown("### 🧪 Final GFP Distributions (cells alive at end)")
    bins = np.linspace(0, 100, 51)
    centers = (bins[:-1] + bins[1:]) / 2.0

    def hist_counts(values, bins):
        counts, _ = np.histogram(np.array(values, dtype=float), bins=bins)
        return counts

    # Aggregate all passives' final cells
    all_passive_final = np.concatenate([np.array(res[-1]) for res in passives]) if passives else np.array([])

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(x=centers, y=hist_counts(final_gfps_driver, bins), name="Driver", opacity=0.65))
    fig_hist.add_trace(go.Bar(x=centers, y=hist_counts(final_gfps_c39, bins), name="Control 39°C", opacity=0.5))
    fig_hist.add_trace(go.Bar(x=centers, y=hist_counts(final_gfps_c30, bins), name="Control 30°C", opacity=0.5))
    if all_passive_final.size > 0:
        fig_hist.add_trace(go.Bar(x=centers, y=hist_counts(all_passive_final, bins), name="Passives (all cells)", opacity=0.45))
    fig_hist.update_layout(barmode="overlay", xaxis_title="GFP (0–100)", yaxis_title="Count", template="plotly_white")
    st.plotly_chart(fig_hist, key="hist_chart", use_container_width=True)

    # ===============================
    # RAW DATA (on demand)
    # ===============================
    with st.expander("Raw division data — Driver"):
        st.dataframe(df_driver, use_container_width=True)

    with st.expander("Raw division data — Controls"):
        st.write("Control 39°C")
        st.dataframe(df_c39, use_container_width=True)
        st.write("Control 30°C")
        st.dataframe(df_c30, use_container_width=True)


# ===============================
# Plot feedback function
# ===============================
def plot_feedback_function():
    x = np.linspace(0, 100, 300)
    y = [feedback_temperature(xx, mode=feedback_mode, base_temp=30, max_temp=39, sensitivity=feedback_sensitivity)
         for xx in x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{feedback_mode}"))
    fig.update_layout(
        title=f"Feedback Function (Mean GFP → Temperature) | mode={feedback_mode}, sens={feedback_sensitivity}",
        xaxis_title="Mean GFP (0–100)",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
    )
    st.plotly_chart(fig, key="feedback_fn_chart", use_container_width=True)

plot_feedback_function()
