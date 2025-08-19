# app.py — Minimal integrated UI: Moran + Learning + Epigenetic Memory
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# import core
from src.main import (
    SimulationParams, GFPParams, MemoryParams,
    run_combined_experiment, calculate_learning_metrics, validate_parameters
)

st.set_page_config(page_title="🧬 Moran Learning + Epigenetic Memory", layout="wide")
st.title("🧬 Learning in a Moran Process + Epigenetic Memory (One-Slider Demo)")
st.caption("Driver well learns via temperature feedback; passives mirror temperature; controls fixed (30/39 °C). "
           "Epigenetic memory listens to dT/dt and biases phenotype switching with inheritance.")

# -------------------------- ONE SLIDER ---------------------------------------
mem_influence = st.slider("Epigenetic influence (0 = off, 1 = strong)", 0.0, 1.0, 0.8, 0.05)

# ----------------------- Optional minimal advanced settings -------------------
with st.expander("Advanced (optional)", expanded=False):
    col = st.columns(6)
    mode            = col[0].selectbox("GFP mode", ["continuous", "binary"], index=0)
    population_size = col[1].slider("Population size", 100, 1000, 200, 50)
    total_time      = col[2].slider("Total time (min)", 300, 3000, 1000, 100)
    dt              = col[3].slider("Δt (min)", 1, 10, 1, 1)
    passives_n      = col[4].slider("Passive wells", 0, 6, 2, 1)
    seed            = int(col[5].number_input("Random seed", 0, 10**9, 42, 1))

    col2 = st.columns(5)
    fb_mode   = col2[0].selectbox("Feedback", ["linear", "sigmoid", "step", "exponential"], index=0)
    fb_sens   = col2[1].slider("Feedback sensitivity", 0.2, 3.0, 1.0, 0.1)
    inertia   = col2[2].slider("Temp inertia", 0.05, 1.0, 0.25, 0.05)
    sel_mode  = col2[3].selectbox("Moran selection", ["fitness", "neutral"], index=0)
    init_sd   = col2[4].slider("Initial GFP SD", 1.0, 30.0, 12.0, 1.0)

run_btn = st.button("Run simulation", type="primary")

if run_btn:
    # defaults if advanced panel unopened
    if "mode" not in locals():
        mode, population_size, total_time, dt = "continuous", 200, 1000, 1
        passives_n, seed = 2, 42
        fb_mode, fb_sens, inertia, sel_mode, init_sd = "linear", 1.0, 0.25, "fitness", 12.0

    sim_p = SimulationParams(
        population_size=population_size, total_time=total_time, time_step=dt,
        num_passive_wells=passives_n, feedback_mode=fb_mode, feedback_sensitivity=fb_sens,
        temp_inertia=inertia, start_at_max_temp=True, selection_mode=sel_mode, random_seed=seed
    )
    gfp_p = GFPParams(init_sd=init_sd)
    mem_p = MemoryParams(influence_strength=float(mem_influence))

    warns = validate_parameters(sim_p, gfp_p, mem_p)
    if warns:
        st.info(" • " + "\n • ".join(warns))

    res = run_combined_experiment(sim_p, gfp_p, mem_p, mode=mode)
    drv = res["driver"]; c30 = res["control_30"]; c39 = res["control_39"]; passives = res["passives"]

    # ----------------------------- Metrics -----------------------------------
    m = calculate_learning_metrics(res)
    k = st.columns(6)
    k[0].metric("Learning score", f"{m.get('learning_score', 0):.2f}")
    k[1].metric("Final temp", f"{m.get('final_temperature', np.nan):.2f} °C")
    k[2].metric("Final driver GFP", f"{m.get('final_gfp', np.nan):.1f}")
    at = m.get("adaptation_time"); et = m.get("establishment_time")
    k[3].metric("Adaptation time", "—" if at is None else f"{at:.0f} min")
    k[4].metric("Establishment", "—" if et is None else f"{et:.0f} min")
    k[5].metric("GFP memory ON (final)", f"{m.get('final_memory_on_fraction', 0):.2f}")

    # ----------------------------- Plots -------------------------------------
    # 1) Temperature (fixed 29–40 °C)
    st.markdown("#### 🌡️ Temperature (Driver, Passives, Controls)")
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=drv["time"], y=drv["temperature"], name="Driver", mode="lines"))
    for i, p in enumerate(passives, start=1):
        fig_t.add_trace(go.Scatter(x=p["time"], y=p["temperature"], name=f"Passive {i}", mode="lines"))
    fig_t.add_trace(go.Scatter(x=c30["time"], y=c30["temperature"], name="Control 30°C", mode="lines"))
    fig_t.add_trace(go.Scatter(x=c39["time"], y=c39["temperature"], name="Control 39°C", mode="lines"))
    fig_t.update_layout(template="plotly_white", height=330, xaxis_title="Time (min)",
                        yaxis_title="°C", yaxis=dict(range=[29, 40]))
    st.plotly_chart(fig_t, use_container_width=True)

    # 2) Mean GFP (0–100)
    st.markdown("#### 🧬 Mean GFP (Driver, Passives, Controls)")
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=drv["time"], y=drv["mean_gfp"], name="Driver", mode="lines"))
    for i, p in enumerate(passives, start=1):
        fig_g.add_trace(go.Scatter(x=p["time"], y=p["mean_gfp"], name=f"Passive {i}", mode="lines"))
    fig_g.add_trace(go.Scatter(x=c30["time"], y=c30["mean_gfp"], name="Control 30°C", mode="lines"))
    fig_g.add_trace(go.Scatter(x=c39["time"], y=c39["mean_gfp"], name="Control 39°C", mode="lines"))
    fig_g.update_layout(template="plotly_white", height=320, xaxis_title="Time (min)",
                        yaxis_title="Mean GFP", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_g, use_container_width=True)

    # 3) High-GFP fraction (0–1) + dashed 0.5
    st.markdown("#### 📈 High-GFP fraction (Driver, Passives, Controls)")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=drv["time"], y=drv["high_gfp_fraction"], name="Driver", mode="lines"))
    for i, p in enumerate(passives, start=1):
        fig_f.add_trace(go.Scatter(x=p["time"], y=p["high_gfp_fraction"], name=f"Passive {i}", mode="lines"))
    fig_f.add_trace(go.Scatter(x=c30["time"], y=c30["high_gfp_fraction"], name="Control 30°C", mode="lines"))
    fig_f.add_trace(go.Scatter(x=c39["time"], y=c39["high_gfp_fraction"], name="Control 39°C", mode="lines"))
    if drv["time"]:
        fig_f.add_shape(type="line", xref="x", yref="y",
                        x0=drv["time"][0], x1=drv["time"][-1], y0=0.5, y1=0.5, line=dict(dash="dash"))
    fig_f.update_layout(template="plotly_white", height=320, xaxis_title="Time (min)",
                        yaxis_title="Fraction GFP+", yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_f, use_container_width=True)

    # 4) Epigenetic memory: ON fraction (0–1) for all wells
    st.markdown("#### 🧠 GFP memory ON fraction (Driver, Passives, Controls)")
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=drv["time"], y=drv["gfp_on_fraction"], name="Driver", mode="lines"))
    for i, p in enumerate(passives, start=1):
        fig_m.add_trace(go.Scatter(x=p["time"], y=p["gfp_on_fraction"], name=f"Passive {i}", mode="lines"))
    fig_m.add_trace(go.Scatter(x=c30["time"], y=c30["gfp_on_fraction"], name="Control 30°C", mode="lines"))
    fig_m.add_trace(go.Scatter(x=c39["time"], y=c39["gfp_on_fraction"], name="Control 39°C", mode="lines"))
    fig_m.update_layout(template="plotly_white", height=320, xaxis_title="Time (min)",
                        yaxis_title="Memory ON fraction", yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_m, use_container_width=True)

    st.caption(
        "Learning: driver cools as mean GFP increases (vs. fixed 39 °C control). "
        "Epigenetic memory listens to the temperature derivative (dT/dt): stronger cooling evidence "
        "→ memory builds → switching toward HIGH GFP is amplified → faster adaptation. "
        "Use the single slider to fade the epigenetic effect in/out."
    )
else:
    st.info("Pick an **Epigenetic influence** and click **Run simulation**. The rest is optional.")
