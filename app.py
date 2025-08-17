import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import json

from main import (
    run_simulation,
    run_simulation_with_external_temperature,
    feedback_temperature,
)

st.set_page_config(page_title="Evolutionary Learning (Moran-style)", layout="wide")

# ===============================
# Sidebar controls
# ===============================
st.sidebar.title("Simulation Parameters (minimal)")

with st.sidebar.expander("Core horizon & population", expanded=True):
    total_time = st.number_input("Total time (min)", 100, 20000, 1000, step=100)
    max_population = st.number_input("Population cap (N)", 10, 20000, 1000, step=10)
    moran_after_cap = st.checkbox("Use Moran birth–death after reaching N", value=True)
    num_passive_wells = st.slider("Passive wells (share driver temp)", 1, 30, 5, 1)

with st.sidebar.expander("Feedback (driver)", expanded=True):
    feedback_mode = st.selectbox("Feedback mode", ["linear", "exp", "sigmoid", "step", "inverse"], index=0)
    feedback_sensitivity = st.slider("Feedback sensitivity  (lower values ⇒ more sensitive)" , 0.1, 5.0, 1.0, 0.1)

with st.sidebar.expander("Trait dynamics", expanded=True):
    inheritance_noise = st.slider("Inheritance noise (GFP SD)", 0.5, 10.0, 2.0, 0.5)
    base_switch_prob = st.slider("Phenotype switch prob @39°C (per min)", 0.0, 0.02, 0.002, 0.001)

with st.sidebar.expander("Fitness & mortality — HIGH IMPACT", expanded=True):
    gfp_cost_strength = st.slider("GFP cost strength (fitness burden) — HIGH impact", 0.0, 2.0, 0.5, 0.1)
    death_mult = st.slider("Mortality multiplier (× base hazard) — HIGH impact", 0.1, 3.0, 1.0, 0.1)

with st.sidebar.expander("Logging", expanded=True):
    enable_stdout = st.checkbox("Log per-minute stats to server terminal (JSON lines)", value=True)
    capture_logs = st.checkbox("Capture logs in app (search & download)", value=True)
    verbose_events = st.checkbox("Verbose event logs (death/division/switch)", value=False)
    show_logs_after_run = st.checkbox("Show logs panel after run", value=True)

with st.sidebar.expander("Reproducibility", expanded=False):
    seed_on = st.checkbox("Set random seed", value=True)
    seed_val = st.number_input("Seed value", 0, 10_000_000, 42, step=1)

run_button = st.sidebar.button("Run simulation")

# ===============================
# Title
# ===============================
st.title("🧬 Evolutionary Learning via GFP Feedback (Continuous trait, Moran-style)")

st.caption(
    "Driver well: temperature updates every minute from *current* mean GFP (full feedback). "
    "Passives follow the driver’s temperature but do not influence it. "
    "Controls are fixed at 39°C and 30°C. After reaching N, births replace random individuals (Moran). "
    "Division fitness integrates temperature *and* GFP metabolic burden; cells also die stochastically. "
    "Sliders for Fitness & Mortality are **high-impact** in this build."
)

# ===============================
# Run simulations
# ===============================
if run_button or ("driver" not in st.session_state):
    if seed_on:
        np.random.seed(int(seed_val))

    with st.spinner("Running driver, passives, and controls..."):
        # Map sliders to stronger internal effects:
        # - cost strength amplified in main.py by 2× internally
        # - death: use squared scaling to increase sensitivity
        death_base_prob = 0.0005 * (death_mult ** 2)

        # DRIVER (feedback)
        driver = run_simulation(
            total_time=total_time,
            max_population=max_population,
            moran_after_cap=moran_after_cap,
            inheritance_noise=inheritance_noise,
            base_switch_prob=base_switch_prob,
            feedback_mode=feedback_mode,
            feedback_sensitivity=feedback_sensitivity,
            control_mode=None,
            start_temp=39,
            # reporting defaults
            high_gfp_threshold=20.0,
            stationarity_window=60,
            stationarity_tol=0.1,
            # fitness + mortality
            gfp_cost_strength=gfp_cost_strength,  # amplified inside main by 2×
            gfp_cost_gamma=1.0,
            death_base_prob=death_base_prob,
            death_temp_weight=1.0 * death_mult,   # extra heat death grows with death_mult
            death_gfp_weight=1.0 * death_mult,    # extra GFP death grows with death_mult
            death_max_prob=0.05,
            # logging
            well_label="driver",
            log_to_stdout=enable_stdout,
            capture_log=capture_logs,
            verbose_events=verbose_events
        )

        # PASSIVES (follow driver temperature)
        passives = []
        _, temp_driver, temp_t_driver, _, _, _, _, _, _, _, _ = driver
        for i in range(num_passive_wells):
            res = run_simulation_with_external_temperature(
                external_temp_times=temp_t_driver,
                external_temp_values=temp_driver,
                total_time=total_time,
                max_population=max_population,
                moran_after_cap=moran_after_cap,
                inheritance_noise=inheritance_noise,
                base_switch_prob=base_switch_prob,
                # reporting defaults
                high_gfp_threshold=20.0,
                stationarity_window=60,
                stationarity_tol=0.1,
                # fitness + mortality
                gfp_cost_strength=gfp_cost_strength,  # amplified inside main by 2×
                gfp_cost_gamma=1.0,
                death_base_prob=death_base_prob,
                death_temp_weight=1.0 * death_mult,
                death_gfp_weight=1.0 * death_mult,
                death_max_prob=0.05,
                # logging
                well_label=f"passive_{i+1}",
                log_to_stdout=enable_stdout,
                capture_log=capture_logs,
                verbose_events=verbose_events
            )
            passives.append(res)

        # CONTROLS
        ctrl39 = run_simulation(
            total_time=total_time,
            max_population=max_population,
            moran_after_cap=moran_after_cap,
            inheritance_noise=inheritance_noise,
            base_switch_prob=base_switch_prob,
            feedback_mode=feedback_mode,
            feedback_sensitivity=feedback_sensitivity,
            control_mode="fixed39",
            start_temp=39,
            gfp_cost_strength=gfp_cost_strength,
            gfp_cost_gamma=1.0,
            death_base_prob=death_base_prob,
            death_temp_weight=1.0 * death_mult,
            death_gfp_weight=1.0 * death_mult,
            well_label="control_39C",
            log_to_stdout=enable_stdout,
            capture_log=capture_logs,
            verbose_events=verbose_events
        )

        ctrl30 = run_simulation(
            total_time=total_time,
            max_population=max_population,
            moran_after_cap=moran_after_cap,
            inheritance_noise=inheritance_noise,
            base_switch_prob=base_switch_prob,
            feedback_mode=feedback_mode,
            feedback_sensitivity=feedback_sensitivity,
            control_mode="fixed30",
            start_temp=30,
            gfp_cost_strength=gfp_cost_strength,
            gfp_cost_gamma=1.0,
            death_base_prob=death_base_prob,
            death_temp_weight=1.0 * death_mult,
            death_gfp_weight=1.0 * death_mult,
            well_label="control_30C",
            log_to_stdout=enable_stdout,
            capture_log=capture_logs,
            verbose_events=verbose_events
        )

        st.session_state.driver = driver
        st.session_state.passives = passives
        st.session_state.ctrl39 = ctrl39
        st.session_state.ctrl30 = ctrl30
        st.session_state.logging_opts = dict(
            enable_stdout=enable_stdout,
            capture_logs=capture_logs,
            verbose_events=verbose_events
        )

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
        log_driver,
    ) = st.session_state.driver

    passives = st.session_state.passives
    (
        df_c39, temp_c39, t_c39, gfp_c39, gen_c39, time_c39, pop_c39,
        t_first_high_c39, t_stat_c39, final_gfps_c39, log_c39,
    ) = st.session_state.ctrl39

    (
        df_c30, temp_c30, t_c30, gfp_c30, gen_c30, time_c30, pop_c30,
        t_first_high_c30, t_stat_c30, final_gfps_c30, log_c30,
    ) = st.session_state.ctrl30

    # ===============================
    # Derived metrics
    # ===============================
    lr = float(np.mean((39.0 - np.array(temp_driver)) / 9.0))

    d_gfp_driver = np.diff(gfp_driver, prepend=gfp_driver[0])
    if passives:
        gfp_passive_avg = np.mean([res[3] for res in passives], axis=0)
    else:
        gfp_passive_avg = np.array(gfp_driver) * np.nan
    d_gfp_passive = np.diff(gfp_passive_avg, prepend=gfp_passive_avg[0])
    d_temp = np.diff(temp_driver, prepend=temp_driver[0])

    counter_idxs = np.where((d_gfp_driver < 0) & (d_gfp_passive > 0) & (d_temp > 0))[0]
    counter_times = [time_driver[i] for i in counter_idxs]

    # ===============================
    # TOP METRICS
    # ===============================
    st.subheader("🔎 Key Events / Endpoints")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Driver: 1st high-GFP (min)", t_first_high_driver if t_first_high_driver is not None else "–")
    with c2:
        lr = float(np.mean((39.0 - np.array(temp_driver)) / 9.0))
        st.metric("Learning ratio (0–1)", f"{lr:.2f}")
    with c3:
        # Keep counting counter-learning for analysis, but don't draw red lines later
        d_gfp_driver = np.diff(gfp_driver, prepend=gfp_driver[0])
        if passives:
            gfp_passive_avg = np.mean([res[3] for res in passives], axis=0)
        else:
            gfp_passive_avg = np.array(gfp_driver) * np.nan
        d_gfp_passive = np.diff(gfp_passive_avg, prepend=gfp_passive_avg[0])
        d_temp = np.diff(temp_driver, prepend=temp_driver[0])
        counter_idxs = np.where((d_gfp_driver < 0) & (d_gfp_passive > 0) & (d_temp > 0))[0]
        st.metric("# Counter-learning episodes", int(len(counter_idxs)))

    # Optional: shorter caption without “stationarity”
    st.caption("Pattern: high-GFP breakthrough → cooling → shorter generation times → faster reproduction → maintenance.")

    # ===============================
    # TEMPERATURE (Driver) — single render, no red lines
    # ===============================
    st.markdown("### 🌡 Driver Temperature Over Time")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=temp_t_driver, y=temp_driver, mode="lines", name="Driver temp"))

    # Optional: first high-GFP marker only (no stationarity, no counter-learning lines)
    if t_first_high_driver is not None:
        fig_temp.add_vline(x=t_first_high_driver, line=dict(color="green", dash="dot"))
        fig_temp.add_annotation(
            x=t_first_high_driver,
            y=temp_driver[min(len(temp_driver) - 1, t_first_high_driver)],
            text="1st high-GFP",
            showarrow=True,
            yshift=20
        )

    fig_temp.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="Temperature (°C)",
        template="plotly_white"
    )

    # IMPORTANT: render ONLY ONCE with this key
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
        fig_gfp.add_trace(go.Scatter(x=time_driver, y=gfp_passive_avg, name=f"Passive (avg of {len(passives)})"))
    fig_gfp.add_trace(go.Scatter(
        x=[time_driver[i] for i in counter_idxs],
        y=[gfp_driver[i] for i in counter_idxs],
        mode="markers",
        name="Counter-learning",
        marker=dict(size=6, symbol="x")
    ))
    fig_gfp.update_layout(xaxis_title="Time (min)", yaxis_title="Mean GFP (0–100)", template="plotly_white")
    st.plotly_chart(fig_gfp, key="mean_gfp_chart", use_container_width=True)

    # ===============================
    # ALL WELLS (UNAGGREGATED GFP)
    # ===============================
    st.markdown("### 🌊 GFP — All Wells (Unaggregated)")
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(x=time_driver, y=gfp_driver, name="Driver", line=dict(width=3)))
    for i, res in enumerate(passives):
        _, _, _, gfp_p, _, time_p, _, _, _, _, _ = res
        fig_all.add_trace(go.Scatter(x=time_p, y=gfp_p, name=f"Passive {i+1}", opacity=0.4, line=dict(width=1)))
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
    st.caption("Interpretation: as mean GFP rises, temperature drops immediately (tight feedback).")

    # ===============================
    # FINAL GFP DENSITY (KDE)
    # ===============================
    st.markdown("### 🧪 Final GFP Density (KDE)")

    # Collect all-passive cells (like before)
    all_passive_final = np.concatenate([np.array(res[-2]) for res in passives]) if passives else np.array([])

    # Simple Gaussian KDE (Silverman's rule), no SciPy required
    def kde_1d(values, grid, bw=None):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return np.zeros_like(grid)
        n = vals.size
        std = np.std(vals, ddof=1) if n > 1 else 1.0
        if bw is None:
            bw = 1.06 * std * (n ** (-1/5))  # Silverman's rule of thumb
            bw = max(bw, 1e-3)
        z = (grid[:, None] - vals[None, :]) / bw
        dens = np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
        return dens

    grid = np.linspace(0, 100, 401)  # smooth 1-D grid over GFP range

    dens_driver = kde_1d(final_gfps_driver, grid)
    dens_c39    = kde_1d(final_gfps_c39, grid)
    dens_c30    = kde_1d(final_gfps_c30, grid)
    dens_pass   = kde_1d(all_passive_final, grid) if all_passive_final.size > 0 else np.zeros_like(grid)

    fig_kde = go.Figure()
    fig_kde.add_trace(go.Scatter(x=grid, y=dens_driver, mode="lines", name="Driver"))
    fig_kde.add_trace(go.Scatter(x=grid, y=dens_c39,    mode="lines", name="Control 39°C", line=dict(dash="dash")))
    fig_kde.add_trace(go.Scatter(x=grid, y=dens_c30,    mode="lines", name="Control 30°C", line=dict(dash="dot")))
    if all_passive_final.size > 0:
        fig_kde.add_trace(go.Scatter(x=grid, y=dens_pass, mode="lines", name="Passives (all cells)", opacity=0.7))

    fig_kde.update_layout(
        xaxis_title="GFP (0–100)",
        yaxis_title="Density",
        template="plotly_white"
    )

    # New key to avoid any collisions with previous histogram
    st.plotly_chart(fig_kde, key="kde_chart", use_container_width=True)


    # ===============================
    # LOGS panel
    # ===============================
    if st.session_state.logging_opts["capture_logs"] and show_logs_after_run:
        st.markdown("## 📜 Logs (search & download)")

        # Combine logs from all wells
        logs_all = []
        def norm(item):
            # make JSON-serializable plain dicts
            if isinstance(item, dict):
                return item
            try:
                return json.loads(item)
            except Exception:
                return {"raw": str(item)}

        logs_all += list(map(norm, log_driver))
        for i, res in enumerate(passives):
            logs_all += list(map(norm, res[-1]))
        logs_all += list(map(norm, log_c39))
        logs_all += list(map(norm, log_c30))

        # As DataFrame
        if logs_all:
            df_logs = pd.DataFrame(logs_all)
        else:
            df_logs = pd.DataFrame(columns=["type","well","t","temp","mean_gfp","pop","divisions","deaths","switches","moran_active"])

        with st.expander("Search / Filter", expanded=True):
            q = st.text_input("Search (substring match across JSON)", "")
            if q:
                mask = df_logs.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False))
                df_logs_view = df_logs[mask.any(axis=1)]
            else:
                df_logs_view = df_logs

            st.dataframe(df_logs_view, use_container_width=True, height=300)

            # Downloads
            json_lines = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in df_logs_view.to_dict(orient="records"))
            st.download_button("⬇️ Download logs (JSON Lines)", data=json_lines, file_name="simulation_logs.jsonl", mime="application/json")

            csv_data = df_logs_view.to_csv(index=False)
            st.download_button("⬇️ Download logs (CSV)", data=csv_data, file_name="simulation_logs.csv", mime="text/csv")

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
