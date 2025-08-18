"""
Evolutionary Learning Simulation - Streamlit Interface
=====================================================

Interactive web application for exploring temperature-feedback driven adaptation
in synthetic yeast populations via a strict Moran (birth–death) process.

Controls are FIXED at 30°C and 39°C for the entire simulation.
Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import json
from typing import Dict, Any, List

from src.main import (
    SimulationParams, GFPParams, run_evolution_experiment,
    calculate_learning_metrics, export_results_to_dataframe,
    create_feedback_function_data, validate_parameters,
    feedback_temperature
)

# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="🧬 Evolutionary Learning - Moran Process",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-repo/evolutionary-learning',
        'Report a bug': "https://github.com/your-repo/evolutionary-learning/issues",
        'About': "# Evolutionary Learning Lab\nStudying adaptation in synthetic biology!"
    }
)

# ===============================
# Custom Styling
# ===============================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 24px rgba(0,0,0,0.08);
        font-size: 1.4rem;
        font-weight: 700;
    }
    .param-display {
        background: linear-gradient(135deg, #667eea 0%, #c3cfe2 50%);
        padding: 0.9rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1.2rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        font-size: 1.05rem;
        color: #2c3e50;
    }
    .metric-card {
        background: white;
        padding: 0.9rem;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e1e5e9;
    }
    .success-box, .warning-box {
        padding: 0.8rem;
        margin: 0.8rem 0;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 1.2rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Helper Functions
# ===============================

def create_parameter_display(sim_params: SimulationParams, gfp_params: GFPParams, mode: str) -> str:
    return f"""
    <div class="param-display">
        <h4 style="margin:0 0 .5rem 0;">🎯 Current Configuration</h4>
        <strong>Mode:</strong> {mode.title()} |
        <strong>Feedback:</strong> {sim_params.feedback_mode.title()} (sens: {sim_params.feedback_sensitivity}) |
        <strong>Pop:</strong> {sim_params.population_size} |
        <strong>Time:</strong> {sim_params.total_time} min (Δt={sim_params.time_step}) |
        <strong>Passives:</strong> {sim_params.num_passive_wells} |
        <strong>GFP Cost:</strong> {gfp_params.cost_strength:.2f} |
        <strong>Moran:</strong> 1 birth + 1 death / step |
        <strong>Controls:</strong> 30 °C & 39 °C (fixed)
    </div>
    """

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal") -> str:
    delta_html = ""
    if delta:
        color = "green" if delta_color == "normal" else "red"
        delta_html = f'<div style="color: {color}; font-size: 0.75rem;">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.4rem;">{title}</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: #2c3e50;">{value}</div>
        {delta_html}
    </div>
    """

def display_warnings(warnings: List[str]):
    if warnings:
        warning_text = "<br>".join(warnings)
        st.markdown(f"""
        <div class="warning-box" style="background:#fffbe6;border:1px solid #ffeaa7;color:#6b5d00;">
            <strong>⚠️ Parameter Warnings:</strong><br>
            {warning_text}
        </div>
        """, unsafe_allow_html=True)


# ===============================
# Generation Time Insights (Heatmap + Slices)
# ===============================

def _gen_time_base_from_temp(temperature: np.ndarray) -> np.ndarray:
    """
    Same function as Cell._calculate_base_generation_time:
    base_time = 60.0 + 120.0 * ((temp - 30)/9)^2, clipped to [30,39] mapping -> [0,1]
    """
    temp_norm = np.clip((temperature - 30.0) / 9.0, 0.0, 1.0)
    return 60.0 + 120.0 * (temp_norm ** 2)

def _gen_time_cost_multiplier(gfp: np.ndarray, mode: str, gfp_params: GFPParams) -> np.ndarray:
    """
    Mirrors Cell._calculate_gfp_cost_multiplier:
      - binary: 1.3 if GFP > 50 else 1.0
      - continuous: 1 + cost_strength * (gfp_norm ** cost_exponent)
    """
    gfp = np.asarray(gfp, dtype=float)
    if mode == "binary":
        return np.where(gfp > 50.0, 1.3, 1.0)
    # continuous
    gfp_norm = np.clip(gfp / 100.0, 0.0, 1.0)
    cost_factor = gfp_params.cost_strength * (gfp_norm ** gfp_params.cost_exponent)
    return 1.0 + cost_factor

def compute_generation_time_grid(base_temp: float, max_temp: float,
                                 mode: str, gfp_params: GFPParams,
                                 n_temp: int = 200, n_gfp: int = 200):
    """
    Returns (gfps, temps, gen_time_matrix) where matrix is shape (n_temp, n_gfp).
    """
    temps = np.linspace(base_temp, max_temp, n_temp)          # shape (n_temp,)
    gfps  = np.linspace(0.0, 100.0, n_gfp)                    # shape (n_gfp,)

    # reshape for broadcasting
    T = temps.reshape(-1, 1)                                  # (n_temp, 1)
    G = gfps.reshape(1, -1)                                   # (1, n_gfp)

    base = _gen_time_base_from_temp(T)                        # (n_temp, 1)
    cost = _gen_time_cost_multiplier(G, mode, gfp_params)     # (1, n_gfp)

    gen_time = base * cost                                    # (n_temp, n_gfp)
    return gfps, temps, gen_time

def render_generation_time_insight(sim_params: SimulationParams,
                                   gfp_params: GFPParams,
                                   mode: str,
                                   results: Dict[str, Any] = None):
    st.markdown("### ⏱️ Generation Time vs Temperature & GFP")

    # grid + figure
    gfps, temps, Z = compute_generation_time_grid(
        base_temp=sim_params.base_temp,
        max_temp=sim_params.max_temp,
        mode=mode,
        gfp_params=gfp_params,
        n_temp=220, n_gfp=220
    )

    heat = go.Heatmap(
        x=gfps, y=temps, z=Z,
        colorbar=dict(title="Generation Time (min)"),
        hovertemplate="GFP: %{x:.1f}<br>T: %{y:.2f}°C<br>Gen time: %{z:.1f} min<extra></extra>"
    )
    contour = go.Contour(
        x=gfps, y=temps, z=Z,
        contours=dict(coloring="lines", showlabels=False),
        showscale=False, line=dict(width=1)
    )

    fig = go.Figure(data=[heat, contour])
    fig.update_layout(
        title="Generation Time Landscape (uses exact model equations)",
        xaxis_title="GFP",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=480,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    # optional: overlay actual driver trajectory from last run
    if results is not None and 'driver' in results and results['driver']['time']:
        traj_x = results['driver']['mean_gfp']
        traj_y = results['driver']['temperature']
        fig.add_trace(go.Scatter(
            x=traj_x, y=traj_y, mode='lines+markers',
            name='Driver trajectory', line=dict(width=2), marker=dict(size=4),
            hovertemplate="Time: %{text} min<br>GFP: %{x:.1f}<br>T: %{y:.2f}°C<extra></extra>",
            text=results['driver']['time']
        ))

    st.plotly_chart(fig, use_container_width=True)


def feedback_preview(gfp_min: float, gfp_max: float, n_points: int,
                     feedback_mode: str, sensitivity: float,
                     base_temp: float, max_temp: float, height: int = 320):
    """Preview feedback using the SAME function as the simulation (driver)."""
    gfp_range = np.linspace(gfp_min, gfp_max, n_points)
    temps = [feedback_temperature(g, feedback_mode, sensitivity, base_temp, max_temp) for g in gfp_range]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gfp_range, y=temps, mode='lines',
        name=f'{feedback_mode.title()} Feedback', line=dict(width=3)
    ))
    fig.update_layout(
        title=f"Temperature Feedback: {feedback_mode.title()} (Sensitivity {sensitivity})",
        xaxis_title="Mean Population GFP",
        yaxis_title=f"Environmental Temperature (°C)",
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Main Application
# ===============================

def main():
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;">🧬 Evolutionary Learning through Moran Processes</h1>
        <p style="font-size: 0.95rem; margin-top: 0.5rem;">
            Temperature-feedback driven adaptation in synthetic yeast populations (strict birth–death updates)
            — controls locked at 30 °C and 39 °C.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar Controls ---
    st.sidebar.markdown("## 🎛️ Simulation Parameters")
    st.sidebar.markdown("Configure your experiment below:")

    with st.sidebar.expander("🔬 **Core Experimental Setup**", expanded=True):
        mode = st.selectbox("GFP Expression Mode", ["continuous", "binary"], index=0,
                            help="Continuous: 0–100 with noise • Binary: High/Low with switching")
        total_time = st.slider("Simulation Time (minutes)", 200, 3000, 1000, step=100)
        time_step = st.slider("Time Step (minutes per Moran event)", 1, 10, 1, step=1,
                              help="One Moran event (1 birth + 1 death) occurs each step.")
        population_size = st.slider("Population Size (cells)", 50, 800, 200, step=25)
        num_passives = st.slider("Number of Passive Wells", 1, 8, 3, step=1)

    with st.sidebar.expander("🌡️ **Temperature Feedback System (Driver)**", expanded=False):
        feedback_mode = st.selectbox("Feedback Function Type", ["linear", "sigmoid", "step", "exponential"], index=0)
        feedback_sensitivity = st.slider("Feedback Sensitivity", 0.2, 4.0, 1.0, step=0.1)
        st.caption("Driver temperature range: 30 °C ↔ 39 °C")

    with st.sidebar.expander("🧬 **Evolution & Fitness Parameters**", expanded=False):
        inherit_noise = st.slider("Inheritance Noise (continuous)", 1.0, 20.0, 5.0, step=1.0)
        switch_rate = st.slider("Stress-Induced Switching Rate", 0.001, 0.08, 0.01, step=0.001)
        fitness_cost = st.slider("GFP Metabolic Cost", 0.0, 1.2, 0.3, step=0.05)
        cost_exponent = st.slider("Cost Function Curvature", 0.5, 3.0, 1.5, step=0.1)

    with st.sidebar.expander("🧊 **Smoothing & Metrics (Driver)**", expanded=False):
        temp_inertia = st.slider("Temperature Inertia (smoothing)", 0.05, 1.0, 0.25, step=0.05,
                                 help="Smaller = smoother temperature changes (driver only).")
        start_at_max_temp = st.checkbox("Start at Max Temp (no-cliff hold @ t=0)", value=True)
        metric_burn_in = st.slider("Metric Burn-in (minutes)", 0, max(100, total_time // 3), 10, step=5,
                                   help="Ignore early window when computing adaptation metrics.")

    with st.sidebar.expander("⚙️ **Advanced Settings**", expanded=False):
        random_seed = st.number_input("Random Seed (reproducibility)", min_value=1, max_value=999999, value=42, step=1)
        enable_validation = st.checkbox("Show Parameter Warnings", value=True)

    # --- Parameter objects ---
    sim_params = SimulationParams(
        total_time=total_time,
        time_step=time_step,
        population_size=population_size,
        num_passive_wells=num_passives,
        feedback_mode=feedback_mode,
        feedback_sensitivity=feedback_sensitivity,
        temp_inertia=temp_inertia,
        start_at_max_temp=start_at_max_temp,
        metric_burn_in=metric_burn_in,
        random_seed=random_seed
        # base_temp/max_temp left as defaults (30/39) to match driver model
    )
    gfp_params = GFPParams(
        inherit_sd=inherit_noise,
        switch_prob_base=switch_rate,
        cost_strength=fitness_cost,
        cost_exponent=cost_exponent
    )

    with st.expander("📈 Quick Preview: GFP → Driver Temperature", expanded=False):
        feedback_preview(0, 100, 300, feedback_mode, feedback_sensitivity,
                         sim_params.base_temp, sim_params.max_temp, height=320)

    # --- Display current parameters ---
    st.markdown('<div class="section-header">📊 Current Parameters</div>', unsafe_allow_html=True)
    st.markdown(create_parameter_display(sim_params, gfp_params, mode), unsafe_allow_html=True)
    if enable_validation:
        warnings = validate_parameters(sim_params, gfp_params)
        if warnings:
            display_warnings(warnings)

    # --- Run button ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_simulation = st.button("🚀 Run Evolution Experiment", type="primary", use_container_width=True)

    if run_simulation:
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            status_text.text("🔬 Initializing populations...")
            progress_bar.progress(10)
            status_text.text("🧬 Running Moran BD simulation...")
            progress_bar.progress(30)
            results = run_evolution_experiment(sim_params, gfp_params, mode)
            progress_bar.progress(80)
            status_text.text("📊 Calculating metrics...")
            metrics = calculate_learning_metrics(results)
            progress_bar.progress(100)
            status_text.text("✅ Simulation completed!")

            st.session_state['results'] = results
            st.session_state['metrics'] = metrics
            st.session_state['params'] = {'sim_params': sim_params, 'gfp_params': gfp_params, 'mode': mode}

            progress_bar.empty(); status_text.empty()
            st.markdown("""
            <div class="success-box">
                <strong>🎉 Simulation completed successfully!</strong><br>
                Scroll down to explore your results.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            progress_bar.empty(); status_text.empty()
            st.error(f"❌ Simulation failed: {str(e)}")
            st.exception(e)

    # --- Results or minimal onboarding ---
    if 'results' in st.session_state and 'metrics' in st.session_state:
        display_results(
            st.session_state['results'],
            st.session_state['metrics'],
            st.session_state['params']
        )
    else:
        with st.expander("🎯 What to Expect (quick primer)", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Learning Dynamics**
                - Driver learns to increase GFP → cooler temps
                - Passives follow temperature only
                - Controls: fixed at 30 °C and 39 °C
                """)
            with col2:
                st.markdown("""
                **Biology**
                - Strict Moran: 1 birth + 1 death per step
                - GFP cost slows division (fitness-proportional parent choice)
                - Heat increases switching
                """)

def display_results(results: Dict[str, Any], metrics: Dict[str, float], params: Dict[str, Any]):
    st.markdown('<div class="section-header">📊 Simulation Results</div>', unsafe_allow_html=True)

    sim_params: SimulationParams = params['sim_params']
    base_T = sim_params.base_temp
    max_T = sim_params.max_temp
    T_span = max_T - base_T
    gfp_params: GFPParams = params['gfp_params']

    # --- Key Metrics Dashboard ---
    st.markdown("### 🎯 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        learning_score = metrics.get('learning_score', 0)
        score_text = "Excellent" if learning_score > 0.7 else "Moderate" if learning_score > 0.3 else "Poor"
        st.markdown(create_metric_card("Learning Score", f"{learning_score:.2f}", score_text), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Final Driver GFP", f"{metrics.get('final_gfp', 0):.1f}", "Target > 60"),
                    unsafe_allow_html=True)
    with col3:
        final_temp = metrics.get('final_temperature', max_T)
        st.markdown(create_metric_card("Final Temp", f"{final_temp:.1f}°C",
                                       f"Cooling {max_T-final_temp:.1f}°C"),
                    unsafe_allow_html=True)
    with col4:
        adaptation_time = metrics.get('adaptation_time', None)
        time_text = f"{adaptation_time:.0f} min" if adaptation_time is not None else "No adaptation"
        st.markdown(create_metric_card("Adaptation Time", time_text, "50% of final"), unsafe_allow_html=True)
    with col5:
        high_gfp_frac = metrics.get('final_high_gfp_fraction', 0)
        st.markdown(create_metric_card("High GFP Fraction", f"{high_gfp_frac:.2f}", "Population success"),
                    unsafe_allow_html=True)

    # --- Temperature Evolution ---
    st.markdown("### 🌡️ Temperature Evolution")
    driver_data = results['driver']
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=driver_data['time'], y=driver_data['temperature'],
        mode='lines', name='Driver Temperature',
        line=dict(width=3),
        hovertemplate='Time: %{x} min<br>Temp: %{y:.1f}°C<extra></extra>'
    ))
    # Reference lines for constant controls
    fig_temp.add_hline(y=30.0, line=dict(dash="dot"), annotation_text="Control 30°C", annotation_position="bottom left")
    fig_temp.add_hline(y=39.0, line=dict(dash="dot"), annotation_text="Control 39°C", annotation_position="top left")
    if metrics.get('adaptation_time') is not None:
        fig_temp.add_vline(x=metrics['adaptation_time'], line=dict(dash="dash"),
                           annotation_text="50% Adaptation")
    fig_temp.update_layout(
        title="Environmental Temperature (Driver Feedback; Controls Fixed)",
        xaxis_title="Time (min)", yaxis_title="Temp (°C)",
        template="plotly_white", hovermode='x unified', height=380
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # --- GFP Evolution ---
    st.markdown("### 🧬 GFP Evolution: Driver vs. Controls")
    fig_gfp = go.Figure()
    fig_gfp.add_trace(go.Scatter(x=driver_data['time'], y=driver_data['mean_gfp'],
                                 mode='lines', name='🎯 Driver (Feedback)', line=dict(width=4)))
    fig_gfp.add_trace(go.Scatter(x=results['control_30']['time'], y=results['control_30']['mean_gfp'],
                                 mode='lines', name='🔴 Control 30°C', line=dict(dash='dash', width=2)))
    fig_gfp.add_trace(go.Scatter(x=results['control_39']['time'], y=results['control_39']['mean_gfp'],
                                 mode='lines', name='🟢 Control 39°C', line=dict(dash='dash', width=2)))
    if results['passives']:
        passive_mean_gfp = np.mean([p['mean_gfp'] for p in results['passives']], axis=0)
        fig_gfp.add_trace(go.Scatter(x=driver_data['time'], y=passive_mean_gfp,
                                     mode='lines', name=f'🟠 Passives (n={len(results["passives"])})',
                                     line=dict(width=2, dash='dot')))
    threshold = 60 if params['mode'] == 'continuous' else 50
    fig_gfp.add_hline(y=threshold, line=dict(dash="dot"), annotation_text=f"High-GFP {threshold}")
    fig_gfp.update_layout(
        title="Mean GFP over Time",
        xaxis_title="Time (min)", yaxis_title="Mean GFP",
        template="plotly_white", hovermode='x unified', height=380
    )
    st.plotly_chart(fig_gfp, use_container_width=True)

    # --- Feedback Function & Evolution Path ---
    st.markdown("### 📈 Feedback Function & Evolution Path")
    gfp_range = np.linspace(0, 100, 300)
    temp_curve = [feedback_temperature(g, sim_params.feedback_mode,
                                       sim_params.feedback_sensitivity,
                                       base_T, max_T) for g in gfp_range]
    fig_feedback = go.Figure()
    fig_feedback.add_trace(go.Scatter(x=gfp_range, y=temp_curve, mode='lines',
                                      name=f'{sim_params.feedback_mode.title()} Feedback',
                                      line=dict(width=4)))
    fig_feedback.add_trace(go.Scatter(x=driver_data['mean_gfp'], y=driver_data['temperature'],
                                      mode='markers+lines', name='Actual Path',
                                      line=dict(width=2, dash='dot'), marker=dict(size=4)))
    fig_feedback.update_layout(
        title=f"Feedback & Trajectory ({sim_params.feedback_mode.title()})",
        xaxis_title="Mean Population GFP",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=420
    )
    st.plotly_chart(fig_feedback, use_container_width=True)

    # --- Detailed Analysis ---
    with st.expander("🔬 Detailed Analysis", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 Learning Progress (Driver)")
            temp_change = np.array(driver_data['temperature'])
            learning_curve = (max_T - temp_change) / max(T_span, 1e-9)
            fig_learning = go.Figure()
            fig_learning.add_trace(go.Scatter(x=driver_data['time'], y=learning_curve, mode='lines',
                                              name='Learning Progress', line=dict(width=2)))
            fig_learning.add_hline(y=0.5, line=dict(dash="dash"), annotation_text="50% Learning")
            fig_learning.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig_learning, use_container_width=True)
        with col2:
            st.markdown("#### 📊 Final State Stats")
            passive_final = (np.mean([p['mean_gfp'][-1] for p in results['passives']])
                             if results['passives'] else np.nan)
            advantage = (driver_data['mean_gfp'][-1] - passive_final) if results['passives'] else np.nan
            final_stats = {
                'Metric': ['Driver Final GFP', 'Control 30°C GFP', 'Control 39°C GFP',
                           'Passive Mean GFP', 'Learning Advantage', 'Temperature Reduction'],
                'Value': [
                    f"{driver_data['mean_gfp'][-1]:.1f}",
                    f"{results['control_30']['mean_gfp'][-1]:.1f}",
                    f"{results['control_39']['mean_gfp'][-1]:.1f}",
                    f"{passive_final:.1f}" if results['passives'] else "N/A",
                    f"{advantage:.1f}" if results['passives'] else "N/A",
                    f"{max_T - driver_data['temperature'][-1]:.1f}°C"
                ]
            }
            stats_df = pd.DataFrame(final_stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            learning_score = metrics.get('learning_score', 0)
            if learning_score > 0.7:
                st.success("🎉 Excellent learning.")
            elif learning_score > 0.3:
                st.warning("⚠️ Moderate learning.")
            else:
                st.error("❌ Poor learning. Try tuning parameters.")

    render_generation_time_insight(sim_params, gfp_params, params['mode'], results)


    # --- Data Export ---
    st.markdown("### 💾 Download Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        df_export = export_results_to_dataframe(results)
        st.download_button("📊 Time Series (CSV)", df_export.to_csv(index=False),
                           file_name=f"evolution_results_{params['mode']}.csv", mime="text/csv",
                           use_container_width=True)
    with col2:
        export_summary = {'simulation_parameters': params, 'learning_metrics': metrics}
        st.download_button("⚙️ Parameters & Metrics (JSON)",
                           json.dumps(export_summary, indent=2, default=str),
                           file_name=f"simulation_config_{params['mode']}.json",
                           mime="application/json", use_container_width=True)
    with col3:
        st.download_button("🔬 Raw Data (JSON)",
                           json.dumps(results, indent=2, default=str),
                           file_name=f"raw_results_{params['mode']}.json",
                           mime="application/json", use_container_width=True)

# ===============================
# Entry Point
# ===============================

if __name__ == "__main__":
    main()
