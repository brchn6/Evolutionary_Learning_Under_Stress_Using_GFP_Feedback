"""
Evolutionary Learning Simulation - Streamlit Interface
=====================================================

Interactive web application for exploring temperature-feedback driven adaptation
in synthetic yeast populations. Provides intuitive controls and rich visualizations
for scientific exploration and education.

Usage:
    streamlit run app.py

Features:
- Binary and continuous GFP expression modes
- Multiple feedback function types
- Real-time visualization of evolutionary dynamics
- Downloadable results for further analysis
"""


import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    initial_sidebar_state="collapsed",  # CHANGED: start collapsed to reduce clutter
    menu_items={
        'Get Help': 'https://github.com/your-repo/evolutionary-learning',
        'Report a bug': "https://github.com/your-repo/evolutionary-learning/issues",
        'About': "# Evolutionary Learning Lab\nStudying adaptation in synthetic biology!"
    }
)

# ===============================
# Custom Styling (slightly smaller/cleaner)
# ===============================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;                 /* CHANGED: tighter */
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;           /* CHANGED */
        box-shadow: 0 6px 24px rgba(0,0,0,0.08);
        font-size: 1.4rem;               /* CHANGED: smaller */
        font-weight: 700;
    }

    .param-display {
        background: linear-gradient(135deg, #667eea 0%, #c3cfe2 50%);
        padding: 0.9rem;                  /* CHANGED: tighter */
        border-radius: 10px;
        border-left: 4px solid #667eea;   /* CHANGED */
        margin-bottom: 1.2rem;            /* CHANGED */
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        font-size: 1.05rem;               /* CHANGED: smaller */
        color: #2c3e50;
    }

    .metric-card {
        background: white;
        padding: 0.9rem;                  /* CHANGED: tighter */
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e1e5e9;
    }

    .success-box, .warning-box {
        padding: 0.8rem;                  /* CHANGED */
        margin: 0.8rem 0;                 /* CHANGED */
        border-radius: 8px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;           /* CHANGED: smaller */
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);      /* CHANGED: subtler */
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    .section-header {
        font-size: 1.2rem;                /* CHANGED: smaller */
        font-weight: 700;
        color: #2c3e50;
        margin: 1.2rem 0 0.8rem 0;        /* CHANGED */
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #667eea; /* CHANGED */
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
        <strong>Time:</strong> {sim_params.total_time} min | 
        <strong>Passives:</strong> {sim_params.num_passive_wells} | 
        <strong>GFP Cost:</strong> {gfp_params.cost_strength:.2f}
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
        warning_text = "\n".join(warnings)
        st.markdown(f"""
        <div class="warning-box" style="background:#fffbe6;border:1px solid #ffeaa7;color:#6b5d00;">
            <strong>⚠️ Parameter Warnings:</strong><br>
            {warning_text.replace('⚠️', '<br>⚠️')}
        </div>
        """, unsafe_allow_html=True)

# ===============================
# Main Application
# ===============================

def main():
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;">🧬 Evolutionary Learning Laboratory</h1>
        <p style="font-size: 0.95rem; margin-top: 0.5rem;">
            Temperature-feedback driven adaptation in synthetic yeast populations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar Controls (slightly minimized) ---
    st.sidebar.markdown("## 🎛️ Simulation Parameters")
    st.sidebar.markdown("Configure your experiment below:")

    with st.sidebar.expander("🔬 **Core Experimental Setup**", expanded=True):
        mode = st.selectbox("GFP Expression Mode", ["continuous", "binary"], index=0,
                            help="Continuous: 0–100 with noise • Binary: High/Low with switching")
        total_time = st.slider("Simulation Time (minutes)", 200, 3000, 1000, step=100)
        population_size = st.slider("Population Size (cells)", 50, 800, 200, step=25)
        num_passives = st.slider("Number of Passive Wells", 1, 8, 3, step=1)

    with st.sidebar.expander("🌡️ **Temperature Feedback System**", expanded=False):  # CHANGED default collapsed
        feedback_mode = st.selectbox("Feedback Function Type", ["linear", "sigmoid", "step", "exponential"], index=0)
        feedback_sensitivity = st.slider("Feedback Sensitivity", 0.2, 4.0, 1.0, step=0.1)
        st.markdown("*Temperature Range: 30 °C ↔ 39 °C*")

    with st.sidebar.expander("🧬 **Evolution & Fitness Parameters**", expanded=False):  # CHANGED default collapsed
        inherit_noise = st.slider("Inheritance Noise (continuous)", 1.0, 20.0, 5.0, step=1.0)
        switch_rate = st.slider("Stress-Induced Switching Rate", 0.001, 0.08, 0.01, step=0.001)
        fitness_cost = st.slider("GFP Metabolic Cost", 0.0, 1.2, 0.3, step=0.05)
        cost_exponent = st.slider("Cost Function Curvature", 0.5, 3.0, 1.5, step=0.1)

    with st.sidebar.expander("⚙️ **Advanced Settings**", expanded=False):
        random_seed = st.number_input("Random Seed (reproducibility)", min_value=1, max_value=999999, value=42, step=1)
        enable_validation = st.checkbox("Show Parameter Warnings", value=True)

    # --- Parameter objects ---
    sim_params = SimulationParams(
        total_time=total_time,
        population_size=population_size,
        num_passive_wells=num_passives,
        feedback_mode=feedback_mode,
        feedback_sensitivity=feedback_sensitivity,
        random_seed=random_seed
    )
    gfp_params = GFPParams(
        inherit_sd=inherit_noise,
        switch_prob_base=switch_rate,
        cost_strength=fitness_cost,
        cost_exponent=cost_exponent
    )

    st.markdown(create_parameter_display(sim_params, gfp_params, mode), unsafe_allow_html=True)
    if enable_validation:
        warnings = validate_parameters(sim_params, gfp_params)
        if warnings:
            display_warnings(warnings)

    # --- ALWAYS show a compact preview BEFORE running (and after, too) ---
    st.markdown('<div class="section-header">📈 Quick Preview: GFP → Temperature</div>', unsafe_allow_html=True)
    display_feedback_preview(feedback_mode, feedback_sensitivity, height=320)  # CHANGED: compact & always shown

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
            status_text.text("🧬 Running evolutionary simulation...")
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
        with st.expander("🎯 What to Expect (quick primer)", expanded=False):  # CHANGED: collapsed by default
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Learning Dynamics**
                - Driver learns to increase GFP → cooler temps
                - Passives follow temperature only
                - Controls at 30 °C and 39 °C
                """)
            with col2:
                st.markdown("""
                **Biology**
                - Moran process birth–death
                - GFP cost slows division
                - Heat increases switching
                """)

def display_feedback_preview(feedback_mode: str, sensitivity: float, height: int = 320):
    """Compact preview of the feedback function (always visible)."""
    gfp_range, temp_range = create_feedback_function_data(feedback_mode, sensitivity)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gfp_range, y=temp_range, mode='lines',
        name=f'{feedback_mode.title()} Feedback', line=dict(width=3)
    ))
    fig.update_layout(
        title=f"Temperature Feedback: {feedback_mode.title()} (Sensitivity {sensitivity})",
        xaxis_title="Mean Population GFP",
        yaxis_title="Environmental Temperature (°C)",
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=50, b=40)  # tighter
    )
    st.plotly_chart(fig, use_container_width=True)

def display_results(results: Dict[str, Any], metrics: Dict[str, float], params: Dict[str, Any]):
    st.markdown('<div class="section-header">📊 Simulation Results</div>', unsafe_allow_html=True)

    # --- Key Metrics Dashboard (compact) ---
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
        final_temp = metrics.get('final_temperature', 39)
        st.markdown(create_metric_card("Final Temp", f"{final_temp:.1f}°C", f"Cooling {39-final_temp:.1f}°C"),
                    unsafe_allow_html=True)
    with col4:
        adaptation_time = metrics.get('adaptation_time', None)
        time_text = f"{adaptation_time:.0f} min" if adaptation_time else "No adaptation"
        st.markdown(create_metric_card("Adaptation Time", time_text, "50% of final"), unsafe_allow_html=True)
    with col5:
        high_gfp_frac = metrics.get('final_high_gfp_fraction', 0)
        st.markdown(create_metric_card("High GFP Fraction", f"{high_gfp_frac:.2f}", "Population success"),
                    unsafe_allow_html=True)

    # --- Temperature Evolution (shorter) ---
    st.markdown("### 🌡️ Temperature Evolution")
    driver_data = results['driver']
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=driver_data['time'], y=driver_data['temperature'],
        mode='lines', name='Driver Temperature',
        line=dict(width=3),
        hovertemplate='Time: %{x} min<br>Temp: %{y:.1f}°C<extra></extra>'
    ))
    if metrics.get('adaptation_time'):
        fig_temp.add_vline(x=metrics['adaptation_time'], line=dict(dash="dash"),
                           annotation_text="50% Adaptation")
    fig_temp.update_layout(
        title="Environmental Temperature (feedback)",
        xaxis_title="Time (min)", yaxis_title="Temp (°C)",
        template="plotly_white", hovermode='x unified', height=380  # CHANGED: lower height
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # --- GFP Evolution (shorter) ---
    st.markdown("### 🧬 GFP Evolution: Driver vs. Controls")
    fig_gfp = go.Figure()
    fig_gfp.add_trace(go.Scatter(x=driver_data['time'], y=driver_data['mean_gfp'],
                                 mode='lines', name='🎯 Driver (Feedback)', line=dict(width=4)))
    fig_gfp.add_trace(go.Scatter(x=results['control_30']['time'], y=results['control_30']['mean_gfp'],
                                 mode='lines', name='🟢 Control 30°C', line=dict(dash='dash', width=2)))
    fig_gfp.add_trace(go.Scatter(x=results['control_39']['time'], y=results['control_39']['mean_gfp'],
                                 mode='lines', name='🔴 Control 39°C', line=dict(dash='dash', width=2)))
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
        template="plotly_white", hovermode='x unified', height=380  # CHANGED
    )
    st.plotly_chart(fig_gfp, use_container_width=True)

    # # --- Phase Plot + Population Events (more compact) ---
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.markdown("#### 🔄 Phase: GFP → Temperature")
    #     fig_phase = go.Figure()
    #     fig_phase.add_trace(go.Scatter(x=driver_data['mean_gfp'], y=driver_data['temperature'],
    #                                    mode='lines+markers', name='Trajectory',
    #                                    marker=dict(size=3)))
    #     fig_phase.add_trace(go.Scatter(x=[driver_data['mean_gfp'][0]], y=[driver_data['temperature'][0]],
    #                                    mode='markers', name='Start',
    #                                    marker=dict(size=10, symbol='circle')))
    #     fig_phase.add_trace(go.Scatter(x=[driver_data['mean_gfp'][-1]], y=[driver_data['temperature'][-1]],
    #                                    mode='markers', name='End',
    #                                    marker=dict(size=10, symbol='star')))
    #     fig_phase.update_layout(template="plotly_white", height=320)  # CHANGED
    #     st.plotly_chart(fig_phase, use_container_width=True)

    # with col2:
    #     st.markdown("#### 👥 Population Dynamics")
    #     fig_pop = make_subplots(rows=2, cols=1, subplot_titles=["High GFP Fraction", "Births & Switches"],
    #                             vertical_spacing=0.12, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
    #     fig_pop.add_trace(go.Scatter(x=driver_data['time'], y=driver_data['high_gfp_fraction'],
    #                                  name='Driver High GFP', line=dict(width=2)), row=1, col=1)
    #     if results['passives']:
    #         passive_high_frac = np.mean([p['high_gfp_fraction'] for p in results['passives']], axis=0)
    #         fig_pop.add_trace(go.Scatter(x=driver_data['time'], y=passive_high_frac,
    #                                      name='Passives High GFP', line=dict(dash='dot', width=2)), row=1, col=1)
    #     fig_pop.add_trace(go.Scatter(x=driver_data['time'], y=driver_data['births'],
    #                                  name='Births', line=dict(width=2)), row=2, col=1)
    #     fig_pop.add_trace(go.Scatter(x=driver_data['time'], y=driver_data['switches'],
    #                                  name='Switches', line=dict(width=2)), row=2, col=1, secondary_y=True)
    #     fig_pop.update_layout(height=320, template="plotly_white")  # CHANGED
    #     fig_pop.update_xaxes(title_text="Time (min)", row=2, col=1)
    #     fig_pop.update_yaxes(title_text="Fraction", row=1, col=1)
    #     fig_pop.update_yaxes(title_text="Births", row=2, col=1)
    #     fig_pop.update_yaxes(title_text="Switches", row=2, col=1, secondary_y=True)
    #     st.plotly_chart(fig_pop, use_container_width=True)

    # --- Feedback Function (full) ---
    st.markdown("### 📈 Feedback Function & Evolution Path")
    gfp_range, temp_range = create_feedback_function_data(
        params['sim_params'].feedback_mode,
        params['sim_params'].feedback_sensitivity, n_points=300
    )
    fig_feedback = go.Figure()
    fig_feedback.add_trace(go.Scatter(x=gfp_range, y=temp_range, mode='lines',
                                      name=f'{params["sim_params"].feedback_mode.title()} Feedback',
                                      line=dict(width=4)))
    fig_feedback.add_trace(go.Scatter(x=driver_data['mean_gfp'], y=driver_data['temperature'],
                                      mode='markers+lines', name='Actual Path',
                                      line=dict(width=2, dash='dot'),
                                      marker=dict(size=4)))
    fig_feedback.add_hrect(y0=30, y1=32, fillcolor="lightgreen", opacity=0.25,
                           annotation_text="Reward Zone", annotation_position="top left")
    fig_feedback.add_hrect(y0=37, y1=39, fillcolor="lightcoral", opacity=0.25,
                           annotation_text="Stress Zone", annotation_position="top left")
    fig_feedback.update_layout(
        title=f"Feedback & Trajectory ({params['sim_params'].feedback_mode.title()})",
        xaxis_title="Mean Population GFP",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=420  # CHANGED: slightly smaller
    )
    st.plotly_chart(fig_feedback, use_container_width=True)

    # --- Detailed Analysis (collapsed) ---
    with st.expander("🔬 Detailed Analysis", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 Learning Progress")
            temp_change = np.array(driver_data['temperature'])
            learning_curve = (39 - temp_change) / 9
            fig_learning = go.Figure()
            fig_learning.add_trace(go.Scatter(x=driver_data['time'], y=learning_curve, mode='lines',
                                              name='Learning Progress', line=dict(width=2)))
            fig_learning.add_hline(y=0.5, line=dict(dash="dash"), annotation_text="50% Learning")
            fig_learning.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig_learning, use_container_width=True)
        with col2:
            st.markdown("#### 📊 Final State Stats")
            final_stats = {
                'Metric': ['Driver Final GFP', 'Control 30°C GFP', 'Control 39°C GFP',
                           'Passive Mean GFP', 'Learning Advantage', 'Temperature Reduction'],
                'Value': [
                    f"{driver_data['mean_gfp'][-1]:.1f}",
                    f"{results['control_30']['mean_gfp'][-1]:.1f}",
                    f"{results['control_39']['mean_gfp'][-1]:.1f}",
                    f"{np.mean([p['mean_gfp'][-1] for p in results['passives']]):.1f}" if results['passives'] else "N/A",
                    f"{driver_data['mean_gfp'][-1] - np.mean([p['mean_gfp'][-1] for p in results['passives']]):.1f}" if results['passives'] else "N/A",
                    f"{39 - driver_data['temperature'][-1]:.1f}°C"
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
