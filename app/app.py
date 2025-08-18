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

# Import our core simulation logic
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
    initial_sidebar_state="expanded",
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
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Parameter display styling */
    .param-display {
        background: linear-gradient(135deg, #667eea 0%, #c3cfe2 50%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #667eea;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        font-size: 1.6rem;
        color: #2c3e50;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e1e5e9;
    }
    
    /* Success/warning styling */
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
    
    /* Custom section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Helper Functions
# ===============================

def create_parameter_display(sim_params: SimulationParams, gfp_params: GFPParams, mode: str) -> str:
    """Create formatted parameter display string"""
    return f"""
    <div class="param-display">
        <h4>🎯 Current Configuration</h4>
        <strong>Mode:</strong> {mode.title()} GFP Expression | 
        <strong>Feedback:</strong> {sim_params.feedback_mode.title()} (sensitivity: {sim_params.feedback_sensitivity}) | 
        <strong>Population:</strong> {sim_params.population_size} cells | 
        <strong>Time:</strong> {sim_params.total_time} min | 
        <strong>Passives:</strong> {sim_params.num_passive_wells} wells | 
        <strong>GFP Cost:</strong> {gfp_params.cost_strength:.2f}
    </div>
    """

def create_metric_card(title: str, value: str, delta: str = None, 
                      delta_color: str = "normal") -> str:
    """Create a custom metric card"""
    delta_html = ""
    if delta:
        color = "green" if delta_color == "normal" else "red"
        delta_html = f'<div style="color: {color}; font-size: 0.8rem;">{delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 2rem; font-weight: 700; color: #2c3e50;">{value}</div>
        {delta_html}
    </div>
    """

def display_warnings(warnings: List[str]):
    """Display parameter validation warnings"""
    if warnings:
        warning_text = "\\n".join(warnings)
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚠️ Parameter Warnings:</strong><br>
            {warning_text.replace('⚠️', '<br>⚠️')}
        </div>
        """, unsafe_allow_html=True)

# ===============================
# Main Application
# ===============================

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Evolutionary Learning Laboratory</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Investigating temperature-feedback driven adaptation in synthetic yeast populations
        </p>
        <p style="font-size: 1rem; opacity: 0.9; margin-top: 0.5rem;">
            Explore how populations learn through environmental feedback • Binary & Continuous GFP Modes • Moran Process Dynamics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===============================
    # Sidebar Controls
    # ===============================
    
    st.sidebar.markdown("## 🎛️ Simulation Parameters")
    st.sidebar.markdown("Configure your evolutionary learning experiment below:")
    
    # Core experimental setup
    with st.sidebar.expander("🔬 **Core Experimental Setup**", expanded=True):
        mode = st.selectbox(
            "GFP Expression Mode", 
            ["continuous", "binary"],
            index=0,
            help="**Continuous**: 0-100 scale with inheritance noise | **Binary**: High/Low states with switching"
        )
        
        total_time = st.slider(
            "Simulation Time (minutes)", 
            200, 3000, 1000, step=100,
            help="**Total evolution time** - Longer allows more adaptation but takes more compute time"
        )
        
        population_size = st.slider(
            "Population Size (cells)", 
            50, 800, 200, step=25,
            help="**Constant size maintained by Moran process** - Larger populations have less drift"
        )
        
        num_passives = st.slider(
            "Number of Passive Wells", 
            1, 8, 3, step=1,
            help="**Wells that follow driver temperature** but don't influence it (breaks learning loop)"
        )
    
    # Temperature feedback system
    with st.sidebar.expander("🌡️ **Temperature Feedback System**", expanded=True):
        feedback_mode = st.selectbox(
            "Feedback Function Type",
            ["linear", "sigmoid", "step", "exponential"],
            index=0,
            help="**How GFP levels translate to temperature**: Linear=proportional, Sigmoid=S-curve, Step=threshold, Exponential=rapid response"
        )
        
        feedback_sensitivity = st.slider(
            "Feedback Sensitivity",
            0.2, 4.0, 1.0, step=0.1,
            help="**System responsiveness to GFP changes** - Higher values = more sensitive feedback"
        )
        
        st.markdown("*Temperature Range: 30°C (reward) ↔ 39°C (stress)*")
    
    # Evolution and fitness parameters
    with st.sidebar.expander("🧬 **Evolution & Fitness Parameters**", expanded=True):
        inherit_noise = st.slider(
            "Inheritance Noise (continuous mode)",
            1.0, 20.0, 5.0, step=1.0,
            help="**Variability in daughter cell GFP** - Higher values increase mutation-like effects"
        )
        
        switch_rate = st.slider(
            "Stress-Induced Switching Rate",
            0.001, 0.08, 0.01, step=0.001,
            help="**Base probability of phenotype switching per minute** at maximum stress (39°C)"
        )
        
        fitness_cost = st.slider(
            "GFP Metabolic Cost",
            0.0, 1.2, 0.3, step=0.05,
            help="**How much high GFP slows cell division** - Realistic metabolic burden"
        )
        
        cost_exponent = st.slider(
            "Cost Function Curvature",
            0.5, 3.0, 1.5, step=0.1,
            help="**Non-linearity of GFP cost** - Higher values = steeper cost at high GFP"
        )
    
    # Advanced settings
    with st.sidebar.expander("⚙️ **Advanced Settings**", expanded=False):
        random_seed = st.number_input(
            "Random Seed (reproducibility)",
            min_value=1, max_value=999999, value=42, step=1,
            help="**Set for reproducible results** - Same seed gives identical simulations"
        )
        
        enable_validation = st.checkbox(
            "Show Parameter Warnings",
            value=True,
            help="**Display warnings for potentially problematic parameter combinations**"
        )
    
    # ===============================
    # Parameter Setup and Validation
    # ===============================
    
    # Create parameter objects
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
    
    # Display current configuration
    st.markdown(create_parameter_display(sim_params, gfp_params, mode), 
                unsafe_allow_html=True)
    
    # Validate parameters and show warnings
    if enable_validation:
        warnings = validate_parameters(sim_params, gfp_params)
        if warnings:
            display_warnings(warnings)
    
    # ===============================
    # Simulation Execution
    # ===============================
    
    # Run simulation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_simulation = st.button(
            "🚀 **Run Evolution Experiment**", 
            type="primary",
            use_container_width=True
        )
    
    if run_simulation:
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Run simulation
            status_text.text("🔬 Initializing populations...")
            progress_bar.progress(10)
            
            status_text.text("🧬 Running evolutionary simulation...")
            progress_bar.progress(30)
            
            results = run_evolution_experiment(sim_params, gfp_params, mode)
            
            progress_bar.progress(80)
            status_text.text("📊 Calculating metrics...")
            
            # Calculate learning metrics
            metrics = calculate_learning_metrics(results)
            
            progress_bar.progress(100)
            status_text.text("✅ Simulation completed!")
            
            # Store results in session state
            st.session_state['results'] = results
            st.session_state['metrics'] = metrics
            st.session_state['params'] = {
                'sim_params': sim_params,
                'gfp_params': gfp_params,
                'mode': mode
            }
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Success message
            st.markdown("""
            <div class="success-box">
                <strong>🎉 Simulation completed successfully!</strong><br>
                Scroll down to explore your results and discover evolutionary dynamics.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Simulation failed: {str(e)}")
            st.exception(e)
    
    # ===============================
    # Results Display
    # ===============================
    
    if 'results' in st.session_state and 'metrics' in st.session_state:
        display_results(
            st.session_state['results'],
            st.session_state['metrics'],
            st.session_state['params']
        )
    else:
        # Show preview of feedback function
        st.markdown('<div class="section-header">📈 Preview: Feedback Function</div>', 
                   unsafe_allow_html=True)
        display_feedback_preview(feedback_mode, feedback_sensitivity)
        
        # Show example results or tutorial
        st.markdown('<div class="section-header">🎯 What to Expect</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🧠 Learning Dynamics
            - **Driver well** learns to increase GFP expression to get cooler temperatures
            - **Passive wells** follow the same temperature but cannot influence it
            - **Control wells** stay at fixed temperatures (30°C and 39°C)
            
            #### 📊 Key Metrics
            - **Learning Score**: How well the driver adapts (0-1 scale)
            - **Adaptation Time**: How quickly learning occurs
            - **Temperature Stability**: Consistency of final state
            """)
        
        with col2:
            st.markdown("""
            #### 🔬 Biological Processes
            - **Moran Process**: Constant population size with birth-death balance
            - **Fitness Costs**: High GFP slows division (metabolic burden)
            - **Stress Switching**: Higher temperature increases GFP switching
            
            #### 🎛️ Experimental Controls
            - **Binary vs Continuous**: Different GFP expression modes
            - **Feedback Functions**: Various response curves
            - **Multiple Replicates**: Passive wells for statistical power
            """)

def display_feedback_preview(feedback_mode: str, sensitivity: float):
    """Display preview of the feedback function"""
    
    gfp_range, temp_range = create_feedback_function_data(feedback_mode, sensitivity)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gfp_range,
        y=temp_range,
        mode='lines',
        name=f'{feedback_mode.title()} Feedback',
        line=dict(color='#667eea', width=3)
    ))
    
    # Add annotations
    fig.add_annotation(
        x=75, y=32,
        text="Reward Zone<br>(Cool Temperature)",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#28a745",
        font=dict(color="#28a745")
    )
    
    fig.add_annotation(
        x=25, y=37,
        text="Stress Zone<br>(Hot Temperature)",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#dc3545",
        font=dict(color="#dc3545")
    )
    
    fig.update_layout(
        title=f"Temperature Feedback Function: {feedback_mode.title()} (Sensitivity: {sensitivity})",
        xaxis_title="Mean Population GFP",
        yaxis_title="Environmental Temperature (°C)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_results(results: Dict[str, Any], metrics: Dict[str, float], 
                   params: Dict[str, Any]):
    """Display comprehensive simulation results"""
    
    st.markdown('<div class="section-header">📊 Simulation Results</div>', 
               unsafe_allow_html=True)
    
    # ===============================
    # Key Metrics Dashboard
    # ===============================
    
    st.markdown("### 🎯 Key Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        learning_score = metrics.get('learning_score', 0)
        score_color = "🟢" if learning_score > 0.7 else "🟡" if learning_score > 0.3 else "🔴"
        st.markdown(create_metric_card(
            "Learning Score", 
            f"{learning_score:.2f}",
            f"{score_color} {'Excellent' if learning_score > 0.7 else 'Moderate' if learning_score > 0.3 else 'Poor'}"
        ), unsafe_allow_html=True)
    
    with col2:
        final_gfp = metrics.get('final_gfp', 0)
        st.markdown(create_metric_card(
            "Final Driver GFP", 
            f"{final_gfp:.1f}",
            f"Target: >60"
        ), unsafe_allow_html=True)
    
    with col3:
        final_temp = metrics.get('final_temperature', 39)
        st.markdown(create_metric_card(
            "Final Temperature", 
            f"{final_temp:.1f}°C",
            f"Cooling: {39-final_temp:.1f}°C"
        ), unsafe_allow_html=True)
    
    with col4:
        adaptation_time = metrics.get('adaptation_time', None)
        time_text = f"{adaptation_time:.0f} min" if adaptation_time else "No adaptation"
        st.markdown(create_metric_card(
            "Adaptation Time", 
            time_text,
            "50% of final state"
        ), unsafe_allow_html=True)
    
    with col5:
        high_gfp_frac = metrics.get('final_high_gfp_fraction', 0)
        st.markdown(create_metric_card(
            "High GFP Fraction", 
            f"{high_gfp_frac:.2f}",
            f"Population success"
        ), unsafe_allow_html=True)
    
    # ===============================
    # Temperature Evolution
    # ===============================
    
    st.markdown("### 🌡️ Temperature Evolution (Driver Feedback Loop)")
    
    driver_data = results['driver']
    
    fig_temp = go.Figure()
    
    # Main temperature trace
    fig_temp.add_trace(go.Scatter(
        x=driver_data['time'],
        y=driver_data['temperature'],
        mode='lines',
        name='Driver Temperature',
        line=dict(color='#e74c3c', width=3),
        hovertemplate='Time: %{x} min<br>Temperature: %{y:.1f}°C<extra></extra>'
    ))
    
    # Add zones
    fig_temp.add_hrect(y0=30, y1=33, fillcolor="lightgreen", opacity=0.2, 
                      annotation_text="Optimal Zone", annotation_position="top left")
    fig_temp.add_hrect(y0=36, y1=39, fillcolor="lightcoral", opacity=0.2,
                      annotation_text="Stress Zone", annotation_position="top left")
    
    # Mark key events
    if metrics.get('adaptation_time'):
        fig_temp.add_vline(
            x=metrics['adaptation_time'], 
            line=dict(color="blue", dash="dash"),
            annotation_text="50% Adaptation"
        )
    
    fig_temp.update_layout(
        title="Environmental Temperature Response to Driver Population GFP",
        xaxis_title="Time (minutes)",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # ===============================
    # GFP Evolution Comparison
    # ===============================
    
    st.markdown("### 🧬 GFP Expression Evolution: Learning vs Controls")
    
    fig_gfp = go.Figure()
    
    # Driver (feedback enabled)
    fig_gfp.add_trace(go.Scatter(
        x=driver_data['time'],
        y=driver_data['mean_gfp'],
        mode='lines',
        name='🎯 Driver (Feedback)',
        line=dict(color='#3498db', width=4),
        hovertemplate='Time: %{x} min<br>GFP: %{y:.1f}<extra></extra>'
    ))
    
    # Control wells
    fig_gfp.add_trace(go.Scatter(
        x=results['control_30']['time'],
        y=results['control_30']['mean_gfp'],
        mode='lines',
        name='🟢 Control 30°C',
        line=dict(color='#27ae60', dash='dash', width=2)
    ))
    
    fig_gfp.add_trace(go.Scatter(
        x=results['control_39']['time'],
        y=results['control_39']['mean_gfp'],
        mode='lines',
        name='🔴 Control 39°C',
        line=dict(color='#e74c3c', dash='dash', width=2)
    ))
    
    # Passive wells (averaged)
    if results['passives']:
        passive_mean_gfp = np.mean([p['mean_gfp'] for p in results['passives']], axis=0)
        passive_std_gfp = np.std([p['mean_gfp'] for p in results['passives']], axis=0)
        
        # Mean line
        fig_gfp.add_trace(go.Scatter(
            x=driver_data['time'],
            y=passive_mean_gfp,
            mode='lines',
            name=f'🟠 Passives (n={len(results["passives"])})',
            line=dict(color='#f39c12', width=2, dash='dot')
        ))
        
        # Confidence interval
        fig_gfp.add_trace(go.Scatter(
            x=driver_data['time'] + driver_data['time'][::-1],
            y=np.concatenate([passive_mean_gfp + passive_std_gfp, 
                             (passive_mean_gfp - passive_std_gfp)[::-1]]),
            fill='tonexty',
            fillcolor='rgba(243, 156, 18, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Passive Range',
            showlegend=False
        ))
    
    # High GFP threshold line
    threshold = 60 if params['mode'] == 'continuous' else 50
    fig_gfp.add_hline(y=threshold, line=dict(color="gray", dash="dot"),
                     annotation_text=f"High GFP Threshold ({threshold})")
    
    fig_gfp.update_layout(
        title="GFP Expression Evolution: Driver Shows Learning, Passives Don't",
        xaxis_title="Time (minutes)",
        yaxis_title="Mean GFP Expression",
        template="plotly_white",
        hovermode='x unified',
        height=500,
        legend=dict(x=0, y=1)
    )
    
    st.plotly_chart(fig_gfp, use_container_width=True)
    
    # ===============================
    # Phase Plot and Population Dynamics
    # ===============================
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 Phase Plot: GFP → Temperature")
        
        fig_phase = go.Figure()
        
        # Trajectory line
        fig_phase.add_trace(go.Scatter(
            x=driver_data['mean_gfp'],
            y=driver_data['temperature'],
            mode='lines+markers',
            name='Evolution Trajectory',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=3, color=driver_data['time'], 
                       colorscale='Viridis', showscale=True,
                       colorbar=dict(title="Time (min)"))
        ))
        
        # Start and end points
        fig_phase.add_trace(go.Scatter(
            x=[driver_data['mean_gfp'][0]], y=[driver_data['temperature'][0]],
            mode='markers', name='Start',
            marker=dict(size=12, color='red', symbol='circle')
        ))
        
        fig_phase.add_trace(go.Scatter(
            x=[driver_data['mean_gfp'][-1]], y=[driver_data['temperature'][-1]],
            mode='markers', name='End',
            marker=dict(size=12, color='green', symbol='star')
        ))
        
        fig_phase.update_layout(
            title="Driver: GFP-Temperature Feedback Loop",
            xaxis_title="Mean GFP",
            yaxis_title="Temperature (°C)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_phase, use_container_width=True)
    
    with col2:
        st.markdown("#### 👥 Population Dynamics")
        
        fig_pop = make_subplots(
            rows=2, cols=1,
            subplot_titles=["High GFP Fraction", "Population Events"],
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # High GFP fraction
        fig_pop.add_trace(
            go.Scatter(x=driver_data['time'], y=driver_data['high_gfp_fraction'], 
                      name='Driver High GFP', line=dict(color='#3498db', width=2)),
            row=1, col=1
        )
        
        if results['passives']:
            passive_high_frac = np.mean([p['high_gfp_fraction'] for p in results['passives']], axis=0)
            fig_pop.add_trace(
                go.Scatter(x=driver_data['time'], y=passive_high_frac, 
                          name='Passives High GFP', line=dict(color='#f39c12', dash='dot')),
                row=1, col=1
            )
        
        # Population events
        fig_pop.add_trace(
            go.Scatter(x=driver_data['time'], y=driver_data['births'], 
                      name='Births', line=dict(color='green')),
            row=2, col=1
        )
        
        fig_pop.add_trace(
            go.Scatter(x=driver_data['time'], y=driver_data['switches'], 
                      name='Switches', line=dict(color='orange')),
            row=2, col=1, secondary_y=True
        )
        
        fig_pop.update_layout(height=400, template="plotly_white")
        fig_pop.update_xaxes(title_text="Time (minutes)", row=2, col=1)
        fig_pop.update_yaxes(title_text="Fraction", row=1, col=1)
        fig_pop.update_yaxes(title_text="Births/Deaths", row=2, col=1)
        fig_pop.update_yaxes(title_text="Switches", row=2, col=1, secondary_y=True)
        
        st.plotly_chart(fig_pop, use_container_width=True)
    
    # ===============================
    # Feedback Function Analysis
    # ===============================
    
    st.markdown("### 📈 Feedback Function Relationship")
    
    # Create comprehensive feedback function visualization
    gfp_range, temp_range = create_feedback_function_data(
        params['sim_params'].feedback_mode, 
        params['sim_params'].feedback_sensitivity, 
        n_points=300
    )
    
    fig_feedback = go.Figure()
    
    # Main feedback curve
    fig_feedback.add_trace(go.Scatter(
        x=gfp_range,
        y=temp_range,
        mode='lines',
        name=f'{params["sim_params"].feedback_mode.title()} Feedback',
        line=dict(color='#2c3e50', width=4),
        hovertemplate='GFP: %{x:.1f}<br>Temperature: %{y:.1f}°C<extra></extra>'
    ))
    
    # Add actual trajectory from simulation
    fig_feedback.add_trace(go.Scatter(
        x=driver_data['mean_gfp'],
        y=driver_data['temperature'],
        mode='markers+lines',
        name='Actual Evolution Path',
        line=dict(color='#e74c3c', width=2, dash='dot'),
        marker=dict(
            size=4,
            color=driver_data['time'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time (min)", x=1.02)
        ),
        opacity=0.8,
        hovertemplate='Time: %{marker.color:.0f} min<br>GFP: %{x:.1f}<br>Temp: %{y:.1f}°C<extra></extra>'
    ))
    
    # Mark start and end points
    fig_feedback.add_trace(go.Scatter(
        x=[driver_data['mean_gfp'][0]], 
        y=[driver_data['temperature'][0]],
        mode='markers',
        name='Start',
        marker=dict(size=15, color='red', symbol='circle', line=dict(width=2, color='white')),
        showlegend=True
    ))
    
    fig_feedback.add_trace(go.Scatter(
        x=[driver_data['mean_gfp'][-1]], 
        y=[driver_data['temperature'][-1]],
        mode='markers',
        name='End',
        marker=dict(size=15, color='green', symbol='star', line=dict(width=2, color='white')),
        showlegend=True
    ))
    
    # Add zones and annotations
    fig_feedback.add_hrect(
        y0=30, y1=32, 
        fillcolor="lightgreen", opacity=0.3,
        annotation_text="Reward Zone", annotation_position="top left"
    )
    
    fig_feedback.add_hrect(
        y0=37, y1=39, 
        fillcolor="lightcoral", opacity=0.3,
        annotation_text="Stress Zone", annotation_position="top left"
    )
    
    # Add sensitivity indicator
    sensitivity_text = f"Sensitivity: {params['sim_params'].feedback_sensitivity:.1f}"
    if params['sim_params'].feedback_sensitivity > 2.0:
        sensitivity_text += " (High - Rapid Response)"
    elif params['sim_params'].feedback_sensitivity < 0.5:
        sensitivity_text += " (Low - Gradual Response)"
    else:
        sensitivity_text += " (Moderate)"
    
    fig_feedback.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=sensitivity_text,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    fig_feedback.update_layout(
        title=f"Feedback Function & Evolution Trajectory ({params['sim_params'].feedback_mode.title()} Mode)",
        xaxis_title="Mean Population GFP",
        yaxis_title="Environmental Temperature (°C)",
        template="plotly_white",
        height=500,
        hovermode='closest',
        legend=dict(x=0.02, y=0.02, bgcolor="rgba(255,255,255,0.8)")
    )
    
    # Add grid for better readability
    fig_feedback.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_feedback.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    st.plotly_chart(fig_feedback, use_container_width=True)
    
    # Add interpretation text
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📊 Function Analysis:**
        - **Theoretical curve** shows ideal feedback relationship
        - **Evolution path** shows actual population trajectory  
        - **Color gradient** indicates time progression
        - **Start/End markers** show learning progress
        """)
    
    with col2:
        # Calculate feedback efficiency
        theoretical_temp = feedback_temperature(
            driver_data['mean_gfp'][-1], 
            params['sim_params'].feedback_mode,
            params['sim_params'].feedback_sensitivity
        )
        actual_temp = driver_data['temperature'][-1]
        efficiency = abs(theoretical_temp - actual_temp)
        
        st.markdown(f"""
        **🎯 Feedback Efficiency:**
        - **Expected final temp:** {theoretical_temp:.1f}°C
        - **Actual final temp:** {actual_temp:.1f}°C  
        - **Tracking error:** {efficiency:.1f}°C
        - **Efficiency:** {"Excellent" if efficiency < 1 else "Good" if efficiency < 3 else "Poor"}
        """)

    # ===============================
    # Detailed Analysis
    # ===============================
    
    with st.expander("🔬 **Detailed Analysis & Statistics**", expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Learning Dynamics")
            
            # Learning curve analysis
            temp_change = np.array(driver_data['temperature'])
            learning_curve = (39 - temp_change) / 9  # Normalize to 0-1
            
            fig_learning = go.Figure()
            fig_learning.add_trace(go.Scatter(
                x=driver_data['time'],
                y=learning_curve,
                mode='lines',
                name='Learning Progress',
                line=dict(color='#e74c3c', width=2)
            ))
            
            fig_learning.add_hline(y=0.5, line=dict(color="gray", dash="dash"),
                                 annotation_text="50% Learning")
            
            fig_learning.update_layout(
                title="Learning Progress Over Time",
                xaxis_title="Time (minutes)",
                yaxis_title="Learning Score (0-1)",
                template="plotly_white",
                height=300
            )
            
            st.plotly_chart(fig_learning, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Final State Statistics")
            
            # Create summary statistics
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
            
            # Interpretation
            st.markdown("##### 🧠 Interpretation")
            if learning_score > 0.7:
                st.success("🎉 Excellent learning! The driver population successfully adapted to exploit the feedback mechanism.")
            elif learning_score > 0.3:
                st.warning("⚠️ Moderate learning observed. Some adaptation occurred but may need parameter tuning.")
            else:
                st.error("❌ Poor learning. The population failed to effectively exploit the feedback. Try adjusting parameters.")
    
    # ===============================
    # Data Export
    # ===============================
    
    st.markdown("### 💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export main results
        df_export = export_results_to_dataframe(results)
        csv_data = df_export.to_csv(index=False)
        st.download_button(
            label="📊 **Download Time Series (CSV)**",
            data=csv_data,
            file_name=f"evolution_results_{params['mode']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export parameters and metrics
        export_summary = {
            'simulation_parameters': params,
            'learning_metrics': metrics,
            'final_statistics': final_stats
        }
        json_data = json.dumps(export_summary, indent=2, default=str)
        st.download_button(
            label="⚙️ **Download Parameters (JSON)**",
            data=json_data,
            file_name=f"simulation_config_{params['mode']}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Export raw results
        raw_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="🔬 **Download Raw Data (JSON)**",
            data=raw_json,
            file_name=f"raw_results_{params['mode']}.json",
            mime="application/json",
            use_container_width=True
        )

# ===============================
# Application Entry Point
# ===============================

if __name__ == "__main__":
    main()