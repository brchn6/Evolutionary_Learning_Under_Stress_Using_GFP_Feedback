"""
Evolutionary Learning Simulation - Core Logic (Controls Fixed at 30°C & 39°C)
============================================================================

This module contains the core biological models and simulation logic for studying
temperature-feedback driven adaptation in synthetic yeast populations.

Update summary (controls lock-in):
- Control wells are HARD-CODED to 30.0 °C and 39.0 °C for the entire simulation.
  They never use feedback, smoothing, or start-up behavior.
- Driver still uses feedback + inertia as before.
- Export now includes control temperatures for verification.

STRICT Moran birth–death process:
- Each time step performs exactly one Moran event: 1 birth (fitness-proportional parent)
  and 1 death (uniform victim), keeping population size constant.
- Trait dynamics (background drift + temp-driven switching) update before fitness is
  computed each step, so fitness reflects current phenotypes.
- Daughter fitness is computed immediately at birth for the current temperature.

Smoothing / metrics features retained:
- Optional no-cliff startup via SimulationParams.start_at_max_temp
- Temperature inertia SimulationParams.temp_inertia for smooth dT/dt (driver only)
- Metrics can ignore an initial burn-in window via SimulationParams.metric_burn_in
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# ===============================
# Configuration Classes
# ===============================

@dataclass
class GFPParams:
    """Parameters for GFP expression dynamics and cellular processes"""
    # Binary mode thresholds
    low_threshold: float = 35.0
    high_threshold: float = 95.0
    # Continuous mode bounds
    min_val: float = 0.0
    max_val: float = 100.0
    # Inheritance parameters
    inherit_sd: float = 2.5
    # Switching probabilities (temperature dependent)
    switch_prob_base: float = 0.01
    switch_boost_mean: float = 15.0
    switch_boost_sd: float = 5.0
    # Fitness cost parameters
    cost_strength: float = 0.3
    cost_exponent: float = 1.5
    # Background dynamics
    drift_rate: float = 0.001
    baseline_gfp: float = 10.0
    noise_sd: float = 0.5

@dataclass
class SimulationParams:
    # Population & timing
    population_size: int = 200
    total_time: int = 1000      # minutes
    time_step: int = 1          # minutes
    num_passive_wells: int = 3

    # Temperature feedback (applies to DRIVER only)
    feedback_mode: str = "linear"   # 'linear', 'sigmoid', 'step', 'exponential'
    feedback_sensitivity: float = 1.0
    base_temp: float = 30.0
    max_temp: float = 39.0

    # Smoothing / starting behavior (applies to DRIVER only)
    temp_inertia: float = 0.2       # 0<inertia<=1; smaller = smoother
    start_at_max_temp: bool = True  # if False: first step starts at target (no jump)

    # Metrics
    metric_burn_in: int = 0         # minutes to ignore when computing adaptation time

    # Reproducibility
    random_seed: Optional[int] = None

# ===============================
# Core Biological Models
# ===============================

class Cell:
    def __init__(self, gfp_value: float, birth_time: float, mode: str = "continuous"):
        self.gfp = float(np.clip(gfp_value, 0.0, 100.0))
        self.birth_time = birth_time
        self.age = 0.0
        self.mode = mode
        self.generation_time = 120.0
        self._last_temp = 39.0  # Cache for efficiency

    def _calculate_base_generation_time(self, temperature: float) -> float:
        # 30->39 °C mapped to [0,1]
        temp_normalized = np.clip((temperature - 30.0) / 9.0, 0.0, 1.0)
        base_time = 60.0 + 120.0 * (temp_normalized ** 2)
        return base_time

    def _calculate_gfp_cost_multiplier(self, params: GFPParams) -> float:
        if self.mode == "binary":
            return 1.3 if self.gfp > 50 else 1.0
        else:
            normalized_gfp = self.gfp / 100.0
            cost_factor = params.cost_strength * (normalized_gfp ** params.cost_exponent)
            return 1.0 + cost_factor

    def update_fitness(self, temperature: float, params: GFPParams):
        base_time = self._calculate_base_generation_time(temperature)
        cost_multiplier = self._calculate_gfp_cost_multiplier(params)
        self.generation_time = base_time * cost_multiplier
        self._last_temp = temperature

    def divide(self, current_time: float, params: GFPParams) -> 'Cell':
        """Create a daughter with inheritance noise/switching behavior baked in."""
        if self.mode == "binary":
            # Binary inheritance with small chance of flipping state at division
            switch_prob = 0.05
            if np.random.random() < switch_prob:
                daughter_gfp = params.high_threshold if self.gfp < 50 else params.low_threshold
            else:
                daughter_gfp = self.gfp
        else:
            noise = np.random.normal(0, params.inherit_sd)
            daughter_gfp = np.clip(self.gfp + noise, params.min_val, params.max_val)

        self.age = 0.0
        daughter = Cell(daughter_gfp, current_time, self.mode)
        # Compute daughter's generation time immediately at current temperature
        daughter.update_fitness(self._last_temp, params)
        return daughter

    def phenotype_switch(self, temperature: float, params: GFPParams, dt: float = 1.0) -> bool:
        temp_factor = np.clip((temperature - 30.0) / 9.0, 0.0, 1.0)
        switch_prob = params.switch_prob_base * temp_factor * dt
        if np.random.random() < switch_prob:
            if self.mode == "binary":
                self.gfp = params.high_threshold if self.gfp < 50 else params.low_threshold
            else:
                boost = np.random.normal(params.switch_boost_mean, params.switch_boost_sd)
                self.gfp = np.clip(self.gfp + boost, params.min_val, params.max_val)
            return True
        return False

    def background_dynamics(self, params: GFPParams, dt: float = 1.0):
        if self.mode == "continuous":
            drift = -params.drift_rate * (self.gfp - params.baseline_gfp) * dt
            noise = np.random.normal(0, params.noise_sd * np.sqrt(dt))
            self.gfp = np.clip(self.gfp + drift + noise, params.min_val, params.max_val)

# ===============================
# Temperature Feedback Functions
# ===============================

def feedback_temperature(mean_gfp: float, mode: str = "linear",
                        sensitivity: float = 1.0, base_temp: float = 30.0,
                        max_temp: float = 39.0) -> float:
    gfp_norm = np.clip(mean_gfp / 100.0, 0.0, 1.0)
    x = np.clip(gfp_norm * sensitivity, 0.0, 1.0)
    if mode == "linear":
        cooling_factor = x
    elif mode == "sigmoid":
        k = 10.0
        cooling_factor = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
    elif mode == "step":
        threshold = 0.5
        cooling_factor = 1.0 if x >= threshold else 0.0
    elif mode == "exponential":
        cooling_factor = 1.0 - np.exp(-3.0 * x)
    else:
        cooling_factor = x
    temperature = max_temp - cooling_factor * (max_temp - base_temp)
    return float(np.clip(temperature, base_temp, max_temp))

# ===============================
# Population Simulation (STRICT Moran)
# ===============================

class Population:
    def __init__(self, target_size: int, mode: str, params: GFPParams,
                 label: str = "population"):
        self.target_size = target_size
        self.mode = mode
        self.params = params
        self.label = label
        self.cells: List[Cell] = []
        self.history = {
            'time': [], 'mean_gfp': [], 'std_gfp': [],
            'population_size': [], 'temperature': [],
            'high_gfp_fraction': [], 'mean_generation_time': [],
            'births': [], 'deaths': [], 'switches': []
        }
        self._initialize_population()

    def _initialize_population(self):
        """Initialize population with low GFP cells"""
        for _ in range(self.target_size):
            if self.mode == "binary":
                initial_gfp = self.params.low_threshold
            else:
                initial_gfp = max(0, np.random.normal(self.params.baseline_gfp, 3.0))
            cell = Cell(initial_gfp, 0.0, self.mode)
            self.cells.append(cell)

    def _calculate_population_stats(self, temperature: float) -> Dict[str, float]:
        if not self.cells:
            return {'mean_gfp': 0.0, 'std_gfp': 0.0,
                    'high_gfp_fraction': 0.0, 'mean_generation_time': 0.0}
        gfp_values = [cell.gfp for cell in self.cells]
        mean_gfp = np.mean(gfp_values)
        std_gfp = np.std(gfp_values)
        mean_gen_time = np.mean([cell.generation_time for cell in self.cells])
        high_threshold = 50.0 if self.mode == "binary" else 60.0
        high_gfp_fraction = sum(1 for gfp in gfp_values if gfp > high_threshold) / len(gfp_values)
        return {'mean_gfp': mean_gfp, 'std_gfp': std_gfp,
                'high_gfp_fraction': high_gfp_fraction, 'mean_generation_time': mean_gen_time}

    def step(self, current_time: float, temperature: float, dt: float = 1.0) -> Dict[str, int]:
        """One STRICT Moran event: exactly one birth and one death.

        Order:
          1) Update traits (age, background drift, switching)
          2) Update fitness from current temperature
          3) Choose a parent fitness-proportionally; create daughter
          4) Kill one random cell; replace with daughter
          5) Record stats
        """
        if not self.cells:
            self.history['time'].append(current_time)
            self.history['mean_gfp'].append(0.0)
            self.history['std_gfp'].append(0.0)
            self.history['population_size'].append(0)
            self.history['temperature'].append(temperature)
            self.history['high_gfp_fraction'].append(0.0)
            self.history['mean_generation_time'].append(0.0)
            self.history['births'].append(0)
            self.history['deaths'].append(0)
            self.history['switches'].append(0)
            return {'births': 0, 'deaths': 0, 'switches': 0}

        # 1) Trait updates before fitness
        switches = 0
        for cell in self.cells:
            cell.age += dt
            if self.mode == "continuous":
                cell.background_dynamics(self.params, dt)
            if cell.phenotype_switch(temperature, self.params, dt):
                switches += 1

        # 2) Update fitness for all cells at current temperature
        for cell in self.cells:
            cell.update_fitness(temperature, self.params)

        # 3) Fitness-proportional parent selection
        fitness_weights = np.array([1.0 / max(cell.generation_time, 1.0) for cell in self.cells], dtype=float)
        total_fitness = fitness_weights.sum()
        p = None if total_fitness <= 0 or not np.isfinite(total_fitness) else fitness_weights / total_fitness

        parent_idx = np.random.choice(len(self.cells), p=p)
        parent = self.cells[parent_idx]
        daughter = parent.divide(current_time, self.params)
        daughter.update_fitness(temperature, self.params)  # ensure exact match to current temp

        # 4) Uniform random death
        victim_idx = np.random.randint(len(self.cells))
        self.cells[victim_idx] = daughter
        births, deaths = 1, 1

        # 5) Record stats
        stats = self._calculate_population_stats(temperature)
        self.history['time'].append(current_time)
        self.history['mean_gfp'].append(stats['mean_gfp'])
        self.history['std_gfp'].append(stats['std_gfp'])
        self.history['population_size'].append(len(self.cells))
        self.history['temperature'].append(temperature)
        self.history['high_gfp_fraction'].append(stats['high_gfp_fraction'])
        self.history['mean_generation_time'].append(stats['mean_generation_time'])
        self.history['births'].append(births)
        self.history['deaths'].append(deaths)
        self.history['switches'].append(switches)

        return {'births': births, 'deaths': deaths, 'switches': switches}

    def get_final_gfp_distribution(self) -> List[float]:
        return [cell.gfp for cell in self.cells]

# ===============================
# Experiment Runner (driver uses feedback; controls fixed)
# ===============================

def run_evolution_experiment(sim_params: SimulationParams,
                             gfp_params: GFPParams,
                             mode: str = "continuous") -> Dict[str, Any]:
    """
    Run complete evolution experiment with driver, passive, and control wells.

    Driver startup & smoothing:
    - First-minute HOLD:
        If start_at_max_temp=True, we hold max_temp at t=0 then begin smoothing at t=1.
        If start_at_max_temp=False, we hold the initial feedback target at t=0.
    - Temperature inertia from t>=1:
        driver_temp <- prev_temp + temp_inertia * (target_temp - prev_temp)

    CONTROLS:
    - control_30 is fixed at EXACTLY 30.0 °C every step.
    - control_39 is fixed at EXACTLY 39.0 °C every step.
    """
    if sim_params.random_seed is not None:
        np.random.seed(sim_params.random_seed)

    # Fixed control temperatures (do not depend on sim_params)
    CONTROL_TEMP_30 = 30.0
    CONTROL_TEMP_39 = 39.0

    # Create wells
    driver = Population(sim_params.population_size, mode, gfp_params, "driver")
    passives = [Population(sim_params.population_size, mode, gfp_params, f"passive_{i+1}")
                for i in range(sim_params.num_passive_wells)]
    control_30 = Population(sim_params.population_size, mode, gfp_params, "control_30")
    control_39 = Population(sim_params.population_size, mode, gfp_params, "control_39")

    # Initial target for driver
    initial_mean_gfp = np.mean([cell.gfp for cell in driver.cells]) if driver.cells else 0.0
    target0 = feedback_temperature(
        initial_mean_gfp,
        mode=sim_params.feedback_mode,
        sensitivity=sim_params.feedback_sensitivity,
        base_temp=sim_params.base_temp,
        max_temp=sim_params.max_temp
    )

    # Initialize driver previous temperature according to start_at_max_temp
    inertia = float(np.clip(sim_params.temp_inertia, 1e-8, 1.0))
    prev_temp = sim_params.max_temp if sim_params.start_at_max_temp else target0

    # Main loop
    for t in range(0, sim_params.total_time, int(sim_params.time_step)):
        current_time = float(t)

        # Driver temperature schedule
        if t == 0:
            driver_temp = prev_temp  # hold at start value for first step
        else:
            driver_mean_gfp = np.mean([cell.gfp for cell in driver.cells]) if driver.cells else 0.0
            target_temp = feedback_temperature(
                driver_mean_gfp,
                mode=sim_params.feedback_mode,
                sensitivity=sim_params.feedback_sensitivity,
                base_temp=sim_params.base_temp,
                max_temp=sim_params.max_temp
            )
            driver_temp = prev_temp + inertia * (target_temp - prev_temp)
            prev_temp = driver_temp  # update memory

        # Update populations
        driver.step(current_time, driver_temp, sim_params.time_step)

        # Passives experience the same temperature schedule as the driver (but no coupling)
        for passive in passives:
            passive.step(current_time, driver_temp, sim_params.time_step)

        # CONTROLS are constant every step (no smoothing/feedback)
        control_30.step(current_time, CONTROL_TEMP_30, sim_params.time_step)
        control_39.step(current_time, CONTROL_TEMP_39, sim_params.time_step)

    results = {
        'driver': driver.history,
        'passives': [passive.history for passive in passives],
        'control_30': control_30.history,
        'control_39': control_39.history,
        'parameters': {'simulation': sim_params, 'gfp': gfp_params, 'mode': mode},
        'final_distributions': {
            'driver': driver.get_final_gfp_distribution(),
            'passives': [passive.get_final_gfp_distribution() for passive in passives],
            'control_30': control_30.get_final_gfp_distribution(),
            'control_39': control_39.get_final_gfp_distribution()
        }
    }
    return results

# ===============================
# Analysis & Utilities
# ===============================

def calculate_learning_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    driver_data = results['driver']
    if not driver_data['time']:
        return {}

    # Access burn-in from SimulationParams if present
    burn_in = 0
    try:
        burn_in = int(getattr(results['parameters']['simulation'], 'metric_burn_in', 0))
    except Exception:
        burn_in = 0

    # Basic stats
    final_gfp = driver_data['mean_gfp'][-1]
    final_temp = driver_data['temperature'][-1]
    initial_temp = driver_data['temperature'][0]

    # Learning score: normalized cooling toward base_temp
    denom = max(initial_temp - 30.0, 1e-9)
    learning_score = np.clip((initial_temp - final_temp) / denom, 0.0, 1.0)

    # Adaptation time: time to cross halfway temperature (with burn-in)
    temp_series = np.array(driver_data['temperature'])
    time_series = np.array(driver_data['time'])
    target_temp_mid = initial_temp - 0.5 * (initial_temp - final_temp)
    adaptation_time = None
    for tm, temp in zip(time_series, temp_series):
        if tm < burn_in:
            continue
        if temp <= target_temp_mid:
            adaptation_time = float(tm)
            break

    # Establishment time: first time high-GFP fraction >= 0.5 (with burn-in)
    high_gfp_fractions = np.array(driver_data['high_gfp_fraction'])
    establishment_time = None
    for tm, frac in zip(time_series, high_gfp_fractions):
        if tm < burn_in:
            continue
        if frac >= 0.5:
            establishment_time = float(tm)
            break

    # Temperature stability over final 20% of the run
    final_portion = int(0.8 * len(temp_series))
    final_temps = temp_series[final_portion:] if final_portion < len(temp_series) else temp_series[-1:]
    temp_stability = 1.0 / (1.0 + np.var(final_temps))

    return {
        'learning_score': float(learning_score),
        'final_gfp': float(final_gfp),
        'final_temperature': float(final_temp),
        'adaptation_time': adaptation_time,
        'establishment_time': establishment_time,
        'temperature_stability': float(temp_stability),
        'final_high_gfp_fraction': float(driver_data['high_gfp_fraction'][-1])
    }

def export_results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    driver_data = results['driver']
    df_data = {
        'time': driver_data['time'],
        'driver_gfp': driver_data['mean_gfp'],
        'driver_temp': driver_data['temperature'],
        'driver_high_frac': driver_data['high_gfp_fraction'],
        'driver_pop_size': driver_data['population_size'],
        'control_30_gfp': results['control_30']['mean_gfp'],
        'control_39_gfp': results['control_39']['mean_gfp'],
        # NEW: include control temperatures to verify constancy
        'control_30_temp': results['control_30']['temperature'],
        'control_39_temp': results['control_39']['temperature'],
    }
    for i, passive_data in enumerate(results['passives']):
        df_data[f'passive_{i+1}_gfp'] = passive_data['mean_gfp']
        df_data[f'passive_{i+1}_high_frac'] = passive_data['high_gfp_fraction']
    return pd.DataFrame(df_data)

def create_feedback_function_data(mode: str = "linear", sensitivity: float = 1.0,
                                 n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    gfp_range = np.linspace(0, 100, n_points)
    temp_range = np.array([
        feedback_temperature(gfp, mode, sensitivity, 30.0, 39.0)
        for gfp in gfp_range
    ])
    return gfp_range, temp_range

def validate_parameters(sim_params: SimulationParams, gfp_params: GFPParams) -> List[str]:
    warnings = []
    if sim_params.total_time < 100:
        warnings.append("⚠️ Short simulation time may not show adaptation")
    if sim_params.population_size < 50:
        warnings.append("⚠️ Small population size increases genetic drift")
    if sim_params.feedback_sensitivity < 0.1:
        warnings.append("⚠️ Very low sensitivity may prevent learning")
    if sim_params.feedback_sensitivity > 5.0:
        warnings.append("⚠️ Very high sensitivity may cause instability")
    if gfp_params.cost_strength > 1.0:
        warnings.append("⚠️ High GFP cost may prevent high expression")
    if gfp_params.switch_prob_base > 0.1:
        warnings.append("⚠️ High switching rate may cause noise")
    if not (0.0 < sim_params.temp_inertia <= 1.0):
        warnings.append("⚠️ temp_inertia should be in (0, 1].")
    if sim_params.base_temp != 30.0 or sim_params.max_temp != 39.0:
        warnings.append("ℹ️ Controls are fixed at 30°C/39°C; driver feedback range differs from controls.")
    return warnings

if __name__ == "__main__":
    print("🧬 Evolutionary Learning Simulation - Core Module (Moran)")
    print("=" * 50)
    sim_params = SimulationParams(
        total_time=500,
        population_size=300,
        feedback_mode="linear",
        feedback_sensitivity=1.8,
        temp_inertia=0.25,
        start_at_max_temp=True,
        metric_burn_in=10,
        random_seed=42
    )
    gfp_params = GFPParams(cost_strength=0.3, switch_prob_base=0.01)
    warnings = validate_parameters(sim_params, gfp_params)
    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"  {warning}")
    print("\nRunning quick test simulation (strict Moran BD updates)...")
    results = run_evolution_experiment(sim_params, gfp_params, "binary")
    metrics = calculate_learning_metrics(results)
    print(f"\nTest Results:")
    print(f"  Learning Score: {metrics.get('learning_score', 0):.2f}")
    print(f"  Final GFP: {metrics.get('final_gfp', 0):.1f}")
    print(f"  Final Temperature: {metrics.get('final_temperature', 39):.1f}°C")
    print(f"  Adaptation Time (w/ burn-in): {metrics.get('adaptation_time', None)} min")
    print("\n✅ Core module (Moran) test completed successfully!")
