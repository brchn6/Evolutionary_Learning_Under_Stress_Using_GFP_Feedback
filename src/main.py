"""
Evolutionary Learning Simulation - Core Logic
===========================================

This module contains the core biological models and simulation logic for studying
temperature-feedback driven adaptation in synthetic yeast populations.

Key Components:
- GFP expression models (binary and continuous)
- Cell class with fitness and division dynamics
- Population class with Moran process
- Temperature feedback functions
- Evolution experiment runner
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json
import logging

# ===============================
# Configuration Classes
# ===============================

@dataclass
class GFPParams:
    """Parameters for GFP expression dynamics and cellular processes"""
    # Binary mode thresholds
    low_threshold: float = 25.0
    high_threshold: float = 75.0
    # Continuous mode bounds
    min_val: float = 0.0
    max_val: float = 100.0
    # Inheritance parameters
    inherit_sd: float = 5.0
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
    """Parameters for overall simulation setup"""
    total_time: int = 1000
    population_size: int = 300
    time_step: float = 1.0
    # Well configuration
    num_passive_wells: int = 3
    # Temperature feedback
    feedback_mode: str = "linear"
    feedback_sensitivity: float = 1.0
    base_temp: float = 30.0
    max_temp: float = 39.0
    # Random seed for reproducibility
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

    def can_divide(self, dt: float = 1.0) -> bool:
        if self.age <= 0:
            return False
        hazard_rate = 0.02
        relative_age = self.age / max(self.generation_time, 1.0)
        prob = 1.0 - np.exp(-hazard_rate * relative_age * dt)
        return np.random.random() < prob

    def divide(self, current_time: float, params: GFPParams) -> 'Cell':
        if self.mode == "binary":
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
        daughter.generation_time = self.generation_time
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
        k = 5.0
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
# Population Simulation
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
                # Keep the population in a 'low' state but don't force temp via GFP
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
        if not self.cells:
            return {'births': 0, 'deaths': 0, 'switches': 0}
        births = deaths = switches = 0
        for cell in self.cells:
            cell.age += dt
            cell.update_fitness(temperature, self.params)
            if self.mode == "continuous":
                cell.background_dynamics(self.params, dt)
            if cell.phenotype_switch(temperature, self.params, dt):
                switches += 1
        if len(self.cells) > 0:
            fitness_weights = [1.0 / max(cell.generation_time, 1.0) for cell in self.cells]
            total_fitness = sum(fitness_weights)
            if total_fitness > 0:
                cells_to_add = []
                for cell in self.cells:
                    if cell.can_divide(dt):
                        daughter = cell.divide(current_time, self.params)
                        cells_to_add.append(daughter)
                        births += 1
                for daughter in cells_to_add:
                    if len(self.cells) >= self.target_size:
                        victim_idx = np.random.randint(len(self.cells))
                        self.cells.pop(victim_idx)
                        deaths += 1
                    self.cells.append(daughter)
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
# Experiment Runner
# ===============================

def run_evolution_experiment(sim_params: SimulationParams,
                             gfp_params: GFPParams,
                             mode: str = "continuous") -> Dict[str, Any]:
    """
    Run complete evolution experiment with driver, passive, and control wells.
    """
    if sim_params.random_seed is not None:
        np.random.seed(sim_params.random_seed)

    driver = Population(sim_params.population_size, mode, gfp_params, "driver")
    passives = [Population(sim_params.population_size, mode, gfp_params, f"passive_{i+1}")
                for i in range(sim_params.num_passive_wells)]
    control_30 = Population(sim_params.population_size, mode, gfp_params, "control_30")
    control_39 = Population(sim_params.population_size, mode, gfp_params, "control_39")

    for t in range(0, sim_params.total_time, int(sim_params.time_step)):
        current_time = float(t)

        # NEW: Force the simulation to *start* at the maximum temperature (e.g., 39°C)
        if t == 0:
            driver_temp = sim_params.max_temp  # ensures time 0 is 39°C regardless of mode/thresholds
        else:
            # After t=0, temperature is governed by the feedback function (and thus can only cool)
            driver_mean_gfp = np.mean([cell.gfp for cell in driver.cells]) if driver.cells else 0.0
            driver_temp = feedback_temperature(
                driver_mean_gfp,
                mode=sim_params.feedback_mode,
                sensitivity=sim_params.feedback_sensitivity,
                base_temp=sim_params.base_temp,
                max_temp=sim_params.max_temp
            )

        # Update populations
        driver.step(current_time, driver_temp, sim_params.time_step)
        for passive in passives:
            passive.step(current_time, driver_temp, sim_params.time_step)
        control_30.step(current_time, 30.0, sim_params.time_step)
        control_39.step(current_time, 39.0, sim_params.time_step)

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
# Analysis & Utilities (unchanged except minor formatting)
# ===============================

def calculate_learning_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    driver_data = results['driver']
    if not driver_data['time']:
        return {}
    final_gfp = driver_data['mean_gfp'][-1]
    final_temp = driver_data['temperature'][-1]
    initial_temp = driver_data['temperature'][0]
    learning_score = (initial_temp - final_temp) / (initial_temp - 30.0)
    learning_score = np.clip(learning_score, 0.0, 1.0)
    temp_change = np.array(driver_data['temperature'])
    target_temp = initial_temp - 0.5 * (initial_temp - final_temp)
    adaptation_time = None
    for i, temp in enumerate(temp_change):
        if temp <= target_temp:
            adaptation_time = driver_data['time'][i]
            break
    high_gfp_fractions = np.array(driver_data['high_gfp_fraction'])
    establishment_time = None
    for i, frac in enumerate(high_gfp_fractions):
        if frac >= 0.5:
            establishment_time = driver_data['time'][i]
            break
    final_portion = int(0.8 * len(driver_data['temperature']))
    final_temps = driver_data['temperature'][final_portion:]
    temp_stability = 1.0 / (1.0 + np.var(final_temps))
    return {
        'learning_score': learning_score,
        'final_gfp': final_gfp,
        'final_temperature': final_temp,
        'adaptation_time': adaptation_time,
        'establishment_time': establishment_time,
        'temperature_stability': temp_stability,
        'final_high_gfp_fraction': driver_data['high_gfp_fraction'][-1]
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
        'control_39_gfp': results['control_39']['mean_gfp']
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
    return warnings

if __name__ == "__main__":
    print("🧬 Evolutionary Learning Simulation - Core Module")
    print("=" * 50)
    sim_params = SimulationParams(total_time=500, population_size=300,
                                  feedback_mode="linear", feedback_sensitivity=1.0)
    gfp_params = GFPParams(cost_strength=0.3, switch_prob_base=0.01)
    warnings = validate_parameters(sim_params, gfp_params)
    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"  {warning}")
    print("\nRunning quick test simulation...")
    results = run_evolution_experiment(sim_params, gfp_params, "continuous")
    metrics = calculate_learning_metrics(results)
    print(f"\nTest Results:")
    print(f"  Learning Score: {metrics.get('learning_score', 0):.2f}")
    print(f"  Final GFP: {metrics.get('final_gfp', 0):.1f}")
    print(f"  Final Temperature: {metrics.get('final_temperature', 39):.1f}°C")
    print("\n✅ Core module test completed successfully!")
