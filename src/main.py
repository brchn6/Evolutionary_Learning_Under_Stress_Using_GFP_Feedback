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
    inherit_sd: float = 5.0  # Standard deviation for inheritance noise
    
    # Switching probabilities (temperature dependent)
    switch_prob_base: float = 0.01  # per minute at 39°C
    switch_boost_mean: float = 15.0  # mean boost size for switches
    switch_boost_sd: float = 5.0     # standard deviation of boost size
    
    # Fitness cost parameters
    cost_strength: float = 0.3  # How much GFP affects generation time
    cost_exponent: float = 1.5  # Non-linearity of the cost
    
    # Background dynamics
    drift_rate: float = 0.001     # per-minute drift toward baseline
    baseline_gfp: float = 10.0    # natural GFP baseline
    noise_sd: float = 0.5         # per-minute random noise

@dataclass
class SimulationParams:
    """Parameters for overall simulation setup"""
    
    total_time: int = 1000           # Total simulation time in minutes
    population_size: int = 300       # Target population size (Moran process)
    time_step: float = 1.0          # Time step in minutes
    
    # Well configuration
    num_passive_wells: int = 3      # Number of passive wells
    
    # Temperature feedback
    feedback_mode: str = "linear"    # linear, sigmoid, step, exponential
    feedback_sensitivity: float = 1.0  # Sensitivity parameter
    base_temp: float = 30.0         # Minimum temperature
    max_temp: float = 39.0          # Maximum temperature
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None

# ===============================
# Core Biological Models
# ===============================

class Cell:
    """
    Individual cell with GFP expression, age tracking, and fitness calculations.
    
    Represents a single yeast cell in the population with:
    - GFP expression level (binary or continuous)
    - Age and division timing
    - Fitness based on temperature and GFP cost
    - Stochastic division and phenotype switching
    """
    
    def __init__(self, gfp_value: float, birth_time: float, mode: str = "continuous"):
        self.gfp = float(np.clip(gfp_value, 0.0, 100.0))
        self.birth_time = birth_time
        self.age = 0.0
        self.mode = mode
        self.generation_time = 120.0  # Will be updated based on conditions
        self._last_temp = 39.0  # Cache for efficiency
        
    def _calculate_base_generation_time(self, temperature: float) -> float:
        """Calculate base generation time from temperature (before GFP cost)"""
        # Biological relationship: cooler = faster growth (within range)
        # 30°C -> 60 min, 39°C -> 180 min (exponential-like relationship)
        temp_normalized = np.clip((temperature - 30.0) / 9.0, 0.0, 1.0)
        base_time = 60.0 + 120.0 * (temp_normalized ** 2)
        return base_time
    
    def _calculate_gfp_cost_multiplier(self, params: GFPParams) -> float:
        """Calculate fitness cost multiplier from GFP expression"""
        if self.mode == "binary":
            # Simple binary cost
            return 1.3 if self.gfp > 50 else 1.0
        else:
            # Continuous cost with configurable strength and curvature
            normalized_gfp = self.gfp / 100.0
            cost_factor = params.cost_strength * (normalized_gfp ** params.cost_exponent)
            return 1.0 + cost_factor
    
    def update_fitness(self, temperature: float, params: GFPParams):
        """Update generation time based on current temperature and GFP cost"""
        base_time = self._calculate_base_generation_time(temperature)
        cost_multiplier = self._calculate_gfp_cost_multiplier(params)
        self.generation_time = base_time * cost_multiplier
        self._last_temp = temperature
    
    def can_divide(self, dt: float = 1.0) -> bool:
        """
        Stochastic division check using exponential hazard model.
        Probability increases with age relative to generation time.
        """
        if self.age <= 0:
            return False
        
        # Hazard rate increases with age/generation_time ratio
        hazard_rate = 0.02  # Base hazard per minute
        relative_age = self.age / max(self.generation_time, 1.0)
        prob = 1.0 - np.exp(-hazard_rate * relative_age * dt)
        
        return np.random.random() < prob
    
    def divide(self, current_time: float, params: GFPParams) -> 'Cell':
        """
        Create daughter cell with inheritance and reset mother's age.
        
        Returns:
            Cell: New daughter cell with inherited GFP (with noise)
        """
        if self.mode == "binary":
            # Binary inheritance with occasional switching
            switch_prob = 0.05  # 5% chance to switch state during division
            if np.random.random() < switch_prob:
                daughter_gfp = params.high_threshold if self.gfp < 50 else params.low_threshold
            else:
                daughter_gfp = self.gfp
        else:
            # Continuous inheritance with Gaussian noise
            noise = np.random.normal(0, params.inherit_sd)
            daughter_gfp = np.clip(self.gfp + noise, params.min_val, params.max_val)
        
        # Reset mother's age (she just divided)
        self.age = 0.0
        
        # Create daughter
        daughter = Cell(daughter_gfp, current_time, self.mode)
        daughter.generation_time = self.generation_time  # Inherit current fitness state
        
        return daughter
    
    def phenotype_switch(self, temperature: float, params: GFPParams, dt: float = 1.0) -> bool:
        """
        Temperature-dependent phenotype switching (stress response).
        Higher temperature increases probability of switching to high GFP.
        
        Returns:
            bool: True if switching occurred
        """
        # Temperature scaling: 0 at 30°C, 1 at 39°C
        temp_factor = np.clip((temperature - 30.0) / 9.0, 0.0, 1.0)
        switch_prob = params.switch_prob_base * temp_factor * dt
        
        if np.random.random() < switch_prob:
            if self.mode == "binary":
                # Binary: switch to opposite state
                self.gfp = params.high_threshold if self.gfp < 50 else params.low_threshold
            else:
                # Continuous: boost GFP with some noise
                boost = np.random.normal(params.switch_boost_mean, params.switch_boost_sd)
                self.gfp = np.clip(self.gfp + boost, params.min_val, params.max_val)
            
            return True
        
        return False
    
    def background_dynamics(self, params: GFPParams, dt: float = 1.0):
        """
        Apply background GFP dynamics (drift toward baseline + noise).
        This represents slow metabolic processes between divisions.
        """
        if self.mode == "continuous":
            # Drift toward baseline
            drift = -params.drift_rate * (self.gfp - params.baseline_gfp) * dt
            
            # Random noise
            noise = np.random.normal(0, params.noise_sd * np.sqrt(dt))
            
            # Apply changes
            self.gfp = np.clip(self.gfp + drift + noise, params.min_val, params.max_val)

# ===============================
# Temperature Feedback Functions
# ===============================

def feedback_temperature(mean_gfp: float, mode: str = "linear", 
                        sensitivity: float = 1.0, base_temp: float = 30.0, 
                        max_temp: float = 39.0) -> float:
    """
    Calculate environmental temperature based on population mean GFP.
    This is the core feedback mechanism that allows "learning".
    
    Args:
        mean_gfp: Population mean GFP (0-100)
        mode: Feedback function type
        sensitivity: How responsive the system is to GFP changes
        base_temp: Minimum temperature (reward)
        max_temp: Maximum temperature (punishment)
    
    Returns:
        float: Temperature in Celsius
    """
    # Normalize GFP to 0-1 scale
    gfp_norm = np.clip(mean_gfp / 100.0, 0.0, 1.0)
    
    # Apply sensitivity scaling
    x = gfp_norm * sensitivity
    x = np.clip(x, 0.0, 1.0)
    
    # Calculate cooling factor based on mode
    if mode == "linear":
        cooling_factor = x
    elif mode == "sigmoid":
        # S-curve for more realistic biological response
        k = 5.0  # Steepness parameter
        cooling_factor = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
    elif mode == "step":
        # Step function at 50% threshold
        threshold = 0.5
        cooling_factor = 1.0 if x >= threshold else 0.0
    elif mode == "exponential":
        # Exponential approach to maximum cooling
        cooling_factor = 1.0 - np.exp(-3.0 * x)
    else:
        # Default to linear
        cooling_factor = x
    
    # Convert cooling factor to temperature
    temperature = max_temp - cooling_factor * (max_temp - base_temp)
    return float(np.clip(temperature, base_temp, max_temp))

# ===============================
# Population Simulation
# ===============================

class Population:
    """
    Population of cells implementing Moran process dynamics.
    
    Maintains constant population size through birth-death balance.
    Tracks population statistics and evolutionary dynamics.
    """
    
    def __init__(self, target_size: int, mode: str, params: GFPParams, 
                 label: str = "population"):
        self.target_size = target_size
        self.mode = mode
        self.params = params
        self.label = label
        self.cells: List[Cell] = []
        
        # History tracking
        self.history = {
            'time': [],
            'mean_gfp': [],
            'std_gfp': [],
            'population_size': [],
            'temperature': [],
            'high_gfp_fraction': [],
            'mean_generation_time': [],
            'births': [],
            'deaths': [],
            'switches': []
        }
        
        # Initialize population with low GFP
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with low GFP cells"""
        for _ in range(self.target_size):
            if self.mode == "binary":
                initial_gfp = self.params.low_threshold
            else:
                # Start with low GFP plus some noise
                initial_gfp = max(0, np.random.normal(self.params.baseline_gfp, 3.0))
            
            cell = Cell(initial_gfp, 0.0, self.mode)
            self.cells.append(cell)
    
    def _calculate_population_stats(self, temperature: float) -> Dict[str, float]:
        """Calculate current population statistics"""
        if not self.cells:
            return {
                'mean_gfp': 0.0,
                'std_gfp': 0.0,
                'high_gfp_fraction': 0.0,
                'mean_generation_time': 0.0
            }
        
        gfp_values = [cell.gfp for cell in self.cells]
        
        # Basic statistics
        mean_gfp = np.mean(gfp_values)
        std_gfp = np.std(gfp_values)
        mean_gen_time = np.mean([cell.generation_time for cell in self.cells])
        
        # High GFP fraction (threshold depends on mode)
        if self.mode == "binary":
            high_threshold = 50.0
        else:
            high_threshold = 60.0
        
        high_gfp_fraction = sum(1 for gfp in gfp_values if gfp > high_threshold) / len(gfp_values)
        
        return {
            'mean_gfp': mean_gfp,
            'std_gfp': std_gfp,
            'high_gfp_fraction': high_gfp_fraction,
            'mean_generation_time': mean_gen_time
        }
    
    def step(self, current_time: float, temperature: float, dt: float = 1.0) -> Dict[str, int]:
        """
        Execute one time step of population dynamics.
        
        Args:
            current_time: Current simulation time
            temperature: Environmental temperature
            dt: Time step size
        
        Returns:
            Dict with counts of births, deaths, switches
        """
        if not self.cells:
            return {'births': 0, 'deaths': 0, 'switches': 0}
        
        births = 0
        deaths = 0
        switches = 0
        
        # Update all cells
        for cell in self.cells:
            cell.age += dt
            cell.update_fitness(temperature, self.params)
            
            # Background dynamics (continuous mode only)
            if self.mode == "continuous":
                cell.background_dynamics(self.params, dt)
            
            # Phenotype switching (stress response)
            if cell.phenotype_switch(temperature, self.params, dt):
                switches += 1
        
        # Moran process: births and deaths
        if len(self.cells) > 0:
            # Calculate fitness-proportional reproduction probabilities
            fitness_weights = [1.0 / max(cell.generation_time, 1.0) for cell in self.cells]
            total_fitness = sum(fitness_weights)
            
            if total_fitness > 0:
                # Attempt divisions
                cells_to_add = []
                
                for cell in self.cells:
                    if cell.can_divide(dt):
                        # Create daughter
                        daughter = cell.divide(current_time, self.params)
                        cells_to_add.append(daughter)
                        births += 1
                
                # Add daughters and maintain population size (Moran replacement)
                for daughter in cells_to_add:
                    if len(self.cells) >= self.target_size:
                        # Remove random cell (death)
                        victim_idx = np.random.randint(len(self.cells))
                        self.cells.pop(victim_idx)
                        deaths += 1
                    
                    self.cells.append(daughter)
        
        # Calculate and record population statistics
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
        """Get final GFP values for all cells"""
        return [cell.gfp for cell in self.cells]

# ===============================
# Experiment Runner
# ===============================

def run_evolution_experiment(sim_params: SimulationParams, 
                           gfp_params: GFPParams, 
                           mode: str = "continuous") -> Dict[str, Any]:
    """
    Run complete evolution experiment with driver, passive, and control wells.
    
    Args:
        sim_params: Simulation configuration
        gfp_params: GFP dynamics parameters
        mode: "binary" or "continuous"
    
    Returns:
        Dictionary containing results from all wells
    """
    # Set random seed if specified
    if sim_params.random_seed is not None:
        np.random.seed(sim_params.random_seed)
    
    # Initialize populations
    driver = Population(sim_params.population_size, mode, gfp_params, "driver")
    
    passives = [
        Population(sim_params.population_size, mode, gfp_params, f"passive_{i+1}")
        for i in range(sim_params.num_passive_wells)
    ]
    
    control_30 = Population(sim_params.population_size, mode, gfp_params, "control_30")
    control_39 = Population(sim_params.population_size, mode, gfp_params, "control_39")
    
    # Simulation loop
    for t in range(0, sim_params.total_time, int(sim_params.time_step)):
        current_time = float(t)
        
        # Driver well determines temperature based on its GFP
        driver_mean_gfp = np.mean([cell.gfp for cell in driver.cells]) if driver.cells else 0.0
        driver_temp = feedback_temperature(
            driver_mean_gfp,
            mode=sim_params.feedback_mode,
            sensitivity=sim_params.feedback_sensitivity,
            base_temp=sim_params.base_temp,
            max_temp=sim_params.max_temp
        )
        
        # Update all populations
        driver.step(current_time, driver_temp, sim_params.time_step)
        
        # Passive wells follow driver temperature
        for passive in passives:
            passive.step(current_time, driver_temp, sim_params.time_step)
        
        # Control wells at fixed temperatures
        control_30.step(current_time, 30.0, sim_params.time_step)
        control_39.step(current_time, 39.0, sim_params.time_step)
    
    # Package results
    results = {
        'driver': driver.history,
        'passives': [passive.history for passive in passives],
        'control_30': control_30.history,
        'control_39': control_39.history,
        'parameters': {
            'simulation': sim_params,
            'gfp': gfp_params,
            'mode': mode
        },
        'final_distributions': {
            'driver': driver.get_final_gfp_distribution(),
            'passives': [passive.get_final_gfp_distribution() for passive in passives],
            'control_30': control_30.get_final_gfp_distribution(),
            'control_39': control_39.get_final_gfp_distribution()
        }
    }
    
    return results

# ===============================
# Analysis Functions
# ===============================

def calculate_learning_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate key learning and adaptation metrics from simulation results.
    
    Args:
        results: Results dictionary from run_evolution_experiment
    
    Returns:
        Dictionary of calculated metrics
    """
    driver_data = results['driver']
    
    if not driver_data['time']:
        return {}
    
    # Final states
    final_gfp = driver_data['mean_gfp'][-1]
    final_temp = driver_data['temperature'][-1]
    initial_temp = driver_data['temperature'][0]
    
    # Learning score (0 = no learning, 1 = perfect learning)
    learning_score = (initial_temp - final_temp) / (initial_temp - 30.0)
    learning_score = np.clip(learning_score, 0.0, 1.0)
    
    # Adaptation rate (time to reach 50% of final adaptation)
    temp_change = np.array(driver_data['temperature'])
    target_temp = initial_temp - 0.5 * (initial_temp - final_temp)
    adaptation_time = None
    
    for i, temp in enumerate(temp_change):
        if temp <= target_temp:
            adaptation_time = driver_data['time'][i]
            break
    
    # High GFP establishment time
    high_gfp_fractions = np.array(driver_data['high_gfp_fraction'])
    establishment_time = None
    
    for i, frac in enumerate(high_gfp_fractions):
        if frac >= 0.5:  # 50% of population has high GFP
            establishment_time = driver_data['time'][i]
            break
    
    # Temperature stability (variance in final 20% of simulation)
    final_portion = int(0.8 * len(driver_data['temperature']))
    final_temps = driver_data['temperature'][final_portion:]
    temp_stability = 1.0 / (1.0 + np.var(final_temps))  # Higher = more stable
    
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
    """
    Convert simulation results to a pandas DataFrame for analysis.
    
    Args:
        results: Results dictionary from run_evolution_experiment
    
    Returns:
        DataFrame with time series data from all wells
    """
    driver_data = results['driver']
    
    # Start with driver data
    df_data = {
        'time': driver_data['time'],
        'driver_gfp': driver_data['mean_gfp'],
        'driver_temp': driver_data['temperature'],
        'driver_high_frac': driver_data['high_gfp_fraction'],
        'driver_pop_size': driver_data['population_size'],
        'control_30_gfp': results['control_30']['mean_gfp'],
        'control_39_gfp': results['control_39']['mean_gfp']
    }
    
    # Add passive wells
    for i, passive_data in enumerate(results['passives']):
        df_data[f'passive_{i+1}_gfp'] = passive_data['mean_gfp']
        df_data[f'passive_{i+1}_high_frac'] = passive_data['high_gfp_fraction']
    
    return pd.DataFrame(df_data)

# ===============================
# Utility Functions
# ===============================

def create_feedback_function_data(mode: str = "linear", sensitivity: float = 1.0, 
                                 n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data points for plotting feedback functions.
    
    Args:
        mode: Feedback function type
        sensitivity: Sensitivity parameter
        n_points: Number of points to generate
    
    Returns:
        Tuple of (gfp_values, temperature_values)
    """
    gfp_range = np.linspace(0, 100, n_points)
    temp_range = np.array([
        feedback_temperature(gfp, mode, sensitivity, 30.0, 39.0) 
        for gfp in gfp_range
    ])
    
    return gfp_range, temp_range

def validate_parameters(sim_params: SimulationParams, gfp_params: GFPParams) -> List[str]:
    """
    Validate simulation parameters and return list of warnings/errors.
    
    Args:
        sim_params: Simulation parameters to validate
        gfp_params: GFP parameters to validate
    
    Returns:
        List of validation messages
    """
    warnings = []
    
    # Check simulation parameters
    if sim_params.total_time < 100:
        warnings.append("⚠️ Short simulation time may not show adaptation")
    
    if sim_params.population_size < 50:
        warnings.append("⚠️ Small population size increases genetic drift")
    
    if sim_params.feedback_sensitivity < 0.1:
        warnings.append("⚠️ Very low sensitivity may prevent learning")
    
    if sim_params.feedback_sensitivity > 5.0:
        warnings.append("⚠️ Very high sensitivity may cause instability")
    
    # Check GFP parameters
    if gfp_params.cost_strength > 1.0:
        warnings.append("⚠️ High GFP cost may prevent high expression")
    
    if gfp_params.switch_prob_base > 0.1:
        warnings.append("⚠️ High switching rate may cause noise")
    
    return warnings

# ===============================
# Main Entry Point for Testing
# ===============================

if __name__ == "__main__":
    # Example usage and testing
    print("🧬 Evolutionary Learning Simulation - Core Module")
    print("=" * 50)
    
    # Create default parameters
    sim_params = SimulationParams(
        total_time=500,
        population_size=300,
        feedback_mode="linear",
        feedback_sensitivity=1.0
    )
    
    gfp_params = GFPParams(
        cost_strength=0.3,
        switch_prob_base=0.01
    )
    
    # Validate parameters
    warnings = validate_parameters(sim_params, gfp_params)
    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    # Run quick test
    print("\nRunning quick test simulation...")
    results = run_evolution_experiment(sim_params, gfp_params, "continuous")
    
    # Calculate metrics
    metrics = calculate_learning_metrics(results)
    print(f"\nTest Results:")
    print(f"  Learning Score: {metrics.get('learning_score', 0):.2f}")
    print(f"  Final GFP: {metrics.get('final_gfp', 0):.1f}")
    print(f"  Final Temperature: {metrics.get('final_temperature', 39):.1f}°C")
    
    print("\n✅ Core module test completed successfully!")