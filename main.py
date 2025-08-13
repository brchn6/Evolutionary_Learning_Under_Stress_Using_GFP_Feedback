import numpy as np
import pandas as pd

# ===============================
# Helpers (biology-informed logic)
# ===============================

def gen_time_func(temp):
    """
    Map temperature (°C) to expected generation time (min).
    Faster growth at lower temp (30°C ~ 20 min), very slow at 39°C (~180 min).
    """
    temp_points = [30, 33, 36, 37, 38, 39]
    gen_times   = [20, 40, 80, 120, 150, 180]
    return np.interp(temp, temp_points, gen_times)

def feedback_temperature(mean_gfp, mode="linear", base_temp=30, max_temp=39, sensitivity=1.0):
    """
    Map mean GFP (0–100) -> temperature (°C).
    sensitivity now affects ALL modes:
      - linear/inverse: scales slope (>=1 steeper, <1 flatter), clamped [0,1]
      - step: shifts threshold (higher sensitivity -> earlier step)
      - exp/sigmoid: unchanged semantics (sensitivity already controls curvature/steepness)
    """
    # normalize to [0,1]
    x = np.clip(mean_gfp / 100.0, 0.0, 1.0)
    s = max(1e-6, float(sensitivity))  # avoid zero/negatives

    if mode == "linear":
        # slope scaled by sensitivity; clamp so it can’t overshoot base_temp
        y = np.clip(s * x, 0.0, 1.0)
        return max_temp - y * (max_temp - base_temp)

    elif mode == "inverse":
        # punishment (hotter with higher GFP); slope scaled by sensitivity
        y = np.clip(s * x, 0.0, 1.0)
        return base_temp + y * (max_temp - base_temp)

    elif mode == "step":
        # move the step threshold with sensitivity:
        # s=1  -> threshold at 50 (x=0.5), as before
        # s>1  -> earlier step (lower threshold), s<1 -> later step (higher threshold)
        threshold = np.clip(0.5 / s, 0.0, 1.0)
        return base_temp if x >= threshold else max_temp

    elif mode == "exp":
        # exponential relief; larger s -> faster drop with GFP
        # note: using normalized x keeps ranges sane
        return base_temp + np.exp(-s * 5.0 * x) * (max_temp - base_temp)

    elif mode == "sigmoid":
        # logistic centered near x=0.5; s controls steepness
        k = 10.0 * s
        c = 0.5
        return base_temp + (max_temp - base_temp) / (1.0 + np.exp(-k * (x - c)))

    else:
        raise ValueError("Unknown feedback mode")

def stress_scaled_boost_prob(temp, base_prob=0.002):
    """
    Rare stress-induced 'boost' chance per minute.
    Scales from 0 at 30°C up to ~base_prob at 39°C.
    """
    return base_prob * max(0.0, (temp - 30.0) / 9.0)

def time_to_stationarity(times, temps, window=60, tol=0.1):
    """
    First time at which temperature variation in the past 'window' minutes <= tol (°C).
    Returns None if never attained.
    """
    if len(times) < window + 1:
        return None
    for i in range(window, len(times)):
        segment = temps[i-window:i+1]
        if (max(segment) - min(segment)) <= tol:
            return times[i]
    return None

# ===============================
# Cell class
# ===============================

class Cell:
    def __init__(self, temp, time_of_birth, mother_div_time, gfp):
        # State
        self.time_of_birth = time_of_birth
        self.mother_division_time = mother_div_time  # minutes since mother's division
        self.time_since_division = 0.0               # minutes counter
        # Traits
        self.generation_time = gen_time_func(temp)   # minutes
        self.gfp = float(gfp)                        # 0..100

    def update_generation_time(self, temp):
        self.generation_time = gen_time_func(temp)

    def division_event(self, scale=0.015):
        """
        Stochastic division hazard; probability rises with waiting time relative to gen time.
        Returns: (divides: bool)
        """
        p_div = 1.0 - np.exp(-scale * (self.time_since_division / max(1e-9, self.generation_time)))
        return np.random.rand() < p_div

# ===============================
# Simulation core (Driver / Controls)
# ===============================

def run_simulation(
    total_time=1000,
    time_step=1,                 # minutes
    feedback_mode="linear",
    max_population=1000,
    moran_after_cap=True,        # <<< KEY: switch to Moran birth–death once cap is reached
    inheritance_noise=2.0,       # stddev of daughter GFP around mother
    base_boost_prob=0.002,       # per-minute rare boost at 39C scale (stress scaled)
    feedback_sensitivity=1.0,
    control_mode=None,           # None, "fixed39", "fixed30"
    start_temp=39,
    high_gfp_threshold=20.0,     # for "first high-GFP" event detection
    stationarity_window=60,
    stationarity_tol=0.1
):
    """
    Returns:
      df_divisions: per-division dataframe (daughter snapshots)
      temp_hist, temp_times
      mean_gfp_hist, mean_gen_hist, times
      pop_size_hist
      first_high_gfp_time (min) -- first time ANY cell gfp >= threshold
      stationarity_time (min)   -- time when temperature stabilized (window/tol criterion)
      final_population_gfps     -- list of GFPs of cells alive at end
    """
    # --- initialize population ---
    population = []
    initial_gfp = float(np.clip(np.random.normal(0.0, 2.0), 0.0, 5.0))  # near-zero starts
    population.append(Cell(start_temp, 0.0, 0.0, initial_gfp))

    # --- histories ---
    temp_hist, temp_times = [], []
    mean_gfp_hist, mean_gen_hist, times = [], [], []
    pop_size_hist = []
    division_records = []

    # --- event flags/metrics ---
    first_high_gfp_time = None

    # --- time loop ---
    current_time = 0
    base_temp, max_temp = 30.0, 39.0

    while current_time < total_time:
        # Environment: immediate feedback each minute
        mean_gfp = np.mean([c.gfp for c in population]) if population else 0.0

        if control_mode == "fixed39":
            temperature = 39.0
        elif control_mode == "fixed30":
            temperature = 30.0
        else:
            temperature = feedback_temperature(
                mean_gfp, mode=feedback_mode, base_temp=base_temp, max_temp=max_temp, sensitivity=feedback_sensitivity
            )

        # Record histories
        temp_hist.append(temperature)
        temp_times.append(current_time)
        mean_gfp_hist.append(mean_gfp)
        mean_gen_hist.append(np.mean([c.generation_time for c in population]) if population else np.nan)
        times.append(current_time)
        pop_size_hist.append(len(population))

        # First high-GFP event
        if first_high_gfp_time is None and any(c.gfp >= high_gfp_threshold for c in population):
            first_high_gfp_time = current_time

        # --- per-cell updates ---
        for cell in list(population):  # iterate over a snapshot of the current population
            # Update generation time to current temperature
            cell.update_generation_time(temperature)

            # Stress-induced rare "boost" upward (more likely at hotter temps)
            prob = stress_scaled_boost_prob(temperature, base_prob=base_boost_prob)
            if np.random.rand() < prob:
                cell.gfp = float(np.clip(cell.gfp + np.random.normal(10.0, 3.0), 0.0, 100.0))

            # Division attempt?
            if cell.division_event():
                div_time = cell.time_since_division
                cell.time_since_division = 0.0

                # Daughter inherits with noise (can go up or down)
                daughter_gfp = float(np.clip(cell.gfp + np.random.normal(0.0, max(1e-9, inheritance_noise)), 0.0, 100.0))
                daughter = Cell(temperature, time_of_birth=current_time, mother_div_time=div_time, gfp=daughter_gfp)

                if (not moran_after_cap) or (len(population) < max_population):
                    # GROWTH PHASE: append until we reach the cap
                    population.append(daughter)
                else:
                    # MORAN PHASE: one-in/one-out replacement at constant size
                    victim_idx = np.random.randint(len(population))  # uniform death
                    population[victim_idx] = daughter

                # Record the division event (daughter snapshot)
                division_records.append({
                    "Time": current_time,
                    "Temperature": temperature,
                    "Mother_Generation_Time": cell.generation_time,
                    "Mother_Division_Time": div_time,
                    "Daughter_GFP": daughter.gfp
                })

            # Aging
            cell.time_since_division += time_step

        # (Important) If moran_after_cap=False, we can still hard-cap by random cull
        if not moran_after_cap:
            while len(population) > max_population:
                population.pop(np.random.randint(len(population)))

        current_time += time_step

    # Stationarity detection (on temperature series)
    stationarity_time = time_to_stationarity(temp_times, temp_hist, window=stationarity_window, tol=stationarity_tol)

    # Final population snapshot
    final_population_gfps = [c.gfp for c in population]

    df = pd.DataFrame(division_records)
    return (
        df,
        temp_hist, temp_times,
        mean_gfp_hist, mean_gen_hist, times,
        pop_size_hist,
        first_high_gfp_time,
        stationarity_time,
        final_population_gfps
    )

# ===============================
# Passive wells (follow external temperature)
# ===============================

def run_simulation_with_external_temperature(
    external_temp_times,
    external_temp_values,
    total_time=1000,
    time_step=1,
    max_population=1000,
    moran_after_cap=True,      # same Moran behavior after cap
    inheritance_noise=2.0,
    base_boost_prob=0.002,
    high_gfp_threshold=20.0,
    stationarity_window=60,
    stationarity_tol=0.1
):
    """
    Passive wells: follow an external temperature trace (the driver's), no feedback of their own.
    Returns analogous outputs to run_simulation (stationarity measured on their own temp trace).
    """
    population = []
    initial_gfp = float(np.clip(np.random.normal(0.0, 2.0), 0.0, 5.0))
    population.append(Cell(39.0, 0.0, 0.0, initial_gfp))

    temp_hist, temp_times = [], []
    mean_gfp_hist, mean_gen_hist, times = [], [], []
    pop_size_hist = []
    division_records = []

    first_high_gfp_time = None

    current_time = 0
    while current_time < total_time:
        temperature = float(np.interp(current_time, external_temp_times, external_temp_values))
        temp_hist.append(temperature)
        temp_times.append(current_time)

        mean_gfp = np.mean([c.gfp for c in population]) if population else 0.0
        mean_gfp_hist.append(mean_gfp)
        mean_gen_hist.append(np.mean([c.generation_time for c in population]) if population else np.nan)
        times.append(current_time)
        pop_size_hist.append(len(population))

        if first_high_gfp_time is None and any(c.gfp >= high_gfp_threshold for c in population):
            first_high_gfp_time = current_time

        for cell in list(population):
            cell.update_generation_time(temperature)

            prob = stress_scaled_boost_prob(temperature, base_prob=base_boost_prob)
            if np.random.rand() < prob:
                cell.gfp = float(np.clip(cell.gfp + np.random.normal(10.0, 3.0), 0.0, 100.0))

            if cell.division_event():
                div_time = cell.time_since_division
                cell.time_since_division = 0.0
                daughter_gfp = float(np.clip(cell.gfp + np.random.normal(0.0, max(1e-9, inheritance_noise)), 0.0, 100.0))
                daughter = Cell(temperature, time_of_birth=current_time, mother_div_time=div_time, gfp=daughter_gfp)

                if (not moran_after_cap) or (len(population) < max_population):
                    population.append(daughter)
                else:
                    victim_idx = np.random.randint(len(population))
                    population[victim_idx] = daughter

                division_records.append({
                    "Time": current_time,
                    "Temperature": temperature,
                    "Mother_Generation_Time": cell.generation_time,
                    "Mother_Division_Time": div_time,
                    "Daughter_GFP": daughter.gfp
                })

            cell.time_since_division += time_step

        if not moran_after_cap:
            while len(population) > max_population:
                population.pop(np.random.randint(len(population)))

        current_time += time_step

    # For passives, we can still compute stationarity on their temp trace if useful:
    stationarity_time = time_to_stationarity(temp_times, temp_hist, window=stationarity_window, tol=stationarity_tol)
    final_population_gfps = [c.gfp for c in population]
    df = pd.DataFrame(division_records)

    return (
        df,
        temp_hist, temp_times,
        mean_gfp_hist, mean_gen_hist, times,
        pop_size_hist,
        first_high_gfp_time,
        stationarity_time,
        final_population_gfps
    )
