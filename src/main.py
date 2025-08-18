import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging, sys, json

# ===============================
# Logging setup (stdout)
# ===============================
logger = logging.getLogger("gfp_sim")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))  # JSON lines
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ===============================
# GFP — formal "process" model
# ===============================

@dataclass(frozen=True)
class GFPParams:
    # hard bounds
    min_val: float = 0.0
    max_val: float = 100.0

    # inheritance (daughter around mother at division)
    inherit_sd: float = 2.0

    # temperature-scaled rare "boost" process
    switch_prob_39C: float = 0.002   # per-minute probability at 39°C
    boost_mu: float = 10.0           # mean boost size if an event happens
    boost_sd: float = 3.0            # SD of boost size

    # gentle background dynamics between divisions (OU-like)
    baseline: float = 0.0            # reversion target
    decay_rate: float = 0.002        # per-minute reversion toward baseline
    drift_sd_per_min: float = 0.05   # Gaussian jitter per minute

    # metabolic burden (fitness cost)
    cost_strength: float = 0.5       # “how much GFP slows division”
    cost_gamma: float = 1.0          # curvature of the cost


class GFPModel:
    def __init__(self, params: GFPParams):
        self.p = params

    # ---------- expression dynamics (between divisions) ----------
    def step(self, gfp: float, temp_c: float, dt: float = 1.0, rng=np.random):
        """
        One-minute update:
          - small OU-like relaxation toward baseline
          - small Gaussian drift
          - rare temperature-scaled "boost" event (Δ~N(mu, sd))
        Returns: (new_gfp: float, did_boost: bool)
        """
        # OU-like decay (pull back toward baseline)
        gfp = gfp + (-self.p.decay_rate * (gfp - self.p.baseline) * dt)
        # small random drift
        gfp = gfp + rng.normal(0.0, self.p.drift_sd_per_min * np.sqrt(dt))
        # rare boost event (more likely at higher temperature)
        ramp = max(0.0, (temp_c - 30.0) / 9.0)  # 0..1
        p_evt = float(self.p.switch_prob_39C * ramp) * dt
        did = rng.rand() < p_evt
        if did:
            gfp = gfp + rng.normal(self.p.boost_mu, self.p.boost_sd)
        return float(np.clip(gfp, self.p.min_val, self.p.max_val)), bool(did)

    # ---------- inheritance at division ----------
    def inherit(self, mother_gfp: float, rng=np.random) -> float:
        g = mother_gfp + rng.normal(0.0, max(1e-9, self.p.inherit_sd))
        return float(np.clip(g, self.p.min_val, self.p.max_val))

    # ---------- fitness burden ----------
    def burden_multiplier(self, gfp: float, cap: float = 5.0) -> float:
        """
        Returns >=1.0; multiply baseline generation time by this.
        """
        x = np.clip(gfp, self.p.min_val, self.p.max_val) / self.p.max_val
        mult = 1.0 + self.p.cost_strength * (x ** max(1e-6, self.p.cost_gamma))
        return float(np.clip(mult, 1.0, 1.0 + cap))


# ===============================
# Helpers (biology-informed logic)
# ===============================

def gen_time_func(temp):
    """
    Map temperature (°C) to baseline generation time (min).
    Cooler -> faster divisions; hotter -> slower.
    """
    temp_points = [30, 33, 36, 37, 38, 39]
    gen_times   = [60, 70, 90, 120, 150, 180]  # minutes
    return np.interp(temp, temp_points, gen_times)

def feedback_temperature(mean_gfp, mode="linear", base_temp=30, max_temp=39, sensitivity=1.0):
    """
    Environmental controller for the DRIVER well.
    Temperature is updated every minute from the *current* mean GFP.
    """
    x = np.clip(mean_gfp / 100.0, 0.0, 1.0)
    s = max(1e-6, float(sensitivity))
    if mode == "linear":
        y = np.clip(s * x, 0.0, 1.0)
        return max_temp - y * (max_temp - base_temp)
    elif mode == "inverse":
        y = np.clip(s * x, 0.0, 1.0)
        return base_temp + y * (max_temp - base_temp)
    elif mode == "step":
        threshold = np.clip(0.5 / s, 0.0, 1.0)
        return base_temp if x >= threshold else max_temp
    elif mode == "exp":
        return base_temp + np.exp(-s * 5.0 * x) * (max_temp - base_temp)
    elif mode == "sigmoid":
        k = 10.0 * s
        c = 0.5
        return max_temp - (max_temp - base_temp) / (1.0 + np.exp(-k * (x - c)))
    else:
        raise ValueError("Unknown feedback mode")

def death_probability(
    temp: float,
    gfp: float,
    base_prob: float = 5e-4,     # neutral/background per-minute death
    temp_weight: float = 1.0,    # scales extra heat death relative to base_prob
    gfp_weight: float = 1.0,     # scales extra GFP death relative to base_prob
    max_prob: float = 0.05
) -> float:
    """
    Per-minute death probability composed of independent hazards:
      - Neutral (background): base_prob
      - Heat stress: ramps 0 @30°C → base_prob*temp_weight @39°C
      - GFP burden: ramps 0 @GFP=0 → base_prob*gfp_weight @GFP=100

    Combined as independent risks:
      p_total = 1 - (1-p_neutral)*(1-p_heat)*(1-p_gfp)
    """
    # Normalize temperature and GFP to [0,1]
    heat = max(0.0, (temp - 30.0) / 9.0)                  # 0..1 from 30→39°C
    burden = np.clip(gfp, 0.0, 100.0) / 100.0             # 0..1 from 0→100 GFP

    # Interpret weights as "extra hazard at max stress" in units of base_prob
    p_neutral = base_prob
    p_heat_max = base_prob * temp_weight                  # extra at 39°C
    p_gfp_max  = base_prob * gfp_weight                   # extra at GFP=100

    # Scale by current heat/GFP levels
    p_heat = p_heat_max * heat
    p_gfp  = p_gfp_max  * burden

    # Combine independent hazards
    p_total = 1.0 - (1.0 - p_neutral) * (1.0 - p_heat) * (1.0 - p_gfp)
    return float(np.clip(p_total, 0.0, max_prob))

def time_to_stationarity(times, temps, window=60, tol=0.1):
    return None


# ===============================
# Cell class
# ===============================

class Cell:
    def __init__(self, temp, time_of_birth, mother_div_time, gfp, gfp_model: GFPModel):
        # State
        self.time_of_birth = time_of_birth
        self.mother_division_time = mother_div_time
        self.time_since_division = 0.0

        # Biology
        self.model = gfp_model
        self.gfp = float(gfp)                        # 0..100
        self.generation_time = gen_time_func(temp)   # minutes (baseline)
        self.update_generation_time(temp)            # apply GFP fitness cost immediately

    def update_generation_time(self, temp):
        base = gen_time_func(temp)
        burden_mult = self.model.burden_multiplier(self.gfp)
        self.generation_time = base * burden_mult

    def division_event(self, scale=0.015):
        """
        Stochastic division hazard; probability rises with waiting time relative to gen time.
        Fitness cost is already embedded by inflating generation_time.
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
    moran_after_cap=True,        # switch to Moran birth–death once cap ever reached
    inheritance_noise=2.0,       # stddev for daughter GFP (now routed to GFPParams.inherit_sd)
    base_switch_prob=0.002,      # per-minute rare boost @39C (now routed to GFPParams.switch_prob_39C)
    feedback_sensitivity=1.0,
    control_mode=None,           # None, "fixed39", "fixed30"
    start_temp=39,
    # hidden (used only for reporting; UI-free)
    high_gfp_threshold=20.0,
    stationarity_window=60,
    stationarity_tol=0.1,
    # Fitness + death knobs (minimal surface in UI)
    gfp_cost_strength=0.5,
    gfp_cost_gamma=1.0,
    death_base_prob=0.0005,
    death_temp_weight=1.0,
    death_gfp_weight=1.0,
    death_max_prob=0.05,
    # Logging
    well_label="driver",
    log_to_stdout=False,
    capture_log=False,
    verbose_events=False
):
    """
    Continuous-trait Moran-style simulation with:
      - GFP fitness cost (slower division at higher GFP).
      - Stochastic death integrating temperature and GFP burden.
      - Stress-scaled phenotype switching (GFP boosts) more likely at hotter temps.
    """
    # --- GFP model shared by all cells in this well ---
    # Increase sensitivity of cost: amplify user strength by 2×
    gfp_params = GFPParams(
        inherit_sd=inheritance_noise,
        switch_prob_39C=base_switch_prob,
        cost_strength=gfp_cost_strength * 2.0,  # HIGHER IMPACT
        cost_gamma=gfp_cost_gamma,
    )
    gfp_model = GFPModel(gfp_params)

    # --- initialize population ---
    population = []
    initial_gfp = float(np.clip(np.random.normal(0.0, 2.0), 0.0, 5.0))
    population.append(
        Cell(start_temp, 0.0, 0.0, initial_gfp, gfp_model=gfp_model)
    )

    # --- histories ---
    temp_hist, temp_times = [], []
    mean_gfp_hist, mean_gen_hist, times = [], [], []
    pop_size_hist = []
    division_records = []
    run_log = []

    # --- event flags/metrics ---
    first_high_gfp_time = None
    moran_active = False  # becomes True once we *hit* the cap at least once

    # --- time loop ---
    current_time = 0
    base_temp, max_temp = 30.0, 39.0

    while current_time < total_time:
        # Environment: immediate feedback each minute (based on current mean GFP)
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

        # First high-GFP event (reporting marker only)
        if first_high_gfp_time is None and any(c.gfp >= high_gfp_threshold for c in population):
            first_high_gfp_time = current_time

        # --- per-minute counters for logs ---
        divisions_this_min = 0
        deaths_this_min = 0
        switches_this_min = 0

        # --- per-cell updates (GFP step -> update gen time -> death -> division -> aging) ---
        to_remove = []

        for idx, cell in enumerate(list(population)):  # snapshot iteration
            # GFP minute update (drift/decay + temp-scaled boost)
            g_before = cell.gfp
            cell.gfp, did_boost = cell.model.step(cell.gfp, temperature, dt=time_step)
            if did_boost:
                switches_this_min += 1
                if verbose_events and (log_to_stdout or capture_log):
                    ev = {"type": "switch", "well": well_label, "t": current_time, "g_before": g_before, "g_after": cell.gfp}
                    if log_to_stdout: logger.info(json.dumps(ev))
                    if capture_log: run_log.append(ev)

            # Update generation time to current temperature with GFP burden
            cell.update_generation_time(temperature)

            # Death hazard (acts before division)
            p_death = death_probability(
                temperature, cell.gfp,
                base_prob=death_base_prob,     # already scaled by death_mult in app.py
                temp_weight=death_temp_weight,
                gfp_weight=death_gfp_weight,
                max_prob=death_max_prob
            )
            if np.random.rand() < p_death:
                deaths_this_min += 1
                if verbose_events and (log_to_stdout or capture_log):
                    ev = {"type": "death", "well": well_label, "t": current_time, "gfp": cell.gfp, "p": p_death}
                    if log_to_stdout: logger.info(json.dumps(ev))
                    if capture_log: run_log.append(ev)
                to_remove.append(idx)
                continue  # dead cells don't divide or age further this tick

            # Division attempt?
            if cell.division_event():
                div_time = cell.time_since_division
                cell.time_since_division = 0.0

                # Daughter inherits with noise (can go up or down)
                daughter_gfp = cell.model.inherit(cell.gfp)
                daughter = Cell(
                    temperature,
                    time_of_birth=current_time,
                    mother_div_time=div_time,
                    gfp=daughter_gfp,
                    gfp_model=gfp_model,
                )

                divisions_this_min += 1
                if verbose_events and (log_to_stdout or capture_log):
                    ev = {"type": "division", "well": well_label, "t": current_time, "mother_gfp": cell.gfp, "daughter_gfp": daughter.gfp}
                    if log_to_stdout: logger.info(json.dumps(ev))
                    if capture_log: run_log.append(ev)

                # Growth vs Moran replacement
                if (not moran_after_cap) or (len(population) < max_population and not moran_active):
                    population.append(daughter)
                    if len(population) >= max_population and moran_after_cap:
                        moran_active = True
                else:
                    # MORAN PHASE: one-in/one-out replacement at constant size
                    victim_idx = np.random.randint(len(population))
                    population[victim_idx] = daughter

                # Record the division event (daughter snapshot)
                division_records.append({
                    "Event": "Division",
                    "Time": current_time,
                    "Temperature": temperature,
                    "Mother_Generation_Time": cell.generation_time,
                    "Mother_Division_Time": div_time,
                    "Daughter_GFP": daughter.gfp
                })

            # Aging
            cell.time_since_division += time_step

        # Remove the dead (handle indices taken from a snapshot)
        if to_remove:
            population = [c for i, c in enumerate(population) if i not in to_remove]

        # MORAN replenishment after deaths: keep size constant once Moran is active
        if moran_after_cap and moran_active and len(population) > 0:
            while len(population) < max_population:
                # choose a random parent to "fill in" the vacancy
                parent = population[np.random.randint(len(population))]
                daughter_gfp = parent.model.inherit(parent.gfp)
                daughter = Cell(
                    temperature,
                    time_of_birth=current_time,
                    mother_div_time=0.0,
                    gfp=daughter_gfp,
                    gfp_model=gfp_model,
                )
                population.append(daughter)
                division_records.append({
                    "Event": "Moran_Replenishment",
                    "Time": current_time,
                    "Temperature": temperature,
                    "Mother_Generation_Time": parent.generation_time,
                    "Mother_Division_Time": parent.time_since_division,
                    "Daughter_GFP": daughter.gfp
                })

        # (Important) If moran_after_cap=False, still hard-cap by random cull
        if not moran_after_cap:
            while len(population) > max_population:
                population.pop(np.random.randint(len(population)))

        # Per-minute aggregate log
        if log_to_stdout or capture_log:
            entry = {
                "type": "minute_agg",
                "well": well_label,
                "t": current_time,
                "temp": float(temperature),
                "mean_gfp": float(np.mean([c.gfp for c in population]) if population else 0.0),
                "pop": int(len(population)),
                "divisions": int(divisions_this_min),
                "deaths": int(deaths_this_min),
                "switches": int(switches_this_min),
                "moran_active": bool(moran_active),
            }
            if log_to_stdout: logger.info(json.dumps(entry))
            if capture_log: run_log.append(entry)

        current_time += time_step

    # Stationarity detection (on temperature series) - reporting only
    stationarity_time = None  # stationarity disabled


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
        final_population_gfps,
        run_log  # NEW: structured JSON lines list
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
    moran_after_cap=True,
    inheritance_noise=2.0,
    base_switch_prob=0.002,
    high_gfp_threshold=20.0,
    stationarity_window=60,
    stationarity_tol=0.1,
    # Fitness + death knobs
    gfp_cost_strength=0.5,
    gfp_cost_gamma=1.0,
    death_base_prob=0.0005,
    death_temp_weight=1.0,
    death_gfp_weight=1.0,
    death_max_prob=0.05,
    # Logging
    well_label="passive",
    log_to_stdout=False,
    capture_log=False,
    verbose_events=False
):
    """
    Passive wells with GFP fitness cost and death process.
    Their temperature is taken from the driver well; they do not influence it.
    """
    # shared GFP model (increase sensitivity of cost: 2×)
    gfp_params = GFPParams(
        inherit_sd=inheritance_noise,
        switch_prob_39C=base_switch_prob,
        cost_strength=gfp_cost_strength * 2.0,  # HIGHER IMPACT
        cost_gamma=gfp_cost_gamma,
    )
    gfp_model = GFPModel(gfp_params)

    population = []
    initial_gfp = float(np.clip(np.random.normal(0.0, 2.0), 0.0, 5.0))
    population.append(
        Cell(39.0, 0.0, 0.0, initial_gfp, gfp_model=gfp_model)
    )

    temp_hist, temp_times = [], []
    mean_gfp_hist, mean_gen_hist, times = [], [], []
    pop_size_hist = []
    division_records = []
    run_log = []

    first_high_gfp_time = None
    moran_active = False

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

        divisions_this_min = 0
        deaths_this_min = 0
        switches_this_min = 0

        to_remove = []
        for idx, cell in enumerate(list(population)):
            # GFP minute update
            g_before = cell.gfp
            cell.gfp, did_boost = cell.model.step(cell.gfp, temperature, dt=time_step)
            if did_boost:
                switches_this_min += 1
                if verbose_events and (log_to_stdout or capture_log):
                    ev = {"type": "switch", "well": well_label, "t": current_time, "g_before": g_before, "g_after": cell.gfp}
                    if log_to_stdout: logger.info(json.dumps(ev))
                    if capture_log: run_log.append(ev)

            # Update generation time to current temperature with GFP burden
            cell.update_generation_time(temperature)

            # Death hazard
            p_death = death_probability(
                temperature, cell.gfp,
                base_prob=death_base_prob,
                temp_weight=death_temp_weight,
                gfp_weight=death_gfp_weight,
                max_prob=death_max_prob
            )
            if np.random.rand() < p_death:
                deaths_this_min += 1
                if verbose_events and (log_to_stdout or capture_log):
                    ev = {"type": "death", "well": well_label, "t": current_time, "gfp": cell.gfp, "p": p_death}
                    if log_to_stdout: logger.info(json.dumps(ev))
                    if capture_log: run_log.append(ev)
                to_remove.append(idx)
                continue

            if cell.division_event():
                div_time = cell.time_since_division
                cell.time_since_division = 0.0
                daughter_gfp = cell.model.inherit(cell.gfp)
                daughter = Cell(
                    temperature,
                    time_of_birth=current_time,
                    mother_div_time=div_time,
                    gfp=daughter_gfp,
                    gfp_model=gfp_model,
                )

                divisions_this_min += 1
                if verbose_events and (log_to_stdout or capture_log):
                    ev = {"type": "division", "well": well_label, "t": current_time, "mother_gfp": cell.gfp, "daughter_gfp": daughter.gfp}
                    if log_to_stdout: logger.info(json.dumps(ev))
                    if capture_log: run_log.append(ev)

                if (not moran_after_cap) or (len(population) < max_population and not moran_active):
                    population.append(daughter)
                    if len(population) >= max_population and moran_after_cap:
                        moran_active = True
                else:
                    victim_idx = np.random.randint(len(population))
                    population[victim_idx] = daughter

                division_records.append({
                    "Event": "Division",
                    "Time": current_time,
                    "Temperature": temperature,
                    "Mother_Generation_Time": cell.generation_time,
                    "Mother_Division_Time": div_time,
                    "Daughter_GFP": daughter.gfp
                })

            cell.time_since_division += time_step

        if to_remove:
            population = [c for i, c in enumerate(population) if i not in to_remove]

        if moran_after_cap and moran_active and len(population) > 0:
            while len(population) < max_population:
                parent = population[np.random.randint(len(population))]
                daughter_gfp = parent.model.inherit(parent.gfp)
                daughter = Cell(
                    temperature,
                    time_of_birth=current_time,
                    mother_div_time=0.0,
                    gfp=daughter_gfp,
                    gfp_model=gfp_model,
                )
                population.append(daughter)
                division_records.append({
                    "Event": "Moran_Replenishment",
                    "Time": current_time,
                    "Temperature": temperature,
                    "Mother_Generation_Time": parent.generation_time,
                    "Mother_Division_Time": parent.time_since_division,
                    "Daughter_GFP": daughter.gfp
                })

        if not moran_after_cap:
            while len(population) > max_population:
                population.pop(np.random.randint(len(population)))

        # Per-minute aggregate log
        if log_to_stdout or capture_log:
            entry = {
                "type": "minute_agg",
                "well": well_label,
                "t": current_time,
                "temp": float(temperature),
                "mean_gfp": float(np.mean([c.gfp for c in population]) if population else 0.0),
                "pop": int(len(population)),
                "divisions": int(divisions_this_min),
                "deaths": int(deaths_this_min),
                "switches": int(switches_this_min),
                "moran_active": bool(moran_active),
            }
            if log_to_stdout: logger.info(json.dumps(entry))
            if capture_log: run_log.append(entry)

        current_time += time_step

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
        final_population_gfps,
        run_log  # NEW
    )


