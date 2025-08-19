#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

# =============================================================================
# ---------------------------- Parameters -------------------------------------
# =============================================================================

@dataclass
class GFPParams:
    # Initial GFP ~ N(0, init_sd) truncated at 0  (positive normal, mean=0)
    init_sd: float = 12.0
    # Binary thresholds
    low_threshold: float = 35.0
    high_threshold: float = 95.0
    # Continuous bounds
    min_val: float = 0.0
    max_val: float = 100.0
    # Inheritance (continuous)
    inherit_sd: float = 2.5
    # Phenotype switching (base, temperature-scaled)
    switch_prob_base: float = 0.01
    switch_boost_mean: float = 15.0
    switch_boost_sd: float = 5.0
    # Fitness cost of GFP (0 = neutral)
    cost_strength: float = 0.3
    cost_exponent: float = 1.5
    # Background dynamics (continuous)
    drift_rate: float = 0.001
    baseline_gfp: float = 10.0
    noise_sd: float = 0.5


@dataclass
class SimulationParams:
    # Population & timing
    population_size: int = 200
    total_time: int = 1000           # minutes
    time_step: int = 1               # minutes
    num_passive_wells: int = 2

    # Learning (driver temperature feedback)
    feedback_mode: str = "linear"    # 'linear', 'sigmoid', 'step', 'exponential'
    feedback_sensitivity: float = 1.0
    base_temp: float = 30.0
    max_temp: float = 39.0
    temp_inertia: float = 0.25       # (0,1], smaller = smoother
    start_at_max_temp: bool = True   # if False, start from target at t=0

    # Moran parent selection
    selection_mode: str = "fitness"  # 'fitness' or 'neutral'

    # Reproducibility
    random_seed: Optional[int] = 42


@dataclass
class MemoryParams:
    # dT/dt → multi-timescale EMAs (minutes)
    tau_short: float = 5.0
    tau_med: float = 30.0
    tau_long: float = 120.0

    # Thresholds for triggering (°C/min)
    th_short: float = 0.06
    th_med: float = 0.03
    th_long: float = 0.015

    # Trigger interpretation
    trigger: str = "cooling"         # 'cooling', 'warming', 'magnitude'
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)  # for combining EMAs

    # Memory dynamics
    on_threshold: float = 0.6
    off_threshold: float = 0.45
    encode_gain: float = 0.12
    decay_tau: float = 180.0
    inherit_retain: float = 0.9
    inherit_jitter_sd: float = 0.02

    # How strongly memory biases switching (0..1)
    influence_strength: float = 0.0  # this is what the app slider controls


# =============================================================================
# --------------------------- Utilities ---------------------------------------
# =============================================================================

def _ema(prev: float, value: float, dt: float, tau: float) -> float:
    alpha = 1.0 - math.exp(-dt / max(tau, 1e-9))
    return prev + alpha * (value - prev)

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _tscale_from_temp(temperature: float) -> float:
    return float(np.clip((temperature - 30.0) / 9.0, 0.0, 1.0))


def feedback_temperature(mean_gfp: float, mode: str = "linear",
                         sensitivity: float = 1.0, base_temp: float = 30.0,
                         max_temp: float = 39.0) -> float:
    """Map population mean GFP ∈ [0,100] to temperature [base,max]."""
    gfp_norm = float(np.clip(mean_gfp / 100.0, 0.0, 1.0))
    x = float(np.clip(gfp_norm * sensitivity, 0.0, 1.0))
    if mode == "linear":
        cooling_factor = x
    elif mode == "sigmoid":
        k = 10.0
        cooling_factor = 1.0 / (1.0 + math.exp(-k * (x - 0.5)))
    elif mode == "step":
        cooling_factor = 1.0 if x >= 0.5 else 0.0
    elif mode == "exponential":
        cooling_factor = 1.0 - math.exp(-3.0 * x)
    else:
        cooling_factor = x
    return float(np.clip(max_temp - cooling_factor * (max_temp - base_temp), base_temp, max_temp))


# =============================================================================
# ---------------------- Environment + Memory Signals -------------------------
# =============================================================================

class WellEnvironment:
    """
    Per-well environment tracking temperature and dT/dt EMAs,
    producing a bidirectional control signal:
      • if too warm  → sign = +1 (push HIGH GFP to cool)
      • if cool enough → sign = -1 (shed GFP cost)
    Strength blends temperature error and derivative evidence.
    """
    __slots__ = ("last_temp", "ema_s", "ema_m", "ema_l", "mem_p", "base_temp", "max_temp")

    def __init__(self, initial_temp: float, mem_p: MemoryParams,
                 base_temp: float = 30.0, max_temp: float = 39.0):
        self.last_temp = float(initial_temp)
        self.ema_s = 0.0
        self.ema_m = 0.0
        self.ema_l = 0.0
        self.mem_p = mem_p
        self.base_temp = float(base_temp)
        self.max_temp = float(max_temp)

    def step(self, new_temp: float, dt: float) -> Dict[str, float]:
        dT = (float(new_temp) - self.last_temp) / dt
        self.ema_s = _ema(self.ema_s, dT, dt, self.mem_p.tau_short)
        self.ema_m = _ema(self.ema_m, dT, dt, self.mem_p.tau_med)
        self.ema_l = _ema(self.ema_l, dT, dt, self.mem_p.tau_long)
        self.last_temp = float(new_temp)
        return {"temp": self.last_temp, "ema_short": self.ema_s, "ema_med": self.ema_m, "ema_long": self.ema_l}

    def trigger_strength_and_sign(self) -> Tuple[float, float]:
        """
        Returns (strength in [0,1], sign ∈ {+1, -1, 0}).
        Uses:
          • error = temp - base_temp   (homeostatic goal)
          • combo = weighted EMA of dT/dt
        Hysteresis: two thresholds around the goal reduce flip-flop.
        """
        temp = self.last_temp
        err = temp - self.base_temp                           # >0 too warm, <0 cool
        err_norm = float(np.clip(abs(err) / max(self.max_temp - self.base_temp, 1e-9), 0.0, 1.0))

        w = np.array(self.mem_p.weights, dtype=float); w /= (w.sum() + 1e-12)
        th_w = float(np.dot(w, np.array([self.mem_p.th_short, self.mem_p.th_med, self.mem_p.th_long])))
        combo = float(np.dot(w, np.array([self.ema_s, self.ema_m, self.ema_l])))  # °C/min

        # derivative “evidence”
        if self.mem_p.trigger == "cooling":
            drv_mag = max(0.0, -combo)               # negative dT/dt → cooling
        elif self.mem_p.trigger == "warming":
            drv_mag = max(0.0, +combo)               # positive dT/dt → warming
        else:                                        # magnitude
            drv_mag = abs(combo)
        drv_excess = max(0.0, (drv_mag - th_w) / (th_w + 1e-12))  # normalized

        # blend error & derivative into a 0..1 urgency
        strength = float(np.clip(0.6 * err_norm + 0.4 * drv_excess, 0.0, 1.0))

        # bidirectional sign (with deadband to avoid chattering)
        # if > goal + 0.3°C → push HIGH; if < goal + 0.1°C → push LOW; else neutral
        if err > 0.3:
            sign = +1.0
        elif err < 0.1:
            sign = -1.0
        else:
            sign = 0.0
        return strength, sign


# =============================================================================
# -------------------------- Epigenetic Memory --------------------------------
# =============================================================================

class EpigeneticMemory:
    __slots__ = ("m", "on", "p")

    def __init__(self, p: MemoryParams):
        self.p = p
        self.m = 0.0
        self.on = False

    def update(self, trig_strength: float, dt: float):
        # decay
        a = 1.0 - math.exp(-dt / max(self.p.decay_tau, 1e-9))
        self.m = (1.0 - a) * self.m
        # encode
        self.m = _clamp(self.m + self.p.encode_gain * trig_strength, 0.0, 1.0)
        # hysteresis
        if not self.on and self.m >= self.p.on_threshold:
            self.on = True
        elif self.on and self.m <= self.p.off_threshold:
            self.on = False

    def inherit_child(self) -> "EpigeneticMemory":
        ch = EpigeneticMemory(self.p)
        mu = self.m * self.p.inherit_retain
        ch.m = _clamp(float(np.random.normal(mu, self.p.inherit_jitter_sd)), 0.0, 1.0)
        if ch.m >= self.p.on_threshold:
            ch.on = True
        elif ch.m <= self.p.off_threshold:
            ch.on = False
        else:
            ch.on = self.on
        return ch


# =============================================================================
# ------------------------------ Cells & Pop ----------------------------------
# =============================================================================

class _Cell:
    __slots__ = ("gfp", "age", "mode", "generation_time", "_last_temp", "_gfp_params", "mem")

    def __init__(self, gfp_value: float, mode: str, gfp_params: GFPParams, mem_params: MemoryParams):
        self.gfp = float(np.clip(gfp_value, 0.0, 100.0))
        self.age = 0.0
        self.mode = mode
        self.generation_time = 120.0
        self._last_temp = 39.0
        self._gfp_params = gfp_params
        self.mem = EpigeneticMemory(mem_params)

    # fitness
    @staticmethod
    def _base_generation_time(temperature: float) -> float:
        t = _tscale_from_temp(temperature)
        return 60.0 + 120.0 * (t ** 2)

    def _cost_multiplier(self) -> float:
        p = self._gfp_params
        if self.mode == "binary":
            return 1.3 if self.gfp > 50.0 else 1.0
        g = self.gfp / 100.0
        return 1.0 + p.cost_strength * (g ** p.cost_exponent)

    def update_fitness(self, temperature: float):
        self.generation_time = self._base_generation_time(temperature) * self._cost_multiplier()
        self._last_temp = temperature

    # phenotype dynamics
    def background_dynamics(self, dt: float):
        p = self._gfp_params
        if self.mode == "continuous":
            drift = -p.drift_rate * (self.gfp - p.baseline_gfp) * dt
            noise = np.random.normal(0.0, p.noise_sd * math.sqrt(dt))
            self.gfp = float(np.clip(self.gfp + drift + noise, p.min_val, p.max_val))

    def phenotype_switch(self, temperature: float, dt: float,
                        mem_influence: float, trig_strength: float, trig_sign: float) -> bool:
        p = self._gfp_params
        tscale = _tscale_from_temp(temperature)
        base = p.switch_prob_base * tscale * dt

        # memory amplifies switching *and* encodes direction
        amp = 1.0 + mem_influence * self.mem.m * trig_strength
        prob = float(np.clip(base * amp, 0.0, 1.0))

        if np.random.random() < prob and trig_sign != 0.0:
            if self.mode == "binary":
                self.gfp = p.high_threshold if trig_sign > 0 else p.low_threshold
            else:
                boost = np.random.normal(p.switch_boost_mean, p.switch_boost_sd)
                delta = +boost if trig_sign > 0 else -boost
                self.gfp = float(np.clip(self.gfp + delta, p.min_val, p.max_val))
            return True
        return False

    def divide(self, mem_params: MemoryParams) -> "_Cell":
        p = self._gfp_params
        if self.mode == "binary":
            # low chance to flip at division
            if np.random.random() < 0.05:
                daughter_gfp = p.high_threshold if self.gfp < 50.0 else p.low_threshold
            else:
                daughter_gfp = self.gfp
        else:
            daughter_gfp = float(np.clip(self.gfp + np.random.normal(0.0, p.inherit_sd),
                                         p.min_val, p.max_val))
        d = _Cell(daughter_gfp, self.mode, p, mem_params)
        d.mem = self.mem.inherit_child()
        d.update_fitness(self._last_temp)
        return d


class Population:
    def __init__(self, N: int, mode: str, gfp_p: GFPParams, mem_p: MemoryParams,
                 label: str, selection_mode: str = "fitness"):
        self.N = int(N)
        self.mode = mode
        self.gfp_p = gfp_p
        self.mem_p = mem_p
        self.label = label
        self.selection_mode = selection_mode
        self.cells: List[_Cell] = []
        self.h: Dict[str, List[float]] = {
            "time": [], "temperature": [], "mean_gfp": [], "high_gfp_fraction": [],
            "mean_generation_time": [], "population_size": [],
            "mean_memory": [], "gfp_on_fraction": [], "switches": []
        }
        self._initialize()

    def _initialize(self):
        for _ in range(self.N):
            g0 = max(0.0, float(np.random.normal(0.0, max(self.gfp_p.init_sd, 1e-9))))
            if self.mode == "continuous":
                g0 = float(np.clip(g0, self.gfp_p.min_val, self.gfp_p.max_val))
            else:  # binary starts low by default given half-normal around 0
                g0 = self.gfp_p.low_threshold
            self.cells.append(_Cell(g0, self.mode, self.gfp_p, self.mem_p))

    def _stats(self) -> Tuple[float, float, float, float]:
        g = np.array([c.gfp for c in self.cells], dtype=float)
        mean_g = float(g.mean()) if len(g) else 0.0
        thr = 50.0 if self.mode == "binary" else 60.0
        frac_high = float((g > thr).mean()) if len(g) else 0.0
        mean_gt = float(np.mean([c.generation_time for c in self.cells])) if self.cells else 0.0
        mean_m = float(np.mean([c.mem.m for c in self.cells])) if self.cells else 0.0
        frac_on = float(np.mean([1.0 if c.mem.on else 0.0 for c in self.cells])) if self.cells else 0.0
        return mean_g, frac_high, mean_gt, mean_m, frac_on

    def step(self, current_time: float, temperature: float,
             env: WellEnvironment, dt: float) -> Dict[str, int]:
        # Memory evidence from environment
        signals = env.step(temperature, dt)
        trig_strength, trig_sign = env.trigger_strength_and_sign()

        # Update cells
        switches = 0
        for c in self.cells:
            c.age += dt
            c.background_dynamics(dt)
            c.mem.update(trig_strength, dt)
            if c.phenotype_switch(temperature, dt, self.mem_p.influence_strength, trig_strength, trig_sign):
                switches += 1
            c.update_fitness(temperature)

        # Moran BD
        if self.selection_mode == "fitness":
            weights = np.array([1.0 / max(c.generation_time, 1.0) for c in self.cells], dtype=float)
            p = weights / (weights.sum() + 1e-12)
            parent_idx = int(np.random.choice(len(self.cells), p=p))
        else:
            parent_idx = int(np.random.randint(0, len(self.cells)))
        victim_idx = int(np.random.randint(0, len(self.cells)))
        self.cells[victim_idx] = self.cells[parent_idx].divide(self.mem_p)

        # Record
        mean_g, frac_high, mean_gt, mean_m, frac_on = self._stats()
        self.h["time"].append(current_time)
        self.h["temperature"].append(temperature)
        self.h["mean_gfp"].append(mean_g)
        self.h["high_gfp_fraction"].append(frac_high)
        self.h["mean_generation_time"].append(mean_gt)
        self.h["population_size"].append(len(self.cells))
        self.h["mean_memory"].append(mean_m)
        self.h["gfp_on_fraction"].append(frac_on)
        self.h["switches"].append(float(switches))
        return {"switches": switches}


# =============================================================================
# ------------------------------ Runner ---------------------------------------
# =============================================================================

def run_combined_experiment(sim_p: SimulationParams, gfp_p: GFPParams,
                            mem_p: MemoryParams, mode: str = "continuous") -> Dict[str, Any]:
    """Run driver + passives (mirror temp) + controls with epigenetic memory integrated."""
    if sim_p.random_seed is not None:
        np.random.seed(int(sim_p.random_seed))

    CONTROL_30 = 30.0
    CONTROL_39 = 39.0

    # Populations
    driver = Population(sim_p.population_size, mode, gfp_p, mem_p, "driver", selection_mode=sim_p.selection_mode)
    passives = [Population(sim_p.population_size, mode, gfp_p, mem_p, f"passive_{i+1}",
                           selection_mode=sim_p.selection_mode) for i in range(sim_p.num_passive_wells)]
    control_30 = Population(sim_p.population_size, mode, gfp_p, mem_p, "control_30",
                            selection_mode=sim_p.selection_mode)
    control_39 = Population(sim_p.population_size, mode, gfp_p, mem_p, "control_39",
                            selection_mode=sim_p.selection_mode)

    # Environments (one per well)
    init_mean = float(np.mean([c.gfp for c in driver.cells])) if driver.cells else 0.0
    target0 = feedback_temperature(init_mean, sim_p.feedback_mode, sim_p.feedback_sensitivity,
                                   sim_p.base_temp, sim_p.max_temp)
    prev_temp = sim_p.max_temp if sim_p.start_at_max_temp else target0

    env_driver   = WellEnvironment(prev_temp, mem_p, base_temp=sim_p.base_temp, max_temp=sim_p.max_temp)
    env_passives = [WellEnvironment(prev_temp, mem_p, base_temp=sim_p.base_temp, max_temp=sim_p.max_temp)
                    for _ in passives]
    env_c30      = WellEnvironment(CONTROL_30, mem_p, base_temp=sim_p.base_temp, max_temp=sim_p.max_temp)
    env_c39      = WellEnvironment(CONTROL_39, mem_p, base_temp=sim_p.base_temp, max_temp=sim_p.max_temp)


    # Time loop
    for t in range(0, sim_p.total_time, int(sim_p.time_step)):
        current = float(t)

        # Driver temperature via learning feedback + inertia
        if t == 0:
            driver_temp = prev_temp
        else:
            mean_gfp_driver = float(np.mean([c.gfp for c in driver.cells])) if driver.cells else 0.0
            tgt = feedback_temperature(mean_gfp_driver, sim_p.feedback_mode, sim_p.feedback_sensitivity,
                                       sim_p.base_temp, sim_p.max_temp)
            driver_temp = prev_temp + float(np.clip(sim_p.temp_inertia, 1e-8, 1.0)) * (tgt - prev_temp)
            prev_temp = driver_temp

        # Step all wells (passives mirror driver temp)
        driver.step(current, driver_temp, env_driver, sim_p.time_step)
        for env_p, pop_p in zip(env_passives, passives):
            pop_p.step(current, driver_temp, env_p, sim_p.time_step)
        control_30.step(current, CONTROL_30, env_c30, sim_p.time_step)
        control_39.step(current, CONTROL_39, env_c39, sim_p.time_step)

    return {
        "driver": driver.h,
        "passives": [p.h for p in passives],
        "control_30": control_30.h,
        "control_39": control_39.h,
        "parameters": {"simulation": sim_p, "gfp": gfp_p, "memory": mem_p, "mode": mode},
    }


# =============================================================================
# --------------------------- Metrics / Validation ----------------------------
# =============================================================================

def calculate_learning_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    d = results["driver"]
    if not d["time"]:
        return {}
    final_temp = float(d["temperature"][-1])
    initial_temp = float(d["temperature"][0])
    denom = max(initial_temp - 30.0, 1e-9)
    learning_score = float(np.clip((initial_temp - final_temp) / denom, 0.0, 1.0))

    times = np.array(d["time"], dtype=float)
    temps = np.array(d["temperature"], dtype=float)
    target_mid = initial_temp - 0.5 * (initial_temp - final_temp)
    adaptation_time = None
    for tm, tp in zip(times, temps):
        if tp <= target_mid:
            adaptation_time = float(tm)
            break

    high_frac = np.array(d["high_gfp_fraction"], dtype=float)
    establishment_time = None
    for tm, hf in zip(times, high_frac):
        if hf >= 0.5:
            establishment_time = float(tm)
            break

    # stability of final segment
    L = max(1, int(0.2 * len(temps)))
    var_tail = float(np.var(temps[-L:]))
    temp_stability = 1.0 / (1.0 + var_tail)

    return {
        "learning_score": learning_score,
        "final_temperature": final_temp,
        "final_gfp": float(d["mean_gfp"][-1]),
        "adaptation_time": adaptation_time,
        "establishment_time": establishment_time,
        "temperature_stability": temp_stability,
        "final_memory_on_fraction": float(d["gfp_on_fraction"][-1]),
        "final_mean_memory": float(d["mean_memory"][-1]),
    }


def validate_parameters(sim_p: SimulationParams, gfp_p: GFPParams, mem_p: MemoryParams) -> List[str]:
    w: List[str] = []
    if sim_p.total_time < 200:
        w.append("⚠️ Short simulation may not show adaptation.")
    if sim_p.population_size < 50:
        w.append("⚠️ Small population size increases drift.")
    if sim_p.feedback_sensitivity < 0.1:
        w.append("⚠️ Very low sensitivity may prevent learning.")
    if not (0.0 < sim_p.temp_inertia <= 1.0):
        w.append("⚠️ temp_inertia should be in (0,1].")
    if mem_p.influence_strength < 0.0 or mem_p.influence_strength > 1.0:
        w.append("⚠️ influence_strength must be in [0,1].")
    return w
