# 🧬 Evolutionary Learning via GFP Feedback (Continuous Trait)

**Authors**: Dan Weinberg, Benel Levy, Bar Cohen
**Course**: Evolution Through Programming
**Project**: Adaptive dynamics in synthetic yeast under feedback control

---

## TL;DR

We simulate a yeast population where **GFP (0–100)** is a **heritable continuous trait**. In the **driver well**, the **environmental temperature updates every minute** based on the **mean GFP** (immediate feedback). **Passive wells** experience the driver’s temperature but do **not** influence it. Two **controls** run at **fixed 39 °C** (stress) and **fixed 30 °C** (benign). Learning shows up as: a **rare high-GFP breakthrough → cooling → faster division → heritable maintenance → temperature stationarity**.

---

## Install & Run

```bash
# (optional) create env
conda create -n gfp-sim python=3.9 -y
conda activate gfp-sim

# install deps
pip install streamlit numpy pandas plotly matplotlib

# run
streamlit run app.py
```

> Want reproducibility? Add `np.random.seed(42)` at the top of `app.py` before running simulations.

---

## What’s in the repo

```
main.py   # simulation engine (continuous GFP, per-minute feedback, metrics)
app.py    # Streamlit UI, plots, per-well visualization
README.md # you are here
```

---

## Biological Model (at a glance)

* **Trait:** GFP ∈ \[0, 100], **continuous**, **inherited with noise** at division.
* **Initialization:** Each well starts from **one cell** at **39 °C** (except Control-30). Initial GFP ≈ 0 (small noise, nonessential trait).
* **Time:** Discrete minutes; **temperature updates every minute**.
* **Division time vs temperature (minutes):**
  30 °C→20, 33 °C→40, 36 °C→80, 37 °C→120, 38 °C→150, 39 °C→180.
* **Division hazard:** Probabilistic; rises with time since last division relative to generation time.
* **Stress exploration:** Rare **stress-scaled “boost”** can nudge a cell’s GFP upward (more likely at hotter temps).
* **Feedback (driver only):** Temperature = f(mean GFP). Immediate, every minute.
* **Passive wells:** Follow the driver’s temperature trace (no influence).
* **Controls:** Fixed **39 °C** and fixed **30 °C**.
* **Population control:** Either **cap** (truncate above cap) or **moran** (replace beyond cap).

---

## The Feedback Functions

Temperature (°C) as a function of **mean GFP** (0–100):

* **linear**: `T = 39 − (meanGFP/100) * (39−30)`
  (0 → 39 °C; 100 → 30 °C)
* **exp**: rapid early relief; shaped by **sensitivity**
* **sigmoid**: soft at ends, steep mid (centered around GFP≈50; **sensitivity** sets slope)
* **step**: hard switch around GFP≈50
* **inverse**: “punishment” (higher GFP → hotter)

Use the **“Feedback Function”** plot in the app to show students the policy before running.

---

## Sidebar Parameters (and how to tune)

| Parameter                        | What it controls                       | Typical values                      | Biological effect / Tuning intuition                                      |
| -------------------------------- | -------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------- |
| **Total time (min)**             | Simulation horizon                     | 1000 (class), 2000–5000 (deep runs) | Longer runs show stationarity/plateaus.                                   |
| **Population cap**               | Max alive cells                        | 500–3000                            | Higher caps = smoother averages, heavier compute.                         |
| **Population model**             | `cap` or `moran`                       | cap (default)                       | `moran` replaces when above cap (Moran-like).                             |
| **Passive wells**                | # of passives to simulate              | 3–10                                | More lines in “All wells” view = more intuition on variability.           |
| **Feedback mode**                | Policy shape                           | linear (default)                    | Start linear for teaching; try sigmoid/exp after.                         |
| **Feedback sensitivity**         | Curvature for exp/sigmoid              | 0.5–3.0                             | Higher = steeper mid-response; lower = gentler.                           |
| **Inheritance noise (GFP SD)**   | Daughter around mother                 | 1–4                                 | ↑ spreads trait faster (noisier); ↓ locks in clones.                      |
| **Max stress boost prob @39 °C** | Rare boost chance at 39 °C             | 0.001–0.005                         | ↑ boosts = more breakthroughs (faster learning); ↓ for rarer discoveries. |
| **High-GFP threshold**           | Event detection for first breakthrough | 10–30                               | Where you consider “meaningful” GFP.                                      |
| **Stationarity window (min)**    | Detection window for stable temp       | 30–120                              | Larger window = stricter stability.                                       |
| **Stationarity tolerance (°C)**  | Allowed temp swing in window           | 0.05–0.2                            | Smaller tol = stricter stability.                                         |

> **Tip:** If nothing “learns”, slightly **increase** the boost prob or inheritance noise. If learning is too explosive, **decrease** them.

---

## What each plot shows (and how to read it)

1. **Feedback Function (policy curve)**
   The rule the environment follows. Use this to **predict** what the driver will try to do.

2. **Driver Temperature Over Time** (with annotations)

   * **First high-GFP time** = first trait breakthrough.
   * **Stationarity time** = environment stabilized (feedback succeeded).
     Expect: spike → discovery → sustained cooling → plateau.

3. **Mean GFP (Driver vs Controls + Passive avg)**
   Driver should rise if learning happens. Control-39 stays low (no relief). Control-30 may drift higher (benign). Passive-avg lags/follows driver.

4. **GFP — All Wells (Unaggregated)**
   Driver, each Passive (separate lines), both Controls. Shows **heterogeneity** and **luck**. Some passives “discover” earlier than others.

5. **Population Size Over Time**
   Cooling → shorter gen times → **faster growth**. A nice, intuitive correlate of learning.

6. **Phase Plot (Driver): Mean GFP → Temperature**
   Watch the control loop close: (low GFP, hot) → (higher GFP, cool) → basin.

7. **Final GFP Distributions**
   Compare trait endpoints across wells. Driver should skew higher vs Control-39; Control-30 shows benign baseline without feedback.

---

## Outputs (under the hood)

For each well we record:

* **Per-division table**: time, temperature, mother’s gen/div times, daughter GFP.
* **Time series**: temperature, mean GFP, mean generation time, population size.
* **Events**: first high-GFP time, **driver** stationarity time.
* **Final snapshot**: GFPs of all cells alive at the end (for histograms).

---

## Recommended presets (for class demos)

1. **Baseline learning (clean)**

   * Mode: **linear**
   * Inheritance noise: **2.0**
   * Boost\@39 °C: **0.002**
   * Passives: **5**
   * Expect: one clear breakthrough, cooling, stable plateau.

2. **Rare discoveries (slow learning)**

   * Noise: **1.0**
   * Boost: **0.001**
   * Expect: later breakthrough, clearer cause-effect.

3. **Explosive exploration (noisy learning)**

   * Noise: **3.0–4.0**
   * Boost: **0.003–0.005**
   * Expect: fast discoveries, divergent passives.

4. **Punishment (anti-learning)**

   * Mode: **inverse**
   * Expect: higher GFP heats the environment; driver avoids high GFP.

---

## Troubleshooting

* **Two Plotly charts causing `StreamlitDuplicateElementId`**
  Add unique `key=` for each chart (already done in this app). Example:
  `st.plotly_chart(fig, key="mean_gfp_chart")`.

* **No learning occurs** (driver stays hot, GFP flat):
  Slightly raise **boost prob** (e.g., 0.001 → 0.002) and/or **inheritance noise** (e.g., 1.0 → 2.0).

* **Too fast / unrealistic learning**:
  Lower **boost prob** or **inheritance noise**; try **sigmoid** feedback (gentler edges).

* **Temperature doesn’t look “immediate”**:
  It updates **every minute** off **current mean GFP**. If it appears smooth, that’s because averages smooth noise or because many passives are plotted (which is expected).

* **Performance**:
  Reduce **total time**, **passive wells**, or **population cap**; keep “All wells” legend manageable.

---

## Assumptions & limitations

* No explicit death model; temperature impacts **division timing** only.
* “Boost” is a stylized proxy for stress-induced exploration (not a mechanistic transcription pathway).
* Mean-field feedback; no spatial structure within wells.
* Population control is stylized (cap/Moran) rather than full chemostat.

---

## Extensions you can add next

* **Periodic dilutions/washes** (bottlenecks) to study drift vs selection.
* **Fitness cost of GFP** (metabolic burden) to shape trade-offs.
* **Parent–daughter correlation** that depends on temperature.
* **Alternative controllers** (saturated linear, PID, delayed/smoothed feedback).
* **Lineage tracking** (clone IDs) for richer evolutionary narratives.

---

## Interpreting “learning”

Learning (without neurons) appears when a population **discovers** a phenotype that **immediately improves** its environment and **inherits** enough of that phenotype to **hold** the improvement. In this app you can **see** the discovery (first high-GFP), the **reward** (cooling), the **advantage** (faster division), and **stability** (stationarity). That’s the story to tell on your plots.

---

Happy exploring!
