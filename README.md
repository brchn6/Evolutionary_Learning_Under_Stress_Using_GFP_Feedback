# 🧬 Evolutionary Learning via GFP Feedback (Continuous Trait)

**Authors**: Dan Weinberg, Benel Levy, Bar Cohen
**Course**: Evolution Through Programming
**Project**: Adaptive dynamics in synthetic yeast under feedback control

---

## TL;DR

We simulate a yeast population where **GFP (0–100)** is a **heritable continuous trait**. In the **driver well**, the **environmental temperature updates every minute** from the **current mean GFP** (tight feedback). **Passive wells** experience this *same* temperature but **do not influence** it. Two **controls** run at **fixed 39 °C** (stress) and **fixed 30 °C** (benign).

**New in this version**

* A **stochastic death process** whose hazard grows with **heat** and **GFP burden**.
* **Fitness** explicitly includes a **GFP metabolic cost** that **slows division**; temperature still sets the **baseline generation time**.
* **Minimal UI**: confusing event/stability knobs removed (kept only as readouts).
* Outputs reveal your “**pattern-breaking**” case via **counter-learning** markers and a **learning ratio** metric.

---

## Install & Run

```bash
# (optional) create env
conda create -n gfp-sim python=3.9 -y
conda activate gfp-sim

# install deps
pip install streamlit numpy pandas plotly

# run
streamlit run app.py
```

> For reproducibility, enable the seed in the sidebar or add `np.random.seed(42)` in `app.py`.

---

## What’s in the repo

```
main.py   # simulation engine (continuous GFP, per-minute feedback, fitness & death)
app.py    # Streamlit UI; learning ratio, counter-learning markers, plots
README.md # this document
```

---

## Biological Model (concise)

* **Trait:** GFP ∈ \[0, 100], **continuous**, **inherited with Normal noise** at division.
* **Initialization:** Each well starts from **one cell** (GFP ≈ 0 ± N(0,2)).
* **Time:** Discrete minutes; **temperature updates every minute**.
* **Division time vs. temperature (min):** 30→60, 33→70, 36→90, 37→120, 38→150, 39→180.
* **Fitness (division speed):**
  baseline **generation time from temperature** × **GFP cost multiplier**
  `mult = 1 + strength * (GFP/100)^gamma` (bounded).
  Higher GFP → **slower division** (metabolic burden).
* **Death (per minute):** Hazard increases with **temperature** and **GFP** (background hazard at 30 °C & GFP≈0).
* **Phenotype switching (“boost”):** rare **stress-scaled** GFP jumps (Δ \~ N(10,3)), **more likely at hotter temps**.
* **Feedback (driver only):** Temperature = f(mean GFP). Immediate (no delay).
* **Passive wells:** **Follow** the driver’s temperature trace (no influence).
* **Controls:** Fixed **39 °C** and fixed **30 °C**.
* **Population control:** Once the cap is reached (if enabled), switch to **Moran** birth–death at constant N.

---

## Driver vs. Passives & Controls

* **Driver well**: its **mean GFP** sets **temperature** each minute.
* **Passive wells**: take the **driver’s temperature** as-is and evolve independently under it.
* **Controls**: constant **39 °C** (stress) and **30 °C** (benign).
  This setup lets you observe the driver’s “learning” while others either follow (passives) or provide baselines (controls).

---

## Minimal Parameters (UI)

| Section         | Parameter                        | Meaning                                 | Typical values                |
| --------------- | -------------------------------- | --------------------------------------- | ----------------------------- |
| Core            | **Total time**                   | Horizon in minutes                      | 1000 (demo), 2000–5000 (deep) |
| Core            | **Population cap N**             | Max concurrent cells                    | 500–3000                      |
| Core            | **Moran after cap**              | Replace at N to keep constant size      | on                            |
| Core            | **Passive wells**                | # wells sharing driver temp             | 3–10                          |
| Feedback        | **Mode**                         | linear / exp / sigmoid / step / inverse | linear                        |
| Feedback        | **Sensitivity**                  | Steepness / response                    | 0.5–3                         |
| Trait           | **Inheritance noise (GFP SD)**   | Daughter around mother                  | 1–4                           |
| Trait           | **Phenotype switch prob @39 °C** | Per-minute probability at 39 °C         | 0.001–0.005                   |
| Fitness & Mort. | **GFP cost strength**            | Burden on division speed                | 0.3–1.0                       |
| Fitness & Mort. | **Mortality multiplier**         | Scales background death hazard          | 0.5–2.0                       |

> We **removed** UI knobs for **High-GFP threshold**, **Stationarity window**, **Stationarity tolerance**. They remain **internally** as readouts/annotations only.

---

## What those old parameters meant (and why we hid them)

* **High-GFP threshold (event):** just a **marker** for the **first meaningful GFP breakthrough** (for annotation). Not mechanistic.
* **Stationarity window (min) / tolerance (°C):** used only to **detect** when **temperature plateaus** (for reporting).
* **“Max stress boost prob @39 °C”:** renamed to **“Phenotype switch prob @39 °C”** (clearer). See formula below.

These are useful for **storytelling/plots**, not as tuning knobs, so they’re hidden to keep the UI minimal.

---

## Probability of phenotype switching (your question)

**Per-minute** probability at temperature **T** (°C):

```
p_switch(T) = p_39C * max(0, (T - 30) / 9)
```

Where **`p_39C`** is the slider **“Phenotype switch prob @39 °C”**.
So: **0 at 30 °C**, ramping linearly to **`p_39C` at 39 °C**. Example with `p_39C = 0.002`:

* 30 °C → 0.0000
* 33 °C → 0.0007
* 36 °C → 0.0013
* 39 °C → 0.0020

This encodes the intuition that **stress (heat) promotes exploratory switching**.

---

## Reading the outputs

1. **Driver Temperature Over Time**
   Annotated with **first high-GFP** and **stationarity**. **Red dotted lines** mark **counter-learning episodes**.

2. **Mean GFP (Driver, Controls, Passive Avg)**
   Expect driver GFP to rise if learning succeeds; Control-39 stays low; Control-30 may drift.

3. **Counter-learning episodes** *(your “break the learning pattern” case)*
   Minutes where **driver GFP decreases**, **passive average GFP increases**, **and** **temperature rises**.
   These capture the moments where the driver “loses the thread” while others still escalate.

4. **Learning ratio (0–1)**
   Average **temperature relief** of the driver, normalized between 39→30 °C:

   ```
   LR = mean((39 − T_driver) / 9)
   ```

   Higher = stronger/steadier learning.

5. **All wells** and **Final distributions**
   Show heterogeneity and end states across driver / passives / controls.

---

## Assumptions & limitations

* Death and GFP costs are stylized hazards/multipliers, not mechanistic physiology.
* Phenotypic “boost” is a coarse stochastic jump, not a full gene-regulatory model.
* No spatial structure; wells are mean-field.
* Controller reads **current** mean GFP each minute (no delay).

---

## Suggested presets

* **Clean learning demo**: linear, sensitivity=1.0, noise=2.0, switch\@39 °C=0.002, cost=0.5, death×=1.0
* **Hard mode (stressful)**: linear, sensitivity=0.7, noise=1.0, switch\@39 °C=0.001, cost=0.8, death×=1.5
* **Anti-learning**: inverse feedback, any other params

---

## Publishing notes

* UI exposes a **minimal knob set** to avoid confusion.
* Metrics emphasize the **story** (breakthrough → cooling → stabilization → occasional pattern breaks).
* Plots include consistent keys to avoid Streamlit widget collisions.

---

Happy exploring!
