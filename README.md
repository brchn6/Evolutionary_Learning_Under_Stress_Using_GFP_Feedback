# 🧬 Evolutionary Learning Simulation via Feedback-Controlled Moran Process

**Authors**: Dan Weinberg, Benel Levy, Bar Cohen  
**Course**: Evolution Through Programming  
**Project**: Simulating Adaptive Dynamics in Synthetic Yeast Populations

---

## 📌 Overview

This simulation models a synthetic biology system where yeast cells express **GFP** (a fluorescent reporter gene), and the **environmental temperature is adjusted based on the population's mean GFP expression**.

This creates a **feedback loop** where population-level behavior (GFP expression) modifies the environment (temperature), which in turn affects individual fitness and phenotypic switching. The goal is to explore **non-cognitive learning-like behavior** through evolutionary dynamics.

---

## 🧠 Conceptual Components

| Component                | Description                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| **Cells**                | Represent individual yeast cells with GFP levels, division timing, and phenotypic state  |
| **GFP Expression**       | Can be in a low (1X) or high (3X) state; cells can stochastically switch                 |
| **Temperature Feedback** | The system dynamically adjusts temperature based on population GFP levels                |
| **Fitness**              | Depends on temperature and GFP state; high-GFP cells have a metabolic cost               |
| **Moran Process**        | The population size is capped using a reproduction-death process with optional selection |

---

## 🧪 What This Code Does

- Simulates a population of yeast cells over time (default: 1000 minutes)
- Dynamically adjusts environmental temperature using multiple **feedback modes**
- Supports both **experimental** (GFP controls temperature) and **control** (temperature is fixed) wells
- Tracks:
  - Cumulative division events
  - Generation times
  - GFP levels
  - Temperature
- Generates plots and summary statistics

---

## 📂 File Structure

| File                    | Purpose                                                 |
| ----------------------- | ------------------------------------------------------- |
| `main.py`               | The main simulation code (with `main()` entry point)    |
| `README.md`             | Documentation (this file)                               |
| *(Optional)* `outputs/` | Folder to store data or images (e.g., exported results) |

---

## 🔧 Simulation Components Breakdown

### 1. `Cell` Class

Represents an individual cell.

**Attributes:**
- `generation_time`: Time until next division
- `gfp`: Current GFP expression level
- `time_since_division`: How long since last division
- `mother_division_time`: Inherited for jump probability calculation

**Key Method:**
- `.divide(...)`: Determines if cell divides based on age and generation time

---

### 2. `feedback_temperature(...)`

Returns the current temperature based on population mean GFP, with selectable modes:

| Mode      | Behavior                                |
| --------- | --------------------------------------- |
| `exp`     | Exponential decay (default)             |
| `linear`  | Linearly decreases with GFP             |
| `sigmoid` | Smooth transition around GFP = 2        |
| `step`    | Drops sharply when GFP > 2              |
| `inverse` | Inverse reward (more GFP = hotter temp) |

---

### 3. `background_switching(...)`

Introduces phenotype switching **independent of division**, based on current temperature. Useful for simulating **noise or biochemical drift**.

---

### 4. `fitness(...)`

Defines a cell's fitness based on:
- How close the temperature is to optimal (30°C)
- Whether the cell pays a cost for being high-GFP

Used to simulate **selection pressure** when removing cells (Moran process).

---

### 5. `run_simulation(...)`

Core simulation loop. Parameters include:

| Parameter                           | Meaning                                     |
| ----------------------------------- | ------------------------------------------- |
| `feedback_mode`                     | Controls feedback shape                     |
| `control_mode`                      | If `True`, disables feedback (control well) |
| `enable_background_switching`       | Allows phenotype changes due to noise       |
| `use_fitness_weighted_reproduction` | Selects cells for death using fitness       |
| `max_switch_rate`                   | Controls noise-driven switching probability |
| `total_time`                        | Total duration of simulation (minutes)      |

Returns:
- DataFrame of all division events
- Time series for temperature, GFP, and generation time

---

### 6. `main()`

Runs two simulations:
- **Experimental**: Feedback enabled
- **Control**: No feedback (fixed temperature)

Produces 5 plots:
1. Cumulative divisions
2. Temperature vs time
3. Generation time vs time
4. Mean GFP vs time
5. Histogram of GFP expression

Also prints:
- Total number of divisions in each condition

---

## 📊 Example Output

After running the simulation, you’ll see side-by-side comparisons between the **experimental** and **control** populations in terms of division rate, GFP dynamics, and temperature regulation.

---

## 🔄 How to Run

```bash
python main.py
