# 🧬 Evolutionary Learning via Feedback-Controlled Moran Process

**Authors**: Dan Weinberg, Benel Levy, Bar Cohen  
**Course**: Evolution Through Programming  
**Project**: Simulating Adaptive Dynamics in Synthetic Yeast Populations

---

## 📌 Overview

This project simulates adaptive evolution in a synthetic biology system where **yeast cells express GFP** (a fluorescent reporter gene), and the **environmental temperature dynamically adjusts based on population-level GFP expression**.

The key idea: **Cells shape the environment through collective gene expression**, and the environment, in turn, influences their fitness and phenotype — forming a feedback loop that mimics **learning without cognition**.

---

## 🧠 Conceptual Components

| Component                | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| **Cells**                | Yeast cells with GFP expression, division time, and stress-based behavior |
| **GFP Expression**       | Binary (low/high) or continuous (Gaussian inheritance)                    |
| **Temperature Feedback** | Adjusted based on mean GFP in the *driver well*                           |
| **Fitness**              | Depends on temperature and GFP (high GFP = higher cost)                   |
| **Phenotypic Switching** | Modeled as noise under stress; can trigger state changes                  |
| **Moran Process**        | Population capped; excess cells are removed randomly or by fitness        |
| **Multi-Well Design**    | Simulates: 1 **driver well**, N **passive wells**, and 1 **control well** |

---

## 🧪 Simulation Features

- Flexible feedback logic (linear, exponential, sigmoid, etc.)
- Optional Gaussian inheritance of GFP
- Multiple passive wells sharing temperature
- Control condition with fixed stress (39°C)
- Tracks:
  - GFP levels
  - Temperature
  - Generation times
  - Cell divisions
- Supports interactive exploration via a **Streamlit app**

---

## 🧬 Biological Modeling

### Learning Analogy:
- **Driver well**: GFP expression affects environment (adaptive feedback)
- **Passive wells**: Share environment, but cannot influence it
- **Control well**: Constant temperature; no learning

This design mimics experimental microfluidics setups where only one well is feedback-coupled.

---

## 📁 File Structure

| File                    | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `main.py`               | Core simulation logic                             |
| `app.py`                | Streamlit UI for parameter tuning & visualization |
| `README.md`             | This documentation                                |
| *(optional)* `outputs/` | Folder for saved results (if added)               |

---

## 🔧 Simulation Function Summary

### `Cell` class
- Handles division logic and phenotype tracking

### `feedback_temperature(mean_gfp, mode, ...)`
- Converts mean GFP into temperature using selectable functions

### `determine_gfp(...)`
- Two inheritance modes:
  - **Binary Jumping**: stress-driven switches between GFP low/high
  - **Gaussian Inheritance**: continuous phenotype with small noise

### `background_switching(...)`
- Allows phenotype switching independent of division

### `fitness(...)`
- Calculates fitness for death-selection based on GFP and temperature

---

## 🚀 How to Run the Simulation (CLI)

You can test the core logic from the command line using `main.py`.

```bash
python main.py



---

## 🌐 How to Run the Interactive App

> **Requirements**: Python ≥ 3.8, Streamlit, NumPy, Pandas, Matplotlib, Plotly

### 1. Install dependencies

If not already installed:

```bash
pip install streamlit pandas numpy matplotlib plotly
```

### 2. Start the Streamlit App

```bash
streamlit run app.py
```

### 3. In the browser:

You’ll be able to:

* Select:

  * Feedback mode
  * Feedback sensitivity
  * Maximum switch rate
  * Number of passive wells
  * **Gaussian inheritance toggle**
* View:

  * GFP dynamics
  * Temperature feedback loop
  * Generation time
  * Final GFP distribution (bimodal or continuous)
* Explore raw data interactively

---

## 📊 Output Visualizations

1. **Mean GFP Expression Over Time**
2. **Temperature (Driven by Feedback)**
3. **Mean Generation Time**
4. **Final GFP Distribution**
5. **Raw Division Tables** (Driver / Passive / Control)

---

## 💡 Future Features

* Export CSV and plot downloads
* Add periodic bottlenecks or dilution events
* Simulate stress memory or epigenetic retention
* Compare multiple feedback topologies (e.g., inverted logic)

---

## 🤝 Acknowledgments

* Microfluidic yeast feedback concepts inspired by synthetic biology research
* Codebase extended from project proposal on **feedback-controlled evolutionary learning**


