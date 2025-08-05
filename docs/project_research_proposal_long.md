# **Proposal: Evolutionary Learning via Feedback-Controlled Moran Process**

*Authors: Dan Weinberg, Benel Levy, Bar Cohen*

## **1. Introduction**

Understanding how microbial populations adapt to dynamic environments remains a central question in evolutionary biology. Recent developments in synthetic biology enable experimental setups that impose artificial feedback mechanisms, making it possible to simulate selection pressures that are designed rather than naturally occurring. In this proposal, we investigate how short-term environmental feedback can lead to adaptation patterns that resemble “learning,” even in the absence of cognition or neural systems.

This work centers on a synthetic microbial system in which *Saccharomyces cerevisiae* (yeast) populations are genetically engineered to link expression of a reporter gene (GFP) with environmental temperature. A custom microfluidic device allows temperature to be dynamically regulated based on the population's mean GFP expression. When more cells express GFP, the system responds by reducing the temperature—an external reward mechanism that mimics reinforcement learning.

Our goal is to simulate this process using a stochastic evolutionary framework and assess under what conditions this population-level feedback leads to stable, adaptive expression patterns. This integrates evolutionary dynamics, phenotypic switching, fitness landscapes, and environmental coupling into a coherent computational model.

---

## **2. Biological Context**

The wet-lab experiment features yeast cells engineered with GFP under the control of the **JEN1** promoter, fused via a T2A ribosomal skip sequence. This allows the GFP reporter to be expressed separately from the JEN1 protein, which naturally responds to carbon source availability but is unrelated to temperature regulation. The synthetic design introduces a **non-natural linkage**: higher GFP expression triggers a **cooling effect** in the microfluidic environment.

### Microfluidic Setup

* **Experimental well**: GFP expression **modulates** temperature via feedback.
* **Control well**: Temperature changes identically to the experimental well but is **decoupled** from GFP expression, allowing a clean comparison.

Cells are initially grown at 30°C, a low-switching, high-fitness environment. After reaching a target population (e.g., 𝑛 = 300), the temperature is raised to 39°C to induce stress. From this point, the feedback loop begins: more High-GFP cells → cooler temperature → enhanced group fitness → selection for High-GFP expression.

---

## **3. Theoretical Foundation**

Our model draws from several key concepts in evolutionary theory:

* **Moran process**: A finite-population stochastic model used to simulate reproduction and death events with selection.
* **Feedback-modulated fitness**: Fitness is not static but depends on environmental variables influenced by population-level behaviors.
* **Phenotypic switching**: Cells can change states (Low-GFP ⇌ High-GFP) with probabilities dependent on environmental conditions.
* **Frequency-dependent selection**: The collective behavior of the population influences individual payoffs—a situation ripe for cooperative dilemmas.
* **Stochasticity and bottlenecks**: Random events, such as periodic dilution (washes), introduce drift and founder effects that shape lineage evolution.

---

## **4. Simulation Logic**

Our computational model simulates discrete time steps using a **modular, event-driven design**. Each cycle performs the following:

### 4.1 Reproduction-Death Dynamics

* A cell is chosen for **reproduction**, with probability proportional to its **temperature-modulated fitness**.
* Another cell is chosen uniformly at random to **die**, keeping the population size constant (classic Moran update).

### 4.2 Phenotypic Switching

* Each cell has a probability $P_{\text{switch}}(T)$ of changing phenotype.
* The switching probability **decreases** as the system approaches 30°C, mimicking stabilization around optimal environmental conditions:

  $$
  P_{\text{switch}}(T) = P_{\max} \cdot \left( \frac{T - 30}{9} \right)
  $$

  where $P_{\max}$ is the maximal switching rate at 39°C.

### 4.3 Environmental Feedback

* The environmental temperature is updated based on the current fraction of High-GFP cells:

  $$
  T(t) = 39^\circ C - \left( \frac{\text{High-GFP}}{N} \cdot X \right)
  $$

  * $N$: current population size
  * $X$: feedback sensitivity parameter (empirically tunable)

This function can be generalized to a **nonlinear form** (e.g., sigmoid or stepwise), allowing investigation of how feedback curvature affects learning-like dynamics.

### 4.4 Fitness Calculation

Each cell’s reproductive success is temperature- and phenotype-dependent:

$$
F_{\text{cell}} = f(T) \cdot \sigma_{\text{phenotype}}
$$

* $f(T) = 1 + \alpha (30 - T)$: linear increase in growth rate as temperature decreases
* $\sigma_{\text{phenotype}} < 1$: penalty for High-GFP cells, modeling the **metabolic cost** of expressing GFP in sugar-rich environments

---

## **5. Experimental Features and Extensions**

### 5.1 Wash Events (Bottlenecks)

Though not essential due to constant media flow, optional simulation of **periodic washes** can model:

* **Dilution effects** (nutrient renewal)
* **Founder effects** (drift from random sampling)

These can profoundly affect population structure, especially under strong selection.

### 5.2 Control vs Experimental Wells

The model includes a **control simulation** where temperature follows the same trajectory as the feedback-coupled system but without GFP influence. This allows us to isolate **feedback-driven adaptation** from passive thermosensitive gene expression.

---

## **6. Research Questions**

1. **Feedback Form**

   * How do linear, sigmoidal, or stepwise feedback curves alter the convergence speed and stability of High-GFP expression?

2. **Negative Feedback ("Punishment")**

   * What happens when GFP expression *increases* temperature (inverse reward)? Can the system learn to avoid maladaptive phenotypes?

3. **Stochasticity and Initial Conditions**

   * How sensitive is adaptation to the number and type of initial founder cells?

4. **Background Noise**

   * Introduce environment-independent phenotype switching or mutation to mimic biochemical noise.

5. **Public Good Dynamics**

   * Is High-GFP a form of cooperative behavior? Can this be interpreted through evolutionary game theory (e.g., Snowdrift Game)?

6. **Parameter Sensitivity**

   * Which model parameters (feedback strength, switching rates, cost coefficients) most influence emergent learning behavior?

7. **Experimental Validation**

   * How closely do simulation outputs match the microfluidic experiment? What are the limits of inference given measurement noise and experimental variation?

---

## **7. Generalization and Broader Implications**

This model, though specialized, lays groundwork for broader applications:

* Adaptive control in synthetic biology
* Evolutionary learning in immune systems or quorum sensing
* Frameworks for studying **group-level adaptation** without cognition

It also invites critical reflection on whether terms like “learning” are metaphorical or functionally meaningful in non-neural biological collectives.

---

## **8. Conclusion**

We present a novel model integrating phenotypic plasticity, environmental feedback, and evolutionary dynamics to explore how simple biological systems can exhibit behaviors akin to learning. Using a feedback-controlled Moran process and validated by real-time microfluidic experiments, this project probes the boundary between adaptation and cognition-free learning. Through both simulation and experimentation, we aim to better understand the role of feedback in shaping evolutionary trajectories—offering insight into how organisms might exploit structured environments to achieve adaptive outcomes.