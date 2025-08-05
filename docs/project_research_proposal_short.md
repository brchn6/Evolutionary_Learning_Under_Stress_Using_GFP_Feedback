**Project Proposal: Evolutionary Learning in a Feedback-Controlled Moran Process**

*Authors: Dan Weinberg, Benel Levy, Bar Cohen*

In this project, we investigate how microbial populations can exhibit adaptive behavior that resembles learning through engineered environmental feedback. Our model is inspired by an ongoing wet-lab experiment in which genetically engineered *Saccharomyces cerevisiae* (yeast) cells express GFP under the control of the JEN1 promoter, a gene unrelated to temperature regulation. In a microfluidic chip, the average GFP expression in a population controls environmental temperature in real-time—creating a synthetic feedback loop where GFP expression is rewarded by cooler conditions.

We simulate this system using a stochastic Moran process with a fixed population of 300 cells. Each cell exists in one of two phenotypic states: Low-GFP or High-GFP. At each time step, a reproduction event (fitness-weighted), a death event (random), and a phenotype switching event (temperature-dependent) are applied. Temperature is updated based on the fraction of High-GFP cells, using both linear and nonlinear feedback functions to reflect different biological scenarios. Fitness is modulated by temperature and penalized for High-GFP expression due to metabolic cost.

Our goals are to explore how the structure of environmental feedback shapes adaptation, and whether collective dynamics can produce behavior analogous to reinforcement learning. We also examine the robustness of such behavior under noise, bottlenecks (e.g., periodic "wash" events), and varying initial conditions.

The project will address several key questions:

* How does the shape of the feedback function (linear vs. sigmoid vs. stepwise) affect the stability and speed of adaptation?
* Can populations evolve to avoid maladaptive phenotypes if the feedback is reversed (i.e., High-GFP increases temperature)?
* How do stochastic switching, founder effects, and bottlenecks impact the likelihood of fixation or loss of High-GFP traits?
* Does the system exhibit frequency-dependent selection or public good dynamics, where cooperative behavior (cooling via GFP) benefits all?
* How well do simulation outcomes align with empirical data from the experimental microfluidic setup?

This project aims to bridge theoretical evolutionary modeling with synthetic biology, using a minimal system to probe the boundary between adaptation and learning-like dynamics in biological collectives.
