# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-03-12 07:09 UTC_

Total papers shown: **4**


---

- **Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming**  
  Yilin Zou, Zhong Zhang, Fanghua Jiang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10711v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Real-time trajectory optimization for nonlinear constrained autonomous systems is critical and typically performed by CPU-based sequential solvers. Specifically, reliance on global sparse linear algebra or the serial nature of dynamic programming algorithms restricts the utilization of massively parallel computing architectures like GPUs. To bridge this gap, we introduce a fully GPU-native trajectory optimization framework that combines sequential convex programming with a consensus-based alternating direction method of multipliers. By applying a temporal splitting strategy, our algorithm decouples the optimization horizon into independent, per-node subproblems that execute massively in parallel. The entire process runs fully on the GPU, eliminating costly memory transfers and large-scale sparse factorizations. This architecture naturally scales to multi-trajectory optimization. We validate the solver on a quadrotor agile flight task and a Mars powered descent problem using an on-board edge computing platform. Benchmarks reveal a sustained 4x throughput speedup and a 51% reduction in energy consumption over a heavily optimized 12-core CPU baseline. Crucially, the framework saturates the hardware, maintaining over 96% active GPU utilization to achieve planning rates exceeding 100 Hz. Furthermore, we demonstrate the solver's extensibility to robust Model Predictive Control by jointly optimizing dynamically coupled scenarios under stochastic disturbances, enabling scalable and safe autonomy.

  </details>



- **GRACE: A Unified 2D Multi-Robot Path Planning Simulator & Benchmark for Grid, Roadmap, And Continuous Environments**  
  Chuanlong Zang, Anna Mannucci, Isabelle Barz, Philipp Schillinger, Florian Lier, Wolfgang Hönig  
  _2026-03-11_ · https://arxiv.org/abs/2603.10858v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Advancing Multi-Agent Pathfinding (MAPF) and Multi-Robot Motion Planning (MRMP) requires platforms that enable transparent, reproducible comparisons across modeling choices. Existing tools either scale under simplifying assumptions (grids, homogeneous agents) or offer higher fidelity with less comparable instrumentation. We present GRACE, a unified 2D simulator+benchmark that instantiates the same task at multiple abstraction levels (grid, roadmap, continuous) via explicit, reproducible operators and a common evaluation protocol. Our empirical results on public maps and representative planners enable commensurate comparisons on a shared instance set. Furthermore, we quantify the expected representation-fidelity trade-offs (MRMP solves instances at higher fidelity but lower speed, while grid/roadmap planners scale farther). By consolidating representation, execution, and evaluation, GRACE thereby aims to make cross-representation studies more comparable and provides a means to advance multi-robot planning research and its translation to practice.

  </details>



- **UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery**  
  Islam Guven, Mehmet Parlak  
  _2026-03-11_ · https://arxiv.org/abs/2603.10528v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Unmanned aerial vehicles (UAVs) are increasingly used to support time-critical medical supply delivery, providing rapid and flexible logistics during emergencies and resource shortages. However, effective deployment of UAV fleets requires coordination mechanisms capable of prioritizing medical requests, allocating limited aerial resources, and adapting delivery schedules under uncertain operational conditions. This paper presents a multi-agent reinforcement learning (MARL) framework for coordinating UAV fleets in stochastic medical delivery scenarios where requests vary in urgency, location, and delivery deadlines. The problem is formulated as a partially observable Markov decision process (POMDP) in which UAV agents maintain awareness of medical delivery demands while having limited visibility of other agents due to communication and localization constraints. The proposed framework employs Proximal Policy Optimization (PPO) as the primary learning algorithm and evaluates several variants, including asynchronous extensions, classical actor--critic methods, and architectural modifications to analyze scalability and performance trade-offs. The model is evaluated using real-world geographic data from selected clinics and hospitals extracted from the OpenStreetMap dataset. The framework provides a decision-support layer that prioritizes medical tasks, reallocates UAV resources in real time, and assists healthcare personnel in managing urgent logistics. Experimental results show that classical PPO achieves superior coordination performance compared to asynchronous and sequential learning strategies, highlighting the potential of reinforcement learning for adaptive and scalable UAV-assisted healthcare logistics.

  </details>



- **6ABOS: An Open-Source Atmospheric Correction Framework for the EnMAP Hyperspectral Mission Based on 6S**  
  Gabriel Caballero Cañas, Bárbara Alvado Arranz, Xavier Sòria-Perpinyà, Antonio Ruiz-Verdú, Jesús Delegido, José Moreno  
  _2026-03-11_ · https://arxiv.org/abs/2603.10856v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The Environmental Mapping and Analysis Program (EnMAP) mission has opened new frontiers in the monitoring of optically complex environments. However, the accurate retrieval of surface reflectance over water bodies remains a significant challenge, as the water-leaving signal typically accounts for only a small fraction of the total radiance, being easily obscured by atmospheric scattering and surface reflection effects. This paper introduces 6ABOS (6S-based Atmospheric Background Offset Subtraction), a novel open-source Python framework designed to automate the atmospheric correction (AC) of EnMAP hyperspectral imagery. By leveraging the Second Simulation of the Satellite Signal in the Solar Spectrum (6S) radiative transfer model, 6ABOS implements a physically-based inversion scheme that accounts for Rayleigh scattering, aerosol interactions, and gaseous absorption. The framework integrates automated EnMAP metadata parsing with dynamic atmospheric parameter retrieval via the Google Earth Engine (GEE) Application Programming Interface (API). Validation was conducted over two Mediterranean inland water reservoirs with contrasting trophic states: the oligotrophic Benag{'e}ber and the hypertrophic Bell{'u}s. Results demonstrate a high degree of spectral similarity between in situ measurements and EnMAP-derived water-leaving reflectances. The Spectral Angle Mapper (SAM) values remained consistently low (SAM $<$ 10$^\circ$) across both study sites. 6ABOS is distributed via conda-forge, providing the scientific community with a scalable, transparent, and reproducible open-science tool for advancing hyperspectral aquatic research in the cloud-computing era.

  </details>


