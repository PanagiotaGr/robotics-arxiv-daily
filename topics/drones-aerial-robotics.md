# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-03-12 07:09 UTC_

Total papers shown: **8**


---

- **Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming**  
  Yilin Zou, Zhong Zhang, Fanghua Jiang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10711v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Real-time trajectory optimization for nonlinear constrained autonomous systems is critical and typically performed by CPU-based sequential solvers. Specifically, reliance on global sparse linear algebra or the serial nature of dynamic programming algorithms restricts the utilization of massively parallel computing architectures like GPUs. To bridge this gap, we introduce a fully GPU-native trajectory optimization framework that combines sequential convex programming with a consensus-based alternating direction method of multipliers. By applying a temporal splitting strategy, our algorithm decouples the optimization horizon into independent, per-node subproblems that execute massively in parallel. The entire process runs fully on the GPU, eliminating costly memory transfers and large-scale sparse factorizations. This architecture naturally scales to multi-trajectory optimization. We validate the solver on a quadrotor agile flight task and a Mars powered descent problem using an on-board edge computing platform. Benchmarks reveal a sustained 4x throughput speedup and a 51% reduction in energy consumption over a heavily optimized 12-core CPU baseline. Crucially, the framework saturates the hardware, maintaining over 96% active GPU utilization to achieve planning rates exceeding 100 Hz. Furthermore, we demonstrate the solver's extensibility to robust Model Predictive Control by jointly optimizing dynamically coupled scenarios under stochastic disturbances, enabling scalable and safe autonomy.

  </details>



- **Bio-Inspired Self-Supervised Learning for Wrist-worn IMU Signals**  
  Prithviraj Tarale, Kiet Chu, Abhishek Varghese, Kai-Chun Liu, Maxwell A Xu, Mohit Iyyer, Sunghoon I. Lee  
  _2026-03-11_ · https://arxiv.org/abs/2603.10961v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Wearable accelerometers have enabled large-scale health and wellness monitoring, yet learning robust human-activity representations has been constrained by the scarcity of labeled data. While self-supervised learning offers a potential remedy, existing approaches treat sensor streams as unstructured time series, overlooking the underlying biological structure of human movement, a factor we argue is critical for effective Human Activity Recognition (HAR). We introduce a novel tokenization strategy grounded in the submovement theory of motor control, which posits that continuous wrist motion is composed of superposed elementary basis functions called submovements. We define our token as the movement segment, a unit of motion composed of a finite sequence of submovements that is readily extractable from wrist accelerometer signals. By treating these segments as tokens, we pretrain a Transformer encoder via masked movement-segment reconstruction to model the temporal dependencies of movement segments, shifting the learning focus beyond local waveform morphology. Pretrained on the NHANES corpus (approximately 28k hours; approximately 11k participants; approximately 10M windows), our representations outperform strong wearable SSL baselines across six subject-disjoint HAR benchmarks. Furthermore, they demonstrate stronger data efficiency in data-scarce settings. Code and pretrained weights will be made publicly available.

  </details>



- **UAV traffic scene understanding: A cross-spectral guided approach and a unified benchmark**  
  Yu Zhang, Zhicheng Zhao, Ze Luo, Chenglong Li, Jin Tang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10722v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Traffic scene understanding from unmanned aerial vehicle (UAV) platforms is crucial for intelligent transportation systems due to its flexible deployment and wide-area monitoring capabilities. However, existing methods face significant challenges in real-world surveillance, as their heavy reliance on optical imagery leads to severe performance degradation under adverse illumination conditions like nighttime and fog. Furthermore, current Visual Question Answering (VQA) models are restricted to elementary perception tasks, lacking the domain-specific regulatory knowledge required to assess complex traffic behaviors. To address these limitations, we propose a novel Cross-spectral Traffic Cognition Network (CTCNet) for robust UAV traffic scene understanding. Specifically, we design a Prototype-Guided Knowledge Embedding (PGKE) module that leverages high-level semantic prototypes from an external Traffic Regulation Memory (TRM) to anchor domain-specific knowledge into visual representations, enabling the model to comprehend complex behaviors and distinguish fine-grained traffic violations. Moreover, we develop a Quality-Aware Spectral Compensation (QASC) module that exploits the complementary characteristics of optical and thermal modalities to perform bidirectional context exchange, effectively compensating for degraded features to ensure robust representation in complex environments. In addition, we construct Traffic-VQA, the first large-scale optical-thermal infrared benchmark for cognitive UAV traffic understanding, comprising 8,180 aligned image pairs and 1.3 million question-answer pairs across 31 diverse types. Extensive experiments demonstrate that CTCNet significantly outperforms state-of-the-art methods in both cognition and perception scenarios. The dataset is available at https://github.com/YuZhang-2004/UAV-traffic-scene-understanding.

  </details>



- **Dynamic Modeling and Attitude Control of a Reaction-Wheel-Based Low-Gravity Bipedal Hopper**  
  Shriram Hari, M Venkata Sai Nikhil, R Prasanth Kumar  
  _2026-03-11_ · https://arxiv.org/abs/2603.10670v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Planetary bodies characterized by low gravitational acceleration, such as the Moon and near-Earth asteroids, impose unique locomotion constraints due to diminished contact forces and extended airborne intervals. Among traversal strategies, hopping locomotion offers high energy efficiency but is prone to mid-flight attitude instability caused by asymmetric thrust generation and uneven terrain interactions. This paper presents an underactuated bipedal hopping robot that employs an internal reaction wheel to regulate body posture during the ballistic flight phase. The system is modeled as a gyrostat, enabling analysis of the dynamic coupling between torso rotation and reaction wheel momentum. The locomotion cycle comprises three phases: a leg-driven propulsive jump, mid-air attitude stabilization via an active momentum exchange controller, and a shock-absorbing landing. A reduced-order model is developed to capture the critical coupling between torso rotation and reaction wheel dynamics. The proposed framework is evaluated in MuJoCo-based simulations under lunar gravity conditions (g = 1.625 m/s^2). Results demonstrate that activation of the reaction wheel controller reduces peak mid-air angular deviation by more than 65% and constrains landing attitude error to within 3.5 degrees at touchdown. Additionally, actuator saturation per hop cycle is reduced, ensuring sufficient control authority. Overall, the approach significantly mitigates in-flight attitude excursions and enables consistent upright landings, providing a practical and control-efficient solution for locomotion on irregular extraterrestrial terrains.

  </details>



- **UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery**  
  Islam Guven, Mehmet Parlak  
  _2026-03-11_ · https://arxiv.org/abs/2603.10528v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Unmanned aerial vehicles (UAVs) are increasingly used to support time-critical medical supply delivery, providing rapid and flexible logistics during emergencies and resource shortages. However, effective deployment of UAV fleets requires coordination mechanisms capable of prioritizing medical requests, allocating limited aerial resources, and adapting delivery schedules under uncertain operational conditions. This paper presents a multi-agent reinforcement learning (MARL) framework for coordinating UAV fleets in stochastic medical delivery scenarios where requests vary in urgency, location, and delivery deadlines. The problem is formulated as a partially observable Markov decision process (POMDP) in which UAV agents maintain awareness of medical delivery demands while having limited visibility of other agents due to communication and localization constraints. The proposed framework employs Proximal Policy Optimization (PPO) as the primary learning algorithm and evaluates several variants, including asynchronous extensions, classical actor--critic methods, and architectural modifications to analyze scalability and performance trade-offs. The model is evaluated using real-world geographic data from selected clinics and hospitals extracted from the OpenStreetMap dataset. The framework provides a decision-support layer that prioritizes medical tasks, reallocates UAV resources in real time, and assists healthcare personnel in managing urgent logistics. Experimental results show that classical PPO achieves superior coordination performance compared to asynchronous and sequential learning strategies, highlighting the potential of reinforcement learning for adaptive and scalable UAV-assisted healthcare logistics.

  </details>



- **ASTER: Attitude-aware Suspended-payload Quadrotor Traversal via Efficient Reinforcement Learning**  
  Dongcheng Cao, Jin Zhou, Shuo Li  
  _2026-03-11_ · https://arxiv.org/abs/2603.10715v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Agile maneuvering of the quadrotor cable-suspended system is significantly hindered by its non-smooth hybrid dynamics. While model-free Reinforcement Learning (RL) circumvents explicit differentiation of complex models, achieving attitude-constrained or inverted flight remains an open challenge due to the extreme reward sparsity under strict orientation requirements. This paper presents ASTER, a robust RL framework that achieves, to our knowledge, the first successful autonomous inverted flight for the cable-suspended system. We propose hybrid-dynamics-informed state seeding (HDSS), an initialization strategy that back-propagates target configurations through physics-consistent kinematic inversions across both taut and slack cable phases. HDSS enables the policy to discover aggressive maneuvers that are unreachable via standard exploration. Extensive simulations and real-world experiments demonstrate remarkable agility, precise attitude alignment, and robust zero-shot sim-to-real transfer across complex trajectories.

  </details>



- **MAVEN: A Meta-Reinforcement Learning Framework for Varying-Dynamics Expertise in Agile Quadrotor Maneuvers**  
  Jin Zhou, Dongcheng Cao, Xian Wang, Shuo Li  
  _2026-03-11_ · https://arxiv.org/abs/2603.10714v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has emerged as a powerful paradigm for achieving online agile navigation with quadrotors. Despite this success, policies trained via standard RL typically fail to generalize across significant dynamic variations, exhibiting a critical lack of adaptability. This work introduces MAVEN, a meta-RL framework that enables a single policy to achieve robust end-to-end navigation across a wide range of quadrotor dynamics. Our approach features a novel predictive context encoder, which learns to infer a latent representation of the system dynamics from interaction history. We demonstrate our method in agile waypoint traversal tasks under two challenging scenarios: large variations in quadrotor mass and severe single-rotor thrust loss. We leverage a GPU-vectorized simulator to distribute tasks across thousands of parallel environments, overcoming the long training times of meta-RL to converge in less than an hour. Through extensive experiments in both simulation and the real world, we validate that MAVEN achieves superior adaptation and agility. The policy successfully executes zero-shot sim-to-real transfer, demonstrating robust online adaptation by performing high-speed maneuvers despite mass variations of up to 66.7% and single-rotor thrust losses as severe as 70%.

  </details>



- **Flexible Multi-Target Angular Emulation for Over-the-Air Testing of Large-Scale ISAC Base Stations: Principle and Experimental Verification**  
  Chunhui Li, Hao Sun, Wei Fan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10629v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Over-the-air (OTA) emulation of diverse sensing target characteristics in a controlled laboratory environment is pivotal for advancing integrated sensing and communication (ISAC) technology, as it facilitates the non-invasive performance evaluation of ISAC base stations (BSs) across complex scenarios. In this work, a flexible multi-target OTA emulation framework based on a wireless cable method is proposed to evaluate the sensing performance of large-scale ISAC BSs. The core concept leverages an amplitude and phase modulation (APM) network to simultaneously establish wireless cables and simulate target spatial characteristics without consuming additional resources on costly radar target emulators. For the wireless cable method, the condition number increases as the number of antennas scales up, which affects the performance of the wireless cable. Although the wireless cable concept has been established for devices-under-test (DUTs) with a limited number of antenna ports, establishing wireless cables for large-scale DUTs remains an open question in the community. We address this problem by optimizing the OTA probe array configuration based on the theoretical properties of strictly diagonally dominant matrices. Experimental results validate the proposed framework, demonstrating high-isolation wireless cables for a 32-element DUT and an extremely low condition number for a 128-element synthetic array. Furthermore, the OTA emulation of a dynamic dual-drone scenario confirms the method's effectiveness and practicality in reproducing complex sensing environments.

  </details>


