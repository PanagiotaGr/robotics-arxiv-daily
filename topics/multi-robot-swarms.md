# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-26 07:13 UTC_

Total papers shown: **7**


---

- **Parallel Continuous-Time Relative Localization with Augmented Clamped Non-Uniform B-Splines**  
  Jiadong Lu, Zhehan Li, Tao Han, Miao Xu, Chao Xu, Yanjun Cao  
  _2026-02-25_ · https://arxiv.org/abs/2602.22006v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate relative localization is critical for multi-robot cooperation. In robot swarms, measurements from different robots arrive asynchronously and with clock time-offsets. Although Continuous-Time (CT) formulations have proved effective for handling asynchronous measurements in single-robot SLAM and calibration, extending CT methods to multi-robot settings faces great challenges to achieve high-accuracy, low-latency, and high-frequency performance. Especially, existing CT methods suffer from the inherent query-time delay of unclamped B-splines and high computational cost. This paper proposes CT-RIO, a novel Continuous-Time Relative-Inertial Odometry framework. We employ Clamped Non-Uniform B-splines (C-NUBS) to represent robot states for the first time, eliminating the query-time delay. We further augment C-NUBS with closed-form extension and shrinkage operations that preserve the spline shape, making it suitable for online estimation and enabling flexible knot management. This flexibility leads to the concept of knot-keyknot strategy, which supports spline extension at high-frequency while retaining sparse keyknots for adaptive relative-motion modeling. We then formulate a sliding-window relative localization problem that operates purely on relative kinematics and inter-robot constraints. To meet the demanding computation required at swarm scale, we decompose the tightly-coupled optimization into robot-wise sub-problems and solve them in parallel using incremental asynchronous block coordinate descent. Extensive experiments show that CT-RIO converges from time-offsets as large as 263 ms to sub-millisecond within 3 s, and achieves RMSEs of 0.046 m and 1.8 °. It consistently outperforms state-of-the-art methods, with improvements of up to 60% under high-speed motion.

  </details>



- **Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning**  
  Tomoya Kawabe, Rin Takano  
  _2026-02-25_ · https://arxiv.org/abs/2602.21670v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-robot task planning requires decomposing natural-language instructions into executable actions for heterogeneous robot teams. Conventional Planning Domain Definition Language (PDDL) planners provide rigorous guarantees but struggle to handle ambiguous or long-horizon missions, while large language models (LLMs) can interpret instructions and propose plans but may hallucinate or produce infeasible actions. We present a hierarchical multi-agent LLM-based planner with prompt optimization: an upper layer decomposes tasks and assigns them to lower-layer agents, which generate PDDL problems solved by a classical planner. When plans fail, the system applies TextGrad-inspired textual-gradient updates to optimize each agent's prompt and thereby improve planning accuracy. In addition, meta-prompts are learned and shared across agents within the same layer, enabling efficient prompt optimization in multi-agent settings. On the MAT-THOR benchmark, our planner achieves success rates of 0.95 on compound tasks, 0.84 on complex tasks, and 0.60 on vague tasks, improving over the previous state-of-the-art LaMMA-P by 2, 7, and 15 percentage points respectively. An ablation study shows that the hierarchical structure, prompt optimization, and meta-prompt sharing contribute roughly +59, +37, and +4 percentage points to the overall success rate.

  </details>



- **ADM-DP: Adaptive Dynamic Modality Diffusion Policy through Vision-Tactile-Graph Fusion for Multi-Agent Manipulation**  
  Enyi Wang, Wen Fan, Dandan Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21622v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-agent robotic manipulation remains challenging due to the combined demands of coordination, grasp stability, and collision avoidance in shared workspaces. To address these challenges, we propose the Adaptive Dynamic Modality Diffusion Policy (ADM-DP), a framework that integrates vision, tactile, and graph-based (multi-agent pose) modalities for coordinated control. ADM-DP introduces four key innovations. First, an enhanced visual encoder merges RGB and point-cloud features via Feature-wise Linear Modulation (FiLM) modulation to enrich perception. Second, a tactile-guided grasping strategy uses Force-Sensitive Resistor (FSR) feedback to detect insufficient contact and trigger corrective grasp refinement, improving grasp stability. Third, a graph-based collision encoder leverages shared tool center point (TCP) positions of multiple agents as structured kinematic context to maintain spatial awareness and reduce inter-agent interference. Fourth, an Adaptive Modality Attention Mechanism (AMAM) dynamically re-weights modalities according to task context, enabling flexible fusion. For scalability and modularity, a decoupled training paradigm is employed in which agents learn independent policies while sharing spatial information. This maintains low interdependence between agents while retaining collective awareness. Across seven multi-agent tasks, ADM-DP achieves 12-25% performance gains over state-of-the-art baselines. Ablation studies show the greatest improvements in tasks requiring multiple sensory modalities, validating our adaptive fusion strategy and demonstrating its robustness for diverse manipulation scenarios.

  </details>



- **Position-Based Flocking for Persistent Alignment without Velocity Sensing**  
  Hossein B. Jond, Veli Bakırcıoğlu, Logan E. Beaver, Nejat Tükenmez, Adel Akbarimajd, Martin Saska  
  _2026-02-25_ · https://arxiv.org/abs/2602.22154v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Coordinated collective motion in bird flocks and fish schools inspires algorithms for cohesive swarm robotics. This paper presents a position-based flocking model that achieves persistent velocity alignment without velocity sensing. By approximating relative velocity differences from changes between current and initial relative positions and incorporating a time- and density-dependent alignment gain with a non-zero minimum threshold to maintain persistent alignment, the model sustains coherent collective motion over extended periods. Simulations with a collective of 50 agents demonstrate that the position-based flocking model attains faster and more sustained directional alignment and results in more compact formations than a velocity-alignment-based baseline. This position-based flocking model is particularly well-suited for real-world robotic swarms, where velocity measurements are unreliable, noisy, or unavailable. Experimental results using a team of nine real wheeled mobile robots are also presented.

  </details>



- **The Swarm Intelligence Freeway-Urban Trajectories (SWIFTraj) Dataset - Part II: A Graph-Based Approach for Trajectory Connection**  
  Xinkai Ji, Pan Liu, Yu Han  
  _2026-02-25_ · https://arxiv.org/abs/2602.21954v1 · `physics.soc-ph`  
  <details><summary>Abstract</summary>

  In Part I of this companion paper series, we introduced SWIFTraj, a new open-source vehicle trajectory dataset collected using a unmanned aerial vehicle (UAV) swarm. The dataset has two distinctive features. First, by connecting trajectories across consecutive UAV videos, it provides long-distance continuous trajectories, with the longest exceeding 4.5 km. Second, it covers an integrated traffic network consisting of both freeways and their connected urban roads. Obtaining such long-distance continuous trajectories from a UAV swarm is challenging, due to the need for accurate time alignment across multiple videos and the irregular spatial distribution of UAVs. To address these challenges, this paper proposes a novel graph-based approach for connecting vehicle trajectories captured by a UAV swarm. An undirected graph is constructed to represent flexible UAV layouts, and an automatic time alignment method based on trajectory matching cost minimization is developed to estimate optimal time offsets across videos. To associate trajectories of the same vehicle observed in different videos, a vehicle matching table is established using the Hungarian algorithm. The proposed approach is evaluated using both simulated and real-world data. Results from real-world experiments show that the time alignment error is within three video frames, corresponding to approximately 0.1 s, and that the vehicle matching achieves an F1-score of about 0.99. These results demonstrate the effectiveness of the proposed method in addressing key challenges in UAV-based trajectory connection and highlight its potential for large-scale vehicle trajectory collection.

  </details>



- **Leaky Coaxial Cable based Generalized Pinching-Antenna Systems with Dual-Port Feeding**  
  Kaidi Wang, Zhiguo Ding, Daniel K. C. So  
  _2026-02-25_ · https://arxiv.org/abs/2602.21856v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  By leveraging the distributed leakage radiation of leaky coaxial cables (LCXs), the concept of pinching antennas can be generalized from the conventional high-frequency waveguide based architectures to cable based structures in lower-frequency scenarios. This paper investigates an LCX based generalized pinching-antenna system with dual-port feeding. By enabling bidirectional excitation along each cable, the proposed design significantly enhances spatial degrees of freedom. A comprehensive channel model is developed to characterize intra-cable attenuation, bidirectional phase progression, slot based radiation, and wireless propagation. Based on this model, both analog and hybrid beamforming frameworks are studied with the objective of maximizing the minimum achievable data rate. For analog transmission, slot activation, port selection, and power allocation are jointly optimized using matching theory, coalitional games, and bisection based power control. For hybrid transmission, zero-forcing (ZF) digital precoding is incorporated to eliminate inter-user interference, thereby simplifying slot activation and enabling closed-form optimal power allocation. Simulation results demonstrate that dual-port feeding provides notable performance gains over single-port LCX systems and fixed-antenna benchmarks, validating the effectiveness of the proposed beamforming and resource allocation designs under various transmit power levels and cable parameters.

  </details>



- **Two-Stage Active Distribution Network Voltage Control via LLM-RL Collaboration: A Hybrid Knowledge-Data-Driven Approach**  
  Xu Yang, Chenhui Lin, Xiang Ma, Dong Liu, Ran Zheng, Haotian Liu, Wenchuan Wu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21715v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  The growing integration of distributed photovoltaics (PVs) into active distribution networks (ADNs) has exacerbated operational challenges, making it imperative to coordinate diverse equipment to mitigate voltage violations and enhance power quality. Although existing data-driven approaches have demonstrated effectiveness in the voltage control problem, they often require extensive trial-and-error exploration and struggle to incorporate heterogeneous information, such as day-ahead forecasts and semantic-based grid codes. Considering the operational scenarios and requirements in real-world ADNs, in this paper, we propose a hybrid knowledge-data-driven approach that leverages dynamic collaboration between a large language model (LLM) agent and a reinforcement learning (RL) agent to achieve two-stage voltage control. In the day-ahead stage, the LLM agent receives coarse region-level forecasts and generates scheduling strategies for on-load tap changer (OLTC) and shunt capacitors (SCs) to regulate the overall voltage profile. Then in the intra-day stage, based on accurate node-level measurements, the RL agent refines terminal voltages by deriving reactive power generation strategies for PV inverters. On top of the LLM-RL collaboration framework, we further propose a self-evolution mechanism for the LLM agent and a pretrain-finetune pipeline for the RL agent, effectively enhancing and coordinating the policies for both agents. The proposed approach not only aligns more closely with practical operational characteristics but also effectively utilizes the inherent knowledge and reasoning capabilities of the LLM agent, significantly improving training efficiency and voltage control performance. Comprehensive comparisons and ablation studies demonstrate the effectiveness of the proposed method.

  </details>


