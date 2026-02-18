# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-18 07:14 UTC_

Total papers shown: **9**


---

- **SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms**  
  Haichao Liu, Yufeng Hu, Shuang Wang, Kangjun Guo, Jun Ma, Jinni Zhou  
  _2026-02-17_ · https://arxiv.org/abs/2602.15633v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous landing of Uncrewed Aerial Vehicles (UAVs) on oscillating marine platforms is severely constrained by wave-induced multi-frequency oscillations, wind disturbances, and prediction phase lags in motion prediction. Existing methods either treat platform motion as a general random process or lack explicit modeling of wave spectral characteristics, leading to suboptimal performance under dynamic sea conditions. To address these limitations, we propose SpecFuse: a novel spectral-temporal fusion predictive control framework that integrates frequency-domain wave decomposition with time-domain recursive state estimation for high-precision 6-DoF motion forecasting of Uncrewed Surface Vehicles (USVs). The framework explicitly models dominant wave harmonics to mitigate phase lags, refining predictions in real time via IMU data without relying on complex calibration. Additionally, we design a hierarchical control architecture featuring a sampling-based HPO-RRT* algorithm for dynamic trajectory planning under non-convex constraints and a learning-augmented predictive controller that fuses data-driven disturbance compensation with optimization-based execution. Extensive validations (2,000 simulations + 8 lake experiments) show our approach achieves a 3.2 cm prediction error, 4.46 cm landing deviation, 98.7% / 87.5% success rates (simulation / real-world), and 82 ms latency on embedded hardware, outperforming state-of-the-art methods by 44%-48% in accuracy. Its robustness to wave-wind coupling disturbances supports critical maritime missions such as search and rescue and environmental monitoring. All code, experimental configurations, and datasets will be released as open-source to facilitate reproducibility.

  </details>



- **Hybrid F' and ROS2 Architecture for Vision-Based Autonomous Flight: Design and Experimental Validation**  
  Abdelrahman Metwally, Monijesu James, Aleksey Fedoseev, Miguel Altamirano Cabrera, Dzmitry Tsetserukou, Andrey Somov  
  _2026-02-17_ · https://arxiv.org/abs/2602.15398v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous aerospace systems require architectures that balance deterministic real-time control with advanced perception capabilities. This paper presents an integrated system combining NASA's F' flight software framework with ROS2 middleware via Protocol Buffers bridging. We evaluate the architecture through a 32.25-minute indoor quadrotor flight test using vision-based navigation. The vision system achieved 87.19 Hz position estimation with 99.90\% data continuity and 11.47 ms mean latency, validating real-time performance requirements. All 15 ground commands executed successfully with 100 % success rate, demonstrating robust F'--PX4 integration. System resource utilization remained low (15.19 % CPU, 1,244 MB RAM) with zero stale telemetry messages, confirming efficient operation on embedded platforms. Results validate the feasibility of hybrid flight-software architectures combining certification-grade determinism with flexible autonomy for autonomous aerial vehicles.

  </details>



- **Fluoroscopy-Constrained Magnetic Robot Control via Zernike-Based Field Modeling and Nonlinear MPC**  
  Xinhao Chen, Hongkun Yao, Anuruddha Bhattacharjee, Suraj Raval, Lamar O. Mair, Yancy Diaz-Mercado, Axel Krieger  
  _2026-02-17_ · https://arxiv.org/abs/2602.15357v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Magnetic actuation enables surgical robots to navigate complex anatomical pathways while reducing tissue trauma and improving surgical precision. However, clinical deployment is limited by the challenges of controlling such systems under fluoroscopic imaging, which provides low frame rate and noisy pose feedback. This paper presents a control framework that remains accurate and stable under such conditions by combining a nonlinear model predictive control (NMPC) framework that directly outputs coil currents, an analytically differentiable magnetic field model based on Zernike polynomials, and a Kalman filter to estimate the robot state. Experimental validation is conducted with two magnetic robots in a 3D-printed fluid workspace and a spine phantom replicating drug delivery in the epidural space. Results show the proposed control method remains highly accurate when feedback is downsampled to 3 Hz with added Gaussian noise (sigma = 2 mm), mimicking clinical fluoroscopy. In the spine phantom experiments, the proposed method successfully executed a drug delivery trajectory with a root mean square (RMS) position error of 1.18 mm while maintaining safe clearance from critical anatomical boundaries.

  </details>



- **One Agent to Guide Them All: Empowering MLLMs for Vision-and-Language Navigation via Explicit World Representation**  
  Zerui Li, Hongpei Zheng, Fangguo Zhao, Aidan Chan, Jian Zhou, Sihao Lin, Shijie Li, Qi Wu  
  _2026-02-17_ · https://arxiv.org/abs/2602.15400v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  A navigable agent needs to understand both high-level semantic instructions and precise spatial perceptions. Building navigation agents centered on Multimodal Large Language Models (MLLMs) demonstrates a promising solution due to their powerful generalization ability. However, the current tightly coupled design dramatically limits system performance. In this work, we propose a decoupled design that separates low-level spatial state estimation from high-level semantic planning. Unlike previous methods that rely on predefined, oversimplified textual maps, we introduce an interactive metric world representation that maintains rich and consistent information, allowing MLLMs to interact with and reason on it for decision-making. Furthermore, counterfactual reasoning is introduced to further elicit MLLMs' capacity, while the metric world representation ensures the physical validity of the produced actions. We conduct comprehensive experiments in both simulated and real-world environments. Our method establishes a new zero-shot state-of-the-art, achieving 48.8\% Success Rate (SR) in R2R-CE and 42.2\% in RxR-CE benchmarks. Furthermore, to validate the versatility of our metric representation, we demonstrate zero-shot sim-to-real transfer across diverse embodiments, including a wheeled TurtleBot 4 and a custom-built aerial drone. These real-world deployments verify that our decoupled framework serves as a robust, domain-invariant interface for embodied Vision-and-Language navigation.

  </details>



- **When Remembering and Planning are Worth it: Navigating under Change**  
  Omid Madani, J. Brian Burns, Reza Eghbali, Thomas L. Dean  
  _2026-02-17_ · https://arxiv.org/abs/2602.15274v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We explore how different types and uses of memory can aid spatial navigation in changing uncertain environments. In the simple foraging task we study, every day, our agent has to find its way from its home, through barriers, to food. Moreover, the world is non-stationary: from day to day, the location of the barriers and food may change, and the agent's sensing such as its location information is uncertain and very limited. Any model construction, such as a map, and use, such as planning, needs to be robust against these challenges, and if any learning is to be useful, it needs to be adequately fast. We look at a range of strategies, from simple to sophisticated, with various uses of memory and learning. We find that an architecture that can incorporate multiple strategies is required to handle (sub)tasks of a different nature, in particular for exploration and search, when food location is not known, and for planning a good path to a remembered (likely) food location. An agent that utilizes non-stationary probability learning techniques to keep updating its (episodic) memories and that uses those memories to build maps and plan on the fly (imperfect maps, i.e. noisy and limited to the agent's experience) can be increasingly and substantially more efficient than the simpler (minimal-memory) agents, as the task difficulties such as distance to goal are raised, as long as the uncertainty, from localization and change, is not too large.

  </details>



- **VLM-DEWM: Dynamic External World Model for Verifiable and Resilient Vision-Language Planning in Manufacturing**  
  Guoqin Tang, Qingxuan Jia, Gang Chen, Tong Li, Zeyuan Huang, Zihang Lv, Ning Ji  
  _2026-02-17_ · https://arxiv.org/abs/2602.15549v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-language model (VLM) shows promise for high-level planning in smart manufacturing, yet their deployment in dynamic workcells faces two critical challenges: (1) stateless operation, they cannot persistently track out-of-view states, causing world-state drift; and (2) opaque reasoning, failures are difficult to diagnose, leading to costly blind retries. This paper presents VLM-DEWM, a cognitive architecture that decouples VLM reasoning from world-state management through a persistent, queryable Dynamic External World Model (DEWM). Each VLM decision is structured into an Externalizable Reasoning Trace (ERT), comprising action proposal, world belief, and causal assumption, which is validated against DEWM before execution. When failures occur, discrepancy analysis between predicted and observed states enables targeted recovery instead of global replanning. We evaluate VLM-DEWM on multi-station assembly, large-scale facility exploration, and real-robot recovery under induced failures. Compared to baseline memory-augmented VLM systems, VLM DEWM improves state-tracking accuracy from 56% to 93%, increases recovery success rate from below 5% to 95%, and significantly reduces computational overhead through structured memory. These results establish VLM-DEWM as a verifiable and resilient solution for long-horizon robotic operations in dynamic manufacturing environments.

  </details>



- **On the Out-of-Distribution Generalization of Reasoning in Multimodal LLMs for Simple Visual Planning Tasks**  
  Yannic Neuhaus, Nicolas Flammarion, Matthias Hein, Francesco Croce  
  _2026-02-17_ · https://arxiv.org/abs/2602.15460v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Integrating reasoning in large language models and large vision-language models has recently led to significant improvement of their capabilities. However, the generalization of reasoning models is still vaguely defined and poorly understood. In this work, we present an evaluation framework to rigorously examine how well chain-of-thought (CoT) approaches generalize on a simple planning task. Specifically, we consider a grid-based navigation task in which a model is provided with a map and must output a sequence of moves that guides a player from a start position to a goal while avoiding obstacles. The versatility of the task and its data allows us to fine-tune model variants using different input representations (visual and textual) and CoT reasoning strategies, and systematically evaluate them under both in-distribution (ID) and out-of-distribution (OOD) test conditions. Our experiments show that, while CoT reasoning improves in-distribution generalization across all representations, out-of-distribution generalization (e.g., to larger maps) remains very limited in most cases when controlling for trivial matches with the ID data. Surprisingly, we find that reasoning traces which combine multiple text formats yield the best (and non-trivial) OOD generalization. Finally, purely text-based models consistently outperform those utilizing image-based inputs, including a recently proposed approach relying on latent space reasoning.

  </details>



- **FAST-EQA: Efficient Embodied Question Answering with Global and Local Region Relevancy**  
  Haochen Zhang, Nirav Savaliya, Faizan Siddiqui, Enna Sachdeva  
  _2026-02-17_ · https://arxiv.org/abs/2602.15813v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Embodied Question Answering (EQA) combines visual scene understanding, goal-directed exploration, spatial and temporal reasoning under partial observability. A central challenge is to confine physical search to question-relevant subspaces while maintaining a compact, actionable memory of observations. Furthermore, for real-world deployment, fast inference time during exploration is crucial. We introduce FAST-EQA, a question-conditioned framework that (i) identifies likely visual targets, (ii) scores global regions of interest to guide navigation, and (iii) employs Chain-of-Thought (CoT) reasoning over visual memory to answer confidently. FAST-EQA maintains a bounded scene memory that stores a fixed-capacity set of region-target hypotheses and updates them online, enabling robust handling of both single and multi-target questions without unbounded growth. To expand coverage efficiently, a global exploration policy treats narrow openings and doors as high-value frontiers, complementing local target seeking with minimal computation. Together, these components focus the agent's attention, improve scene coverage, and improve answer reliability while running substantially faster than prior approaches. On HMEQA and EXPRESS-Bench, FAST-EQA achieves state-of-the-art performance, while performing competitively on OpenEQA and MT-HM3D.

  </details>



- **Learning to Retrieve Navigable Candidates for Efficient Vision-and-Language Navigation**  
  Shutian Gu, Chengkai Huang, Ruoyu Wang, Lina Yao  
  _2026-02-17_ · https://arxiv.org/abs/2602.15724v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-and-Language Navigation (VLN) requires an agent to follow natural-language instructions and navigate through previously unseen environments. Recent approaches increasingly employ large language models (LLMs) as high-level navigators due to their flexibility and reasoning capability. However, prompt-based LLM navigation often suffers from inefficient decision-making, as the model must repeatedly interpret instructions from scratch and reason over noisy and verbose navigable candidates at each step. In this paper, we propose a retrieval-augmented framework to improve the efficiency and stability of LLM-based VLN without modifying or fine-tuning the underlying language model. Our approach introduces retrieval at two complementary levels. At the episode level, an instruction-level embedding retriever selects semantically similar successful navigation trajectories as in-context exemplars, providing task-specific priors for instruction grounding. At the step level, an imitation-learned candidate retriever prunes irrelevant navigable directions before LLM inference, reducing action ambiguity and prompt complexity. Both retrieval modules are lightweight, modular, and trained independently of the LLM. We evaluate our method on the Room-to-Room (R2R) benchmark. Experimental results demonstrate consistent improvements in Success Rate, Oracle Success Rate, and SPL on both seen and unseen environments. Ablation studies further show that instruction-level exemplar retrieval and candidate pruning contribute complementary benefits to global guidance and step-wise decision efficiency. These results indicate that retrieval-augmented decision support is an effective and scalable strategy for enhancing LLM-based vision-and-language navigation.

  </details>


