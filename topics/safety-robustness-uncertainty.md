# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-03-07 06:56 UTC_

Total papers shown: **20**


---

- **VinePT-Map: Pole-Trunk Semantic Mapping for Resilient Autonomous Robotics in Vineyards**  
  Giorgio Audrito, Mauro Martini, Alessandro Navone, Giorgia Galluzzo, Marcello Chiaberge  
  _2026-03-05_ · https://arxiv.org/abs/2603.05070v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable long-term deployment of autonomous robots in agricultural environments remains challenging due to perceptual aliasing, seasonal variability, and the dynamic nature of crop canopies. Vineyards, characterized by repetitive row structures and significant visual changes across phenological stages, represent a pivotal field challenge, limiting the robustness of conventional feature-based localization and mapping approaches. This paper introduces VinePT-Map, a semantic mapping framework that leverages vine trunks and support poles as persistent structural landmarks to enable season-agnostic and resilient robot localization. The proposed method formulates the mapping problem as a factor graph, integrating GPS, IMU, and RGB-D observations through robust geometrical constraints that exploit vineyard structure. An efficient perception pipeline based on instance segmentation and tracking, combined with a clustering filter for outlier rejection and pose refinement, enables accurate landmark detection using low-cost sensors and onboard computation. To validate the pipeline, we present a multi-season dataset for trunk and pole segmentation and tracking. Extensive field experiments conducted across diverse seasons demonstrate the robustness and accuracy of the proposed approach, highlighting its suitability for long-term autonomous operation in agricultural environments.

  </details>



- **SPIRIT: Perceptive Shared Autonomy for Robust Robotic Manipulation under Deep Learning Uncertainty**  
  Jongseok Lee, Ribin Balachandran, Harsimran Singh, Jianxiang Feng, Hrishik Mishra, Marco De Stefano, Rudolph Triebel, Alin Albu-Schaeffer, Konstantin Kondak  
  _2026-03-05_ · https://arxiv.org/abs/2603.05111v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deep learning (DL) has enabled impressive advances in robotic perception, yet its limited robustness and lack of interpretability hinder reliable deployment in safety critical applications. We propose a concept termed perceptive shared autonomy, in which uncertainty estimates from DL based perception are used to regulate the level of autonomy. Specifically, when the robot's perception is confident, semi-autonomous manipulation is enabled to improve performance; when uncertainty increases, control transitions to haptic teleoperation for maintaining robustness. In this way, high-performing but uninterpretable DL methods can be integrated safely into robotic systems. A key technical enabler is an uncertainty aware DL based point cloud registration approach based on the so called Neural Tangent Kernels (NTK). We evaluate perceptive shared autonomy on challenging aerial manipulation tasks through a user study of 15 participants and realization of mock-up industrial scenarios, demonstrating reliable robotic manipulation despite failures in DL based perception. The resulting system, named SPIRIT, improves both manipulation performance and system reliability. SPIRIT was selected as a finalist of a major industrial innovation award.

  </details>



- **Omni-Manip: Beyond-FOV Large-Workspace Humanoid Manipulation with Omnidirectional 3D Perception**  
  Pei Qu, Zheng Li, Yufei Jia, Ziyun Liu, Liang Zhu, Haoang Li, Jinni Zhou, Jun Ma  
  _2026-03-05_ · https://arxiv.org/abs/2603.05355v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The deployment of humanoid robots for dexterous manipulation in unstructured environments remains challenging due to perceptual limitations that constrain the effective workspace. In scenarios where physical constraints prevent the robot from repositioning itself, maintaining omnidirectional awareness becomes far more critical than color or semantic information. While recent advances in visuomotor policy learning have improved manipulation capabilities, conventional RGB-D solutions suffer from narrow fields of view (FOV) and self-occlusion, requiring frequent base movements that introduce motion uncertainty and safety risks. Existing approaches to expanding perception, including active vision systems and third-view cameras, introduce mechanical complexity, calibration dependencies, and latency that hinder reliable real-time performance. In this work, We propose Omni-Manip, an end-to-end LiDAR-driven 3D visuomotor policy that enables robust manipulation in large workspaces. Our method processes panoramic point clouds through a Time-Aware Attention Pooling mechanism, efficiently encoding sparse 3D data while capturing temporal dependencies. This 360° perception allows the robot to interact with objects across wide areas without frequent repositioning. To support policy learning, we develop a whole-body teleoperation system for efficient data collection on full-body coordination. Extensive experiments in simulation and real-world environments show that Omni-Manip achieves robust performance in large-workspace and cluttered scenarios, outperforming baselines that rely on egocentric depth cameras.

  </details>



- **CT-Enabled Patient-Specific Simulation and Contact-Aware Robotic Planning for Cochlear Implantation**  
  Lingxiao Xun, Gang Zheng, Alexandre Kruszewski, Renato Torres  
  _2026-03-05_ · https://arxiv.org/abs/2603.05333v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic cochlear-implant (CI) insertion requires precise prediction and regulation of contact forces to minimize intracochlear trauma and prevent failure modes such as locking and buckling. Aligned with the integration of advanced medical imaging and robotics for autonomous, precision interventions, this paper presents a unified CT-to-simulation pipeline for contact-aware insertion planning and validation. We develop a low-dimensional, differentiable Cosserat-rod model of the electrode array coupled with frictional contact and pseudo-dynamics regularization to ensure continuous stick-slip transitions. Patient-specific cochlear anatomy is reconstructed from CT imaging and encoded via an analytic parametrization of the scala-tympani lumen, enabling efficient and differentiable contact queries through closest-point projection. Based on a differentiated equilibrium-constraint formulation, we derive an online direction-update law under an RCM-like constraint that suppresses lateral insertion forces while maintaining axial advancement. Simulations and benchtop experiments validate deformation and force trends, demonstrating reduced locking/buckling risk and improved insertion depth. The study highlights how CT-based imaging enhances modeling, planning, and safety capabilities in robot-assisted inner-ear procedures.

  </details>



- **Safe-SAGE: Social-Semantic Adaptive Guidance for Safe Engagement through Laplace-Modulated Poisson Safety Functions**  
  Lizhi Yang, Ryan M. Bena, Meg Wilkinson, Gilbert Bahati, Andy Navarro Brenes, Ryan K. Cosner, Aaron D. Ames  
  _2026-03-05_ · https://arxiv.org/abs/2603.05497v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Traditional safety-critical control methods, such as control barrier functions, suffer from semantic blindness, exhibiting the same behavior around obstacles regardless of contextual significance. This limitation leads to the uniform treatment of all obstacles, despite their differing semantic meanings. We present Safe-SAGE (Social-Semantic Adaptive Guidance for Safe Engagement), a unified framework that bridges the gap between high-level semantic understanding and low-level safety-critical control through a Poisson safety function (PSF) modulated using a Laplace guidance field. Our approach perceives the environment by fusing multi-sensor point clouds with vision-based instance segmentation and persistent object tracking to maintain up-to-date semantics beyond the camera's field of view. A multi-layer safety filter is then used to modulate system inputs to achieve safe navigation using this semantic understanding of the environment. This safety filter consists of both a model predictive control layer and a control barrier function layer. Both layers utilize the PSF and flux modulation of the guidance field to introduce varying levels of conservatism and multi-agent passing norms for different obstacles in the environment. Our framework enables legged robots to navigate semantically rich, dynamic environments with context-dependent safety margins while maintaining rigorous safety guarantees.

  </details>



- **Critic in the Loop: A Tri-System VLA Framework for Robust Long-Horizon Manipulation**  
  Pengfei Yi, Yingjie Ma, Wenjiang Xu, Yanan Hao, Shuai Gan, Wanting Li, Shanlin Zhong  
  _2026-03-05_ · https://arxiv.org/abs/2603.05185v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Balancing high-level semantic reasoning with low-level reactive control remains a core challenge in visual robotic manipulation. While Vision-Language Models (VLMs) excel at cognitive planning, their inference latency precludes real-time execution. Conversely, fast Vision-Language-Action (VLA) models often lack the semantic depth required for complex, long-horizon tasks. To bridge this gap, we introduce Critic in the Loop, an adaptive hierarchical framework driven by dynamic VLM-Expert scheduling. At its core is a bionic Tri-System architecture comprising a VLM brain for global reasoning, a VLA cerebellum for reactive execution, and a lightweight visual Critic. By continuously monitoring the workspace, the Critic dynamically routes control authority. It sustains rapid closed-loop execution via the VLA for routine subtasks, and adaptively triggers the VLM for replanning upon detecting execution anomalies such as task stagnation or failures. Furthermore, our architecture seamlessly integrates human-inspired rules to intuitively break infinite retry loops. This visually-grounded scheduling minimizes expensive VLM queries, while substantially enhancing system robustness and autonomy in out-of-distribution (OOD) scenarios. Comprehensive experiments on challenging, long-horizon manipulation benchmarks reveal that our approach achieves state-of-the-art performance.

  </details>



- **cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots**  
  Balakumar Sundaralingam, Adithyavairavan Murali, Stan Birchfield  
  _2026-03-05_ · https://arxiv.org/abs/2603.05493v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Effective robot autonomy requires motion generation that is safe, feasible, and reactive. Current methods are fragmented: fast planners output physically unexecutable trajectories, reactive controllers struggle with high-fidelity perception, and existing solvers fail on high-DoF systems. We present cuRoboV2, a unified framework with three key innovations: (1) B-spline trajectory optimization that enforces smoothness and torque limits; (2) a GPU-native TSDF/ESDF perception pipeline that generates dense signed distance fields covering the full workspace, unlike existing methods that only provide distances within sparsely allocated blocks, up to 10x faster and in 8x less memory than the state-of-the-art at manipulation scale, with up to 99% collision recall; and (3) scalable GPU-native whole-body computation, namely topology-aware kinematics, differentiable inverse dynamics, and map-reduce self-collision, that achieves up to 61x speedup while also extending to high-DoF humanoids (where previous GPU implementations fail). On benchmarks, cuRoboV2 achieves 99.7% success under 3kg payload (where baselines achieve only 72--77%), 99.6% collision-free IK on a 48-DoF humanoid (where prior methods fail entirely), and 89.5% retargeting constraint satisfaction (vs. 61% for PyRoki); these collision-free motions yield locomotion policies with 21% lower tracking error than PyRoki and 12x lower cross-seed variance than mink. A ground-up codebase redesign for discoverability enabled LLM coding assistants to author up to 73% of new modules, including hand-optimized CUDA kernels, demonstrating that well-structured robotics code can unlock productive human--LLM collaboration. Together, these advances provide a unified, dynamics-aware motion generation stack that scales from single-arm manipulators to full humanoids.

  </details>



- **Building AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned**  
  Nghi D. Q. Bui  
  _2026-03-05_ · https://arxiv.org/abs/2603.05344v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The landscape of AI coding assistance is undergoing a fundamental shift from complex IDE plugins to versatile, terminal-native agents. Operating directly where developers manage source control, execute builds, and deploy environments, CLI-based agents offer unprecedented autonomy for long-horizon development tasks. In this paper, we present OPENDEV, an open-source, command-line coding agent engineered specifically for this new paradigm. Effective autonomous assistance requires strict safety controls and highly efficient context management to prevent context bloat and reasoning degradation. OPENDEV overcomes these challenges through a compound AI system architecture with workload-specialized model routing, a dual-agent architecture separating planning from execution, lazy tool discovery, and adaptive context compaction that progressively reduces older observations. Furthermore, it employs an automated memory system to accumulate project-specific knowledge across sessions and counteracts instruction fade-out through event-driven system reminders. By enforcing explicit reasoning phases and prioritizing context efficiency, OPENDEV provides a secure, extensible foundation for terminal-first AI assistance, offering a blueprint for robust autonomous software engineering.

  </details>



- **Decoupling Task and Behavior: A Two-Stage Reward Curriculum in Reinforcement Learning for Robotics**  
  Kilian Freitag, Knut Åkesson, Morteza Haghir Chehreghani  
  _2026-03-05_ · https://arxiv.org/abs/2603.05113v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Deep Reinforcement Learning is a promising tool for robotic control, yet practical application is often hindered by the difficulty of designing effective reward functions. Real-world tasks typically require optimizing multiple objectives simultaneously, necessitating precise tuning of their weights to learn a policy with the desired characteristics. To address this, we propose a two-stage reward curriculum where we decouple task-specific objectives from behavioral terms. In our method, we first train the agent on a simplified task-only reward function to ensure effective exploration before introducing the full reward that includes auxiliary behavior-related terms such as energy efficiency. Further, we analyze various transition strategies and demonstrate that reusing samples between phases is critical for training stability. We validate our approach on the DeepMind Control Suite, ManiSkill3, and a mobile robot environment, modified to include auxiliary behavioral objectives. Our method proves to be simple yet effective, substantially outperforming baselines trained directly on the full reward while exhibiting higher robustness to specific reward weightings.

  </details>



- **GaussTwin: Unified Simulation and Correction with Gaussian Splatting for Robotic Digital Twins**  
  Yichen Cai, Paul Jansonnie, Cristiana de Farias, Oleg Arenz, Jan Peters  
  _2026-03-05_ · https://arxiv.org/abs/2603.05108v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Digital twins promise to enhance robotic manipulation by maintaining a consistent link between real-world perception and simulation. However, most existing systems struggle with the lack of a unified model, complex dynamic interactions, and the real-to-sim gap, which limits downstream applications such as model predictive control. Thus, we propose GaussTwin, a real-time digital twin that combines position-based dynamics with discrete Cosserat rod formulations for physically grounded simulation, and Gaussian splatting for efficient rendering and visual correction. By anchoring Gaussians to physical primitives and enforcing coherent SE(3) updates driven by photometric error and segmentation masks, GaussTwin achieves stable prediction-correction while preserving physical fidelity. Through experiments in both simulation and on a Franka Research 3 platform, we show that GaussTwin consistently improves tracking accuracy and robustness compared to shape-matching and rigid-only baselines, while also enabling downstream tasks such as push-based planning. These results highlight GaussTwin as a step toward unified, physically meaningful digital twins that can support closed-loop robotic interaction and learning.

  </details>



- **Act, Think or Abstain: Complexity-Aware Adaptive Inference for Vision-Language-Action Models**  
  Riccardo Andrea Izzo, Gianluca Bardaro, Matteo Matteucci  
  _2026-03-05_ · https://arxiv.org/abs/2603.05147v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Current research on Vision-Language-Action (VLA) models predominantly focuses on enhancing generalization through established reasoning techniques. While effective, these improvements invariably increase computational complexity and inference latency. Furthermore, these mechanisms are typically applied indiscriminately, resulting in the inefficient allocation of resources for trivial tasks while simultaneously failing to provide the uncertainty estimation necessary to prevent catastrophic failure on out-of-distribution tasks. Inspired by human cognition, we propose an adaptive framework that dynamically routes VLA execution based on the complexity of the perceived state. Our approach transforms the VLA's vision-language backbone into an active detection tool by projecting latent embeddings into an ensemble of parametric and non-parametric estimators. This allows the system to execute known tasks immediately (Act), reason about ambiguous scenarios (Think), and preemptively halt execution when encountering significant physical or semantic anomalies (Abstain). In our empirical analysis, we observe a phenomenon where visual embeddings alone are superior for inferring task complexity due to the semantic invariance of language. Evaluated on the LIBERO and LIBERO-PRO benchmarks as well as on a real robot, our vision-only configuration achieves 80% F1-Score using as little as 5% of training data, establishing itself as a reliable and efficient task complexity detector.

  </details>



- **A 360-degree Multi-camera System for Blue Emergency Light Detection Using Color Attention RT-DETR and the ABLDataset**  
  Francisco Vacalebri-Lloret, Lucas Banchero, Jose J. Lopez, Jose M. Mossi  
  _2026-03-05_ · https://arxiv.org/abs/2603.05058v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This study presents an advanced system for detecting blue lights on emergency vehicles, developed using ABLDataset, a curated dataset that includes images of European emergency vehicles under various climatic and geographic conditions. The system employs a configuration of four fisheye cameras, each with a 180-degree horizontal field of view, mounted on the sides of the vehicle. A calibration process enables the azimuthal localization of the detections. Additionally, a comparative analysis of major deep neural network algorithms was conducted, including YOLO (v5, v8, and v10), RetinaNet, Faster R-CNN, and RT-DETR. RT-DETR was selected as the base model and enhanced through the incorporation of a color attention block, achieving an accuracy of 94.7 percent and a recall of 94.1 percent on the test set, with field test detections reaching up to 70 meters. Furthermore, the system estimates the approach angle of the emergency vehicle relative to the center of the car using geometric transformations. Designed for integration into a multimodal system that combines visual and acoustic data, this system has demonstrated high efficiency, offering a promising approach to enhancing Advanced Driver Assistance Systems (ADAS) and road safety.

  </details>



- **Loop Closure via Maximal Cliques in 3D LiDAR-Based SLAM**  
  Javier Laserna, Saurabh Gupta, Oscar Martinez Mozos, Cyrill Stachniss, Pablo San Segundo  
  _2026-03-05_ · https://arxiv.org/abs/2603.05397v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable loop closure detection remains a critical challenge in 3D LiDAR-based SLAM, especially under sensor noise, environmental ambiguity, and viewpoint variation conditions. RANSAC is often used in the context of loop closures for geometric model fitting in the presence of outliers. However, this approach may fail, leading to map inconsistency. We introduce a novel deterministic algorithm, CliReg, for loop closure validation that replaces RANSAC verification with a maximal clique search over a compatibility graph of feature correspondences. This formulation avoids random sampling and increases robustness in the presence of noise and outliers. We integrated our approach into a real- time pipeline employing binary 3D descriptors and a Hamming distance embedding binary search tree-based matching. We evaluated it on multiple real-world datasets featuring diverse LiDAR sensors. The results demonstrate that our proposed technique consistently achieves a lower pose error and more reliable loop closures than RANSAC, especially in sparse or ambiguous conditions. Additional experiments on 2D projection-based maps confirm its generality across spatial domains, making our approach a robust and efficient alternative for loop closure detection.

  </details>



- **S5-SHB Agent: Society 5.0 enabled Multi-model Agentic Blockchain Framework for Smart Home**  
  Janani Rangila, Akila Siriweera, Incheon Paik, Keitaro Naruse, Isuru Jayanada, Vishmika Devindi  
  _2026-03-05_ · https://arxiv.org/abs/2603.05027v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The smart home is a key application domain within the Society 5.0 vision for a human-centered society. As smart home ecosystems expand with heterogeneous IoT protocols, diverse devices, and evolving threats, autonomous systems must manage comfort, security, energy, and safety for residents. Such autonomous decision-making requires a trust anchor, making blockchain a preferred foundation for transparent and accountable smart home governance. However, realizing this vision requires blockchain-governed smart homes to simultaneously address adaptive consensus, intelligent multi-agent coordination, and resident-controlled governance aligned with the principles of Society 5.0. Existing frameworks rely solely on rigid smart contracts with fixed consensus protocols, employ at most a single AI model without multi-agent coordination, and offer no governance mechanism for residents to control automation behaviour. To address these limitations, this paper presents the Society 5.0-driven human-centered governance-enabled smart home blockchain agent (S5-SHB-Agent). The framework orchestrates ten specialized agents using interchangeable large language models to make decisions across the safety, security, comfort, energy, privacy, and health domains. An adaptive PoW blockchain adjusts mining difficulty based on transaction volume and emergency conditions, with digital signatures and Merkle tree anchoring to ensure tamper evident auditability. A four-tier governance model enables residents to control automation through tiered preferences from routine adjustments to immutable safety thresholds. Evaluation confirms that resident governance correctly separates adjustable comfort priorities from immutable safety thresholds across all tested configurations, while adaptive consensus commits emergency blocks.

  </details>



- **Towards Multimodal Lifelong Understanding: A Dataset and Agentic Baseline**  
  Guo Chen, Lidong Lu, Yicheng Liu, Liangrui Dong, Lidong Zou, Jixin Lv, Zhenquan Li, Xinyi Mao, Baoqi Pei, Shihao Wang, et al.  
  _2026-03-05_ · https://arxiv.org/abs/2603.05484v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  While datasets for video understanding have scaled to hour-long durations, they typically consist of densely concatenated clips that differ from natural, unscripted daily life. To bridge this gap, we introduce MM-Lifelong, a dataset designed for Multimodal Lifelong Understanding. Comprising 181.1 hours of footage, it is structured across Day, Week, and Month scales to capture varying temporal densities. Extensive evaluations reveal two critical failure modes in current paradigms: end-to-end MLLMs suffer from a Working Memory Bottleneck due to context saturation, while representative agentic baselines experience Global Localization Collapse when navigating sparse, month-long timelines. To address this, we propose the Recursive Multimodal Agent (ReMA), which employs dynamic memory management to iteratively update a recursive belief state, significantly outperforming existing methods. Finally, we establish dataset splits designed to isolate temporal and domain biases, providing a rigorous foundation for future research in supervised learning and out-of-distribution generalization.

  </details>



- **EdgeDAM: Real-time Object Tracking for Mobile Devices**  
  Syed Muhammad Raza, Syed Murtaza Hussain Abidi, Khawar Islam, Muhammad Ibrahim, Ajmal Saeed Mian  
  _2026-03-05_ · https://arxiv.org/abs/2603.05463v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Single-object tracking (SOT) on edge devices is a critical computer vision task, requiring accurate and continuous target localization across video frames under occlusion, distractor interference, and fast motion. However, recent state-of-the-art distractor-aware memory mechanisms are largely built on segmentation-based trackers and rely on mask prediction and attention-driven memory updates, which introduce substantial computational overhead and limit real-time deployment on resource-constrained hardware; meanwhile, lightweight trackers sustain high throughput but are prone to drift when visually similar distractors appear. To address these challenges, we propose EdgeDAM, a lightweight detection-guided tracking framework that reformulates distractor-aware memory for bounding-box tracking under strict edge constraints. EdgeDAM introduces two key strategies: (1) Dual-Buffer Distractor-Aware Memory (DAM), which integrates a Recent-Aware Memory to preserve temporally consistent target hypotheses and a Distractor-Resolving Memory to explicitly store hard negative candidates and penalize their re-selection during recovery; and (2) Confidence-Driven Switching with Held-Box Stabilization, where tracker reliability and temporal consistency criteria adaptively activate detection and memory-guided re-identification during occlusion, while a held-box mechanism temporarily freezes and expands the estimate to suppress distractor contamination. Extensive experiments on five benchmarks, including the distractor-focused DiDi dataset, demonstrate improved robustness under occlusion and fast motion while maintaining real-time performance on mobile devices, achieving 88.2% accuracy on DiDi and 25 FPS on an iPhone 15. Code will be released.

  </details>



- **Residual RL--MPC for Robust Microrobotic Cell Pushing Under Time-Varying Flow**  
  Yanda Yang, Sambeeta Das  
  _2026-03-05_ · https://arxiv.org/abs/2603.05448v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Contact-rich micromanipulation in microfluidic flow is challenging because small disturbances can break pushing contact and induce large lateral drift. We study planar cell pushing with a magnetic rolling microrobot that tracks a waypoint-sampled reference curve under time-varying Poiseuille flow. We propose a hybrid controller that augments a nominal MPC with a learned residual policy trained by SAC. The policy outputs a bounded 2D velocity correction that is contact-gated, so residual actions are applied only during robot--cell contact, preserving reliable approach behavior and stabilizing learning. All methods share the same actuation interface and speed envelope for fair comparisons. Experiments show improved robustness and tracking accuracy over pure MPC and PID under nonstationary flow, with generalization from a clover training curve to unseen circle and square trajectories. A residual-bound sweep identifies an intermediate correction limit as the best trade-off, which we use in all benchmarks.

  </details>



- **X-RAY: Mapping LLM Reasoning Capability via Formalized and Calibrated Probes**  
  Gao Tianxi, Cai Yufan, Yuan Yusi, Dong Jin Song  
  _2026-03-05_ · https://arxiv.org/abs/2603.05290v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) achieve promising performance, yet their ability to reason remains poorly understood. Existing evaluations largely emphasize task-level accuracy, often conflating pattern matching with reasoning capability. We present X-RAY, an explainable reasoning analysis system that maps the LLM reasoning capability using calibrated, formally verified probes. We model reasoning capability as a function of extractable \textit{structure}, operationalized through formal properties such as constraint interaction, reasoning depth, and solution-space geometry. X-Ray generates probes via formal tools with controlled structural variations, enabling precise isolation of incremental structural information through formal calibration and verification. We evaluate state-of-the-art LLMs on problems ranging from junior-level to advanced in mathematics, physics, and chemistry. Our analysis reveals a systematic asymmetry in LLM reasoning: models are relatively robust to constraint refinement, where additional conditions shrink an existing solution space, but degrade sharply under solution-space restructuring, where modifications alter the underlying structural form of the solution manifold. Moreover, calibrated formal probes differentiate models that appear indistinguishable on standard benchmarks and reveal failure modes that are structurally interpretable rather than opaque. Beyond evaluation, our framework is contamination-free and supports the training and testing of reasoning models.

  </details>



- **From Code to Road: A Vehicle-in-the-Loop and Digital Twin-Based Framework for Central Car Server Testing in Autonomous Driving**  
  Chengdong Wu, Sven Kirchner, Nils Purschke, Axel Torschmied, Norbert Kroth, Yinglei Song, André Schamschurko, Erik Leo Haß, Kuo-Yi Chao, Yi Zhang, et al.  
  _2026-03-05_ · https://arxiv.org/abs/2603.05279v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Simulation is one of the most essential parts in the development stage of automotive software. However, purely virtual simulations often struggle to accurately capture all real-world factors due to limitations in modeling. To address this challenge, this work presents a test framework for automotive software on the centralized E/E architecture, which is a central car server in our case, based on Vehicle-in-the-Loop (ViL) and digital twin technology. The framework couples a physical test vehicle on a dynamometer test bench with its synchronized virtual counterpart in a simulation environment. Our approach provides a safe, reproducible, realistic, and cost-effective platform for validating autonomous driving algorithms with a centralized architecture. This test method eliminates the need to test individual physical ECUs and their communication protocols separately. In contrast to traditional ViL methods, the proposed framework runs the full autonomous driving software directly on the vehicle hardware after the simulation process, eliminating flashing and intermediate layers while enabling seamless virtual-physical integration and accurately reflecting centralized E/E behavior. In addition, incorporating mixed testing in both simulated and physical environments reduces the need for full hardware integration during the early stages of automotive development. Experimental case studies demonstrate the effectiveness of the framework in different test scenarios. These findings highlight the potential to reduce development and integration efforts for testing autonomous driving pipelines in the future.

  </details>



- **Beyond Word Error Rate: Auditing the Diversity Tax in Speech Recognition through Dataset Cartography**  
  Ting-Hui Cheng, Line H. Clemmensen, Sneha Das  
  _2026-03-05_ · https://arxiv.org/abs/2603.05267v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Automatic speech recognition (ASR) systems are predominantly evaluated using the Word Error Rate (WER). However, raw token-level metrics fail to capture semantic fidelity and routinely obscures the `diversity tax', the disproportionate burden on marginalized and atypical speaker due to systematic recognition failures. In this paper, we explore the limitations of relying solely on lexical counts by systematically evaluating a broader class of non-linear and semantic metrics. To enable rigorous model auditing, we introduce the sample difficulty index (SDI), a novel metric that quantifies how intrinsic demographic and acoustic factors drive model failure. By mapping SDI on data cartography, we demonstrate that metrics EmbER and SemDist expose hidden systemic biases and inter-model disagreements that WER ignores. Finally, our findings are the first steps towards a robust audit framework for prospective safety analysis, empowering developers to audit and mitigate ASR disparities prior to deployment.

  </details>


