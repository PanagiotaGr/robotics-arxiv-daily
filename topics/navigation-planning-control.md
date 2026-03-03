# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-03-03 07:06 UTC_

Total papers shown: **10**


---

- **CHOP: Counterfactual Human Preference Labels Improve Obstacle Avoidance in Visuomotor Navigation Policies**  
  Gershom Seneviratne, Jianyu An, Vaibhav Shende, Sahire Ellahy, Yaxita Amin, Kondapi Manasanjani, Samarth Chopra, Jonathan Deepak Kannan, Dinesh Manocha  
  _2026-03-02_ · https://arxiv.org/abs/2603.02004v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visuomotor navigation policies have shown strong perception-action coupling for embodied agents, yet they often struggle with safe navigation and dynamic obstacle avoidance in complex real-world environments. We introduce CHOP, a novel approach that leverages Counterfactual Human Preference Labels to align visuomotor navigation policies towards human intuition of safety and obstacle avoidance in navigation. In CHOP, for each visual observation, the robot's executed trajectory is included among a set of counterfactual navigation trajectories: alternative trajectories the robot could have followed under identical conditions. Human annotators provide pairwise preference labels over these trajectories based on anticipated outcomes such as collision risk and path efficiency. These aggregated preferences are then used to fine-tune visuomotor navigation policies, aligning their behavior with human preferences in navigation. Experiments on the SCAND dataset show that visuomotor navigation policies fine-tuned with CHOP reduce near-collision events by 49.7%, decrease deviation from human-preferred trajectories by 45.0%, and increase average obstacle clearance by 19.8% on average across multiple state-of-the-art models, compared to their pretrained baselines. These improvements transfer to real-world deployments on a Ghost Robotics Vision60 quadruped, where CHOP-aligned policies improve average goal success rates by 24.4%, increase minimum obstacle clearance by 6.8%, reduce collision and intervention events by 45.7%, and improve normalized path completion by 38.6% on average across navigation scenarios, compared to their pretrained baselines. Our results highlight the value of counterfactual preference supervision in bridging the gap between large-scale visuomotor policies and human-aligned, safety-aware embodied navigation.

  </details>



- **Learning Vision-Based Omnidirectional Navigation: A Teacher-Student Approach Using Monocular Depth Estimation**  
  Jan Finke, Wayne Paul Martis, Adrian Schmelter, Lars Erbach, Christian Jestel, Marvin Wiedemann  
  _2026-03-02_ · https://arxiv.org/abs/2603.01999v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable obstacle avoidance in industrial settings demands 3D scene understanding, but widely used 2D LiDAR sensors perceive only a single horizontal slice of the environment, missing critical obstacles above or below the scan plane. We present a teacher-student framework for vision-based mobile robot navigation that eliminates the need for LiDAR sensors. A teacher policy trained via Proximal Policy Optimization (PPO) in NVIDIA Isaac Lab leverages privileged 2D LiDAR observations that account for the full robot footprint to learn robust navigation. The learned behavior is distilled into a student policy that relies solely on monocular depth maps predicted by a fine-tuned Depth Anything V2 model from four RGB cameras. The complete inference pipeline, comprising monocular depth estimation (MDE), policy execution, and motor control, runs entirely onboard an NVIDIA Jetson Orin AGX mounted on a DJI RoboMaster platform, requiring no external computation for inference. In simulation, the student achieves success rates of 82-96.5%, consistently outperforming the standard 2D LiDAR teacher (50-89%). In real-world experiments, the MDE-based student outperforms the 2D LiDAR teacher when navigating around obstacles with complex 3D geometries, such as overhanging structures and low-profile objects, that fall outside the single scan plane of a 2D LiDAR.

  </details>



- **SaferPath: Hierarchical Visual Navigation with Learned Guidance and Safety-Constrained Control**  
  Lingjie Zhang, Zeyu Jiang, Changhao Chen  
  _2026-03-02_ · https://arxiv.org/abs/2603.01898v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual navigation is a core capability for mobile robots, yet end-to-end learning-based methods often struggle with generalization and safety in unseen, cluttered, or narrow environments. These limitations are especially pronounced in dense indoor settings, where collisions are likely and end-to-end models frequently fail. To address this, we propose SaferPath, a hierarchical visual navigation framework that leverages learned guidance from existing end-to-end models and refines it through a safety-constrained optimization-control module. SaferPath transforms visual observations into a traversable-area map and refines guidance trajectories using Model Predictive Stein Variational Evolution Strategy (MP-SVES), efficiently generating safe trajectories in only a few iterations. The refined trajectories are tracked by an MPC controller, ensuring robust navigation in complex environments. Extensive experiments in scenarios with unseen obstacles, dense unstructured spaces, and narrow corridors demonstrate that SaferPath consistently improves success rates and reduces collisions, outperforming representative baselines such as ViNT and NoMaD, and enabling safe navigation in challenging real-world settings.

  </details>



- **Real-Time Thermal-Inertial Odometry on Embedded Hardware for High-Speed GPS-Denied Flight**  
  Austin Stone, Mark Petersen, Cammy Peterson  
  _2026-03-02_ · https://arxiv.org/abs/2603.02114v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a real-time monocular thermal-inertial odometry system designed for high-velocity, GPS-denied flight on embedded hardware. The system fuses measurements from a FLIR Boson+ 640 longwave infrared camera, a high-rate IMU, a laser range finder, a barometer, and a magnetometer within a fixed-lag factor graph. To sustain reliable feature tracks under motion blur, low contrast, and rapid viewpoint changes, we employ a lightweight thermal-optimized front-end with multi-stage feature filtering. Laser range finder measurements provide per-feature depth priors that stabilize scale during weakly observable motion. High-rate inertial data is first pre-filtered using a Chebyshev Type II infinite impulse response (IIR) filter and then preintegrated, improving robustness to airframe vibrations during aggressive maneuvers. To address barometric altitude errors induced at high airspeeds, we train an uncertainty-aware gated recurrent unit (GRU) network that models the temporal dynamics of static pressure distortion, outperforming polynomial and multi-layer perceptron (MLP) baselines. Integrated on an NVIDIA Jetson Xavier NX, the complete system supports closed-loop quadrotor flight at 30 m/s with drift under 2% over kilometer-scale trajectories. These contributions expand the operational envelope of thermal-inertial navigation, enabling reliable high-speed flight in visually degraded and GPS-denied environments.

  </details>



- **Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling**  
  Alexander Prutsch, David Schinagl, Horst Possegger  
  _2026-03-02_ · https://arxiv.org/abs/2603.01864v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Future trajectories of neighboring traffic agents have a significant influence on the path planning and decision-making of autonomous vehicles. While trajectory forecasting is a well-studied field, research mainly focuses on snapshot-based prediction, where each scenario is treated independently of its global temporal context. However, real-world autonomous driving systems need to operate in a continuous setting, requiring real-time processing of data streams with low latency and consistent predictions over successive timesteps. We leverage this continuous setting to propose a lightweight yet highly accurate streaming-based trajectory forecasting approach. We integrate valuable information from previous predictions with a novel endpoint-aware modeling scheme. Our temporal context propagation uses the trajectory endpoints of the previous forecasts as anchors to extract targeted scenario context encodings. Our approach efficiently guides its scene encoder to extract highly relevant context information without needing refinement iterations or segment-wise decoding. Our experiments highlight that our approach effectively relays information across consecutive timesteps. Unlike methods using multi-stage refinement processing, our approach significantly reduces inference latency, making it well-suited for real-world deployment. We achieve state-of-the-art streaming trajectory prediction results on the Argoverse~2 multi-agent and single-agent benchmarks, while requiring substantially fewer resources.

  </details>



- **Tiny-DroNeRF: Tiny Neural Radiance Fields aboard Federated Learning-enabled Nano-drones**  
  Ilenia Carboni, Elia Cereda, Lorenzo Lamberti, Daniele Malpetti, Francesco Conti, Daniele Palossi  
  _2026-03-02_ · https://arxiv.org/abs/2603.01850v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Sub-30g nano-sized aerial robots can leverage their agility and form factor to autonomously explore cluttered and narrow environments, like in industrial inspection and search and rescue missions. However, the price for their tiny size is a strong limit in their resources, i.e., sub-100 mW microcontroller units (MCUs) delivering $\sim$100 GOps/s at best, and memory budgets well below 100 MB. Despite these strict constraints, we aim to enable complex vision-based tasks aboard nano-drones, such as dense 3D scene reconstruction: a key robotic task underlying fundamental capabilities like spatial awareness and motion planning. Top-performing 3D reconstruction methods leverage neural radiance fields (NeRF) models, which require GBs of memory and massive computation, usually delivered by high-end GPUs consuming 100s of Watts. Our work introduces Tiny-DroNeRF, a lightweight NeRF model, based on Instant-NGP, and optimized for running on a GAP9 ultra-low-power (ULP) MCU aboard our nano-drones. Then, we further empower our Tiny-DroNeRF by leveraging a collaborative federated learning scheme, which distributes the model training among multiple nano-drones. Our experimental results show a 96% reduction in Tiny-DroNeRF's memory footprint compared to Instant-NGP, with only a 5.7 dB drop in reconstruction accuracy. Finally, our federated learning scheme allows Tiny-DroNeRF to train with an amount of data otherwise impossible to keep in a single drone's memory, increasing the overall reconstruction accuracy. Ultimately, our work combines, for the first time, NeRF training on an ULP MCU with federated learning on nano-drones.

  </details>



- **MMNavAgent: Multi-Magnification WSI Navigation Agent for Clinically Consistent Whole-Slide Analysis**  
  Zhengyang Xu, Han Li, Jingsong Liu, Linrui Xie, Xun Ma, Xin You, Shihui Zu, Ayako Ito, Xinyu Hao, Hongming Xu, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.02079v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent AI navigation approaches aim to improve Whole-Slide Image (WSI) diagnosis by modeling spatial exploration and selecting diagnostically relevant regions, yet most operate at a single fixed magnification or rely on predefined magnification traversal. In clinical practice, pathologists examine slides across multiple magnifications and selectively inspect only necessary scales, dynamically integrating global and cellular evidence in a sequential manner. This mismatch prevents existing methods from modeling cross-magnification interactions and adaptive magnification selection inherent to real diagnostic workflows. To these, we propose a clinically consistent Multi-Magnification WSI Navigation Agent (MMNavAgent) that explicitly models multi magnification interaction and adaptive magnification selection. Specifically, we introduce a Cross-Magnification navigation Tool (CMT) that aggregates contextual information from adjacent magnifications to enhance discriminative representations along the navigation path. We further introduce a Magnification Selection Tool (MST) that leverages memory-driven reasoning within the agent framework to enable interactive and adaptive magnification selection, mimicking the sequential decision process of pathologists. Extensive experiments on a public dataset demonstrate improved diagnostic performance, with 1.45% gain of AUC and 2.93% gain of BACC over a non-agent baseline. Code will be public upon acceptance.

  </details>



- **A Resource-Rational Principle for Modeling Visual Attention Control**  
  Yunpeng Bai  
  _2026-03-02_ · https://arxiv.org/abs/2603.02056v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Understanding how people allocate visual attention is central to Human-Computer Interaction (HCI), yet existing computational models of attention are often either descriptive, task-specific, or difficult to interpret. My dissertation develops a resource-rational, simulation-based framework for modeling visual attention as a sequential decision-making process under perceptual, memory, and time constraints. I formalize visual tasks, such as reading and multitasking, as bounded-optimal control problems using Partially Observable Markov Decision Processes, enabling eye-movement behaviors such as fixation and attention switching to emerge from rational adaptation rather than being hand-coded or purely data-driven. These models are instantiated in simulation environments spanning traditional text reading and reading-while-walking with smart glasses, where they reproduce classic empirical effects, explain observed trade-offs between comprehension and safety, and generate novel predictions under time pressure and interface variation. Collectively, this work contributes a unified computational account of visual attention, offering new tools for theory-driven and resource-efficient HCI design.

  </details>



- **LOCUS: A Distribution-Free Loss-Quantile Score for Risk-Aware Predictions**  
  Matheus Barreto, Mário de Castro, Thiago R. Ramos, Denis Valle, Rafael Izbicki  
  _2026-03-02_ · https://arxiv.org/abs/2603.01971v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  Modern machine learning models can be accurate on average yet still make mistakes that dominate deployment cost. We introduce Locus, a distribution-free wrapper that produces a per-input loss-scale reliability score for a fixed prediction function. Rather than quantifying uncertainty about the label, Locus models the realized loss of the prediction function using any engine that outputs a predictive distribution for the loss given an input. A simple split-calibration step turns this function into a distribution-free interpretable score that is comparable across inputs and can be read as an upper loss level. The score is useful on its own for ranking, and it can optionally be thresholded to obtain a transparent flagging rule with distribution-free control of large-loss events. Experiments across 13 regression benchmarks show that Locus yields effective risk ranking and reduces large-loss frequency compared to standard heuristics.

  </details>



- **SSMG-Nav: Enhancing Lifelong Object Navigation with Semantic Skeleton Memory Graph**  
  Haochen Niu, Lantao Zhang, Xingwu Ji, Rendong Ying, Peilin Liu, Fei Wen  
  _2026-03-02_ · https://arxiv.org/abs/2603.01813v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Navigating to out-of-sight targets from human instructions in unfamiliar environments is a core capability for service robots. Despite substantial progress, most approaches underutilize reusable, persistent memory, constraining performance in lifelong settings. Many are additionally limited to single-modality inputs and employ myopic greedy policies, which often induce inefficient back-and-forth maneuvers (BFMs). To address such limitations, we introduce SSMG-Nav, a framework for object navigation built on a \textit{Semantic Skeleton Memory Graph} (SSMG) that consolidates past observations into a spatially aligned, persistent memory anchored by topological keypoints (e.g., junctions, room centers). SSMG clusters nearby entities into subgraphs, unifying entity- and space-level semantics to yield a compact set of candidate destinations. To support multimodal targets (images, objects, and text), we integrate a vision-language model (VLM). For each subgraph, a multimodal prompt synthesized from memory guides the VLM to infer a target belief over destinations. A long-horizon planner then trades off this belief against traversability costs to produce a visit sequence that minimizes expected path length, thereby reducing backtracking. Extensive experiments on challenging lifelong benchmarks and standard ObjectNav benchmarks demonstrate that, compared to strong baselines, our method achieves higher success rates and greater path efficiency, validating the effectiveness of SSMG-Nav.

  </details>


