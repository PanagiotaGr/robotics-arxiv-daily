# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-01-28 06:52 UTC_

Total papers shown: **6**


---

- **Self-Supervised Path Planning in Unstructured Environments via Global-Guided Differentiable Hard Constraint Projection**  
  Ziqian Wang, Chenxi Fang, Zhen Zhang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19354v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deploying deep learning agents for autonomous navigation in unstructured environments faces critical challenges regarding safety, data scarcity, and limited computational resources. Traditional solvers often suffer from high latency, while emerging learning-based approaches struggle to ensure deterministic feasibility. To bridge the gap from embodied to embedded intelligence, we propose a self-supervised framework incorporating a differentiable hard constraint projection layer for runtime assurance. To mitigate data scarcity, we construct a Global-Guided Artificial Potential Field (G-APF), which provides dense supervision signals without manual labeling. To enforce actuator limitations and geometric constraints efficiently, we employ an adaptive neural projection layer, which iteratively rectifies the coarse network output onto the feasible manifold. Extensive benchmarks on a test set of 20,000 scenarios demonstrate an 88.75\% success rate, substantiating the enhanced operational safety. Closed-loop experiments in CARLA further validate the physical realizability of the planned paths under dynamic constraints. Furthermore, deployment verification on an NVIDIA Jetson Orin NX confirms an inference latency of 94 ms, showing real-time feasibility on resource-constrained embedded hardware. This framework offers a generalized paradigm for embedding physical laws into neural architectures, providing a viable direction for solving constrained optimization in mechatronics. Source code is available at: https://github.com/wzq-13/SSHC.git.

  </details>



- **How Does Delegation in Social Interaction Evolve Over Time? Navigation with a Robot for Blind People**  
  Rayna Hata, Masaki Kuribayashi, Allan Wang, Hironobu Takagi, Chieko Asakawa  
  _2026-01-27_ · https://arxiv.org/abs/2601.19851v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Autonomy and independent navigation are vital to daily life but remain challenging for individuals with blindness. Robotic systems can enhance mobility and confidence by providing intelligent navigation assistance. However, fully autonomous systems may reduce users' sense of control, even when they wish to remain actively involved. Although collaboration between user and robot has been recognized as important, little is known about how perceptions of this relationship change with repeated use. We present a repeated exposure study with six blind participants who interacted with a navigation-assistive robot in a real-world museum. Participants completed tasks such as navigating crowds, approaching lines, and encountering obstacles. Findings show that participants refined their strategies over time, developing clearer preferences about when to rely on the robot versus act independently. This work provides insights into how strategies and preferences evolve with repeated interaction and offers design implications for robots that adapt to user needs over time.

  </details>



- **ScenePilot-Bench: A Large-Scale Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving**  
  Yujin Wang, Yutong Zheng, Wenxian Fan, Tianyi Wang, Hongqing Chu, Daxin Tian, Bingzhao Gao, Jianqiang Wang, Hong Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19582v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In this paper, we introduce ScenePilot-Bench, a large-scale first-person driving benchmark designed to evaluate vision-language models (VLMs) in autonomous driving scenarios. ScenePilot-Bench is built upon ScenePilot-4K, a diverse dataset comprising 3,847 hours of driving videos, annotated with multi-granularity information including scene descriptions, risk assessments, key participant identification, ego trajectories, and camera parameters. The benchmark features a four-axis evaluation suite that assesses VLM capabilities in scene understanding, spatial perception, motion planning, and GPT-Score, with safety-aware metrics and cross-region generalization settings. We benchmark representative VLMs on ScenePilot-Bench, providing empirical analyses that clarify current performance boundaries and identify gaps for driving-oriented reasoning. ScenePilot-Bench offers a comprehensive framework for evaluating and advancing VLMs in safety-critical autonomous driving contexts.

  </details>



- **Instance-Guided Radar Depth Estimation for 3D Object Detection**  
  Chen-Chou Lo, Patrick Vandewalle  
  _2026-01-27_ · https://arxiv.org/abs/2601.19314v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate depth estimation is fundamental to 3D perception in autonomous driving, supporting tasks such as detection, tracking, and motion planning. However, monocular camera-based 3D detection suffers from depth ambiguity and reduced robustness under challenging conditions. Radar provides complementary advantages such as resilience to poor lighting and adverse weather, but its sparsity and low resolution limit its direct use in detection frameworks. This motivates the need for effective Radar-camera fusion with improved preprocessing and depth estimation strategies. We propose an end-to-end framework that enhances monocular 3D object detection through two key components. First, we introduce InstaRadar, an instance segmentation-guided expansion method that leverages pre-trained segmentation masks to enhance Radar density and semantic alignment, producing a more structured representation. InstaRadar achieves state-of-the-art results in Radar-guided depth estimation, showing its effectiveness in generating high-quality depth features. Second, we integrate the pre-trained RCDPT into the BEVDepth framework as a replacement for its depth module. With InstaRadar-enhanced inputs, the RCDPT integration consistently improves 3D detection performance. Overall, these components yield steady gains over the baseline BEVDepth model, demonstrating the effectiveness of InstaRadar and the advantage of explicit depth supervision in 3D object detection. Although the framework lags behind Radar-camera fusion models that directly extract BEV features, since Radar serves only as guidance rather than an independent feature stream, this limitation highlights potential for improvement. Future work will extend InstaRadar to point cloud-like representations and integrate a dedicated Radar branch with temporal cues for enhanced BEV fusion.

  </details>



- **A DVL Aided Loosely Coupled Inertial Navigation Strategy for AUVs with Attitude Error Modeling and Variance Propagation**  
  Jin Huang, Zichen Liu, Haoda Li, Zhikun Wang, Ying Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19509v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In underwater navigation systems, strap-down inertial navigation system/Doppler velocity log (SINS/DVL)-based loosely coupled architectures are widely adopted. Conventional approaches project DVL velocities from the body coordinate system to the navigation coordinate system using SINS-derived attitude; however, accumulated attitude estimation errors introduce biases into velocity projection and degrade navigation performance during long-term operation. To address this issue, two complementary improvements are introduced. First, a vehicle attitude error-aware DVL velocity transformation model is formulated by incorporating attitude error terms into the observation equation to reduce projection-induced velocity bias. Second, a covariance matrix-based variance propagation method is developed to transform DVL measurement uncertainty across coordinate systems, introducing an expectation-based attitude error compensation term to achieve statistically consistent noise modeling. Simulation and field experiment results demonstrate that both improvements individually enhance navigation accuracy and confirm that accumulated attitude errors affect both projected velocity measurements and their associated uncertainty. When jointly applied, long-term error divergence is effectively suppressed. Field experimental results show that the proposed approach achieves a 78.3% improvement in 3D position RMSE and a 71.8% reduction in the maximum component-wise position error compared with the baseline IMU+DVL method, providing a robust solution for improving long-term SINS/DVL navigation performance.

  </details>



- **Dynamic Worlds, Dynamic Humans: Generating Virtual Human-Scene Interaction Motion in Dynamic Scenes**  
  Yin Wang, Zhiying Leng, Haitian Liu, Frederick W. B. Li, Mu Li, Xiaohui Liang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19484v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Scenes are continuously undergoing dynamic changes in the real world. However, existing human-scene interaction generation methods typically treat the scene as static, which deviates from reality. Inspired by world models, we introduce Dyn-HSI, the first cognitive architecture for dynamic human-scene interaction, which endows virtual humans with three humanoid components. (1)Vision (human eyes): we equip the virtual human with a Dynamic Scene-Aware Navigation, which continuously perceives changes in the surrounding environment and adaptively predicts the next waypoint. (2)Memory (human brain): we equip the virtual human with a Hierarchical Experience Memory, which stores and updates experiential data accumulated during training. This allows the model to leverage prior knowledge during inference for context-aware motion priming, thereby enhancing both motion quality and generalization. (3) Control (human body): we equip the virtual human with Human-Scene Interaction Diffusion Model, which generates high-fidelity interaction motions conditioned on multimodal inputs. To evaluate performance in dynamic scenes, we extend the existing static human-scene interaction datasets to construct a dynamic benchmark, Dyn-Scenes. We conduct extensive qualitative and quantitative experiments to validate Dyn-HSI, showing that our method consistently outperforms existing approaches and generates high-quality human-scene interaction motions in both static and dynamic settings.

  </details>


