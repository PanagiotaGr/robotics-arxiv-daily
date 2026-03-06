# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-03-06 07:04 UTC_

Total papers shown: **9**


---

- **Direct Contact-Tolerant Motion Planning With Vision Language Models**  
  He Li, Jian Sun, Chengyang Li, Guoliang Li, Qiyu Ruan, Shuai Wang, Chengzhong Xu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05017v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Navigation in cluttered environments often requires robots to tolerate contact with movable or deformable objects to maintain efficiency. Existing contact-tolerant motion planning (CTMP) methods rely on indirect spatial representations (e.g., prebuilt map, obstacle set), resulting in inaccuracies and a lack of adaptiveness to environmental uncertainties. To address this issue, we propose a direct contact-tolerant (DCT) planner, which integrates vision-language models (VLMs) into direct point perception and navigation, including two key components. The first one is VLM point cloud partitioner (VPP), which performs contact-tolerance reasoning in image space using VLM, caches inference masks, propagates them across frames using odometry, and projects them onto the current scan to generate a contact-aware point cloud. The second innovation is VPP guided navigation (VGN), which formulates CTMP as a perception-to-control optimization problem under direct contact-aware point cloud constraints, which is further solved by a specialized deep neural network (DNN). We implement DCT in Isaac Sim and a real car-like robot, demonstrating that DCT achieves robust and efficient navigation in cluttered environments with movable obstacles, outperforming representative baselines across diverse metrics. The code is available at: https://github.com/ChrisLeeUM/DCT.

  </details>



- **Digital Twin Driven Textile Classification and Foreign Object Recognition in Automated Sorting Systems**  
  Serkan Ergun, Tobias Mitterer, Hubert Zangl  
  _2026-03-05_ · https://arxiv.org/abs/2603.05230v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The increasing demand for sustainable textile recycling requires robust automation solutions capable of handling deformable garments and detecting foreign objects in cluttered environments. This work presents a digital twin driven robotic sorting system that integrates grasp prediction, multi modal perception, and semantic reasoning for real world textile classification. A dual arm robotic cell equipped with RGBD sensing, capacitive tactile feedback, and collision-aware motion planning autonomously separates garments from an unsorted basket, transfers them to an inspection zone, and classifies them using state of the art Visual Language Models (VLMs). We benchmark nine VLM s from five model families on a dataset of 223 inspection scenarios comprising shirts, socks, trousers, underwear, foreign objects (including garments outside of the aforementioned classes), and empty scenes. The evaluation assesses per class accuracy, hallucination behavior, and computational performance under practical hardware constraints. Results show that the Qwen model family achieves the highest overall accuracy (up to 87.9 %), with strong foreign object detection performance, while lighter models such as Gemma3 offer competitive speed accuracy trade offs for edge deployment. A digital twin combined with MoveIt enables collision aware path planning and integrates segmented 3D point clouds of inspected garments into the virtual environment for improved manipulation reliability. The presented system demonstrates the feasibility of combining semantic VLM reasoning with conventional grasp detection and digital twin technology for scalable, autonomous textile sorting in realistic industrial settings.

  </details>



- **Safe-SAGE: Social-Semantic Adaptive Guidance for Safe Engagement through Laplace-Modulated Poisson Safety Functions**  
  Lizhi Yang, Ryan M. Bena, Meg Wilkinson, Gilbert Bahati, Andy Navarro Brenes, Ryan K. Cosner, Aaron D. Ames  
  _2026-03-05_ · https://arxiv.org/abs/2603.05497v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Traditional safety-critical control methods, such as control barrier functions, suffer from semantic blindness, exhibiting the same behavior around obstacles regardless of contextual significance. This limitation leads to the uniform treatment of all obstacles, despite their differing semantic meanings. We present Safe-SAGE (Social-Semantic Adaptive Guidance for Safe Engagement), a unified framework that bridges the gap between high-level semantic understanding and low-level safety-critical control through a Poisson safety function (PSF) modulated using a Laplace guidance field. Our approach perceives the environment by fusing multi-sensor point clouds with vision-based instance segmentation and persistent object tracking to maintain up-to-date semantics beyond the camera's field of view. A multi-layer safety filter is then used to modulate system inputs to achieve safe navigation using this semantic understanding of the environment. This safety filter consists of both a model predictive control layer and a control barrier function layer. Both layers utilize the PSF and flux modulation of the guidance field to introduce varying levels of conservatism and multi-agent passing norms for different obstacles in the environment. Our framework enables legged robots to navigate semantically rich, dynamic environments with context-dependent safety margins while maintaining rigorous safety guarantees.

  </details>



- **Accelerating Sampling-Based Control via Learned Linear Koopman Dynamics**  
  Wenjian Hao, Yuxuan Fang, Zehui Lu, Shaoshuai Mou  
  _2026-03-05_ · https://arxiv.org/abs/2603.05385v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents an efficient model predictive path integral (MPPI) control framework for systems with complex nonlinear dynamics. To improve the computational efficiency of classic MPPI while preserving control performance, we replace the nonlinear dynamics used for trajectory propagation with a learned linear deep Koopman operator (DKO) model, enabling faster rollout and more efficient trajectory sampling. The DKO dynamics are learned directly from interaction data, eliminating the need for analytical system models. The resulting controller, termed MPPI-DK, is evaluated in simulation on pendulum balancing and surface vehicle navigation tasks, and validated on hardware through reference-tracking experiments on a quadruped robot. Experimental results demonstrate that MPPI-DK achieves control performance close to MPPI with true dynamics while substantially reducing computational cost, enabling efficient real-time control on robotic platforms.

  </details>



- **OpenFrontier: General Navigation with Visual-Language Grounded Frontiers**  
  Esteban Padilla, Boyang Sun, Marc Pollefeys, Hermann Blum  
  _2026-03-05_ · https://arxiv.org/abs/2603.05377v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Open-world navigation requires robots to make decisions in complex everyday environments while adapting to flexible task requirements. Conventional navigation approaches often rely on dense 3D reconstruction and hand-crafted goal metrics, which limits their generalization across tasks and environments. Recent advances in vision--language navigation (VLN) and vision--language--action (VLA) models enable end-to-end policies conditioned on natural language, but typically require interactive training, large-scale data collection, or task-specific fine-tuning with a mobile agent. We formulate navigation as a sparse subgoal identification and reaching problem and observe that providing visual anchoring targets for high-level semantic priors enables highly efficient goal-conditioned navigation. Based on this insight, we select navigation frontiers as semantic anchors and propose OpenFrontier, a training-free navigation framework that seamlessly integrates diverse vision--language prior models. OpenFrontier enables efficient navigation with a lightweight system design, without dense 3D mapping, policy training, or model fine-tuning. We evaluate OpenFrontier across multiple navigation benchmarks and demonstrate strong zero-shot performance, as well as effective real-world deployment on a mobile robot.

  </details>



- **GaussTwin: Unified Simulation and Correction with Gaussian Splatting for Robotic Digital Twins**  
  Yichen Cai, Paul Jansonnie, Cristiana de Farias, Oleg Arenz, Jan Peters  
  _2026-03-05_ · https://arxiv.org/abs/2603.05108v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Digital twins promise to enhance robotic manipulation by maintaining a consistent link between real-world perception and simulation. However, most existing systems struggle with the lack of a unified model, complex dynamic interactions, and the real-to-sim gap, which limits downstream applications such as model predictive control. Thus, we propose GaussTwin, a real-time digital twin that combines position-based dynamics with discrete Cosserat rod formulations for physically grounded simulation, and Gaussian splatting for efficient rendering and visual correction. By anchoring Gaussians to physical primitives and enforcing coherent SE(3) updates driven by photometric error and segmentation masks, GaussTwin achieves stable prediction-correction while preserving physical fidelity. Through experiments in both simulation and on a Franka Research 3 platform, we show that GaussTwin consistently improves tracking accuracy and robustness compared to shape-matching and rigid-only baselines, while also enabling downstream tasks such as push-based planning. These results highlight GaussTwin as a step toward unified, physically meaningful digital twins that can support closed-loop robotic interaction and learning.

  </details>



- **Observing and Controlling Features in Vision-Language-Action Models**  
  Hugo Buurmeijer, Carmen Amo Alonso, Aiden Swann, Marco Pavone  
  _2026-03-05_ · https://arxiv.org/abs/2603.05487v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action Models (VLAs) have shown remarkable progress towards embodied intelligence. While their architecture partially resembles that of Large Language Models (LLMs), VLAs exhibit higher complexity due to their multi-modal inputs/outputs and often hybrid nature of transformer and diffusion heads. This is part of the reason why insights from mechanistic interpretability in LLMs, which explain how the internal model representations relate to their output behavior, do not trivially transfer to VLA counterparts. In this work, we propose to close this gap by introducing and analyzing two main concepts: feature-observability and feature-controllability. In particular, we first study features that are linearly encoded in representation space, and show how they can be observed by means of a linear classifier. Then, we use a minimal linear intervention grounded in optimal control to accurately place internal representations and steer the VLA's output towards a desired region. Our results show that targeted, lightweight interventions can reliably steer a robot's behavior while preserving closed-loop capabilities. We demonstrate on different VLA architectures ($π_{0.5}$ and OpenVLA) through simulation experiments that VLAs possess interpretable internal structure amenable to online adaptation without fine-tuning, enabling real-time alignment with user preferences and task requirements.

  </details>



- **Critic in the Loop: A Tri-System VLA Framework for Robust Long-Horizon Manipulation**  
  Pengfei Yi, Yingjie Ma, Wenjiang Xu, Yanan Hao, Shuai Gan, Wanting Li, Shanlin Zhong  
  _2026-03-05_ · https://arxiv.org/abs/2603.05185v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Balancing high-level semantic reasoning with low-level reactive control remains a core challenge in visual robotic manipulation. While Vision-Language Models (VLMs) excel at cognitive planning, their inference latency precludes real-time execution. Conversely, fast Vision-Language-Action (VLA) models often lack the semantic depth required for complex, long-horizon tasks. To bridge this gap, we introduce Critic in the Loop, an adaptive hierarchical framework driven by dynamic VLM-Expert scheduling. At its core is a bionic Tri-System architecture comprising a VLM brain for global reasoning, a VLA cerebellum for reactive execution, and a lightweight visual Critic. By continuously monitoring the workspace, the Critic dynamically routes control authority. It sustains rapid closed-loop execution via the VLA for routine subtasks, and adaptively triggers the VLM for replanning upon detecting execution anomalies such as task stagnation or failures. Furthermore, our architecture seamlessly integrates human-inspired rules to intuitively break infinite retry loops. This visually-grounded scheduling minimizes expensive VLM queries, while substantially enhancing system robustness and autonomy in out-of-distribution (OOD) scenarios. Comprehensive experiments on challenging, long-horizon manipulation benchmarks reveal that our approach achieves state-of-the-art performance.

  </details>



- **Residual RL--MPC for Robust Microrobotic Cell Pushing Under Time-Varying Flow**  
  Yanda Yang, Sambeeta Das  
  _2026-03-05_ · https://arxiv.org/abs/2603.05448v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Contact-rich micromanipulation in microfluidic flow is challenging because small disturbances can break pushing contact and induce large lateral drift. We study planar cell pushing with a magnetic rolling microrobot that tracks a waypoint-sampled reference curve under time-varying Poiseuille flow. We propose a hybrid controller that augments a nominal MPC with a learned residual policy trained by SAC. The policy outputs a bounded 2D velocity correction that is contact-gated, so residual actions are applied only during robot--cell contact, preserving reliable approach behavior and stabilizing learning. All methods share the same actuation interface and speed envelope for fair comparisons. Experiments show improved robustness and tracking accuracy over pure MPC and PID under nonstationary flow, with generalization from a clover training curve to unseen circle and square trajectories. A residual-bound sweep identifies an intermediate correction limit as the best trade-off, which we use in all benchmarks.

  </details>


