# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-19 07:13 UTC_

Total papers shown: **13**


---

- **Reactive Motion Generation With Particle-Based Perception in Dynamic Environments**  
  Xiyuan Zhao, Huijun Li, Lifeng Zhu, Zhikai Wei, Xianyi Zhu, Aiguo Song  
  _2026-02-18_ · https://arxiv.org/abs/2602.16462v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reactive motion generation in dynamic and unstructured scenarios is typically subject to essentially static perception and system dynamics. Reliably modeling dynamic obstacles and optimizing collision-free trajectories under perceptive and control uncertainty are challenging. This article focuses on revealing tight connection between reactive planning and dynamic mapping for manipulators from a model-based perspective. To enable efficient particle-based perception with expressively dynamic property, we present a tensorized particle weight update scheme that explicitly maintains obstacle velocities and covariance meanwhile. Building upon this dynamic representation, we propose an obstacle-aware MPPI-based planning formulation that jointly propagates robot-obstacle dynamics, allowing future system motion to be predicted and evaluated under uncertainty. The model predictive method is shown to significantly improve safety and reactivity with dynamic surroundings. By applying our complete framework in simulated and noisy real-world environments, we demonstrate that explicit modeling of robot-obstacle dynamics consistently enhances performance over state-of-the-art MPPI-based perception-planning baselines avoiding multiple static and dynamic obstacles.

  </details>



- **Decentralized and Fully Onboard: Range-Aided Cooperative Localization and Navigation on Micro Aerial Vehicles**  
  Abhishek Goudar, Angela P. Schoellig  
  _2026-02-18_ · https://arxiv.org/abs/2602.16594v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Controlling a team of robots in a coordinated manner is challenging because centralized approaches (where all computation is performed on a central machine) scale poorly, and globally referenced external localization systems may not always be available. In this work, we consider the problem of range-aided decentralized localization and formation control. In such a setting, each robot estimates its relative pose by combining data only from onboard odometry sensors and distance measurements to other robots in the team. Additionally, each robot calculates the control inputs necessary to collaboratively navigate an environment to accomplish a specific task, for example, moving in a desired formation while monitoring an area. We present a block coordinate descent approach to localization that does not require strict coordination between the robots. We present a novel formulation for formation control as inference on factor graphs that takes into account the state estimation uncertainty and can be solved efficiently. Our approach to range-aided localization and formation-based navigation is completely decentralized, does not require specialized trajectories to maintain formation, and achieves decimeter-level positioning and formation control accuracy. We demonstrate our approach through multiple real experiments involving formation flights in diverse indoor and outdoor environments.

  </details>



- **SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks**  
  Zirui Zang, Ahmad Amine, Nick-Marios T. Kokolakis, Truong X. Nghiem, Ugo Rosolia, Rahul Mangharam  
  _2026-02-18_ · https://arxiv.org/abs/2602.16187v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots executing iterative tasks in complex, uncertain environments require control strategies that balance robustness, safety, and high performance. This paper introduces a safe information-theoretic learning model predictive control (SIT-LMPC) algorithm for iterative tasks. Specifically, we design an iterative control framework based on an information-theoretic model predictive control algorithm to address a constrained infinite-horizon optimal control problem for discrete-time nonlinear stochastic systems. An adaptive penalty method is developed to ensure safety while balancing optimality. Trajectories from previous iterations are utilized to learn a value function using normalizing flows, which enables richer uncertainty modeling compared to Gaussian priors. SIT-LMPC is designed for highly parallel execution on graphics processing units, allowing efficient real-time optimization. Benchmark simulations and hardware experiments demonstrate that SIT-LMPC iteratively improves system performance while robustly satisfying system constraints.

  </details>



- **Benchmarking Adversarial Robustness and Adversarial Training Strategies for Object Detection**  
  Alexis Winter, Jean-Vincent Martini, Romaric Audigier, Angelique Loesch, Bertrand Luvison  
  _2026-02-18_ · https://arxiv.org/abs/2602.16494v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Object detection models are critical components of automated systems, such as autonomous vehicles and perception-based robots, but their sensitivity to adversarial attacks poses a serious security risk. Progress in defending these models lags behind classification, hindered by a lack of standardized evaluation. It is nearly impossible to thoroughly compare attack or defense methods, as existing work uses different datasets, inconsistent efficiency metrics, and varied measures of perturbation cost. This paper addresses this gap by investigating three key questions: (1) How can we create a fair benchmark to impartially compare attacks? (2) How well do modern attacks transfer across different architectures, especially from Convolutional Neural Networks to Vision Transformers? (3) What is the most effective adversarial training strategy for robust defense? To answer these, we first propose a unified benchmark framework focused on digital, non-patch-based attacks. This framework introduces specific metrics to disentangle localization and classification errors and evaluates attack cost using multiple perceptual metrics. Using this benchmark, we conduct extensive experiments on state-of-the-art attacks and a wide range of detectors. Our findings reveal two major conclusions: first, modern adversarial attacks against object detection models show a significant lack of transferability to transformer-based architectures. Second, we demonstrate that the most robust adversarial training strategy leverages a dataset composed of a mix of high-perturbation attacks with different objectives (e.g., spatial and semantic), which outperforms training on any single attack.

  </details>



- **Uncertainty-Guided Inference-Time Depth Adaptation for Transformer-Based Visual Tracking**  
  Patrick Poggi, Divake Kumar, Theja Tulabandhula, Amit Ranjan Trivedi  
  _2026-02-18_ · https://arxiv.org/abs/2602.16160v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Transformer-based single-object trackers achieve state-of-the-art accuracy but rely on fixed-depth inference, executing the full encoder--decoder stack for every frame regardless of visual complexity, thereby incurring unnecessary computational cost in long video sequences dominated by temporally coherent frames. We propose UncL-STARK, an architecture-preserving approach that enables dynamic, uncertainty-aware depth adaptation in transformer-based trackers without modifying the underlying network or adding auxiliary heads. The model is fine-tuned to retain predictive robustness at multiple intermediate depths using random-depth training with knowledge distillation, thus enabling safe inference-time truncation. At runtime, we derive a lightweight uncertainty estimate directly from the model's corner localization heatmaps and use it in a feedback-driven policy that selects the encoder and decoder depth for the next frame based on the prediction confidence by exploiting temporal coherence in video. Extensive experiments on GOT-10k and LaSOT demonstrate up to 12\% GFLOPs reduction, 8.9\% latency reduction, and 10.8\% energy savings while maintaining tracking accuracy within 0.2\% of the full-depth baseline across both short-term and long-term sequences.

  </details>



- **Dual-Quadruped Collaborative Transportation in Narrow Environments via Safe Reinforcement Learning**  
  Zhezhi Lei, Zhihai Bi, Wenxin Wang, Jun Ma  
  _2026-02-18_ · https://arxiv.org/abs/2602.16353v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Collaborative transportation, where multiple robots collaboratively transport a payload, has garnered significant attention in recent years. While ensuring safe and high-performance inter-robot collaboration is critical for effective task execution, it is difficult to pursue in narrow environments where the feasible region is extremely limited. To address this challenge, we propose a novel approach for dual-quadruped collaborative transportation via safe reinforcement learning (RL). Specifically, we model the task as a fully cooperative constrained Markov game, where collision avoidance is formulated as constraints. We introduce a cost-advantage decomposition method that enforces the sum of team constraints to remain below an upper bound, thereby guaranteeing task safety within an RL framework. Furthermore, we propose a constraint allocation method that assigns shared constraints to individual robots to maximize the overall task reward, encouraging autonomous task-assignment among robots, thereby improving collaborative task performance. Simulation and real-time experimental results demonstrate that the proposed approach achieves superior performance and a higher success rate in dual-quadruped collaborative transportation compared to existing methods.

  </details>



- **Docking and Persistent Operations for a Resident Underwater Vehicle**  
  Leonard Günzel, Gabrielė Kasparavičiūtė, Ambjørn Grimsrud Waldum, Bjørn-Magnus Moslått, Abubakar Aliyu Badawi, Celil Yılmaz, Md Shamin Yeasher Yousha, Robert Staven, Martin Ludvigsen  
  _2026-02-18_ · https://arxiv.org/abs/2602.16360v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Our understanding of the oceans remains limited by sparse and infrequent observations, primarily because current methods are constrained by the high cost and logistical effort of underwater monitoring, relying either on sporadic surveys across broad areas or on long-term measurements at fixed locations. To overcome these limitations, monitoring systems must enable persistent and autonomous operations without the need for continuous surface support. Despite recent advances, resident underwater vehicles remain uncommon due to persistent challenges in autonomy, robotic resilience, and mechanical robustness, particularly under long-term deployment in harsh and remote environments. This work addresses these problems by presenting the development, deployment, and operation of a resident infrastructure using a docking station with a mini-class Remotely Operated Vehicle (ROV) at 90m depth. The ROVis equipped with enhanced onboard processing and perception, allowing it to autonomously navigate using USBL signals, dock via ArUco marker-based visual localisation fused through an Extended Kalman Filter, and carry out local inspection routines. The system demonstrated a 90% autonomous docking success rate and completed full inspection missions within four minutes, validating the integration of acoustic and visual navigation in real-world conditions. These results show that reliable, untethered operations at depth are feasible, highlighting the potential of resident ROV systems for scalable, cost-effective underwater monitoring.

  </details>



- **SCAR: Satellite Imagery-Based Calibration for Aerial Recordings**  
  Henry Hölzemann, Michael Schleiss  
  _2026-02-18_ · https://arxiv.org/abs/2602.16349v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We introduce SCAR, a method for long-term auto-calibration refinement of aerial visual-inertial systems that exploits georeferenced satellite imagery as a persistent global reference. SCAR estimates both intrinsic and extrinsic parameters by aligning aerial images with 2D--3D correspondences derived from publicly available orthophotos and elevation models. In contrast to existing approaches that rely on dedicated calibration maneuvers or manually surveyed ground control points, our method leverages external geospatial data to detect and correct calibration degradation under field deployment conditions. We evaluate our approach on six large-scale aerial campaigns conducted over two years under diverse seasonal and environmental conditions. Across all sequences, SCAR consistently outperforms established baselines (Kalibr, COLMAP, VINS-Mono), reducing median reprojection error by a large margin, and translating these calibration gains into substantially lower visual localization rotation errors and higher pose accuracy. These results demonstrate that SCAR provides accurate, robust, and reproducible calibration over long-term aerial operations without the need for manual intervention.

  </details>



- **World Model Failure Classification and Anomaly Detection for Autonomous Inspection**  
  Michelle Ho, Muhammad Fadhil Ginting, Isaac R. Ward, Andrzej Reinke, Mykel J. Kochenderfer, Ali-akbar Agha-Mohammadi, Shayegan Omidshafiei  
  _2026-02-18_ · https://arxiv.org/abs/2602.16182v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous inspection robots for monitoring industrial sites can reduce costs and risks associated with human-led inspection. However, accurate readings can be challenging due to occlusions, limited viewpoints, or unexpected environmental conditions. We propose a hybrid framework that combines supervised failure classification with anomaly detection, enabling classification of inspection tasks as a success, known failure, or anomaly (i.e., out-of-distribution) case. Our approach uses a world model backbone with compressed video inputs. This policy-agnostic, distribution-free framework determines classifications based on two decision functions set by conformal prediction (CP) thresholds before a human observer does. We evaluate the framework on gauge inspection feeds collected from office and industrial sites and demonstrate real-time deployment on a Boston Dynamics Spot. Experiments show over 90% accuracy in distinguishing between successes, failures, and OOD cases, with classifications occurring earlier than a human observer. These results highlight the potential for robust, anticipatory failure detection in autonomous inspection tasks or as a feedback signal for model training to assess and improve the quality of training data. Project website: https://autoinspection-classification.github.io

  </details>



- **Discrete Stochastic Localization for Non-autoregressive Generation**  
  Yunshu Wu, Jiayi Cheng, Partha Thakuria, Rob Brekelmans, Evangelos E. Papalexakis, Greg Ver Steeg  
  _2026-02-18_ · https://arxiv.org/abs/2602.16169v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Non-autoregressive (NAR) generation reduces decoding latency by predicting many tokens in parallel, but iterative refinement often suffers from error accumulation and distribution shift under self-generated drafts. Masked diffusion language models (MDLMs) and their remasking samplers (e.g., ReMDM) can be viewed as modern NAR iterative refinement, where generation repeatedly revises a partially observed draft. In this work we show that \emph{training alone} can substantially improve the step-efficiency of MDLM/ReMDM sampling. We propose \textsc{DSL} (Discrete Stochastic Localization), which trains a single SNR-invariant denoiser across a continuum of corruption levels, bridging intermediate draft noise and mask-style endpoint corruption within one Diffusion Transformer. On OpenWebText, \textsc{DSL} fine-tuning yields large MAUVE gains at low step budgets, surpassing the MDLM+ReMDM baseline with \(\sim\)4$\times$ fewer denoiser evaluations, and matches autoregressive quality at high budgets. Analyses show improved self-correction and uncertainty calibration, making remasking markedly more compute-efficient.

  </details>



- **Image Measurement Method for Automatic Insertion of Forks into Inclined Pallet**  
  Nobuyuki Kita, Takuro Kato  
  _2026-02-18_ · https://arxiv.org/abs/2602.16178v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In order to insert a fork into a hole of a pallet by a forklift located in front of a pallet, it is necessary to control the height position, reach position, and tilt angle of the fork to match the position and orientation of the hole of the pallet. In order to make AGF (Autonomous Guided Forklift) do this automatically, we propose an image measurement method to measure the pitch inclination of the pallet in the camera coordinate system from an image obtained by using a wide-angle camera. In addition, we propose an image measurement method to easily acquire the calibration information between the camera coordinate system and the fork coordinate system necessary to apply the measurements in the camera coordinate system to the fork control. In the experiment space, a wide-angle camera was fixed at the backrest of a reach type forklift. The wide-angle images taken by placing a pallet in front of the camera were processed. As a result of evaluating the error by comparing the image measurement value with the hand measurement value when changing the pitch inclination angle of the pallet, the relative height of the pallet and the fork, and whether the pallet is loaded or not, it was confirmed that the error was within the allowable range for safely inserting the fork.

  </details>



- **VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection**  
  Yingyuan Yang, Tian Lan, Yifei Gao, Yimeng Lu, Wenjun He, Meng Wang, Chenghao Liu, Chen Zhang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16681v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Time-series anomaly detection (TSAD) requires identifying both immediate Point Anomalies and long-range Context Anomalies. However, existing foundation models face a fundamental trade-off: 1D temporal models provide fine-grained pointwise localization but lack a global contextual perspective, while 2D vision-based models capture global patterns but suffer from information bottlenecks due to a lack of temporal alignment and coarse-grained pointwise detection. To resolve this dilemma, we propose VETime, the first TSAD framework that unifies temporal and visual modalities through fine-grained visual-temporal alignment and dynamic fusion. VETime introduces a Reversible Image Conversion and a Patch-Level Temporal Alignment module to establish a shared visual-temporal timeline, preserving discriminative details while maintaining temporal sensitivity. Furthermore, we design an Anomaly Window Contrastive Learning mechanism and a Task-Adaptive Multi-Modal Fusion to adaptively integrate the complementary perceptual strengths of both modalities. Extensive experiments demonstrate that VETime significantly outperforms state-of-the-art models in zero-shot scenarios, achieving superior localization precision with lower computational overhead than current vision-based approaches. Code available at: https://github.com/yyyangcoder/VETime.

  </details>



- **Parameter-Free Adaptive Multi-Scale Channel-Spatial Attention Aggregation framework for 3D Indoor Semantic Scene Completion Toward Assisting Visually Impaired**  
  Qi He, XiangXiang Wang, Jingtao Zhang, Yongbin Yu, Hongxiang Chu, Manping Fan, JingYe Cai, Zhenglin Yang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16385v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In indoor assistive perception for visually impaired users, 3D Semantic Scene Completion (SSC) is expected to provide structurally coherent and semantically consistent occupancy under strictly monocular vision for safety-critical scene understanding. However, existing monocular SSC approaches often lack explicit modeling of voxel-feature reliability and regulated cross-scale information propagation during 2D-3D projection and multi-scale fusion, making them vulnerable to projection diffusion and feature entanglement and thus limiting structural stability.To address these challenges, this paper presents an Adaptive Multi-scale Attention Aggregation (AMAA) framework built upon the MonoScene pipeline. Rather than introducing a heavier backbone, AMAA focuses on reliability-oriented feature regulation within a monocular SSC framework. Specifically, lifted voxel features are jointly calibrated in semantic and spatial dimensions through parallel channel-spatial attention aggregation, while multi-scale encoder-decoder fusion is stabilized via a hierarchical adaptive feature-gating strategy that regulates information injection across scales.Experiments on the NYUv2 benchmark demonstrate consistent improvements over MonoScene without significantly increasing system complexity: AMAA achieves 27.25% SSC mIoU (+0.31) and 43.10% SC IoU (+0.59). In addition, system-level deployment on an NVIDIA Jetson platform verifies that the complete AMAA framework can be executed stably on embedded hardware. Overall, AMAA improves monocular SSC quality and provides a reliable and deployable perception framework for indoor assistive systems targeting visually impaired users.

  </details>


