# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-11 07:16 UTC_

Total papers shown: **12**


---

- **A Collision-Free Sway Damping Model Predictive Controller for Safe and Reactive Forestry Crane Navigation**  
  Marc-Philip Ecker, Christoph Fröhlich, Johannes Huemer, David Gruber, Bernhard Bischof, Tobias Glück, Wolfgang Kemmetmüller  
  _2026-02-10_ · https://arxiv.org/abs/2602.10035v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Forestry cranes operate in dynamic, unstructured outdoor environments where simultaneous collision avoidance and payload sway control are critical for safe navigation. Existing approaches address these challenges separately, either focusing on sway damping with predefined collision-free paths or performing collision avoidance only at the global planning level. We present the first collision-free, sway-damping model predictive controller (MPC) for a forestry crane that unifies both objectives in a single control framework. Our approach integrates LiDAR-based environment mapping directly into the MPC using online Euclidean distance fields (EDF), enabling real-time environmental adaptation. The controller simultaneously enforces collision constraints while damping payload sway, allowing it to (i) replan upon quasi-static environmental changes, (ii) maintain collision-free operation under disturbances, and (iii) provide safe stopping when no bypass exists. Experimental validation on a real forestry crane demonstrates effective sway damping and successful obstacle avoidance. A video can be found at https://youtu.be/tEXDoeLLTxA.

  </details>



- **Fast Motion Planning for Non-Holonomic Mobile Robots via a Rectangular Corridor Representation of Structured Environments**  
  Alejandro Gonzalez-Garcia, Sebastiaan Wyns, Sonia De Santis, Jan Swevers, Wilm Decré  
  _2026-02-10_ · https://arxiv.org/abs/2602.09714v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a complete framework for fast motion planning of non-holonomic autonomous mobile robots in highly complex but structured environments. Conventional grid-based planners struggle with scalability, while many kinematically-feasible planners impose a significant computational burden due to their search space complexity. To overcome these limitations, our approach introduces a deterministic free-space decomposition that creates a compact graph of overlapping rectangular corridors. This method enables a significant reduction in the search space, without sacrificing path resolution. The framework then performs online motion planning by finding a sequence of rectangles and generating a near-time-optimal, kinematically-feasible trajectory using an analytical planner. The result is a highly efficient solution for large-scale navigation. We validate our framework through extensive simulations and on a physical robot. The implementation is publicly available as open-source software.

  </details>



- **AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild**  
  Xiaolou Sun, Wufei Si, Wenhui Ni, Yuntian Li, Dongming Wu, Fei Xie, Runwei Guan, He-Yang Xu, Henghui Ding, Yuan Wu, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09657v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-language navigation (VLN) requires intelligent agents to navigate environments by interpreting linguistic instructions alongside visual observations, serving as a cornerstone task in Embodied AI. Current VLN research for unmanned aerial vehicles (UAVs) relies on detailed, pre-specified instructions to guide the UAV along predetermined routes. However, real-world outdoor exploration typically occurs in unknown environments where detailed navigation instructions are unavailable. Instead, only coarse-grained positional or directional guidance can be provided, requiring UAVs to autonomously navigate through continuous planning and obstacle avoidance. To bridge this gap, we propose AutoFly, an end-to-end Vision-Language-Action (VLA) model for autonomous UAV navigation. AutoFly incorporates a pseudo-depth encoder that derives depth-aware features from RGB inputs to enhance spatial reasoning, coupled with a progressive two-stage training strategy that effectively aligns visual, depth, and linguistic representations with action policies. Moreover, existing VLN datasets have fundamental limitations for real-world autonomous navigation, stemming from their heavy reliance on explicit instruction-following over autonomous decision-making and insufficient real-world data. To address these issues, we construct a novel autonomous navigation dataset that shifts the paradigm from instruction-following to autonomous behavior modeling through: (1) trajectory collection emphasizing continuous obstacle avoidance, autonomous planning, and recognition workflows; (2) comprehensive real-world data integration. Experimental results demonstrate that AutoFly achieves a 3.9% higher success rate compared to state-of-the-art VLA baselines, with consistent performance across simulated and real environments.

  </details>



- **Decoupled MPPI-Based Multi-Arm Motion Planning**  
  Dan Evron, Elias Goldsztejn, Ronen I. Brafman  
  _2026-02-10_ · https://arxiv.org/abs/2602.10114v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advances in sampling-based motion planning algorithms for high DOF arms leverage GPUs to provide SOTA performance. These algorithms can be used to control multiple arms jointly, but this approach scales poorly. To address this, we extend STORM, a sampling-based model-predictive-control (MPC) motion planning algorithm, to handle multiple robots in a distributed fashion. First, we modify STORM to handle dynamic obstacles. Then, we let each arm compute its own motion plan prefix, which it shares with the other arms, which treat it as a dynamic obstacle. Finally, we add a dynamic priority scheme. The new algorithm, MR-STORM, demonstrates clear empirical advantages over SOTA algorithms when operating with both static and dynamic obstacles.

  </details>



- **Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction**  
  Sizhe Yang, Linning Xu, Hao Li, Juncheng Mu, Jia Zeng, Dahua Lin, Jiangmiao Pang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10101v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  3D spatial perception is fundamental to generalizable robotic manipulation, yet obtaining reliable, high-quality 3D geometry remains challenging. Depth sensors suffer from noise and material sensitivity, while existing reconstruction models lack the precision and metric consistency required for physical interaction. We introduce Robo3R, a feed-forward, manipulation-ready 3D reconstruction model that predicts accurate, metric-scale scene geometry directly from RGB images and robot states in real time. Robo3R jointly infers scale-invariant local geometry and relative camera poses, which are unified into the scene representation in the canonical robot frame via a learned global similarity transformation. To meet the precision demands of manipulation, Robo3R employs a masked point head for sharp, fine-grained point clouds, and a keypoint-based Perspective-n-Point (PnP) formulation to refine camera extrinsics and global alignment. Trained on Robo3R-4M, a curated large-scale synthetic dataset with four million high-fidelity annotated frames, Robo3R consistently outperforms state-of-the-art reconstruction methods and depth sensors. Across downstream tasks including imitation learning, sim-to-real transfer, grasp synthesis, and collision-free motion planning, we observe consistent gains in performance, suggesting the promise of this alternative 3D sensing module for robotic manipulation.

  </details>



- **Hydra-Nav: Object Navigation via Adaptive Dual-Process Reasoning**  
  Zixuan Wang, Huang Fang, Shaoan Wang, Yuanfei Luo, Heng Dong, Wei Li, Yiming Gan  
  _2026-02-10_ · https://arxiv.org/abs/2602.09972v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While large vision-language models (VLMs) show promise for object goal navigation, current methods still struggle with low success rates and inefficient localization of unseen objects--failures primarily attributed to weak temporal-spatial reasoning. Meanwhile, recent attempts to inject reasoning into VLM-based agents improve success rates but incur substantial computational overhead. To address both the ineffectiveness and inefficiency of existing approaches, we introduce Hydra-Nav, a unified VLM architecture that adaptively switches between a deliberative slow system for analyzing exploration history and formulating high-level plans, and a reactive fast system for efficient execution. We train Hydra-Nav through a three-stage curriculum: (i) spatial-action alignment to strengthen trajectory planning, (ii) memory-reasoning integration to enhance temporal-spatial reasoning over long-horizon exploration, and (iii) iterative rejection fine-tuning to enable selective reasoning at critical decision points. Extensive experiments demonstrate that Hydra-Nav achieves state-of-the-art performance on the HM3D, MP3D, and OVON benchmarks, outperforming the second-best methods by 11.1%, 17.4%, and 21.2%, respectively. Furthermore, we introduce SOT (Success weighted by Operation Time), a new metric to measure search efficiency across VLMs with varying reasoning intensity. Results show that adaptive reasoning significantly enhances search efficiency over fixed-frequency baselines.

  </details>



- **NavDreamer: Video Models as Zero-Shot 3D Navigators**  
  Xijie Huang, Weiqi Gai, Tianyue Wu, Congyu Wang, Zhiyang Liu, Xin Zhou, Yuze Wu, Fei Gao  
  _2026-02-10_ · https://arxiv.org/abs/2602.09765v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Previous Vision-Language-Action models face critical limitations in navigation: scarce, diverse data from labor-intensive collection and static representations that fail to capture temporal dynamics and physical laws. We propose NavDreamer, a video-based framework for 3D navigation that leverages generative video models as a universal interface between language instructions and navigation trajectories. Our main hypothesis is that video's ability to encode spatiotemporal information and physical dynamics, combined with internet-scale availability, enables strong zero-shot generalization in navigation. To mitigate the stochasticity of generative predictions, we introduce a sampling-based optimization method that utilizes a VLM for trajectory scoring and selection. An inverse dynamics model is employed to decode executable waypoints from generated video plans for navigation. To systematically evaluate this paradigm in several video model backbones, we introduce a comprehensive benchmark covering object navigation, precise navigation, spatial grounding, language control, and scene reasoning. Extensive experiments demonstrate robust generalization across novel objects and unseen environments, with ablation studies revealing that navigation's high-level decision-making nature makes it particularly suited for video-based planning.

  </details>



- **Code2World: A GUI World Model via Renderable Code Generation**  
  Yuhao Zheng, Li'an Zhong, Yi Wang, Rui Dai, Kaikui Liu, Xiangxiang Chu, Linyuan Lv, Philip Torr, Kevin Qinghong Lin  
  _2026-02-10_ · https://arxiv.org/abs/2602.09856v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous GUI agents interact with environments by perceiving interfaces and executing actions. As a virtual sandbox, the GUI World model empowers agents with human-like foresight by enabling action-conditioned prediction. However, existing text- and pixel-based approaches struggle to simultaneously achieve high visual fidelity and fine-grained structural controllability. To this end, we propose Code2World, a vision-language coder that simulates the next visual state via renderable code generation. Specifically, to address the data scarcity problem, we construct AndroidCode by translating GUI trajectories into high-fidelity HTML and refining synthesized code through a visual-feedback revision mechanism, yielding a corpus of over 80K high-quality screen-action pairs. To adapt existing VLMs into code prediction, we first perform SFT as a cold start for format layout following, then further apply Render-Aware Reinforcement Learning which uses rendered outcome as the reward signal by enforcing visual semantic fidelity and action consistency. Extensive experiments demonstrate that Code2World-8B achieves the top-performing next UI prediction, rivaling the competitive GPT-5 and Gemini-3-Pro-Image. Notably, Code2World significantly enhances downstream navigation success rates in a flexible manner, boosting Gemini-2.5-Flash by +9.5% on AndroidWorld navigation. The code is available at https://github.com/AMAP-ML/Code2World.

  </details>



- **Robust Vision Systems for Connected and Autonomous Vehicles: Security Challenges and Attack Vectors**  
  Sandeep Gupta, Roberto Passerone  
  _2026-02-10_ · https://arxiv.org/abs/2602.09740v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This article investigates the robustness of vision systems in Connected and Autonomous Vehicles (CAVs), which is critical for developing Level-5 autonomous driving capabilities. Safe and reliable CAV navigation undeniably depends on robust vision systems that enable accurate detection of objects, lane markings, and traffic signage. We analyze the key sensors and vision components essential for CAV navigation to derive a reference architecture for CAV vision system (CAVVS). This reference architecture provides a basis for identifying potential attack surfaces of CAVVS. Subsequently, we elaborate on identified attack vectors targeting each attack surface, rigorously evaluating their implications for confidentiality, integrity, and availability (CIA). Our study provides a comprehensive understanding of attack vector dynamics in vision systems, which is crucial for formulating robust security measures that can uphold the principles of the CIA triad.

  </details>



- **Doppler Effect: Analyses and Applications in Wireless Sensing and Communications**  
  Lie-Liang Yang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09955v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  This chapter is motivated by the need for a rigorous and comprehensive analysis of the Doppler effects encountered by electromagnetic and acoustic signals across a diverse spectrum of modern applications. These include land mobile communications, various Internet of Things (IoT) networks, machine-type communications (MTC), and various radar and satellite-based systems for navigation and sensing, as well as the emerging regime of integrated sensing and communications (ISAC). A wide array of kinematic profiles is investigated, ranging from uniform motion and constant acceleration to more complex general motion. Consequently, the multi-faceted factors influencing the Doppler shift are addressed in detail, encompassing classical kinematics, special and general relativity, atmospheric dynamics, and the properties of the propagation medium. This work is intended to establish a definitive theoretical foundation for both the general enthusiast and the specialized researcher seeking to master the complexities of signal frequency shifts in modern wireless sensing and communications systems.

  </details>



- **Bladder Vessel Segmentation using a Hybrid Attention-Convolution Framework**  
  Franziska Krauß, Matthias Ege, Zoltan Lovasz, Albrecht Bartz-Schmidt, Igor Tsaur, Oliver Sawodny, Carina Veil  
  _2026-02-10_ · https://arxiv.org/abs/2602.09949v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Urinary bladder cancer surveillance requires tracking tumor sites across repeated interventions, yet the deformable and hollow bladder lacks stable landmarks for orientation. While blood vessels visible during endoscopy offer a patient-specific "vascular fingerprint" for navigation, automated segmentation is challenged by imperfect endoscopic data, including sparse labels, artifacts like bubbles or variable lighting, continuous deformation, and mucosal folds that mimic vessels. State-of-the-art vessel segmentation methods often fail to address these domain-specific complexities. We introduce a Hybrid Attention-Convolution (HAC) architecture that combines Transformers to capture global vessel topology prior with a CNN that learns a residual refinement map to precisely recover thin-vessel details. To prioritize structural connectivity, the Transformer is trained on optimized ground truth data that exclude short and terminal branches. Furthermore, to address data scarcity, we employ a physics-aware pretraining, that is a self-supervised strategy using clinically grounded augmentations on unlabeled data. Evaluated on the BlaVeS dataset, consisting of endoscopic video frames, our approach achieves high accuracy (0.94) and superior precision (0.61) and clDice (0.66) compared to state-of-the-art medical segmentation models. Crucially, our method successfully suppresses false positives from mucosal folds that dynamically appear and vanish as the bladder fills and empties during surgery. Hence, HAC provides the reliable structural stability required for clinical navigation.

  </details>



- **Optimal Control of Microswimmers for Trajectory Tracking Using Bayesian Optimization**  
  Lucas Palazzolo, Mickaël Binois, Laëtitia Giraldi  
  _2026-02-10_ · https://arxiv.org/abs/2602.09563v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Trajectory tracking for microswimmers remains a key challenge in microrobotics, where low-Reynolds-number dynamics make control design particularly complex. In this work, we formulate the trajectory tracking problem as an optimal control problem and solve it using a combination of B-spline parametrization with Bayesian optimization, allowing the treatment of high computational costs without requiring complex gradient computations. Applied to a flagellated magnetic swimmer, the proposed method reproduces a variety of target trajectories, including biologically inspired paths observed in experimental studies. We further evaluate the approach on a three-sphere swimmer model, demonstrating that it can adapt to and partially compensate for wall-induced hydrodynamic effects. The proposed optimization strategy can be applied consistently across models of different fidelity, from low-dimensional ODE-based models to high-fidelity PDE-based simulations, showing its robustness and generality. These results highlight the potential of Bayesian optimization as a versatile tool for optimal control strategies in microscale locomotion under complex fluid-structure interactions.

  </details>


