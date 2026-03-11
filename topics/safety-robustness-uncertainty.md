# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-03-11 07:08 UTC_

Total papers shown: **20**


---

- **RESBev: Making BEV Perception More Robust**  
  Lifeng Zhuo, Kefan Jin, Zhe Liu, Hesheng Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09529v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Bird's-eye-view (BEV) perception has emerged as a cornerstone of autonomous driving systems, providing a structured, ego-centric representation critical for downstream planning and control. However, real-world deployment faces challenges from sensor degradation and adversarial attacks, which can cause severe perceptual anomalies and ultimately compromise the safety of autonomous driving systems. To address this, we propose a resilient and plug-and-play BEV perception method, RESBev, which can be easily applied to existing BEV perception methods to enhance their robustness to diverse disturbances. Specifically, we reframe perception robustness as a latent semantic prediction problem. A latent world model is constructed to extract spatiotemporal correlations across sequential BEV observations, thereby learning the underlying BEV state transitions to predict clean BEV features for reconstructing corrupted observations. The proposed framework operates at the semantic feature level of the Lift-Splat-Shoot pipeline, enabling recovery that generalizes across both natural disturbances and adversarial attacks without modifying the underlying backbone. Extensive experiments on the nuScenes dataset demonstrate that, with few-shot fine-tuning, RESBev significantly improves the robustness of existing BEV perception models against various external disturbances and adversarial attacks.

  </details>



- **Towards Terrain-Aware Safe Locomotion for Quadrupedal Robots Using Proprioceptive Sensing**  
  Peiyu Yang, Jiatao Ding, Wei Pan, Claudio Semini, Cosimo Della Santina  
  _2026-03-10_ · https://arxiv.org/abs/2603.09585v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Achieving safe quadrupedal locomotion in real-world environments has attracted much attention in recent years. When walking over uneven terrain, achieving reliable estimation and realising safety-critical control based on the obtained information is still an open question. To address this challenge, especially for low-cost robots equipped solely with proprioceptive sensors (e.g., IMUs, joint encoders, and contact force sensors), this work first presents an estimation framework that generates a 2.5-D terrain map and extracts support plane parameters, which are then integrated into contact and state estimation. Then, we integrate this estimation framework into a safety-critical control pipeline by formulating control barrier functions that provide rigorous safety guarantees. Experiments demonstrate that the proposed terrain estimation method provides smooth terrain representations. Moreover, the coupled estimation framework of terrain, state, and contact reduces the mean absolute error of base position estimation by 64.8%, decreases the estimation variance by 47.2%, and improves the robustness of contact estimation compared to a decoupled framework. The terrain-informed CBFs integrate historical terrain information and current proprioceptive measurements to ensure global safety by keeping the robot out of hazardous areas and local safety by preventing body-terrain collision, relying solely on proprioceptive sensing.

  </details>



- **VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM**  
  Anh Thuan Tran, Jana Kosecka  
  _2026-03-10_ · https://arxiv.org/abs/2603.09673v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties. To this end, we introduce VarSplat, an uncertainty-aware 3DGS-SLAM system that explicitly learns per-splat appearance variance. By using the law of total variance with alpha compositing, we then render differentiable per-pixel uncertainty map via efficient, single-pass rasterization. This map guides tracking, submap registration, and loop detection toward focusing on reliable regions and contributes to more stable optimization. Experimental results on Replica (synthetic) and TUM-RGBD, ScanNet, and ScanNet++ (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing studies for dense RGB-D SLAM.

  </details>



- **SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments**  
  Shiyi Chen, Mingye Yang, Haiyan Mao, Jiaqi Zhang, Haiyi Liu, Shuheng He, Debing Zhang, Zihao Qiu, Chun Zhang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09460v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Efficiently training quadruped robot navigation in densely cluttered environments remains a significant challenge. Existing methods are either limited by a lack of safety and agility in simple obstacle distributions or suffer from slow locomotion in complex environments, often requiring excessively long training phases. To this end, we propose SEA-Nav (Safe, Efficient, and Agile Navigation), a reinforcement learning framework for quadruped navigation. Within diverse and dense obstacle environments, a differentiable control barrier function (CBF)-based shield constraints the navigation policy to output safe velocity commands. An adaptive collision replay mechanism and hazardous exploration rewards are introduced to increase the probability of learning from critical experiences, guiding efficient exploration and exploitation. Finally, kinematic action constraints are incorporated to ensure safe velocity commands, facilitating successful physical deployment. To the best of our knowledge, this is the first approach that achieves highly challenging quadruped navigation in the real world with minute-level training time.

  </details>



- **OTPL-VIO: Robust Visual-Inertial Odometry with Optimal Transport Line Association and Adaptive Uncertainty**  
  Zikun Chen, Wentao Zhao, Yihe Niu, Tianchen Deng, Jingchuan Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09653v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Robust stereo visual-inertial odometry (VIO) remains challenging in low-texture scenes and under abrupt illumination changes, where point features become sparse and unstable, leading to ambiguous association and under-constrained estimation. Line structures offer complementary geometric cues, yet many efficient point-line systems still rely on point-guided line association, which can break down when point support is weak and may lead to biased constraints. We present a stereo point-line VIO system in which line segments are equipped with dedicated deep descriptors and matched using an entropy-regularized optimal transport formulation, enabling globally consistent correspondences under ambiguity, outliers, and partial observations. The proposed descriptor is training-free and is computed by sampling and pooling network feature maps. To improve estimation stability, we analyze the impact of line measurement noise and introduce reliability-adaptive weighting to regulate the influence of line constraints during optimization. Experiments on EuRoC and UMA-VI, together with real-world deployments in low-texture and illumination-challenging environments, demonstrate improved accuracy and robustness over representative baselines while maintaining real-time performance.

  </details>



- **Lightweight 3D LiDAR-Based UAV Tracking: An Adaptive Extended Kalman Filtering Approach**  
  Nivand Khosravi, Meysam Basiri, Rodrigo Ventura  
  _2026-03-10_ · https://arxiv.org/abs/2603.09783v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate relative positioning is crucial for swarm aerial robotics, enabling coordinated flight and collision avoidance. Although vision-based tracking has been extensively studied, 3D LiDAR-based methods remain underutilized despite their robustness under varying lighting conditions. Existing systems often rely on bulky, power-intensive sensors, making them impractical for small UAVs with strict payload and energy constraints. This paper presents a lightweight LiDAR-based UAV tracking system incorporating an Adaptive Extended Kalman Filter (AEKF) framework. Our approach effectively addresses the challenges posed by sparse, noisy, and nonuniform point cloud data generated by non-repetitive scanning 3D LiDARs, ensuring reliable tracking while remaining suitable for small drones with strict payload constraints. Unlike conventional filtering techniques, the proposed method dynamically adjusts the noise covariance matrices using innovation and residual statistics, thereby enhancing tracking accuracy under real-world conditions. Additionally, a recovery mechanism ensures continuity of tracking during temporary detection failures caused by scattered LiDAR returns or occlusions. Experimental validation was performed using a Livox Mid-360 LiDAR mounted on a DJI F550 UAV in real-world flight scenarios. The proposed method demonstrated robust UAV tracking performance under sparse LiDAR returns and intermittent detections, consistently outperforming both standard Kalman filtering and particle filtering approaches during aggressive maneuvers. These results confirm that the framework enables reliable relative positioning in GPS-denied environments without the need for multi-sensor arrays or external infrastructure.

  </details>



- **From Flow to One Step: Real-Time Multi-Modal Trajectory Policies via Implicit Maximum Likelihood Estimation-based Distribution Distillation**  
  Ju Dong, Liding Zhang, Lei Zhang, Yu Fu, Kaixin Bai, Zoltan-Csaba Marton, Zhenshan Bing, Zhaopeng Chen, Alois Christian Knoll, Jianwei Zhang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09415v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Generative policies based on diffusion and flow matching achieve strong performance in robotic manipulation by modeling multi-modal human demonstrations. However, their reliance on iterative Ordinary Differential Equation (ODE) integration introduces substantial latency, limiting high-frequency closed-loop control. Recent single-step acceleration methods alleviate this overhead but often exhibit distributional collapse, producing averaged trajectories that fail to execute coherent manipulation strategies. We propose a framework that distills a Conditional Flow Matching (CFM) expert into a fast single-step student via Implicit Maximum Likelihood Estimation (IMLE). A bi-directional Chamfer distance provides a set-level objective that promotes both mode coverage and fidelity, enabling preservation of the teacher multi-modal action distribution in a single forward pass. A unified perception encoder further integrates multi-view RGB, depth, point clouds, and proprioception into a geometry-aware representation. The resulting high-frequency control supports real-time receding-horizon re-planning and improved robustness under dynamic disturbances.

  </details>



- **$M^2$-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs**  
  Kaixin Lin, Kunyu Peng, Di Wen, Yufan Chen, Ruiping Liu, Kailun Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09737v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Semantic occupancy prediction enables dense 3D geometric and semantic understanding for autonomous driving. However, existing camera-based approaches implicitly assume complete surround-view observations, an assumption that rarely holds in real-world deployment due to occlusion, hardware malfunction, or communication failures. We study semantic occupancy prediction under incomplete multi-camera inputs and introduce $M^2$-Occ, a framework designed to preserve geometric structure and semantic coherence when views are missing. $M^2$-Occ addresses two complementary challenges. First, a Multi-view Masked Reconstruction (MMR) module leverages the spatial overlap among neighboring cameras to recover missing-view representations directly in the feature space. Second, a Feature Memory Module (FMM) introduces a learnable memory bank that stores class-level semantic prototypes. By retrieving and integrating these global priors, the FMM refines ambiguous voxel features, ensuring semantic consistency even when observational evidence is incomplete. We introduce a systematic missing-view evaluation protocol on the nuScenes-based SurroundOcc benchmark, encompassing both deterministic single-view failures and stochastic multi-view dropout scenarios. Under the safety-critical missing back-view setting, $M^2$-Occ improves the IoU by 4.93%. As the number of missing cameras increases, the robustness gap further widens; for instance, under the setting with five missing views, our method boosts the IoU by 5.01%. These gains are achieved without compromising full-view performance. The source code will be publicly released at https://github.com/qixi7up/M2-Occ.

  </details>



- **PRECEPT: Planning Resilience via Experience, Context Engineering & Probing Trajectories A Unified Framework for Test-Time Adaptation with Compositional Rule Learning and Pareto-Guided Prompt Evolution**  
  Arash Shahmansoori  
  _2026-03-10_ · https://arxiv.org/abs/2603.09641v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  LLM agents that store knowledge as natural language suffer steep retrieval degradation as condition count grows, often struggle to compose learned rules reliably, and typically lack explicit mechanisms to detect stale or adversarial knowledge. We introduce PRECEPT, a unified framework for test-time adaptation with three tightly coupled components: (1) deterministic exact-match rule retrieval over structured condition keys, (2) conflict-aware memory with Bayesian source reliability and threshold-based rule invalidation, and (3) COMPASS, a Pareto-guided prompt-evolution outer loop. Exact retrieval eliminates partial-match interpretation errors on the deterministic path (0% by construction, vs 94.4% under Theorem~B.6's independence model at N=10) and supports compositional stacking through a semantic tier hierarchy; conflict-aware memory resolves static--dynamic disagreements and supports drift adaptation; COMPASS evaluates prompts through the same end-to-end execution pipeline. Results (9--10 seeds): PRECEPT achieves a +41.1pp first-try advantage over Full Reflexion (d>1.9), +33.3pp compositional generalization (d=1.55), 100% $P_1$ on 2-way logistics compositions (d=2.64), +40--55pp continuous learning gains, strong eventual robustness under adversarial static knowledge (100% logistics with adversarial SK active; partial recovery on integration), +55.0pp drift recovery (d=0.95, p=0.031), and 61% fewer steps. Core comparisons are statistically significant, often at p<0.001.

  </details>



- **What Do We Care About in Bandits with Noncompliance? BRACE: Bandits with Recommendations, Abstention, and Certified Effects**  
  Nicolás Della Penna  
  _2026-03-10_ · https://arxiv.org/abs/2603.09532v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  Bandits with noncompliance separate the learner's recommendation from the treatment actually delivered, so the learning target itself must be chosen. A platform may care about recommendation welfare in the current mediated workflow, treatment learning for a future direct-control regime, or anytime-valid uncertainty for one of those targets. These objectives need not agree. We formalize this objective-choice problem, identify the direct-control regime in which recommendation and treatment objectives collapse, and show by example that recommendation welfare can strictly exceed every learner-measurable treatment policy when downstream actors use private information. For finite-context square-IV problems we propose BRACE, a parameter-free phase-doubling algorithm that performs IV inversion only after matrix certification and otherwise returns full-range but honest structural intervals. BRACE delivers simultaneous policy-value validity, fixed-gap identification of the operationally optimal recommendation policy, and fixed-gap identification of the structurally optimal treatment policy under contextual homogeneity and invertibility. We complement the theory with a finite-context empirical benchmark spanning direct control, mediated present-versus-future tradeoffs, weak identification, homogeneity failure, and rectangular overidentification. The experiments show that safety appears as regret on easy problems, as abstention and wide valid intervals under weak identification, as a reason to prefer recommendation welfare under homogeneity failure, and as tighter structural uncertainty when extra instruments are available. For rich contexts, we also derive an orthogonal score whose conditional bias factorizes into compliance-model and outcome-model errors, clarifying what must be stabilized for anytime-valid semiparametric IV inference.

  </details>



- **Robust Cooperative Localization in Featureless Environments: A Comparative Study of DCL, StCL, CCL, CI, and Standard-CL**  
  Nivand Khosravi, Meysam Basiri, Rodrigo Ventura  
  _2026-03-10_ · https://arxiv.org/abs/2603.09886v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Cooperative localization (CL) enables accurate position estimation in multi-robot systems operating in GPS-denied environments. This paper presents a comparative study of five CL approaches: Centralized Cooperative Localization (CCL), Decentralized Cooperative Localization (DCL), Sequential Cooperative Localization (StCL), Covariance Intersection (CI), and Standard Cooperative Localization (Standard-CL). All methods are implemented in ROS and evaluated through Monte Carlo simulations under two conditions: weak data association and robust detection. Our analysis reveals fundamental trade-offs among the methods. StCL and Standard-CL achieve the lowest position errors but exhibit severe filter inconsistency, making them unsuitable for safety-critical applications. DCL demonstrates remarkable stability under challenging conditions due to its measurement stride mechanism, which provides implicit regularization against outliers. CI emerges as the most balanced approach, achieving near-optimal consistency while maintaining competitive accuracy. CCL provides theoretically optimal estimation but shows sensitivity to measurement outliers. These findings offer practical guidance for selecting CL algorithms based on application requirements.

  </details>



- **Let's Reward Step-by-Step: Step-Aware Contrastive Alignment for Vision-Language Navigation in Continuous Environments**  
  Haoyuan Li, Rui Liu, Hehe Fan, Yi Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09740v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language Navigation in Continuous Environments (VLN-CE) requires agents to learn complex reasoning from long-horizon human interactions. While Multi-modal Large Language Models (MLLMs) have driven recent progress, current training paradigms struggle to balance generalization capability, error recovery and training stability. Specifically, (i) policies derived from SFT suffer from compounding errors, struggling to recover from out-of-distribution states, and (ii) Reinforcement Fine-Tuning (RFT) methods e.g. GRPO are bottlenecked by sparse outcome rewards. Their binary feedback fails to assign credit to individual steps, leading to gradient signal collapse in failure dominant batches. To address these challenges, we introduce Step-Aware Contrastive Alignment (SACA), a framework designed to extract dense supervision from imperfect trajectories. At its core, the Perception-Grounded Step-Aware auditor evaluates progress step-by-step, disentangling failed trajectories into valid prefixes and exact divergence points. Leveraging these signals, Scenario-Conditioned Group Construction mechanism dynamically routes batches to specialized resampling and optimization strategies. Extensive experiments on VLN-CE benchmarks demonstrate that SACA achieves state-of-the-art performance.

  </details>



- **DRIFT: Dual-Representation Inter-Fusion Transformer for Automated Driving Perception with 4D Radar Point Clouds**  
  Siqi Pei, Andras Palffy, Dariu M. Gavrila  
  _2026-03-10_ · https://arxiv.org/abs/2603.09695v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  4D radars, which provide 3D point cloud data along with Doppler velocity, are attractive components of modern automated driving systems due to their low cost and robustness under adverse weather conditions. However, they provide a significantly lower point cloud density than LiDAR sensors. This makes it important to exploit not only local but also global contextual scene information. This paper proposes DRIFT, a model that effectively captures and fuses both local and global contexts through a dual-path architecture. The model incorporates a point path to aggregate fine-grained local features and a pillar path to encode coarse-grained global features. These two parallel paths are intertwined via novel feature-sharing layers at multiple stages, enabling full utilization of both representations. DRIFT is evaluated on the widely used View-of-Delft (VoD) dataset and a proprietary internal dataset. It outperforms the baselines on the tasks of object detection and/or free road estimation. For example, DRIFT achieves a mean average precision (mAP) of 52.6\% (compared to, say, 45.4\% of CenterPoint) on the VoD dataset.

  </details>



- **MM-tau-p$^2$: Persona-Adaptive Prompting for Robust Multi-Modal Agent Evaluation in Dual-Control Settings**  
  Anupam Purwar, Aditya Choudhary  
  _2026-03-10_ · https://arxiv.org/abs/2603.09643v1 · `cs.ET`  
  <details><summary>Abstract</summary>

  Current evaluation frameworks and benchmarks for LLM powered agents focus on text chat driven agents, these frameworks do not expose the persona of user to the agent, thus operating in a user agnostic environment. Importantly, in customer experience management domain, the agent's behaviour evolves as the agent learns about user personality. With proliferation of real time TTS and multi-modal language models, LLM based agents are gradually going to become multi-modal. Towards this, we propose the MM-tau-p$^2$ benchmark with metrics for evaluating the robustness of multi-modal agents in dual control setting with and without persona adaption of user, while also taking user inputs in the planning process to resolve a user query. In particular, our work shows that even with state of-the-art frontier LLMs like GPT-5, GPT 4.1, there are additional considerations measured using metrics viz. multi-modal robustness, turn overhead while introducing multi-modality into LLM based agents. Overall, MM-tau-p$^2$ builds on our prior work FOCAL and provides a holistic way of evaluating multi-modal agents in an automated way by introducing 12 novel metrics. We also provide estimates of these metrics on the telecom and retail domains by using the LLM-as-judge approach using carefully crafted prompts with well defined rubrics for evaluating each conversation.

  </details>



- **Declarative Scenario-based Testing with RoadLogic**  
  Ezio Bartocci, Alessio Gambi, Felix Gigler, Cristinel Mateis, Dejan Ničković  
  _2026-03-10_ · https://arxiv.org/abs/2603.09455v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Scenario-based testing is a key method for cost-effective and safe validation of autonomous vehicles (AVs). Existing approaches rely on imperative scenario definitions, requiring developers to manually enumerate numerous variants to achieve coverage. Declarative languages, such as OpenSCENARIO DSL (OS2), raise the abstraction level but lack systematic methods for instantiating concrete, specification-compliant scenarios as simulations. To our knowledge, currently, no open-source solution provides this capability. We present RoadLogic that bridges declarative OS2 specifications and executable simulations. It uses Answer Set Programming to generate abstract plans satisfying scenario constraints, motion planning to refine the plans into feasible trajectories, and specification-based monitoring to verify correctness. We evaluate RoadLogic on instantiating representative OS2 scenarios as simulations in the CommonRoad framework. Results show that RoadLogic consistently produces realistic, specification-satisfying simulations within minutes and captures diverse behavioral variants through parameter sampling, thus opening the door to systematic scenario-based testing for autonomous driving systems.

  </details>



- **AI-Enabled Data-driven Intelligence for Spectrum Demand Estimation**  
  Colin Brown, Mohamad Alkadamani, Halim Yanikomeroglu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09916v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Accurately forecasting spectrum demand is a key component for efficient spectrum resource allocation and management. With the rapid growth in demand for wireless services, mobile network operators and regulators face increasing challenges in ensuring adequate spectrum availability. This paper presents a data-driven approach leveraging artificial intelligence (AI) and machine learning (ML) to estimate and manage spectrum demand. The approach uses multiple proxies of spectrum demand, drawing from site license data and derived from crowdsourced data. These proxies are validated against real-world mobile network traffic data to ensure reliability, achieving an R$^2$ value of 0.89 for an enhanced proxy. The proposed ML models are tested and validated across five major Canadian cities, demonstrating their generalizability and robustness. These contributions assist spectrum regulators in dynamic spectrum planning, enabling better resource allocation and policy adjustments to meet future network demands.

  </details>



- **DISPLAY: Directable Human-Object Interaction Video Generation via Sparse Motion Guidance and Multi-Task Auxiliary**  
  Jiazhi Guan, Quanwei Yang, Luying Huang, Junhao Liang, Borong Liang, Haocheng Feng, Wei He, Kaisiyuan Wang, Hang Zhou, Jingdong Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09883v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Human-centric video generation has advanced rapidly, yet existing methods struggle to produce controllable and physically consistent Human-Object Interaction (HOI) videos. Existing works rely on dense control signals, template videos, or carefully crafted text prompts, which limit flexibility and generalization to novel objects. We introduce a framework, namely DISPLAY, guided by Sparse Motion Guidance, composed only of wrist joint coordinates and a shape-agnostic object bounding box. This lightweight guidance alleviates the imbalance between human and object representations and enables intuitive user control. To enhance fidelity under such sparse conditions, we propose an Object-Stressed Attention mechanism that improves object robustness. To address the scarcity of high-quality HOI data, we further develop a Multi-Task Auxiliary Training strategy with a dedicated data curation pipeline, allowing the model to benefit from both reliable HOI samples and auxiliary tasks. Comprehensive experiments show that our method achieves high-fidelity, controllable HOI generation across diverse tasks. The project page can be found at \href{https://mumuwei.github.io/DISPLAY/}.

  </details>



- **VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models**  
  Shuhao Kang, Youqi Liao, Peijie Wang, Wenlong Liao, Qilin Zhang, Benjamin Busam, Xieyuanli Chen, Yun Liu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09826v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Text-to-point-cloud (T2P) localization aims to infer precise spatial positions within 3D point cloud maps from natural language descriptions, reflecting how humans perceive and communicate spatial layouts through language. However, existing methods largely rely on shallow text-point cloud correspondence without effective spatial reasoning, limiting their accuracy in complex environments. To address this limitation, we propose VLM-Loc, a framework that leverages the spatial reasoning capability of large vision-language models (VLMs) for T2P localization. Specifically, we transform point clouds into bird's-eye-view (BEV) images and scene graphs that jointly encode geometric and semantic context, providing structured inputs for the VLM to learn cross-modal representations bridging linguistic and spatial semantics. On top of these representations, we introduce a partial node assignment mechanism that explicitly associates textual cues with scene graph nodes, enabling interpretable spatial reasoning for accurate localization. To facilitate systematic evaluation across diverse scenes, we present CityLoc, a benchmark built from multi-source point clouds for fine-grained T2P localization. Experiments on CityLoc demonstrate VLM-Loc achieves superior accuracy and robustness compared to state-of-the-art methods. Our code, model, and dataset are available at \href{https://github.com/MCG-NKU/nku-3d-vision}{repository}.

  </details>



- **TIMID: Time-Dependent Mistake Detection in Videos of Robot Executions**  
  Nerea Gallego, Fernando Salanova, Claudio Mannarano, Cristian Mahulea, Eduardo Montijano  
  _2026-03-10_ · https://arxiv.org/abs/2603.09782v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  As robotic systems execute increasingly difficult task sequences, so does the number of ways in which they can fail. Video Anomaly Detection (VAD) frameworks typically focus on singular, low-level kinematic or action failures, struggling to identify more complex temporal or spatial task violations, because they do not necessarily manifest as low-level execution errors. To address this problem, the main contribution of this paper is a new VAD-inspired architecture, TIMID, which is able to detect robot time-dependent mistakes when executing high-level tasks. Our architecture receives as inputs a video and prompts of the task and the potential mistake, and returns a frame-level prediction in the video of whether the mistake is present or not. By adopting a VAD formulation, the model can be trained with weak supervision, requiring only a single label per video. Additionally, to alleviate the problem of data scarcity of incorrect executions, we introduce a multi-robot simulation dataset with controlled temporal errors and real executions for zero-shot sim-to-real evaluation. Our experiments demonstrate that out-of-the-box VLMs lack the explicit temporal reasoning required for this task, whereas our framework successfully detects different types of temporal errors. Project: https://ropertunizar.github.io/TIMID/

  </details>



- **PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments**  
  Guoliang Zhu, Wanjun Jia, Caoyang Shao, Yuheng Zhang, Zhiyong Li, Kailun Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09760v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Global perception is essential for embodied agents in 360° spaces, yet current affordance grounding remains largely object-centric and restricted to perspective views. To bridge this gap, we introduce a novel task: Holistic Affordance Grounding in 360° Indoor Environments. This task faces unique challenges, including severe geometric distortions from Equirectangular Projection (ERP), semantic dispersion, and cross-scale alignment difficulties. We propose PanoAffordanceNet, an end-to-end framework featuring a Distortion-Aware Spectral Modulator (DASM) for latitude-dependent calibration and an Omni-Spherical Densification Head (OSDH) to restore topological continuity from sparse activations. By integrating multi-level constraints comprising pixel-wise, distributional, and region-text contrastive objectives, our framework effectively suppresses semantic drift under low supervision. Furthermore, we construct 360-AGD, the first high-quality panoramic affordance grounding dataset. Extensive experiments demonstrate that PanoAffordanceNet significantly outperforms existing methods, establishing a solid baseline for scene-level perception in embodied intelligence. The source code and benchmark dataset will be made publicly available at https://github.com/GL-ZHU925/PanoAffordanceNet.

  </details>


