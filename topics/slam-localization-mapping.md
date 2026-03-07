# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-03-07 06:56 UTC_

Total papers shown: **14**


---

- **VinePT-Map: Pole-Trunk Semantic Mapping for Resilient Autonomous Robotics in Vineyards**  
  Giorgio Audrito, Mauro Martini, Alessandro Navone, Giorgia Galluzzo, Marcello Chiaberge  
  _2026-03-05_ · https://arxiv.org/abs/2603.05070v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable long-term deployment of autonomous robots in agricultural environments remains challenging due to perceptual aliasing, seasonal variability, and the dynamic nature of crop canopies. Vineyards, characterized by repetitive row structures and significant visual changes across phenological stages, represent a pivotal field challenge, limiting the robustness of conventional feature-based localization and mapping approaches. This paper introduces VinePT-Map, a semantic mapping framework that leverages vine trunks and support poles as persistent structural landmarks to enable season-agnostic and resilient robot localization. The proposed method formulates the mapping problem as a factor graph, integrating GPS, IMU, and RGB-D observations through robust geometrical constraints that exploit vineyard structure. An efficient perception pipeline based on instance segmentation and tracking, combined with a clustering filter for outlier rejection and pose refinement, enables accurate landmark detection using low-cost sensors and onboard computation. To validate the pipeline, we present a multi-season dataset for trunk and pole segmentation and tracking. Extensive field experiments conducted across diverse seasons demonstrate the robustness and accuracy of the proposed approach, highlighting its suitability for long-term autonomous operation in agricultural environments.

  </details>



- **Direct Contact-Tolerant Motion Planning With Vision Language Models**  
  He Li, Jian Sun, Chengyang Li, Guoliang Li, Qiyu Ruan, Shuai Wang, Chengzhong Xu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05017v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Navigation in cluttered environments often requires robots to tolerate contact with movable or deformable objects to maintain efficiency. Existing contact-tolerant motion planning (CTMP) methods rely on indirect spatial representations (e.g., prebuilt map, obstacle set), resulting in inaccuracies and a lack of adaptiveness to environmental uncertainties. To address this issue, we propose a direct contact-tolerant (DCT) planner, which integrates vision-language models (VLMs) into direct point perception and navigation, including two key components. The first one is VLM point cloud partitioner (VPP), which performs contact-tolerance reasoning in image space using VLM, caches inference masks, propagates them across frames using odometry, and projects them onto the current scan to generate a contact-aware point cloud. The second innovation is VPP guided navigation (VGN), which formulates CTMP as a perception-to-control optimization problem under direct contact-aware point cloud constraints, which is further solved by a specialized deep neural network (DNN). We implement DCT in Isaac Sim and a real car-like robot, demonstrating that DCT achieves robust and efficient navigation in cluttered environments with movable obstacles, outperforming representative baselines across diverse metrics. The code is available at: https://github.com/ChrisLeeUM/DCT.

  </details>



- **AIM-SLAM: Dense Monocular SLAM via Adaptive and Informative Multi-View Keyframe Prioritization with Foundation Model**  
  Jinwoo Jeon, Dong-Uk Seo, Eungchang Mason Lee, Hyun Myung  
  _2026-03-05_ · https://arxiv.org/abs/2603.05097v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advances in geometric foundation models have emerged as a promising alternative for addressing the challenge of dense reconstruction in monocular visual simultaneous localization and mapping (SLAM). Although geometric foundation models enable SLAM to leverage variable input views, the previous methods remain confined to two-view pairs or fixed-length inputs without sufficient deliberation of geometric context for view selection. To tackle this problem, we propose AIM-SLAM, a dense monocular SLAM framework that exploits an adaptive and informative multi-view keyframe prioritization with dense pointmap predictions from visual geometry grounded transformer (VGGT). Specifically, we introduce the selective information- and geometric-aware multi-view adaptation (SIGMA) module, which employs voxel overlap and information gain to retrieve a candidate set of keyframes and adaptively determine its size. Furthermore, we formulate a joint multi-view Sim(3) optimization that enforces consistent alignment across selected views, substantially improving pose estimation accuracy. The effectiveness of AIM-SLAM is demonstrated on real-world datasets, where it achieves state-of-the-art performance in both pose estimation and dense reconstruction. Our system supports ROS integration, with code is available at https://aimslam.github.io/.

  </details>



- **Loop Closure via Maximal Cliques in 3D LiDAR-Based SLAM**  
  Javier Laserna, Saurabh Gupta, Oscar Martinez Mozos, Cyrill Stachniss, Pablo San Segundo  
  _2026-03-05_ · https://arxiv.org/abs/2603.05397v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable loop closure detection remains a critical challenge in 3D LiDAR-based SLAM, especially under sensor noise, environmental ambiguity, and viewpoint variation conditions. RANSAC is often used in the context of loop closures for geometric model fitting in the presence of outliers. However, this approach may fail, leading to map inconsistency. We introduce a novel deterministic algorithm, CliReg, for loop closure validation that replaces RANSAC verification with a maximal clique search over a compatibility graph of feature correspondences. This formulation avoids random sampling and increases robustness in the presence of noise and outliers. We integrated our approach into a real- time pipeline employing binary 3D descriptors and a Hamming distance embedding binary search tree-based matching. We evaluated it on multiple real-world datasets featuring diverse LiDAR sensors. The results demonstrate that our proposed technique consistently achieves a lower pose error and more reliable loop closures than RANSAC, especially in sparse or ambiguous conditions. Additional experiments on 2D projection-based maps confirm its generality across spatial domains, making our approach a robust and efficient alternative for loop closure detection.

  </details>



- **OpenFrontier: General Navigation with Visual-Language Grounded Frontiers**  
  Esteban Padilla, Boyang Sun, Marc Pollefeys, Hermann Blum  
  _2026-03-05_ · https://arxiv.org/abs/2603.05377v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Open-world navigation requires robots to make decisions in complex everyday environments while adapting to flexible task requirements. Conventional navigation approaches often rely on dense 3D reconstruction and hand-crafted goal metrics, which limits their generalization across tasks and environments. Recent advances in vision--language navigation (VLN) and vision--language--action (VLA) models enable end-to-end policies conditioned on natural language, but typically require interactive training, large-scale data collection, or task-specific fine-tuning with a mobile agent. We formulate navigation as a sparse subgoal identification and reaching problem and observe that providing visual anchoring targets for high-level semantic priors enables highly efficient goal-conditioned navigation. Based on this insight, we select navigation frontiers as semantic anchors and propose OpenFrontier, a training-free navigation framework that seamlessly integrates diverse vision--language prior models. OpenFrontier enables efficient navigation with a lightweight system design, without dense 3D mapping, policy training, or model fine-tuning. We evaluate OpenFrontier across multiple navigation benchmarks and demonstrate strong zero-shot performance, as well as effective real-world deployment on a mobile robot.

  </details>



- **Constraint-Free Static Modeling of Continuum Parallel Robot**  
  Lingxiao Xun, Matyas Diezinger, Azad Artinian, Guillaume Laurent, Brahim Tamadazte  
  _2026-03-05_ · https://arxiv.org/abs/2603.05309v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Continuum parallel robots (CPR) combine rigid actuation mechanisms with multiple elastic rods in a closed-loop topology, making forward statics challenging when rigid--continuum junctions are enforced by explicit kinematic constraints. Such constraint-based formulations typically introduce additional algebraic variables and complicate both numerical solution and downstream control. This paper presents a geometric exact, configuration-based and constraint-free static model of CPR that remains valid under geometrically nonlinear, large-deformation and large-rotation conditions. Connectivity constraints are eliminated by kinematic embedding, yielding a reduced unconstrained problem. Each rod of CPR is discretized by nodal poses on SE(3), while the element-wise strain field is reconstructed through a linear strain parameterization. A fourth-order Magnus approximation yields an explicit and geometrically consistent mapping between element end poses and the strain. Rigid attachments at the motor-driven base and the end-effector platforms are handled through kinematic embeddings. Based on total potential energy and virtual work, we derive assembly-ready residuals and explicit Newton tangents, and solve the resulting nonlinear equilibrium equations using a Riemannian Newton iteration on the product manifold. Experiments on a three-servomotor, six-rod prototype validate the model by showing good agreement between simulation and measurements for both unloaded motions and externally loaded cases.

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



- **SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning**  
  Ye-Chan Kim, SeungJu Cha, Si-Woo Kim, Minju Jeon, Hyungee Kim, Dong-Jin Kim  
  _2026-03-05_ · https://arxiv.org/abs/2603.05437v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Weakly-Supervised Dense Video Captioning aims to localize and describe events in videos trained only on caption annotations, without temporal boundaries. Prior work introduced an implicit supervision paradigm based on Gaussian masking and complementary captioning. However, existing method focuses merely on generating non-overlapping masks without considering their semantic relationship to corresponding events, resulting in simplistic, uniformly distributed masks that fail to capture semantically meaningful regions. Moreover, relying solely on ground-truth captions leads to sub-optimal performance due to the inherent sparsity of existing datasets. In this work, we propose SAIL, which constructs semantically-aware masks through cross-modal alignment. Our similarity aware training objective guides masks to emphasize video regions with high similarity to their corresponding event captions. Furthermore, to guide more accurate mask generation under sparse annotation settings, we introduce an LLM-based augmentation strategy that generates synthetic captions to provide additional alignment signals. These synthetic captions are incorporated through an inter-mask mechanism, providing auxiliary guidance for precise temporal localization without degrading the main objective. Experiments on ActivityNet Captions and YouCook2 demonstrate state-of-the-art performance on both captioning and localization metrics.

  </details>



- **X-RAY: Mapping LLM Reasoning Capability via Formalized and Calibrated Probes**  
  Gao Tianxi, Cai Yufan, Yuan Yusi, Dong Jin Song  
  _2026-03-05_ · https://arxiv.org/abs/2603.05290v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) achieve promising performance, yet their ability to reason remains poorly understood. Existing evaluations largely emphasize task-level accuracy, often conflating pattern matching with reasoning capability. We present X-RAY, an explainable reasoning analysis system that maps the LLM reasoning capability using calibrated, formally verified probes. We model reasoning capability as a function of extractable \textit{structure}, operationalized through formal properties such as constraint interaction, reasoning depth, and solution-space geometry. X-Ray generates probes via formal tools with controlled structural variations, enabling precise isolation of incremental structural information through formal calibration and verification. We evaluate state-of-the-art LLMs on problems ranging from junior-level to advanced in mathematics, physics, and chemistry. Our analysis reveals a systematic asymmetry in LLM reasoning: models are relatively robust to constraint refinement, where additional conditions shrink an existing solution space, but degrade sharply under solution-space restructuring, where modifications alter the underlying structural form of the solution manifold. Moreover, calibrated formal probes differentiate models that appear indistinguishable on standard benchmarks and reveal failure modes that are structurally interpretable rather than opaque. Beyond evaluation, our framework is contamination-free and supports the training and testing of reasoning models.

  </details>



- **Beyond Word Error Rate: Auditing the Diversity Tax in Speech Recognition through Dataset Cartography**  
  Ting-Hui Cheng, Line H. Clemmensen, Sneha Das  
  _2026-03-05_ · https://arxiv.org/abs/2603.05267v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Automatic speech recognition (ASR) systems are predominantly evaluated using the Word Error Rate (WER). However, raw token-level metrics fail to capture semantic fidelity and routinely obscures the `diversity tax', the disproportionate burden on marginalized and atypical speaker due to systematic recognition failures. In this paper, we explore the limitations of relying solely on lexical counts by systematically evaluating a broader class of non-linear and semantic metrics. To enable rigorous model auditing, we introduce the sample difficulty index (SDI), a novel metric that quantifies how intrinsic demographic and acoustic factors drive model failure. By mapping SDI on data cartography, we demonstrate that metrics EmbER and SemDist expose hidden systemic biases and inter-model disagreements that WER ignores. Finally, our findings are the first steps towards a robust audit framework for prospective safety analysis, empowering developers to audit and mitigate ASR disparities prior to deployment.

  </details>



- **C2-Faith: Benchmarking LLM Judges for Causal and Coverage Faithfulness in Chain-of-Thought Reasoning**  
  Avni Mittal, Rauno Arike  
  _2026-03-05_ · https://arxiv.org/abs/2603.05167v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) are increasingly used as judges of chain-of-thought (CoT) reasoning, but it remains unclear whether they can reliably assess process faithfulness rather than just answer plausibility. We introduce C2-Faith, a benchmark built from PRM800K that targets two complementary dimensions of faithfulness: causality (does each step logically follow from prior context?) and coverage (are essential intermediate inferences present?). Using controlled perturbations, we create examples with known causal error positions by replacing a single step with an acausal variant, and with controlled coverage deletions at varying deletion rates (scored against reference labels). We evaluate three frontier judges under three tasks: binary causal detection, causal step localization, and coverage scoring. The results show that model rankings depend strongly on task framing, with no single judge dominating all settings; all judges exhibit a substantial gap between detecting an error and localizing it; and coverage judgments are systematically inflated for incomplete reasoning. These findings clarify when LLM judges are dependable and where they fail, and provide practical guidance for selecting judges in process-level evaluation

  </details>



- **GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization through EM-Guided Decomposition and Temporal Refinement**  
  Xiaodong Zhu, Yuanming Zheng, Suting Wang, Junqi Yang, Yuhong Yang, Weiping Tu, Zhongyuan Wang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05095v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Temporal Forgery Localization (TFL) aims to precisely identify manipulated segments within videos or audio streams, providing interpretable evidence for multimedia forensics and security. While most existing TFL methods rely on dense frame-level labels in a fully supervised manner, Weakly Supervised TFL (WS-TFL) reduces labeling cost by learning only from binary video-level labels. However, current WS-TFL approaches suffer from mismatched training and inference objectives, limited supervision from binary labels, gradient blockage caused by non-differentiable top-k aggregation, and the absence of explicit modeling of inter-proposal relationships. To address these issues, we propose GEM-TFL (Graph-based EM-powered Temporal Forgery Localization), a two-phase classification-regression framework that effectively bridges the supervision gap between training and inference. Built upon this foundation, (1) we enhance weak supervision by reformulating binary labels into multi-dimensional latent attributes through an EM-based optimization process; (2) we introduce a training-free temporal consistency refinement that realigns frame-level predictions for smoother temporal dynamics; and (3) we design a graph-based proposal refinement module that models temporal-semantic relationships among proposals for globally consistent confidence estimation. Extensive experiments on benchmark datasets demonstrate that GEM-TFL achieves more accurate and robust temporal forgery localization, substantially narrowing the gap with fully supervised methods.

  </details>



- **A 360-degree Multi-camera System for Blue Emergency Light Detection Using Color Attention RT-DETR and the ABLDataset**  
  Francisco Vacalebri-Lloret, Lucas Banchero, Jose J. Lopez, Jose M. Mossi  
  _2026-03-05_ · https://arxiv.org/abs/2603.05058v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This study presents an advanced system for detecting blue lights on emergency vehicles, developed using ABLDataset, a curated dataset that includes images of European emergency vehicles under various climatic and geographic conditions. The system employs a configuration of four fisheye cameras, each with a 180-degree horizontal field of view, mounted on the sides of the vehicle. A calibration process enables the azimuthal localization of the detections. Additionally, a comparative analysis of major deep neural network algorithms was conducted, including YOLO (v5, v8, and v10), RetinaNet, Faster R-CNN, and RT-DETR. RT-DETR was selected as the base model and enhanced through the incorporation of a color attention block, achieving an accuracy of 94.7 percent and a recall of 94.1 percent on the test set, with field test detections reaching up to 70 meters. Furthermore, the system estimates the approach angle of the emergency vehicle relative to the center of the car using geometric transformations. Designed for integration into a multimodal system that combines visual and acoustic data, this system has demonstrated high efficiency, offering a promising approach to enhancing Advanced Driver Assistance Systems (ADAS) and road safety.

  </details>


