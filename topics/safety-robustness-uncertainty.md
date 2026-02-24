# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-24 07:13 UTC_

Total papers shown: **20**


---

- **Contextual Safety Reasoning and Grounding for Open-World Robots**  
  Zachary Ravichadran, David Snyder, Alexander Robey, Hamed Hassani, Vijay Kumar, George J. Pappas  
  _2026-02-23_ · https://arxiv.org/abs/2602.19983v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots are increasingly operating in open-world environments where safe behavior depends on context: the same hallway may require different navigation strategies when crowded versus empty, or during an emergency versus normal operations. Traditional safety approaches enforce fixed constraints in user-specified contexts, limiting their ability to handle the open-ended contextual variability of real-world deployment. We address this gap via CORE, a safety framework that enables online contextual reasoning, grounding, and enforcement without prior knowledge of the environment (e.g., maps or safety specifications). CORE uses a vision-language model (VLM) to continuously reason about context-dependent safety rules directly from visual observations, grounds these rules in the physical environment, and enforces the resulting spatially-defined safe sets via control barrier functions. We provide probabilistic safety guarantees for CORE that account for perceptual uncertainty, and we demonstrate through simulation and real-world experiments that CORE enforces contextually appropriate behavior in unseen environments, significantly outperforming prior semantic safety methods that lack online contextual reasoning. Ablation studies validate our theoretical guarantees and underscore the importance of both VLM-based reasoning and spatial grounding for enforcing contextual safety in novel settings. We provide additional resources at https://zacravichandran.github.io/CORE.

  </details>



- **Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception**  
  Yihang Tao, Senkang Hu, Haonan An, Zhengru Fang, Hangcheng Cao, Yuguang Fang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19596v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Collaborative perception (CP) enables data sharing among connected and autonomous vehicles (CAVs) to enhance driving safety. However, CP systems are vulnerable to adversarial attacks where malicious agents forge false objects via feature-level perturbations. Current defensive systems use threshold-based consensus verification by comparing collaborative and ego detection results. Yet, these defenses remain vulnerable to more sophisticated attack strategies that could exploit two critical weaknesses: (i) lack of robustness against attacks with systematic timing and target region optimization, and (ii) inadvertent disclosure of vulnerability knowledge through implicit confidence information in shared collaboration data. In this paper, we propose MVIG attack, a novel adaptive adversarial CP framework learning to capture vulnerability knowledge disclosed by different defensive CP systems from a unified mutual view information graph (MVIG) representation. Our approach combines MVIG representation with temporal graph learning to generate evolving fabrication risk maps and employs entropy-aware vulnerability search to optimize attack location, timing and persistence, enabling adaptive attacks with generalizability across various defensive configurations. Extensive evaluations on OPV2V and Adv-OPV2V datasets demonstrate that MVIG attack reduces defense success rates by up to 62\% against state-of-the-art defenses while achieving 47\% lower detection for persistent attacks at 29.9 FPS, exposing critical security gaps in CP systems. Code will be released at https://github.com/yihangtao/MVIG.git

  </details>



- **MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving**  
  Junli Wang, Xueyi Liu, Yinan Zheng, Zebing Xing, Pengfei Li, Guang Li, Kun Ma, Guang Chen, Hangjun Ye, Zhongpu Xia, et al.  
  _2026-02-23_ · https://arxiv.org/abs/2602.20060v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Generative models have shown great potential in trajectory planning. Recent studies demonstrate that anchor-guided generative models are effective in modeling the uncertainty of driving behaviors and improving overall performance. However, these methods rely on discrete anchor vocabularies that must sufficiently cover the trajectory distribution during testing to ensure robustness, inducing an inherent trade-off between vocabulary size and model performance. To overcome this limitation, we propose MeanFuser, an end-to-end autonomous driving method that enhances both efficiency and robustness through three key designs. (1) We introduce Gaussian Mixture Noise (GMN) to guide generative sampling, enabling a continuous representation of the trajectory space and eliminating the dependency on discrete anchor vocabularies. (2) We adapt ``MeanFlow Identity" to end-to-end planning, which models the mean velocity field between GMN and trajectory distribution instead of the instantaneous velocity field used in vanilla flow matching methods, effectively eliminating numerical errors from ODE solvers and significantly accelerating inference. (3) We design a lightweight Adaptive Reconstruction Module (ARM) that enables the model to implicitly select from all sampled proposals or reconstruct a new trajectory when none is satisfactory via attention weights. Experiments on the NAVSIM closed-loop benchmark demonstrate that MeanFuser achieves outstanding performance without the supervision of the PDM Score. and exceptional inference efficiency, offering a robust and efficient solution for end-to-end autonomous driving. Our code and model are available at https://github.com/wjl2244/MeanFuser.

  </details>



- **VGGT-MPR: VGGT-Enhanced Multimodal Place Recognition in Autonomous Driving Environments**  
  Jingyi Xu, Zhangshuo Qi, Zhongmiao Yan, Xuyu Gao, Qianyun Jiao, Songpengcheng Xia, Xieyuanli Chen, Ling Pei  
  _2026-02-23_ · https://arxiv.org/abs/2602.19735v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In autonomous driving, robust place recognition is critical for global localization and loop closure detection. While inter-modality fusion of camera and LiDAR data in multimodal place recognition (MPR) has shown promise in overcoming the limitations of unimodal counterparts, existing MPR methods basically attend to hand-crafted fusion strategies and heavily parameterized backbones that require costly retraining. To address this, we propose VGGT-MPR, a multimodal place recognition framework that adopts the Visual Geometry Grounded Transformer (VGGT) as a unified geometric engine for both global retrieval and re-ranking. In the global retrieval stage, VGGT extracts geometrically-rich visual embeddings through prior depth-aware and point map supervision, and densifies sparse LiDAR point clouds with predicted depth maps to improve structural representation. This enhances the discriminative ability of fused multimodal features and produces global descriptors for fast retrieval. Beyond global retrieval, we design a training-free re-ranking mechanism that exploits VGGT's cross-view keypoint-tracking capability. By combining mask-guided keypoint extraction with confidence-aware correspondence scoring, our proposed re-ranking mechanism effectively refines retrieval results without additional parameter optimization. Extensive experiments on large-scale autonomous driving benchmarks and our self-collected data demonstrate that VGGT-MPR achieves state-of-the-art performance, exhibiting strong robustness to severe environmental changes, viewpoint shifts, and occlusions. Our code and data will be made publicly available.

  </details>



- **BarrierSteer: LLM Safety via Learning Barrier Steering**  
  Thanh Q. Tran, Arun Verma, Kiwan Wong, Bryan Kian Hsiang Low, Daniela Rus, Wei Xiao  
  _2026-02-23_ · https://arxiv.org/abs/2602.20102v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Despite the state-of-the-art performance of large language models (LLMs) across diverse tasks, their susceptibility to adversarial attacks and unsafe content generation remains a major obstacle to deployment, particularly in high-stakes settings. Addressing this challenge requires safety mechanisms that are both practically effective and supported by rigorous theory. We introduce BarrierSteer, a novel framework that formalizes response safety by embedding learned non-linear safety constraints directly into the model's latent representation space. BarrierSteer employs a steering mechanism based on Control Barrier Functions (CBFs) to efficiently detect and prevent unsafe response trajectories during inference with high precision. By enforcing multiple safety constraints through efficient constraint merging, without modifying the underlying LLM parameters, BarrierSteer preserves the model's original capabilities and performance. We provide theoretical results establishing that applying CBFs in latent space offers a principled and computationally efficient approach to enforcing safety. Our experiments across multiple models and datasets show that BarrierSteer substantially reduces adversarial success rates, decreases unsafe generations, and outperforms existing methods.

  </details>



- **The LLMbda Calculus: AI Agents, Conversations, and Information Flow**  
  Zac Garby, Andrew D. Gordon, David Sands  
  _2026-02-23_ · https://arxiv.org/abs/2602.20064v1 · `cs.PL`  
  <details><summary>Abstract</summary>

  A conversation with a large language model (LLM) is a sequence of prompts and responses, with each response generated from the preceding conversation. AI agents build such conversations automatically: given an initial human prompt, a planner loop interleaves LLM calls with tool invocations and code execution. This tight coupling creates a new and poorly understood attack surface. A malicious prompt injected into a conversation can compromise later reasoning, trigger dangerous tool calls, or distort final outputs. Despite the centrality of such systems, we currently lack a principled semantic foundation for reasoning about their behaviour and safety. We address this gap by introducing an untyped call-by-value lambda calculus enriched with dynamic information-flow control and a small number of primitives for constructing prompt-response conversations. Our language includes a primitive that invokes an LLM: it serializes a value, sends it to the model as a prompt, and parses the response as a new term. This calculus faithfully represents planner loops and their vulnerabilities, including the mechanisms by which prompt injection alters subsequent computation. The semantics explicitly captures conversations, and so supports reasoning about defenses such as quarantined sub-conversations, isolation of generated code, and information-flow restrictions on what may influence an LLM call. A termination-insensitive noninterference theorem establishes integrity and confidentiality guarantees, demonstrating that a formal calculus can provide rigorous foundations for safe agentic programming.

  </details>



- **Agentic AI for Scalable and Robust Optical Systems Control**  
  Zehao Wang, Mingzhe Han, Wei Cheng, Yue-Kai Huang, Philip Ji, Denton Wu, Mahdi Safari, Flemming Holtorf, Kenaish AlQubaisi, Norbert M. Linke, et al.  
  _2026-02-23_ · https://arxiv.org/abs/2602.20144v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  We present AgentOptics, an agentic AI framework for high-fidelity, autonomous optical system control built on the Model Context Protocol (MCP). AgentOptics interprets natural language tasks and executes protocol-compliant actions on heterogeneous optical devices through a structured tool abstraction layer. We implement 64 standardized MCP tools across 8 representative optical devices and construct a 410-task benchmark to evaluate request understanding, role-aware responses, multi-step coordination, robustness to linguistic variation, and error handling. We assess two deployment configurations--commercial online LLMs and locally hosted open-source LLMs--and compare them with LLM-based code generation baselines. AgentOptics achieves 87.7%--99.0% average task success rates, significantly outperforming code-generation approaches, which reach up to 50% success. We further demonstrate broader applicability through five case studies extending beyond device-level control to system orchestration, monitoring, and closed-loop optimization. These include DWDM link provisioning and coordinated monitoring of coherent 400 GbE and analog radio-over-fiber (ARoF) channels; autonomous characterization and bias optimization of a wideband ARoF link carrying 5G fronthaul traffic; multi-span channel provisioning with launch power optimization; closed-loop fiber polarization stabilization; and distributed acoustic sensing (DAS)-based fiber monitoring with LLM-assisted event detection. These results establish AgentOptics as a scalable, robust paradigm for autonomous control and orchestration of heterogeneous optical systems.

  </details>



- **Athena: An Autonomous Open-Hardware Tracked Rescue Robot Platform**  
  Stefan Fabian, Aljoscha Schmidt, Jonas Süß, Dishant, Aum Oza, Oskar von Stryk  
  _2026-02-23_ · https://arxiv.org/abs/2602.19898v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In disaster response and situation assessment, robots have great potential in reducing the risks to the safety and health of first responders. As the situations encountered and the required capabilities of the robots deployed in such missions differ wildly and are often not known in advance, heterogeneous fleets of robots are needed to cover a wide range of mission requirements. While UAVs can quickly survey the mission environment, their ability to carry heavy payloads such as sensors and manipulators is limited. UGVs can carry required payloads to assess and manipulate the mission environment, but need to be able to deal with difficult and unstructured terrain such as rubble and stairs. The ability of tracked platforms with articulated arms (flippers) to reconfigure their geometry makes them particularly effective for navigating challenging terrain. In this paper, we present Athena, an open-hardware rescue ground robot research platform with four individually reconfigurable flippers and a reliable low-cost remote emergency stop (E-Stop) solution. A novel mounting solution using an industrial PU belt and tooth inserts allows the replacement and testing of different track profiles. The manipulator with a maximum reach of 1.54m can be used to operate doors, valves, and other objects of interest. Full CAD & PCB files, as well as all low-level software, are released as open-source contributions.

  </details>



- **OpenClaw, Moltbook, and ClawdLab: From Agent-Only Social Networks to Autonomous Scientific Research**  
  Lukas Weidener, Marko Brkić, Mihailo Jovanović, Ritvik Singh, Emre Ulgac, Aakaash Meduri  
  _2026-02-23_ · https://arxiv.org/abs/2602.19810v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  In January 2026, the open-source agent framework OpenClaw and the agent-only social network Moltbook produced a large-scale dataset of autonomous AI-to-AI interaction, attracting six academic publications within fourteen days. This study conducts a multivocal literature review of that ecosystem and presents ClawdLab, an open-source platform for autonomous scientific research, as a design science response to the architectural failure modes identified. The literature documents emergent collective phenomena, security vulnerabilities spanning 131 agent skills and over 15,200 exposed control panels, and five recurring architectural patterns. ClawdLab addresses these failure modes through hard role restrictions, structured adversarial critique, PI-led governance, multi-model orchestration, and domain-specific evidence requirements encoded as protocol constraints that ground validation in computational tool outputs rather than social consensus; the architecture provides emergent Sybil resistance as a structural consequence. A three-tier taxonomy distinguishes single-agent pipelines, predetermined multi-agent workflows, and fully decentralised systems, analysing why leading AI co-scientist platforms remain confined to the first two tiers. ClawdLab's composable third-tier architecture, in which foundation models, capabilities, governance, and evidence requirements are independently modifiable, enables compounding improvement as the broader AI ecosystem advances.

  </details>



- **BayesFusion-SDF: Probabilistic Signed Distance Fusion with View Planning on CPU**  
  Soumya Mazumdar, Vineet Kumar Rakesh, Tapas Samanta  
  _2026-02-23_ · https://arxiv.org/abs/2602.19697v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Key part of robotics, augmented reality, and digital inspection is dense 3D reconstruction from depth observations. Traditional volumetric fusion techniques, including truncated signed distance functions (TSDF), enable efficient and deterministic geometry reconstruction; however, they depend on heuristic weighting and fail to transparently convey uncertainty in a systematic way. Recent neural implicit methods, on the other hand, get very high fidelity but usually need a lot of GPU power for optimization and aren't very easy to understand for making decisions later on. This work presents BayesFusion-SDF, a CPU-centric probabilistic signed distance fusion framework that conceptualizes geometry as a sparse Gaussian random field with a defined posterior distribution over voxel distances. First, a rough TSDF reconstruction is used to create an adaptive narrow-band domain. Then, depth observations are combined using a heteroscedastic Bayesian formulation that is solved using sparse linear algebra and preconditioned conjugate gradients. Randomized diagonal estimators are a quick way to get an idea of posterior uncertainty. This makes it possible to extract surfaces and plan the next best view while taking into account uncertainty. Tests on a controlled ablation scene and a CO3D object sequence show that the new method is more accurate geometrically than TSDF baselines and gives useful estimates of uncertainty for active sensing. The proposed formulation provides a clear and easy-to-use alternative to GPU-heavy neural reconstruction methods while still being able to be understood in a probabilistic way and acting in a predictable way. GitHub: https://mazumdarsoumya.github.io/BayesFusionSDF

  </details>



- **Simulation-Ready Cluttered Scene Estimation via Physics-aware Joint Shape and Pose Optimization**  
  Wei-Cheng Huang, Jiaheng Han, Xiaohan Ye, Zherong Pan, Kris Hauser  
  _2026-02-23_ · https://arxiv.org/abs/2602.20150v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Estimating simulation-ready scenes from real-world observations is crucial for downstream planning and policy learning tasks. Regretfully, existing methods struggle in cluttered environments, often exhibiting prohibitive computational cost, poor robustness, and restricted generality when scaling to multiple interacting objects. We propose a unified optimization-based formulation for real-to-sim scene estimation that jointly recovers the shapes and poses of multiple rigid objects under physical constraints. Our method is built on two key technical innovations. First, we leverage the recently introduced shape-differentiable contact model, whose global differentiability permits joint optimization over object geometry and pose while modeling inter-object contacts. Second, we exploit the structured sparsity of the augmented Lagrangian Hessian to derive an efficient linear system solver whose computational cost scales favorably with scene complexity. Building on this formulation, we develop an end-to-end real-to-sim scene estimation pipeline that integrates learning-based object initialization, physics-constrained joint shape-pose optimization, and differentiable texture refinement. Experiments on cluttered scenes with up to 5 objects and 22 convex hulls demonstrate that our approach robustly reconstructs physically valid, simulation-ready object shapes and poses.

  </details>



- **Modeling Epidemiological Dynamics Under Adversarial Data and User Deception**  
  Yiqi Su, Christo Kurisummoottil Thomas, Walid Saad, Bud Mishra, Naren Ramakrishnan  
  _2026-02-23_ · https://arxiv.org/abs/2602.20134v1 · `cs.GT`  
  <details><summary>Abstract</summary>

  Epidemiological models increasingly rely on self-reported behavioral data such as vaccination status, mask usage, and social distancing adherence to forecast disease transmission and assess the impact of non-pharmaceutical interventions (NPIs). While such data provide valuable real-time insights, they are often subject to strategic misreporting, driven by individual incentives to avoid penalties, access benefits, or express distrust in public health authorities. To account for such human behavior, in this paper, we introduce a game-theoretic framework that models the interaction between the population and a public health authority as a signaling game. Individuals (senders) choose how to report their behaviors, while the public health authority (receiver) updates their epidemiological model(s) based on potentially distorted signals. Focusing on deception around masking and vaccination, we characterize analytically game equilibrium outcomes and evaluate the degree to which deception can be tolerated while maintaining epidemic control through policy interventions. Our results show that separating equilibria-with minimal deception-drive infections to near zero over time. Remarkably, even under pervasive dishonesty in pooling equilibria, well-designed sender and receiver strategies can still maintain effective epidemic control. This work advances the understanding of adversarial data in epidemiology and offers tools for designing more robust public health models in the presence of strategic user behavior.

  </details>



- **Robust Taylor-Lagrange Control for Safety-Critical Systems**  
  Wei Xiao, Christos Cassandras, Anni Li  
  _2026-02-23_ · https://arxiv.org/abs/2602.20076v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Solving safety-critical control problem has widely adopted the Control Barrier Function (CBF) method. However, the existence of a CBF is only a sufficient condition for system safety. The recently proposed Taylor-Lagrange Control (TLC) method addresses this limitation, but is vulnerable to the feasibility preservation problem (e.g., inter-sampling effect). In this paper, we propose a robust TLC (rTLC) method to address the feasibility preservation problem. Specifically, the rTLC method expands the safety function at an order higher than the relative degree of the function using Taylor's expansion with Lagrange remainder, which allows the control to explicitly show up at the current time instead of the future time in the TLC method. The rTLC method naturally addresses the feasibility preservation problem with only one hyper-parameter (the discretization time interval size during implementation), which is much less than its counterparts. Finally, we illustrate the effectiveness of the proposed rTLC method through an adaptive cruise control problem, and compare it with existing safety-critical control methods.

  </details>



- **AdaWorldPolicy: World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation**  
  Ge Yuan, Qiyuan Qiao, Jing Zhang, Dong Xu  
  _2026-02-23_ · https://arxiv.org/abs/2602.20057v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. In this work, we introduce a unified framework, World-Model-Driven Diffusion Policy with Online Adaptive Learning (AdaWorldPolicy) to enhance robotic manipulation under dynamic conditions with minimal human involvement. Our core insight is that world models provide strong supervision signals, enabling online adaptive learning in dynamic environments, which can be complemented by force-torque feedback to mitigate dynamic force shifts. Our AdaWorldPolicy integrates a world model, an action expert, and a force predictor-all implemented as interconnected Flow Matching Diffusion Transformers (DiT). They are interconnected via the multi-modal self-attention layers, enabling deep feature exchange for joint learning while preserving their distinct modularity characteristics. We further propose a novel Online Adaptive Learning (AdaOL) strategy that dynamically switches between an Action Generation mode and a Future Imagination mode to drive reactive updates across all three modules. This creates a powerful closed-loop mechanism that adapts to both visual and physical domain shifts with minimal overhead. Across a suite of simulated and real-robot benchmarks, our AdaWorldPolicy achieves state-of-the-art performance, with dynamical adaptive capacity to out-of-distribution scenarios.

  </details>



- **Agents of Chaos**  
  Natalie Shapira, Chris Wendler, Avery Yen, Gabriele Sarti, Koyena Pal, Olivia Floody, Adam Belfki, Alex Loftus, Aditya Ratan Jannali, Nikhil Prakash, et al.  
  _2026-02-23_ · https://arxiv.org/abs/2602.20021v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We report an exploratory red-teaming study of autonomous language-model-powered agents deployed in a live laboratory environment with persistent memory, email accounts, Discord access, file systems, and shell execution. Over a two-week period, twenty AI researchers interacted with the agents under benign and adversarial conditions. Focusing on failures emerging from the integration of language models with autonomy, tool use, and multi-party communication, we document eleven representative case studies. Observed behaviors include unauthorized compliance with non-owners, disclosure of sensitive information, execution of destructive system-level actions, denial-of-service conditions, uncontrolled resource consumption, identity spoofing vulnerabilities, cross-agent propagation of unsafe practices, and partial system takeover. In several cases, agents reported task completion while the underlying system state contradicted those reports. We also report on some of the failed attempts. Our findings establish the existence of security-, privacy-, and governance-relevant vulnerabilities in realistic deployment settings. These behaviors raise unresolved questions regarding accountability, delegated authority, and responsibility for downstream harms, and warrant urgent attention from legal scholars, policymakers, and researchers across disciplines. This report serves as an initial empirical contribution to that broader conversation.

  </details>



- **A Secure and Private Distributed Bayesian Federated Learning Design**  
  Nuocheng Yang, Sihua Wang, Zhaohui Yang, Mingzhe Chen, Changchuan Yin, Kaibin Huang  
  _2026-02-23_ · https://arxiv.org/abs/2602.20003v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Distributed Federated Learning (DFL) enables decentralized model training across large-scale systems without a central parameter server. However, DFL faces three critical challenges: privacy leakage from honest-but-curious neighbors, slow convergence due to the lack of central coordination, and vulnerability to Byzantine adversaries aiming to degrade model accuracy. To address these issues, we propose a novel DFL framework that integrates Byzantine robustness, privacy preservation, and convergence acceleration. Within this framework, each device trains a local model using a Bayesian approach and independently selects an optimal subset of neighbors for posterior exchange. We formulate this neighbor selection as an optimization problem to minimize the global loss function under security and privacy constraints. Solving this problem is challenging because devices only possess partial network information, and the complex coupling between topology, security, and convergence remains unclear. To bridge this gap, we first analytically characterize the trade-offs between dynamic connectivity, Byzantine detection, privacy levels, and convergence speed. Leveraging these insights, we develop a fully distributed Graph Neural Network (GNN)-based Reinforcement Learning (RL) algorithm. This approach enables devices to make autonomous connection decisions based on local observations. Simulation results demonstrate that our method achieves superior robustness and efficiency with significantly lower overhead compared to traditional security and privacy schemes.

  </details>



- **Multivariate time-series forecasting of ASTRI-Horn monitoring data: A Normal Behavior Model**  
  Federico Incardona, Alessandro Costa, Farida Farsian, Francesco Franchina, Giuseppe Leto, Emilio Mastriani, Kevin Munari, Giovanni Pareschi, Salvatore Scuderi, Sebastiano Spinello, et al.  
  _2026-02-23_ · https://arxiv.org/abs/2602.19984v1 · `astro-ph.IM`  
  <details><summary>Abstract</summary>

  This study presents a Normal Behavior Model (NBM) developed to forecast monitoring time-series data from the ASTRI-Horn Cherenkov telescope under normal operating conditions. The analysis focused on 15 physical variables acquired by the Telescope Control Unit between September 2022 and July 2024, representing sensor measurements from the Azimuth and Elevation motors. After data cleaning, resampling, feature selection, and correlation analysis, the dataset was segmented into fixed-length intervals, in which the first I samples represented the input sequence provided to the model, while the forecast length, T, indicated the number of future time steps to be predicted. A sliding-window technique was then applied to increase the number of intervals. A Multi-Layer Perceptron (MLP) was trained to perform multivariate forecasting across all features simultaneously. Model performance was evaluated using the Mean Squared Error (MSE) and the Normalized Median Absolute Deviation (NMAD), and it was also benchmarked against a Long Short-Term Memory (LSTM) network. The MLP model demonstrated consistent results across different features and I-T configurations, and matched the performance of the LSTM while converging faster. It achieved an MSE of 0.019+/-0.003 and an NMAD of 0.032+/-0.009 on the test set under its best configuration (4 hidden layers, 720 units per layer, and I-T lengths of 300 samples each, corresponding to 5 hours at 1-minute resolution). Extending the forecast horizon up to 6.5 hours-the maximum allowed by this configuration-did not degrade performance, confirming the model's effectiveness in providing reliable hour-scale predictions. The proposed NBM provides a powerful tool for enabling early anomaly detection in online ASTRI-Horn monitoring time series, offering a basis for the future development of a prognostics and health management system that supports predictive maintenance.

  </details>



- **Active IoT User Detection in Near-Field with Location Information**  
  Gabriel Martins de Jesus, Richard Demo Souza, Onel Luis Alcaraz López  
  _2026-02-23_ · https://arxiv.org/abs/2602.19613v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  In this paper, we address active users detection (AUD) in near-field Internet of Things (IoT) networks by exploring prior knowledge of users' locations. We consider a scenario where users are distributed in a semi-circular area within the Rayleigh distance of a multi-antenna base station (BS). We propose the BS to use location estimates of the users to reconstruct their line-of-sight (LoS) channel components, hence assisting the AUD process. For this, the BS combines these reconstructed channels with users' pilot sequences, enhancing the correlation between received signals and active users. We formulate the location-aided AUD as a convex optimization problem, solved via the alternating direction method of multipliers (ADMM). {Our proposal has a higher computational complexity compared to the baseline ADMM approach where location information is not used. Moreover, the proposal requires location information of users, which can be readily informed if users are static, or inferred via established localization algorithms if they are mobile.} Simulation results compare our proposal against the baseline across varying systems parameters, such as number of users, pilot length and LoS component strength. We demonstrate that under perfect location estimation and strong LoS, our proposed method significantly outperforms the baseline. Furthermore, robustness analysis shows that performance gains persist under imperfect location estimation, provided the estimation error remains within bounds determined by the system parameters.

  </details>



- **RAID: Retrieval-Augmented Anomaly Detection**  
  Mingxiu Cai, Zhe Zhang, Gaochang Wu, Tianyou Chai, Xiatian Zhu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19611v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Unsupervised Anomaly Detection (UAD) aims to identify abnormal regions by establishing correspondences between test images and normal templates. Existing methods primarily rely on image reconstruction or template retrieval but face a fundamental challenge: matching between test images and normal templates inevitably introduces noise due to intra-class variations, imperfect correspondences, and limited templates. Observing that Retrieval-Augmented Generation (RAG) leverages retrieved samples directly in the generation process, we reinterpret UAD through this lens and introduce \textbf{RAID}, a retrieval-augmented UAD framework designed for noise-resilient anomaly detection and localization. Unlike standard RAG that enriches context or knowledge, we focus on using retrieved normal samples to guide noise suppression in anomaly map generation. RAID retrieves class-, semantic-, and instance-level representations from a hierarchical vector database, forming a coarse-to-fine pipeline. A matching cost volume correlates the input with retrieved exemplars, followed by a guided Mixture-of-Experts (MoE) network that leverages the retrieved samples to adaptively suppress matching noise and produce fine-grained anomaly maps. RAID achieves state-of-the-art performance across full-shot, few-shot, and multi-dataset settings on MVTec, VisA, MPDD, and BTAD benchmarks. \href{https://github.com/Mingxiu-Cai/RAID}{https://github.com/Mingxiu-Cai/RAID}.

  </details>



- **Advantage-based Temporal Attack in Reinforcement Learning**  
  Shenghong He  
  _2026-02-23_ · https://arxiv.org/abs/2602.19582v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Extensive research demonstrates that Deep Reinforcement Learning (DRL) models are susceptible to adversarially constructed inputs (i.e., adversarial examples), which can mislead the agent to take suboptimal or unsafe actions. Recent methods improve attack effectiveness by leveraging future rewards to guide adversarial perturbation generation over sequential time steps (i.e., reward-based attacks). However, these methods are unable to capture dependencies between different time steps in the perturbation generation process, resulting in a weak temporal correlation between the current perturbation and previous perturbations.In this paper, we propose a novel method called Advantage-based Adversarial Transformer (AAT), which can generate adversarial examples with stronger temporal correlations (i.e., time-correlated adversarial examples) to improve the attack performance. AAT employs a multi-scale causal self-attention (MSCSA) mechanism to dynamically capture dependencies between historical information from different time periods and the current state, thus enhancing the correlation between the current perturbation and the previous perturbation. Moreover, AAT introduces a weighted advantage mechanism, which quantifies the effectiveness of a perturbation in a given state and guides the generation process toward high-performance adversarial examples by sampling high-advantage regions. Extensive experiments demonstrate that the performance of AAT matches or surpasses mainstream adversarial attack baselines on Atari, DeepMind Control Suite and Google football tasks.

  </details>


