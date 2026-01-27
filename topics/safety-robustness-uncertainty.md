# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-01-27 06:52 UTC_

Total papers shown: **20**


---

- **$α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks**  
  Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah  
  _2026-01-26_ · https://arxiv.org/abs/2601.18754v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Autonomous unmanned aerial vehicle (UAV) systems are increasingly deployed in safety-critical, networked environments where they must operate reliably in the presence of malicious adversaries. While recent benchmarks have evaluated large language model (LLM)-based UAV agents in reasoning, navigation, and efficiency, systematic assessment of security, resilience, and trust under adversarial conditions remains largely unexplored, particularly in emerging 6G-enabled settings. We introduce $α^{3}$-SecBench, the first large-scale evaluation suite for assessing the security-aware autonomy of LLM-based UAV agents under realistic adversarial interference. Building on multi-turn conversational UAV missions from $α^{3}$-Bench, the framework augments benign episodes with 20,000 validated security overlay attack scenarios targeting seven autonomy layers, including sensing, perception, planning, control, communication, edge/cloud infrastructure, and LLM reasoning. $α^{3}$-SecBench evaluates agents across three orthogonal dimensions: security (attack detection and vulnerability attribution), resilience (safe degradation behavior), and trust (policy-compliant tool usage). We evaluate 23 state-of-the-art LLMs from major industrial providers and leading AI labs using thousands of adversarially augmented UAV episodes sampled from a corpus of 113,475 missions spanning 175 threat types. While many models reliably detect anomalous behavior, effective mitigation, vulnerability attribution, and trustworthy control actions remain inconsistent. Normalized overall scores range from 12.9% to 57.1%, highlighting a significant gap between anomaly detection and security-aware autonomous decision-making. We release $α^{3}$-SecBench on GitHub: https://github.com/maferrag/AlphaSecBench

  </details>



- **Constraint-Aware Discrete-Time PID Gain Optimization for Robotic Joint Control Under Actuator Saturation**  
  Ojasva Mishra, Xiaolong Wu, Min Xu  
  _2026-01-26_ · https://arxiv.org/abs/2601.18639v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The precise regulation of rotary actuation is fundamental in autonomous robotics, yet practical PID loops deviate from continuous-time theory due to discrete-time execution, actuator saturation, and small delays and measurement imperfections. We present an implementation-aware analysis and tuning workflow for saturated discrete-time joint control. We (i) derive PI stability regions under Euler and exact zero-order-hold (ZOH) discretizations using the Jury criterion, (ii) evaluate a discrete back-calculation anti-windup realization under saturation-dominant regimes, and (iii) propose a hybrid-certified Bayesian optimization workflow that screens analytically unstable candidates and behaviorally unsafe transients while optimizing a robust IAE objective with soft penalties on overshoot and saturation duty. Baseline sweeps ($τ=1.0$~s, $Δt=0.01$~s, $u\in[-10,10]$) quantify rise/settle trends for P/PI/PID. Under a randomized model family emulating uncertainty, delay, noise, quantization, and tighter saturation, robustness-oriented tuning improves median IAE from $0.843$ to $0.430$ while keeping median overshoot below $2\%$. In simulation-only tuning, the certification screen rejects $11.6\%$ of randomly sampled gains within bounds before full robust evaluation, improving sample efficiency without hardware experiments.

  </details>



- **Co-PLNet: A Collaborative Point-Line Network for Prompt-Guided Wireframe Parsing**  
  Chao Wang, Xuanying Li, Cheng Dai, Jinglei Feng, Yuxiang Luo, Yuqi Ouyang, Hao Qin  
  _2026-01-26_ · https://arxiv.org/abs/2601.18252v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Wireframe parsing aims to recover line segments and their junctions to form a structured geometric representation useful for downstream tasks such as Simultaneous Localization and Mapping (SLAM). Existing methods predict lines and junctions separately and reconcile them post-hoc, causing mismatches and reduced robustness. We present Co-PLNet, a point-line collaborative framework that exchanges spatial cues between the two tasks, where early detections are converted into spatial prompts via a Point-Line Prompt Encoder (PLP-Encoder), which encodes geometric attributes into compact and spatially aligned maps. A Cross-Guidance Line Decoder (CGL-Decoder) then refines predictions with sparse attention conditioned on complementary prompts, enforcing point-line consistency and efficiency. Experiments on Wireframe and YorkUrban show consistent improvements in accuracy and robustness, together with favorable real-time efficiency, demonstrating our effectiveness for structured geometry perception.

  </details>



- **Goal-oriented Communication for Fast and Robust Robotic Fault Detection and Recovery**  
  Shutong Chen, Adnan Aijaz, Yansha Deng  
  _2026-01-26_ · https://arxiv.org/abs/2601.18765v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous robotic systems are widely deployed in smart factories and operate in dynamic, uncertain, and human-involved environments that require low-latency and robust fault detection and recovery (FDR). However, existing FDR frameworks exhibit various limitations, such as significant delays in communication and computation, and unreliability in robot motion/trajectory generation, mainly because the communication-computation-control (3C) loop is designed without considering the downstream FDR goal. To address this, we propose a novel Goal-oriented Communication (GoC) framework that jointly designs the 3C loop tailored for fast and robust robotic FDR, with the goal of minimising the FDR time while maximising the robotic task (e.g., workpiece sorting) success rate. For fault detection, our GoC framework innovatively defines and extracts the 3D scene graph (3D-SG) as the semantic representation via our designed representation extractor, and detects faults by monitoring spatial relationship changes in the 3D-SG. For fault recovery, we fine-tune a small language model (SLM) via Low-Rank Adaptation (LoRA) and enhance its reasoning and generalization capabilities via knowledge distillation to generate recovery motions for robots. We also design a lightweight goal-oriented digital twin reconstruction module to refine the recovery motions generated by the SLM when fine-grained robotic control is required, using only task-relevant object contours for digital twin reconstruction. Extensive simulations demonstrate that our GoC framework reduces the FDR time by up to 82.6% and improves the task success rate by up to 76%, compared to the state-of-the-art frameworks that rely on vision language models for fault detection and large language models for fault recovery.

  </details>



- **TC-IDM: Grounding Video Generation for Executable Zero-shot Robot Motion**  
  Weishi Mi, Yong Bao, Xiaowei Chi, Xiaozhu Ju, Zhiyuan Qin, Kuangzhi Ge, Kai Tang, Peidong Jia, Shanghang Zhang, Jian Tang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18323v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The vision-language-action (VLA) paradigm has enabled powerful robotic control by leveraging vision-language models, but its reliance on large-scale, high-quality robot data limits its generalization. Generative world models offer a promising alternative for general-purpose embodied AI, yet a critical gap remains between their pixel-level plans and physically executable actions. To this end, we propose the Tool-Centric Inverse Dynamics Model (TC-IDM). By focusing on the tool's imagined trajectory as synthesized by the world model, TC-IDM establishes a robust intermediate representation that bridges the gap between visual planning and physical control. TC-IDM extracts the tool's point cloud trajectories via segmentation and 3D motion estimation from generated videos. Considering diverse tool attributes, our architecture employs decoupled action heads to project these planned trajectories into 6-DoF end-effector motions and corresponding control signals. This plan-and-translate paradigm not only supports a wide range of end-effectors but also significantly improves viewpoint invariance. Furthermore, it exhibits strong generalization capabilities across long-horizon and out-of-distribution tasks, including interacting with deformable objects. In real-world evaluations, the world model with TC-IDM achieves an average success rate of 61.11 percent, with 77.7 percent on simple tasks and 38.46 percent on zero-shot deformable object tasks. It substantially outperforms end-to-end VLA-style baselines and other inverse dynamics models.

  </details>



- **Trust, Don't Trust, or Flip: Robust Preference-Based Reinforcement Learning with Multi-Expert Feedback**  
  Seyed Amir Hosseini, Maryam Abdolali, Amirhosein Tavakkoli, Fardin Ayar, Ehsan Javanmardi, Manabu Tsukada, Mahdi Javanmardi  
  _2026-01-26_ · https://arxiv.org/abs/2601.18751v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Preference-based reinforcement learning (PBRL) offers a promising alternative to explicit reward engineering by learning from pairwise trajectory comparisons. However, real-world preference data often comes from heterogeneous annotators with varying reliability; some accurate, some noisy, and some systematically adversarial. Existing PBRL methods either treat all feedback equally or attempt to filter out unreliable sources, but both approaches fail when faced with adversarial annotators who systematically provide incorrect preferences. We introduce TriTrust-PBRL (TTP), a unified framework that jointly learns a shared reward model and expert-specific trust parameters from multi-expert preference feedback. The key insight is that trust parameters naturally evolve during gradient-based optimization to be positive (trust), near zero (ignore), or negative (flip), enabling the model to automatically invert adversarial preferences and recover useful signal rather than merely discarding corrupted feedback. We provide theoretical analysis establishing identifiability guarantees and detailed gradient analysis that explains how expert separation emerges naturally during training without explicit supervision. Empirically, we evaluate TTP on four diverse domains spanning manipulation tasks (MetaWorld) and locomotion (DM Control) under various corruption scenarios. TTP achieves state-of-the-art robustness, maintaining near-oracle performance under adversarial corruption while standard PBRL methods fail catastrophically. Notably, TTP outperforms existing baselines by successfully learning from mixed expert pools containing both reliable and adversarial annotators, all while requiring no expert features beyond identification indices and integrating seamlessly with existing PBRL pipelines.

  </details>



- **Out-of-Distribution Radar Detection with Complex VAEs: Theory, Whitening, and ANMF Fusion**  
  Yadang Alexis Rouzoumka, Jean Pinsolle, Eugénie Terreaux, Christèle Morisseau, Jean-Philippe Ovarlez, Chengfang Ren  
  _2026-01-26_ · https://arxiv.org/abs/2601.18677v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  We investigate the detection of weak complex-valued signals immersed in non-Gaussian, range-varying interference, with emphasis on maritime radar scenarios. The proposed methodology exploits a Complex-valued Variational AutoEncoder (CVAE) trained exclusively on clutter-plus-noise to perform Out-Of-Distribution detection. By operating directly on in-phase / quadrature samples, the CVAE preserves phase and Doppler structure and is assessed in two configurations: (i) using unprocessed range profiles and (ii) after local whitening, where per-range covariance estimates are obtained from neighboring profiles. Using extensive simulations together with real sea-clutter data from the CSIR maritime dataset, we benchmark performance against classical and adaptive detectors (MF, NMF, AMF-SCM, ANMF-SCM, ANMF-Tyler). In both configurations, the CVAE yields a higher detection probability Pd at matched false-alarm rate Pfa, with the most notable improvements observed under whitening. We further integrate the CVAE with the ANMF through a weighted log-p fusion rule at the decision level, attaining enhanced robustness in strongly non-Gaussian clutter and enabling empirically calibrated Pfa control under H0. Overall, the results demonstrate that statistical normalization combined with complex-valued generative modeling substantively improves detection in realistic sea-clutter conditions, and that the fused CVAE-ANMF scheme constitutes a competitive alternative to established model-based detectors.

  </details>



- **AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security**  
  Dongrui Liu, Qihan Ren, Chen Qian, Shuai Shao, Yuejin Xie, Yu Li, Zhonghao Yang, Haoyu Luo, Peng Wang, Qingyu Liu, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18491v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy that orthogonally categorizes agentic risks by their source (where), failure mode (how), and consequence (what). Guided by this structured and hierarchical taxonomy, we introduce a new fine-grained agentic safety benchmark (ATBench) and a Diagnostic Guardrail framework for agent safety and security (AgentDoG). AgentDoG provides fine-grained and contextual monitoring across agent trajectories. More Crucially, AgentDoG can diagnose the root causes of unsafe actions and seemingly safe but unreasonable actions, offering provenance and transparency beyond binary labels to facilitate effective agent alignment. AgentDoG variants are available in three sizes (4B, 7B, and 8B parameters) across Qwen and Llama model families. Extensive experimental results demonstrate that AgentDoG achieves state-of-the-art performance in agentic safety moderation in diverse and complex interactive scenarios. All models and datasets are openly released.

  </details>



- **SG-CADVLM: A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation**  
  Hongyi Zhao, Shuo Wang, Qijie He, Ziyuan Pu  
  _2026-01-26_ · https://arxiv.org/abs/2601.18442v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous vehicle safety validation requires testing on safety-critical scenarios, but these events are rare in real-world driving and costly to test due to collision risks. Crash reports provide authentic specifications of safety-critical events, offering a vital alternative to scarce real-world collision trajectory data. This makes them valuable sources for generating realistic high-risk scenarios through simulation. Existing approaches face significant limitations because data-driven methods lack diversity due to their reliance on existing latent distributions, whereas adversarial methods often produce unrealistic scenarios lacking physical fidelity. Large Language Model (LLM) and Vision Language Model (VLM)-based methods show significant promise. However, they suffer from context suppression issues where internal parametric knowledge overrides crash specifications, producing scenarios that deviate from actual accident characteristics. This paper presents SG-CADVLM (A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation), a framework that integrates Context-Aware Decoding with multi-modal input processing to generate safety-critical scenarios from crash reports and road network diagrams. The framework mitigates VLM hallucination issues while enabling the simultaneous generation of road geometry and vehicle trajectories. The experimental results demonstrate that SG-CADVLM generates critical risk scenarios at a rate of 84.4% compared to 12.5% for the baseline methods, representing an improvement of 469%, while producing executable simulations for autonomous vehicle testing.

  </details>



- **Cognitive Fusion of ZC Sequences and Time-Frequency Images for Out-of-Distribution Detection of Drone Signals**  
  Jie Li, Jing Li, Lu Lv, Zhanyu Ju, Fengkui Gong  
  _2026-01-26_ · https://arxiv.org/abs/2601.18326v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We propose a drone signal out-of-distribution detection (OODD) algorithm based on the cognitive fusion of Zadoff-Chu (ZC) sequences and time-frequency images (TFI). ZC sequences are identified by analyzing the communication protocols of DJI drones, while TFI capture the time-frequency characteristics of drone signals with unknown or non-standard communication protocols. Both modalities are used jointly to enable OODD in the drone remote identification (RID) task. Specifically, ZC sequence features and TFI features are generated from the received radio frequency signals, which are then processed through dedicated feature extraction module to enhance and align them. The resultant multi-modal features undergo multi-modal feature interaction, single-modal feature fusion, and multi-modal feature fusion to produce features that integrate and complement information across modalities. Discrimination scores are computed from the fused features along both spatial and channel dimensions to capture time-frequency characteristic differences dictated by the communication protocols, and these scores will be transformed into adaptive attention weights. The weighted features are then passed through a Softmax function to produce the signal classification results. Simulation results demonstrate that the proposed algorithm outperforms existing algorithms and achieves 1.7% and 7.5% improvements in RID and OODD metrics, respectively. The proposed algorithm also performs strong robustness under varying flight conditions and across different drone types.

  </details>



- **What Do Learned Models Measure?**  
  Indrė Žliobaitė  
  _2026-01-26_ · https://arxiv.org/abs/2601.18278v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  In many scientific and data-driven applications, machine learning models are increasingly used as measurement instruments, rather than merely as predictors of predefined labels. When the measurement function is learned from data, the mapping from observations to quantities is determined implicitly by the training distribution and inductive biases, allowing multiple inequivalent mappings to satisfy standard predictive evaluation criteria. We formalize learned measurement functions as a distinct focus of evaluation and introduce measurement stability, a property capturing invariance of the measured quantity across admissible realizations of the learning process and across contexts. We show that standard evaluation criteria in machine learning, including generalization error, calibration, and robustness, do not guarantee measurement stability. Through a real-world case study, we show that models with comparable predictive performance can implement systematically inequivalent measurement functions, with distribution shift providing a concrete illustration of this failure. Taken together, our results highlight a limitation of existing evaluation frameworks in settings where learned model outputs are identified as measurements, motivating the need for an additional evaluative dimension.

  </details>



- **Quest2ROS2: A ROS 2 Framework for Bi-manual VR Teleoperation**  
  Jialong Li, Zhenguo Wang, Tianci Wang, Maj Stenmark, Volker Krueger  
  _2026-01-26_ · https://arxiv.org/abs/2601.18289v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Quest2ROS2 is an open-source ROS2 framework for bi-manual teleoperation designed to scale robot data collection. Extending Quest2ROS, it overcomes workspace limitations via relative motion-based control, calculating robot movement from VR controller pose changes to enable intuitive, pose-independent operation. The framework integrates essential usability and safety features, including real-time RViz visualization, streamlined gripper control, and a pause-and-reset function for smooth transitions. We detail a modular architecture that supports "Side-by-Side" and "Mirror" control modes to optimize operator experience across diverse platforms. Code is available at: https://github.com/Taokt/Quest2ROS2.

  </details>



- **Multi-Objective Reinforcement Learning for Efficient Tactical Decision Making for Trucks in Highway Traffic**  
  Deepthi Pathare, Leo Laine, Morteza Haghir Chehreghani  
  _2026-01-26_ · https://arxiv.org/abs/2601.18783v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Balancing safety, efficiency, and operational costs in highway driving poses a challenging decision-making problem for heavy-duty vehicles. A central difficulty is that conventional scalar reward formulations, obtained by aggregating these competing objectives, often obscure the structure of their trade-offs. We present a Proximal Policy Optimization based multi-objective reinforcement learning framework that learns a continuous set of policies explicitly representing these trade-offs and evaluates it on a scalable simulation platform for tactical decision making in trucks. The proposed approach learns a continuous set of Pareto-optimal policies that capture the trade-offs among three conflicting objectives: safety, quantified in terms of collisions and successful completion; energy efficiency and time efficiency, quantified using energy cost and driver cost, respectively. The resulting Pareto frontier is smooth and interpretable, enabling flexibility in choosing driving behavior along different conflicting objectives. This framework allows seamless transitions between different driving policies without retraining, yielding a robust and adaptive decision-making strategy for autonomous trucking applications.

  </details>



- **Counterfactual Explanations on Robust Perceptual Geodesics**  
  Eslam Zaher, Maciej Trzaskowski, Quan Nguyen, Fred Roosta  
  _2026-01-26_ · https://arxiv.org/abs/2601.18678v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Latent-space optimization methods for counterfactual explanations - framed as minimal semantic perturbations that change model predictions - inherit the ambiguity of Wachter et al.'s objective: the choice of distance metric dictates whether perturbations are meaningful or adversarial. Existing approaches adopt flat or misaligned geometries, leading to off-manifold artifacts, semantic drift, or adversarial collapse. We introduce Perceptual Counterfactual Geodesics (PCG), a method that constructs counterfactuals by tracing geodesics under a perceptually Riemannian metric induced from robust vision features. This geometry aligns with human perception and penalizes brittle directions, enabling smooth, on-manifold, semantically valid transitions. Experiments on three vision datasets show that PCG outperforms baselines and reveals failure modes hidden under standard metrics.

  </details>



- **Physics-Informed Uncertainty Enables Reliable AI-driven Design**  
  Tingkai Xue, Chin Chun Ooi, Yang Jiang, Luu Trung Pham Duong, Pao-Hsiung Chiu, Weijiang Zhao, Nagarajan Raghavan, My Ha Dao  
  _2026-01-26_ · https://arxiv.org/abs/2601.18638v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Inverse design is a central goal in much of science and engineering, including frequency-selective surfaces (FSS) that are critical to microelectronics for telecommunications and optical metamaterials. Traditional surrogate-assisted optimization methods using deep learning can accelerate the design process but do not usually incorporate uncertainty quantification, leading to poorer optimization performance due to erroneous predictions in data-sparse regions. Here, we introduce and validate a fundamentally different paradigm of Physics-Informed Uncertainty, where the degree to which a model's prediction violates fundamental physical laws serves as a computationally-cheap and effective proxy for predictive uncertainty. By integrating physics-informed uncertainty into a multi-fidelity uncertainty-aware optimization workflow to design complex frequency-selective surfaces within the 20 - 30 GHz range, we increase the success rate of finding performant solutions from less than 10% to over 50%, while simultaneously reducing computational cost by an order of magnitude compared to the sole use of a high-fidelity solver. These results highlight the necessity of incorporating uncertainty quantification in machine-learning-driven inverse design for high-dimensional problems, and establish physics-informed uncertainty as a viable alternative to quantifying uncertainty in surrogate models for physical systems, thereby setting the stage for autonomous scientific discovery systems that can efficiently and robustly explore and evaluate candidate designs.

  </details>



- **ExoGS: A 4D Real-to-Sim-to-Real Framework for Scalable Manipulation Data Collection**  
  Yiming Wang, Ruogu Zhang, Minyang Li, Hao Shi, Junbo Wang, Deyi Li, Jieji Ren, Wenhai Liu, Weiming Wang, Hao-Shu Fang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18629v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Real-to-Sim-to-Real technique is gaining increasing interest for robotic manipulation, as it can generate scalable data in simulation while having narrower sim-to-real gap. However, previous methods mainly focused on environment-level visual real-to-sim transfer, ignoring the transfer of interactions, which could be challenging and inefficient to obtain purely in simulation especially for contact-rich tasks. We propose ExoGS, a robot-free 4D Real-to-Sim-to-Real framework that captures both static environments and dynamic interactions in the real world and transfers them seamlessly to a simulated environment. It provides a new solution for scalable manipulation data collection and policy learning. ExoGS employs a self-designed robot-isomorphic passive exoskeleton AirExo-3 to capture kinematically consistent trajectories with millimeter-level accuracy and synchronized RGB observations during direct human demonstrations. The robot, objects, and environment are reconstructed as editable 3D Gaussian Splatting assets, enabling geometry-consistent replay and large-scale data augmentation. Additionally, a lightweight Mask Adapter injects instance-level semantics into the policy to enhance robustness under visual domain shifts. Real-world experiments demonstrate that ExoGS significantly improves data efficiency and policy generalization compared to teleoperation-based baselines. Code and hardware files have been released on https://github.com/zaixiabalala/ExoGS.

  </details>



- **Fast and Safe Trajectory Optimization for Mobile Manipulators With Neural Configuration Space Distance Field**  
  Yulin Li, Zhiyuan Song, Yiming Li, Zhicheng Song, Kai Chen, Chunxin Zheng, Zhihai Bi, Jiahang Cao, Sylvain Calinon, Fan Shi, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18548v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mobile manipulators promise agile, long-horizon behavior by coordinating base and arm motion, yet whole-body trajectory optimization in cluttered, confined spaces remains difficult due to high-dimensional nonconvexity and the need for fast, accurate collision reasoning. Configuration Space Distance Fields (CDF) enable fixed-base manipulators to model collisions directly in configuration space via smooth, implicit distances. This representation holds strong potential to bypass the nonlinear configuration-to-workspace mapping while preserving accurate whole-body geometry and providing optimization-friendly collision costs. Yet, extending this capability to mobile manipulators is hindered by unbounded workspaces and tighter base-arm coupling. We lift this promise to mobile manipulation with Generalized Configuration Space Distance Fields (GCDF), extending CDF to robots with both translational and rotational joints in unbounded workspaces with tighter base-arm coupling. We prove that GCDF preserves Euclidean-like local distance structure and accurately encodes whole-body geometry in configuration space, and develop a data generation and training pipeline that yields continuous neural GCDFs with accurate values and gradients, supporting efficient GPU-batched queries. Building on this representation, we develop a high-performance sequential convex optimization framework centered on GCDF-based collision reasoning. The solver scales to large numbers of implicit constraints through (i) online specification of neural constraints, (ii) sparsity-aware active-set detection with parallel batched evaluation across thousands of constraints, and (iii) incremental constraint management for rapid replanning under scene changes.

  </details>



- **SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction**  
  Linyong Gan, Zimo Li, Wenxin Xu, Xingjian Li, Jianhua Z. Huang, Enmei Tu, Shuhang Chen  
  _2026-01-26_ · https://arxiv.org/abs/2601.18537v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate long-horizon vessel trajectory prediction remains challenging due to compounded uncertainty from complex navigation behaviors and environmental factors. Existing methods often struggle to maintain global directional consistency, leading to drifting or implausible trajectories when extrapolated over long time horizons. To address this issue, we propose a semantic-key-point-conditioned trajectory modeling framework, in which future trajectories are predicted by conditioning on a high-level Next Key Point (NKP) that captures navigational intent. This formulation decomposes long-horizon prediction into global semantic decision-making and local motion modeling, effectively restricting the support of future trajectories to semantically feasible subsets. To efficiently estimate the NKP prior from historical observations, we adopt a pretrain-finetune strategy. Extensive experiments on real-world AIS data demonstrate that the proposed method consistently outperforms state-of-the-art approaches, particularly for long travel durations, directional accuracy, and fine-grained trajectory prediction.

  </details>



- **LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models**  
  Kai Hu, Haoqi Hu, Matt Fredrikson  
  _2026-01-26_ · https://arxiv.org/abs/2601.18513v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Lipschitz-based certification offers efficient, deterministic robustness guarantees but has struggled to scale in model size, training efficiency, and ImageNet performance. We introduce \emph{LipNeXt}, the first \emph{constraint-free} and \emph{convolution-free} 1-Lipschitz architecture for certified robustness. LipNeXt is built using two techniques: (1) a manifold optimization procedure that updates parameters directly on the orthogonal manifold and (2) a \emph{Spatial Shift Module} to model spatial pattern without convolutions. The full network uses orthogonal projections, spatial shifts, a simple 1-Lipschitz $β$-Abs nonlinearity, and $L_2$ spatial pooling to maintain tight Lipschitz control while enabling expressive feature mixing. Across CIFAR-10/100 and Tiny-ImageNet, LipNeXt achieves state-of-the-art clean and certified robust accuracy (CRA), and on ImageNet it scales to 1-2B large models, improving CRA over prior Lipschitz models (e.g., up to $+8\%$ at $\varepsilon{=}1$) while retaining efficient, stable low-precision training. These results demonstrate that Lipschitz-based certification can benefit from modern scaling trends without sacrificing determinism or efficiency.

  </details>



- **Dualband OFDM Delay Estimation for Multi-Target Localization**  
  Jialun Kou, Achiel Colpaert, Zhuangzhuang Cui, Sofie Pollin  
  _2026-01-26_ · https://arxiv.org/abs/2601.18478v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Integrated localization and communication systems aim to reuse communication waveforms for simultaneous data transmission and localization, but delay resolution is fundamentally limited by the available bandwidth. In practice, large contiguous bandwidths are difficult to obtain due to hardware constraints and spectrum fragmentation. Aggregating non-contiguous narrow bands can increase the effective frequency span, but a non-contiguous frequency layout introduces challenges such as elevated sidelobes and ambiguity in delay estimation. This paper introduces a point-spread-function (PSF)-centric framework for dual-band OFDM delay estimation. We model the observed delay profile as the convolution of the true target response with a PSF determined by the dual-band subcarrier selection pattern, explicitly linking band configuration to resolution and ambiguity. To suppress PSF-induced artifacts, we adapt the RELAX algorithm for dual-band multi-target delay estimation. Simulations demonstrate improved robustness and accuracy in dual-band scenarios, supporting ILC under fragmented spectrum.

  </details>


