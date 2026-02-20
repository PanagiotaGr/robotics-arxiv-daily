# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-20 07:10 UTC_

Total papers shown: **20**


---

- **Conditional Flow Matching for Continuous Anomaly Detection in Autonomous Driving on a Manifold-Aware Spectral Space**  
  Antonio Guillen-Perez  
  _2026-02-19_ · https://arxiv.org/abs/2602.17586v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Safety validation for Level 4 autonomous vehicles (AVs) is currently bottlenecked by the inability to scale the detection of rare, high-risk long-tail scenarios using traditional rule-based heuristics. We present Deep-Flow, an unsupervised framework for safety-critical anomaly detection that utilizes Optimal Transport Conditional Flow Matching (OT-CFM) to characterize the continuous probability density of expert human driving behavior. Unlike standard generative approaches that operate in unstable, high-dimensional coordinate spaces, Deep-Flow constrains the generative process to a low-rank spectral manifold via a Principal Component Analysis (PCA) bottleneck. This ensures kinematic smoothness by design and enables the computation of the exact Jacobian trace for numerically stable, deterministic log-likelihood estimation. To resolve multi-modal ambiguity at complex junctions, we utilize an Early Fusion Transformer encoder with lane-aware goal conditioning, featuring a direct skip-connection to the flow head to maintain intent-integrity throughout the network. We introduce a kinematic complexity weighting scheme that prioritizes high-energy maneuvers (quantified via path tortuosity and jerk) during the simulation-free training process. Evaluated on the Waymo Open Motion Dataset (WOMD), our framework achieves an AUC-ROC of 0.766 against a heuristic golden set of safety-critical events. More significantly, our analysis reveals a fundamental distinction between kinematic danger and semantic non-compliance. Deep-Flow identifies a critical predictability gap by surfacing out-of-distribution behaviors, such as lane-boundary violations and non-normative junction maneuvers, that traditional safety filters overlook. This work provides a mathematically rigorous foundation for defining statistical safety gates, enabling objective, data-driven validation for the safe deployment of autonomous fleets.

  </details>



- **Multi-session Localization and Mapping Exploiting Topological Information**  
  Lorenzo Montano-Olivan, Julio A. Placed, Luis Montano, Maria T. Lazaro  
  _2026-02-19_ · https://arxiv.org/abs/2602.17226v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Operating in previously visited environments is becoming increasingly crucial for autonomous systems, with direct applications in autonomous driving, surveying, and warehouse or household robotics. This repeated exposure to observing the same areas poses significant challenges for mapping and localization -- key components for enabling any higher-level task. In this work, we propose a novel multi-session framework that builds on map-based localization, in contrast to the common practice of greedily running full SLAM sessions and trying to find correspondences between the resulting maps. Our approach incorporates a topology-informed, uncertainty-aware decision-making mechanism that analyzes the pose-graph structure to detect low-connectivity regions, selectively triggering mapping and loop closing modules. The resulting map and pose-graph are seamlessly integrated into the existing model, reducing accumulated error and enhancing global consistency. We validate our method on overlapping sequences from datasets and demonstrate its effectiveness in a real-world mine-like environment.

  </details>



- **What Breaks Embodied AI Security:LLM Vulnerabilities, CPS Flaws,or Something Else?**  
  Boyang Ma, Hechuan Guo, Peizhuo Lv, Minghui Xu, Xuelong Dai, YeChao Zhang, Yijun Yang, Yue Zhang  
  _2026-02-19_ · https://arxiv.org/abs/2602.17345v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Embodied AI systems (e.g., autonomous vehicles, service robots, and LLM-driven interactive agents) are rapidly transitioning from controlled environments to safety critical real-world deployments. Unlike disembodied AI, failures in embodied intelligence lead to irreversible physical consequences, raising fundamental questions about security, safety, and reliability. While existing research predominantly analyzes embodied AI through the lenses of Large Language Model (LLM) vulnerabilities or classical Cyber-Physical System (CPS) failures, this survey argues that these perspectives are individually insufficient to explain many observed breakdowns in modern embodied systems. We posit that a significant class of failures arises from embodiment-induced system-level mismatches, rather than from isolated model flaws or traditional CPS attacks. Specifically, we identify four core insights that explain why embodied AI is fundamentally harder to secure: (i) semantic correctness does not imply physical safety, as language-level reasoning abstracts away geometry, dynamics, and contact constraints; (ii) identical actions can lead to drastically different outcomes across physical states due to nonlinear dynamics and state uncertainty; (iii) small errors propagate and amplify across tightly coupled perception-decision-action loops; and (iv) safety is not compositional across time or system layers, enabling locally safe decisions to accumulate into globally unsafe behavior. These insights suggest that securing embodied AI requires moving beyond component-level defenses toward system-level reasoning about physical risk, uncertainty, and failure propagation.

  </details>



- **RA-Nav: A Risk-Aware Navigation System Based on Semantic Segmentation for Aerial Robots in Unpredictable Environments**  
  Ziyi Zong, Xin Dong, Jinwu Xiang, Daochun Li, Zhan Tu  
  _2026-02-19_ · https://arxiv.org/abs/2602.17515v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Existing aerial robot navigation systems typically plan paths around static and dynamic obstacles, but fail to adapt when a static obstacle suddenly moves. Integrating environmental semantic awareness enables estimation of potential risks posed by suddenly moving obstacles. In this paper, we propose RA- Nav, a risk-aware navigation framework based on semantic segmentation. A lightweight multi-scale semantic segmentation network identifies obstacle categories in real time. These obstacles are further classified into three types: stationary, temporarily static, and dynamic. For each type, corresponding risk estimation functions are designed to enable real-time risk prediction, based on which a complete local risk map is constructed. Based on this map, the risk-informed path search algorithm is designed to guarantee planning that balances path efficiency and safety. Trajectory optimization is then applied to generate trajectories that are safe, smooth, and dynamically feasible. Comparative simulations demonstrate that RA-Nav achieves higher success rates than baselines in sudden obstacle state transition scenarios. Its effectiveness is further validated in simulations using real- world data.

  </details>



- **Distributed Virtual Model Control for Scalable Human-Robot Collaboration in Shared Workspace**  
  Yi Zhang, Omar Faris, Chapa Sirithunge, Kai-Fung Chu, Fumiya Iida, Fulvio Forni  
  _2026-02-19_ · https://arxiv.org/abs/2602.17415v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a decentralized, agent agnostic, and safety-aware control framework for human-robot collaboration based on Virtual Model Control (VMC). In our approach, both humans and robots are embedded in the same virtual-component-shaped workspace, where motion is the result of the interaction with virtual springs and dampers rather than explicit trajectory planning. A decentralized, force-based stall detector identifies deadlocks, which are resolved through negotiation. This reduces the probability of robots getting stuck in the block placement task from up to 61.2% to zero in our experiments. The framework scales without structural changes thanks to the distributed implementation: in experiments we demonstrate safe collaboration with up to two robots and two humans, and in simulation up to four robots, maintaining inter-agent separation at around 20 cm. Results show that the method shapes robot behavior intuitively by adjusting control parameters and achieves deadlock-free operation across team sizes in all tested scenarios.

  </details>



- **Attachment Anchors: A Novel Framework for Laparoscopic Grasping Point Prediction in Colorectal Surgery**  
  Dennis N. Schneider, Lars Wagner, Daniel Rueckert, Dirk Wilhelm  
  _2026-02-19_ · https://arxiv.org/abs/2602.17310v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate grasping point prediction is a key challenge for autonomous tissue manipulation in minimally invasive surgery, particularly in complex and variable procedures such as colorectal interventions. Due to their complexity and prolonged duration, colorectal procedures have been underrepresented in current research. At the same time, they pose a particularly interesting learning environment due to repetitive tissue manipulation, making them a promising entry point for autonomous, machine learning-driven support. Therefore, in this work, we introduce attachment anchors, a structured representation that encodes the local geometric and mechanical relationships between tissue and its anatomical attachments in colorectal surgery. This representation reduces uncertainty in grasping point prediction by normalizing surgical scenes into a consistent local reference frame. We demonstrate that attachment anchors can be predicted from laparoscopic images and incorporated into a grasping framework based on machine learning. Experiments on a dataset of 90 colorectal surgeries demonstrate that attachment anchors improve grasping point prediction compared to image-only baselines. There are particularly strong gains in out-of-distribution settings, including unseen procedures and operating surgeons. These results suggest that attachment anchors are an effective intermediate representation for learning-based tissue manipulation in colorectal surgery.

  </details>



- **When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs**  
  Yu Fang, Yuchun Feng, Dong Jing, Jiaqi Liu, Yue Yang, Zhenyu Wei, Daniel Szafir, Mingyu Ding  
  _2026-02-19_ · https://arxiv.org/abs/2602.17659v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language-Action models (VLAs) promise to ground language instructions in robot control, yet in practice often fail to faithfully follow language. When presented with instructions that lack strong scene-specific supervision, VLAs suffer from counterfactual failures: they act based on vision shortcuts induced by dataset biases, repeatedly executing well-learned behaviors and selecting objects frequently seen during training regardless of language intent. To systematically study it, we introduce LIBERO-CF, the first counterfactual benchmark for VLAs that evaluates language following capability by assigning alternative instructions under visually plausible LIBERO layouts. Our evaluation reveals that counterfactual failures are prevalent yet underexplored across state-of-the-art VLAs. We propose Counterfactual Action Guidance (CAG), a simple yet effective dual-branch inference scheme that explicitly regularizes language conditioning in VLAs. CAG combines a standard VLA policy with a language-unconditioned Vision-Action (VA) module, enabling counterfactual comparison during action selection. This design reduces reliance on visual shortcuts, improves robustness on under-observed tasks, and requires neither additional demonstrations nor modifications to existing architectures or pretrained models. Extensive experiments demonstrate its plug-and-play integration across diverse VLAs and consistent improvements. For example, on LIBERO-CF, CAG improves $π_{0.5}$ by 9.7% in language following accuracy and 3.6% in task success on under-observed tasks using a training-free strategy, with further gains of 15.5% and 8.5%, respectively, when paired with a VA model. In real-world evaluations, CAG reduces counterfactual failures of 9.4% and improves task success by 17.2% on average.

  </details>



- **Toward a Fully Autonomous, AI-Native Particle Accelerator**  
  Chris Tennant  
  _2026-02-19_ · https://arxiv.org/abs/2602.17536v1 · `physics.acc-ph`  
  <details><summary>Abstract</summary>

  This position paper presents a vision for self-driving particle accelerators that operate autonomously with minimal human intervention. We propose that future facilities be designed through artificial intelligence (AI) co-design, where AI jointly optimizes the accelerator lattice, diagnostics, and science application from inception to maximize performance while enabling autonomous operation. Rather than retrofitting AI onto human-centric systems, we envision facilities designed from the ground up as AI-native platforms. We outline nine critical research thrusts spanning agentic control architectures, knowledge integration, adaptive learning, digital twins, health monitoring, safety frameworks, modular hardware design, multimodal data fusion, and cross-domain collaboration. This roadmap aims to guide the accelerator community toward a future where AI-driven design and operation deliver unprecedented science output and reliability.

  </details>



- **Voice-Driven Semantic Perception for UAV-Assisted Emergency Networks**  
  Nuno Saavedra, Pedro Ribeiro, André Coelho, Rui Campos  
  _2026-02-19_ · https://arxiv.org/abs/2602.17394v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  Unmanned Aerial Vehicle (UAV)-assisted networks are increasingly foreseen as a promising approach for emergency response, providing rapid, flexible, and resilient communications in environments where terrestrial infrastructure is degraded or unavailable. In such scenarios, voice radio communications remain essential for first responders due to their robustness; however, their unstructured nature prevents direct integration with automated UAV-assisted network management. This paper proposes SIREN, an AI-driven framework that enables voice-driven perception for UAV-assisted networks. By integrating Automatic Speech Recognition (ASR) with Large Language Model (LLM)-based semantic extraction and Natural Language Processing (NLP) validation, SIREN converts emergency voice traffic into structured, machine-readable information, including responding units, location references, emergency severity, and Quality-of-Service (QoS) requirements. SIREN is evaluated using synthetic emergency scenarios with controlled variations in language, speaker count, background noise, and message complexity. The results demonstrate robust transcription and reliable semantic extraction across diverse operating conditions, while highlighting speaker diarization and geographic ambiguity as the main limiting factors. These findings establish the feasibility of voice-driven situational awareness for UAV-assisted networks and show a practical foundation for human-in-the-loop decision support and adaptive network management in emergency response operations.

  </details>



- **MDP Planning as Policy Inference**  
  David Tolpin  
  _2026-02-19_ · https://arxiv.org/abs/2602.17375v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We cast episodic Markov decision process (MDP) planning as Bayesian inference over _policies_. A policy is treated as the latent variable and is assigned an unnormalized probability of optimality that is monotone in its expected return, yielding a posterior distribution whose modes coincide with return-maximizing solutions while posterior dispersion represents uncertainty over optimal behavior. To approximate this posterior in discrete domains, we adapt variational sequential Monte Carlo (VSMC) to inference over deterministic policies under stochastic dynamics, introducing a sweep that enforces policy consistency across revisited states and couples transition randomness across particles to avoid confounding from simulator noise. Acting is performed by posterior predictive sampling, which induces a stochastic control policy through a Thompson-sampling interpretation rather than entropy regularization. Across grid worlds, Blackjack, Triangle Tireworld, and Academic Advising, we analyze the structure of inferred policy distributions and compare the resulting behavior to discrete Soft Actor-Critic, highlighting qualitative and statistical differences that arise from policy-level uncertainty.

  </details>



- **Nonlinear Predictive Control of the Continuum and Hybrid Dynamics of a Suspended Deformable Cable for Aerial Pick and Place**  
  Antonio Rapuano, Yaolei Shen, Federico Califano, Chiara Gabellieri, Antonio Franchi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17199v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a framework for aerial manipulation of an extensible cable that combines a high-fidelity model based on partial differential equations (PDEs) with a reduced-order representation suitable for real-time control. The PDEs are discretised using a finite-difference method, and proper orthogonal decomposition is employed to extract a reduced-order model (ROM) that retains the dominant deformation modes while significantly reducing computational complexity. Based on this ROM, a nonlinear model predictive control scheme is formulated, capable of stabilizing cable oscillations and handling hybrid transitions such as payload attachment and detachment. Simulation results confirm the stability, efficiency, and robustness of the ROM, as well as the effectiveness of the controller in regulating cable dynamics under a range of operating conditions. Additional simulations illustrate the application of the ROM for trajectory planning in constrained environments, demonstrating the versatility of the proposed approach. Overall, the framework enables real-time, dynamics-aware control of unmanned aerial vehicles (UAVs) carrying suspended flexible cables.

  </details>



- **3D Scene Rendering with Multimodal Gaussian Splatting**  
  Chi-Shiang Gau, Konstantinos D. Polyzos, Athanasios Bacharis, Saketh Madhuvarasu, Tara Javidi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17124v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  3D scene reconstruction and rendering are core tasks in computer vision, with applications spanning industrial monitoring, robotics, and autonomous driving. Recent advances in 3D Gaussian Splatting (GS) and its variants have achieved impressive rendering fidelity while maintaining high computational and memory efficiency. However, conventional vision-based GS pipelines typically rely on a sufficient number of camera views to initialize the Gaussian primitives and train their parameters, typically incurring additional processing cost during initialization while falling short in conditions where visual cues are unreliable, such as adverse weather, low illumination, or partial occlusions. To cope with these challenges, and motivated by the robustness of radio-frequency (RF) signals to weather, lighting, and occlusions, we introduce a multimodal framework that integrates RF sensing, such as automotive radar, with GS-based rendering as a more efficient and robust alternative to vision-only GS rendering. The proposed approach enables efficient depth prediction from only sparse RF-based depth measurements, yielding a high-quality 3D point cloud for initializing Gaussian functions across diverse GS architectures. Numerical tests demonstrate the merits of judiciously incorporating RF sensing into GS pipelines, achieving high-fidelity 3D scene rendering driven by RF-informed structural accuracy.

  </details>



- **Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs**  
  Luke Huang, Zhuoyang Zhang, Qinghao Hu, Shang Yang, Song Han  
  _2026-02-19_ · https://arxiv.org/abs/2602.17616v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) is widely used to improve large language models on reasoning tasks, and asynchronous RL training is attractive because it increases end-to-end throughput. However, for widely adopted critic-free policy-gradient methods such as REINFORCE and GRPO, high asynchrony makes the policy-gradient estimator markedly $\textbf{higher variance}$: training on stale rollouts creates heavy-tailed importance ratios, causing a small fraction of samples to dominate updates. This amplification makes gradients noisy and learning unstable relative to matched on-policy training. Across math and general reasoning benchmarks, we find collapse is reliably predicted by effective sample size (ESS) and unstable gradient norms. Motivated by this diagnosis, we propose $\textbf{V}$ariance $\textbf{C}$ontrolled $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{VCPO}$), a general stabilization method for REINFORCE/GRPO-style algorithms that (i) scales learning rate based on effective sample size to dampen unreliable updates, and (ii) applies a closed-form minimum-variance baseline for the off-policy setting, avoiding an auxiliary value model and adding minimal overhead. Empirically, VCPO substantially improves robustness for asynchronous training across math, general reasoning, and tool-use tasks, outperforming a broad suite of baselines spanning masking/clipping stabilizers and algorithmic variants. This reduces long-context, multi-turn training time by 2.5$\times$ while matching synchronous performance, demonstrating that explicit control of policy-gradient variance is key for reliable asynchronous RL at scale.

  </details>



- **Proximal powered knee placement: a case study**  
  Kyle R. Embry, Lorenzo Vianello, Jim Lipsey, Frank Ursetta, Michael Stephens, Zhi Wang, Ann M. Simon, Andrea J. Ikeda, Suzanne B. Finucane, Shawana Anarwala, et al.  
  _2026-02-19_ · https://arxiv.org/abs/2602.17502v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Lower limb amputation affects millions worldwide, leading to impaired mobility, reduced walking speed, and limited participation in daily and social activities. Powered prosthetic knees can partially restore mobility by actively assisting knee joint torque, improving gait symmetry, sit-to-stand transitions, and walking speed. However, added mass from powered components may diminish these benefits, negatively affecting gait mechanics and increasing metabolic cost. Consequently, optimizing mass distribution, rather than simply minimizing total mass, may provide a more effective and practical solution. In this exploratory study, we evaluated the feasibility of above-knee powertrain placement for a powered prosthetic knee in a small cohort. Compared to below-knee placement, the above-knee configuration demonstrated improved walking speed (+9.2% for one participant) and cadence (+3.6%), with mixed effects on gait symmetry. Kinematic measures indicated similar knee range of motion and peak velocity across configurations. Additional testing on ramps and stairs confirmed the robustness of the control strategy across multiple locomotion tasks. These preliminary findings suggest that above-knee placement is functionally feasible and that careful mass distribution can preserve the benefits of powered assistance while mitigating adverse effects of added weight. Further studies are needed to confirm these trends and guide design and clinical recommendations.

  </details>



- **Inferring Height from Earth Embeddings: First insights using Google AlphaEarth**  
  Alireza Hamoudzadeh, Valeria Belloni, Roberta Ravanelli  
  _2026-02-19_ · https://arxiv.org/abs/2602.17250v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This study investigates whether the geospatial and multimodal features encoded in \textit{Earth Embeddings} can effectively guide deep learning (DL) regression models for regional surface height mapping. In particular, we focused on AlphaEarth Embeddings at 10 m spatial resolution and evaluated their capability to support terrain height inference using a high-quality Digital Surface Model (DSM) as reference. U-Net and U-Net++ architectures were thus employed as lightweight convolutional decoders to assess how well the geospatial information distilled in the embeddings can be translated into accurate surface height estimates. Both architectures achieved strong training performance (both with $R^2 = 0.97$), confirming that the embeddings encode informative and decodable height-related signals. On the test set, performance decreased due to distribution shifts in height frequency between training and testing areas. Nevertheless, U-Net++ shows better generalization ($R^2 = 0.84$, median difference = -2.62 m) compared with the standard U-Net ($R^2 = 0.78$, median difference = -7.22 m), suggesting enhanced robustness to distribution mismatch. While the testing RMSE (approximately 16 m for U-Net++) and residual bias highlight remaining challenges in generalization, strong correlations indicate that the embeddings capture transferable topographic patterns. Overall, the results demonstrate the promising potential of AlphaEarth Embeddings to guide DL-based height mapping workflows, particularly when combined with spatially aware convolutional architectures, while emphasizing the need to address bias for improved regional transferability.

  </details>



- **HiMAP: History-aware Map-occupancy Prediction with Fallback**  
  Yiming Xu, Yi Yang, Hao Cheng, Monika Sester  
  _2026-02-19_ · https://arxiv.org/abs/2602.17231v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate motion forecasting is critical for autonomous driving, yet most predictors rely on multi-object tracking (MOT) with identity association, assuming that objects are correctly and continuously tracked. When tracking fails due to, e.g., occlusion, identity switches, or missed detections, prediction quality degrades and safety risks increase. We present \textbf{HiMAP}, a tracking-free, trajectory prediction framework that remains reliable under MOT failures. HiMAP converts past detections into spatiotemporally invariant historical occupancy maps and introduces a historical query module that conditions on the current agent state to iteratively retrieve agent-specific history from unlabeled occupancy representations. The retrieved history is summarized by a temporal map embedding and, together with the final query and map context, drives a DETR-style decoder to produce multi-modal future trajectories. This design lifts identity reliance, supports streaming inference via reusable encodings, and serves as a robust fallback when tracking is unavailable. On Argoverse~2, HiMAP achieves performance comparable to tracking-based methods while operating without IDs, and it substantially outperforms strong baselines in the no-tracking setting, yielding relative gains of 11\% in FDE, 12\% in ADE, and a 4\% reduction in MR over a fine-tuned QCNet. Beyond aggregate metrics, HiMAP delivers stable forecasts for all agents simultaneously without waiting for tracking to recover, highlighting its practical value for safety-critical autonomy. The code is available under: https://github.com/XuYiMing83/HiMAP.

  </details>



- **Continual uncertainty learning**  
  Heisei Yonezawa, Ansei Yonezawa, Itsuro Kajiwara  
  _2026-02-19_ · https://arxiv.org/abs/2602.17174v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Robust control of mechanical systems with multiple uncertainties remains a fundamental challenge, particularly when nonlinear dynamics and operating-condition variations are intricately intertwined. While deep reinforcement learning (DRL) combined with domain randomization has shown promise in mitigating the sim-to-real gap, simultaneously handling all sources of uncertainty often leads to sub-optimal policies and poor learning efficiency. This study formulates a new curriculum-based continual learning framework for robust control problems involving nonlinear dynamical systems in which multiple sources of uncertainty are simultaneously superimposed. The key idea is to decompose a complex control problem with multiple uncertainties into a sequence of continual learning tasks, in which strategies for handling each uncertainty are acquired sequentially. The original system is extended into a finite set of plants whose dynamic uncertainties are gradually expanded and diversified as learning progresses. The policy is stably updated across the entire plant sets associated with tasks defined by different uncertainty configurations without catastrophic forgetting. To ensure learning efficiency, we jointly incorporate a model-based controller (MBC), which guarantees a shared baseline performance across the plant sets, into the learning process to accelerate the convergence. This residual learning scheme facilitates task-specific optimization of the DRL agent for each uncertainty, thereby enhancing sample efficiency. As a practical industrial application, this study applies the proposed method to designing an active vibration controller for automotive powertrains. We verified that the resulting controller is robust against structural nonlinearities and dynamic variations, realizing successful sim-to-real transfer.

  </details>



- **BadCLIP++: Stealthy and Persistent Backdoors in Multimodal Contrastive Learning**  
  Siyuan Liang, Yongcheng Jing, Yingjie Wang, Jiaxing Huang, Ee-chien Chang, Dacheng Tao  
  _2026-02-19_ · https://arxiv.org/abs/2602.17168v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Research on backdoor attacks against multimodal contrastive learning models faces two key challenges: stealthiness and persistence. Existing methods often fail under strong detection or continuous fine-tuning, largely due to (1) cross-modal inconsistency that exposes trigger patterns and (2) gradient dilution at low poisoning rates that accelerates backdoor forgetting. These coupled causes remain insufficiently modeled and addressed. We propose BadCLIP++, a unified framework that tackles both challenges. For stealthiness, we introduce a semantic-fusion QR micro-trigger that embeds imperceptible patterns near task-relevant regions, preserving clean-data statistics while producing compact trigger distributions. We further apply target-aligned subset selection to strengthen signals at low injection rates. For persistence, we stabilize trigger embeddings via radius shrinkage and centroid alignment, and stabilize model parameters through curvature control and elastic weight consolidation, maintaining solutions within a low-curvature wide basin resistant to fine-tuning. We also provide the first theoretical analysis showing that, within a trust region, gradients from clean fine-tuning and backdoor objectives are co-directional, yielding a non-increasing upper bound on attack success degradation. Experiments demonstrate that with only 0.3% poisoning, BadCLIP++ achieves 99.99% attack success rate (ASR) in digital settings, surpassing baselines by 11.4 points. Across nineteen defenses, ASR remains above 99.90% with less than 0.8% drop in clean accuracy. The method further attains 65.03% success in physical attacks and shows robustness against watermark removal defenses.

  </details>



- **Assessing Ionospheric Scintillation Risk for Direct-to-Cellular Communications using Frequency-Scaled GNSS Observations**  
  Abdollah Masoud Darya, Muhammad Mubasshir Shaikh  
  _2026-02-19_ · https://arxiv.org/abs/2602.17143v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  One of the key issues facing Direct-to-Cellular (D2C) satellite communication systems is ionospheric scintillation on the uplink and downlink, which can significantly degrade link quality. This work investigates the spatial and temporal characteristics of amplitude scintillation at D2C frequencies by scaling L-band scintillation observations from Global Navigation Satellite Systems (GNSS) receivers to bands relevant to D2C operation, including the low-band, and 3GPP's N255 and N256. These observations are then compared to scaled radio-occultation scintillation observations from the FORMOSAT-7/COSMIC-2 (F7/C2) mission, which can be used in regions that do not possess ground-based scintillation monitoring stations. As a proof of concept, five years of ground-based GNSS scintillation data from Sharjah, United Arab Emirates, together with two years of F7/C2 observations over the same region, corresponding to the ascending phase of Solar Cycle 25, are analyzed. Both space-based and ground-based observations indicate a pronounced diurnal scintillation peak between 20--22 local time, particularly during the equinoxes, with occurrence rates increasing with solar activity. Ground-based observations also reveal a strong azimuth dependence, with most scintillation events occurring on southward satellite links. The scintillation occurrence rate at the low-band is more than twice that observed at N255 and N256, highlighting the increased robustness of higher D2C bands to ionospheric scintillation. These results demonstrate how GNSS scintillation observations can be leveraged to characterize and anticipate scintillation-induced D2C link impairments, which help in D2C system design and the implementation of scintillation mitigation strategies.

  </details>



- **Benchmarking the Effects of Object Pose Estimation and Reconstruction on Robotic Grasping Success**  
  Varun Burde, Pavel Burget, Torsten Sattler  
  _2026-02-19_ · https://arxiv.org/abs/2602.17101v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  3D reconstruction serves as the foundational layer for numerous robotic perception tasks, including 6D object pose estimation and grasp pose generation. Modern 3D reconstruction methods for objects can produce visually and geometrically impressive meshes from multi-view images, yet standard geometric evaluations do not reflect how reconstruction quality influences downstream tasks such as robotic manipulation performance. This paper addresses this gap by introducing a large-scale, physics-based benchmark that evaluates 6D pose estimators and 3D mesh models based on their functional efficacy in grasping. We analyze the impact of model fidelity by generating grasps on various reconstructed 3D meshes and executing them on the ground-truth model, simulating how grasp poses generated with an imperfect model affect interaction with the real object. This assesses the combined impact of pose error, grasp robustness, and geometric inaccuracies from 3D reconstruction. Our results show that reconstruction artifacts significantly decrease the number of grasp pose candidates but have a negligible effect on grasping performance given an accurately estimated pose. Our results also reveal that the relationship between grasp success and pose error is dominated by spatial error, and even a simple translation error provides insight into the success of the grasping pose of symmetric objects. This work provides insight into how perception systems relate to object manipulation using robots.

  </details>


