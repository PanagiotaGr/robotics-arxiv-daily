# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-03 07:06 UTC_

Total papers shown: **17**


---

- **Before Autonomy Takes Control: Software Testing in Robotics**  
  Nils Chur, Thiago Santos de Moura, Argentina Ortega, Sven Peldszus, Thorsten Berger, Nico Hochgeschwender, Yannic Noller  
  _2026-02-02_ · https://arxiv.org/abs/2602.02293v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Robotic systems are complex and safety-critical software systems. As such, they need to be tested thoroughly. Unfortunately, robot software is intrinsically hard to test compared to traditional software, mainly since the software needs to closely interact with hardware, account for uncertainty in its operational environment, handle disturbances, and act highly autonomously. However, given the large space in which robots operate, anticipating possible failures when designing tests is challenging. This paper presents a mapping study by considering robotics testing papers and relating them to the software testing theory. We consider 247 robotics testing papers and map them to software testing, discussing the state-of-the-art software testing in robotics with an illustrated example, and discuss current challenges. Forming the basis to introduce both the robotics and software engineering communities to software testing challenges. Finally, we identify open questions and lessons learned.

  </details>



- **David vs. Goliath: Verifiable Agent-to-Agent Jailbreaking via Reinforcement Learning**  
  Samuel Nellessen, Tal Kachman  
  _2026-02-02_ · https://arxiv.org/abs/2602.02395v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The evolution of large language models into autonomous agents introduces adversarial failures that exploit legitimate tool privileges, transforming safety evaluation in tool-augmented environments from a subjective NLP task into an objective control problem. We formalize this threat model as Tag-Along Attacks: a scenario where a tool-less adversary "tags along" on the trusted privileges of a safety-aligned Operator to induce prohibited tool use through conversation alone. To validate this threat, we present Slingshot, a 'cold-start' reinforcement learning framework that autonomously discovers emergent attack vectors, revealing a critical insight: in our setting, learned attacks tend to converge to short, instruction-like syntactic patterns rather than multi-turn persuasion. On held-out extreme-difficulty tasks, Slingshot achieves a 67.0% success rate against a Qwen2.5-32B-Instruct-AWQ Operator (vs. 1.7% baseline), reducing the expected attempts to first success (on solved tasks) from 52.3 to 1.3. Crucially, Slingshot transfers zero-shot to several model families, including closed-source models like Gemini 2.5 Flash (56.0% attack success rate) and defensive-fine-tuned open-source models like Meta-SecAlign-8B (39.2% attack success rate). Our work establishes Tag-Along Attacks as a first-class, verifiable threat model and shows that effective agentic attacks can be elicited from off-the-shelf open-weight models through environment interaction alone.

  </details>



- **Bridging the Sim-to-Real Gap with multipanda ros2: A Real-Time ROS2 Framework for Multimanual Systems**  
  Jon Škerlj, Seongjin Bien, Abdeldjallil Naceri, Sami Haddadin  
  _2026-02-02_ · https://arxiv.org/abs/2602.02269v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present $multipanda\_ros2$, a novel open-source ROS2 architecture for multi-robot control of Franka Robotics robots. Leveraging ros2 control, this framework provides native ROS2 interfaces for controlling any number of robots from a single process. Our core contributions address key challenges in real-time torque control, including interaction control and robot-environment modeling. A central focus of this work is sustaining a 1kHz control frequency, a necessity for real-time control and a minimum frequency required by safety standards. Moreover, we introduce a controllet-feature design pattern that enables controller-switching delays of $\le 2$ ms, facilitating reproducible benchmarking and complex multi-robot interaction scenarios. To bridge the simulation-to-reality (sim2real) gap, we integrate a high-fidelity MuJoCo simulation with quantitative metrics for both kinematic accuracy and dynamic consistency (torques, forces, and control errors). Furthermore, we demonstrate that real-world inertial parameter identification can significantly improve force and torque accuracy, providing a methodology for iterative physics refinement. Our work extends approaches from soft robotics to rigid dual-arm, contact-rich tasks, showcasing a promising method to reduce the sim2real gap and providing a robust, reproducible platform for advanced robotics research.

  </details>



- **FD-VLA: Force-Distilled Vision-Language-Action Model for Contact-Rich Manipulation**  
  Ruiteng Zhao, Wenshuo Wang, Yicheng Ma, Xiaocong Li, Francis E. H. Tay, Marcelo H. Ang, Haiyue Zhu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02142v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Force sensing is a crucial modality for Vision-Language-Action (VLA) frameworks, as it enables fine-grained perception and dexterous manipulation in contact-rich tasks. We present Force-Distilled VLA (FD-VLA), a novel framework that integrates force awareness into contact-rich manipulation without relying on physical force sensors. The core of our approach is a Force Distillation Module (FDM), which distills force by mapping a learnable query token, conditioned on visual observations and robot states, into a predicted force token aligned with the latent representation of actual force signals. During inference, this distilled force token is injected into the pretrained VLM, enabling force-aware reasoning while preserving the integrity of its vision-language semantics. This design provides two key benefits: first, it allows practical deployment across a wide range of robots that lack expensive or fragile force-torque sensors, thereby reducing hardware cost and complexity; second, the FDM introduces an additional force-vision-state fusion prior to the VLM, which improves cross-modal alignment and enhances perception-action robustness in contact-rich scenarios. Surprisingly, our physical experiments show that the distilled force token outperforms direct sensor force measurements as well as other baselines, which highlights the effectiveness of this force-distilled VLA approach.

  </details>



- **SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration**  
  Qingni Wang, Yue Fan, Xin Eric Wang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02419v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Graphical User Interface (GUI) grounding aims to translate natural language instructions into executable screen coordinates, enabling automated GUI interaction. Nevertheless, incorrect grounding can result in costly, hard-to-reverse actions (e.g., erroneous payment approvals), raising concerns about model reliability. In this paper, we introduce SafeGround, an uncertainty-aware framework for GUI grounding models that enables risk-aware predictions through calibrations before testing. SafeGround leverages a distribution-aware uncertainty quantification method to capture the spatial dispersion of stochastic samples from outputs of any given model. Then, through the calibration process, SafeGround derives a test-time decision threshold with statistically guaranteed false discovery rate (FDR) control. We apply SafeGround on multiple GUI grounding models for the challenging ScreenSpot-Pro benchmark. Experimental results show that our uncertainty measure consistently outperforms existing baselines in distinguishing correct from incorrect predictions, while the calibrated threshold reliably enables rigorous risk control and potentials of substantial system-level accuracy improvements. Across multiple GUI grounding models, SafeGround improves system-level accuracy by up to 5.38\% percentage points over Gemini-only inference.

  </details>



- **Flow Policy Gradients for Robot Control**  
  Brent Yi, Hongsuk Choi, Himanshu Gaurav Singh, Xiaoyu Huang, Takara E. Truong, Carmelo Sferrazza, Yi Ma, Rocky Duan, Pieter Abbeel, Guanya Shi, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02481v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Likelihood-based policy gradient methods are the dominant approach for training robot control policies from rewards. These methods rely on differentiable action likelihoods, which constrain policy outputs to simple distributions like Gaussians. In this work, we show how flow matching policy gradients -- a recent framework that bypasses likelihood computation -- can be made effective for training and fine-tuning more expressive policies in challenging robot control settings. We introduce an improved objective that enables success in legged locomotion, humanoid motion tracking, and manipulation tasks, as well as robust sim-to-real transfer on two humanoid robots. We then present ablations and analysis on training dynamics. Results show how policies can exploit the flow representation for exploration when training from scratch, as well as improved fine-tuning robustness over baselines.

  </details>



- **Drift-Bench: Diagnosing Cooperative Breakdowns in LLM Agents under Input Faults via Multi-Turn Interaction**  
  Han Bao, Zheyuan Zhang, Pengcheng Jing, Zhengqing Yuan, Kaiwen Shi, Yanfang Ye  
  _2026-02-02_ · https://arxiv.org/abs/2602.02455v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  As Large Language Models transition to autonomous agents, user inputs frequently violate cooperative assumptions (e.g., implicit intent, missing parameters, false presuppositions, or ambiguous expressions), creating execution risks that text-only evaluations do not capture. Existing benchmarks typically assume well-specified instructions or restrict evaluation to text-only, single-turn clarification, and thus do not measure multi-turn disambiguation under grounded execution risk. We introduce \textbf{Drift-Bench}, the first diagnostic benchmark that evaluates agentic pragmatics under input faults through multi-turn clarification across state-oriented and service-oriented execution environments. Grounded in classical theories of communication, \textbf{Drift-Bench} provides a unified taxonomy of cooperative breakdowns and employs a persona-driven user simulator with the \textbf{Rise} evaluation protocol. Experiments show substantial performance drops under these faults, with clarification effectiveness varying across user personas and fault types. \MethodName bridges clarification research and agent safety evaluation, enabling systematic diagnosis of failures that can lead to unsafe executions.

  </details>



- **Multi-Agent Monte Carlo Tree Search for Makespan-Efficient Object Rearrangement in Cluttered Spaces**  
  Hanwen Ren, Junyong Kim, Aathman Tharmasanthiran, Ahmed H. Qureshi  
  _2026-02-02_ · https://arxiv.org/abs/2602.02411v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Object rearrangement planning in complex, cluttered environments is a common challenge in warehouses, households, and rescue sites. Prior studies largely address monotone instances, whereas real-world tasks are often non-monotone-objects block one another and must be temporarily relocated to intermediate positions before reaching their final goals. In such settings, effective multi-agent collaboration can substantially reduce the time required to complete tasks. This paper introduces Centralized, Asynchronous, Multi-agent Monte Carlo Tree Search (CAM-MCTS), a novel framework for general-purpose makespan-efficient object rearrangement planning in challenging environments. CAM-MCTS combines centralized task assignment-where agents remain aware of each other's intended actions to facilitate globally optimized planning-with an asynchronous task execution strategy that enables agents to take on new tasks at appropriate time steps, rather than waiting for others, guided by a one-step look-ahead cost estimate. This design minimizes idle time, prevents unnecessary synchronization delays, and enhances overall system efficiency. We evaluate CAM-MCTS across a diverse set of monotone and non-monotone tasks in cluttered environments, demonstrating consistent reductions in makespan compared to strong baselines. Finally, we validate our approach on a real-world multi-agent system under different configurations, further confirming its effectiveness and robustness.

  </details>



- **Self-Supervised Learning from Structural Invariance**  
  Yipeng Zhang, Hafez Ghaemi, Jungyoon Lee, Shahab Bakhtiari, Eilif B. Muller, Laurent Charlin  
  _2026-02-02_ · https://arxiv.org/abs/2602.02381v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Joint-embedding self-supervised learning (SSL), the key paradigm for unsupervised representation learning from visual data, learns from invariances between semantically-related data pairs. We study the one-to-many mapping problem in SSL, where each datum may be mapped to multiple valid targets. This arises when data pairs come from naturally occurring generative processes, e.g., successive video frames. We show that existing methods struggle to flexibly capture this conditional uncertainty. As a remedy, we introduce a latent variable to account for this uncertainty and derive a variational lower bound on the mutual information between paired embeddings. Our derivation yields a simple regularization term for standard SSL objectives. The resulting method, which we call AdaSSL, applies to both contrastive and distillation-based SSL objectives, and we empirically show its versatility in causal representation learning, fine-grained image understanding, and world modeling on videos.

  </details>



- **From Sycophancy to Sensemaking: Premise Governance for Human-AI Decision Making**  
  Raunak Jain, Mudita Khurana, John Stephens, Srinivas Dharmasanam, Shankar Venkataraman  
  _2026-02-02_ · https://arxiv.org/abs/2602.02378v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  As LLMs expand from assistance to decision support, a dangerous pattern emerges: fluent agreement without calibrated judgment. Low-friction assistants can become sycophantic, baking in implicit assumptions and pushing verification costs onto experts, while outcomes arrive too late to serve as reward signals. In deep-uncertainty decisions (where objectives are contested and reversals are costly), scaling fluent agreement amplifies poor commitments faster than it builds expertise. We argue reliable human-AI partnership requires a shift from answer generation to collaborative premise governance over a knowledge substrate, negotiating only what is decision-critical. A discrepancy-driven control loop operates over this substrate: detecting conflicts, localizing misalignment via typed discrepancies (teleological, epistemic, procedural), and triggering bounded negotiation through decision slices. Commitment gating blocks action on uncommitted load-bearing premises unless overridden under logged risk; value-gated challenge allocates probing under interaction cost. Trust then attaches to auditable premises and evidence standards, not conversational fluency. We illustrate with tutoring and propose falsifiable evaluation criteria.

  </details>



- **Enhancing Indoor Occupancy Prediction via Sparse Query-Based Multi-Level Consistent Knowledge Distillation**  
  Xiang Li, Yupeng Zheng, Pengfei Li, Yilun Chen, Ya-Qin Zhang, Wenchao Ding  
  _2026-02-02_ · https://arxiv.org/abs/2602.02318v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Occupancy prediction provides critical geometric and semantic understanding for robotics but faces efficiency-accuracy trade-offs. Current dense methods suffer computational waste on empty voxels, while sparse query-based approaches lack robustness in diverse and complex indoor scenes. In this paper, we propose DiScene, a novel sparse query-based framework that leverages multi-level distillation to achieve efficient and robust occupancy prediction. In particular, our method incorporates two key innovations: (1) a Multi-level Consistent Knowledge Distillation strategy, which transfers hierarchical representations from large teacher models to lightweight students through coordinated alignment across four levels, including encoder-level feature alignment, query-level feature matching, prior-level spatial guidance, and anchor-level high-confidence knowledge transfer and (2) a Teacher-Guided Initialization policy, employing optimized parameter warm-up to accelerate model convergence. Validated on the Occ-Scannet benchmark, DiScene achieves 23.2 FPS without depth priors while outperforming our baseline method, OPUS, by 36.1% and even better than the depth-enhanced version, OPUS†. With depth integration, DiScene† attains new SOTA performance, surpassing EmbodiedOcc by 3.7% with 1.62$\times$ faster inference speed. Furthermore, experiments on the Occ3D-nuScenes benchmark and in-the-wild scenarios demonstrate the versatility of our approach in various environments. Code and models can be accessed at https://github.com/getterupper/DiScene.

  </details>



- **SurvKAN: A Fully Parametric Survival Model Based on Kolmogorov-Arnold Networks**  
  Marina Mastroleo, Alberto Archetti, Federico Mastroleo, Matteo Matteucci  
  _2026-02-02_ · https://arxiv.org/abs/2602.02179v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Accurate prediction of time-to-event outcomes is critical for clinical decision-making, treatment planning, and resource allocation in modern healthcare. While classical survival models such as Cox remain widely adopted in standard practice, they rely on restrictive assumptions, including linear covariate relationships and proportional hazards over time, that often fail to capture real-world clinical dynamics. Recent deep learning approaches like DeepSurv and DeepHit offer improved expressivity but sacrifice interpretability, limiting clinical adoption where trust and transparency are paramount. Hybrid models incorporating Kolmogorov-Arnold Networks (KANs), such as CoxKAN, have begun to address this trade-off but remain constrained by the semi-parametric Cox framework. In this work we introduce SurvKAN, a fully parametric, time-continuous survival model based on KAN architectures that eliminates the proportional hazards constraint. SurvKAN treats time as an explicit input to a KAN that directly predicts the log-hazard function, enabling end-to-end training on the full survival likelihood. Our architecture preserves interpretability through learnable univariate functions that indicate how individual features influence risk over time. Extensive experiments on standard survival benchmarks demonstrate that SurvKAN achieves competitive or superior performance compared to classical and state-of-the-art baselines across concordance and calibration metrics. Additionally, interpretability analyses reveal clinically meaningful patterns that align with medical domain knowledge.

  </details>



- **Lung Nodule Image Synthesis Driven by Two-Stage Generative Adversarial Networks**  
  Lu Cao, Xiquan He, Junying Zeng, Chaoyun Mai, Min Luo  
  _2026-02-02_ · https://arxiv.org/abs/2602.02171v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The limited sample size and insufficient diversity of lung nodule CT datasets severely restrict the performance and generalization ability of detection models. Existing methods generate images with insufficient diversity and controllability, suffering from issues such as monotonous texture features and distorted anatomical structures. Therefore, we propose a two-stage generative adversarial network (TSGAN) to enhance the diversity and spatial controllability of synthetic data by decoupling the morphological structure and texture features of lung nodules. In the first stage, StyleGAN is used to generate semantic segmentation mask images, encoding lung nodules and tissue backgrounds to control the anatomical structure of lung nodule images; The second stage uses the DL-Pix2Pix model to translate the mask map into CT images, employing local importance attention to capture local features, while utilizing dynamic weight multi-head window attention to enhance the modeling capability of lung nodule texture and background. Compared to the original dataset, the accuracy improved by 4.6% and mAP by 4% on the LUNA16 dataset. Experimental results demonstrate that TSGAN can enhance the quality of synthetic images and the performance of detection models.

  </details>



- **ECHO: Entropy-Confidence Hybrid Optimization for Test-Time Reinforcement Learning**  
  Chu Zhao, Enneng Yang, Yuting Liu, Jianzhe Zhao, Guibing Guo  
  _2026-02-02_ · https://arxiv.org/abs/2602.02150v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Test-time reinforcement learning generates multiple candidate answers via repeated rollouts and performs online updates using pseudo-labels constructed by majority voting. To reduce overhead and improve exploration, prior work introduces tree structured rollouts, which share reasoning prefixes and branch at key nodes to improve sampling efficiency. However, this paradigm still faces two challenges: (1) high entropy branching can trigger rollout collapse, where the branching budget concentrates on a few trajectories with consecutive high-entropy segments, rapidly reducing the number of effective branches; (2) early pseudo-labels are noisy and biased, which can induce self-reinforcing overfitting, causing the policy to sharpen prematurely and suppress exploration. To address these issues, we propose Entropy Confidence Hybrid Group Relative Policy Optimization (ECHO). During rollout, ECHO jointly leverages local entropy and group level confidence to adaptively control branch width, and further introduces online confidence-based pruning to terminate persistently low confidence branches, avoiding high entropy traps and mitigating collapse. During policy updates, ECHO employs confidence adaptive clipping and an entropy confidence hybrid advantage shaping approach to enhance training robustness and mitigate early stage bias. Experiments demonstrate that ECHO achieves consistent gains on multiple mathematical and visual reasoning benchmarks, and generalizes more effectively under a limited rollout budget.

  </details>



- **RIS-Aided Wireless Amodal Sensing for Single-View 3D Reconstruction**  
  Yuhan Wang, Haobo Zhang, Qingyu Liu, Hongliang Zhang, Lingyang Song  
  _2026-02-02_ · https://arxiv.org/abs/2602.02148v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Amodal sensing is critical for various real-world sensing applications because it can recover the complete shapes of partially occluded objects in complex environments. Among various amodal sensing paradigms, wireless amodal sensing is a potential solution due to its advantages of environmental robustness, privacy preservation, and low cost. However, the sensing data obtained by wireless system is sparse for shape reconstruction because of the low spatial resolution, and this issue is further intensified in complex environments with occlusion. To address this issue, we propose a Reconfigurable Intelligent Surface (RIS)-aided wireless amodal sensing scheme that leverages a large-scale RIS to enhance the spatial resolution and create reflection paths that can bypass the obstacles. A generative learning model is also employed to reconstruct the complete shape based on the sensing data captured from the viewpoint of the RIS. In such a system, it is challenging to optimize the RIS phase shifts because the relationship between RIS phase shifts and amodal sensing accuracy is complex and the closed-form expression is unknown. To tackle this challenge, we develop an error prediction model that learns the mapping from RIS phase shifts to amodal sensing accuracy, and optimizes RIS phase shifts based on this mapping. Experimental results on the benchmark dataset show that our method achieves at least a 56.73% reduction in reconstruction error compared to conventional schemes under the same number of RIS configurations.

  </details>



- **DCoPilot: Generative AI-Empowered Policy Adaptation for Dynamic Data Center Operations**  
  Minghao Li, Ruihang Wang, Rui Tan, Yonggang Wen  
  _2026-02-02_ · https://arxiv.org/abs/2602.02137v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Modern data centers (DCs) hosting artificial intelligence (AI)-dedicated devices operate at high power densities with rapidly varying workloads, making minute-level adaptation essential for safe and energy-efficient operation. However, manually designing piecewise deep reinforcement learning (DRL) agents cannot keep pace with frequent dynamics shifts and service-level agreement (SLA) changes of an evolving DC. This specification-to-policy lag causes a lack of timely, effective control policies, which may lead to service outages. To bridge the gap, we present DCoPilot, a hybrid framework for generative control policies in dynamic DC operation. DCoPilot synergizes two distinct generative paradigms, i.e., a large language model (LLM) that performs symbolic generation of structured reward forms, and a hypernetwork that conducts parametric generation of policy weights. DCoPilot operates through three coordinated phases: (i) simulation scale-up, which stress-tests reward candidates across diverse simulation-ready (SimReady) scenes; (ii) meta policy distillation, where a hypernetwork is trained to output policy weights conditioned on SLA and scene embeddings; and (iii) online adaptation, enabling zero-shot policy generation in response to updated specifications. Evaluated across five control task families spanning diverse DC components, DCoPilot achieves near-zero constraint violations and outperforms all baselines across specification variations. Ablation studies validate the effectiveness of LLM-based unified reward generation in enabling stable hypernetwork convergence.

  </details>



- **No Global Plan in Chain-of-Thought: Uncover the Latent Planning Horizon of LLMs**  
  Liyan Xu, Mo Yu, Fandong Meng, Jie Zhou  
  _2026-02-02_ · https://arxiv.org/abs/2602.02103v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  This work stems from prior complementary observations on the dynamics of Chain-of-Thought (CoT): Large Language Models (LLMs) is shown latent planning of subsequent reasoning prior to CoT emergence, thereby diminishing the significance of explicit CoT; whereas CoT remains critical for tasks requiring multi-step reasoning. To deepen the understanding between LLM's internal states and its verbalized reasoning trajectories, we investigate the latent planning strength of LLMs, through our probing method, Tele-Lens, applying to hidden states across diverse task domains. Our empirical results indicate that LLMs exhibit a myopic horizon, primarily conducting incremental transitions without precise global planning. Leveraging this characteristic, we propose a hypothesis on enhancing uncertainty estimation of CoT, which we validate that a small subset of CoT positions can effectively represent the uncertainty of the entire path. We further underscore the significance of exploiting CoT dynamics, and demonstrate that automatic recognition of CoT bypass can be achieved without performance degradation. Our code, data and models are released at https://github.com/lxucs/tele-lens.

  </details>


