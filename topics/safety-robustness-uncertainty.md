# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-07 06:57 UTC_

Total papers shown: **11**


---

- **HiCrowd: Hierarchical Crowd Flow Alignment for Dense Human Environments**  
  Yufei Zhu, Shih-Min Yang, Martin Magnusson, Allan Wang  
  _2026-02-05_ · https://arxiv.org/abs/2602.05608v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Navigating through dense human crowds remains a significant challenge for mobile robots. A key issue is the freezing robot problem, where the robot struggles to find safe motions and becomes stuck within the crowd. To address this, we propose HiCrowd, a hierarchical framework that integrates reinforcement learning (RL) with model predictive control (MPC). HiCrowd leverages surrounding pedestrian motion as guidance, enabling the robot to align with compatible crowd flows. A high-level RL policy generates a follow point to align the robot with a suitable pedestrian group, while a low-level MPC safely tracks this guidance with short horizon planning. The method combines long-term crowd aware decision making with safe short-term execution. We evaluate HiCrowd against reactive and learning-based baselines in offline setting (replaying recorded human trajectories) and online setting (human trajectories are updated to react to the robot in simulation). Experiments on a real-world dataset and a synthetic crowd dataset show that our method outperforms in navigation efficiency and safety, while reducing freezing behaviors. Our results suggest that leveraging human motion as guidance, rather than treating humans solely as dynamic obstacles, provides a powerful principle for safe and efficient robot navigation in crowds.

  </details>



- **Perception-Based Beliefs for POMDPs with Visual Observations**  
  Miriam Schäfers, Merlijn Krale, Thiago D. Simão, Nils Jansen, Maximilian Weininger  
  _2026-02-05_ · https://arxiv.org/abs/2602.05679v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Partially observable Markov decision processes (POMDPs) are a principled planning model for sequential decision-making under uncertainty. Yet, real-world problems with high-dimensional observations, such as camera images, remain intractable for traditional belief- and filtering-based solvers. To tackle this problem, we introduce the Perception-based Beliefs for POMDPs framework (PBP), which complements such solvers with a perception model. This model takes the form of an image classifier which maps visual observations to probability distributions over states. PBP incorporates these distributions directly into belief updates, so the underlying solver does not need to reason explicitly over high-dimensional observation spaces. We show that the belief update of PBP coincides with the standard belief update if the image classifier is exact. Moreover, to handle classifier imprecision, we incorporate uncertainty quantification and introduce two methods to adjust the belief update accordingly. We implement PBP using two traditional POMDP solvers and empirically show that (1) it outperforms existing end-to-end deep RL methods and (2) uncertainty quantification improves robustness of PBP against visual corruption.

  </details>



- **Regularized Calibration with Successive Rounding for Post-Training Quantization**  
  Seohyeon Cha, Huancheng Chen, Dongjun Kim, Haoran Zhang, Kevin Chan, Gustavo de Veciana, Haris Vikalo  
  _2026-02-05_ · https://arxiv.org/abs/2602.05902v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) deliver robust performance across diverse applications, yet their deployment often faces challenges due to the memory and latency costs of storing and accessing billions of parameters. Post-training quantization (PTQ) enables efficient inference by mapping pretrained weights to low-bit formats without retraining, but its effectiveness depends critically on both the quantization objective and the rounding procedure used to obtain low-bit weight representations. In this work, we show that interpolating between symmetric and asymmetric calibration acts as a form of regularization that preserves the standard quadratic structure used in PTQ while providing robustness to activation mismatch. Building on this perspective, we derive a simple successive rounding procedure that naturally incorporates asymmetric calibration, as well as a bounded-search extension that allows for an explicit trade-off between quantization quality and the compute cost. Experiments across multiple LLM families, quantization bit-widths, and benchmarks demonstrate that the proposed bounded search based on a regularized asymmetric calibration objective consistently improves perplexity and accuracy over PTQ baselines, while incurring only modest and controllable additional computational cost.

  </details>



- **Depth as Prior Knowledge for Object Detection**  
  Moussa Kassem Sbeyti, Nadja Klein  
  _2026-02-05_ · https://arxiv.org/abs/2602.05730v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Detecting small and distant objects remains challenging for object detectors due to scale variation, low resolution, and background clutter. Safety-critical applications require reliable detection of these objects for safe planning. Depth information can improve detection, but existing approaches require complex, model-specific architectural modifications. We provide a theoretical analysis followed by an empirical investigation of the depth-detection relationship. Together, they explain how depth causes systematic performance degradation and why depth-informed supervision mitigates it. We introduce DepthPrior, a framework that uses depth as prior knowledge rather than as a fused feature, providing comparable benefits without modifying detector architectures. DepthPrior consists of Depth-Based Loss Weighting (DLW) and Depth-Based Loss Stratification (DLS) during training, and Depth-Aware Confidence Thresholding (DCT) during inference. The only overhead is the initial cost of depth estimation. Experiments across four benchmarks (KITTI, MS COCO, VisDrone, SUN RGB-D) and two detectors (YOLOv11, EfficientDet) demonstrate the effectiveness of DepthPrior, achieving up to +9% mAP$_S$ and +7% mAR$_S$ for small objects, with inference recovery rates as high as 95:1 (true vs. false detections). DepthPrior offers these benefits without additional sensors, architectural changes, or performance costs. Code is available at https://github.com/mos-ks/DepthPrior.

  </details>



- **ROMAN: Reward-Orchestrated Multi-Head Attention Network for Autonomous Driving System Testing**  
  Jianlei Chi, Yuzhen Wu, Jiaxuan Hou, Xiaodong Zhang, Ming Fan, Suhui Sun, Weijun Dai, Bo Li, Jianguo Sun, Jun Sun  
  _2026-02-05_ · https://arxiv.org/abs/2602.05629v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Automated Driving System (ADS) acts as the brain of autonomous vehicles, responsible for their safety and efficiency. Safe deployment requires thorough testing in diverse real-world scenarios and compliance with traffic laws like speed limits, signal obedience, and right-of-way rules. Violations like running red lights or speeding pose severe safety risks. However, current testing approaches face significant challenges: limited ability to generate complex and high-risk law-breaking scenarios, and failing to account for complex interactions involving multiple vehicles and critical situations. To address these challenges, we propose ROMAN, a novel scenario generation approach for ADS testing that combines a multi-head attention network with a traffic law weighting mechanism. ROMAN is designed to generate high-risk violation scenarios to enable more thorough and targeted ADS evaluation. The multi-head attention mechanism models interactions among vehicles, traffic signals, and other factors. The traffic law weighting mechanism implements a workflow that leverages an LLM-based risk weighting module to evaluate violations based on the two dimensions of severity and occurrence. We have evaluated ROMAN by testing the Baidu Apollo ADS within the CARLA simulation platform and conducting extensive experiments to measure its performance. Experimental results demonstrate that ROMAN surpassed state-of-the-art tools ABLE and LawBreaker by achieving 7.91% higher average violation count than ABLE and 55.96% higher than LawBreaker, while also maintaining greater scenario diversity. In addition, only ROMAN successfully generated violation scenarios for every clause of the input traffic laws, enabling it to identify more high-risk violations than existing approaches.

  </details>



- **IDSOR: Intensity- and Distance-Aware Statistical Outlier Removal for Weather-Robust LiDAR Point Clouds**  
  Chenyang Yan, Mats Bengtsson  
  _2026-02-05_ · https://arxiv.org/abs/2602.05876v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  LiDAR point clouds captured in rain or snow are often corrupted by weather-induced returns, which can degrade perception and safety-critical scene understanding. This paper proposes Intensity- and Distance-Aware Statistical Outlier Removal (IDSOR), a range-adaptive filtering method that jointly exploits intensity cues and neighborhood sparsity. By incorporating an empirical, range-dependent distribution of weather returns into the threshold design, IDSOR suppresses weather-induced points while preserving fine structural details without cumbersome manual parameter tuning. We also propose a variant that uses a previously proposed method to estimate the weather return distribution from data, and integrates it into IDSOR. Experiments on simulation-augmented level-crossing measurements and on the Winter Adverse Driving dataset (WADS) demonstrate that IDSOR achieves a favorable precision-recall trade-off, maintaining both precision and recall above 90% on WADS.

  </details>



- **Scalable and General Whole-Body Control for Cross-Humanoid Locomotion**  
  Yufei Xue, YunFeng Lin, Wentao Dong, Yang Tang, Jingbo Wang, Jiangmiao Pang, Ming Zhou, Minghuan Liu, Weinan Zhang  
  _2026-02-05_ · https://arxiv.org/abs/2602.05791v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning-based whole-body controllers have become a key driver for humanoid robots, yet most existing approaches require robot-specific training. In this paper, we study the problem of cross-embodiment humanoid control and show that a single policy can robustly generalize across a wide range of humanoid robot designs with one-time training. We introduce XHugWBC, a novel cross-embodiment training framework that enables generalist humanoid control through: (1) physics-consistent morphological randomization, (2) semantically aligned observation and action spaces across diverse humanoid robots, and (3) effective policy architectures modeling morphological and dynamical properties. XHugWBC is not tied to any specific robot. Instead, it internalizes a broad distribution of morphological and dynamical characteristics during training. By learning motion priors from diverse randomized embodiments, the policy acquires a strong structural bias that supports zero-shot transfer to previously unseen robots. Experiments on twelve simulated humanoids and seven real-world robots demonstrate the strong generalization and robustness of the resulting universal controller.

  </details>



- **From Bench to Flight: Translating Drone Impact Tests into Operational Safety Limits**  
  Aziz Mohamed Mili, Louis Catar, Paul Gérard, Ilyass Tabiai, David St-Onge  
  _2026-02-05_ · https://arxiv.org/abs/2602.05922v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Indoor micro-aerial vehicles (MAVs) are increasingly used for tasks that require close proximity to people, yet practitioners lack practical methods to tune motion limits based on measured impact risk. We present an end-to-end, open toolchain that converts benchtop impact tests into deployable safety governors for drones. First, we describe a compact and replicable impact rig and protocol for capturing force-time profiles across drone classes and contact surfaces. Second, we provide data-driven models that map pre-impact speed to impulse and contact duration, enabling direct computation of speed bounds for a target force limit. Third, we release scripts and a ROS2 node that enforce these bounds online and log compliance, with support for facility-specific policies. We validate the workflow on multiple commercial off-the-shelf quadrotors and representative indoor assets, demonstrating that the derived governors preserve task throughput while meeting force constraints specified by safety stakeholders. Our contribution is a practical bridge from measured impacts to runtime limits, with shareable datasets, code, and a repeatable process that teams can adopt to certify indoor MAV operations near humans.

  </details>



- **Residual Reinforcement Learning for Waste-Container Lifting Using Large-Scale Cranes with Underactuated Tools**  
  Qi Li, Karsten Berns  
  _2026-02-05_ · https://arxiv.org/abs/2602.05895v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper studies the container lifting phase of a waste-container recycling task in urban environments, performed by a hydraulic loader crane equipped with an underactuated discharge unit, and proposes a residual reinforcement learning (RRL) approach that combines a nominal Cartesian controller with a learned residual policy. All experiments are conducted in simulation, where the task is characterized by tight geometric tolerances between the discharge-unit hooks and the container rings relative to the overall crane scale, making precise trajectory tracking and swing suppression essential. The nominal controller uses admittance control for trajectory tracking and pendulum-aware swing damping, followed by damped least-squares inverse kinematics with a nullspace posture term to generate joint velocity commands. A PPO-trained residual policy in Isaac Lab compensates for unmodeled dynamics and parameter variations, improving precision and robustness without requiring end-to-end learning from scratch. We further employ randomized episode initialization and domain randomization over payload properties, actuator gains, and passive joint parameters to enhance generalization. Simulation results demonstrate improved tracking accuracy, reduced oscillations, and higher lifting success rates compared to the nominal controller alone.

  </details>



- **DFPO: Scaling Value Modeling via Distributional Flow towards Robust and Generalizable LLM Post-Training**  
  Dingwei Zhu, Zhiheng Xi, Shihan Dou, Jiahan Li, Chenhao Huang, Junjie Ye, Sixian Li, Mingxu Chai, Yuhui Wang, Yajie Yang, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05890v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Training reinforcement learning (RL) systems in real-world environments remains challenging due to noisy supervision and poor out-of-domain (OOD) generalization, especially in LLM post-training. Recent distributional RL methods improve robustness by modeling values with multiple quantile points, but they still learn each quantile independently as a scalar. This results in rough-grained value representations that lack fine-grained conditioning on state information, struggling under complex and OOD conditions. We propose DFPO (Distributional Value Flow Policy Optimization with Conditional Risk and Consistency Control), a robust distributional RL framework that models values as continuous flows across time steps. By scaling value modeling through learning of a value flow field instead of isolated quantile predictions, DFPO captures richer state information for more accurate advantage estimation. To stabilize training under noisy feedback, DFPO further integrates conditional risk control and consistency constraints along value flow trajectories. Experiments on dialogue, math reasoning, and scientific tasks show that DFPO outperforms PPO, FlowRL, and other robust baselines under noisy supervision, achieving improved training stability and generalization.

  </details>



- **Agent2Agent Threats in Safety-Critical LLM Assistants: A Human-Centric Taxonomy**  
  Lukas Stappen, Ahmet Erkan Turan, Johann Hagerer, Georg Groh  
  _2026-02-05_ · https://arxiv.org/abs/2602.05877v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The integration of Large Language Model (LLM)-based conversational agents into vehicles creates novel security challenges at the intersection of agentic AI, automotive safety, and inter-agent communication. As these intelligent assistants coordinate with external services via protocols such as Google's Agent-to-Agent (A2A), they establish attack surfaces where manipulations can propagate through natural language payloads, potentially causing severe consequences ranging from driver distraction to unauthorized vehicle control. Existing AI security frameworks, while foundational, lack the rigorous "separation of concerns" standard in safety-critical systems engineering by co-mingling the concepts of what is being protected (assets) with how it is attacked (attack paths). This paper addresses this methodological gap by proposing a threat modeling framework called AgentHeLLM (Agent Hazard Exploration for LLM Assistants) that formally separates asset identification from attack path analysis. We introduce a human-centric asset taxonomy derived from harm-oriented "victim modeling" and inspired by the Universal Declaration of Human Rights, and a formal graph-based model that distinguishes poison paths (malicious data propagation) from trigger paths (activation actions). We demonstrate the framework's practical applicability through an open-source attack path suggestion tool AgentHeLLM Attack Path Generator that automates multi-stage threat discovery using a bi-level search strategy.

  </details>


