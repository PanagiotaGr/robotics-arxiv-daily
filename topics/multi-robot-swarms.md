# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-03-06 07:04 UTC_

Total papers shown: **6**


---

- **S5-SHB Agent: Society 5.0 enabled Multi-model Agentic Blockchain Framework for Smart Home**  
  Janani Rangila, Akila Siriweera, Incheon Paik, Keitaro Naruse, Isuru Jayanada, Vishmika Devindi  
  _2026-03-05_ · https://arxiv.org/abs/2603.05027v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The smart home is a key application domain within the Society 5.0 vision for a human-centered society. As smart home ecosystems expand with heterogeneous IoT protocols, diverse devices, and evolving threats, autonomous systems must manage comfort, security, energy, and safety for residents. Such autonomous decision-making requires a trust anchor, making blockchain a preferred foundation for transparent and accountable smart home governance. However, realizing this vision requires blockchain-governed smart homes to simultaneously address adaptive consensus, intelligent multi-agent coordination, and resident-controlled governance aligned with the principles of Society 5.0. Existing frameworks rely solely on rigid smart contracts with fixed consensus protocols, employ at most a single AI model without multi-agent coordination, and offer no governance mechanism for residents to control automation behaviour. To address these limitations, this paper presents the Society 5.0-driven human-centered governance-enabled smart home blockchain agent (S5-SHB-Agent). The framework orchestrates ten specialized agents using interchangeable large language models to make decisions across the safety, security, comfort, energy, privacy, and health domains. An adaptive PoW blockchain adjusts mining difficulty based on transaction volume and emergency conditions, with digital signatures and Merkle tree anchoring to ensure tamper evident auditability. A four-tier governance model enables residents to control automation through tiered preferences from routine adjustments to immutable safety thresholds. Evaluation confirms that resident governance correctly separates adjustable comfort priorities from immutable safety thresholds across all tested configurations, while adaptive consensus commits emergency blocks.

  </details>



- **Omni-Manip: Beyond-FOV Large-Workspace Humanoid Manipulation with Omnidirectional 3D Perception**  
  Pei Qu, Zheng Li, Yufei Jia, Ziyun Liu, Liang Zhu, Haoang Li, Jinni Zhou, Jun Ma  
  _2026-03-05_ · https://arxiv.org/abs/2603.05355v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The deployment of humanoid robots for dexterous manipulation in unstructured environments remains challenging due to perceptual limitations that constrain the effective workspace. In scenarios where physical constraints prevent the robot from repositioning itself, maintaining omnidirectional awareness becomes far more critical than color or semantic information. While recent advances in visuomotor policy learning have improved manipulation capabilities, conventional RGB-D solutions suffer from narrow fields of view (FOV) and self-occlusion, requiring frequent base movements that introduce motion uncertainty and safety risks. Existing approaches to expanding perception, including active vision systems and third-view cameras, introduce mechanical complexity, calibration dependencies, and latency that hinder reliable real-time performance. In this work, We propose Omni-Manip, an end-to-end LiDAR-driven 3D visuomotor policy that enables robust manipulation in large workspaces. Our method processes panoramic point clouds through a Time-Aware Attention Pooling mechanism, efficiently encoding sparse 3D data while capturing temporal dependencies. This 360° perception allows the robot to interact with objects across wide areas without frequent repositioning. To support policy learning, we develop a whole-body teleoperation system for efficient data collection on full-body coordination. Extensive experiments in simulation and real-world environments show that Omni-Manip achieves robust performance in large-workspace and cluttered scenarios, outperforming baselines that rely on egocentric depth cameras.

  </details>



- **PhysiFlow: Physics-Aware Humanoid Whole-Body VLA via Multi-Brain Latent Flow Matching and Robust Tracking**  
  Weikai Qin, Sichen Wu, Ci Chen, Mengfan Liu, Linxi Feng, Xinru Cui, Haoqi Han, Hesheng Wang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05410v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In the domain of humanoid robot control, the fusion of Vision-Language-Action (VLA) with whole-body control is essential for semantically guided execution of real-world tasks. However, existing methods encounter challenges in terms of low VLA inference efficiency or an absence of effective semantic guidance for whole-body control, resulting in instability in dynamic limb-coordinated tasks. To bridge this gap, we present a semantic-motion intent guided, physics-aware multi-brain VLA framework for humanoid whole-body control. A series of experiments was conducted to evaluate the performance of the proposed framework. The experimental results demonstrated that the framework enabled reliable vision-language-guided full-body coordination for humanoid robots.

  </details>



- **RoboPocket: Improve Robot Policies Instantly with Your Phone**  
  Junjie Fang, Wendi Chen, Han Xue, Fangyuan Zhou, Tian Le, Yi Wang, Yuting Zhang, Jun Lv, Chuan Wen, Cewu Lu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05504v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Scaling imitation learning is fundamentally constrained by the efficiency of data collection. While handheld interfaces have emerged as a scalable solution for in-the-wild data acquisition, they predominantly operate in an open-loop manner: operators blindly collect demonstrations without knowing the underlying policy's weaknesses, leading to inefficient coverage of critical state distributions. Conversely, interactive methods like DAgger effectively address covariate shift but rely on physical robot execution, which is costly and difficult to scale. To reconcile this trade-off, we introduce RoboPocket, a portable system that enables Robot-Free Instant Policy Iteration using single consumer smartphones. Its core innovation is a Remote Inference framework that visualizes the policy's predicted trajectory via Augmented Reality (AR) Visual Foresight. This immersive feedback allows collectors to proactively identify potential failures and focus data collection on the policy's weak regions without requiring a physical robot. Furthermore, we implement an asynchronous Online Finetuning pipeline that continuously updates the policy with incoming data, effectively closing the learning loop in minutes. Extensive experiments demonstrate that RoboPocket adheres to data scaling laws and doubles the data efficiency compared to offline scaling strategies, overcoming their long-standing efficiency bottleneck. Moreover, our instant iteration loop also boosts sample efficiency by up to 2$\times$ in distributed environments a small number of interactive corrections per person. Project page and videos: https://robo-pocket.github.io.

  </details>



- **SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning**  
  Ye-Chan Kim, SeungJu Cha, Si-Woo Kim, Minju Jeon, Hyungee Kim, Dong-Jin Kim  
  _2026-03-05_ · https://arxiv.org/abs/2603.05437v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Weakly-Supervised Dense Video Captioning aims to localize and describe events in videos trained only on caption annotations, without temporal boundaries. Prior work introduced an implicit supervision paradigm based on Gaussian masking and complementary captioning. However, existing method focuses merely on generating non-overlapping masks without considering their semantic relationship to corresponding events, resulting in simplistic, uniformly distributed masks that fail to capture semantically meaningful regions. Moreover, relying solely on ground-truth captions leads to sub-optimal performance due to the inherent sparsity of existing datasets. In this work, we propose SAIL, which constructs semantically-aware masks through cross-modal alignment. Our similarity aware training objective guides masks to emphasize video regions with high similarity to their corresponding event captions. Furthermore, to guide more accurate mask generation under sparse annotation settings, we introduce an LLM-based augmentation strategy that generates synthetic captions to provide additional alignment signals. These synthetic captions are incorporated through an inter-mask mechanism, providing auxiliary guidance for precise temporal localization without degrading the main objective. Experiments on ActivityNet Captions and YouCook2 demonstrate state-of-the-art performance on both captioning and localization metrics.

  </details>



- **RelaxFlow: Text-Driven Amodal 3D Generation**  
  Jiayin Zhu, Guoji Fu, Xiaolu Liu, Qiyuan He, Yicong Li, Angela Yao  
  _2026-03-05_ · https://arxiv.org/abs/2603.05425v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Image-to-3D generation faces inherent semantic ambiguity under occlusion, where partial observation alone is often insufficient to determine object category. In this work, we formalize text-driven amodal 3D generation, where text prompts steer the completion of unseen regions while strictly preserving input observation. Crucially, we identify that these objectives demand distinct control granularities: rigid control for the observation versus relaxed structural control for the prompt. To this end, we propose RelaxFlow, a training-free dual-branch framework that decouples control granularity via a Multi-Prior Consensus Module and a Relaxation Mechanism. Theoretically, we prove that our relaxation is equivalent to applying a low-pass filter on the generative vector field, which suppresses high-frequency instance details to isolate geometric structure that accommodates the observation. To facilitate evaluation, we introduce two diagnostic benchmarks, ExtremeOcc-3D and AmbiSem-3D. Extensive experiments demonstrate that RelaxFlow successfully steers the generation of unseen regions to match the prompt intent without compromising visual fidelity.

  </details>


