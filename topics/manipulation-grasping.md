# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-10 07:19 UTC_

Total papers shown: **7**


---

- **Dexterous Manipulation Policies from RGB Human Videos via 4D Hand-Object Trajectory Reconstruction**  
  Hongyi Chen, Tony Dong, Tiancheng Wu, Liquan Wang, Yash Jangir, Yaru Niu, Yufei Ye, Homanga Bharadhwaj, Zackory Erickson, Jeffrey Ichnowski  
  _2026-02-09_ · https://arxiv.org/abs/2602.09013v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-finger robotic hand manipulation and grasping are challenging due to the high-dimensional action space and the difficulty of acquiring large-scale training data. Existing approaches largely rely on human teleoperation with wearable devices or specialized sensing equipment to capture hand-object interactions, which limits scalability. In this work, we propose VIDEOMANIP, a device-free framework that learns dexterous manipulation directly from RGB human videos. Leveraging recent advances in computer vision, VIDEOMANIP reconstructs explicit 4D robot-object trajectories from monocular videos by estimating human hand poses, object meshes, and retargets the reconstructed human motions to robotic hands for manipulation learning. To make the reconstructed robot data suitable for dexterous manipulation training, we introduce hand-object contact optimization with interaction-centric grasp modeling, as well as a demonstration synthesis strategy that generates diverse training trajectories from a single video, enabling generalizable policy learning without additional robot demonstrations. In simulation, the learned grasping model achieves a 70.25% success rate across 20 diverse objects using the Inspire Hand. In the real world, manipulation policies trained from RGB videos achieve an average 62.86% success rate across seven tasks using the LEAP Hand, outperforming retargeting-based methods by 15.87%. Project videos are available at videomanip.github.io.

  </details>



- **A Precise Real-Time Force-Aware Grasping System for Robust Aerial Manipulation**  
  Kenghou Hoi, Yuze Wu, Annan Ding, Junjie Wang, Anke Zhao, Chengqian Zhang, Fei Gao  
  _2026-02-09_ · https://arxiv.org/abs/2602.08599v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Aerial manipulation requires force-aware capabilities to enable safe and effective grasping and physical interaction. Previous works often rely on heavy, expensive force sensors unsuitable for typical quadrotor platforms, or perform grasping without force feedback, risking damage to fragile objects. To address these limitations, we propose a novel force-aware grasping framework incorporating six low-cost, sensitive skin-like tactile sensors. We introduce a magnetic-based tactile sensing module that provides high-precision three-dimensional force measurements. We eliminate geomagnetic interference through a reference Hall sensor and simplify the calibration process compared to previous work. The proposed framework enables precise force-aware grasping control, allowing safe manipulation of fragile objects and real-time weight measurement of grasped items. The system is validated through comprehensive real-world experiments, including balloon grasping, dynamic load variation tests, and ablation studies, demonstrating its effectiveness in various aerial manipulation scenarios. Our approach achieves fully onboard operation without external motion capture systems, significantly enhancing the practicality of force-sensitive aerial manipulation. The supplementary video is available at: https://www.youtube.com/watch?v=mbcZkrJEf1I.

  </details>



- **Mimic Intent, Not Just Trajectories**  
  Renming Huang, Chendong Zeng, Wenjing Tang, Jingtian Cai, Cewu Lu, Panpan Cai  
  _2026-02-09_ · https://arxiv.org/abs/2602.08602v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While imitation learning (IL) has achieved impressive success in dexterous manipulation through generative modeling and pretraining, state-of-the-art approaches like Vision-Language-Action (VLA) models still struggle with adaptation to environmental changes and skill transfer. We argue this stems from mimicking raw trajectories without understanding the underlying intent. To address this, we propose explicitly disentangling behavior intent from execution details in end-2-end IL: \textit{``Mimic Intent, Not just Trajectories'' (MINT)}. We achieve this via \textit{multi-scale frequency-space tokenization}, which enforces a spectral decomposition of action chunk representation. We learn action tokens with a multi-scale coarse-to-fine structure, and force the coarsest token to capture low-frequency global structure and finer tokens to encode high-frequency details. This yields an abstract \textit{Intent token} that facilitates planning and transfer, and multi-scale \textit{Execution tokens} that enable precise adaptation to environmental dynamics. Building on this hierarchy, our policy generates trajectories through \textit{next-scale autoregression}, performing progressive \textit{intent-to-execution reasoning}, thus boosting learning efficiency and generalization. Crucially, this disentanglement enables \textit{one-shot transfer} of skills, by simply injecting the Intent token from a demonstration into the autoregressive generation process. Experiments on several manipulation benchmarks and on a real robot demonstrate state-of-the-art success rates, superior inference efficiency, robust generalization against disturbances, and effective one-shot transfer.

  </details>



- **Mind the Gap: Learning Implicit Impedance in Visuomotor Policies via Intent-Execution Mismatch**  
  Cuijie Xu, Shurui Zheng, Zihao Su, Yuanfan Xu, Tinghao Yi, Xudong Zhang, Jian Wang, Yu Wang, Jinchen Yu  
  _2026-02-09_ · https://arxiv.org/abs/2602.08776v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Teleoperation inherently relies on the human operator acting as a closed-loop controller to actively compensate for hardware imperfections, including latency, mechanical friction, and lack of explicit force feedback. Standard Behavior Cloning (BC), by mimicking the robot's executed trajectory, fundamentally ignores this compensatory mechanism. In this work, we propose a Dual-State Conditioning framework that shifts the learning objective to "Intent Cloning" (master command). We posit that the Intent-Execution Mismatch, the discrepancy between master command and slave response, is not noise, but a critical signal that physically encodes implicit interaction forces and algorithmically reveals the operator's strategy for overcoming system dynamics. By predicting the master intent, our policy learns to generate a "virtual equilibrium point", effectively realizing implicit impedance control. Furthermore, by explicitly conditioning on the history of this mismatch, the model performs implicit system identification, perceiving tracking errors as external forces to close the control loop. To bridge the temporal gap caused by inference latency, we further formulate the policy as a trajectory inpainter to ensure continuous control. We validate our approach on a sensorless, low-cost bi-manual setup. Empirical results across tasks requiring contact-rich manipulation and dynamic tracking reveal a decisive gap: while standard execution-cloning fails due to the inability to overcome contact stiffness and tracking lag, our mismatch-aware approach achieves robust success. This presents a minimalist behavior cloning framework for low-cost hardware, enabling force perception and dynamic compensation without relying on explicit force sensing. Videos are available on the \href{https://xucj98.github.io/mind-the-gap-page/}{project page}.

  </details>



- **TwinRL-VLA: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation**  
  Qinwen Xu, Jiaming Liu, Rui Zhou, Shaojun Shi, Nuowei Han, Zhuoyang Liu, Chenyang Gu, Shuo Gu, Yang Yue, Gao Huang, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.09023v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Despite strong generalization capabilities, Vision-Language-Action (VLA) models remain constrained by the high cost of expert demonstrations and insufficient real-world interaction. While online reinforcement learning (RL) has shown promise in improving general foundation models, applying RL to VLA manipulation in real-world settings is still hindered by low exploration efficiency and a restricted exploration space. Through systematic real-world experiments, we observe that the effective exploration space of online RL is closely tied to the data distribution of supervised fine-tuning (SFT). Motivated by this observation, we propose TwinRL, a digital twin-real-world collaborative RL framework designed to scale and guide exploration for VLA models. First, a high-fidelity digital twin is efficiently reconstructed from smartphone-captured scenes, enabling realistic bidirectional transfer between real and simulated environments. During the SFT warm-up stage, we introduce an exploration space expansion strategy using digital twins to broaden the support of the data trajectory distribution. Building on this enhanced initialization, we propose a sim-to-real guided exploration strategy to further accelerate online RL. Specifically, TwinRL performs efficient and parallel online RL in the digital twin prior to deployment, effectively bridging the gap between offline and online training stages. Subsequently, we exploit efficient digital twin sampling to identify failure-prone yet informative configurations, which are used to guide targeted human-in-the-loop rollouts on the real robot. In our experiments, TwinRL approaches 100% success in both in-distribution regions covered by real-world demonstrations and out-of-distribution regions, delivering at least a 30% speedup over prior real-world RL methods and requiring only about 20 minutes on average across four tasks.

  </details>



- **Contact-Anchored Policies: Contact Conditioning Creates Strong Robot Utility Models**  
  Zichen Jeff Cui, Omar Rayyan, Haritheja Etukuru, Bowen Tan, Zavier Andrianarivo, Zicheng Teng, Yihang Zhou, Krish Mehta, Nicholas Wojno, Kevin Yuanbo Wu, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.09017v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The prevalent paradigm in robot learning attempts to generalize across environments, embodiments, and tasks with language prompts at runtime. A fundamental tension limits this approach: language is often too abstract to guide the concrete physical understanding required for robust manipulation. In this work, we introduce Contact-Anchored Policies (CAP), which replace language conditioning with points of physical contact in space. Simultaneously, we structure CAP as a library of modular utility models rather than a monolithic generalist policy. This factorization allows us to implement a real-to-sim iteration cycle: we build EgoGym, a lightweight simulation benchmark, to rapidly identify failure modes and refine our models and datasets prior to real-world deployment. We show that by conditioning on contact and iterating via simulation, CAP generalizes to novel environments and embodiments out of the box on three fundamental manipulation skills while using only 23 hours of demonstration data, and outperforms large, state-of-the-art VLAs in zero-shot evaluations by 56%. All model checkpoints, codebase, hardware, simulation, and datasets will be open-sourced. Project page: https://cap-policy.github.io/

  </details>



- **Constrained Sampling to Guide Universal Manipulation RL**  
  Marc Toussaint, Cornelius V. Braun, Eckart Cobo-Briesewitz, Sayantan Auddy, Armand Jordana, Justin Carpentier  
  _2026-02-09_ · https://arxiv.org/abs/2602.08557v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We consider how model-based solvers can be leveraged to guide training of a universal policy to control from any feasible start state to any feasible goal in a contact-rich manipulation setting. While Reinforcement Learning (RL) has demonstrated its strength in such settings, it may struggle to sufficiently explore and discover complex manipulation strategies, especially in sparse-reward settings. Our approach is based on the idea of a lower-dimensional manifold of feasible, likely-visited states during such manipulation and to guide RL with a sampler from this manifold. We propose Sample-Guided RL, which uses model-based constraint solvers to efficiently sample feasible configurations (satisfying differentiable collision, contact, and force constraints) and leverage them to guide RL for universal (goal-conditioned) manipulation policies. We study using this data directly to bias state visitation, as well as using black-box optimization of open-loop trajectories between random configurations to impose a state bias and optionally add a behavior cloning loss. In a minimalistic double sphere manipulation setting, Sample-Guided RL discovers complex manipulation strategies and achieves high success rates in reaching any statically stable state. In a more challenging panda arm setting, our approach achieves a significant success rate over a near-zero baseline, and demonstrates a breadth of complex whole-body-contact manipulation strategies.

  </details>


