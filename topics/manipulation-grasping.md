# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-03-11 07:08 UTC_

Total papers shown: **5**


---

- **From Flow to One Step: Real-Time Multi-Modal Trajectory Policies via Implicit Maximum Likelihood Estimation-based Distribution Distillation**  
  Ju Dong, Liding Zhang, Lei Zhang, Yu Fu, Kaixin Bai, Zoltan-Csaba Marton, Zhenshan Bing, Zhaopeng Chen, Alois Christian Knoll, Jianwei Zhang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09415v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Generative policies based on diffusion and flow matching achieve strong performance in robotic manipulation by modeling multi-modal human demonstrations. However, their reliance on iterative Ordinary Differential Equation (ODE) integration introduces substantial latency, limiting high-frequency closed-loop control. Recent single-step acceleration methods alleviate this overhead but often exhibit distributional collapse, producing averaged trajectories that fail to execute coherent manipulation strategies. We propose a framework that distills a Conditional Flow Matching (CFM) expert into a fast single-step student via Implicit Maximum Likelihood Estimation (IMLE). A bi-directional Chamfer distance provides a set-level objective that promotes both mode coverage and fidelity, enabling preservation of the teacher multi-modal action distribution in a single forward pass. A unified perception encoder further integrates multi-view RGB, depth, point clouds, and proprioception into a geometry-aware representation. The resulting high-frequency control supports real-time receding-horizon re-planning and improved robustness under dynamic disturbances.

  </details>



- **TiPToP: A Modular Open-Vocabulary Planning System for Robotic Manipulation**  
  William Shen, Nishanth Kumar, Sahit Chintalapudi, Jie Wang, Christopher Watson, Edward Hu, Jing Cao, Dinesh Jayaraman, Leslie Pack Kaelbling, Tomás Lozano-Pérez  
  _2026-03-10_ · https://arxiv.org/abs/2603.09971v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present TiPToP, an extensible modular system that combines pretrained vision foundation models with an existing Task and Motion Planner (TAMP) to solve multi-step manipulation tasks directly from input RGB images and natural-language instructions. Our system aims to be simple and easy-to-use: it can be installed and run on a standard DROID setup in under one hour and adapted to new embodiments with minimal effort. We evaluate TiPToP -- which requires zero robot data -- over 28 tabletop manipulation tasks in simulation and the real world and find it matches or outperforms $π_{0.5}\text{-DROID}$, a vision-language-action (VLA) model fine-tuned on 350 hours of embodiment-specific demonstrations. TiPToP's modular architecture enables us to analyze the system's failure modes at the component level. We analyze results from an evaluation of 173 trials and identify directions for improvement. We release TiPToP open-source to further research on modular manipulation systems and tighter integration between learning and planning. Project website and code: https://tiptop-robot.github.io

  </details>



- **Beyond Short-Horizon: VQ-Memory for Robust Long-Horizon Manipulation in Non-Markovian Simulation Benchmarks**  
  Wang Honghui, Jing Zhi, Ao Jicong, Song Shiji, Li Xuelong, Huang Gao, Bai Chenjia  
  _2026-03-10_ · https://arxiv.org/abs/2603.09513v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The high cost of collecting real-robot data has made robotic simulation a scalable platform for both evaluation and data generation. Yet most existing benchmarks concentrate on simple manipulation tasks such as pick-and-place, failing to capture the non-Markovian characteristics of real-world tasks and the complexity of articulated object interactions. To address this limitation, we present RuleSafe, a new articulated manipulation benchmark built upon a scalable LLM-aided simulation framework. RuleSafe features safes with diverse unlocking mechanisms, such as key locks, password locks, and logic locks, which require different multi-stage reasoning and manipulation strategies. These LLM-generated rules produce non-Markovian and long-horizon tasks that require temporal modeling and memory-based reasoning. We further propose VQ-Memory, a compact and structured temporal representation that uses vector-quantized variational autoencoders (VQ-VAEs) to encode past proprioceptive states into discrete latent tokens. This representation filters low-level noise while preserving high-level task-phase context, providing lightweight yet robust temporal cues that are compatible with existing Vision-Language-Action models (VLA). Extensive experiments on state-of-the-art VLA models and diffusion policies show that VQ-Memory consistently improves long-horizon planning, enhances generalization to unseen configurations, and enables more efficient manipulation with reduced computational cost. Project page: vqmemory.github.io

  </details>



- **Robotic Scene Cloning:Advancing Zero-Shot Robotic Scene Adaptation in Manipulation via Visual Prompt Editing**  
  Binyuan Huang, Yuqing Wen, Yucheng Zhao, Yaosi Hu, Tiancai Wang, Chang Wen Chen, Haoqiang Fan, Zhenzhong Chen  
  _2026-03-10_ · https://arxiv.org/abs/2603.09712v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Modern robots can perform a wide range of simple tasks and adapt to diverse scenarios in the well-trained environment. However, deploying pre-trained robot models in real-world user scenarios remains challenging due to their limited zero-shot capabilities, often necessitating extensive on-site data collection. To address this issue, we propose Robotic Scene Cloning (RSC), a novel method designed for scene-specific adaptation by editing existing robot operation trajectories. RSC achieves accurate and scene-consistent sample generation by leveraging a visual prompting mechanism and a carefully tuned condition injection module. Not only transferring textures but also performing moderate shape adaptations in response to the visual prompts, RSC demonstrates reliable task performance across a variety of object types. Experiments across various simulated and real-world environments demonstrate that RSC significantly enhances policy generalization in target environments.

  </details>



- **Stein Variational Ergodic Surface Coverage with SE(3) Constraints**  
  Jiayun Li, Yufeng Jin, Sangli Teng, Dejian Gong, Georgia Chalvatzaki  
  _2026-03-10_ · https://arxiv.org/abs/2603.09458v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Surface manipulation tasks require robots to generate trajectories that comprehensively cover complex 3D surfaces while maintaining precise end-effector poses. Existing ergodic trajectory optimization (TO) methods demonstrate success in coverage tasks, while struggling with point-cloud targets due to the nonconvex optimization landscapes and the inadequate handling of SE(3) constraints in sampling-as-optimization (SAO) techniques. In this work, we introduce a preconditioned SE(3) Stein Variational Gradient Descent (SVGD) approach for SAO ergodic trajectory generation. Our proposed approach comprises multiple innovations. First, we reformulate point-cloud ergodic coverage as a manifold-aware sampling problem. Second, we derive SE(3)-specific SVGD particle updates, and, third, we develop a preconditioner to accelerate TO convergence. Our sampling-based framework consistently identifies superior local optima compared to strong optimization-based and SAO baselines while preserving the SE(3) geometric structure. Experiments on a 3D point-cloud surface coverage benchmark and robotic surface drawing tasks demonstrate that our method achieves superior coverage quality with tractable computation in our setting relative to existing TO and SAO approaches, and is validated in real-world robot experiments.

  </details>


