# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-25 07:14 UTC_

Total papers shown: **9**


---

- **Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics**  
  Abdulaziz Almuzairee, Henrik I. Christensen  
  _2026-02-24_ · https://arxiv.org/abs/2602.21203v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual reinforcement learning is appealing for robotics but expensive -- off-policy methods are sample-efficient yet slow; on-policy methods parallelize well but waste samples. Recent work has shown that off-policy methods can train faster than on-policy methods in wall-clock time for state-based control. Extending this to vision remains challenging, where high-dimensional input images complicate training dynamics and introduce substantial storage and encoding overhead. To address these challenges, we introduce Squint, a visual Soft Actor Critic method that achieves faster wall-clock training than prior visual off-policy and on-policy methods. Squint achieves this via parallel simulation, a distributional critic, resolution squinting, layer normalization, a tuned update-to-data ratio, and an optimized implementation. We evaluate on the SO-101 Task Set, a new suite of eight manipulation tasks in ManiSkill3 with heavy domain randomization, and demonstrate sim-to-real transfer to a real SO-101 robot. We train policies for 15 minutes on a single RTX 3090 GPU, with most tasks converging in under 6 minutes.

  </details>



- **Robot Local Planner: A Periodic Sampling-Based Motion Planner with Minimal Waypoints for Home Environments**  
  Keisuke Takeshita, Takahiro Yamazaki, Tomohiro Ono, Takashi Yamamoto  
  _2026-02-24_ · https://arxiv.org/abs/2602.20645v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The objective of this study is to enable fast and safe manipulation tasks in home environments. Specifically, we aim to develop a system that can recognize its surroundings and identify target objects while in motion, enabling it to plan and execute actions accordingly. We propose a periodic sampling-based whole-body trajectory planning method, called the "Robot Local Planner (RLP)." This method leverages unique features of home environments to enhance computational efficiency, motion optimality, and robustness against recognition and control errors, all while ensuring safety. The RLP minimizes computation time by planning with minimal waypoints and generating safe trajectories. Furthermore, overall motion optimality is improved by periodically executing trajectory planning to select more optimal motions. This approach incorporates inverse kinematics that are robust to base position errors, further enhancing robustness. Evaluation experiments demonstrated that the RLP outperformed existing methods in terms of motion planning time, motion duration, and robustness, confirming its effectiveness in home environments. Moreover, application experiments using a tidy-up task achieved high success rates and short operation times, thereby underscoring its practical feasibility.

  </details>



- **Notes-to-Self: Scratchpad Augmented VLAs for Memory Dependent Manipulation Tasks**  
  Sanjay Haresh, Daniel Dijkman, Apratim Bhattacharyya, Roland Memisevic  
  _2026-02-24_ · https://arxiv.org/abs/2602.21013v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many dexterous manipulation tasks are non-markovian in nature, yet little attention has been paid to this fact in the recent upsurge of the vision-language-action (VLA) paradigm. Although they are successful in bringing internet-scale semantic understanding to robotics, existing VLAs are primarily "stateless" and struggle with memory-dependent long horizon tasks. In this work, we explore a way to impart both spatial and temporal memory to a VLA by incorporating a language scratchpad. The scratchpad makes it possible to memorize task-specific information, such as object positions, and it allows the model to keep track of a plan and progress towards subgoals within that plan. We evaluate this approach on a split of memory-dependent tasks from the ClevrSkills environment, on MemoryBench, as well as on a challenging real-world pick-and-place task. We show that incorporating a language scratchpad significantly improves generalization on these tasks for both non-recurrent and recurrent models.

  </details>



- **ActionReasoning: Robot Action Reasoning in 3D Space with LLM for Robotic Brick Stacking**  
  Guangming Wang, Qizhen Ying, Yixiong Jing, Olaf Wysocki, Brian Sheil  
  _2026-02-24_ · https://arxiv.org/abs/2602.21161v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Classical robotic systems typically rely on custom planners designed for constrained environments. While effective in restricted settings, these systems lack generalization capabilities, limiting the scalability of embodied AI and general-purpose robots. Recent data-driven Vision-Language-Action (VLA) approaches aim to learn policies from large-scale simulation and real-world data. However, the continuous action space of the physical world significantly exceeds the representational capacity of linguistic tokens, making it unclear if scaling data alone can yield general robotic intelligence. To address this gap, we propose ActionReasoning, an LLM-driven framework that performs explicit action reasoning to produce physics-consistent, prior-guided decisions for robotic manipulation. ActionReasoning leverages the physical priors and real-world knowledge already encoded in Large Language Models (LLMs) and structures them within a multi-agent architecture. We instantiate this framework on a tractable case study of brick stacking, where the environment states are assumed to be already accurately measured. The environmental states are then serialized and passed to a multi-agent LLM framework that generates physics-aware action plans. The experiments demonstrate that the proposed multi-agent LLM framework enables stable brick placement while shifting effort from low-level domain-specific coding to high-level tool invocation and prompting, highlighting its potential for broader generalization. This work introduces a promising approach to bridging perception and execution in robotic manipulation by integrating physical reasoning with LLMs.

  </details>



- **SOM-VQ: Topology-Aware Tokenization for Interactive Generative Models**  
  Alessandro Londei, Denise Lanzieri, Matteo Benati  
  _2026-02-24_ · https://arxiv.org/abs/2602.21133v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Vector-quantized representations enable powerful discrete generative models but lack semantic structure in token space, limiting interpretable human control. We introduce SOM-VQ, a tokenization method that combines vector quantization with Self-Organizing Maps to learn discrete codebooks with explicit low-dimensional topology. Unlike standard VQ-VAE, SOM-VQ uses topology-aware updates that preserve neighborhood structure: nearby tokens on a learned grid correspond to semantically similar states, enabling direct geometric manipulation of the latent space. We demonstrate that SOM-VQ produces more learnable token sequences in the evaluated domains while providing an explicit navigable geometry in code space. Critically, the topological organization enables intuitive human-in-the-loop control: users can steer generation by manipulating distances in token space, achieving semantic alignment without frame-level constraints. We focus on human motion generation - a domain where kinematic structure, smooth temporal continuity, and interactive use cases (choreography, rehabilitation, HCI) make topology-aware control especially natural - demonstrating controlled divergence and convergence from reference sequences through simple grid-based sampling. SOM-VQ provides a general framework for interpretable discrete representations applicable to music, gesture, and other interactive generative domains.

  </details>



- **From Perception to Action: An Interactive Benchmark for Vision Reasoning**  
  Yuhao Wu, Maojia Song, Yihuai Lan, Lei Wang, Zhiqiang Hu, Yao Xiao, Heng Zhou, Weihua Zheng, Dylan Raharja, Soujanya Poria, et al.  
  _2026-02-24_ · https://arxiv.org/abs/2602.21015v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Understanding the physical structure is essential for real-world applications such as embodied agents, interactive design, and long-horizon manipulation. Yet, prevailing Vision-Language Model (VLM) evaluations still center on structure-agnostic, single-turn setups (e.g., VQA), which fail to assess agents' ability to reason about how geometry, contact, and support relations jointly constrain what actions are possible in a dynamic environment. To address this gap, we introduce the Causal Hierarchy of Actions and Interactions (CHAIN) benchmark, an interactive 3D, physics-driven testbed designed to evaluate whether models can understand, plan, and execute structured action sequences grounded in physical constraints. CHAIN shifts evaluation from passive perception to active problem solving, spanning tasks such as interlocking mechanical puzzles and 3D stacking and packing. We conduct a comprehensive study of state-of-the-art VLMs and diffusion-based models under unified interactive settings. Our results show that top-performing models still struggle to internalize physical structure and causal constraints, often failing to produce reliable long-horizon plans and cannot robustly translate perceived structure into effective actions. The project is available at https://social-ai-studio.github.io/CHAIN/.

  </details>



- **See and Fix the Flaws: Enabling VLMs and Diffusion Models to Comprehend Visual Artifacts via Agentic Data Synthesis**  
  Jaehyun Park, Minyoung Ahn, Minkyu Kim, Jonghyun Lee, Jae-Gil Lee, Dongmin Park  
  _2026-02-24_ · https://arxiv.org/abs/2602.20951v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Despite recent advances in diffusion models, AI generated images still often contain visual artifacts that compromise realism. Although more thorough pre-training and bigger models might reduce artifacts, there is no assurance that they can be completely eliminated, which makes artifact mitigation a highly crucial area of study. Previous artifact-aware methodologies depend on human-labeled artifact datasets, which are costly and difficult to scale, underscoring the need for an automated approach to reliably acquire artifact-annotated datasets. In this paper, we propose ArtiAgent, which efficiently creates pairs of real and artifact-injected images. It comprises three agents: a perception agent that recognizes and grounds entities and subentities from real images, a synthesis agent that introduces artifacts via artifact injection tools through novel patch-wise embedding manipulation within a diffusion transformer, and a curation agent that filters the synthesized artifacts and generates both local and global explanations for each instance. Using ArtiAgent, we synthesize 100K images with rich artifact annotations and demonstrate both efficacy and versatility across diverse applications. Code is available at link.

  </details>



- **IG-RFT: An Interaction-Guided RL Framework for VLA Models in Long-Horizon Robotic Manipulation**  
  Zhian Su, Weijie Kong, Haonan Dong, Huixu Dong  
  _2026-02-24_ · https://arxiv.org/abs/2602.20715v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have demonstrated significant potential for generalist robotic policies; however, they struggle to generalize to long-horizon complex tasks in novel real-world domains due to distribution shifts and the scarcity of high-quality demonstrations. Although reinforcement learning (RL) offers a promising avenue for policy improvement, applying it to real-world VLA fine-tuning faces challenges regarding exploration efficiency, training stability, and sample cost. To address these issues, we propose IG-RFT, a novel Interaction-Guided Reinforced Fine-Tuning system designed for flow-based VLA models. Firstly, to facilitate effective policy optimization, we introduce Interaction-Guided Advantage Weighted Regression (IG-AWR), an RL algorithm that dynamically modulates exploration intensity based on the robot's interaction status. Furthermore, to address the limitations of sparse or task-specific rewards, we design a novel hybrid dense reward function that integrates the trajectory-level reward and the subtask-level reward. Finally, we construct a three-stage RL system comprising SFT, Offline RL, and Human-in-the-Loop RL for fine-tuning VLA models. Extensive real-world experiments on four challenging long-horizon tasks demonstrate that IG-RFT achieves an average success rate of 85.0%, significantly outperforming SFT (18.8%) and standard Offline RL baselines (40.0%). Ablation studies confirm the critical contributions of IG-AWR and hybrid reward shaping. In summary, our work establishes and validates a novel reinforced fine-tuning system for VLA models in real-world robotic manipulation.

  </details>



- **Recursive Belief Vision Language Model**  
  Vaidehi Bagaria, Bijo Sebastian, Nirav Patel  
  _2026-02-24_ · https://arxiv.org/abs/2602.20659v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Current vision-language-action (VLA) models struggle with long-horizon manipulation under partial observability. Most existing approaches remain observation-driven, relying on short context windows or repeated queries to vision-language models (VLMs). This leads to loss of task progress, action repetition under perceptual aliasing, and high inference latency. Semantic reasoning alone is not the primary bottleneck in long-horizon manipulation. Instead, VLAs lack persistent, action-conditioned state representations and exhibit limited temporal and physical reasoning, making them ill-suited for multi-stage control. This paper introduces RB-VLA, a belief-centric architecture trained with self-supervised world-model objectives that maintains a compact latent state encoding task-relevant history, dynamics, and object interactions. Queried once for high-level intent, the VLM provides task specification, while the belief tracks task progress and enables phase-aware, causally grounded control under partial observability without storing raw observations or scaling memory with time. The belief and intent jointly condition a diffusion policy for robust closed-loop execution. RB-VLA outperforms prior VLAs on long-horizon benchmarks, achieving 52.5% and 37.5% higher success on multi-stage pick-and-place and stacking tasks, respectively, compared to π0. It also reduces inference latency by up to 5x relative to baselines and eliminates memory growth across timesteps observed in existing VLAs. Ablations show that the belief module is the primary driver of performance, increasing success rates from 32.5% to 77.5%. These results demonstrate the effectiveness of belief-based state representations for long-horizon VLA policies.

  </details>


