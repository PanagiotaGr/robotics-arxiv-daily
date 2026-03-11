# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-03-11 07:08 UTC_

Total papers shown: **16**


---

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



- **EvoDriveVLA: Evolving Autonomous Driving Vision-Language-Action Model via Collaborative Perception-Planning Distillation**  
  Jiajun Cao, Xiaoan Zhang, Xiaobao Wei, Liyuqiu Huang, Wang Zijian, Hanzhen Zhang, Zhengyu Jia, Wei Mao, Hao Wang, Xianming Liu, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09465v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language-Action models have shown great promise for autonomous driving, yet they suffer from degraded perception after unfreezing the visual encoder and struggle with accumulated instability in long-term planning. To address these challenges, we propose EvoDriveVLA-a novel collaborative perception-planning distillation framework that integrates self-anchored perceptual constraints and oracle-guided trajectory optimization. Specifically, self-anchored visual distillation leverages self-anchor teacher to deliver visual anchoring constraints, regularizing student representations via trajectory-guided key-region awareness. In parallel, oracle-guided trajectory distillation employs a future-aware oracle teacher with coarse-to-fine trajectory refinement and Monte Carlo dropout sampling to produce high-quality trajectory candidates, thereby selecting the optimal trajectory to guide the student's prediction. EvoDriveVLA achieves SOTA performance in open-loop evaluation and significantly enhances performance in closed-loop evaluation. Our code is available at: https://github.com/hey-cjj/EvoDriveVLA.

  </details>



- **SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments**  
  Shiyi Chen, Mingye Yang, Haiyan Mao, Jiaqi Zhang, Haiyi Liu, Shuheng He, Debing Zhang, Zihao Qiu, Chun Zhang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09460v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Efficiently training quadruped robot navigation in densely cluttered environments remains a significant challenge. Existing methods are either limited by a lack of safety and agility in simple obstacle distributions or suffer from slow locomotion in complex environments, often requiring excessively long training phases. To this end, we propose SEA-Nav (Safe, Efficient, and Agile Navigation), a reinforcement learning framework for quadruped navigation. Within diverse and dense obstacle environments, a differentiable control barrier function (CBF)-based shield constraints the navigation policy to output safe velocity commands. An adaptive collision replay mechanism and hazardous exploration rewards are introduced to increase the probability of learning from critical experiences, guiding efficient exploration and exploitation. Finally, kinematic action constraints are incorporated to ensure safe velocity commands, facilitating successful physical deployment. To the best of our knowledge, this is the first approach that achieves highly challenging quadruped navigation in the real world with minute-level training time.

  </details>



- **Influencing LLM Multi-Agent Dialogue via Policy-Parameterized Prompts**  
  Hongbo Bo, Jingyu Hu, Weiru Liu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09890v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have emerged as a new paradigm for multi-agent systems. However, existing research on the behaviour of LLM-based multi-agents relies on ad hoc prompts and lacks a principled policy perspective. Different from reinforcement learning, we investigate whether prompt-as-action can be parameterized so as to construct a lightweight policy which consists of a sequence of state-action pairs to influence conversational behaviours without training. Our framework regards prompts as actions executed by LLMs, and dynamically constructs prompts through five components based on the current state of the agent. To test the effectiveness of parameterized control, we evaluated the dialogue flow based on five indicators: responsiveness, rebuttal, evidence usage, non-repetition, and stance shift. We conduct experiments using different LLM-driven agents in two discussion scenarios related to the general public and show that prompt parameterization can influence the dialogue dynamics. This result shows that policy-parameterised prompts offer a simple and effective mechanism to influence the dialogue process, which will help the research of multi-agent systems in the direction of social simulation.

  </details>



- **World2Mind: Cognition Toolkit for Allocentric Spatial Reasoning in Foundation Models**  
  Shouwei Ruan, Bin Wang, Zhenyu Wu, Qihui Zhu, Yuxiang Zhang, Hang Su, Yubin Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09774v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Achieving robust spatial reasoning remains a fundamental challenge for current Multimodal Foundation Models (MFMs). Existing methods either overfit statistical shortcuts via 3D grounding data or remain confined to 2D visual perception, limiting both spatial reasoning accuracy and generalization in unseen scenarios. Inspired by the spatial cognitive mapping mechanisms of biological intelligence, we propose World2Mind, a training-free spatial intelligence toolkit. At its core, World2Mind leverages 3D reconstruction and instance segmentation models to construct structured spatial cognitive maps, empowering MFMs to proactively acquire targeted spatial knowledge regarding interested landmarks and routes of interest. To provide robust geometric-topological priors, World2Mind synthesizes an Allocentric-Spatial Tree (AST) that uses elliptical parameters to model the top-down layout of landmarks accurately. To mitigate the inherent inaccuracies of 3D reconstruction, we introduce a three-stage reasoning chain comprising tool invocation assessment, modality-decoupled cue collection, and geometry-semantics interwoven reasoning. Extensive experiments demonstrate that World2Mind boosts the performance of frontier models, such as GPT-5.2, by 5%~18%. Astonishingly, relying solely on the AST-structured text, purely text-only foundation models can perform complex 3D spatial reasoning, achieving performance approaching that of advanced multimodal models.

  </details>



- **MM-tau-p$^2$: Persona-Adaptive Prompting for Robust Multi-Modal Agent Evaluation in Dual-Control Settings**  
  Anupam Purwar, Aditya Choudhary  
  _2026-03-10_ · https://arxiv.org/abs/2603.09643v1 · `cs.ET`  
  <details><summary>Abstract</summary>

  Current evaluation frameworks and benchmarks for LLM powered agents focus on text chat driven agents, these frameworks do not expose the persona of user to the agent, thus operating in a user agnostic environment. Importantly, in customer experience management domain, the agent's behaviour evolves as the agent learns about user personality. With proliferation of real time TTS and multi-modal language models, LLM based agents are gradually going to become multi-modal. Towards this, we propose the MM-tau-p$^2$ benchmark with metrics for evaluating the robustness of multi-modal agents in dual control setting with and without persona adaption of user, while also taking user inputs in the planning process to resolve a user query. In particular, our work shows that even with state of-the-art frontier LLMs like GPT-5, GPT 4.1, there are additional considerations measured using metrics viz. multi-modal robustness, turn overhead while introducing multi-modality into LLM based agents. Overall, MM-tau-p$^2$ builds on our prior work FOCAL and provides a holistic way of evaluating multi-modal agents in an automated way by introducing 12 novel metrics. We also provide estimates of these metrics on the telecom and retail domains by using the LLM-as-judge approach using carefully crafted prompts with well defined rubrics for evaluating each conversation.

  </details>



- **StyleVLA: Driving Style-Aware Vision Language Action Model for Autonomous Driving**  
  Yuan Gao, Dengyuan Hua, Mattia Piccinini, Finn Rasmus Schäfer, Korbinian Moller, Lin Li, Johannes Betz  
  _2026-03-10_ · https://arxiv.org/abs/2603.09482v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision Language Models (VLMs) bridge visual perception and linguistic reasoning. In Autonomous Driving (AD), this synergy has enabled Vision Language Action (VLA) models, which translate high-level multimodal understanding into driving behaviors, typically represented as future trajectories. However, existing VLA models mainly generate generic collision-free trajectories. Beyond collision avoidance, adapting to diverse driving styles (e.g., sporty, comfortable) is essential for personalized driving. Moreover, many methods treat trajectory generation as naive token prediction, which can produce kinematically infeasible actions. To address these limitations, we present StyleVLA, a physics-informed VLA framework for generating diverse and physically plausible driving behaviors. We introduce a hybrid loss that combines a kinematic consistency constraint with a continuous regression head to improve trajectory feasibility. To train StyleVLA, built on Qwen3-VL-4B, we construct a large-scale instruction dataset with over 1.2k scenarios, 76k Bird's Eye View (BEV) samples, and 42k First Person View (FPV) samples, with ground-truth trajectories for five driving styles and natural-language instructions. Experiments show that our 4B-parameter StyleVLA significantly outperforms proprietary models (e.g., Gemini-3-Pro) and state-of-the-art VLA models. Using a composite driving score measuring success rate, physical feasibility, and style adherence, StyleVLA achieves 0.55 on BEV and 0.51 on FPV, versus 0.32 and 0.35 for Gemini-3-Pro. These results show that a specialized, physics-informed, lightweight model can surpass closed-source models on domain-specific tasks.

  </details>



- **Kinodynamic Motion Retargeting for Humanoid Locomotion via Multi-Contact Whole-Body Trajectory Optimization**  
  Xiaoyu Zhang, Steven Haener, Varun Madabushi, Maegan Tucker  
  _2026-03-10_ · https://arxiv.org/abs/2603.09956v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present the KinoDynamic Motion Retargeting (KDMR) framework, a novel approach for humanoid locomotion that models the retargeting process as a multi-contact, whole-body trajectory optimization problem. Conventional kinematics-based retargeting methods rely solely on spatial motion capture (MoCap) data, inevitably introducing physically inconsistent artifacts, such as foot sliding and ground penetration, that severely degrade the performance of downstream imitation learning policies. To bridge this gap, KDMR extends beyond pure kinematics by explicitly enforcing rigid-body dynamics and contact complementarity constraints. Further, by integrating ground reaction force (GRF) measurements alongside MoCap data, our method automatically detects heel-toe contact events to accurately replicate complex human-like contact patterns. We evaluate KDMR against the state-of-the-art baseline, GMR, across three key dimensions: 1) the dynamic feasibility and smoothness of the retargeted motions, 2) the accuracy of GRF tracking compared to raw source data, and 3) the training efficiency and final performance of downstream control policies trained via the BeyondMimic framework. Experimental results demonstrate that KDMR significantly outperforms purely kinematic methods, yielding dynamically viable reference trajectories that accelerate policy convergence and enhance overall locomotion stability. Our end-to-end pipeline will be open-sourced upon publication.

  </details>



- **When Learning Rates Go Wrong: Early Structural Signals in PPO Actor-Critic**  
  Alberto Fernández-Hernández, Cristian Pérez-Corral, Jose I. Mestre, Manuel F. Dolz, Jose Duato, Enrique S. Quintana-Ortí  
  _2026-03-10_ · https://arxiv.org/abs/2603.09950v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Deep Reinforcement Learning systems are highly sensitive to the learning rate (LR), and selecting stable and performant training runs often requires extensive hyperparameter search. In Proximal Policy Optimization (PPO) actor--critic methods, small LR values lead to slow convergence, whereas large LR values may induce instability or collapse. We analyse this phenomenon from the behavior of the hidden neurons in the network using the Overfitting-Underfitting Indicator (OUI), a metric that quantifies the balance of binary activation patterns over a fixed probe batch. We introduce an efficient batch-based formulation of OUI and derive a theoretical connection between LR and activation sign changes, clarifying how a correct evolution of the neuron's inner structure depends on the step size. Empirically, across three discrete-control environments and multiple seeds, we show that OUI measured at only 10\% of training already discriminates between LR regimes. We observe a consistent asymmetry: critic networks achieving highest return operate in an intermediate OUI band (avoiding saturation), whereas actor networks achieving highest return exhibit comparatively high OUI values. We then compare OUI-based screening rules against early return, clip-based, divergence-based, and flip-based criteria under matched recall over successful runs. In this setting, OUI provides the strongest early screening signal: OUI alone achieves the best precision at broader recall, while combining early return with OUI yields the highest precision in best-performing screening regimes, enabling aggressive pruning of unpromising runs without requiring full training.

  </details>



- **AutoAgent: Evolving Cognition and Elastic Memory Orchestration for Adaptive Agents**  
  Xiaoxing Wang, Ning Liao, Shikun Wei, Chen Tang, Feiyu Xiong  
  _2026-03-10_ · https://arxiv.org/abs/2603.09716v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Autonomous agent frameworks still struggle to reconcile long-term experiential learning with real-time, context-sensitive decision-making. In practice, this gap appears as static cognition, rigid workflow dependence, and inefficient context usage, which jointly limit adaptability in open-ended and non-stationary environments. To address these limitations, we present AutoAgent, a self-evolving multi-agent framework built on three tightly coupled components: evolving cognition, on-the-fly contextual decision-making, and elastic memory orchestration. At the core of AutoAgent, each agent maintains structured prompt-level cognition over tools, self-capabilities, peer expertise, and task knowledge. During execution, this cognition is combined with live task context to select actions from a unified space that includes tool calls, LLM-based generation, and inter-agent requests. To support efficient long-horizon reasoning, an Elastic Memory Orchestrator dynamically organizes interaction history by preserving raw records, compressing redundant trajectories, and constructing reusable episodic abstractions, thereby reducing token overhead while retaining decision-critical evidence. These components are integrated through a closed-loop cognitive evolution process that aligns intended actions with observed outcomes to continuously update cognition and expand reusable skills, without external retraining. Empirical results across retrieval-augmented reasoning, tool-augmented agent benchmarks, and embodied task environments show that AutoAgent consistently improves task success, tool-use efficiency, and collaborative robustness over static and memory-augmented baselines. Overall, AutoAgent provides a unified and practical foundation for adaptive autonomous agents that must learn from experience while making reliable context-aware decisions in dynamic environments.

  </details>



- **PRECEPT: Planning Resilience via Experience, Context Engineering & Probing Trajectories A Unified Framework for Test-Time Adaptation with Compositional Rule Learning and Pareto-Guided Prompt Evolution**  
  Arash Shahmansoori  
  _2026-03-10_ · https://arxiv.org/abs/2603.09641v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  LLM agents that store knowledge as natural language suffer steep retrieval degradation as condition count grows, often struggle to compose learned rules reliably, and typically lack explicit mechanisms to detect stale or adversarial knowledge. We introduce PRECEPT, a unified framework for test-time adaptation with three tightly coupled components: (1) deterministic exact-match rule retrieval over structured condition keys, (2) conflict-aware memory with Bayesian source reliability and threshold-based rule invalidation, and (3) COMPASS, a Pareto-guided prompt-evolution outer loop. Exact retrieval eliminates partial-match interpretation errors on the deterministic path (0% by construction, vs 94.4% under Theorem~B.6's independence model at N=10) and supports compositional stacking through a semantic tier hierarchy; conflict-aware memory resolves static--dynamic disagreements and supports drift adaptation; COMPASS evaluates prompts through the same end-to-end execution pipeline. Results (9--10 seeds): PRECEPT achieves a +41.1pp first-try advantage over Full Reflexion (d>1.9), +33.3pp compositional generalization (d=1.55), 100% $P_1$ on 2-way logistics compositions (d=2.64), +40--55pp continuous learning gains, strong eventual robustness under adversarial static knowledge (100% logistics with adversarial SK active; partial recovery on integration), +55.0pp drift recovery (d=0.95, p=0.031), and 61% fewer steps. Core comparisons are statistically significant, often at p<0.001.

  </details>



- **X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models**  
  Yueen Ma, Irwin King  
  _2026-03-10_ · https://arxiv.org/abs/2603.09632v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods are isolated, focusing on specific domains such as online SLAM, semantic enrichment, or 3DGS for unposed images. In this paper, we introduce X-GS, an extensible open framework that unifies a broad range of techniques to enable real-time 3DGS-based online SLAM enriched with semantics, bridging the gap to downstream multimodal models. At the core of X-GS is a highly efficient pipeline called X-GS-Perceiver, capable of taking unposed RGB (or optionally RGB-D) video streams as input to co-optimize geometry and poses, and distill high-dimensional semantic features from vision foundation models into the 3D Gaussians. We achieve real-time performance through a novel online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The semantic 3D Gaussians can then be utilized by vision-language models within the X-GS-Thinker component, enabling downstream tasks such as object detection, zero-shot caption generation, and potentially embodied tasks. Experimental results on real-world datasets showcase the efficacy, efficiency, and newly unlocked multimodal capabilities of the X-GS framework.

  </details>



- **GenePlan: Evolving Better Generalized PDDL Plans using Large Language Models**  
  Andrew Murray, Danial Dervovic, Alberto Pozanco, Michael Cashmore  
  _2026-03-10_ · https://arxiv.org/abs/2603.09481v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We present GenePlan (GENeralized Evolutionary Planner), a novel framework that leverages large language model (LLM) assisted evolutionary algorithms to generate domain-dependent generalized planners for classical planning tasks described in PDDL. By casting generalized planning as an optimization problem, GenePlan iteratively evolves interpretable Python planners that minimize plan length across diverse problem instances. In empirical evaluation across six existing benchmark domains and two new domains, GenePlan achieved an average SAT score of 0.91, closely matching the performance of the state-of-the-art planners (SAT score 0.93), and significantly outperforming other LLM-based baselines such as chain-of-thought (CoT) prompting (average SAT score 0.64). The generated planners solve new instances rapidly (average 0.49 seconds per task) and at low cost (average $1.82 per domain using GPT-4o).

  </details>



- **TopoOR: A Unified Topological Scene Representation for the Operating Room**  
  Tony Danjun Wang, Ka Young Kim, Tolga Birdal, Nassir Navab, Lennart Bastian  
  _2026-03-10_ · https://arxiv.org/abs/2603.09466v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Surgical Scene Graphs abstract the complexity of surgical operating rooms (OR) into a structure of entities and their relations, but existing paradigms suffer from strictly dyadic structural limitations. Frameworks that predominantly rely on pairwise message passing or tokenized sequences flatten the manifold geometry inherent to relational structures and lose structure in the process. We introduce TopoOR, a new paradigm that models multimodal operating rooms as a higher-order structure, innately preserving pairwise and group relationships. By lifting interactions between entities into higher-order topological cells, TopoOR natively models complex dynamics and multimodality present in the OR. This topological representation subsumes traditional scene graphs, thereby offering strictly greater expressivity. We also propose a higher-order attention mechanism that explicitly preserves manifold structure and modality-specific features throughout hierarchical relational attention. In this way, we circumvent combining 3D geometry, audio, and robot kinematics into a single joint latent representation, preserving the precise multimodal structure required for safety-critical reasoning, unlike existing methods. Extensive experiments demonstrate that our approach outperforms traditional graph and LLM-based baselines across sterility breach detection, robot phase prediction, and next-action anticipation

  </details>



- **Impact of Markov Decision Process Design on Sim-to-Real Reinforcement Learning**  
  Tatjana Krau, Jorge Mandlmaier, Tobias Damm, Frieder Heieck  
  _2026-03-10_ · https://arxiv.org/abs/2603.09427v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement Learning (RL) has demonstrated strong potential for industrial process control, yet policies trained in simulation often suffer from a significant sim-to-real gap when deployed on physical hardware. This work systematically analyzes how core Markov Decision Process (MDP) design choices -- state composition, target inclusion, reward formulation, termination criteria, and environment dynamics models -- affect this transfer. Using a color mixing task, we evaluate different MDP configurations and mixing dynamics across simulation and real-world experiments. We validate our findings on physical hardware, demonstrating that physics-based dynamics models achieve up to 50% real-world success under strict precision constraints where simplified models fail entirely. Our results provide practical MDP design guidelines for deploying RL in industrial process control.

  </details>


