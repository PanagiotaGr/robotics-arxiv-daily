# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-03-12 07:09 UTC_

Total papers shown: **20**


---

- **Learning Adaptive Force Control for Contact-Rich Sample Scraping with Heterogeneous Materials**  
  Cenk Cetin, Shreyas Pouli, Gabriella Pizzuto  
  _2026-03-11_ · https://arxiv.org/abs/2603.10979v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The increasing demand for accelerated scientific discovery, driven by global challenges, highlights the need for advanced AI-driven robotics. Deploying robotic chemists in human-centric labs is key for the next horizon of autonomous discovery, as complex tasks still demand the dexterity of human scientists. Robotic manipulation in this context is uniquely challenged by handling diverse chemicals (granular, powdery, or viscous liquids), under varying lab conditions. For example, humans use spatulas for scraping materials from vial walls. Automating this process is challenging because it goes beyond simple robotic insertion tasks and traditional lab automation, requiring the execution of fine-granular movements within a constrained environment (the sample vial). Our work proposes an adaptive control framework to address this, relying on a low-level Cartesian impedance controller for stable and compliant physical interaction and a high-level reinforcement learning agent that learns to dynamically adjust interaction forces at the end-effector. The agent is guided by perception feedback, which provides the material's location. We first created a task-representative simulation environment with a Franka Research 3 robot, a scraping tool, and a sample vial containing heterogeneous materials. To facilitate the learning of an adaptive policy and model diverse characteristics, the sample is modelled as a collection of spheres, where each sphere is assigned a unique dislodgement force threshold, which is procedurally generated using Perlin noise. We train an agent to autonomously learn and adapt the optimal contact wrench for a sample scraping task in simulation and then successfully transfer this policy to a real robotic setup. Our method was evaluated across five different material setups, outperforming a fixed-wrench baseline by an average of 10.9%.

  </details>



- **Learning Bimanual Cloth Manipulation with Vision-based Tactile Sensing via Single Robotic Arm**  
  Dongmyoung Lee, Wei Chen, Xiaoshuai Chen, Rui Zong, Petar Kormushev  
  _2026-03-11_ · https://arxiv.org/abs/2603.10609v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic cloth manipulation remains challenging due to the high-dimensional state space of fabrics, their deformable nature, and frequent occlusions that limit vision-based sensing. Although dual-arm systems can mitigate some of these issues, they increase hardware and control complexity. This paper presents Touch G.O.G., a compact vision-based tactile gripper and perception/control framework for single-arm bimanual cloth manipulation. The proposed framework combines three key components: (1) a novel gripper design and control strategy for in-gripper cloth sliding with a single robot arm, (2) a Vision Foundation Model-backboned Vision Transformer pipeline for cloth part classification (PC-Net) and edge pose estimation (PE-Net) using real and synthetic tactile images, and (3) an encoder-decoder synthetic data generator (SD-Net) that reduces manual annotation by producing high-fidelity tactile images. Experiments show 96% accuracy in distinguishing edges, corners, interior regions, and grasp failures, together with sub-millimeter edge localization and 4.5° orientation error. Real-world results demonstrate reliable cloth unfolding, even for crumpled fabrics, using only a single robotic arm. These results highlight Touch G.O.G. as a compact and cost-effective solution for deformable object manipulation.

  </details>



- **FG-CLTP: Fine-Grained Contrastive Language Tactile Pretraining for Robotic Manipulation**  
  Wenxuan Ma, Chaofan Zhang, Yinghao Cai, Guocai Yao, Shaowei Cui, Shuo Wang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10871v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advancements in integrating tactile sensing into vision-language-action (VLA) models have demonstrated transformative potential for robotic perception. However, existing tactile representations predominantly rely on qualitative descriptors (e.g., texture), neglecting quantitative contact states such as force magnitude, contact geometry, and principal axis orientation, which are indispensable for fine-grained manipulation. To bridge this gap, we propose FG-CLTP, a fine-grained contrastive language tactile pretraining framework. We first introduce a novel dataset comprising over 100k tactile 3D point cloud-language pairs that explicitly capture multidimensional contact states from the sensor's perspective. We then implement a discretized numerical tokenization mechanism to achieve quantitative-semantic alignment, effectively injecting explicit physical metrics into the multimodal feature space. The proposed FG-CLTP model yields a 95.9% classification accuracy and reduces the regression error (MAE) by 52.6% compared to state-of-the-art methods. Furthermore, the integration of 3D point cloud representations establishes a sensor-agnostic foundation with a minimal sim-to-real gap of 3.5%. Building upon this fine-grained representation, we develop a 3D tactile-language-action (3D-TLA) architecture driven by a flow matching policy to enable multimodal reasoning and control. Extensive experiments demonstrate that our framework significantly outperforms strong baselines in contact-rich manipulation tasks, providing a robust and generalizable foundation for tactile-language-action models.

  </details>



- **FutureVLA: Joint Visuomotor Prediction for Vision-Language-Action Model**  
  Xiaoxu Xu, Hao Li, Jinhui Ye, Yilun Chen, Jia Zeng, Xinyi Chen, Linning Xu, Dahua Lin, Weixin Li, Jiangmiao Pang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10712v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Predictive foresight is important to intelligent embodied agents. Since the motor execution of a robot is intrinsically constrained by its visual perception of environmental geometry, effectively anticipating the future requires capturing this tightly coupled visuomotor interplay. While recent vision-language-action models attempt to incorporate future guidance, they struggle with this joint modeling. Existing explicit methods divert capacity to task-irrelevant visual details, whereas implicit methods relying on sparse frame pairs disrupt temporal continuity. By heavily relying on visual reconstruction, these methods become visually dominated, entangling static scene context with dynamic action intent. We argue that effective joint visuomotor predictive modeling requires both temporal continuity and visually-conditioned supervision decoupling. To this end, we propose FutureVLA, featuring a novel Joint Visuomotor Predictive Architecture. FutureVLA is designed to extract joint visuomotor embeddings by first decoupling visual and motor information, and then jointly encoding generalized physical priors. Specifically, in the pretraining stage, we leverage heterogeneous manipulation datasets and introduce a Joint Visuomotor Gating mechanism to structurally separate visual state preservation from temporal action modeling. It allows the motor stream to focus on continuous physical dynamics while explicitly querying visual tokens for environmental constraints, yielding highly generalizable joint visuomotor embeddings. Subsequently, in the post-training stage, we employ a latent embeddings alignment strategy, enabling diverse downstream VLA models to internalize these temporal priors without modifying their inference architectures. Extensive experiments demonstrate that FutureVLA consistently improves VLA frameworks.

  </details>



- **Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents**  
  Yuanhao Li, Haozhe Wang, Geyong Min, Nektarios Georgalas, Wang Miao  
  _2026-03-11_ · https://arxiv.org/abs/2603.10564v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The integration of Generative AI models into AI-native network systems offers a transformative path toward achieving autonomous and adaptive control. However, the application of such models to continuous control tasks is impeded by intrinsic architectural limitations, including finite context windows, the lack of explicit reward signals, and the degradation of the long context. This paper posits that the key to unlocking robust continuous control is enabling agents to internalize experience by distilling it into their parameters, rather than relying on prompt-based memory. To this end, we propose a novel self-finetuning framework that enables agentic systems to learn continuously through direct interaction with the environment, bypassing the need for handcrafted rewards. Our framework implements a bi-perspective reflection mechanism that generates autonomous linguistic feedback to construct preference datasets from interaction history. A subsequent preference-based fine-tuning process distills long-horizon experiences into the model's parameters. We evaluate our approach on a dynamic Radio Access Network (RAN) slicing task, a challenging multi-objective control problem that requires the resolution of acute trade-offs between spectrum efficiency, service quality, and reconfiguration stability under volatile network conditions. Experimental results show that our framework outperforms standard Reinforcement Learning (RL) baselines and existing Large Language Model (LLM)-based agents in sample efficiency, stability, and multi-metric optimization. These findings demonstrate the potential of self-improving generative agents for continuous control tasks, paving the way for future AI-native network infrastructure.

  </details>



- **RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion**  
  Andrea Patrizi, Carlo Rizzardo, Arturo Laurenzi, Francesco Ruscelli, Luca Rossini, Nikos G. Tsagarakis  
  _2026-03-11_ · https://arxiv.org/abs/2603.10878v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We propose a contact-explicit hierarchical architecture coupling Reinforcement Learning (RL) and Model Predictive Control (MPC), where a high-level RL agent provides gait and navigation commands to a low-level locomotion MPC. This offloads the combinatorial burden of contact timing from the MPC by learning acyclic gaits through trial and error in simulation. We show that only a minimal set of rewards and limited tuning are required to obtain effective policies. We validate the architecture in simulation across robotic platforms spanning 50 kg to 120 kg and different MPC implementations, observing the emergence of acyclic gaits and timing adaptations in flat-terrain legged and hybrid locomotion, and further demonstrating extensibility to non-flat terrains. Across all platforms, we achieve zero-shot sim-to-sim transfer without domain randomization, and we further demonstrate zero-shot sim-to-real transfer without domain randomization on Centauro, our 120 kg wheeled-legged humanoid robot. We make our software framework and evaluation results publicly available at https://github.com/AndrePatri/AugMPC.

  </details>



- **Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control**  
  Matti Vahs, Joris Verhagen, Jana Tumova  
  _2026-03-11_ · https://arxiv.org/abs/2603.10572v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Partially Observable Markov Decision Processes (POMDPs) provide a principled framework for robot decision-making under uncertainty. Solving reach-avoid POMDPs, however, requires coordinating three distinct behaviors: goal reaching, safety, and active information gathering to reduce uncertainty. Existing online POMDP solvers attempt to address all three within a single belief tree search, but this unified approach struggles with the conflicting time scales inherent to these objectives. We propose a layered, certificate-based control architecture that operates directly in belief space, decoupling goal reaching, information gathering, and safety into modular components. We introduce Belief Control Lyapunov Functions (BCLFs) that formalize information gathering as a Lyapunov convergence problem in belief space, and show how they can be learned via reinforcement learning. For safety, we develop Belief Control Barrier Functions (BCBFs) that leverage conformal prediction to provide probabilistic safety guarantees over finite horizons. The resulting control synthesis reduces to lightweight quadratic programs solvable in real time, even for non-Gaussian belief representations with dimension $>10^4$. Experiments in simulation and on a space-robotics platform demonstrate real-time performance and improved safety and task success compared to state-of-the-art constrained POMDP solvers.

  </details>



- **Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models**  
  Yixiu Mao, Yun Qu, Qi Wang, Heming Zou, Xiangyang Ji  
  _2026-03-11_ · https://arxiv.org/abs/2603.10887v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) finetuning has become a key technique for enhancing the reasoning abilities of large language models (LLMs). However, its effectiveness critically depends on the selection of training data. Recent advances underscore the importance of online prompt selection methods, which typically concentrate training on partially solved or moderately challenging examples under the current policy, thereby yielding more effective model updates. While significantly accelerating RL finetuning in terms of training steps, they also incur substantial computational overhead by requiring extensive LLM rollouts over large candidate batches to identify informative samples, an expense that can outweigh the finetuning process itself. To address this challenge, this work proposes Dynamics-Predictive Sampling (DPS), which online predicts and selects informative prompts by inferring their learning dynamics prior to costly rollouts. Specifically, we introduce a new perspective by modeling each prompt's solving progress during RL finetuning as a dynamical system, where the extent of solving is represented as the state and the transition is characterized by a hidden Markov model. Using historical rollout reward signals, we perform online Bayesian inference to estimate evolving state distributions, and the inference outcome provides a predictive prior for efficient prompt selection without rollout-intensive filtering. Empirical results across diverse reasoning tasks, including mathematics, planning, and visual geometry, demonstrate that DPS substantially reduces redundant rollouts, accelerates the training process, and achieves superior reasoning performance.

  </details>



- **Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning**  
  Zhaowei Zhang, Xiaohan Liu, Xuekai Zhu, Junchao Huang, Ceyao Zhang, Zhiyuan Feng, Yaodong Yang, Xiaoyuan Yi, Xing Xie  
  _2026-03-11_ · https://arxiv.org/abs/2603.10588v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Reinforcement learning with verifiable rewards (RLVR) has achieved remarkable success in logical reasoning tasks, yet whether large language model (LLM) alignment requires fundamentally different approaches remains unclear. Given the apparent tolerance for multiple valid responses in moral reasoning, a natural hypothesis is that alignment tasks inherently require diversity-seeking distribution-matching algorithms rather than reward-maximizing policy-based methods. We conduct the first comprehensive empirical study comparing both paradigms on MoReBench. To enable stable RLVR training, we build a rubric-grounded reward pipeline by training a Qwen3-1.7B judge model. Contrary to our hypothesis, we find that distribution-matching approaches do not demonstrate significant advantages over reward-maximizing methods as expected on alignment tasks. Through semantic visualization mapping high-reward responses to semantic space, we demonstrate that moral reasoning exhibits more concentrated high-reward distributions than mathematical reasoning, where diverse solution strategies yield similarly high rewards. This counter-intuitive finding explains why mode-seeking optimization proves equally or more effective for alignment tasks. Our results suggest that alignment tasks do not inherently require diversity-preserving algorithms, and standard reward-maximizing RLVR methods can effectively transfer to moral reasoning without explicit diversity mechanisms.

  </details>



- **Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning**  
  Zichao Li, Jie Lou, Fangchen Dong, Zhiyuan Fan, Mengjie Ren, Hongyu Lin, Xianpei Han, Debing Zhang, Le Sun, Yaojie Lu, et al.  
  _2026-03-11_ · https://arxiv.org/abs/2603.10535v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning significantly enhances LLM capabilities but suffers from a critical issue: length inflation, where models adopt verbosity or inefficient reasoning to maximize rewards. Prior approaches struggle to address this challenge in a general and lossless manner, primarily because additive penalties introduce a compensatory effect that creates optimization shortcuts, while heuristic gating strategies lack generality beyond binary feedback. To bridge this gap, we present Group Relative Reward Rescaling (GR$^3$), which reframes length control as a multiplicative rescaling paradigm, effectively establishing a generalized, continuous, and reward-dependent gating mechanism. To further ensure lossless optimization, we incorporate group-relative regularization and advantage-aware calibration, which dynamically adapt length budgets to instance difficulty and preserve the advantage signal of high-quality trajectories. Empirically, across both RLHF and RLVR settings, GR$^3$~maintains training dynamics and downstream performance comparable to standard GRPO while significantly mitigating length inflation, outperforming state-of-the-art length-regularized baselines.

  </details>



- **Resource-constrained Amazons chess decision framework integrating large language models and graph attention**  
  Tianhao Qian, Zhuoxuan Li, Jinde Cao, Xinli Shi, Hanjie Liu, Leszek Rutkowski  
  _2026-03-11_ · https://arxiv.org/abs/2603.10512v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Artificial intelligence has advanced significantly through the development of intelligent game-playing systems, providing rigorous testbeds for decision-making, strategic planning, and adaptive learning. However, resource-constrained environments pose critical challenges, as conventional deep learning methods heavily rely on extensive datasets and computational resources. In this paper, we propose a lightweight hybrid framework for the Game of the Amazons, which explores the paradigm of weak-to-strong generalization by integrating the structural reasoning of graph-based learning with the generative capabilities of large language models. Specifically, we leverage a Graph Attention Autoencoder to inform a multi-step Monte Carlo Tree Search, utilize a Stochastic Graph Genetic Algorithm to optimize evaluation signals, and harness GPT-4o-mini to generate synthetic training data. Unlike traditional approaches that rely on expert demonstrations, our framework learns from noisy and imperfect supervision. We demonstrate that the Graph Attention mechanism effectively functions as a structural filter, denoising the LLM's outputs. Experiments on a 10$\times$10 Amazons board show that our hybrid approach not only achieves a 15\%--56\% improvement in decision accuracy over baselines but also significantly outperforms its teacher model (GPT-4o-mini), achieving a competitive win rate of 45.0\% at N=30 nodes and a decisive 66.5\% at only N=50 nodes. These results verify the feasibility of evolving specialized, high-performance game AI from general-purpose foundation models under stringent computational constraints.

  </details>



- **DepthCache: Depth-Guided Training-Free Visual Token Merging for Vision-Language-Action Model Inference**  
  Yuquan Li, Lianjie Ma, Han Ding, Lijun Zhu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10469v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models enable generalist robotic manipulation but suffer from high inference latency. This bottleneck stems from the massive number of visual tokens processed by large language backbones. Existing methods either prune or merge tokens uniformly, degrading the spatial reasoning essential for robotic control. We present DepthCache, a training-free framework that leverages depth as a structural prior for visual token compression. It partitions observations into depth-based regions and applies spatially differentiated merge ratios, preserving the near-field workspace while compressing the distant background. To exploit temporal redundancy, DepthCache distributes the merging process across consecutive frames, ensuring consistent representations while reducing per-step computation. A motion-adaptive pipeline further optimizes auxiliary view compression based on end-effector dynamics. The framework requires no model modification, generalizing across diverse VLA architectures. On the LIBERO benchmark, DepthCache achieves up to 1.28x inference speedup with less than 1% average success rate degradation across three VLA models (pi_0.5, OpenVLA, GR00T), whereas pruning and merging baselines incur 4--24% degradation at comparable compression. Real-world experiments on a physical manipulator demonstrate that DepthCache enables faster task throughput and more responsive closed-loop control in latency-sensitive scenarios.

  </details>



- **UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery**  
  Islam Guven, Mehmet Parlak  
  _2026-03-11_ · https://arxiv.org/abs/2603.10528v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Unmanned aerial vehicles (UAVs) are increasingly used to support time-critical medical supply delivery, providing rapid and flexible logistics during emergencies and resource shortages. However, effective deployment of UAV fleets requires coordination mechanisms capable of prioritizing medical requests, allocating limited aerial resources, and adapting delivery schedules under uncertain operational conditions. This paper presents a multi-agent reinforcement learning (MARL) framework for coordinating UAV fleets in stochastic medical delivery scenarios where requests vary in urgency, location, and delivery deadlines. The problem is formulated as a partially observable Markov decision process (POMDP) in which UAV agents maintain awareness of medical delivery demands while having limited visibility of other agents due to communication and localization constraints. The proposed framework employs Proximal Policy Optimization (PPO) as the primary learning algorithm and evaluates several variants, including asynchronous extensions, classical actor--critic methods, and architectural modifications to analyze scalability and performance trade-offs. The model is evaluated using real-world geographic data from selected clinics and hospitals extracted from the OpenStreetMap dataset. The framework provides a decision-support layer that prioritizes medical tasks, reallocates UAV resources in real time, and assists healthcare personnel in managing urgent logistics. Experimental results show that classical PPO achieves superior coordination performance compared to asynchronous and sequential learning strategies, highlighting the potential of reinforcement learning for adaptive and scalable UAV-assisted healthcare logistics.

  </details>



- **DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving**  
  Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, et al.  
  _2026-03-11_ · https://arxiv.org/abs/2603.11041v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT.

  </details>



- **Safe RLHF Beyond Expectation: Stochastic Dominance for Universal Spectral Risk Control**  
  Yaswanth Chittepu, Ativ Joshi, Rajarshi Bhattacharjee, Scott Niekum  
  _2026-03-11_ · https://arxiv.org/abs/2603.10938v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Safe Reinforcement Learning from Human Feedback (RLHF) typically enforces safety through expected cost constraints, but the expectation captures only a single statistic of the cost distribution and fails to account for distributional uncertainty, particularly under heavy tails or rare catastrophic events. This limitation is problematic when robustness and risk sensitivity are critical. Stochastic dominance offers a principled alternative by comparing entire cost distributions rather than just their averages, enabling direct control over tail risks and potential out-of-distribution failures that expectation-based constraints may overlook. In this work, we propose Risk-sensitive Alignment via Dominance (RAD), a novel alignment framework that replaces scalar expected cost constraints with First-Order Stochastic Dominance (FSD) constraints. We operationalize this constraint by comparing the target policy's cost distribution to that of a reference policy within an Optimal Transport (OT) framework, using entropic regularization and Sinkhorn iterations to obtain a differentiable and computationally efficient objective for stable end-to-end optimization. Furthermore, we introduce quantile-weighted FSD constraints and show that weighted FSD universally controls a broad class of Spectral Risk Measures (SRMs), so that improvements under weighted dominance imply guaranteed improvements in the corresponding spectral risk. This provides a principled mechanism for tuning a model's risk profile via the quantile weighting function. Empirical results demonstrate that RAD improves harmlessness over baselines while remaining competitive in helpfulness, and exhibits greater robustness on out-of-distribution harmlessness evaluations.

  </details>



- **Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment**  
  Fanqi Yu, Matteo Tiezzi, Tommaso Apicella, Cigdem Beyan, Vittorio Murino  
  _2026-03-11_ · https://arxiv.org/abs/2603.10929v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We introduce a lifelong imitation learning framework that enables continual policy refinement across sequential tasks under realistic memory and data constraints. Our approach departs from conventional experience replay by operating entirely in a multimodal latent space, where compact representations of visual, linguistic, and robot's state information are stored and reused to support future learning. To further stabilize adaptation, we introduce an incremental feature adjustment mechanism that regularizes the evolution of task embeddings through an angular margin constraint, preserving inter-task distinctiveness. Our method establishes a new state of the art in the LIBERO benchmarks, achieving 10-17 point gains in AUC and up to 65% less forgetting compared to previous leading methods. Ablation studies confirm the effectiveness of each component, showing consistent gains over alternative strategies. The code is available at: https://github.com/yfqi/lifelong_mlr_ifa.

  </details>



- **A Hybrid Knowledge-Grounded Framework for Safety and Traceability in Prescription Verification**  
  Yichi Zhu, Kan Ling, Xu Liu, Hengrun Zhang, Huiqun Yu, Guisheng Fan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10891v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Medication errors pose a significant threat to patient safety, making pharmacist verification (PV) a critical, yet heavily burdened, final safeguard. The direct application of Large Language Models (LLMs) to this zero-tolerance domain is untenable due to their inherent factual unreliability, lack of traceability, and weakness in complex reasoning. To address these challenges, we introduce PharmGraph-Auditor, a novel system designed for safe and evidence-grounded prescription auditing. The core of our system is a trustworthy Hybrid Pharmaceutical Knowledge Base (HPKB), implemented under the Virtual Knowledge Graph (VKG) paradigm. This architecture strategically unifies a relational component for set constraint satisfaction and a graph component for topological reasoning via a rigorous mapping layer. To construct this HPKB, we propose the Iterative Schema Refinement (ISR) algorithm, a framework that enables the co-evolution of both graph and relational schemas from medical texts. For auditing, we introduce the KB-grounded Chain of Verification (CoV), a new reasoning paradigm that transforms the LLM from an unreliable generator into a transparent reasoning engine. CoV decomposes the audit task into a sequence of verifiable queries against the HPKB, generating hybrid query plans to retrieve evidence from the most appropriate data store. Experimental results demonstrate robust knowledge extraction capabilities and show promises of using PharmGraph-Auditor to enable pharmacists to achieve safer and faster prescription verification.

  </details>



- **ReTabSyn: Realistic Tabular Data Synthesis via Reinforcement Learning**  
  Xiaofeng Lin, Seungbae Kim, Zhuoya Li, Zachary DeSoto, Charles Fleming, Guang Cheng  
  _2026-03-11_ · https://arxiv.org/abs/2603.10823v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  Deep generative models can help with data scarcity and privacy by producing synthetic training data, but they struggle in low-data, imbalanced tabular settings to fully learn the complex data distribution. We argue that striving for the full joint distribution could be overkill; for greater data efficiency, models should prioritize learning the conditional distribution $P(y\mid \bm{X})$, as suggested by recent theoretical analysis. Therefore, we overcome this limitation with \textbf{ReTabSyn}, a \textbf{Re}inforced \textbf{Tab}ular \textbf{Syn}thesis pipeline that provides direct feedback on feature correlation preservation during synthesizer training. This objective encourages the generator to prioritize the most useful predictive signals when training data is limited, thereby strengthening downstream model utility. We empirically fine-tune a language model-based generator using this approach, and across benchmarks with small sample sizes, class imbalance, and distribution shift, ReTabSyn consistently outperforms state-of-the-art baselines. Moreover, our approach can be readily extended to control various aspects of synthetic tabular data, such as applying expert-specified constraints on generated observations.

  </details>



- **ASTER: Attitude-aware Suspended-payload Quadrotor Traversal via Efficient Reinforcement Learning**  
  Dongcheng Cao, Jin Zhou, Shuo Li  
  _2026-03-11_ · https://arxiv.org/abs/2603.10715v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Agile maneuvering of the quadrotor cable-suspended system is significantly hindered by its non-smooth hybrid dynamics. While model-free Reinforcement Learning (RL) circumvents explicit differentiation of complex models, achieving attitude-constrained or inverted flight remains an open challenge due to the extreme reward sparsity under strict orientation requirements. This paper presents ASTER, a robust RL framework that achieves, to our knowledge, the first successful autonomous inverted flight for the cable-suspended system. We propose hybrid-dynamics-informed state seeding (HDSS), an initialization strategy that back-propagates target configurations through physics-consistent kinematic inversions across both taut and slack cable phases. HDSS enables the policy to discover aggressive maneuvers that are unreachable via standard exploration. Extensive simulations and real-world experiments demonstrate remarkable agility, precise attitude alignment, and robust zero-shot sim-to-real transfer across complex trajectories.

  </details>



- **MAVEN: A Meta-Reinforcement Learning Framework for Varying-Dynamics Expertise in Agile Quadrotor Maneuvers**  
  Jin Zhou, Dongcheng Cao, Xian Wang, Shuo Li  
  _2026-03-11_ · https://arxiv.org/abs/2603.10714v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has emerged as a powerful paradigm for achieving online agile navigation with quadrotors. Despite this success, policies trained via standard RL typically fail to generalize across significant dynamic variations, exhibiting a critical lack of adaptability. This work introduces MAVEN, a meta-RL framework that enables a single policy to achieve robust end-to-end navigation across a wide range of quadrotor dynamics. Our approach features a novel predictive context encoder, which learns to infer a latent representation of the system dynamics from interaction history. We demonstrate our method in agile waypoint traversal tasks under two challenging scenarios: large variations in quadrotor mass and severe single-rotor thrust loss. We leverage a GPU-vectorized simulator to distribute tasks across thousands of parallel environments, overcoming the long training times of meta-RL to converge in less than an hour. Through extensive experiments in both simulation and the real world, we validate that MAVEN achieves superior adaptation and agility. The policy successfully executes zero-shot sim-to-real transfer, demonstrating robust online adaptation by performing high-speed maneuvers despite mass variations of up to 66.7% and single-rotor thrust losses as severe as 70%.

  </details>


