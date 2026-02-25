# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-25 07:14 UTC_

Total papers shown: **19**


---

- **ActionReasoning: Robot Action Reasoning in 3D Space with LLM for Robotic Brick Stacking**  
  Guangming Wang, Qizhen Ying, Yixiong Jing, Olaf Wysocki, Brian Sheil  
  _2026-02-24_ · https://arxiv.org/abs/2602.21161v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Classical robotic systems typically rely on custom planners designed for constrained environments. While effective in restricted settings, these systems lack generalization capabilities, limiting the scalability of embodied AI and general-purpose robots. Recent data-driven Vision-Language-Action (VLA) approaches aim to learn policies from large-scale simulation and real-world data. However, the continuous action space of the physical world significantly exceeds the representational capacity of linguistic tokens, making it unclear if scaling data alone can yield general robotic intelligence. To address this gap, we propose ActionReasoning, an LLM-driven framework that performs explicit action reasoning to produce physics-consistent, prior-guided decisions for robotic manipulation. ActionReasoning leverages the physical priors and real-world knowledge already encoded in Large Language Models (LLMs) and structures them within a multi-agent architecture. We instantiate this framework on a tractable case study of brick stacking, where the environment states are assumed to be already accurately measured. The environmental states are then serialized and passed to a multi-agent LLM framework that generates physics-aware action plans. The experiments demonstrate that the proposed multi-agent LLM framework enables stable brick placement while shifting effort from low-level domain-specific coding to high-level tool invocation and prompting, highlighting its potential for broader generalization. This work introduces a promising approach to bridging perception and execution in robotic manipulation by integrating physical reasoning with LLMs.

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



- **Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics**  
  Abdulaziz Almuzairee, Henrik I. Christensen  
  _2026-02-24_ · https://arxiv.org/abs/2602.21203v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual reinforcement learning is appealing for robotics but expensive -- off-policy methods are sample-efficient yet slow; on-policy methods parallelize well but waste samples. Recent work has shown that off-policy methods can train faster than on-policy methods in wall-clock time for state-based control. Extending this to vision remains challenging, where high-dimensional input images complicate training dynamics and introduce substantial storage and encoding overhead. To address these challenges, we introduce Squint, a visual Soft Actor Critic method that achieves faster wall-clock training than prior visual off-policy and on-policy methods. Squint achieves this via parallel simulation, a distributional critic, resolution squinting, layer normalization, a tuned update-to-data ratio, and an optimized implementation. We evaluate on the SO-101 Task Set, a new suite of eight manipulation tasks in ManiSkill3 with heavy domain randomization, and demonstrate sim-to-real transfer to a real SO-101 robot. We train policies for 15 minutes on a single RTX 3090 GPU, with most tasks converging in under 6 minutes.

  </details>



- **VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving**  
  Jie Wang, Guang Li, Zhijian Huang, Chenxu Dang, Hangjun Ye, Yahong Han, Long Chen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20794v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The significance of cross-view 3D geometric modeling capabilities for autonomous driving is self-evident, yet existing Vision-Language Models (VLMs) inherently lack this capability, resulting in their mediocre performance. While some promising approaches attempt to mitigate this by constructing Q&A data for auxiliary training, they still fail to fundamentally equip VLMs with the ability to comprehensively handle diverse evaluation protocols. We thus chart a new course, advocating for the infusion of VLMs with the cross-view geometric grounding of mature 3D foundation models, closing this critical capability gap in autonomous driving. In this spirit, we propose a novel architecture, VGGDrive, which empowers Vision-language models with cross-view Geometric Grounding for autonomous Driving. Concretely, to bridge the cross-view 3D geometric features from the frozen visual 3D model with the VLM's 2D visual features, we introduce a plug-and-play Cross-View 3D Geometric Enabler (CVGE). The CVGE decouples the base VLM architecture and effectively empowers the VLM with 3D features through a hierarchical adaptive injection mechanism. Extensive experiments show that VGGDrive enhances base VLM performance across five autonomous driving benchmarks, including tasks like cross-view risk perception, motion prediction, and trajectory planning. It's our belief that mature 3D foundation models can empower autonomous driving tasks through effective integration, and we hope our initial exploration demonstrates the potential of this paradigm to the autonomous driving community.

  </details>



- **NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning**  
  Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan  
  _2026-02-24_ · https://arxiv.org/abs/2602.21172v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models are advancing autonomous driving by replacing modular pipelines with unified end-to-end architectures. However, current VLAs face two expensive requirements: (1) massive dataset collection, and (2) dense reasoning annotations. In this work, we address both challenges with \modelname (\textbf{No} \textbf{R}easoning for \textbf{D}riving). Compared to existing VLAs, \modelname achieves competitive performance while being fine-tuned on $<$60\% of the data and no reasoning annotations, resulting in 3$\times$ fewer tokens. We identify that standard Group Relative Policy Optimization (GRPO) fails to yield significant improvements when applied to policies trained on such small, reasoning-free datasets. We show that this limitation stems from difficulty bias, which disproportionately penalizes reward signals from scenarios that produce high-variance rollouts within GRPO. \modelname overcomes this by incorporating Dr.~GRPO, a recent algorithm designed to mitigate difficulty bias in LLMs. As a result, \modelname achieves competitive performance on Waymo and NAVSIM with a fraction of the training data and no reasoning overhead, enabling more efficient autonomous systems.

  </details>



- **Notes-to-Self: Scratchpad Augmented VLAs for Memory Dependent Manipulation Tasks**  
  Sanjay Haresh, Daniel Dijkman, Apratim Bhattacharyya, Roland Memisevic  
  _2026-02-24_ · https://arxiv.org/abs/2602.21013v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many dexterous manipulation tasks are non-markovian in nature, yet little attention has been paid to this fact in the recent upsurge of the vision-language-action (VLA) paradigm. Although they are successful in bringing internet-scale semantic understanding to robotics, existing VLAs are primarily "stateless" and struggle with memory-dependent long horizon tasks. In this work, we explore a way to impart both spatial and temporal memory to a VLA by incorporating a language scratchpad. The scratchpad makes it possible to memorize task-specific information, such as object positions, and it allows the model to keep track of a plan and progress towards subgoals within that plan. We evaluate this approach on a split of memory-dependent tasks from the ClevrSkills environment, on MemoryBench, as well as on a challenging real-world pick-and-place task. We show that incorporating a language scratchpad significantly improves generalization on these tasks for both non-recurrent and recurrent models.

  </details>



- **Architecting AgentOS: From Token-Level Context to Emergent System-Level Intelligence**  
  ChengYou Li, XiaoDong Liu, XiangBao Meng, XinYu Zhao  
  _2026-02-24_ · https://arxiv.org/abs/2602.20934v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The paradigm of Large Language Models is undergoing a fundamental transition from static inference engines to dynamic autonomous cognitive systems.While current research primarily focuses on scaling context windows or optimizing prompt engineering the theoretical bridge between micro scale token processing and macro scale systemic intelligence remains fragmented.This paper proposes AgentOS,a holistic conceptual framework that redefines the LLM as a "Reasoning Kernel" governed by structured operating system logic.Central to this architecture is Deep Context Management which conceptualizes the context window as an Addressable Semantic Space rather than a passive buffer.We systematically deconstruct the transition from discrete sequences to coherent cognitive states introducing mechanisms for Semantic Slicing and Temporal Alignment to mitigate cognitive drift in multi-agent orchestration.By mapping classical OS abstractions such as memory paging interrupt handling and process scheduling onto LLM native constructs, this review provides a rigorous roadmap for architecting resilient scalable and self-evolving cognitive environments.Our analysis asserts that the next frontier of AGI development lies in the architectural efficiency of system-level coordination.

  </details>



- **SoK: Agentic Skills -- Beyond Tool Use in LLM Agents**  
  Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, Guangsheng Yu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20867v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Agentic systems increasingly rely on reusable procedural capabilities, \textit{a.k.a., agentic skills}, to execute long-horizon workflows reliably. These capabilities are callable modules that package procedural knowledge with explicit applicability conditions, execution policies, termination criteria, and reusable interfaces. Unlike one-off plans or atomic tool calls, skills operate (and often do well) across tasks. This paper maps the skill layer across the full lifecycle (discovery, practice, distillation, storage, composition, evaluation, and update) and introduces two complementary taxonomies. The first is a system-level set of \textbf{seven design patterns} capturing how skills are packaged and executed in practice, from metadata-driven progressive disclosure and executable code skills to self-evolving libraries and marketplace distribution. The second is an orthogonal \textbf{representation $\times$ scope} taxonomy describing what skills \emph{are} (natural language, code, policy, hybrid) and what environments they operate over (web, OS, software engineering, robotics). We analyze the security and governance implications of skill-based agents, covering supply-chain risks, prompt injection via skill payloads, and trust-tiered execution, grounded by a case study of the ClawHavoc campaign in which nearly 1{,}200 malicious skills infiltrated a major agent marketplace, exfiltrating API keys, cryptocurrency wallets, and browser credentials at scale. We further survey deterministic evaluation approaches, anchored by recent benchmark evidence that curated skills can substantially improve agent success rates while self-generated skills may degrade them. We conclude with open challenges toward robust, verifiable, and certifiable skills for real-world autonomous agents.

  </details>



- **SparkMe: Adaptive Semi-Structured Interviewing for Qualitative Insight Discovery**  
  David Anugraha, Vishakh Padmakumar, Diyi Yang  
  _2026-02-24_ · https://arxiv.org/abs/2602.21136v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Qualitative insights from user experiences are critical for informing product and policy decisions, but collecting such data at scale is constrained by the time and availability of experts to conduct semi-structured interviews. Recent work has explored using large language models (LLMs) to automate interviewing, yet existing systems lack a principled mechanism for balancing systematic coverage of predefined topics with adaptive exploration, or the ability to pursue follow-ups, deep dives, and emergent themes that arise organically during conversation. In this work, we formulate adaptive semi-structured interviewing as an optimization problem over the interviewer's behavior. We define interview utility as a trade-off between coverage of a predefined interview topic guide, discovery of relevant emergent themes, and interview cost measured by length. Based on this formulation, we introduce SparkMe, a multi-agent LLM interviewer that performs deliberative planning via simulated conversation rollouts to select questions with high expected utility. We evaluate SparkMe through controlled experiments with LLM-based interviewees, showing that it achieves higher interview utility, improving topic guide coverage (+4.7% over the best baseline) and eliciting richer emergent insights while using fewer conversational turns than prior LLM interviewing approaches. We further validate SparkMe in a user study with 70 participants across 7 professions on the impact of AI on their workflows. Domain experts rate SparkMe as producing high-quality adaptive interviews that surface helpful profession-specific insights not captured by prior approaches. The code, datasets, and evaluation protocols for SparkMe are available as open-source at https://github.com/SALT-NLP/SparkMe.

  </details>



- **"Are You Sure?": An Empirical Study of Human Perception Vulnerability in LLM-Driven Agentic Systems**  
  Xinfeng Li, Shenyu Dai, Kelong Zheng, Yue Xiao, Gelei Deng, Wei Dong, Xiaofeng Wang  
  _2026-02-24_ · https://arxiv.org/abs/2602.21127v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Large language model (LLM) agents are rapidly becoming trusted copilots in high-stakes domains like software development and healthcare. However, this deepening trust introduces a novel attack surface: Agent-Mediated Deception (AMD), where compromised agents are weaponized against their human users. While extensive research focuses on agent-centric threats, human susceptibility to deception by a compromised agent remains unexplored. We present the first large-scale empirical study with 303 participants to measure human susceptibility to AMD. This is based on HAT-Lab (Human-Agent Trust Laboratory), a high-fidelity research platform we develop, featuring nine carefully crafted scenarios spanning everyday and professional domains (e.g., healthcare, software development, human resources). Our 10 key findings reveal significant vulnerabilities and provide future defense perspectives. Specifically, only 8.6% of participants perceive AMD attacks, while domain experts show increased susceptibility in certain scenarios. We identify six cognitive failure modes in users and find that their risk awareness often fails to translate to protective behavior. The defense analysis reveals that effective warnings should interrupt workflows with low verification costs. With experiential learning based on HAT-Lab, over 90% of users who perceive risks report increased caution against AMD. This work provides empirical evidence and a platform for human-centric agent security research.

  </details>



- **Cooperative-Competitive Team Play of Real-World Craft Robots**  
  Rui Zhao, Xihui Li, Yizheng Zhang, Yuzhen Liu, Zhong Zhang, Yufeng Zhang, Cheng Zhou, Zhengyou Zhang, Lei Han  
  _2026-02-24_ · https://arxiv.org/abs/2602.21119v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-agent deep Reinforcement Learning (RL) has made significant progress in developing intelligent game-playing agents in recent years. However, the efficient training of collective robots using multi-agent RL and the transfer of learned policies to real-world applications remain open research questions. In this work, we first develop a comprehensive robotic system, including simulation, distributed learning framework, and physical robot components. We then propose and evaluate reinforcement learning techniques designed for efficient training of cooperative and competitive policies on this platform. To address the challenges of multi-agent sim-to-real transfer, we introduce Out of Distribution State Initialization (OODSI) to mitigate the impact of the sim-to-real gap. In the experiments, OODSI improves the Sim2Real performance by 20%. We demonstrate the effectiveness of our approach through experiments with a multi-robot car competitive game and a cooperative task in real-world settings.

  </details>



- **HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG**  
  Yuqi Huang, Ning Liao, Kai Yang, Anning Hu, Shengchao Hu, Xiaoxing Wang, Junchi Yan  
  _2026-02-24_ · https://arxiv.org/abs/2602.20926v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) often struggle with inherent knowledge boundaries and hallucinations, limiting their reliability in knowledge-intensive tasks. While Retrieval-Augmented Generation (RAG) mitigates these issues, it frequently overlooks structural interdependencies essential for multi-hop reasoning. Graph-based RAG approaches attempt to bridge this gap, yet they typically face trade-offs between accuracy and efficiency due to challenges such as costly graph traversals and semantic noise in LLM-generated summaries. In this paper, we propose HyperNode Expansion and Logical Path-Guided Evidence Localization strategies for GraphRAG (HELP), a novel framework designed to balance accuracy with practical efficiency through two core strategies: 1) HyperNode Expansion, which iteratively chains knowledge triplets into coherent reasoning paths abstracted as HyperNodes to capture complex structural dependencies and ensure retrieval accuracy; and 2) Logical Path-Guided Evidence Localization, which leverages precomputed graph-text correlations to map these paths directly to the corpus for superior efficiency. HELP avoids expensive random walks and semantic distortion, preserving knowledge integrity while drastically reducing retrieval latency. Extensive experiments demonstrate that HELP achieves competitive performance across multiple simple and multi-hop QA benchmarks and up to a 28.8$\times$ speedup over leading Graph-based RAG baselines.

  </details>



- **LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding**  
  Jihao Qiu, Lingxi Xie, Xinyue Huo, Qi Tian, Qixiang Ye  
  _2026-02-24_ · https://arxiv.org/abs/2602.20913v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This paper addresses the critical and underexplored challenge of long video understanding with low computational budgets. We propose LongVideo-R1, an active, reasoning-equipped multimodal large language model (MLLM) agent designed for efficient video context navigation, avoiding the redundancy of exhaustive search. At the core of LongVideo-R1 lies a reasoning module that leverages high-level visual cues to infer the most informative video clip for subsequent processing. During inference, the agent initiates traversal from top-level visual summaries and iteratively refines its focus, immediately halting the exploration process upon acquiring sufficient knowledge to answer the query. To facilitate training, we first extract hierarchical video captions from CGBench, a video corpus with grounding annotations, and guide GPT-5 to generate 33K high-quality chain-of-thought-with-tool trajectories. The LongVideo-R1 agent is fine-tuned upon the Qwen-3-8B model through a two-stage paradigm: supervised fine-tuning (SFT) followed by reinforcement learning (RL), where RL employs a specifically designed reward function to maximize selective and efficient clip navigation. Experiments on multiple long video benchmarks validate the effectiveness of name, which enjoys superior tradeoff between QA accuracy and efficiency. All curated data and source code are provided in the supplementary material and will be made publicly available. Code and data are available at: https://github.com/qiujihao19/LongVideo-R1

  </details>



- **Regret-Guided Search Control for Efficient Learning in AlphaZero**  
  Yun-Jui Tsai, Wei-Yu Chen, Yan-Ru Ju, Yu-Hung Chang, Ti-Rong Wu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20809v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) agents achieve remarkable performance but remain far less learning-efficient than humans. While RL agents require extensive self-play games to extract useful signals, humans often need only a few games, improving rapidly by repeatedly revisiting states where mistakes occurred. This idea, known as search control, aims to restart from valuable states rather than always from the initial state. In AlphaZero, prior work Go-Exploit applies this idea by sampling past states from self-play or search trees, but it treats all states equally, regardless of their learning potential. We propose Regret-Guided Search Control (RGSC), which extends AlphaZero with a regret network that learns to identify high-regret states, where the agent's evaluation diverges most from the actual outcome. These states are collected from both self-play trajectories and MCTS nodes, stored in a prioritized regret buffer, and reused as new starting positions. Across 9x9 Go, 10x10 Othello, and 11x11 Hex, RGSC outperforms AlphaZero and Go-Exploit by an average of 77 and 89 Elo, respectively. When training on a well-trained 9x9 Go model, RGSC further improves the win rate against KataGo from 69.3% to 78.2%, while both baselines show no improvement. These results demonstrate that RGSC provides an effective mechanism for search control, improving both efficiency and robustness of AlphaZero training. Our code is available at https://rlg.iis.sinica.edu.tw/papers/rgsc.

  </details>



- **Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty**  
  Xu Wan, Chao Yang, Cheng Yang, Jie Song, Mingyang Sun  
  _2026-02-24_ · https://arxiv.org/abs/2602.20729v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Safe Reinforcement Learning (RL) is crucial for achieving high performance while ensuring safety in real-world applications. However, the complex interplay of multiple uncertainty sources in real environments poses significant challenges for interpretable risk assessment and robust decision-making. To address these challenges, we propose Fuz-RL, a fuzzy measure-guided robust framework for safe RL. Specifically, our framework develops a novel fuzzy Bellman operator for estimating robust value functions using Choquet integrals. Theoretically, we prove that solving the Fuz-RL problem (in Constrained Markov Decision Process (CMDP) form) is equivalent to solving distributionally robust safe RL problems (in robust CMDP form), effectively avoiding min-max optimization. Empirical analyses on safe-control-gym and safety-gymnasium scenarios demonstrate that Fuz-RL effectively integrates with existing safe RL baselines in a model-free manner, significantly improving both safety and control performance under various types of uncertainties in observation, action, and dynamics.

  </details>



- **Balancing Multiple Objectives in Urban Traffic Control with Reinforcement Learning from AI Feedback**  
  Chenyang Zhao, Vinny Cahill, Ivana Dusparic  
  _2026-02-24_ · https://arxiv.org/abs/2602.20728v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Reward design has been one of the central challenges for real world reinforcement learning (RL) deployment, especially in settings with multiple objectives. Preference-based RL offers an appealing alternative by learning from human preferences over pairs of behavioural outcomes. More recently, RL from AI feedback (RLAIF) has demonstrated that large language models (LLMs) can generate preference labels at scale, mitigating the reliance on human annotators. However, existing RLAIF work typically focuses only on single-objective tasks, leaving the open question of how RLAIF handles systems that involve multiple objectives. In such systems trade-offs among conflicting objectives are difficult to specify, and policies risk collapsing into optimizing for a dominant goal. In this paper, we explore the extension of the RLAIF paradigm to multi-objective self-adaptive systems. We show that multi-objective RLAIF can produce policies that yield balanced trade-offs reflecting different user priorities without laborious reward engineering. We argue that integrating RLAIF into multi-objective RL offers a scalable path toward user-aligned policy learning in domains with inherently conflicting objectives.

  </details>



- **Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning**  
  Xu Wan, Yansheng Wang, Wenqi Huang, Mingyang Sun  
  _2026-02-24_ · https://arxiv.org/abs/2602.20722v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Traditional on-policy Reinforcement Learning with Verifiable Rewards (RLVR) frameworks suffer from experience waste and reward homogeneity, which directly hinders learning efficiency on difficult samples during large language models post-training. In this paper, we introduce Batch Adaptation Policy Optimization (BAPO), an off-policy RLVR framework to improve the data efficiency in large language models post-training. It dynamically selects training batches by re-evaluating historically difficult samples and reusing high-quality ones, while holding a lower bound guarantee for policy improvement. Extensive experiments further demonstrate that BAPO achieves an average 12.5% improvement over GRPO across mathematics, planning, and visual reasoning tasks. Crucially, BAPO successfully resolves 40.7% of problems that base models consistently fail to solve.

  </details>



- **TrajGPT-R: Generating Urban Mobility Trajectory with Reinforcement Learning-Enhanced Generative Pre-trained Transformer**  
  Jiawei Wang, Chuang Yang, Jiawei Yong, Xiaohang Xu, Hongjun Wang, Noboru Koshizuka, Shintaro Fukushima, Ryosuke Shibasaki, Renhe Jiang  
  _2026-02-24_ · https://arxiv.org/abs/2602.20643v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Mobility trajectories are essential for understanding urban dynamics and enhancing urban planning, yet access to such data is frequently hindered by privacy concerns. This research introduces a transformative framework for generating large-scale urban mobility trajectories, employing a novel application of a transformer-based model pre-trained and fine-tuned through a two-phase process. Initially, trajectory generation is conceptualized as an offline reinforcement learning (RL) problem, with a significant reduction in vocabulary space achieved during tokenization. The integration of Inverse Reinforcement Learning (IRL) allows for the capture of trajectory-wise reward signals, leveraging historical data to infer individual mobility preferences. Subsequently, the pre-trained model is fine-tuned using the constructed reward model, effectively addressing the challenges inherent in traditional RL-based autoregressive methods, such as long-term credit assignment and handling of sparse reward environments. Comprehensive evaluations on multiple datasets illustrate that our framework markedly surpasses existing models in terms of reliability and diversity. Our findings not only advance the field of urban mobility modeling but also provide a robust methodology for simulating urban data, with significant implications for traffic management and urban development planning. The implementation is publicly available at https://github.com/Wangjw6/TrajGPT_R.

  </details>


