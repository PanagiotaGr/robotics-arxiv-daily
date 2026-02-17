# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-17 07:12 UTC_

Total papers shown: **15**


---

- **DriveFine: Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving**  
  Chenxu Dang, Sining Ang, Yongkang Li, Haochen Tian, Jie Wang, Guang Li, Hangjun Ye, Jie Ma, Long Chen, Yan Wang  
  _2026-02-16_ · https://arxiv.org/abs/2602.14577v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models for autonomous driving increasingly adopt generative planners trained with imitation learning followed by reinforcement learning. Diffusion-based planners suffer from modality alignment difficulties, low training efficiency, and limited generalization. Token-based planners are plagued by cumulative causal errors and irreversible decoding. In summary, the two dominant paradigms exhibit complementary strengths and weaknesses. In this paper, we propose DriveFine, a masked diffusion VLA model that combines flexible decoding with self-correction capabilities. In particular, we design a novel plug-and-play block-MoE, which seamlessly injects a refinement expert on top of the generation expert. By enabling explicit expert selection during inference and gradient blocking during training, the two experts are fully decoupled, preserving the foundational capabilities and generic patterns of the pretrained weights, which highlights the flexibility and extensibility of the block-MoE design. Furthermore, we design a hybrid reinforcement learning strategy that encourages effective exploration of refinement expert while maintaining training stability. Extensive experiments on NAVSIM v1, v2, and Navhard benchmarks demonstrate that DriveFine exhibits strong efficacy and robustness. The code will be released at https://github.com/MSunDYY/DriveFine.

  </details>



- **DM0: An Embodied-Native Vision-Language-Action Model towards Physical AI**  
  En Yu, Haoran Lv, Jianjian Sun, Kangheng Lin, Ruitao Zhang, Yukang Shi, Yuyang Chen, Ze Chen, Ziheng Zhang, Fan Jia, et al.  
  _2026-02-16_ · https://arxiv.org/abs/2602.14974v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Moving beyond the traditional paradigm of adapting internet-pretrained models to physical tasks, we present DM0, an Embodied-Native Vision-Language-Action (VLA) framework designed for Physical AI. Unlike approaches that treat physical grounding as a fine-tuning afterthought, DM0 unifies embodied manipulation and navigation by learning from heterogeneous data sources from the onset. Our methodology follows a comprehensive three-stage pipeline: Pretraining, Mid-Training, and Post-Training. First, we conduct large-scale unified pretraining on the Vision-Language Model (VLM) using diverse corpora--seamlessly integrating web text, autonomous driving scenarios, and embodied interaction logs-to jointly acquire semantic knowledge and physical priors. Subsequently, we build a flow-matching action expert atop the VLM. To reconcile high-level reasoning with low-level control, DM0 employs a hybrid training strategy: for embodied data, gradients from the action expert are not backpropagated to the VLM to preserve generalized representations, while the VLM remains trainable on non-embodied data. Furthermore, we introduce an Embodied Spatial Scaffolding strategy to construct spatial Chain-of-Thought (CoT) reasoning, effectively constraining the action solution space. Experiments on the RoboChallenge benchmark demonstrate that DM0 achieves state-of-the-art performance in both Specialist and Generalist settings on Table30.

  </details>



- **ManeuverNet: A Soft Actor-Critic Framework for Precise Maneuvering of Double-Ackermann-Steering Robots with Optimized Reward Functions**  
  Kohio Deflesselle, Mélodie Daniel, Aly Magassouba, Miguel Aranda, Olivier Ly  
  _2026-02-16_ · https://arxiv.org/abs/2602.14726v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous control of double-Ackermann-steering robots is essential in agricultural applications, where robots must execute precise and complex maneuvers within a limited space. Classical methods, such as the Timed Elastic Band (TEB) planner, can address this problem, but they rely on parameter tuning, making them highly sensitive to changes in robot configuration or environment and impractical to deploy without constant recalibration. At the same time, end-to-end deep reinforcement learning (DRL) methods often fail due to unsuitable reward functions for non-holonomic constraints, resulting in sub-optimal policies and poor generalization. To address these challenges, this paper presents ManeuverNet, a DRL framework tailored for double-Ackermann systems, combining Soft Actor-Critic with CrossQ. Furthermore, ManeuverNet introduces four specifically designed reward functions to support maneuver learning. Unlike prior work, ManeuverNet does not depend on expert data or handcrafted guidance. We extensively evaluate ManeuverNet against both state-of-the-art DRL baselines and the TEB planner. Experimental results demonstrate that our framework substantially improves maneuverability and success rates, achieving more than a 40% gain over DRL baselines. Moreover, ManeuverNet effectively mitigates the strong parameter sensitivity observed in the TEB planner. In real-world trials, ManeuverNet achieved up to a 90% increase in maneuvering trajectory efficiency, highlighting its robustness and practical applicability.

  </details>



- **MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design**  
  Gen Zhou, Sugitha Janarthanan, Lianghong Chen, Pingzhao Hu  
  _2026-02-16_ · https://arxiv.org/abs/2602.14926v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  To address the global health threat of antimicrobial resistance, antimicrobial peptides (AMP) are being explored for their potent and promising ability to fight resistant pathogens. While artificial intelligence (AI) is being employed to advance AMP discovery and design, most AMP design models struggle to balance key goals like activity, toxicity, and novelty, using rigid or unclear scoring methods that make results hard to interpret and optimize. As the capabilities of Large Language Models (LLM) advance and evolve swiftly, we turn to AI multi-agent collaboration based on such models (multi-agent LLMs), which show rapidly rising potential in complex scientific design scenarios. Based on this, we introduce MAC-AMP, a closed-loop multi-agent collaboration (MAC) system for multi-objective AMP design. The system implements a fully autonomous simulated peer review-adaptive reinforcement learning framework that requires only a task description and example dataset to design novel AMPs. The novelty of our work lies in introducing a closed-loop multi-agent system for AMP design, with cross-domain transferability, that supports multi-objective optimization while remaining explainable rather than a 'black box'. Experiments show that MAC-AMP outperforms other AMP generative models by effectively optimizing AMP generation for multiple key molecular properties, demonstrating exceptional results in antibacterial activity, AMP likeliness, toxicity compliance, and structural reliability.

  </details>



- **Arbor: A Framework for Reliable Navigation of Critical Conversation Flows**  
  Luís Silva, Diogo Gonçalves, Catarina Farinha, Clara Matos, Luís Ungaro  
  _2026-02-16_ · https://arxiv.org/abs/2602.14643v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large language models struggle to maintain strict adherence to structured workflows in high-stakes domains such as healthcare triage. Monolithic approaches that encode entire decision structures within a single prompt are prone to instruction-following degradation as prompt length increases, including lost-in-the-middle effects and context window overflow. To address this gap, we present Arbor, a framework that decomposes decision tree navigation into specialized, node-level tasks. Decision trees are standardized into an edge-list representation and stored for dynamic retrieval. At runtime, a directed acyclic graph (DAG)-based orchestration mechanism iteratively retrieves only the outgoing edges of the current node, evaluates valid transitions via a dedicated LLM call, and delegates response generation to a separate inference step. The framework is agnostic to the underlying decision logic and model provider. Evaluated against single-prompt baselines across 10 foundation models using annotated turns from real clinical triage conversations. Arbor improves mean turn accuracy by 29.4 percentage points, reduces per-turn latency by 57.1%, and achieves an average 14.4x reduction in per-turn cost. These results indicate that architectural decomposition reduces dependence on intrinsic model capability, enabling smaller models to match or exceed larger models operating under single-prompt baselines.

  </details>



- **LACONIC: Length-Aware Constrained Reinforcement Learning for LLM**  
  Chang Liu, Yiran Zhao, Lawrence Liu, Yaoqi Ye, Csaba Szepesvári, Lin F. Yang  
  _2026-02-16_ · https://arxiv.org/abs/2602.14468v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has enhanced the capabilities of large language models (LLMs) through reward-driven training. Nevertheless, this process can introduce excessively long responses, inflating inference latency and computational overhead. Prior length-control approaches typically rely on fixed heuristic reward shaping, which can misalign with the task objective and require brittle tuning. In this work, we propose LACONIC, a reinforcement learning method that enforces a target token budget during training. Specifically, we update policy models using an augmented objective that combines the task reward with a length-based cost. To balance brevity and task performance, the cost scale is adaptively adjusted throughout training. This yields robust length control while preserving task reward. We provide a theoretical guarantee that support the method. Across mathematical reasoning models and datasets, LACONIC preserves or improves pass@1 while reducing output length by over 50%. It maintains out-of-domain performance on general knowledge and multilingual benchmarks with 44% fewer tokens. Moreover, LACONIC integrates into standard RL-tuning with no inference changes and minimal deployment overhead.

  </details>



- **Decoupled Continuous-Time Reinforcement Learning via Hamiltonian Flow**  
  Minh Nguyen  
  _2026-02-16_ · https://arxiv.org/abs/2602.14587v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Many real-world control problems, ranging from finance to robotics, evolve in continuous time with non-uniform, event-driven decisions. Standard discrete-time reinforcement learning (RL), based on fixed-step Bellman updates, struggles in this setting: as time gaps shrink, the $Q$-function collapses to the value function $V$, eliminating action ranking. Existing continuous-time methods reintroduce action information via an advantage-rate function $q$. However, they enforce optimality through complicated martingale losses or orthogonality constraints, which are sensitive to the choice of test processes. These approaches entangle $V$ and $q$ into a large, complex optimization problem that is difficult to train reliably. To address these limitations, we propose a novel decoupled continuous-time actor-critic algorithm with alternating updates: $q$ is learned from diffusion generators on $V$, and $V$ is updated via a Hamiltonian-based value flow that remains informative under infinitesimal time steps, where standard max/softmax backups fail. Theoretically, we prove rigorous convergence via new probabilistic arguments, sidestepping the challenge that generator-based Hamiltonians lack Bellman-style contraction under the sup-norm. Empirically, our method outperforms prior continuous-time and leading discrete-time baselines across challenging continuous-control benchmarks and a real-world trading task, achieving 21% profit over a single quarter$-$nearly doubling the second-best method.

  </details>



- **Simulation-based Learning of Electrical Cabinet Assembly Using Robot Skills**  
  Arik Laemmle, Balázs András Bálint, Philipp Tenbrock, Frank Naegele, David Traunecker, József Váncza, Marco F. Huber  
  _2026-02-16_ · https://arxiv.org/abs/2602.14561v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a simulation-driven approach for automating the force-controlled assembly of electrical terminals on DIN-rails, a task traditionally hindered by high programming effort and product variability. The proposed method integrates deep reinforcement learning (DRL) with parameterizable robot skills in a physics-based simulation environment. To realistically model the snap-fit assembly process, we develop and evaluate two types of joining models: analytical models based on beam theory and rigid-body models implemented in the MuJoCo physics engine. These models enable accurate simulation of interaction forces, essential for training DRL agents. The robot skills are structured using the pitasc framework, allowing modular, reusable control strategies. Training is conducted in simulation using Soft Actor-Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithms. Domain randomization is applied to improve robustness. The trained policies are transferred to a physical UR10e robot system without additional tuning. Experimental results demonstrate high success rates (up to 100%) in both simulation and real-world settings, even under significant positional and rotational deviations. The system generalizes well to new terminal types and positions, significantly reducing manual programming effort. This work highlights the potential of combining simulation-based learning with modular robot skills for flexible, scalable automation in small-batch manufacturing. Future work will explore hybrid learning methods, automated environment parameterization, and further refinement of joining models for design integration.

  </details>



- **MoRL: Reinforced Reasoning for Unified Motion Understanding and Generation**  
  Hongpeng Wang, Zeyu Zhang, Wenhao Li, Hao Tang  
  _2026-02-16_ · https://arxiv.org/abs/2602.14534v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Human motion understanding and generation are crucial for vision and robotics but remain limited in reasoning capability and test-time planning. We propose MoRL, a unified multimodal motion model trained with supervised fine-tuning and reinforcement learning with verifiable rewards. Our task-specific reward design combines semantic alignment and reasoning coherence for understanding with physical plausibility and text-motion consistency for generation, improving both logical reasoning and perceptual realism. To further enhance inference, we introduce Chain-of-Motion (CoM), a test-time reasoning method that enables step-by-step planning and reflection. We also construct two large-scale CoT datasets, MoUnd-CoT-140K and MoGen-CoT-140K, to align motion sequences with reasoning traces and action descriptions. Experiments on HumanML3D and KIT-ML show that MoRL achieves significant gains over state-of-the-art baselines. Code: https://github.com/AIGeeksGroup/MoRL. Website: https://aigeeksgroup.github.io/MoRL.

  </details>



- **TWISTED-RL: Hierarchical Skilled Agents for Knot-Tying without Human Demonstrations**  
  Guy Freund, Tom Jurgenson, Matan Sudry, Erez Karpas  
  _2026-02-16_ · https://arxiv.org/abs/2602.14526v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic knot-tying represents a fundamental challenge in robotics due to the complex interactions between deformable objects and strict topological constraints. We present TWISTED-RL, a framework that improves upon the previous state-of-the-art in demonstration-free knot-tying (TWISTED), which smartly decomposed a single knot-tying problem into manageable subproblems, each addressed by a specialized agent. Our approach replaces TWISTED's single-step inverse model that was learned via supervised learning with a multi-step Reinforcement Learning policy conditioned on abstract topological actions rather than goal states. This change allows more delicate topological state transitions while avoiding costly and ineffective data collection protocols, thus enabling better generalization across diverse knot configurations. Experimental results demonstrate that TWISTED-RL manages to solve previously unattainable knots of higher complexity, including commonly used knots such as the Figure-8 and the Overhand. Furthermore, the increase in success rates and drop in planning time establishes TWISTED-RL as the new state-of-the-art in robotic knot-tying without human demonstrations.

  </details>



- **Learning Transferability: A Two-Stage Reinforcement Learning Approach for Enhancing Quadruped Robots' Performance in U-Shaped Stair Climbing**  
  Baixiao Huang, Baiyu Huang, Yu Hou  
  _2026-02-16_ · https://arxiv.org/abs/2602.14473v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Quadruped robots are employed in various scenarios in building construction. However, autonomous stair climbing across different indoor staircases remains a major challenge for robot dogs to complete building construction tasks. In this project, we employed a two-stage end-to-end deep reinforcement learning (RL) approach to optimize a robot's performance on U-shaped stairs. The training robot-dog modality, Unitree Go2, was first trained to climb stairs on Isaac Lab's pyramid-stair terrain, and then to climb a U-shaped indoor staircase using the learned policies. This project explores end-to-end RL methods that enable robot dogs to autonomously climb stairs. The results showed (1) the successful goal reached for robot dogs climbing U-shaped stairs with a stall penalty, and (2) the transferability from the policy trained on U-shaped stairs to deployment on straight, L-shaped, and spiral stair terrains, and transferability from other stair models to deployment on U-shaped terrain.

  </details>



- **EditCtrl: Disentangled Local and Global Control for Real-Time Generative Video Editing**  
  Yehonathan Litman, Shikun Liu, Dario Seyb, Nicholas Milef, Yang Zhou, Carl Marshall, Shubham Tulsiani, Caleb Leak  
  _2026-02-16_ · https://arxiv.org/abs/2602.15031v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  High-fidelity generative video editing has seen significant quality improvements by leveraging pre-trained video foundation models. However, their computational cost is a major bottleneck, as they are often designed to inefficiently process the full video context regardless of the inpainting mask's size, even for sparse, localized edits. In this paper, we introduce EditCtrl, an efficient video inpainting control framework that focuses computation only where it is needed. Our approach features a novel local video context module that operates solely on masked tokens, yielding a computational cost proportional to the edit size. This local-first generation is then guided by a lightweight temporal global context embedder that ensures video-wide context consistency with minimal overhead. Not only is EditCtrl 10 times more compute efficient than state-of-the-art generative editing methods, it even improves editing quality compared to methods designed with full-attention. Finally, we showcase how EditCtrl unlocks new capabilities, including multi-region editing with text prompts and autoregressive content propagation.

  </details>



- **BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames**  
  Max Sobol Mark, Jacky Liang, Maria Attarian, Chuyuan Fu, Debidatta Dwibedi, Dhruv Shah, Aviral Kumar  
  _2026-02-16_ · https://arxiv.org/abs/2602.15010v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many robot tasks require attending to the history of past observations. For example, finding an item in a room requires remembering which places have already been searched. However, the best-performing robot policies typically condition only on the current observation, limiting their applicability to such tasks. Naively conditioning on past observations often fails due to spurious correlations: policies latch onto incidental features of training histories that do not generalize to out-of-distribution trajectories upon deployment. We analyze why policies latch onto these spurious correlations and find that this problem stems from limited coverage over the space of possible histories during training, which grows exponentially with horizon. Existing regularization techniques provide inconsistent benefits across tasks, as they do not fundamentally address this coverage problem. Motivated by these findings, we propose Big Picture Policies (BPP), an approach that conditions on a minimal set of meaningful keyframes detected by a vision-language model. By projecting diverse rollouts onto a compact set of task-relevant events, BPP substantially reduces distribution shift between training and deployment, without sacrificing expressivity. We evaluate BPP on four challenging real-world manipulation tasks and three simulation tasks, all requiring history conditioning. BPP achieves 70% higher success rates than the best comparison on real-world evaluations.

  </details>



- **RF-GPT: Teaching AI to See the Wireless World**  
  Hang Zou, Yu Tian, Bohao Wang, Lina Bariah, Samson Lasaulce, Chongwen Huang, Mérouane Debbah  
  _2026-02-16_ · https://arxiv.org/abs/2602.14833v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) and multimodal models have become powerful general-purpose reasoning systems. However, radio-frequency (RF) signals, which underpin wireless systems, are still not natively supported by these models. Existing LLM-based approaches for telecom focus mainly on text and structured data, while conventional RF deep-learning models are built separately for specific signal-processing tasks, highlighting a clear gap between RF perception and high-level reasoning. To bridge this gap, we introduce RF-GPT, a radio-frequency language model (RFLM) that utilizes the visual encoders of multimodal LLMs to process and understand RF spectrograms. In this framework, complex in-phase/quadrature (IQ) waveforms are mapped to time-frequency spectrograms and then passed to pretrained visual encoders. The resulting representations are injected as RF tokens into a decoder-only LLM, which generates RF-grounded answers, explanations, and structured outputs. To train RF-GPT, we perform supervised instruction fine-tuning of a pretrained multimodal LLM using a fully synthetic RF corpus. Standards-compliant waveform generators produce wideband scenes for six wireless technologies, from which we derive time-frequency spectrograms, exact configuration metadata, and dense captions. A text-only LLM then converts these captions into RF-grounded instruction-answer pairs, yielding roughly 12,000 RF scenes and 0.625 million instruction examples without any manual labeling. Across benchmarks for wideband modulation classification, overlap analysis, wireless-technology recognition, WLAN user counting, and 5G NR information extraction, RF-GPT achieves strong multi-task performance, whereas general-purpose VLMs with no RF grounding largely fail.

  </details>



- **RNM-TD3: N:M Semi-structured Sparse Reinforcement Learning From Scratch**  
  Isam Vrce, Andreas Kassler, Gökçe Aydos  
  _2026-02-16_ · https://arxiv.org/abs/2602.14578v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Sparsity is a well-studied technique for compressing deep neural networks (DNNs) without compromising performance. In deep reinforcement learning (DRL), neural networks with up to 5% of their original weights can still be trained with minimal performance loss compared to their dense counterparts. However, most existing methods rely on unstructured fine-grained sparsity, which limits hardware acceleration opportunities due to irregular computation patterns. Structured coarse-grained sparsity enables hardware acceleration, yet typically degrades performance and increases pruning complexity. In this work, we present, to the best of our knowledge, the first study on N:M structured sparsity in RL, which balances compression, performance, and hardware efficiency. Our framework enforces row-wise N:M sparsity throughout training for all networks in off-policy RL (TD3), maintaining compatibility with accelerators that support N:M sparse matrix operations. Experiments on continuous-control benchmarks show that RNM-TD3, our N:M sparse agent, outperforms its dense counterpart at 50%-75% sparsity (e.g., 2:4 and 1:4), achieving up to a 14% increase in performance at 2:4 sparsity on the Ant environment. RNM-TD3 remains competitive even at 87.5% sparsity (1:8), while enabling potential training speedups.

  </details>


