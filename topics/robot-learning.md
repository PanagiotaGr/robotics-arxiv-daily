# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-04 07:06 UTC_

Total papers shown: **12**


---

- **QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization**  
  Yuhao Xu, Yantai Yang, Zhenyang Fan, Yufan Liu, Yuming Li, Bing Li, Zhipeng Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03782v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The advent of Vision-Language-Action (VLA) models represents a significant leap for embodied intelligence, yet their immense computational demands critically hinder deployment on resource-constrained robotic platforms. Intuitively, low-bit quantization is a prevalent and preferred technique for large-scale model compression. However, we find that a systematic analysis of VLA model's quantization is fundamentally lacking. We argue that naively applying uniform-bit quantization from Large Language Models (LLMs) to robotics is flawed, as these methods prioritize passive data fidelity while ignoring how minor action deviations compound into catastrophic task failures. To bridge this gap, we introduce QVLA, the first action-centric quantization framework specifically designed for embodied control. In a sharp departure from the rigid, uniform-bit quantization of LLM-based methods, QVLA introduces a highly granular, channel-wise bit allocation strategy. Its core mechanism is to directly measure the final action-space sensitivity when quantizing each individual channel to various bit-widths. This process yields a precise, per-channel importance metric that guides a global optimization, which elegantly unifies quantization and pruning (0-bit) into a single, cohesive framework. Extensive evaluations on different baselines demonstrate the superiority of our approach. In the LIBERO, the quantization version of OpenVLA-OFT with our method requires only 29.2% of the original model's VRAM while maintaining 98.9% of its original performance and achieving a 1.49x speedup. This translates to a 22.6% performance improvement over the LLM-derived method SmoothQuant. Our work establishes a new, principled foundation for compressing VLA models in robotics, paving the way for deploying powerful, large-scale models on real-world hardware. Code will be released.

  </details>



- **FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation**  
  Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Mingjie Zhan, Hongsheng Li  
  _2026-02-03_ · https://arxiv.org/abs/2602.03798v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Assisting non-expert users to develop complex interactive websites has become a popular task for LLM-powered code agents. However, existing code agents tend to only generate frontend web pages, masking the lack of real full-stack data processing and storage with fancy visual effects. Notably, constructing production-level full-stack web applications is far more challenging than only generating frontend web pages, demanding careful control of data flow, comprehensive understanding of constantly updating packages and dependencies, and accurate localization of obscure bugs in the codebase. To address these difficulties, we introduce FullStack-Agent, a unified agent system for full-stack agentic coding that consists of three parts: (1) FullStack-Dev, a multi-agent framework with strong planning, code editing, codebase navigation, and bug localization abilities. (2) FullStack-Learn, an innovative data-scaling and self-improving method that back-translates crawled and synthesized website repositories to improve the backbone LLM of FullStack-Dev. (3) FullStack-Bench, a comprehensive benchmark that systematically tests the frontend, backend and database functionalities of the generated website. Our FullStack-Dev outperforms the previous state-of-the-art method by 8.7%, 38.2%, and 15.9% on the frontend, backend, and database test cases respectively. Additionally, FullStack-Learn raises the performance of a 30B model by 9.7%, 9.5%, and 2.8% on the three sets of test cases through self-improvement, demonstrating the effectiveness of our approach. The code is released at https://github.com/mnluzimu/FullStack-Agent.

  </details>



- **Enhancing Navigation Efficiency of Quadruped Robots via Leveraging Personal Transportation Platforms**  
  Minsung Yoon, Sung-Eui Yoon  
  _2026-02-03_ · https://arxiv.org/abs/2602.03397v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Quadruped robots face limitations in long-range navigation efficiency due to their reliance on legs. To ameliorate the limitations, we introduce a Reinforcement Learning-based Active Transporter Riding method (\textit{RL-ATR}), inspired by humans' utilization of personal transporters, including Segways. The \textit{RL-ATR} features a transporter riding policy and two state estimators. The policy devises adequate maneuvering strategies according to transporter-specific control dynamics, while the estimators resolve sensor ambiguities in non-inertial frames by inferring unobservable robot and transporter states. Comprehensive evaluations in simulation validate proficient command tracking abilities across various transporter-robot models and reduced energy consumption compared to legged locomotion. Moreover, we conduct ablation studies to quantify individual component contributions within the \textit{RL-ATR}. This riding ability could broaden the locomotion modalities of quadruped robots, potentially expanding the operational range and efficiency.

  </details>



- **MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction**  
  Jung Min Lee, Dohyeok Lee, Seokhun Ju, Taehyun Cho, Jin Woo Koo, Li Zhao, Sangwoo Hong, Jungwoo Lee  
  _2026-02-03_ · https://arxiv.org/abs/2602.03668v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning \emph{latent actions} from diverse human videos enables scaling robot learning beyond embodiment-specific robot datasets, and these latent actions have recently been used as pseudo-action labels for vision-language-action (VLA) model pretraining. To make VLA pretraining effective, latent actions should contain information about the underlying agent's actions despite the absence of ground-truth labels. We propose \textbf{M}ulti-\textbf{V}iew\textbf{P}oint \textbf{L}atent \textbf{A}ction \textbf{M}odel (\textbf{MVP-LAM}), which learns discrete latent actions that are highly informative about ground-truth actions from time-synchronized multi-view videos. MVP-LAM trains latent actions with a \emph{cross-viewpoint reconstruction} objective, so that a latent action inferred from one view must explain the future in another view, reducing reliance on viewpoint-specific cues. On Bridge V2, MVP-LAM produces more action-centric latent actions, achieving higher mutual information with ground-truth actions and improved action prediction, including under out-of-distribution evaluation. Finally, pretraining VLAs with MVP-LAM latent actions improves downstream manipulation performance on the SIMPLER and LIBERO-Long benchmarks.

  </details>



- **Can LLMs Do Rocket Science? Exploring the Limits of Complex Reasoning with GTOC 12**  
  Iñaki del Campo, Pablo Cuervo, Victor Rodriguez-Fernandez, Roberto Armellin, Jack Yarndley  
  _2026-02-03_ · https://arxiv.org/abs/2602.03630v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have demonstrated remarkable proficiency in code generation and general reasoning, yet their capacity for autonomous multi-stage planning in high-dimensional, physically constrained environments remains an open research question. This study investigates the limits of current AI agents by evaluating them against the 12th Global Trajectory Optimization Competition (GTOC 12), a complex astrodynamics challenge requiring the design of a large-scale asteroid mining campaign. We adapt the MLE-Bench framework to the domain of orbital mechanics and deploy an AIDE-based agent architecture to autonomously generate and refine mission solutions. To assess performance beyond binary validity, we employ an "LLM-as-a-Judge" methodology, utilizing a rubric developed by domain experts to evaluate strategic viability across five structural categories. A comparative analysis of models, ranging from GPT-4-Turbo to reasoning-enhanced architectures like Gemini 2.5 Pro, and o3, reveals a significant trend: the average strategic viability score has nearly doubled in the last two years (rising from 9.3 to 17.2 out of 26). However, we identify a critical capability gap between strategy and execution. While advanced models demonstrate sophisticated conceptual understanding, correctly framing objective functions and mission architectures, they consistently fail at implementation due to physical unit inconsistencies, boundary condition errors, and inefficient debugging loops. We conclude that, while current LLMs often demonstrate sufficient knowledge and intelligence to tackle space science tasks, they remain limited by an implementation barrier, functioning as powerful domain facilitators rather than fully autonomous engineers.

  </details>



- **CMR: Contractive Mapping Embeddings for Robust Humanoid Locomotion on Unstructured Terrains**  
  Qixin Zeng, Hongyin Zhang, Shangke Lyu, Junxi Jin, Donglin Wang, Chao Huang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03511v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robust disturbance rejection remains a longstanding challenge in humanoid locomotion, particularly on unstructured terrains where sensing is unreliable and model mismatch is pronounced. While perception information, such as height map, enhances terrain awareness, sensor noise and sim-to-real gaps can destabilize policies in practice. In this work, we provide theoretical analysis that bounds the return gap under observation noise, when the induced latent dynamics are contractive. Furthermore, we present Contractive Mapping for Robustness (CMR) framework that maps high-dimensional, disturbance-prone observations into a latent space, where local perturbations are attenuated over time. Specifically, this approach couples contrastive representation learning with Lipschitz regularization to preserve task-relevant geometry while explicitly controlling sensitivity. Notably, the formulation can be incorporated into modern deep reinforcement learning pipelines as an auxiliary loss term with minimal additional technical effort required. Further, our extensive humanoid experiments show that CMR potently outperforms other locomotion algorithms under increased noise.

  </details>



- **PLATE: Plasticity-Tunable Efficient Adapters for Geometry-Aware Continual Learning**  
  Romain Cosentino  
  _2026-02-03_ · https://arxiv.org/abs/2602.03846v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We develop a continual learning method for pretrained models that \emph{requires no access to old-task data}, addressing a practical barrier in foundation model adaptation where pretraining distributions are often unavailable. Our key observation is that pretrained networks exhibit substantial \emph{geometric redundancy}, and that this redundancy can be exploited in two complementary ways. First, redundant neurons provide a proxy for dominant pretraining-era feature directions, enabling the construction of approximately protected update subspaces directly from pretrained weights. Second, redundancy offers a natural bias for \emph{where} to place plasticity: by restricting updates to a subset of redundant neurons and constraining the remaining degrees of freedom, we obtain update families with reduced functional drift on the old-data distribution and improved worst-case retention guarantees. These insights lead to \textsc{PLATE} (\textbf{Pla}sticity-\textbf{T}unable \textbf{E}fficient Adapters), a continual learning method requiring no past-task data that provides explicit control over the plasticity-retention trade-off. PLATE parameterizes each layer with a structured low-rank update $ΔW = B A Q^\top$, where $B$ and $Q$ are computed once from pretrained weights and kept frozen, and only $A$ is trained on the new task. The code is available at https://github.com/SalesforceAIResearch/PLATE.

  </details>



- **Digital-Twin Empowered Deep Reinforcement Learning For Site-Specific Radio Resource Management in NextG Wireless Aerial Corridor**  
  Pulok Tarafder, Zoheb Hassan, Imtiaz Ahmed, Danda B. Rawat, Kamrul Hasan, Cong Pu  
  _2026-02-03_ · https://arxiv.org/abs/2602.03801v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Joint base station (BS) association and beam selection in multi-UAV aerial corridors constitutes a challenging radio resource management (RRM) problem. It is driven by high-dimensional action spaces, need for substantial overhead to acquire global channel state information (CSI), rapidly varying propagation channels, and stringent latency requirements. Conventional combinatorial optimization methods, while near-optimal, are computationally prohibitive for real-time operation in such dynamic environments. While learning-based approaches can mitigate computational complexity and CSI overhead, the need for extensive site-specific (SS) datasets for model training remains a key challenge. To address these challenges, we develop a Digital Twin (DT)-enabled two-stage optimization framework that couples physics-based beam gain modeling with DRL for scalable online decision-making. In the first stage, a channel twin (CT) is constructed using a high-fidelity ray-tracing solver with geo-spatial contexts, and network information to capture SS propagation characteristics, and dual annealing algorithm is employed to precompute optimal transmission beam directions. In the second stage, a Multi-Head Proximal Policy Optimization (MH-PPO) agent, equipped with a scalable multi-head actor-critic architecture, is trained on the DT-generated channel dataset to directly map complex channel and beam states to jointly execute UAV-BS-beam association decisions. The proposed PPO agent achieves a 44%-121% improvement over DQN and 249%-807% gain over traditional heuristic based optimization schemes in a dense UAV scenario, while reducing inference latency by several orders of magnitude. These results demonstrate that DT-driven training pipelines can deliver high-performance, low-latency RRM policies tailored to SS deployments suitable for real-time resource management in next-generation aerial corridor networks.

  </details>



- **Cognitively Diverse Multiple-Choice Question Generation: A Hybrid Multi-Agent Framework with Large Language Models**  
  Yu Tian, Linh Huynh, Katerina Christhilf, Shubham Chakraborty, Micah Watanabe, Tracy Arner, Danielle McNamara  
  _2026-02-03_ · https://arxiv.org/abs/2602.03704v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Recent advances in large language models (LLMs) have made automated multiple-choice question (MCQ) generation increasingly feasible; however, reliably producing items that satisfy controlled cognitive demands remains a challenge. To address this gap, we introduce ReQUESTA, a hybrid, multi-agent framework for generating cognitively diverse MCQs that systematically target text-based, inferential, and main idea comprehension. ReQUESTA decomposes MCQ authoring into specialized subtasks and coordinates LLM-powered agents with rule-based components to support planning, controlled generation, iterative evaluation, and post-processing. We evaluated the framework in a large-scale reading comprehension study using academic expository texts, comparing ReQUESTA-generated MCQs with those produced by a single-pass GPT-5 zero-shot baseline. Psychometric analyses of learner responses assessed item difficulty and discrimination, while expert raters evaluated question quality across multiple dimensions, including topic relevance and distractor quality. Results showed that ReQUESTA-generated items were consistently more challenging, more discriminative, and more strongly aligned with overall reading comprehension performance. Expert evaluations further indicated stronger alignment with central concepts and superior distractor linguistic consistency and semantic plausibility, particularly for inferential questions. These findings demonstrate that hybrid, agentic orchestration can systematically improve the reliability and controllability of LLM-based generation, highlighting workflow design as a key lever for structured artifact generation beyond single-pass prompting.

  </details>



- **Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems**  
  Haibo Jin, Kuang Peng, Ye Yu, Xiaopeng Yuan, Haohan Wang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03695v1 · `cs.MA`  
  <details><summary>Abstract</summary>

  While existing multi-agent systems (MAS) can handle complex problems by enabling collaboration among multiple agents, they are often highly task-specific, relying on manually crafted agent roles and interaction prompts, which leads to increased architectural complexity and limited reusability across tasks. Moreover, most MAS communicate primarily through natural language, making them vulnerable to error accumulation and instability in long-context, multi-stage interactions within internal agent histories. In this work, we propose \textbf{Agent Primitives}, a set of reusable latent building blocks for LLM-based MAS. Inspired by neural network design, where complex models are built from reusable components, we observe that many existing MAS architectures can be decomposed into a small number of recurring internal computation patterns. Based on this observation, we instantiate three primitives: Review, Voting and Selection, and Planning and Execution. All primitives communicate internally via key-value (KV) cache, which improves both robustness and efficiency by mitigating information degradation across multi-stage interactions. To enable automatic system construction, an Organizer agent selects and composes primitives for each query, guided by a lightweight knowledge pool of previously successful configurations, forming a primitive-based MAS. Experiments show that primitives-based MAS improve average accuracy by 12.0-16.5\% over single-agent baselines, reduce token usage and inference latency by approximately 3$\times$-4$\times$ compared to text-based MAS, while incurring only 1.3$\times$-1.6$\times$ overhead relative to single-agent inference and providing more stable performance across model backbones.

  </details>



- **When Single Answer Is Not Enough: Rethinking Single-Step Retrosynthesis Benchmarks for LLMs**  
  Bogdan Zagribelnyy, Ivan Ilin, Maksim Kuznetsov, Nikita Bondarev, Roman Schutski, Thomas MacDougall, Rim Shayakhmetov, Zulfat Miftakhutdinov, Mikolaj Mizera, Vladimir Aladinskiy, et al.  
  _2026-02-03_ · https://arxiv.org/abs/2602.03554v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Recent progress has expanded the use of large language models (LLMs) in drug discovery, including synthesis planning. However, objective evaluation of retrosynthesis performance remains limited. Existing benchmarks and metrics typically rely on published synthetic procedures and Top-K accuracy based on single ground-truth, which does not capture the open-ended nature of real-world synthesis planning. We propose a new benchmarking framework for single-step retrosynthesis that evaluates both general-purpose and chemistry-specialized LLMs using ChemCensor, a novel metric for chemical plausibility. By emphasizing plausibility over exact match, this approach better aligns with human synthesis planning practices. We also introduce CREED, a novel dataset comprising millions of ChemCensor-validated reaction records for LLM training, and use it to train a model that improves over the LLM baselines under this benchmark.

  </details>



- **On the Entropy Dynamics in Reinforcement Fine-Tuning of Large Language Models**  
  Shumin Wang, Yuexiang Xie, Wenhao Zhang, Yuchang Sun, Yanxi Chen, Yaliang Li, Yanyong Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03392v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Entropy serves as a critical metric for measuring the diversity of outputs generated by large language models (LLMs), providing valuable insights into their exploration capabilities. While recent studies increasingly focus on monitoring and adjusting entropy to better balance exploration and exploitation in reinforcement fine-tuning (RFT), a principled understanding of entropy dynamics during this process is yet to be thoroughly investigated. In this paper, we establish a theoretical framework for analyzing the entropy dynamics during the RFT process, which begins with a discriminant expression that quantifies entropy change under a single logit update. This foundation enables the derivation of a first-order expression for entropy change, which can be further extended to the update formula of Group Relative Policy Optimization (GRPO). The corollaries and insights drawn from the theoretical analysis inspire the design of entropy control methods, and also offer a unified lens for interpreting various entropy-based methods in existing studies. We provide empirical evidence to support the main conclusions of our analysis and demonstrate the effectiveness of the derived entropy-discriminator clipping methods. This study yields novel insights into RFT training dynamics, providing theoretical support and practical strategies for optimizing the exploration-exploitation balance during LLM fine-tuning.

  </details>


