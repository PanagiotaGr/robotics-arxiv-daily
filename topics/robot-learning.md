# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-01-22 06:51 UTC_

Total papers shown: **9**


---

- **FARE: Fast-Slow Agentic Robotic Exploration**  
  Shuhao Liao, Xuxin Lv, Jeric Lew, Shizhe Zhang, Jingsong Liang, Peizhuo Li, Yuhong Cao, Wenjun Wu, Guillaume Sartoretti  
  _2026-01-21_ · https://arxiv.org/abs/2601.14681v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This work advances autonomous robot exploration by integrating agent-level semantic reasoning with fast local control. We introduce FARE, a hierarchical autonomous exploration framework that integrates a large language model (LLM) for global reasoning with a reinforcement learning (RL) policy for local decision making. FARE follows a fast-slow thinking paradigm. The slow-thinking LLM module interprets a concise textual description of the unknown environment and synthesizes an agent-level exploration strategy, which is then grounded into a sequence of global waypoints through a topological graph. To further improve reasoning efficiency, this module employs a modularity-based pruning mechanism that reduces redundant graph structures. The fast-thinking RL module executes exploration by reacting to local observations while being guided by the LLM-generated global waypoints. The RL policy is additionally shaped by a reward term that encourages adherence to the global waypoints, enabling coherent and robust closed-loop behavior. This architecture decouples semantic reasoning from geometric decision, allowing each module to operate in its appropriate temporal and spatial scale. In challenging simulated environments, our results show that FARE achieves substantial improvements in exploration efficiency over state-of-the-art baselines. We further deploy FARE on hardware and validate it in complex, large scale $200m\times130m$ building environment.

  </details>



- **The Why Behind the Action: Unveiling Internal Drivers via Agentic Attribution**  
  Chen Qian, Peng Wang, Dongrui Liu, Junyao Yang, Dadi Guo, Ling Tang, Jilin Mei, Qihan Ren, Shuai Shao, Yong Liu, et al.  
  _2026-01-21_ · https://arxiv.org/abs/2601.15075v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Model (LLM)-based agents are widely used in real-world applications such as customer service, web navigation, and software engineering. As these systems become more autonomous and are deployed at scale, understanding why an agent takes a particular action becomes increasingly important for accountability and governance. However, existing research predominantly focuses on \textit{failure attribution} to localize explicit errors in unsuccessful trajectories, which is insufficient for explaining the reasoning behind agent behaviors. To bridge this gap, we propose a novel framework for \textbf{general agentic attribution}, designed to identify the internal factors driving agent actions regardless of the task outcome. Our framework operates hierarchically to manage the complexity of agent interactions. Specifically, at the \textit{component level}, we employ temporal likelihood dynamics to identify critical interaction steps; then at the \textit{sentence level}, we refine this localization using perturbation-based analysis to isolate the specific textual evidence. We validate our framework across a diverse suite of agentic scenarios, including standard tool use and subtle reliability risks like memory-induced bias. Experimental results demonstrate that the proposed framework reliably pinpoints pivotal historical events and sentences behind the agent behavior, offering a critical step toward safer and more accountable agentic systems.

  </details>



- **BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries**  
  Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, Kai Chen  
  _2026-01-21_ · https://arxiv.org/abs/2601.15197v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.

  </details>



- **TIDAL: Temporally Interleaved Diffusion and Action Loop for High-Frequency VLA Control**  
  Yuteng Sun, Haoran Wang, Ruofei Bai, Zhengguo Li, Jun Li, Meng Yee, Chuah, Wei Yun Yau  
  _2026-01-21_ · https://arxiv.org/abs/2601.14945v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Large-scale Vision-Language-Action (VLA) models offer semantic generalization but suffer from high inference latency, limiting them to low-frequency batch-and-execute paradigm. This frequency mismatch creates an execution blind spot, causing failures in dynamic environments where targets move during the open-loop execution window. We propose TIDAL (Temporally Interleaved Diffusion and Action Loop), a hierarchical framework that decouples semantic reasoning from high-frequency actuation. TIDAL operates as a backbone-agnostic module for diffusion-based VLAs, using a dual-frequency architecture to redistribute the computational budget. Specifically, a low-frequency macro-intent loop caches semantic embeddings, while a high-frequency micro-control loop interleaves single-step flow integration with execution. This design enables approximately 9 Hz control updates on edge hardware (vs. approximately 2.4 Hz baselines) without increasing marginal overhead. To handle the resulting latency shift, we introduce a temporally misaligned training strategy where the policy learns predictive compensation using stale semantic intent alongside real-time proprioception. Additionally, we address the insensitivity of static vision encoders to velocity by incorporating a differential motion predictor. TIDAL is architectural, making it orthogonal to system-level optimizations. Experiments show a 2x performance gain over open-loop baselines in dynamic interception tasks. Despite a marginal regression in static success rates, our approach yields a 4x increase in feedback frequency and extends the effective horizon of semantic embeddings beyond the native action chunk size. Under non-paused inference protocols, TIDAL remains robust where standard baselines fail due to latency.

  </details>



- **Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control**  
  Jannis Becktepe, Aleksandra Franz, Nils Thuerey, Sebastian Peitz  
  _2026-01-21_ · https://arxiv.org/abs/2601.15015v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has shown promising results in active flow control (AFC), yet progress in the field remains difficult to assess as existing studies rely on heterogeneous observation and actuation schemes, numerical setups, and evaluation protocols. Current AFC benchmarks attempt to address these issues but heavily rely on external computational fluid dynamics (CFD) solvers, are not fully differentiable, and provide limited 3D and multi-agent support. To overcome these limitations, we introduce FluidGym, the first standalone, fully differentiable benchmark suite for RL in AFC. Built entirely in PyTorch on top of the GPU-accelerated PICT solver, FluidGym runs in a single Python stack, requires no external CFD software, and provides standardized evaluation protocols. We present baseline results with PPO and SAC and release all environments, datasets, and trained models as public resources. FluidGym enables systematic comparison of control methods, establishes a scalable foundation for future research in learning-based flow control, and is available at https://github.com/safe-autonomous-systems/fluidgym.

  </details>



- **How to Build AI Agents by Augmenting LLMs with Codified Human Expert Domain Knowledge? A Software Engineering Framework**  
  Choro Ulan uulu, Mikhail Kulyabin, Iris Fuhrmann, Jan Joosten, Nuno Miguel Martins Pacheco, Filippos Petridis, Rebecca Johnson, Jan Bosch, Helena Holmström Olsson  
  _2026-01-21_ · https://arxiv.org/abs/2601.15153v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Critical domain knowledge typically resides with few experts, creating organizational bottlenecks in scalability and decision-making. Non-experts struggle to create effective visualizations, leading to suboptimal insights and diverting expert time. This paper investigates how to capture and embed human domain knowledge into AI agent systems through an industrial case study. We propose a software engineering framework to capture human domain knowledge for engineering AI agents in simulation data visualization by augmenting a Large Language Model (LLM) with a request classifier, Retrieval-Augmented Generation (RAG) system for code generation, codified expert rules, and visualization design principles unified in an agent demonstrating autonomous, reactive, proactive, and social behavior. Evaluation across five scenarios spanning multiple engineering domains with 12 evaluators demonstrates 206% improvement in output quality, with our agent achieving expert-level ratings in all cases versus baseline's poor performance, while maintaining superior code quality with lower variance. Our contributions are: an automated agent-based system for visualization generation and a validated framework for systematically capturing human domain knowledge and codifying tacit expert knowledge into AI agents, demonstrating that non-experts can achieve expert-level outcomes in specialized domains.

  </details>



- **CI4A: Semantic Component Interfaces for Agents Empowering Web Automation**  
  Zhi Qiu, Jiazheng Sun, Chenxiao Xia, Jun Zheng, Xin Peng  
  _2026-01-21_ · https://arxiv.org/abs/2601.14790v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  While Large Language Models demonstrate remarkable proficiency in high-level semantic planning, they remain limited in handling fine-grained, low-level web component manipulations. To address this limitation, extensive research has focused on enhancing model grounding capabilities through techniques such as Reinforcement Learning. However, rather than compelling agents to adapt to human-centric interfaces, we propose constructing interaction interfaces specifically optimized for agents. This paper introduces Component Interface for Agent (CI4A), a semantic encapsulation mechanism that abstracts the complex interaction logic of UI components into a set of unified tool primitives accessible to agents. We implemented CI4A within Ant Design, an industrial-grade front-end framework, covering 23 categories of commonly used UI components. Furthermore, we developed a hybrid agent featuring an action space that dynamically updates according to the page state, enabling flexible invocation of available CI4A tools. Leveraging the CI4A-integrated Ant Design, we refactored and upgraded the WebArena benchmark to evaluate existing SoTA methods. Experimental results demonstrate that the CI4A-based agent significantly outperforms existing approaches, achieving a new SoTA task success rate of 86.3%, alongside substantial improvements in execution efficiency.

  </details>



- **Case-Guided Sequential Assay Planning in Drug Discovery**  
  Tianchi Chen, Jan Bima, Sean L. Wu, Otto Ritter, Bingjia Yang, Xiang Yu  
  _2026-01-21_ · https://arxiv.org/abs/2601.14710v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Optimally sequencing experimental assays in drug discovery is a high-stakes planning problem under severe uncertainty and resource constraints. A primary obstacle for standard reinforcement learning (RL) is the absence of an explicit environment simulator or transition data $(s, a, s')$; planning must rely solely on a static database of historical outcomes. We introduce the Implicit Bayesian Markov Decision Process (IBMDP), a model-based RL framework designed for such simulator-free settings. IBMDP constructs a case-guided implicit model of transition dynamics by forming a nonparametric belief distribution using similar historical outcomes. This mechanism enables Bayesian belief updating as evidence accumulates and employs ensemble MCTS planning to generate stable policies that balance information gain toward desired outcomes with resource efficiency. We validate IBMDP through comprehensive experiments. On a real-world central nervous system (CNS) drug discovery task, IBMDP reduced resource consumption by up to 92\% compared to established heuristics while maintaining decision confidence. To rigorously assess decision quality, we also benchmarked IBMDP in a synthetic environment with a computable optimal policy. Our framework achieves significantly higher alignment with this optimal policy than a deterministic value iteration alternative that uses the same similarity-based model, demonstrating the superiority of our ensemble planner. IBMDP offers a practical solution for sequential experimental design in data-rich but simulator-poor domains.

  </details>



- **Beyond Error-Based Optimization: Experience-Driven Symbolic Regression with Goal-Conditioned Reinforcement Learning**  
  Jianwen Sun, Xinrui Li, Fuqing Li, Xiaoxuan Shen  
  _2026-01-21_ · https://arxiv.org/abs/2601.14693v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Symbolic Regression aims to automatically identify compact and interpretable mathematical expressions that model the functional relationship between input and output variables. Most existing search-based symbolic regression methods typically rely on the fitting error to inform the search process. However, in the vast expression space, numerous candidate expressions may exhibit similar error values while differing substantially in structure, leading to ambiguous search directions and hindering convergence to the underlying true function. To address this challenge, we propose a novel framework named EGRL-SR (Experience-driven Goal-conditioned Reinforcement Learning for Symbolic Regression). In contrast to traditional error-driven approaches, EGRL-SR introduces a new perspective: leveraging precise historical trajectories and optimizing the action-value network to proactively guide the search process, thereby achieving a more robust expression search. Specifically, we formulate symbolic regression as a goal-conditioned reinforcement learning problem and incorporate hindsight experience replay, allowing the action-value network to generalize common mapping patterns from diverse input-output pairs. Moreover, we design an all-point satisfaction binary reward function that encourages the action-value network to focus on structural patterns rather than low-error expressions, and concurrently propose a structure-guided heuristic exploration strategy to enhance search diversity and space coverage. Experiments on public benchmarks show that EGRL-SR consistently outperforms state-of-the-art methods in recovery rate and robustness, and can recover more complex expressions under the same search budget. Ablation results validate that the action-value network effectively guides the search, with both the reward function and the exploration strategy playing critical roles.

  </details>


