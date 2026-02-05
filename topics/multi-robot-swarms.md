# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-05 07:13 UTC_

Total papers shown: **6**


---

- **Act, Sense, Act: Learning Non-Markovian Active Perception Strategies from Large-Scale Egocentric Human Data**  
  Jialiang Li, Yi Qiao, Yunhan Guo, Changwen Chen, Wenzhao Lian  
  _2026-02-04_ · https://arxiv.org/abs/2602.04600v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Achieving generalizable manipulation in unconstrained environments requires the robot to proactively resolve information uncertainty, i.e., the capability of active perception. However, existing methods are often confined in limited types of sensing behaviors, restricting their applicability to complex environments. In this work, we formalize active perception as a non-Markovian process driven by information gain and decision branching, providing a structured categorization of visual active perception paradigms. Building on this perspective, we introduce CoMe-VLA, a cognitive and memory-aware vision-language-action (VLA) framework that leverages large-scale human egocentric data to learn versatile exploration and manipulation priors. Our framework integrates a cognitive auxiliary head for autonomous sub-task transitions and a dual-track memory system to maintain consistent self and environmental awareness by fusing proprioceptive and visual temporal contexts. By aligning human and robot hand-eye coordination behaviors in a unified egocentric action space, we train the model progressively in three stages. Extensive experiments on a wheel-based humanoid have demonstrated strong robustness and adaptability of our proposed method across diverse long-horizon tasks spanning multiple active perception scenarios.

  </details>



- **TACO: Temporal Consensus Optimization for Continual Neural Mapping**  
  Xunlan Zhou, Hongrui Zhao, Negar Mehr  
  _2026-02-04_ · https://arxiv.org/abs/2602.04516v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Neural implicit mapping has emerged as a powerful paradigm for robotic navigation and scene understanding. However, real-world robotic deployment requires continual adaptation to changing environments under strict memory and computation constraints, which existing mapping systems fail to support. Most prior methods rely on replaying historical observations to preserve consistency and assume static scenes. As a result, they cannot adapt to continual learning in dynamic robotic settings. To address these challenges, we propose TACO (TemporAl Consensus Optimization), a replay-free framework for continual neural mapping. We reformulate mapping as a temporal consensus optimization problem, where we treat past model snapshots as temporal neighbors. Intuitively, our approach resembles a model consulting its own past knowledge. We update the current map by enforcing weighted consensus with historical representations. Our method allows reliable past geometry to constrain optimization while permitting unreliable or outdated regions to be revised in response to new observations. TACO achieves a balance between memory efficiency and adaptability without storing or replaying previous data. Through extensive simulated and real-world experiments, we show that TACO robustly adapts to scene changes, and consistently outperforms other continual learning baselines.

  </details>



- **A Unified Complementarity-based Approach for Rigid-Body Manipulation and Motion Prediction**  
  Bingkun Huang, Xin Ma, Nilanjan Chakraborty, Riddhiman Laha  
  _2026-02-04_ · https://arxiv.org/abs/2602.04522v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic manipulation in unstructured environments requires planners to reason jointly about free-space motion and sustained, frictional contact with the environment. Existing (local) planning and simulation frameworks typically separate these regimes or rely on simplified contact representations, particularly when modeling non-convex or distributed contact patches. Such approximations limit the fidelity of contact-mode transitions and hinder the robust execution of contact-rich behaviors in real time. This paper presents a unified discrete-time modeling framework for robotic manipulation that consistently captures both free motion and frictional contact within a single mathematical formalism (Unicomp). Building on complementarity-based rigid-body dynamics, we formulate free-space motion and contact interactions as coupled linear and nonlinear complementarity problems, enabling principled transitions between contact modes without enforcing fixed-contact assumptions. For planar patch contact, we derive a frictional contact model from the maximum power dissipation principle in which the set of admissible contact wrenches is represented by an ellipsoidal limit surface. This representation captures coupled force-moment effects, including torsional friction, while remaining agnostic to the underlying pressure distribution across the contact patch. The resulting formulation yields a discrete-time predictive model that relates generalized velocities and contact wrenches through quadratic constraints and is suitable for real-time optimization-based planning. Experimental results show that the proposed approach enables stable, physically consistent behavior at interactive speeds across tasks, from planar pushing to contact-rich whole-body maneuvers.

  </details>



- **ReThinker: Scientific Reasoning by Rethinking with Guided Reflection and Confidence Control**  
  Zhentao Tang, Yuqi Cui, Shixiong Kai, Wenqian Zhao, Ke Ye, Xing Li, Anxin Tian, Zehua Pei, Hui-Ling Zhen, Shoubo Hu, et al.  
  _2026-02-04_ · https://arxiv.org/abs/2602.04496v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Expert-level scientific reasoning remains challenging for large language models, particularly on benchmarks such as Humanity's Last Exam (HLE), where rigid tool pipelines, brittle multi-agent coordination, and inefficient test-time scaling often limit performance. We introduce ReThinker, a confidence-aware agentic framework that orchestrates retrieval, tool use, and multi-agent reasoning through a stage-wise Solver-Critic-Selector architecture. Rather than following a fixed pipeline, ReThinker dynamically allocates computation based on model confidence, enabling adaptive tool invocation, guided multi-dimensional reflection, and robust confidence-weighted selection. To support scalable training without human annotation, we further propose a reverse data synthesis pipeline and an adaptive trajectory recycling strategy that transform successful reasoning traces into high-quality supervision. Experiments on HLE, GAIA, and XBench demonstrate that ReThinker consistently outperforms state-of-the-art foundation models with tools and existing deep research systems, achieving state-of-the-art results on expert-level reasoning tasks.

  </details>



- **LLM-Empowered Cooperative Content Caching in Vehicular Fog Caching-Assisted Platoon Networks**  
  Bowen Tan, Qiong Wu, Pingyi Fan, Kezhi Wang, Nan Cheng, Wen Chen  
  _2026-02-04_ · https://arxiv.org/abs/2602.04471v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  This letter proposes a novel three-tier content caching architecture for Vehicular Fog Caching (VFC)-assisted platoon, where the VFC is formed by the vehicles driving near the platoon. The system strategically coordinates storage across local platoon vehicles, dynamic VFC clusters, and cloud server (CS) to minimize content retrieval latency. To efficiently manage distributed storage, we integrate large language models (LLMs) for real-time and intelligent caching decisions. The proposed approach leverages LLMs' ability to process heterogeneous information, including user profiles, historical data, content characteristics, and dynamic system states. Through a designed prompting framework encoding task objectives and caching constraints, the LLMs formulate caching as a decision-making task, and our hierarchical deterministic caching mapping strategy enables adaptive requests prediction and precise content placement across three tiers without frequent retraining. Simulation results demonstrate the advantages of our proposed caching scheme.

  </details>



- **SPEAR: An Engineering Case Study of Multi-Agent Coordination for Smart Contract Auditing**  
  Arnab Mallick, Indraveni Chebolu, Harmesh Rana  
  _2026-02-04_ · https://arxiv.org/abs/2602.04418v1 · `cs.MA`  
  <details><summary>Abstract</summary>

  We present SPEAR, a multi-agent coordination framework for smart contract auditing that applies established MAS patterns in a realistic security analysis workflow. SPEAR models auditing as a coordinated mission carried out by specialized agents: a Planning Agent prioritizes contracts using risk-aware heuristics, an Execution Agent allocates tasks via the Contract Net protocol, and a Repair Agent autonomously recovers from brittle generated artifacts using a programmatic-first repair policy. Agents maintain local beliefs updated through AGM-compliant revision, coordinate via negotiation and auction protocols, and revise plans as new information becomes available. An empirical study compares the multi-agent design with centralized and pipeline-based alternatives under controlled failure scenarios, focusing on coordination, recovery behavior, and resource use.

  </details>


