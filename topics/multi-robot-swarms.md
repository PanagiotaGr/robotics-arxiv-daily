# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-01-29 07:02 UTC_

Total papers shown: **4**


---

- **MeCo: Enhancing LLM-Empowered Multi-Robot Collaboration via Similar Task Memoization**  
  Baiqing Wang, Helei Cui, Bo Zhang, Xiaolong Zheng, Bin Guo, Zhiwen Yu  
  _2026-01-28_ · https://arxiv.org/abs/2601.20577v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-robot systems have been widely deployed in real-world applications, providing significant improvements in efficiency and reductions in labor costs. However, most existing multi-robot collaboration methods rely on extensive task-specific training, which limits their adaptability to new or diverse scenarios. Recent research leverages the language understanding and reasoning capabilities of large language models (LLMs) to enable more flexible collaboration without specialized training. Yet, current LLM-empowered approaches remain inefficient: when confronted with identical or similar tasks, they must replan from scratch because they omit task-level similarities. To address this limitation, we propose MeCo, a similarity-aware multi-robot collaboration framework that applies the principle of ``cache and reuse'' (a.k.a., memoization) to reduce redundant computation. Unlike simple task repetition, identifying and reusing solutions for similar but not identical tasks is far more challenging, particularly in multi-robot settings. To this end, MeCo introduces a new similarity testing method that retrieves previously solved tasks with high relevance, enabling effective plan reuse without re-invoking LLMs. Furthermore, we present MeCoBench, the first benchmark designed to evaluate performance on similar-task collaboration scenarios. Experimental results show that MeCo substantially reduces planning costs and improves success rates compared with state-of-the-art approaches.

  </details>



- **Investigating the Development of Task-Oriented Communication in Vision-Language Models**  
  Boaz Carmeli, Orr Paradise, Shafi Goldwasser, Yonatan Belinkov, Ron Meir  
  _2026-01-28_ · https://arxiv.org/abs/2601.20641v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We investigate whether \emph{LLM-based agents} can develop task-oriented communication protocols that differ from standard natural language in collaborative reasoning tasks. Our focus is on two core properties such task-oriented protocols may exhibit: Efficiency -- conveying task-relevant information more concisely than natural language, and Covertness -- becoming difficult for external observers to interpret, raising concerns about transparency and control. To investigate these aspects, we use a referential-game framework in which vision-language model (VLM) agents communicate, providing a controlled, measurable setting for evaluating language variants. Experiments show that VLMs can develop effective, task-adapted communication patterns. At the same time, they can develop covert protocols that are difficult for humans and external agents to interpret. We also observe spontaneous coordination between similar models without explicitly shared protocols. These findings highlight both the potential and the risks of task-oriented communication, and position referential games as a valuable testbed for future work in this area.

  </details>



- **A Practical Framework of Key Performance Indicators for Multi-Robot Lunar and Planetary Field Tests**  
  Julia Richter, David Oberacker, Gabriela Ligeza, Valentin T. Bickel, Philip Arm, William Talbot, Marvin Grosse Besselmann, Florian Kehl, Tristan Schnell, Hendrik Kolvenbach, et al.  
  _2026-01-28_ · https://arxiv.org/abs/2601.20529v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic prospecting for critical resources on the Moon, such as ilmenite, rare earth elements, and water ice, requires robust exploration methods given the diverse terrain and harsh environmental conditions. Although numerous analog field trials address these goals, comparing their results remains challenging because of differences in robot platforms and experimental setups. These missions typically assess performance using selected, scenario-specific engineering metrics that fail to establish a clear link between field performance and science-driven objectives. In this paper, we address this gap by deriving a structured framework of KPI from three realistic multi-robot lunar scenarios reflecting scientific objectives and operational constraints. Our framework emphasizes scenario-dependent priorities in efficiency, robustness, and precision, and is explicitly designed for practical applicability in field deployments. We validated the framework in a multi-robot field test and found it practical and easy to apply for efficiency- and robustness-related KPI, whereas precision-oriented KPI require reliable ground-truth data that is not always feasible to obtain in outdoor analog environments. Overall, we propose this framework as a common evaluation standard enabling consistent, goal-oriented comparison of multi-robot field trials and supporting systematic development of robotic systems for future planetary exploration.

  </details>



- **Decentralized Stochastic Constrained Optimization via Prox-Linearization**  
  Shivangi Dubey Sharma, Basil M. Idrees, Lavish Arora, Ketan Rajawat  
  _2026-01-28_ · https://arxiv.org/abs/2601.20345v1 · `math.OC`  
  <details><summary>Abstract</summary>

  This paper studies consensus-based decentralized stochastic optimization for minimizing possibly non-convex expected objectives with convex non-smooth regularizers and nonlinear functional inequality constraints. We reformulate the constrained problem using the exact-penalty model and develop two algorithms that require only local stochastic gradients and first-order constraint information. The first method, Decentralized Stochastic Momentum-based Prox-Linear Algorithm (D-SMPL), combines constraint linearization with a prox-linear step, resulting in a linearly constrained quadratic subproblem per iteration. Building on this approach, we propose a successive convex approximation (SCA) variant, Decentralized SCA Momentum-based Prox-Linear (D-SCAMPL), which handles additional objective structure through strongly convex surrogate subproblems while still allowing infeasible initialization. Both methods incorporate recursive momentum-based gradient estimators and a consensus mechanism requiring only two communication rounds per iteration. Under standard smoothness and regularity assumptions, both algorithms achieve an oracle complexity of $\mathcal{O}(ε^{-3/2})$, matching the optimal rate known for unconstrained centralized stochastic non-convex optimization. Numerical experiments on energy-optimal ocean trajectory planning corroborate the theory and demonstrate improved performance over existing decentralized baselines.

  </details>


