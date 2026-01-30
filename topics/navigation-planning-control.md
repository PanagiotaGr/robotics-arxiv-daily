# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-01-30 07:03 UTC_

Total papers shown: **4**


---

- **LLM-Driven Scenario-Aware Planning for Autonomous Driving**  
  He Li, Zhaowei Chen, Rui Gao, Guoliang Li, Qi Hao, Shuai Wang, Chengzhong Xu  
  _2026-01-29_ · https://arxiv.org/abs/2601.21876v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Hybrid planner switching framework (HPSF) for autonomous driving needs to reconcile high-speed driving efficiency with safe maneuvering in dense traffic. Existing HPSF methods often fail to make reliable mode transitions or sustain efficient driving in congested environments, owing to heuristic scene recognition and low-frequency control updates. To address the limitation, this paper proposes LAP, a large language model (LLM) driven, adaptive planning method, which switches between high-speed driving in low-complexity scenes and precise driving in high-complexity scenes, enabling high qualities of trajectory generation through confined gaps. This is achieved by leveraging LLM for scene understanding and integrating its inference into the joint optimization of mode configuration and motion planning. The joint optimization is solved using tree-search model predictive control and alternating minimization. We implement LAP by Python in Robot Operating System (ROS). High-fidelity simulation results show that the proposed LAP outperforms other benchmarks in terms of both driving time and success rate.

  </details>



- **CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty**  
  Johannes Kirmayr, Lukas Stappen, Elisabeth André  
  _2026-01-29_ · https://arxiv.org/abs/2601.22027v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Existing benchmarks for Large Language Model (LLM) agents focus on task completion under idealistic settings but overlook reliability in real-world, user-facing applications. In domains, such as in-car voice assistants, users often issue incomplete or ambiguous requests, creating intrinsic uncertainty that agents must manage through dialogue, tool use, and policy adherence. We introduce CAR-bench, a benchmark for evaluating consistency, uncertainty handling, and capability awareness in multi-turn, tool-using LLM agents in an in-car assistant domain. The environment features an LLM-simulated user, domain policies, and 58 interconnected tools spanning navigation, productivity, charging, and vehicle control. Beyond standard task completion, CAR-bench introduces Hallucination tasks that test agents' limit-awareness under missing tools or information, and Disambiguation tasks that require resolving uncertainty through clarification or internal information gathering. Baseline results reveal large gaps between occasional and consistent success on all task types. Even frontier reasoning LLMs achieve less than 50% consistent pass rate on Disambiguation tasks due to premature actions, and frequently violate policies or fabricate information to satisfy user requests in Hallucination tasks, underscoring the need for more reliable and self-aware LLM agents in real-world settings.

  </details>



- **Dynamic Topology Awareness: Breaking the Granularity Rigidity in Vision-Language Navigation**  
  Jiankun Peng, Jianyuan Guo, Ying Xu, Yue Liu, Jiashuang Yan, Xuanwei Ye, Houhua Li, Xiaoming Wang  
  _2026-01-29_ · https://arxiv.org/abs/2601.21751v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language Navigation in Continuous Environments (VLN-CE) presents a core challenge: grounding high-level linguistic instructions into precise, safe, and long-horizon spatial actions. Explicit topological maps have proven to be a vital solution for providing robust spatial memory in such tasks. However, existing topological planning methods suffer from a "Granularity Rigidity" problem. Specifically, these methods typically rely on fixed geometric thresholds to sample nodes, which fails to adapt to varying environmental complexities. This rigidity leads to a critical mismatch: the model tends to over-sample in simple areas, causing computational redundancy, while under-sampling in high-uncertainty regions, increasing collision risks and compromising precision. To address this, we propose DGNav, a framework for Dynamic Topological Navigation, introducing a context-aware mechanism to modulate map density and connectivity on-the-fly. Our approach comprises two core innovations: (1) A Scene-Aware Adaptive Strategy that dynamically modulates graph construction thresholds based on the dispersion of predicted waypoints, enabling "densification on demand" in challenging environments; (2) A Dynamic Graph Transformer that reconstructs graph connectivity by fusing visual, linguistic, and geometric cues into dynamic edge weights, enabling the agent to filter out topological noise and enhancing instruction adherence. Extensive experiments on the R2R-CE and RxR-CE benchmarks demonstrate DGNav exhibits superior navigation performance and strong generalization capabilities. Furthermore, ablation studies confirm that our framework achieves an optimal trade-off between navigation efficiency and safe exploration. The code is available at https://github.com/shannanshouyin/DGNav.

  </details>



- **WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents**  
  Yao Zhang, Shijie Tang, Zeyu Li, Zhen Han, Volker Tresp  
  _2026-01-29_ · https://arxiv.org/abs/2601.21872v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Web agents hold great potential for automating complex computer tasks, yet their interactions involve long-horizon, sequential decision-making with irreversible actions. In such settings, outcome-based supervision is sparse and delayed, often rewarding incorrect trajectories and failing to support inference-time scaling. This motivates the use of Process Reward Models (WebPRMs) for web navigation, but existing approaches remain limited: scalar WebPRMs collapse progress into coarse, weakly grounded signals, while checklist-based WebPRMs rely on brittle template matching that fails under layout or semantic changes and often mislabels superficially correct actions as successful, providing little insight or interpretability. To address these challenges, we introduce WebArbiter, a reasoning-first, principle-inducing WebPRM that formulates reward modeling as text generation, producing structured justifications that conclude with a preference verdict and identify the action most conducive to task completion under the current context. Training follows a two-stage pipeline: reasoning distillation equips the model with coherent principle-guided reasoning, and reinforcement learning corrects teacher biases by directly aligning verdicts with correctness, enabling stronger generalization. To support systematic evaluation, we release WebPRMBench, a comprehensive benchmark spanning four diverse web environments with rich tasks and high-quality preference annotations. On WebPRMBench, WebArbiter-7B outperforms the strongest baseline, GPT-5, by 9.1 points. In reward-guided trajectory search on WebArena-Lite, it surpasses the best prior WebPRM by up to 7.2 points, underscoring its robustness and practical value in real-world complex web tasks.

  </details>


