# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-12 07:15 UTC_

Total papers shown: **4**


---

- **Min-Sum Uniform Coverage Problem by Autonomous Mobile Robots**  
  Animesh Maiti, Abhinav Chakraborty, Bibhuti Das, Subhash Bhagat, Krishnendu Mukhopadhyaya  
  _2026-02-11_ · https://arxiv.org/abs/2602.11125v1 · `cs.DC`  
  <details><summary>Abstract</summary>

  We study the \textit{min-sum uniform coverage} problem for a swarm of $n$ mobile robots on a given finite line segment and on a circle having finite positive radius, where the circle is given as an input. The robots must coordinate their movements to reach a uniformly spaced configuration that minimizes the total distance traveled by all robots. The robots are autonomous, anonymous, identical, and homogeneous, and operate under the \textit{Look-Compute-Move} (LCM) model with \textit{non-rigid} motion controlled by a fair asynchronous scheduler. They are oblivious and silent, possessing neither persistent memory nor a means of explicit communication. In the \textbf{line-segment setting}, the \textit{min-sum uniform coverage} problem requires placing the robots at uniformly spaced points along the segment so as to minimize the total distance traveled by all robots. In the \textbf{circle setting} for this problem, the robots have to arrange themselves uniformly around the given circle to form a regular $n$-gon. There is no fixed orientation or designated starting vertex, and the goal is to minimize the total distance traveled by all the robots. We present a deterministic distributed algorithm that achieves uniform coverage in the line-segment setting with minimum total movement cost. For the circle setting, we characterize all initial configurations for which the \textit{min-sum uniform coverage} problem is deterministically unsolvable under the considered robot model. For all the other remaining configurations, we provide a deterministic distributed algorithm that achieves uniform coverage while minimizing the total distance traveled. These results characterize the deterministic solvability of min-sum coverage for oblivious robots and achieve optimal cost whenever solvable.

  </details>



- **Multi-UAV Trajectory Optimization for Bearing-Only Localization in GPS Denied Environments**  
  Alfonso Sciacchitano, Liraz Mudrik, Sean Kragelund, Isaac Kaminer  
  _2026-02-11_ · https://arxiv.org/abs/2602.11116v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Accurate localization of maritime targets by unmanned aerial vehicles (UAVs) remains challenging in GPS-denied environments. UAVs equipped with gimballed electro-optical sensors are typically used to localize targets, however, reliance on these sensors increases mechanical complexity, cost, and susceptibility to single-point failures, limiting scalability and robustness in multi-UAV operations. This work presents a new trajectory optimization framework that enables cooperative target localization using UAVs with fixed, non-gimballed cameras operating in coordination with a surface vessel. This estimation-aware optimization generates dynamically feasible trajectories that explicitly account for mission constraints, platform dynamics, and out-of-frame events. Estimation-aware trajectories outperform heuristic paths by reducing localization error by more than a factor of two, motivating their use in cooperative operations. Results further demonstrate that coordinated UAVs with fixed, non-gimballed cameras achieve localization accuracy that meets or exceeds that of single gimballed systems, while substantially lowering system complexity and cost, enabling scalability, and enhancing mission resilience.

  </details>



- **DeepImageSearch: Benchmarking Multimodal Agents for Context-Aware Image Retrieval in Visual Histories**  
  Chenlong Deng, Mengjie Deng, Junjie Wu, Dun Zeng, Teng Wang, Qingsong Xie, Jiadeng Huang, Shengjie Ma, Changwang Zhang, Zhaoxiang Wang, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10809v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Existing multimodal retrieval systems excel at semantic matching but implicitly assume that query-image relevance can be measured in isolation. This paradigm overlooks the rich dependencies inherent in realistic visual streams, where information is distributed across temporal sequences rather than confined to single snapshots. To bridge this gap, we introduce DeepImageSearch, a novel agentic paradigm that reformulates image retrieval as an autonomous exploration task. Models must plan and perform multi-step reasoning over raw visual histories to locate targets based on implicit contextual cues. We construct DISBench, a challenging benchmark built on interconnected visual data. To address the scalability challenge of creating context-dependent queries, we propose a human-model collaborative pipeline that employs vision-language models to mine latent spatiotemporal associations, effectively offloading intensive context discovery before human verification. Furthermore, we build a robust baseline using a modular agent framework equipped with fine-grained tools and a dual-memory system for long-horizon navigation. Extensive experiments demonstrate that DISBench poses significant challenges to state-of-the-art models, highlighting the necessity of incorporating agentic reasoning into next-generation retrieval systems.

  </details>



- **Beyond Task Performance: A Metric-Based Analysis of Sequential Cooperation in Heterogeneous Multi-Agent Destructive Foraging**  
  Alejandro Mendoza Barrionuevo, Samuel Yanes Luis, Daniel Gutiérrez Reina, Sergio L. Toral Marín  
  _2026-02-11_ · https://arxiv.org/abs/2602.10685v1 · `cs.MA`  
  <details><summary>Abstract</summary>

  This work addresses the problem of analyzing cooperation in heterogeneous multi-agent systems which operate under partial observability and temporal role dependency, framed within a destructive multi-agent foraging setting. Unlike most previous studies, which focus primarily on algorithmic performance with respect to task completion, this article proposes a systematic set of general-purpose cooperation metrics aimed at characterizing not only efficiency, but also coordination and dependency between teams and agents, fairness, and sensitivity. These metrics are designed to be transferable to different multi-agent sequential domains similar to foraging. The proposed suite of metrics is structured into three main categories that jointly provide a multilevel characterization of cooperation: primary metrics, inter-team metrics, and intra-team metrics. They have been validated in a realistic destructive foraging scenario inspired by dynamic aquatic surface cleaning using heterogeneous autonomous vehicles. It involves two specialized teams with sequential dependencies: one focused on the search of resources, and another on their destruction. Several representative approaches have been evaluated, covering both learning-based algorithms and classical heuristic paradigms.

  </details>


