# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-07 06:57 UTC_

Total papers shown: **3**


---

- **Location-Aware Dispersion on Anonymous Graphs**  
  Himani, Supantha Pandit, Gokarna Sharma  
  _2026-02-05_ · https://arxiv.org/abs/2602.05948v1 · `cs.DC`  
  <details><summary>Abstract</summary>

  The well-studied DISPERSION problem is a fundamental coordination problem in distributed robotics, where a set of mobile robots must relocate so that each occupies a distinct node of a network. DISPERSION assumes that a robot can settle at any node as long as no other robot settles on that node. In this work, we introduce LOCATION-AWARE DISPERSION, a novel generalization of DISPERSION that incorporates location awareness: Let $G = (V, E)$ be an anonymous, connected, undirected graph with $n = |V|$ nodes, each labeled with a color $\sf{col}(v) \in C = \{c_1, \dots, c_t\}, t\leq n$. A set $R = \{r_1, \dots, r_k\}$ of $k \leq n$ mobile robots is given, where each robot $r_i$ has an associated color $\mathsf{col}(r_i) \in C$. Initially placed arbitrarily on the graph, the goal is to relocate the robots so that each occupies a distinct node of the same color. When $|C|=1$, LOCATION-AWARE DISPERSION reduces to DISPERSION. There is a solution to DISPERSION in graphs with any $k\leq n$ without knowing $k,n$. Like DISPERSION, the goal is to solve LOCATION-AWARE DISPERSION minimizing both time and memory requirement at each agent. We develop several deterministic algorithms with guaranteed bounds on both time and memory requirement. We also give an impossibility and a lower bound for any deterministic algorithm for LOCATION-AWARE DISPERSION. To the best of our knowledge, the presented results collectively establish the algorithmic feasibility of LOCATION-AWARE DISPERSION in anonymous networks and also highlight the challenges on getting an efficient solution compared to the solutions for DISPERSION.

  </details>



- **InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions**  
  Sirui Xu, Samuel Schulter, Morteza Ziyadi, Xialin He, Xiaohan Fei, Yu-Xiong Wang, Liangyan Gui  
  _2026-02-05_ · https://arxiv.org/abs/2602.06035v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Humans rarely plan whole-body interactions with objects at the level of explicit whole-body movements. High-level intentions, such as affordance, define the goal, while coordinated balance, contact, and manipulation can emerge naturally from underlying physical and motor priors. Scaling such priors is key to enabling humanoids to compose and generalize loco-manipulation skills across diverse contexts while maintaining physically coherent whole-body coordination. To this end, we introduce InterPrior, a scalable framework that learns a unified generative controller through large-scale imitation pretraining and post-training by reinforcement learning. InterPrior first distills a full-reference imitation expert into a versatile, goal-conditioned variational policy that reconstructs motion from multimodal observations and high-level intent. While the distilled policy reconstructs training behaviors, it does not generalize reliably due to the vast configuration space of large-scale human-object interactions. To address this, we apply data augmentation with physical perturbations, and then perform reinforcement learning finetuning to improve competence on unseen goals and initializations. Together, these steps consolidate the reconstructed latent skills into a valid manifold, yielding a motion prior that generalizes beyond the training data, e.g., it can incorporate new behaviors such as interactions with unseen objects. We further demonstrate its effectiveness for user-interactive control and its potential for real robot deployment.

  </details>



- **Reactive Knowledge Representation and Asynchronous Reasoning**  
  Simon Kohaut, Benedict Flade, Julian Eggert, Kristian Kersting, Devendra Singh Dhami  
  _2026-02-05_ · https://arxiv.org/abs/2602.05625v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Exact inference in complex probabilistic models often incurs prohibitive computational costs. This challenge is particularly acute for autonomous agents in dynamic environments that require frequent, real-time belief updates. Existing methods are often inefficient for ongoing reasoning, as they re-evaluate the entire model upon any change, failing to exploit that real-world information streams have heterogeneous update rates. To address this, we approach the problem from a reactive, asynchronous, probabilistic reasoning perspective. We first introduce Resin (Reactive Signal Inference), a probabilistic programming language that merges probabilistic logic with reactive programming. Furthermore, to provide efficient and exact semantics for Resin, we propose Reactive Circuits (RCs). Formulated as a meta-structure over Algebraic Circuits and asynchronous data streams, RCs are time-dynamic Directed Acyclic Graphs that autonomously adapt themselves based on the volatility of input signals. In high-fidelity drone swarm simulations, our approach achieves several orders of magnitude of speedup over frequency-agnostic inference. We demonstrate that RCs' structural adaptations successfully capture environmental dynamics, significantly reducing latency and facilitating reactive real-time reasoning. By partitioning computations based on the estimated Frequency of Change in the asynchronous inputs, large inference tasks can be decomposed into individually memoized sub-problems. This ensures that only the specific components of a model affected by new information are re-evaluated, drastically reducing redundant computation in streaming contexts.

  </details>


