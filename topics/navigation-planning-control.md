# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-13 07:12 UTC_

Total papers shown: **9**


---

- **LAMP: Implicit Language Map for Robot Navigation**  
  Sibaek Lee, Hyeonwoo Yu, Giseop Kim, Sunwook Choi  
  _2026-02-12_ · https://arxiv.org/abs/2602.11862v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advances in vision-language models have made zero-shot navigation feasible, enabling robots to follow natural language instructions without requiring labeling. However, existing methods that explicitly store language vectors in grid or node-based maps struggle to scale to large environments due to excessive memory requirements and limited resolution for fine-grained planning. We introduce LAMP (Language Map), a novel neural language field-based navigation framework that learns a continuous, language-driven map and directly leverages it for fine-grained path generation. Unlike prior approaches, our method encodes language features as an implicit neural field rather than storing them explicitly at every location. By combining this implicit representation with a sparse graph, LAMP supports efficient coarse path planning and then performs gradient-based optimization in the learned field to refine poses near the goal. This coarse-to-fine pipeline, language-driven, gradient-guided optimization is the first application of an implicit language map for precise path generation. This refinement is particularly effective at selecting goal regions not directly observed by leveraging semantic similarities in the learned feature space. To further enhance robustness, we adopt a Bayesian framework that models embedding uncertainty via the von Mises-Fisher distribution, thereby improving generalization to unobserved regions. To scale to large environments, LAMP employs a graph sampling strategy that prioritizes spatial coverage and embedding confidence, retaining only the most informative nodes and substantially reducing computational overhead. Our experimental results, both in NVIDIA Isaac Sim and on a real multi-floor building, demonstrate that LAMP outperforms existing explicit methods in both memory efficiency and fine-grained goal-reaching accuracy.

  </details>



- **Safety Beyond the Training Data: Robust Out-of-Distribution MPC via Conformalized System Level Synthesis**  
  Anutam Srinivasan, Antoine Leeman, Glen Chou  
  _2026-02-12_ · https://arxiv.org/abs/2602.12047v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a novel framework for robust out-of-distribution planning and control using conformal prediction (CP) and system level synthesis (SLS), addressing the challenge of ensuring safety and robustness when using learned dynamics models beyond the training data distribution. We first derive high-confidence model error bounds using weighted CP with a learned, state-control-dependent covariance model. These bounds are integrated into an SLS-based robust nonlinear model predictive control (MPC) formulation, which performs constraint tightening over the prediction horizon via volume-optimized forward reachable sets. We provide theoretical guarantees on coverage and robustness under distributional drift, and analyze the impact of data density and trajectory tube size on prediction coverage. Empirically, we demonstrate our method on nonlinear systems of increasing complexity, including a 4D car and a {12D} quadcopter, improving safety and robustness compared to fixed-bound and non-robust baselines, especially outside of the data distribution.

  </details>



- **3DGSNav: Enhancing Vision-Language Model Reasoning for Object Navigation via Active 3D Gaussian Splatting**  
  Wancai Zheng, Hao Chen, Xianlong Lu, Linlin Ou, Xinyi Yu  
  _2026-02-12_ · https://arxiv.org/abs/2602.12159v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Object navigation is a core capability of embodied intelligence, enabling an agent to locate target objects in unknown environments. Recent advances in vision-language models (VLMs) have facilitated zero-shot object navigation (ZSON). However, existing methods often rely on scene abstractions that convert environments into semantic maps or textual representations, causing high-level decision making to be constrained by the accuracy of low-level perception. In this work, we present 3DGSNav, a novel ZSON framework that embeds 3D Gaussian Splatting (3DGS) as persistent memory for VLMs to enhance spatial reasoning. Through active perception, 3DGSNav incrementally constructs a 3DGS representation of the environment, enabling trajectory-guided free-viewpoint rendering of frontier-aware first-person views. Moreover, we design structured visual prompts and integrate them with Chain-of-Thought (CoT) prompting to further improve VLM reasoning. During navigation, a real-time object detector filters potential targets, while VLM-driven active viewpoint switching performs target re-verification, ensuring efficient and reliable recognition. Extensive evaluations across multiple benchmarks and real-world experiments on a quadruped robot demonstrate that our method achieves robust and competitive performance against state-of-the-art approaches.The Project Page:https://aczheng-cai.github.io/3dgsnav.github.io/

  </details>



- **Multi Graph Search for High-Dimensional Robot Motion Planning**  
  Itamar Mishani, Maxim Likhachev  
  _2026-02-12_ · https://arxiv.org/abs/2602.12096v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Efficient motion planning for high-dimensional robotic systems, such as manipulators and mobile manipulators, is critical for real-time operation and reliable deployment. Although advances in planning algorithms have enhanced scalability to high-dimensional state spaces, these improvements often come at the cost of generating unpredictable, inconsistent motions or requiring excessive computational resources and memory. In this work, we introduce Multi-Graph Search (MGS), a search-based motion planning algorithm that generalizes classical unidirectional and bidirectional search to a multi-graph setting. MGS maintains and incrementally expands multiple implicit graphs over the state space, focusing exploration on high-potential regions while allowing initially disconnected subgraphs to be merged through feasible transitions as the search progresses. We prove that MGS is complete and bounded-suboptimal, and empirically demonstrate its effectiveness on a range of manipulation and mobile manipulation tasks. Demonstrations, benchmarks and code are available at https://multi-graph-search.github.io/.

  </details>



- **Multi UAVs Preflight Planning in a Shared and Dynamic Airspace**  
  Amath Sow, Mauricio Rodriguez Cesen, Fabiola Martins Campos de Oliveira, Mariusz Wzorek, Daniel de Leng, Mattias Tiger, Fredrik Heintz, Christian Esteve Rothenberg  
  _2026-02-12_ · https://arxiv.org/abs/2602.12055v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Preflight planning for large-scale Unmanned Aerial Vehicle (UAV) fleets in dynamic, shared airspace presents significant challenges, including temporal No-Fly Zones (NFZs), heterogeneous vehicle profiles, and strict delivery deadlines. While Multi-Agent Path Finding (MAPF) provides a formal framework, existing methods often lack the scalability and flexibility required for real-world Unmanned Traffic Management (UTM). We propose DTAPP-IICR: a Delivery-Time Aware Prioritized Planning method with Incremental and Iterative Conflict Resolution. Our framework first generates an initial solution by prioritizing missions based on urgency. Secondly, it computes roundtrip trajectories using SFIPP-ST, a novel 4D single-agent planner (Safe Flight Interval Path Planning with Soft and Temporal Constraints). SFIPP-ST handles heterogeneous UAVs, strictly enforces temporal NFZs, and models inter-agent conflicts as soft constraints. Subsequently, an iterative Large Neighborhood Search, guided by a geometric conflict graph, efficiently resolves any residual conflicts. A completeness-preserving directional pruning technique further accelerates the 3D search. On benchmarks with temporal NFZs, DTAPP-IICR achieves near-100% success with fleets of up to 1,000 UAVs and gains up to 50% runtime reduction from pruning, outperforming batch Enhanced Conflict-Based Search in the UTM context. Scaling successfully in realistic city-scale operations where other priority-based methods fail even at moderate deployments, DTAPP-IICR is positioned as a practical and scalable solution for preflight planning in dense, dynamic urban airspace.

  </details>



- **Adaptive-Horizon Conflict-Based Search for Closed-Loop Multi-Agent Path Finding**  
  Jiarui Li, Federico Pecora, Runyu Zhang, Gioele Zardini  
  _2026-02-12_ · https://arxiv.org/abs/2602.12024v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  MAPF is a core coordination problem for large robot fleets in automated warehouses and logistics. Existing approaches are typically either open-loop planners, which generate fixed trajectories and struggle to handle disturbances, or closed-loop heuristics without reliable performance guarantees, limiting their use in safety-critical deployments. This paper presents ACCBS, a closed-loop algorithm built on a finite-horizon variant of CBS with a horizon-changing mechanism inspired by iterative deepening in MPC. ACCBS dynamically adjusts the planning horizon based on the available computational budget, and reuses a single constraint tree to enable seamless transitions between horizons. As a result, it produces high-quality feasible solutions quickly while being asymptotically optimal as the budget increases, exhibiting anytime behavior. Extensive case studies demonstrate that ACCBS combines flexibility to disturbances with strong performance guarantees, effectively bridging the gap between theoretical optimality and practical robustness for large-scale robot deployment.

  </details>



- **Learning to Control: The iUzawa-Net for Nonsmooth Optimal Control of Linear PDEs**  
  Yongcun Song, Xiaoming Yuan, Hangrui Yue, Tianyou Zeng  
  _2026-02-12_ · https://arxiv.org/abs/2602.12273v1 · `math.OC`  
  <details><summary>Abstract</summary>

  We propose an optimization-informed deep neural network approach, named iUzawa-Net, aiming for the first solver that enables real-time solutions for a class of nonsmooth optimal control problems of linear partial differential equations (PDEs). The iUzawa-Net unrolls an inexact Uzawa method for saddle point problems, replacing classical preconditioners and PDE solvers with specifically designed learnable neural networks. We prove universal approximation properties and establish the asymptotic $\varepsilon$-optimality for the iUzawa-Net, and validate its promising numerical efficiency through nonsmooth elliptic and parabolic optimal control problems. Our techniques offer a versatile framework for designing and analyzing various optimization-informed deep learning approaches to optimal control and other PDE-constrained optimization problems. The proposed learning-to-control approach synergizes model-based optimization algorithms and data-driven deep learning techniques, inheriting the merits of both methodologies.

  </details>



- **Pack it in: Packing into Partially Filled Containers Through Contact**  
  David Russell, Zisong Xu, Maximo A. Roa, Mehmet Dogar  
  _2026-02-12_ · https://arxiv.org/abs/2602.12095v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The automation of warehouse operations is crucial for improving productivity and reducing human exposure to hazardous environments. One operation frequently performed in warehouses is bin-packing where items need to be placed into containers, either for delivery to a customer, or for temporary storage in the warehouse. Whilst prior bin-packing works have largely been focused on packing items into empty containers and have adopted collision-free strategies, it is often the case that containers will already be partially filled with items, often in suboptimal arrangements due to transportation about a warehouse. This paper presents a contact-aware packing approach that exploits purposeful interactions with previously placed objects to create free space and enable successful placement of new items. This is achieved by using a contact-based multi-object trajectory optimizer within a model predictive controller, integrated with a physics-aware perception system that estimates object poses even during inevitable occlusions, and a method that suggests physically-feasible locations to place the object inside the container.

  </details>



- **AlphaPROBE: Alpha Mining via Principled Retrieval and On-graph biased evolution**  
  Taian Guo, Haiyang Shen, Junyu Luo, Binqi Chen, Hongjun Ding, Jinsheng Huang, Luchen Liu, Yun Ma, Ming Zhang  
  _2026-02-12_ · https://arxiv.org/abs/2602.11917v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Extracting signals through alpha factor mining is a fundamental challenge in quantitative finance. Existing automated methods primarily follow two paradigms: Decoupled Factor Generation, which treats factor discovery as isolated events, and Iterative Factor Evolution, which focuses on local parent-child refinements. However, both paradigms lack a global structural view, often treating factor pools as unstructured collections or fragmented chains, which leads to redundant search and limited diversity. To address these limitations, we introduce AlphaPROBE (Alpha Mining via Principled Retrieval and On-graph Biased Evolution), a framework that reframes alpha mining as the strategic navigation of a Directed Acyclic Graph (DAG). By modeling factors as nodes and evolutionary links as edges, AlphaPROBE treats the factor pool as a dynamic, interconnected ecosystem. The framework consists of two core components: a Bayesian Factor Retriever that identifies high-potential seeds by balancing exploitation and exploration through a posterior probability model, and a DAG-aware Factor Generator that leverages the full ancestral trace of factors to produce context-aware, nonredundant optimizations. Extensive experiments on three major Chinese stock market datasets against 8 competitive baselines demonstrate that AlphaPROBE significantly gains enhanced performance in predictive accuracy, return stability and training efficiency. Our results confirm that leveraging global evolutionary topology is essential for efficient and robust automated alpha discovery. We have open-sourced our implementation at https://github.com/gta0804/AlphaPROBE.

  </details>


