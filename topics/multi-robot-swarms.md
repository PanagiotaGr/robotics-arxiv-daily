# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-14 07:00 UTC_

Total papers shown: **4**


---

- **Decentralized Multi-Robot Obstacle Detection and Tracking in a Maritime Scenario**  
  Muhammad Farhan Ahmed, Vincent Frémont  
  _2026-02-12_ · https://arxiv.org/abs/2602.12012v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous aerial-surface robot teams are promising for maritime monitoring. Robust deployment requires reliable perception over reflective water and scalable coordination under limited communication. We present a decentralized multi-robot framework for detecting and tracking floating containers using multiple UAVs cooperating with an autonomous surface vessel. Each UAV performs YOLOv8 and stereo-disparity-based visual detection, then tracks targets with per-object EKFs using uncertainty-aware data association. Compact track summaries are exchanged and fused conservatively via covariance intersection, ensuring consistency under unknown correlations. An information-driven assignment module allocates targets and selects UAV hover viewpoints by trading expected uncertainty reduction against travel effort and safety separation. Simulation results in a maritime scenario demonstrate improved coverage, localization accuracy, and tracking consistency while maintaining modest communication requirements.

  </details>



- **RF-Modulated Adaptive Communication Improves Multi-Agent Robotic Exploration**  
  Lorin Achey, Breanne Crockett, Christoffer Heckman, Bradley Hayes  
  _2026-02-12_ · https://arxiv.org/abs/2602.12074v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable coordination and efficient communication are critical challenges for multi-agent robotic exploration of environments where communication is limited. This work introduces Adaptive-RF Transmission (ART), a novel communication-aware planning algorithm that dynamically modulates transmission location based on signal strength and data payload size, enabling heterogeneous robot teams to share information efficiently without unnecessary backtracking. We further explore an extension to this approach called ART-SST, which enforces signal strength thresholds for high-fidelity data delivery. Through over 480 simulations across three cave-inspired environments, ART consistently outperforms existing strategies, including full rendezvous and minimum-signal heuristic approaches, achieving up to a 58% reduction in distance traveled and up to 52% faster exploration times compared to baseline methods. These results demonstrate that adaptive, payload-aware communication significantly improves coverage efficiency and mission speed in complex, communication-constrained environments, offering a promising foundation for future planetary exploration and search-and-rescue missions.

  </details>



- **Adaptive-Horizon Conflict-Based Search for Closed-Loop Multi-Agent Path Finding**  
  Jiarui Li, Federico Pecora, Runyu Zhang, Gioele Zardini  
  _2026-02-12_ · https://arxiv.org/abs/2602.12024v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  MAPF is a core coordination problem for large robot fleets in automated warehouses and logistics. Existing approaches are typically either open-loop planners, which generate fixed trajectories and struggle to handle disturbances, or closed-loop heuristics without reliable performance guarantees, limiting their use in safety-critical deployments. This paper presents ACCBS, a closed-loop algorithm built on a finite-horizon variant of CBS with a horizon-changing mechanism inspired by iterative deepening in MPC. ACCBS dynamically adjusts the planning horizon based on the available computational budget, and reuses a single constraint tree to enable seamless transitions between horizons. As a result, it produces high-quality feasible solutions quickly while being asymptotically optimal as the budget increases, exhibiting anytime behavior. Extensive case studies demonstrate that ACCBS combines flexibility to disturbances with strong performance guarantees, effectively bridging the gap between theoretical optimality and practical robustness for large-scale robot deployment.

  </details>



- **Resource-Aware Deployment Optimization for Collaborative Intrusion Detection in Layered Networks**  
  André García Gómez, Ines Rieger, Wolfgang Hotwagner, Max Landauer, Markus Wurzenberger, Florian Skopik, Edgar Weippl  
  _2026-02-12_ · https://arxiv.org/abs/2602.11851v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Collaborative Intrusion Detection Systems (CIDS) are increasingly adopted to counter cyberattacks, as their collaborative nature enables them to adapt to diverse scenarios across heterogeneous environments. As distributed critical infrastructure operates in rapidly evolving environments, such as drones in both civil and military domains, there is a growing need for CIDS architectures that can flexibly accommodate these dynamic changes. In this study, we propose a novel CIDS framework designed for easy deployment across diverse distributed environments. The framework dynamically optimizes detector allocation per node based on available resources and data types, enabling rapid adaptation to new operational scenarios with minimal computational overhead. We first conducted a comprehensive literature review to identify key characteristics of existing CIDS architectures. Based on these insights and real-world use cases, we developed our CIDS framework, which we evaluated using several distributed datasets that feature different attack chains and network topologies. Notably, we introduce a public dataset based on a realistic cyberattack targeting a ground drone aimed at sabotaging critical infrastructure. Experimental results demonstrate that the proposed CIDS framework can achieve adaptive, efficient intrusion detection in distributed settings, automatically reconfiguring detectors to maintain an optimal configuration, without requiring heavy computation, since all experiments were conducted on edge devices.

  </details>


