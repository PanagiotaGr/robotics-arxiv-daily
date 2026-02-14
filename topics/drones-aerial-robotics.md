# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-14 07:00 UTC_

Total papers shown: **3**


---

- **Decentralized Multi-Robot Obstacle Detection and Tracking in a Maritime Scenario**  
  Muhammad Farhan Ahmed, Vincent Frémont  
  _2026-02-12_ · https://arxiv.org/abs/2602.12012v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous aerial-surface robot teams are promising for maritime monitoring. Robust deployment requires reliable perception over reflective water and scalable coordination under limited communication. We present a decentralized multi-robot framework for detecting and tracking floating containers using multiple UAVs cooperating with an autonomous surface vessel. Each UAV performs YOLOv8 and stereo-disparity-based visual detection, then tracks targets with per-object EKFs using uncertainty-aware data association. Compact track summaries are exchanged and fused conservatively via covariance intersection, ensuring consistency under unknown correlations. An information-driven assignment module allocates targets and selects UAV hover viewpoints by trading expected uncertainty reduction against travel effort and safety separation. Simulation results in a maritime scenario demonstrate improved coverage, localization accuracy, and tracking consistency while maintaining modest communication requirements.

  </details>



- **Multi UAVs Preflight Planning in a Shared and Dynamic Airspace**  
  Amath Sow, Mauricio Rodriguez Cesen, Fabiola Martins Campos de Oliveira, Mariusz Wzorek, Daniel de Leng, Mattias Tiger, Fredrik Heintz, Christian Esteve Rothenberg  
  _2026-02-12_ · https://arxiv.org/abs/2602.12055v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Preflight planning for large-scale Unmanned Aerial Vehicle (UAV) fleets in dynamic, shared airspace presents significant challenges, including temporal No-Fly Zones (NFZs), heterogeneous vehicle profiles, and strict delivery deadlines. While Multi-Agent Path Finding (MAPF) provides a formal framework, existing methods often lack the scalability and flexibility required for real-world Unmanned Traffic Management (UTM). We propose DTAPP-IICR: a Delivery-Time Aware Prioritized Planning method with Incremental and Iterative Conflict Resolution. Our framework first generates an initial solution by prioritizing missions based on urgency. Secondly, it computes roundtrip trajectories using SFIPP-ST, a novel 4D single-agent planner (Safe Flight Interval Path Planning with Soft and Temporal Constraints). SFIPP-ST handles heterogeneous UAVs, strictly enforces temporal NFZs, and models inter-agent conflicts as soft constraints. Subsequently, an iterative Large Neighborhood Search, guided by a geometric conflict graph, efficiently resolves any residual conflicts. A completeness-preserving directional pruning technique further accelerates the 3D search. On benchmarks with temporal NFZs, DTAPP-IICR achieves near-100% success with fleets of up to 1,000 UAVs and gains up to 50% runtime reduction from pruning, outperforming batch Enhanced Conflict-Based Search in the UTM context. Scaling successfully in realistic city-scale operations where other priority-based methods fail even at moderate deployments, DTAPP-IICR is positioned as a practical and scalable solution for preflight planning in dense, dynamic urban airspace.

  </details>



- **Resource-Aware Deployment Optimization for Collaborative Intrusion Detection in Layered Networks**  
  André García Gómez, Ines Rieger, Wolfgang Hotwagner, Max Landauer, Markus Wurzenberger, Florian Skopik, Edgar Weippl  
  _2026-02-12_ · https://arxiv.org/abs/2602.11851v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Collaborative Intrusion Detection Systems (CIDS) are increasingly adopted to counter cyberattacks, as their collaborative nature enables them to adapt to diverse scenarios across heterogeneous environments. As distributed critical infrastructure operates in rapidly evolving environments, such as drones in both civil and military domains, there is a growing need for CIDS architectures that can flexibly accommodate these dynamic changes. In this study, we propose a novel CIDS framework designed for easy deployment across diverse distributed environments. The framework dynamically optimizes detector allocation per node based on available resources and data types, enabling rapid adaptation to new operational scenarios with minimal computational overhead. We first conducted a comprehensive literature review to identify key characteristics of existing CIDS architectures. Based on these insights and real-world use cases, we developed our CIDS framework, which we evaluated using several distributed datasets that feature different attack chains and network topologies. Notably, we introduce a public dataset based on a realistic cyberattack targeting a ground drone aimed at sabotaging critical infrastructure. Experimental results demonstrate that the proposed CIDS framework can achieve adaptive, efficient intrusion detection in distributed settings, automatically reconfiguring detectors to maintain an optimal configuration, without requiring heavy computation, since all experiments were conducted on edge devices.

  </details>


