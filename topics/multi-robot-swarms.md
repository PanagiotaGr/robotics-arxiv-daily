# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-01-31 06:55 UTC_

Total papers shown: **4**


---

- **Liquid Interfaces: A Dynamic Ontology for the Interoperability of Autonomous Systems**  
  Dhiogo de Sá, Carlos Schmiedel, Carlos Pereira Lopes  
  _2026-01-29_ · https://arxiv.org/abs/2601.21993v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Contemporary software architectures struggle to support autonomous agents whose reasoning is adaptive, probabilistic, and context-dependent, while system integration remains dominated by static interfaces and deterministic contracts. This paper introduces Liquid Interfaces, a coordination paradigm in which interfaces are not persistent technical artifacts, but ephemeral relational events that emerge through intention articulation and semantic negotiation at runtime.We formalize this model and present the Liquid Interface Protocol (LIP),which governs intention-driven interaction, negotiated execution, and enforce ephemerality under semantic uncertainty. We further discuss the governance implications of this approach and describe a reference architecture that demonstrates practical feasibility. Liquid Interfaces provide a principled foundation for adaptive coordination in agent-based systems

  </details>



- **Multi-Modular MANTA-RAY: A Modular Soft Surface Platform for Distributed Multi-Object Manipulation**  
  Pratik Ingle, Jørn Lambertsen, Kasper Støy, Andres Faina  
  _2026-01-29_ · https://arxiv.org/abs/2601.21884v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Manipulation surfaces control objects by actively deforming their shape rather than directly grasping them. While dense actuator arrays can generate complex deformations, they also introduce high degrees of freedom (DOF), increasing system complexity and limiting scalability. The MANTA-RAY (Manipulation with Adaptive Non-rigid Textile Actuation with Reduced Actuation densitY) platform addresses these challenges by leveraging a soft, fabric-based surface with reduced actuator density to manipulate fragile and heterogeneous objects. Previous studies focused on single-module implementations supported by four actuators, whereas the feasibility and benefits of a scalable, multi-module configuration remain unexplored. In this work, we present a distributed, modular, and scalable variant of the MANTA-RAY platform that maintains manipulation performance with a reduced actuator density. The proposed multi-module MANTA-RAY platform and control strategy employs object passing between modules and a geometric transformation driven PID controller that directly maps tilt-angle control outputs to actuator commands, eliminating the need for extensive data-driven or black-box training. We evaluate system performance in simulation across surface configurations of varying modules (3x3 and 4x4) and validate its feasibility through experiments on a physical 2x2 hardware prototype. The system successfully manipulates objects with diverse geometries, masses, and textures including fragile items such as eggs and apples as well as enabling parallel manipulation. The results demonstrate that the multi-module MANTA-RAY improves scalability and enables coordinated manipulation of multiple objects across larger areas, highlighting its potential for practical, real-world applications.

  </details>



- **CORE:Toward Ubiquitous 6G Intelligence Through Collaborative Orchestration of Large Language Model Agents Over Hierarchical Edge**  
  Zitong Yu, Boquan Sun, Yang Li, Zheyan Qu, Xing Zhang  
  _2026-01-29_ · https://arxiv.org/abs/2601.21822v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Rapid advancements in sixth-generation (6G) networks and large language models (LLMs) have paved the way for ubiquitous intelligence, wherein seamless connectivity and distributed artificial intelligence (AI) have revolutionized various aspects of our lives.However, realizing this vision faces significant challenges owing to the fragmented and heterogeneous computing resources across hierarchical networks, which are insufficient for individual LLM agents to perform complex reasoning tasks.To address this issue, we propose Collaborative Orchestration Role at Edge (CORE), an innovative framework that employs a collaborative learning system in which multiple LLMs, each assigned a distinct functional role, are distributed across mobile devices and tiered edge servers. The system integrates three optimization modules, encompassing real-time perception,dynamic role orchestration, and pipeline-parallel execution, to facilitate efficient and rapid collaboration among distributed agents. Furthermore, we introduce a novel role affinity scheduling algorithm for dynamically orchestrating LLM role assignments across the hierarchical edge infrastructure, intelligently matching computational demands with available dispersed resources.Finally, comprehensive case studies and performance evaluations across various 6G application scenarios demonstrated the efficacy of CORE, revealing significant enhancements in the system efficiency and task completion rates. Building on these promising outcomes, we further validated the practical applicability of CORE by deploying it on a real-world edge-computing platform,that exhibits robust performance in operational environments.

  </details>



- **Flocking behavior for dynamic and complex swarm structures**  
  Carmen D. R. Pita-Romero, Pedro Arias-Perez, Miguel Fernandez-Cortizas, Rafael Perez-Segui, Pascual Campoy  
  _2026-01-29_ · https://arxiv.org/abs/2601.21772v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Maintaining the formation of complex structures with multiple UAVs and achieving complex trajectories remains a major challenge. This work presents an algorithm for implementing the flocking behavior of UAVs based on the concept of Virtual Centroid to easily develop a structure for the flock. The approach builds on the classical virtual-based behavior, providing a theoretical framework for incorporating enhancements to dynamically control both the number of agents and the formation of the structure. Simulation tests and real-world experiments were conducted, demonstrating its simplicity even with complex formations and complex trajectories.

  </details>


