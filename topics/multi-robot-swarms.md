# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-03-13 07:08 UTC_

Total papers shown: **7**


---

- **Decentralized Cooperative Localization for Multi-Robot Systems with Asynchronous Sensor Fusion**  
  Nivand Khosravi, Niusha Khosravi, Mohammad Bozorg, Masoud S. Bahraini  
  _2026-03-12_ · https://arxiv.org/abs/2603.12075v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Decentralized cooperative localization (DCL) is a promising approach for nonholonomic mobile robots operating in GPS-denied environments with limited communication infrastructure. This paper presents a DCL framework in which each robot performs localization locally using an Extended Kalman Filter, while sharing measurement information during update stages only when communication links are available and companion robots are successfully detected by LiDAR. The framework preserves cross-correlation consistency among robot state estimates while handling asynchronous sensor data with heterogeneous sampling rates and accommodating accelerations during dynamic maneuvers. Unlike methods that require pre-aligned coordinate systems, the proposed approach allows robots to initialize with arbitrary reference-frame orientations and achieves automatic alignment through transformation matrices in both the prediction and update stages. To improve robustness in feature-sparse environments, we introduce a dual-landmark evaluation framework that exploits both static environmental features and mobile robots as dynamic landmarks. The proposed framework enables reliable detection and feature extraction during sharp turns, while prediction accuracy is improved through information sharing from mutual observations. Experimental results in both Gazebo simulation and real-world basement environments show that DCL outperforms centralized cooperative localization (CCL), achieving a 34% reduction in RMSE, while the dual-landmark variant yields an improvement of 56%. These results demonstrate the applicability of DCL to challenging domains such as enclosed spaces, underwater environments, and feature-sparse terrains where conventional localization methods are ineffective.

  </details>



- **Decentralized Orchestration Architecture for Fluid Computing: A Secure Distributed AI Use Case**  
  Diego Cajaraville-Aboy, Ana Fernández-Vilas, Rebeca P. Díaz-Redondo, Manuel Fernández-Veiga, Pablo Picallo-López  
  _2026-03-12_ · https://arxiv.org/abs/2603.12001v1 · `cs.DC`  
  <details><summary>Abstract</summary>

  Distributed AI and IoT applications increasingly execute across heterogeneous resources spanning end devices, edge/fog infrastructure, and cloud platforms, often under different administrative domains. Fluid Computing has emerged as a promising paradigm for enhancing massive resource management across the computing continuum by treating such resources as a unified fabric, enabling optimal service-agnostic deployments driven by application requirements. However, existing solutions remain largely centralized and often do not explicitly address multi-domain considerations. This paper proposes an agnostic multi-domain orchestration architecture for fluid computing environments. The orchestration plane enables decentralized coordination among domains that maintain local autonomy while jointly realizing intent-based deployment requests from tenants, ensuring end-to-end placement and execution. To this end, the architecture elevates domain-side control services as first-class capabilities to support application-level enhancement at runtime. As a representative use case, we consider a multi-domain Decentralized Federated Learning (DFL) deployment under Byzantine threats. We leverage domain-side capabilities to enhance Byzantine security by introducing FU-HST, an SDN-enabled multi-domain anomaly detection mechanism that complements Byzantine-robust aggregation. We validate the approach via simulation in single- and multi-domain settings, evaluating anomaly detection, DFL performance, and computation/communication overhead.

  </details>



- **Learning Visuomotor Policy for Multi-Robot Laser Tag Game**  
  Kai Li, Shiyu Zhao  
  _2026-03-12_ · https://arxiv.org/abs/2603.11980v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In this paper, we study multi robot laser tag, a simplified yet practical shooting-game-style task. Classic modular approaches on these tasks face challenges such as limited observability and reliance on depth mapping and inter robot communication. To overcome these issues, we present an end-to-end visuomotor policy that maps images directly to robot actions. We train a high performing teacher policy with multi agent reinforcement learning and distill its knowledge into a vision-based student policy. Technical designs, including a permutation-invariant feature extractor and depth heatmap input, improve performance over standard architectures. Our policy outperforms classic methods by 16.7% in hitting accuracy and 6% in collision avoidance, and is successfully deployed on real robots. Code will be released publicly.

  </details>



- **Security Considerations for Artificial Intelligence Agents**  
  Ninghui Li, Kaiyuan Zhang, Kyle Polley, Jerry Ma  
  _2026-03-12_ · https://arxiv.org/abs/2603.12230v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  This article, a lightly adapted version of Perplexity's response to NIST/CAISI Request for Information 2025-0035, details our observations and recommendations concerning the security of frontier AI agents. These insights are informed by Perplexity's experience operating general-purpose agentic systems used by millions of users and thousands of enterprises in both controlled and open-world environments. Agent architectures change core assumptions around code-data separation, authority boundaries, and execution predictability, creating new confidentiality, integrity, and availability failure modes. We map principal attack surfaces across tools, connectors, hosting boundaries, and multi-agent coordination, with particular emphasis on indirect prompt injection, confused-deputy behavior, and cascading failures in long-running workflows. We then assess current defenses as a layered stack: input-level and model-level mitigations, sandboxed execution, and deterministic policy enforcement for high-consequence actions. Finally, we identify standards and research gaps, including adaptive security benchmarks, policy models for delegation and privilege control, and guidance for secure multi-agent system design aligned with NIST risk management principles.

  </details>



- **WORKSWORLD: A Domain for Integrated Numeric Planning and Scheduling of Distributed Pipelined Workflows**  
  Taylor Paul, William Regli  
  _2026-03-12_ · https://arxiv.org/abs/2603.12214v1 · `cs.DC`  
  <details><summary>Abstract</summary>

  This work pursues automated planning and scheduling of distributed data pipelines, or workflows. We develop a general workflow and resource graph representation that includes both data processing and sharing components with corresponding network interfaces for scheduling. Leveraging these graphs, we introduce WORKSWORLD, a new domain for numeric domain-independent planners designed for permanently scheduled workflows, like ingest pipelines. Our framework permits users to define data sources, available workflow components, and desired data destinations and formats without explicitly declaring the entire workflow graph as a goal. The planner solves a joint planning and scheduling problem, producing a plan that both builds the workflow graph and schedules its components on the resource graph. We empirically show that a state-of-the-art numeric planner running on commodity hardware with one hour of CPU time and 30GB of memory can solve linear-chain workflows of up to 14 components across eight sites.

  </details>



- **Cascade: Composing Software-Hardware Attack Gadgets for Adversarial Threat Amplification in Compound AI Systems**  
  Sarbartha Banerjee, Prateek Sahu, Anjo Vahldiek-Oberwagner, Jose Sanchez Vicarte, Mohit Tiwari  
  _2026-03-12_ · https://arxiv.org/abs/2603.12023v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Rapid progress in generative AI has given rise to Compound AI systems - pipelines comprised of multiple large language models (LLM), software tools and database systems. Compound AI systems are constructed on a layered traditional software stack running on a distributed hardware infrastructure. Many of the diverse software components are vulnerable to traditional security flaws documented in the Common Vulnerabilities and Exposures (CVE) database, while the underlying distributed hardware infrastructure remains exposed to timing attacks, bit-flip faults, and power-based side channels. Today, research targets LLM-specific risks like model extraction, training data leakage, and unsafe generation -- overlooking the impact of traditional system vulnerabilities. This work investigates how traditional software and hardware vulnerabilities can complement LLM-specific algorithmic attacks to compromise the integrity of a compound AI pipeline. We demonstrate two novel attacks that combine system-level vulnerabilities with algorithmic weaknesses: (1) Exploiting a software code injection flaw along with a guardrail Rowhammer attack to inject an unaltered jailbreak prompt into an LLM, resulting in an AI safety violation, and (2) Manipulating a knowledge database to redirect an LLM agent to transmit sensitive user data to a malicious application, thus breaching confidentiality. These attacks highlight the need to address traditional vulnerabilities; we systematize the attack primitives and analyze their composition by grouping vulnerabilities by their objective and mapping them to distinct stages of an attack lifecycle. This approach enables a rigorous red-teaming exercise and lays the groundwork for future defense strategies.

  </details>



- **Hybrid Human-Agent Social Dilemmas in Energy Markets**  
  Isuri Perera, Frits de Nijs, Julian Garcia  
  _2026-03-12_ · https://arxiv.org/abs/2603.11834v1 · `cs.MA`  
  <details><summary>Abstract</summary>

  In hybrid populations where humans delegate strategic decision-making to autonomous agents, understanding when and how cooperative behaviors can emerge remains a key challenge. We study this problem in the context of energy load management: consumer agents schedule their appliance use under demand-dependent pricing. This structure can create a social dilemma where everybody would benefit from coordination, but in equilibrium agents often choose to incur the congestion costs that cooperative turn-taking would avoid. To address the problem of coordination, we introduce artificial agents that use globally observable signals to increase coordination. Using evolutionary dynamics, and reinforcement learning experiments, we show that artificial agents can shift the learning dynamics to favour coordination outcomes. An often neglected problem is partial adoption: what happens when the technology of artificial agents is in the early adoption stages? We analyze mixed populations of adopters and non-adopters, demonstrating that unilateral entry is feasible: adopters are not structurally penalized, and partial adoption can still improve aggregate outcomes. However, in some parameter regimes, non-adopters may benefit disproportionately from the cooperation induced by adopters. This asymmetry, while not precluding beneficial entry, warrants consideration in deployment, and highlights strategic issues around the adoption of AI technology in multiagent settings.

  </details>


