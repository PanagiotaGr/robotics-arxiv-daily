# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-24 07:13 UTC_

Total papers shown: **8**


---

- **Chasing Ghosts: A Simulation-to-Real Olfactory Navigation Stack with Optional Vision Augmentation**  
  Kordel K. France, Ovidiu Daescu, Latifur Khan, Rohith Peddi  
  _2026-02-23_ · https://arxiv.org/abs/2602.19577v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous odor source localization remains a challenging problem for aerial robots due to turbulent airflow, sparse and delayed sensory signals, and strict payload and compute constraints. While prior unmanned aerial vehicle (UAV)-based olfaction systems have demonstrated gas distribution mapping or reactive plume tracing, they rely on predefined coverage patterns, external infrastructure, or extensive sensing and coordination. In this work, we present a complete, open-source UAV system for online odor source localization using a minimal sensor suite. The system integrates custom olfaction hardware, onboard sensing, and a learning-based navigation policy trained in simulation and deployed on a real quadrotor. Through our minimal framework, the UAV is able to navigate directly toward an odor source without constructing an explicit gas distribution map or relying on external positioning systems. Vision is incorporated as an optional complementary modality to accelerate navigation under certain conditions. We validate the proposed system through real-world flight experiments in a large indoor environment using an ethanol source, demonstrating consistent source-finding behavior under realistic airflow conditions. The primary contribution of this work is a reproducible system and methodological framework for UAV-based olfactory navigation and source finding under minimal sensing assumptions. We elaborate on our hardware design and open source our UAV firmware, simulation code, olfaction-vision dataset, and circuit board to the community. Code, data, and designs will be made available at https://github.com/KordelFranceTech/ChasingGhosts.

  </details>



- **Agentic AI for Scalable and Robust Optical Systems Control**  
  Zehao Wang, Mingzhe Han, Wei Cheng, Yue-Kai Huang, Philip Ji, Denton Wu, Mahdi Safari, Flemming Holtorf, Kenaish AlQubaisi, Norbert M. Linke, et al.  
  _2026-02-23_ · https://arxiv.org/abs/2602.20144v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  We present AgentOptics, an agentic AI framework for high-fidelity, autonomous optical system control built on the Model Context Protocol (MCP). AgentOptics interprets natural language tasks and executes protocol-compliant actions on heterogeneous optical devices through a structured tool abstraction layer. We implement 64 standardized MCP tools across 8 representative optical devices and construct a 410-task benchmark to evaluate request understanding, role-aware responses, multi-step coordination, robustness to linguistic variation, and error handling. We assess two deployment configurations--commercial online LLMs and locally hosted open-source LLMs--and compare them with LLM-based code generation baselines. AgentOptics achieves 87.7%--99.0% average task success rates, significantly outperforming code-generation approaches, which reach up to 50% success. We further demonstrate broader applicability through five case studies extending beyond device-level control to system orchestration, monitoring, and closed-loop optimization. These include DWDM link provisioning and coordinated monitoring of coherent 400 GbE and analog radio-over-fiber (ARoF) channels; autonomous characterization and bias optimization of a wideband ARoF link carrying 5G fronthaul traffic; multi-span channel provisioning with launch power optimization; closed-loop fiber polarization stabilization; and distributed acoustic sensing (DAS)-based fiber monitoring with LLM-assisted event detection. These results establish AgentOptics as a scalable, robust paradigm for autonomous control and orchestration of heterogeneous optical systems.

  </details>



- **A Secure and Private Distributed Bayesian Federated Learning Design**  
  Nuocheng Yang, Sihua Wang, Zhaohui Yang, Mingzhe Chen, Changchuan Yin, Kaibin Huang  
  _2026-02-23_ · https://arxiv.org/abs/2602.20003v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Distributed Federated Learning (DFL) enables decentralized model training across large-scale systems without a central parameter server. However, DFL faces three critical challenges: privacy leakage from honest-but-curious neighbors, slow convergence due to the lack of central coordination, and vulnerability to Byzantine adversaries aiming to degrade model accuracy. To address these issues, we propose a novel DFL framework that integrates Byzantine robustness, privacy preservation, and convergence acceleration. Within this framework, each device trains a local model using a Bayesian approach and independently selects an optimal subset of neighbors for posterior exchange. We formulate this neighbor selection as an optimization problem to minimize the global loss function under security and privacy constraints. Solving this problem is challenging because devices only possess partial network information, and the complex coupling between topology, security, and convergence remains unclear. To bridge this gap, we first analytically characterize the trade-offs between dynamic connectivity, Byzantine detection, privacy levels, and convergence speed. Leveraging these insights, we develop a fully distributed Graph Neural Network (GNN)-based Reinforcement Learning (RL) algorithm. This approach enables devices to make autonomous connection decisions based on local observations. Simulation results demonstrate that our method achieves superior robustness and efficiency with significantly lower overhead compared to traditional security and privacy schemes.

  </details>



- **OpenClaw, Moltbook, and ClawdLab: From Agent-Only Social Networks to Autonomous Scientific Research**  
  Lukas Weidener, Marko Brkić, Mihailo Jovanović, Ritvik Singh, Emre Ulgac, Aakaash Meduri  
  _2026-02-23_ · https://arxiv.org/abs/2602.19810v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  In January 2026, the open-source agent framework OpenClaw and the agent-only social network Moltbook produced a large-scale dataset of autonomous AI-to-AI interaction, attracting six academic publications within fourteen days. This study conducts a multivocal literature review of that ecosystem and presents ClawdLab, an open-source platform for autonomous scientific research, as a design science response to the architectural failure modes identified. The literature documents emergent collective phenomena, security vulnerabilities spanning 131 agent skills and over 15,200 exposed control panels, and five recurring architectural patterns. ClawdLab addresses these failure modes through hard role restrictions, structured adversarial critique, PI-led governance, multi-model orchestration, and domain-specific evidence requirements encoded as protocol constraints that ground validation in computational tool outputs rather than social consensus; the architecture provides emergent Sybil resistance as a structural consequence. A three-tier taxonomy distinguishes single-agent pipelines, predetermined multi-agent workflows, and fully decentralised systems, analysing why leading AI co-scientist platforms remain confined to the first two tiers. ClawdLab's composable third-tier architecture, in which foundation models, capabilities, governance, and evidence requirements are independently modifiable, enables compounding improvement as the broader AI ecosystem advances.

  </details>



- **Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception**  
  Yihang Tao, Senkang Hu, Haonan An, Zhengru Fang, Hangcheng Cao, Yuguang Fang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19596v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Collaborative perception (CP) enables data sharing among connected and autonomous vehicles (CAVs) to enhance driving safety. However, CP systems are vulnerable to adversarial attacks where malicious agents forge false objects via feature-level perturbations. Current defensive systems use threshold-based consensus verification by comparing collaborative and ego detection results. Yet, these defenses remain vulnerable to more sophisticated attack strategies that could exploit two critical weaknesses: (i) lack of robustness against attacks with systematic timing and target region optimization, and (ii) inadvertent disclosure of vulnerability knowledge through implicit confidence information in shared collaboration data. In this paper, we propose MVIG attack, a novel adaptive adversarial CP framework learning to capture vulnerability knowledge disclosed by different defensive CP systems from a unified mutual view information graph (MVIG) representation. Our approach combines MVIG representation with temporal graph learning to generate evolving fabrication risk maps and employs entropy-aware vulnerability search to optimize attack location, timing and persistence, enabling adaptive attacks with generalizability across various defensive configurations. Extensive evaluations on OPV2V and Adv-OPV2V datasets demonstrate that MVIG attack reduces defense success rates by up to 62\% against state-of-the-art defenses while achieving 47\% lower detection for persistent attacks at 29.9 FPS, exposing critical security gaps in CP systems. Code will be released at https://github.com/yihangtao/MVIG.git

  </details>



- **Interaction Theater: A case of LLM Agents Interacting at Scale**  
  Sarath Shekkizhar, Adam Earle  
  _2026-02-23_ · https://arxiv.org/abs/2602.20059v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  As multi-agent architectures and agent-to-agent protocols proliferate, a fundamental question arises: what actually happens when autonomous LLM agents interact at scale? We study this question empirically using data from Moltbook, an AI-agent-only social platform, with 800K posts, 3.5M comments, and 78K agent profiles. We combine lexical metrics (Jaccard specificity), embedding-based semantic similarity, and LLM-as-judge validation to characterize agent interaction quality. Our findings reveal agents produce diverse, well-formed text that creates the surface appearance of active discussion, but the substance is largely absent. Specifically, while most agents ($67.5\%$) vary their output across contexts, $65\%$ of comments share no distinguishing content vocabulary with the post they appear under, and information gain from additional comments decays rapidly. LLM judge based metrics classify the dominant comment types as spam ($28\%$) and off-topic content ($22\%$). Embedding-based semantic analysis confirms that lexically generic comments are also semantically generic. Agents rarely engage in threaded conversation ($5\%$ of comments), defaulting instead to independent top-level responses. We discuss implications for multi-agent interaction design, arguing that coordination mechanisms must be explicitly designed; without them, even large populations of capable agents produce parallel output rather than productive exchange.

  </details>



- **Scalable Low-Density Distributed Manipulation Using an Interconnected Actuator Array**  
  Bailey Dacre, Rodrigo Moreno, Jørn Lambertsen, Kasper Stoy, Andrés Faíña  
  _2026-02-23_ · https://arxiv.org/abs/2602.19653v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Distributed Manipulator Systems, composed of arrays of robotic actuators necessitate dense actuator arrays to effectively manipulate small objects. This paper presents a system composed of modular 3-DoF robotic tiles interconnected by a compliant surface layer, forming a continuous, controllable manipulation surface. The compliant layer permits increased actuator spacing without compromising object manipulation capabilities, significantly reducing actuator density while maintaining robust control, even for smaller objects. We characterize the coupled workspace of the array and develop a manipulation strategy capable of translating objects to arbitrary positions within an N X N array. The approach is validated experimentally using a minimal 2 X 2 prototype, demonstrating the successful manipulation of objects with varied shapes and sizes.

  </details>



- **Active IoT User Detection in Near-Field with Location Information**  
  Gabriel Martins de Jesus, Richard Demo Souza, Onel Luis Alcaraz López  
  _2026-02-23_ · https://arxiv.org/abs/2602.19613v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  In this paper, we address active users detection (AUD) in near-field Internet of Things (IoT) networks by exploring prior knowledge of users' locations. We consider a scenario where users are distributed in a semi-circular area within the Rayleigh distance of a multi-antenna base station (BS). We propose the BS to use location estimates of the users to reconstruct their line-of-sight (LoS) channel components, hence assisting the AUD process. For this, the BS combines these reconstructed channels with users' pilot sequences, enhancing the correlation between received signals and active users. We formulate the location-aided AUD as a convex optimization problem, solved via the alternating direction method of multipliers (ADMM). {Our proposal has a higher computational complexity compared to the baseline ADMM approach where location information is not used. Moreover, the proposal requires location information of users, which can be readily informed if users are static, or inferred via established localization algorithms if they are mobile.} Simulation results compare our proposal against the baseline across varying systems parameters, such as number of users, pilot length and LoS component strength. We demonstrate that under perfect location estimation and strong LoS, our proposed method significantly outperforms the baseline. Furthermore, robustness analysis shows that performance gains persist under imperfect location estimation, provided the estimation error remains within bounds determined by the system parameters.

  </details>


