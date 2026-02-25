# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-25 07:14 UTC_

Total papers shown: **3**


---

- **Cooperative-Competitive Team Play of Real-World Craft Robots**  
  Rui Zhao, Xihui Li, Yizheng Zhang, Yuzhen Liu, Zhong Zhang, Yufeng Zhang, Cheng Zhou, Zhengyou Zhang, Lei Han  
  _2026-02-24_ · https://arxiv.org/abs/2602.21119v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-agent deep Reinforcement Learning (RL) has made significant progress in developing intelligent game-playing agents in recent years. However, the efficient training of collective robots using multi-agent RL and the transfer of learned policies to real-world applications remain open research questions. In this work, we first develop a comprehensive robotic system, including simulation, distributed learning framework, and physical robot components. We then propose and evaluate reinforcement learning techniques designed for efficient training of cooperative and competitive policies on this platform. To address the challenges of multi-agent sim-to-real transfer, we introduce Out of Distribution State Initialization (OODSI) to mitigate the impact of the sim-to-real gap. In the experiments, OODSI improves the Sim2Real performance by 20%. We demonstrate the effectiveness of our approach through experiments with a multi-robot car competitive game and a cooperative task in real-world settings.

  </details>



- **Architecting AgentOS: From Token-Level Context to Emergent System-Level Intelligence**  
  ChengYou Li, XiaoDong Liu, XiangBao Meng, XinYu Zhao  
  _2026-02-24_ · https://arxiv.org/abs/2602.20934v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The paradigm of Large Language Models is undergoing a fundamental transition from static inference engines to dynamic autonomous cognitive systems.While current research primarily focuses on scaling context windows or optimizing prompt engineering the theoretical bridge between micro scale token processing and macro scale systemic intelligence remains fragmented.This paper proposes AgentOS,a holistic conceptual framework that redefines the LLM as a "Reasoning Kernel" governed by structured operating system logic.Central to this architecture is Deep Context Management which conceptualizes the context window as an Addressable Semantic Space rather than a passive buffer.We systematically deconstruct the transition from discrete sequences to coherent cognitive states introducing mechanisms for Semantic Slicing and Temporal Alignment to mitigate cognitive drift in multi-agent orchestration.By mapping classical OS abstractions such as memory paging interrupt handling and process scheduling onto LLM native constructs, this review provides a rigorous roadmap for architecting resilient scalable and self-evolving cognitive environments.Our analysis asserts that the next frontier of AGI development lies in the architectural efficiency of system-level coordination.

  </details>



- **A Micro-Macro Model of Encounter-Driven Information Diffusion in Robot Swarms**  
  Davis S. Catherman, Carlo Pinciroli  
  _2026-02-24_ · https://arxiv.org/abs/2602.21148v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In this paper, we propose the problem of Encounter-Driven Information Diffusion (EDID). In EDID, robots are allowed to exchange information only upon meeting. Crucially, EDID assumes that the robots are not allowed to schedule their meetings. As such, the robots have no means to anticipate when, where, and who they will meet. As a step towards the design of storage and routing algorithms for EDID, in this paper we propose a model of information diffusion that captures the essential dynamics of EDID. The model is derived from first principles and is composed of two levels: a micro model, based on a generalization of the concept of `mean free path'; and a macro model, which captures the global dynamics of information diffusion. We validate the model through extensive robot simulations, in which we consider swarm size, communication range, environment size, and different random motion regimes. We conclude the paper with a discussion of the implications of this model on the algorithms that best support information diffusion according to the parameters of interest.

  </details>


