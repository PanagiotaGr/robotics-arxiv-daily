# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-24 07:13 UTC_

Total papers shown: **15**


---

- **To Move or Not to Move: Constraint-based Planning Enables Zero-Shot Generalization for Interactive Navigation**  
  Apoorva Vashisth, Manav Kulshrestha, Pranav Bakshi, Damon Conover, Guillaume Sartoretti, Aniket Bera  
  _2026-02-23_ · https://arxiv.org/abs/2602.20055v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual navigation typically assumes the existence of at least one obstacle-free path between start and goal, which must be discovered/planned by the robot. However, in real-world scenarios, such as home environments and warehouses, clutter can block all routes. Targeted at such cases, we introduce the Lifelong Interactive Navigation problem, where a mobile robot with manipulation abilities can move clutter to forge its own path to complete sequential object- placement tasks - each involving placing an given object (eg. Alarm clock, Pillow) onto a target object (eg. Dining table, Desk, Bed). To address this lifelong setting - where effects of environment changes accumulate and have long-term effects - we propose an LLM-driven, constraint-based planning framework with active perception. Our framework allows the LLM to reason over a structured scene graph of discovered objects and obstacles, deciding which object to move, where to place it, and where to look next to discover task-relevant information. This coupling of reasoning and active perception allows the agent to explore the regions expected to contribute to task completion rather than exhaustively mapping the environment. A standard motion planner then executes the corresponding navigate-pick-place, or detour sequence, ensuring reliable low-level control. Evaluated in physics-enabled ProcTHOR-10k simulator, our approach outperforms non-learning and learning-based baselines. We further demonstrate our approach qualitatively on real-world hardware.

  </details>



- **Universal Pose Pretraining for Generalizable Vision-Language-Action Policies**  
  Haitao Lin, Hanyang Yu, Jingshun Huang, He Zhang, Yonggen Ling, Ping Tan, Xiangyang Xue, Yanwei Fu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19710v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Existing Vision-Language-Action (VLA) models often suffer from feature collapse and low training efficiency because they entangle high-level perception with sparse, embodiment-specific action supervision. Since these models typically rely on VLM backbones optimized for Visual Question Answering (VQA), they excel at semantic identification but often overlook subtle 3D state variations that dictate distinct action patterns. To resolve these misalignments, we propose Pose-VLA, a decoupled paradigm that separates VLA training into a pre-training phase for extracting universal 3D spatial priors in a unified camera-centric space, and a post-training phase for efficient embodiment alignment within robot-specific action space. By introducing discrete pose tokens as a universal representation, Pose-VLA seamlessly integrates spatial grounding from diverse 3D datasets with geometry-level trajectories from robotic demonstrations. Our framework follows a two-stage pre-training pipeline, establishing fundamental spatial grounding via poses followed by motion alignment through trajectory supervision. Extensive evaluations demonstrate that Pose-VLA achieves state-of-the-art results on RoboTwin 2.0 with a 79.5% average success rate and competitive performance on LIBERO at 96.0%. Real-world experiments further showcase robust generalization across diverse objects using only 100 demonstrations per task, validating the efficiency of our pre-training paradigm.

  </details>



- **Agentic AI for Scalable and Robust Optical Systems Control**  
  Zehao Wang, Mingzhe Han, Wei Cheng, Yue-Kai Huang, Philip Ji, Denton Wu, Mahdi Safari, Flemming Holtorf, Kenaish AlQubaisi, Norbert M. Linke, et al.  
  _2026-02-23_ · https://arxiv.org/abs/2602.20144v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  We present AgentOptics, an agentic AI framework for high-fidelity, autonomous optical system control built on the Model Context Protocol (MCP). AgentOptics interprets natural language tasks and executes protocol-compliant actions on heterogeneous optical devices through a structured tool abstraction layer. We implement 64 standardized MCP tools across 8 representative optical devices and construct a 410-task benchmark to evaluate request understanding, role-aware responses, multi-step coordination, robustness to linguistic variation, and error handling. We assess two deployment configurations--commercial online LLMs and locally hosted open-source LLMs--and compare them with LLM-based code generation baselines. AgentOptics achieves 87.7%--99.0% average task success rates, significantly outperforming code-generation approaches, which reach up to 50% success. We further demonstrate broader applicability through five case studies extending beyond device-level control to system orchestration, monitoring, and closed-loop optimization. These include DWDM link provisioning and coordinated monitoring of coherent 400 GbE and analog radio-over-fiber (ARoF) channels; autonomous characterization and bias optimization of a wideband ARoF link carrying 5G fronthaul traffic; multi-span channel provisioning with launch power optimization; closed-loop fiber polarization stabilization; and distributed acoustic sensing (DAS)-based fiber monitoring with LLM-assisted event detection. These results establish AgentOptics as a scalable, robust paradigm for autonomous control and orchestration of heterogeneous optical systems.

  </details>



- **Open-vocabulary 3D scene perception in industrial environments**  
  Keno Moenck, Adrian Philip Florea, Julian Koch, Thorsten Schüppstuhl  
  _2026-02-23_ · https://arxiv.org/abs/2602.19823v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous vision applications in production, intralogistics, or manufacturing environments require perception capabilities beyond a small, fixed set of classes. Recent open-vocabulary methods, leveraging 2D Vision-Language Foundation Models (VLFMs), target this task but often rely on class-agnostic segmentation models pre-trained on non-industrial datasets (e.g., household scenes). In this work, we first demonstrate that such models fail to generalize, performing poorly on common industrial objects. Therefore, we propose a training-free, open-vocabulary 3D perception pipeline that overcomes this limitation. Instead of using a pre-trained model to generate instance proposals, our method simply generates masks by merging pre-computed superpoints based on their semantic features. Following, we evaluate the domain-adapted VLFM "IndustrialCLIP" on a representative 3D industrial workshop scene for open-vocabulary querying. Our qualitative results demonstrate successful segmentation of industrial objects.

  </details>



- **OpenClaw, Moltbook, and ClawdLab: From Agent-Only Social Networks to Autonomous Scientific Research**  
  Lukas Weidener, Marko Brkić, Mihailo Jovanović, Ritvik Singh, Emre Ulgac, Aakaash Meduri  
  _2026-02-23_ · https://arxiv.org/abs/2602.19810v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  In January 2026, the open-source agent framework OpenClaw and the agent-only social network Moltbook produced a large-scale dataset of autonomous AI-to-AI interaction, attracting six academic publications within fourteen days. This study conducts a multivocal literature review of that ecosystem and presents ClawdLab, an open-source platform for autonomous scientific research, as a design science response to the architectural failure modes identified. The literature documents emergent collective phenomena, security vulnerabilities spanning 131 agent skills and over 15,200 exposed control panels, and five recurring architectural patterns. ClawdLab addresses these failure modes through hard role restrictions, structured adversarial critique, PI-led governance, multi-model orchestration, and domain-specific evidence requirements encoded as protocol constraints that ground validation in computational tool outputs rather than social consensus; the architecture provides emergent Sybil resistance as a structural consequence. A three-tier taxonomy distinguishes single-agent pipelines, predetermined multi-agent workflows, and fully decentralised systems, analysing why leading AI co-scientist platforms remain confined to the first two tiers. ClawdLab's composable third-tier architecture, in which foundation models, capabilities, governance, and evidence requirements are independently modifiable, enables compounding improvement as the broader AI ecosystem advances.

  </details>



- **CACTO-BIC: Scalable Actor-Critic Learning via Biased Sampling and GPU-Accelerated Trajectory Optimization**  
  Elisa Alboni, Pietro Noah Crestaz, Elias Fontanari, Andrea Del Prete  
  _2026-02-23_ · https://arxiv.org/abs/2602.19699v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Trajectory Optimization (TO) and Reinforcement Learning (RL) offer complementary strengths for solving optimal control problems. TO efficiently computes locally optimal solutions but can struggle with non-convexity, while RL is more robust to non-convexity at the cost of significantly higher computational demands. CACTO (Continuous Actor-Critic with Trajectory Optimization) was introduced to combine these advantages by learning a warm-start policy that guides the TO solver towards low-cost trajectories. However, scalability remains a key limitation, as increasing system complexity significantly raises the computational cost of TO. This work introduces CACTO-BIC to address these challenges. CACTO-BIC improves data efficiency by biasing initial-state sampling leveraging a property of the value function associated with locally optimal policies; moreover, it reduces computation time by exploiting GPU acceleration. Empirical evaluations show improved sample efficiency and faster computation compared to CACTO. Comparisons with PPO demonstrate that our approach can achieve similar solutions in less time. Finally, experiments on the AlienGO quadruped robot demonstrate that CACTO-BIC can scale to high-dimensional systems and is suitable for real-time applications.

  </details>



- **BarrierSteer: LLM Safety via Learning Barrier Steering**  
  Thanh Q. Tran, Arun Verma, Kiwan Wong, Bryan Kian Hsiang Low, Daniela Rus, Wei Xiao  
  _2026-02-23_ · https://arxiv.org/abs/2602.20102v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Despite the state-of-the-art performance of large language models (LLMs) across diverse tasks, their susceptibility to adversarial attacks and unsafe content generation remains a major obstacle to deployment, particularly in high-stakes settings. Addressing this challenge requires safety mechanisms that are both practically effective and supported by rigorous theory. We introduce BarrierSteer, a novel framework that formalizes response safety by embedding learned non-linear safety constraints directly into the model's latent representation space. BarrierSteer employs a steering mechanism based on Control Barrier Functions (CBFs) to efficiently detect and prevent unsafe response trajectories during inference with high precision. By enforcing multiple safety constraints through efficient constraint merging, without modifying the underlying LLM parameters, BarrierSteer preserves the model's original capabilities and performance. We provide theoretical results establishing that applying CBFs in latent space offers a principled and computationally efficient approach to enforcing safety. Our experiments across multiple models and datasets show that BarrierSteer substantially reduces adversarial success rates, decreases unsafe generations, and outperforms existing methods.

  </details>



- **The LLMbda Calculus: AI Agents, Conversations, and Information Flow**  
  Zac Garby, Andrew D. Gordon, David Sands  
  _2026-02-23_ · https://arxiv.org/abs/2602.20064v1 · `cs.PL`  
  <details><summary>Abstract</summary>

  A conversation with a large language model (LLM) is a sequence of prompts and responses, with each response generated from the preceding conversation. AI agents build such conversations automatically: given an initial human prompt, a planner loop interleaves LLM calls with tool invocations and code execution. This tight coupling creates a new and poorly understood attack surface. A malicious prompt injected into a conversation can compromise later reasoning, trigger dangerous tool calls, or distort final outputs. Despite the centrality of such systems, we currently lack a principled semantic foundation for reasoning about their behaviour and safety. We address this gap by introducing an untyped call-by-value lambda calculus enriched with dynamic information-flow control and a small number of primitives for constructing prompt-response conversations. Our language includes a primitive that invokes an LLM: it serializes a value, sends it to the model as a prompt, and parses the response as a new term. This calculus faithfully represents planner loops and their vulnerabilities, including the mechanisms by which prompt injection alters subsequent computation. The semantics explicitly captures conversations, and so supports reasoning about defenses such as quarantined sub-conversations, isolation of generated code, and information-flow restrictions on what may influence an LLM call. A termination-insensitive noninterference theorem establishes integrity and confidentiality guarantees, demonstrating that a formal calculus can provide rigorous foundations for safe agentic programming.

  </details>



- **Interaction Theater: A case of LLM Agents Interacting at Scale**  
  Sarath Shekkizhar, Adam Earle  
  _2026-02-23_ · https://arxiv.org/abs/2602.20059v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  As multi-agent architectures and agent-to-agent protocols proliferate, a fundamental question arises: what actually happens when autonomous LLM agents interact at scale? We study this question empirically using data from Moltbook, an AI-agent-only social platform, with 800K posts, 3.5M comments, and 78K agent profiles. We combine lexical metrics (Jaccard specificity), embedding-based semantic similarity, and LLM-as-judge validation to characterize agent interaction quality. Our findings reveal agents produce diverse, well-formed text that creates the surface appearance of active discussion, but the substance is largely absent. Specifically, while most agents ($67.5\%$) vary their output across contexts, $65\%$ of comments share no distinguishing content vocabulary with the post they appear under, and information gain from additional comments decays rapidly. LLM judge based metrics classify the dominant comment types as spam ($28\%$) and off-topic content ($22\%$). Embedding-based semantic analysis confirms that lexically generic comments are also semantically generic. Agents rarely engage in threaded conversation ($5\%$ of comments), defaulting instead to independent top-level responses. We discuss implications for multi-agent interaction design, arguing that coordination mechanisms must be explicitly designed; without them, even large populations of capable agents produce parallel output rather than productive exchange.

  </details>



- **AdaWorldPolicy: World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation**  
  Ge Yuan, Qiyuan Qiao, Jing Zhang, Dong Xu  
  _2026-02-23_ · https://arxiv.org/abs/2602.20057v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. In this work, we introduce a unified framework, World-Model-Driven Diffusion Policy with Online Adaptive Learning (AdaWorldPolicy) to enhance robotic manipulation under dynamic conditions with minimal human involvement. Our core insight is that world models provide strong supervision signals, enabling online adaptive learning in dynamic environments, which can be complemented by force-torque feedback to mitigate dynamic force shifts. Our AdaWorldPolicy integrates a world model, an action expert, and a force predictor-all implemented as interconnected Flow Matching Diffusion Transformers (DiT). They are interconnected via the multi-modal self-attention layers, enabling deep feature exchange for joint learning while preserving their distinct modularity characteristics. We further propose a novel Online Adaptive Learning (AdaOL) strategy that dynamically switches between an Action Generation mode and a Future Imagination mode to drive reactive updates across all three modules. This creates a powerful closed-loop mechanism that adapts to both visual and physical domain shifts with minimal overhead. Across a suite of simulated and real-robot benchmarks, our AdaWorldPolicy achieves state-of-the-art performance, with dynamical adaptive capacity to out-of-distribution scenarios.

  </details>



- **AgenticSum: An Agentic Inference-Time Framework for Faithful Clinical Text Summarization**  
  Fahmida Liza Piya, Rahmatollah Beheshti  
  _2026-02-23_ · https://arxiv.org/abs/2602.20040v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) offer substantial promise for automating clinical text summarization, yet maintaining factual consistency remains challenging due to the length, noise, and heterogeneity of clinical documentation. We present AgenticSum, an inference-time, agentic framework that separates context selection, generation, verification, and targeted correction to reduce hallucinated content. The framework decomposes summarization into coordinated stages that compress task-relevant context, generate an initial draft, identify weakly supported spans using internal attention grounding signals, and selectively revise flagged content under supervisory control. We evaluate AgenticSum on two public datasets, using reference-based metrics, LLM-as-a-judge assessment, and human evaluation. Across various measures, AgenticSum demonstrates consistent improvements compared to vanilla LLMs and other strong baselines. Our results indicate that structured, agentic design with targeted correction offers an effective inference time solution to improve clinical note summarization using LLMs.

  </details>



- **A Secure and Private Distributed Bayesian Federated Learning Design**  
  Nuocheng Yang, Sihua Wang, Zhaohui Yang, Mingzhe Chen, Changchuan Yin, Kaibin Huang  
  _2026-02-23_ · https://arxiv.org/abs/2602.20003v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Distributed Federated Learning (DFL) enables decentralized model training across large-scale systems without a central parameter server. However, DFL faces three critical challenges: privacy leakage from honest-but-curious neighbors, slow convergence due to the lack of central coordination, and vulnerability to Byzantine adversaries aiming to degrade model accuracy. To address these issues, we propose a novel DFL framework that integrates Byzantine robustness, privacy preservation, and convergence acceleration. Within this framework, each device trains a local model using a Bayesian approach and independently selects an optimal subset of neighbors for posterior exchange. We formulate this neighbor selection as an optimization problem to minimize the global loss function under security and privacy constraints. Solving this problem is challenging because devices only possess partial network information, and the complex coupling between topology, security, and convergence remains unclear. To bridge this gap, we first analytically characterize the trade-offs between dynamic connectivity, Byzantine detection, privacy levels, and convergence speed. Leveraging these insights, we develop a fully distributed Graph Neural Network (GNN)-based Reinforcement Learning (RL) algorithm. This approach enables devices to make autonomous connection decisions based on local observations. Simulation results demonstrate that our method achieves superior robustness and efficiency with significantly lower overhead compared to traditional security and privacy schemes.

  </details>



- **Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation**  
  Filip Wolf, Blaž Rolih, Luka Čehovin Zajc  
  _2026-02-23_ · https://arxiv.org/abs/2602.19863v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Foundation models are transforming Earth Observation (EO), yet the diversity of EO sensors and modalities makes a single universal model unrealistic. Multiple specialized EO foundation models (EOFMs) will likely coexist, making efficient knowledge transfer across modalities essential. Most existing EO pretraining relies on masked image modeling, which emphasizes local reconstruction but provides limited control over global semantic structure. To address this, we propose a dual-teacher contrastive distillation framework for multispectral imagery that aligns the student's pretraining objective with the contrastive self-distillation paradigm of modern optical vision foundation models (VFMs). Our approach combines a multispectral teacher with an optical VFM teacher, enabling coherent cross-modal representation learning. Experiments across diverse optical and multispectral benchmarks show that our model adapts to multispectral data without compromising performance on optical-only inputs, achieving state-of-the-art results in both settings, with an average improvement of 3.64 percentage points in semantic segmentation, 1.2 in change detection, and 1.31 in classification tasks. This demonstrates that contrastive distillation provides a principled and efficient approach to scalable representation learning across heterogeneous EO data sources. Code: Coming soon.

  </details>



- **TextShield-R1: Reinforced Reasoning for Tampered Text Detection**  
  Chenfan Qu, Yiwu Zhong, Jian Liu, Xuekang Zhu, Bohan Yu, Lianwen Jin  
  _2026-02-23_ · https://arxiv.org/abs/2602.19828v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The growing prevalence of tampered images poses serious security threats, highlighting the urgent need for reliable detection methods. Multimodal large language models (MLLMs) demonstrate strong potential in analyzing tampered images and generating interpretations. However, they still struggle with identifying micro-level artifacts, exhibit low accuracy in localizing tampered text regions, and heavily rely on expensive annotations for forgery interpretation. To this end, we introduce TextShield-R1, the first reinforcement learning based MLLM solution for tampered text detection and reasoning. Specifically, our approach introduces Forensic Continual Pre-training, an easy-to-hard curriculum that well prepares the MLLM for tampered text detection by harnessing the large-scale cheap data from natural image forensic and OCR tasks. During fine-tuning, we perform Group Relative Policy Optimization with novel reward functions to reduce annotation dependency and improve reasoning capabilities. At inference time, we enhance localization accuracy via OCR Rectification, a method that leverages the MLLM's strong text recognition abilities to refine its predictions. Furthermore, to support rigorous evaluation, we introduce the Text Forensics Reasoning (TFR) benchmark, comprising over 45k real and tampered images across 16 languages, 10 tampering techniques, and diverse domains. Rich reasoning-style annotations are included, allowing for comprehensive assessment. Our TFR benchmark simultaneously addresses seven major limitations of existing benchmarks and enables robust evaluation under cross-style, cross-method, and cross-language conditions. Extensive experiments demonstrate that TextShield-R1 significantly advances the state of the art in interpretable tampered text detection.

  </details>



- **Advantage-based Temporal Attack in Reinforcement Learning**  
  Shenghong He  
  _2026-02-23_ · https://arxiv.org/abs/2602.19582v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Extensive research demonstrates that Deep Reinforcement Learning (DRL) models are susceptible to adversarially constructed inputs (i.e., adversarial examples), which can mislead the agent to take suboptimal or unsafe actions. Recent methods improve attack effectiveness by leveraging future rewards to guide adversarial perturbation generation over sequential time steps (i.e., reward-based attacks). However, these methods are unable to capture dependencies between different time steps in the perturbation generation process, resulting in a weak temporal correlation between the current perturbation and previous perturbations.In this paper, we propose a novel method called Advantage-based Adversarial Transformer (AAT), which can generate adversarial examples with stronger temporal correlations (i.e., time-correlated adversarial examples) to improve the attack performance. AAT employs a multi-scale causal self-attention (MSCSA) mechanism to dynamically capture dependencies between historical information from different time periods and the current state, thus enhancing the correlation between the current perturbation and the previous perturbation. Moreover, AAT introduces a weighted advantage mechanism, which quantifies the effectiveness of a perturbation in a given state and guides the generation process toward high-performance adversarial examples by sampling high-advantage regions. Extensive experiments demonstrate that the performance of AAT matches or surpasses mainstream adversarial attack baselines on Atari, DeepMind Control Suite and Google football tasks.

  </details>


