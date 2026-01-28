# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-01-28 06:52 UTC_

Total papers shown: **19**


---

- **ALRM: Agentic LLM for Robotic Manipulation**  
  Vitor Gaboardi dos Santos, Ibrahim Khadraoui, Ibrahim Farhat, Hamza Yous, Samy Teffahi, Hakim Hacid  
  _2026-01-27_ · https://arxiv.org/abs/2601.19510v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.

  </details>



- **LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation**  
  Chongjun Xia, Yanchun Peng, Xianzhi Wang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19585v1 · `cs.IR`  
  <details><summary>Abstract</summary>

  Interactive recommender systems can dynamically adapt to user feedback, but often suffer from content homogeneity and filter bubble effects due to overfitting short-term user preferences. While recent efforts aim to improve content diversity, they predominantly operate in static or one-shot settings, neglecting the long-term evolution of user interests. Reinforcement learning provides a principled framework for optimizing long-term user satisfaction by modeling sequential decision-making processes. However, its application in recommendation is hindered by sparse, long-tailed user-item interactions and limited semantic planning capabilities. In this work, we propose LLM-Enhanced Reinforcement Learning (LERL), a novel hierarchical recommendation framework that integrates the semantic planning power of LLM with the fine-grained adaptability of RL. LERL consists of a high-level LLM-based planner that selects semantically diverse content categories, and a low-level RL policy that recommends personalized items within the selected semantic space. This hierarchical design narrows the action space, enhances planning efficiency, and mitigates overexposure to redundant content. Extensive experiments on real-world datasets demonstrate that LERL significantly improves long-term user satisfaction when compared with state-of-the-art baselines. The implementation of LERL is available at https://anonymous.4open.science/r/code3-18D3/.

  </details>



- **Task-Centric Policy Optimization from Misaligned Motion Priors**  
  Ziang Zheng, Kai Feng, Yi Nie, Shentao Qin  
  _2026-01-27_ · https://arxiv.org/abs/2601.19411v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Humanoid control often leverages motion priors from human demonstrations to encourage natural behaviors. However, such demonstrations are frequently suboptimal or misaligned with robotic tasks due to embodiment differences, retargeting errors, and task-irrelevant variations, causing naïve imitation to degrade task performance. Conversely, task-only reinforcement learning admits many task-optimal solutions, often resulting in unnatural or unstable motions. This exposes a fundamental limitation of linear reward mixing in adversarial imitation learning. We propose \emph{Task-Centric Motion Priors} (TCMP), a task-priority adversarial imitation framework that treats imitation as a conditional regularizer rather than a co-equal objective. TCMP maximizes task improvement while incorporating imitation signals only when they are compatible with task progress, yielding an adaptive, geometry-aware update that preserves task-feasible descent and suppresses harmful imitation under misalignment. We provide theoretical analysis of gradient conflict and task-priority stationary points, and validate our claims through humanoid control experiments demonstrating robust task performance with consistent motion style under noisy demonstrations.

  </details>



- **HARMONI: Multimodal Personalization of Multi-User Human-Robot Interactions with LLMs**  
  Jeanne Malécot, Hamed Rahimi, Jeanne Cattoni, Marie Samson, Mouad Abrini, Mahdi Khoramshahi, Maribel Pino, Mohamed Chetouani  
  _2026-01-27_ · https://arxiv.org/abs/2601.19839v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Existing human-robot interaction systems often lack mechanisms for sustained personalization and dynamic adaptation in multi-user environments, limiting their effectiveness in real-world deployments. We present HARMONI, a multimodal personalization framework that leverages large language models to enable socially assistive robots to manage long-term multi-user interactions. The framework integrates four key modules: (i) a perception module that identifies active speakers and extracts multimodal input; (ii) a world modeling module that maintains representations of the environment and short-term conversational context; (iii) a user modeling module that updates long-term speaker-specific profiles; and (iv) a generation module that produces contextually grounded and ethically informed responses. Through extensive evaluation and ablation studies on four datasets, as well as a real-world scenario-driven user-study in a nursing home environment, we demonstrate that HARMONI supports robust speaker identification, online memory updating, and ethically aligned personalization, outperforming baseline LLM-driven approaches in user modeling accuracy, personalization quality, and user satisfaction.

  </details>



- **Agentic Design Patterns: A System-Theoretic Framework**  
  Minh-Dung Dao, Quy Minh Le, Hoang Thanh Lam, Duc-Trong Le, Quoc-Viet Pham, Barry O'Sullivan, Hoang D. Nguyen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19752v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  With the development of foundation model (FM), agentic AI systems are getting more attention, yet their inherent issues like hallucination and poor reasoning, coupled with the frequent ad-hoc nature of system design, lead to unreliable and brittle applications. Existing efforts to characterise agentic design patterns often lack a rigorous systems-theoretic foundation, resulting in high-level or convenience-based taxonomies that are difficult to implement. This paper addresses this gap by introducing a principled methodology for engineering robust AI agents. We propose two primary contributions: first, a novel system-theoretic framework that deconstructs an agentic AI system into five core, interacting functional subsystems: Reasoning & World Model, Perception & Grounding, Action Execution, Learning & Adaptation, and Inter-Agent Communication. Second, derived from this architecture and directly mapped to a comprehensive taxonomy of agentic challenges, we present a collection of 12 agentic design patterns. These patterns - categorised as Foundational, Cognitive & Decisional, Execution & Interaction, and Adaptive & Learning - offer reusable, structural solutions to recurring problems in agent design. The utility of the framework is demonstrated by a case study on the ReAct framework, showing how the proposed patterns can rectify systemic architectural deficiencies. This work provides a foundational language and a structured methodology to standardise agentic design communication among researchers and engineers, leading to more modular, understandable, and reliable autonomous systems.

  </details>



- **ComAgent: Multi-LLM based Agentic AI Empowered Intelligent Wireless Networks**  
  Haoyun Li, Ming Xiao, Kezhi Wang, Robert Schober, Dong In Kim, Yong Liang Guan  
  _2026-01-27_ · https://arxiv.org/abs/2601.19607v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Emerging 6G networks rely on complex cross-layer optimization, yet manually translating high-level intents into mathematical formulations remains a bottleneck. While Large Language Models (LLMs) offer promise, monolithic approaches often lack sufficient domain grounding, constraint awareness, and verification capabilities. To address this, we present ComAgent, a multi-LLM agentic AI framework. ComAgent employs a closed-loop Perception-Planning-Action-Reflection cycle, coordinating specialized agents for literature search, coding, and scoring to autonomously generate solver-ready formulations and reproducible simulations. By iteratively decomposing problems and self-correcting errors, the framework effectively bridges the gap between user intent and execution. Evaluations demonstrate that ComAgent achieves expert-comparable performance in complex beamforming optimization and outperforms monolithic LLMs across diverse wireless tasks, highlighting its potential for automating design in emerging wireless networks.

  </details>



- **Reinforcement Learning Goal-Reaching Control with Guaranteed Lyapunov-Like Stabilizer for Mobile Robots**  
  Mehdi Heydari Shahna, Seyed Adel Alizadeh Kolagar, Jouni Mattila  
  _2026-01-27_ · https://arxiv.org/abs/2601.19499v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) can be highly effective at learning goal-reaching policies, but it typically does not provide formal guarantees that the goal will always be reached. A common approach to provide formal goal-reaching guarantees is to introduce a shielding mechanism that restricts the agent to actions that satisfy predefined safety constraints. The main challenge here is integrating this mechanism with RL so that learning and exploration remain effective without becoming overly conservative. Hence, this paper proposes an RL-based control framework that provides formal goal-reaching guarantees for wheeled mobile robots operating in unstructured environments. We first design a real-time RL policy with a set of 15 carefully defined reward terms. These rewards encourage the robot to reach both static and dynamic goals while generating sufficiently smooth command signals that comply with predefined safety specifications, which is critical in practice. Second, a Lyapunov-like stabilizer layer is integrated into the benchmark RL framework as a policy supervisor to formally strengthen the goal-reaching control while preserving meaningful exploration of the state action space. The proposed framework is suitable for real-time deployment in challenging environments, as it provides a formal guarantee of convergence to the intended goal states and compensates for uncertainties by generating real-time control signals based on the current state, while respecting real-world motion constraints. The experimental results show that the proposed Lyapunov-like stabilizer consistently improves the benchmark RL policies, boosting the goal-reaching rate from 84.6% to 99.0%, sharply reducing failures, and improving efficiency.

  </details>



- **Towards Gold-Standard Depth Estimation for Tree Branches in UAV Forestry: Benchmarking Deep Stereo Matching Methods**  
  Yida Lin, Bing Xue, Mengjie Zhang, Sam Schofield, Richard Green  
  _2026-01-27_ · https://arxiv.org/abs/2601.19461v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous UAV forestry operations require robust depth estimation with strong cross-domain generalization, yet existing evaluations focus on urban and indoor scenarios, leaving a critical gap for vegetation-dense environments. We present the first systematic zero-shot evaluation of eight stereo methods spanning iterative refinement, foundation model, diffusion-based, and 3D CNN paradigms. All methods use officially released pretrained weights (trained on Scene Flow) and are evaluated on four standard benchmarks (ETH3D, KITTI 2012/2015, Middlebury) plus a novel 5,313-pair Canterbury Tree Branches dataset ($1920 \times 1080$). Results reveal scene-dependent patterns: foundation models excel on structured scenes (BridgeDepth: 0.23 px on ETH3D; DEFOM: 4.65 px on Middlebury), while iterative methods show variable cross-benchmark performance (IGEV++: 0.36 px on ETH3D but 6.77 px on Middlebury; IGEV: 0.33 px on ETH3D but 4.99 px on Middlebury). Qualitative evaluation on the Tree Branches dataset establishes DEFOM as the gold-standard baseline for vegetation depth estimation, with superior cross-domain consistency (consistently ranking 1st-2nd across benchmarks, average rank 1.75). DEFOM predictions will serve as pseudo-ground-truth for future benchmarking.

  </details>



- **Reimagining Social Robots as Recommender Systems: Foundations, Framework, and Applications**  
  Jin Huang, Fethiye Irmak Doğan, Hatice Gunes  
  _2026-01-27_ · https://arxiv.org/abs/2601.19761v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Personalization in social robots refers to the ability of the robot to meet the needs and/or preferences of an individual user. Existing approaches typically rely on large language models (LLMs) to generate context-aware responses based on user metadata and historical interactions or on adaptive methods such as reinforcement learning (RL) to learn from users' immediate reactions in real time. However, these approaches fall short of comprehensively capturing user preferences-including long-term, short-term, and fine-grained aspects-, and of using them to rank and select actions, proactively personalize interactions, and ensure ethically responsible adaptations. To address the limitations, we propose drawing on recommender systems (RSs), which specialize in modeling user preferences and providing personalized recommendations. To ensure the integration of RS techniques is well-grounded and seamless throughout the social robot pipeline, we (i) align the paradigms underlying social robots and RSs, (ii) identify key techniques that can enhance personalization in social robots, and (iii) design them as modular, plug-and-play components. This work not only establishes a framework for integrating RS techniques into social robots but also opens a pathway for deep collaboration between the RS and HRI communities, accelerating innovation in both fields.

  </details>



- **Improving Policy Exploitation in Online Reinforcement Learning with Instant Retrospect Action**  
  Gong Gao, Weidong Zhao, Xianhui Liu, Ning Jia  
  _2026-01-27_ · https://arxiv.org/abs/2601.19720v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Existing value-based online reinforcement learning (RL) algorithms suffer from slow policy exploitation due to ineffective exploration and delayed policy updates. To address these challenges, we propose an algorithm called Instant Retrospect Action (IRA). Specifically, we propose Q-Representation Discrepancy Evolution (RDE) to facilitate Q-network representation learning, enabling discriminative representations for neighboring state-action pairs. In addition, we adopt an explicit method to policy constraints by enabling Greedy Action Guidance (GAG). This is achieved through backtracking historical actions, which effectively enhances the policy update process. Our proposed method relies on providing the learning algorithm with accurate $k$-nearest-neighbor action value estimates and learning to design a fast-adaptable policy through policy constraints. We further propose the Instant Policy Update (IPU) mechanism, which enhances policy exploitation by systematically increasing the frequency of policy updates. We further discover that the early-stage training conservatism of the IRA method can alleviate the overestimation bias problem in value-based RL. Experimental results show that IRA can significantly improve the learning efficiency and final performance of online RL algorithms on eight MuJoCo continuous control tasks.

  </details>



- **Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow**  
  Yunyue Wei, Chenhui Zuo, Yanan Sui  
  _2026-01-27_ · https://arxiv.org/abs/2601.19707v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Controlling high-dimensional systems in biological and robotic applications is challenging due to expansive state-action spaces, where effective exploration is critical. Commonly used exploration strategies in reinforcement learning are largely undirected with sharp degradation as action dimensionality grows. Many existing methods resort to dimensionality reduction, which constrains policy expressiveness and forfeits system flexibility. We introduce Q-guided Flow Exploration (Qflex), a scalable reinforcement learning method that conducts exploration directly in the native high-dimensional action space. During training, Qflex traverses actions from a learnable source distribution along a probability flow induced by the learned value function, aligning exploration with task-relevant gradients rather than isotropic noise. Our proposed method substantially outperforms representative online reinforcement learning baselines across diverse high-dimensional continuous-control benchmarks. Qflex also successfully controls a full-body human musculoskeletal model to perform agile, complex movements, demonstrating superior scalability and sample efficiency in very high-dimensional settings. Our results indicate that value-guided flows offer a principled and practical route to exploration at scale.

  </details>



- **Out-of-Distribution Generalization via Invariant Trajectories for Multimodal Large Language Model Editing**  
  Jiajie Su, Haoyuan Wang, Xiaohua Feng, Yunshan Ma, Xiaobo Xia, Yuyuan Li, Xiaolin Zheng, Jianmao Xiao, Chaochao Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19700v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Knowledge editing emerges as a crucial technique for efficiently correcting incorrect or outdated knowledge in large language models (LLM). Existing editing methods for unimodal LLM rely on a rigid parameter-to-output mapping, which causes causal-underfit and causal-overfit in cascaded reasoning for Multimodal LLM (MLLM). In this paper, we reformulate MLLM editing as an out-of-distribution (OOD) generalization problem, where the goal is to discern semantic shift with factual shift and thus achieve robust editing among diverse cross-modal prompting. The key challenge of this OOD problem lies in identifying invariant causal trajectories that generalize accurately while suppressing spurious correlations. To address it, we propose ODEdit, a plug-and-play invariant learning based framework that optimizes the tripartite OOD risk objective to simultaneously enhance editing reliability, locality, and generality.We further introduce an edit trajectory invariant learning method, which integrates a total variation penalty into the risk minimization objective to stabilize edit trajectories against environmental variations. Theoretical analysis and extensive experiments demonstrate the effectiveness of ODEdit.

  </details>



- **Towards Governance-Oriented Low-Altitude Intelligence: A Management-Centric Multi-Modal Benchmark With Implicitly Coordinated Vision-Language Reasoning Framework**  
  Hao Chang, Zhihui Wang, Lingxiang Wu, Peijin Wang, Wenhui Diao, Jinqiao Wang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19640v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Low-altitude vision systems are becoming a critical infrastructure for smart city governance. However, existing object-centric perception paradigms and loosely coupled vision-language pipelines are still difficult to support management-oriented anomaly understanding required in real-world urban governance. To bridge this gap, we introduce GovLA-10K, the first management-oriented multi-modal benchmark for low-altitude intelligence, along with GovLA-Reasoner, a unified vision-language reasoning framework tailored for governance-aware aerial perception. Unlike existing studies that aim to exhaustively annotate all visible objects, GovLA-10K is deliberately designed around functionally salient targets that directly correspond to practical management needs, and further provides actionable management suggestions grounded in these observations. To effectively coordinate the fine-grained visual grounding with high-level contextual language reasoning, GovLA-Reasoner introduces an efficient feature adapter that implicitly coordinates discriminative representation sharing between the visual detector and the large language model (LLM). Extensive experiments show that our method significantly improves performance while avoiding the need of fine-tuning for any task-specific individual components. We believe our work offers a new perspective and foundation for future studies on management-aware low-altitude vision-language systems.

  </details>



- **PALM: Enhanced Generalizability for Local Visuomotor Policies via Perception Alignment**  
  Ruiyu Wang, Zheyu Zhuang, Danica Kragic, Florian T. Pokorny  
  _2026-01-27_ · https://arxiv.org/abs/2601.19514v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Generalizing beyond the training domain in image-based behavior cloning remains challenging. Existing methods address individual axes of generalization, workspace shifts, viewpoint changes, and cross-embodiment transfer, yet they are typically developed in isolation and often rely on complex pipelines. We introduce PALM (Perception Alignment for Local Manipulation), which leverages the invariance of local action distributions between out-of-distribution (OOD) and demonstrated domains to address these OOD shifts concurrently, without additional input modalities, model changes, or data collection. PALM modularizes the manipulation policy into coarse global components and a local policy for fine-grained actions. We reduce the discrepancy between in-domain and OOD inputs at the local policy level by enforcing local visual focus and consistent proprioceptive representation, allowing the policy to retrieve invariant local actions under OOD conditions. Experiments show that PALM limits OOD performance drops to 8% in simulation and 24% in the real world, compared to 45% and 77% for baselines.

  </details>



- **Bridging Information Asymmetry: A Hierarchical Framework for Deterministic Blind Face Restoration**  
  Zhengjian Yao, Jiakui Hu, Kaiwen Li, Hangzhou He, Xinliang Zhang, Shuang Zeng, Lei Zhu, Yanye Lu  
  _2026-01-27_ · https://arxiv.org/abs/2601.19506v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Blind face restoration remains a persistent challenge due to the inherent ill-posedness of reconstructing holistic structures from severely constrained observations. Current generative approaches, while capable of synthesizing realistic textures, often suffer from information asymmetry -- the intrinsic disparity between the information-sparse low quality inputs and the information-dense high quality outputs. This imbalance leads to a one-to-many mapping, where insufficient constraints result in stochastic uncertainty and hallucinatory artifacts. To bridge this gap, we present \textbf{Pref-Restore}, a hierarchical framework that integrates discrete semantic logic with continuous texture generation to achieve deterministic, preference-aligned restoration. Our methodology fundamentally addresses this information disparity through two complementary strategies: (1) Augmenting Input Density: We employ an auto-regressive integrator to reformulate textual instructions into dense latent queries, injecting high-level semantic stability to constrain the degraded signals; (2) Pruning Output Distribution: We pioneer the integration of on-policy reinforcement learning directly into the diffusion restoration loop. By transforming human preferences into differentiable constraints, we explicitly penalize stochastic deviations, thereby sharpening the posterior distribution toward the desired high-fidelity outcomes. Extensive experiments demonstrate that Pref-Restore achieves state-of-the-art performance across synthetic and real-world benchmarks. Furthermore, empirical analysis confirms that our preference-aligned strategy significantly reduces solution entropy, establishing a robust pathway toward reliable and deterministic blind restoration.

  </details>



- **PROTEUS: SLA-Aware Routing via Lagrangian RL for Multi-LLM Serving Systems**  
  Amit Singh Bhatti, Vishal Vaddina, Dagnachew Birru  
  _2026-01-27_ · https://arxiv.org/abs/2601.19402v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Production LLM deployments serve diverse workloads where cost and quality requirements vary by customer tier, time of day, and query criticality. Model serving systems accept latency SLOs directly. LLM routers do not. They force operators to tune parameters offline and guess what accuracy might result. The relationship between parameters and outcomes is indirect, non-monotonic, and dataset-dependent. Operators need to specify accuracy targets, not infer them from opaque settings. We present PROTEUS (Polymorphic Router for Operational Target Enforcement with Unified SLA), a router that accepts accuracy targets tau as runtime input. PROTEUS uses Lagrangian dual control. A learned dual variable lambda tracks constraint violations during training and conditions the policy network. This lets the router translate specified tau values into routing decisions that satisfy them. A single trained model serves the full accuracy spectrum without retraining.We evaluate on RouterBench (11 models, 405K queries) and SPROUT (14 models, 45K queries). PROTEUS achieves consistent floor compliance where accuracy meets or exceeds tau. The target-response correlation reaches 0.97 to 0.98. The closest baseline, OmniRouter, meets floors only 22% of the time despite also using Lagrangian optimization. PROTEUS operates across tau in [0.85, 0.95] from a single model. On RouterBench it achieves 90.1% accuracy, within 1.3% of oracle. On SPROUT it achieves 94.0% accuracy, within 4.6% of oracle. Cost savings reach 89.8% versus the best fixed model.

  </details>



- **Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection**  
  Quy-Anh Dang, Chris Ngo  
  _2026-01-27_ · https://arxiv.org/abs/2601.19375v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Despite significant progress in alignment, large language models (LLMs) remain vulnerable to adversarial attacks that elicit harmful behaviors. Activation steering techniques offer a promising inference-time intervention approach, but existing methods suffer from critical limitations: activation addition requires careful coefficient tuning and is sensitive to layer-specific norm variations, while directional ablation provides only binary control. Recent work on Angular Steering introduces continuous control via rotation in a 2D subspace, but its practical implementation violates norm preservation, causing distribution shift and generation collapse, particularly in models below 7B parameters. We propose Selective Steering, which addresses these limitations through two key innovations: (1) a mathematically rigorous norm-preserving rotation formulation that maintains activation distribution integrity, and (2) discriminative layer selection that applies steering only where feature representations exhibit opposite-signed class alignment. Experiments across nine models demonstrate that Selective Steering achieves 5.5x higher attack success rates than prior methods while maintaining zero perplexity violations and approximately 100\% capability retention on standard benchmarks. Our approach provides a principled, efficient framework for controllable and stable LLM behavior modification. Code: https://github.com/knoveleng/steering

  </details>



- **From Observations to Events: Event-Aware World Model for Reinforcement Learning**  
  Zhao-Han Peng, Shaohui Li, Zhi Li, Shulan Ruan, Yu Liu, You He  
  _2026-01-27_ · https://arxiv.org/abs/2601.19336v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  While model-based reinforcement learning (MBRL) improves sample efficiency by learning world models from raw observations, existing methods struggle to generalize across structurally similar scenes and remain vulnerable to spurious variations such as textures or color shifts. From a cognitive science perspective, humans segment continuous sensory streams into discrete events and rely on these key events for decision-making. Motivated by this principle, we propose the Event-Aware World Model (EAWM), a general framework that learns event-aware representations to streamline policy learning without requiring handcrafted labels. EAWM employs an automated event generator to derive events from raw observations and introduces a Generic Event Segmentor (GES) to identify event boundaries, which mark the start and end time of event segments. Through event prediction, the representation space is shaped to capture meaningful spatio-temporal transitions. Beyond this, we present a unified formulation of seemingly distinct world model architectures and show the broad applicability of our methods. Experiments on Atari 100K, Craftax 1M, and DeepMind Control 500K, DMC-GB2 500K demonstrate that EAWM consistently boosts the performance of strong MBRL baselines by 10%-45%, setting new state-of-the-art results across benchmarks. Our code is released at https://github.com/MarquisDarwin/EAWM.

  </details>



- **Output Feedback Stabilization of Linear Systems via Policy Gradient Methods**  
  Ankang Zhang, Ming Chi, Xiaoling Wang, Lintao Ye  
  _2026-01-27_ · https://arxiv.org/abs/2601.19284v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Stabilizing a dynamical system is a fundamental problem that serves as a cornerstone for many complex tasks in the field of control systems. The problem becomes challenging when the system model is unknown. Among the Reinforcement Learning (RL) algorithms that have been successfully applied to solve problems pertaining to unknown linear dynamical systems, the policy gradient (PG) method stands out due to its ease of implementation and can solve the problem in a model-free manner. However, most of the existing works on PG methods for unknown linear dynamical systems assume full-state feedback. In this paper, we take a step towards model-free learning for partially observable linear dynamical systems with output feedback and focus on the fundamental stabilization problem of the system. We propose an algorithmic framework that stretches the boundary of PG methods to the problem without global convergence guarantees. We show that by leveraging zeroth-order PG update based on system trajectories and its convergence to stationary points, the proposed algorithms return a stabilizing output feedback policy for discrete-time linear dynamical systems. We also explicitly characterize the sample complexity of our algorithm and verify the effectiveness of the algorithm using numerical examples.

  </details>


