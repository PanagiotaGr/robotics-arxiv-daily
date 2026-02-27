# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-27 07:08 UTC_

Total papers shown: **15**


---

- **Simple Models, Real Swimming: Digital Twins for Tendon-Driven Underwater Robots**  
  Mike Y. Michelis, Nana Obayashi, Josie Hughes, Robert K. Katzschmann  
  _2026-02-26_ · https://arxiv.org/abs/2602.23283v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mimicking the graceful motion of swimming animals remains a core challenge in soft robotics due to the complexity of fluid-structure interaction and the difficulty of controlling soft, biomimetic bodies. Existing modeling approaches are often computationally expensive and impractical for complex control or reinforcement learning needed for realistic motions to emerge in robotic systems. In this work, we present a tendon-driven fish robot modeled in an efficient underwater swimmer environment using a simplified, stateless hydrodynamics formulation implemented in the widespread robotics framework MuJoCo. With just two real-world swimming trajectories, we identify five fluid parameters that allow a matching to experimental behavior and generalize across a range of actuation frequencies. We show that this stateless fluid model can generalize to unseen actuation and outperform classical analytical models such as the elongated body theory. This simulation environment runs faster than real-time and can easily enable downstream learning algorithms such as reinforcement learning for target tracking, reaching a 93% success rate. Due to the simplicity and ease of use of the model and our open-source simulation environment, our results show that even simple, stateless models -- when carefully matched to physical data -- can serve as effective digital twins for soft underwater robots, opening up new directions for scalable learning and control in aquatic environments.

  </details>



- **Towards Intelligible Human-Robot Interaction: An Active Inference Approach to Occluded Pedestrian Scenarios**  
  Kai Chen, Yuyao Huang, Guang Chen  
  _2026-02-26_ · https://arxiv.org/abs/2602.23109v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The sudden appearance of occluded pedestrians presents a critical safety challenge in autonomous driving. Conventional rule-based or purely data-driven approaches struggle with the inherent high uncertainty of these long-tail scenarios. To tackle this challenge, we propose a novel framework grounded in Active Inference, which endows the agent with a human-like, belief-driven mechanism. Our framework leverages a Rao-Blackwellized Particle Filter (RBPF) to efficiently estimate the pedestrian's hybrid state. To emulate human-like cognitive processes under uncertainty, we introduce a Conditional Belief Reset mechanism and a Hypothesis Injection technique to explicitly model beliefs about the pedestrian's multiple latent intentions. Planning is achieved via a Cross-Entropy Method (CEM) enhanced Model Predictive Path Integral (MPPI) controller, which synergizes the efficient, iterative search of CEM with the inherent robustness of MPPI. Simulation experiments demonstrate that our approach significantly reduces the collision rate compared to reactive, rule-based, and reinforcement learning (RL) baselines, while also exhibiting explainable and human-like driving behavior that reflects the agent's internal belief state.

  </details>



- **A Perspective on Open Challenges in Deformable Object Manipulation**  
  Ryan Paul McKennaa, John Oyekan  
  _2026-02-26_ · https://arxiv.org/abs/2602.22998v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deformable object manipulation (DOM) represents a critical challenge in robotics, with applications spanning healthcare, manufacturing, food processing, and beyond. Unlike rigid objects, deformable objects exhibit infinite dimensionality, dynamic shape changes, and complex interactions with their environment, posing significant hurdles for perception, modeling, and control. This paper reviews the state of the art in DOM, focusing on key challenges such as occlusion handling, task generalization, and scalable, real-time solutions. It highlights advancements in multimodal perception systems, including the integration of multi-camera setups, active vision, and tactile sensing, which collectively address occlusion and improve adaptability in unstructured environments. Cutting-edge developments in physically informed reinforcement learning (RL) and differentiable simulations are explored, showcasing their impact on efficiency, precision, and scalability. The review also emphasizes the potential of simulated expert demonstrations and generative neural networks to standardize task specifications and bridge the simulation-to-reality gap. Finally, future directions are proposed, including the adoption of graph neural networks for high-level decision-making and the creation of comprehensive datasets to enhance DOM's real-world applicability. By addressing these challenges, DOM research can pave the way for versatile robotic systems capable of handling diverse and dynamic tasks with deformable objects.

  </details>



- **DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation**  
  Zebin Yang, Yijiahao Qi, Tong Xie, Bo Yu, Shaoshan Liu, Meng Li  
  _2026-02-26_ · https://arxiv.org/abs/2602.22896v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have shown remarkable success in robotic tasks like manipulation by fusing a language model's reasoning with a vision model's 3D understanding. However, their high computational cost remains a major obstacle for real-world applications that require real-time performance. We observe that the actions within a task have varying levels of importance: critical steps demand high precision, while less important ones can tolerate more variance. Leveraging this insight, we propose DySL-VLA, a novel framework that addresses computational cost by dynamically skipping VLA layers based on each action's importance. DySL-VLA categorizes its layers into two types: informative layers, which are consistently executed, and incremental layers, which can be selectively skipped. To intelligently skip layers without sacrificing accuracy, we invent a prior-post skipping guidance mechanism to determine when to initiate layer-skipping. We also propose a skip-aware two-stage knowledge distillation algorithm to efficiently train a standard VLA into a DySL-VLA. Our experiments indicate that DySL-VLA achieves 2.1% improvement in success length over Deer-VLA on the Calvin dataset, while simultaneously reducing trainable parameters by a factor of 85.7 and providing a 3.75x speedup relative to the RoboFlamingo baseline at iso-accuracy. Our code is available on https://github.com/PKU-SEC-Lab/DYSL_VLA.

  </details>



- **GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion**  
  Enda Xiang, Haoxiang Ma, Xinzhu Ma, Zicheng Liu, Di Huang  
  _2026-02-26_ · https://arxiv.org/abs/2602.22862v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper focuses on enhancing the grasping precision and generalization of manipulation policies learned via imitation learning. Diffusion-based policy learning methods have recently become the mainstream approach for robotic manipulation tasks. As grasping is a critical subtask in manipulation, the ability of imitation-learned policies to execute precise and generalizable grasps merits particular attention. Existing imitation learning techniques for grasping often suffer from imprecise grasp executions, limited spatial generalization, and poor object generalization. To address these challenges, we incorporate grasp prior knowledge into the diffusion policy framework. In particular, we employ a latent diffusion policy to guide action chunk decoding with grasp pose prior, ensuring that generated motion trajectories adhere closely to feasible grasp configurations. Furthermore, we introduce a self-supervised reconstruction objective during diffusion to embed the graspness prior: at each reverse diffusion step, we reconstruct wrist-camera images back-projected the graspness from the intermediate representations. Both simulation and real robot experiments demonstrate that our approach significantly outperforms baseline methods and exhibits strong dynamic grasping capabilities.

  </details>



- **Physics Informed Viscous Value Representations**  
  Hrishikesh Viswanath, Juanwu Lu, S. Talha Bukhari, Damon Conover, Ziran Wang, Aniket Bera  
  _2026-02-26_ · https://arxiv.org/abs/2602.23280v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Offline goal-conditioned reinforcement learning (GCRL) learns goal-conditioned policies from static pre-collected datasets. However, accurate value estimation remains a challenge due to the limited coverage of the state-action space. Recent physics-informed approaches have sought to address this by imposing physical and geometric constraints on the value function through regularization defined over first-order partial differential equations (PDEs), such as the Eikonal equation. However, these formulations can often be ill-posed in complex, high-dimensional environments. In this work, we propose a physics-informed regularization derived from the viscosity solution of the Hamilton-Jacobi-Bellman (HJB) equation. By providing a physics-based inductive bias, our approach grounds the learning process in optimal control theory, explicitly regularizing and bounding updates during value iterations. Furthermore, we leverage the Feynman-Kac theorem to recast the PDE solution as an expectation, enabling a tractable Monte Carlo estimation of the objective that avoids numerical instability in higher-order gradients. Experiments demonstrate that our method improves geometric consistency, making it broadly applicable to navigation and high-dimensional, complex manipulation tasks. Open-source codes are available at https://github.com/HrishikeshVish/phys-fk-value-GCRL.

  </details>



- **Risk-Aware World Model Predictive Control for Generalizable End-to-End Autonomous Driving**  
  Jiangxin Sun, Feng Xue, Teng Long, Chang Liu, Jian-Fang Hu, Wei-Shi Zheng, Nicu Sebe  
  _2026-02-26_ · https://arxiv.org/abs/2602.23259v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  With advances in imitation learning (IL) and large-scale driving datasets, end-to-end autonomous driving (E2E-AD) has made great progress recently. Currently, IL-based methods have become a mainstream paradigm: models rely on standard driving behaviors given by experts, and learn to minimize the discrepancy between their actions and expert actions. However, this objective of "only driving like the expert" suffers from limited generalization: when encountering rare or unseen long-tail scenarios outside the distribution of expert demonstrations, models tend to produce unsafe decisions in the absence of prior experience. This raises a fundamental question: Can an E2E-AD system make reliable decisions without any expert action supervision? Motivated by this, we propose a unified framework named Risk-aware World Model Predictive Control (RaWMPC) to address this generalization dilemma through robust control, without reliance on expert demonstrations. Practically, RaWMPC leverages a world model to predict the consequences of multiple candidate actions and selects low-risk actions through explicit risk evaluation. To endow the world model with the ability to predict the outcomes of risky driving behaviors, we design a risk-aware interaction strategy that systematically exposes the world model to hazardous behaviors, making catastrophic outcomes predictable and thus avoidable. Furthermore, to generate low-risk candidate actions at test time, we introduce a self-evaluation distillation method to distill riskavoidance capabilities from the well-trained world model into a generative action proposal network without any expert demonstration. Extensive experiments show that RaWMPC outperforms state-of-the-art methods in both in-distribution and out-of-distribution scenarios, while providing superior decision interpretability.

  </details>



- **ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering**  
  Elzo Brito dos Santos Filho  
  _2026-02-26_ · https://arxiv.org/abs/2602.23193v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Autonomous agents based on Large Language Models (LLMs) have evolved from reactive assistants to systems capable of planning, executing actions via tools, and iterating over environment observations. However, they remain vulnerable to structural limitations: lack of native state, context degradation over long horizons, and the gap between probabilistic generation and deterministic execution requirements. This paper presents the ESAA (Event Sourcing for Autonomous Agents) architecture, which separates the agent's cognitive intention from the project's state mutation, inspired by the Event Sourcing pattern. In ESAA, agents emit only structured intentions in validated JSON (agent.result or issue.report); a deterministic orchestrator validates, persists events in an append-only log (activity.jsonl), applies file-writing effects, and projects a verifiable materialized view (roadmap.json). The proposal incorporates boundary contracts (AGENT_CONTRACT.yaml), metaprompting profiles (PARCER), and replay verification with hashing (esaa verify), ensuring the immutability of completed tasks and forensic traceability. Two case studies validate the architecture: (i) a landing page project (9 tasks, 49 events, single-agent composition) and (ii) a clinical dashboard system (50 tasks, 86 events, 4 concurrent agents across 8 phases), both concluding with run.status=success and verify_status=ok. The multi-agent case study demonstrates real concurrent orchestration with heterogeneous LLMs (Claude Sonnet 4.6, Codex GPT-5, Antigravity/Gemini 3 Pro, and Claude Opus 4.6), providing empirical evidence of the architecture's scalability beyond single-agent scenarios.

  </details>



- **Toward Expert Investment Teams:A Multi-Agent LLM System with Fine-Grained Trading Tasks**  
  Kunihiro Miyazaki, Takanobu Kawahara, Stephen Roberts, Stefan Zohren  
  _2026-02-26_ · https://arxiv.org/abs/2602.23330v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The advancement of large language models (LLMs) has accelerated the development of autonomous financial trading systems. While mainstream approaches deploy multi-agent systems mimicking analyst and manager roles, they often rely on abstract instructions that overlook the intricacies of real-world workflows, which can lead to degraded inference performance and less transparent decision-making. Therefore, we propose a multi-agent LLM trading framework that explicitly decomposes investment analysis into fine-grained tasks, rather than providing coarse-grained instructions. We evaluate the proposed framework using Japanese stock data, including prices, financial statements, news, and macro information, under a leakage-controlled backtesting setting. Experimental results show that fine-grained task decomposition significantly improves risk-adjusted returns compared to conventional coarse-grained designs. Crucially, further analysis of intermediate agent outputs suggests that alignment between analytical outputs and downstream decision preferences is a critical driver of system performance. Moreover, we conduct standard portfolio optimization, exploiting low correlation with the stock index and the variance of each system's output. This approach achieves superior performance. These findings contribute to the design of agent structure and task configuration when applying LLM agents to trading systems in practical settings.

  </details>



- **PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering**  
  Junkai Lu, Peng Chen, Xingjian Wu, Yang Shu, Chenjuan Guo, Christian S. Jensen, Bin Yang  
  _2026-02-26_ · https://arxiv.org/abs/2602.23161v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Time series reasoning demands both the perception of complex dynamics and logical depth. However, existing LLM-based approaches exhibit two limitations: they often treat time series merely as text or images, failing to capture the patterns like trends and seasonalities needed to answer specific questions; and when trained on a mix of simple and complex tasks, simpler objectives often dominate the learning process, hindering the development of deep reasoning capabilities. To address these limitations, we propose the Pattern-Aware Alignment and Balanced Reasoning model (PATRA), introducing a pattern-aware mechanism that extracts trend and seasonality patterns from time series to achieve deep alignment. Furthermore, we design a task-aware balanced reward to harmonize learning across tasks of varying difficulty, incentivizing the generation of coherent Chains of Thought. Extensive experiments show that PATRA outperforms strong baselines across diverse Time Series Question Answering (TSQA) tasks, demonstrating superior cross-modal understanding and reasoning capability.

  </details>



- **Multi-Agent Large Language Model Based Emotional Detoxification Through Personalized Intensity Control for Consumer Protection**  
  Keito Inoshita  
  _2026-02-26_ · https://arxiv.org/abs/2602.23123v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  In the attention economy, sensational content exposes consumers to excessive emotional stimulation, hindering calm decision-making. This study proposes Multi-Agent LLM-based Emotional deToxification (MALLET), a multi-agent information sanitization system consisting of four agents: Emotion Analysis, Emotion Adjustment, Balance Monitoring, and Personal Guide. The Emotion Analysis Agent quantifies stimulus intensity using a 6-emotion BERT classifier, and the Emotion Adjustment Agent rewrites texts into two presentation modes, BALANCED (neutralized text) and COOL (neutralized text + supplementary text), using an LLM. The Balance Monitoring Agent aggregates weekly information consumption patterns and generates personalized advice, while the Personal Guide Agent recommends a presentation mode according to consumer sensitivity. Experiments on 800 AG News articles demonstrated significant stimulus score reduction (up to 19.3%) and improved emotion balance while maintaining semantic preservation. Near-zero correlation between stimulus reduction and semantic preservation confirmed that the two are independently controllable. Category-level analysis revealed substantial reduction (17.8-33.8%) in Sports, Business, and Sci/Tech, whereas the effect was limited in the World category, where facts themselves are inherently high-stimulus. The proposed system provides a framework for supporting calm information reception of consumers without restricting access to the original text.

  </details>



- **Three AI-agents walk into a bar . . . . `Lord of the Flies' tribalism emerges among smart AI-Agents**  
  Dhwanil M. Mori, Neil F. Johnson  
  _2026-02-26_ · https://arxiv.org/abs/2602.23093v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Near-future infrastructure systems may be controlled by autonomous AI agents that repeatedly request access to limited resources such as energy, bandwidth, or computing power. We study a simplified version of this setting using a framework where N AI-agents independently decide at each round whether to request one unit from a system with fixed capacity C. An AI version of "Lord of the Flies" arises in which controlling tribes emerge with their own collective character and identity. The LLM agents do not reduce overload or improve resource use, and often perform worse than if they were flipping coins to make decisions. Three main tribal types emerge: Aggressive (27.3%), Conservative (24.7%), and Opportunistic (48.1%). The more capable AI-agents actually increase the rate of systemic failure. Overall, our findings show that smarter AI-agents can behave dumber as a result of forming tribes.

  </details>



- **GeoWorld: Geometric World Models**  
  Zeyu Zhang, Danning Li, Ian Reid, Richard Hartley  
  _2026-02-26_ · https://arxiv.org/abs/2602.23058v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Energy-based predictive world models provide a powerful approach for multi-step visual planning by reasoning over latent energy landscapes rather than generating pixels. However, existing approaches face two major challenges: (i) their latent representations are typically learned in Euclidean space, neglecting the underlying geometric and hierarchical structure among states, and (ii) they struggle with long-horizon prediction, which leads to rapid degradation across extended rollouts. To address these challenges, we introduce GeoWorld, a geometric world model that preserves geometric structure and hierarchical relations through a Hyperbolic JEPA, which maps latent representations from Euclidean space onto hyperbolic manifolds. We further introduce Geometric Reinforcement Learning for energy-based optimization, enabling stable multi-step planning in hyperbolic latent space. Extensive experiments on CrossTask and COIN demonstrate around 3% SR improvement in 3-step planning and 2% SR improvement in 4-step planning compared to the state-of-the-art V-JEPA 2. Project website: https://steve-zeyu-zhang.github.io/GeoWorld.

  </details>



- **Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space**  
  Xingcheng Fu, Shengpeng Wang, Yisen Gao, Xianxian Li, Chunpei Li, Qingyun Sun, Dongran Yu  
  _2026-02-26_ · https://arxiv.org/abs/2602.22879v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Knowledge Tracing (KT) diagnoses students' concept mastery through continuous learning state monitoring in education.Existing methods primarily focus on studying behavioral sequences based on ID or textual information.While existing methods rely on ID-based sequences or shallow textual features, they often fail to capture (1) the hierarchical evolution of cognitive states and (2) individualized problem difficulty perception due to limited semantic modeling. Therefore, this paper proposes a Large Language Model Hyperbolic Aligned Knowledge Tracing(L-HAKT). First, the teacher agent deeply parses question semantics and explicitly constructs hierarchical dependencies of knowledge points; the student agent simulates learning behaviors to generate synthetic data. Then, contrastive learning is performed between synthetic and real data in hyperbolic space to reduce distribution differences in key features such as question difficulty and forgetting patterns. Finally, by optimizing hyperbolic curvature, we explicitly model the tree-like hierarchical structure of knowledge points, precisely characterizing differences in learning curve morphology for knowledge points at different levels. Extensive experiments on four real-world educational datasets validate the effectiveness of our Large Language Model Hyperbolic Aligned Knowledge Tracing (L-HAKT) framework.

  </details>



- **From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models**  
  Hongrui Jia, Chaoya Jiang, Shikun Zhang, Wei Ye  
  _2026-02-26_ · https://arxiv.org/abs/2602.22859v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  As Large Multimodal Models (LMMs) scale up and reinforcement learning (RL) methods mature, LMMs have made notable progress in complex reasoning and decision making. Yet training still relies on static data and fixed recipes, making it difficult to diagnose capability blind spots or provide dynamic, targeted reinforcement. Motivated by findings that test driven error exposure and feedback based correction outperform repetitive practice, we propose Diagnostic-driven Progressive Evolution (DPE), a spiral loop where diagnosis steers data generation and reinforcement, and each iteration re-diagnoses the updated model to drive the next round of targeted improvement. DPE has two key components. First, multiple agents annotate and quality control massive unlabeled multimodal data, using tools such as web search and image editing to produce diverse, realistic samples. Second, DPE attributes failures to specific weaknesses, dynamically adjusts the data mixture, and guides agents to generate weakness focused data for targeted reinforcement. Experiments on Qwen3-VL-8B-Instruct and Qwen2.5-VL-7B-Instruct show stable, continual gains across eleven benchmarks, indicating DPE as a scalable paradigm for continual LMM training under open task distributions. Our code, models, and data are publicly available at https://github.com/hongruijia/DPE.

  </details>


