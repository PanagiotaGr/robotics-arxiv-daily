# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-27 07:08 UTC_

Total papers shown: **10**


---

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



- **Marinarium: a New Arena to Bring Maritime Robotics Closer to Shore**  
  Ignacio Torroba, David Dorner, Victor Nan Fernandez-Ayala, Mart Kartasev, Joris Verhagen, Elias Krantz, Gregorio Marchesini, Carl Ljung, Pedro Roque, Chelsea Sidrane, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.23053v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents the Marinarium, a modular and stand-alone underwater research facility designed to provide a realistic testbed for maritime and space-analog robotic experimentation in a resource-efficient manner. The Marinarium combines a fully instrumented underwater and aerial operational volume, extendable via a retractable roof for real-weather conditions, a digital twin in the SMaRCSim simulator and tight integration with a space robotics laboratory. All of these result from design choices aimed at bridging simulation, laboratory validation, and field conditions. We compare the Marinarium to similar existing infrastructures and illustrate how its design enables a set of experiments in four open research areas within field robotics. First, we exploit high-fidelity dynamics data from the tank to demonstrate the potential of learning-based system identification approaches applied to underwater vehicles. We further highlight the versatility of the multi-domain operating volume via a rendezvous mission with a heterogeneous fleet of robots across underwater, surface, and air. We then illustrate how the presented digital twin can be utilized to reduce the reality gap in underwater simulation. Finally, we demonstrate the potential of underwater surrogates for spacecraft navigation validation by executing spatiotemporally identical inspection tasks on a planar space-robot emulator and a neutrally buoyant \gls{rov}. In this work, by sharing the insights obtained and rationale behind the design and construction of the Marinarium, we hope to provide the field robotics research community with a blueprint for bridging the gap between controlled and real offshore and space robotics experimentation.

  </details>



- **WaterVideoQA: ASV-Centric Perception and Rule-Compliant Reasoning via Multi-Modal Agents**  
  Runwei Guan, Shaofeng Liang, Ningwei Ouyang, Weichen Fei, Shanliang Yao, Wei Dai, Chenhao Ge, Penglei Sun, Xiaohui Zhu, Tao Huang, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.22923v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  While autonomous navigation has achieved remarkable success in passive perception (e.g., object detection and segmentation), it remains fundamentally constrained by a void in knowledge-driven, interactive environmental cognition. In the high-stakes domain of maritime navigation, the ability to bridge the gap between raw visual perception and complex cognitive reasoning is not merely an enhancement but a critical prerequisite for Autonomous Surface Vessels to execute safe and precise maneuvers. To this end, we present WaterVideoQA, the first large-scale, comprehensive Video Question Answering benchmark specifically engineered for all-waterway environments. This benchmark encompasses 3,029 video clips across six distinct waterway categories, integrating multifaceted variables such as volatile lighting and dynamic weather to rigorously stress-test ASV capabilities across a five-tier hierarchical cognitive framework. Furthermore, we introduce NaviMind, a pioneering multi-agent neuro-symbolic system designed for open-ended maritime reasoning. By synergizing Adaptive Semantic Routing, Situation-Aware Hierarchical Reasoning, and Autonomous Self-Reflective Verification, NaviMind transitions ASVs from superficial pattern matching to regulation-compliant, interpretable decision-making. Experimental results demonstrate that our framework significantly transcends existing baselines, establishing a new paradigm for intelligent, trustworthy interaction in dynamic maritime environments.

  </details>



- **LineGraph2Road: Structural Graph Reasoning on Line Graphs for Road Network Extraction**  
  Zhengyang Wei, Renzhi Jing, Yiyi He, Jenny Suckale  
  _2026-02-26_ · https://arxiv.org/abs/2602.23290v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The accurate and automatic extraction of roads from satellite imagery is critical for applications in navigation and urban planning, significantly reducing the need for manual annotation. Many existing methods decompose this task into keypoint extraction and connectedness prediction, but often struggle to capture long-range dependencies and complex topologies. Here, we propose LineGraph2Road, a framework that improves connectedness prediction by formulating it as binary classification over edges in a constructed global but sparse Euclidean graph, where nodes are keypoints extracted from segmentation masks and edges connect node pairs within a predefined distance threshold, representing potential road segments. To better learn structural link representation, we transform the original graph into its corresponding line graph and apply a Graph Transformer on it for connectedness prediction. This formulation overcomes the limitations of endpoint-embedding fusion on set-isomorphic links, enabling rich link representations and effective relational reasoning over the global structure. Additionally, we introduce an overpass/underpass head to resolve multi-level crossings and a coupled NMS strategy to preserve critical connections. We evaluate LineGraph2Road on three benchmarks: City-scale, SpaceNet, and Global-scale, and show that it achieves state-of-the-art results on two key metrics, TOPO-F1 and APLS. It also captures fine visual details critical for real-world deployment. We will make our code publicly available.

  </details>



- **UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception**  
  Mohammad Mahdavian, Gordon Tan, Binbin Xu, Yuan Ren, Dongfeng Bai, Bingbing Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23224v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present UniScale, a unified, scale-aware multi-view 3D reconstruction framework for robotic applications that flexibly integrates geometric priors through a modular, semantically informed design. In vision-based robotic navigation, the accurate extraction of environmental structure from raw image sequences is critical for downstream tasks. UniScale addresses this challenge with a single feed-forward network that jointly estimates camera intrinsics and extrinsics, scale-invariant depth and point maps, and the metric scale of a scene from multi-view images, while optionally incorporating auxiliary geometric priors when available. By combining global contextual reasoning with camera-aware feature representations, UniScale is able to recover the metric-scale of the scene. In robotic settings where camera intrinsics are known, they can be easily incorporated to improve performance, with additional gains obtained when camera poses are also available. This co-design enables robust, metric-aware 3D reconstruction within a single unified model. Importantly, UniScale does not require training from scratch, and leverages world priors exhibited in pre-existing models without geometric encoding strategies, making it particularly suitable for resource-constrained robotic teams. We evaluate UniScale on multiple benchmarks, demonstrating strong generalization and consistent performance across diverse environments. We will release our implementation upon acceptance.

  </details>



- **Spatio-Temporal Token Pruning for Efficient High-Resolution GUI Agents**  
  Zhou Xu, Bowen Zhou, Qi Wang, Shuwen Feng, Jingyu Xiao  
  _2026-02-26_ · https://arxiv.org/abs/2602.23235v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Pure-vision GUI agents provide universal interaction capabilities but suffer from severe efficiency bottlenecks due to the massive spatiotemporal redundancy inherent in high-resolution screenshots and historical trajectories. We identify two critical misalignments in existing compression paradigms: the temporal mismatch, where uniform history encoding diverges from the agent's "fading memory" attention pattern, and the spatial topology conflict, where unstructured pruning compromises the grid integrity required for precise coordinate grounding, inducing spatial hallucinations. To address these challenges, we introduce GUIPruner, a training-free framework tailored for high-resolution GUI navigation. It synergizes Temporal-Adaptive Resolution (TAR), which eliminates historical redundancy via decay-based resizing, and Stratified Structure-aware Pruning (SSP), which prioritizes interactive foregrounds and semantic anchors while safeguarding global layout. Extensive evaluations across diverse benchmarks demonstrate that GUIPruner consistently achieves state-of-the-art performance, effectively preventing the collapse observed in large-scale models under high compression. Notably, on Qwen2-VL-2B, our method delivers a 3.4x reduction in FLOPs and a 3.3x speedup in vision encoding latency while retaining over 94% of the original performance, enabling real-time, high-precision navigation with minimal resource consumption.

  </details>



- **AgentVista: Evaluating Multimodal Agents in Ultra-Challenging Realistic Visual Scenarios**  
  Zhaochen Su, Jincheng Gao, Hangyu Guo, Zhenhua Liu, Lueyang Zhang, Xinyu Geng, Shijue Huang, Peng Xia, Guanyu Jiang, Cheng Wang, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.23166v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Real-world multimodal agents solve multi-step workflows grounded in visual evidence. For example, an agent can troubleshoot a device by linking a wiring photo to a schematic and validating the fix with online documentation, or plan a trip by interpreting a transit map and checking schedules under routing constraints. However, existing multimodal benchmarks mainly evaluate single-turn visual reasoning or specific tool skills, and they do not fully capture the realism, visual subtlety, and long-horizon tool use that practical agents require. We introduce AgentVista, a benchmark for generalist multimodal agents that spans 25 sub-domains across 7 categories, pairing realistic and detail-rich visual scenarios with natural hybrid tool use. Tasks require long-horizon tool interactions across modalities, including web search, image search, page navigation, and code-based operations for both image processing and general programming. Comprehensive evaluation of state-of-the-art models exposes significant gaps in their ability to carry out long-horizon multimodal tool use. Even the best model in our evaluation, Gemini-3-Pro with tools, achieves only 27.3% overall accuracy, and hard instances can require more than 25 tool-calling turns. We expect AgentVista to accelerate the development of more capable and reliable multimodal agents for realistic and ultra-challenging problem solving.

  </details>



- **Automated Robotic Needle Puncture for Percutaneous Dilatational Tracheostomy**  
  Yuan Tang, Bruno V. Adorno, Brendan A. McGrath, Andrew Weightman  
  _2026-02-26_ · https://arxiv.org/abs/2602.22952v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Percutaneous dilatational tracheostomy (PDT) is frequently performed on patients in intensive care units for prolonged mechanical ventilation. The needle puncture, as the most critical step of PDT, could lead to adverse consequences such as major bleeding and posterior tracheal wall perforation if performed inaccurately. Current practices of PDT puncture are all performed manually with no navigation assistance, which leads to large position and angular errors (5 mm and 30 degree). To improve the accuracy and reduce the difficulty of the PDT procedure, we propose a system that automates the needle insertion using a velocity-controlled robotic manipulator. Guided using pose data from two electromagnetic sensors, one at the needle tip and the other inside the trachea, the robotic system uses an adaptive constrained controller to adapt the uncertain kinematic parameters online and avoid collisions with the patient's body and tissues near the target. Simulations were performed to validate the controller's implementation, and then four hundred PDT punctures were performed on a mannequin to evaluate the position and angular accuracy. The absolute median puncture position error was 1.7 mm (IQR: 1.9 mm) and midline deviation was 4.13 degree (IQR: 4.55 degree), measured by the sensor inside the trachea. The small deviations from the nominal puncture in a simulated experimental setup and formal guarantees of collision-free insertions suggest the feasibility of the robotic PDT puncture.

  </details>



- **Considering Perspectives for Automated Driving Ethics: Collective Risk in Vehicular Motion Planning**  
  Leon Tolksdorf, Arturo Tejada, Christian Birkner, Nathan van de Wouw  
  _2026-02-26_ · https://arxiv.org/abs/2602.22940v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent automated vehicle (AV) motion planning strategies evolve around minimizing risk in road traffic. However, they exclusively consider risk from the AV's perspective and, as such, do not address the ethicality of its decisions for other road users. We argue that this does not reduce the risk of each road user, as risk may be different from the perspective of each road user. Indeed, minimizing the risk from the AV's perspective may not imply that the risk from the perspective of other road users is also being minimized; in fact, it may even increase. To test this hypothesis, we propose an AV motion planning strategy that supports switching risk minimization strategies between all road user perspectives. We find that the risk from the perspective of other road users can generally be considered different to the risk from the AV's perspective. Taking a collective risk perspective, i.e., balancing the risks of all road users, we observe an AV that minimizes overall traffic risk the best, while putting itself at slightly higher risk for the benefit of others, which is consistent with human driving behavior. In addition, adopting a collective risk minimization strategy can also be beneficial to the AV's travel efficiency by acting assertively when other road users maintain a low risk estimate of the AV. Yet, the AV drives conservatively when its planned actions are less predictable to other road users, i.e., associated with high risk. We argue that such behavior is a form of self-reflection and a natural prerequisite for socially acceptable AV behavior. We conclude that to facilitate ethicality in road traffic that includes AVs, the risk-perspective of each road user must be considered in the decision-making of AVs.

  </details>


