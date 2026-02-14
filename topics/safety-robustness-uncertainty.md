# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-14 07:00 UTC_

Total papers shown: **12**


---

- **6G Empowering Future Robotics: A Vision for Next-Generation Autonomous Systems**  
  Mona Ghassemian, Andrés Meseguer Valenzuela, Ana Garcia Armada, Dejan Vukobratovic, Periklis Chatzimisios, Kaspar Althoefer, Ranga Rao Venkatesha Prasad  
  _2026-02-12_ · https://arxiv.org/abs/2602.12246v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  The convergence of robotics and next-generation communication is a critical driver of technological advancement. As the world transitions from 5G to 6G, the foundational capabilities of wireless networks are evolving to support increasingly complex and autonomous robotic systems. This paper examines the transformative impact of 6G on enhancing key robotics functionalities. It provides a systematic mapping of IMT-2030 key performance indicators to robotic functional blocks including sensing, perception, cognition, actuation and self-learning. Building upon this mapping, we propose a high-level architectural framework integrating robotic, intelligent, and network service planes, underscoring the need for a holistic approach. As an example use case, we present a real-time, dynamic safety framework enabled by IMT-2030 capabilities for safe and efficient human-robot collaboration in shared spaces.

  </details>



- **Decentralized Multi-Robot Obstacle Detection and Tracking in a Maritime Scenario**  
  Muhammad Farhan Ahmed, Vincent Frémont  
  _2026-02-12_ · https://arxiv.org/abs/2602.12012v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous aerial-surface robot teams are promising for maritime monitoring. Robust deployment requires reliable perception over reflective water and scalable coordination under limited communication. We present a decentralized multi-robot framework for detecting and tracking floating containers using multiple UAVs cooperating with an autonomous surface vessel. Each UAV performs YOLOv8 and stereo-disparity-based visual detection, then tracks targets with per-object EKFs using uncertainty-aware data association. Compact track summaries are exchanged and fused conservatively via covariance intersection, ensuring consistency under unknown correlations. An information-driven assignment module allocates targets and selects UAV hover viewpoints by trading expected uncertainty reduction against travel effort and safety separation. Simulation results in a maritime scenario demonstrate improved coverage, localization accuracy, and tracking consistency while maintaining modest communication requirements.

  </details>



- **Safety Beyond the Training Data: Robust Out-of-Distribution MPC via Conformalized System Level Synthesis**  
  Anutam Srinivasan, Antoine Leeman, Glen Chou  
  _2026-02-12_ · https://arxiv.org/abs/2602.12047v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a novel framework for robust out-of-distribution planning and control using conformal prediction (CP) and system level synthesis (SLS), addressing the challenge of ensuring safety and robustness when using learned dynamics models beyond the training data distribution. We first derive high-confidence model error bounds using weighted CP with a learned, state-control-dependent covariance model. These bounds are integrated into an SLS-based robust nonlinear model predictive control (MPC) formulation, which performs constraint tightening over the prediction horizon via volume-optimized forward reachable sets. We provide theoretical guarantees on coverage and robustness under distributional drift, and analyze the impact of data density and trajectory tube size on prediction coverage. Empirically, we demonstrate our method on nonlinear systems of increasing complexity, including a 4D car and a {12D} quadcopter, improving safety and robustness compared to fixed-bound and non-robust baselines, especially outside of the data distribution.

  </details>



- **LAMP: Implicit Language Map for Robot Navigation**  
  Sibaek Lee, Hyeonwoo Yu, Giseop Kim, Sunwook Choi  
  _2026-02-12_ · https://arxiv.org/abs/2602.11862v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advances in vision-language models have made zero-shot navigation feasible, enabling robots to follow natural language instructions without requiring labeling. However, existing methods that explicitly store language vectors in grid or node-based maps struggle to scale to large environments due to excessive memory requirements and limited resolution for fine-grained planning. We introduce LAMP (Language Map), a novel neural language field-based navigation framework that learns a continuous, language-driven map and directly leverages it for fine-grained path generation. Unlike prior approaches, our method encodes language features as an implicit neural field rather than storing them explicitly at every location. By combining this implicit representation with a sparse graph, LAMP supports efficient coarse path planning and then performs gradient-based optimization in the learned field to refine poses near the goal. This coarse-to-fine pipeline, language-driven, gradient-guided optimization is the first application of an implicit language map for precise path generation. This refinement is particularly effective at selecting goal regions not directly observed by leveraging semantic similarities in the learned feature space. To further enhance robustness, we adopt a Bayesian framework that models embedding uncertainty via the von Mises-Fisher distribution, thereby improving generalization to unobserved regions. To scale to large environments, LAMP employs a graph sampling strategy that prioritizes spatial coverage and embedding confidence, retaining only the most informative nodes and substantially reducing computational overhead. Our experimental results, both in NVIDIA Isaac Sim and on a real multi-floor building, demonstrate that LAMP outperforms existing explicit methods in both memory efficiency and fine-grained goal-reaching accuracy.

  </details>



- **Adaptive-Horizon Conflict-Based Search for Closed-Loop Multi-Agent Path Finding**  
  Jiarui Li, Federico Pecora, Runyu Zhang, Gioele Zardini  
  _2026-02-12_ · https://arxiv.org/abs/2602.12024v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  MAPF is a core coordination problem for large robot fleets in automated warehouses and logistics. Existing approaches are typically either open-loop planners, which generate fixed trajectories and struggle to handle disturbances, or closed-loop heuristics without reliable performance guarantees, limiting their use in safety-critical deployments. This paper presents ACCBS, a closed-loop algorithm built on a finite-horizon variant of CBS with a horizon-changing mechanism inspired by iterative deepening in MPC. ACCBS dynamically adjusts the planning horizon based on the available computational budget, and reuses a single constraint tree to enable seamless transitions between horizons. As a result, it produces high-quality feasible solutions quickly while being asymptotically optimal as the budget increases, exhibiting anytime behavior. Extensive case studies demonstrate that ACCBS combines flexibility to disturbances with strong performance guarantees, effectively bridging the gap between theoretical optimality and practical robustness for large-scale robot deployment.

  </details>



- **SafeNeuron: Neuron-Level Safety Alignment for Large Language Models**  
  Zhaoxin Wang, Jiaming Liang, Fengbin Zhu, Weixiang Zhao, Junfeng Fang, Jiayi Ji, Handing Wang, Tat-Seng Chua  
  _2026-02-12_ · https://arxiv.org/abs/2602.12158v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) and multimodal LLMs are typically safety-aligned before release to prevent harmful content generation. However, recent studies show that safety behaviors are concentrated in a small subset of parameters, making alignment brittle and easily bypassed through neuron-level attacks. Moreover, most existing alignment methods operate at the behavioral level, offering limited control over the model's internal safety mechanisms. In this work, we propose SafeNeuron, a neuron-level safety alignment framework that improves robustness by redistributing safety representations across the network. SafeNeuron first identifies safety-related neurons, then freezes these neurons during preference optimization to prevent reliance on sparse safety pathways and force the model to construct redundant safety representations. Extensive experiments across models and modalities demonstrate that SafeNeuron significantly improves robustness against neuron pruning attacks, reduces the risk of open-source models being repurposed as red-team generators, and preserves general capabilities. Furthermore, our layer-wise analysis reveals that safety behaviors are governed by stable and shared internal representations. Overall, SafeNeuron provides an interpretable and robust perspective for model alignment.

  </details>



- **General Humanoid Whole-Body Control via Pretraining and Fast Adaptation**  
  Zepeng Wang, Jiangxing Wang, Shiqing Yao, Yu Zhang, Ziluo Ding, Ming Yang, Yuxuan Wang, Haobin Jiang, Chao Ma, Xiaochuan Shi, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.11929v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning a general whole-body controller for humanoid robots remains challenging due to the diversity of motion distributions, the difficulty of fast adaptation, and the need for robust balance in high-dynamic scenarios. Existing approaches often require task-specific training or suffer from performance degradation when adapting to new motions. In this paper, we present FAST, a general humanoid whole-body control framework that enables Fast Adaptation and Stable Motion Tracking. FAST introduces Parseval-Guided Residual Policy Adaptation, which learns a lightweight delta action policy under orthogonality and KL constraints, enabling efficient adaptation to out-of-distribution motions while mitigating catastrophic forgetting. To further improve physical robustness, we propose Center-of-Mass-Aware Control, which incorporates CoM-related observations and objectives to enhance balance when tracking challenging reference motions. Extensive experiments in simulation and real-world deployment demonstrate that FAST consistently outperforms state-of-the-art baselines in robustness, adaptation efficiency, and generalization.

  </details>



- **Safe Fairness Guarantees Without Demographics in Classification: Spectral Uncertainty Set Perspective**  
  Ainhize Barrainkua, Santiago Mazuelas, Novi Quadrianto, Jose A. Lozano  
  _2026-02-12_ · https://arxiv.org/abs/2602.11785v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  As automated classification systems become increasingly prevalent, concerns have emerged over their potential to reinforce and amplify existing societal biases. In the light of this issue, many methods have been proposed to enhance the fairness guarantees of classifiers. Most of the existing interventions assume access to group information for all instances, a requirement rarely met in practice. Fairness without access to demographic information has often been approached through robust optimization techniques,which target worst-case outcomes over a set of plausible distributions known as the uncertainty set. However, their effectiveness is strongly influenced by the chosen uncertainty set. In fact, existing approaches often overemphasize outliers or overly pessimistic scenarios, compromising both overall performance and fairness. To overcome these limitations, we introduce SPECTRE, a minimax-fair method that adjusts the spectrum of a simple Fourier feature mapping and constrains the extent to which the worst-case distribution can deviate from the empirical distribution. We perform extensive experiments on the American Community Survey datasets involving 20 states. The safeness of SPECTRE comes as it provides the highest average values on fairness guarantees together with the smallest interquartile range in comparison to state-of-the-art approaches, even compared to those with access to demographic group information. In addition, we provide a theoretical analysis that derives computable bounds on the worst-case error for both individual groups and the overall population, as well as characterizes the worst-case distributions responsible for these extremal performances

  </details>



- **Multi UAVs Preflight Planning in a Shared and Dynamic Airspace**  
  Amath Sow, Mauricio Rodriguez Cesen, Fabiola Martins Campos de Oliveira, Mariusz Wzorek, Daniel de Leng, Mattias Tiger, Fredrik Heintz, Christian Esteve Rothenberg  
  _2026-02-12_ · https://arxiv.org/abs/2602.12055v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Preflight planning for large-scale Unmanned Aerial Vehicle (UAV) fleets in dynamic, shared airspace presents significant challenges, including temporal No-Fly Zones (NFZs), heterogeneous vehicle profiles, and strict delivery deadlines. While Multi-Agent Path Finding (MAPF) provides a formal framework, existing methods often lack the scalability and flexibility required for real-world Unmanned Traffic Management (UTM). We propose DTAPP-IICR: a Delivery-Time Aware Prioritized Planning method with Incremental and Iterative Conflict Resolution. Our framework first generates an initial solution by prioritizing missions based on urgency. Secondly, it computes roundtrip trajectories using SFIPP-ST, a novel 4D single-agent planner (Safe Flight Interval Path Planning with Soft and Temporal Constraints). SFIPP-ST handles heterogeneous UAVs, strictly enforces temporal NFZs, and models inter-agent conflicts as soft constraints. Subsequently, an iterative Large Neighborhood Search, guided by a geometric conflict graph, efficiently resolves any residual conflicts. A completeness-preserving directional pruning technique further accelerates the 3D search. On benchmarks with temporal NFZs, DTAPP-IICR achieves near-100% success with fleets of up to 1,000 UAVs and gains up to 50% runtime reduction from pruning, outperforming batch Enhanced Conflict-Based Search in the UTM context. Scaling successfully in realistic city-scale operations where other priority-based methods fail even at moderate deployments, DTAPP-IICR is positioned as a practical and scalable solution for preflight planning in dense, dynamic urban airspace.

  </details>



- **Robot-DIFT: Distilling Diffusion Features for Geometrically Consistent Visuomotor Control**  
  Yu Deng, Yufeng Jin, Xiaogang Jia, Jiahong Xue, Gerhard Neumann, Georgia Chalvatzaki  
  _2026-02-12_ · https://arxiv.org/abs/2602.11934v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We hypothesize that a key bottleneck in generalizable robot manipulation is not solely data scale or policy capacity, but a structural mismatch between current visual backbones and the physical requirements of closed-loop control. While state-of-the-art vision encoders (including those used in VLAs) optimize for semantic invariance to stabilize classification, manipulation typically demands geometric sensitivity the ability to map millimeter-level pose shifts to predictable feature changes. Their discriminative objective creates a "blind spot" for fine-grained control, whereas generative diffusion models inherently encode geometric dependencies within their latent manifolds, encouraging the preservation of dense multi-scale spatial structure. However, directly deploying stochastic diffusion features for control is hindered by stochastic instability, inference latency, and representation drift during fine-tuning. To bridge this gap, we propose Robot-DIFT, a framework that decouples the source of geometric information from the process of inference via Manifold Distillation. By distilling a frozen diffusion teacher into a deterministic Spatial-Semantic Feature Pyramid Network (S2-FPN), we retain the rich geometric priors of the generative model while ensuring temporal stability, real-time execution, and robustness against drift. Pretrained on the large-scale DROID dataset, Robot-DIFT demonstrates superior geometric consistency and control performance compared to leading discriminative baselines, supporting the view that how a model learns to see dictates how well it can learn to act.

  </details>



- **Data-Driven Trajectory Imputation for Vessel Mobility Analysis**  
  Giannis Spiliopoulos, Alexandros Troupiotis-Kapeliaris, Kostas Patroumpas, Nikolaos Liapis, Dimitrios Skoutas, Dimitris Zissis, Nikos Bikakis  
  _2026-02-12_ · https://arxiv.org/abs/2602.11890v1 · `cs.DB`  
  <details><summary>Abstract</summary>

  Modeling vessel activity at sea is critical for a wide range of applications, including route planning, transportation logistics, maritime safety, and environmental monitoring. Over the past two decades, the Automatic Identification System (AIS) has enabled real-time monitoring of hundreds of thousands of vessels, generating huge amounts of data daily. One major challenge in using AIS data is the presence of large gaps in vessel trajectories, often caused by coverage limitations or intentional transmission interruptions. These gaps can significantly degrade data quality, resulting in inaccurate or incomplete analysis. State-of-the-art imputation approaches have mainly been devised to tackle gaps in vehicle trajectories, even when the underlying road network is not considered. But the motion patterns of sailing vessels differ substantially, e.g., smooth turns, maneuvering near ports, or navigating in adverse weather conditions. In this application paper, we propose HABIT, a lightweight, configurable H3 Aggregation-Based Imputation framework for vessel Trajectories. This data-driven framework provides a valuable means to impute missing trajectory segments by extracting, analyzing, and indexing motion patterns from historical AIS data. Our empirical study over AIS data across various timeframes, densities, and vessel types reveals that HABIT produces maritime trajectory imputations performing comparably to baseline methods in terms of accuracy, while performing better in terms of latency while accounting for vessel characteristics and their motion patterns.

  </details>



- **CAAL: Confidence-Aware Active Learning for Heteroscedastic Atmospheric Regression**  
  Fei Jiang, Jiyang Xia, Junjie Yu, Mingfei Sun, Hugh Coe, David Topping, Dantong Liu, Zhenhui Jessie Li, Zhonghua Zheng  
  _2026-02-12_ · https://arxiv.org/abs/2602.11825v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Quantifying the impacts of air pollution on health and climate relies on key atmospheric particle properties such as toxicity and hygroscopicity. However, these properties typically require complex observational techniques or expensive particle-resolved numerical simulations, limiting the availability of labeled data. We therefore estimate these hard-to-measure particle properties from routinely available observations (e.g., air pollutant concentrations and meteorological conditions). Because routine observations only indirectly reflect particle composition and structure, the mapping from routine observations to particle properties is noisy and input-dependent, yielding a heteroscedastic regression setting. With a limited and costly labeling budget, the central challenge is to select which samples to measure or simulate. While active learning is a natural approach, most acquisition strategies rely on predictive uncertainty. Under heteroscedastic noise, this signal conflates reducible epistemic uncertainty with irreducible aleatoric uncertainty, causing limited budgets to be wasted in noise-dominated regions. To address this challenge, we propose a confidence-aware active learning framework (CAAL) for efficient and robust sample selection in heteroscedastic settings. CAAL consists of two components: a decoupled uncertainty-aware training objective that separately optimises the predictive mean and noise level to stabilise uncertainty estimation, and a confidence-aware acquisition function that dynamically weights epistemic uncertainty using predicted aleatoric uncertainty as a reliability signal. Experiments on particle-resolved numerical simulations and real atmospheric observations show that CAAL consistently outperforms standard AL baselines. The proposed framework provides a practical and general solution for the efficient expansion of high-cost atmospheric particle property databases.

  </details>


