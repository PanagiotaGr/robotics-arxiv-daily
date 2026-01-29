# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-01-29 07:02 UTC_

Total papers shown: **4**


---

- **Li-ViP3D++: Query-Gated Deformable Camera-LiDAR Fusion for End-to-End Perception and Trajectory Prediction**  
  Matej Halinkovic, Nina Masarykova, Alexey Vinel, Marek Galinski  
  _2026-01-28_ · https://arxiv.org/abs/2601.20720v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  End-to-end perception and trajectory prediction from raw sensor data is one of the key capabilities for autonomous driving. Modular pipelines restrict information flow and can amplify upstream errors. Recent query-based, fully differentiable perception-and-prediction (PnP) models mitigate these issues, yet the complementarity of cameras and LiDAR in the query-space has not been sufficiently explored. Models often rely on fusion schemes that introduce heuristic alignment and discrete selection steps which prevent full utilization of available information and can introduce unwanted bias. We propose Li-ViP3D++, a query-based multimodal PnP framework that introduces Query-Gated Deformable Fusion (QGDF) to integrate multi-view RGB and LiDAR in query space. QGDF (i) aggregates image evidence via masked attention across cameras and feature levels, (ii) extracts LiDAR context through fully differentiable BEV sampling with learned per-query offsets, and (iii) applies query-conditioned gating to adaptively weight visual and geometric cues per agent. The resulting architecture jointly optimizes detection, tracking, and multi-hypothesis trajectory forecasting in a single end-to-end model. On nuScenes, Li-ViP3D++ improves end-to-end behavior and detection quality, achieving higher EPA (0.335) and mAP (0.502) while substantially reducing false positives (FP ratio 0.147), and it is faster than the prior Li-ViP3D variant (139.82 ms vs. 145.91 ms). These results indicate that query-space, fully differentiable camera-LiDAR fusion can increase robustness of end-to-end PnP without sacrificing deployability.

  </details>



- **Learning Contextual Runtime Monitors for Safe AI-Based Autonomy**  
  Alejandro Luque-Cerpa, Mengyuan Wang, Emil Carlsson, Sanjit A. Seshia, Devdatt Dubhashi, Hazem Torfah  
  _2026-01-28_ · https://arxiv.org/abs/2601.20666v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We introduce a novel framework for learning context-aware runtime monitors for AI-based control ensembles. Machine-learning (ML) controllers are increasingly deployed in (autonomous) cyber-physical systems because of their ability to solve complex decision-making tasks. However, their accuracy can degrade sharply in unfamiliar environments, creating significant safety concerns. Traditional ensemble methods aim to improve robustness by averaging or voting across multiple controllers, yet this often dilutes the specialized strengths that individual controllers exhibit in different operating contexts. We argue that, rather than blending controller outputs, a monitoring framework should identify and exploit these contextual strengths. In this paper, we reformulate the design of safe AI-based control ensembles as a contextual monitoring problem. A monitor continuously observes the system's context and selects the controller best suited to the current conditions. To achieve this, we cast monitor learning as a contextual learning task and draw on techniques from contextual multi-armed bandits. Our approach comes with two key benefits: (1) theoretical safety guarantees during controller selection, and (2) improved utilization of controller diversity. We validate our framework in two simulated autonomous driving scenarios, demonstrating significant improvements in both safety and performance compared to non-contextual baselines.

  </details>



- **Unsupervised Anomaly Detection in Multi-Agent Trajectory Prediction via Transformer-Based Models**  
  Qing Lyu, Zhe Fu, Alexandre Bayen  
  _2026-01-28_ · https://arxiv.org/abs/2601.20367v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Identifying safety-critical scenarios is essential for autonomous driving, but the rarity of such events makes supervised labeling impractical. Traditional rule-based metrics like Time-to-Collision are too simplistic to capture complex interaction risks, and existing methods lack a systematic way to verify whether statistical anomalies truly reflect physical danger. To address this gap, we propose an unsupervised anomaly detection framework based on a multi-agent Transformer that models normal driving and measures deviations through prediction residuals. A dual evaluation scheme has been proposed to assess both detection stability and physical alignment: Stability is measured using standard ranking metrics in which Kendall Rank Correlation Coefficient captures rank agreement and Jaccard index captures the consistency of the top-K selected items; Physical alignment is assessed through correlations with established Surrogate Safety Measures (SSM). Experiments on the NGSIM dataset demonstrate our framework's effectiveness: We show that the maximum residual aggregator achieves the highest physical alignment while maintaining stability. Furthermore, our framework identifies 388 unique anomalies missed by Time-to-Collision and statistical baselines, capturing subtle multi-agent risks like reactive braking under lateral drift. The detected anomalies are further clustered into four interpretable risk types, offering actionable insights for simulation and testing.

  </details>



- **Decentralized Stochastic Constrained Optimization via Prox-Linearization**  
  Shivangi Dubey Sharma, Basil M. Idrees, Lavish Arora, Ketan Rajawat  
  _2026-01-28_ · https://arxiv.org/abs/2601.20345v1 · `math.OC`  
  <details><summary>Abstract</summary>

  This paper studies consensus-based decentralized stochastic optimization for minimizing possibly non-convex expected objectives with convex non-smooth regularizers and nonlinear functional inequality constraints. We reformulate the constrained problem using the exact-penalty model and develop two algorithms that require only local stochastic gradients and first-order constraint information. The first method, Decentralized Stochastic Momentum-based Prox-Linear Algorithm (D-SMPL), combines constraint linearization with a prox-linear step, resulting in a linearly constrained quadratic subproblem per iteration. Building on this approach, we propose a successive convex approximation (SCA) variant, Decentralized SCA Momentum-based Prox-Linear (D-SCAMPL), which handles additional objective structure through strongly convex surrogate subproblems while still allowing infeasible initialization. Both methods incorporate recursive momentum-based gradient estimators and a consensus mechanism requiring only two communication rounds per iteration. Under standard smoothness and regularity assumptions, both algorithms achieve an oracle complexity of $\mathcal{O}(ε^{-3/2})$, matching the optimal rate known for unconstrained centralized stochastic non-convex optimization. Numerical experiments on energy-optimal ocean trajectory planning corroborate the theory and demonstrate improved performance over existing decentralized baselines.

  </details>


