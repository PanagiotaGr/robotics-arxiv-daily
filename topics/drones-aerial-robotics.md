# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-01-28 06:52 UTC_

Total papers shown: **3**


---

- **Perception-to-Pursuit: Track-Centric Temporal Reasoning for Open-World Drone Detection and Autonomous Chasing**  
  Venkatakrishna Reddy Oruganti  
  _2026-01-27_ · https://arxiv.org/abs/2601.19318v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous drone pursuit requires not only detecting drones but also predicting their trajectories in a manner that enables kinematically feasible interception. Existing tracking methods optimize for prediction accuracy but ignore pursuit feasibility, resulting in trajectories that are physically impossible to intercept 99.9% of the time. We propose Perception-to-Pursuit (P2P), a track-centric temporal reasoning framework that bridges detection and actionable pursuit planning. Our method represents drone motion as compact 8-dimensional tokens capturing velocity, acceleration, scale, and smoothness, enabling a 12-frame causal transformer to reason about future behavior. We introduce the Intercept Success Rate (ISR) metric to measure pursuit feasibility under realistic interceptor constraints. Evaluated on the Anti-UAV-RGBT dataset with 226 real drone sequences, P2P achieves 28.12 pixel average displacement error and 0.597 ISR, representing a 77% improvement in trajectory prediction and 597x improvement in pursuit feasibility over tracking-only baselines, while maintaining perfect drone classification accuracy (100%). Our work demonstrates that temporal reasoning over motion patterns enables both accurate prediction and actionable pursuit planning.

  </details>



- **A DVL Aided Loosely Coupled Inertial Navigation Strategy for AUVs with Attitude Error Modeling and Variance Propagation**  
  Jin Huang, Zichen Liu, Haoda Li, Zhikun Wang, Ying Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19509v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In underwater navigation systems, strap-down inertial navigation system/Doppler velocity log (SINS/DVL)-based loosely coupled architectures are widely adopted. Conventional approaches project DVL velocities from the body coordinate system to the navigation coordinate system using SINS-derived attitude; however, accumulated attitude estimation errors introduce biases into velocity projection and degrade navigation performance during long-term operation. To address this issue, two complementary improvements are introduced. First, a vehicle attitude error-aware DVL velocity transformation model is formulated by incorporating attitude error terms into the observation equation to reduce projection-induced velocity bias. Second, a covariance matrix-based variance propagation method is developed to transform DVL measurement uncertainty across coordinate systems, introducing an expectation-based attitude error compensation term to achieve statistically consistent noise modeling. Simulation and field experiment results demonstrate that both improvements individually enhance navigation accuracy and confirm that accumulated attitude errors affect both projected velocity measurements and their associated uncertainty. When jointly applied, long-term error divergence is effectively suppressed. Field experimental results show that the proposed approach achieves a 78.3% improvement in 3D position RMSE and a 71.8% reduction in the maximum component-wise position error compared with the baseline IMU+DVL method, providing a robust solution for improving long-term SINS/DVL navigation performance.

  </details>



- **Towards Gold-Standard Depth Estimation for Tree Branches in UAV Forestry: Benchmarking Deep Stereo Matching Methods**  
  Yida Lin, Bing Xue, Mengjie Zhang, Sam Schofield, Richard Green  
  _2026-01-27_ · https://arxiv.org/abs/2601.19461v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous UAV forestry operations require robust depth estimation with strong cross-domain generalization, yet existing evaluations focus on urban and indoor scenarios, leaving a critical gap for vegetation-dense environments. We present the first systematic zero-shot evaluation of eight stereo methods spanning iterative refinement, foundation model, diffusion-based, and 3D CNN paradigms. All methods use officially released pretrained weights (trained on Scene Flow) and are evaluated on four standard benchmarks (ETH3D, KITTI 2012/2015, Middlebury) plus a novel 5,313-pair Canterbury Tree Branches dataset ($1920 \times 1080$). Results reveal scene-dependent patterns: foundation models excel on structured scenes (BridgeDepth: 0.23 px on ETH3D; DEFOM: 4.65 px on Middlebury), while iterative methods show variable cross-benchmark performance (IGEV++: 0.36 px on ETH3D but 6.77 px on Middlebury; IGEV: 0.33 px on ETH3D but 4.99 px on Middlebury). Qualitative evaluation on the Tree Branches dataset establishes DEFOM as the gold-standard baseline for vegetation depth estimation, with superior cross-domain consistency (consistently ranking 1st-2nd across benchmarks, average rank 1.75). DEFOM predictions will serve as pseudo-ground-truth for future benchmarking.

  </details>


