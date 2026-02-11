# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-11 07:16 UTC_

Total papers shown: **4**


---

- **AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild**  
  Xiaolou Sun, Wufei Si, Wenhui Ni, Yuntian Li, Dongming Wu, Fei Xie, Runwei Guan, He-Yang Xu, Henghui Ding, Yuan Wu, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09657v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-language navigation (VLN) requires intelligent agents to navigate environments by interpreting linguistic instructions alongside visual observations, serving as a cornerstone task in Embodied AI. Current VLN research for unmanned aerial vehicles (UAVs) relies on detailed, pre-specified instructions to guide the UAV along predetermined routes. However, real-world outdoor exploration typically occurs in unknown environments where detailed navigation instructions are unavailable. Instead, only coarse-grained positional or directional guidance can be provided, requiring UAVs to autonomously navigate through continuous planning and obstacle avoidance. To bridge this gap, we propose AutoFly, an end-to-end Vision-Language-Action (VLA) model for autonomous UAV navigation. AutoFly incorporates a pseudo-depth encoder that derives depth-aware features from RGB inputs to enhance spatial reasoning, coupled with a progressive two-stage training strategy that effectively aligns visual, depth, and linguistic representations with action policies. Moreover, existing VLN datasets have fundamental limitations for real-world autonomous navigation, stemming from their heavy reliance on explicit instruction-following over autonomous decision-making and insufficient real-world data. To address these issues, we construct a novel autonomous navigation dataset that shifts the paradigm from instruction-following to autonomous behavior modeling through: (1) trajectory collection emphasizing continuous obstacle avoidance, autonomous planning, and recognition workflows; (2) comprehensive real-world data integration. Experimental results demonstrate that AutoFly achieves a 3.9% higher success rate compared to state-of-the-art VLA baselines, with consistent performance across simulated and real environments.

  </details>



- **A Collision-Free Sway Damping Model Predictive Controller for Safe and Reactive Forestry Crane Navigation**  
  Marc-Philip Ecker, Christoph Fröhlich, Johannes Huemer, David Gruber, Bernhard Bischof, Tobias Glück, Wolfgang Kemmetmüller  
  _2026-02-10_ · https://arxiv.org/abs/2602.10035v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Forestry cranes operate in dynamic, unstructured outdoor environments where simultaneous collision avoidance and payload sway control are critical for safe navigation. Existing approaches address these challenges separately, either focusing on sway damping with predefined collision-free paths or performing collision avoidance only at the global planning level. We present the first collision-free, sway-damping model predictive controller (MPC) for a forestry crane that unifies both objectives in a single control framework. Our approach integrates LiDAR-based environment mapping directly into the MPC using online Euclidean distance fields (EDF), enabling real-time environmental adaptation. The controller simultaneously enforces collision constraints while damping payload sway, allowing it to (i) replan upon quasi-static environmental changes, (ii) maintain collision-free operation under disturbances, and (iii) provide safe stopping when no bypass exists. Experimental validation on a real forestry crane demonstrates effective sway damping and successful obstacle avoidance. A video can be found at https://youtu.be/tEXDoeLLTxA.

  </details>



- **Acoustic Drone Package Delivery Detection**  
  François Marcoux, François Grondin  
  _2026-02-10_ · https://arxiv.org/abs/2602.09991v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In recent years, the illicit use of unmanned aerial vehicles (UAVs) for deliveries in restricted area such as prisons became a significant security challenge. While numerous studies have focused on UAV detection or localization, little attention has been given to delivery events identification. This study presents the first acoustic package delivery detection algorithm using a ground-based microphone array. The proposed method estimates both the drone's propeller speed and the delivery event using solely acoustic features. A deep neural network detects the presence of a drone and estimates the propeller's rotation speed or blade passing frequency (BPF) from a mel spectrogram. The algorithm analyzes the BPFs to identify probable delivery moments based on sudden changes before and after a specific time. Results demonstrate a mean absolute error of the blade passing frequency estimator of 16 Hz when the drone is less than 150 meters away from the microphone array. The drone presence detection estimator has a accuracy of 97%. The delivery detection algorithm correctly identifies 96% of events with a false positive rate of 8%. This study shows that deliveries can be identified using acoustic signals up to a range of 100 meters.

  </details>



- **HAPS-RIS and UAV Integrated Networks: A Unified Joint Multi-objective Framework**  
  Arman Azizi, Mostafa Rahmani Ghourtani, Mustafa A. Kishk, Hamed Ahmadi, Arman Farhang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09960v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Future 6G non-terrestrial networks aim to deliver ubiquitous connectivity to remote and undeserved regions, but unmanned aerial vehicle (UAV) base stations face fundamental challenges such as limited numbers and power budgets. To overcome these obstacles, high-altitude platform station (HAPS) equipped with a reconfigurable intelligent surface (RIS), so-called HAPS-RIS, is a promising candidate. We propose a novel unified joint multi-objective framework where UAVs and HAPS-RIS are fully integrated to extend coverage and enhance network performance. This joint multi-objective design maximizes the number of users served by the HAPS-RIS, minimizes the number of UAVs deployed and minimizes the total average UAV path loss subject to quality-of-service (QoS) and resource constraints. We propose a novel low-complexity solution strategy by proving the equivalence between minimizing the total average UAV path loss upper bound and k-means clustering, deriving a practical closed-form RIS phase-shift design, and introducing a mapping technique that collapses the combinatorial assignments into a zone radius and a bandwidth-portioning factor. Then, we propose a dynamic Pareto optimization technique to solve the transformed optimization problem. Extensive simulation results demonstrate that the proposed framework adapts seamlessly across operating regimes. A HAPS-RIS-only setup achieves full coverage at low data rates, but UAV assistance becomes indispensable as rate demands increase. By tuning a single bandwidth portioning factor, the model recovers UAV-only, HAPS-RIS-only and equal bandwidth portioning baselines within one formulation and consistently surpasses them across diverse rate requirements. The simulations also quantify a tangible trade-off between RIS scale and UAV deployment, enabling designers to trade increased RIS elements for fewer UAVs as service demands evolve.

  </details>


