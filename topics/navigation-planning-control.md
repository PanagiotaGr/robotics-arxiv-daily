# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-20 07:10 UTC_

Total papers shown: **9**


---

- **NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting**  
  Jiwei Shan, Zeyu Cai, Yirui Li, Yongbo Chen, Lijun Han, Yun-hui Liu, Hesheng Wang, Shing Shin Cheng  
  _2026-02-19_ · https://arxiv.org/abs/2602.17182v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Visual simultaneous localization and mapping (V-SLAM) is a fundamental capability for autonomous perception and navigation. However, endoscopic scenes violate the rigidity assumption due to persistent soft-tissue deformations, creating a strong coupling ambiguity between camera ego-motion and intrinsic deformation. Although recent monocular non-rigid SLAM methods have made notable progress, they often lack effective decoupling mechanisms and rely on sparse or low-fidelity scene representations, which leads to tracking drift and limited reconstruction quality. To address these limitations, we propose NRGS-SLAM, a monocular non-rigid SLAM system for endoscopy based on 3D Gaussian Splatting. To resolve the coupling ambiguity, we introduce a deformation-aware 3D Gaussian map that augments each Gaussian primitive with a learnable deformation probability, optimized via a Bayesian self-supervision strategy without requiring external non-rigidity labels. Building on this representation, we design a deformable tracking module that performs robust coarse-to-fine pose estimation by prioritizing low-deformation regions, followed by efficient per-frame deformation updates. A carefully designed deformable mapping module progressively expands and refines the map, balancing representational capacity and computational efficiency. In addition, a unified robust geometric loss incorporates external geometric priors to mitigate the inherent ill-posedness of monocular non-rigid SLAM. Extensive experiments on multiple public endoscopic datasets demonstrate that NRGS-SLAM achieves more accurate camera pose estimation (up to 50\% reduction in RMSE) and higher-quality photo-realistic reconstructions than state-of-the-art methods. Comprehensive ablation studies further validate the effectiveness of our key design choices. Source code will be publicly available upon paper acceptance.

  </details>



- **Dodging the Moose: Experimental Insights in Real-Life Automated Collision Avoidance**  
  Leila Gharavi, Simone Baldi, Yuki Hosomi, Tona Sato, Bart De Schutter, Binh-Minh Nguyen, Hiroshi Fujimoto  
  _2026-02-19_ · https://arxiv.org/abs/2602.17512v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  The sudden appearance of a static obstacle on the road, i.e. the moose test, is a well-known emergency scenario in collision avoidance for automated driving. Model Predictive Control (MPC) has long been employed for planning and control of automated vehicles in the state of the art. However, real-time implementation of automated collision avoidance in emergency scenarios such as the moose test remains unaddressed due to the high computational demand of MPC for evasive action in such hazardous scenarios. This paper offers new insights into real-time collision avoidance via the experimental imple- mentation of MPC for motion planning after a sudden and unexpected appearance of a static obstacle. As the state-of-the-art nonlinear MPC shows limited capability to provide an acceptable solution in real-time, we propose a human-like feed-forward planner to assist when the MPC optimization problem is either infeasible or unable to find a suitable solution due to the poor quality of its initial guess. We introduce the concept of maximum steering maneuver to design the feed-forward planner and mimic a human-like reaction after detecting the static obstacle on the road. Real-life experiments are conducted across various speeds and level of emergency using FPEV2-Kanon electric vehicle. Moreover, we demonstrate the effectiveness of our planning strategy via comparison with the state-of- the-art MPC motion planner.

  </details>



- **RA-Nav: A Risk-Aware Navigation System Based on Semantic Segmentation for Aerial Robots in Unpredictable Environments**  
  Ziyi Zong, Xin Dong, Jinwu Xiang, Daochun Li, Zhan Tu  
  _2026-02-19_ · https://arxiv.org/abs/2602.17515v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Existing aerial robot navigation systems typically plan paths around static and dynamic obstacles, but fail to adapt when a static obstacle suddenly moves. Integrating environmental semantic awareness enables estimation of potential risks posed by suddenly moving obstacles. In this paper, we propose RA- Nav, a risk-aware navigation framework based on semantic segmentation. A lightweight multi-scale semantic segmentation network identifies obstacle categories in real time. These obstacles are further classified into three types: stationary, temporarily static, and dynamic. For each type, corresponding risk estimation functions are designed to enable real-time risk prediction, based on which a complete local risk map is constructed. Based on this map, the risk-informed path search algorithm is designed to guarantee planning that balances path efficiency and safety. Trajectory optimization is then applied to generate trajectories that are safe, smooth, and dynamically feasible. Comparative simulations demonstrate that RA-Nav achieves higher success rates than baselines in sudden obstacle state transition scenarios. Its effectiveness is further validated in simulations using real- world data.

  </details>



- **Bluetooth Phased-array Aided Inertial Navigation Using Factor Graphs: Experimental Verification**  
  Glen Hjelmerud Mørkbak Sørensen, Torleiv H. Bryne, Kristoffer Gryte, Tor Arne Johansen  
  _2026-02-19_ · https://arxiv.org/abs/2602.17407v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Phased-array Bluetooth systems have emerged as a low-cost alternative for performing aided inertial navigation in GNSS-denied use cases such as warehouse logistics, drone landings, and autonomous docking. Basing a navigation system off of commercial-off-the-shelf components may reduce the barrier of entry for phased-array radio navigation systems, albeit at the cost of significantly noisier measurements and relatively short feasible range. In this paper, we compare robust estimation strategies for a factor graph optimisation-based estimator using experimental data collected from multirotor drone flight. We evaluate performance in loss-of-GNSS scenarios when aided by Bluetooth angular measurements, as well as range or barometric pressure.

  </details>



- **Hybrid System Planning using a Mixed-Integer ADMM Heuristic and Hybrid Zonotopes**  
  Joshua A. Robbins, Andrew F. Thompson, Jonah J. Glunt, Herschel C. Pangborn  
  _2026-02-19_ · https://arxiv.org/abs/2602.17574v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Embedded optimization-based planning for hybrid systems is challenging due to the use of mixed-integer programming, which is computationally intensive and often sensitive to the specific numerical formulation. To address that challenge, this article proposes a framework for motion planning of hybrid systems that pairs hybrid zonotopes - an advanced set representation - with a new alternating direction method of multipliers (ADMM) mixed-integer programming heuristic. A general treatment of piecewise affine (PWA) system reachability analysis using hybrid zonotopes is presented and extended to formulate optimal planning problems. Sets produced using the proposed identities have lower memory complexity and tighter convex relaxations than equivalent sets produced from preexisting techniques. The proposed ADMM heuristic makes efficient use of the hybrid zonotope structure. For planning problems formulated as hybrid zonotopes, the proposed heuristic achieves improved convergence rates as compared to state-of-the-art mixed-integer programming heuristics. The proposed methods for hybrid system planning on embedded hardware are experimentally applied in a combined behavior and motion planning scenario for autonomous driving.

  </details>



- **Nonlinear Predictive Control of the Continuum and Hybrid Dynamics of a Suspended Deformable Cable for Aerial Pick and Place**  
  Antonio Rapuano, Yaolei Shen, Federico Califano, Chiara Gabellieri, Antonio Franchi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17199v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a framework for aerial manipulation of an extensible cable that combines a high-fidelity model based on partial differential equations (PDEs) with a reduced-order representation suitable for real-time control. The PDEs are discretised using a finite-difference method, and proper orthogonal decomposition is employed to extract a reduced-order model (ROM) that retains the dominant deformation modes while significantly reducing computational complexity. Based on this ROM, a nonlinear model predictive control scheme is formulated, capable of stabilizing cable oscillations and handling hybrid transitions such as payload attachment and detachment. Simulation results confirm the stability, efficiency, and robustness of the ROM, as well as the effectiveness of the controller in regulating cable dynamics under a range of operating conditions. Additional simulations illustrate the application of the ROM for trajectory planning in constrained environments, demonstrating the versatility of the proposed approach. Overall, the framework enables real-time, dynamics-aware control of unmanned aerial vehicles (UAVs) carrying suspended flexible cables.

  </details>



- **Graph Neural Model Predictive Control for High-Dimensional Systems**  
  Patrick Benito Eberhard, Luis Pabon, Daniele Gammelli, Hugo Buurmeijer, Amon Lahr, Mark Leone, Andrea Carron, Marco Pavone  
  _2026-02-19_ · https://arxiv.org/abs/2602.17601v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The control of high-dimensional systems, such as soft robots, requires models that faithfully capture complex dynamics while remaining computationally tractable. This work presents a framework that integrates Graph Neural Network (GNN)-based dynamics models with structure-exploiting Model Predictive Control to enable real-time control of high-dimensional systems. By representing the system as a graph with localized interactions, the GNN preserves sparsity, while a tailored condensing algorithm eliminates state variables from the control problem, ensuring efficient computation. The complexity of our condensing algorithm scales linearly with the number of system nodes, and leverages Graphics Processing Unit (GPU) parallelization to achieve real-time performance. The proposed approach is validated in simulation and experimentally on a physical soft robotic trunk. Results show that our method scales to systems with up to 1,000 nodes at 100 Hz in closed-loop, and demonstrates real-time reference tracking on hardware with sub-centimeter accuracy, outperforming baselines by 63.6%. Finally, we show the capability of our method to achieve effective full-body obstacle avoidance.

  </details>



- **Flickering Multi-Armed Bandits**  
  Sourav Chakraborty, Amit Kiran Rege, Claire Monteleoni, Lijun Chen  
  _2026-02-19_ · https://arxiv.org/abs/2602.17315v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We introduce Flickering Multi-Armed Bandits (FMAB), a new MAB framework where the set of available arms (or actions) can change at each round, and the available set at any time may depend on the agent's previously selected arm. We model this constrained, evolving availability using random graph processes, where arms are nodes and the agent's movement is restricted to its local neighborhood. We analyze this problem under two random graph models: an i.i.d. Erdős--Rényi (ER) process and an Edge-Markovian process. We propose and analyze a two-phase algorithm that employs a lazy random walk for exploration to efficiently identify the optimal arm, followed by a navigation and commitment phase for exploitation. We establish high-probability and expected sublinear regret bounds for both graph settings. We show that the exploration cost of our algorithm is near-optimal by establishing a matching information-theoretic lower bound for this problem class, highlighting the fundamental cost of exploration under local-move constraints. We complement our theoretical guarantees with numerical simulations, including a scenario of a robotic ground vehicle scouting a disaster-affected region.

  </details>



- **Assessing Ionospheric Scintillation Risk for Direct-to-Cellular Communications using Frequency-Scaled GNSS Observations**  
  Abdollah Masoud Darya, Muhammad Mubasshir Shaikh  
  _2026-02-19_ · https://arxiv.org/abs/2602.17143v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  One of the key issues facing Direct-to-Cellular (D2C) satellite communication systems is ionospheric scintillation on the uplink and downlink, which can significantly degrade link quality. This work investigates the spatial and temporal characteristics of amplitude scintillation at D2C frequencies by scaling L-band scintillation observations from Global Navigation Satellite Systems (GNSS) receivers to bands relevant to D2C operation, including the low-band, and 3GPP's N255 and N256. These observations are then compared to scaled radio-occultation scintillation observations from the FORMOSAT-7/COSMIC-2 (F7/C2) mission, which can be used in regions that do not possess ground-based scintillation monitoring stations. As a proof of concept, five years of ground-based GNSS scintillation data from Sharjah, United Arab Emirates, together with two years of F7/C2 observations over the same region, corresponding to the ascending phase of Solar Cycle 25, are analyzed. Both space-based and ground-based observations indicate a pronounced diurnal scintillation peak between 20--22 local time, particularly during the equinoxes, with occurrence rates increasing with solar activity. Ground-based observations also reveal a strong azimuth dependence, with most scintillation events occurring on southward satellite links. The scintillation occurrence rate at the low-band is more than twice that observed at N255 and N256, highlighting the increased robustness of higher D2C bands to ionospheric scintillation. These results demonstrate how GNSS scintillation observations can be leveraged to characterize and anticipate scintillation-induced D2C link impairments, which help in D2C system design and the implementation of scintillation mitigation strategies.

  </details>


