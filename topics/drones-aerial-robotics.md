# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-12 07:15 UTC_

Total papers shown: **6**


---

- **Biomimetic Mantaray robot toward the underwater autonomous -- Experimental verification of swimming and diving by flapping motion -**  
  Kenta Tabata, Ryosuke Oku, Jun Ito, Renato Miyagusuku, Koichi Ozaki  
  _2026-02-11_ · https://arxiv.org/abs/2602.10904v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This study presents the development and experimental verification of a biomimetic manta ray robot for underwater autonomous exploration. Inspired by manta rays, the robot uses flapping motion for propulsion to minimize seabed disturbance and enhance efficiency compared to traditional screw propulsion. The robot features pectoral fins driven by servo motors and a streamlined control box to reduce fluid resistance. The control system, powered by a Raspberry Pi 3B, includes an IMU and pressure sensor for real-time monitoring and control. Experiments in a pool assessed the robot's swimming and diving capabilities. Results show stable swimming and diving motions with PD control. The robot is suitable for applications in environments like aquariums and fish nurseries, requiring minimal disturbance and efficient maneuverability. Our findings demonstrate the potential of bio-inspired robotic designs to improve ecological monitoring and underwater exploration.

  </details>



- **Multi-Task Reinforcement Learning of Drone Aerobatics by Exploiting Geometric Symmetries**  
  Zhanyu Guo, Zikang Yin, Guobin Zhu, Shiliang Guo, Shiyu Zhao  
  _2026-02-11_ · https://arxiv.org/abs/2602.10997v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Flight control for autonomous micro aerial vehicles (MAVs) is evolving from steady flight near equilibrium points toward more aggressive aerobatic maneuvers, such as flips, rolls, and Power Loop. Although reinforcement learning (RL) has shown great potential in these tasks, conventional RL methods often suffer from low data efficiency and limited generalization. This challenge becomes more pronounced in multi-task scenarios where a single policy is required to master multiple maneuvers. In this paper, we propose a novel end-to-end multi-task reinforcement learning framework, called GEAR (Geometric Equivariant Aerobatics Reinforcement), which fully exploits the inherent SO(2) rotational symmetry in MAV dynamics and explicitly incorporates this property into the policy network architecture. By integrating an equivariant actor network, FiLM-based task modulation, and a multi-head critic, GEAR achieves both efficiency and flexibility in learning diverse aerobatic maneuvers, enabling a data-efficient, robust, and unified framework for aerobatic control. GEAR attains a 98.85\% success rate across various aerobatic tasks, significantly outperforming baseline methods. In real-world experiments, GEAR demonstrates stable execution of multiple maneuvers and the capability to combine basic motion primitives to complete complex aerobatics.

  </details>



- **(MGS)$^2$-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization**  
  Minglei Li, Mengfan He, Chao Chen, Ziyang Meng  
  _2026-02-11_ · https://arxiv.org/abs/2602.10704v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Cross-view geo-localization (CVGL) is pivotal for GNSS-denied UAV navigation but remains brittle under the drastic geometric misalignment between oblique aerial views and orthographic satellite references. Existing methods predominantly operate within a 2D manifold, neglecting the underlying 3D geometry where view-dependent vertical facades (macro-structure) and scale variations (micro-scale) severely corrupt feature alignment. To bridge this gap, we propose (MGS)$^2$, a geometry-grounded framework. The core of our innovation is the Macro-Geometric Structure Filtering (MGSF) module. Unlike pixel-wise matching sensitive to noise, MGSF leverages dilated geometric gradients to physically filter out high-frequency facade artifacts while enhancing the view-invariant horizontal plane, directly addressing the domain shift. To guarantee robust input for this structural filtering, we explicitly incorporate a Micro-Geometric Scale Adaptation (MGSA) module. MGSA utilizes depth priors to dynamically rectify scale discrepancies via multi-branch feature fusion. Furthermore, a Geometric-Appearance Contrastive Distillation (GACD) loss is designed to strictly discriminate against oblique occlusions. Extensive experiments demonstrate that (MGS)$^2$ achieves state-of-the-art performance, recording a Recall@1 of 97.5\% on University-1652 and 97.02\% on SUES-200. Furthermore, the framework exhibits superior cross-dataset generalization against geometric ambiguity. The code is available at: \href{https://github.com/GabrielLi1473/MGS-Net}{https://github.com/GabrielLi1473/MGS-Net}.

  </details>



- **Multi-UAV Trajectory Optimization for Bearing-Only Localization in GPS Denied Environments**  
  Alfonso Sciacchitano, Liraz Mudrik, Sean Kragelund, Isaac Kaminer  
  _2026-02-11_ · https://arxiv.org/abs/2602.11116v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Accurate localization of maritime targets by unmanned aerial vehicles (UAVs) remains challenging in GPS-denied environments. UAVs equipped with gimballed electro-optical sensors are typically used to localize targets, however, reliance on these sensors increases mechanical complexity, cost, and susceptibility to single-point failures, limiting scalability and robustness in multi-UAV operations. This work presents a new trajectory optimization framework that enables cooperative target localization using UAVs with fixed, non-gimballed cameras operating in coordination with a surface vessel. This estimation-aware optimization generates dynamically feasible trajectories that explicitly account for mission constraints, platform dynamics, and out-of-frame events. Estimation-aware trajectories outperform heuristic paths by reducing localization error by more than a factor of two, motivating their use in cooperative operations. Results further demonstrate that coordinated UAVs with fixed, non-gimballed cameras achieve localization accuracy that meets or exceeds that of single gimballed systems, while substantially lowering system complexity and cost, enabling scalability, and enhancing mission resilience.

  </details>



- **Transfer to Sky: Unveil Low-Altitude Route-Level Radio Maps via Ground Crowdsourced Data**  
  Wenlihan Lu, Huacong Chen, Ruiyang Duan, Weijie Yuan, Shijian Gao  
  _2026-02-11_ · https://arxiv.org/abs/2602.10736v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  The expansion of the low-altitude economy is contingent on reliable cellular connectivity for unmanned aerial vehicles (UAVs). A key challenge in pre-flight planning is predicting communication link quality along proposed and pre-defined routes, a task hampered by sparse measurements that render existing radio map methods ineffective. This paper introduces a transfer learning framework for high-fidelity route-level radio map prediction. Our key insight is to leverage abundant crowdsourced ground signals as auxiliary supervision. To bridge the significant domain gap between ground and aerial data and address spatial sparsity, our framework learns general propagation priors from simulation, performs adversarial alignment of the feature spaces, and is fine-tuned on limited real UAV measurements. Extensive experiments on a real-world dataset from Meituan show that our method achieves over 50% higher accuracy in predicting Route RSRP compared to state-of-the-art baselines.

  </details>



- **Omnidirectional Dual-Arm Aerial Manipulator with Proprioceptive Contact Localization for Landing on Slanted Roofs**  
  Martijn B. J. Brummelhuis, Nathan F. Lepora, Salua Hamaza  
  _2026-02-11_ · https://arxiv.org/abs/2602.10703v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Operating drones in urban environments often means they need to land on rooftops, which can have different geometries and surface irregularities. Accurately detecting roof inclination using conventional sensing methods, such as vision-based or acoustic techniques, can be unreliable, as measurement quality is strongly influenced by external factors including weather conditions and surface materials. To overcome these challenges, we propose a novel unmanned aerial manipulator morphology featuring a dual-arm aerial manipulator with an omnidirectional 3D workspace and extended reach. Building on this design, we develop a proprioceptive contact detection and contact localization strategy based on a momentum-based torque observer. This enables the UAM to infer the inclination of slanted surfaces blindly - through physical interaction - prior to touchdown. We validate the approach in flight experiments, demonstrating robust landings on surfaces with inclinations of up to 30.5 degrees and achieving an average surface inclination estimation error of 2.87 degrees over 9 experiments at different incline angles.

  </details>


