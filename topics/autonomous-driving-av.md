# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-20 07:10 UTC_

Total papers shown: **8**


---

- **HiMAP: History-aware Map-occupancy Prediction with Fallback**  
  Yiming Xu, Yi Yang, Hao Cheng, Monika Sester  
  _2026-02-19_ · https://arxiv.org/abs/2602.17231v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate motion forecasting is critical for autonomous driving, yet most predictors rely on multi-object tracking (MOT) with identity association, assuming that objects are correctly and continuously tracked. When tracking fails due to, e.g., occlusion, identity switches, or missed detections, prediction quality degrades and safety risks increase. We present \textbf{HiMAP}, a tracking-free, trajectory prediction framework that remains reliable under MOT failures. HiMAP converts past detections into spatiotemporally invariant historical occupancy maps and introduces a historical query module that conditions on the current agent state to iteratively retrieve agent-specific history from unlabeled occupancy representations. The retrieved history is summarized by a temporal map embedding and, together with the final query and map context, drives a DETR-style decoder to produce multi-modal future trajectories. This design lifts identity reliance, supports streaming inference via reusable encodings, and serves as a robust fallback when tracking is unavailable. On Argoverse~2, HiMAP achieves performance comparable to tracking-based methods while operating without IDs, and it substantially outperforms strong baselines in the no-tracking setting, yielding relative gains of 11\% in FDE, 12\% in ADE, and a 4\% reduction in MR over a fine-tuned QCNet. Beyond aggregate metrics, HiMAP delivers stable forecasts for all agents simultaneously without waiting for tracking to recover, highlighting its practical value for safety-critical autonomy. The code is available under: https://github.com/XuYiMing83/HiMAP.

  </details>



- **Multi-session Localization and Mapping Exploiting Topological Information**  
  Lorenzo Montano-Olivan, Julio A. Placed, Luis Montano, Maria T. Lazaro  
  _2026-02-19_ · https://arxiv.org/abs/2602.17226v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Operating in previously visited environments is becoming increasingly crucial for autonomous systems, with direct applications in autonomous driving, surveying, and warehouse or household robotics. This repeated exposure to observing the same areas poses significant challenges for mapping and localization -- key components for enabling any higher-level task. In this work, we propose a novel multi-session framework that builds on map-based localization, in contrast to the common practice of greedily running full SLAM sessions and trying to find correspondences between the resulting maps. Our approach incorporates a topology-informed, uncertainty-aware decision-making mechanism that analyzes the pose-graph structure to detect low-connectivity regions, selectively triggering mapping and loop closing modules. The resulting map and pose-graph are seamlessly integrated into the existing model, reducing accumulated error and enhancing global consistency. We validate our method on overlapping sequences from datasets and demonstrate its effectiveness in a real-world mine-like environment.

  </details>



- **Distributed Virtual Model Control for Scalable Human-Robot Collaboration in Shared Workspace**  
  Yi Zhang, Omar Faris, Chapa Sirithunge, Kai-Fung Chu, Fumiya Iida, Fulvio Forni  
  _2026-02-19_ · https://arxiv.org/abs/2602.17415v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a decentralized, agent agnostic, and safety-aware control framework for human-robot collaboration based on Virtual Model Control (VMC). In our approach, both humans and robots are embedded in the same virtual-component-shaped workspace, where motion is the result of the interaction with virtual springs and dampers rather than explicit trajectory planning. A decentralized, force-based stall detector identifies deadlocks, which are resolved through negotiation. This reduces the probability of robots getting stuck in the block placement task from up to 61.2% to zero in our experiments. The framework scales without structural changes thanks to the distributed implementation: in experiments we demonstrate safe collaboration with up to two robots and two humans, and in simulation up to four robots, maintaining inter-agent separation at around 20 cm. Results show that the method shapes robot behavior intuitively by adjusting control parameters and achieves deadlock-free operation across team sizes in all tested scenarios.

  </details>



- **Conditional Flow Matching for Continuous Anomaly Detection in Autonomous Driving on a Manifold-Aware Spectral Space**  
  Antonio Guillen-Perez  
  _2026-02-19_ · https://arxiv.org/abs/2602.17586v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Safety validation for Level 4 autonomous vehicles (AVs) is currently bottlenecked by the inability to scale the detection of rare, high-risk long-tail scenarios using traditional rule-based heuristics. We present Deep-Flow, an unsupervised framework for safety-critical anomaly detection that utilizes Optimal Transport Conditional Flow Matching (OT-CFM) to characterize the continuous probability density of expert human driving behavior. Unlike standard generative approaches that operate in unstable, high-dimensional coordinate spaces, Deep-Flow constrains the generative process to a low-rank spectral manifold via a Principal Component Analysis (PCA) bottleneck. This ensures kinematic smoothness by design and enables the computation of the exact Jacobian trace for numerically stable, deterministic log-likelihood estimation. To resolve multi-modal ambiguity at complex junctions, we utilize an Early Fusion Transformer encoder with lane-aware goal conditioning, featuring a direct skip-connection to the flow head to maintain intent-integrity throughout the network. We introduce a kinematic complexity weighting scheme that prioritizes high-energy maneuvers (quantified via path tortuosity and jerk) during the simulation-free training process. Evaluated on the Waymo Open Motion Dataset (WOMD), our framework achieves an AUC-ROC of 0.766 against a heuristic golden set of safety-critical events. More significantly, our analysis reveals a fundamental distinction between kinematic danger and semantic non-compliance. Deep-Flow identifies a critical predictability gap by surfacing out-of-distribution behaviors, such as lane-boundary violations and non-normative junction maneuvers, that traditional safety filters overlook. This work provides a mathematically rigorous foundation for defining statistical safety gates, enabling objective, data-driven validation for the safe deployment of autonomous fleets.

  </details>



- **Hybrid System Planning using a Mixed-Integer ADMM Heuristic and Hybrid Zonotopes**  
  Joshua A. Robbins, Andrew F. Thompson, Jonah J. Glunt, Herschel C. Pangborn  
  _2026-02-19_ · https://arxiv.org/abs/2602.17574v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Embedded optimization-based planning for hybrid systems is challenging due to the use of mixed-integer programming, which is computationally intensive and often sensitive to the specific numerical formulation. To address that challenge, this article proposes a framework for motion planning of hybrid systems that pairs hybrid zonotopes - an advanced set representation - with a new alternating direction method of multipliers (ADMM) mixed-integer programming heuristic. A general treatment of piecewise affine (PWA) system reachability analysis using hybrid zonotopes is presented and extended to formulate optimal planning problems. Sets produced using the proposed identities have lower memory complexity and tighter convex relaxations than equivalent sets produced from preexisting techniques. The proposed ADMM heuristic makes efficient use of the hybrid zonotope structure. For planning problems formulated as hybrid zonotopes, the proposed heuristic achieves improved convergence rates as compared to state-of-the-art mixed-integer programming heuristics. The proposed methods for hybrid system planning on embedded hardware are experimentally applied in a combined behavior and motion planning scenario for autonomous driving.

  </details>



- **Toward a Fully Autonomous, AI-Native Particle Accelerator**  
  Chris Tennant  
  _2026-02-19_ · https://arxiv.org/abs/2602.17536v1 · `physics.acc-ph`  
  <details><summary>Abstract</summary>

  This position paper presents a vision for self-driving particle accelerators that operate autonomously with minimal human intervention. We propose that future facilities be designed through artificial intelligence (AI) co-design, where AI jointly optimizes the accelerator lattice, diagnostics, and science application from inception to maximize performance while enabling autonomous operation. Rather than retrofitting AI onto human-centric systems, we envision facilities designed from the ground up as AI-native platforms. We outline nine critical research thrusts spanning agentic control architectures, knowledge integration, adaptive learning, digital twins, health monitoring, safety frameworks, modular hardware design, multimodal data fusion, and cross-domain collaboration. This roadmap aims to guide the accelerator community toward a future where AI-driven design and operation deliver unprecedented science output and reliability.

  </details>



- **Nonlinear Predictive Control of the Continuum and Hybrid Dynamics of a Suspended Deformable Cable for Aerial Pick and Place**  
  Antonio Rapuano, Yaolei Shen, Federico Califano, Chiara Gabellieri, Antonio Franchi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17199v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a framework for aerial manipulation of an extensible cable that combines a high-fidelity model based on partial differential equations (PDEs) with a reduced-order representation suitable for real-time control. The PDEs are discretised using a finite-difference method, and proper orthogonal decomposition is employed to extract a reduced-order model (ROM) that retains the dominant deformation modes while significantly reducing computational complexity. Based on this ROM, a nonlinear model predictive control scheme is formulated, capable of stabilizing cable oscillations and handling hybrid transitions such as payload attachment and detachment. Simulation results confirm the stability, efficiency, and robustness of the ROM, as well as the effectiveness of the controller in regulating cable dynamics under a range of operating conditions. Additional simulations illustrate the application of the ROM for trajectory planning in constrained environments, demonstrating the versatility of the proposed approach. Overall, the framework enables real-time, dynamics-aware control of unmanned aerial vehicles (UAVs) carrying suspended flexible cables.

  </details>



- **3D Scene Rendering with Multimodal Gaussian Splatting**  
  Chi-Shiang Gau, Konstantinos D. Polyzos, Athanasios Bacharis, Saketh Madhuvarasu, Tara Javidi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17124v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  3D scene reconstruction and rendering are core tasks in computer vision, with applications spanning industrial monitoring, robotics, and autonomous driving. Recent advances in 3D Gaussian Splatting (GS) and its variants have achieved impressive rendering fidelity while maintaining high computational and memory efficiency. However, conventional vision-based GS pipelines typically rely on a sufficient number of camera views to initialize the Gaussian primitives and train their parameters, typically incurring additional processing cost during initialization while falling short in conditions where visual cues are unreliable, such as adverse weather, low illumination, or partial occlusions. To cope with these challenges, and motivated by the robustness of radio-frequency (RF) signals to weather, lighting, and occlusions, we introduce a multimodal framework that integrates RF sensing, such as automotive radar, with GS-based rendering as a more efficient and robust alternative to vision-only GS rendering. The proposed approach enables efficient depth prediction from only sparse RF-based depth measurements, yielding a high-quality 3D point cloud for initializing Gaussian functions across diverse GS architectures. Numerical tests demonstrate the merits of judiciously incorporating RF sensing into GS pipelines, achieving high-fidelity 3D scene rendering driven by RF-informed structural accuracy.

  </details>


