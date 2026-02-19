# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-19 07:13 UTC_

Total papers shown: **4**


---

- **Decentralized and Fully Onboard: Range-Aided Cooperative Localization and Navigation on Micro Aerial Vehicles**  
  Abhishek Goudar, Angela P. Schoellig  
  _2026-02-18_ · https://arxiv.org/abs/2602.16594v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Controlling a team of robots in a coordinated manner is challenging because centralized approaches (where all computation is performed on a central machine) scale poorly, and globally referenced external localization systems may not always be available. In this work, we consider the problem of range-aided decentralized localization and formation control. In such a setting, each robot estimates its relative pose by combining data only from onboard odometry sensors and distance measurements to other robots in the team. Additionally, each robot calculates the control inputs necessary to collaboratively navigate an environment to accomplish a specific task, for example, moving in a desired formation while monitoring an area. We present a block coordinate descent approach to localization that does not require strict coordination between the robots. We present a novel formulation for formation control as inference on factor graphs that takes into account the state estimation uncertainty and can be solved efficiently. Our approach to range-aided localization and formation-based navigation is completely decentralized, does not require specialized trajectories to maintain formation, and achieves decimeter-level positioning and formation control accuracy. We demonstrate our approach through multiple real experiments involving formation flights in diverse indoor and outdoor environments.

  </details>



- **Markerless Robot Detection and 6D Pose Estimation for Multi-Agent SLAM**  
  Markus Rueggeberg, Maximilian Ulmer, Maximilian Durner, Wout Boerdijk, Marcus Gerhard Mueller, Rudolph Triebel, Riccardo Giubilato  
  _2026-02-18_ · https://arxiv.org/abs/2602.16308v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The capability of multi-robot SLAM approaches to merge localization history and maps from different observers is often challenged by the difficulty in establishing data association. Loop closure detection between perceptual inputs of different robotic agents is easily compromised in the context of perceptual aliasing, or when perspectives differ significantly. For this reason, direct mutual observation among robots is a powerful way to connect partial SLAM graphs, but often relies on the presence of calibrated arrays of fiducial markers (e.g., AprilTag arrays), which severely limits the range of observations and frequently fails under sharp lighting conditions, e.g., reflections or overexposure. In this work, we propose a novel solution to this problem leveraging recent advances in Deep-Learning-based 6D pose estimation. We feature markerless pose estimation as part of a decentralized multi-robot SLAM system and demonstrate the benefit to the relative localization accuracy among the robotic team. The solution is validated experimentally on data recorded in a test field campaign on a planetary analogous environment.

  </details>



- **Dynamic Modeling and MPC for Locomotion of Tendon-Driven Soft Quadruped**  
  Saumya Karan, Neerav Maram, Suraj Borate, Madhu Vadali  
  _2026-02-18_ · https://arxiv.org/abs/2602.16371v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  SLOT (Soft Legged Omnidirectional Tetrapod), a tendon-driven soft quadruped robot with 3D-printed TPU legs, is presented to study physics-informed modeling and control of compliant legged locomotion using only four actuators. Each leg is modeled as a deformable continuum using discrete Cosserat rod theory, enabling the capture of large bending deformations, distributed elasticity, tendon actuation, and ground contact interactions. A modular whole-body modeling framework is introduced, in which compliant leg dynamics are represented through physically consistent reaction forces applied to a rigid torso, providing a scalable interface between continuum soft limbs and rigid-body locomotion dynamics. This formulation allows efficient whole-body simulation and real-time control without sacrificing physical fidelity. The proposed model is embedded into a convex model predictive control framework that optimizes ground reaction forces over a 0.495 s prediction horizon and maps them to tendon actuation through a physics-informed force-angle relationship. The resulting controller achieves asymptotic stability under diverse perturbations. The framework is experimentally validated on a physical prototype during crawling and walking gaits, achieving high accuracy with less than 5 mm RMSE in center of mass trajectories. These results demonstrate a generalizable approach for integrating continuum soft legs into model-based locomotion control, advancing scalable and reusable modeling and control methods for soft quadruped robots.

  </details>



- **Causal and Compositional Abstraction**  
  Robin Lorenz, Sean Tull  
  _2026-02-18_ · https://arxiv.org/abs/2602.16612v1 · `cs.LO`  
  <details><summary>Abstract</summary>

  Abstracting from a low level to a more explanatory high level of description, and ideally while preserving causal structure, is fundamental to scientific practice, to causal inference problems, and to robust, efficient and interpretable AI. We present a general account of abstractions between low and high level models as natural transformations, focusing on the case of causal models. This provides a new formalisation of causal abstraction, unifying several notions in the literature, including constructive causal abstraction, Q-$τ$ consistency, abstractions based on interchange interventions, and `distributed' causal abstractions. Our approach is formalised in terms of category theory, and uses the general notion of a compositional model with a given set of queries and semantics in a monoidal, cd- or Markov category; causal models and their queries such as interventions being special cases. We identify two basic notions of abstraction: downward abstractions mapping queries from high to low level; and upward abstractions, mapping concrete queries such as Do-interventions from low to high. Although usually presented as the latter, we show how common causal abstractions may, more fundamentally, be understood in terms of the former. Our approach also leads us to consider a new stronger notion of `component-level' abstraction, applying to the individual components of a model. In particular, this yields a novel, strengthened form of constructive causal abstraction at the mechanism-level, for which we prove characterisation results. Finally, we show that abstraction can be generalised to further compositional models, including those with a quantum semantics implemented by quantum circuits, and we take first steps in exploring abstractions between quantum compositional circuit models and high-level classical causal models as a means to explainable quantum AI.

  </details>


