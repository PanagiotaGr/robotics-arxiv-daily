# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-19 07:13 UTC_

Total papers shown: **8**


---

- **Decentralized and Fully Onboard: Range-Aided Cooperative Localization and Navigation on Micro Aerial Vehicles**  
  Abhishek Goudar, Angela P. Schoellig  
  _2026-02-18_ · https://arxiv.org/abs/2602.16594v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Controlling a team of robots in a coordinated manner is challenging because centralized approaches (where all computation is performed on a central machine) scale poorly, and globally referenced external localization systems may not always be available. In this work, we consider the problem of range-aided decentralized localization and formation control. In such a setting, each robot estimates its relative pose by combining data only from onboard odometry sensors and distance measurements to other robots in the team. Additionally, each robot calculates the control inputs necessary to collaboratively navigate an environment to accomplish a specific task, for example, moving in a desired formation while monitoring an area. We present a block coordinate descent approach to localization that does not require strict coordination between the robots. We present a novel formulation for formation control as inference on factor graphs that takes into account the state estimation uncertainty and can be solved efficiently. Our approach to range-aided localization and formation-based navigation is completely decentralized, does not require specialized trajectories to maintain formation, and achieves decimeter-level positioning and formation control accuracy. We demonstrate our approach through multiple real experiments involving formation flights in diverse indoor and outdoor environments.

  </details>



- **Dynamic Modeling and MPC for Locomotion of Tendon-Driven Soft Quadruped**  
  Saumya Karan, Neerav Maram, Suraj Borate, Madhu Vadali  
  _2026-02-18_ · https://arxiv.org/abs/2602.16371v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  SLOT (Soft Legged Omnidirectional Tetrapod), a tendon-driven soft quadruped robot with 3D-printed TPU legs, is presented to study physics-informed modeling and control of compliant legged locomotion using only four actuators. Each leg is modeled as a deformable continuum using discrete Cosserat rod theory, enabling the capture of large bending deformations, distributed elasticity, tendon actuation, and ground contact interactions. A modular whole-body modeling framework is introduced, in which compliant leg dynamics are represented through physically consistent reaction forces applied to a rigid torso, providing a scalable interface between continuum soft limbs and rigid-body locomotion dynamics. This formulation allows efficient whole-body simulation and real-time control without sacrificing physical fidelity. The proposed model is embedded into a convex model predictive control framework that optimizes ground reaction forces over a 0.495 s prediction horizon and maps them to tendon actuation through a physics-informed force-angle relationship. The resulting controller achieves asymptotic stability under diverse perturbations. The framework is experimentally validated on a physical prototype during crawling and walking gaits, achieving high accuracy with less than 5 mm RMSE in center of mass trajectories. These results demonstrate a generalizable approach for integrating continuum soft legs into model-based locomotion control, advancing scalable and reusable modeling and control methods for soft quadruped robots.

  </details>



- **Nonplanar Model Predictive Control for Autonomous Vehicles with Recursive Sparse Gaussian Process Dynamics**  
  Ahmad Amine, Kabir Puri, Viet-Anh Le, Rahul Mangharam  
  _2026-02-18_ · https://arxiv.org/abs/2602.16206v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper proposes a nonplanar model predictive control (MPC) framework for autonomous vehicles operating on nonplanar terrain. To approximate complex vehicle dynamics in such environments, we develop a geometry-aware modeling approach that learns a residual Gaussian Process (GP). By utilizing a recursive sparse GP, the framework enables real-time adaptation to varying terrain geometry. The effectiveness of the learned model is demonstrated in a reference-tracking task using a Model Predictive Path Integral (MPPI) controller. Validation within a custom Isaac Sim environment confirms the framework's capability to maintain high tracking accuracy on challenging 3D surfaces.

  </details>



- **Learning Humanoid End-Effector Control for Open-Vocabulary Visual Loco-Manipulation**  
  Runpei Dong, Ziyan Li, Xialin He, Saurabh Gupta  
  _2026-02-18_ · https://arxiv.org/abs/2602.16705v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual loco-manipulation of arbitrary objects in the wild with humanoid robots requires accurate end-effector (EE) control and a generalizable understanding of the scene via visual inputs (e.g., RGB-D images). Existing approaches are based on real-world imitation learning and exhibit limited generalization due to the difficulty in collecting large-scale training datasets. This paper presents a new paradigm, HERO, for object loco-manipulation with humanoid robots that combines the strong generalization and open-vocabulary understanding of large vision models with strong control performance from simulated training. We achieve this by designing an accurate residual-aware EE tracking policy. This EE tracking policy combines classical robotics with machine learning. It uses a) inverse kinematics to convert residual end-effector targets into reference trajectories, b) a learned neural forward model for accurate forward kinematics, c) goal adjustment, and d) replanning. Together, these innovations help us cut down the end-effector tracking error by 3.2x. We use this accurate end-effector tracker to build a modular system for loco-manipulation, where we use open-vocabulary large vision models for strong visual generalization. Our system is able to operate in diverse real-world environments, from offices to coffee shops, where the robot is able to reliably manipulate various everyday objects (e.g., mugs, apples, toys) on surfaces ranging from 43cm to 92cm in height. Systematic modular and end-to-end tests in simulation and the real world demonstrate the effectiveness of our proposed design. We believe the advances in this paper can open up new ways of training humanoid robots to interact with daily objects.

  </details>



- **PredMapNet: Future and Historical Reasoning for Consistent Online HD Vectorized Map Construction**  
  Bo Lang, Nirav Savaliya, Zhihao Zheng, Jinglun Feng, Zheng-Hang Yeh, Mooi Choo Chuah  
  _2026-02-18_ · https://arxiv.org/abs/2602.16669v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  High-definition (HD) maps are crucial to autonomous driving, providing structured representations of road elements to support navigation and planning. However, existing query-based methods often employ random query initialization and depend on implicit temporal modeling, which lead to temporal inconsistencies and instabilities during the construction of a global map. To overcome these challenges, we introduce a novel end-to-end framework for consistent online HD vectorized map construction, which jointly performs map instance tracking and short-term prediction. First, we propose a Semantic-Aware Query Generator that initializes queries with spatially aligned semantic masks to capture scene-level context globally. Next, we design a History Rasterized Map Memory to store fine-grained instance-level maps for each tracked instance, enabling explicit historical priors. A History-Map Guidance Module then integrates rasterized map information into track queries, improving temporal continuity. Finally, we propose a Short-Term Future Guidance module to forecast the immediate motion of map instances based on the stored history trajectories. These predicted future locations serve as hints for tracked instances to further avoid implausible predictions and keep temporal consistency. Extensive experiments on the nuScenes and Argoverse2 datasets demonstrate that our proposed method outperforms state-of-the-art (SOTA) methods with good efficiency.

  </details>



- **Docking and Persistent Operations for a Resident Underwater Vehicle**  
  Leonard Günzel, Gabrielė Kasparavičiūtė, Ambjørn Grimsrud Waldum, Bjørn-Magnus Moslått, Abubakar Aliyu Badawi, Celil Yılmaz, Md Shamin Yeasher Yousha, Robert Staven, Martin Ludvigsen  
  _2026-02-18_ · https://arxiv.org/abs/2602.16360v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Our understanding of the oceans remains limited by sparse and infrequent observations, primarily because current methods are constrained by the high cost and logistical effort of underwater monitoring, relying either on sporadic surveys across broad areas or on long-term measurements at fixed locations. To overcome these limitations, monitoring systems must enable persistent and autonomous operations without the need for continuous surface support. Despite recent advances, resident underwater vehicles remain uncommon due to persistent challenges in autonomy, robotic resilience, and mechanical robustness, particularly under long-term deployment in harsh and remote environments. This work addresses these problems by presenting the development, deployment, and operation of a resident infrastructure using a docking station with a mini-class Remotely Operated Vehicle (ROV) at 90m depth. The ROVis equipped with enhanced onboard processing and perception, allowing it to autonomously navigate using USBL signals, dock via ArUco marker-based visual localisation fused through an Extended Kalman Filter, and carry out local inspection routines. The system demonstrated a 90% autonomous docking success rate and completed full inspection missions within four minutes, validating the integration of acoustic and visual navigation in real-world conditions. These results show that reliable, untethered operations at depth are feasible, highlighting the potential of resident ROV systems for scalable, cost-effective underwater monitoring.

  </details>



- **SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks**  
  Zirui Zang, Ahmad Amine, Nick-Marios T. Kokolakis, Truong X. Nghiem, Ugo Rosolia, Rahul Mangharam  
  _2026-02-18_ · https://arxiv.org/abs/2602.16187v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots executing iterative tasks in complex, uncertain environments require control strategies that balance robustness, safety, and high performance. This paper introduces a safe information-theoretic learning model predictive control (SIT-LMPC) algorithm for iterative tasks. Specifically, we design an iterative control framework based on an information-theoretic model predictive control algorithm to address a constrained infinite-horizon optimal control problem for discrete-time nonlinear stochastic systems. An adaptive penalty method is developed to ensure safety while balancing optimality. Trajectories from previous iterations are utilized to learn a value function using normalizing flows, which enables richer uncertainty modeling compared to Gaussian priors. SIT-LMPC is designed for highly parallel execution on graphics processing units, allowing efficient real-time optimization. Benchmark simulations and hardware experiments demonstrate that SIT-LMPC iteratively improves system performance while robustly satisfying system constraints.

  </details>



- **System Identification under Constraints and Disturbance: A Bayesian Estimation Approach**  
  Sergi Martinez, Steve Tonneau, Carlos Mastalli  
  _2026-02-18_ · https://arxiv.org/abs/2602.16358v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We introduce a Bayesian system identification (SysID) framework for jointly estimating robot's state trajectories and physical parameters with high accuracy. It embeds physically consistent inverse dynamics, contact and loop-closure constraints, and fully featured joint friction models as hard, stage-wise equality constraints. It relies on energy-based regressors to enhance parameter observability, supports both equality and inequality priors on inertial and actuation parameters, enforces dynamically consistent disturbance projections, and augments proprioceptive measurements with energy observations to disambiguate nonlinear friction effects. To ensure scalability, we derive a parameterized equality-constrained Riccati recursion that preserves the banded structure of the problem, achieving linear complexity in the time horizon, and develop computationally efficient derivatives. Simulation studies on representative robotic systems, together with hardware experiments on a Unitree B1 equipped with a Z1 arm, demonstrate faster convergence, lower inertial and friction estimation errors, and improved contact consistency compared to forward-dynamics and decoupled identification baselines. When deployed within model predictive control frameworks, the resulting models yield measurable improvements in tracking performance during locomotion over challenging environments.

  </details>


