# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-20 07:10 UTC_

Total papers shown: **4**


---

- **Contact-Anchored Proprioceptive Odometry for Quadruped Robots**  
  Minxing Sun, Yao Mao  
  _2026-02-19_ · https://arxiv.org/abs/2602.17393v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable odometry for legged robots without cameras or LiDAR remains challenging due to IMU drift and noisy joint velocity sensing. This paper presents a purely proprioceptive state estimator that uses only IMU and motor measurements to jointly estimate body pose and velocity, with a unified formulation applicable to biped, quadruped, and wheel-legged robots. The key idea is to treat each contacting leg as a kinematic anchor: joint-torque--based foot wrench estimation selects reliable contacts, and the corresponding footfall positions provide intermittent world-frame constraints that suppress long-term drift. To prevent elevation drift during extended traversal, we introduce a lightweight height clustering and time-decay correction that snaps newly recorded footfall heights to previously observed support planes. To improve foot velocity observations under encoder quantization, we apply an inverse-kinematics cubature Kalman filter that directly filters foot-end velocities from joint angles and velocities. The implementation further mitigates yaw drift through multi-contact geometric consistency and degrades gracefully to a kinematics-derived heading reference when IMU yaw constraints are unavailable or unreliable. We evaluate the method on four quadruped platforms (three Astrall robots and a Unitree Go2 EDU) using closed-loop trajectories. On Astrall point-foot robot~A, a $\sim$200\,m horizontal loop and a $\sim$15\,m vertical loop return with 0.1638\,m and 0.219\,m error, respectively; on wheel-legged robot~B, the corresponding errors are 0.2264\,m and 0.199\,m. On wheel-legged robot~C, a $\sim$700\,m horizontal loop yields 7.68\,m error and a $\sim$20\,m vertical loop yields 0.540\,m error. Unitree Go2 EDU closes a $\sim$120\,m horizontal loop with 2.2138\,m error and a $\sim$8\,m vertical loop with less than 0.1\,m vertical error. github.com/ShineMinxing/Ros2Go2Estimator.git

  </details>



- **Bluetooth Phased-array Aided Inertial Navigation Using Factor Graphs: Experimental Verification**  
  Glen Hjelmerud Mørkbak Sørensen, Torleiv H. Bryne, Kristoffer Gryte, Tor Arne Johansen  
  _2026-02-19_ · https://arxiv.org/abs/2602.17407v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Phased-array Bluetooth systems have emerged as a low-cost alternative for performing aided inertial navigation in GNSS-denied use cases such as warehouse logistics, drone landings, and autonomous docking. Basing a navigation system off of commercial-off-the-shelf components may reduce the barrier of entry for phased-array radio navigation systems, albeit at the cost of significantly noisier measurements and relatively short feasible range. In this paper, we compare robust estimation strategies for a factor graph optimisation-based estimator using experimental data collected from multirotor drone flight. We evaluate performance in loss-of-GNSS scenarios when aided by Bluetooth angular measurements, as well as range or barometric pressure.

  </details>



- **Voice-Driven Semantic Perception for UAV-Assisted Emergency Networks**  
  Nuno Saavedra, Pedro Ribeiro, André Coelho, Rui Campos  
  _2026-02-19_ · https://arxiv.org/abs/2602.17394v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  Unmanned Aerial Vehicle (UAV)-assisted networks are increasingly foreseen as a promising approach for emergency response, providing rapid, flexible, and resilient communications in environments where terrestrial infrastructure is degraded or unavailable. In such scenarios, voice radio communications remain essential for first responders due to their robustness; however, their unstructured nature prevents direct integration with automated UAV-assisted network management. This paper proposes SIREN, an AI-driven framework that enables voice-driven perception for UAV-assisted networks. By integrating Automatic Speech Recognition (ASR) with Large Language Model (LLM)-based semantic extraction and Natural Language Processing (NLP) validation, SIREN converts emergency voice traffic into structured, machine-readable information, including responding units, location references, emergency severity, and Quality-of-Service (QoS) requirements. SIREN is evaluated using synthetic emergency scenarios with controlled variations in language, speaker count, background noise, and message complexity. The results demonstrate robust transcription and reliable semantic extraction across diverse operating conditions, while highlighting speaker diarization and geographic ambiguity as the main limiting factors. These findings establish the feasibility of voice-driven situational awareness for UAV-assisted networks and show a practical foundation for human-in-the-loop decision support and adaptive network management in emergency response operations.

  </details>



- **Graph Neural Model Predictive Control for High-Dimensional Systems**  
  Patrick Benito Eberhard, Luis Pabon, Daniele Gammelli, Hugo Buurmeijer, Amon Lahr, Mark Leone, Andrea Carron, Marco Pavone  
  _2026-02-19_ · https://arxiv.org/abs/2602.17601v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The control of high-dimensional systems, such as soft robots, requires models that faithfully capture complex dynamics while remaining computationally tractable. This work presents a framework that integrates Graph Neural Network (GNN)-based dynamics models with structure-exploiting Model Predictive Control to enable real-time control of high-dimensional systems. By representing the system as a graph with localized interactions, the GNN preserves sparsity, while a tailored condensing algorithm eliminates state variables from the control problem, ensuring efficient computation. The complexity of our condensing algorithm scales linearly with the number of system nodes, and leverages Graphics Processing Unit (GPU) parallelization to achieve real-time performance. The proposed approach is validated in simulation and experimentally on a physical soft robotic trunk. Results show that our method scales to systems with up to 1,000 nodes at 100 Hz in closed-loop, and demonstrates real-time reference tracking on hardware with sub-centimeter accuracy, outperforming baselines by 63.6%. Finally, we show the capability of our method to achieve effective full-body obstacle avoidance.

  </details>


