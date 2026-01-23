# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-01-23 06:51 UTC_

Total papers shown: **2**


---

- **AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning**  
  Zichen Yan, Yuchen Hou, Shenao Wang, Yichao Gao, Rui Huang, Lin Zhao  
  _2026-01-22_ · https://arxiv.org/abs/2601.15614v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Object-Goal Navigation (ObjectNav) requires an agent to autonomously explore an unknown environment and navigate toward target objects specified by a semantic label. While prior work has primarily studied zero-shot ObjectNav under 2D locomotion, extending it to aerial platforms with 3D locomotion capability remains underexplored. Aerial robots offer superior maneuverability and search efficiency, but they also introduce new challenges in spatial perception, dynamic control, and safety assurance. In this paper, we propose AION for vision-based aerial ObjectNav without relying on external localization or global maps. AION is an end-to-end dual-policy reinforcement learning (RL) framework that decouples exploration and goal-reaching behaviors into two specialized policies. We evaluate AION on the AI2-THOR benchmark and further assess its real-time performance in IsaacSim using high-fidelity drone models. Experimental results show that AION achieves superior performance across comprehensive evaluation metrics in exploration, navigation efficiency, and safety. The video can be found at https://youtu.be/TgsUm6bb7zg.

  </details>



- **Glove2UAV: A Wearable IMU-Based Glove for Intuitive Control of UAV**  
  Amir Habel, Ivan Snegirev, Elizaveta Semenyakina, Miguel Altamirano Cabrera, Jeffrin Sam, Fawad Mehboob, Roohan Ahmed Khan, Muhammad Ahsan Mustafa, Dzmitry Tsetserukou  
  _2026-01-22_ · https://arxiv.org/abs/2601.15775v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents Glove2UAV, a wearable IMU-glove interface for intuitive UAV control through hand and finger gestures, augmented with vibrotactile warnings for exceeding predefined speed thresholds. To promote safer and more predictable interaction in dynamic flight, Glove2UAV is designed as a lightweight and easily deployable wearable interface intended for real-time operation. Glove2UAV streams inertial measurements in real time and estimates palm and finger orientations using a compact processing pipeline that combines median-based outlier suppression with Madgwick-based orientation estimation. The resulting motion estimations are mapped to a small set of control primitives for directional flight (forward/backward and lateral motion) and, when supported by the platform, to object-interaction commands. Vibrotactile feedback is triggered when flight speed exceeds predefined threshold values, providing an additional alert channel during operation. We validate real-time feasibility by synchronizing glove signals with UAV telemetry in both simulation and real-world flights. The results show fast gesture-based command execution, stable coupling between gesture dynamics and platform motion, correct operation of the core command set in our trials, and timely delivery of vibratile warning cues.

  </details>


