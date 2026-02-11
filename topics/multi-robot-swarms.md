# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-11 07:16 UTC_

Total papers shown: **3**


---

- **TriPilot-FF: Coordinated Whole-Body Teleoperation with Force Feedback**  
  Zihao Li, Yanan Zhou, Ranpeng Qiu, Hangyu Wu, Guoqiang Ren, Weiming Zhi  
  _2026-02-10_ · https://arxiv.org/abs/2602.09888v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mobile manipulators broaden the operational envelope for robot manipulation. However, the whole-body teleoperation of such robots remains a problem: operators must coordinate a wheeled base and two arms while reasoning about obstacles and contact. Existing interfaces are predominantly hand-centric (e.g., VR controllers and joysticks), leaving foot-operated channels underexplored for continuous base control. We present TriPilot-FF, an open-source whole-body teleoperation system for a custom bimanual mobile manipulator that introduces a foot-operated pedal with lidar-driven pedal haptics, coupled with upper-body bimanual leader-follower teleoperation. Using only a low-cost base-mounted lidar, TriPilot-FF renders a resistive pedal cue from proximity-to-obstacle signals in the commanded direction, shaping operator commands toward collision-averse behaviour without an explicit collision-avoidance controller. The system also supports arm-side force reflection for contact awareness and provides real-time force and visual guidance of bimanual manipulability to prompt mobile base repositioning, thereby improving reach. We demonstrate the capability of TriPilot-FF to effectively ``co-pilot'' the human operator over long time-horizons and tasks requiring precise mobile base movement and coordination. Finally, we incorporate teleoperation feedback signals into an Action Chunking with Transformers (ACT) policy and demonstrate improved performance when the additional information is available. We release the pedal device design, full software stack, and conduct extensive real-world evaluations on a bimanual wheeled platform. The project page of TriPilot-FF is http://bit.ly/46H3ZJT.

  </details>



- **RANT: Ant-Inspired Multi-Robot Rainforest Exploration Using Particle Filter Localisation and Virtual Pheromone Coordination**  
  Ameer Alhashemi, Layan Abdulhadi, Karam Abuodeh, Tala Baghdadi, Suryanarayana Datla  
  _2026-02-10_ · https://arxiv.org/abs/2602.09661v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents RANT, an ant-inspired multi-robot exploration framework for noisy, uncertain environments. A team of differential-drive robots navigates a 10 x 10 m terrain, collects noisy probe measurements of a hidden richness field, and builds local probabilistic maps while the supervisor maintains a global evaluation. RANT combines particle-filter localisation, a behaviour-based controller with gradient-driven hotspot exploitation, and a lightweight no-revisit coordination mechanism based on virtual pheromone blocking. We experimentally analyse how team size, localisation fidelity, and coordination influence coverage, hotspot recall, and redundancy. Results show that particle filtering is essential for reliable hotspot engagement, coordination substantially reduces overlap, and increasing team size improves coverage but yields diminishing returns due to interference.

  </details>



- **Decoupled MPPI-Based Multi-Arm Motion Planning**  
  Dan Evron, Elias Goldsztejn, Ronen I. Brafman  
  _2026-02-10_ · https://arxiv.org/abs/2602.10114v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advances in sampling-based motion planning algorithms for high DOF arms leverage GPUs to provide SOTA performance. These algorithms can be used to control multiple arms jointly, but this approach scales poorly. To address this, we extend STORM, a sampling-based model-predictive-control (MPC) motion planning algorithm, to handle multiple robots in a distributed fashion. First, we modify STORM to handle dynamic obstacles. Then, we let each arm compute its own motion plan prefix, which it shares with the other arms, which treat it as a dynamic obstacle. Finally, we add a dynamic priority scheme. The new algorithm, MR-STORM, demonstrates clear empirical advantages over SOTA algorithms when operating with both static and dynamic obstacles.

  </details>


