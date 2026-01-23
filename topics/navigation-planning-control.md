# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-01-23 06:51 UTC_

Total papers shown: **5**


---

- **AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning**  
  Zichen Yan, Yuchen Hou, Shenao Wang, Yichao Gao, Rui Huang, Lin Zhao  
  _2026-01-22_ · https://arxiv.org/abs/2601.15614v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Object-Goal Navigation (ObjectNav) requires an agent to autonomously explore an unknown environment and navigate toward target objects specified by a semantic label. While prior work has primarily studied zero-shot ObjectNav under 2D locomotion, extending it to aerial platforms with 3D locomotion capability remains underexplored. Aerial robots offer superior maneuverability and search efficiency, but they also introduce new challenges in spatial perception, dynamic control, and safety assurance. In this paper, we propose AION for vision-based aerial ObjectNav without relying on external localization or global maps. AION is an end-to-end dual-policy reinforcement learning (RL) framework that decouples exploration and goal-reaching behaviors into two specialized policies. We evaluate AION on the AI2-THOR benchmark and further assess its real-time performance in IsaacSim using high-fidelity drone models. Experimental results show that AION achieves superior performance across comprehensive evaluation metrics in exploration, navigation efficiency, and safety. The video can be found at https://youtu.be/TgsUm6bb7zg.

  </details>



- **A Beacon Based Solution for Autonomous UUVs GNSS-Denied Stealthy Navigation**  
  Alexandre Albore, Humbert Fiorino, Damien Pellier  
  _2026-01-22_ · https://arxiv.org/abs/2601.15802v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous Unmanned Underwater Vehicles (UUVs) enable military and civilian covert operations in coastal areas without relying on support vessels or Global Navigation Satellite Systems (GNSS). Such operations are critical when surface access is not possible and stealthy navigation is required in restricted environments such as protected zones or dangerous areas under access ban. GNSS denied navigation is then essential to maintaining concealment as surfacing could expose UUVs to detection. To ensure a precise fleet positioning a constellation of beacons deployed by aerial or surface drones establish a synthetic landmark network that will guide the fleet of UUVs along an optimized path from the continental shelf to the goal on the shore. These beacons either submerged or floating emit acoustic signals for UUV localisation and navigation. A hierarchical planner generates an adaptive route for the drones executing primitive actions while continuously monitoring and replanning as needed to maintain trajectory accuracy.

  </details>



- **DualShield: Safe Model Predictive Diffusion via Reachability Analysis for Interactive Autonomous Driving**  
  Rui Yang, Lei Zheng, Ruoyu Yao, Jun Ma  
  _2026-01-22_ · https://arxiv.org/abs/2601.15729v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Diffusion models have emerged as a powerful approach for multimodal motion planning in autonomous driving. However, their practical deployment is typically hindered by the inherent difficulty in enforcing vehicle dynamics and a critical reliance on accurate predictions of other agents, making them prone to safety issues under uncertain interactions. To address these limitations, we introduce DualShield, a planning and control framework that leverages Hamilton-Jacobi (HJ) reachability value functions in a dual capacity. First, the value functions act as proactive guidance, steering the diffusion denoising process towards safe and dynamically feasible regions. Second, they form a reactive safety shield using control barrier-value functions (CBVFs) to modify the executed actions and ensure safety. This dual mechanism preserves the rich exploration capabilities of diffusion models while providing principled safety assurance under uncertain and even adversarial interactions. Simulations in challenging unprotected U-turn scenarios demonstrate that DualShield significantly improves both safety and task efficiency compared to leading methods from different planning paradigms under uncertainty.

  </details>



- **Improve the autonomy of the SE2(3) group based Extended Kalman Filter for Integrated Navigation: Application**  
  Jiarui Cui, Maosong Wang, Wenqi Wu, Peiqi Li, Xianfei Pan  
  _2026-01-22_ · https://arxiv.org/abs/2601.16078v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  One of the core advantages of SE2(3) Lie group framework for navigation modeling lies in the autonomy of error propagation. In the previous paper, the theoretical analysis of autonomy property of navigation model in inertial, earth and world frames was given. A construction method for SE2(3) group navigation model is proposed to improve the non-inertial navigation model toward full autonomy. This paper serves as a counterpart to previous paper and conducts the real-world strapdown inertial navigation system (SINS)/odometer(ODO) experiments as well as Monte-Carlo simulations to demonstrate the performance of improved SE2(3) group based high-precision navigation models.

  </details>



- **Improve the autonomy of the SE2(3) group based Extended Kalman Filter for Integrated Navigation: Theoretical Analysis**  
  Jiarui Cui, Maosong Wang, Wenqi Wu, Peiqi Li, Xianfei Pan  
  _2026-01-22_ · https://arxiv.org/abs/2601.16062v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  One of core advantages of the SE2(3) Lie group framework for navigation modeling lies in the autonomy of error propagation. Current research on Lie group based extended Kalman filters has demonstrated that error propagation autonomy holds in low-precision applications, such as in micro electromechanical system (MEMS) based integrated navigation without considering earth rotation and inertial device biases. However, in high-precision navigation state estimation, maintaining autonomy is extremely difficult when considering with earth rotation and inertial device biases. This paper presents the theoretical analysis on the autonomy of SE2(3) group based high-precision navigation models under inertial, earth and world frame respectively. Through theoretical analysis, we find that the limitation of the traditional, trivial SE2(3) group navigation modeling method is that the presence of Coriolis force terms introduced by velocity in non-inertial frame. Therefore, a construction method for SE2(3) group navigation models is proposed, which brings the navigation models closer to full autonomy.

  </details>


