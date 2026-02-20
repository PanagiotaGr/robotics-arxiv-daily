# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-20 07:10 UTC_

Total papers shown: **5**


---

- **Benchmarking the Effects of Object Pose Estimation and Reconstruction on Robotic Grasping Success**  
  Varun Burde, Pavel Burget, Torsten Sattler  
  _2026-02-19_ · https://arxiv.org/abs/2602.17101v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  3D reconstruction serves as the foundational layer for numerous robotic perception tasks, including 6D object pose estimation and grasp pose generation. Modern 3D reconstruction methods for objects can produce visually and geometrically impressive meshes from multi-view images, yet standard geometric evaluations do not reflect how reconstruction quality influences downstream tasks such as robotic manipulation performance. This paper addresses this gap by introducing a large-scale, physics-based benchmark that evaluates 6D pose estimators and 3D mesh models based on their functional efficacy in grasping. We analyze the impact of model fidelity by generating grasps on various reconstructed 3D meshes and executing them on the ground-truth model, simulating how grasp poses generated with an imperfect model affect interaction with the real object. This assesses the combined impact of pose error, grasp robustness, and geometric inaccuracies from 3D reconstruction. Our results show that reconstruction artifacts significantly decrease the number of grasp pose candidates but have a negligible effect on grasping performance given an accurately estimated pose. Our results also reveal that the relationship between grasp success and pose error is dominated by spatial error, and even a simple translation error provides insight into the success of the grasping pose of symmetric objects. This work provides insight into how perception systems relate to object manipulation using robots.

  </details>



- **Physical Human-Robot Interaction for Grasping in Augmented Reality via Rigid-Soft Robot Synergy**  
  Huishi Huang, Jack Klusmann, Haozhe Wang, Shuchen Ji, Fengkang Ying, Yiyuan Zhang, John Nassour, Gordon Cheng, Daniela Rus, Jun Liu, et al.  
  _2026-02-19_ · https://arxiv.org/abs/2602.17128v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Hybrid rigid-soft robots combine the precision of rigid manipulators with the compliance and adaptability of soft arms, offering a promising approach for versatile grasping in unstructured environments. However, coordinating hybrid robots remains challenging, due to difficulties in modeling, perception, and cross-domain kinematics. In this work, we present a novel augmented reality (AR)-based physical human-robot interaction framework that enables direct teleoperation of a hybrid rigid-soft robot for simple reaching and grasping tasks. Using an AR headset, users can interact with a simulated model of the robotic system integrated into a general-purpose physics engine, which is superimposed on the real system, allowing simulated execution prior to real-world deployment. To ensure consistent behavior between the virtual and physical robots, we introduce a real-to-simulation parameter identification pipeline that leverages the inherent geometric properties of the soft robot, enabling accurate modeling of its static and dynamic behavior as well as the control system's response.

  </details>



- **Attachment Anchors: A Novel Framework for Laparoscopic Grasping Point Prediction in Colorectal Surgery**  
  Dennis N. Schneider, Lars Wagner, Daniel Rueckert, Dirk Wilhelm  
  _2026-02-19_ · https://arxiv.org/abs/2602.17310v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate grasping point prediction is a key challenge for autonomous tissue manipulation in minimally invasive surgery, particularly in complex and variable procedures such as colorectal interventions. Due to their complexity and prolonged duration, colorectal procedures have been underrepresented in current research. At the same time, they pose a particularly interesting learning environment due to repetitive tissue manipulation, making them a promising entry point for autonomous, machine learning-driven support. Therefore, in this work, we introduce attachment anchors, a structured representation that encodes the local geometric and mechanical relationships between tissue and its anatomical attachments in colorectal surgery. This representation reduces uncertainty in grasping point prediction by normalizing surgical scenes into a consistent local reference frame. We demonstrate that attachment anchors can be predicted from laparoscopic images and incorporated into a grasping framework based on machine learning. Experiments on a dataset of 90 colorectal surgeries demonstrate that attachment anchors improve grasping point prediction compared to image-only baselines. There are particularly strong gains in out-of-distribution settings, including unseen procedures and operating surgeons. These results suggest that attachment anchors are an effective intermediate representation for learning-based tissue manipulation in colorectal surgery.

  </details>



- **Nonlinear Predictive Control of the Continuum and Hybrid Dynamics of a Suspended Deformable Cable for Aerial Pick and Place**  
  Antonio Rapuano, Yaolei Shen, Federico Califano, Chiara Gabellieri, Antonio Franchi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17199v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a framework for aerial manipulation of an extensible cable that combines a high-fidelity model based on partial differential equations (PDEs) with a reduced-order representation suitable for real-time control. The PDEs are discretised using a finite-difference method, and proper orthogonal decomposition is employed to extract a reduced-order model (ROM) that retains the dominant deformation modes while significantly reducing computational complexity. Based on this ROM, a nonlinear model predictive control scheme is formulated, capable of stabilizing cable oscillations and handling hybrid transitions such as payload attachment and detachment. Simulation results confirm the stability, efficiency, and robustness of the ROM, as well as the effectiveness of the controller in regulating cable dynamics under a range of operating conditions. Additional simulations illustrate the application of the ROM for trajectory planning in constrained environments, demonstrating the versatility of the proposed approach. Overall, the framework enables real-time, dynamics-aware control of unmanned aerial vehicles (UAVs) carrying suspended flexible cables.

  </details>



- **Grasp Synthesis Matching From Rigid To Soft Robot Grippers Using Conditional Flow Matching**  
  Tanisha Parulekar, Ge Shi, Josh Pinskier, David Howard, Jen Jen Chung  
  _2026-02-19_ · https://arxiv.org/abs/2602.17110v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  A representation gap exists between grasp synthesis for rigid and soft grippers. Anygrasp [1] and many other grasp synthesis methods are designed for rigid parallel grippers, and adapting them to soft grippers often fails to capture their unique compliant behaviors, resulting in data-intensive and inaccurate models. To bridge this gap, this paper proposes a novel framework to map grasp poses from a rigid gripper model to a soft Fin-ray gripper. We utilize Conditional Flow Matching (CFM), a generative model, to learn this complex transformation. Our methodology includes a data collection pipeline to generate paired rigid-soft grasp poses. A U-Net autoencoder conditions the CFM model on the object's geometry from a depth image, allowing it to learn a continuous mapping from an initial Anygrasp pose to a stable Fin-ray gripper pose. We validate our approach on a 7-DOF robot, demonstrating that our CFM-generated poses achieve a higher overall success rate for seen and unseen objects (34% and 46% respectively) compared to the baseline rigid poses (6% and 25% respectively) when executed by the soft gripper. The model shows significant improvements, particularly for cylindrical (50% and 100% success for seen and unseen objects) and spherical objects (25% and 31% success for seen and unseen objects), and successfully generalizes to unseen objects. This work presents CFM as a data-efficient and effective method for transferring grasp strategies, offering a scalable methodology for other soft robotic systems.

  </details>


