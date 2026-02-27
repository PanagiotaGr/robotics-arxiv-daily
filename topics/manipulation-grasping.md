# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-27 07:08 UTC_

Total papers shown: **7**


---

- **A Perspective on Open Challenges in Deformable Object Manipulation**  
  Ryan Paul McKennaa, John Oyekan  
  _2026-02-26_ · https://arxiv.org/abs/2602.22998v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deformable object manipulation (DOM) represents a critical challenge in robotics, with applications spanning healthcare, manufacturing, food processing, and beyond. Unlike rigid objects, deformable objects exhibit infinite dimensionality, dynamic shape changes, and complex interactions with their environment, posing significant hurdles for perception, modeling, and control. This paper reviews the state of the art in DOM, focusing on key challenges such as occlusion handling, task generalization, and scalable, real-time solutions. It highlights advancements in multimodal perception systems, including the integration of multi-camera setups, active vision, and tactile sensing, which collectively address occlusion and improve adaptability in unstructured environments. Cutting-edge developments in physically informed reinforcement learning (RL) and differentiable simulations are explored, showcasing their impact on efficiency, precision, and scalability. The review also emphasizes the potential of simulated expert demonstrations and generative neural networks to standardize task specifications and bridge the simulation-to-reality gap. Finally, future directions are proposed, including the adoption of graph neural networks for high-level decision-making and the creation of comprehensive datasets to enhance DOM's real-world applicability. By addressing these challenges, DOM research can pave the way for versatile robotic systems capable of handling diverse and dynamic tasks with deformable objects.

  </details>



- **Grasp, Slide, Roll: Comparative Analysis of Contact Modes for Tactile-Based Shape Reconstruction**  
  Chung Hee Kim, Shivani Kamtikar, Tye Brady, Taskin Padir, Joshua Migdal  
  _2026-02-26_ · https://arxiv.org/abs/2602.23206v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Tactile sensing allows robots to gather detailed geometric information about objects through physical interaction, complementing vision-based approaches. However, efficiently acquiring useful tactile data remains challenging due to the time-consuming nature of physical contact and the need to strategically choose contact locations that maximize information gain while minimizing physical interactions. This paper studies how different contact modes affect object shape reconstruction using a tactile-enabled dexterous gripper. We compare three contact interaction modes: grasp-releasing, sliding induced by finger-grazing, and palm-rolling. These contact modes are combined with an information-theoretic exploration framework that guides subsequent sampling locations using a shape completion model. Our results show that the improved tactile sensing efficiency of finger-grazing and palm-rolling translates into faster convergence in shape reconstruction, requiring 34% fewer physical interactions while improving reconstruction accuracy by 55%. We validate our approach using a UR5e robot arm equipped with an Inspire-Robots Dexterous Hand, showing robust performance across primitive object geometries.

  </details>



- **DigiArm: An Anthropomorphic 3D-Printed Prosthetic Hand with Enhanced Dexterity for Typing Tasks**  
  Dean Zadok, Tom Naamani, Yuval Bar-Ratson, Elisha Barash, Oren Salzman, Alon Wolf, Alex M. Bronstein, Nili Krausz  
  _2026-02-26_ · https://arxiv.org/abs/2602.23017v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Despite recent advancements, existing prosthetic limbs are unable to replicate the dexterity and intuitive control of the human hand. Current control systems for prosthetic hands are often limited to grasping, and commercial prosthetic hands lack the precision needed for dexterous manipulation or applications that require fine finger motions. Thus, there is a critical need for accessible and replicable prosthetic designs that enable individuals to interact with electronic devices and perform precise finger pressing, such as keyboard typing or piano playing, while preserving current prosthetic capabilities. This paper presents a low-cost, lightweight, 3D-printed robotic prosthetic hand, specifically engineered for enhanced dexterity with electronic devices such as a computer keyboard or piano, as well as general object manipulation. The robotic hand features a mechanism to adjust finger abduction/adduction spacing, a 2-D wrist with the inclusion of controlled ulnar/radial deviation optimized for typing, and control of independent finger pressing. We conducted a study to demonstrate how participants can use the robotic hand to perform keyboard typing and piano playing in real time, with different levels of finger and wrist motion. This supports the notion that our proposed design can allow for the execution of key typing motions more effectively than before, aiming to enhance the functionality of prosthetic hands.

  </details>



- **GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion**  
  Enda Xiang, Haoxiang Ma, Xinzhu Ma, Zicheng Liu, Di Huang  
  _2026-02-26_ · https://arxiv.org/abs/2602.22862v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper focuses on enhancing the grasping precision and generalization of manipulation policies learned via imitation learning. Diffusion-based policy learning methods have recently become the mainstream approach for robotic manipulation tasks. As grasping is a critical subtask in manipulation, the ability of imitation-learned policies to execute precise and generalizable grasps merits particular attention. Existing imitation learning techniques for grasping often suffer from imprecise grasp executions, limited spatial generalization, and poor object generalization. To address these challenges, we incorporate grasp prior knowledge into the diffusion policy framework. In particular, we employ a latent diffusion policy to guide action chunk decoding with grasp pose prior, ensuring that generated motion trajectories adhere closely to feasible grasp configurations. Furthermore, we introduce a self-supervised reconstruction objective during diffusion to embed the graspness prior: at each reverse diffusion step, we reconstruct wrist-camera images back-projected the graspness from the intermediate representations. Both simulation and real robot experiments demonstrate that our approach significantly outperforms baseline methods and exhibits strong dynamic grasping capabilities.

  </details>



- **Physics Informed Viscous Value Representations**  
  Hrishikesh Viswanath, Juanwu Lu, S. Talha Bukhari, Damon Conover, Ziran Wang, Aniket Bera  
  _2026-02-26_ · https://arxiv.org/abs/2602.23280v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Offline goal-conditioned reinforcement learning (GCRL) learns goal-conditioned policies from static pre-collected datasets. However, accurate value estimation remains a challenge due to the limited coverage of the state-action space. Recent physics-informed approaches have sought to address this by imposing physical and geometric constraints on the value function through regularization defined over first-order partial differential equations (PDEs), such as the Eikonal equation. However, these formulations can often be ill-posed in complex, high-dimensional environments. In this work, we propose a physics-informed regularization derived from the viscosity solution of the Hamilton-Jacobi-Bellman (HJB) equation. By providing a physics-based inductive bias, our approach grounds the learning process in optimal control theory, explicitly regularizing and bounding updates during value iterations. Furthermore, we leverage the Feynman-Kac theorem to recast the PDE solution as an expectation, enabling a tractable Monte Carlo estimation of the objective that avoids numerical instability in higher-order gradients. Experiments demonstrate that our method improves geometric consistency, making it broadly applicable to navigation and high-dimensional, complex manipulation tasks. Open-source codes are available at https://github.com/HrishikeshVish/phys-fk-value-GCRL.

  </details>



- **InCoM: Intent-Driven Perception and Structured Coordination for Whole-Body Mobile Manipulation**  
  Jiahao Liu, Cui Wenbo, Haoran Li, Dongbin Zhao  
  _2026-02-26_ · https://arxiv.org/abs/2602.23024v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Whole-body mobile manipulation is a fundamental capability for general-purpose robotic agents, requiring both coordinated control of the mobile base and manipulator and robust perception under dynamically changing viewpoints. However, existing approaches face two key challenges: strong coupling between base and arm actions complicates whole-body control optimization, and perceptual attention is often poorly allocated as viewpoints shift during mobile manipulation. We propose InCoM, an intent-driven perception and structured coordination framework for whole-body mobile manipulation. InCoM infers latent motion intent to dynamically reweight multi-scale perceptual features, enabling stage-adaptive allocation of perceptual attention. To support robust cross-modal perception, InCoM further incorporates a geometric-semantic structured alignment mechanism that enhances multimodal correspondence. On the control side, we design a decoupled coordinated flow matching action decoder that explicitly models coordinated base-arm action generation, alleviating optimization difficulties caused by control coupling. Without access to privileged perceptual information, InCoM outperforms state-of-the-art methods on three ManiSkill-HAB scenarios by 28.2%, 26.1%, and 23.6% in success rate, demonstrating strong effectiveness for whole-body mobile manipulation.

  </details>



- **DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation**  
  Zebin Yang, Yijiahao Qi, Tong Xie, Bo Yu, Shaoshan Liu, Meng Li  
  _2026-02-26_ · https://arxiv.org/abs/2602.22896v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have shown remarkable success in robotic tasks like manipulation by fusing a language model's reasoning with a vision model's 3D understanding. However, their high computational cost remains a major obstacle for real-world applications that require real-time performance. We observe that the actions within a task have varying levels of importance: critical steps demand high precision, while less important ones can tolerate more variance. Leveraging this insight, we propose DySL-VLA, a novel framework that addresses computational cost by dynamically skipping VLA layers based on each action's importance. DySL-VLA categorizes its layers into two types: informative layers, which are consistently executed, and incremental layers, which can be selectively skipped. To intelligently skip layers without sacrificing accuracy, we invent a prior-post skipping guidance mechanism to determine when to initiate layer-skipping. We also propose a skip-aware two-stage knowledge distillation algorithm to efficiently train a standard VLA into a DySL-VLA. Our experiments indicate that DySL-VLA achieves 2.1% improvement in success length over Deer-VLA on the Calvin dataset, while simultaneously reducing trainable parameters by a factor of 85.7 and providing a 3.75x speedup relative to the RoboFlamingo baseline at iso-accuracy. Our code is available on https://github.com/PKU-SEC-Lab/DYSL_VLA.

  </details>


