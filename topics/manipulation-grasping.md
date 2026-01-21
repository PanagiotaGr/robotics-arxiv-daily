# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-01-21 06:53 UTC_

Total papers shown: **4**


---

- **DroneVLA: VLA based Aerial Manipulation**  
  Fawad Mehboob, Monijesu James, Amir Habel, Jeffrin Sam, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13809v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  As aerial platforms evolve from passive observers to active manipulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally. This work introduces a novel concept of autonomous aerial manipulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user. The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom-built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera. VLA performs semantic reasoning to interpret the intent of a user prompt and generates a prioritized task queue for grasping of relevant objects in the scene. Grounding DINO and dynamic A* planning algorithm are used to navigate and safely relocate the object. To ensure safe and natural interaction during the handover phase, the system employs a human-centric controller driven by MediaPipe. This module provides real-time human pose estimation, allowing the drone to employ visual servoing to maintain a stable, distinct position directly in front of the user, facilitating a comfortable handover. We demonstrate the system's efficacy through real-world experiments for localization and navigation, which resulted in a 0.164m, 0.070m, and 0.084m of max, mean euclidean, and root-mean squared errors, respectively, highlighting the feasibility of VLA for aerial manipulation operations.

  </details>



- **SandWorm: Event-based Visuotactile Perception with Active Vibration for Screw-Actuated Robot in Granular Media**  
  Shoujie Li, Changqing Guo, Junhao Gong, Chenxin Liang, Wenhua Ding, Wenbo Ding  
  _2026-01-20_ · https://arxiv.org/abs/2601.14128v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Perception in granular media remains challenging due to unpredictable particle dynamics. To address this challenge, we present SandWorm, a biomimetic screw-actuated robot augmented by peristaltic motion to enhance locomotion, and SWTac, a novel event-based visuotactile sensor with an actively vibrated elastomer. The event camera is mechanically decoupled from vibrations by a spring isolation mechanism, enabling high-quality tactile imaging of both dynamic and stationary objects. For algorithm design, we propose an IMU-guided temporal filter to enhance imaging consistency, improving MSNR by 24%. Moreover, we systematically optimize SWTac with vibration parameters, event camera settings and elastomer properties. Motivated by asymmetric edge features, we also implement contact surface estimation by U-Net. Experimental validation demonstrates SWTac's 0.2 mm texture resolution, 98% stone classification accuracy, and 0.15 N force estimation error, while SandWorm demonstrates versatile locomotion (up to 12.5 mm/s) in challenging terrains, successfully executes pipeline dredging and subsurface exploration in complex granular media (observed 90% success rate). Field experiments further confirm the system's practical performance.

  </details>



- **Active Cross-Modal Visuo-Tactile Perception of Deformable Linear Objects**  
  Raffaele Mazza, Ciro Natale, Pietro Falco  
  _2026-01-20_ · https://arxiv.org/abs/2601.13979v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a novel cross-modal visuo-tactile perception framework for the 3D shape reconstruction of deformable linear objects (DLOs), with a specific focus on cables subject to severe visual occlusions. Unlike existing methods relying predominantly on vision, whose performance degrades under varying illumination, background clutter, or partial visibility, the proposed approach integrates foundation-model-based visual perception with adaptive tactile exploration. The visual pipeline exploits SAM for instance segmentation and Florence for semantic refinement, followed by skeletonization, endpoint detection, and point-cloud extraction. Occluded cable segments are autonomously identified and explored with a tactile sensor, which provides local point clouds that are merged with the visual data through Euclidean clustering and topology-preserving fusion. A B-spline interpolation driven by endpoint-guided point sorting yields a smooth and complete reconstruction of the cable shape. Experimental validation using a robotic manipulator equipped with an RGB-D camera and a tactile pad demonstrates that the proposed framework accurately reconstructs both simple and highly curved single or multiple cable configurations, even when large portions are occluded. These results highlight the potential of foundation-model-enhanced cross-modal perception for advancing robotic manipulation of deformable objects.

  </details>



- **TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers**  
  Bin Yu, Shijie Lian, Xiaopeng Lin, Yuliang Wei, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Xinming Wang, Bailing Wang, Cong Huang, et al.  
  _2026-01-20_ · https://arxiv.org/abs/2601.14133v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Standard Vision-Language-Action (VLA) models typically fine-tune a monolithic Vision-Language Model (VLM) backbone explicitly for robotic control. However, this approach creates a critical tension between maintaining high-level general semantic understanding and learning low-level, fine-grained sensorimotor skills, often leading to "catastrophic forgetting" of the model's open-world capabilities. To resolve this conflict, we introduce TwinBrainVLA, a novel architecture that coordinates a generalist VLM retaining universal semantic understanding and a specialist VLM dedicated to embodied proprioception for joint robotic control. TwinBrainVLA synergizes a frozen "Left Brain", which retains robust general visual reasoning, with a trainable "Right Brain", specialized for embodied perception, via a novel Asymmetric Mixture-of-Transformers (AsyMoT) mechanism. This design allows the Right Brain to dynamically query semantic knowledge from the frozen Left Brain and fuse it with proprioceptive states, providing rich conditioning for a Flow-Matching Action Expert to generate precise continuous controls. Extensive experiments on SimplerEnv and RoboCasa benchmarks demonstrate that TwinBrainVLA achieves superior manipulation performance compared to state-of-the-art baselines while explicitly preserving the comprehensive visual understanding capabilities of the pre-trained VLM, offering a promising direction for building general-purpose robots that simultaneously achieve high-level semantic understanding and low-level physical dexterity.

  </details>


