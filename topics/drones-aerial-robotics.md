# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-01-21 06:53 UTC_

Total papers shown: **7**


---

- **Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep Reinforcement Learning**  
  Myong-Yol Choi, Hankyoul Ko, Hanse Cho, Changseung Kim, Seunghwan Kim, Jaemin Seo, Hyondong Oh  
  _2026-01-20_ · https://arxiv.org/abs/2601.13657v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a deep reinforcement learning (DRL) based controller for collective navigation of unmanned aerial vehicle (UAV) swarms in communication-denied environments, enabling robust operation in complex, obstacle-rich environments. Inspired by biological swarms where informed individuals guide groups without explicit communication, we employ an implicit leader-follower framework. In this paradigm, only the leader possesses goal information, while follower UAVs learn robust policies using only onboard LiDAR sensing, without requiring any inter-agent communication or leader identification. Our system utilizes LiDAR point clustering and an extended Kalman filter for stable neighbor tracking, providing reliable perception independent of external positioning systems. The core of our approach is a DRL controller, trained in GPU-accelerated Nvidia Isaac Sim, that enables followers to learn complex emergent behaviors - balancing flocking and obstacle avoidance - using only local perception. This allows the swarm to implicitly follow the leader while robustly addressing perceptual challenges such as occlusion and limited field-of-view. The robustness and sim-to-real transfer of our approach are confirmed through extensive simulations and challenging real-world experiments with a swarm of five UAVs, which successfully demonstrated collective navigation across diverse indoor and outdoor environments without any communication or external localization.

  </details>



- **DroneVLA: VLA based Aerial Manipulation**  
  Fawad Mehboob, Monijesu James, Amir Habel, Jeffrin Sam, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13809v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  As aerial platforms evolve from passive observers to active manipulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally. This work introduces a novel concept of autonomous aerial manipulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user. The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom-built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera. VLA performs semantic reasoning to interpret the intent of a user prompt and generates a prioritized task queue for grasping of relevant objects in the scene. Grounding DINO and dynamic A* planning algorithm are used to navigate and safely relocate the object. To ensure safe and natural interaction during the handover phase, the system employs a human-centric controller driven by MediaPipe. This module provides real-time human pose estimation, allowing the drone to employ visual servoing to maintain a stable, distinct position directly in front of the user, facilitating a comfortable handover. We demonstrate the system's efficacy through real-world experiments for localization and navigation, which resulted in a 0.164m, 0.070m, and 0.084m of max, mean euclidean, and root-mean squared errors, respectively, highlighting the feasibility of VLA for aerial manipulation operations.

  </details>



- **HoverAI: An Embodied Aerial Agent for Natural Human-Drone Interaction**  
  Yuhua Jin, Nikita Kuzmin, Georgii Demianchuk, Mariya Lezina, Fawad Mehboob, Issatay Tokmurziyev, Miguel Altamirano Cabrera, Muhammad Ahsan Mustafa, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13801v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Drones operating in human-occupied spaces suffer from insufficient communication mechanisms that create uncertainty about their intentions. We present HoverAI, an embodied aerial agent that integrates drone mobility, infrastructure-independent visual projection, and real-time conversational AI into a unified platform. Equipped with a MEMS laser projector, onboard semi-rigid screen, and RGB camera, HoverAI perceives users through vision and voice, responding via lip-synced avatars that adapt appearance to user demographics. The system employs a multimodal pipeline combining VAD, ASR (Whisper), LLM-based intent classification, RAG for dialogue, face analysis for personalization, and voice synthesis (XTTS v2). Evaluation demonstrates high accuracy in command recognition (F1: 0.90), demographic estimation (gender F1: 0.89, age MAE: 5.14 years), and speech transcription (WER: 0.181). By uniting aerial robotics with adaptive conversational AI and self-contained visual output, HoverAI introduces a new class of spatially-aware, socially responsive embodied agents for applications in guidance, assistance, and human-centered interaction.

  </details>



- **SandWorm: Event-based Visuotactile Perception with Active Vibration for Screw-Actuated Robot in Granular Media**  
  Shoujie Li, Changqing Guo, Junhao Gong, Chenxin Liang, Wenhua Ding, Wenbo Ding  
  _2026-01-20_ · https://arxiv.org/abs/2601.14128v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Perception in granular media remains challenging due to unpredictable particle dynamics. To address this challenge, we present SandWorm, a biomimetic screw-actuated robot augmented by peristaltic motion to enhance locomotion, and SWTac, a novel event-based visuotactile sensor with an actively vibrated elastomer. The event camera is mechanically decoupled from vibrations by a spring isolation mechanism, enabling high-quality tactile imaging of both dynamic and stationary objects. For algorithm design, we propose an IMU-guided temporal filter to enhance imaging consistency, improving MSNR by 24%. Moreover, we systematically optimize SWTac with vibration parameters, event camera settings and elastomer properties. Motivated by asymmetric edge features, we also implement contact surface estimation by U-Net. Experimental validation demonstrates SWTac's 0.2 mm texture resolution, 98% stone classification accuracy, and 0.15 N force estimation error, while SandWorm demonstrates versatile locomotion (up to 12.5 mm/s) in challenging terrains, successfully executes pipeline dredging and subsurface exploration in complex granular media (observed 90% success rate). Field experiments further confirm the system's practical performance.

  </details>



- **GuideTouch: An Obstacle Avoidance Device for Visually Impaired**  
  Timofei Kozlov, Artem Trandofilov, Georgii Gazaryan, Issatay Tokmurziyev, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13813v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Safe navigation for the visually impaired individuals remains a critical challenge, especially concerning head-level obstacles, which traditional mobility aids often fail to detect. We introduce GuideTouch, a compact, affordable, standalone wearable device designed for autonomous obstacle avoidance. The system integrates two vertically aligned Time-of-Flight (ToF) sensors, enabling three-dimensional environmental perception, and four vibrotactile actuators that provide directional haptic feedback. Proximity and direction information is communicated via an intuitive 4-point vibrotactile feedback system located across the user's shoulders and upper chest. For real-world robustness, the device includes a unique centrifugal self-cleaning optical cover mechanism and a sound alarm system for location if the device is dropped. We evaluated the haptic perception accuracy across 22 participants (17 male and 5 female, aged 21-48, mean 25.7, sd 6.1). Statistical analysis confirmed a significant difference between the perception accuracy of different patterns. The system demonstrated high recognition accuracy, achieving an average of 92.9% for single and double motor (primary directional) patterns. Furthermore, preliminary experiments with 14 visually impaired users validated this interface, showing a recognition accuracy of 93.75% for primary directional cues. The results demonstrate that GuideTouch enables intuitive spatial perception and could significantly improve the safety, confidence, and autonomy of users with visual impairments during independent navigation.

  </details>



- **MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network**  
  Yiwei Lu, Hao Huang, Tao Yan  
  _2026-01-20_ · https://arxiv.org/abs/2601.13715v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Glass surface ubiquitous in both daily life and professional environments presents a potential threat to vision-based systems, such as robot and drone navigation. To solve this challenge, most recent studies have shown significant interest in Video Glass Surface Detection (VGSD). We observe that objects in the reflection (or transmission) layer appear farther from the glass surfaces. Consequently, in video motion scenarios, the notable reflected (or transmitted) objects on the glass surface move slower than objects in non-glass regions within the same spatial plane, and this motion inconsistency can effectively reveal the presence of glass surfaces. Based on this observation, we propose a novel network, named MVGD-Net, for detecting glass surfaces in videos by leveraging motion inconsistency cues. Our MVGD-Net features three novel modules: the Cross-scale Multimodal Fusion Module (CMFM) that integrates extracted spatial features and estimated optical flow maps, the History Guided Attention Module (HGAM) and Temporal Cross Attention Module (TCAM), both of which further enhances temporal features. A Temporal-Spatial Decoder (TSD) is also introduced to fuse the spatial and temporal features for generating the glass region mask. Furthermore, for learning our network, we also propose a large-scale dataset, which comprises 312 diverse glass scenarios with a total of 19,268 frames. Extensive experiments demonstrate that our MVGD-Net outperforms relevant state-of-the-art methods.

  </details>



- **Optimizing Energy and Data Collection in UAV-aided IoT Networks using Attention-based Multi-Objective Reinforcement Learning**  
  Babacar Toure, Dimitrios Tsilimantos, Omid Esrafilian, Marios Kountouris  
  _2026-01-20_ · https://arxiv.org/abs/2601.14092v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Due to their adaptability and mobility, Unmanned Aerial Vehicles (UAVs) are becoming increasingly essential for wireless network services, particularly for data harvesting tasks. In this context, Artificial Intelligence (AI)-based approaches have gained significant attention for addressing UAV path planning tasks in large and complex environments, bridging the gap with real-world deployments. However, many existing algorithms suffer from limited training data, which hampers their performance in highly dynamic environments. Moreover, they often overlook the inherently multi-objective nature of the task, treating it in an overly simplistic manner. To address these limitations, we propose an attention-based Multi-Objective Reinforcement Learning (MORL) architecture that explicitly handles the trade-off between data collection and energy consumption in urban environments, even without prior knowledge of wireless channel conditions. Our method develops a single model capable of adapting to varying trade-off preferences and dynamic scenario parameters without the need for fine-tuning or retraining. Extensive simulations show that our approach achieves substantial improvements in performance, model compactness, sample efficiency, and most importantly, generalization to previously unseen scenarios, outperforming existing RL solutions.

  </details>


