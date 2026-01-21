# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-01-21 06:53 UTC_

Total papers shown: **10**


---

- **DroneVLA: VLA based Aerial Manipulation**  
  Fawad Mehboob, Monijesu James, Amir Habel, Jeffrin Sam, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13809v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  As aerial platforms evolve from passive observers to active manipulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally. This work introduces a novel concept of autonomous aerial manipulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user. The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom-built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera. VLA performs semantic reasoning to interpret the intent of a user prompt and generates a prioritized task queue for grasping of relevant objects in the scene. Grounding DINO and dynamic A* planning algorithm are used to navigate and safely relocate the object. To ensure safe and natural interaction during the handover phase, the system employs a human-centric controller driven by MediaPipe. This module provides real-time human pose estimation, allowing the drone to employ visual servoing to maintain a stable, distinct position directly in front of the user, facilitating a comfortable handover. We demonstrate the system's efficacy through real-world experiments for localization and navigation, which resulted in a 0.164m, 0.070m, and 0.084m of max, mean euclidean, and root-mean squared errors, respectively, highlighting the feasibility of VLA for aerial manipulation operations.

  </details>



- **Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep Reinforcement Learning**  
  Myong-Yol Choi, Hankyoul Ko, Hanse Cho, Changseung Kim, Seunghwan Kim, Jaemin Seo, Hyondong Oh  
  _2026-01-20_ · https://arxiv.org/abs/2601.13657v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a deep reinforcement learning (DRL) based controller for collective navigation of unmanned aerial vehicle (UAV) swarms in communication-denied environments, enabling robust operation in complex, obstacle-rich environments. Inspired by biological swarms where informed individuals guide groups without explicit communication, we employ an implicit leader-follower framework. In this paradigm, only the leader possesses goal information, while follower UAVs learn robust policies using only onboard LiDAR sensing, without requiring any inter-agent communication or leader identification. Our system utilizes LiDAR point clustering and an extended Kalman filter for stable neighbor tracking, providing reliable perception independent of external positioning systems. The core of our approach is a DRL controller, trained in GPU-accelerated Nvidia Isaac Sim, that enables followers to learn complex emergent behaviors - balancing flocking and obstacle avoidance - using only local perception. This allows the swarm to implicitly follow the leader while robustly addressing perceptual challenges such as occlusion and limited field-of-view. The robustness and sim-to-real transfer of our approach are confirmed through extensive simulations and challenging real-world experiments with a swarm of five UAVs, which successfully demonstrated collective navigation across diverse indoor and outdoor environments without any communication or external localization.

  </details>



- **ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins**  
  Xinhao Liu, Yu Wang, Xiansheng Guo, Gordon Owusu Boateng, Yu Cao, Haonan Si, Xingchen Guo, Nirwan Ansari  
  _2026-01-20_ · https://arxiv.org/abs/2601.13706v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  High-fidelity parking-lot digital twins provide essential priors for path planning, collision checking, and perception validation in Automated Valet Parking (AVP). Yet robot-oriented reconstruction faces a trilemma: sparse forward-facing views cause weak parallax and ill-posed geometry; dynamic occlusions and extreme lighting hinder stable texture fusion; and neural rendering typically needs expensive offline optimization, violating edge-side streaming constraints. We propose ParkingTwin, a training-free, lightweight system for online streaming 3D reconstruction. First, OSM-prior-driven geometric construction uses OpenStreetMap semantic topology to directly generate a metric-consistent TSDF, replacing blind geometric search with deterministic mapping and avoiding costly optimization. Second, geometry-aware dynamic filtering employs a quad-modal constraint field (normal/height/depth consistency) to reject moving vehicles and transient occlusions in real time. Third, illumination-robust fusion in CIELAB decouples luminance and chromaticity via adaptive L-channel weighting and depth-gradient suppression, reducing seams under abrupt lighting changes. ParkingTwin runs at 30+ FPS on an entry-level GTX 1660. On a 68,000 m^2 real-world dataset, it achieves SSIM 0.87 (+16.0%), delivers about 15x end-to-end speedup, and reduces GPU memory by 83.3% compared with state-of-the-art 3D Gaussian Splatting (3DGS) that typically requires high-end GPUs (RTX 4090D). The system outputs explicit triangle meshes compatible with Unity/Unreal digital-twin pipelines. Project page: https://mihoutao-liu.github.io/ParkingTwin/

  </details>



- **FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation**  
  Jing Zuo, Lingzhou Mu, Fan Jiang, Chengcheng Ma, Mu Xu, Yonggang Qi  
  _2026-01-20_ · https://arxiv.org/abs/2601.13976v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Achieving human-level performance in Vision-and-Language Navigation (VLN) requires an embodied agent to jointly understand multimodal instructions and visual-spatial context while reasoning over long action sequences. Recent works, such as NavCoT and NavGPT-2, demonstrate the potential of Chain-of-Thought (CoT) reasoning for improving interpretability and long-horizon planning. Moreover, multimodal extensions like OctoNav-R1 and CoT-VLA further validate CoT as a promising pathway toward human-like navigation reasoning. However, existing approaches face critical drawbacks: purely textual CoTs lack spatial grounding and easily overfit to sparse annotated reasoning steps, while multimodal CoTs incur severe token inflation by generating imagined visual observations, making real-time navigation impractical. In this work, we propose FantasyVLN, a unified implicit reasoning framework that preserves the benefits of CoT reasoning without explicit token overhead. Specifically, imagined visual tokens are encoded into a compact latent space using a pretrained Visual AutoRegressor (VAR) during CoT reasoning training, and the model jointly learns from textual, visual, and multimodal CoT modes under a unified multi-CoT strategy. At inference, our model performs direct instruction-to-action mapping while still enjoying reasoning-aware representations. Extensive experiments on LH-VLN show that our approach achieves reasoning-aware yet real-time navigation, improving success rates and efficiency while reducing inference latency by an order of magnitude compared to explicit CoT methods.

  </details>



- **GuideTouch: An Obstacle Avoidance Device for Visually Impaired**  
  Timofei Kozlov, Artem Trandofilov, Georgii Gazaryan, Issatay Tokmurziyev, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13813v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Safe navigation for the visually impaired individuals remains a critical challenge, especially concerning head-level obstacles, which traditional mobility aids often fail to detect. We introduce GuideTouch, a compact, affordable, standalone wearable device designed for autonomous obstacle avoidance. The system integrates two vertically aligned Time-of-Flight (ToF) sensors, enabling three-dimensional environmental perception, and four vibrotactile actuators that provide directional haptic feedback. Proximity and direction information is communicated via an intuitive 4-point vibrotactile feedback system located across the user's shoulders and upper chest. For real-world robustness, the device includes a unique centrifugal self-cleaning optical cover mechanism and a sound alarm system for location if the device is dropped. We evaluated the haptic perception accuracy across 22 participants (17 male and 5 female, aged 21-48, mean 25.7, sd 6.1). Statistical analysis confirmed a significant difference between the perception accuracy of different patterns. The system demonstrated high recognition accuracy, achieving an average of 92.9% for single and double motor (primary directional) patterns. Furthermore, preliminary experiments with 14 visually impaired users validated this interface, showing a recognition accuracy of 93.75% for primary directional cues. The results demonstrate that GuideTouch enables intuitive spatial perception and could significantly improve the safety, confidence, and autonomy of users with visual impairments during independent navigation.

  </details>



- **PAtt: A Pattern Attention Network for ETA Prediction Using Historical Speed Profiles**  
  ByeoungDo Kim, JunYeop Na, Kyungwook Tak, JunTae Kim, DongHyeon Kim, Duckky Kim  
  _2026-01-20_ · https://arxiv.org/abs/2601.13793v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  In this paper, we propose an ETA model (Estimated Time of Arrival) that leverages an attention mechanism over historical road speed patterns. As autonomous driving and intelligent transportation systems become increasingly prevalent, the need for accurate and reliable ETA estimation has grown, playing a vital role in navigation, mobility planning, and traffic management. However, predicting ETA remains a challenging task due to the dynamic and complex nature of traffic flow. Traditional methods often combine real-time and historical traffic data in simplistic ways, or rely on complex rule-based computations. While recent deep learning models have shown potential, they often require high computational costs and do not effectively capture the spatio-temporal patterns crucial for ETA prediction. ETA prediction inherently involves spatio-temporal causality, and our proposed model addresses this by leveraging attention mechanisms to extract and utilize temporal features accumulated at each spatio-temporal point along a route. This architecture enables efficient and accurate ETA estimation while keeping the model lightweight and scalable. We validate our approach using real-world driving datasets and demonstrate that our approach outperforms existing baselines by effectively integrating road characteristics, real-time traffic conditions, and historical speed patterns in a task-aware manner.

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



- **Remapping and navigation of an embedding space via error minimization: a fundamental organizational principle of cognition in natural and artificial systems**  
  Benedikt Hartl, Léo Pio-Lopez, Chris Fields, Michael Levin  
  _2026-01-20_ · https://arxiv.org/abs/2601.14096v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The emerging field of diverse intelligence seeks an integrated view of problem-solving in agents of very different provenance, composition, and substrates. From subcellular chemical networks to swarms of organisms, and across evolved, engineered, and chimeric systems, it is hypothesized that scale-invariant principles of decision-making can be discovered. We propose that cognition in both natural and synthetic systems can be characterized and understood by the interplay between two equally important invariants: (1) the remapping of embedding spaces, and (2) the navigation within these spaces. Biological collectives, from single cells to entire organisms (and beyond), remap transcriptional, morphological, physiological, or 3D spaces to maintain homeostasis and regenerate structure, while navigating these spaces through distributed error correction. Modern Artificial Intelligence (AI) systems, including transformers, diffusion models, and neural cellular automata enact analogous processes by remapping data into latent embeddings and refining them iteratively through contextualization. We argue that this dual principle - remapping and navigation of embedding spaces via iterative error minimization - constitutes a substrate-independent invariant of cognition. Recognizing this shared mechanism not only illuminates deep parallels between living systems and artificial models, but also provides a unifying framework for engineering adaptive intelligence across scales.

  </details>



- **TSN-IoT: A Two-Stage NOMA-Enabled Framework for Prioritized Traffic Handling in Dense IoT Networks**  
  Shama Siddiqui, Anwar Ahmed Khan, Nicola Marchetti  
  _2026-01-20_ · https://arxiv.org/abs/2601.13680v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  With the growing applications of the Internet of Things (IoT), a major challenge is to ensure continuous connectivity while providing prioritized access. In dense IoT scenarios, synchronization may be disrupted either by the movement of nodes away from base stations or by the unavailability of reliable Global Navigation Satellite System (GNSS) signals, which can be affected by physical obstructions, multipath fading, or environmental interference, such as such as walls, buildings, moving objects, or electromagnetic noise from surrounding devices. In such contexts, distributed synchronization through Non-Orthogonal Multiple Access (NOMA) offers a promising solution, as it enables simultaneous transmission to multiple users with different power levels, supporting efficient synchronization while minimizing the signaling overhead. Moreover, NOMA also plays a vital role for dynamic priority management in dense and heterogeneous IoT environments. In this article, we proposed a Two-Stage NOMA-Enabled Framework "TSN-IoT" that integrates the mechanisms of conventional Precision Time Protocol (PTP) based synchronization, distributed synchronization and data transmission. The framework is designed as a four-tier architecture that facilitates prioritized data delivery from sensor nodes to the central base station. We demonstrated the performance of "TSN-IoT" through a healthcare use case, where intermittent connectivity and varying data priority levels present key challenges for reliable communication. Synchronization speed and end-to-end delay were evaluated through a series of simulations implemented in Python. Results show that, compared to priority-based Orthogonal Frequency Division Multiple Access (OFDMA), TSN-IoT achieves significantly better performance by offering improved synchronization opportunities and enabling parallel transmissions over the same sub-carrier.

  </details>


