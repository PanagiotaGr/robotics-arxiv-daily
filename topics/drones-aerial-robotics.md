# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-03-03 07:06 UTC_

Total papers shown: **8**


---

- **Real-Time Thermal-Inertial Odometry on Embedded Hardware for High-Speed GPS-Denied Flight**  
  Austin Stone, Mark Petersen, Cammy Peterson  
  _2026-03-02_ · https://arxiv.org/abs/2603.02114v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a real-time monocular thermal-inertial odometry system designed for high-velocity, GPS-denied flight on embedded hardware. The system fuses measurements from a FLIR Boson+ 640 longwave infrared camera, a high-rate IMU, a laser range finder, a barometer, and a magnetometer within a fixed-lag factor graph. To sustain reliable feature tracks under motion blur, low contrast, and rapid viewpoint changes, we employ a lightweight thermal-optimized front-end with multi-stage feature filtering. Laser range finder measurements provide per-feature depth priors that stabilize scale during weakly observable motion. High-rate inertial data is first pre-filtered using a Chebyshev Type II infinite impulse response (IIR) filter and then preintegrated, improving robustness to airframe vibrations during aggressive maneuvers. To address barometric altitude errors induced at high airspeeds, we train an uncertainty-aware gated recurrent unit (GRU) network that models the temporal dynamics of static pressure distortion, outperforming polynomial and multi-layer perceptron (MLP) baselines. Integrated on an NVIDIA Jetson Xavier NX, the complete system supports closed-loop quadrotor flight at 30 m/s with drift under 2% over kilometer-scale trajectories. These contributions expand the operational envelope of thermal-inertial navigation, enabling reliable high-speed flight in visually degraded and GPS-denied environments.

  </details>



- **CHOP: Counterfactual Human Preference Labels Improve Obstacle Avoidance in Visuomotor Navigation Policies**  
  Gershom Seneviratne, Jianyu An, Vaibhav Shende, Sahire Ellahy, Yaxita Amin, Kondapi Manasanjani, Samarth Chopra, Jonathan Deepak Kannan, Dinesh Manocha  
  _2026-03-02_ · https://arxiv.org/abs/2603.02004v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visuomotor navigation policies have shown strong perception-action coupling for embodied agents, yet they often struggle with safe navigation and dynamic obstacle avoidance in complex real-world environments. We introduce CHOP, a novel approach that leverages Counterfactual Human Preference Labels to align visuomotor navigation policies towards human intuition of safety and obstacle avoidance in navigation. In CHOP, for each visual observation, the robot's executed trajectory is included among a set of counterfactual navigation trajectories: alternative trajectories the robot could have followed under identical conditions. Human annotators provide pairwise preference labels over these trajectories based on anticipated outcomes such as collision risk and path efficiency. These aggregated preferences are then used to fine-tune visuomotor navigation policies, aligning their behavior with human preferences in navigation. Experiments on the SCAND dataset show that visuomotor navigation policies fine-tuned with CHOP reduce near-collision events by 49.7%, decrease deviation from human-preferred trajectories by 45.0%, and increase average obstacle clearance by 19.8% on average across multiple state-of-the-art models, compared to their pretrained baselines. These improvements transfer to real-world deployments on a Ghost Robotics Vision60 quadruped, where CHOP-aligned policies improve average goal success rates by 24.4%, increase minimum obstacle clearance by 6.8%, reduce collision and intervention events by 45.7%, and improve normalized path completion by 38.6% on average across navigation scenarios, compared to their pretrained baselines. Our results highlight the value of counterfactual preference supervision in bridging the gap between large-scale visuomotor policies and human-aligned, safety-aware embodied navigation.

  </details>



- **Learning Vision-Based Omnidirectional Navigation: A Teacher-Student Approach Using Monocular Depth Estimation**  
  Jan Finke, Wayne Paul Martis, Adrian Schmelter, Lars Erbach, Christian Jestel, Marvin Wiedemann  
  _2026-03-02_ · https://arxiv.org/abs/2603.01999v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable obstacle avoidance in industrial settings demands 3D scene understanding, but widely used 2D LiDAR sensors perceive only a single horizontal slice of the environment, missing critical obstacles above or below the scan plane. We present a teacher-student framework for vision-based mobile robot navigation that eliminates the need for LiDAR sensors. A teacher policy trained via Proximal Policy Optimization (PPO) in NVIDIA Isaac Lab leverages privileged 2D LiDAR observations that account for the full robot footprint to learn robust navigation. The learned behavior is distilled into a student policy that relies solely on monocular depth maps predicted by a fine-tuned Depth Anything V2 model from four RGB cameras. The complete inference pipeline, comprising monocular depth estimation (MDE), policy execution, and motor control, runs entirely onboard an NVIDIA Jetson Orin AGX mounted on a DJI RoboMaster platform, requiring no external computation for inference. In simulation, the student achieves success rates of 82-96.5%, consistently outperforming the standard 2D LiDAR teacher (50-89%). In real-world experiments, the MDE-based student outperforms the 2D LiDAR teacher when navigating around obstacles with complex 3D geometries, such as overhanging structures and low-profile objects, that fall outside the single scan plane of a 2D LiDAR.

  </details>



- **Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control**  
  Peng Yu, Xin Wang, Ning Tan  
  _2026-03-02_ · https://arxiv.org/abs/2603.01751v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Continuum robots possess high flexibility and redundancy, making them well suited for safe interaction in complex environments, yet their continuous deformation and nonlinear dynamics pose fundamental challenges to perception, modeling, and control. Existing vision-based control approaches often rely on end-to-end learning, achieving shape regulation without explicit awareness of robot geometry or its interaction with the environment. Here, we introduce a shape-interpretable visual self-modeling framework for continuum robots that enables geometry-aware control. Robot shapes are encoded from multi-view planar images using a Bezier-curve representation, transforming visual observations into a compact and physically meaningful shape space that uniquely characterizes the robot's three-dimensional configuration. Based on this representation, neural ordinary differential equations are employed to self-model both shape and end-effector dynamics directly from data, enabling hybrid shape-position control without analytical models or dense body markers. The explicit geometric structure of the learned shape space allows the robot to reason about its body and surroundings, supporting environment-aware behaviors such as obstacle avoidance and self-motion while maintaining end-effector objectives. Experiments on a cable-driven continuum robot demonstrate accurate shape-position regulation and tracking, with shape errors within 1.56% of image resolution and end-effector errors within 2% of robot length, as well as robust performance in constrained environments. By elevating visual shape representations from two-dimensional observations to an interpretable three-dimensional self-model, this work establishes a principled alternative to vision-based end-to-end control and advances autonomous, geometry-aware manipulation for continuum robots.

  </details>



- **Stereo-Inertial Poser: Towards Metric-Accurate Shape-Aware Motion Capture Using Sparse IMUs and a Single Stereo Camera**  
  Tutian Tang, Xingyu Ji, Yutong Li, MingHao Liu, Wenqiang Xu, Cewu Lu  
  _2026-03-02_ · https://arxiv.org/abs/2603.02130v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advancements in visual-inertial motion capture systems have demonstrated the potential of combining monocular cameras with sparse inertial measurement units (IMUs) as cost-effective solutions, which effectively mitigate occlusion and drift issues inherent in single-modality systems. However, they are still limited by metric inaccuracies in global translations stemming from monocular depth ambiguity, and shape-agnostic local motion estimations that ignore anthropometric variations. We present Stereo-Inertial Poser, a real-time motion capture system that leverages a single stereo camera and six IMUs to estimate metric-accurate and shape-aware 3D human motion. By replacing the monocular RGB with stereo vision, our system resolves depth ambiguity through calibrated baseline geometry, enabling direct 3D keypoint extraction and body shape parameter estimation. IMU data and visual cues are fused for predicting drift-compensated joint positions and root movements, while a novel shape-aware fusion module dynamically harmonizes anthropometry variations with global translations. Our end-to-end pipeline achieves over 200 FPS without optimization-based post-processing, enabling real-time deployment. Quantitative evaluations across various datasets demonstrate state-of-the-art performance. Qualitative results show our method produces drift-free global translation under a long recording time and reduces foot-skating effects.

  </details>



- **BAWSeg: A UAV Multispectral Benchmark for Barley Weed Segmentation**  
  Haitian Wang, Xinyu Wang, Muhammad Ibrahim, Dustin Severtson, Ajmal Mian  
  _2026-03-02_ · https://arxiv.org/abs/2603.01932v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate weed mapping in cereal fields requires pixel-level segmentation from UAV imagery that remains reliable across fields, seasons, and illumination. Existing multispectral pipelines often depend on thresholded vegetation indices, which are brittle under radiometric drift and mixed crop--weed pixels, or on single-stream CNN and Transformer backbones that ingest stacked bands and indices, where radiance cues and normalized index cues interfere and reduce sensitivity to small weed clusters embedded in crop canopies. We propose VISA (Vegetation-Index and Spectral Attention), a two-stream segmentation network that decouples these cues and fuses them at native resolution. The radiance stream learns from calibrated five-band reflectance using residual spectral-spatial attention to preserve fine textures and row boundaries that are attenuated by ratio indices. The index stream operates on vegetation-index maps with windowed self-attention to model local structure efficiently, state-space layers to propagate field-scale context without quadratic attention cost, and Slot Attention to form stable region descriptors that improve discrimination of sparse weeds under canopy mixing. To support supervised training and deployment-oriented evaluation, we introduce BAWSeg, a four-year UAV multispectral dataset collected over commercial barley paddocks in Western Australia, providing radiometrically calibrated blue, green, red, red edge, and near-infrared orthomosaics, derived vegetation indices, and dense crop, weed, and other labels with leakage-free block splits. On BAWSeg, VISA achieves 75.6% mIoU and 63.5% weed IoU with 22.8M parameters, outperforming a multispectral SegFormer-B1 baseline by 1.2 mIoU and 1.9 weed IoU. Under cross-plot and cross-year protocols, VISA maintains 71.2% and 69.2% mIoU, respectively. The BAWSeg data, VISA code, and trained models will be released upon publication.

  </details>



- **Tiny-DroNeRF: Tiny Neural Radiance Fields aboard Federated Learning-enabled Nano-drones**  
  Ilenia Carboni, Elia Cereda, Lorenzo Lamberti, Daniele Malpetti, Francesco Conti, Daniele Palossi  
  _2026-03-02_ · https://arxiv.org/abs/2603.01850v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Sub-30g nano-sized aerial robots can leverage their agility and form factor to autonomously explore cluttered and narrow environments, like in industrial inspection and search and rescue missions. However, the price for their tiny size is a strong limit in their resources, i.e., sub-100 mW microcontroller units (MCUs) delivering $\sim$100 GOps/s at best, and memory budgets well below 100 MB. Despite these strict constraints, we aim to enable complex vision-based tasks aboard nano-drones, such as dense 3D scene reconstruction: a key robotic task underlying fundamental capabilities like spatial awareness and motion planning. Top-performing 3D reconstruction methods leverage neural radiance fields (NeRF) models, which require GBs of memory and massive computation, usually delivered by high-end GPUs consuming 100s of Watts. Our work introduces Tiny-DroNeRF, a lightweight NeRF model, based on Instant-NGP, and optimized for running on a GAP9 ultra-low-power (ULP) MCU aboard our nano-drones. Then, we further empower our Tiny-DroNeRF by leveraging a collaborative federated learning scheme, which distributes the model training among multiple nano-drones. Our experimental results show a 96% reduction in Tiny-DroNeRF's memory footprint compared to Instant-NGP, with only a 5.7 dB drop in reconstruction accuracy. Finally, our federated learning scheme allows Tiny-DroNeRF to train with an amount of data otherwise impossible to keep in a single drone's memory, increasing the overall reconstruction accuracy. Ultimately, our work combines, for the first time, NeRF training on an ULP MCU with federated learning on nano-drones.

  </details>



- **Event-Only Drone Trajectory Forecasting with RPM-Modulated Kalman Filtering**  
  Hari Prasanth S. M., Pejman Habibiroudkenar, Eerik Alamikkotervo, Dimitrios Bouzoulas, Risto Ojala  
  _2026-03-02_ · https://arxiv.org/abs/2603.01997v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Event cameras provide high-temporal-resolution visual sensing that is well suited for observing fast-moving aerial objects; however, their use for drone trajectory prediction remains limited. This work introduces an event-only drone forecasting method that exploits propeller-induced motion cues. Propeller rotational speed are extracted directly from raw event data and fused within an RPM-aware Kalman filtering framework. Evaluations on the FRED dataset show that the proposed method outperforms learning-based approaches and vanilla kalman filter in terms of average distance error and final distance error at 0.4s and 0.8s forecasting horizons. The results demonstrate robust and accurate short- and medium-horizon trajectory forecasting without reliance on RGB imagery or training data.

  </details>


