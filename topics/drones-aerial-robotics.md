# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-25 07:14 UTC_

Total papers shown: **4**


---

- **Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones**  
  Rong Zou, Marco Cannici, Davide Scaramuzza  
  _2026-02-24_ · https://arxiv.org/abs/2602.21101v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.

  </details>



- **EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations**  
  Luka Šiktar, Branimir Ćaran, Bojan Šekoranja, Marko Švaco  
  _2026-02-24_ · https://arxiv.org/abs/2602.20958v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Search and rescue (SAR) operations require rapid responses to save lives or property. Unmanned Aerial Vehicles (UAVs) equipped with vision-based systems support these missions through prior terrain investigation or real-time assistance during the mission itself. Vision-based UAV frameworks aid human search tasks by detecting and recognizing specific individuals, then tracking and following them while maintaining a safe distance. A key safety requirement for UAV following is the accurate estimation of the distance between camera and target object under real-world conditions, achieved by fusing multiple image modalities. UAVs with deep learning-based vision systems offer a new approach to the planning and execution of SAR operations. As part of the system for automatic people detection and face recognition using deep learning, in this paper we present the fusion of depth camera measurements and monocular camera-to-body distance estimation for robust tracking and following. Deep learning-based filtering of depth camera data and estimation of camera-to-body distance from a monocular camera are achieved with YOLO-pose, enabling real-time fusion of depth information using the Extended Kalman Filter (EKF) algorithm. The proposed subsystem, designed for use in drones, estimates and measures the distance between the depth camera and the human body keypoints, to maintain the safe distance between the drone and the human target. Our system provides an accurate estimated distance, which has been validated against motion capture ground truth data. The system has been tested in real time indoors, where it reduces the average errors, root mean square error (RMSE) and standard deviations of distance estimation up to 15,3\% in three tested scenarios.

  </details>



- **Visual Cooperative Drone Tracking for Open-Path Gas Measurements**  
  Marius Schaab, Alisha Kiefer, Thomas Wiedemann, Patrick Hinsen, Achim J. Lilienthal  
  _2026-02-24_ · https://arxiv.org/abs/2602.20768v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Open-path Tunable Diode Laser Absorption Spectroscopy offers an effective method for measuring, mapping, and monitoring gas concentrations, such as leaking CO2 or methane. Compared to spatial sampling of gas distributions using in-situ sensors, open-path sensors in combination with gas tomography algorithms can cover large outdoor environments faster in a non-invasive way. However, the requirement of a dedicated reflection surface for the open-path laser makes automating the spatial sampling process challenging. This publication presents a robotic system for collecting open-path measurements, making use of a sensor mounted on a ground-based pan-tilt unit and a small drone carrying a reflector. By means of a zoom camera, the ground unit visually tracks red LED markers mounted on the drone and aligns the sensor's laser beam with the reflector. Incorporating GNSS position information provided by the drone's flight controller further improves the tracking approach. Outdoor experiments validated the system's performance, demonstrating successful autonomous tracking and valid CO2 measurements at distances up to 60 meters. Furthermore, the system successfully measured a CO2 plume without interference from the drone's propulsion system, demonstrating its superiority compared to flying in-situ sensors.

  </details>



- **ParkDiffusion++: Ego Intention Conditioned Joint Multi-Agent Trajectory Prediction for Automated Parking using Diffusion Models**  
  Jiarong Wei, Anna Rehr, Christian Feist, Abhinav Valada  
  _2026-02-24_ · https://arxiv.org/abs/2602.20923v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Automated parking is a challenging operational domain for advanced driver assistance systems, requiring robust scene understanding and interaction reasoning. The key challenge is twofold: (i) predict multiple plausible ego intentions according to context and (ii) for each intention, predict the joint responses of surrounding agents, enabling effective what-if decision-making. However, existing methods often fall short, typically treating these interdependent problems in isolation. We propose ParkDiffusion++, which jointly learns a multi-modal ego intention predictor and an ego-conditioned multi-agent joint trajectory predictor for automated parking. Our approach makes several key contributions. First, we introduce an ego intention tokenizer that predicts a small set of discrete endpoint intentions from agent histories and vectorized map polylines. Second, we perform ego-intention-conditioned joint prediction, yielding socially consistent predictions of the surrounding agents for each possible ego intention. Third, we employ a lightweight safety-guided denoiser with different constraints to refine joint scenes during training, thus improving accuracy and safety. Fourth, we propose counterfactual knowledge distillation, where an EMA teacher refined by a frozen safety-guided denoiser provides pseudo-targets that capture how agents react to alternative ego intentions. Extensive evaluations demonstrate that ParkDiffusion++ achieves state-of-the-art performance on the Dragon Lake Parking (DLP) dataset and the Intersections Drone (inD) dataset. Importantly, qualitative what-if visualizations show that other agents react appropriately to different ego intentions.

  </details>


