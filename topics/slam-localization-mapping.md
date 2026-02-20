# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-20 07:10 UTC_

Total papers shown: **13**


---

- **NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting**  
  Jiwei Shan, Zeyu Cai, Yirui Li, Yongbo Chen, Lijun Han, Yun-hui Liu, Hesheng Wang, Shing Shin Cheng  
  _2026-02-19_ · https://arxiv.org/abs/2602.17182v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Visual simultaneous localization and mapping (V-SLAM) is a fundamental capability for autonomous perception and navigation. However, endoscopic scenes violate the rigidity assumption due to persistent soft-tissue deformations, creating a strong coupling ambiguity between camera ego-motion and intrinsic deformation. Although recent monocular non-rigid SLAM methods have made notable progress, they often lack effective decoupling mechanisms and rely on sparse or low-fidelity scene representations, which leads to tracking drift and limited reconstruction quality. To address these limitations, we propose NRGS-SLAM, a monocular non-rigid SLAM system for endoscopy based on 3D Gaussian Splatting. To resolve the coupling ambiguity, we introduce a deformation-aware 3D Gaussian map that augments each Gaussian primitive with a learnable deformation probability, optimized via a Bayesian self-supervision strategy without requiring external non-rigidity labels. Building on this representation, we design a deformable tracking module that performs robust coarse-to-fine pose estimation by prioritizing low-deformation regions, followed by efficient per-frame deformation updates. A carefully designed deformable mapping module progressively expands and refines the map, balancing representational capacity and computational efficiency. In addition, a unified robust geometric loss incorporates external geometric priors to mitigate the inherent ill-posedness of monocular non-rigid SLAM. Extensive experiments on multiple public endoscopic datasets demonstrate that NRGS-SLAM achieves more accurate camera pose estimation (up to 50\% reduction in RMSE) and higher-quality photo-realistic reconstructions than state-of-the-art methods. Comprehensive ablation studies further validate the effectiveness of our key design choices. Source code will be publicly available upon paper acceptance.

  </details>



- **Multi-session Localization and Mapping Exploiting Topological Information**  
  Lorenzo Montano-Olivan, Julio A. Placed, Luis Montano, Maria T. Lazaro  
  _2026-02-19_ · https://arxiv.org/abs/2602.17226v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Operating in previously visited environments is becoming increasingly crucial for autonomous systems, with direct applications in autonomous driving, surveying, and warehouse or household robotics. This repeated exposure to observing the same areas poses significant challenges for mapping and localization -- key components for enabling any higher-level task. In this work, we propose a novel multi-session framework that builds on map-based localization, in contrast to the common practice of greedily running full SLAM sessions and trying to find correspondences between the resulting maps. Our approach incorporates a topology-informed, uncertainty-aware decision-making mechanism that analyzes the pose-graph structure to detect low-connectivity regions, selectively triggering mapping and loop closing modules. The resulting map and pose-graph are seamlessly integrated into the existing model, reducing accumulated error and enhancing global consistency. We validate our method on overlapping sequences from datasets and demonstrate its effectiveness in a real-world mine-like environment.

  </details>



- **Contact-Anchored Proprioceptive Odometry for Quadruped Robots**  
  Minxing Sun, Yao Mao  
  _2026-02-19_ · https://arxiv.org/abs/2602.17393v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable odometry for legged robots without cameras or LiDAR remains challenging due to IMU drift and noisy joint velocity sensing. This paper presents a purely proprioceptive state estimator that uses only IMU and motor measurements to jointly estimate body pose and velocity, with a unified formulation applicable to biped, quadruped, and wheel-legged robots. The key idea is to treat each contacting leg as a kinematic anchor: joint-torque--based foot wrench estimation selects reliable contacts, and the corresponding footfall positions provide intermittent world-frame constraints that suppress long-term drift. To prevent elevation drift during extended traversal, we introduce a lightweight height clustering and time-decay correction that snaps newly recorded footfall heights to previously observed support planes. To improve foot velocity observations under encoder quantization, we apply an inverse-kinematics cubature Kalman filter that directly filters foot-end velocities from joint angles and velocities. The implementation further mitigates yaw drift through multi-contact geometric consistency and degrades gracefully to a kinematics-derived heading reference when IMU yaw constraints are unavailable or unreliable. We evaluate the method on four quadruped platforms (three Astrall robots and a Unitree Go2 EDU) using closed-loop trajectories. On Astrall point-foot robot~A, a $\sim$200\,m horizontal loop and a $\sim$15\,m vertical loop return with 0.1638\,m and 0.219\,m error, respectively; on wheel-legged robot~B, the corresponding errors are 0.2264\,m and 0.199\,m. On wheel-legged robot~C, a $\sim$700\,m horizontal loop yields 7.68\,m error and a $\sim$20\,m vertical loop yields 0.540\,m error. Unitree Go2 EDU closes a $\sim$120\,m horizontal loop with 2.2138\,m error and a $\sim$8\,m vertical loop with less than 0.1\,m vertical error. github.com/ShineMinxing/Ros2Go2Estimator.git

  </details>



- **A Multi-modal Detection System for Infrastructure-based Freight Signal Priority**  
  Ziyan Zhang, Chuheng Wei, Xuanpeng Zhao, Siyan Li, Will Snyder, Mike Stas, Peng Hao, Kanok Boriboonsomsin, Guoyuan Wu  
  _2026-02-19_ · https://arxiv.org/abs/2602.17252v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Freight vehicles approaching signalized intersections require reliable detection and motion estimation to support infrastructure-based Freight Signal Priority (FSP). Accurate and timely perception of vehicle type, position, and speed is essential for enabling effective priority control strategies. This paper presents the design, deployment, and evaluation of an infrastructure-based multi-modal freight vehicle detection system integrating LiDAR and camera sensors. A hybrid sensing architecture is adopted, consisting of an intersection-mounted subsystem and a midblock subsystem, connected via wireless communication for synchronized data transmission. The perception pipeline incorporates both clustering-based and deep learning-based detection methods with Kalman filter tracking to achieve stable real-time performance. LiDAR measurements are registered into geodetic reference frames to support lane-level localization and consistent vehicle tracking. Field evaluations demonstrate that the system can reliably monitor freight vehicle movements at high spatio-temporal resolution. The design and deployment provide practical insights for developing infrastructure-based sensing systems to support FSP applications.

  </details>



- **GraphThinker: Reinforcing Video Reasoning with Event Graph Thinking**  
  Zixu Cheng, Da Li, Jian Hu, Ziquan Liu, Wei Li, Shaogang Gong  
  _2026-02-19_ · https://arxiv.org/abs/2602.17555v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Video reasoning requires understanding the causal relationships between events in a video. However, such relationships are often implicit and costly to annotate manually. While existing multimodal large language models (MLLMs) often infer event relations through dense captions or video summaries for video reasoning, such modeling still lacks causal understanding. Without explicit causal structure modeling within and across video events, these models suffer from hallucinations during the video reasoning. In this work, we propose GraphThinker, a reinforcement finetuning-based method that constructs structural event-level scene graphs and enhances visual grounding to jointly reduce hallucinations in video reasoning. Specifically, we first employ an MLLM to construct an event-based video scene graph (EVSG) that explicitly models both intra- and inter-event relations, and incorporate these formed scene graphs into the MLLM as an intermediate thinking process. We also introduce a visual attention reward during reinforcement finetuning, which strengthens video grounding and further mitigates hallucinations. We evaluate GraphThinker on two datasets, RexTime and VidHalluc, where it shows superior ability to capture object and event relations with more precise event localization, reducing hallucinations in video reasoning compared to prior methods.

  </details>



- **Device-Centric ISAC for Exposure Control via Opportunistic Virtual Aperture Sensing**  
  Marouan Mizmizi, Zhibin Yu, Guanglong Du, Umberto Spagnolini  
  _2026-02-19_ · https://arxiv.org/abs/2602.17609v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Regulatory limits on Maximum Permissible Exposure (MPE) require handheld devices to reduce transmit power when operated near the user's body. Current proximity sensors provide only binary detection, triggering conservative power back-off that degrades link quality. If the device could measure its distance from the body, transmit power could be adjusted proportionally, improving throughput while maintaining compliance. This paper develops a device-centric integrated sensing and communication (ISAC) method for the device to measure this distance. The uplink communication waveform is exploited for sensing, and the natural motion of the user's hand creates a virtual aperture that provides the angular resolution necessary for localization. Virtual aperture processing requires precise knowledge of the device trajectory, which in this scenario is opportunistic and unknown. One can exploit onboard inertial sensors to estimate the device trajectory; however, the inertial sensors accuracy is not sufficient. To address this, we develop an autofocus algorithm based on extended Kalman filtering that jointly tracks the trajectory and compensates residual errors using phase observations from strong scatterers. The Bayesian Cramér-Rao bound for localization is derived under correlated inertial errors. Numerical results at 28GHz demonstrate centimeter-level accuracy with realistic sensor parameters.

  </details>



- **TimeOmni-VL: Unified Models for Time Series Understanding and Generation**  
  Tong Guan, Sheng Pan, Johan Barthelemy, Zhao Li, Yujun Cai, Cesare Alippi, Ming Jin, Shirui Pan  
  _2026-02-19_ · https://arxiv.org/abs/2602.17149v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Recent time series modeling faces a sharp divide between numerical generation and semantic understanding, with research showing that generation models often rely on superficial pattern matching, while understanding-oriented models struggle with high-fidelity numerical output. Although unified multimodal models (UMMs) have bridged this gap in vision, their potential for time series remains untapped. We propose TimeOmni-VL, the first vision-centric framework that unifies time series understanding and generation through two key innovations: (1) Fidelity-preserving bidirectional mapping between time series and images (Bi-TSI), which advances Time Series-to-Image (TS2I) and Image-to-Time Series (I2TS) conversions to ensure near-lossless transformations. (2) Understanding-guided generation. We introduce TSUMM-Suite, a novel dataset consists of six understanding tasks rooted in time series analytics that are coupled with two generation tasks. With a calibrated Chain-of-Thought, TimeOmni-VL is the first to leverage time series understanding as an explicit control signal for high-fidelity generation. Experiments confirm that this unified approach significantly improves both semantic understanding and numerical precision, establishing a new frontier for multimodal time series modeling.

  </details>



- **Grasp Synthesis Matching From Rigid To Soft Robot Grippers Using Conditional Flow Matching**  
  Tanisha Parulekar, Ge Shi, Josh Pinskier, David Howard, Jen Jen Chung  
  _2026-02-19_ · https://arxiv.org/abs/2602.17110v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  A representation gap exists between grasp synthesis for rigid and soft grippers. Anygrasp [1] and many other grasp synthesis methods are designed for rigid parallel grippers, and adapting them to soft grippers often fails to capture their unique compliant behaviors, resulting in data-intensive and inaccurate models. To bridge this gap, this paper proposes a novel framework to map grasp poses from a rigid gripper model to a soft Fin-ray gripper. We utilize Conditional Flow Matching (CFM), a generative model, to learn this complex transformation. Our methodology includes a data collection pipeline to generate paired rigid-soft grasp poses. A U-Net autoencoder conditions the CFM model on the object's geometry from a depth image, allowing it to learn a continuous mapping from an initial Anygrasp pose to a stable Fin-ray gripper pose. We validate our approach on a 7-DOF robot, demonstrating that our CFM-generated poses achieve a higher overall success rate for seen and unseen objects (34% and 46% respectively) compared to the baseline rigid poses (6% and 25% respectively) when executed by the soft gripper. The model shows significant improvements, particularly for cylindrical (50% and 100% success for seen and unseen objects) and spherical objects (25% and 31% success for seen and unseen objects), and successfully generalizes to unseen objects. This work presents CFM as a data-efficient and effective method for transferring grasp strategies, offering a scalable methodology for other soft robotic systems.

  </details>



- **FoundationPose-Initialized 3D-2D Liver Registration for Surgical Augmented Reality**  
  Hanyuan Zhang, Lucas He, Runlong He, Abdolrahim Kadkhodamohammadi, Danail Stoyanov, Brian R. Davidson, Evangelos B. Mazomenos, Matthew J. Clarkson  
  _2026-02-19_ · https://arxiv.org/abs/2602.17517v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Augmented reality can improve tumor localization in laparoscopic liver surgery. Existing registration pipelines typically depend on organ contours; deformable (non-rigid) alignment is often handled with finite-element (FE) models coupled to dimensionality-reduction or machine-learning components. We integrate laparoscopic depth maps with a foundation pose estimator for camera-liver pose estimation and replace FE-based deformation with non-rigid iterative closest point (NICP) to lower engineering/modeling complexity and expertise requirements. On real patient data, the depth-augmented foundation pose approach achieved 9.91 mm mean registration error in 3 cases. Combined rigid-NICP registration outperformed rigid-only registration, demonstrating NICP as an efficient substitute for finite-element deformable models. This pipeline achieves clinically relevant accuracy while offering a lightweight, engineering-friendly alternative to FE-based deformation.

  </details>



- **Tree crop mapping of South America reveals links to deforestation and conservation**  
  Yuchang Jiang, Anton Raichuk, Xiaoye Tong, Vivien Sainte Fare Garnot, Daniel Ortiz-Gonzalo, Dan Morris, Konrad Schindler, Jan Dirk Wegner, Maxim Neumann  
  _2026-02-19_ · https://arxiv.org/abs/2602.17372v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Monitoring tree crop expansion is vital for zero-deforestation policies like the European Union's Regulation on Deforestation-free Products (EUDR). However, these efforts are hindered by a lack of highresolution data distinguishing diverse agricultural systems from forests. Here, we present the first 10m-resolution tree crop map for South America, generated using a multi-modal, spatio-temporal deep learning model trained on Sentinel-1 and Sentinel-2 satellite imagery time series. The map identifies approximately 11 million hectares of tree crops, 23% of which is linked to 2000-2020 forest cover loss. Critically, our analysis reveals that existing regulatory maps supporting the EUDR often classify established agriculture, particularly smallholder agroforestry, as "forest". This discrepancy risks false deforestation alerts and unfair penalties for small-scale farmers. Our work mitigates this risk by providing a high-resolution baseline, supporting conservation policies that are effective, inclusive, and equitable.

  </details>



- **Inferring Height from Earth Embeddings: First insights using Google AlphaEarth**  
  Alireza Hamoudzadeh, Valeria Belloni, Roberta Ravanelli  
  _2026-02-19_ · https://arxiv.org/abs/2602.17250v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This study investigates whether the geospatial and multimodal features encoded in \textit{Earth Embeddings} can effectively guide deep learning (DL) regression models for regional surface height mapping. In particular, we focused on AlphaEarth Embeddings at 10 m spatial resolution and evaluated their capability to support terrain height inference using a high-quality Digital Surface Model (DSM) as reference. U-Net and U-Net++ architectures were thus employed as lightweight convolutional decoders to assess how well the geospatial information distilled in the embeddings can be translated into accurate surface height estimates. Both architectures achieved strong training performance (both with $R^2 = 0.97$), confirming that the embeddings encode informative and decodable height-related signals. On the test set, performance decreased due to distribution shifts in height frequency between training and testing areas. Nevertheless, U-Net++ shows better generalization ($R^2 = 0.84$, median difference = -2.62 m) compared with the standard U-Net ($R^2 = 0.78$, median difference = -7.22 m), suggesting enhanced robustness to distribution mismatch. While the testing RMSE (approximately 16 m for U-Net++) and residual bias highlight remaining challenges in generalization, strong correlations indicate that the embeddings capture transferable topographic patterns. Overall, the results demonstrate the promising potential of AlphaEarth Embeddings to guide DL-based height mapping workflows, particularly when combined with spatially aware convolutional architectures, while emphasizing the need to address bias for improved regional transferability.

  </details>



- **Dynamic Decision-Making under Model Misspecification: A Stochastic Stability Approach**  
  Xinyu Dai, Daniel Chen, Yian Qian  
  _2026-02-19_ · https://arxiv.org/abs/2602.17086v1 · `econ.TH`  
  <details><summary>Abstract</summary>

  Dynamic decision-making under model uncertainty is central to many economic environments, yet existing bandit and reinforcement learning algorithms rely on the assumption of correct model specification. This paper studies the behavior and performance of one of the most commonly used Bayesian reinforcement learning algorithms, Thompson Sampling (TS), when the model class is misspecified. We first provide a complete dynamic classification of posterior evolution in a misspecified two-armed Gaussian bandit, identifying distinct regimes: correct model concentration, incorrect model concentration, and persistent belief mixing, characterized by the direction of statistical evidence and the model-action mapping. These regimes yield sharp predictions for limiting beliefs, action frequencies, and asymptotic regret. We then extend the analysis to a general finite model class and develop a unified stochastic stability framework that represents posterior evolution as a Markov process on the belief simplex. This approach characterizes two sufficient conditions to classify the ergodic and transient behaviors and provides inductive dimensional reductions of the posterior dynamics. Our results offer the first qualitative and geometric classification of TS under misspecification, bridging Bayesian learning with evolutionary dynamics, and also build the foundations of robust decision-making in structured bandits.

  </details>



- **ComptonUNet: A Deep Learning Model for GRB Localization with Compton Cameras under Noisy and Low-Statistic Conditions**  
  Shogo Sato, Kazuo Tanaka, Shojun Ogasawara, Kazuki Yamamoto, Kazuhiko Murasaki, Ryuichi Tanida, Jun Kataoka  
  _2026-02-19_ · https://arxiv.org/abs/2602.17085v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Gamma-ray bursts (GRBs) are among the most energetic transient phenomena in the universe and serve as powerful probes for high-energy astrophysical processes. In particular, faint GRBs originating from a distant universe may provide unique insights into the early stages of star formation. However, detecting and localizing such weak sources remains challenging owing to low photon statistics and substantial background noise. Although recent machine learning models address individual aspects of these challenges, they often struggle to balance the trade-off between statistical robustness and noise suppression. Consequently, we propose ComptonUNet, a hybrid deep learning framework that jointly processes raw data and reconstructs images for robust GRB localization. ComptonUNet was designed to operate effectively under conditions of limited photon statistics and strong background contamination by combining the statistical efficiency of direct reconstruction models with the denoising capabilities of image-based architectures. We perform realistic simulations of GRB-like events embedded in background environments representative of low-Earth orbit missions to evaluate the performance of ComptonUNet. Our results demonstrate that ComptonUNet significantly outperforms existing approaches, achieving improved localization accuracy across a wide range of low-statistic and high-background scenarios.

  </details>


