# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-26 07:13 UTC_

Total papers shown: **5**


---

- **Dream-SLAM: Dreaming the Unseen for Active SLAM in Dynamic Environments**  
  Xiangqi Meng, Pengxu Hou, Zhenjun Zhao, Javier Civera, Daniel Cremers, Hesheng Wang, Haoang Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21967v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In addition to the core tasks of simultaneous localization and mapping (SLAM), active SLAM additionally in- volves generating robot actions that enable effective and efficient exploration of unknown environments. However, existing active SLAM pipelines are limited by three main factors. First, they inherit the restrictions of the underlying SLAM modules that they may be using. Second, their motion planning strategies are typically shortsighted and lack long-term vision. Third, most approaches struggle to handle dynamic scenes. To address these limitations, we propose a novel monocular active SLAM method, Dream-SLAM, which is based on dreaming cross-spatio-temporal images and semantically plausible structures of partially observed dynamic environments. The generated cross-spatio-temporal im- ages are fused with real observations to mitigate noise and data incompleteness, leading to more accurate camera pose estimation and a more coherent 3D scene representation. Furthermore, we integrate dreamed and observed scene structures to enable long- horizon planning, producing farsighted trajectories that promote efficient and thorough exploration. Extensive experiments on both public and self-collected datasets demonstrate that Dream-SLAM outperforms state-of-the-art methods in localization accuracy, mapping quality, and exploration efficiency. Source code will be publicly available upon paper acceptance.

  </details>



- **UNet-Based Keypoint Regression for 3D Cone Localization in Autonomous Racing**  
  Mariia Baidachna, James Carty, Aidan Ferguson, Joseph Agrane, Varad Kulkarni, Aubrey Agub, Michael Baxendale, Aaron David, Rachel Horton, Elliott Atkinson  
  _2026-02-25_ · https://arxiv.org/abs/2602.21904v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate cone localization in 3D space is essential in autonomous racing for precise navigation around the track. Approaches that rely on traditional computer vision algorithms are sensitive to environmental variations, and neural networks are often trained on limited data and are infeasible to run in real time. We present a UNet-based neural network for keypoint detection on cones, leveraging the largest custom-labeled dataset we have assembled. Our approach enables accurate cone position estimation and the potential for color prediction. Our model achieves substantial improvements in keypoint accuracy over conventional methods. Furthermore, we leverage our predicted keypoints in the perception pipeline and evaluate the end-to-end autonomous system. Our results show high-quality performance across all metrics, highlighting the effectiveness of this approach and its potential for adoption in competitive autonomous racing systems.

  </details>



- **GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL**  
  Rui Yang, Qianhui Wu, Zhaoyang Wang, Hanyang Chen, Ke Yang, Hao Cheng, Huaxiu Yao, Baoling Peng, Huan Zhang, Jianfeng Gao, et al.  
  _2026-02-25_ · https://arxiv.org/abs/2602.22190v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Open-source native GUI agents still lag behind closed-source systems on long-horizon navigation tasks. This gap stems from two limitations: a shortage of high-quality, action-aligned reasoning data, and the direct adoption of generic post-training pipelines that overlook the unique challenges of GUI agents. We identify two fundamental issues in these pipelines: (i) standard SFT with CoT reasoning often hurts grounding, and (ii) step-wise RLVR-tyle training faces partial verifiability, where multiple actions can be correct but only a single demonstrated action is used for verification. This makes offline step-wise metrics weak predictors of online task success. In this work, we present GUI-Libra, a tailored training recipe that addresses these challenges. First, to mitigate the scarcity of action-aligned reasoning data, we introduce a data construction and filtering pipeline and release a curated 81K GUI reasoning dataset. Second, to reconcile reasoning with grounding, we propose action-aware SFT that mixes reasoning-then-action and direct-action data and reweights tokens to emphasize action and grounding. Third, to stabilize RL under partial verifiability, we identify the overlooked importance of KL regularization in RLVR and show that a KL trust region is critical for improving offline-to-online predictability; we further introduce success-adaptive scaling to downweight unreliable negative gradients. Across diverse web and mobile benchmarks, GUI-Libra consistently improves both step-wise accuracy and end-to-end task completion. Our results suggest that carefully designed post-training and data curation can unlock significantly stronger task-solving capabilities without costly online data collection. We release our dataset, code, and models to facilitate further research on data-efficient post-training for reasoning-capable GUI agents.

  </details>



- **EndoDDC: Learning Sparse to Dense Reconstruction for Endoscopic Robotic Navigation via Diffusion Depth Completion**  
  Yinheng Lin, Yiming Huang, Beilei Cui, Long Bai, Huxin Gao, Hongliang Ren, Jiewen Lai  
  _2026-02-25_ · https://arxiv.org/abs/2602.21893v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate depth estimation plays a critical role in the navigation of endoscopic surgical robots, forming the foundation for 3D reconstruction and safe instrument guidance. Fine-tuning pretrained models heavily relies on endoscopic surgical datasets with precise depth annotations. While existing self-supervised depth estimation techniques eliminate the need for accurate depth annotations, their performance degrades in environments with weak textures and variable lighting, leading to sparse reconstruction with invalid depth estimation. Depth completion using sparse depth maps can mitigate these issues and improve accuracy. Despite the advances in depth completion techniques in general fields, their application in endoscopy remains limited. To overcome these limitations, we propose EndoDDC, an endoscopy depth completion method that integrates images, sparse depth information with depth gradient features, and optimizes depth maps through a diffusion model, addressing the issues of weak texture and light reflection in endoscopic environments. Extensive experiments on two publicly available endoscopy datasets show that our approach outperforms state-of-the-art models in both depth accuracy and robustness. This demonstrates the potential of our method to reduce visual errors in complex endoscopic environments. Our code will be released at https://github.com/yinheng-lin/EndoDDC.

  </details>



- **Pilot-Free Optimal Control over Wireless Networks: A Control-Aided Channel Prediction Approach**  
  Minjie Tang, Zunqi Li, Photios A. Stavrou, Marios Kountouris  
  _2026-02-25_ · https://arxiv.org/abs/2602.21752v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  A recurring theme in optimal controller design for wireless networked control systems (WNCS) is the reliance on real-time channel state information (CSI). However, acquiring accurate CSI a priori is notoriously challenging due to the time-varying nature of wireless channels. In this work, we propose a pilot-free framework for optimal control over wireless channels in which control commands are generated from plant states together with control-aided channel prediction. For linear plants operating over an orthogonal frequency-division multiplexing (OFDM) architecture, channel prediction is performed via a Kalman filter (KF), and the optimal control policy is derived from the Bellman principle. To alleviate the curse of dimensionality in computing the optimal control policy, we approximate the solution using a coupled algebraic Riccati equation (CARE), which can be computed efficiently via a stochastic approximation (SA) algorithm. Rigorous performance guarantees are established by proving the stability of both the channel predictor and the closed-loop system under the resulting control policy, providing sufficient conditions for the existence and uniqueness of a stabilizing approximate CARE solution, and establishing convergence of the SA-based control algorithm. The framework is further extended to nonlinear plants under general wireless architectures by combining a KalmanNet-based predictor with a Markov-modulated deep deterministic policy gradient (MM-DDPG) controller. Numerical results show that the proposed pilot-free approach outperforms benchmark schemes in both control performance and channel prediction accuracy for linear and nonlinear scenarios.

  </details>


