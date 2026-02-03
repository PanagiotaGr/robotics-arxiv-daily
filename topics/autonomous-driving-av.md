# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-03 07:06 UTC_

Total papers shown: **4**


---

- **LiFlow: Flow Matching for 3D LiDAR Scene Completion**  
  Andrea Matteazzi, Dietmar Tutsch  
  _2026-02-02_ · https://arxiv.org/abs/2602.02232v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In autonomous driving scenarios, the collected LiDAR point clouds can be challenged by occlusion and long-range sparsity, limiting the perception of autonomous driving systems. Scene completion methods can infer the missing parts of incomplete 3D LiDAR scenes. Recent methods adopt local point-level denoising diffusion probabilistic models, which require predicting Gaussian noise, leading to a mismatch between training and inference initial distributions. This paper introduces the first flow matching framework for 3D LiDAR scene completion, improving upon diffusion-based methods by ensuring consistent initial distributions between training and inference. The model employs a nearest neighbor flow matching loss and a Chamfer distance loss to enhance both local structure and global coverage in the alignment of point clouds. LiFlow achieves state-of-the-art performance across multiple metrics. Code: https://github.com/matteandre/LiFlow.

  </details>



- **Real-Time 2D LiDAR Object Detection Using Three-Frame RGB Scan Encoding**  
  Soheil Behnam Roudsari, Alexandre S. Brandão, Felipe N. Martins  
  _2026-02-02_ · https://arxiv.org/abs/2602.02167v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Indoor service robots need perception that is robust, more privacy-friendly than RGB video, and feasible on embedded hardware. We present a camera-free 2D LiDAR object detection pipeline that encodes short-term temporal context by stacking three consecutive scans as RGB channels, yielding a compact YOLOv8n input without occupancy-grid construction while preserving angular structure and motion cues. Evaluated in Webots across 160 randomized indoor scenarios with strict scenario-level holdout, the method achieves 98.4% mAP@0.5 (0.778 mAP@0.5:0.95) with 94.9% precision and 94.7% recall on four object classes. On a Raspberry Pi 5, it runs in real time with a mean post-warm-up end-to-end latency of 47.8ms per frame, including scan encoding and postprocessing. Relative to a closely related occupancy-grid LiDAR-YOLO pipeline reported on the same platform, the proposed representation is associated with substantially lower reported end-to-end latency. Although results are simulation-based, they suggest that lightweight temporal encoding can enable accurate and real-time LiDAR-only detection for embedded indoor robotics without capturing RGB appearance.

  </details>



- **Enhancing Indoor Occupancy Prediction via Sparse Query-Based Multi-Level Consistent Knowledge Distillation**  
  Xiang Li, Yupeng Zheng, Pengfei Li, Yilun Chen, Ya-Qin Zhang, Wenchao Ding  
  _2026-02-02_ · https://arxiv.org/abs/2602.02318v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Occupancy prediction provides critical geometric and semantic understanding for robotics but faces efficiency-accuracy trade-offs. Current dense methods suffer computational waste on empty voxels, while sparse query-based approaches lack robustness in diverse and complex indoor scenes. In this paper, we propose DiScene, a novel sparse query-based framework that leverages multi-level distillation to achieve efficient and robust occupancy prediction. In particular, our method incorporates two key innovations: (1) a Multi-level Consistent Knowledge Distillation strategy, which transfers hierarchical representations from large teacher models to lightweight students through coordinated alignment across four levels, including encoder-level feature alignment, query-level feature matching, prior-level spatial guidance, and anchor-level high-confidence knowledge transfer and (2) a Teacher-Guided Initialization policy, employing optimized parameter warm-up to accelerate model convergence. Validated on the Occ-Scannet benchmark, DiScene achieves 23.2 FPS without depth priors while outperforming our baseline method, OPUS, by 36.1% and even better than the depth-enhanced version, OPUS†. With depth integration, DiScene† attains new SOTA performance, surpassing EmbodiedOcc by 3.7% with 1.62$\times$ faster inference speed. Furthermore, experiments on the Occ3D-nuScenes benchmark and in-the-wild scenarios demonstrate the versatility of our approach in various environments. Code and models can be accessed at https://github.com/getterupper/DiScene.

  </details>



- **Online Fine-Tuning of Pretrained Controllers for Autonomous Driving via Real-Time Recurrent RL**  
  Julian Lemmel, Felix Resch, Mónika Farsang, Ramin Hasani, Daniela Rus, Radu Grosu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02236v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deploying pretrained policies in real-world applications presents substantial challenges that fundamentally limit the practical applicability of learning-based control systems. When autonomous systems encounter environmental changes in system dynamics, sensor drift, or task objectives, fixed policies rapidly degrade in performance. We show that employing Real-Time Recurrent Reinforcement Learning (RTRRL), a biologically plausible algorithm for online adaptation, can effectively fine-tune a pretrained policy to improve autonomous agents' performance on driving tasks. We further show that RTRRL synergizes with a recent biologically inspired recurrent network model, the Liquid-Resistance Liquid-Capacitance RNN. We demonstrate the effectiveness of this closed-loop approach in a simulated CarRacing environment and in a real-world line-following task with a RoboRacer car equipped with an event camera.

  </details>


