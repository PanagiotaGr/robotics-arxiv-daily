# 3D Gaussian Splatting & Neural Rendering (Robotics)

_Robotics arXiv Daily_

_Updated: 2026-02-26 07:13 UTC_

Total papers shown: **3**


---

- **DAGS-SLAM: Dynamic-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty-Aware Scheduling**  
  Li Zhang, Yu-An Liu, Xijia Jiang, Conghao Huang, Danyang Li, Yanyong Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21644v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mobile robots and IoT devices demand real-time localization and dense reconstruction under tight compute and energy budgets. While 3D Gaussian Splatting (3DGS) enables efficient dense SLAM, dynamic objects and occlusions still degrade tracking and mapping. Existing dynamic 3DGS-SLAM often relies on heavy optical flow and per-frame segmentation, which is costly for mobile deployment and brittle under challenging illumination. We present DAGS-SLAM, a dynamic-aware 3DGS-SLAM system that maintains a spatiotemporal motion probability (MP) state per Gaussian and triggers semantics on demand via an uncertainty-aware scheduler. DAGS-SLAM fuses lightweight YOLO instance priors with geometric cues to estimate and temporally update MP, propagates MP to the front-end for dynamic-aware correspondence selection, and suppresses dynamic artifacts in the back-end via MP-guided optimization. Experiments on public dynamic RGB-D benchmarks show improved reconstruction and robust tracking while sustaining real-time throughput on a commodity GPU, demonstrating a practical speed-accuracy tradeoff with reduced semantic invocations toward mobile deployment.

  </details>



- **Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context**  
  JiaKui Hu, Jialun Liu, Liying Yang, Xinliang Zhang, Kaiwen Li, Shuang Zeng, Yuanwei Li, Haibin Huang, Chi Zhang, Yanye Lu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21929v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Scene-consistent video generation aims to create videos that explore 3D scenes based on a camera trajectory. Previous methods rely on video generation models with external memory for consistency, or iterative 3D reconstruction and inpainting, which accumulate errors during inference due to incorrect intermediary outputs, non-differentiable processes, and separate models. To overcome these limitations, we introduce ``geometry-as-context". It iteratively completes the following steps using an autoregressive camera-controlled video generation model: (1) estimates the geometry of the current view necessary for 3D reconstruction, and (2) simulates and restores novel view images rendered by the 3D scene. Under this multi-task framework, we develop the camera gated attention module to enhance the model's capability to effectively leverage camera poses. During the training phase, text contexts are utilized to ascertain whether geometric or RGB images should be generated. To ensure that the model can generate RGB-only outputs during inference, the geometry context is randomly dropped from the interleaved text-image-geometry training sequence. The method has been tested on scene video generation with one-direction and forth-and-back trajectories. The results show its superiority over previous approaches in maintaining scene consistency and camera control.

  </details>



- **EndoDDC: Learning Sparse to Dense Reconstruction for Endoscopic Robotic Navigation via Diffusion Depth Completion**  
  Yinheng Lin, Yiming Huang, Beilei Cui, Long Bai, Huxin Gao, Hongliang Ren, Jiewen Lai  
  _2026-02-25_ · https://arxiv.org/abs/2602.21893v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate depth estimation plays a critical role in the navigation of endoscopic surgical robots, forming the foundation for 3D reconstruction and safe instrument guidance. Fine-tuning pretrained models heavily relies on endoscopic surgical datasets with precise depth annotations. While existing self-supervised depth estimation techniques eliminate the need for accurate depth annotations, their performance degrades in environments with weak textures and variable lighting, leading to sparse reconstruction with invalid depth estimation. Depth completion using sparse depth maps can mitigate these issues and improve accuracy. Despite the advances in depth completion techniques in general fields, their application in endoscopy remains limited. To overcome these limitations, we propose EndoDDC, an endoscopy depth completion method that integrates images, sparse depth information with depth gradient features, and optimizes depth maps through a diffusion model, addressing the issues of weak texture and light reflection in endoscopic environments. Extensive experiments on two publicly available endoscopy datasets show that our approach outperforms state-of-the-art models in both depth accuracy and robustness. This demonstrates the potential of our method to reduce visual errors in complex endoscopic environments. Our code will be released at https://github.com/yinheng-lin/EndoDDC.

  </details>


