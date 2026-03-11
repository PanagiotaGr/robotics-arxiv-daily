# 3D Gaussian Splatting & Neural Rendering (Robotics)

_Robotics arXiv Daily_

_Updated: 2026-03-11 07:08 UTC_

Total papers shown: **4**


---

- **VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM**  
  Anh Thuan Tran, Jana Kosecka  
  _2026-03-10_ · https://arxiv.org/abs/2603.09673v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties. To this end, we introduce VarSplat, an uncertainty-aware 3DGS-SLAM system that explicitly learns per-splat appearance variance. By using the law of total variance with alpha compositing, we then render differentiable per-pixel uncertainty map via efficient, single-pass rasterization. This map guides tracking, submap registration, and loop detection toward focusing on reliable regions and contributes to more stable optimization. Experimental results on Replica (synthetic) and TUM-RGBD, ScanNet, and ScanNet++ (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing studies for dense RGB-D SLAM.

  </details>



- **X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models**  
  Yueen Ma, Irwin King  
  _2026-03-10_ · https://arxiv.org/abs/2603.09632v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods are isolated, focusing on specific domains such as online SLAM, semantic enrichment, or 3DGS for unposed images. In this paper, we introduce X-GS, an extensible open framework that unifies a broad range of techniques to enable real-time 3DGS-based online SLAM enriched with semantics, bridging the gap to downstream multimodal models. At the core of X-GS is a highly efficient pipeline called X-GS-Perceiver, capable of taking unposed RGB (or optionally RGB-D) video streams as input to co-optimize geometry and poses, and distill high-dimensional semantic features from vision foundation models into the 3D Gaussians. We achieve real-time performance through a novel online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The semantic 3D Gaussians can then be utilized by vision-language models within the X-GS-Thinker component, enabling downstream tasks such as object detection, zero-shot caption generation, and potentially embodied tasks. Experimental results on real-world datasets showcase the efficacy, efficiency, and newly unlocked multimodal capabilities of the X-GS framework.

  </details>



- **World2Mind: Cognition Toolkit for Allocentric Spatial Reasoning in Foundation Models**  
  Shouwei Ruan, Bin Wang, Zhenyu Wu, Qihui Zhu, Yuxiang Zhang, Hang Su, Yubin Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09774v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Achieving robust spatial reasoning remains a fundamental challenge for current Multimodal Foundation Models (MFMs). Existing methods either overfit statistical shortcuts via 3D grounding data or remain confined to 2D visual perception, limiting both spatial reasoning accuracy and generalization in unseen scenarios. Inspired by the spatial cognitive mapping mechanisms of biological intelligence, we propose World2Mind, a training-free spatial intelligence toolkit. At its core, World2Mind leverages 3D reconstruction and instance segmentation models to construct structured spatial cognitive maps, empowering MFMs to proactively acquire targeted spatial knowledge regarding interested landmarks and routes of interest. To provide robust geometric-topological priors, World2Mind synthesizes an Allocentric-Spatial Tree (AST) that uses elliptical parameters to model the top-down layout of landmarks accurately. To mitigate the inherent inaccuracies of 3D reconstruction, we introduce a three-stage reasoning chain comprising tool invocation assessment, modality-decoupled cue collection, and geometry-semantics interwoven reasoning. Extensive experiments demonstrate that World2Mind boosts the performance of frontier models, such as GPT-5.2, by 5%~18%. Astonishingly, relying solely on the AST-structured text, purely text-only foundation models can perform complex 3D spatial reasoning, achieving performance approaching that of advanced multimodal models.

  </details>



- **ConfCtrl: Enabling Precise Camera Control in Video Diffusion via Confidence-Aware Interpolation**  
  Liudi Yang, George Eskandar, Fengyi Shen, Mohammad Altillawi, Yang Bai, Chi Zhang, Ziyuan Liu, Abhinav Valada  
  _2026-03-10_ · https://arxiv.org/abs/2603.09819v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We address the challenge of novel view synthesis from only two input images under large viewpoint changes. Existing regression-based methods lack the capacity to reconstruct unseen regions, while camera-guided diffusion models often deviate from intended trajectories due to noisy point cloud projections or insufficient conditioning from camera poses. To address these issues, we propose ConfCtrl, a confidence-aware video interpolation framework that enables diffusion models to follow prescribed camera poses while completing unseen regions. ConfCtrl initializes the diffusion process by combining a confidence-weighted projected point cloud latent with noise as the conditioning input. It then applies a Kalman-inspired predict-update mechanism, treating the projected point cloud as a noisy measurement and using learned residual corrections to balance pose-driven predictions with noisy geometric observations. This allows the model to rely on reliable projections while down-weighting uncertain regions, yielding stable, geometry-aware generation. Experiments on multiple datasets show that ConfCtrl produces geometrically consistent and visually plausible novel views, effectively reconstructing occluded regions under large viewpoint changes.

  </details>


