# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-01-28 06:52 UTC_

Total papers shown: **4**


---

- **Instance-Guided Radar Depth Estimation for 3D Object Detection**  
  Chen-Chou Lo, Patrick Vandewalle  
  _2026-01-27_ · https://arxiv.org/abs/2601.19314v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate depth estimation is fundamental to 3D perception in autonomous driving, supporting tasks such as detection, tracking, and motion planning. However, monocular camera-based 3D detection suffers from depth ambiguity and reduced robustness under challenging conditions. Radar provides complementary advantages such as resilience to poor lighting and adverse weather, but its sparsity and low resolution limit its direct use in detection frameworks. This motivates the need for effective Radar-camera fusion with improved preprocessing and depth estimation strategies. We propose an end-to-end framework that enhances monocular 3D object detection through two key components. First, we introduce InstaRadar, an instance segmentation-guided expansion method that leverages pre-trained segmentation masks to enhance Radar density and semantic alignment, producing a more structured representation. InstaRadar achieves state-of-the-art results in Radar-guided depth estimation, showing its effectiveness in generating high-quality depth features. Second, we integrate the pre-trained RCDPT into the BEVDepth framework as a replacement for its depth module. With InstaRadar-enhanced inputs, the RCDPT integration consistently improves 3D detection performance. Overall, these components yield steady gains over the baseline BEVDepth model, demonstrating the effectiveness of InstaRadar and the advantage of explicit depth supervision in 3D object detection. Although the framework lags behind Radar-camera fusion models that directly extract BEV features, since Radar serves only as guidance rather than an independent feature stream, this limitation highlights potential for improvement. Future work will extend InstaRadar to point cloud-like representations and integrate a dedicated Radar branch with temporal cues for enhanced BEV fusion.

  </details>



- **ScenePilot-Bench: A Large-Scale Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving**  
  Yujin Wang, Yutong Zheng, Wenxian Fan, Tianyi Wang, Hongqing Chu, Daxin Tian, Bingzhao Gao, Jianqiang Wang, Hong Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19582v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In this paper, we introduce ScenePilot-Bench, a large-scale first-person driving benchmark designed to evaluate vision-language models (VLMs) in autonomous driving scenarios. ScenePilot-Bench is built upon ScenePilot-4K, a diverse dataset comprising 3,847 hours of driving videos, annotated with multi-granularity information including scene descriptions, risk assessments, key participant identification, ego trajectories, and camera parameters. The benchmark features a four-axis evaluation suite that assesses VLM capabilities in scene understanding, spatial perception, motion planning, and GPT-Score, with safety-aware metrics and cross-region generalization settings. We benchmark representative VLMs on ScenePilot-Bench, providing empirical analyses that clarify current performance boundaries and identify gaps for driving-oriented reasoning. ScenePilot-Bench offers a comprehensive framework for evaluating and advancing VLMs in safety-critical autonomous driving contexts.

  </details>



- **Towards Gold-Standard Depth Estimation for Tree Branches in UAV Forestry: Benchmarking Deep Stereo Matching Methods**  
  Yida Lin, Bing Xue, Mengjie Zhang, Sam Schofield, Richard Green  
  _2026-01-27_ · https://arxiv.org/abs/2601.19461v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous UAV forestry operations require robust depth estimation with strong cross-domain generalization, yet existing evaluations focus on urban and indoor scenarios, leaving a critical gap for vegetation-dense environments. We present the first systematic zero-shot evaluation of eight stereo methods spanning iterative refinement, foundation model, diffusion-based, and 3D CNN paradigms. All methods use officially released pretrained weights (trained on Scene Flow) and are evaluated on four standard benchmarks (ETH3D, KITTI 2012/2015, Middlebury) plus a novel 5,313-pair Canterbury Tree Branches dataset ($1920 \times 1080$). Results reveal scene-dependent patterns: foundation models excel on structured scenes (BridgeDepth: 0.23 px on ETH3D; DEFOM: 4.65 px on Middlebury), while iterative methods show variable cross-benchmark performance (IGEV++: 0.36 px on ETH3D but 6.77 px on Middlebury; IGEV: 0.33 px on ETH3D but 4.99 px on Middlebury). Qualitative evaluation on the Tree Branches dataset establishes DEFOM as the gold-standard baseline for vegetation depth estimation, with superior cross-domain consistency (consistently ranking 1st-2nd across benchmarks, average rank 1.75). DEFOM predictions will serve as pseudo-ground-truth for future benchmarking.

  </details>



- **Generative Latent Alignment for Interpretable Radar Based Occupancy Detection in Ambient Assisted Living**  
  Huy Trinh  
  _2026-01-27_ · https://arxiv.org/abs/2601.19853v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  In this work, we study how to make mmWave radar presence detection more interpretable for Ambient Assisted Living (AAL) settings, where camera-based sensing raises privacy concerns. We propose a Generative Latent Alignment (GLA) framework that combines a lightweight convolutional variational autoencoder with a frozen CLIP text encoder to learn a low-dimensional latent representation of radar Range-Angle (RA) heatmaps. The latent space is softly aligned with two semantic anchors corresponding to "empty room" and "person present", and Grad-CAM is applied in this aligned latent space to visualize which spatial regions support each presence decision. On our mmWave radar dataset, we qualitatively observe that the "person present" class produces compact Grad-CAM blobs that coincide with strong RA returns, whereas "empty room" samples yield diffuse or no evidence. We also conduct an ablation study using unrelated text prompts, which degrades both reconstruction and localization, suggesting that radar-specific anchors are important for meaningful explanations in this setting.

  </details>


