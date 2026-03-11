# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-03-11 07:08 UTC_

Total papers shown: **12**


---

- **VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM**  
  Anh Thuan Tran, Jana Kosecka  
  _2026-03-10_ · https://arxiv.org/abs/2603.09673v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties. To this end, we introduce VarSplat, an uncertainty-aware 3DGS-SLAM system that explicitly learns per-splat appearance variance. By using the law of total variance with alpha compositing, we then render differentiable per-pixel uncertainty map via efficient, single-pass rasterization. This map guides tracking, submap registration, and loop detection toward focusing on reliable regions and contributes to more stable optimization. Experimental results on Replica (synthetic) and TUM-RGBD, ScanNet, and ScanNet++ (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing studies for dense RGB-D SLAM.

  </details>



- **Robust Cooperative Localization in Featureless Environments: A Comparative Study of DCL, StCL, CCL, CI, and Standard-CL**  
  Nivand Khosravi, Meysam Basiri, Rodrigo Ventura  
  _2026-03-10_ · https://arxiv.org/abs/2603.09886v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Cooperative localization (CL) enables accurate position estimation in multi-robot systems operating in GPS-denied environments. This paper presents a comparative study of five CL approaches: Centralized Cooperative Localization (CCL), Decentralized Cooperative Localization (DCL), Sequential Cooperative Localization (StCL), Covariance Intersection (CI), and Standard Cooperative Localization (Standard-CL). All methods are implemented in ROS and evaluated through Monte Carlo simulations under two conditions: weak data association and robust detection. Our analysis reveals fundamental trade-offs among the methods. StCL and Standard-CL achieve the lowest position errors but exhibit severe filter inconsistency, making them unsuitable for safety-critical applications. DCL demonstrates remarkable stability under challenging conditions due to its measurement stride mechanism, which provides implicit regularization against outliers. CI emerges as the most balanced approach, achieving near-optimal consistency while maintaining competitive accuracy. CCL provides theoretically optimal estimation but shows sensitivity to measurement outliers. These findings offer practical guidance for selecting CL algorithms based on application requirements.

  </details>



- **RA-SSU: Towards Fine-Grained Audio-Visual Learning with Region-Aware Sound Source Understanding**  
  Muyi Sun, Yixuan Wang, Hong Wang, Chen Su, Man Zhang, Xingqun Qi, Qi Li, Zhenan Sun  
  _2026-03-10_ · https://arxiv.org/abs/2603.09809v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Audio-Visual Learning (AVL) is one fundamental task of multi-modality learning and embodied intelligence, displaying the vital role in scene understanding and interaction. However, previous researchers mostly focus on exploring downstream tasks from a coarse-grained perspective (e.g., audio-visual correspondence, sound source localization, and audio-visual event localization). Considering providing more specific scene perception details, we newly define a fine-grained Audio-Visual Learning task, termed Region-Aware Sound Source Understanding (RA-SSU), which aims to achieve region-aware, frame-level, and high-quality sound source understanding. To support this goal, we innovatively construct two corresponding datasets, i.e. fine-grained Music (f-Music) and fine-grained Lifescene (f-Lifescene), each containing annotated sound source masks and frame-by-frame textual descriptions. The f-Music dataset includes 3,976 samples across 22 scene types related to specific application scenarios, focusing on music scenes with complex instrument mixing. The f-Lifescene dataset contains 6,156 samples across 61 types representing diverse sounding objects in life scenarios. Moreover, we propose SSUFormer, a Sound-Source Understanding TransFormer benchmark that facilitates both the sound source segmentation and sound region description with a multi-modal input and multi-modal output architecture. Specifically, we design two modules for this framework, Mask Collaboration Module (MCM) and Mixture of Hierarchical-prompted Experts (MoHE), to respectively enhance the accuracy and enrich the elaboration of the sound source description. Extensive experiments are conducted on our two datasets to verify the feasibility of the task, evaluate the availability of the datasets, and demonstrate the superiority of the SSUFormer, which achieves SOTA performance on the Sound Source Understanding benchmark.

  </details>



- **World2Mind: Cognition Toolkit for Allocentric Spatial Reasoning in Foundation Models**  
  Shouwei Ruan, Bin Wang, Zhenyu Wu, Qihui Zhu, Yuxiang Zhang, Hang Su, Yubin Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09774v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Achieving robust spatial reasoning remains a fundamental challenge for current Multimodal Foundation Models (MFMs). Existing methods either overfit statistical shortcuts via 3D grounding data or remain confined to 2D visual perception, limiting both spatial reasoning accuracy and generalization in unseen scenarios. Inspired by the spatial cognitive mapping mechanisms of biological intelligence, we propose World2Mind, a training-free spatial intelligence toolkit. At its core, World2Mind leverages 3D reconstruction and instance segmentation models to construct structured spatial cognitive maps, empowering MFMs to proactively acquire targeted spatial knowledge regarding interested landmarks and routes of interest. To provide robust geometric-topological priors, World2Mind synthesizes an Allocentric-Spatial Tree (AST) that uses elliptical parameters to model the top-down layout of landmarks accurately. To mitigate the inherent inaccuracies of 3D reconstruction, we introduce a three-stage reasoning chain comprising tool invocation assessment, modality-decoupled cue collection, and geometry-semantics interwoven reasoning. Extensive experiments demonstrate that World2Mind boosts the performance of frontier models, such as GPT-5.2, by 5%~18%. Astonishingly, relying solely on the AST-structured text, purely text-only foundation models can perform complex 3D spatial reasoning, achieving performance approaching that of advanced multimodal models.

  </details>



- **OTPL-VIO: Robust Visual-Inertial Odometry with Optimal Transport Line Association and Adaptive Uncertainty**  
  Zikun Chen, Wentao Zhao, Yihe Niu, Tianchen Deng, Jingchuan Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09653v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Robust stereo visual-inertial odometry (VIO) remains challenging in low-texture scenes and under abrupt illumination changes, where point features become sparse and unstable, leading to ambiguous association and under-constrained estimation. Line structures offer complementary geometric cues, yet many efficient point-line systems still rely on point-guided line association, which can break down when point support is weak and may lead to biased constraints. We present a stereo point-line VIO system in which line segments are equipped with dedicated deep descriptors and matched using an entropy-regularized optimal transport formulation, enabling globally consistent correspondences under ambiguity, outliers, and partial observations. The proposed descriptor is training-free and is computed by sampling and pooling network feature maps. To improve estimation stability, we analyze the impact of line measurement noise and introduce reliability-adaptive weighting to regulate the influence of line constraints during optimization. Experiments on EuRoC and UMA-VI, together with real-world deployments in low-texture and illumination-challenging environments, demonstrate improved accuracy and robustness over representative baselines while maintaining real-time performance.

  </details>



- **SurgFed: Language-guided Multi-Task Federated Learning for Surgical Video Understanding**  
  Zheng Fang, Ziwei Niu, Ziyue Wang, Zhu Zhuo, Haofeng Liu, Shuyang Qian, Jun Xia, Yueming Jin  
  _2026-03-10_ · https://arxiv.org/abs/2603.09496v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Surgical scene Multi-Task Federated Learning (MTFL) is essential for robot-assisted minimally invasive surgery (RAS) but remains underexplored in surgical video understanding due to two key challenges: (1) Tissue Diversity: Local models struggle to adapt to site-specific tissue features, limiting their effectiveness in heterogeneous clinical environments and leading to poor local predictions. (2) Task Diversity: Server-side aggregation, relying solely on gradient-based clustering, often produces suboptimal or incorrect parameter updates due to inter-site task heterogeneity, resulting in inaccurate localization. In light of these two issues, we propose SurgFed, a multi-task federated learning framework, enabling federated learning for surgical scene segmentation and depth estimation across diverse surgical types. SurgFed is powered by two appealing designs, i.e., Language-guided Channel Selection (LCS) and Language-guided Hyper Aggregation (LHA), to address the challenge of fully exploration on corss-site and cross-task. Technically, the LCS is first designed a lightweight personalized channel selection network that enhances site-specific adaptation using pre-defined text inputs, which optimally the local model learn the specific embeddings. We further introduce the LHA that employs a layer-wise cross-attention mechanism with pre-defined text inputs to model task interactions across sites and guide a hypernetwork for personalized parameter updates. Extensive empirical evidence shows that SurgFed yields improvements over the state-of-the-art methods in five public datasets across four surgical types. The code is available at https://anonymous.4open.science/r/SurgFed-070E/.

  </details>



- **Stepping VLMs onto the Court: Benchmarking Spatial Intelligence in Sports**  
  Yuchen Yang, Yuqing Shao, Duxiu Huang, Linfeng Dong, Yifei Liu, Suixin Tang, Xiang Zhou, Yuanyuan Gao, Wei Wang, Yue Zhou, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09896v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Sports have long attracted broad attention as they push the limits of human physical and cognitive capabilities. Amid growing interest in spatial intelligence for vision-language models (VLMs), sports provide a natural testbed for understanding high-intensity human motion and dynamic object interactions. To this end, we present CourtSI, the first large-scale spatial intelligence dataset tailored to sports scenarios. CourtSI contains over 1M QA pairs, organized under a holistic taxonomy that systematically covers spatial counting, distance measurement, localization, and relational reasoning, across representative net sports including badminton, tennis, and table tennis. Leveraging well-defined court geometry as metric anchors, we develop a semi-automatic data engine to reconstruct sports scenes, enabling scalable curation of CourtSI. In addition, we introduce CourtSI-Bench, a high-quality evaluation benchmark comprising 3,686 QA pairs with rigorous human verification. We evaluate 25 proprietary and open-source VLMs on CourtSI-Bench, revealing a remaining human-AI performance gap and limited generalization from existing spatial intelligence benchmarks. These findings indicate that sports scenarios expose limitations in spatial intelligence capabilities captured by existing benchmarks. Further, fine-tuning Qwen3-VL-8B on CourtSI improves accuracy on CourtSI-Bench by 23.5 percentage points. The adapted model also generalizes effectively to CourtSI-Ext, an evaluation set built on a similar but unseen sport, and demonstrates enhanced spatial-aware commentary generation. Together, these findings demonstrate that CourtSI provides a scalable pathway toward advancing spatial intelligence of VLMs in sports.

  </details>



- **SCENEBench: An Audio Understanding Benchmark Grounded in Assistive and Industrial Use Cases**  
  Laya Iyer, Angelina Wang, Sanmi Koyejo  
  _2026-03-10_ · https://arxiv.org/abs/2603.09853v1 · `cs.SD`  
  <details><summary>Abstract</summary>

  Advances in large language models (LLMs) have enabled significant capabilities in audio processing, resulting in state-of-the-art models now known as Large Audio Language Models (LALMs). However, minimal work has been done to measure audio understanding beyond automatic speech recognition (ASR). This paper closes that gap by proposing a benchmark suite, SCENEBench (Spatial, Cross-lingual, Environmental, Non-speech Evaluation), that targets a broad form of audio comprehension across four real-world categories: background sound understanding, noise localization, cross-linguistic speech understanding, and vocal characterizer recognition. These four categories are selected based on understudied needs from accessibility technology and industrial noise monitoring. In addition to performance, we also measure model latency. The purpose of this benchmark suite is to assess audio beyond just what words are said - rather, how they are said and the non-speech components of the audio. Because our audio samples are synthetically constructed (e.g., by overlaying two natural audio samples), we further validate our benchmark against 20 natural audio items per task, sub-sampled from existing datasets to match our task criteria, to assess ecological validity. We assess five state-of-the-art LALMs and find critical gaps: performance varies across tasks, with some tasks performing below random chance and others achieving high accuracy. These results provide direction for targeted improvements in model capabilities.

  </details>



- **VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models**  
  Shuhao Kang, Youqi Liao, Peijie Wang, Wenlong Liao, Qilin Zhang, Benjamin Busam, Xieyuanli Chen, Yun Liu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09826v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Text-to-point-cloud (T2P) localization aims to infer precise spatial positions within 3D point cloud maps from natural language descriptions, reflecting how humans perceive and communicate spatial layouts through language. However, existing methods largely rely on shallow text-point cloud correspondence without effective spatial reasoning, limiting their accuracy in complex environments. To address this limitation, we propose VLM-Loc, a framework that leverages the spatial reasoning capability of large vision-language models (VLMs) for T2P localization. Specifically, we transform point clouds into bird's-eye-view (BEV) images and scene graphs that jointly encode geometric and semantic context, providing structured inputs for the VLM to learn cross-modal representations bridging linguistic and spatial semantics. On top of these representations, we introduce a partial node assignment mechanism that explicitly associates textual cues with scene graph nodes, enabling interpretable spatial reasoning for accurate localization. To facilitate systematic evaluation across diverse scenes, we present CityLoc, a benchmark built from multi-source point clouds for fine-grained T2P localization. Experiments on CityLoc demonstrate VLM-Loc achieves superior accuracy and robustness compared to state-of-the-art methods. Our code, model, and dataset are available at \href{https://github.com/MCG-NKU/nku-3d-vision}{repository}.

  </details>



- **X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models**  
  Yueen Ma, Irwin King  
  _2026-03-10_ · https://arxiv.org/abs/2603.09632v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods are isolated, focusing on specific domains such as online SLAM, semantic enrichment, or 3DGS for unposed images. In this paper, we introduce X-GS, an extensible open framework that unifies a broad range of techniques to enable real-time 3DGS-based online SLAM enriched with semantics, bridging the gap to downstream multimodal models. At the core of X-GS is a highly efficient pipeline called X-GS-Perceiver, capable of taking unposed RGB (or optionally RGB-D) video streams as input to co-optimize geometry and poses, and distill high-dimensional semantic features from vision foundation models into the 3D Gaussians. We achieve real-time performance through a novel online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The semantic 3D Gaussians can then be utilized by vision-language models within the X-GS-Thinker component, enabling downstream tasks such as object detection, zero-shot caption generation, and potentially embodied tasks. Experimental results on real-world datasets showcase the efficacy, efficiency, and newly unlocked multimodal capabilities of the X-GS framework.

  </details>



- **Decoder-Free Distillation for Quantized Image Restoration**  
  S. M. A. Sharif, Abdur Rehman, Seongwan Kim, Jaeho Lee  
  _2026-03-10_ · https://arxiv.org/abs/2603.09624v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Quantization-Aware Training (QAT), combined with Knowledge Distillation (KD), holds immense promise for compressing models for edge deployment. However, joint optimization for precision-sensitive image restoration (IR) to recover visual quality from degraded images remains largely underexplored. Directly adapting QAT-KD to low-level vision reveals three critical bottlenecks: teacher-student capacity mismatch, spatial error amplification during decoder distillation, and an optimization "tug-of-war" between reconstruction and distillation losses caused by quantization noise. To tackle these, we introduce Quantization-aware Distilled Restoration (QDR), a framework for edge-deployed IR. QDR eliminates capacity mismatch via FP32 self-distillation and prevents error amplification through Decoder-Free Distillation (DFD), which corrects quantization errors strictly at the network bottleneck. To stabilize the optimization tug-of-war, we propose a Learnable Magnitude Reweighting (LMR) that dynamically balances competing gradients. Finally, we design an Edge-Friendly Model (EFM) featuring a lightweight Learnable Degradation Gating (LDG) to dynamically modulate spatial degradation localization. Extensive experiments across four IR tasks demonstrate that our Int8 model recovers 96.5% of FP32 performance, achieves 442 frames per second (FPS) on an NVIDIA Jetson Orin, and boosts downstream object detection by 16.3 mAP

  </details>



- **Benchmarking Dataset for Presence-Only Passive Reconnaissance in Wireless Smart-Grid Communications**  
  Bochra Al Agha, Razane Tajeddine  
  _2026-03-10_ · https://arxiv.org/abs/2603.09590v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Benchmarking presence-only passive reconnaissance in smart-grid communications is challenging because the adversary is receive-only, yet nearby observers can still alter propagation through additional shadowing and multipath that reshapes channel coherence. Public smart-grid cybersecurity datasets largely target active protocol- or measurement-layer attacks and rarely provide propagation-driven observables with tiered topology context, which limits reproducible evaluation under strictly passive threat models. This paper introduces an IEEE-inspired, literature-anchored benchmark dataset generator for passive reconnaissance over a tiered Home Area Network (HAN), Neighborhood Area Network (NAN), and Wide Area Network (WAN) communication graph with heterogeneous wireless and wireline links. Node-level time series are produced through a physically consistent channel-to-metrics mapping where channel state information (CSI) is represented via measurement-realistic amplitude and phase proxies that drive inferred signal-to-noise ratio (SNR), packet error behavior, and delay dynamics. Passive attacks are modeled only as windowed excess attenuation and coherence degradation with increased channel innovation, so reliability and latency deviations emerge through the same causal mapping without labels or feature shortcuts. The release provides split-independent realizations with burn-in removal, strictly causal temporal descriptors, adjacency-weighted neighbor aggregates and deviation features, and federated-ready per-node train, validation, and test partitions with train-only normalization metadata. Baseline federated experiments highlight technology-dependent detectability and enable standardized benchmarking of graph-temporal and federated detectors for passive reconnaissance.

  </details>


