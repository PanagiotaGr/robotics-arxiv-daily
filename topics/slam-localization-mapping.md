# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-24 07:13 UTC_

Total papers shown: **12**


---

- **Chasing Ghosts: A Simulation-to-Real Olfactory Navigation Stack with Optional Vision Augmentation**  
  Kordel K. France, Ovidiu Daescu, Latifur Khan, Rohith Peddi  
  _2026-02-23_ · https://arxiv.org/abs/2602.19577v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous odor source localization remains a challenging problem for aerial robots due to turbulent airflow, sparse and delayed sensory signals, and strict payload and compute constraints. While prior unmanned aerial vehicle (UAV)-based olfaction systems have demonstrated gas distribution mapping or reactive plume tracing, they rely on predefined coverage patterns, external infrastructure, or extensive sensing and coordination. In this work, we present a complete, open-source UAV system for online odor source localization using a minimal sensor suite. The system integrates custom olfaction hardware, onboard sensing, and a learning-based navigation policy trained in simulation and deployed on a real quadrotor. Through our minimal framework, the UAV is able to navigate directly toward an odor source without constructing an explicit gas distribution map or relying on external positioning systems. Vision is incorporated as an optional complementary modality to accelerate navigation under certain conditions. We validate the proposed system through real-world flight experiments in a large indoor environment using an ethanol source, demonstrating consistent source-finding behavior under realistic airflow conditions. The primary contribution of this work is a reproducible system and methodological framework for UAV-based olfactory navigation and source finding under minimal sensing assumptions. We elaborate on our hardware design and open source our UAV firmware, simulation code, olfaction-vision dataset, and circuit board to the community. Code, data, and designs will be made available at https://github.com/KordelFranceTech/ChasingGhosts.

  </details>



- **To Move or Not to Move: Constraint-based Planning Enables Zero-Shot Generalization for Interactive Navigation**  
  Apoorva Vashisth, Manav Kulshrestha, Pranav Bakshi, Damon Conover, Guillaume Sartoretti, Aniket Bera  
  _2026-02-23_ · https://arxiv.org/abs/2602.20055v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual navigation typically assumes the existence of at least one obstacle-free path between start and goal, which must be discovered/planned by the robot. However, in real-world scenarios, such as home environments and warehouses, clutter can block all routes. Targeted at such cases, we introduce the Lifelong Interactive Navigation problem, where a mobile robot with manipulation abilities can move clutter to forge its own path to complete sequential object- placement tasks - each involving placing an given object (eg. Alarm clock, Pillow) onto a target object (eg. Dining table, Desk, Bed). To address this lifelong setting - where effects of environment changes accumulate and have long-term effects - we propose an LLM-driven, constraint-based planning framework with active perception. Our framework allows the LLM to reason over a structured scene graph of discovered objects and obstacles, deciding which object to move, where to place it, and where to look next to discover task-relevant information. This coupling of reasoning and active perception allows the agent to explore the regions expected to contribute to task completion rather than exhaustively mapping the environment. A standard motion planner then executes the corresponding navigate-pick-place, or detour sequence, ensuring reliable low-level control. Evaluated in physics-enabled ProcTHOR-10k simulator, our approach outperforms non-learning and learning-based baselines. We further demonstrate our approach qualitatively on real-world hardware.

  </details>



- **VGGT-MPR: VGGT-Enhanced Multimodal Place Recognition in Autonomous Driving Environments**  
  Jingyi Xu, Zhangshuo Qi, Zhongmiao Yan, Xuyu Gao, Qianyun Jiao, Songpengcheng Xia, Xieyuanli Chen, Ling Pei  
  _2026-02-23_ · https://arxiv.org/abs/2602.19735v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In autonomous driving, robust place recognition is critical for global localization and loop closure detection. While inter-modality fusion of camera and LiDAR data in multimodal place recognition (MPR) has shown promise in overcoming the limitations of unimodal counterparts, existing MPR methods basically attend to hand-crafted fusion strategies and heavily parameterized backbones that require costly retraining. To address this, we propose VGGT-MPR, a multimodal place recognition framework that adopts the Visual Geometry Grounded Transformer (VGGT) as a unified geometric engine for both global retrieval and re-ranking. In the global retrieval stage, VGGT extracts geometrically-rich visual embeddings through prior depth-aware and point map supervision, and densifies sparse LiDAR point clouds with predicted depth maps to improve structural representation. This enhances the discriminative ability of fused multimodal features and produces global descriptors for fast retrieval. Beyond global retrieval, we design a training-free re-ranking mechanism that exploits VGGT's cross-view keypoint-tracking capability. By combining mask-guided keypoint extraction with confidence-aware correspondence scoring, our proposed re-ranking mechanism effectively refines retrieval results without additional parameter optimization. Extensive experiments on large-scale autonomous driving benchmarks and our self-collected data demonstrate that VGGT-MPR achieves state-of-the-art performance, exhibiting strong robustness to severe environmental changes, viewpoint shifts, and occlusions. Our code and data will be made publicly available.

  </details>



- **TraceVision: Trajectory-Aware Vision-Language Model for Human-Like Spatial Understanding**  
  Fan Yang, Shurong Zheng, Hongyin Zhao, Yufei Zhan, Xin Li, Yousong Zhu, Chaoyang Zhao Ming Tang, Jinqiao Wang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19768v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities in image understanding and natural language generation. However, current approaches focus predominantly on global image understanding, struggling to simulate human visual attention trajectories and explain associations between descriptions and specific regions. We propose TraceVision, a unified vision-language model integrating trajectory-aware spatial understanding in an end-to-end framework. TraceVision employs a Trajectory-aware Visual Perception (TVP) module for bidirectional fusion of visual features and trajectory information. We design geometric simplification to extract semantic keypoints from raw trajectories and propose a three-stage training pipeline where trajectories guide description generation and region localization. We extend TraceVision to trajectory-guided segmentation and video scene understanding, enabling cross-frame tracking and temporal attention analysis. We construct the Reasoning-based Interactive Localized Narratives (RILN) dataset to enhance logical reasoning and interpretability. Extensive experiments on trajectory-guided captioning, text-guided trajectory prediction, understanding, and segmentation demonstrate that TraceVision achieves state-of-the-art performance, establishing a foundation for intuitive spatial interaction and interpretable visual understanding.

  </details>



- **PerturbDiff: Functional Diffusion for Single-Cell Perturbation Modeling**  
  Xinyu Yuan, Xixian Liu, Ya Shi Zhang, Zuobai Zhang, Hongyu Guo, Jian Tang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19685v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Building Virtual Cells that can accurately simulate cellular responses to perturbations is a long-standing goal in systems biology. A fundamental challenge is that high-throughput single-cell sequencing is destructive: the same cell cannot be observed both before and after a perturbation. Thus, perturbation prediction requires mapping unpaired control and perturbed populations. Existing models address this by learning maps between distributions, but typically assume a single fixed response distribution when conditioned on observed cellular context (e.g., cell type) and the perturbation type. In reality, responses vary systematically due to unobservable latent factors such as microenvironmental fluctuations and complex batch effects, forming a manifold of possible distributions for the same observed conditions. To account for this variability, we introduce PerturbDiff, which shifts modeling from individual cells to entire distributions. By embedding distributions as points in a Hilbert space, we define a diffusion-based generative process operating directly over probability distributions. This allows PerturbDiff to capture population-level response shifts across hidden factors. Benchmarks on established datasets show that PerturbDiff achieves state-of-the-art performance in single-cell response prediction and generalizes substantially better to unseen perturbations. See our project page (https://katarinayuan.github.io/PerturbDiff-ProjectPage/), where code and data will be made publicly available (https://github.com/DeepGraphLearning/PerturbDiff).

  </details>



- **Closing the gap in multimodal medical representation alignment**  
  Eleonora Grassucci, Giordano Cicchetti, Danilo Comminiello  
  _2026-02-23_ · https://arxiv.org/abs/2602.20046v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In multimodal learning, CLIP has emerged as the de-facto approach for mapping different modalities into a shared latent space by bringing semantically similar representations closer while pushing apart dissimilar ones. However, CLIP-based contrastive losses exhibit unintended behaviors that negatively impact true semantic alignment, leading to sparse and fragmented latent spaces. This phenomenon, known as the modality gap, has been partially mitigated for standard text and image pairs but remains unknown and unresolved in more complex multimodal settings, such as the medical domain. In this work, we study this phenomenon in the latter case, revealing that the modality gap is present also in medical alignment, and we propose a modality-agnostic framework that closes this gap, ensuring that semantically related representations are more aligned, regardless of their source modality. Our method enhances alignment between radiology images and clinical text, improving cross-modal retrieval and image captioning.

  </details>



- **Discover, Segment, and Select: A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation**  
  Yilong Yang, Jianxin Tian, Shengchuan Zhang, Liujuan Cao  
  _2026-02-23_ · https://arxiv.org/abs/2602.19944v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Current zero-shot Camouflaged Object Segmentation methods typically employ a two-stage pipeline (discover-then-segment): using MLLMs to obtain visual prompts, followed by SAM segmentation. However, relying solely on MLLMs for camouflaged object discovery often leads to inaccurate localization, false positives, and missed detections. To address these issues, we propose the \textbf{D}iscover-\textbf{S}egment-\textbf{S}elect (\textbf{DSS}) mechanism, a progressive framework designed to refine segmentation step by step. The proposed method contains a Feature-coherent Object Discovery (FOD) module that leverages visual features to generate diverse object proposals, a segmentation module that refines these proposals through SAM segmentation, and a Semantic-driven Mask Selection (SMS) module that employs MLLMs to evaluate and select the optimal segmentation mask from multiple candidates. Without requiring any training or supervision, DSS achieves state-of-the-art performance on multiple COS benchmarks, especially in multiple-instance scenes.

  </details>



- **TextShield-R1: Reinforced Reasoning for Tampered Text Detection**  
  Chenfan Qu, Yiwu Zhong, Jian Liu, Xuekang Zhu, Bohan Yu, Lianwen Jin  
  _2026-02-23_ · https://arxiv.org/abs/2602.19828v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The growing prevalence of tampered images poses serious security threats, highlighting the urgent need for reliable detection methods. Multimodal large language models (MLLMs) demonstrate strong potential in analyzing tampered images and generating interpretations. However, they still struggle with identifying micro-level artifacts, exhibit low accuracy in localizing tampered text regions, and heavily rely on expensive annotations for forgery interpretation. To this end, we introduce TextShield-R1, the first reinforcement learning based MLLM solution for tampered text detection and reasoning. Specifically, our approach introduces Forensic Continual Pre-training, an easy-to-hard curriculum that well prepares the MLLM for tampered text detection by harnessing the large-scale cheap data from natural image forensic and OCR tasks. During fine-tuning, we perform Group Relative Policy Optimization with novel reward functions to reduce annotation dependency and improve reasoning capabilities. At inference time, we enhance localization accuracy via OCR Rectification, a method that leverages the MLLM's strong text recognition abilities to refine its predictions. Furthermore, to support rigorous evaluation, we introduce the Text Forensics Reasoning (TFR) benchmark, comprising over 45k real and tampered images across 16 languages, 10 tampering techniques, and diverse domains. Rich reasoning-style annotations are included, allowing for comprehensive assessment. Our TFR benchmark simultaneously addresses seven major limitations of existing benchmarks and enables robust evaluation under cross-style, cross-method, and cross-language conditions. Extensive experiments demonstrate that TextShield-R1 significantly advances the state of the art in interpretable tampered text detection.

  </details>



- **Drift Localization using Conformal Predictions**  
  Fabian Hinder, Valerie Vaquet, Johannes Brinkrolf, Barbara Hammer  
  _2026-02-23_ · https://arxiv.org/abs/2602.19790v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Concept drift -- the change of the distribution over time -- poses significant challenges for learning systems and is of central interest for monitoring. Understanding drift is thus paramount, and drift localization -- determining which samples are affected by the drift -- is essential. While several approaches exist, most rely on local testing schemes, which tend to fail in high-dimensional, low-signal settings. In this work, we consider a fundamentally different approach based on conformal predictions. We discuss and show the shortcomings of common approaches and demonstrate the performance of our approach on state-of-the-art image datasets.

  </details>



- **Active IoT User Detection in Near-Field with Location Information**  
  Gabriel Martins de Jesus, Richard Demo Souza, Onel Luis Alcaraz López  
  _2026-02-23_ · https://arxiv.org/abs/2602.19613v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  In this paper, we address active users detection (AUD) in near-field Internet of Things (IoT) networks by exploring prior knowledge of users' locations. We consider a scenario where users are distributed in a semi-circular area within the Rayleigh distance of a multi-antenna base station (BS). We propose the BS to use location estimates of the users to reconstruct their line-of-sight (LoS) channel components, hence assisting the AUD process. For this, the BS combines these reconstructed channels with users' pilot sequences, enhancing the correlation between received signals and active users. We formulate the location-aided AUD as a convex optimization problem, solved via the alternating direction method of multipliers (ADMM). {Our proposal has a higher computational complexity compared to the baseline ADMM approach where location information is not used. Moreover, the proposal requires location information of users, which can be readily informed if users are static, or inferred via established localization algorithms if they are mobile.} Simulation results compare our proposal against the baseline across varying systems parameters, such as number of users, pilot length and LoS component strength. We demonstrate that under perfect location estimation and strong LoS, our proposed method significantly outperforms the baseline. Furthermore, robustness analysis shows that performance gains persist under imperfect location estimation, provided the estimation error remains within bounds determined by the system parameters.

  </details>



- **RAID: Retrieval-Augmented Anomaly Detection**  
  Mingxiu Cai, Zhe Zhang, Gaochang Wu, Tianyou Chai, Xiatian Zhu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19611v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Unsupervised Anomaly Detection (UAD) aims to identify abnormal regions by establishing correspondences between test images and normal templates. Existing methods primarily rely on image reconstruction or template retrieval but face a fundamental challenge: matching between test images and normal templates inevitably introduces noise due to intra-class variations, imperfect correspondences, and limited templates. Observing that Retrieval-Augmented Generation (RAG) leverages retrieved samples directly in the generation process, we reinterpret UAD through this lens and introduce \textbf{RAID}, a retrieval-augmented UAD framework designed for noise-resilient anomaly detection and localization. Unlike standard RAG that enriches context or knowledge, we focus on using retrieved normal samples to guide noise suppression in anomaly map generation. RAID retrieves class-, semantic-, and instance-level representations from a hierarchical vector database, forming a coarse-to-fine pipeline. A matching cost volume correlates the input with retrieved exemplars, followed by a guided Mixture-of-Experts (MoE) network that leverages the retrieved samples to adaptively suppress matching noise and produce fine-grained anomaly maps. RAID achieves state-of-the-art performance across full-shot, few-shot, and multi-dataset settings on MVTec, VisA, MPDD, and BTAD benchmarks. \href{https://github.com/Mingxiu-Cai/RAID}{https://github.com/Mingxiu-Cai/RAID}.

  </details>



- **CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning**  
  Chunlei Meng, Guanhong Huang, Rong Fu, Runmin Jian, Zhongxue Gan, Chun Ouyang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19605v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Multimodal learning aims to capture both shared and private information from multiple modalities. However, existing methods that project all modalities into a single latent space for fusion often overlook the asynchronous, multi-level semantic structure of multimodal data. This oversight induces semantic misalignment and error propagation, thereby degrading representation quality. To address this issue, we propose Cross-Level Co-Representation (CLCR), which explicitly organizes each modality's features into a three-level semantic hierarchy and specifies level-wise constraints for cross-modal interactions. First, a semantic hierarchy encoder aligns shallow, mid, and deep features across modalities, establishing a common basis for interaction. And then, at each level, an Intra-Level Co-Exchange Domain (IntraCED) factorizes features into shared and private subspaces and restricts cross-modal attention to the shared subspace via a learnable token budget. This design ensures that only shared semantics are exchanged and prevents leakage from private channels. To integrate information across levels, the Inter-Level Co-Aggregation Domain (InterCAD) synchronizes semantic scales using learned anchors, selectively fuses the shared representations, and gates private cues to form a compact task representation. We further introduce regularization terms to enforce separation of shared and private features and to minimize cross-level interference. Experiments on six benchmarks spanning emotion recognition, event localization, sentiment analysis, and action recognition show that CLCR achieves strong performance and generalizes well across tasks.

  </details>


