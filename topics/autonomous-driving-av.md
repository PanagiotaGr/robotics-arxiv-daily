# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-07 06:57 UTC_

Total papers shown: **5**


---

- **LSA: Localized Semantic Alignment for Enhancing Temporal Consistency in Traffic Video Generation**  
  Mirlan Karimov, Teodora Spasojevic, Markus Braun, Julian Wiederer, Vasileios Belagiannis, Marc Pollefeys  
  _2026-02-05_ · https://arxiv.org/abs/2602.05966v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Controllable video generation has emerged as a versatile tool for autonomous driving, enabling realistic synthesis of traffic scenarios. However, existing methods depend on control signals at inference time to guide the generative model towards temporally consistent generation of dynamic objects, limiting their utility as scalable and generalizable data engines. In this work, we propose Localized Semantic Alignment (LSA), a simple yet effective framework for fine-tuning pre-trained video generation models. LSA enhances temporal consistency by aligning semantic features between ground-truth and generated video clips. Specifically, we compare the output of an off-the-shelf feature extraction model between the ground-truth and generated video clips localized around dynamic objects inducing a semantic feature consistency loss. We fine-tune the base model by combining this loss with the standard diffusion loss. The model fine-tuned for a single epoch with our novel loss outperforms the baselines in common video generation evaluation metrics. To further test the temporal consistency in generated videos we adapt two additional metrics from object detection task, namely mAP and mIoU. Extensive experiments on nuScenes and KITTI datasets show the effectiveness of our approach in enhancing temporal consistency in video generation without the need for external control signals during inference and any computational overheads.

  </details>



- **Unified Sensor Simulation for Autonomous Driving**  
  Nikolay Patakin, Arsenii Shirokov, Anton Konushin, Dmitry Senushkin  
  _2026-02-05_ · https://arxiv.org/abs/2602.05617v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In this work, we introduce \textbf{XSIM}, a sensor simulation framework for autonomous driving. XSIM extends 3DGUT splatting with a generalized rolling-shutter modeling tailored for autonomous driving applications. Our framework provides a unified and flexible formulation for appearance and geometric sensor modeling, enabling rendering of complex sensor distortions in dynamic environments. We identify spherical cameras, such as LiDARs, as a critical edge case for existing 3DGUT splatting due to cyclic projection and time discontinuities at azimuth boundaries leading to incorrect particle projection. To address this issue, we propose a phase modeling mechanism that explicitly accounts temporal and shape discontinuities of Gaussians projected by the Unscented Transform at azimuth borders. In addition, we introduce an extended 3D Gaussian representation that incorporates two distinct opacity parameters to resolve mismatches between geometry and color distributions. As a result, our framework provides enhanced scene representations with improved geometric consistency and photorealistic appearance. We evaluate our framework extensively on multiple autonomous driving datasets, including Waymo Open Dataset, Argoverse 2, and PandaSet. Our framework consistently outperforms strong recent baselines and achieves state-of-the-art performance across all datasets. The source code is publicly available at \href{https://github.com/whesense/XSIM}{https://github.com/whesense/XSIM}.

  </details>



- **Thinking with Geometry: Active Geometry Integration for Spatial Reasoning**  
  Haoyuan Li, Qihang Cao, Tao Tang, Kun Xiang, Zihan Guo, Jianhua Han, Hang Xu, Xiaodan Liang  
  _2026-02-05_ · https://arxiv.org/abs/2602.06037v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent progress in spatial reasoning with Multimodal Large Language Models (MLLMs) increasingly leverages geometric priors from 3D encoders. However, most existing integration strategies remain passive: geometry is exposed as a global stream and fused in an indiscriminate manner, which often induces semantic-geometry misalignment and redundant signals. We propose GeoThinker, a framework that shifts the paradigm from passive fusion to active perception. Instead of feature mixing, GeoThinker enables the model to selectively retrieve geometric evidence conditioned on its internal reasoning demands. GeoThinker achieves this through Spatial-Grounded Fusion applied at carefully selected VLM layers, where semantic visual priors selectively query and integrate task-relevant geometry via frame-strict cross-attention, further calibrated by Importance Gating that biases per-frame attention toward task-relevant structures. Comprehensive evaluation results show that GeoThinker sets a new state-of-the-art in spatial intelligence, achieving a peak score of 72.6 on the VSI-Bench. Furthermore, GeoThinker demonstrates robust generalization and significantly improved spatial perception across complex downstream scenarios, including embodied referring and autonomous driving. Our results indicate that the ability to actively integrate spatial structures is essential for next-generation spatial intelligence. Code can be found at https://github.com/Li-Hao-yuan/GeoThinker.

  </details>



- **Depth as Prior Knowledge for Object Detection**  
  Moussa Kassem Sbeyti, Nadja Klein  
  _2026-02-05_ · https://arxiv.org/abs/2602.05730v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Detecting small and distant objects remains challenging for object detectors due to scale variation, low resolution, and background clutter. Safety-critical applications require reliable detection of these objects for safe planning. Depth information can improve detection, but existing approaches require complex, model-specific architectural modifications. We provide a theoretical analysis followed by an empirical investigation of the depth-detection relationship. Together, they explain how depth causes systematic performance degradation and why depth-informed supervision mitigates it. We introduce DepthPrior, a framework that uses depth as prior knowledge rather than as a fused feature, providing comparable benefits without modifying detector architectures. DepthPrior consists of Depth-Based Loss Weighting (DLW) and Depth-Based Loss Stratification (DLS) during training, and Depth-Aware Confidence Thresholding (DCT) during inference. The only overhead is the initial cost of depth estimation. Experiments across four benchmarks (KITTI, MS COCO, VisDrone, SUN RGB-D) and two detectors (YOLOv11, EfficientDet) demonstrate the effectiveness of DepthPrior, achieving up to +9% mAP$_S$ and +7% mAR$_S$ for small objects, with inference recovery rates as high as 95:1 (true vs. false detections). DepthPrior offers these benefits without additional sensors, architectural changes, or performance costs. Code is available at https://github.com/mos-ks/DepthPrior.

  </details>



- **ROMAN: Reward-Orchestrated Multi-Head Attention Network for Autonomous Driving System Testing**  
  Jianlei Chi, Yuzhen Wu, Jiaxuan Hou, Xiaodong Zhang, Ming Fan, Suhui Sun, Weijun Dai, Bo Li, Jianguo Sun, Jun Sun  
  _2026-02-05_ · https://arxiv.org/abs/2602.05629v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Automated Driving System (ADS) acts as the brain of autonomous vehicles, responsible for their safety and efficiency. Safe deployment requires thorough testing in diverse real-world scenarios and compliance with traffic laws like speed limits, signal obedience, and right-of-way rules. Violations like running red lights or speeding pose severe safety risks. However, current testing approaches face significant challenges: limited ability to generate complex and high-risk law-breaking scenarios, and failing to account for complex interactions involving multiple vehicles and critical situations. To address these challenges, we propose ROMAN, a novel scenario generation approach for ADS testing that combines a multi-head attention network with a traffic law weighting mechanism. ROMAN is designed to generate high-risk violation scenarios to enable more thorough and targeted ADS evaluation. The multi-head attention mechanism models interactions among vehicles, traffic signals, and other factors. The traffic law weighting mechanism implements a workflow that leverages an LLM-based risk weighting module to evaluate violations based on the two dimensions of severity and occurrence. We have evaluated ROMAN by testing the Baidu Apollo ADS within the CARLA simulation platform and conducting extensive experiments to measure its performance. Experimental results demonstrate that ROMAN surpassed state-of-the-art tools ABLE and LawBreaker by achieving 7.91% higher average violation count than ABLE and 55.96% higher than LawBreaker, while also maintaining greater scenario diversity. In addition, only ROMAN successfully generated violation scenarios for every clause of the input traffic laws, enabling it to identify more high-risk violations than existing approaches.

  </details>


