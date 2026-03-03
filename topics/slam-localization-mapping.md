# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-03-03 07:06 UTC_

Total papers shown: **10**


---

- **Real-Time Thermal-Inertial Odometry on Embedded Hardware for High-Speed GPS-Denied Flight**  
  Austin Stone, Mark Petersen, Cammy Peterson  
  _2026-03-02_ · https://arxiv.org/abs/2603.02114v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a real-time monocular thermal-inertial odometry system designed for high-velocity, GPS-denied flight on embedded hardware. The system fuses measurements from a FLIR Boson+ 640 longwave infrared camera, a high-rate IMU, a laser range finder, a barometer, and a magnetometer within a fixed-lag factor graph. To sustain reliable feature tracks under motion blur, low contrast, and rapid viewpoint changes, we employ a lightweight thermal-optimized front-end with multi-stage feature filtering. Laser range finder measurements provide per-feature depth priors that stabilize scale during weakly observable motion. High-rate inertial data is first pre-filtered using a Chebyshev Type II infinite impulse response (IIR) filter and then preintegrated, improving robustness to airframe vibrations during aggressive maneuvers. To address barometric altitude errors induced at high airspeeds, we train an uncertainty-aware gated recurrent unit (GRU) network that models the temporal dynamics of static pressure distortion, outperforming polynomial and multi-layer perceptron (MLP) baselines. Integrated on an NVIDIA Jetson Xavier NX, the complete system supports closed-loop quadrotor flight at 30 m/s with drift under 2% over kilometer-scale trajectories. These contributions expand the operational envelope of thermal-inertial navigation, enabling reliable high-speed flight in visually degraded and GPS-denied environments.

  </details>



- **BAWSeg: A UAV Multispectral Benchmark for Barley Weed Segmentation**  
  Haitian Wang, Xinyu Wang, Muhammad Ibrahim, Dustin Severtson, Ajmal Mian  
  _2026-03-02_ · https://arxiv.org/abs/2603.01932v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate weed mapping in cereal fields requires pixel-level segmentation from UAV imagery that remains reliable across fields, seasons, and illumination. Existing multispectral pipelines often depend on thresholded vegetation indices, which are brittle under radiometric drift and mixed crop--weed pixels, or on single-stream CNN and Transformer backbones that ingest stacked bands and indices, where radiance cues and normalized index cues interfere and reduce sensitivity to small weed clusters embedded in crop canopies. We propose VISA (Vegetation-Index and Spectral Attention), a two-stream segmentation network that decouples these cues and fuses them at native resolution. The radiance stream learns from calibrated five-band reflectance using residual spectral-spatial attention to preserve fine textures and row boundaries that are attenuated by ratio indices. The index stream operates on vegetation-index maps with windowed self-attention to model local structure efficiently, state-space layers to propagate field-scale context without quadratic attention cost, and Slot Attention to form stable region descriptors that improve discrimination of sparse weeds under canopy mixing. To support supervised training and deployment-oriented evaluation, we introduce BAWSeg, a four-year UAV multispectral dataset collected over commercial barley paddocks in Western Australia, providing radiometrically calibrated blue, green, red, red edge, and near-infrared orthomosaics, derived vegetation indices, and dense crop, weed, and other labels with leakage-free block splits. On BAWSeg, VISA achieves 75.6% mIoU and 63.5% weed IoU with 22.8M parameters, outperforming a multispectral SegFormer-B1 baseline by 1.2 mIoU and 1.9 weed IoU. Under cross-plot and cross-year protocols, VISA maintains 71.2% and 69.2% mIoU, respectively. The BAWSeg data, VISA code, and trained models will be released upon publication.

  </details>



- **LEAR: Learning Edge-Aware Representations for Event-to-LiDAR Localization**  
  Kuangyi Chen, Jun Zhang, Yuxi Hu, Yi Zhou, Friedrich Fraundorfer  
  _2026-03-02_ · https://arxiv.org/abs/2603.01839v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Event cameras offer high-temporal-resolution sensing that remains reliable under high-speed motion and challenging lighting, making them promising for localization from LiDAR point clouds in GPS-denied and visually degraded environments. However, aligning sparse, asynchronous events with dense LiDAR maps is fundamentally ill-posed, as direct correspondence estimation suffers from modality gaps. We propose LEAR, a dual-task learning framework that jointly estimates edge structures and dense event-depth flow fields to bridge the sensing-modality divide. Instead of treating edges as a post-hoc aid, LEAR couples them with flow estimation through a cross-modal fusion mechanism that injects modality-invariant geometric cues into the motion representation, and an iterative refinement strategy that enforces mutual consistency between the two tasks over multiple update steps. This synergy produces edge-aware, depth-aligned flow fields that enable more robust and accurate pose recovery via Perspective-n-Point (PnP) solvers. On several popular and challenging datasets, LEAR achieves superior performance over the best prior method. The source code, trained models, and demo videos are made publicly available online.

  </details>



- **Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation**  
  Han Xue, Nan Min, Xiaotong Liu, Wendi Chen, Yuan Fang, Jun Lv, Cewu Lu, Chuan Wen  
  _2026-03-02_ · https://arxiv.org/abs/2603.02139v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The adoption of fisheye cameras in robotic manipulation, driven by their exceptionally wide Field of View (FoV), is rapidly outpacing a systematic understanding of their downstream effects on policy learning. This paper presents the first comprehensive empirical study to bridge this gap, rigorously analyzing the properties of wrist-mounted fisheye cameras for imitation learning. Through extensive experiments in both simulation and the real world, we investigate three critical research questions: spatial localization, scene generalization, and hardware generalization. Our investigation reveals that: (1) The wide FoV significantly enhances spatial localization, but this benefit is critically contingent on the visual complexity of the environment. (2) Fisheye-trained policies, while prone to overfitting in simple scenes, unlock superior scene generalization when trained with sufficient environmental diversity. (3) While naive cross-camera transfer leads to failures, we identify the root cause as scale overfitting and demonstrate that hardware generalization performance can be improved with a simple Random Scale Augmentation (RSA) strategy. Collectively, our findings provide concrete, actionable guidance for the large-scale collection and effective use of fisheye datasets in robotic learning. More results and videos are available on https://robo-fisheye.github.io/

  </details>



- **Learning to Read Where to Look: Disease-Aware Vision-Language Pretraining for 3D CT**  
  Simon Ging, Philipp Arnold, Sebastian Walter, Hani Alnahas, Hannah Bast, Elmar Kotter, Jiancheng Yang, Behzad Bozorgtabar, Thomas Brox  
  _2026-03-02_ · https://arxiv.org/abs/2603.02026v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent 3D CT vision-language models align volumes with reports via contrastive pretraining, but typically rely on limited public data and provide only coarse global supervision. We train a 3D CT vision-language model on 98k report-volume pairs (50k patients) collected at a single hospital, combined with public datasets, using SigLIP-style contrastive pretraining together with prompt-based disease supervision in the shared vision-text embedding space. On CT-RATE, our model achieves state-of-the-art text-to-image retrieval (R@10 31.5 vs. 22.2) and competitive disease classification (AUC 83.8 vs. 83.8), with consistent results on Rad-ChestCT (AUC 77.0 vs. 77.3). We further observe that radiologists routinely reference specific images within their reports (e.g., ``series X, image Y''), linking textual descriptions to precise axial locations. We automatically mine 262k such snippet-slice pairs and introduce the task of intra-scan snippet localization -- predicting the axial depth referred to by a text snippet -- reducing mean absolute error to 36.3 mm at 12 mm feature resolution, compared with 67.0 mm for the best baseline. Adding this localization objective leaves retrieval and classification broadly unchanged within confidence bounds, yielding a single unified model for retrieval, classification, and intra-scan grounding.

  </details>



- **Zero-shot Low-Field MRI Enhancement via Diffusion-Based Adaptive Contrast Transport**  
  Muyu Liu, Chenhe Du, Xuanyu Tian, Qing Wu, Xiao Wang, Haonan Zhang, Hongjiang Wei, Yuyao Zhang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01913v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Low-field (LF) magnetic resonance imaging (MRI) democratizes access to diagnostic imaging but is fundamentally limited by low signal-to-noise ratio and significant tissue contrast distortion due to field-dependent relaxation dynamics. Reconstructing high-field (HF) quality images from LF data is a blind inverse problem, severely challenged by the scarcity of paired training data and the unknown, non-linear contrast transformation operator. Existing zero-shot methods, which assume simplified linear degradation, often fail to recover authentic tissue contrast. In this paper, we propose DACT(Diffusion-Based Adaptive Contrast Transport), a novel zero-shot framework that restores HF-quality images without paired supervision. DACT synergizes a pre-trained HF diffusion prior to ensure anatomical fidelity with a physically-informed adaptive forward model. Specifically, we introduce a differentiable Sinkhorn optimal transport module that explicitly models and corrects the intensity distribution shift between LF and HF domains during the reverse diffusion process. This allows the framework to dynamically learn the intractable contrast mapping while preserving topological consistency. Extensive experiments on simulated and real clinical LF datasets demonstrate that DACT achieves state-of-the-art performance, yielding reconstructions with superior structural detail and correct tissue contrast.

  </details>



- **Agentic Code Reasoning**  
  Shubham Ugare, Satish Chandra  
  _2026-03-02_ · https://arxiv.org/abs/2603.01896v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Can LLM agents explore codebases and reason about code semantics without executing the code? We study this capability, which we call agentic code reasoning, and introduce semi-formal reasoning: a structured prompting methodology that requires agents to construct explicit premises, trace execution paths, and derive formal conclusions. Unlike unstructured chain-of-thought, semi-formal reasoning acts as a certificate: the agent cannot skip cases or make unsupported claims. We evaluate across three tasks (patch equivalence verification, fault localization, and code question answering) and show that semi-formal reasoning consistently improves accuracy on all of them. For patch equivalence, accuracy improves from 78% to 88% on curated examples and reaches 93% on real-world agent-generated patches, approaching the reliability needed for execution-free RL reward signals. For code question answering on RubberDuckBench Mohammad et al. (2026), semi-formal reasoning achieves 87% accuracy. For fault localization on Defects4J Just et al. (2014), semi-formal reasoning improves Top-5 accuracy by 5 percentage points over standard reasoning. These results demonstrate that structured agentic reasoning enables meaningful semantic code analysis without execution, opening practical applications in RL training pipelines, code review, and static program analysis.

  </details>



- **Generative Visual Chain-of-Thought for Image Editing**  
  Zijin Yin, Tiankai Hang, Yiji Cheng, Shiyi Zhang, Runze He, Yu Xu, Chunyu Wang, Bing Li, Zheng Chang, Kongming Liang, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01893v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Existing image editing methods struggle to perceive where to edit, especially under complex scenes and nuanced spatial instructions. To address this issue, we propose Generative Visual Chain-of-Thought (GVCoT), a unified framework that performs native visual reasoning by first generating spatial cues to localize the target region and then executing the edit. Unlike prior text-only CoT or tool-dependent visual CoT paradigms, GVCoT jointly optimizes visual tokens generated during the reasoning and editing phases in an end-to-end manner. This way fosters the emergence of innate spatial reasoning ability and enables more effective utilization of visual-domain cues. The main challenge of training GCVoT lies in the scarcity of large-scale editing data with precise edit region annotations; to this end, we construct GVCoT-Edit-Instruct, a dataset of 1.8M high-quality samples spanning 19 tasks. We adopt a progressive training strategy: supervised fine-tuning to build foundational localization ability in reasoning trace before final editing, followed by reinforcement learning to further improve reasoning and editing quality. Finally, we introduce SREdit-Bench, a new benchmark designed to comprehensively stress-test models under sophisticated scenes and fine-grained referring expressions. Experiments demonstrate that GVCoT consistently outperforms state-of-the-art models on SREdit-Bench and ImgEdit. We hope our GVCoT will inspire future research toward interpretable and precise image editing.

  </details>



- **Diagnosing Generalization Failures from Representational Geometry Markers**  
  Chi-Ning Chou, Artem Kirsanov, Yao-Yuan Yang, SueYeon Chung  
  _2026-03-02_ · https://arxiv.org/abs/2603.01879v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Generalization, the ability to perform well beyond the training context, is a hallmark of biological and artificial intelligence, yet anticipating unseen failures remains a central challenge. Conventional approaches often take a ``bottom-up'' mechanistic route by reverse-engineering interpretable features or circuits to build explanatory models. While insightful, these methods often struggle to provide the high-level, predictive signals for anticipating failure in real-world deployment. Here, we propose using a ``top-down'' approach to studying generalization failures inspired by medical biomarkers: identifying system-level measurements that serve as robust indicators of a model's future performance. Rather than mapping out detailed internal mechanisms, we systematically design and test network markers to probe structure, function links, identify prognostic indicators, and validate predictions in real-world settings. In image classification, we find that task-relevant geometric properties of in-distribution (ID) object manifolds consistently forecast poor out-of-distribution (OOD) generalization. In particular, reductions in two geometric measures, effective manifold dimensionality and utility, predict weaker OOD performance across diverse architectures, optimizers, and datasets. We apply this finding to transfer learning with ImageNet-pretrained models. We consistently find that the same geometric patterns predict OOD transfer performance more reliably than ID accuracy. This work demonstrates that representational geometry can expose hidden vulnerabilities, offering more robust guidance for model selection and AI interpretability.

  </details>



- **Neural Operator-Grounded Continuous Tensor Function Representation and Its Applications**  
  Ruoyang Su, Xi-Le Zhao, Sheng Liu, Wei-Hao Wu, Yisi Luo, Michael K. Ng  
  _2026-03-02_ · https://arxiv.org/abs/2603.01812v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recently, continuous tensor functions have attracted increasing attention, because they can unifiedly represent data both on mesh grids and beyond mesh grids. However, since mode-$n$ product is essentially discrete and linear, the potential of current continuous tensor function representations is still locked. To break this bottleneck, we suggest neural operator-grounded mode-$n$ operators as a continuous and nonlinear alternative of discrete and linear mode-$n$ product. Instead of mapping the discrete core tensor to the discrete target tensor, proposed mode-$n$ operator directly maps the continuous core tensor function to the continuous target tensor function, which provides a genuine continuous representation of real-world data and can ameliorate discretization artifacts. Empowering with continuous and nonlinear mode-$n$ operators, we propose a neural operator-grounded continuous tensor function representation (abbreviated as NO-CTR), which can more faithfully represent complex real-world data compared with classic discrete tensor representations and continuous tensor function representations. Theoretically, we also prove that any continuous tensor function can be approximated by NO-CTR. To examine the capability of NO-CTR, we suggest an NO-CTR-based multi-dimensional data completion model. Extensive experiments across various data on regular mesh grids (multi-spectral images and color videos), on mesh girds with different resolutions (Sentinel-2 images) and beyond mesh grids (point clouds) demonstrate the superiority of NO-CTR.

  </details>


