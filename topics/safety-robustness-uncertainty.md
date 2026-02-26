# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-26 07:13 UTC_

Total papers shown: **14**


---

- **Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos**  
  Matthew Strong, Wei-Jer Chang, Quentin Herau, Jiezhi Yang, Yihan Hu, Chensheng Peng, Wei Zhan  
  _2026-02-25_ · https://arxiv.org/abs/2602.22091v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Ego-centric driving videos available online provide an abundant source of visual data for autonomous driving, yet their lack of annotations makes it difficult to learn representations that capture both semantic structure and 3D geometry. Recent advances in large feedforward spatial models demonstrate that point maps and ego-motion can be inferred in a single forward pass, suggesting a promising direction for scalable driving perception. We therefore propose a label-free, teacher-guided framework for learning autonomous driving representations directly from unposed videos. Unlike prior self-supervised approaches that focus primarily on frame-to-frame consistency, we posit that safe and reactive driving depends critically on temporal context. To this end, we leverage a feedforward architecture equipped with a lightweight autoregressive module, trained using multi-modal supervisory signals that guide the model to jointly predict current and future point maps, camera poses, semantic segmentation, and motion masks. Multi-modal teachers provide sequence-level pseudo-supervision, enabling LFG to learn a unified pseudo-4D representation from raw YouTube videos without poses, labels, or LiDAR. The resulting encoder not only transfers effectively to downstream autonomous driving planning on the NAVSIM benchmark, surpassing multi-camera and LiDAR baselines with only a single monocular camera, but also yields strong performance when evaluated on a range of semantic, geometric, and qualitative motion prediction tasks. These geometry and motion-aware features position LFG as a compelling video-centric foundation model for autonomous driving.

  </details>



- **Parallel Continuous-Time Relative Localization with Augmented Clamped Non-Uniform B-Splines**  
  Jiadong Lu, Zhehan Li, Tao Han, Miao Xu, Chao Xu, Yanjun Cao  
  _2026-02-25_ · https://arxiv.org/abs/2602.22006v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate relative localization is critical for multi-robot cooperation. In robot swarms, measurements from different robots arrive asynchronously and with clock time-offsets. Although Continuous-Time (CT) formulations have proved effective for handling asynchronous measurements in single-robot SLAM and calibration, extending CT methods to multi-robot settings faces great challenges to achieve high-accuracy, low-latency, and high-frequency performance. Especially, existing CT methods suffer from the inherent query-time delay of unclamped B-splines and high computational cost. This paper proposes CT-RIO, a novel Continuous-Time Relative-Inertial Odometry framework. We employ Clamped Non-Uniform B-splines (C-NUBS) to represent robot states for the first time, eliminating the query-time delay. We further augment C-NUBS with closed-form extension and shrinkage operations that preserve the spline shape, making it suitable for online estimation and enabling flexible knot management. This flexibility leads to the concept of knot-keyknot strategy, which supports spline extension at high-frequency while retaining sparse keyknots for adaptive relative-motion modeling. We then formulate a sliding-window relative localization problem that operates purely on relative kinematics and inter-robot constraints. To meet the demanding computation required at swarm scale, we decompose the tightly-coupled optimization into robot-wise sub-problems and solve them in parallel using incremental asynchronous block coordinate descent. Extensive experiments show that CT-RIO converges from time-offsets as large as 263 ms to sub-millisecond within 3 s, and achieves RMSEs of 0.046 m and 1.8 °. It consistently outperforms state-of-the-art methods, with improvements of up to 60% under high-speed motion.

  </details>



- **Compact Circulant Layers with Spectral Priors**  
  Joseph Margaryan, Thomas Hamelryck  
  _2026-02-25_ · https://arxiv.org/abs/2602.21965v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Critical applications in areas such as medicine, robotics and autonomous systems require compact (i.e., memory efficient), uncertainty-aware neural networks suitable for edge and other resource-constrained deployments. We study compact spectral circulant and block-circulant-with-circulant-blocks (BCCB) layers: FFT-diagonalizable circular convolutions whose weights live directly in the real FFT (RFFT) half (1D) or half-plane (2D). Parameterizing filters in the frequency domain lets us impose simple spectral structure, perform structured variational inference in a low-dimensional weight space, and calculate exact layer spectral norms, enabling inexpensive global Lipschitz bounds and margin-based robustness diagnostics. By placing independent complex Gaussians on the Hermitian support we obtain a discrete instance of the spectral representation of stationary kernels, inducing an exact stationary Gaussian-process prior over filters on the discrete circle/torus. We exploit this to define a practical spectral prior and a Hermitian-aware low-rank-plus-diagonal variational posterior in real coordinates. Empirically, spectral circulant/BCCB layers are effective compact building blocks in both (variational) Bayesian and point estimate regimes: compact Bayesian neural networks on MNIST->Fashion-MNIST, variational heads on frozen CIFAR-10 features, and deterministic ViT projections on CIFAR-10/Tiny ImageNet; spectral layers match strong baselines while using substantially fewer parameters and with tighter Lipschitz certificates.

  </details>



- **LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration**  
  Aditya Ranjan Dash, Ramy Battrawy, René Schuster, Didier Stricker  
  _2026-02-25_ · https://arxiv.org/abs/2602.21754v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Advanced autonomous systems rely on multi-sensor fusion for safer and more robust perception. To enable effective fusion, calibrating directly from natural driving scenes (i.e., target-free) with high accuracy is crucial for precise multi-sensor alignment. Existing learning-based calibration methods are typically designed for only a single pair of sensor modalities (i.e., a bi-modal setup). Unlike these methods, we propose LiREC-Net, a target-free, learning-based calibration network that jointly calibrates multiple sensor modality pairs, including LiDAR, RGB, and event data, within a unified framework. To reduce redundant computation and improve efficiency, we introduce a shared LiDAR representation that leverages features from both its 3D nature and projected depth map, ensuring better consistency across modalities. Trained and evaluated on established datasets, such as KITTI and DSEC, our LiREC-Net achieves competitive performance to bi-modal models and sets a new strong baseline for the tri-modal use case.

  </details>



- **DAGS-SLAM: Dynamic-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty-Aware Scheduling**  
  Li Zhang, Yu-An Liu, Xijia Jiang, Conghao Huang, Danyang Li, Yanyong Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21644v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mobile robots and IoT devices demand real-time localization and dense reconstruction under tight compute and energy budgets. While 3D Gaussian Splatting (3DGS) enables efficient dense SLAM, dynamic objects and occlusions still degrade tracking and mapping. Existing dynamic 3DGS-SLAM often relies on heavy optical flow and per-frame segmentation, which is costly for mobile deployment and brittle under challenging illumination. We present DAGS-SLAM, a dynamic-aware 3DGS-SLAM system that maintains a spatiotemporal motion probability (MP) state per Gaussian and triggers semantics on demand via an uncertainty-aware scheduler. DAGS-SLAM fuses lightweight YOLO instance priors with geometric cues to estimate and temporally update MP, propagates MP to the front-end for dynamic-aware correspondence selection, and suppresses dynamic artifacts in the back-end via MP-guided optimization. Experiments on public dynamic RGB-D benchmarks show improved reconstruction and robust tracking while sustaining real-time throughput on a commodity GPU, demonstrating a practical speed-accuracy tradeoff with reduced semantic invocations toward mobile deployment.

  </details>



- **EndoDDC: Learning Sparse to Dense Reconstruction for Endoscopic Robotic Navigation via Diffusion Depth Completion**  
  Yinheng Lin, Yiming Huang, Beilei Cui, Long Bai, Huxin Gao, Hongliang Ren, Jiewen Lai  
  _2026-02-25_ · https://arxiv.org/abs/2602.21893v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate depth estimation plays a critical role in the navigation of endoscopic surgical robots, forming the foundation for 3D reconstruction and safe instrument guidance. Fine-tuning pretrained models heavily relies on endoscopic surgical datasets with precise depth annotations. While existing self-supervised depth estimation techniques eliminate the need for accurate depth annotations, their performance degrades in environments with weak textures and variable lighting, leading to sparse reconstruction with invalid depth estimation. Depth completion using sparse depth maps can mitigate these issues and improve accuracy. Despite the advances in depth completion techniques in general fields, their application in endoscopy remains limited. To overcome these limitations, we propose EndoDDC, an endoscopy depth completion method that integrates images, sparse depth information with depth gradient features, and optimizes depth maps through a diffusion model, addressing the issues of weak texture and light reflection in endoscopic environments. Extensive experiments on two publicly available endoscopy datasets show that our approach outperforms state-of-the-art models in both depth accuracy and robustness. This demonstrates the potential of our method to reduce visual errors in complex endoscopic environments. Our code will be released at https://github.com/yinheng-lin/EndoDDC.

  </details>



- **Force Policy: Learning Hybrid Force-Position Control Policy under Interaction Frame for Contact-Rich Manipulation**  
  Hongjie Fang, Shirun Tang, Mingyu Mei, Haoxiang Qin, Zihao He, Jingjing Chen, Ying Feng, Chenxi Wang, Wanxi Liu, Zaixing He, et al.  
  _2026-02-25_ · https://arxiv.org/abs/2602.22088v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Contact-rich manipulation demands human-like integration of perception and force feedback: vision should guide task progress, while high-frequency interaction control must stabilize contact under uncertainty. Existing learning-based policies often entangle these roles in a monolithic network, trading off global generalization against stable local refinement, while control-centric approaches typically assume a known task structure or learn only controller parameters rather than the structure itself. In this paper, we formalize a physically grounded interaction frame, an instantaneous local basis that decouples force regulation from motion execution, and propose a method to recover it from demonstrations. Based on this, we address both issues by proposing Force Policy, a global-local vision-force policy in which a global policy guides free-space actions using vision, and upon contact, a high-frequency local policy with force feedback estimates the interaction frame and executes hybrid force-position control for stable interaction. Real-world experiments across diverse contact-rich tasks show consistent gains over strong baselines, with more robust contact establishment, more accurate force regulation, and reliable generalization to novel objects with varied geometries and physical properties, ultimately improving both contact stability and execution quality. Project page: https://force-policy.github.io/

  </details>



- **ADM-DP: Adaptive Dynamic Modality Diffusion Policy through Vision-Tactile-Graph Fusion for Multi-Agent Manipulation**  
  Enyi Wang, Wen Fan, Dandan Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21622v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-agent robotic manipulation remains challenging due to the combined demands of coordination, grasp stability, and collision avoidance in shared workspaces. To address these challenges, we propose the Adaptive Dynamic Modality Diffusion Policy (ADM-DP), a framework that integrates vision, tactile, and graph-based (multi-agent pose) modalities for coordinated control. ADM-DP introduces four key innovations. First, an enhanced visual encoder merges RGB and point-cloud features via Feature-wise Linear Modulation (FiLM) modulation to enrich perception. Second, a tactile-guided grasping strategy uses Force-Sensitive Resistor (FSR) feedback to detect insufficient contact and trigger corrective grasp refinement, improving grasp stability. Third, a graph-based collision encoder leverages shared tool center point (TCP) positions of multiple agents as structured kinematic context to maintain spatial awareness and reduce inter-agent interference. Fourth, an Adaptive Modality Attention Mechanism (AMAM) dynamically re-weights modalities according to task context, enabling flexible fusion. For scalability and modularity, a decoupled training paradigm is employed in which agents learn independent policies while sharing spatial information. This maintains low interdependence between agents while retaining collective awareness. Across seven multi-agent tasks, ADM-DP achieves 12-25% performance gains over state-of-the-art baselines. Ablation studies show the greatest improvements in tasks requiring multiple sensory modalities, validating our adaptive fusion strategy and demonstrating its robustness for diverse manipulation scenarios.

  </details>



- **WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs**  
  Yulin Zhang, Cheng Shi, Sibei Yang  
  _2026-02-25_ · https://arxiv.org/abs/2602.22142v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advances in Multimodal Large Language Models have greatly improved visual understanding and reasoning, yet their quadratic attention and offline training protocols make them ill-suited for streaming settings where frames arrive sequentially and future observations are inaccessible. We diagnose a core limitation of current Video-LLMs, namely Time-Agnosticism, in which videos are treated as an unordered bag of evidence rather than a causally ordered sequence, yielding two failures in streams: temporal order ambiguity, in which the model cannot follow or reason over the correct chronological order, and past-current focus blindness where it fails to distinguish present observations from accumulated history. We present WeaveTime, a simple, efficient, and model agnostic framework that first teaches order and then uses order. We introduce a lightweight Temporal Reconstruction objective-our Streaming Order Perception enhancement-that instills order aware representations with minimal finetuning and no specialized streaming data. At inference, a Past-Current Dynamic Focus Cache performs uncertainty triggered, coarse-to-fine retrieval, expanding history only when needed. Plugged into exsiting Video-LLM without architectural changes, WeaveTime delivers consistent gains on representative streaming benchmarks, improving accuracy while reducing latency. These results establish WeaveTime as a practical path toward time aware stream Video-LLMs under strict online, time causal constraints. Code and weights will be made publicly available. Project Page: https://zhangyl4.github.io/publications/weavetime/

  </details>



- **Learning Quantum Data Distribution via Chaotic Quantum Diffusion Model**  
  Quoc Hoan Tran, Koki Chinzei, Yasuhiro Endo, Hirotaka Oshima  
  _2026-02-25_ · https://arxiv.org/abs/2602.22061v1 · `quant-ph`  
  <details><summary>Abstract</summary>

  Generative models for quantum data pose significant challenges but hold immense potential in fields such as chemoinformatics and quantum physics. Quantum denoising diffusion probabilistic models (QuDDPMs) enable efficient learning of quantum data distributions by progressively scrambling and denoising quantum states; however, existing implementations typically rely on circuit-based random unitary dynamics that can be costly to realize and sensitive to control imperfections, particularly on analog quantum hardware. We propose the chaotic quantum diffusion model, a framework that generates projected ensembles via chaotic Hamiltonian time evolution, providing a flexible and hardware-compatible diffusion mechanism. Requiring only global, time-independent control, our approach substantially reduces implementation overhead across diverse analog quantum platforms while achieving accuracy comparable to QuDDPMs. This method improves trainability and robustness, broadening the applicability of quantum generative modeling.

  </details>



- **RT-RMOT: A Dataset and Framework for RGB-Thermal Referring Multi-Object Tracking**  
  Yanqiu Yu, Zhifan Jin, Sijia Chen, Tongfei Chu, En Yu, Liman Liu, Wenbing Tao  
  _2026-02-25_ · https://arxiv.org/abs/2602.22033v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Referring Multi-Object Tracking has attracted increasing attention due to its human-friendly interactive characteristics, yet it exhibits limitations in low-visibility conditions, such as nighttime, smoke, and other challenging scenarios. To overcome this limitation, we propose a new RGB-Thermal RMOT task, named RT-RMOT, which aims to fuse RGB appearance features with the illumination robustness of the thermal modality to enable all-day referring multi-object tracking. To promote research on RT-RMOT, we construct the first Referring Multi-Object Tracking dataset under RGB-Thermal modality, named RefRT. It contains 388 language descriptions, 1,250 tracked targets, and 166,147 Language-RGB-Thermal (L-RGB-T) triplets. Furthermore, we propose RTrack, a framework built upon a multimodal large language model (MLLM) that integrates RGB, thermal, and textual features. Since the initial framework still leaves room for improvement, we introduce a Group Sequence Policy Optimization (GSPO) strategy to further exploit the model's potential. To alleviate training instability during RL fine-tuning, we introduce a Clipped Advantage Scaling (CAS) strategy to suppress gradient explosion. In addition, we design Structured Output Reward and Comprehensive Detection Reward to balance exploration and exploitation, thereby improving the completeness and accuracy of target perception. Extensive experiments on the RefRT dataset demonstrate the effectiveness of the proposed RTrack framework.

  </details>



- **PatchDenoiser: Parameter-efficient multi-scale patch learning and fusion denoiser for medical images**  
  Jitindra Fartiyal, Pedro Freire, Sergei K. Turitsyn, Sergei G. Solovski  
  _2026-02-25_ · https://arxiv.org/abs/2602.21987v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Medical images are essential for diagnosis, treatment planning, and research, but their quality is often degraded by noise from low-dose acquisition, patient motion, or scanner limitations, affecting both clinical interpretation and downstream analysis. Traditional filtering approaches often over-smooth and lose fine anatomical details, while deep learning methods, including CNNs, GANs, and transformers, may struggle to preserve such details or require large, computationally expensive models, limiting clinical practicality. We propose PatchDenoiser, a lightweight, energy-efficient multi-scale patch-based denoising framework. It decomposes denoising into local texture extraction and global context aggregation, fused via a spatially aware patch fusion strategy. This design enables effective noise suppression while preserving fine structural and anatomical details. PatchDenoiser is ultra-lightweight, with far fewer parameters and lower computational complexity than CNN-, GAN-, and transformer-based denoisers. On the 2016 Mayo Low-Dose CT dataset, PatchDenoiser consistently outperforms state-of-the-art CNN- and GAN-based methods in PSNR and SSIM. It is robust to variations in slice thickness, reconstruction kernels, and HU windows, generalizes across scanners without fine-tuning, and reduces parameters by ~9x and energy consumption per inference by ~27x compared with conventional CNN denoisers. PatchDenoiser thus provides a practical, scalable, and computationally efficient solution for medical image denoising, balancing performance, robustness, and clinical deployability.

  </details>



- **TIRAuxCloud: A Thermal Infrared Dataset for Day and Night Cloud Detection**  
  Alexis Apostolakis, Vasileios Botsos, Niklas Wölki, Andrea Spichtinger, Nikolaos Ioannis Bountos, Ioannis Papoutsis, Panayiotis Tsanakas  
  _2026-02-25_ · https://arxiv.org/abs/2602.21905v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Clouds are a major obstacle in Earth observation, limiting the usability and reliability of critical remote sensing applications such as fire disaster response, urban heat island monitoring, and snow and ice cover mapping. Therefore, the ability to detect clouds 24/7 is of paramount importance. While visible and near-infrared bands are effective for daytime cloud detection, their dependence on solar illumination makes them unsuitable for nighttime monitoring. In contrast, thermal infrared (TIR) imagery plays a crucial role in detecting clouds at night, when sunlight is absent. Due to their generally lower temperatures, clouds emit distinct thermal signatures that are detectable in TIR bands. Despite this, accurate nighttime cloud detection remains challenging due to limited spectral information and the typically lower spatial resolution of TIR imagery. To address these challenges, we present TIRAuxCloud, a multi-modal dataset centered around thermal spectral data to facilitate cloud segmentation under both daytime and nighttime conditions. The dataset comprises a unique combination of multispectral data (TIR, optical, and near-infrared bands) from Landsat and VIIRS, aligned with auxiliary information layers. Elevation, land cover, meteorological variables, and cloud-free reference images are included to help reduce surface-cloud ambiguity and cloud formation uncertainty. To overcome the scarcity of manual cloud labels, we include a large set of samples with automated cloud masks and a smaller manually annotated subset to further evaluate and improve models. Comprehensive benchmarks are presented to establish performance baselines through supervised and transfer learning, demonstrating the dataset's value in advancing the development of innovative methods for day and night time cloud detection.

  </details>



- **Send Less, Perceive More: Masked Quantized Point Cloud Communication for Loss-Tolerant Collaborative Perception**  
  Sheng Xu, Enshu Wang, Hongfei Xue, Jian Teng, Bingyi Liu, Yi Zhu, Pu Wang, Libing Wu, Chunming Qiao  
  _2026-02-25_ · https://arxiv.org/abs/2602.21667v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Collaborative perception allows connected vehicles to overcome occlusions and limited viewpoints by sharing sensory information. However, existing approaches struggle to achieve high accuracy under strict bandwidth constraints and remain highly vulnerable to random transmission packet loss. We introduce QPoint2Comm, a quantized point-cloud communication framework that dramatically reduces bandwidth while preserving high-fidelity 3D information. Instead of transmitting intermediate features, QPoint2Comm directly communicates quantized point-cloud indices using a shared codebook, enabling efficient reconstruction with lower bandwidth than feature-based methods. To ensure robustness to possible communication packet loss, we employ a masked training strategy that simulates random packet loss, allowing the model to maintain strong performance even under severe transmission failures. In addition, a cascade attention fusion module is proposed to enhance multi-vehicle information integration. Extensive experiments on both simulated and real-world datasets demonstrate that QPoint2Comm sets a new state of the art in accuracy, communication efficiency, and resilience to packet loss.

  </details>


