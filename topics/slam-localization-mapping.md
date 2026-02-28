# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-28 06:54 UTC_

Total papers shown: **10**


---

- **FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time**  
  David Dirnfeld, Fabien Delattre, Pedro Miraldo, Erik Learned-Miller  
  _2026-02-26_ · https://arxiv.org/abs/2602.23115v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Estimating camera motion from monocular video is a fundamental problem in computer vision, central to tasks such as SLAM, visual odometry, and structure-from-motion. Existing methods that recover the camera's heading under known rotation, whether from an IMU or an optimization algorithm, tend to perform well in low-noise, low-outlier conditions, but often decrease in accuracy or become computationally expensive as noise and outlier levels increase. To address these limitations, we propose a novel generalization of the Hough transform on the unit sphere (S(2)) to estimate the camera's heading. First, the method extracts correspondences between two frames and generates a great circle of directions compatible with each pair of correspondences. Then, by discretizing the unit sphere using a Fibonacci lattice as bin centers, each great circle casts votes for a range of directions, ensuring that features unaffected by noise or dynamic objects vote consistently for the correct motion direction. Experimental results on three datasets demonstrate that the proposed method is on the Pareto frontier of accuracy versus efficiency. Additionally, experiments on SLAM show that the proposed method reduces RMSE by correcting the heading during camera pose initialization.

  </details>



- **WARM-CAT: : Warm-Started Test-Time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning**  
  Xudong Yan, Songhe Feng, Jiaxin Wang, Xin Su, Yi Jin  
  _2026-02-26_ · https://arxiv.org/abs/2602.23114v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Compositional Zero-Shot Learning (CZSL) aims to recognize novel attribute-object compositions based on the knowledge learned from seen ones. Existing methods suffer from performance degradation caused by the distribution shift of label space at test time, which stems from the inclusion of unseen compositions recombined from attributes and objects. To overcome the challenge, we propose a novel approach that accumulates comprehensive knowledge in both textual and visual modalities from unsupervised data to update multimodal prototypes at test time. Building on this, we further design an adaptive update weight to control the degree of prototype adjustment, enabling the model to flexibly adapt to distribution shift during testing. Moreover, a dynamic priority queue is introduced that stores high-confidence images to acquire visual prototypes from historical images for inference. Since the model tends to favor compositions already stored in the queue during testing, we warm-start the queue by initializing it with training images for visual prototypes of seen compositions and generating unseen visual prototypes using the mapping learned between seen and unseen textual prototypes. Considering the semantic consistency of multimodal knowledge, we align textual and visual prototypes by multimodal collaborative representation learning. To provide a more reliable evaluation for CZSL, we introduce a new benchmark dataset, C-Fashion, and refine the widely used but noisy MIT-States dataset. Extensive experiments indicate that our approach achieves state-of-the-art performance on four benchmark datasets under both closed-world and open-world settings. The source code and datasets are available at https://github.com/xud-yan/WARM-CAT .

  </details>



- **VGG-T$^3$: Offline Feed-Forward 3D Reconstruction at Scale**  
  Sven Elflein, Ruilong Li, Sérgio Agostinho, Zan Gojcic, Laura Leal-Taixé, Qunjie Zhou, Aljosa Osep  
  _2026-02-26_ · https://arxiv.org/abs/2602.23361v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present a scalable 3D reconstruction model that addresses a critical limitation in offline feed-forward methods: their computational and memory requirements grow quadratically w.r.t. the number of input images. Our approach is built on the key insight that this bottleneck stems from the varying-length Key-Value (KV) space representation of scene geometry, which we distill into a fixed-size Multi-Layer Perceptron (MLP) via test-time training. VGG-T$^3$ (Visual Geometry Grounded Test Time Training) scales linearly w.r.t. the number of input views, similar to online models, and reconstructs a $1k$ image collection in just $54$ seconds, achieving a $11.6\times$ speed-up over baselines that rely on softmax attention. Since our method retains global scene aggregation capability, our point map reconstruction error outperforming other linear-time methods by large margins. Finally, we demonstrate visual localization capabilities of our model by querying the scene representation with unseen images.

  </details>



- **Towards Long-Form Spatio-Temporal Video Grounding**  
  Xin Gu, Bing Fan, Jiali Yao, Zhipeng Zhang, Yan Huang, Cheng Han, Heng Fan, Libo Zhang  
  _2026-02-26_ · https://arxiv.org/abs/2602.23294v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In real scenarios, videos can span several minutes or even hours. However, existing research on spatio-temporal video grounding (STVG), given a textual query, mainly focuses on localizing targets in short videos of tens of seconds, typically less than one minute, which limits real-world applications. In this paper, we explore Long-Form STVG (LF-STVG), which aims to locate targets in long-term videos. Compared with short videos, long-term videos contain much longer temporal spans and more irrelevant information, making it difficult for existing STVG methods that process all frames at once. To address this challenge, we propose an AutoRegressive Transformer architecture for LF-STVG, termed ART-STVG. Unlike conventional STVG methods that require the entire video sequence to make predictions at once, ART-STVG treats the video as streaming input and processes frames sequentially, enabling efficient handling of long videos. To model spatio-temporal context, we design spatial and temporal memory banks and apply them to the decoders. Since memories from different moments are not always relevant to the current frame, we introduce simple yet effective memory selection strategies to provide more relevant information to the decoders, significantly improving performance. Furthermore, instead of parallel spatial and temporal localization, we propose a cascaded spatio-temporal design that connects the spatial decoder to the temporal decoder, allowing fine-grained spatial cues to assist complex temporal localization in long videos. Experiments on newly extended LF-STVG datasets show that ART-STVG significantly outperforms state-of-the-art methods, while achieving competitive performance on conventional short-form STVG.

  </details>



- **Motion-aware Event Suppression for Event Cameras**  
  Roberto Pellerito, Nico Messikommer, Giovanni Cioffi, Marco Cannici, Davide Scaramuzza  
  _2026-02-26_ · https://arxiv.org/abs/2602.23204v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In this work, we introduce the first framework for Motion-aware Event Suppression, which learns to filter events triggered by IMOs and ego-motion in real time. Our model jointly segments IMOs in the current event stream while predicting their future motion, enabling anticipatory suppression of dynamic events before they occur. Our lightweight architecture achieves 173 Hz inference on consumer-grade GPUs with less than 1 GB of memory usage, outperforming previous state-of-the-art methods on the challenging EVIMO benchmark by 67\% in segmentation accuracy while operating at a 53\% higher inference rate. Moreover, we demonstrate significant benefits for downstream applications: our method accelerates Vision Transformer inference by 83\% via token pruning and improves event-based visual odometry accuracy, reducing Absolute Trajectory Error (ATE) by 13\%.

  </details>



- **From Agnostic to Specific: Latent Preference Diffusion for Multi-Behavior Sequential Recommendation**  
  Ruochen Yang, Xiaodong Li, Jiawei Sheng, Jiangxia Cao, Xinkui Lin, Shen Wang, Shuang Yang, Zhaojie Liu, Tingwen Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23132v1 · `cs.IR`  
  <details><summary>Abstract</summary>

  Multi-behavior sequential recommendation (MBSR) aims to learn the dynamic and heterogeneous interactions of users' multi-behavior sequences, so as to capture user preferences under target behavior for the next interacted item prediction. Unlike previous methods that adopt unidirectional modeling by mapping auxiliary behaviors to target behavior, recent concerns are shifting from behavior-fixed to behavior-specific recommendation. However, these methods still ignore the user's latent preference that underlying decision-making, leading to suboptimal solutions. Meanwhile, due to the asymmetric deterministic between items and behaviors, discriminative paradigm based on preference scoring is unsuitable to capture the uncertainty from low-entropy behaviors to high-entropy items, failing to provide efficient and diverse recommendation. To address these challenges, we propose \textbf{FatsMB}, a framework based diffusion model that guides preference generation \textit{\textbf{F}rom Behavior-\textbf{A}gnostic \textbf{T}o Behavior-\textbf{S}pecific} in latent spaces, enabling diverse and accurate \textit{\textbf{M}ulti-\textbf{B}ehavior Sequential Recommendation}. Specifically, we design a Multi-Behavior AutoEncoder (MBAE) to construct a unified user latent preference space, facilitating interaction and collaboration across Behaviors, within Behavior-aware RoPE (BaRoPE) employed for multiple information fusion. Subsequently, we conduct target behavior-specific preference transfer in the latent space, enriching with informative priors. A Multi-Condition Guided Layer Normalization (MCGLN) is introduced for the denoising. Extensive experiments on real-world datasets demonstrate the effectiveness of our model.

  </details>



- **TriLite: Efficient Weakly Supervised Object Localization with Universal Visual Features and Tri-Region Disentanglement**  
  Arian Sabaghi, José Oramas  
  _2026-02-26_ · https://arxiv.org/abs/2602.23120v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Weakly supervised object localization (WSOL) aims to localize target objects in images using only image-level labels. Despite recent progress, many approaches still rely on multi-stage pipelines or full fine-tuning of large backbones, which increases training cost, while the broader WSOL community continues to face the challenge of partial object coverage. We present TriLite, a single-stage WSOL framework that leverages a frozen Vision Transformer with Dinov2 pre-training in a self-supervised manner, and introduces only a minimal number of trainable parameters (fewer than 800K on ImageNet-1K) for both classification and localization. At its core is the proposed TriHead module, which decomposes patch features into foreground, background, and ambiguous regions, thereby improving object coverage while suppressing spurious activations. By disentangling classification and localization objectives, TriLite effectively exploits the universal representations learned by self-supervised ViTs without requiring expensive end-to-end training. Extensive experiments on CUB-200-2011, ImageNet-1K, and OpenImages demonstrate that TriLite sets a new state of the art, while remaining significantly more parameter-efficient and easier to train than prior methods. The code will be released soon.

  </details>



- **Quantity Convergence, Quality Divergence: Disentangling Fluency and Accuracy in L2 Mandarin Prosody**  
  Yuqi Shi, Hao Yang, Xiyao Lu, Jinsong Zhang  
  _2026-02-26_ · https://arxiv.org/abs/2602.23071v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  While second language (L2) learners may acquire target syntactic word order, mapping this syntax onto appropriate prosodic structures remains a persistent challenge. This study investigates the fossilization and stability of the L2 syntax-prosody interface by comparing 67 native Mandarin speakers with 67 Vietnamese learners using the BLCU-SAIT corpus. By integrating C-ToBI boundary annotation with Dependency Grammar analysis, we examined both the quantity of prosodic boundaries and their mapping to syntactic relations. Results reveal a non-linear acquisition: although high-proficiency learners (VNH) converge to the native baseline in boundary quantity at the Major Phrase level (B3), their structural mapping significantly diverges. Specifically, VNH demote the prosodic boundary at the Subject-Verb (SBV) interface (Major Phrase B3 -> Prosodic Word B1), while erroneously promoting the boundary at the Verb-Object (VOB) interface (Prosodic Word B1 -> Major Phrase B3). This strategy allows learners to maintain high long phrasal output at the expense of structural accuracy. This results in a distorted prosodic hierarchy where the native pattern is inverted.

  </details>



- **HELMLAB: An Analytical, Data-Driven Color Space for Perceptual Distance in UI Design Systems**  
  Gorkem Yildiz  
  _2026-02-26_ · https://arxiv.org/abs/2602.23010v1 · `cs.GR`  
  <details><summary>Abstract</summary>

  We present HELMLAB, a 72-parameter analytical color space for UI design systems. The forward transform maps CIE XYZ to a perceptually-organized Lab representation through learned matrices, per-channel power compression, Fourier hue correction, and embedded Helmholtz-Kohlrausch lightness adjustment. A post-pipeline neutral correction guarantees that achromatic colors map to a=b=0 (chroma < 10^-6), and a rigid rotation of the chromatic plane improves hue-angle alignment without affecting the distance metric, which is invariant under isometries. On the COMBVD dataset (3,813 color pairs), HELMLAB achieves a STRESS of 23.22, a 20.4% reduction from CIEDE2000 (29.18). Cross-validation on He et al. 2022 and MacAdam 1974 shows competitive cross-dataset performance. The transform is invertible with round-trip errors below 10^-14. Gamut mapping, design-token export, and dark/light mode adaptation utilities are included for use in web and mobile design systems.

  </details>



- **Digital Twin-Based Beamforming for Interference Mitigation in AF Relay MIMO Systems**  
  Alexander Bonora, Anna V. Guglielmi, Davide Scazzoli, Marco Giordani, Maurizio Magarini, Vineeth Teeda, Stefano Tomasin  
  _2026-02-26_ · https://arxiv.org/abs/2602.22991v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Beamforming in multiple-input multiple-output (MIMO) systems should take interference mitigation into account. However, for beamform design, accurate channel state information (CSI) is needed, which is often difficult to obtain due to channel variability, feedback overhead, or hardware constraints. For example, amplify-and-forward (AF) relays passively forward signals without measurement, precluding full CSI acquisition to and from the relay. To address these issues, this paper introduces a novel prediction-assisted optimization (PAO) framework for beamform design in AF relay-assisted multiuser MIMO systems. The proposed solution in the AF relay aims at maximizing the signal-plus-interference-to-noise ratio (SINR). Unlike other methods, PAO relies solely on received power measurements, making it suitable for scenarios where CSI is unreliable or unavailable. PAO consists of two stages: a supervised-learning-based neural network (NN) that predicts the positions of transmitters using signal observations, and an optimization algorithm, guided by a digital twin (DT), that iteratively refines the beam direction of the relay in a simulated radio environment. As a key contribution, we validate the proposed framework using realistic measurements collected on a custom-built experimental millimeter wave (mmWave) platform, which enables training of the NN model under practical wireless conditions. The estimated information is then used to update the digital twin with knowledge of the surrounding environment, enabling online optimization. Numerical results show the trade-off between localization accuracy and beamforming performance and confirm that PAO maintains robustness even in the presence of localization errors while reducing the need for real-world measurements.

  </details>


