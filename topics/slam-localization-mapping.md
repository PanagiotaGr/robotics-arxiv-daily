# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-03-14 07:02 UTC_

Total papers shown: **12**


---

- **ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control**  
  Chetan Borse, Zhixian Xie, Wei-Cheng Huang, Wanxin Jin  
  _2026-03-12_ · https://arxiv.org/abs/2603.12185v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Physics simulation for contact-rich robotics is often bottlenecked by contact resolution: mainstream engines enforce non-penetration and Coulomb friction via complementarity constraints or constrained optimization, requiring per-step iterative solves whose cost grows superlinearly with contact density. We present ComFree-Sim, a GPU-parallelized analytical contact physics engine built on complementarity-free contact modeling. ComFree-Sim computes contact impulses in closed form via an impedance-style prediction--correction update in the dual cone of Coulomb friction. Contact computation decouples across contact pairs and becomes separable across cone facets, mapping naturally to GPU kernels and yielding near-linear runtime scaling with the number of contacts. We further extend the formulation to a unified 6D contact model capturing tangential, torsional, and rolling friction, and introduce a practical dual-cone impedance heuristic. ComFree-Sim is implemented in Warp and exposed through a MuJoCo-compatible interface as a drop-in backend alternative to MuJoCo Warp (MJWarp). Experiments benchmark penetration, friction behaviors, stability, and simulation runtime scaling against MJWarp, demonstrating near-linear scaling and 2--3 times higher throughput in dense contact scenes with comparable physical fidelity. We deploy ComFree-Sim in real-time MPC for in-hand dexterous manipulation on a real-world multi-fingered LEAP hand and in dynamics-aware motion retargeting, demonstrating that low-latency simulation yields higher closed-loop success rates and enables practical high-frequency control in contact-rich tasks.

  </details>



- **Decentralized Cooperative Localization for Multi-Robot Systems with Asynchronous Sensor Fusion**  
  Nivand Khosravi, Niusha Khosravi, Mohammad Bozorg, Masoud S. Bahraini  
  _2026-03-12_ · https://arxiv.org/abs/2603.12075v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Decentralized cooperative localization (DCL) is a promising approach for nonholonomic mobile robots operating in GPS-denied environments with limited communication infrastructure. This paper presents a DCL framework in which each robot performs localization locally using an Extended Kalman Filter, while sharing measurement information during update stages only when communication links are available and companion robots are successfully detected by LiDAR. The framework preserves cross-correlation consistency among robot state estimates while handling asynchronous sensor data with heterogeneous sampling rates and accommodating accelerations during dynamic maneuvers. Unlike methods that require pre-aligned coordinate systems, the proposed approach allows robots to initialize with arbitrary reference-frame orientations and achieves automatic alignment through transformation matrices in both the prediction and update stages. To improve robustness in feature-sparse environments, we introduce a dual-landmark evaluation framework that exploits both static environmental features and mobile robots as dynamic landmarks. The proposed framework enables reliable detection and feature extraction during sharp turns, while prediction accuracy is improved through information sharing from mutual observations. Experimental results in both Gazebo simulation and real-world basement environments show that DCL outperforms centralized cooperative localization (CCL), achieving a 34% reduction in RMSE, while the dual-landmark variant yields an improvement of 56%. These results demonstrate the applicability of DCL to challenging domains such as enclosed spaces, underwater environments, and feature-sparse terrains where conventional localization methods are ineffective.

  </details>



- **Dense Dynamic Scene Reconstruction and Camera Pose Estimation from Multi-View Videos**  
  Shuo Sun, Unal Artan, Malcolm Mielle, Achim J. Lilienthaland, Martin Magnusson  
  _2026-03-12_ · https://arxiv.org/abs/2603.12064v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We address the challenging problem of dense dynamic scene reconstruction and camera pose estimation from multiple freely moving cameras -- a setting that arises naturally when multiple observers capture a shared event. Prior approaches either handle only single-camera input or require rigidly mounted, pre-calibrated camera rigs, limiting their practical applicability. We propose a two-stage optimization framework that decouples the task into robust camera tracking and dense depth refinement. In the first stage, we extend single-camera visual SLAM to the multi-camera setting by constructing a spatiotemporal connection graph that exploits both intra-camera temporal continuity and inter-camera spatial overlap, enabling consistent scale and robust tracking. To ensure robustness under limited overlap, we introduce a wide-baseline initialization strategy using feed-forward reconstruction models. In the second stage, we refine depth and camera poses by optimizing dense inter- and intra-camera consistency using wide-baseline optical flow. Additionally, we introduce MultiCamRobolab, a new real-world dataset with ground-truth poses from a motion capture system. Finally, we demonstrate that our method significantly outperforms state-of-the-art feed-forward models on both synthetic and real-world benchmarks, while requiring less memory.

  </details>



- **Learning Visuomotor Policy for Multi-Robot Laser Tag Game**  
  Kai Li, Shiyu Zhao  
  _2026-03-12_ · https://arxiv.org/abs/2603.11980v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In this paper, we study multi robot laser tag, a simplified yet practical shooting-game-style task. Classic modular approaches on these tasks face challenges such as limited observability and reliance on depth mapping and inter robot communication. To overcome these issues, we present an end-to-end visuomotor policy that maps images directly to robot actions. We train a high performing teacher policy with multi agent reinforcement learning and distill its knowledge into a vision-based student policy. Technical designs, including a permutation-invariant feature extractor and depth heatmap input, improve performance over standard architectures. Our policy outperforms classic methods by 16.7% in hitting accuracy and 6% in collision avoidance, and is successfully deployed on real robots. Code will be released publicly.

  </details>



- **Exhaustive Circuit Mapping of a Single-Cell Foundation Model Reveals Massive Redundancy, Heavy-Tailed Hub Architecture, and Layer-Dependent Differentiation Control**  
  Ihor Kendiukhov  
  _2026-03-12_ · https://arxiv.org/abs/2603.11940v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Mechanistic interpretability of biological foundation models has relied on selective feature sampling, pairwise interaction testing, and observational trajectory analysis. Each of these can introduce systematic bias. Here we present three experiments that address these limitations through exhaustive circuit tracing, higher order combinatorial ablation, and causal trajectory steering in Geneformer, a transformer based single cell foundation model. First, exhaustive tracing of all 4065 active sparse autoencoder features at layer 5 yields 1393850 significant downstream edges, a 27 fold expansion over selective sampling. This reveals a heavy tailed hub distribution in which 1.8 percent of features account for disproportionate connectivity and 40 percent of the top 20 hubs lack biological annotation. These results indicate systematic annotation bias in prior selective analyses. Second, three way combinatorial ablation across 8 feature triplets shows that redundancy deepens monotonically with interaction order, with a three way ratio of 0.59 versus a pairwise ratio of 0.74, and with zero synergy. This confirms that the model architecture is subadditive at all tested orders. Third, trajectory guided feature steering establishes a causal link between layer position and differentiation directionality. Late layer features at L17 consistently push cell states toward maturity, with fraction positive equal to 1.0. Early and mid layer features at L0 and L11 mostly push away from maturity, with fraction positive ranging from 0.00 to 0.58. Together these results move from correlation toward causal evidence for layer dependent control of cell state.

  </details>



- **BiGain: Unified Token Compression for Joint Generation and Classification**  
  Jiacheng Liu, Shengkun Tang, Jiacheng Cui, Dongkuan Xu, Zhiqiang Shen  
  _2026-03-12_ · https://arxiv.org/abs/2603.12240v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Acceleration methods for diffusion models (e.g., token merging or downsampling) typically optimize synthesis quality under reduced compute, yet often ignore discriminative capacity. We revisit token compression with a joint objective and present BiGain, a training-free, plug-and-play framework that preserves generation quality while improving classification in accelerated diffusion models. Our key insight is frequency separation: mapping feature-space signals into a frequency-aware representation disentangles fine detail from global semantics, enabling compression that respects both generative fidelity and discriminative utility. BiGain reflects this principle with two frequency-aware operators: (1) Laplacian-gated token merging, which encourages merges among spectrally smooth tokens while discouraging merges of high-contrast tokens, thereby retaining edges and textures; and (2) Interpolate-Extrapolate KV Downsampling, which downsamples keys/values via a controllable interextrapolation between nearest and average pooling while keeping queries intact, thereby conserving attention precision. Across DiT- and U-Net-based backbones and ImageNet-1K, ImageNet-100, Oxford-IIIT Pets, and COCO-2017, our operators consistently improve the speed-accuracy trade-off for diffusion-based classification, while maintaining or enhancing generation quality under comparable acceleration. For instance, on ImageNet-1K, with 70% token merging on Stable Diffusion 2.0, BiGain increases classification accuracy by 7.15% while improving FID by 0.34 (1.85%). Our analyses indicate that balanced spectral retention, preserving high-frequency detail and low/mid-frequency semantics, is a reliable design rule for token compression in diffusion models. To our knowledge, BiGain is the first framework to jointly study and advance both generation and classification under accelerated diffusion, supporting lower-cost deployment.

  </details>



- **A Two-Stage Dual-Modality Model for Facial Emotional Expression Recognition**  
  Jiajun Sun, Zhe Gao  
  _2026-03-12_ · https://arxiv.org/abs/2603.12221v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This paper addresses the expression (EXPR) recognition challenge in the 10th Affective Behavior Analysis in-the-Wild (ABAW) workshop and competition, which requires frame-level classification of eight facial emotional expressions from unconstrained videos. This task is challenging due to inaccurate face localization, large pose and scale variations, motion blur, temporal instability, and other confounding factors across adjacent frames. We propose a two-stage dual-modal (audio-visual) model to address these difficulties. Stage I focuses on robust visual feature extraction with a pretrained DINOv2-based encoder. Specifically, DINOv2 ViT-L/14 is used as the backbone, a padding-aware augmentation (PadAug) strategy is employed for image padding and data preprocessing from raw videos, and a mixture-of-experts (MoE) training head is introduced to enhance classifier diversity. Stage II addresses modality fusion and temporal consistency. For the visual modality, faces are re-cropped from raw videos at multiple scales, and the extracted visual features are averaged to form a robust frame-level representation. Concurrently, frame-aligned Wav2Vec 2.0 audio features are derived from short audio windows to provide complementary acoustic cues. These dual-modal features are integrated via a lightweight gated fusion module, followed by inference-time temporal smoothing. Experiments on the ABAW dataset demonstrate the effectiveness of the proposed method. The two-stage model achieves a Macro-F1 score of 0.5368 on the official validation set and 0.5122 +/- 0.0277 under 5-fold cross-validation, outperforming the official baselines.

  </details>



- **RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images**  
  Bin Wan, Runmin Cong, Xiaofei Zhou, Hao Fang, Yaoqi Sun, Sam Kwong  
  _2026-03-12_ · https://arxiv.org/abs/2603.12215v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Salient object detection (SOD) in remote sensing images faces significant challenges due to large variations in object sizes, the computational cost of self-attention mechanisms, and the limitations of CNN-based extractors in capturing global context and long-range dependencies. Existing methods that rely on fixed convolution kernels often struggle to adapt to diverse object scales, leading to detail loss or irrelevant feature aggregation. To address these issues, this work aims to enhance robustness to scale variations and achieve precise object localization. We propose the Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network (RDNet), which replaces the CNN backbone with the SwinTransformer for global context modeling and introduces three key modules: (1) the Dynamic Adaptive Detail-aware (DAD) module, which applies varied convolution kernels guided by object region proportions; (2) the Frequency-matching Context Enhancement (FCE) module, which enriches contextual information through wavelet interactions and attention; and (3) the Region Proportion-aware Localization (RPL) module, which employs cross-attention to highlight semantic details and integrates a Proportion Guidance (PG) block to assist the DAD module. By combining these modules, RDNet achieves robustness against scale variations and accurate localization, delivering superior detection performance compared with state-of-the-art methods.

  </details>



- **Cascade: Composing Software-Hardware Attack Gadgets for Adversarial Threat Amplification in Compound AI Systems**  
  Sarbartha Banerjee, Prateek Sahu, Anjo Vahldiek-Oberwagner, Jose Sanchez Vicarte, Mohit Tiwari  
  _2026-03-12_ · https://arxiv.org/abs/2603.12023v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Rapid progress in generative AI has given rise to Compound AI systems - pipelines comprised of multiple large language models (LLM), software tools and database systems. Compound AI systems are constructed on a layered traditional software stack running on a distributed hardware infrastructure. Many of the diverse software components are vulnerable to traditional security flaws documented in the Common Vulnerabilities and Exposures (CVE) database, while the underlying distributed hardware infrastructure remains exposed to timing attacks, bit-flip faults, and power-based side channels. Today, research targets LLM-specific risks like model extraction, training data leakage, and unsafe generation -- overlooking the impact of traditional system vulnerabilities. This work investigates how traditional software and hardware vulnerabilities can complement LLM-specific algorithmic attacks to compromise the integrity of a compound AI pipeline. We demonstrate two novel attacks that combine system-level vulnerabilities with algorithmic weaknesses: (1) Exploiting a software code injection flaw along with a guardrail Rowhammer attack to inject an unaltered jailbreak prompt into an LLM, resulting in an AI safety violation, and (2) Manipulating a knowledge database to redirect an LLM agent to transmit sensitive user data to a malicious application, thus breaching confidentiality. These attacks highlight the need to address traditional vulnerabilities; we systematize the attack primitives and analyze their composition by grouping vulnerabilities by their objective and mapping them to distinct stages of an attack lifecycle. This approach enables a rigorous red-teaming exercise and lays the groundwork for future defense strategies.

  </details>



- **Indirect and Direct Multiuser Hybrid Beamforming for Far-Field and Near-Field Communications: A Deep Learning Approach**  
  Xinyang Li, Songjie Yang, Boyu Ning, Zongmiao He, Xiang Ling, Chau Yuen  
  _2026-03-12_ · https://arxiv.org/abs/2603.11918v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Hybrid beamforming for extremely large-scale multiple-input multiple-output (XL-MIMO) systems is challenging in the near field because the channel depends jointly on angle and distance, and the multiuser interference (MUI) is strong. Existing deep learning methods typically follow either a decoupled design that optimizes analog beamforming without explicitly accounting for MUI, or an end-to-end (E2E) joint analog-digital optimization that can be unstable under nonconvex constant-modulus (CM), pronounced analog-digital coupling, and gradient pattern of sum-rate loss. To address both issues, we develop a complex-valued E2E framework based on a variant minimum mean square error (variant-MMSE) criterion, where the digital precoder is eliminated in closed form via Karush-Kuhn-Tucker (KKT) conditions so that analog learning is trained with a stable objective. The network employs a grouped complex-convolution sensing front-end for uplink (UL) measurements, a shared complex multi-layer perceptron (MLP) for per-user feature extraction, and a merged constant-modulus head to output the analog precoder. In the indirect mode, the network designs hybrid beamformers from estimated channel state information (CSI). In the direct mode where explicit CSI is unavailable, the network learns the sensing operator and the analog mapping from short pilots, after which additional pilots estimate the equivalent channel and enable a KKT closed-form digital precoder. Simulations show that the indirect mode approaches the performance of iterative variant-MMSE optimization with a complexity reduction proportional to the antenna number. In the direct mode, the proposed method improves spectral efficiency over sparse-recovery pipelines and recent deep learning baselines under the same pilot budget.

  </details>



- **A Diffeomorphism Groupoid and Algebroid Framework for Discontinuous Image Registration**  
  Lili Bao, Bin Xiao, Shihui Ying, Stefan Sommer  
  _2026-03-12_ · https://arxiv.org/abs/2603.11806v1 · `math.GR`  
  <details><summary>Abstract</summary>

  In this paper, we propose a novel mathematical framework for piecewise diffeomorphic image registration that involves discontinuous sliding motion using a diffeomorphism groupoid and algebroid approach. The traditional Large Deformation Diffeomorphic Metric Mapping (LDDMM) registration method builds on Lie groups, which assume continuity and smoothness in velocity fields, limiting its applicability in handling discontinuous sliding motion. To overcome this limitation, we extend the diffeomorphism Lie groups to a framework of discontinuous diffeomorphism Lie groupoids, allowing for discontinuities along sliding boundaries while maintaining diffeomorphism within homogeneous regions. We provide a rigorous analysis of the associated mathematical structures, including Lie algebroids and their duals, and derive specific Euler-Arnold equations to govern optimal flows for discontinuous deformations. Some numerical tests are performed to validate the efficiency of the proposed approach.

  </details>



- **DocSage: An Information Structuring Agent for Multi-Doc Multi-Entity Question Answering**  
  Teng Lin, Yizhang Zhu, Zhengxuan Zhang, Yuyu Luo, Nan Tang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11798v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Multi-document Multi-entity Question Answering inherently demands models to track implicit logic between multiple entities across scattered documents. However, existing Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) frameworks suffer from critical limitations: standard RAG's vector similarity-based coarse-grained retrieval often omits critical facts, graph-based RAG fails to efficiently integrate fragmented complex relationship networks, and both lack schema awareness, leading to inadequate cross-document evidence chain construction and inaccurate entity relationship deduction. To address these challenges, we propose DocSage, an end-to-end agentic framework that integrates dynamic schema discovery, structured information extraction, and schema-aware relational reasoning with error guarantees. DocSage operates through three core modules: (1) A schema discovery module dynamically infers query-specific minimal joinable schemas to capture essential entities and relationships; (2) An extraction module transforms unstructured text into semantically coherent relational tables, enhanced by error-aware correction mechanisms to reduce extraction errors; (3) A reasoning module performs multi-hop relational reasoning over structured tables, leveraging schema awareness to efficiently align cross-document entities and aggregate evidence. This agentic design offers three key advantages: precise fact localization via SQL-powered indexing, natural support for cross-document entity joins through relational tables, and mitigated LLM attention diffusion via structured representation. Evaluations on two MDMEQA benchmarks demonstrate that DocSage significantly outperforms state-of-the-art long-context LLMs and RAG systems, achieving more than 27% accuracy improvements respectively.

  </details>


