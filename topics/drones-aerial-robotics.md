# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-01-27 06:52 UTC_

Total papers shown: **4**


---

- **$α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks**  
  Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah  
  _2026-01-26_ · https://arxiv.org/abs/2601.18754v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Autonomous unmanned aerial vehicle (UAV) systems are increasingly deployed in safety-critical, networked environments where they must operate reliably in the presence of malicious adversaries. While recent benchmarks have evaluated large language model (LLM)-based UAV agents in reasoning, navigation, and efficiency, systematic assessment of security, resilience, and trust under adversarial conditions remains largely unexplored, particularly in emerging 6G-enabled settings. We introduce $α^{3}$-SecBench, the first large-scale evaluation suite for assessing the security-aware autonomy of LLM-based UAV agents under realistic adversarial interference. Building on multi-turn conversational UAV missions from $α^{3}$-Bench, the framework augments benign episodes with 20,000 validated security overlay attack scenarios targeting seven autonomy layers, including sensing, perception, planning, control, communication, edge/cloud infrastructure, and LLM reasoning. $α^{3}$-SecBench evaluates agents across three orthogonal dimensions: security (attack detection and vulnerability attribution), resilience (safe degradation behavior), and trust (policy-compliant tool usage). We evaluate 23 state-of-the-art LLMs from major industrial providers and leading AI labs using thousands of adversarially augmented UAV episodes sampled from a corpus of 113,475 missions spanning 175 threat types. While many models reliably detect anomalous behavior, effective mitigation, vulnerability attribution, and trustworthy control actions remain inconsistent. Normalized overall scores range from 12.9% to 57.1%, highlighting a significant gap between anomaly detection and security-aware autonomous decision-making. We release $α^{3}$-SecBench on GitHub: https://github.com/maferrag/AlphaSecBench

  </details>



- **EFSI-DETR: Efficient Frequency-Semantic Integration for Real-Time Small Object Detection in UAV Imagery**  
  Yu Xia, Chang Liu, Tianqi Xiang, Zhigang Tu  
  _2026-01-26_ · https://arxiv.org/abs/2601.18597v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Real-time small object detection in Unmanned Aerial Vehicle (UAV) imagery remains challenging due to limited feature representation and ineffective multi-scale fusion. Existing methods underutilize frequency information and rely on static convolutional operations, which constrain the capacity to obtain rich feature representations and hinder the effective exploitation of deep semantic features. To address these issues, we propose EFSI-DETR, a novel detection framework that integrates efficient semantic feature enhancement with dynamic frequency-spatial guidance. EFSI-DETR comprises two main components: (1) a Dynamic Frequency-Spatial Unified Synergy Network (DyFusNet) that jointly exploits frequency and spatial cues for robust multi-scale feature fusion, (2) an Efficient Semantic Feature Concentrator (ESFC) that enables deep semantic extraction with minimal computational cost. Furthermore, a Fine-grained Feature Retention (FFR) strategy is adopted to incorporate spatially rich shallow features during fusion to preserve fine-grained details, crucial for small object detection in UAV imagery. Extensive experiments on VisDrone and CODrone benchmarks demonstrate that our EFSI-DETR achieves the state-of-the-art performance with real-time efficiency, yielding improvement of \textbf{1.6}\% and \textbf{5.8}\% in AP and AP$_{s}$ on VisDrone, while obtaining \textbf{188} FPS inference speed on a single RTX 4090 GPU.

  </details>



- **Discriminability-Driven Spatial-Channel Selection with Gradient Norm for Drone Signal OOD Detection**  
  Chuhan Feng, Jing Li, Jie Li, Lu Lv, Fengkui Gong  
  _2026-01-26_ · https://arxiv.org/abs/2601.18329v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We propose a drone signal out-of-distribution (OOD) detection algorithm based on discriminability-driven spatial-channel selection with a gradient norm. Time-frequency image features are adaptively weighted along both spatial and channel dimensions by quantifying inter-class similarity and variance based on protocol-specific time-frequency characteristics. Subsequently, a gradient-norm metric is introduced to measure perturbation sensitivity for capturing the inherent instability of OOD samples, which is then fused with energy-based scores for joint inference. Simulation results demonstrate that the proposed algorithm provides superior discriminative power and robust performance via SNR and various drone types.

  </details>



- **Cognitive Fusion of ZC Sequences and Time-Frequency Images for Out-of-Distribution Detection of Drone Signals**  
  Jie Li, Jing Li, Lu Lv, Zhanyu Ju, Fengkui Gong  
  _2026-01-26_ · https://arxiv.org/abs/2601.18326v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We propose a drone signal out-of-distribution detection (OODD) algorithm based on the cognitive fusion of Zadoff-Chu (ZC) sequences and time-frequency images (TFI). ZC sequences are identified by analyzing the communication protocols of DJI drones, while TFI capture the time-frequency characteristics of drone signals with unknown or non-standard communication protocols. Both modalities are used jointly to enable OODD in the drone remote identification (RID) task. Specifically, ZC sequence features and TFI features are generated from the received radio frequency signals, which are then processed through dedicated feature extraction module to enhance and align them. The resultant multi-modal features undergo multi-modal feature interaction, single-modal feature fusion, and multi-modal feature fusion to produce features that integrate and complement information across modalities. Discrimination scores are computed from the fused features along both spatial and channel dimensions to capture time-frequency characteristic differences dictated by the communication protocols, and these scores will be transformed into adaptive attention weights. The weighted features are then passed through a Softmax function to produce the signal classification results. Simulation results demonstrate that the proposed algorithm outperforms existing algorithms and achieves 1.7% and 7.5% improvements in RID and OODD metrics, respectively. The proposed algorithm also performs strong robustness under varying flight conditions and across different drone types.

  </details>


