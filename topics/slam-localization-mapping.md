# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-11 07:16 UTC_

Total papers shown: **15**


---

- **A Collision-Free Sway Damping Model Predictive Controller for Safe and Reactive Forestry Crane Navigation**  
  Marc-Philip Ecker, Christoph Fröhlich, Johannes Huemer, David Gruber, Bernhard Bischof, Tobias Glück, Wolfgang Kemmetmüller  
  _2026-02-10_ · https://arxiv.org/abs/2602.10035v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Forestry cranes operate in dynamic, unstructured outdoor environments where simultaneous collision avoidance and payload sway control are critical for safe navigation. Existing approaches address these challenges separately, either focusing on sway damping with predefined collision-free paths or performing collision avoidance only at the global planning level. We present the first collision-free, sway-damping model predictive controller (MPC) for a forestry crane that unifies both objectives in a single control framework. Our approach integrates LiDAR-based environment mapping directly into the MPC using online Euclidean distance fields (EDF), enabling real-time environmental adaptation. The controller simultaneously enforces collision constraints while damping payload sway, allowing it to (i) replan upon quasi-static environmental changes, (ii) maintain collision-free operation under disturbances, and (iii) provide safe stopping when no bypass exists. Experimental validation on a real forestry crane demonstrates effective sway damping and successful obstacle avoidance. A video can be found at https://youtu.be/tEXDoeLLTxA.

  </details>



- **RoboSubtaskNet: Temporal Sub-task Segmentation for Human-to-Robot Skill Transfer in Real-World Environments**  
  Dharmendra Sharma, Archit Sharma, John Reberio, Vaibhav Kesharwani, Peeyush Thakur, Narendra Kumar Dhar, Laxmidhar Behera  
  _2026-02-10_ · https://arxiv.org/abs/2602.10015v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Temporally locating and classifying fine-grained sub-task segments in long, untrimmed videos is crucial to safe human-robot collaboration. Unlike generic activity recognition, collaborative manipulation requires sub-task labels that are directly robot-executable. We present RoboSubtaskNet, a multi-stage human-to-robot sub-task segmentation framework that couples attention-enhanced I3D features (RGB plus optical flow) with a modified MS-TCN employing a Fibonacci dilation schedule to capture better short-horizon transitions such as reach-pick-place. The network is trained with a composite objective comprising cross-entropy and temporal regularizers (truncated MSE and a transition-aware term) to reduce over-segmentation and to encourage valid sub-task progressions. To close the gap between vision benchmarks and control, we introduce RoboSubtask, a dataset of healthcare and industrial demonstrations annotated at the sub-task level and designed for deterministic mapping to manipulator primitives. Empirically, RoboSubtaskNet outperforms MS-TCN and MS-TCN++ on GTEA and our RoboSubtask benchmark (boundary-sensitive and sequence metrics), while remaining competitive on the long-horizon Breakfast benchmark. Specifically, RoboSubtaskNet attains F1 @ 50 = 79.5%, Edit = 88.6%, Acc = 78.9% on GTEA; F1 @ 50 = 30.4%, Edit = 52.0%, Acc = 53.5% on Breakfast; and F1 @ 50 = 94.2%, Edit = 95.6%, Acc = 92.2% on RoboSubtask. We further validate the full perception-to-execution pipeline on a 7-DoF Kinova Gen3 manipulator, achieving reliable end-to-end behavior in physical trials (overall task success approx 91.25%). These results demonstrate a practical path from sub-task level video understanding to deployed robotic manipulation in real-world settings.

  </details>



- **Acoustic Drone Package Delivery Detection**  
  François Marcoux, François Grondin  
  _2026-02-10_ · https://arxiv.org/abs/2602.09991v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In recent years, the illicit use of unmanned aerial vehicles (UAVs) for deliveries in restricted area such as prisons became a significant security challenge. While numerous studies have focused on UAV detection or localization, little attention has been given to delivery events identification. This study presents the first acoustic package delivery detection algorithm using a ground-based microphone array. The proposed method estimates both the drone's propeller speed and the delivery event using solely acoustic features. A deep neural network detects the presence of a drone and estimates the propeller's rotation speed or blade passing frequency (BPF) from a mel spectrogram. The algorithm analyzes the BPFs to identify probable delivery moments based on sudden changes before and after a specific time. Results demonstrate a mean absolute error of the blade passing frequency estimator of 16 Hz when the drone is less than 150 meters away from the microphone array. The drone presence detection estimator has a accuracy of 97%. The delivery detection algorithm correctly identifies 96% of events with a false positive rate of 8%. This study shows that deliveries can be identified using acoustic signals up to a range of 100 meters.

  </details>



- **Hydra-Nav: Object Navigation via Adaptive Dual-Process Reasoning**  
  Zixuan Wang, Huang Fang, Shaoan Wang, Yuanfei Luo, Heng Dong, Wei Li, Yiming Gan  
  _2026-02-10_ · https://arxiv.org/abs/2602.09972v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While large vision-language models (VLMs) show promise for object goal navigation, current methods still struggle with low success rates and inefficient localization of unseen objects--failures primarily attributed to weak temporal-spatial reasoning. Meanwhile, recent attempts to inject reasoning into VLM-based agents improve success rates but incur substantial computational overhead. To address both the ineffectiveness and inefficiency of existing approaches, we introduce Hydra-Nav, a unified VLM architecture that adaptively switches between a deliberative slow system for analyzing exploration history and formulating high-level plans, and a reactive fast system for efficient execution. We train Hydra-Nav through a three-stage curriculum: (i) spatial-action alignment to strengthen trajectory planning, (ii) memory-reasoning integration to enhance temporal-spatial reasoning over long-horizon exploration, and (iii) iterative rejection fine-tuning to enable selective reasoning at critical decision points. Extensive experiments demonstrate that Hydra-Nav achieves state-of-the-art performance on the HM3D, MP3D, and OVON benchmarks, outperforming the second-best methods by 11.1%, 17.4%, and 21.2%, respectively. Furthermore, we introduce SOT (Success weighted by Operation Time), a new metric to measure search efficiency across VLMs with varying reasoning intensity. Results show that adaptive reasoning significantly enhances search efficiency over fixed-frequency baselines.

  </details>



- **HAPS-RIS and UAV Integrated Networks: A Unified Joint Multi-objective Framework**  
  Arman Azizi, Mostafa Rahmani Ghourtani, Mustafa A. Kishk, Hamed Ahmadi, Arman Farhang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09960v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Future 6G non-terrestrial networks aim to deliver ubiquitous connectivity to remote and undeserved regions, but unmanned aerial vehicle (UAV) base stations face fundamental challenges such as limited numbers and power budgets. To overcome these obstacles, high-altitude platform station (HAPS) equipped with a reconfigurable intelligent surface (RIS), so-called HAPS-RIS, is a promising candidate. We propose a novel unified joint multi-objective framework where UAVs and HAPS-RIS are fully integrated to extend coverage and enhance network performance. This joint multi-objective design maximizes the number of users served by the HAPS-RIS, minimizes the number of UAVs deployed and minimizes the total average UAV path loss subject to quality-of-service (QoS) and resource constraints. We propose a novel low-complexity solution strategy by proving the equivalence between minimizing the total average UAV path loss upper bound and k-means clustering, deriving a practical closed-form RIS phase-shift design, and introducing a mapping technique that collapses the combinatorial assignments into a zone radius and a bandwidth-portioning factor. Then, we propose a dynamic Pareto optimization technique to solve the transformed optimization problem. Extensive simulation results demonstrate that the proposed framework adapts seamlessly across operating regimes. A HAPS-RIS-only setup achieves full coverage at low data rates, but UAV assistance becomes indispensable as rate demands increase. By tuning a single bandwidth portioning factor, the model recovers UAV-only, HAPS-RIS-only and equal bandwidth portioning baselines within one formulation and consistently surpasses them across diverse rate requirements. The simulations also quantify a tangible trade-off between RIS scale and UAV deployment, enabling designers to trade increased RIS elements for fewer UAVs as service demands evolve.

  </details>



- **Can Image Splicing and Copy-Move Forgery Be Detected by the Same Model? Forensim: An Attention-Based State-Space Approach**  
  Soumyaroop Nandi, Prem Natarajan  
  _2026-02-10_ · https://arxiv.org/abs/2602.10079v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We introduce Forensim, an attention-based state-space framework for image forgery detection that jointly localizes both manipulated (target) and source regions. Unlike traditional approaches that rely solely on artifact cues to detect spliced or forged areas, Forensim is designed to capture duplication patterns crucial for understanding context. In scenarios such as protest imagery, detecting only the forged region, for example a duplicated act of violence inserted into a peaceful crowd, can mislead interpretation, highlighting the need for joint source-target localization. Forensim outputs three-class masks (pristine, source, target) and supports detection of both splicing and copy-move forgeries within a unified architecture. We propose a visual state-space model that leverages normalized attention maps to identify internal similarities, paired with a region-based block attention module to distinguish manipulated regions. This design enables end-to-end training and precise localization. Forensim achieves state-of-the-art performance on standard benchmarks. We also release CMFD-Anything, a new dataset addressing limitations of existing copy-move forgery datasets.

  </details>



- **Supervised Metric Regularization Through Alternating Optimization for Multi-Regime Physics-Informed Neural Networks**  
  Enzo Nicolas Spotorno, Josafat Ribeiro Leal, Antonio Augusto Frohlich  
  _2026-02-10_ · https://arxiv.org/abs/2602.09980v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Standard Physics-Informed Neural Networks (PINNs) often face challenges when modeling parameterized dynamical systems with sharp regime transitions, such as bifurcations. In these scenarios, the continuous mapping from parameters to solutions can result in spectral bias or "mode collapse", where the network averages distinct physical behaviors. We propose a Topology-Aware PINN (TAPINN) that aims to mitigate this challenge by structuring the latent space via Supervised Metric Regularization. Unlike standard parametric PINNs that map physical parameters directly to solutions, our method conditions the solver on a latent state optimized to reflect the metric-based separation between regimes, showing ~49% lower physics residual (0.082 vs. 0.160). We train this architecture using a phase-based Alternating Optimization (AO) schedule to manage gradient conflicts between the metric and physics objectives. Preliminary experiments on the Duffing Oscillator demonstrate that while standard baselines suffer from spectral bias and high-capacity Hypernetworks overfit (memorizing data while violating physics), our approach achieves stable convergence with 2.18x lower gradient variance than a multi-output Sobolev Error baseline, and 5x fewer parameters than a hypernetwork-based alternative.

  </details>



- **Learning to Detect Baked Goods with Limited Supervision**  
  Thomas H. Schmitt, Maximilian Bundscherer, Tobias Bocklet  
  _2026-02-10_ · https://arxiv.org/abs/2602.09979v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Monitoring leftover products provides valuable insights that can be used to optimize future production. This is especially important for German bakeries because freshly baked goods have a very short shelf life. Automating this process can reduce labor costs, improve accuracy, and streamline operations. We propose automating this process using an object detection model to identify baked goods from images. However, the large diversity of German baked goods makes fully supervised training prohibitively expensive and limits scalability. Although open-vocabulary detectors (e.g., OWLv2, Grounding DINO) offer lexibility, we demonstrate that they are insufficient for our task. While motivated by bakeries, our work addresses the broader challenges of deploying computer vision in industries, where tasks are specialized and annotated datasets are scarce. We compile dataset splits with varying supervision levels, covering 19 classes of baked goods. We propose two training workflows to train an object detection model with limited supervision. First, we combine OWLv2 and Grounding DINO localization with image-level supervision to train the model in a weakly supervised manner. Second, we improve viewpoint robustness by fine-tuning on video frames annotated using Segment Anything 2 as a pseudo-label propagation model. Using these workflows, we train YOLOv11 for our detection task due to its favorable speed accuracy tradeoff. Relying solely on image-level supervision, the model achieves a mean Average Precision (mAP) of 0.91. Finetuning with pseudo-labels raises model performance by 19.3% under non-ideal deployment conditions. Combining these workflows trains a model that surpasses our fully-supervised baseline model under non-ideal deployment conditions, despite relying only on image-level supervision.

  </details>



- **Robust Processing and Learning: Principles, Methods, and Wireless Applications**  
  Shixiong Wang, Wei Dai, Li-Chun Wang, Geoffrey Ye Li  
  _2026-02-10_ · https://arxiv.org/abs/2602.09848v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  This tutorial-style overview article examines the fundamental principles and methods of robustness, using wireless sensing and communication (WSC) as the narrative and exemplifying framework. First, we formalize the conceptual and mathematical foundations of robustness, highlighting the interpretations and relations across robust statistics, optimization, and machine learning. Key techniques, such as robust estimation and testing, distributionally robust optimization, and regularized and adversary training, are investigated. Together, the costs of robustness in system design, for example, the compromised nominal performances and the extra computational burdens, are discussed. Second, we review recent robust signal processing solutions for WSC that address model mismatch, data scarcity, adversarial perturbation, and distributional shift. Specific applications include robust ranging-based localization, modality sensing, channel estimation, receive combining, waveform design, and federated learning. Through this effort, we aim to introduce the classical developments and recent advances in robustness theory to the general signal processing community, exemplifying how robust statistical, optimization, and machine learning approaches can address the uncertainties inherent in WSC systems.

  </details>



- **Where Do Images Come From? Analyzing Captions to Geographically Profile Datasets**  
  Abhipsa Basu, Yugam Bahl, Kirti Bhagat, Preethi Seshadri, R. Venkatesh Babu, Danish Pruthi  
  _2026-02-10_ · https://arxiv.org/abs/2602.09775v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent studies show that text-to-image models often fail to generate geographically representative images, raising concerns about the representativeness of their training data and motivating the question: which parts of the world do these training examples come from? We geographically profile large-scale multimodal datasets by mapping image-caption pairs to countries based on location information extracted from captions using LLMs. Studying English captions from three widely used datasets (Re-LAION, DataComp1B, and Conceptual Captions) across $20$ common entities (e.g., house, flag), we find that the United States, the United Kingdom, and Canada account for $48.0\%$ of samples, while South American and African countries are severely under-represented with only $1.8\%$ and $3.8\%$ of images, respectively. We observe a strong correlation between a country's GDP and its representation in the data ($ρ= 0.82$). Examining non-English subsets for $4$ languages from the Re-LAION dataset, we find that representation skews heavily toward countries where these languages are predominantly spoken. Additionally, we find that higher representation does not necessarily translate to greater visual or semantic diversity. Finally, analyzing country-specific images generated by Stable Diffusion v1.3 trained on Re-LAION, we show that while generations appear realistic, they are severely limited in their coverage compared to real-world images.

  </details>



- **Grounding LTL Tasks in Sub-Symbolic RL Environments for Zero-Shot Generalization**  
  Matteo Pannacci, Andrea Fanti, Elena Umili, Roberto Capobianco  
  _2026-02-10_ · https://arxiv.org/abs/2602.09761v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  In this work we address the problem of training a Reinforcement Learning agent to follow multiple temporally-extended instructions expressed in Linear Temporal Logic in sub-symbolic environments. Previous multi-task work has mostly relied on knowledge of the mapping between raw observations and symbols appearing in the formulae. We drop this unrealistic assumption by jointly training a multi-task policy and a symbol grounder with the same experience. The symbol grounder is trained only from raw observations and sparse rewards via Neural Reward Machines in a semi-supervised fashion. Experiments on vision-based environments show that our method achieves performance comparable to using the true symbol grounding and significantly outperforms state-of-the-art methods for sub-symbolic environments.

  </details>



- **Contextual and Seasonal LSTMs for Time Series Anomaly Detection**  
  Lingpei Zhang, Qingming Li, Yong Yang, Jiahao Chen, Rui Zeng, Chenyang Lyu, Shouling Ji  
  _2026-02-10_ · https://arxiv.org/abs/2602.09690v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Univariate time series (UTS), where each timestamp records a single variable, serve as crucial indicators in web systems and cloud servers. Anomaly detection in UTS plays an essential role in both data mining and system reliability management. However, existing reconstruction-based and prediction-based methods struggle to capture certain subtle anomalies, particularly small point anomalies and slowly rising anomalies. To address these challenges, we propose a novel prediction-based framework named Contextual and Seasonal LSTMs (CS-LSTMs). CS-LSTMs are built upon a noise decomposition strategy and jointly leverage contextual dependencies and seasonal patterns, thereby strengthening the detection of subtle anomalies. By integrating both time-domain and frequency-domain representations, CS-LSTMs achieve more accurate modeling of periodic trends and anomaly localization. Extensive evaluations on public benchmark datasets demonstrate that CS-LSTMs consistently outperform state-of-the-art methods, highlighting their effectiveness and practical value in robust time series anomaly detection.

  </details>



- **Towards Training-free Multimodal Hate Localisation with Large Language Models**  
  Yueming Sun, Long Yang, Jianbo Jiao, Zeyu Fu  
  _2026-02-10_ · https://arxiv.org/abs/2602.09637v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The proliferation of hateful content in online videos poses severe threats to individual well-being and societal harmony. However, existing solutions for video hate detection either rely heavily on large-scale human annotations or lack fine-grained temporal precision. In this work, we propose LELA, the first training-free Large Language Model (LLM) based framework for hate video localization. Distinct from state-of-the-art models that depend on supervised pipelines, LELA leverages LLMs and modality-specific captioning to detect and temporally localize hateful content in a training-free manner. Our method decomposes a video into five modalities, including image, speech, OCR, music, and video context, and uses a multi-stage prompting scheme to compute fine-grained hateful scores for each frame. We further introduce a composition matching mechanism to enhance cross-modal reasoning. Experiments on two challenging benchmarks, HateMM and MultiHateClip, demonstrate that LELA outperforms all existing training-free baselines by a large margin. We also provide extensive ablations and qualitative visualizations, establishing LELA as a strong foundation for scalable and interpretable hate video localization.

  </details>



- **ECG-IMN: Interpretable Mesomorphic Neural Networks for 12-Lead Electrocardiogram Interpretation**  
  Vajira Thambawita, Jonas L. Isaksen, Jørgen K. Kanters, Hugo L. Hammer, Pål Halvorsen  
  _2026-02-10_ · https://arxiv.org/abs/2602.09566v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Deep learning has achieved expert-level performance in automated electrocardiogram (ECG) diagnosis, yet the "black-box" nature of these models hinders their clinical deployment. Trust in medical AI requires not just high accuracy but also transparency regarding the specific physiological features driving predictions. Existing explainability methods for ECGs typically rely on post-hoc approximations (e.g., Grad-CAM and SHAP), which can be unstable, computationally expensive, and unfaithful to the model's actual decision-making process. In this work, we propose the ECG-IMN, an Interpretable Mesomorphic Neural Network tailored for high-resolution 12-lead ECG classification. Unlike standard classifiers, the ECG-IMN functions as a hypernetwork: a deep convolutional backbone generates the parameters of a strictly linear model specific to each input sample. This architecture enforces intrinsic interpretability, as the decision logic is mathematically transparent and the generated weights (W) serve as exact, high-resolution feature attribution maps. We introduce a transition decoder that effectively maps latent features to sample-wise weights, enabling precise localization of pathological evidence (e.g., ST-elevation, T-wave inversion) in both time and lead dimensions. We evaluate our approach on the PTB-XL dataset for classification tasks, demonstrating that the ECG-IMN achieves competitive predictive performance (AUROC comparable to black-box baselines) while providing faithful, instance-specific explanations. By explicitly decoupling parameter generation from prediction execution, our framework bridges the gap between deep learning capability and clinical trustworthiness, offering a principled path toward "white-box" cardiac diagnostics.

  </details>



- **Scalpel: Fine-Grained Alignment of Attention Activation Manifolds via Mixture Gaussian Bridges to Mitigate Multimodal Hallucination**  
  Ziqiang Shi, Rujie Liu, Shanshan Yu, Satoshi Munakata, Koichi Shirahata  
  _2026-02-10_ · https://arxiv.org/abs/2602.09541v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Rapid progress in large vision-language models (LVLMs) has achieved unprecedented performance in vision-language tasks. However, due to the strong prior of large language models (LLMs) and misaligned attention across modalities, LVLMs often generate outputs inconsistent with visual content - termed hallucination. To address this, we propose \textbf{Scalpel}, a method that reduces hallucination by refining attention activation distributions toward more credible regions. Scalpel predicts trusted attention directions for each head in Transformer layers during inference and adjusts activations accordingly. It employs a Gaussian mixture model to capture multi-peak distributions of attention in trust and hallucination manifolds, and uses entropic optimal transport (equivalent to Schrödinger bridge problem) to map Gaussian components precisely. During mitigation, Scalpel dynamically adjusts intervention strength and direction based on component membership and mapping relationships between hallucination and trust activations. Extensive experiments across multiple datasets and benchmarks demonstrate that Scalpel effectively mitigates hallucinations, outperforming previous methods and achieving state-of-the-art performance. Moreover, Scalpel is model- and data-agnostic, requiring no additional computation, only a single decoding step.

  </details>


