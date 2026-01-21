# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-01-21 06:53 UTC_

Total papers shown: **15**


---

- **DroneVLA: VLA based Aerial Manipulation**  
  Fawad Mehboob, Monijesu James, Amir Habel, Jeffrin Sam, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13809v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  As aerial platforms evolve from passive observers to active manipulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally. This work introduces a novel concept of autonomous aerial manipulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user. The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom-built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera. VLA performs semantic reasoning to interpret the intent of a user prompt and generates a prioritized task queue for grasping of relevant objects in the scene. Grounding DINO and dynamic A* planning algorithm are used to navigate and safely relocate the object. To ensure safe and natural interaction during the handover phase, the system employs a human-centric controller driven by MediaPipe. This module provides real-time human pose estimation, allowing the drone to employ visual servoing to maintain a stable, distinct position directly in front of the user, facilitating a comfortable handover. We demonstrate the system's efficacy through real-world experiments for localization and navigation, which resulted in a 0.164m, 0.070m, and 0.084m of max, mean euclidean, and root-mean squared errors, respectively, highlighting the feasibility of VLA for aerial manipulation operations.

  </details>



- **Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep Reinforcement Learning**  
  Myong-Yol Choi, Hankyoul Ko, Hanse Cho, Changseung Kim, Seunghwan Kim, Jaemin Seo, Hyondong Oh  
  _2026-01-20_ · https://arxiv.org/abs/2601.13657v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a deep reinforcement learning (DRL) based controller for collective navigation of unmanned aerial vehicle (UAV) swarms in communication-denied environments, enabling robust operation in complex, obstacle-rich environments. Inspired by biological swarms where informed individuals guide groups without explicit communication, we employ an implicit leader-follower framework. In this paradigm, only the leader possesses goal information, while follower UAVs learn robust policies using only onboard LiDAR sensing, without requiring any inter-agent communication or leader identification. Our system utilizes LiDAR point clustering and an extended Kalman filter for stable neighbor tracking, providing reliable perception independent of external positioning systems. The core of our approach is a DRL controller, trained in GPU-accelerated Nvidia Isaac Sim, that enables followers to learn complex emergent behaviors - balancing flocking and obstacle avoidance - using only local perception. This allows the swarm to implicitly follow the leader while robustly addressing perceptual challenges such as occlusion and limited field-of-view. The robustness and sim-to-real transfer of our approach are confirmed through extensive simulations and challenging real-world experiments with a swarm of five UAVs, which successfully demonstrated collective navigation across diverse indoor and outdoor environments without any communication or external localization.

  </details>



- **ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins**  
  Xinhao Liu, Yu Wang, Xiansheng Guo, Gordon Owusu Boateng, Yu Cao, Haonan Si, Xingchen Guo, Nirwan Ansari  
  _2026-01-20_ · https://arxiv.org/abs/2601.13706v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  High-fidelity parking-lot digital twins provide essential priors for path planning, collision checking, and perception validation in Automated Valet Parking (AVP). Yet robot-oriented reconstruction faces a trilemma: sparse forward-facing views cause weak parallax and ill-posed geometry; dynamic occlusions and extreme lighting hinder stable texture fusion; and neural rendering typically needs expensive offline optimization, violating edge-side streaming constraints. We propose ParkingTwin, a training-free, lightweight system for online streaming 3D reconstruction. First, OSM-prior-driven geometric construction uses OpenStreetMap semantic topology to directly generate a metric-consistent TSDF, replacing blind geometric search with deterministic mapping and avoiding costly optimization. Second, geometry-aware dynamic filtering employs a quad-modal constraint field (normal/height/depth consistency) to reject moving vehicles and transient occlusions in real time. Third, illumination-robust fusion in CIELAB decouples luminance and chromaticity via adaptive L-channel weighting and depth-gradient suppression, reducing seams under abrupt lighting changes. ParkingTwin runs at 30+ FPS on an entry-level GTX 1660. On a 68,000 m^2 real-world dataset, it achieves SSIM 0.87 (+16.0%), delivers about 15x end-to-end speedup, and reduces GPU memory by 83.3% compared with state-of-the-art 3D Gaussian Splatting (3DGS) that typically requires high-end GPUs (RTX 4090D). The system outputs explicit triangle meshes compatible with Unity/Unreal digital-twin pipelines. Project page: https://mihoutao-liu.github.io/ParkingTwin/

  </details>



- **FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation**  
  Jing Zuo, Lingzhou Mu, Fan Jiang, Chengcheng Ma, Mu Xu, Yonggang Qi  
  _2026-01-20_ · https://arxiv.org/abs/2601.13976v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Achieving human-level performance in Vision-and-Language Navigation (VLN) requires an embodied agent to jointly understand multimodal instructions and visual-spatial context while reasoning over long action sequences. Recent works, such as NavCoT and NavGPT-2, demonstrate the potential of Chain-of-Thought (CoT) reasoning for improving interpretability and long-horizon planning. Moreover, multimodal extensions like OctoNav-R1 and CoT-VLA further validate CoT as a promising pathway toward human-like navigation reasoning. However, existing approaches face critical drawbacks: purely textual CoTs lack spatial grounding and easily overfit to sparse annotated reasoning steps, while multimodal CoTs incur severe token inflation by generating imagined visual observations, making real-time navigation impractical. In this work, we propose FantasyVLN, a unified implicit reasoning framework that preserves the benefits of CoT reasoning without explicit token overhead. Specifically, imagined visual tokens are encoded into a compact latent space using a pretrained Visual AutoRegressor (VAR) during CoT reasoning training, and the model jointly learns from textual, visual, and multimodal CoT modes under a unified multi-CoT strategy. At inference, our model performs direct instruction-to-action mapping while still enjoying reasoning-aware representations. Extensive experiments on LH-VLN show that our approach achieves reasoning-aware yet real-time navigation, improving success rates and efficiency while reducing inference latency by an order of magnitude compared to explicit CoT methods.

  </details>



- **LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR**  
  Said Taghadouini, Adrien Cavaillès, Baptiste Aubertin  
  _2026-01-20_ · https://arxiv.org/abs/2601.14251v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present \textbf{LightOnOCR-2-1B}, a 1B-parameter end-to-end multilingual vision--language model that converts document images (e.g., PDFs) into clean, naturally ordered text without brittle OCR pipelines. Trained on a large-scale, high-quality distillation mix with strong coverage of scans, French documents, and scientific PDFs, LightOnOCR-2 achieves state-of-the-art results on OlmOCR-Bench while being 9$\times$ smaller and substantially faster than prior best-performing models. We further extend the output format to predict normalized bounding boxes for embedded images, introducing localization during pretraining via a resume strategy and refining it with RLVR using IoU-based rewards. Finally, we improve robustness with checkpoint averaging and task-arithmetic merging. We release model checkpoints under Apache 2.0, and publicly release the dataset and \textbf{LightOnOCR-bbox-bench} evaluation under their respective licenses.

  </details>



- **Robust Localization in OFDM-Based Massive MIMO through Phase Offset Calibration**  
  Qing Zhang, Adham Sakhnini, Robbert Beerten, Haoqiu Xiong, Zhuangzhuang Cui, Yang Miao, Sofie Pollin  
  _2026-01-20_ · https://arxiv.org/abs/2601.14244v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Accurate localization in Orthogonal Frequency Division Multiplexing (OFDM)-based massive Multiple-Input Multiple-Output (MIMO) systems depends critically on phase coherence across subcarriers and antennas. However, practical systems suffer from frequency-dependent and (spatial) antenna-dependent phase offsets, degrading localization accuracy. This paper analytically studies the impact of phase incoherence on localization performance under a static User Equipment (UE) and Line-of-Sight (LoS) scenario. We use two complementary tools. First, we derive the Cramér-Rao Lower Bound (CRLB) to quantify the theoretical limits under phase offsets. Then, we develop a Spatial Ambiguity Function (SAF)-based model to characterize ambiguity patterns. Simulation results reveal that spatial phase offsets severely degrade localization performance, while frequency phase offsets have a minor effect in the considered system configuration. To address this, we propose a robust Channel State Information (CSI) calibration framework and validate it using real-world measurements from a practical massive MIMO testbed. The experimental results confirm that the proposed calibration framework significantly improves the localization Root Mean Squared Error (RMSE) from 5 m to 1.2 cm, aligning well with the theoretical predictions.

  </details>



- **Learning to Explain: Supervised Token Attribution from Transformer Attention Patterns**  
  George Mihaila  
  _2026-01-20_ · https://arxiv.org/abs/2601.14112v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Explainable AI (XAI) has become critical as transformer-based models are deployed in high-stakes applications including healthcare, legal systems, and financial services, where opacity hinders trust and accountability. Transformers self-attention mechanisms have proven valuable for model interpretability, with attention weights successfully used to understand model focus and behavior (Xu et al., 2015); (Wiegreffe and Pinter, 2019). However, existing attention-based explanation methods rely on manually defined aggregation strategies and fixed attribution rules (Abnar and Zuidema, 2020a); (Chefer et al., 2021), while model-agnostic approaches (LIME, SHAP) treat the model as a black box and incur significant computational costs through input perturbation. We introduce Explanation Network (ExpNet), a lightweight neural network that learns an explicit mapping from transformer attention patterns to token-level importance scores. Unlike prior methods, ExpNet discovers optimal attention feature combinations automatically rather than relying on predetermined rules. We evaluate ExpNet in a challenging cross-task setting and benchmark it against a broad spectrum of model-agnostic methods and attention-based techniques spanning four methodological families.

  </details>



- **Causal feature selection framework for stable soft sensor modeling based on time-delayed cross mapping**  
  Shi-Shun Chen, Xiao-Yang Li, Enrico Zio  
  _2026-01-20_ · https://arxiv.org/abs/2601.14099v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Soft sensor modeling plays a crucial role in process monitoring. Causal feature selection can enhance the performance of soft sensor models in industrial applications. However, existing methods ignore two critical characteristics of industrial processes. Firstly, causal relationships between variables always involve time delays, whereas most causal feature selection methods investigate causal relationships in the same time dimension. Secondly, variables in industrial processes are often interdependent, which contradicts the decorrelation assumption of traditional causal inference methods. Consequently, soft sensor models based on existing causal feature selection approaches often lack sufficient accuracy and stability. To overcome these challenges, this paper proposes a causal feature selection framework based on time-delayed cross mapping. Time-delayed cross mapping employs state space reconstruction to effectively handle interdependent variables in causality analysis, and considers varying causal strength across time delay. Time-delayed convergent cross mapping (TDCCM) is introduced for total causal inference, and time-delayed partial cross mapping (TDPCM) is developed for direct causal inference. Then, in order to achieve automatic feature selection, an objective feature selection strategy is presented. The causal threshold is automatically determined based on the model performance on the validation set, and the causal features are then selected. Two real-world case studies show that TDCCM achieves the highest average performance, while TDPCM improves soft sensor stability and performance in the worst scenario. The code is publicly available at https://github.com/dirge1/TDPCM.

  </details>



- **Fine-Grained Zero-Shot Composed Image Retrieval with Complementary Visual-Semantic Integration**  
  Yongcong Ye, Kai Zhang, Yanghai Zhang, Enhong Chen, Longfei Li, Jun Zhou  
  _2026-01-20_ · https://arxiv.org/abs/2601.14060v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Zero-shot composed image retrieval (ZS-CIR) is a rapidly growing area with significant practical applications, allowing users to retrieve a target image by providing a reference image and a relative caption describing the desired modifications. Existing ZS-CIR methods often struggle to capture fine-grained changes and integrate visual and semantic information effectively. They primarily rely on either transforming the multimodal query into a single text using image-to-text models or employing large language models for target image description generation, approaches that often fail to capture complementary visual information and complete semantic context. To address these limitations, we propose a novel Fine-Grained Zero-Shot Composed Image Retrieval method with Complementary Visual-Semantic Integration (CVSI). Specifically, CVSI leverages three key components: (1) Visual Information Extraction, which not only extracts global image features but also uses a pre-trained mapping network to convert the image into a pseudo token, combining it with the modification text and the objects most likely to be added. (2) Semantic Information Extraction, which involves using a pre-trained captioning model to generate multiple captions for the reference image, followed by leveraging an LLM to generate the modified captions and the objects most likely to be added. (3) Complementary Information Retrieval, which integrates information extracted from both the query and database images to retrieve the target image, enabling the system to efficiently handle retrieval queries in a variety of situations. Extensive experiments on three public datasets (e.g., CIRR, CIRCO, and FashionIQ) demonstrate that CVSI significantly outperforms existing state-of-the-art methods. Our code is available at https://github.com/yyc6631/CVSI.

  </details>



- **Decoder-Free Supervoxel GNN for Accurate Brain-Tumor Localization in Multi-Modal MRI**  
  Andrea Protani, Marc Molina Van Den Bosch, Lorenzo Giusti, Heloisa Barbosa Da Silva, Paolo Cacace, Albert Sund Aillet, Miguel Angel Gonzalez Ballester, Friedhelm Hummel, Luigi Serio  
  _2026-01-20_ · https://arxiv.org/abs/2601.14055v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Modern vision backbones for 3D medical imaging typically process dense voxel grids through parameter-heavy encoder-decoder structures, a design that allocates a significant portion of its parameters to spatial reconstruction rather than feature learning. Our approach introduces SVGFormer, a decoder-free pipeline built upon a content-aware grouping stage that partitions the volume into a semantic graph of supervoxels. Its hierarchical encoder learns rich node representations by combining a patch-level Transformer with a supervoxel-level Graph Attention Network, jointly modeling fine-grained intra-region features and broader inter-regional dependencies. This design concentrates all learnable capacity on feature encoding and provides inherent, dual-scale explainability from the patch to the region level. To validate the framework's flexibility, we trained two specialized models on the BraTS dataset: one for node-level classification and one for tumor proportion regression. Both models achieved strong performance, with the classification model achieving a F1-score of 0.875 and the regression model a MAE of 0.028, confirming the encoder's ability to learn discriminative and localized features. Our results establish that a graph-based, encoder-only paradigm offers an accurate and inherently interpretable alternative for 3D medical image representation.

  </details>



- **Credible CO2 Comparisons: A Machine Learning Approach to Vehicle Powertrain Assessment**  
  Rodrigo Pereira David, Luciano Araujo Dourado Filho, Daniel Marques da Silva, João Alfredo Cal-Braz  
  _2026-01-20_ · https://arxiv.org/abs/2601.14022v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Decarbonizing road transport requires consistent and transparent methods for comparing CO2 emissions across vehicle technologies. This paper proposes a machine learning-based framework for like-for-like operational assessment of internal combustion engine vehicles (ICEVs) and electric vehicles (EVs) under identical, real-world driving conditions. The approach isolates technology-specific effects by holding the observed speed profile and environmental context fixed, enabling direct comparison of powertrain performance. Recurrent neural network models are trained independently for each domain to learn the mapping from contextual driving variables (speed, acceleration, temperature) to internal actuation variables (torque, throttle) and instantaneous CO2-equivalent emission rates. This structure allows the construction of counterfactual scenarios that answer: What emissions would an EV have generated if it had followed the same driving profile as an ICEV? By aligning both vehicle types on a unified instantaneous emissions metric, the framework enables fair and reproducible evaluation of powertrain technologies. It offers a scalable foundation for credible, data-driven assessments of vehicle carbon performance under real-world operating conditions.

  </details>



- **Auditory Brain Passage Retrieval: Cross-Sensory EEG Training for Neural Information Retrieval**  
  Niall McGuire, Yashar Moshfeghi  
  _2026-01-20_ · https://arxiv.org/abs/2601.14001v1 · `cs.IR`  
  <details><summary>Abstract</summary>

  Query formulation from internal information needs remains fundamentally challenging across all Information Retrieval paradigms due to cognitive complexity and physical impairments. Brain Passage Retrieval (BPR) addresses this by directly mapping EEG signals to passage representations without intermediate text translation. However, existing BPR research exclusively uses visual stimuli, leaving critical questions unanswered: Can auditory EEG enable effective retrieval for voice-based interfaces and visually impaired users? Can training on combined EEG datasets from different sensory modalities improve performance despite severe data scarcity? We present the first systematic investigation of auditory EEG for BPR and evaluate cross-sensory training benefits. Using dual encoder architectures with four pooling strategies (CLS, mean, max, multi-vector), we conduct controlled experiments comparing auditory-only, visual-only, and combined training on the Alice (auditory) and Nieuwland (visual) datasets. Results demonstrate that auditory EEG consistently outperforms visual EEG, and cross-sensory training with CLS pooling achieves substantial improvements over individual training: 31% in MRR (0.474), 43% in Hit@1 (0.314), and 28% in Hit@10 (0.858). Critically, combined auditory EEG models surpass BM25 text baselines (MRR: 0.474 vs 0.428), establishing neural queries as competitive with traditional retrieval whilst enabling accessible interfaces. These findings validate auditory neural interfaces for IR tasks and demonstrate that cross-sensory training addresses data scarcity whilst outperforming single-modality approaches Code: https://github.com/NiallMcguire/Audio_BPR

  </details>



- **DExTeR: Weakly Semi-Supervised Object Detection with Class and Instance Experts for Medical Imaging**  
  Adrien Meyer, Didier Mutter, Nicolas Padoy  
  _2026-01-20_ · https://arxiv.org/abs/2601.13954v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Detecting anatomical landmarks in medical imaging is essential for diagnosis and intervention guidance. However, object detection models rely on costly bounding box annotations, limiting scalability. Weakly Semi-Supervised Object Detection (WSSOD) with point annotations proposes annotating each instance with a single point, minimizing annotation time while preserving localization signals. A Point-to-Box teacher model, trained on a small box-labeled subset, converts these point annotations into pseudo-box labels to train a student detector. Yet, medical imagery presents unique challenges, including overlapping anatomy, variable object sizes, and elusive structures, which hinder accurate bounding box inference. To overcome these challenges, we introduce DExTeR (DETR with Experts), a transformer-based Point-to-Box regressor tailored for medical imaging. Built upon Point-DETR, DExTeR encodes single-point annotations as object queries, refining feature extraction with the proposed class-guided deformable attention, which guides attention sampling using point coordinates and class labels to capture class-specific characteristics. To improve discrimination in complex structures, it introduces CLICK-MoE (CLass, Instance, and Common Knowledge Mixture of Experts), decoupling class and instance representations to reduce confusion among adjacent or overlapping instances. Finally, we implement a multi-point training strategy which promotes prediction consistency across different point placements, improving robustness to annotation variability. DExTeR achieves state-of-the-art performance across three datasets spanning different medical domains (endoscopy, chest X-rays, and endoscopic ultrasound) highlighting its potential to reduce annotation costs while maintaining high detection accuracy.

  </details>



- **On the Role of Rotation Equivariance in Monocular 3D Human Pose Estimation**  
  Pavlo Melnyk, Cuong Le, Urs Waldmann, Per-Erik Forssén, Bastian Wandt  
  _2026-01-20_ · https://arxiv.org/abs/2601.13913v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Estimating 3D from 2D is one of the central tasks in computer vision. In this work, we consider the monocular setting, i.e. single-view input, for 3D human pose estimation (HPE). Here, the task is to predict a 3D point set of human skeletal joints from a single 2D input image. While by definition this is an ill-posed problem, recent work has presented methods that solve it with up to several-centimetre error. Typically, these methods employ a two-step approach, where the first step is to detect the 2D skeletal joints in the input image, followed by the step of 2D-to-3D lifting. We find that common lifting models fail when encountering a rotated input. We argue that learning a single human pose along with its in-plane rotations is considerably easier and more geometrically grounded than directly learning a point-to-point mapping. Furthermore, our intuition is that endowing the model with the notion of rotation equivariance without explicitly constraining its parameter space should lead to a more straightforward learning process than one with equivariance by design. Utilising the common HPE benchmarks, we confirm that the 2D rotation equivariance per se improves the model performance on human poses akin to rotations in the image plane, and can be efficiently and straightforwardly learned by augmentation, outperforming state-of-the-art equivariant-by-design methods.

  </details>



- **Towards Token-Level Text Anomaly Detection**  
  Yang Cao, Bicheng Yu, Sikun Yang, Ming Liu, Yujiu Yang  
  _2026-01-20_ · https://arxiv.org/abs/2601.13644v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Despite significant progress in text anomaly detection for web applications such as spam filtering and fake news detection, existing methods are fundamentally limited to document-level analysis, unable to identify which specific parts of a text are anomalous. We introduce token-level anomaly detection, a novel paradigm that enables fine-grained localization of anomalies within text. We formally define text anomalies at both document and token-levels, and propose a unified detection framework that operates across multiple levels. To facilitate research in this direction, we collect and annotate three benchmark datasets spanning spam, reviews and grammar errors with token-level labels. Experimental results demonstrate that our framework get better performance than other 6 baselines, opening new possibilities for precise anomaly localization in text. All the codes and data are publicly available on https://github.com/charles-cao/TokenCore.

  </details>


