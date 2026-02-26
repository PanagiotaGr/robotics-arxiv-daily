# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-26 07:13 UTC_

Total papers shown: **8**


---

- **SunnyParking: Multi-Shot Trajectory Generation and Motion State Awareness for Human-like Parking**  
  Jishu Miao, Han Chen, Jiankun Zhai, Qi Liu, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi  
  _2026-02-25_ · https://arxiv.org/abs/2602.21682v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous parking fundamentally differs from on-road driving due to its frequent direction changes and complex maneuvering requirements. However, existing End-to-End (E2E) planning methods often simplify the parking task into a geometric path regression problem, neglecting explicit modeling of the vehicle's kinematic state. This "dimensionality deficiency" easily leads to physically infeasible trajectories and deviates from real human driving behavior, particularly at critical gear-shift points in multi-shot parking scenarios. In this paper, we propose SunnyParking, a novel dual-branch E2E architecture that achieves motion state awareness by jointly predicting spatial trajectories and discrete motion state sequences (e.g., forward/reverse). Additionally, we introduce a Fourier feature-based representation of target parking slots to overcome the resolution limitations of traditional bird's-eye view (BEV) approaches, enabling high-precision target interactions. Experimental results demonstrate that our framework generates more robust and human-like trajectories in complex multi-shot parking scenarios, while significantly improving gear-shift point localization accuracy compared to state-of-the-art methods. We open-source a new parking dataset of the CARLA simulator, specifically designed to evaluate full prediction capabilities under complex maneuvers.

  </details>



- **Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos**  
  Matthew Strong, Wei-Jer Chang, Quentin Herau, Jiezhi Yang, Yihan Hu, Chensheng Peng, Wei Zhan  
  _2026-02-25_ · https://arxiv.org/abs/2602.22091v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Ego-centric driving videos available online provide an abundant source of visual data for autonomous driving, yet their lack of annotations makes it difficult to learn representations that capture both semantic structure and 3D geometry. Recent advances in large feedforward spatial models demonstrate that point maps and ego-motion can be inferred in a single forward pass, suggesting a promising direction for scalable driving perception. We therefore propose a label-free, teacher-guided framework for learning autonomous driving representations directly from unposed videos. Unlike prior self-supervised approaches that focus primarily on frame-to-frame consistency, we posit that safe and reactive driving depends critically on temporal context. To this end, we leverage a feedforward architecture equipped with a lightweight autoregressive module, trained using multi-modal supervisory signals that guide the model to jointly predict current and future point maps, camera poses, semantic segmentation, and motion masks. Multi-modal teachers provide sequence-level pseudo-supervision, enabling LFG to learn a unified pseudo-4D representation from raw YouTube videos without poses, labels, or LiDAR. The resulting encoder not only transfers effectively to downstream autonomous driving planning on the NAVSIM benchmark, surpassing multi-camera and LiDAR baselines with only a single monocular camera, but also yields strong performance when evaluated on a range of semantic, geometric, and qualitative motion prediction tasks. These geometry and motion-aware features position LFG as a compelling video-centric foundation model for autonomous driving.

  </details>



- **MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving**  
  Lingjun Zhang, Yujian Yuan, Changjie Wu, Xinyuan Chang, Xin Cai, Shuang Zeng, Linzhe Shi, Sijin Wang, Hang Zhang, Mu Xu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21952v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLM) exhibit strong reasoning capabilities, showing promise for end-to-end autonomous driving systems. Chain-of-Thought (CoT), as VLM's widely used reasoning strategy, is facing critical challenges. Existing textual CoT has a large gap between text semantic space and trajectory physical space. Although the recent approach utilizes future image to replace text as CoT process, it lacks clear planning-oriented objective guidance to generate images with accurate scene evolution. To address these, we innovatively propose MindDriver, a progressive multimodal reasoning framework that enables VLM to imitate human-like progressive thinking for autonomous driving. MindDriver presents semantic understanding, semantic-to-physical space imagination, and physical-space trajectory planning. To achieve aligned reasoning processes in MindDriver, we develop a feedback-guided automatic data annotation pipeline to generate aligned multimodal reasoning training data. Furthermore, we develop a progressive reinforcement fine-tuning method to optimize the alignment through progressive high- level reward-based learning. MindDriver demonstrates superior performance in both nuScences open-loop and Bench2Drive closed-loop evaluation. Codes are available at https://github.com/hotdogcheesewhite/MindDriver.

  </details>



- **PanoEnv: Exploring 3D Spatial Intelligence in Panoramic Environments with Reinforcement Learning**  
  Zekai Lin, Xu Zheng  
  _2026-02-25_ · https://arxiv.org/abs/2602.21992v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  360 panoramic images are increasingly used in virtual reality, autonomous driving, and robotics for holistic scene understanding. However, current Vision-Language Models (VLMs) struggle with 3D spatial reasoning on Equirectangular Projection (ERP) images due to geometric distortion and limited 3D supervision. We introduce PanoEnv, a large-scale VQA benchmark built from synthetic 3D environments, containing 14.8K questions across five categories (e.g., relative position, volume comparison) grounded in accurate 3D annotations including depth, segmentation, and bounding boxes. Benchmarking 14 state-of-the-art VLMs reveals limited 3D understanding, achieving only 49.34% overall accuracy and 8.36% on open-ended (OE) questions. To enhance 3D reasoning, we propose a reinforcement learning post-training framework based on Group Relative Policy Optimization (GRPO) with a ground-truth-guided reward that incorporates five geometry-aware strategies such as distance tolerance and spatial consistency. A two-stage curriculum further mitigates catastrophic forgetting: Stage 1 trains on structured tasks (true/false and multiple choice), and Stage 2 fine-tunes on mixed open-ended data to improve generalization. Our 7B model achieves new state-of-the-art performance, improving overall accuracy to 52.93% (+3.59%) and open-ended accuracy to 14.83% while maintaining structured-task performance. It also achieves top semantic evaluation scores (Q-Score 6.24, P-Score 5.95), surpassing 32B models. These results demonstrate that PanoEnv-QA and our curriculum-based RL framework effectively instill 3D spatial intelligence in VLMs for omnidirectional perception.

  </details>



- **LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration**  
  Aditya Ranjan Dash, Ramy Battrawy, René Schuster, Didier Stricker  
  _2026-02-25_ · https://arxiv.org/abs/2602.21754v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Advanced autonomous systems rely on multi-sensor fusion for safer and more robust perception. To enable effective fusion, calibrating directly from natural driving scenes (i.e., target-free) with high accuracy is crucial for precise multi-sensor alignment. Existing learning-based calibration methods are typically designed for only a single pair of sensor modalities (i.e., a bi-modal setup). Unlike these methods, we propose LiREC-Net, a target-free, learning-based calibration network that jointly calibrates multiple sensor modality pairs, including LiDAR, RGB, and event data, within a unified framework. To reduce redundant computation and improve efficiency, we introduce a shared LiDAR representation that leverages features from both its 3D nature and projected depth map, ensuring better consistency across modalities. Trained and evaluated on established datasets, such as KITTI and DSEC, our LiREC-Net achieves competitive performance to bi-modal models and sets a new strong baseline for the tri-modal use case.

  </details>



- **WeatherCity: Urban Scene Reconstruction with Controllable Multi-Weather Transformation**  
  Wenhua Wu, Huai Guan, Zhe Liu, Hesheng Wang  
  _2026-02-25_ · https://arxiv.org/abs/2602.22096v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Editable high-fidelity 4D scenes are crucial for autonomous driving, as they can be applied to end-to-end training and closed-loop simulation. However, existing reconstruction methods are primarily limited to replicating observed scenes and lack the capability for diverse weather simulation. While image-level weather editing methods tend to introduce scene artifacts and offer poor controllability over the weather effects. To address these limitations, we propose WeatherCity, a novel framework for 4D urban scene reconstruction and weather editing. Specifically, we leverage a text-guided image editing model to achieve flexible editing of image weather backgrounds. To tackle the challenge of multi-weather modeling, we introduce a novel weather Gaussian representation based on shared scene features and dedicated weather-specific decoders. This representation is further enhanced with a content consistency optimization, ensuring coherent modeling across different weather conditions. Additionally, we design a physics-driven model that simulates dynamic weather effects through particles and motion patterns. Extensive experiments on multiple datasets and various scenes demonstrate that WeatherCity achieves flexible controllability, high fidelity, and temporal consistency in 4D reconstruction and weather editing. Our framework not only enables fine-grained control over weather conditions (e.g., light rain and heavy snow) but also supports object-level manipulation within the scene.

  </details>



- **RT-RMOT: A Dataset and Framework for RGB-Thermal Referring Multi-Object Tracking**  
  Yanqiu Yu, Zhifan Jin, Sijia Chen, Tongfei Chu, En Yu, Liman Liu, Wenbing Tao  
  _2026-02-25_ · https://arxiv.org/abs/2602.22033v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Referring Multi-Object Tracking has attracted increasing attention due to its human-friendly interactive characteristics, yet it exhibits limitations in low-visibility conditions, such as nighttime, smoke, and other challenging scenarios. To overcome this limitation, we propose a new RGB-Thermal RMOT task, named RT-RMOT, which aims to fuse RGB appearance features with the illumination robustness of the thermal modality to enable all-day referring multi-object tracking. To promote research on RT-RMOT, we construct the first Referring Multi-Object Tracking dataset under RGB-Thermal modality, named RefRT. It contains 388 language descriptions, 1,250 tracked targets, and 166,147 Language-RGB-Thermal (L-RGB-T) triplets. Furthermore, we propose RTrack, a framework built upon a multimodal large language model (MLLM) that integrates RGB, thermal, and textual features. Since the initial framework still leaves room for improvement, we introduce a Group Sequence Policy Optimization (GSPO) strategy to further exploit the model's potential. To alleviate training instability during RL fine-tuning, we introduce a Clipped Advantage Scaling (CAS) strategy to suppress gradient explosion. In addition, we design Structured Output Reward and Comprehensive Detection Reward to balance exploration and exploitation, thereby improving the completeness and accuracy of target perception. Extensive experiments on the RefRT dataset demonstrate the effectiveness of the proposed RTrack framework.

  </details>



- **GFPL: Generative Federated Prototype Learning for Resource-Constrained and Data-Imbalanced Vision Task**  
  Shiwei Lu, Yuhang He, Jiashuo Li, Qiang Wang, Yihong Gong  
  _2026-02-25_ · https://arxiv.org/abs/2602.21873v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Federated learning (FL) facilitates the secure utilization of decentralized images, advancing applications in medical image recognition and autonomous driving. However, conventional FL faces two critical challenges in real-world deployment: ineffective knowledge fusion caused by model updates biased toward majority-class features, and prohibitive communication overhead due to frequent transmissions of high-dimensional model parameters. Inspired by the human brain's efficiency in knowledge integration, we propose a novel Generative Federated Prototype Learning (GFPL) framework to address these issues. Within this framework, a prototype generation method based on Gaussian Mixture Model (GMM) captures the statistical information of class-wise features, while a prototype aggregation strategy using Bhattacharyya distance effectively fuses semantically similar knowledge across clients. In addition, these fused prototypes are leveraged to generate pseudo-features, thereby mitigating feature distribution imbalance across clients. To further enhance feature alignment during local training, we devise a dual-classifier architecture, optimized via a hybrid loss combining Dot Regression and Cross-Entropy. Extensive experiments on benchmarks show that GFPL improves model accuracy by 3.6% under imbalanced data settings while maintaining low communication cost.

  </details>


