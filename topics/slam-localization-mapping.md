# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-26 07:13 UTC_

Total papers shown: **19**


---

- **Dream-SLAM: Dreaming the Unseen for Active SLAM in Dynamic Environments**  
  Xiangqi Meng, Pengxu Hou, Zhenjun Zhao, Javier Civera, Daniel Cremers, Hesheng Wang, Haoang Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21967v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In addition to the core tasks of simultaneous localization and mapping (SLAM), active SLAM additionally in- volves generating robot actions that enable effective and efficient exploration of unknown environments. However, existing active SLAM pipelines are limited by three main factors. First, they inherit the restrictions of the underlying SLAM modules that they may be using. Second, their motion planning strategies are typically shortsighted and lack long-term vision. Third, most approaches struggle to handle dynamic scenes. To address these limitations, we propose a novel monocular active SLAM method, Dream-SLAM, which is based on dreaming cross-spatio-temporal images and semantically plausible structures of partially observed dynamic environments. The generated cross-spatio-temporal im- ages are fused with real observations to mitigate noise and data incompleteness, leading to more accurate camera pose estimation and a more coherent 3D scene representation. Furthermore, we integrate dreamed and observed scene structures to enable long- horizon planning, producing farsighted trajectories that promote efficient and thorough exploration. Extensive experiments on both public and self-collected datasets demonstrate that Dream-SLAM outperforms state-of-the-art methods in localization accuracy, mapping quality, and exploration efficiency. Source code will be publicly available upon paper acceptance.

  </details>



- **Parallel Continuous-Time Relative Localization with Augmented Clamped Non-Uniform B-Splines**  
  Jiadong Lu, Zhehan Li, Tao Han, Miao Xu, Chao Xu, Yanjun Cao  
  _2026-02-25_ · https://arxiv.org/abs/2602.22006v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate relative localization is critical for multi-robot cooperation. In robot swarms, measurements from different robots arrive asynchronously and with clock time-offsets. Although Continuous-Time (CT) formulations have proved effective for handling asynchronous measurements in single-robot SLAM and calibration, extending CT methods to multi-robot settings faces great challenges to achieve high-accuracy, low-latency, and high-frequency performance. Especially, existing CT methods suffer from the inherent query-time delay of unclamped B-splines and high computational cost. This paper proposes CT-RIO, a novel Continuous-Time Relative-Inertial Odometry framework. We employ Clamped Non-Uniform B-splines (C-NUBS) to represent robot states for the first time, eliminating the query-time delay. We further augment C-NUBS with closed-form extension and shrinkage operations that preserve the spline shape, making it suitable for online estimation and enabling flexible knot management. This flexibility leads to the concept of knot-keyknot strategy, which supports spline extension at high-frequency while retaining sparse keyknots for adaptive relative-motion modeling. We then formulate a sliding-window relative localization problem that operates purely on relative kinematics and inter-robot constraints. To meet the demanding computation required at swarm scale, we decompose the tightly-coupled optimization into robot-wise sub-problems and solve them in parallel using incremental asynchronous block coordinate descent. Extensive experiments show that CT-RIO converges from time-offsets as large as 263 ms to sub-millisecond within 3 s, and achieves RMSEs of 0.046 m and 1.8 °. It consistently outperforms state-of-the-art methods, with improvements of up to 60% under high-speed motion.

  </details>



- **DAGS-SLAM: Dynamic-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty-Aware Scheduling**  
  Li Zhang, Yu-An Liu, Xijia Jiang, Conghao Huang, Danyang Li, Yanyong Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21644v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mobile robots and IoT devices demand real-time localization and dense reconstruction under tight compute and energy budgets. While 3D Gaussian Splatting (3DGS) enables efficient dense SLAM, dynamic objects and occlusions still degrade tracking and mapping. Existing dynamic 3DGS-SLAM often relies on heavy optical flow and per-frame segmentation, which is costly for mobile deployment and brittle under challenging illumination. We present DAGS-SLAM, a dynamic-aware 3DGS-SLAM system that maintains a spatiotemporal motion probability (MP) state per Gaussian and triggers semantics on demand via an uncertainty-aware scheduler. DAGS-SLAM fuses lightweight YOLO instance priors with geometric cues to estimate and temporally update MP, propagates MP to the front-end for dynamic-aware correspondence selection, and suppresses dynamic artifacts in the back-end via MP-guided optimization. Experiments on public dynamic RGB-D benchmarks show improved reconstruction and robust tracking while sustaining real-time throughput on a commodity GPU, demonstrating a practical speed-accuracy tradeoff with reduced semantic invocations toward mobile deployment.

  </details>



- **UNet-Based Keypoint Regression for 3D Cone Localization in Autonomous Racing**  
  Mariia Baidachna, James Carty, Aidan Ferguson, Joseph Agrane, Varad Kulkarni, Aubrey Agub, Michael Baxendale, Aaron David, Rachel Horton, Elliott Atkinson  
  _2026-02-25_ · https://arxiv.org/abs/2602.21904v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate cone localization in 3D space is essential in autonomous racing for precise navigation around the track. Approaches that rely on traditional computer vision algorithms are sensitive to environmental variations, and neural networks are often trained on limited data and are infeasible to run in real time. We present a UNet-based neural network for keypoint detection on cones, leveraging the largest custom-labeled dataset we have assembled. Our approach enables accurate cone position estimation and the potential for color prediction. Our model achieves substantial improvements in keypoint accuracy over conventional methods. Furthermore, we leverage our predicted keypoints in the perception pipeline and evaluate the end-to-end autonomous system. Our results show high-quality performance across all metrics, highlighting the effectiveness of this approach and its potential for adoption in competitive autonomous racing systems.

  </details>



- **Enhancing Cellular-enabled Collaborative Robots Planning through GNSS data for SAR Scenarios**  
  Arnau Romero, Carmen Delgado, Jana Baguer, Raúl Suárez, Xavier Costa-Pérez  
  _2026-02-25_ · https://arxiv.org/abs/2602.21899v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Cellular-enabled collaborative robots are becoming paramount in Search-and-Rescue (SAR) and emergency response. Crucially dependent on resilient mobile network connectivity, they serve as invaluable assets for tasks like rapid victim localization and the exploration of hazardous, otherwise unreachable areas. However, their reliance on battery power and the need for persistent, low-latency communication limit operational time and mobility. To address this, and considering the evolving capabilities of 5G/6G networks, we propose a novel SAR framework that includes Mission Planning and Mission Execution phases and that optimizes robot deployment. By considering parameters such as the exploration area size, terrain elevation, robot fleet size, communication-influenced energy profiles, desired exploration rate, and target response time, our framework determines the minimum number of robots required and their optimal paths to ensure effective coverage and timely data backhaul over mobile networks. Our results demonstrate the trade-offs between number of robots, explored area, and response time for wheeled and quadruped robots. Further, we quantify the impact of terrain elevation data on mission time and energy consumption, showing the benefits of incorporating real-world environmental factors that might also affect mobile signal propagation and connectivity into SAR planning. This framework provides critical insights for leveraging next-generation mobile networks to enhance autonomous SAR operations.

  </details>



- **AdaSpot: Spend Resolution Where It Matters for Precise Event Spotting**  
  Artur Xarles, Sergio Escalera, Thomas B. Moeslund, Albert Clapés  
  _2026-02-25_ · https://arxiv.org/abs/2602.22073v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Precise Event Spotting aims to localize fast-paced actions or events in videos with high temporal precision, a key task for applications in sports analytics, robotics, and autonomous systems. Existing methods typically process all frames uniformly, overlooking the inherent spatio-temporal redundancy in video data. This leads to redundant computation on non-informative regions while limiting overall efficiency. To remain tractable, they often spatially downsample inputs, losing fine-grained details crucial for precise localization. To address these limitations, we propose \textbf{AdaSpot}, a simple yet effective framework that processes low-resolution videos to extract global task-relevant features while adaptively selecting the most informative region-of-interest in each frame for high-resolution processing. The selection is performed via an unsupervised, task-aware strategy that maintains spatio-temporal consistency across frames and avoids the training instability of learnable alternatives. This design preserves essential fine-grained visual cues with a marginal computational overhead compared to low-resolution-only baselines, while remaining far more efficient than uniform high-resolution processing. Experiments on standard PES benchmarks demonstrate that \textbf{AdaSpot} achieves state-of-the-art performance under strict evaluation metrics (\eg, $+3.96$ and $+2.26$ mAP$@0$ frames on Tennis and FineDiving), while also maintaining strong results under looser metrics. Code is available at: \href{https://github.com/arturxe2/AdaSpot}{https://github.com/arturxe2/AdaSpot}.

  </details>



- **RGB-Event HyperGraph Prompt for Kilometer Marker Recognition based on Pre-trained Foundation Models**  
  Xiaoyu Xian, Shiao Wang, Xiao Wang, Daxin Tian, Yan Tian  
  _2026-02-25_ · https://arxiv.org/abs/2602.22026v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Metro trains often operate in highly complex environments, characterized by illumination variations, high-speed motion, and adverse weather conditions. These factors pose significant challenges for visual perception systems, especially those relying solely on conventional RGB cameras. To tackle these difficulties, we explore the integration of event cameras into the perception system, leveraging their advantages in low-light conditions, high-speed scenarios, and low power consumption. Specifically, we focus on Kilometer Marker Recognition (KMR), a critical task for autonomous metro localization under GNSS-denied conditions. In this context, we propose a robust baseline method based on a pre-trained RGB OCR foundation model, enhanced through multi-modal adaptation. Furthermore, we construct the first large-scale RGB-Event dataset, EvMetro5K, containing 5,599 pairs of synchronized RGB-Event samples, split into 4,479 training and 1,120 testing samples. Extensive experiments on EvMetro5K and other widely used benchmarks demonstrate the effectiveness of our approach for KMR. Both the dataset and source code will be released on https://github.com/Event-AHU/EvMetro5K_benchmark

  </details>



- **SF3D-RGB: Scene Flow Estimation from Monocular Camera and Sparse LiDAR**  
  Rajai Alhimdiat, Ramy Battrawy, René Schuster, Didier Stricker, Wesam Ashour  
  _2026-02-25_ · https://arxiv.org/abs/2602.21699v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Scene flow estimation is an extremely important task in computer vision to support the perception of dynamic changes in the scene. For robust scene flow, learning-based approaches have recently achieved impressive results using either image-based or LiDAR-based modalities. However, these methods have tended to focus on the use of a single modality. To tackle these problems, we present a deep learning architecture, SF3D-RGB, that enables sparse scene flow estimation using 2D monocular images and 3D point clouds (e.g., acquired by LiDAR) as inputs. Our architecture is an end-to-end model that first encodes information from each modality into features and fuses them together. Then, the fused features enhance a graph matching module for better and more robust mapping matrix computation to generate an initial scene flow. Finally, a residual scene flow module further refines the initial scene flow. Our model is designed to strike a balance between accuracy and efficiency. Furthermore, experiments show that our proposed method outperforms single-modality methods and achieves better scene flow accuracy on real-world datasets while using fewer parameters compared to other state-of-the-art methods with fusion.

  </details>



- **SunnyParking: Multi-Shot Trajectory Generation and Motion State Awareness for Human-like Parking**  
  Jishu Miao, Han Chen, Jiankun Zhai, Qi Liu, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi  
  _2026-02-25_ · https://arxiv.org/abs/2602.21682v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous parking fundamentally differs from on-road driving due to its frequent direction changes and complex maneuvering requirements. However, existing End-to-End (E2E) planning methods often simplify the parking task into a geometric path regression problem, neglecting explicit modeling of the vehicle's kinematic state. This "dimensionality deficiency" easily leads to physically infeasible trajectories and deviates from real human driving behavior, particularly at critical gear-shift points in multi-shot parking scenarios. In this paper, we propose SunnyParking, a novel dual-branch E2E architecture that achieves motion state awareness by jointly predicting spatial trajectories and discrete motion state sequences (e.g., forward/reverse). Additionally, we introduce a Fourier feature-based representation of target parking slots to overcome the resolution limitations of traditional bird's-eye view (BEV) approaches, enabling high-precision target interactions. Experimental results demonstrate that our framework generates more robust and human-like trajectories in complex multi-shot parking scenarios, while significantly improving gear-shift point localization accuracy compared to state-of-the-art methods. We open-source a new parking dataset of the CARLA simulator, specifically designed to evaluate full prediction capabilities under complex maneuvers.

  </details>



- **Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models**  
  Zheyuan Gu, Qingsong Zhao, Yusong Wang, Zhaohong Huang, Xinqi Li, Cheng Yuan, Jiaowei Shao, Chi Zhang, Xuelong Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21779v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Current Vision-Language Models (VLMs) for deepfake detection excel at identifying spatial artifacts but overlook a critical dimension: temporal inconsistencies in video forgeries. Adapting VLMs to reason about these dynamic cues remains a distinct challenge. To bridge this gap, we propose Forensic Answer-Questioning (FAQ), a large-scale benchmark that formulates temporal deepfake analysis as a multiple-choice task. FAQ introduces a three-level hierarchy to progressively evaluate and equip VLMs with forensic capabilities: (1) Facial Perception, testing the ability to identify static visual artifacts; (2) Temporal Deepfake Grounding, requiring the localization of dynamic forgery artifacts across frames; and (3) Forensic Reasoning, challenging models to synthesize evidence for final authenticity verdicts. We evaluate a range of VLMs on FAQ and generate a corresponding instruction-tuning set, FAQ-IT. Extensive experiments show that models fine-tuned on FAQ-IT achieve advanced performance on both in-domain and cross-dataset detection benchmarks. Ablation studies further validate the impact of our key design choices, confirming that FAQ is the driving force behind the temporal reasoning capabilities of these VLMs.

  </details>



- **Tacmap: Bridging the Tactile Sim-to-Real Gap via Geometry-Consistent Penetration Depth Map**  
  Lei Su, Zhijie Peng, Renyuan Ren, Shengping Mao, Juan Du, Kaifeng Zhang, Xuezhou Zhu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21625v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Based Tactile Sensors (VBTS) are essential for achieving dexterous robotic manipulation, yet the tactile sim-to-real gap remains a fundamental bottleneck. Current tactile simulations suffer from a persistent dilemma: simplified geometric projections lack physical authenticity, while high-fidelity Finite Element Methods (FEM) are too computationally prohibitive for large-scale reinforcement learning. In this work, we present Tacmap, a high-fidelity, computationally efficient tactile simulation framework anchored in volumetric penetration depth. Our key insight is to bridge the tactile sim-to-real gap by unifying both domains through a shared deform map representation. Specifically, we compute 3D intersection volumes as depth maps in simulation, while in the real world, we employ an automated data-collection rig to learn a robust mapping from raw tactile images to ground-truth depth maps. By aligning simulation and real-world in this unified geometric space, Tacmap minimizes domain shift while maintaining physical consistency. Quantitative evaluations across diverse contact scenarios demonstrate that Tacmap's deform maps closely mirror real-world measurements. Moreover, we validate the utility of Tacmap through an in-hand rotation task, where a policy trained exclusively in simulation achieves zero-shot transfer to a physical robot.

  </details>



- **Recovered in Translation: Efficient Pipeline for Automated Translation of Benchmarks and Datasets**  
  Hanna Yukhymenko, Anton Alexandrov, Martin Vechev  
  _2026-02-25_ · https://arxiv.org/abs/2602.22207v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  The reliability of multilingual Large Language Model (LLM) evaluation is currently compromised by the inconsistent quality of translated benchmarks. Existing resources often suffer from semantic drift and context loss, which can lead to misleading performance metrics. In this work, we present a fully automated framework designed to address these challenges by enabling scalable, high-quality translation of datasets and benchmarks. We demonstrate that adapting test-time compute scaling strategies, specifically Universal Self-Improvement (USI) and our proposed multi-round ranking method, T-RANK, allows for significantly higher quality outputs compared to traditional pipelines. Our framework ensures that benchmarks preserve their original task structure and linguistic nuances during localization. We apply this approach to translate popular benchmarks and datasets into eight Eastern and Southern European languages (Ukrainian, Bulgarian, Slovak, Romanian, Lithuanian, Estonian, Turkish, Greek). Evaluations using both reference-based metrics and LLM-as-a-judge show that our translations surpass existing resources, resulting in more accurate downstream model assessment. We release both the framework and the improved benchmarks to facilitate robust and reproducible multilingual AI development.

  </details>



- **CoLoGen: Progressive Learning of Concept`-`Localization Duality for Unified Image Generation**  
  YuXin Song, Yu Lu, Haoyuan Sun, Huanjin Yao, Fanglong Liu, Yifan Sun, Haocheng Feng, Hang Zhou, Jingdong Wang  
  _2026-02-25_ · https://arxiv.org/abs/2602.22150v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Unified conditional image generation remains difficult because different tasks depend on fundamentally different internal representations. Some require conceptual understanding for semantic synthesis, while others rely on localization cues for spatial precision. Forcing these heterogeneous tasks to share a single representation leads to concept`-`localization representational conflict. To address this issue, we propose CoLoGen, a unified diffusion framework that progressively learns and reconciles this concept`-`localization duality. CoLoGen uses a staged curriculum that first builds core conceptual and localization abilities, then adapts them to diverse visual conditions, and finally refines their synergy for complex instruction`-`driven tasks. Central to this process is the Progressive Representation Weaving (PRW) module, which dynamically routes features to specialized experts and stably integrates their outputs across stages. Experiments on editing, controllable generation, and customized generation show that CoLoGen achieves competitive or superior performance, offering a principled representational perspective for unified image generation.

  </details>



- **Probing the Geometry of Diffusion Models with the String Method**  
  Elio Moreau, Florentin Coeurdoux, Grégoire Ferre, Eric Vanden-Eijnden  
  _2026-02-25_ · https://arxiv.org/abs/2602.22122v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  Understanding the geometry of learned distributions is fundamental to improving and interpreting diffusion models, yet systematic tools for exploring their landscape remain limited. Standard latent-space interpolations fail to respect the structure of the learned distribution, often traversing low-density regions. We introduce a framework based on the string method that computes continuous paths between samples by evolving curves under the learned score function. Operating on pretrained models without retraining, our approach interpolates between three regimes: pure generative transport, which yields continuous sample paths; gradient-dominated dynamics, which recover minimum energy paths (MEPs); and finite-temperature string dynamics, which compute principal curves -- self-consistent paths that balance energy and entropy. We demonstrate that the choice of regime matters in practice. For image diffusion models, MEPs contain high-likelihood but unrealistic ''cartoon'' images, confirming prior observations that likelihood maxima appear unrealistic; principal curves instead yield realistic morphing sequences despite lower likelihood. For protein structure prediction, our method computes transition pathways between metastable conformers directly from models trained on static structures, yielding paths with physically plausible intermediates. Together, these results establish the string method as a principled tool for probing the modal structure of diffusion models -- identifying modes, characterizing barriers, and mapping connectivity in complex learned distributions.

  </details>



- **Brain3D: Brain Report Automation via Inflated Vision Transformers in 3D**  
  Mariano Barone, Francesco Di Serio, Giuseppe Riccio, Antonio Romano, Marco Postiglione, Antonino Ferraro, Vincenzo Moscato  
  _2026-02-25_ · https://arxiv.org/abs/2602.22098v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Current medical vision-language models (VLMs) process volumetric brain MRI using 2D slice-based approximations, fragmenting the spatial context required for accurate neuroradiological interpretation. We developed \textbf{Brain3D}, a staged vision-language framework for automated radiology report generation from 3D brain tumor MRI. Our approach inflates a pretrained 2D medical encoder into a native 3D architecture and progressively aligns it with a causal language model through three stages: contrastive grounding, supervised projector warmup, and LoRA-based linguistic specialization. Unlike generalist 3D medical VLMs, \textbf{Brain3D} is tailored to neuroradiology, where hemispheric laterality, tumor infiltration patterns, and anatomical localization are critical. Evaluated on 468 subjects (BraTS pathological cases plus healthy controls), our model achieves a Clinical Pathology F1 of 0.951 versus 0.413 for a strong 2D baseline while maintaining perfect specificity on healthy scans. The staged alignment proves essential: contrastive grounding establishes visual-textual correspondence, projector warmup stabilizes conditioning, and LoRA adaptation shifts output from verbose captions to structured clinical reports\footnote{Our code is publicly available for transparency and reproducibility

  </details>



- **When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters**  
  Liangwei Lyu, Jiaqi Xu, Jianwei Ding, Qiyao Deng  
  _2026-02-25_ · https://arxiv.org/abs/2602.21977v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Low-Rank Adaptation (LoRA) has emerged as a leading technique for efficiently fine-tuning text-to-image diffusion models, and its widespread adoption on open-source platforms has fostered a vibrant culture of model sharing and customization. However, the same modular and plug-and-play flexibility that makes LoRA appealing also introduces a broader attack surface. To highlight this risk, we propose Masquerade-LoRA (MasqLoRA), the first systematic attack framework that leverages an independent LoRA module as the attack vehicle to stealthily inject malicious behavior into text-to-image diffusion models. MasqLoRA operates by freezing the base model parameters and updating only the low-rank adapter weights using a small number of "trigger word-target image" pairs. This enables the attacker to train a standalone backdoor LoRA module that embeds a hidden cross-modal mapping: when the module is loaded and a specific textual trigger is provided, the model produces a predefined visual output; otherwise, it behaves indistinguishably from the benign model, ensuring the stealthiness of the attack. Experimental results demonstrate that MasqLoRA can be trained with minimal resource overhead and achieves a high attack success rate of 99.8%. MasqLoRA reveals a severe and unique threat in the AI supply chain, underscoring the urgent need for dedicated defense mechanisms for the LoRA-centric sharing ecosystem.

  </details>



- **TIRAuxCloud: A Thermal Infrared Dataset for Day and Night Cloud Detection**  
  Alexis Apostolakis, Vasileios Botsos, Niklas Wölki, Andrea Spichtinger, Nikolaos Ioannis Bountos, Ioannis Papoutsis, Panayiotis Tsanakas  
  _2026-02-25_ · https://arxiv.org/abs/2602.21905v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Clouds are a major obstacle in Earth observation, limiting the usability and reliability of critical remote sensing applications such as fire disaster response, urban heat island monitoring, and snow and ice cover mapping. Therefore, the ability to detect clouds 24/7 is of paramount importance. While visible and near-infrared bands are effective for daytime cloud detection, their dependence on solar illumination makes them unsuitable for nighttime monitoring. In contrast, thermal infrared (TIR) imagery plays a crucial role in detecting clouds at night, when sunlight is absent. Due to their generally lower temperatures, clouds emit distinct thermal signatures that are detectable in TIR bands. Despite this, accurate nighttime cloud detection remains challenging due to limited spectral information and the typically lower spatial resolution of TIR imagery. To address these challenges, we present TIRAuxCloud, a multi-modal dataset centered around thermal spectral data to facilitate cloud segmentation under both daytime and nighttime conditions. The dataset comprises a unique combination of multispectral data (TIR, optical, and near-infrared bands) from Landsat and VIIRS, aligned with auxiliary information layers. Elevation, land cover, meteorological variables, and cloud-free reference images are included to help reduce surface-cloud ambiguity and cloud formation uncertainty. To overcome the scarcity of manual cloud labels, we include a large set of samples with automated cloud masks and a smaller manually annotated subset to further evaluate and improve models. Comprehensive benchmarks are presented to establish performance baselines through supervised and transfer learning, demonstrating the dataset's value in advancing the development of innovative methods for day and night time cloud detection.

  </details>



- **From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors**  
  Liangbing Zhao, Le Zhuo, Sayak Paul, Hongsheng Li, Mohamed Elhoseiny  
  _2026-02-25_ · https://arxiv.org/abs/2602.21778v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Instruction-based image editing has achieved remarkable success in semantic alignment, yet state-of-the-art models frequently fail to render physically plausible results when editing involves complex causal dynamics, such as refraction or material deformation. We attribute this limitation to the dominant paradigm that treats editing as a discrete mapping between image pairs, which provides only boundary conditions and leaves transition dynamics underspecified. To address this, we reformulate physics-aware editing as predictive physical state transitions and introduce PhysicTran38K, a large-scale video-based dataset comprising 38K transition trajectories across five physical domains, constructed via a two-stage filtering and constraint-aware annotation pipeline. Building on this supervision, we propose PhysicEdit, an end-to-end framework equipped with a textual-visual dual-thinking mechanism. It combines a frozen Qwen2.5-VL for physically grounded reasoning with learnable transition queries that provide timestep-adaptive visual guidance to a diffusion backbone. Experiments show that PhysicEdit improves over Qwen-Image-Edit by 5.9% in physical realism and 10.1% in knowledge-grounded editing, setting a new state-of-the-art for open-source methods, while remaining competitive with leading proprietary models.

  </details>



- **PPCR-IM: A System for Multi-layer DAG-based Public Policy Consequence Reasoning and Social Indicator Mapping**  
  Zichen Song, Weijia Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21650v1 · `cs.SI`  
  <details><summary>Abstract</summary>

  Public policy decisions are typically justified using a narrow set of headline indicators, leaving many downstream social impacts unstructured and difficult to compare across policies. We propose PPCR-IM, a system for multi-layer DAG-based consequence reasoning and social indicator mapping that addresses this gap. Given a policy description and its context, PPCR-IM uses an LLM-driven, layer-wise generator to construct a directed acyclic graph of intermediate consequences, allowing child nodes to have multiple parents to capture joint influences. A mapping module then aligns these nodes to a fixed indicator set and assigns one of three qualitative impact directions: increase, decrease, or ambiguous change. For each policy episode, the system outputs a structured record containing the DAG, indicator mappings, and three evaluation measures: an expected-indicator coverage score, a discovery rate for overlooked but relevant indicators, and a relative focus ratio comparing the systems coverage to that of the government. PPCR-IM is available both as an online demo and as a configurable XLSX-to-JSON batch pipeline.

  </details>


