# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-01-22 06:51 UTC_

Total papers shown: **20**


---

- **MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI**  
  Stavrow A. Bahnam, Robin Ferede, Till M. Blaha, Anton E. Lang, Erin Lucassen, Quentin Missinne, Aderik E. C. Verraest, Christophe De Wagter, Guido C. H. E. de Croon  
  _2026-01-21_ · https://arxiv.org/abs/2601.15222v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous drone racing represents a major frontier in robotics research. It requires an Artificial Intelligence (AI) that can run on board light-weight flying robots under tight resource and time constraints, while pushing the physical system to its limits. The state of the art in this area consists of a system with a stereo camera and an inertial measurement unit (IMU) that beat human drone racing champions in a controlled indoor environment. Here, we present MonoRace: an onboard drone racing approach that uses a monocular, rolling-shutter camera and IMU that generalizes to a competition environment without any external motion tracking system. The approach features robust state estimation that combines neural-network-based gate segmentation with a drone model. Moreover, it includes an offline optimization procedure that leverages the known geometry of gates to refine any state estimation parameter. This offline optimization is based purely on onboard flight data and is important for fine-tuning the vital external camera calibration parameters. Furthermore, the guidance and control are performed by a neural network that foregoes inner loop controllers by directly sending motor commands. This small network runs on the flight controller at 500Hz. The proposed approach won the 2025 Abu Dhabi Autonomous Drone Racing Competition (A2RL), outperforming all competing AI teams and three human world champion pilots in a direct knockout tournament. It set a new milestone in autonomous drone racing research, reaching speeds up to 100 km/h on the competition track and successfully coping with problems such as camera interference and IMU saturation.

  </details>



- **HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV**  
  Faryal Batool, Iana Zhura, Valerii Serpiva, Roohan Ahmed Khan, Ivan Valuev, Issatay Tokmurziyev, Dzmitry Tsetserukou  
  _2026-01-21_ · https://arxiv.org/abs/2601.14973v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable human--robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments. This paper presents HumanDiffusion, a lightweight image-conditioned diffusion planner that generates human-aware navigation trajectories directly from RGB imagery. The system combines YOLO-11--based human detection with diffusion-driven trajectory generation, enabling a quadrotor to approach a target person and deliver medical assistance without relying on prior maps or computationally intensive planning pipelines. Trajectories are predicted in pixel space, ensuring smooth motion and a consistent safety margin around humans. We evaluate HumanDiffusion in simulation and real-world indoor mock-disaster scenarios. On a 300-sample test set, the model achieves a mean squared error of 0.02 in pixel-space trajectory reconstruction. Real-world experiments demonstrate an overall mission success rate of 80% across accident-response and search-and-locate tasks with partial occlusions. These results indicate that human-conditioned diffusion planning offers a practical and robust solution for human-aware UAV navigation in time-critical assistance settings.

  </details>



- **DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration**  
  Dominik Rößle, Xujun Xie, Adithya Mohan, Venkatesh Thirugnana Sambandham, Daniel Cremers, Torsten Schön  
  _2026-01-21_ · https://arxiv.org/abs/2601.15260v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Perception is a cornerstone of autonomous driving, enabling vehicles to understand their surroundings and make safe, reliable decisions. Developing robust perception algorithms requires large-scale, high-quality datasets that cover diverse driving conditions and support thorough evaluation. Existing datasets often lack a high-fidelity digital twin, limiting systematic testing, edge-case simulation, sensor modification, and sim-to-real evaluations. To address this gap, we present DrivIng, a large-scale multimodal dataset with a complete geo-referenced digital twin of a ~18 km route spanning urban, suburban, and highway segments. Our dataset provides continuous recordings from six RGB cameras, one LiDAR, and high-precision ADMA-based localization, captured across day, dusk, and night. All sequences are annotated at 10 Hz with 3D bounding boxes and track IDs across 12 classes, yielding ~1.2 million annotated instances. Alongside the benefits of a digital twin, DrivIng enables a 1-to-1 transfer of real traffic into simulation, preserving agent interactions while enabling realistic and flexible scenario testing. To support reproducible research and robust validation, we benchmark DrivIng with state-of-the-art perception models and publicly release the dataset, digital twin, HD map, and codebase.

  </details>



- **On-the-fly hand-eye calibration for the da Vinci surgical robot**  
  Zejian Cui, Ferdinando Rodriguez y Baena  
  _2026-01-21_ · https://arxiv.org/abs/2601.14871v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In Robot-Assisted Minimally Invasive Surgery (RMIS), accurate tool localization is crucial to ensure patient safety and successful task execution. However, this remains challenging for cable-driven robots, such as the da Vinci robot, because erroneous encoder readings lead to pose estimation errors. In this study, we propose a calibration framework to produce accurate tool localization results through computing the hand-eye transformation matrix on-the-fly. The framework consists of two interrelated algorithms: the feature association block and the hand-eye calibration block, which provide robust correspondences for key points detected on monocular images without pre-training, and offer the versatility to accommodate various surgical scenarios by adopting an array of filter approaches, respectively. To validate its efficacy, we test the framework extensively on publicly available video datasets that feature multiple surgical instruments conducting tasks in both in vitro and ex vivo scenarios, under varying illumination conditions and with different levels of key point measurement accuracy. The results show a significant reduction in tool localization errors under the proposed calibration framework, with accuracies comparable to other state-of-the-art methods while being more time-efficient.

  </details>



- **Moving Beyond Compliance in Soft-Robotic Catheters Through Modularity for Precision Therapies**  
  B. Calmé, N. J. Greenidge, A. Metcalf, A. Bacchetti, G. Loza, D. Kpeglo, P. Lloyd, V. Pensabene, J. H. Chandler, P. Valdastri  
  _2026-01-21_ · https://arxiv.org/abs/2601.14837v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Soft robotic instruments could navigate delicate, tortuous anatomy more safely than rigid tools, but clinical adoption is limited by insufficient tip functionalization and real-time feedback at the tissue interface. Few sensing and therapeutic modules are compact, robust, and adaptable enough to measure, and respond to, subtle physiological cues during intraluminal procedures. We present a 1.47 mm diameter modular soft robotic catheter that integrates sensing, actuation, and therapy while retaining the compliance needed for safe endoluminal navigation. Validated across multiple in vivo settings, we emphasize its utility in endoscopic retrograde cholangiopancreatography (ERCP), a highly technical procedure and a key access route to the pancreas, an organ that is fragile, difficult to instrument, and central to diseases such as pancreatic cancer. Our architecture supports up to four independently controlled functional units, allowing customizable combinations of anchoring, manipulation, sensing, and targeted drug delivery. In a live porcine model, we demonstrate semi-autonomous deployment into the pancreatic duct and 7.5 cm of endoscopic navigation within it, a region currently inaccessible with standard catheters. A closed-loop autonomous/shared-control system that combines a learned model, magnetic actuation, onboard shape sensing, and visual marker tracking further improves cannulation accuracy. Together, these results establish a scalable platform for multifunctional soft robotic catheters and a new paradigm for complex endoluminal interventions, with potential to reduce radiation exposure, shorten training, and accelerate clinical translation of soft robotic technologies.

  </details>



- **Stochastic Decision-Making Framework for Human-Robot Collaboration in Industrial Applications**  
  Muhammad Adel Yusuf, Ali Nasir, Zeeshan Hameed Khan  
  _2026-01-21_ · https://arxiv.org/abs/2601.14809v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Collaborative robots, or cobots, are increasingly integrated into various industrial and service settings to work efficiently and safely alongside humans. However, for effective human-robot collaboration, robots must reason based on human factors such as motivation level and aggression level. This paper proposes an approach for decision-making in human-robot collaborative (HRC) environments utilizing stochastic modeling. By leveraging probabilistic models and control strategies, the proposed method aims to anticipate human actions and emotions, enabling cobots to adapt their behavior accordingly. So far, most of the research has been done to detect the intentions of human co-workers. This paper discusses the theoretical framework, implementation strategies, simulation results, and potential applications of the bilateral collaboration approach for safety and efficiency in collaborative robotics.

  </details>



- **HyperNet-Adaptation for Diffusion-Based Test Case Generation**  
  Oliver Weißl, Vincenzo Riccio, Severin Kacianka, Andrea Stocco  
  _2026-01-21_ · https://arxiv.org/abs/2601.15041v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The increasing deployment of deep learning systems requires systematic evaluation of their reliability in real-world scenarios. Traditional gradient-based adversarial attacks introduce small perturbations that rarely correspond to realistic failures and mainly assess robustness rather than functional behavior. Generative test generation methods offer an alternative but are often limited to simple datasets or constrained input domains. Although diffusion models enable high-fidelity image synthesis, their computational cost and limited controllability restrict their applicability to large-scale testing. We present HyNeA, a generative testing method that enables direct and efficient control over diffusion-based generation. HyNeA provides dataset-free controllability through hypernetworks, allowing targeted manipulation of the generative process without relying on architecture-specific conditioning mechanisms or dataset-driven adaptations such as fine-tuning. HyNeA employs a distinct training strategy that supports instance-level tuning to identify failure-inducing test cases without requiring datasets that explicitly contain examples of similar failures. This approach enables the targeted generation of realistic failure cases at substantially lower computational cost than search-based methods. Experimental results show that HyNeA improves controllability and test diversity compared to existing generative test generators and generalizes to domains where failure-labeled training data is unavailable.

  </details>



- **Risk Estimation for Automated Driving**  
  Leon Tolksdorf, Arturo Tejada, Jonas Bauernfeind, Christian Birkner, Nathan van de Wouw  
  _2026-01-21_ · https://arxiv.org/abs/2601.15018v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Safety is a central requirement for automated vehicles. As such, the assessment of risk in automated driving is key in supporting both motion planning technologies and safety evaluation. In automated driving, risk is characterized by two aspects. The first aspect is the uncertainty on the state estimates of other road participants by an automated vehicle. The second aspect is the severity of a collision event with said traffic participants. Here, the uncertainty aspect typically causes the risk to be non-zero for near-collision events. This makes risk particularly useful for automated vehicle motion planning. Namely, constraining or minimizing risk naturally navigates the automated vehicle around traffic participants while keeping a safety distance based on the level of uncertainty and the potential severity of the impending collision. Existing approaches to calculate the risk either resort to empirical modeling or severe approximations, and, hence, lack generalizability and accuracy. In this paper, we combine recent advances in collision probability estimation with the concept of collision severity to develop a general method for accurate risk estimation. The proposed method allows us to assign individual severity functions for different collision constellations, such as, e.g., frontal or side collisions. Furthermore, we show that the proposed approach is computationally efficient, which is beneficial, e.g., in real-time motion planning applications. The programming code for an exemplary implementation of Gaussian uncertainties is also provided.

  </details>



- **From Observation to Prediction: LSTM for Vehicle Lane Change Forecasting on Highway On/Off-Ramps**  
  Mohamed Abouras, Catherine M. Elias  
  _2026-01-21_ · https://arxiv.org/abs/2601.14848v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  On and off-ramps are understudied road sections even though they introduce a higher level of variation in highway interactions. Predicting vehicles' behavior in these areas can decrease the impact of uncertainty and increase road safety. In this paper, the difference between this Area of Interest (AoI) and a straight highway section is studied. Multi-layered LSTM architecture to train the AoI model with ExiD drone dataset is utilized. In the process, different prediction horizons and different models' workflow are tested. The results show great promise on horizons up to 4 seconds with prediction accuracy starting from about 76% for the AoI and 94% for the general highway scenarios on the maximum horizon.

  </details>



- **Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control**  
  Jannis Becktepe, Aleksandra Franz, Nils Thuerey, Sebastian Peitz  
  _2026-01-21_ · https://arxiv.org/abs/2601.15015v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has shown promising results in active flow control (AFC), yet progress in the field remains difficult to assess as existing studies rely on heterogeneous observation and actuation schemes, numerical setups, and evaluation protocols. Current AFC benchmarks attempt to address these issues but heavily rely on external computational fluid dynamics (CFD) solvers, are not fully differentiable, and provide limited 3D and multi-agent support. To overcome these limitations, we introduce FluidGym, the first standalone, fully differentiable benchmark suite for RL in AFC. Built entirely in PyTorch on top of the GPU-accelerated PICT solver, FluidGym runs in a single Python stack, requires no external CFD software, and provides standardized evaluation protocols. We present baseline results with PPO and SAC and release all environments, datasets, and trained models as public resources. FluidGym enables systematic comparison of control methods, establishes a scalable foundation for future research in learning-based flow control, and is available at https://github.com/safe-autonomous-systems/fluidgym.

  </details>



- **Finite-Sample Inference for Sparsely Permuted Linear Regression**  
  Hirofumi Ota, Masaaki Imaizumi  
  _2026-01-21_ · https://arxiv.org/abs/2601.14872v1 · `math.ST`  
  <details><summary>Abstract</summary>

  We study a noisy linear observation model with an unknown permutation called permuted/shuffled linear regression, where responses and covariates are mismatched and the permutation forms a discrete, factorial-size parameter. This unknown permutation is a key component of the data-generating process, yet its statistical investigation remains challenging due to its discrete nature. In this study, we develop a general statistical inference framework on the permutation and regression coefficients. First, we introduce a localization step that reduces the permutation space to a small candidate set building on recent advances in the repro samples method, whose miscoverage decays polynomially with the number of Monte Carlo samples. Then, based on this localized set, we provide statistical inference procedures: a conditional Monte Carlo test of permutation structures with valid finite-sample Type-I error control. We also develop coefficient inference that remains valid under alignment uncertainty of permutations. For computational purposes, we develop a linear assignment problem computable in polynomial time complexity and demonstrate that its solution asymptotically converges to that of the conventional least squares problem with large computational cost. Extensions to partially permuted designs and ridge regularization are also discussed. Extensive simulations and an application to Beijing air-quality data corroborate finite-sample validity, strong power to detect mismatches, and practical scalability.

  </details>



- **AutoDriDM: An Explainable Benchmark for Decision-Making of Vision-Language Models in Autonomous Driving**  
  Zecong Tang, Zixu Wang, Yifei Wang, Weitong Lian, Tianjian Gao, Haoran Li, Tengju Ru, Lingyi Meng, Zhejun Cui, Yichen Zhu, et al.  
  _2026-01-21_ · https://arxiv.org/abs/2601.14702v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Autonomous driving is a highly challenging domain that requires reliable perception and safe decision-making in complex scenarios. Recent vision-language models (VLMs) demonstrate reasoning and generalization abilities, opening new possibilities for autonomous driving; however, existing benchmarks and metrics overemphasize perceptual competence and fail to adequately assess decision-making processes. In this work, we present AutoDriDM, a decision-centric, progressive benchmark with 6,650 questions across three dimensions - Object, Scene, and Decision. We evaluate mainstream VLMs to delineate the perception-to-decision capability boundary in autonomous driving, and our correlation analysis reveals weak alignment between perception and decision-making performance. We further conduct explainability analyses of models' reasoning processes, identifying key failure modes such as logical reasoning errors, and introduce an analyzer model to automate large-scale annotation. AutoDriDM bridges the gap between perception-centered and decision-centered evaluation, providing guidance toward safer and more reliable VLMs for real-world autonomous driving.

  </details>



- **BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries**  
  Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, Kai Chen  
  _2026-01-21_ · https://arxiv.org/abs/2601.15197v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.

  </details>



- **M2I2HA: A Multi-modal Object Detection Method Based on Intra- and Inter-Modal Hypergraph Attention**  
  Xiaofan Yang, Yubin Liu, Wei Pan, Guoqing Chu, Junming Zhang, Jie Zhao, Zhuoqi Man, Xuanming Cao  
  _2026-01-21_ · https://arxiv.org/abs/2601.14776v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advances in multi-modal detection have significantly improved detection accuracy in challenging environments (e.g., low light, overexposure). By integrating RGB with modalities such as thermal and depth, multi-modal fusion increases data redundancy and system robustness. However, significant challenges remain in effectively extracting task-relevant information both within and across modalities, as well as in achieving precise cross-modal alignment. While CNNs excel at feature extraction, they are limited by constrained receptive fields, strong inductive biases, and difficulty in capturing long-range dependencies. Transformer-based models offer global context but suffer from quadratic computational complexity and are confined to pairwise correlation modeling. Mamba and other State Space Models (SSMs), on the other hand, are hindered by their sequential scanning mechanism, which flattens 2D spatial structures into 1D sequences, disrupting topological relationships and limiting the modeling of complex higher-order dependencies. To address these issues, we propose a multi-modal perception network based on hypergraph theory called M2I2HA. Our architecture includes an Intra-Hypergraph Enhancement module to capture global many-to-many high-order relationships within each modality, and an Inter-Hypergraph Fusion module to align, enhance, and fuse cross-modal features by bridging configuration and spatial gaps between data sources. We further introduce a M2-FullPAD module to enable adaptive multi-level fusion of multi-modal enhanced features within the network, meanwhile enhancing data distribution and flow across the architecture. Extensive object detection experiments on multiple public datasets against baselines demonstrate that M2I2HA achieves state-of-the-art performance in multi-modal object detection tasks.

  </details>



- **SimD3: A Synthetic drone Dataset with Payload and Bird Distractor Modeling for Robust Detection**  
  Ami Pandat, Kanyala Muvva, Punna Rajasekhar, Gopika Vinod, Rohit Shukla  
  _2026-01-21_ · https://arxiv.org/abs/2601.14742v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Reliable drone detection is challenging due to limited annotated real-world data, large appearance variability, and the presence of visually similar distractors such as birds. To address these challenges, this paper introduces SimD3, a large-scale high-fidelity synthetic dataset designed for robust drone detection in complex aerial environments. Unlike existing synthetic drone datasets, SimD3 explicitly models drones with heterogeneous payloads, incorporates multiple bird species as realistic distractors, and leverages diverse Unreal Engine 5 environments with controlled weather, lighting, and flight trajectories captured using a 360 six-camera rig. Using SimD3, we conduct an extensive experimental evaluation within the YOLOv5 detection framework, including an attention-enhanced variant termed Yolov5m+C3b, where standard bottleneck-based C3 blocks are replaced with C3b modules. Models are evaluated on synthetic data, combined synthetic and real data, and multiple unseen real-world benchmarks to assess robustness and generalization. Experimental results show that SimD3 provides effective supervision for small-object drone detection and that Yolov5m+C3b consistently outperforms the baseline across in-domain and cross-dataset evaluations. These findings highlight the utility of SimD3 for training and benchmarking robust drone detection models under diverse and challenging conditions.

  </details>



- **Safeguarding Facial Identity against Diffusion-based Face Swapping via Cascading Pathway Disruption**  
  Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo  
  _2026-01-21_ · https://arxiv.org/abs/2601.14738v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The rapid evolution of diffusion models has democratized face swapping but also raises concerns about privacy and identity security. Existing proactive defenses, often adapted from image editing attacks, prove ineffective in this context. We attribute this failure to an oversight of the structural resilience and the unique static conditional guidance mechanism inherent in face swapping systems. To address this, we propose VoidFace, a systemic defense method that views face swapping as a coupled identity pathway. By injecting perturbations at critical bottlenecks, VoidFace induces cascading disruption throughout the pipeline. Specifically, we first introduce localization disruption and identity erasure to degrade physical regression and semantic embeddings, thereby impairing the accurate modeling of the source face. We then intervene in the generative domain by decoupling attention mechanisms to sever identity injection, and corrupting intermediate diffusion features to prevent the reconstruction of source identity. To ensure visual imperceptibility, we perform adversarial search in the latent manifold, guided by a perceptual adaptive strategy to balance attack potency with image quality. Extensive experiments show that VoidFace outperforms existing defenses across various diffusion-based swapping models, while producing adversarial faces with superior visual quality.

  </details>



- **Case-Guided Sequential Assay Planning in Drug Discovery**  
  Tianchi Chen, Jan Bima, Sean L. Wu, Otto Ritter, Bingjia Yang, Xiang Yu  
  _2026-01-21_ · https://arxiv.org/abs/2601.14710v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Optimally sequencing experimental assays in drug discovery is a high-stakes planning problem under severe uncertainty and resource constraints. A primary obstacle for standard reinforcement learning (RL) is the absence of an explicit environment simulator or transition data $(s, a, s')$; planning must rely solely on a static database of historical outcomes. We introduce the Implicit Bayesian Markov Decision Process (IBMDP), a model-based RL framework designed for such simulator-free settings. IBMDP constructs a case-guided implicit model of transition dynamics by forming a nonparametric belief distribution using similar historical outcomes. This mechanism enables Bayesian belief updating as evidence accumulates and employs ensemble MCTS planning to generate stable policies that balance information gain toward desired outcomes with resource efficiency. We validate IBMDP through comprehensive experiments. On a real-world central nervous system (CNS) drug discovery task, IBMDP reduced resource consumption by up to 92\% compared to established heuristics while maintaining decision confidence. To rigorously assess decision quality, we also benchmarked IBMDP in a synthetic environment with a computable optimal policy. Our framework achieves significantly higher alignment with this optimal policy than a deterministic value iteration alternative that uses the same similarity-based model, demonstrating the superiority of our ensemble planner. IBMDP offers a practical solution for sequential experimental design in data-rich but simulator-poor domains.

  </details>



- **When Text-as-Vision Meets Semantic IDs in Generative Recommendation: An Empirical Study**  
  Shutong Qiao, Wei Yuan, Tong Chen, Xiangyu Zhao, Quoc Viet Hung Nguyen, Hongzhi Yin  
  _2026-01-21_ · https://arxiv.org/abs/2601.14697v1 · `cs.IR`  
  <details><summary>Abstract</summary>

  Semantic ID learning is a key interface in Generative Recommendation (GR) models, mapping items to discrete identifiers grounded in side information, most commonly via a pretrained text encoder. However, these text encoders are primarily optimized for well-formed natural language. In real-world recommendation data, item descriptions are often symbolic and attribute-centric, containing numerals, units, and abbreviations. These text encoders can break these signals into fragmented tokens, weakening semantic coherence and distorting relationships among attributes. Worse still, when moving to multimodal GR, relying on standard text encoders introduces an additional obstacle: text and image embeddings often exhibit mismatched geometric structures, making cross-modal fusion less effective and less stable. In this paper, we revisit representation design for Semantic ID learning by treating text as a visual signal. We conduct a systematic empirical study of OCR-based text representations, obtained by rendering item descriptions into images and encoding them with vision-based OCR models. Experiments across four datasets and two generative backbones show that OCR-text consistently matches or surpasses standard text embeddings for Semantic ID learning in both unimodal and multimodal settings. Furthermore, we find that OCR-based Semantic IDs remain robust under extreme spatial-resolution compression, indicating strong robustness and efficiency in practical deployments.

  </details>



- **Beyond Error-Based Optimization: Experience-Driven Symbolic Regression with Goal-Conditioned Reinforcement Learning**  
  Jianwen Sun, Xinrui Li, Fuqing Li, Xiaoxuan Shen  
  _2026-01-21_ · https://arxiv.org/abs/2601.14693v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Symbolic Regression aims to automatically identify compact and interpretable mathematical expressions that model the functional relationship between input and output variables. Most existing search-based symbolic regression methods typically rely on the fitting error to inform the search process. However, in the vast expression space, numerous candidate expressions may exhibit similar error values while differing substantially in structure, leading to ambiguous search directions and hindering convergence to the underlying true function. To address this challenge, we propose a novel framework named EGRL-SR (Experience-driven Goal-conditioned Reinforcement Learning for Symbolic Regression). In contrast to traditional error-driven approaches, EGRL-SR introduces a new perspective: leveraging precise historical trajectories and optimizing the action-value network to proactively guide the search process, thereby achieving a more robust expression search. Specifically, we formulate symbolic regression as a goal-conditioned reinforcement learning problem and incorporate hindsight experience replay, allowing the action-value network to generalize common mapping patterns from diverse input-output pairs. Moreover, we design an all-point satisfaction binary reward function that encourages the action-value network to focus on structural patterns rather than low-error expressions, and concurrently propose a structure-guided heuristic exploration strategy to enhance search diversity and space coverage. Experiments on public benchmarks show that EGRL-SR consistently outperforms state-of-the-art methods in recovery rate and robustness, and can recover more complex expressions under the same search budget. Ablation results validate that the action-value network effectively guides the search, with both the reward function and the exploration strategy playing critical roles.

  </details>



- **Beyond Denial-of-Service: The Puppeteer's Attack for Fine-Grained Control in Ranking-Based Federated Learning**  
  Zhihao Chen, Zirui Gong, Jianting Ning, Yanjun Zhang, Leo Yu Zhang  
  _2026-01-21_ · https://arxiv.org/abs/2601.14687v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Federated Rank Learning (FRL) is a promising Federated Learning (FL) paradigm designed to be resilient against model poisoning attacks due to its discrete, ranking-based update mechanism. Unlike traditional FL methods that rely on model updates, FRL leverages discrete rankings as a communication parameter between clients and the server. This approach significantly reduces communication costs and limits an adversary's ability to scale or optimize malicious updates in the continuous space, thereby enhancing its robustness. This makes FRL particularly appealing for applications where system security and data privacy are crucial, such as web-based auction and bidding platforms. While FRL substantially reduces the attack surface, we demonstrate that it remains vulnerable to a new class of local model poisoning attack, i.e., fine-grained control attacks. We introduce the Edge Control Attack (ECA), the first fine-grained control attack tailored to ranking-based FL frameworks. Unlike conventional denial-of-service (DoS) attacks that cause conspicuous disruptions, ECA enables an adversary to precisely degrade a competitor's accuracy to any target level while maintaining a normal-looking convergence trajectory, thereby avoiding detection. ECA operates in two stages: (i) identifying and manipulating Ascending and Descending Edges to align the global model with the target model, and (ii) widening the selection boundary gap to stabilize the global model at the target accuracy. Extensive experiments across seven benchmark datasets and nine Byzantine-robust aggregation rules (AGRs) show that ECA achieves fine-grained accuracy control with an average error of only 0.224%, outperforming the baseline by up to 17x. Our findings highlight the need for stronger defenses against advanced poisoning attacks. Our code is available at: https://github.com/Chenzh0205/ECA

  </details>


