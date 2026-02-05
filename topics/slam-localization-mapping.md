# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-05 07:13 UTC_

Total papers shown: **17**


---

- **Radar-Inertial Odometry For Computationally Constrained Aerial Navigation**  
  Jan Michalczyk  
  _2026-02-04_ · https://arxiv.org/abs/2602.04631v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recently, the progress in the radar sensing technology consisting in the miniaturization of the packages and increase in measuring precision has drawn the interest of the robotics research community. Indeed, a crucial task enabling autonomy in robotics is to precisely determine the pose of the robot in space. To fulfill this task sensor fusion algorithms are often used, in which data from one or several exteroceptive sensors like, for example, LiDAR, camera, laser ranging sensor or GNSS are fused together with the Inertial Measurement Unit (IMU) measurements to obtain an estimate of the navigation states of the robot. Nonetheless, owing to their particular sensing principles, some exteroceptive sensors are often incapacitated in extreme environmental conditions, like extreme illumination or presence of fine particles in the environment like smoke or fog. Radars are largely immune to aforementioned factors thanks to the characteristics of electromagnetic waves they use. In this thesis, we present Radar-Inertial Odometry (RIO) algorithms to fuse the information from IMU and radar in order to estimate the navigation states of a (Uncrewed Aerial Vehicle) UAV capable of running on a portable resource-constrained embedded computer in real-time and making use of inexpensive, consumer-grade sensors. We present novel RIO approaches relying on the multi-state tightly-coupled Extended Kalman Filter (EKF) and Factor Graphs (FG) fusing instantaneous velocities of and distances to 3D points delivered by a lightweight, low-cost, off-the-shelf Frequency Modulated Continuous Wave (FMCW) radar with IMU readings. We also show a novel way to exploit advances in deep learning to retrieve 3D point correspondences in sparse and noisy radar point clouds.

  </details>



- **How to rewrite the stars: Mapping your orchard over time through constellations of fruits**  
  Gonçalo P. Matos, Carlos Santiago, João P. Costeira, Ricardo L. Saldanha, Ernesto M. Morgado  
  _2026-02-04_ · https://arxiv.org/abs/2602.04722v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Following crop growth through the vegetative cycle allows farmers to predict fruit setting and yield in early stages, but it is a laborious and non-scalable task if performed by a human who has to manually measure fruit sizes with a caliper or dendrometers. In recent years, computer vision has been used to automate several tasks in precision agriculture, such as detecting and counting fruits, and estimating their size. However, the fundamental problem of matching the exact same fruits from one video, collected on a given date, to the fruits visible in another video, collected on a later date, which is needed to track fruits' growth through time, remains to be solved. Few attempts were made, but they either assume that the camera always starts from the same known position and that there are sufficiently distinct features to match, or they used other sources of data like GPS. Here we propose a new paradigm to tackle this problem, based on constellations of 3D centroids, and introduce a descriptor for very sparse 3D point clouds that can be used to match fruits across videos. Matching constellations instead of individual fruits is key to deal with non-rigidity, occlusions and challenging imagery with few distinct visual features to track. The results show that the proposed method can be successfully used to match fruits across videos and through time, and also to build an orchard map and later use it to locate the camera pose in 6DoF, thus providing a method for autonomous navigation of robots in the orchard and for selective fruit picking, for example.

  </details>



- **S-MUSt3R: Sliding Multi-view 3D Reconstruction**  
  Leonid Antsfeld, Boris Chidlovskii, Yohann Cabon, Vincent Leroy, Jerome Revaud  
  _2026-02-04_ · https://arxiv.org/abs/2602.04517v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The recent paradigm shift in 3D vision led to the rise of foundation models with remarkable capabilities in 3D perception from uncalibrated images. However, extending these models to large-scale RGB stream 3D reconstruction remains challenging due to memory limitations. This work proposes S-MUSt3R, a simple and efficient pipeline that extends the limits of foundation models for monocular 3D reconstruction. Our approach addresses the scalability bottleneck of foundation models through a simple strategy of sequence segmentation followed by segment alignment and lightweight loop closure optimization. Without model retraining, we benefit from remarkable 3D reconstruction capacities of MUSt3R model and achieve trajectory and reconstruction performance comparable to traditional methods with more complex architecture. We evaluate S-MUSt3R on TUM, 7-Scenes and proprietary robot navigation datasets and show that S-MUSt3R runs successfully on long RGB sequences and produces accurate and consistent 3D reconstruction. Our results highlight the potential of leveraging the MUSt3R model for scalable monocular 3D scene in real-world settings, with an important advantage of making predictions directly in the metric space.

  </details>



- **Score-Based Change-Point Detection and Region Localization for Spatio-Temporal Point Processes**  
  Wenbin Zhou, Liyan Xie, Shixiang Zhu  
  _2026-02-04_ · https://arxiv.org/abs/2602.04798v1 · `stat.ME`  
  <details><summary>Abstract</summary>

  We study sequential change-point detection for spatio-temporal point processes, where actionable detection requires not only identifying when a distributional change occurs but also localizing where it manifests in space. While classical quickest change detection methods provide strong guarantees on detection delay and false-alarm rates, existing approaches for point-process data predominantly focus on temporal changes and do not explicitly infer affected spatial regions. We propose a likelihood-free, score-based detection framework that jointly estimates the change time and the change region in continuous space-time without assuming parametric knowledge of the pre- or post-change dynamics. The method leverages a localized and conditionally weighted Hyvärinen score to quantify event-level deviations from nominal behavior and aggregates these scores using a spatio-temporal CUSUM-type statistic over a prescribed class of spatial regions. Operating sequentially, the procedure outputs both a stopping time and an estimated change region, enabling real-time detection with spatial interpretability. We establish theoretical guarantees on false-alarm control, detection delay, and spatial localization accuracy, and demonstrate the effectiveness of the proposed approach through simulations and real-world spatio-temporal event data.

  </details>



- **Mitigating Long-Tail Bias via Prompt-Controlled Diffusion Augmentation**  
  Buddhi Wijenayake, Nichula Wasalathilake, Roshan Godaliyadda, Vijitha Herath, Parakrama Ekanayake, Vishal M. Patel  
  _2026-02-04_ · https://arxiv.org/abs/2602.04749v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Semantic segmentation of high-resolution remote-sensing imagery is critical for urban mapping and land-cover monitoring, yet training data typically exhibits severe long-tailed pixel imbalance. In the dataset LoveDA, this challenge is compounded by an explicit Urban/Rural split with distinct appearance and inconsistent class-frequency statistics across domains. We present a prompt-controlled diffusion augmentation framework that synthesizes paired label--image samples with explicit control of both domain and semantic composition. Stage~A uses a domain-aware, masked ratio-conditioned discrete diffusion model to generate layouts that satisfy user-specified class-ratio targets while respecting learned co-occurrence structure. Stage~B translates layouts into photorealistic, domain-consistent images using Stable Diffusion with ControlNet guidance. Mixing the resulting ratio and domain-controlled synthetic pairs with real data yields consistent improvements across multiple segmentation backbones, with gains concentrated on minority classes and improved Urban and Rural generalization, demonstrating controllable augmentation as a practical mechanism to mitigate long-tail bias in remote-sensing segmentation. Source codes, pretrained models, and synthetic datasets are available at \href{https://github.com/Buddhi19/SyntheticGen.git}{Github}

  </details>



- **Relational Scene Graphs for Object Grounding of Natural Language Commands**  
  Julia Kuhn, Francesco Verdoja, Tsvetomila Mihaylova, Ville Kyrki  
  _2026-02-04_ · https://arxiv.org/abs/2602.04635v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots are finding wider adoption in human environments, increasing the need for natural human-robot interaction. However, understanding a natural language command requires the robot to infer the intended task and how to decompose it into executable actions, and to ground those actions in the robot's knowledge of the environment, including relevant objects, agents, and locations. This challenge can be addressed by combining the capabilities of Large language models (LLMs) to understand natural language with 3D scene graphs (3DSGs) for grounding inferred actions in a semantic representation of the environment. However, many 3DSGs lack explicit spatial relations between objects, even though humans often rely on these relations to describe an environment. This paper investigates whether incorporating open- or closed-vocabulary spatial relations into 3DSGs can improve the ability of LLMs to interpret natural language commands. To address this, we propose an LLM-based pipeline for target object grounding from open-vocabulary language commands and a vision language model (VLM)-based pipeline to add open-vocabulary spatial edges to 3DSGs from images captured while mapping. Finally, two LLMs are evaluated in a study assessing their performance on the downstream task of target object grounding. Our study demonstrates that explicit spatial relations improve the ability of LLMs to ground objects. Moreover, open-vocabulary relation generation with VLMs proves feasible from robot-captured images, but their advantage over closed-vocabulary relations is found to be limited.

  </details>



- **TACO: Temporal Consensus Optimization for Continual Neural Mapping**  
  Xunlan Zhou, Hongrui Zhao, Negar Mehr  
  _2026-02-04_ · https://arxiv.org/abs/2602.04516v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Neural implicit mapping has emerged as a powerful paradigm for robotic navigation and scene understanding. However, real-world robotic deployment requires continual adaptation to changing environments under strict memory and computation constraints, which existing mapping systems fail to support. Most prior methods rely on replaying historical observations to preserve consistency and assume static scenes. As a result, they cannot adapt to continual learning in dynamic robotic settings. To address these challenges, we propose TACO (TemporAl Consensus Optimization), a replay-free framework for continual neural mapping. We reformulate mapping as a temporal consensus optimization problem, where we treat past model snapshots as temporal neighbors. Intuitively, our approach resembles a model consulting its own past knowledge. We update the current map by enforcing weighted consensus with historical representations. Our method allows reliable past geometry to constrain optimization while permitting unreliable or outdated regions to be revised in response to new observations. TACO achieves a balance between memory efficiency and adaptability without storing or replaying previous data. Through extensive simulated and real-world experiments, we show that TACO robustly adapts to scene changes, and consistently outperforms other continual learning baselines.

  </details>



- **Generative Modeling via Drifting**  
  Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He  
  _2026-02-04_ · https://arxiv.org/abs/2602.04770v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Generative modeling can be formulated as learning a mapping f such that its pushforward distribution matches the data distribution. The pushforward behavior can be carried out iteratively at inference time, for example in diffusion and flow-based models. In this paper, we propose a new paradigm called Drifting Models, which evolve the pushforward distribution during training and naturally admit one-step inference. We introduce a drifting field that governs the sample movement and achieves equilibrium when the distributions match. This leads to a training objective that allows the neural network optimizer to evolve the distribution. In experiments, our one-step generator achieves state-of-the-art results on ImageNet at 256 x 256 resolution, with an FID of 1.54 in latent space and 1.61 in pixel space. We hope that our work opens up new opportunities for high-quality one-step generation.

  </details>



- **Resilient Channel Charting Under Varying Radio Link Availability**  
  Jonas Pirkl, Jonathan Ott, Maximilian Stahlke, George Yammine, Tobias Feigl, Christopher Mutschler  
  _2026-02-04_ · https://arxiv.org/abs/2602.04704v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Channel charting (CC) has become a key technology for RF-based localization, enabling unsupervised radio fingerprinting, even in non line of sight scenarios, with a minimum of reference position labels. However, most CC models assume fixed-size inputs, such as a constant number of antennas or channel measurements. In practical systems, antennas may fail, signals may be blocked, or antenna sets may change during handovers, making fixed-input architectures fragile. Existing radio-fingerprinting approaches address this by training separate models for each antenna configuration, but the resulting training effort scales prohibitively with the array size. We propose Adaptive Positioning (AdaPos), a CC architecture that natively handles variable numbers of channel measurements. AdaPos combines convolutional feature extraction with a transformer-based encoder using learnable antenna identifiers and self-attention to fuse arbitrary subsets of CSI inputs. Experiments on two public real-world datasets (SISO and MIMO) show that AdaPos maintains state-of-the-art accuracy under missing-antenna conditions and replaces roughly 57 configuration-specific models with a single unified model. With AdaPos and our novel training strategies, we provide resilience to both individual antenna failures and full-array outages.

  </details>



- **Knowledge Distillation for mmWave Beam Prediction Using Sub-6 GHz Channels**  
  Sina Tavakolian, Nhan Thanh Nguyen, Ahmed Alkhateeb, Markku Juntti  
  _2026-02-04_ · https://arxiv.org/abs/2602.04703v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Beamforming in millimeter-wave (mmWave) high-mobility environments typically incurs substantial training overhead. While prior studies suggest that sub-6 GHz channels can be exploited to predict optimal mmWave beams, existing methods depend on large deep learning (DL) models with prohibitive computational and memory requirements. In this paper, we propose a computationally efficient framework for sub-6 GHz channel-mmWave beam mapping based on the knowledge distillation (KD) technique. We develop two compact student DL architectures based on individual and relational distillation strategies, which retain only a few hidden layers yet closely mimic the performance of large teacher DL models. Extensive simulations demonstrate that the proposed student models achieve the teacher's beam prediction accuracy and spectral efficiency while reducing trainable parameters and computational complexity by 99%.

  </details>



- **A Human-Centered Privacy Approach (HCP) to AI**  
  Luyi Sun, Wei Xu, Zaifeng Gao  
  _2026-02-04_ · https://arxiv.org/abs/2602.04616v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  As the paradigm of Human-Centered AI (HCAI) gains prominence, its benefits to society are accompanied by significant ethical concerns, one of which is the protection of individual privacy. This chapter provides a comprehensive overview of privacy within HCAI, proposing a human-centered privacy (HCP) framework, providing integrated solution from technology, ethics, and human factors perspectives. The chapter begins by mapping privacy risks across each stage of AI development lifecycle, from data collection to deployment and reuse, highlighting the impact of privacy risks on the entire system. The chapter then introduces privacy-preserving techniques such as federated learning and dif erential privacy. Subsequent chapters integrate the crucial user perspective by examining mental models, alongside the evolving regulatory and ethical landscapes as well as privacy governance. Next, advice on design guidelines is provided based on the human-centered privacy framework. After that, we introduce practical case studies across diverse fields. Finally, the chapter discusses persistent open challenges and future research directions, concluding that a multidisciplinary approach, merging technical, design, policy, and ethical expertise, is essential to successfully embed privacy into the core of HCAI, thereby ensuring these technologies advance in a manner that respects and ensures human autonomy, trust and dignity.

  </details>



- **SLUM-i: Semi-supervised Learning for Urban Mapping of Informal Settlements and Data Quality Benchmarking**  
  Muhammad Taha Mukhtar, Syed Musa Ali Kazmi, Khola Naseem, Muhammad Ali Chattha, Andreas Dengel, Sheraz Ahmed, Muhammad Naseer Bajwa, Muhammad Imran Malik  
  _2026-02-04_ · https://arxiv.org/abs/2602.04525v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Rapid urban expansion has fueled the growth of informal settlements in major cities of low- and middle-income countries, with Lahore and Karachi in Pakistan and Mumbai in India serving as prominent examples. However, large-scale mapping of these settlements is severely constrained not only by the scarcity of annotations but by inherent data quality challenges, specifically high spectral ambiguity between formal and informal structures and significant annotation noise. We address this by introducing a benchmark dataset for Lahore, constructed from scratch, along with companion datasets for Karachi and Mumbai, which were derived from verified administrative boundaries, totaling 1,869 $\text{km}^2$ of area. To evaluate the global robustness of our framework, we extend our experiments to five additional established benchmarks, encompassing eight cities across three continents, and provide comprehensive data quality assessments of all datasets. We also propose a new semi-supervised segmentation framework designed to mitigate the class imbalance and feature degradation inherent in standard semi-supervised learning pipelines. Our method integrates a Class-Aware Adaptive Thresholding mechanism that dynamically adjusts confidence thresholds to prevent minority class suppression and a Prototype Bank System that enforces semantic consistency by anchoring predictions to historically learned high-fidelity feature representations. Extensive experiments across a total of eight cities spanning three continents demonstrate that our approach outperforms state-of-the-art semi-supervised baselines. Most notably, our method demonstrates superior domain transfer capability whereby a model trained on only 10% of source labels reaches a 0.461 mIoU on unseen geographies and outperforms the zero-shot generalization of fully supervised models.

  </details>



- **LLM-Empowered Cooperative Content Caching in Vehicular Fog Caching-Assisted Platoon Networks**  
  Bowen Tan, Qiong Wu, Pingyi Fan, Kezhi Wang, Nan Cheng, Wen Chen  
  _2026-02-04_ · https://arxiv.org/abs/2602.04471v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  This letter proposes a novel three-tier content caching architecture for Vehicular Fog Caching (VFC)-assisted platoon, where the VFC is formed by the vehicles driving near the platoon. The system strategically coordinates storage across local platoon vehicles, dynamic VFC clusters, and cloud server (CS) to minimize content retrieval latency. To efficiently manage distributed storage, we integrate large language models (LLMs) for real-time and intelligent caching decisions. The proposed approach leverages LLMs' ability to process heterogeneous information, including user profiles, historical data, content characteristics, and dynamic system states. Through a designed prompting framework encoding task objectives and caching constraints, the LLMs formulate caching as a decision-making task, and our hierarchical deterministic caching mapping strategy enables adaptive requests prediction and precise content placement across three tiers without frequent retraining. Simulation results demonstrate the advantages of our proposed caching scheme.

  </details>



- **Rigid Body Localization via Gaussian Belief Propagation with Quadratic Angle Approximation**  
  Niclas Führling, Hyeon Seok Rou, Giuseppe Abreu, David González G., Osvaldo Gonsa  
  _2026-02-04_ · https://arxiv.org/abs/2602.04410v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Gaussian belief propagation (GaBP) is a technique that relies on linearized error and input-output models to yield low-complexity solutions to complex estimation problems, which has been recently shown to be effective in the design of range-based GaBP schemes for stationary and moving rigid body localization (RBL) in three-dimensional (3D) space, as long as an accurate prior on the orientation of the target rigid body is available. In this article we present a novel range-based RBL scheme via GaBP that removes the latter limitation. To this end, the proposed method incorporates a quadratic angle approximation to linearize the relative orientation between the prior and the target rigid body, enabling high precision estimates of corresponding rotation angles even for large deviations. Leveraging the resulting linearized model, we derive the corresponding message-passing (MP) rules to obtain estimates of the translation vector and rotation matrix of the target rigid body, relative to a prior reference frame. Numerical results corroborate the good performance of the proposed angle approximation itself, as well as the consequent RBL performance in terms of root mean square errors (RMSEs) in comparison to the state-of-the-art (SotA), while maintaining a low computational complexity

  </details>



- **Theory of Speciation Transitions in Diffusion Models with General Class Structure**  
  Beatrice Achilli, Marco Benedetti, Giulio Biroli, Marc Mézard  
  _2026-02-04_ · https://arxiv.org/abs/2602.04404v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Diffusion Models generate data by reversing a stochastic diffusion process, progressively transforming noise into structured samples drawn from a target distribution. Recent theoretical work has shown that this backward dynamics can undergo sharp qualitative transitions, known as speciation transitions, during which trajectories become dynamically committed to data classes. Existing theoretical analyses, however, are limited to settings where classes are identifiable through first moments, such as mixtures of Gaussians with well-separated means. In this work, we develop a general theory of speciation in diffusion models that applies to arbitrary target distributions admitting well-defined classes. We formalize the notion of class structure through Bayes classification and characterize speciation times in terms of free-entropy difference between classes. This criterion recovers known results in previously studied Gaussian-mixture models, while extending to situations in which classes are not distinguishable by first moments and may instead differ through higher-order or collective features. Our framework also accommodates multiple classes and predicts the existence of successive speciation times associated with increasingly fine-grained class commitment. We illustrate the theory on two analytically tractable examples: mixtures of one-dimensional Ising models at different temperatures and mixtures of zero-mean Gaussians with distinct covariance structures. In the Ising case, we obtain explicit expressions for speciation times by mapping the problem onto a random-field Ising model and solving it via the replica method. Our results provide a unified and broadly applicable description of speciation transitions in diffusion-based generative models.

  </details>



- **Reducing the labeling burden in time-series mapping using Common Ground: a semi-automated approach to tracking changes in land cover and species over time**  
  Geethen Singh, Jasper A Slingsby, Tamara B Robinson, Glenn Moncrieff  
  _2026-02-04_ · https://arxiv.org/abs/2602.04373v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reliable classification of Earth Observation data depends on consistent, up-to-date reference labels. However, collecting new labelled data at each time step remains expensive and logistically difficult, especially in dynamic or remote ecological systems. As a response to this challenge, we demonstrate that a model with access to reference data solely from time step t0 can perform competitively on both t0 and a future time step t1, outperforming models trained separately on time-specific reference data (the gold standard). This finding suggests that effective temporal generalization can be achieved without requiring manual updates to reference labels beyond the initial time step t0. Drawing on concepts from change detection and semi-supervised learning (SSL), the most performant approach, "Common Ground", uses a semi-supervised framework that leverages temporally stable regions-areas with little to no change in spectral or semantic characteristics between time steps-as a source of implicit supervision for dynamic regions. We evaluate this strategy across multiple classifiers, sensors (Landsat-8, Sentinel-2 satellite multispectral and airborne imaging spectroscopy), and ecological use cases. For invasive tree species mapping, we observed a 21-40% improvement in classification accuracy using Common Ground compared to naive temporal transfer, where models trained at a single time step are directly applied to a future time step. We also observe a 10 -16% higher accuracy for the introduced approach compared to a gold-standard approach. In contrast, when broad land cover categories were mapped across Europe, we observed a more modest 2% increase in accuracy compared to both the naive and gold-standard approaches. These results underscore the effectiveness of combining stable reference screening with SSL for scalable and label-efficient multi-temporal remote sensing classification.

  </details>



- **SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration**  
  Zekun Li, Ning Wang, Tongxin Bai, Changwang Mei, Peisong Wang, Shuang Qiu, Jian Cheng  
  _2026-02-04_ · https://arxiv.org/abs/2602.04361v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Visual AutoRegressive (VAR) modeling has garnered significant attention for its innovative next-scale prediction paradigm. However, mainstream VAR paradigms attend to all tokens across historical scales at each autoregressive step. As the next scale resolution grows, the computational complexity of attention increases quartically with resolution, causing substantial latency. Prior accelerations often skip high-resolution scales, which speeds up inference but discards high-frequency details and harms image quality. To address these problems, we present SparVAR, a training-free acceleration framework that exploits three properties of VAR attention: (i) strong attention sinks, (ii) cross-scale activation similarity, and (iii) pronounced locality. Specifically, we dynamically predict the sparse attention pattern of later high-resolution scales from a sparse decision scale, and construct scale self-similar sparse attention via an efficient index-mapping mechanism, enabling high-efficiency sparse attention computation at large scales. Furthermore, we propose cross-scale local sparse attention and implement an efficient block-wise sparse kernel, which achieves $\mathbf{> 5\times}$ faster forward speed than FlashAttention. Extensive experiments demonstrate that the proposed SparseVAR can reduce the generation time of an 8B model producing $1024\times1024$ high-resolution images to the 1s, without skipping the last scales. Compared with the VAR baseline accelerated by FlashAttention, our method achieves a $\mathbf{1.57\times}$ speed-up while preserving almost all high-frequency details. When combined with existing scale-skipping strategies, SparseVAR attains up to a $\mathbf{2.28\times}$ acceleration, while maintaining competitive visual generation quality. Code is available at https://github.com/CAS-CLab/SparVAR.

  </details>


