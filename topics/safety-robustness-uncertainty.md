# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-01-29 07:02 UTC_

Total papers shown: **20**


---

- **Learning Contextual Runtime Monitors for Safe AI-Based Autonomy**  
  Alejandro Luque-Cerpa, Mengyuan Wang, Emil Carlsson, Sanjit A. Seshia, Devdatt Dubhashi, Hazem Torfah  
  _2026-01-28_ · https://arxiv.org/abs/2601.20666v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We introduce a novel framework for learning context-aware runtime monitors for AI-based control ensembles. Machine-learning (ML) controllers are increasingly deployed in (autonomous) cyber-physical systems because of their ability to solve complex decision-making tasks. However, their accuracy can degrade sharply in unfamiliar environments, creating significant safety concerns. Traditional ensemble methods aim to improve robustness by averaging or voting across multiple controllers, yet this often dilutes the specialized strengths that individual controllers exhibit in different operating contexts. We argue that, rather than blending controller outputs, a monitoring framework should identify and exploit these contextual strengths. In this paper, we reformulate the design of safe AI-based control ensembles as a contextual monitoring problem. A monitor continuously observes the system's context and selects the controller best suited to the current conditions. To achieve this, we cast monitor learning as a contextual learning task and draw on techniques from contextual multi-armed bandits. Our approach comes with two key benefits: (1) theoretical safety guarantees during controller selection, and (2) improved utilization of controller diversity. We validate our framework in two simulated autonomous driving scenarios, demonstrating significant improvements in both safety and performance compared to non-contextual baselines.

  </details>



- **Learning From a Steady Hand: A Weakly Supervised Agent for Robot Assistance under Microscopy**  
  Huanyu Tian, Martin Huber, Lingyun Zeng, Zhe Han, Wayne Bennett, Giuseppe Silvestri, Gerardo Mendizabal-Ruiz, Tom Vercauteren, Alejandro Chavez-Badiola, Christos Bergeles  
  _2026-01-28_ · https://arxiv.org/abs/2601.20776v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper rethinks steady-hand robotic manipulation by using a weakly supervised framework that fuses calibration-aware perception with admittance control. Unlike conventional automation that relies on labor-intensive 2D labeling, our framework leverages reusable warm-up trajectories to extract implicit spatial information, thereby achieving calibration-aware, depth-resolved perception without the need for external fiducials or manual depth annotation. By explicitly characterizing residuals from observation and calibration models, the system establishes a task-space error budget from recorded warm-ups. The uncertainty budget yields a lateral closed-loop accuracy of approx. 49 micrometers at 95% confidence (worst-case testing subset) and a depth accuracy of <= 291 micrometers at 95% confidence bound during large in-plane moves. In a within-subject user study (N=8), the learned agent reduces overall NASA-TLX workload by 77.1% relative to the simple steady-hand assistance baseline. These results demonstrate that the weakly supervised agent improves the reliability of microscope-guided biomedical micromanipulation without introducing complex setup requirements, offering a practical framework for microscope-guided intervention.

  </details>



- **Vibro-Sense: Robust Vibration-based Impulse Response Localization and Trajectory Tracking for Robotic Hands**  
  Wadhah Zai El Amri, Nicolás Navarro-Guerrero  
  _2026-01-28_ · https://arxiv.org/abs/2601.20555v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Rich contact perception is crucial for robotic manipulation, yet traditional tactile skins remain expensive and complex to integrate. This paper presents a scalable alternative: high-accuracy whole-body touch localization via vibro-acoustic sensing. By equipping a robotic hand with seven low-cost piezoelectric microphones and leveraging an Audio Spectrogram Transformer, we decode the vibrational signatures generated during physical interaction. Extensive evaluation across stationary and dynamic tasks reveals a localization error of under 5 mm in static conditions. Furthermore, our analysis highlights the distinct influence of material properties: stiff materials (e.g., metal) excel in impulse response localization due to sharp, high-bandwidth responses, whereas textured materials (e.g., wood) provide superior friction-based features for trajectory tracking. The system demonstrates robustness to the robot's own motion, maintaining effective tracking even during active operation. Our primary contribution is demonstrating that complex physical contact dynamics can be effectively decoded from simple vibrational signals, offering a viable pathway to widespread, affordable contact perception in robotics. To accelerate research, we provide our full datasets, models, and experimental setups as open-source resources.

  </details>



- **Li-ViP3D++: Query-Gated Deformable Camera-LiDAR Fusion for End-to-End Perception and Trajectory Prediction**  
  Matej Halinkovic, Nina Masarykova, Alexey Vinel, Marek Galinski  
  _2026-01-28_ · https://arxiv.org/abs/2601.20720v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  End-to-end perception and trajectory prediction from raw sensor data is one of the key capabilities for autonomous driving. Modular pipelines restrict information flow and can amplify upstream errors. Recent query-based, fully differentiable perception-and-prediction (PnP) models mitigate these issues, yet the complementarity of cameras and LiDAR in the query-space has not been sufficiently explored. Models often rely on fusion schemes that introduce heuristic alignment and discrete selection steps which prevent full utilization of available information and can introduce unwanted bias. We propose Li-ViP3D++, a query-based multimodal PnP framework that introduces Query-Gated Deformable Fusion (QGDF) to integrate multi-view RGB and LiDAR in query space. QGDF (i) aggregates image evidence via masked attention across cameras and feature levels, (ii) extracts LiDAR context through fully differentiable BEV sampling with learned per-query offsets, and (iii) applies query-conditioned gating to adaptively weight visual and geometric cues per agent. The resulting architecture jointly optimizes detection, tracking, and multi-hypothesis trajectory forecasting in a single end-to-end model. On nuScenes, Li-ViP3D++ improves end-to-end behavior and detection quality, achieving higher EPA (0.335) and mAP (0.502) while substantially reducing false positives (FP ratio 0.147), and it is faster than the prior Li-ViP3D variant (139.82 ms vs. 145.91 ms). These results indicate that query-space, fully differentiable camera-LiDAR fusion can increase robustness of end-to-end PnP without sacrificing deployability.

  </details>



- **STORM: Slot-based Task-aware Object-centric Representation for robotic Manipulation**  
  Alexandre Chapin, Emmanuel Dellandréa, Liming Chen  
  _2026-01-28_ · https://arxiv.org/abs/2601.20381v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual foundation models provide strong perceptual features for robotics, but their dense representations lack explicit object-level structure, limiting robustness and contractility in manipulation tasks. We propose STORM (Slot-based Task-aware Object-centric Representation for robotic Manipulation), a lightweight object-centric adaptation module that augments frozen visual foundation models with a small set of semantic-aware slots for robotic manipulation. Rather than retraining large backbones, STORM employs a multi-phase training strategy: object-centric slots are first stabilized through visual--semantic pretraining using language embeddings, then jointly adapted with a downstream manipulation policy. This staged learning prevents degenerate slot formation and preserves semantic consistency while aligning perception with task objectives. Experiments on object discovery benchmarks and simulated manipulation tasks show that STORM improves generalization to visual distractors, and control performance compared to directly using frozen foundation model features or training object-centric representations end-to-end. Our results highlight multi-phase adaptation as an efficient mechanism for transforming generic foundation model features into task-aware object-centric representations for robotic control.

  </details>



- **A New Dataset and Framework for Robust Road Surface Classification via Camera-IMU Fusion**  
  Willams de Lima Costa, Thifany Ketuli Silva de Souza, Jonas Ferreira Silva, Carlos Gabriel Bezerra Pereira, Bruno Reis Vila Nova, Leonardo Silvino Brito, Rafael Raider Leoni, Juliano Silva, Valter Ferreira, Sibele Miguel Soares Neto, et al.  
  _2026-01-28_ · https://arxiv.org/abs/2601.20847v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Road surface classification (RSC) is a key enabler for environment-aware predictive maintenance systems. However, existing RSC techniques often fail to generalize beyond narrow operational conditions due to limited sensing modalities and datasets that lack environmental diversity. This work addresses these limitations by introducing a multimodal framework that fuses images and inertial measurements using a lightweight bidirectional cross-attention module followed by an adaptive gating layer that adjusts modality contributions under domain shifts. Given the limitations of current benchmarks, especially regarding lack of variability, we introduce ROAD, a new dataset composed of three complementary subsets: (i) real-world multimodal recordings with RGB-IMU streams synchronized using a gold-standard industry datalogger, captured across diverse lighting, weather, and surface conditions; (ii) a large vision-only subset designed to assess robustness under adverse illumination and heterogeneous capture setups; and (iii) a synthetic subset generated to study out-of-distribution generalization in scenarios difficult to obtain in practice. Experiments show that our method achieves a +1.4 pp improvement over the previous state-of-the-art on the PVS benchmark and an +11.6 pp improvement on our multimodal ROAD subset, with consistently higher F1-scores on minority classes. The framework also demonstrates stable performance across challenging visual conditions, including nighttime, heavy rain, and mixed-surface transitions. These findings indicate that combining affordable camera and IMU sensors with multimodal attention mechanisms provides a scalable, robust foundation for road surface understanding, particularly relevant for regions where environmental variability and cost constraints limit the adoption of high-end sensing suites.

  </details>



- **Unsupervised Anomaly Detection in Multi-Agent Trajectory Prediction via Transformer-Based Models**  
  Qing Lyu, Zhe Fu, Alexandre Bayen  
  _2026-01-28_ · https://arxiv.org/abs/2601.20367v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Identifying safety-critical scenarios is essential for autonomous driving, but the rarity of such events makes supervised labeling impractical. Traditional rule-based metrics like Time-to-Collision are too simplistic to capture complex interaction risks, and existing methods lack a systematic way to verify whether statistical anomalies truly reflect physical danger. To address this gap, we propose an unsupervised anomaly detection framework based on a multi-agent Transformer that models normal driving and measures deviations through prediction residuals. A dual evaluation scheme has been proposed to assess both detection stability and physical alignment: Stability is measured using standard ranking metrics in which Kendall Rank Correlation Coefficient captures rank agreement and Jaccard index captures the consistency of the top-K selected items; Physical alignment is assessed through correlations with established Surrogate Safety Measures (SSM). Experiments on the NGSIM dataset demonstrate our framework's effectiveness: We show that the maximum residual aggregator achieves the highest physical alignment while maintaining stability. Furthermore, our framework identifies 388 unique anomalies missed by Time-to-Collision and statistical baselines, capturing subtle multi-agent risks like reactive braking under lateral drift. The detected anomalies are further clustered into four interpretable risk types, offering actionable insights for simulation and testing.

  </details>



- **End-to-end example-based sim-to-real RL policy transfer based on neural stylisation with application to robotic cutting**  
  Jamie Hathaway, Alireza Rastegarpanah, Rustam Stolkin  
  _2026-01-28_ · https://arxiv.org/abs/2601.20846v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Whereas reinforcement learning has been applied with success to a range of robotic control problems in complex, uncertain environments, reliance on extensive data - typically sourced from simulation environments - limits real-world deployment due to the domain gap between simulated and physical systems, coupled with limited real-world sample availability. We propose a novel method for sim-to-real transfer of reinforcement learning policies, based on a reinterpretation of neural style transfer from image processing to synthesise novel training data from unpaired unlabelled real world datasets. We employ a variational autoencoder to jointly learn self-supervised feature representations for style transfer and generate weakly paired source-target trajectories to improve physical realism of synthesised trajectories. We demonstrate the application of our approach based on the case study of robot cutting of unknown materials. Compared to baseline methods, including our previous work, CycleGAN, and conditional variational autoencoder-based time series translation, our approach achieves improved task completion time and behavioural stability with minimal real-world data. Our framework demonstrates robustness to geometric and material variation, and highlights the feasibility of policy adaptation in challenging contact-rich tasks where real-world reward information is unavailable.

  </details>



- **Post-Training Fairness Control: A Single-Train Framework for Dynamic Fairness in Recommendation**  
  Weixin Chen, Li Chen, Yuhan Zhao  
  _2026-01-28_ · https://arxiv.org/abs/2601.20848v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Despite growing efforts to mitigate unfairness in recommender systems, existing fairness-aware methods typically fix the fairness requirement at training time and provide limited post-training flexibility. However, in real-world scenarios, diverse stakeholders may demand differing fairness requirements over time, so retraining for different fairness requirements becomes prohibitive. To address this limitation, we propose Cofair, a single-train framework that enables post-training fairness control in recommendation. Specifically, Cofair introduces a shared representation layer with fairness-conditioned adapter modules to produce user embeddings specialized for varied fairness levels, along with a user-level regularization term that guarantees user-wise monotonic fairness improvements across these levels. We theoretically establish that the adversarial objective of Cofair upper bounds demographic parity and the regularization term enforces progressive fairness at user level. Comprehensive experiments on multiple datasets and backbone models demonstrate that our framework provides dynamic fairness at different levels, delivering comparable or better fairness-accuracy curves than state-of-the-art baselines, without the need to retrain for each new fairness requirement. Our code is publicly available at https://github.com/weixinchen98/Cofair.

  </details>



- **VSCOUT: A Hybrid Variational Autoencoder Approach to Outlier Detection in High-Dimensional Retrospective Monitoring**  
  Waldyn G. Martinez  
  _2026-01-28_ · https://arxiv.org/abs/2601.20830v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  Modern industrial and service processes generate high-dimensional, non-Gaussian, and contamination-prone data that challenge the foundational assumptions of classical Statistical Process Control (SPC). Heavy tails, multimodality, nonlinear dependencies, and sparse special-cause observations can distort baseline estimation, mask true anomalies, and prevent reliable identification of an in-control (IC) reference set. To address these challenges, we introduce VSCOUT, a distribution-free framework designed specifically for retrospective (Phase I) monitoring in high-dimensional settings. VSCOUT combines an Automatic Relevance Determination Variational Autoencoder (ARD-VAE) architecture with ensemble-based latent outlier filtering and changepoint detection. The ARD prior isolates the most informative latent dimensions, while the ensemble and changepoint filters identify pointwise and structural contamination within the determined latent space. A second-stage retraining step removes flagged observations and re-estimates the latent structure using only the retained inliers, mitigating masking and stabilizing the IC latent manifold. This two-stage refinement produces a clean and reliable IC baseline suitable for subsequent Phase II deployment. Extensive experiments across benchmark datasets demonstrate that VSCOUT achieves superior sensitivity to special-cause structure while maintaining controlled false alarms, outperforming classical SPC procedures, robust estimators, and modern machine-learning baselines. Its scalability, distributional flexibility, and resilience to complex contamination patterns position VSCOUT as a practical and effective method for retrospective modeling and anomaly detection in AI-enabled environments.

  </details>



- **Smoothing the Black-Box: Signed-Distance Supervision for Black-Box Model Copying**  
  Rubén Jiménez, Oriol Pujol  
  _2026-01-28_ · https://arxiv.org/abs/2601.20773v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Deployed machine learning systems must continuously evolve as data, architectures, and regulations change, often without access to original training data or model internals. In such settings, black-box copying provides a practical refactoring mechanism, i.e. upgrading legacy models by learning replicas from input-output queries alone. When restricted to hard-label outputs, copying turns into a discontinuous surface reconstruction problem from pointwise queries, severely limiting the ability to recover boundary geometry efficiently. We propose a distance-based copying (distillation) framework that replaces hard-label supervision with signed distances to the teacher's decision boundary, converting copying into a smooth regression problem that exploits local geometry. We develop an $α$-governed smoothing and regularization scheme with Hölder/Lipschitz control over the induced target surface, and introduce two model-agnostic algorithms to estimate signed distances under label-only access. Experiments on synthetic problems and UCI benchmarks show consistent improvements in fidelity and generalization accuracy over hard-label baselines, while enabling distance outputs as uncertainty-related signals for black-box replicas.

  </details>



- **Less is More: Clustered Cross-Covariance Control for Offline RL**  
  Nan Qiao, Sheng Yue, Shuning Wang, Yongheng Deng, Ju Ren  
  _2026-01-28_ · https://arxiv.org/abs/2601.20765v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  A fundamental challenge in offline reinforcement learning is distributional shift. Scarce data or datasets dominated by out-of-distribution (OOD) areas exacerbate this issue. Our theoretical analysis and experiments show that the standard squared error objective induces a harmful TD cross covariance. This effect amplifies in OOD areas, biasing optimization and degrading policy learning. To counteract this mechanism, we develop two complementary strategies: partitioned buffer sampling that restricts updates to localized replay partitions, attenuates irregular covariance effects, and aligns update directions, yielding a scheme that is easy to integrate with existing implementations, namely Clustered Cross-Covariance Control for TD (C^4). We also introduce an explicit gradient-based corrective penalty that cancels the covariance induced bias within each update. We prove that buffer partitioning preserves the lower bound property of the maximization objective, and that these constraints mitigate excessive conservatism in extreme OOD areas without altering the core behavior of policy constrained offline reinforcement learning. Empirically, our method showcases higher stability and up to 30% improvement in returns over prior methods, especially with small datasets and splits that emphasize OOD areas.

  </details>



- **Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?**  
  Hao Liang, Jiayu Cheng, Sean R. Sinclair, Yali Du  
  _2026-01-28_ · https://arxiv.org/abs/2601.20694v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Exogenous MDPs (Exo-MDPs) capture sequential decision-making where uncertainty comes solely from exogenous inputs that evolve independently of the learner's actions. This structure is especially common in operations research applications such as inventory control, energy storage, and resource allocation, where exogenous randomness (e.g., demand, arrivals, or prices) drives system behavior. Despite decades of empirical evidence that greedy, exploitation-only methods work remarkably well in these settings, theory has lagged behind: all existing regret guarantees for Exo-MDPs rely on explicit exploration or tabular assumptions. We show that exploration is unnecessary. We propose Pure Exploitation Learning (PEL) and prove the first general finite-sample regret bounds for exploitation-only algorithms in Exo-MDPs. In the tabular case, PEL achieves $\widetilde{O}(H^2|Ξ|\sqrt{K})$. For large, continuous endogenous state spaces, we introduce LSVI-PE, a simple linear-approximation method whose regret is polynomial in the feature dimension, exogenous state space, and horizon, independent of the endogenous state and action spaces. Our analysis introduces two new tools: counterfactual trajectories and Bellman-closed feature transport, which together allow greedy policies to have accurate value estimates without optimism. Experiments on synthetic and resource-management tasks show that PEL consistently outperforming baselines. Overall, our results overturn the conventional wisdom that exploration is required, demonstrating that in Exo-MDPs, pure exploitation is enough.

  </details>



- **Decoupling Perception and Calibration: Label-Efficient Image Quality Assessment Framework**  
  Xinyue Li, Zhichao Zhang, Zhiming Xu, Shubo Xu, Xiongkuo Min, Yitong Chen, Guangtao Zhai  
  _2026-01-28_ · https://arxiv.org/abs/2601.20689v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent multimodal large language models (MLLMs) have demonstrated strong capabilities in image quality assessment (IQA) tasks. However, adapting such large-scale models is computationally expensive and still relies on substantial Mean Opinion Score (MOS) annotations. We argue that for MLLM-based IQA, the core bottleneck lies not in the quality perception capacity of MLLMs, but in MOS scale calibration. Therefore, we propose LEAF, a Label-Efficient Image Quality Assessment Framework that distills perceptual quality priors from an MLLM teacher into a lightweight student regressor, enabling MOS calibration with minimal human supervision. Specifically, the teacher conducts dense supervision through point-wise judgments and pair-wise preferences, with an estimate of decision reliability. Guided by these signals, the student learns the teacher's quality perception patterns through joint distillation and is calibrated on a small MOS subset to align with human annotations. Experiments on both user-generated and AI-generated IQA benchmarks demonstrate that our method significantly reduces the need for human annotations while maintaining strong MOS-aligned correlations, making lightweight IQA practical under limited annotation budgets.

  </details>



- **CLEAR-Mamba:Towards Accurate, Adaptive and Trustworthy Multi-Sequence Ophthalmic Angiography Classification**  
  Zhuonan Wang, Wenjie Yan, Wenqiao Zhang, Xiaohui Song, Jian Ma, Ke Yao, Yibo Yu, Beng Chin Ooi  
  _2026-01-28_ · https://arxiv.org/abs/2601.20601v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Medical image classification is a core task in computer-aided diagnosis (CAD), playing a pivotal role in early disease detection, treatment planning, and patient prognosis assessment. In ophthalmic practice, fluorescein fundus angiography (FFA) and indocyanine green angiography (ICGA) provide hemodynamic and lesion-structural information that conventional fundus photography cannot capture. However, due to the single-modality nature, subtle lesion patterns, and significant inter-device variability, existing methods still face limitations in generalization and high-confidence prediction. To address these challenges, we propose CLEAR-Mamba, an enhanced framework built upon MedMamba with optimizations in both architecture and training strategy. Architecturally, we introduce HaC, a hypernetwork-based adaptive conditioning layer that dynamically generates parameters according to input feature distributions, thereby improving cross-domain adaptability. From a training perspective, we develop RaP, a reliability-aware prediction scheme built upon evidential uncertainty learning, which encourages the model to emphasize low-confidence samples and improves overall stability and reliability. We further construct a large-scale ophthalmic angiography dataset covering both FFA and ICGA modalities, comprising multiple retinal disease categories for model training and evaluation. Experimental results demonstrate that CLEAR-Mamba consistently outperforms multiple baseline models, including the original MedMamba, across various metrics-showing particular advantages in multi-disease classification and reliability-aware prediction. This study provides an effective solution that balances generalizability and reliability for modality-specific medical image classification tasks.

  </details>



- **SegRap2025: A Benchmark of Gross Tumor Volume and Lymph Node Clinical Target Volume Segmentation for Radiotherapy Planning of Nasopharyngeal Carcinoma**  
  Jia Fu, Litingyu Wang, He Li, Zihao Luo, Huamin Wang, Chenyuan Bian, Zijun Gao, Chunbin Gu, Xin Weng, Jianghao Wu, et al.  
  _2026-01-28_ · https://arxiv.org/abs/2601.20575v1 · `eess.IV`  
  <details><summary>Abstract</summary>

  Accurate delineation of Gross Tumor Volume (GTV), Lymph Node Clinical Target Volume (LN CTV), and Organ-at-Risk (OAR) from Computed Tomography (CT) scans is essential for precise radiotherapy planning in Nasopharyngeal Carcinoma (NPC). Building upon SegRap2023, which focused on OAR and GTV segmentation using single-center paired non-contrast CT (ncCT) and contrast-enhanced CT (ceCT) scans, the SegRap2025 challenge aims to enhance the generalizability and robustness of segmentation models across imaging centers and modalities. SegRap2025 comprises two tasks: Task01 addresses GTV segmentation using paired CT from the SegRap2023 dataset, with an additional external testing set to evaluate cross-center generalization, and Task02 focuses on LN CTV segmentation using multi-center training data and an unseen external testing set, where each case contains paired CT scans or a single modality, emphasizing both cross-center and cross-modality robustness. This paper presents the challenge setup and provides a comprehensive analysis of the solutions submitted by ten participating teams. For GTV segmentation task, the top-performing models achieved average Dice Similarity Coefficient (DSC) of 74.61% and 56.79% on the internal and external testing cohorts, respectively. For LN CTV segmentation task, the highest average DSC values reached 60.24%, 60.50%, and 57.23% on paired CT, ceCT-only, and ncCT-only subsets, respectively. SegRap2025 establishes a large-scale multi-center, multi-modality benchmark for evaluating the generalization and robustness in radiotherapy target segmentation, providing valuable insights toward clinically applicable automated radiotherapy planning systems. The benchmark is available at: https://hilab-git.github.io/SegRap2025_Challenge.

  </details>



- **A Practical Framework of Key Performance Indicators for Multi-Robot Lunar and Planetary Field Tests**  
  Julia Richter, David Oberacker, Gabriela Ligeza, Valentin T. Bickel, Philip Arm, William Talbot, Marvin Grosse Besselmann, Florian Kehl, Tristan Schnell, Hendrik Kolvenbach, et al.  
  _2026-01-28_ · https://arxiv.org/abs/2601.20529v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic prospecting for critical resources on the Moon, such as ilmenite, rare earth elements, and water ice, requires robust exploration methods given the diverse terrain and harsh environmental conditions. Although numerous analog field trials address these goals, comparing their results remains challenging because of differences in robot platforms and experimental setups. These missions typically assess performance using selected, scenario-specific engineering metrics that fail to establish a clear link between field performance and science-driven objectives. In this paper, we address this gap by deriving a structured framework of KPI from three realistic multi-robot lunar scenarios reflecting scientific objectives and operational constraints. Our framework emphasizes scenario-dependent priorities in efficiency, robustness, and precision, and is explicitly designed for practical applicability in field deployments. We validated the framework in a multi-robot field test and found it practical and easy to apply for efficiency- and robustness-related KPI, whereas precision-oriented KPI require reliable ground-truth data that is not always feasible to obtain in outdoor analog environments. Overall, we propose this framework as a common evaluation standard enabling consistent, goal-oriented comparison of multi-robot field trials and supporting systematic development of robotic systems for future planetary exploration.

  </details>



- **Youtu-Parsing: Perception, Structuring and Recognition via High-Parallelism Decoding**  
  Kun Yin, Yunfei Wu, Bing Liu, Zhongpeng Cai, Xiaotian Li, Huang Chen, Xin Li, Haoyu Cao, Yinsong Liu, Deqiang Jiang, et al.  
  _2026-01-28_ · https://arxiv.org/abs/2601.20430v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This paper presents Youtu-Parsing, an efficient and versatile document parsing model designed for high-performance content extraction. The architecture employs a native Vision Transformer (ViT) featuring a dynamic-resolution visual encoder to extract shared document features, coupled with a prompt-guided Youtu-LLM-2B language model for layout analysis and region-prompted decoding. Leveraging this decoupled and feature-reusable framework, we introduce a high-parallelism decoding strategy comprising two core components: token parallelism and query parallelism. The token parallelism strategy concurrently generates up to 64 candidate tokens per inference step, which are subsequently validated through a verification mechanism. This approach yields a 5--11x speedup over traditional autoregressive decoding and is particularly well-suited for highly structured scenarios, such as table recognition. To further exploit the advantages of region-prompted decoding, the query parallelism strategy enables simultaneous content prediction for multiple bounding boxes (up to five), providing an additional 2x acceleration while maintaining output quality equivalent to standard decoding. Youtu-Parsing encompasses a diverse range of document elements, including text, formulas, tables, charts, seals, and hierarchical structures. Furthermore, the model exhibits strong robustness when handling rare characters, multilingual text, and handwritten content. Extensive evaluations demonstrate that Youtu-Parsing achieves state-of-the-art (SOTA) performance on both the OmniDocBench and olmOCR-bench benchmarks. Overall, Youtu-Parsing demonstrates significant experimental value and practical utility for large-scale document intelligence applications.

  </details>



- **Dual-Modality IoT Framework for Integrated Access Control and Environmental Safety Monitoring with Real-Time Cloud Analytics**  
  Abdul Hasib, A. S. M. Ahsanul Sarkar Akib, Nihal Das Ankur, Anish Giri  
  _2026-01-28_ · https://arxiv.org/abs/2601.20366v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The integration of physical security systems with environmental safety monitoring represents a critical advancement in smart infrastructure management. Traditional approaches maintain these systems as independent silos, creating operational inefficiencies, delayed emergency responses, and increased management complexity. This paper presents a comprehensive dual-modality Internet of Things framework that seamlessly integrates RFID-based access control with multi-sensor environmental safety monitoring through a unified cloud architecture. The system comprises two coordinated subsystems: Subsystem 1 implements RFID authentication with servo-actuated gate control and real-time Google Sheets logging, while Subsystem 2 provides comprehensive safety monitoring incorporating flame detection, water flow measurement, LCD status display, and personnel identification. Both subsystems utilize ESP32 microcontrollers for edge processing and wireless connectivity. Experimental evaluation over 45 days demonstrates exceptional performance metrics: 99.2\% RFID authentication accuracy with 0.82-second average response time, 98.5\% flame detection reliability within 5-meter range, and 99.8\% cloud data logging success rate. The system maintains operational integrity during network disruptions through intelligent local caching mechanisms and achieves total implementation cost of 5,400 BDT (approximately \$48), representing an 82\% reduction compared to commercial integrated solutions. This research establishes a practical framework for synergistic security-safety integration, demonstrating that professional-grade performance can be achieved through careful architectural design and component optimization while maintaining exceptional cost-effectiveness and accessibility for diverse application scenarios.

  </details>



- **Bridging the Applicator Gap with Data-Doping:Dual-Domain Learning for Precise Bladder Segmentation in CT-Guided Brachytherapy**  
  Suresh Das, Siladittya Manna, Sayantari Ghosh  
  _2026-01-28_ · https://arxiv.org/abs/2601.20302v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Performance degradation due to covariate shift remains a major challenge for deep learning models in medical image segmentation. An open question is whether samples from a shifted distribution can effectively support learning when combined with limited target domain data. We investigate this problem in the context of bladder segmentation in CT guided gynecological brachytherapy, a critical task for accurate dose optimization and organ at risk sparing. While CT scans without brachytherapy applicators (no applicator: NA) are widely available, scans with applicators inserted (with applicator: WA) are scarce and exhibit substantial anatomical deformation and imaging artifacts, making automated segmentation particularly difficult. We propose a dual domain learning strategy that integrates NA and WA CT data to improve robustness and generalizability under covariate shift. Using a curated assorted dataset, we show that NA data alone fail to capture the anatomical and artifact related characteristics of WA images. However, introducing a modest proportion of WA data into a predominantly NA training set leads to significant performance improvements. Through systematic experiments across axial, coronal, and sagittal planes using multiple deep learning architectures, we demonstrate that doping only 10 to 30 percent WA data achieves segmentation performance comparable to models trained exclusively on WA data. The proposed approach attains Dice similarity coefficients of up to 0.94 and Intersection over Union scores of up to 0.92, indicating effective domain adaptation and improved clinical reliability. This study highlights the value of integrating anatomically similar but distribution shifted datasets to overcome data scarcity and enhance deep learning based segmentation for brachytherapy treatment planning.

  </details>


