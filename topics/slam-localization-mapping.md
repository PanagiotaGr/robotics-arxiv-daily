# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-19 07:13 UTC_

Total papers shown: **14**


---

- **Decentralized and Fully Onboard: Range-Aided Cooperative Localization and Navigation on Micro Aerial Vehicles**  
  Abhishek Goudar, Angela P. Schoellig  
  _2026-02-18_ · https://arxiv.org/abs/2602.16594v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Controlling a team of robots in a coordinated manner is challenging because centralized approaches (where all computation is performed on a central machine) scale poorly, and globally referenced external localization systems may not always be available. In this work, we consider the problem of range-aided decentralized localization and formation control. In such a setting, each robot estimates its relative pose by combining data only from onboard odometry sensors and distance measurements to other robots in the team. Additionally, each robot calculates the control inputs necessary to collaboratively navigate an environment to accomplish a specific task, for example, moving in a desired formation while monitoring an area. We present a block coordinate descent approach to localization that does not require strict coordination between the robots. We present a novel formulation for formation control as inference on factor graphs that takes into account the state estimation uncertainty and can be solved efficiently. Our approach to range-aided localization and formation-based navigation is completely decentralized, does not require specialized trajectories to maintain formation, and achieves decimeter-level positioning and formation control accuracy. We demonstrate our approach through multiple real experiments involving formation flights in diverse indoor and outdoor environments.

  </details>



- **Markerless Robot Detection and 6D Pose Estimation for Multi-Agent SLAM**  
  Markus Rueggeberg, Maximilian Ulmer, Maximilian Durner, Wout Boerdijk, Marcus Gerhard Mueller, Rudolph Triebel, Riccardo Giubilato  
  _2026-02-18_ · https://arxiv.org/abs/2602.16308v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The capability of multi-robot SLAM approaches to merge localization history and maps from different observers is often challenged by the difficulty in establishing data association. Loop closure detection between perceptual inputs of different robotic agents is easily compromised in the context of perceptual aliasing, or when perspectives differ significantly. For this reason, direct mutual observation among robots is a powerful way to connect partial SLAM graphs, but often relies on the presence of calibrated arrays of fiducial markers (e.g., AprilTag arrays), which severely limits the range of observations and frequently fails under sharp lighting conditions, e.g., reflections or overexposure. In this work, we propose a novel solution to this problem leveraging recent advances in Deep-Learning-based 6D pose estimation. We feature markerless pose estimation as part of a decentralized multi-robot SLAM system and demonstrate the benefit to the relative localization accuracy among the robotic team. The solution is validated experimentally on data recorded in a test field campaign on a planetary analogous environment.

  </details>



- **Reactive Motion Generation With Particle-Based Perception in Dynamic Environments**  
  Xiyuan Zhao, Huijun Li, Lifeng Zhu, Zhikai Wei, Xianyi Zhu, Aiguo Song  
  _2026-02-18_ · https://arxiv.org/abs/2602.16462v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reactive motion generation in dynamic and unstructured scenarios is typically subject to essentially static perception and system dynamics. Reliably modeling dynamic obstacles and optimizing collision-free trajectories under perceptive and control uncertainty are challenging. This article focuses on revealing tight connection between reactive planning and dynamic mapping for manipulators from a model-based perspective. To enable efficient particle-based perception with expressively dynamic property, we present a tensorized particle weight update scheme that explicitly maintains obstacle velocities and covariance meanwhile. Building upon this dynamic representation, we propose an obstacle-aware MPPI-based planning formulation that jointly propagates robot-obstacle dynamics, allowing future system motion to be predicted and evaluated under uncertainty. The model predictive method is shown to significantly improve safety and reactivity with dynamic surroundings. By applying our complete framework in simulated and noisy real-world environments, we demonstrate that explicit modeling of robot-obstacle dynamics consistently enhances performance over state-of-the-art MPPI-based perception-planning baselines avoiding multiple static and dynamic obstacles.

  </details>



- **Towards Autonomous Robotic Kidney Ultrasound: Spatial-Efficient Volumetric Imaging via Template Guided Optimal Pivoting**  
  Xihan Ma, Haichong Zhang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16641v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Medical ultrasound (US) imaging is a frontline tool for the diagnosis of kidney diseases. However, traditional freehand imaging procedure suffers from inconsistent, operator-dependent outcomes, lack of 3D localization information, and risks of work-related musculoskeletal disorders. While robotic ultrasound (RUS) systems offer the potential for standardized, operator-independent 3D kidney data acquisition, the existing scanning methods lack the ability to determine the optimal imaging window for efficient imaging. As a result, the scan is often blindly performed with excessive probe footprint, which frequently leads to acoustic shadowing and incomplete organ coverage. Consequently, there is a critical need for a spatially efficient imaging technique that can maximize the kidney coverage through minimum probe footprint. Here, we propose an autonomous workflow to achieve efficient kidney imaging via template-guided optimal pivoting. The system first performs an explorative imaging to generate partial observations of the kidney. This data is then registered to a kidney template to estimate the organ pose. With the kidney localized, the robot executes a fixed-point pivoting sweep where the imaging plane is aligned with the kidney long axis to minimize the probe translation. The proposed method was validated in simulation and in-vivo. Simulation results indicate that a 60% exploration ratio provides optimal balance between kidney localization accuracy and scanning efficiency. In-vivo evaluation on two male subjects demonstrates a kidney localization accuracy up to 7.36 mm and 13.84 degrees. Moreover, the optimal pivoting approach shortened the probe footprint by around 75 mm when compared with the baselines. These results valid our approach of leveraging anatomical templates to align the probe optimally for volumetric sweep.

  </details>



- **Benchmarking Adversarial Robustness and Adversarial Training Strategies for Object Detection**  
  Alexis Winter, Jean-Vincent Martini, Romaric Audigier, Angelique Loesch, Bertrand Luvison  
  _2026-02-18_ · https://arxiv.org/abs/2602.16494v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Object detection models are critical components of automated systems, such as autonomous vehicles and perception-based robots, but their sensitivity to adversarial attacks poses a serious security risk. Progress in defending these models lags behind classification, hindered by a lack of standardized evaluation. It is nearly impossible to thoroughly compare attack or defense methods, as existing work uses different datasets, inconsistent efficiency metrics, and varied measures of perturbation cost. This paper addresses this gap by investigating three key questions: (1) How can we create a fair benchmark to impartially compare attacks? (2) How well do modern attacks transfer across different architectures, especially from Convolutional Neural Networks to Vision Transformers? (3) What is the most effective adversarial training strategy for robust defense? To answer these, we first propose a unified benchmark framework focused on digital, non-patch-based attacks. This framework introduces specific metrics to disentangle localization and classification errors and evaluates attack cost using multiple perceptual metrics. Using this benchmark, we conduct extensive experiments on state-of-the-art attacks and a wide range of detectors. Our findings reveal two major conclusions: first, modern adversarial attacks against object detection models show a significant lack of transferability to transformer-based architectures. Second, we demonstrate that the most robust adversarial training strategy leverages a dataset composed of a mix of high-perturbation attacks with different objectives (e.g., spatial and semantic), which outperforms training on any single attack.

  </details>



- **SCAR: Satellite Imagery-Based Calibration for Aerial Recordings**  
  Henry Hölzemann, Michael Schleiss  
  _2026-02-18_ · https://arxiv.org/abs/2602.16349v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We introduce SCAR, a method for long-term auto-calibration refinement of aerial visual-inertial systems that exploits georeferenced satellite imagery as a persistent global reference. SCAR estimates both intrinsic and extrinsic parameters by aligning aerial images with 2D--3D correspondences derived from publicly available orthophotos and elevation models. In contrast to existing approaches that rely on dedicated calibration maneuvers or manually surveyed ground control points, our method leverages external geospatial data to detect and correct calibration degradation under field deployment conditions. We evaluate our approach on six large-scale aerial campaigns conducted over two years under diverse seasonal and environmental conditions. Across all sequences, SCAR consistently outperforms established baselines (Kalibr, COLMAP, VINS-Mono), reducing median reprojection error by a large margin, and translating these calibration gains into substantially lower visual localization rotation errors and higher pose accuracy. These results demonstrate that SCAR provides accurate, robust, and reproducible calibration over long-term aerial operations without the need for manual intervention.

  </details>



- **Visual Self-Refine: A Pixel-Guided Paradigm for Accurate Chart Parsing**  
  Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Dahua Lin  
  _2026-02-18_ · https://arxiv.org/abs/2602.16455v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  While Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities for reasoning and self-correction at the textual level, these strengths provide minimal benefits for complex tasks centered on visual perception, such as Chart Parsing. Existing models often struggle with visually dense charts, leading to errors like data omission, misalignment, and hallucination. Inspired by the human strategy of using a finger as a ``visual anchor'' to ensure accuracy when reading complex charts, we propose a new paradigm named Visual Self-Refine (VSR). The core idea of VSR is to enable a model to generate pixel-level localization outputs, visualize them, and then feed these visualizations back to itself, allowing it to intuitively inspect and correct its own potential visual perception errors. We instantiate the VSR paradigm in the domain of Chart Parsing by proposing ChartVSR. This model decomposes the parsing process into two stages: a Refine Stage, where it iteratively uses visual feedback to ensure the accuracy of all data points' Pixel-level Localizations, and a Decode Stage, where it uses these verified localizations as precise visual anchors to parse the final structured data. To address the limitations of existing benchmarks, we also construct ChartP-Bench, a new and highly challenging benchmark for chart parsing. Our work also highlights VSR as a general-purpose visual feedback mechanism, offering a promising new direction for enhancing accuracy on a wide range of vision-centric tasks.

  </details>



- **Knowledge-Embedded Latent Projection for Robust Representation Learning**  
  Weijing Tang, Ming Yuan, Zongqi Xia, Tianxi Cai  
  _2026-02-18_ · https://arxiv.org/abs/2602.16709v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Latent space models are widely used for analyzing high-dimensional discrete data matrices, such as patient-feature matrices in electronic health records (EHRs), by capturing complex dependence structures through low-dimensional embeddings. However, estimation becomes challenging in the imbalanced regime, where one matrix dimension is much larger than the other. In EHR applications, cohort sizes are often limited by disease prevalence or data availability, whereas the feature space remains extremely large due to the breadth of medical coding system. Motivated by the increasing availability of external semantic embeddings, such as pre-trained embeddings of clinical concepts in EHRs, we propose a knowledge-embedded latent projection model that leverages semantic side information to regularize representation learning. Specifically, we model column embeddings as smooth functions of semantic embeddings via a mapping in a reproducing kernel Hilbert space. We develop a computationally efficient two-step estimation procedure that combines semantically guided subspace construction via kernel principal component analysis with scalable projected gradient descent. We establish estimation error bounds that characterize the trade-off between statistical error and approximation error induced by the kernel projection. Furthermore, we provide local convergence guarantees for our non-convex optimization procedure. Extensive simulation studies and a real-world EHR application demonstrate the effectiveness of the proposed method.

  </details>



- **Causality is Key for Interpretability Claims to Generalise**  
  Shruti Joshi, Aaron Mueller, David Klindt, Wieland Brendel, Patrik Reizinger, Dhanya Sridhar  
  _2026-02-18_ · https://arxiv.org/abs/2602.16698v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Interpretability research on large language models (LLMs) has yielded important insights into model behaviour, yet recurring pitfalls persist: findings that do not generalise, and causal interpretations that outrun the evidence. Our position is that causal inference specifies what constitutes a valid mapping from model activations to invariant high-level structures, the data or assumptions needed to achieve it, and the inferences it can support. Specifically, Pearl's causal hierarchy clarifies what an interpretability study can justify. Observations establish associations between model behaviour and internal components. Interventions (e.g., ablations or activation patching) support claims how these edits affect a behavioural metric (\eg, average change in token probabilities) over a set of prompts. However, counterfactual claims -- i.e., asking what the model output would have been for the same prompt under an unobserved intervention -- remain largely unverifiable without controlled supervision. We show how causal representation learning (CRL) operationalises this hierarchy, specifying which variables are recoverable from activations and under what assumptions. Together, these motivate a diagnostic framework that helps practitioners select methods and evaluations matching claims to evidence such that findings generalise.

  </details>



- **VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection**  
  Yingyuan Yang, Tian Lan, Yifei Gao, Yimeng Lu, Wenjun He, Meng Wang, Chenghao Liu, Chen Zhang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16681v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Time-series anomaly detection (TSAD) requires identifying both immediate Point Anomalies and long-range Context Anomalies. However, existing foundation models face a fundamental trade-off: 1D temporal models provide fine-grained pointwise localization but lack a global contextual perspective, while 2D vision-based models capture global patterns but suffer from information bottlenecks due to a lack of temporal alignment and coarse-grained pointwise detection. To resolve this dilemma, we propose VETime, the first TSAD framework that unifies temporal and visual modalities through fine-grained visual-temporal alignment and dynamic fusion. VETime introduces a Reversible Image Conversion and a Patch-Level Temporal Alignment module to establish a shared visual-temporal timeline, preserving discriminative details while maintaining temporal sensitivity. Furthermore, we design an Anomaly Window Contrastive Learning mechanism and a Task-Adaptive Multi-Modal Fusion to adaptively integrate the complementary perceptual strengths of both modalities. Extensive experiments demonstrate that VETime significantly outperforms state-of-the-art models in zero-shot scenarios, achieving superior localization precision with lower computational overhead than current vision-based approaches. Code available at: https://github.com/yyyangcoder/VETime.

  </details>



- **Causal and Compositional Abstraction**  
  Robin Lorenz, Sean Tull  
  _2026-02-18_ · https://arxiv.org/abs/2602.16612v1 · `cs.LO`  
  <details><summary>Abstract</summary>

  Abstracting from a low level to a more explanatory high level of description, and ideally while preserving causal structure, is fundamental to scientific practice, to causal inference problems, and to robust, efficient and interpretable AI. We present a general account of abstractions between low and high level models as natural transformations, focusing on the case of causal models. This provides a new formalisation of causal abstraction, unifying several notions in the literature, including constructive causal abstraction, Q-$τ$ consistency, abstractions based on interchange interventions, and `distributed' causal abstractions. Our approach is formalised in terms of category theory, and uses the general notion of a compositional model with a given set of queries and semantics in a monoidal, cd- or Markov category; causal models and their queries such as interventions being special cases. We identify two basic notions of abstraction: downward abstractions mapping queries from high to low level; and upward abstractions, mapping concrete queries such as Do-interventions from low to high. Although usually presented as the latter, we show how common causal abstractions may, more fundamentally, be understood in terms of the former. Our approach also leads us to consider a new stronger notion of `component-level' abstraction, applying to the individual components of a model. In particular, this yields a novel, strengthened form of constructive causal abstraction at the mechanism-level, for which we prove characterisation results. Finally, we show that abstraction can be generalised to further compositional models, including those with a quantum semantics implemented by quantum circuits, and we take first steps in exploring abstractions between quantum compositional circuit models and high-level classical causal models as a means to explainable quantum AI.

  </details>



- **Rethinking Input Domains in Physics-Informed Neural Networks via Geometric Compactification Mappings**  
  Zhenzhen Huang, Haoyu Bian, Jiaquan Zhang, Yibei Liu, Kuien Liu, Caiyan Qin, Guoqing Wang, Yang Yang, Chaoning Zhang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16193v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Several complex physical systems are governed by multi-scale partial differential equations (PDEs) that exhibit both smooth low-frequency components and localized high-frequency structures. Existing physics-informed neural network (PINN) methods typically train with fixed coordinate system inputs, where geometric misalignment with these structures induces gradient stiffness and ill-conditioning that hinder convergence. To address this issue, we introduce a mapping paradigm that reshapes the input coordinates through differentiable geometric compactification mappings and couples the geometric structure of PDEs with the spectral properties of residual operators. Based on this paradigm, we propose Geometric Compactification (GC)-PINN, a framework that introduces three mapping strategies for periodic boundaries, far-field scale expansion, and localized singular structures in the input domain without modifying the underlying PINN architecture. Extensive empirical evaluation demonstrates that this approach yields more uniform residual distributions and higher solution accuracy on representative 1D and 2D PDEs, while improving training stability and convergence speed.

  </details>



- **Discrete Stochastic Localization for Non-autoregressive Generation**  
  Yunshu Wu, Jiayi Cheng, Partha Thakuria, Rob Brekelmans, Evangelos E. Papalexakis, Greg Ver Steeg  
  _2026-02-18_ · https://arxiv.org/abs/2602.16169v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Non-autoregressive (NAR) generation reduces decoding latency by predicting many tokens in parallel, but iterative refinement often suffers from error accumulation and distribution shift under self-generated drafts. Masked diffusion language models (MDLMs) and their remasking samplers (e.g., ReMDM) can be viewed as modern NAR iterative refinement, where generation repeatedly revises a partially observed draft. In this work we show that \emph{training alone} can substantially improve the step-efficiency of MDLM/ReMDM sampling. We propose \textsc{DSL} (Discrete Stochastic Localization), which trains a single SNR-invariant denoiser across a continuum of corruption levels, bridging intermediate draft noise and mask-style endpoint corruption within one Diffusion Transformer. On OpenWebText, \textsc{DSL} fine-tuning yields large MAUVE gains at low step budgets, surpassing the MDLM+ReMDM baseline with \(\sim\)4$\times$ fewer denoiser evaluations, and matches autoregressive quality at high budgets. Analyses show improved self-correction and uncertainty calibration, making remasking markedly more compute-efficient.

  </details>



- **Uncertainty-Guided Inference-Time Depth Adaptation for Transformer-Based Visual Tracking**  
  Patrick Poggi, Divake Kumar, Theja Tulabandhula, Amit Ranjan Trivedi  
  _2026-02-18_ · https://arxiv.org/abs/2602.16160v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Transformer-based single-object trackers achieve state-of-the-art accuracy but rely on fixed-depth inference, executing the full encoder--decoder stack for every frame regardless of visual complexity, thereby incurring unnecessary computational cost in long video sequences dominated by temporally coherent frames. We propose UncL-STARK, an architecture-preserving approach that enables dynamic, uncertainty-aware depth adaptation in transformer-based trackers without modifying the underlying network or adding auxiliary heads. The model is fine-tuned to retain predictive robustness at multiple intermediate depths using random-depth training with knowledge distillation, thus enabling safe inference-time truncation. At runtime, we derive a lightweight uncertainty estimate directly from the model's corner localization heatmaps and use it in a feedback-driven policy that selects the encoder and decoder depth for the next frame based on the prediction confidence by exploiting temporal coherence in video. Extensive experiments on GOT-10k and LaSOT demonstrate up to 12\% GFLOPs reduction, 8.9\% latency reduction, and 10.8\% energy savings while maintaining tracking accuracy within 0.2\% of the full-depth baseline across both short-term and long-term sequences.

  </details>


