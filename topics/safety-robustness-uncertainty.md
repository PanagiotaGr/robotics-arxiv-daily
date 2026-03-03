# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-03-03 07:06 UTC_

Total papers shown: **20**


---

- **CHOP: Counterfactual Human Preference Labels Improve Obstacle Avoidance in Visuomotor Navigation Policies**  
  Gershom Seneviratne, Jianyu An, Vaibhav Shende, Sahire Ellahy, Yaxita Amin, Kondapi Manasanjani, Samarth Chopra, Jonathan Deepak Kannan, Dinesh Manocha  
  _2026-03-02_ · https://arxiv.org/abs/2603.02004v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visuomotor navigation policies have shown strong perception-action coupling for embodied agents, yet they often struggle with safe navigation and dynamic obstacle avoidance in complex real-world environments. We introduce CHOP, a novel approach that leverages Counterfactual Human Preference Labels to align visuomotor navigation policies towards human intuition of safety and obstacle avoidance in navigation. In CHOP, for each visual observation, the robot's executed trajectory is included among a set of counterfactual navigation trajectories: alternative trajectories the robot could have followed under identical conditions. Human annotators provide pairwise preference labels over these trajectories based on anticipated outcomes such as collision risk and path efficiency. These aggregated preferences are then used to fine-tune visuomotor navigation policies, aligning their behavior with human preferences in navigation. Experiments on the SCAND dataset show that visuomotor navigation policies fine-tuned with CHOP reduce near-collision events by 49.7%, decrease deviation from human-preferred trajectories by 45.0%, and increase average obstacle clearance by 19.8% on average across multiple state-of-the-art models, compared to their pretrained baselines. These improvements transfer to real-world deployments on a Ghost Robotics Vision60 quadruped, where CHOP-aligned policies improve average goal success rates by 24.4%, increase minimum obstacle clearance by 6.8%, reduce collision and intervention events by 45.7%, and improve normalized path completion by 38.6% on average across navigation scenarios, compared to their pretrained baselines. Our results highlight the value of counterfactual preference supervision in bridging the gap between large-scale visuomotor policies and human-aligned, safety-aware embodied navigation.

  </details>



- **Real-Time Thermal-Inertial Odometry on Embedded Hardware for High-Speed GPS-Denied Flight**  
  Austin Stone, Mark Petersen, Cammy Peterson  
  _2026-03-02_ · https://arxiv.org/abs/2603.02114v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a real-time monocular thermal-inertial odometry system designed for high-velocity, GPS-denied flight on embedded hardware. The system fuses measurements from a FLIR Boson+ 640 longwave infrared camera, a high-rate IMU, a laser range finder, a barometer, and a magnetometer within a fixed-lag factor graph. To sustain reliable feature tracks under motion blur, low contrast, and rapid viewpoint changes, we employ a lightweight thermal-optimized front-end with multi-stage feature filtering. Laser range finder measurements provide per-feature depth priors that stabilize scale during weakly observable motion. High-rate inertial data is first pre-filtered using a Chebyshev Type II infinite impulse response (IIR) filter and then preintegrated, improving robustness to airframe vibrations during aggressive maneuvers. To address barometric altitude errors induced at high airspeeds, we train an uncertainty-aware gated recurrent unit (GRU) network that models the temporal dynamics of static pressure distortion, outperforming polynomial and multi-layer perceptron (MLP) baselines. Integrated on an NVIDIA Jetson Xavier NX, the complete system supports closed-loop quadrotor flight at 30 m/s with drift under 2% over kilometer-scale trajectories. These contributions expand the operational envelope of thermal-inertial navigation, enabling reliable high-speed flight in visually degraded and GPS-denied environments.

  </details>



- **Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control**  
  Peng Yu, Xin Wang, Ning Tan  
  _2026-03-02_ · https://arxiv.org/abs/2603.01751v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Continuum robots possess high flexibility and redundancy, making them well suited for safe interaction in complex environments, yet their continuous deformation and nonlinear dynamics pose fundamental challenges to perception, modeling, and control. Existing vision-based control approaches often rely on end-to-end learning, achieving shape regulation without explicit awareness of robot geometry or its interaction with the environment. Here, we introduce a shape-interpretable visual self-modeling framework for continuum robots that enables geometry-aware control. Robot shapes are encoded from multi-view planar images using a Bezier-curve representation, transforming visual observations into a compact and physically meaningful shape space that uniquely characterizes the robot's three-dimensional configuration. Based on this representation, neural ordinary differential equations are employed to self-model both shape and end-effector dynamics directly from data, enabling hybrid shape-position control without analytical models or dense body markers. The explicit geometric structure of the learned shape space allows the robot to reason about its body and surroundings, supporting environment-aware behaviors such as obstacle avoidance and self-motion while maintaining end-effector objectives. Experiments on a cable-driven continuum robot demonstrate accurate shape-position regulation and tracking, with shape errors within 1.56% of image resolution and end-effector errors within 2% of robot length, as well as robust performance in constrained environments. By elevating visual shape representations from two-dimensional observations to an interpretable three-dimensional self-model, this work establishes a principled alternative to vision-based end-to-end control and advances autonomous, geometry-aware manipulation for continuum robots.

  </details>



- **Conformal Policy Control**  
  Drew Prinster, Clara Fannjiang, Ji Won Park, Kyunghyun Cho, Anqi Liu, Suchi Saria, Samuel Stanton  
  _2026-03-02_ · https://arxiv.org/abs/2603.02196v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  An agent must try new behaviors to explore and improve. In high-stakes environments, an agent that violates safety constraints may cause harm and must be taken offline, curtailing any future interaction. Imitating old behavior is safe, but excessive conservatism discourages exploration. How much behavior change is too much? We show how to use any safe reference policy as a probabilistic regulator for any optimized but untested policy. Conformal calibration on data from the safe policy determines how aggressively the new policy can act, while provably enforcing the user's declared risk tolerance. Unlike conservative optimization methods, we do not assume the user has identified the correct model class nor tuned any hyperparameters. Unlike previous conformal methods, our theory provides finite-sample guarantees even for non-monotonic bounded constraint functions. Our experiments on applications ranging from natural language question answering to biomolecular engineering show that safe exploration is not only possible from the first moment of deployment, but can also improve performance.

  </details>



- **LAD-Drive: Bridging Language and Trajectory with Action-Aware Diffusion Transformers**  
  Fabian Schmidt, Karol Fedurko, Markus Enzweiler, Abhinav Valada  
  _2026-03-02_ · https://arxiv.org/abs/2603.02035v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While multimodal large language models (MLLMs) provide advanced reasoning for autonomous driving, translating their discrete semantic knowledge into continuous trajectories remains a fundamental challenge. Existing methods often rely on unimodal planning heads that inherently limit their ability to represent multimodal driving behavior. Furthermore, most generative approaches frequently condition on one-hot encoded actions, discarding the nuanced navigational uncertainty critical for complex scenarios. To resolve these limitations, we introduce LAD-Drive, a generative framework that structurally disentangles high-level intention from low-level spatial planning. LAD-Drive employs an action decoder to infer a probabilistic meta-action distribution, establishing an explicit belief state that preserves the nuanced intent typically lost by one-hot encodings. This distribution, fused with the vehicle's kinematic state, conditions an action-aware diffusion decoder that utilizes a truncated denoising process to refine learned motion anchors into safe, kinematically feasible trajectories. Extensive evaluations on the LangAuto benchmark demonstrate that LAD-Drive achieves state-of-the-art results, outperforming competitive baselines by up to 59% in Driving Score while significantly reducing route deviations and collisions. We will publicly release the code and models on https://github.com/iis-esslingen/lad-drive.

  </details>



- **SaferPath: Hierarchical Visual Navigation with Learned Guidance and Safety-Constrained Control**  
  Lingjie Zhang, Zeyu Jiang, Changhao Chen  
  _2026-03-02_ · https://arxiv.org/abs/2603.01898v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual navigation is a core capability for mobile robots, yet end-to-end learning-based methods often struggle with generalization and safety in unseen, cluttered, or narrow environments. These limitations are especially pronounced in dense indoor settings, where collisions are likely and end-to-end models frequently fail. To address this, we propose SaferPath, a hierarchical visual navigation framework that leverages learned guidance from existing end-to-end models and refines it through a safety-constrained optimization-control module. SaferPath transforms visual observations into a traversable-area map and refines guidance trajectories using Model Predictive Stein Variational Evolution Strategy (MP-SVES), efficiently generating safe trajectories in only a few iterations. The refined trajectories are tracked by an MPC controller, ensuring robust navigation in complex environments. Extensive experiments in scenarios with unseen obstacles, dense unstructured spaces, and narrow corridors demonstrate that SaferPath consistently improves success rates and reduces collisions, outperforming representative baselines such as ViNT and NoMaD, and enabling safe navigation in challenging real-world settings.

  </details>



- **LaST-VLA: Thinking in Latent Spatio-Temporal Space for Vision-Language-Action in Autonomous Driving**  
  Yuechen Luo, Fang Li, Shaoqing Xu, Yang Ji, Zehan Zhang, Bing Wang, Yuannan Shen, Jianwei Cui, Long Chen, Guang Chen, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01928v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  While Vision-Language-Action (VLA) models have revolutionized autonomous driving by unifying perception and planning, their reliance on explicit textual Chain-of-Thought (CoT) leads to semantic-perceptual decoupling and perceptual-symbolic conflicts. Recent shifts toward latent reasoning attempt to bypass these bottlenecks by thinking in continuous hidden space. However, without explicit intermediate constraints, standard latent CoT often operates as a physics-agnostic representation. To address this, we propose the Latent Spatio-Temporal VLA (LaST-VLA), a framework shifting the reasoning paradigm from discrete symbolic processing into a physically grounded Latent Spatio-Temporal CoT. By implementing a dual-feature alignment mechanism, we distill geometric constraints from 3D foundation models and dynamic foresight from world models directly into the latent space. Coupled with a progressive SFT training strategy that transitions from feature alignment to trajectory generation, and refined via Reinforcement Learning with Group Relative Policy Optimization (GRPO) to ensure safety and rule compliance. \method~setting a new record on NAVSIM v1 (91.3 PDMS) and NAVSIM v2 (87.1 EPDMS), while excelling in spatial-temporal reasoning on SURDS and NuDynamics benchmarks.

  </details>



- **On the Rate of Convergence of GD in Non-linear Neural Networks: An Adversarial Robustness Perspective**  
  Guy Smorodinsky, Sveta Gimpleson, Itay Safran  
  _2026-03-02_ · https://arxiv.org/abs/2603.02095v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We study the convergence dynamics of Gradient Descent (GD) in a minimal binary classification setting, consisting of a two-neuron ReLU network and two training instances. We prove that even under these strong simplifying assumptions, while GD successfully converges to an optimal robustness margin, effectively maximizing the distance between the decision boundary and the training points, this convergence occurs at a prohibitively slow rate, scaling strictly as $Θ(1/\ln(t))$. To the best of our knowledge, this establishes the first explicit lower bound on the convergence rate of the robustness margin in a non-linear model. Through empirical simulations, we further demonstrate that this inherent failure mode is pervasive, exhibiting the exact same tight convergence rate across multiple natural network initializations. Our theoretical guarantees are derived via a rigorous analysis of the GD trajectories across the distinct activation patterns of the model. Specifically, we develop tight control over the system's dynamics to bound the trajectory of the decision boundary, overcoming the primary technical challenge introduced by the non-linear nature of the architecture.

  </details>



- **LOCUS: A Distribution-Free Loss-Quantile Score for Risk-Aware Predictions**  
  Matheus Barreto, Mário de Castro, Thiago R. Ramos, Denis Valle, Rafael Izbicki  
  _2026-03-02_ · https://arxiv.org/abs/2603.01971v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  Modern machine learning models can be accurate on average yet still make mistakes that dominate deployment cost. We introduce Locus, a distribution-free wrapper that produces a per-input loss-scale reliability score for a fixed prediction function. Rather than quantifying uncertainty about the label, Locus models the realized loss of the prediction function using any engine that outputs a predictive distribution for the loss given an input. A simple split-calibration step turns this function into a distribution-free interpretable score that is comparable across inputs and can be read as an upper loss level. The score is useful on its own for ranking, and it can optionally be thresholded to obtain a transparent flagging rule with distribution-free control of large-loss events. Experiments across 13 regression benchmarks show that Locus yields effective risk ranking and reduces large-loss frequency compared to standard heuristics.

  </details>



- **LiveCultureBench: a Multi-Agent, Multi-Cultural Benchmark for Large Language Models in Dynamic Social Simulations**  
  Viet-Thanh Pham, Lizhen Qu, Thuy-Trang Vu, Gholamreza Haffari, Dinh Phung  
  _2026-03-02_ · https://arxiv.org/abs/2603.01952v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) are increasingly deployed as autonomous agents, yet evaluations focus primarily on task success rather than cultural appropriateness or evaluator reliability. We introduce LiveCultureBench, a multi-cultural, dynamic benchmark that embeds LLMs as agents in a simulated town and evaluates them on both task completion and adherence to socio-cultural norms. The simulation models a small city as a location graph with synthetic residents having diverse demographic and cultural profiles. Each episode assigns one resident a daily goal while others provide social context. An LLM-based verifier generates structured judgments on norm violations and task progress, which we aggregate into metrics capturing task-norm trade-offs and verifier uncertainty. Using LiveCultureBench across models and cultural profiles, we study (i) cross-cultural robustness of LLM agents, (ii) how they balance effectiveness against norm sensitivity, and (iii) when LLM-as-a-judge evaluation is reliable for automated benchmarking versus when human oversight is needed.

  </details>



- **From Leaderboard to Deployment: Code Quality Challenges in AV Perception Repositories**  
  Mateus Karvat, Bram Adams, Sidney Givigi  
  _2026-03-02_ · https://arxiv.org/abs/2603.02194v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous vehicle (AV) perception models are typically evaluated solely on benchmark performance metrics, with limited attention to code quality, production readiness and long-term maintainability. This creates a significant gap between research excellence and real-world deployment in safety-critical systems subject to international safety standards. To address this gap, we present the first large-scale empirical study of software quality in AV perception repositories, systematically analyzing 178 unique models from the KITTI and NuScenes 3D Object Detection leaderboards. Using static analysis tools (Pylint, Bandit, and Radon), we evaluated code errors, security vulnerabilities, maintainability, and development practices. Our findings revealed that only 7.3% of the studied repositories meet basic production-readiness criteria, defined as having zero critical errors and no high-severity security vulnerabilities. Security issues are highly concentrated, with the top five issues responsible for almost 80% of occurrences, which prompted us to develop a set of actionable guidelines to prevent them. Additionally, the adoption of Continuous Integration/Continuous Deployment pipelines was correlated with better code maintainability. Our findings highlight that leaderboard performance does not reflect production readiness and that targeted interventions could substantially improve the quality and safety of AV perception code.

  </details>



- **$π$-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs**  
  Siting Wang, Xiaofeng Wang, Zheng Zhu, Minnan Pei, Xinyu Cui, Cheng Deng, Jian Zhao, Guan Huang, Haifeng Zhang, Jun Wang  
  _2026-03-02_ · https://arxiv.org/abs/2603.02083v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Flow-based vision-language-action (VLA) models excel in embodied control but suffer from intractable likelihoods during multi-step sampling, hindering online reinforcement learning. We propose \textbf{\textit{$\boldsymbolπ$-StepNFT}} (Step-wise Negative-aware Fine-Tuning), a critic-and-likelihood-free framework that requires only a single forward pass per optimization step and eliminates auxiliary value networks. We identify that wider exploration spaces necessitate finer-grained, step-wise guidance for alignment. Empirically, $π$-StepNFT unlocks latent potential on LIBERO with competitive few-shot robustness. Moreover, it achieves superior generalization on ManiSkill, outperforming value-based baselines in OOD scenarios by preventing overfitting to multimodal features. This property offers a scalable solution promising for complex real-world applications.

  </details>



- **A Resource-Rational Principle for Modeling Visual Attention Control**  
  Yunpeng Bai  
  _2026-03-02_ · https://arxiv.org/abs/2603.02056v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Understanding how people allocate visual attention is central to Human-Computer Interaction (HCI), yet existing computational models of attention are often either descriptive, task-specific, or difficult to interpret. My dissertation develops a resource-rational, simulation-based framework for modeling visual attention as a sequential decision-making process under perceptual, memory, and time constraints. I formalize visual tasks, such as reading and multitasking, as bounded-optimal control problems using Partially Observable Markov Decision Processes, enabling eye-movement behaviors such as fixation and attention switching to emerge from rational adaptation rather than being hand-coded or purely data-driven. These models are instantiated in simulation environments spanning traditional text reading and reading-while-walking with smart glasses, where they reproduce classic empirical effects, explain observed trade-offs between comprehension and safety, and generate novel predictions under time pressure and interface variation. Collectively, this work contributes a unified computational account of visual attention, offering new tools for theory-driven and resource-efficient HCI design.

  </details>



- **CausalWrap: Model-Agnostic Causal Constraint Wrappers for Tabular Synthetic Data**  
  Amir Asiaee, Zhuohui J. Liang, Chao Yan  
  _2026-03-02_ · https://arxiv.org/abs/2603.02015v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Tabular synthetic data generators are typically trained to match observational distributions, which can yield high conventional utility (e.g., column correlations, predictive accuracy) yet poor preservation of structural relations relevant to causal analysis and out-of-distribution (OOD) reasoning. When the downstream use of synthetic data involves causal reasoning -- estimating treatment effects, evaluating policies, or testing mediation pathways -- merely matching the observational distribution is insufficient: structural fidelity and treatment-mechanism preservation become essential. We propose CausalWrap (CW), a model-agnostic wrapper that injects partial causal knowledge (PCK) -- trusted edges, forbidden edges, and qualitative/monotonic constraints -- into any pretrained base generator (GAN, VAE, or diffusion model), without requiring access to its internals. CW learns a lightweight, differentiable post-hoc correction map applied to samples from the base generator, optimized with causal penalty terms under an augmented-Lagrangian schedule. We provide theoretical results connecting penalty-based optimization to constraint satisfaction and relating approximate factorization to joint distributional control. We validate CW on simulated structural causal models (SCMs) with known ground-truth interventions, semi-synthetic causal benchmarks (IHDP and an ACIC-style suite), and a real-world ICU cohort (MIMIC-IV) with expert-elicited partial graphs. CW improves causal fidelity across diverse base generators -- e.g., reducing average treatment effect (ATE) error by up to 63% on ACIC and lifting ATE agreement from 0.00 to 0.38 on the intensive care unit (ICU) cohort -- while largely retaining conventional utility.

  </details>



- **Efficient RLVR Training via Weighted Mutual Information Data Selection**  
  Xinyu Zhou, Boyu Zhu, Haotian Zhang, Huiming Wang, Zhijiang Guo  
  _2026-03-02_ · https://arxiv.org/abs/2603.01907v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) plays a central role in improving the reasoning and alignment of large language models, yet its efficiency critically depends on how training data are selected. Existing online selection strategies predominantly rely on difficulty-based heuristics, favouring datapoints with intermediate success rates, implicitly equating difficulty with informativeness and neglecting epistemic uncertainty arising from limited evidence. We introduce InSight, an INformation-guided data SamplInG metHod for RL Training, grounded in a weighted mutual information objective. By modeling data outcomes with Bayesian latent success rates, we show that expected uncertainty reduction decomposes into complementary difficulty- and evidence-dependent components, revealing a fundamental limitation of difficulty-only selection. Leveraging this observation, InSight constructs a stable acquisition score based on the mean belief of datapoints' success rather than noisy sampled outcomes, and naturally extends to multi-rollout settings common in reinforcement learning with verifiable rewards (RLVR). Extensive experiments demonstrate that InSight consistently achieves state-of-the-art performance and improves training efficiency, including a +1.41 average gain on Planning & Mathmatics benchmarks, +1.01 improvement on general reasoning, and up to ~2.2x acceleration, with negligible additional computational overhead.

  </details>



- **ScreenAnt: Transparent On-Screen Antennas for 6G**  
  Shun Zhuge, Qing Wang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01906v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  6G will require on-device antenna systems to operate at ultra-high frequency bands, achieve robust beamforming on the compact user devices, and be blockage-robust. Conventional edge-mounted antennas on devices have limited apertures, suffer from the 'death grip' caused by user-induced blockage, and have poor scalability at mmWave and sub-THz bands. To address these issues, motivated by the rapid evolution of transparent materials and antennas, we propose ScreenAnt in this work--which integrates a transparent antenna array onto the screens of future mobile devices. Specifically, we propose using a transparent on-screen uniform planar array and develop a framework to model its electromagnetic property, spatial configuration, and blockage robustness under realistic user-induced blockage. We also design a gradient-ascent-based algorithm to efficiently optimize power and phase control of on-screen antennas to maximize ScreenAnt's spectral efficiency. Our thorough simulations show that the proposed ScreenAnt can increase the uplink spectral efficiency by over 50% compared to edge-mounted antennas at 28 GHz, and by more than 150% at 300 GHz. ScreenAnt also demonstrates strong robustness against user-induced blockage, paving the way for practical and high-capacity 6G user device designs.

  </details>



- **Diagnosing Generalization Failures from Representational Geometry Markers**  
  Chi-Ning Chou, Artem Kirsanov, Yao-Yuan Yang, SueYeon Chung  
  _2026-03-02_ · https://arxiv.org/abs/2603.01879v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Generalization, the ability to perform well beyond the training context, is a hallmark of biological and artificial intelligence, yet anticipating unseen failures remains a central challenge. Conventional approaches often take a ``bottom-up'' mechanistic route by reverse-engineering interpretable features or circuits to build explanatory models. While insightful, these methods often struggle to provide the high-level, predictive signals for anticipating failure in real-world deployment. Here, we propose using a ``top-down'' approach to studying generalization failures inspired by medical biomarkers: identifying system-level measurements that serve as robust indicators of a model's future performance. Rather than mapping out detailed internal mechanisms, we systematically design and test network markers to probe structure, function links, identify prognostic indicators, and validate predictions in real-world settings. In image classification, we find that task-relevant geometric properties of in-distribution (ID) object manifolds consistently forecast poor out-of-distribution (OOD) generalization. In particular, reductions in two geometric measures, effective manifold dimensionality and utility, predict weaker OOD performance across diverse architectures, optimizers, and datasets. We apply this finding to transfer learning with ImageNet-pretrained models. We consistently find that the same geometric patterns predict OOD transfer performance more reliably than ID accuracy. This work demonstrates that representational geometry can expose hidden vulnerabilities, offering more robust guidance for model selection and AI interpretability.

  </details>



- **GroupEnsemble: Efficient Uncertainty Estimation for DETR-based Object Detection**  
  Yutong Yang, Katarina Popović, Julian Wiederer, Markus Braun, Vasileios Belagiannis, Bin Yang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01847v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Detection Transformer (DETR) and its variants show strong performance on object detection, a key task for autonomous systems. However, a critical limitation of these models is that their confidence scores only reflect semantic uncertainty, failing to capture the equally important spatial uncertainty. This results in an incomplete assessment of the detection reliability. On the other hand, Deep Ensembles can tackle this by providing high-quality spatial uncertainty estimates. However, their immense memory consumption makes them impractical for real-world applications. A cheaper alternative, Monte Carlo (MC) Dropout, suffers from high latency due to the need of multiple forward passes during inference to estimate uncertainty. To address these limitations, we introduce GroupEnsemble, an efficient and effective uncertainty estimation method for DETR-like models. GroupEnsemble simultaneously predicts multiple individual detection sets by feeding additional diverse groups of object queries to the transformer decoder during inference. Each query group is transformed by the shared decoder in isolation and predicts a complete detection set for the same input. An attention mask is applied to the decoder to prevent inter-group query interactions, ensuring each group detects independently to achieve reliable ensemble-based uncertainty estimation. By leveraging the decoder's inherent parallelism, GroupEnsemble efficiently estimates uncertainty in a single forward pass without sequential repetition. We validated our method under autonomous driving scenes and common daily scenes using the Cityscapes and COCO datasets, respectively. The results show that a hybrid approach combining MC-Dropout and GroupEnsemble outperforms Deep Ensembles on several metrics at a fraction of the cost. The code is available at https://github.com/yutongy98/GroupEnsemble.

  </details>



- **PleaSQLarify: Visual Pragmatic Repair for Natural Language Database Querying**  
  Robin Shing Moon Chan, Rita Sevastjanova, Mennatallah El-Assady  
  _2026-03-02_ · https://arxiv.org/abs/2603.01795v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Natural language database interfaces broaden data access, yet they remain brittle under input ambiguity. Standard approaches often collapse uncertainty into a single query, offering little support for mismatches between user intent and system interpretation. We reframe this challenge through pragmatic inference: while users economize expressions, systems operate on priors over the action space that may not align with the users'. In this view, pragmatic repair -- incremental clarification through minimal interaction -- is a natural strategy for resolving underspecification. We present \textsc{PleaSQLarify}, which operationalizes pragmatic repair by structuring interaction around interpretable decision variables that enable efficient clarification. A visual interface complements this by surfacing the action space for exploration, requesting user disambiguation, and making belief updates traceable across turns. In a study with twelve participants, \textsc{PleaSQLarify} helped users recognize alternative interpretations and efficiently resolve ambiguity. Our findings highlight pragmatic repair as a design principle that fosters effective user control in natural language interfaces.

  </details>



- **Discrete World Models via Regularization**  
  Davide Bizzaro, Luciano Serafini  
  _2026-03-02_ · https://arxiv.org/abs/2603.01748v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  World models aim to capture the states and dynamics of an environment in a compact latent space. Moreover, using Boolean state representations is particularly useful for search heuristics and symbolic reasoning and planning. Existing approaches keep latents informative via decoder-based reconstruction, or instead via contrastive or reward signals. In this work, we introduce Discrete World Models via Regularization (DWMR): a reconstruction-free and contrastive-free method for unsupervised Boolean world-model learning. In particular, we introduce a novel world-modeling loss that couples latent prediction with specialized regularizers. Such regularizers maximize the entropy and independence of the representation bits through variance, correlation, and coskewness penalties, while simultaneously enforcing a locality prior for sparse action changes. To enable effective optimization, we also introduce a novel training scheme improving robustness to discrete roll-outs. Experiments on two benchmarks with underlying combinatorial structure show that DWMR learns more accurate representations and transitions than reconstruction-based alternatives. Finally, DWMR can also be paired with an auxiliary reconstruction decoder, and this combination yields additional gains.

  </details>


