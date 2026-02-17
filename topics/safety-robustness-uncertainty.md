# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-17 07:12 UTC_

Total papers shown: **13**


---

- **Multimodal Covariance Steering in Belief Space with Active Probing and Influence for Autonomous Driving**  
  Devodita Chakravarty, John Dolan, Yiwei Lyu  
  _2026-02-16_ · https://arxiv.org/abs/2602.14540v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous driving in complex traffic requires reasoning under uncertainty. Common approaches rely on prediction-based planning or risk-aware control, but these are typically treated in isolation, limiting their ability to capture the coupled nature of action and inference in interactive settings. This gap becomes especially critical in uncertain scenarios, where simply reacting to predictions can lead to unsafe maneuvers or overly conservative behavior. Our central insight is that safe interaction requires not only estimating human behavior but also shaping it when ambiguity poses risks. To this end, we introduce a hierarchical belief model that structures human behavior across coarse discrete intents and fine motion modes, updated via Bayesian inference for interpretable multi-resolution reasoning. On top of this, we develop an active probing strategy that identifies when multimodal ambiguity in human predictions may compromise safety and plans disambiguating actions that both reveal intent and gently steer human decisions toward safer outcomes. Finally, a runtime risk-evaluation layer based on Conditional Value-at-Risk (CVaR) ensures that all probing actions remain within human risk tolerance during influence. Our simulations in lane-merging and unsignaled intersection scenarios demonstrate that our approach achieves higher success rates and shorter completion times compared to existing methods. These results highlight the benefit of coupling belief inference, probing, and risk monitoring, yielding a principled and interpretable framework for planning under uncertainty.

  </details>



- **ManeuverNet: A Soft Actor-Critic Framework for Precise Maneuvering of Double-Ackermann-Steering Robots with Optimized Reward Functions**  
  Kohio Deflesselle, Mélodie Daniel, Aly Magassouba, Miguel Aranda, Olivier Ly  
  _2026-02-16_ · https://arxiv.org/abs/2602.14726v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous control of double-Ackermann-steering robots is essential in agricultural applications, where robots must execute precise and complex maneuvers within a limited space. Classical methods, such as the Timed Elastic Band (TEB) planner, can address this problem, but they rely on parameter tuning, making them highly sensitive to changes in robot configuration or environment and impractical to deploy without constant recalibration. At the same time, end-to-end deep reinforcement learning (DRL) methods often fail due to unsuitable reward functions for non-holonomic constraints, resulting in sub-optimal policies and poor generalization. To address these challenges, this paper presents ManeuverNet, a DRL framework tailored for double-Ackermann systems, combining Soft Actor-Critic with CrossQ. Furthermore, ManeuverNet introduces four specifically designed reward functions to support maneuver learning. Unlike prior work, ManeuverNet does not depend on expert data or handcrafted guidance. We extensively evaluate ManeuverNet against both state-of-the-art DRL baselines and the TEB planner. Experimental results demonstrate that our framework substantially improves maneuverability and success rates, achieving more than a 40% gain over DRL baselines. Moreover, ManeuverNet effectively mitigates the strong parameter sensitivity observed in the TEB planner. In real-world trials, ManeuverNet achieved up to a 90% increase in maneuvering trajectory efficiency, highlighting its robustness and practical applicability.

  </details>



- **The Well-Tempered Classifier: Some Elementary Properties of Temperature Scaling**  
  Pierre-Alexandre Mattei, Bruno Loureiro  
  _2026-02-16_ · https://arxiv.org/abs/2602.14862v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  Temperature scaling is a simple method that allows to control the uncertainty of probabilistic models. It is mostly used in two contexts: improving the calibration of classifiers and tuning the stochasticity of large language models (LLMs). In both cases, temperature scaling is the most popular method for the job. Despite its popularity, a rigorous theoretical analysis of the properties of temperature scaling has remained elusive. We investigate here some of these properties. For classification, we show that increasing the temperature increases the uncertainty in the model in a very general sense (and in particular increases its entropy). However, for LLMs, we challenge the common claim that increasing temperature increases diversity. Furthermore, we introduce two new characterisations of temperature scaling. The first one is geometric: the tempered model is shown to be the information projection of the original model onto the set of models with a given entropy. The second characterisation clarifies the role of temperature scaling as a submodel of more general linear scalers such as matrix scaling and Dirichlet calibration: we show that temperature scaling is the only linear scaler that does not change the hard predictions of the model.

  </details>



- **Advances in Global Solvers for 3D Vision**  
  Zhenjun Zhao, Heng Yang, Bangyan Liao, Yingping Zeng, Shaocheng Yan, Yingdong Gu, Peidong Liu, Yi Zhou, Haoang Li, Javier Civera  
  _2026-02-16_ · https://arxiv.org/abs/2602.14662v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Global solvers have emerged as a powerful paradigm for 3D vision, offering certifiable solutions to nonconvex geometric optimization problems traditionally addressed by local or heuristic methods. This survey presents the first systematic review of global solvers in geometric vision, unifying the field through a comprehensive taxonomy of three core paradigms: Branch-and-Bound (BnB), Convex Relaxation (CR), and Graduated Non-Convexity (GNC). We present their theoretical foundations, algorithmic designs, and practical enhancements for robustness and scalability, examining how each addresses the fundamental nonconvexity of geometric estimation problems. Our analysis spans ten core vision tasks, from Wahba problem to bundle adjustment, revealing the optimality-robustness-scalability trade-offs that govern solver selection. We identify critical future directions: scaling algorithms while maintaining guarantees, integrating data-driven priors with certifiable optimization, establishing standardized benchmarks, and addressing societal implications for safety-critical deployment. By consolidating theoretical foundations, practical advances, and broader impacts, this survey provides a unified perspective and roadmap toward certifiable, trustworthy perception for real-world applications. A continuously-updated literature summary and companion code tutorials are available at https://github.com/ericzzj1989/Awesome-Global-Solvers-for-3D-Vision.

  </details>



- **Towards Selection as Power: Bounding Decision Authority in Autonomous Agents**  
  Jose Manuel de la Chica Rodriguez, Juan Manuel Vera Díaz  
  _2026-02-16_ · https://arxiv.org/abs/2602.14606v1 · `cs.MA`  
  <details><summary>Abstract</summary>

  Autonomous agentic systems are increasingly deployed in regulated, high-stakes domains where decisions may be irreversible and institutionally constrained. Existing safety approaches emphasize alignment, interpretability, or action-level filtering. We argue that these mechanisms are necessary but insufficient because they do not directly govern selection power: the authority to determine which options are generated, surfaced, and framed for decision. We propose a governance architecture that separates cognition, selection, and action into distinct domains and models autonomy as a vector of sovereignty. Cognitive autonomy remains unconstrained, while selection and action autonomy are bounded through mechanically enforced primitives operating outside the agent's optimization space. The architecture integrates external candidate generation (CEFL), a governed reducer, commit-reveal entropy isolation, rationale validation, and fail-loud circuit breakers. We evaluate the system across multiple regulated financial scenarios under adversarial stress targeting variance manipulation, threshold gaming, framing skew, ordering effects, and entropy probing. Metrics quantify selection concentration, narrative diversity, governance activation cost, and failure visibility. Results show that mechanical selection governance is implementable, auditable, and prevents deterministic outcome capture while preserving reasoning capacity. Although probabilistic concentration remains, the architecture measurably bounds selection authority relative to conventional scalar pipelines. This work reframes governance as bounded causal power rather than internal intent alignment, offering a foundation for deploying autonomous agents where silent failure is unacceptable.

  </details>



- **Kalman Filtering Based Flight Management System Modeling for AAM Aircraft**  
  Balram Kandoria, Aryaman Singh Samyal  
  _2026-02-16_ · https://arxiv.org/abs/2602.14948v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Advanced Aerial Mobility (AAM) operations require strategic flight planning services that predict both spatial and temporal uncertainties to safely validate flight plans against hazards such as weather cells, restricted airspaces, and CNS disruption areas. Current uncertainty estimation methods for AAM vehicles rely on conservative linear models due to limited real-world performance data. This paper presents a novel Kalman Filter-based uncertainty propagation method that models AAM Flight Management System (FMS) architectures through sigmoid-blended measurement noise covariance. Unlike existing approaches with fixed uncertainty thresholds, our method continuously adapts the filter's measurement trust based on progress toward waypoints, enabling FMS correction behavior to emerge naturally. The approach scales proportionally with control inputs and is tunable to match specific aircraft characteristics or route conditions. We validate the method using real ADS-B data from general aviation aircraft divided into training and verification sets. Uncertainty propagation parameters were tuned on the training set, achieving 76% accuracy in predicting arrival times when compared against the verification dataset, demonstrating the method's effectiveness for strategic flight plan validation in AAM operations.

  </details>



- **EmbeWebAgent: Embedding Web Agents into Any Customized UI**  
  Chenyang Ma, Clyde Fare, Matthew Wilson, Dave Braines  
  _2026-02-16_ · https://arxiv.org/abs/2602.14865v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Most web agents operate at the human interface level, observing screenshots or raw DOM trees without application-level access, which limits robustness and action expressiveness. In enterprise settings, however, explicit control of both the frontend and backend is available. We present EmbeWebAgent, a framework for embedding agents directly into existing UIs using lightweight frontend hooks (curated ARIA and URL-based observations, and a per-page function registry exposed via a WebSocket) and a reusable backend workflow that performs reasoning and takes actions. EmbeWebAgent is stack-agnostic (e.g., React or Angular), supports mixed-granularity actions ranging from GUI primitives to higher-level composites, and orchestrates navigation, manipulation, and domain-specific analytics via MCP tools. Our demo shows minimal retrofitting effort and robust multi-step behaviors grounded in a live UI setting. Live Demo: https://youtu.be/Cy06Ljee1JQ

  </details>



- **Simulation-based Learning of Electrical Cabinet Assembly Using Robot Skills**  
  Arik Laemmle, Balázs András Bálint, Philipp Tenbrock, Frank Naegele, David Traunecker, József Váncza, Marco F. Huber  
  _2026-02-16_ · https://arxiv.org/abs/2602.14561v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a simulation-driven approach for automating the force-controlled assembly of electrical terminals on DIN-rails, a task traditionally hindered by high programming effort and product variability. The proposed method integrates deep reinforcement learning (DRL) with parameterizable robot skills in a physics-based simulation environment. To realistically model the snap-fit assembly process, we develop and evaluate two types of joining models: analytical models based on beam theory and rigid-body models implemented in the MuJoCo physics engine. These models enable accurate simulation of interaction forces, essential for training DRL agents. The robot skills are structured using the pitasc framework, allowing modular, reusable control strategies. Training is conducted in simulation using Soft Actor-Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithms. Domain randomization is applied to improve robustness. The trained policies are transferred to a physical UR10e robot system without additional tuning. Experimental results demonstrate high success rates (up to 100%) in both simulation and real-world settings, even under significant positional and rotational deviations. The system generalizes well to new terminal types and positions, significantly reducing manual programming effort. This work highlights the potential of combining simulation-based learning with modular robot skills for flexible, scalable automation in small-batch manufacturing. Future work will explore hybrid learning methods, automated environment parameterization, and further refinement of joining models for design integration.

  </details>



- **BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames**  
  Max Sobol Mark, Jacky Liang, Maria Attarian, Chuyuan Fu, Debidatta Dwibedi, Dhruv Shah, Aviral Kumar  
  _2026-02-16_ · https://arxiv.org/abs/2602.15010v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many robot tasks require attending to the history of past observations. For example, finding an item in a room requires remembering which places have already been searched. However, the best-performing robot policies typically condition only on the current observation, limiting their applicability to such tasks. Naively conditioning on past observations often fails due to spurious correlations: policies latch onto incidental features of training histories that do not generalize to out-of-distribution trajectories upon deployment. We analyze why policies latch onto these spurious correlations and find that this problem stems from limited coverage over the space of possible histories during training, which grows exponentially with horizon. Existing regularization techniques provide inconsistent benefits across tasks, as they do not fundamentally address this coverage problem. Motivated by these findings, we propose Big Picture Policies (BPP), an approach that conditions on a minimal set of meaningful keyframes detected by a vision-language model. By projecting diverse rollouts onto a compact set of task-relevant events, BPP substantially reduces distribution shift between training and deployment, without sacrificing expressivity. We evaluate BPP on four challenging real-world manipulation tasks and three simulation tasks, all requiring history conditioning. BPP achieves 70% higher success rates than the best comparison on real-world evaluations.

  </details>



- **Fault Detection in Electrical Distribution System using Autoencoders**  
  Sidharthenee Nayak, Victor Sam Moses Babu, Chandrashekhar Narayan Bhende, Pratyush Chakraborty, Mayukha Pal  
  _2026-02-16_ · https://arxiv.org/abs/2602.14939v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  In recent times, there has been considerable interest in fault detection within electrical power systems, garnering attention from both academic researchers and industry professionals. Despite the development of numerous fault detection methods and their adaptations over the past decade, their practical application remains highly challenging. Given the probabilistic nature of fault occurrences and parameters, certain decision-making tasks could be approached from a probabilistic standpoint. Protective systems are tasked with the detection, classification, and localization of faulty voltage and current line magnitudes, culminating in the activation of circuit breakers to isolate the faulty line. An essential aspect of designing effective fault detection systems lies in obtaining reliable data for training and testing, which is often scarce. Leveraging deep learning techniques, particularly the powerful capabilities of pattern classifiers in learning, generalizing, and parallel processing, offers promising avenues for intelligent fault detection. To address this, our paper proposes an anomaly-based approach for fault detection in electrical power systems, employing deep autoencoders. Additionally, we utilize Convolutional Autoencoders (CAE) for dimensionality reduction, which, due to its fewer parameters, requires less training time compared to conventional autoencoders. The proposed method demonstrates superior performance and accuracy compared to alternative detection approaches by achieving an accuracy of 97.62% and 99.92% on simulated and publicly available datasets.

  </details>



- **GOT-JEPA: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture**  
  Shih-Fang Chen, Jun-Cheng Chen, I-Hong Jhuo, Yen-Yu Lin  
  _2026-02-16_ · https://arxiv.org/abs/2602.14771v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The human visual system tracks objects by integrating current observations with previously observed information, adapting to target and scene changes, and reasoning about occlusion at fine granularity. In contrast, recent generic object trackers are often optimized for training targets, which limits robustness and generalization in unseen scenarios, and their occlusion reasoning remains coarse, lacking detailed modeling of occlusion patterns. To address these limitations in generalization and occlusion perception, we propose GOT-JEPA, a model-predictive pretraining framework that extends JEPA from predicting image features to predicting tracking models. Given identical historical information, a teacher predictor generates pseudo-tracking models from a clean current frame, and a student predictor learns to predict the same pseudo-tracking models from a corrupted version of the current frame. This design provides stable pseudo supervision and explicitly trains the predictor to produce reliable tracking models under occlusions, distractors, and other adverse observations, improving generalization to dynamic environments. Building on GOT-JEPA, we further propose OccuSolver to enhance occlusion perception for object tracking. OccuSolver adapts a point-centric point tracker for object-aware visibility estimation and detailed occlusion-pattern capture. Conditioned on object priors iteratively generated by the tracker, OccuSolver incrementally refines visibility states, strengthens occlusion handling, and produces higher-quality reference labels that progressively improve subsequent model predictions. Extensive evaluations on seven benchmarks show that our method effectively enhances tracker generalization and robustness.

  </details>



- **Solving Inverse Parametrized Problems via Finite Elements and Extreme Learning Networks**  
  Erik Burman, Mats G. Larson, Karl Larsson, Jonatan Vallin  
  _2026-02-16_ · https://arxiv.org/abs/2602.14757v1 · `math.NA`  
  <details><summary>Abstract</summary>

  We develop an interpolation-based reduced-order modeling framework for parameter-dependent partial differential equations arising in control, inverse problems, and uncertainty quantification. The solution is discretized in the physical domain using finite element methods, while the dependence on a finite-dimensional parameter is approximated separately. We establish existence, uniqueness, and regularity of the parametric solution and derive rigorous error estimates that explicitly quantify the interplay between spatial discretization and parameter approximation. In low-dimensional parameter spaces, classical interpolation schemes yield algebraic convergence rates based on Sobolev regularity in the parameter variable. In higher-dimensional parameter spaces, we replace classical interpolation by extreme learning machine (ELM) surrogates and obtain error bounds under explicit approximation and stability assumptions. The proposed framework is applied to inverse problems in quantitative photoacoustic tomography, where we derive potential and parameter reconstruction error estimates and demonstrate substantial computational savings compared to standard approaches, without sacrificing accuracy.

  </details>



- **DriveFine: Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving**  
  Chenxu Dang, Sining Ang, Yongkang Li, Haochen Tian, Jie Wang, Guang Li, Hangjun Ye, Jie Ma, Long Chen, Yan Wang  
  _2026-02-16_ · https://arxiv.org/abs/2602.14577v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models for autonomous driving increasingly adopt generative planners trained with imitation learning followed by reinforcement learning. Diffusion-based planners suffer from modality alignment difficulties, low training efficiency, and limited generalization. Token-based planners are plagued by cumulative causal errors and irreversible decoding. In summary, the two dominant paradigms exhibit complementary strengths and weaknesses. In this paper, we propose DriveFine, a masked diffusion VLA model that combines flexible decoding with self-correction capabilities. In particular, we design a novel plug-and-play block-MoE, which seamlessly injects a refinement expert on top of the generation expert. By enabling explicit expert selection during inference and gradient blocking during training, the two experts are fully decoupled, preserving the foundational capabilities and generic patterns of the pretrained weights, which highlights the flexibility and extensibility of the block-MoE design. Furthermore, we design a hybrid reinforcement learning strategy that encourages effective exploration of refinement expert while maintaining training stability. Extensive experiments on NAVSIM v1, v2, and Navhard benchmarks demonstrate that DriveFine exhibits strong efficacy and robustness. The code will be released at https://github.com/MSunDYY/DriveFine.

  </details>


