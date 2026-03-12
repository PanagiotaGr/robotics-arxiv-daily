# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-03-12 07:09 UTC_

Total papers shown: **19**


---

- **Safe RLHF Beyond Expectation: Stochastic Dominance for Universal Spectral Risk Control**  
  Yaswanth Chittepu, Ativ Joshi, Rajarshi Bhattacharjee, Scott Niekum  
  _2026-03-11_ · https://arxiv.org/abs/2603.10938v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Safe Reinforcement Learning from Human Feedback (RLHF) typically enforces safety through expected cost constraints, but the expectation captures only a single statistic of the cost distribution and fails to account for distributional uncertainty, particularly under heavy tails or rare catastrophic events. This limitation is problematic when robustness and risk sensitivity are critical. Stochastic dominance offers a principled alternative by comparing entire cost distributions rather than just their averages, enabling direct control over tail risks and potential out-of-distribution failures that expectation-based constraints may overlook. In this work, we propose Risk-sensitive Alignment via Dominance (RAD), a novel alignment framework that replaces scalar expected cost constraints with First-Order Stochastic Dominance (FSD) constraints. We operationalize this constraint by comparing the target policy's cost distribution to that of a reference policy within an Optimal Transport (OT) framework, using entropic regularization and Sinkhorn iterations to obtain a differentiable and computationally efficient objective for stable end-to-end optimization. Furthermore, we introduce quantile-weighted FSD constraints and show that weighted FSD universally controls a broad class of Spectral Risk Measures (SRMs), so that improvements under weighted dominance imply guaranteed improvements in the corresponding spectral risk. This provides a principled mechanism for tuning a model's risk profile via the quantile weighting function. Empirical results demonstrate that RAD improves harmlessness over baselines while remaining competitive in helpfulness, and exhibits greater robustness on out-of-distribution harmlessness evaluations.

  </details>



- **Splat2Real: Novel-view Scaling for Physical AI with 3D Gaussian Splatting**  
  Hansol Lim, Jongseong Brad Choi  
  _2026-03-11_ · https://arxiv.org/abs/2603.10638v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Physical AI faces viewpoint shift between training and deployment, and novel-view robustness is essential for monocular RGB-to-3D perception. We cast Real2Render2Real monocular depth pretraining as imitation-learning-style supervision from a digital twin oracle: a student depth network imitates expert metric depth/visibility rendered from a scene mesh, while 3DGS supplies scalable novel-view observations. We present Splat2Real, centered on novel-view scaling: performance depends more on which views are added than on raw view count. We introduce CN-Coverage, a coverage+novelty curriculum that greedily selects views by geometry gain and an extrapolation penalty, plus a quality-aware guardrail fallback for low-reliability teachers. Across 20 TUM RGB-D sequences with step-matched budgets (N=0 to 2000 additional rendered views, with N unique <= 500 and resampling for larger budgets), naive scaling is unstable; CN-Coverage mitigates worst-case regressions relative to Robot/Coverage policies, and GOL-Gated CN-Coverage provides the strongest medium-high-budget stability with the lowest high-novelty tail error. Downstream control-proxy results versus N provides embodied-relevance evidence by shifting safety/progress trade-offs under viewpoint shift.

  </details>



- **Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control**  
  Matti Vahs, Joris Verhagen, Jana Tumova  
  _2026-03-11_ · https://arxiv.org/abs/2603.10572v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Partially Observable Markov Decision Processes (POMDPs) provide a principled framework for robot decision-making under uncertainty. Solving reach-avoid POMDPs, however, requires coordinating three distinct behaviors: goal reaching, safety, and active information gathering to reduce uncertainty. Existing online POMDP solvers attempt to address all three within a single belief tree search, but this unified approach struggles with the conflicting time scales inherent to these objectives. We propose a layered, certificate-based control architecture that operates directly in belief space, decoupling goal reaching, information gathering, and safety into modular components. We introduce Belief Control Lyapunov Functions (BCLFs) that formalize information gathering as a Lyapunov convergence problem in belief space, and show how they can be learned via reinforcement learning. For safety, we develop Belief Control Barrier Functions (BCBFs) that leverage conformal prediction to provide probabilistic safety guarantees over finite horizons. The resulting control synthesis reduces to lightweight quadratic programs solvable in real time, even for non-Gaussian belief representations with dimension $>10^4$. Experiments in simulation and on a space-robotics platform demonstrate real-time performance and improved safety and task success compared to state-of-the-art constrained POMDP solvers.

  </details>



- **Evaluating randomized smoothing as a defense against adversarial attacks in trajectory prediction**  
  Julian F. Schumann, Eduardo Figueiredo, Frederik Baymler Mathiesen, Luca Laurenti, Jens Kober, Arkady Zgonnikov  
  _2026-03-11_ · https://arxiv.org/abs/2603.10821v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Accurate and robust trajectory prediction is essential for safe and efficient autonomous driving, yet recent work has shown that even state-of-the-art prediction models are highly vulnerable to inputs being mildly perturbed by adversarial attacks. Although model vulnerabilities to such attacks have been studied, work on effective countermeasures remains limited. In this work, we develop and evaluate a new defense mechanism for trajectory prediction models based on randomized smoothing -- an approach previously applied successfully in other domains. We evaluate its ability to improve model robustness through a series of experiments that test different strategies of randomized smoothing. We show that our approach can consistently improve prediction robustness of multiple base trajectory prediction models in various datasets without compromising accuracy in non-adversarial settings. Our results demonstrate that randomized smoothing offers a simple and computationally inexpensive technique for mitigating adversarial attacks in trajectory prediction.

  </details>



- **CUPID: A Plug-in Framework for Joint Aleatoric and Epistemic Uncertainty Estimation with a Single Model**  
  Xinran Xu, Xiuyi Fan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10745v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Accurate estimation of uncertainty in deep learning is critical for deploying models in high-stakes domains such as medical diagnosis and autonomous decision-making, where overconfident predictions can lead to harmful outcomes. In practice, understanding the reason behind a model's uncertainty and the type of uncertainty it represents can support risk-aware decisions, enhance user trust, and guide additional data collection. However, many existing methods only address a single type of uncertainty or require modifications and retraining of the base model, making them difficult to adopt in real-world systems. We introduce CUPID (Comprehensive Uncertainty Plug-in estImation moDel), a general-purpose module that jointly estimates aleatoric and epistemic uncertainty without modifying or retraining the base model. CUPID can be flexibly inserted into any layer of a pretrained network. It models aleatoric uncertainty through a learned Bayesian identity mapping and captures epistemic uncertainty by analyzing the model's internal responses to structured perturbations. We evaluate CUPID across a range of tasks, including classification, regression, and out-of-distribution detection. The results show that it consistently delivers competitive performance while offering layer-wise insights into the origins of uncertainty. By making uncertainty estimation modular, interpretable, and model-agnostic, CUPID supports more transparent and trustworthy AI. Related code and data are available at https://github.com/a-Fomalhaut-a/CUPID.

  </details>



- **Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming**  
  Yilin Zou, Zhong Zhang, Fanghua Jiang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10711v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Real-time trajectory optimization for nonlinear constrained autonomous systems is critical and typically performed by CPU-based sequential solvers. Specifically, reliance on global sparse linear algebra or the serial nature of dynamic programming algorithms restricts the utilization of massively parallel computing architectures like GPUs. To bridge this gap, we introduce a fully GPU-native trajectory optimization framework that combines sequential convex programming with a consensus-based alternating direction method of multipliers. By applying a temporal splitting strategy, our algorithm decouples the optimization horizon into independent, per-node subproblems that execute massively in parallel. The entire process runs fully on the GPU, eliminating costly memory transfers and large-scale sparse factorizations. This architecture naturally scales to multi-trajectory optimization. We validate the solver on a quadrotor agile flight task and a Mars powered descent problem using an on-board edge computing platform. Benchmarks reveal a sustained 4x throughput speedup and a 51% reduction in energy consumption over a heavily optimized 12-core CPU baseline. Crucially, the framework saturates the hardware, maintaining over 96% active GPU utilization to achieve planning rates exceeding 100 Hz. Furthermore, we demonstrate the solver's extensibility to robust Model Predictive Control by jointly optimizing dynamically coupled scenarios under stochastic disturbances, enabling scalable and safe autonomy.

  </details>



- **OnFly: Onboard Zero-Shot Aerial Vision-Language Navigation toward Safety and Efficiency**  
  Guiyong Zheng, Yueting Ban, Mingjie Zhang, Juepeng Zheng, Boyu Zhou  
  _2026-03-11_ · https://arxiv.org/abs/2603.10682v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Aerial vision-language navigation (AVLN) enables UAVs to follow natural-language instructions in complex 3D environments. However, existing zero-shot AVLN methods often suffer from unstable single-stream Vision-Language Model decision-making, unreliable long-horizon progress monitoring, and a trade-off between safety and efficiency. We propose OnFly, a fully onboard, real-time framework for zero-shot AVLN. OnFly adopts a shared-perception dual-agent architecture that decouples high-frequency target generation from low-frequency progress monitoring, thereby stabilizing decision-making. It further employs a hybrid keyframe-recent-frame memory to preserve global trajectory context while maintaining KV-cache prefix stability, enabling reliable long-horizon monitoring with termination and recovery signals. In addition, a semantic-geometric verifier refines VLM-predicted targets for instruction consistency and geometric safety using VLM features and depth cues, while a receding-horizon planner generates optimized collision-free trajectories under geometric safety constraints, improving both safety and efficiency. In simulation, OnFly improves task success from 26.4% to 67.8%, compared with the strongest state-of-the-art baseline, while fully onboard real-world flights validate its feasibility for real-time deployment. The code will be released at https://github.com/Robotics-STAR-Lab/OnFly

  </details>



- **Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction**  
  Hao Zhou, Lu Qi, Jason Li, Jie Zhang, Yi Liu, Xu Yang, Mingyu Fan, Fei Luo  
  _2026-03-11_ · https://arxiv.org/abs/2603.10597v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Trajectory prediction is critical for autonomous driving, enabling safe and efficient planning in dense, dynamic traffic. Most existing methods optimize prediction accuracy under fixed-length observations. However, real-world driving often yields variable-length, incomplete observations, posing a challenge to these methods. A common strategy is to directly map features from incomplete observations to those from complete ones. This one-shot mapping, however, struggles to learn accurate representations for short trajectories due to significant information gaps. To address this issue, we propose a Progressive Retrospective Framework (PRF), which gradually aligns features from incomplete observations with those from complete ones via a cascade of retrospective units. Each unit consists of a Retrospective Distillation Module (RDM) and a Retrospective Prediction Module (RPM), where RDM distills features and RPM recovers previous timesteps using the distilled features. Moreover, we propose a Rolling-Start Training Strategy (RSTS) that enhances data efficiency during PRF training. PRF is plug-and-play with existing methods. Extensive experiments on datasets Argoverse 2 and Argoverse 1 demonstrate the effectiveness of PRF. Code is available at https://github.com/zhouhao94/PRF.

  </details>



- **Path Planning for Sound Speed Profile Estimation**  
  Ludvig Lindström, Tadas Paskevicius, Andreas Jakobsson, Isaac Skog  
  _2026-03-11_ · https://arxiv.org/abs/2603.10585v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Accurate estimation of the sound speed profile (SSP) is essential for underwater acoustic communication, sonar performance, and navigation, as the acoustic wave propagation depends strongly on the SSP. This work considers SSP estimation in a region of interest using an autonomous underwater vehicle (AUV) equipped with a conductivity-temperature-depth (CTD) sensor and an acoustic receiver measuring transmission loss (TL) from a sonar transmitter. The SSP is modeled using a linear basis-function expansion and is sequentially estimated with an unscented Kalman filter that fuses local CTD measurements with TL measurements. A receding-horizon path planning scheme is also employed to select future AUV positions by minimizing the predicted total sound speed variance. Simulations using the Bellhop acoustic wave propagation solver show that CTD measurements provide accurate local SSP estimates, whereas TL measurements are seen to capture the global characteristics of the SSP, with their joint use improving the reconstruction of both local variations and large-scale SSP behavior. The results also indicate that the proposed path planning strategy reduces the estimation uncertainty compared to constant-velocity motion, thereby enabling improved environmental characterization for underwater acoustic systems.

  </details>



- **A Hybrid Knowledge-Grounded Framework for Safety and Traceability in Prescription Verification**  
  Yichi Zhu, Kan Ling, Xu Liu, Hengrun Zhang, Huiqun Yu, Guisheng Fan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10891v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Medication errors pose a significant threat to patient safety, making pharmacist verification (PV) a critical, yet heavily burdened, final safeguard. The direct application of Large Language Models (LLMs) to this zero-tolerance domain is untenable due to their inherent factual unreliability, lack of traceability, and weakness in complex reasoning. To address these challenges, we introduce PharmGraph-Auditor, a novel system designed for safe and evidence-grounded prescription auditing. The core of our system is a trustworthy Hybrid Pharmaceutical Knowledge Base (HPKB), implemented under the Virtual Knowledge Graph (VKG) paradigm. This architecture strategically unifies a relational component for set constraint satisfaction and a graph component for topological reasoning via a rigorous mapping layer. To construct this HPKB, we propose the Iterative Schema Refinement (ISR) algorithm, a framework that enables the co-evolution of both graph and relational schemas from medical texts. For auditing, we introduce the KB-grounded Chain of Verification (CoV), a new reasoning paradigm that transforms the LLM from an unreliable generator into a transparent reasoning engine. CoV decomposes the audit task into a sequence of verifiable queries against the HPKB, generating hybrid query plans to retrieve evidence from the most appropriate data store. Experimental results demonstrate robust knowledge extraction capabilities and show promises of using PharmGraph-Auditor to enable pharmacists to achieve safer and faster prescription verification.

  </details>



- **Evaluating Few-Shot Pill Recognition Under Visual Domain Shift**  
  W. I. Chu, G. Tarroni, L. Li  
  _2026-03-11_ · https://arxiv.org/abs/2603.10833v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Adverse drug events are a significant source of preventable harm, which has led to the development of automated pill recognition systems to enhance medication safety. Real-world deployment of these systems is hindered by visually complex conditions, including cluttered scenes, overlapping pills, reflections, and diverse acquisition environments. This study investigates few-shot pill recognition from a deployment-oriented perspective, prioritizing generalization under realistic cross-dataset domain shifts over architectural innovation. A two-stage object detection framework is employed, involving base training followed by few-shot fine-tuning. Models are adapted to novel pill classes using one, five, or ten labeled examples per class and are evaluated on a separate deployment dataset featuring multi-object, cluttered scenes. The evaluation focuses on classification-centric and error-based metrics to address heterogeneous annotation strategies. Findings indicate that semantic pill recognition adapts rapidly with few-shot supervision, with classification performance reaching saturation even with a single labeled example. However, stress testing under overlapping and occluded conditions demonstrates a marked decline in localization and recall, despite robust semantic classification. Models trained on visually realistic, multi-pill data consistently exhibit greater robustness in low-shot scenarios, underscoring the importance of training data realism and the diagnostic utility of few-shot fine-tuning for deployment readiness.

  </details>



- **CUAAudit: Meta-Evaluation of Vision-Language Models as Auditors of Autonomous Computer-Use Agents**  
  Marta Sumyk, Oleksandr Kosovan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10577v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Computer-Use Agents (CUAs) are emerging as a new paradigm in human-computer interaction, enabling autonomous execution of tasks in desktop environment by perceiving high-level natural-language instructions. As such agents become increasingly capable and are deployed across diverse desktop environments, evaluating their behavior in a scalable and reliable manner becomes a critical challenge. Existing evaluation pipelines rely on static benchmarks, rule-based success checks, or manual inspection, which are brittle, costly, and poorly aligned with real-world usage. In this work, we study Vision-Language Models (VLMs) as autonomous auditors for assessing CUA task completion directly from observable interactions and conduct a large-scale meta-evaluation of five VLMs that judge task success given a natural-language instruction and the final environment state. Our evaluation spans three widely used CUA benchmarks across macOS, Windows, and Linux environments and analyzes auditor behavior along three complementary dimensions: accuracy, calibration of confidence estimates, and inter-model agreement. We find that while state-of-the-art VLMs achieve strong accuracy and calibration, all auditors exhibit notable performance degradation in more complex or heterogeneous environments, and even high-performing models show significant disagreement in their judgments. These results expose fundamental limitations of current model-based auditing approaches and highlight the need to explicitly account for evaluator reliability, uncertainty, and variance when deploying autonomous CUAs in real-world settings.

  </details>



- **When should we trust the annotation? Selective prediction for molecular structure retrieval from mass spectra**  
  Mira Jürgens, Gaetan De Waele, Morteza Rakhshaninejad, Willem Waegeman  
  _2026-03-11_ · https://arxiv.org/abs/2603.10950v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Machine learning methods for identifying molecular structures from tandem mass spectra (MS/MS) have advanced rapidly, yet current approaches still exhibit significant error rates. In high-stakes applications such as clinical metabolomics and environmental screening, incorrect annotations can have serious consequences, making it essential to determine when a prediction can be trusted. We introduce a selective prediction framework for molecular structure retrieval from MS/MS spectra, enabling models to abstain from predictions when uncertainty is too high. We formulate the problem within the risk-coverage tradeoff framework and comprehensively evaluate uncertainty quantification strategies at two levels of granularity: fingerprint-level uncertainty over predicted molecular fingerprint bits, and retrieval-level uncertainty over candidate rankings. We compare scoring functions including first-order confidence measures, aleatoric and epistemic uncertainty estimates from second-order distributions, as well as distance-based measures in the latent space. All experiments are conducted on the MassSpecGym benchmark. Our analysis reveals that while fingerprint-level uncertainty scores are poor proxies for retrieval success, computationally inexpensive first-order confidence measures and retrieval-level aleatoric uncertainty achieve strong risk-coverage tradeoffs across evaluation settings. We demonstrate that by applying distribution-free risk control via generalization bounds, practitioners can specify a tolerable error rate and obtain a subset of annotations satisfying that constraint with high probability.

  </details>



- **STADA: Specification-based Testing for Autonomous Driving Agents**  
  Joy Saha, Trey Woodlief, Sebastian Elbaum, Matthew B. Dwyer  
  _2026-03-11_ · https://arxiv.org/abs/2603.10940v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Simulation-based testing has become a standard approach to validating autonomous driving agents prior to real-world deployment. A high-quality validation campaign will exercise an agent in diverse contexts comprised of varying static environments, e.g., lanes, intersections, signage, and dynamic elements, e.g., vehicles and pedestrians. To achieve this, existing test generation techniques rely on template-based, manually constructed, or random scenario generation. When applied to validate formally specified safety requirements, such methods either require significant human effort or run the risk of missing important behavior related to the requirement. To address this gap, we present STADA, a Specification-based Test generation framework for Autonomous Driving Agents that systematically generates the space of scenarios defined by a formal specification expressed in temporal logic (LTLf). Given a specification, STADA constructs all distinct initial scenes, a diverse space of continuations of those scenes, and simulations that reflect the behaviors of the specification. Evaluation of STADA on a variety of LTLf specifications formalized in SCENEFLOW using three complementary coverage criteria demonstrates that STADA yields more than 2x higher coverage than the best baseline on the finest criteria and a 75% increase for the coarsest criteria. Moreover, it matches the coverage of the best baseline with 6 times fewer simulations. While set in the context of autonomous driving, the approach is applicable to other domains with rich simulation environments.

  </details>



- **Surrogate models for nuclear fusion with parametric Shallow Recurrent Decoder Networks: applications to magnetohydrodynamics**  
  M. Lo Verso, C. Introini, E. Cervi, L. Savoldi, J. N. Kutz, A. Cammi  
  _2026-03-11_ · https://arxiv.org/abs/2603.10678v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Magnetohydrodynamic (MHD) effects play a key role in the design and operation of nuclear fusion systems, where electrically conducting fluids (such as liquid metals or molten salts in reactor blankets) interact with magnetic fields of varying intensity and orientation, which affect the resulting flow. The numerical resolution of MHD models involves highly nonlinear multiphysics systems of equations and can become computationally expensive, particularly in multi-query, parametric, or real-time contexts. This work investigates a fully data-driven framework for MHD state reconstruction that combines dimensionality reduction via Singular Value Decomposition (SVD) with the SHallow REcurrent Decoder (SHRED), a neural network architecture designed to recover the full spatio-temporal state from sparse time-series measurements of a limited number of observables. The methodology is applied to a parametric MHD test case involving compressible lead-lithium flow in a stepped channel subjected to thermal gradients and magnetic fields spanning a broad range of intensities. To improve efficiency, the full-order dataset is first compressed using SVD, yielding a reduced representation used as reference truth for training. Only temperature measurements from three sensors are provided as input, while the network reconstructs the full fields of velocity, pressure, and temperature. To assess robustness with respect to sensor placement, thirty randomly generated sensor configurations are tested in ensemble mode. Results show that SHRED accurately reconstructs the full MHD state even for magnetic field intensities not included in the training set. These findings demonstrate the potential of SHRED as a computationally efficient surrogate modeling strategy for fusion-relevant multiphysics problems, enabling low-cost state estimation with possible applications in real-time monitoring and control.

  </details>



- **Spatio-Temporal Attention Graph Neural Network: Explaining Causalities With Attention**  
  Kosti Koistinen, Kirsi Hellsten, Joni Herttuainen, Kimmo K. Kaski  
  _2026-03-11_ · https://arxiv.org/abs/2603.10676v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Industrial Control Systems (ICS) underpin critical infrastructure and face growing cyber-physical threats due to the convergence of operational technology and networked environments. While machine learning-based anomaly detection approaches in ICS shows strong theoretical performance, deployment is often limited by poor explainability, high false-positive rates, and sensitivity to evolving system behavior, i.e., baseline drifting. We propose a Spatio-Temporal Attention Graph Neural Network (STA-GNN) for unsupervised and explainable anomaly detection in ICS that models both temporal dynamics and relational structure of the system. Sensors, controllers, and network entities are represented as nodes in a dynamically learned graph, enabling the model to capture inter-dependencies across physical processes and communication patterns. Attention mechanisms provide influential relationships, supporting inspection of correlations and potential causal pathways behind detected events. The approach supports multiple data modalities, including SCADA point measurements, network flow features, and payload features, and thus enables unified cyber-physical analysis. To address operational requirements, we incorporate a conformal prediction strategy to control false alarm rates and monitor performance degradation under drifting of the environment. Our findings highlight the possibilities and limitations of model evaluation and common pitfalls in anomaly detection in ICS. Our findings emphasise the importance of explainable, drift-aware evaluation for reliable deployment of learning-based security monitoring systems.

  </details>



- **AdaClearGrasp: Learning Adaptive Clearing for Zero-Shot Robust Dexterous Grasping in Densely Cluttered Environments**  
  Zixuan Chen, Wenquan Zhang, Jing Fang, Ruiming Zeng, Zhixuan Xu, Yiwen Hou, Xinke Wang, Jieqi Shi, Jing Huo, Yang Gao  
  _2026-03-11_ · https://arxiv.org/abs/2603.10616v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In densely cluttered environments, physical interference, visual occlusions, and unstable contacts often cause direct dexterous grasping to fail, while aggressive singulation strategies may compromise safety. Enabling robots to adaptively decide whether to clear surrounding objects or directly grasp the target is therefore crucial for robust manipulation. We propose AdaClearGrasp, a closed-loop decision-execution framework for adaptive clearing and zero-shot dexterous grasping in densely cluttered environments. The framework formulates manipulation as a controllable high-level decision process that determines whether to directly grasp the target or first clear surrounding objects. A pretrained vision-language model (VLM) interprets visual observations and language task descriptions to reason about grasp interference and generate a high-level planning skeleton, which invokes structured atomic skills through a unified action interface. For dexterous grasping, we train a reinforcement learning policy with a relative hand-object distance representation, enabling zero-shot generalization across diverse object geometries and physical properties. During execution, visual feedback monitors outcomes and triggers replanning upon failures, forming a closed-loop correction mechanism. To evaluate language-conditioned dexterous grasping in clutter, we introduce Clutter-Bench, the first simulation benchmark with graded clutter complexity. It includes seven target objects across three clutter levels, yielding 210 task scenarios. We further perform sim-to-real experiments on three objects under three clutter levels (18 scenarios). Results demonstrate that AdaClearGrasp significantly improves grasp success rates in densely cluttered environments. For more videos and code, please visit our project website: https://chenzixuan99.github.io/adaclear-grasp.github.io/.

  </details>



- **Prompting with the human-touch: evaluating model-sensitivity of foundation models for musculoskeletal CT segmentation**  
  Caroline Magg, Maaike A. ter Wee, Johannes G. G. Dobbe, Geert J. Streekstra, Leendert Blankevoort, Clara I. Sánchez, Hoel Kervadec  
  _2026-03-11_ · https://arxiv.org/abs/2603.10541v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Promptable Foundation Models (FMs), initially introduced for natural image segmentation, have also revolutionized medical image segmentation. The increasing number of models, along with evaluations varying in datasets, metrics, and compared models, makes direct performance comparison between models difficult and complicates the selection of the most suitable model for specific clinical tasks. In our study, 11 promptable FMs are tested using non-iterative 2D and 3D prompting strategies on a private and public dataset focusing on bone and implant segmentation in four anatomical regions (wrist, shoulder, hip and lower leg). The Pareto-optimal models are identified and further analyzed using human prompts collected through a dedicated observer study. Our findings are: 1) The segmentation performance varies a lot between FMs and prompting strategies; 2) The Pareto-optimal models in 2D are SAM and SAM2.1, in 3D nnInteractive and Med-SAM2; 3) Localization accuracy and rater consistency vary with anatomical structures, with higher consistency for simple structures (wrist bones) and lower consistency for complex structures (pelvis, tibia, implants); 4) The segmentation performance drops using human prompts, suggesting that performance reported on "ideal" prompts extracted from reference labels might overestimate the performance in a human-driven setting; 5) All models were sensitive to prompt variations. While two models demonstrated intra-rater robustness, it did not scale to inter-rater settings. We conclude that the selection of the most optimal FM for a human-driven setting remains challenging, with even high-performing FMs being sensitive to variations in human input prompts. Our code base for prompt extraction and model inference is available: https://github.com/CarolineMagg/segmentation-FM-benchmark/

  </details>



- **Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning**  
  Zichao Li, Jie Lou, Fangchen Dong, Zhiyuan Fan, Mengjie Ren, Hongyu Lin, Xianpei Han, Debing Zhang, Le Sun, Yaojie Lu, et al.  
  _2026-03-11_ · https://arxiv.org/abs/2603.10535v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning significantly enhances LLM capabilities but suffers from a critical issue: length inflation, where models adopt verbosity or inefficient reasoning to maximize rewards. Prior approaches struggle to address this challenge in a general and lossless manner, primarily because additive penalties introduce a compensatory effect that creates optimization shortcuts, while heuristic gating strategies lack generality beyond binary feedback. To bridge this gap, we present Group Relative Reward Rescaling (GR$^3$), which reframes length control as a multiplicative rescaling paradigm, effectively establishing a generalized, continuous, and reward-dependent gating mechanism. To further ensure lossless optimization, we incorporate group-relative regularization and advantage-aware calibration, which dynamically adapt length budgets to instance difficulty and preserve the advantage signal of high-quality trajectories. Empirically, across both RLHF and RLVR settings, GR$^3$~maintains training dynamics and downstream performance comparable to standard GRPO while significantly mitigating length inflation, outperforming state-of-the-art length-regularized baselines.

  </details>


