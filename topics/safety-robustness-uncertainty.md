# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-01-28 06:52 UTC_

Total papers shown: **20**


---

- **Tactile Memory with Soft Robot: Robust Object Insertion via Masked Encoding and Soft Wrist**  
  Tatsuya Kamijo, Mai Nishimura, Cristian C. Beltran-Hernandez, Nodoka Shibasaki, Masashi Hamaya  
  _2026-01-27_ · https://arxiv.org/abs/2601.19275v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Tactile memory, the ability to store and retrieve touch-based experience, is critical for contact-rich tasks such as key insertion under uncertainty. To replicate this capability, we introduce Tactile Memory with Soft Robot (TaMeSo-bot), a system that integrates a soft wrist with tactile retrieval-based control to enable safe and robust manipulation. The soft wrist allows safe contact exploration during data collection, while tactile memory reuses past demonstrations via retrieval for flexible adaptation to unseen scenarios. The core of this system is the Masked Tactile Trajectory Transformer (MAT$^\text{3}$), which jointly models spatiotemporal interactions between robot actions, distributed tactile feedback, force-torque measurements, and proprioceptive signals. Through masked-token prediction, MAT$^\text{3}$ learns rich spatiotemporal representations by inferring missing sensory information from context, autonomously extracting task-relevant features without explicit subtask segmentation. We validate our approach on peg-in-hole tasks with diverse pegs and conditions in real-robot experiments. Our extensive evaluation demonstrates that MAT$^\text{3}$ achieves higher success rates than the baselines over all conditions and shows remarkable capability to adapt to unseen pegs and conditions.

  </details>



- **ScenePilot-Bench: A Large-Scale Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving**  
  Yujin Wang, Yutong Zheng, Wenxian Fan, Tianyi Wang, Hongqing Chu, Daxin Tian, Bingzhao Gao, Jianqiang Wang, Hong Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19582v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In this paper, we introduce ScenePilot-Bench, a large-scale first-person driving benchmark designed to evaluate vision-language models (VLMs) in autonomous driving scenarios. ScenePilot-Bench is built upon ScenePilot-4K, a diverse dataset comprising 3,847 hours of driving videos, annotated with multi-granularity information including scene descriptions, risk assessments, key participant identification, ego trajectories, and camera parameters. The benchmark features a four-axis evaluation suite that assesses VLM capabilities in scene understanding, spatial perception, motion planning, and GPT-Score, with safety-aware metrics and cross-region generalization settings. We benchmark representative VLMs on ScenePilot-Bench, providing empirical analyses that clarify current performance boundaries and identify gaps for driving-oriented reasoning. ScenePilot-Bench offers a comprehensive framework for evaluating and advancing VLMs in safety-critical autonomous driving contexts.

  </details>



- **Self-Supervised Path Planning in Unstructured Environments via Global-Guided Differentiable Hard Constraint Projection**  
  Ziqian Wang, Chenxi Fang, Zhen Zhang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19354v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deploying deep learning agents for autonomous navigation in unstructured environments faces critical challenges regarding safety, data scarcity, and limited computational resources. Traditional solvers often suffer from high latency, while emerging learning-based approaches struggle to ensure deterministic feasibility. To bridge the gap from embodied to embedded intelligence, we propose a self-supervised framework incorporating a differentiable hard constraint projection layer for runtime assurance. To mitigate data scarcity, we construct a Global-Guided Artificial Potential Field (G-APF), which provides dense supervision signals without manual labeling. To enforce actuator limitations and geometric constraints efficiently, we employ an adaptive neural projection layer, which iteratively rectifies the coarse network output onto the feasible manifold. Extensive benchmarks on a test set of 20,000 scenarios demonstrate an 88.75\% success rate, substantiating the enhanced operational safety. Closed-loop experiments in CARLA further validate the physical realizability of the planned paths under dynamic constraints. Furthermore, deployment verification on an NVIDIA Jetson Orin NX confirms an inference latency of 94 ms, showing real-time feasibility on resource-constrained embedded hardware. This framework offers a generalized paradigm for embedding physical laws into neural architectures, providing a viable direction for solving constrained optimization in mechatronics. Source code is available at: https://github.com/wzq-13/SSHC.git.

  </details>



- **Instance-Guided Radar Depth Estimation for 3D Object Detection**  
  Chen-Chou Lo, Patrick Vandewalle  
  _2026-01-27_ · https://arxiv.org/abs/2601.19314v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate depth estimation is fundamental to 3D perception in autonomous driving, supporting tasks such as detection, tracking, and motion planning. However, monocular camera-based 3D detection suffers from depth ambiguity and reduced robustness under challenging conditions. Radar provides complementary advantages such as resilience to poor lighting and adverse weather, but its sparsity and low resolution limit its direct use in detection frameworks. This motivates the need for effective Radar-camera fusion with improved preprocessing and depth estimation strategies. We propose an end-to-end framework that enhances monocular 3D object detection through two key components. First, we introduce InstaRadar, an instance segmentation-guided expansion method that leverages pre-trained segmentation masks to enhance Radar density and semantic alignment, producing a more structured representation. InstaRadar achieves state-of-the-art results in Radar-guided depth estimation, showing its effectiveness in generating high-quality depth features. Second, we integrate the pre-trained RCDPT into the BEVDepth framework as a replacement for its depth module. With InstaRadar-enhanced inputs, the RCDPT integration consistently improves 3D detection performance. Overall, these components yield steady gains over the baseline BEVDepth model, demonstrating the effectiveness of InstaRadar and the advantage of explicit depth supervision in 3D object detection. Although the framework lags behind Radar-camera fusion models that directly extract BEV features, since Radar serves only as guidance rather than an independent feature stream, this limitation highlights potential for improvement. Future work will extend InstaRadar to point cloud-like representations and integrate a dedicated Radar branch with temporal cues for enhanced BEV fusion.

  </details>



- **SETA: Statistical Fault Attribution for Compound AI Systems**  
  Sayak Chowdhury, Meenakshi D'Souza  
  _2026-01-27_ · https://arxiv.org/abs/2601.19337v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Modern AI systems increasingly comprise multiple interconnected neural networks to tackle complex inference tasks. Testing such systems for robustness and safety entails significant challenges. Current state-of-the-art robustness testing techniques, whether black-box or white-box, have been proposed and implemented for single-network models and do not scale well to multi-network pipelines. We propose a modular robustness testing framework that applies a given set of perturbations to test data. Our testing framework supports (1) a component-wise system analysis to isolate errors and (2) reasoning about error propagation across the neural network modules. The testing framework is architecture and modality agnostic and can be applied across domains. We apply the framework to a real-world autonomous rail inspection system composed of multiple deep networks and successfully demonstrate how our approach enables fine-grained robustness analysis beyond conventional end-to-end metrics.

  </details>



- **Enhancing Worker Safety in Harbors Using Quadruped Robots**  
  Zoe Betta, Davide Corongiu, Carmine Tommaso Recchiuto, Antonio Sgorbissa  
  _2026-01-27_ · https://arxiv.org/abs/2601.19643v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Infrastructure inspection is becoming increasingly relevant in the field of robotics due to its significant impact on ensuring workers' safety. The harbor environment presents various challenges in designing a robotic solution for inspection, given the complexity of daily operations. This work introduces an initial phase to identify critical areas within the port environment. Following this, a preliminary solution using a quadruped robot for inspecting these critical areas is analyzed.

  </details>



- **Enhancing Inverse Perspective Mapping for Automatic Vectorized Road Map Generation**  
  Hongji Liu, Linwei Zheng, Yongjian Li, Mingkai Tang, Xiaoyang Yan, Ming Liu, Jun Ma  
  _2026-01-27_ · https://arxiv.org/abs/2601.19536v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In this study, we present a low-cost and unified framework for vectorized road mapping leveraging enhanced inverse perspective mapping (IPM). In this framework, Catmull-Rom splines are utilized to characterize lane lines, and all the other ground markings are depicted using polygons uniformly. The results from instance segmentation serve as references to refine the three-dimensional position of spline control points and polygon corner points. In conjunction with this process, the homography matrix of IPM and vehicle poses are optimized simultaneously. Our proposed framework significantly reduces the mapping errors associated with IPM. It also improves the accuracy of the initial IPM homography matrix and the predicted vehicle poses. Furthermore, it addresses the limitations imposed by the coplanarity assumption in IPM. These enhancements enable IPM to be effectively applied to vectorized road mapping, which serves a cost-effective solution with enhanced accuracy. In addition, our framework generalizes road map elements to include all common ground markings and lane lines. The proposed framework is evaluated in two different practical scenarios, and the test results show that our method can automatically generate high-precision maps with near-centimeter-level accuracy. Importantly, the optimized IPM matrix achieves an accuracy comparable to that of manual calibration, while the accuracy of vehicle poses is also significantly improved.

  </details>



- **A DVL Aided Loosely Coupled Inertial Navigation Strategy for AUVs with Attitude Error Modeling and Variance Propagation**  
  Jin Huang, Zichen Liu, Haoda Li, Zhikun Wang, Ying Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19509v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In underwater navigation systems, strap-down inertial navigation system/Doppler velocity log (SINS/DVL)-based loosely coupled architectures are widely adopted. Conventional approaches project DVL velocities from the body coordinate system to the navigation coordinate system using SINS-derived attitude; however, accumulated attitude estimation errors introduce biases into velocity projection and degrade navigation performance during long-term operation. To address this issue, two complementary improvements are introduced. First, a vehicle attitude error-aware DVL velocity transformation model is formulated by incorporating attitude error terms into the observation equation to reduce projection-induced velocity bias. Second, a covariance matrix-based variance propagation method is developed to transform DVL measurement uncertainty across coordinate systems, introducing an expectation-based attitude error compensation term to achieve statistically consistent noise modeling. Simulation and field experiment results demonstrate that both improvements individually enhance navigation accuracy and confirm that accumulated attitude errors affect both projected velocity measurements and their associated uncertainty. When jointly applied, long-term error divergence is effectively suppressed. Field experimental results show that the proposed approach achieves a 78.3% improvement in 3D position RMSE and a 71.8% reduction in the maximum component-wise position error compared with the baseline IMU+DVL method, providing a robust solution for improving long-term SINS/DVL navigation performance.

  </details>



- **Reinforcement Learning Goal-Reaching Control with Guaranteed Lyapunov-Like Stabilizer for Mobile Robots**  
  Mehdi Heydari Shahna, Seyed Adel Alizadeh Kolagar, Jouni Mattila  
  _2026-01-27_ · https://arxiv.org/abs/2601.19499v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) can be highly effective at learning goal-reaching policies, but it typically does not provide formal guarantees that the goal will always be reached. A common approach to provide formal goal-reaching guarantees is to introduce a shielding mechanism that restricts the agent to actions that satisfy predefined safety constraints. The main challenge here is integrating this mechanism with RL so that learning and exploration remain effective without becoming overly conservative. Hence, this paper proposes an RL-based control framework that provides formal goal-reaching guarantees for wheeled mobile robots operating in unstructured environments. We first design a real-time RL policy with a set of 15 carefully defined reward terms. These rewards encourage the robot to reach both static and dynamic goals while generating sufficiently smooth command signals that comply with predefined safety specifications, which is critical in practice. Second, a Lyapunov-like stabilizer layer is integrated into the benchmark RL framework as a policy supervisor to formally strengthen the goal-reaching control while preserving meaningful exploration of the state action space. The proposed framework is suitable for real-time deployment in challenging environments, as it provides a formal guarantee of convergence to the intended goal states and compensates for uncertainties by generating real-time control signals based on the current state, while respecting real-world motion constraints. The experimental results show that the proposed Lyapunov-like stabilizer consistently improves the benchmark RL policies, boosting the goal-reaching rate from 84.6% to 99.0%, sharply reducing failures, and improving efficiency.

  </details>



- **Physical Human-Robot Interaction: A Critical Review of Safety Constraints**  
  Riccardo Zanella, Federico Califano, Stefano Stramigioli  
  _2026-01-27_ · https://arxiv.org/abs/2601.19462v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  This paper aims to provide a clear and rigorous understanding of commonly recognized safety constraints in physical human-robot interaction, i.e. ISO/TS 15066, by examining how they are obtained and which assumptions support them. We clarify the interpretation and practical impact of key simplifying assumptions, show how these modeling choices affect both safety and performance across the system, and indicate specific design parameters that can be adjusted in safety-critical control implementations. Numerical examples are provided to quantify performance degradation induced by common approximations and simplifying design choices. Furthermore, the fundamental role of energy in safety assessment is emphasized, and focused insights are offered on the existing body of work concerning energy-based safety methodologies.

  </details>



- **Estimating Trust in Human-Robot Collaboration through Behavioral Indicators and Explainability**  
  Giulio Campagna, Marta Lagomarsino, Marta Lorenzini, Dimitrios Chrysostomou, Matthias Rehm, Arash Ajoudani  
  _2026-01-27_ · https://arxiv.org/abs/2601.19856v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Industry 5.0 focuses on human-centric collaboration between humans and robots, prioritizing safety, comfort, and trust. This study introduces a data-driven framework to assess trust using behavioral indicators. The framework employs a Preference-Based Optimization algorithm to generate trust-enhancing trajectories based on operator feedback. This feedback serves as ground truth for training machine learning models to predict trust levels from behavioral indicators. The framework was tested in a chemical industry scenario where a robot assisted a human operator in mixing chemicals. Machine learning models classified trust with over 80\% accuracy, with the Voting Classifier achieving 84.07\% accuracy and an AUC-ROC score of 0.90. These findings underscore the effectiveness of data-driven methods in assessing trust within human-robot collaboration, emphasizing the valuable role behavioral indicators play in predicting the dynamics of human trust.

  </details>



- **When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering**  
  Mahdi Astaraki, Mohammad Arshi Saloot, Ali Shiraee Kasmaee, Hamidreza Mahyar, Soheila Samiee  
  _2026-01-27_ · https://arxiv.org/abs/2601.19827v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Retrieval-Augmented Generation (RAG) extends large language models (LLMs) beyond parametric knowledge, yet it is unclear when iterative retrieval-reasoning loops meaningfully outperform static RAG, particularly in scientific domains with multi-hop reasoning, sparse domain knowledge, and heterogeneous evidence. We provide the first controlled, mechanism-level diagnostic study of whether synchronized iterative retrieval and reasoning can surpass an idealized static upper bound (Gold Context) RAG. We benchmark eleven state-of-the-art LLMs under three regimes: (i) No Context, measuring reliance on parametric memory; (ii) Gold Context, where all oracle evidence is supplied at once; and (iii) Iterative RAG, a training-free controller that alternates retrieval, hypothesis refinement, and evidence-aware stopping. Using the chemistry-focused ChemKGMultiHopQA dataset, we isolate questions requiring genuine retrieval and analyze behavior with diagnostics spanning retrieval coverage gaps, anchor-carry drop, query quality, composition fidelity, and control calibration. Across models, Iterative RAG consistently outperforms Gold Context, with gains up to 25.6 percentage points, especially for non-reasoning fine-tuned models. Staged retrieval reduces late-hop failures, mitigates context overload, and enables dynamic correction of early hypothesis drift, but remaining failure modes include incomplete hop coverage, distractor latch trajectories, early stopping miscalibration, and high composition failure rates even with perfect retrieval. Overall, staged retrieval is often more influential than the mere presence of ideal evidence; we provide practical guidance for deploying and diagnosing RAG systems in specialized scientific settings and a foundation for more reliable, controllable iterative retrieval-reasoning frameworks.

  </details>



- **Out-of-Distribution Generalization via Invariant Trajectories for Multimodal Large Language Model Editing**  
  Jiajie Su, Haoyuan Wang, Xiaohua Feng, Yunshan Ma, Xiaobo Xia, Yuyuan Li, Xiaolin Zheng, Jianmao Xiao, Chaochao Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19700v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Knowledge editing emerges as a crucial technique for efficiently correcting incorrect or outdated knowledge in large language models (LLM). Existing editing methods for unimodal LLM rely on a rigid parameter-to-output mapping, which causes causal-underfit and causal-overfit in cascaded reasoning for Multimodal LLM (MLLM). In this paper, we reformulate MLLM editing as an out-of-distribution (OOD) generalization problem, where the goal is to discern semantic shift with factual shift and thus achieve robust editing among diverse cross-modal prompting. The key challenge of this OOD problem lies in identifying invariant causal trajectories that generalize accurately while suppressing spurious correlations. To address it, we propose ODEdit, a plug-and-play invariant learning based framework that optimizes the tripartite OOD risk objective to simultaneously enhance editing reliability, locality, and generality.We further introduce an edit trajectory invariant learning method, which integrates a total variation penalty into the risk minimization objective to stabilize edit trajectories against environmental variations. Theoretical analysis and extensive experiments demonstrate the effectiveness of ODEdit.

  </details>



- **PALM: Enhanced Generalizability for Local Visuomotor Policies via Perception Alignment**  
  Ruiyu Wang, Zheyu Zhuang, Danica Kragic, Florian T. Pokorny  
  _2026-01-27_ · https://arxiv.org/abs/2601.19514v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Generalizing beyond the training domain in image-based behavior cloning remains challenging. Existing methods address individual axes of generalization, workspace shifts, viewpoint changes, and cross-embodiment transfer, yet they are typically developed in isolation and often rely on complex pipelines. We introduce PALM (Perception Alignment for Local Manipulation), which leverages the invariance of local action distributions between out-of-distribution (OOD) and demonstrated domains to address these OOD shifts concurrently, without additional input modalities, model changes, or data collection. PALM modularizes the manipulation policy into coarse global components and a local policy for fine-grained actions. We reduce the discrepancy between in-domain and OOD inputs at the local policy level by enforcing local visual focus and consistent proprioceptive representation, allowing the policy to retrieve invariant local actions under OOD conditions. Experiments show that PALM limits OOD performance drops to 8% in simulation and 24% in the real world, compared to 45% and 77% for baselines.

  </details>



- **Bridging Information Asymmetry: A Hierarchical Framework for Deterministic Blind Face Restoration**  
  Zhengjian Yao, Jiakui Hu, Kaiwen Li, Hangzhou He, Xinliang Zhang, Shuang Zeng, Lei Zhu, Yanye Lu  
  _2026-01-27_ · https://arxiv.org/abs/2601.19506v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Blind face restoration remains a persistent challenge due to the inherent ill-posedness of reconstructing holistic structures from severely constrained observations. Current generative approaches, while capable of synthesizing realistic textures, often suffer from information asymmetry -- the intrinsic disparity between the information-sparse low quality inputs and the information-dense high quality outputs. This imbalance leads to a one-to-many mapping, where insufficient constraints result in stochastic uncertainty and hallucinatory artifacts. To bridge this gap, we present \textbf{Pref-Restore}, a hierarchical framework that integrates discrete semantic logic with continuous texture generation to achieve deterministic, preference-aligned restoration. Our methodology fundamentally addresses this information disparity through two complementary strategies: (1) Augmenting Input Density: We employ an auto-regressive integrator to reformulate textual instructions into dense latent queries, injecting high-level semantic stability to constrain the degraded signals; (2) Pruning Output Distribution: We pioneer the integration of on-policy reinforcement learning directly into the diffusion restoration loop. By transforming human preferences into differentiable constraints, we explicitly penalize stochastic deviations, thereby sharpening the posterior distribution toward the desired high-fidelity outcomes. Extensive experiments demonstrate that Pref-Restore achieves state-of-the-art performance across synthetic and real-world benchmarks. Furthermore, empirical analysis confirms that our preference-aligned strategy significantly reduces solution entropy, establishing a robust pathway toward reliable and deterministic blind restoration.

  </details>



- **Cortex-Grounded Diffusion Models for Brain Image Generation**  
  Fabian Bongratz, Yitong Li, Sama Elbaroudy, Christian Wachinger  
  _2026-01-27_ · https://arxiv.org/abs/2601.19498v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Synthetic neuroimaging data can mitigate critical limitations of real-world datasets, including the scarcity of rare phenotypes, domain shifts across scanners, and insufficient longitudinal coverage. However, existing generative models largely rely on weak conditioning signals, such as labels or text, which lack anatomical grounding and often produce biologically implausible outputs. To this end, we introduce Cor2Vox, a cortex-grounded generative framework for brain magnetic resonance image (MRI) synthesis that ties image generation to continuous structural priors of the cerebral cortex. It leverages high-resolution cortical surfaces to guide a 3D shape-to-image Brownian bridge diffusion process, enabling topologically faithful synthesis and precise control over underlying anatomies. To support the generation of new, realistic brain shapes, we developed a large-scale statistical shape model of cortical morphology derived from over 33,000 UK Biobank scans. We validated the fidelity of Cor2Vox based on traditional image quality metrics, advanced cortical surface reconstruction, and whole-brain segmentation quality, outperforming many baseline methods. Across three applications, namely (i) anatomically consistent synthesis, (ii) simulation of progressive gray matter atrophy, and (iii) harmonization of in-house frontotemporal dementia scans with public datasets, Cor2Vox preserved fine-grained cortical morphology at the sub-voxel level, exhibiting remarkable robustness to variations in cortical geometry and disease phenotype without retraining.

  </details>



- **Task-Centric Policy Optimization from Misaligned Motion Priors**  
  Ziang Zheng, Kai Feng, Yi Nie, Shentao Qin  
  _2026-01-27_ · https://arxiv.org/abs/2601.19411v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Humanoid control often leverages motion priors from human demonstrations to encourage natural behaviors. However, such demonstrations are frequently suboptimal or misaligned with robotic tasks due to embodiment differences, retargeting errors, and task-irrelevant variations, causing naïve imitation to degrade task performance. Conversely, task-only reinforcement learning admits many task-optimal solutions, often resulting in unnatural or unstable motions. This exposes a fundamental limitation of linear reward mixing in adversarial imitation learning. We propose \emph{Task-Centric Motion Priors} (TCMP), a task-priority adversarial imitation framework that treats imitation as a conditional regularizer rather than a co-equal objective. TCMP maximizes task improvement while incorporating imitation signals only when they are compatible with task progress, yielding an adaptive, geometry-aware update that preserves task-feasible descent and suppresses harmful imitation under misalignment. We provide theoretical analysis of gradient conflict and task-priority stationary points, and validate our claims through humanoid control experiments demonstrating robust task performance with consistent motion style under noisy demonstrations.

  </details>



- **DSP-Reg: Domain-Sensitive Parameter Regularization for Robust Domain Generalization**  
  Xudong Han, Senkang Hu, Yihang Tao, Yu Guo, Philip Birch, Sam Tak Wu Kwong, Yuguang Fang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19394v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Domain Generalization (DG) is a critical area that focuses on developing models capable of performing well on data from unseen distributions, which is essential for real-world applications. Existing approaches primarily concentrate on learning domain-invariant features, which assume that a model robust to variations in the source domains will generalize well to unseen target domains. However, these approaches neglect a deeper analysis at the parameter level, which makes the model hard to explicitly differentiate between parameters sensitive to domain shifts and those robust, potentially hindering its overall ability to generalize. In order to address these limitations, we first build a covariance-based parameter sensitivity analysis framework to quantify the sensitivity of each parameter in a model to domain shifts. By computing the covariance of parameter gradients across multiple source domains, we can identify parameters that are more susceptible to domain variations, which serves as our theoretical foundation. Based on this, we propose Domain-Sensitive Parameter Regularization (DSP-Reg), a principled framework that guides model optimization by a soft regularization technique that encourages the model to rely more on domain-invariant parameters while suppressing those that are domain-specific. This approach provides a more granular control over the model's learning process, leading to improved robustness and generalization to unseen domains. Extensive experiments on benchmarks, such as PACS, VLCS, OfficeHome, and DomainNet, demonstrate that DSP-Reg outperforms state-of-the-art approaches, achieving an average accuracy of 66.7\% and surpassing all baselines.

  </details>



- **Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection**  
  Quy-Anh Dang, Chris Ngo  
  _2026-01-27_ · https://arxiv.org/abs/2601.19375v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Despite significant progress in alignment, large language models (LLMs) remain vulnerable to adversarial attacks that elicit harmful behaviors. Activation steering techniques offer a promising inference-time intervention approach, but existing methods suffer from critical limitations: activation addition requires careful coefficient tuning and is sensitive to layer-specific norm variations, while directional ablation provides only binary control. Recent work on Angular Steering introduces continuous control via rotation in a 2D subspace, but its practical implementation violates norm preservation, causing distribution shift and generation collapse, particularly in models below 7B parameters. We propose Selective Steering, which addresses these limitations through two key innovations: (1) a mathematically rigorous norm-preserving rotation formulation that maintains activation distribution integrity, and (2) discriminative layer selection that applies steering only where feature representations exhibit opposite-signed class alignment. Experiments across nine models demonstrate that Selective Steering achieves 5.5x higher attack success rates than prior methods while maintaining zero perplexity violations and approximately 100\% capability retention on standard benchmarks. Our approach provides a principled, efficient framework for controllable and stable LLM behavior modification. Code: https://github.com/knoveleng/steering

  </details>



- **Curiosity Driven Knowledge Retrieval for Mobile Agents**  
  Sijia Li, Xiaoyu Tan, Shahir Ali, Niels Schmidt, Gengchen Ma, Xihe Qiu  
  _2026-01-27_ · https://arxiv.org/abs/2601.19306v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Mobile agents have made progress toward reliable smartphone automation, yet performance in complex applications remains limited by incomplete knowledge and weak generalization to unseen environments. We introduce a curiosity driven knowledge retrieval framework that formalizes uncertainty during execution as a curiosity score. When this score exceeds a threshold, the system retrieves external information from documentation, code repositories, and historical trajectories. Retrieved content is organized into structured AppCards, which encode functional semantics, parameter conventions, interface mappings, and interaction patterns. During execution, an enhanced agent selectively integrates relevant AppCards into its reasoning process, thereby compensating for knowledge blind spots and improving planning reliability. Evaluation on the AndroidWorld benchmark shows consistent improvements across backbones, with an average gain of six percentage points and a new state of the art success rate of 88.8\% when combined with GPT-5. Analysis indicates that AppCards are particularly effective for multi step and cross application tasks, while improvements depend on the backbone model. Case studies further confirm that AppCards reduce ambiguity, shorten exploration, and support stable execution trajectories. Task trajectories are publicly available at https://lisalsj.github.io/Droidrun-appcard/.

  </details>


