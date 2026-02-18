# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-18 07:14 UTC_

Total papers shown: **17**


---

- **SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms**  
  Haichao Liu, Yufeng Hu, Shuang Wang, Kangjun Guo, Jun Ma, Jinni Zhou  
  _2026-02-17_ · https://arxiv.org/abs/2602.15633v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous landing of Uncrewed Aerial Vehicles (UAVs) on oscillating marine platforms is severely constrained by wave-induced multi-frequency oscillations, wind disturbances, and prediction phase lags in motion prediction. Existing methods either treat platform motion as a general random process or lack explicit modeling of wave spectral characteristics, leading to suboptimal performance under dynamic sea conditions. To address these limitations, we propose SpecFuse: a novel spectral-temporal fusion predictive control framework that integrates frequency-domain wave decomposition with time-domain recursive state estimation for high-precision 6-DoF motion forecasting of Uncrewed Surface Vehicles (USVs). The framework explicitly models dominant wave harmonics to mitigate phase lags, refining predictions in real time via IMU data without relying on complex calibration. Additionally, we design a hierarchical control architecture featuring a sampling-based HPO-RRT* algorithm for dynamic trajectory planning under non-convex constraints and a learning-augmented predictive controller that fuses data-driven disturbance compensation with optimization-based execution. Extensive validations (2,000 simulations + 8 lake experiments) show our approach achieves a 3.2 cm prediction error, 4.46 cm landing deviation, 98.7% / 87.5% success rates (simulation / real-world), and 82 ms latency on embedded hardware, outperforming state-of-the-art methods by 44%-48% in accuracy. Its robustness to wave-wind coupling disturbances supports critical maritime missions such as search and rescue and environmental monitoring. All code, experimental configurations, and datasets will be released as open-source to facilitate reproducibility.

  </details>



- **Selective Perception for Robot: Task-Aware Attention in Multimodal VLA**  
  Young-Chae Son, Jung-Woo Lee, Yoon-Ji Choi, Dae-Kwan Ko, Soo-Chul Lim  
  _2026-02-17_ · https://arxiv.org/abs/2602.15543v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In robotics, Vision-Language-Action (VLA) models that integrate diverse multimodal signals from multi-view inputs have emerged as an effective approach. However, most prior work adopts static fusion that processes all visual inputs uniformly, which incurs unnecessary computational overhead and allows task-irrelevant background information to act as noise. Inspired by the principles of human active perception, we propose a dynamic information fusion framework designed to maximize the efficiency and robustness of VLA models. Our approach introduces a lightweight adaptive routing architecture that analyzes the current text prompt and observations from a wrist-mounted camera in real-time to predict the task-relevance of multiple camera views. By conditionally attenuating computations for views with low informational utility and selectively providing only essential visual features to the policy network, Our framework achieves computation efficiency proportional to task relevance. Furthermore, to efficiently secure large-scale annotation data for router training, we established an automated labeling pipeline utilizing Vision-Language Models (VLMs) to minimize data collection and annotation costs. Experimental results in real-world robotic manipulation scenarios demonstrate that the proposed approach achieves significant improvements in both inference efficiency and control performance compared to existing VLA models, validating the effectiveness and practicality of dynamic information fusion in resource-constrained, real-time robot control environments.

  </details>



- **Estimating Human Muscular Fatigue in Dynamic Collaborative Robotic Tasks with Learning-Based Models**  
  Feras Kiki, Pouya P. Niaz, Alireza Madani, Cagatay Basdogan  
  _2026-02-17_ · https://arxiv.org/abs/2602.15684v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Assessing human muscle fatigue is critical for optimizing performance and safety in physical human-robot interaction(pHRI). This work presents a data-driven framework to estimate fatigue in dynamic, cyclic pHRI using arm-mounted surface electromyography(sEMG). Subject-specific machine-learning regression models(Random Forest, XGBoost, and Linear Regression predict the fraction of cycles to fatigue(FCF) from three frequency-domain and one time-domain EMG features, and are benchmarked against a convolutional neural network(CNN) that ingests spectrograms of filtered EMG. Framing fatigue estimation as regression (rather than classification) captures continuous progression toward fatigue, supporting earlier detection, timely intervention, and adaptive robot control. In experiments with ten participants, a collaborative robot under admittance control guided repetitive lateral (left-right) end-effector motions until muscular fatigue. Average FCF RMSE across participants was 20.8+/-4.3% for the CNN, 23.3+/-3.8% for Random Forest, 24.8+/-4.5% for XGBoost, and 26.9+/-6.1% for Linear Regression. To probe cross-task generalization, one participant additionally performed unseen vertical (up-down) and circular repetitions; models trained only on lateral data were tested directly and largely retained accuracy, indicating robustness to changes in movement direction, arm kinematics, and muscle recruitment, while Linear Regression deteriorated. Overall, the study shows that both feature-based ML and spectrogram-based DL can estimate remaining work capacity during repetitive pHRI, with the CNN delivering the lowest error and the tree-based models close behind. The reported transfer to new motion patterns suggests potential for practical fatigue monitoring without retraining for every task, improving operator protection and enabling fatigue-aware shared autonomy, for safer fatigue-adaptive pHRI control.

  </details>



- **Dex4D: Task-Agnostic Point Track Policy for Sim-to-Real Dexterous Manipulation**  
  Yuxuan Kuang, Sungjae Park, Katerina Fragkiadaki, Shubham Tulsiani  
  _2026-02-17_ · https://arxiv.org/abs/2602.15828v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning generalist policies capable of accomplishing a plethora of everyday tasks remains an open challenge in dexterous manipulation. In particular, collecting large-scale manipulation data via real-world teleoperation is expensive and difficult to scale. While learning in simulation provides a feasible alternative, designing multiple task-specific environments and rewards for training is similarly challenging. We propose Dex4D, a framework that instead leverages simulation for learning task-agnostic dexterous skills that can be flexibly recomposed to perform diverse real-world manipulation tasks. Specifically, Dex4D learns a domain-agnostic 3D point track conditioned policy capable of manipulating any object to any desired pose. We train this 'Anypose-to-Anypose' policy in simulation across thousands of objects with diverse pose configurations, covering a broad space of robot-object interactions that can be composed at test time. At deployment, this policy can be zero-shot transferred to real-world tasks without finetuning, simply by prompting it with desired object-centric point tracks extracted from generated videos. During execution, Dex4D uses online point tracking for closed-loop perception and control. Extensive experiments in simulation and on real robots show that our method enables zero-shot deployment for diverse dexterous manipulation tasks and yields consistent improvements over prior baselines. Furthermore, we demonstrate strong generalization to novel objects, scene layouts, backgrounds, and trajectories, highlighting the robustness and scalability of the proposed framework.

  </details>



- **Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching**  
  Zhen Wu, Xiaoyu Huang, Lujie Yang, Yuanhang Zhang, Koushil Sreenath, Xi Chen, Pieter Abbeel, Rocky Duan, Angjoo Kanazawa, Carmelo Sferrazza, et al.  
  _2026-02-17_ · https://arxiv.org/abs/2602.15827v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While recent advances in humanoid locomotion have achieved stable walking on varied terrains, capturing the agility and adaptivity of highly dynamic human motions remains an open challenge. In particular, agile parkour in complex environments demands not only low-level robustness, but also human-like motion expressiveness, long-horizon skill composition, and perception-driven decision-making. In this paper, we present Perceptive Humanoid Parkour (PHP), a modular framework that enables humanoid robots to autonomously perform long-horizon, vision-based parkour across challenging obstacle courses. Our approach first leverages motion matching, formulated as nearest-neighbor search in a feature space, to compose retargeted atomic human skills into long-horizon kinematic trajectories. This framework enables the flexible composition and smooth transition of complex skill chains while preserving the elegance and fluidity of dynamic human motions. Next, we train motion-tracking reinforcement learning (RL) expert policies for these composed motions, and distill them into a single depth-based, multi-skill student policy, using a combination of DAgger and RL. Crucially, the combination of perception and skill composition enables autonomous, context-aware decision-making: using only onboard depth sensing and a discrete 2D velocity command, the robot selects and executes whether to step over, climb onto, vault or roll off obstacles of varying geometries and heights. We validate our framework with extensive real-world experiments on a Unitree G1 humanoid robot, demonstrating highly dynamic parkour skills such as climbing tall obstacles up to 1.25m (96% robot height), as well as long-horizon multi-obstacle traversal with closed-loop adaptation to real-time obstacle perturbations.

  </details>



- **When Remembering and Planning are Worth it: Navigating under Change**  
  Omid Madani, J. Brian Burns, Reza Eghbali, Thomas L. Dean  
  _2026-02-17_ · https://arxiv.org/abs/2602.15274v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We explore how different types and uses of memory can aid spatial navigation in changing uncertain environments. In the simple foraging task we study, every day, our agent has to find its way from its home, through barriers, to food. Moreover, the world is non-stationary: from day to day, the location of the barriers and food may change, and the agent's sensing such as its location information is uncertain and very limited. Any model construction, such as a map, and use, such as planning, needs to be robust against these challenges, and if any learning is to be useful, it needs to be adequately fast. We look at a range of strategies, from simple to sophisticated, with various uses of memory and learning. We find that an architecture that can incorporate multiple strategies is required to handle (sub)tasks of a different nature, in particular for exploration and search, when food location is not known, and for planning a good path to a remembered (likely) food location. An agent that utilizes non-stationary probability learning techniques to keep updating its (episodic) memories and that uses those memories to build maps and plan on the fly (imperfect maps, i.e. noisy and limited to the agent's experience) can be increasingly and substantially more efficient than the simpler (minimal-memory) agents, as the task difficulties such as distance to goal are raised, as long as the uncertainty, from localization and change, is not too large.

  </details>



- **Constraining Streaming Flow Models for Adapting Learned Robot Trajectory Distributions**  
  Jieting Long, Dechuan Liu, Weidong Cai, Ian Manchester, Weiming Zhi  
  _2026-02-17_ · https://arxiv.org/abs/2602.15567v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robot motion distributions often exhibit multi-modality and require flexible generative models for accurate representation. Streaming Flow Policies (SFPs) have recently emerged as a powerful paradigm for generating robot trajectories by integrating learned velocity fields directly in action space, enabling smooth and reactive control. However, existing formulations lack mechanisms for adapting trajectories post-training to enforce safety and task-specific constraints. We propose Constraint-Aware Streaming Flow (CASF), a framework that augments streaming flow policies with constraint-dependent metrics that reshape the learned velocity field during execution. CASF models each constraint, defined in either the robot's workspace or configuration space, as a differentiable distance function that is converted into a local metric and pulled back into the robot's control space. Far from restricted regions, the resulting metric reduces to the identity; near constraint boundaries, it smoothly attenuates or redirects motion, effectively deforming the underlying flow to maintain safety. This allows trajectories to be adapted in real time, ensuring that robot actions respect joint limits, avoid collisions, and remain within feasible workspaces, while preserving the multi-modal and reactive properties of streaming flow policies. We demonstrate CASF in simulated and real-world manipulation tasks, showing that it produces constraint-satisfying trajectories that remain smooth, feasible, and dynamically consistent, outperforming standard post-hoc projection baselines.

  </details>



- **On the Out-of-Distribution Generalization of Reasoning in Multimodal LLMs for Simple Visual Planning Tasks**  
  Yannic Neuhaus, Nicolas Flammarion, Matthias Hein, Francesco Croce  
  _2026-02-17_ · https://arxiv.org/abs/2602.15460v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Integrating reasoning in large language models and large vision-language models has recently led to significant improvement of their capabilities. However, the generalization of reasoning models is still vaguely defined and poorly understood. In this work, we present an evaluation framework to rigorously examine how well chain-of-thought (CoT) approaches generalize on a simple planning task. Specifically, we consider a grid-based navigation task in which a model is provided with a map and must output a sequence of moves that guides a player from a start position to a goal while avoiding obstacles. The versatility of the task and its data allows us to fine-tune model variants using different input representations (visual and textual) and CoT reasoning strategies, and systematically evaluate them under both in-distribution (ID) and out-of-distribution (OOD) test conditions. Our experiments show that, while CoT reasoning improves in-distribution generalization across all representations, out-of-distribution generalization (e.g., to larger maps) remains very limited in most cases when controlling for trivial matches with the ID data. Surprisingly, we find that reasoning traces which combine multiple text formats yield the best (and non-trivial) OOD generalization. Finally, purely text-based models consistently outperform those utilizing image-based inputs, including a recently proposed approach relying on latent space reasoning.

  </details>



- **Lyapunov-Based $\mathcal{L}_2$-Stable PI-Like Control of a Four-Wheel Independently Driven and Steered Robot**  
  Branimir Ćaran, Vladimir Milić, Bojan Jerbić  
  _2026-02-17_ · https://arxiv.org/abs/2602.15424v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In this letter, Lyapunov-based synthesis of a PI-like controller is proposed for $\mathcal{L}_2$-stable motion control of an independently driven and steered four-wheel mobile robot. An explicit, structurally verified model is used to enable systematic controller design with stability and performance guarantees suitable for real-time operation. A Lyapunov function is constructed to yield explicit bounds and $\mathcal{L}_2$ stability results, supporting feedback synthesis that reduces configuration dependent effects. The resulting control law maintains a PI-like form suitable for standard embedded implementation while preserving rigorous stability properties. Effectiveness and robustness are demonstrated experimentally on a real four-wheel mobile robot platform.

  </details>



- **Fluoroscopy-Constrained Magnetic Robot Control via Zernike-Based Field Modeling and Nonlinear MPC**  
  Xinhao Chen, Hongkun Yao, Anuruddha Bhattacharjee, Suraj Raval, Lamar O. Mair, Yancy Diaz-Mercado, Axel Krieger  
  _2026-02-17_ · https://arxiv.org/abs/2602.15357v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Magnetic actuation enables surgical robots to navigate complex anatomical pathways while reducing tissue trauma and improving surgical precision. However, clinical deployment is limited by the challenges of controlling such systems under fluoroscopic imaging, which provides low frame rate and noisy pose feedback. This paper presents a control framework that remains accurate and stable under such conditions by combining a nonlinear model predictive control (NMPC) framework that directly outputs coil currents, an analytically differentiable magnetic field model based on Zernike polynomials, and a Kalman filter to estimate the robot state. Experimental validation is conducted with two magnetic robots in a 3D-printed fluid workspace and a spine phantom replicating drug delivery in the epidural space. Results show the proposed control method remains highly accurate when feedback is downsampled to 3 Hz with added Gaussian noise (sigma = 2 mm), mimicking clinical fluoroscopy. In the spine phantom experiments, the proposed method successfully executed a drug delivery trajectory with a root mean square (RMS) position error of 1.18 mm while maintaining safe clearance from critical anatomical boundaries.

  </details>



- **Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning**  
  Oswin So, Eric Yang Yu, Songyuan Zhang, Matthew Cleaveland, Mitchell Black, Chuchu Fan  
  _2026-02-17_ · https://arxiv.org/abs/2602.15817v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Recent advances in deep reinforcement learning (RL) have achieved strong results on high-dimensional control tasks, but applying RL to reachability problems raises a fundamental mismatch: reachability seeks to maximize the set of states from which a system remains safe indefinitely, while RL optimizes expected returns over a user-specified distribution. This mismatch can result in policies that perform poorly on low-probability states that are still within the safe set. A natural alternative is to frame the problem as a robust optimization over a set of initial conditions that specify the initial state, dynamics and safe set, but whether this problem has a solution depends on the feasibility of the specified set, which is unknown a priori. We propose Feasibility-Guided Exploration (FGE), a method that simultaneously identifies a subset of feasible initial conditions under which a safe policy exists, and learns a policy to solve the reachability problem over this set of initial conditions. Empirical results demonstrate that FGE learns policies with over 50% more coverage than the best existing method for challenging initial conditions across tasks in the MuJoCo simulator and the Kinetix simulator with pixel observations.

  </details>



- **CAMEL: An ECG Language Model for Forecasting Cardiac Events**  
  Neelay Velingker, Alaia Solko-Breslin, Mayank Keoliya, Seewon Choi, Jiayi Xin, Anika Marathe, Alireza Oraii, Rajat Deo, Sameed Khatana, Rajeev Alur, et al.  
  _2026-02-17_ · https://arxiv.org/abs/2602.15677v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Electrocardiograms (ECG) are electrical recordings of the heart that are critical for diagnosing cardiovascular conditions. ECG language models (ELMs) have recently emerged as a promising framework for ECG classification accompanied by report generation. However, current models cannot forecast future cardiac events despite the immense clinical value for planning earlier intervention. To address this gap, we propose CAMEL, the first ELM that is capable of inference over longer signal durations which enables its forecasting capability. Our key insight is a specialized ECG encoder which enables cross-understanding of ECG signals with text. We train CAMEL using established LLM training procedures, combining LoRA adaptation with a curriculum learning pipeline. Our curriculum includes ECG classification, metrics calculations, and multi-turn conversations to elicit reasoning. CAMEL demonstrates strong zero-shot performance across 6 tasks and 9 datasets, including ECGForecastBench, a new benchmark that we introduce for forecasting arrhythmias. CAMEL is on par with or surpasses ELMs and fully supervised baselines both in- and out-of-distribution, achieving SOTA results on ECGBench (+7.0% absolute average gain) as well as ECGForecastBench (+12.4% over fully supervised models and +21.1% over zero-shot ELMs).

  </details>



- **Spatially-Aware Adaptive Trajectory Optimization with Controller-Guided Feedback for Autonomous Racing**  
  Alexander Wachter, Alexander Willert, Marc-Philip Ecker, Christian Hartl-Nesic  
  _2026-02-17_ · https://arxiv.org/abs/2602.15642v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a closed-loop framework for autonomous raceline optimization that combines NURBS-based trajectory representation, CMA-ES global trajectory optimization, and controller-guided spatial feedback. Instead of treating tracking errors as transient disturbances, our method exploits them as informative signals of local track characteristics via a Kalman-inspired spatial update. This enables the construction of an adaptive, acceleration-based constraint map that iteratively refines trajectories toward near-optimal performance under spatially varying track and vehicle behavior. In simulation, our approach achieves a 17.38% lap time reduction compared to a controller parametrized with maximum static acceleration. On real hardware, tested with different tire compounds ranging from high to low friction, we obtain a 7.60% lap time improvement without explicitly parametrizing friction. This demonstrates robustness to changing grip conditions in real-world scenarios.

  </details>



- **Latency-aware Human-in-the-Loop Reinforcement Learning for Semantic Communications**  
  Peizheng Li, Xinyi Lin, Adnan Aijaz  
  _2026-02-17_ · https://arxiv.org/abs/2602.15640v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Semantic communication promises task-aligned transmission but must reconcile semantic fidelity with stringent latency guarantees in immersive and safety-critical services. This paper introduces a time-constrained human-in-the-loop reinforcement learning (TC-HITL-RL) framework that embeds human feedback, semantic utility, and latency control within a semantic-aware Open radio access network (RAN) architecture. We formulate semantic adaptation driven by human feedback as a constrained Markov decision process (CMDP) whose state captures semantic quality, human preferences, queue slack, and channel dynamics, and solve it via a primal--dual proximal policy optimization algorithm with action shielding and latency-aware reward shaping. The resulting policy preserves PPO-level semantic rewards while tightening the variability of both air-interface and near-real-time RAN intelligent controller processing budgets. Simulations over point-to-multipoint links with heterogeneous deadlines show that TC-HITL-RL consistently meets per-user timing constraints, outperforms baseline schedulers in reward, and stabilizes resource consumption, providing a practical blueprint for latency-aware semantic adaptation.

  </details>



- **A Comparison of Bayesian Prediction Techniques for Mobile Robot Trajectory Tracking**  
  Jose Luis Peralta-Cabezas, Miguel Torres-Torriti, Marcelo Guarini-Hermann  
  _2026-02-17_ · https://arxiv.org/abs/2602.15354v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a performance comparison of different estimation and prediction techniques applied to the problem of tracking multiple robots. The main performance criteria are the magnitude of the estimation or prediction error, the computational effort and the robustness of each method to non-Gaussian noise. Among the different techniques compared are the well known Kalman filters and their different variants (e.g. extended and unscented), and the more recent techniques relying on Sequential Monte Carlo Sampling methods, such as particle filters and Gaussian Mixture Sigma Point Particle Filter.

  </details>



- **High-Fidelity Network Management for Federated AI-as-a-Service: Cross-Domain Orchestration**  
  Merve Saimler, Mohaned Chraiti, Ozgur Ercetin  
  _2026-02-17_ · https://arxiv.org/abs/2602.15281v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  To support the emergence of AI-as-a-Service (AIaaS), communication service providers (CSPs) are on the verge of a radical transformation-from pure connectivity providers to AIaaS a managed network service (control-and-orchestration plane that exposes AI models). In this model, the CSP is responsible not only for transport/communications, but also for intent-to-model resolution and joint network-compute orchestration, i.e., reliable and timely end-to-end delivery. The resulting end-to-end AIaaS service thus becomes governed by communications impairments (delay, loss) and inference impairments (latency, error). A central open problem is an operational AIaaS control-and-orchestration framework that enforces high fidelity, particularly under multi-domain federation. This paper introduces an assurance-oriented AIaaS management plane based on Tail-Risk Envelopes (TREs): signed, composable per-domain descriptors that combine deterministic guardrails with stochastic rate-latency-impairment models. Using stochastic network calculus, we derive bounds on end-to-end delay violation probabilities across tandem domains and obtain an optimization-ready risk-budget decomposition. We show that tenant-level reservations prevent bursty traffic from inflating tail latency under TRE contracts. An auditing layer then uses runtime telemetry to estimate extreme-percentile performance, quantify uncertainty, and attribute tail-risk to each domain for accountability. Packet-level Monte-Carlo simulations demonstrate improved p99.9 compliance under overload via admission control and robust tenant isolation under correlated burstiness.

  </details>



- **Enhancing Diversity and Feasibility: Joint Population Synthesis from Multi-source Data Using Generative Models**  
  Farbod Abbasi, Zachary Patterson, Bilal Farooq  
  _2026-02-17_ · https://arxiv.org/abs/2602.15270v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Generating realistic synthetic populations is essential for agent-based models (ABM) in transportation and urban planning. Current methods face two major limitations. First, many rely on a single dataset or follow a sequential data fusion and generation process, which means they fail to capture the complex interplay between features. Second, these approaches struggle with sampling zeros (valid but unobserved attribute combinations) and structural zeros (infeasible combinations due to logical constraints), which reduce the diversity and feasibility of the generated data. This study proposes a novel method to simultaneously integrate and synthesize multi-source datasets using a Wasserstein Generative Adversarial Network (WGAN) with gradient penalty. This joint learning method improves both the diversity and feasibility of synthetic data by defining a regularization term (inverse gradient penalty) for the generator loss function. For the evaluation, we implement a unified evaluation metric for similarity, and place special emphasis on measuring diversity and feasibility through recall, precision, and the F1 score. Results show that the proposed joint approach outperforms the sequential baseline, with recall increasing by 7\% and precision by 15\%. Additionally, the regularization term further improves diversity and feasibility, reflected in a 10\% increase in recall and 1\% in precision. We assess similarity distributions using a five-metric score. The joint approach performs better overall, and reaches a score of 88.1 compared to 84.6 for the sequential method. Since synthetic populations serve as a key input for ABM, this multi-source generative approach has the potential to significantly enhance the accuracy and reliability of ABM.

  </details>


