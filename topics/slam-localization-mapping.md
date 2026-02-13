# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-13 07:12 UTC_

Total papers shown: **11**


---

- **6G Empowering Future Robotics: A Vision for Next-Generation Autonomous Systems**  
  Mona Ghassemian, Andrés Meseguer Valenzuela, Ana Garcia Armada, Dejan Vukobratovic, Periklis Chatzimisios, Kaspar Althoefer, Ranga Rao Venkatesha Prasad  
  _2026-02-12_ · https://arxiv.org/abs/2602.12246v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  The convergence of robotics and next-generation communication is a critical driver of technological advancement. As the world transitions from 5G to 6G, the foundational capabilities of wireless networks are evolving to support increasingly complex and autonomous robotic systems. This paper examines the transformative impact of 6G on enhancing key robotics functionalities. It provides a systematic mapping of IMT-2030 key performance indicators to robotic functional blocks including sensing, perception, cognition, actuation and self-learning. Building upon this mapping, we propose a high-level architectural framework integrating robotic, intelligent, and network service planes, underscoring the need for a holistic approach. As an example use case, we present a real-time, dynamic safety framework enabled by IMT-2030 capabilities for safe and efficient human-robot collaboration in shared spaces.

  </details>



- **Decentralized Multi-Robot Obstacle Detection and Tracking in a Maritime Scenario**  
  Muhammad Farhan Ahmed, Vincent Frémont  
  _2026-02-12_ · https://arxiv.org/abs/2602.12012v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous aerial-surface robot teams are promising for maritime monitoring. Robust deployment requires reliable perception over reflective water and scalable coordination under limited communication. We present a decentralized multi-robot framework for detecting and tracking floating containers using multiple UAVs cooperating with an autonomous surface vessel. Each UAV performs YOLOv8 and stereo-disparity-based visual detection, then tracks targets with per-object EKFs using uncertainty-aware data association. Compact track summaries are exchanged and fused conservatively via covariance intersection, ensuring consistency under unknown correlations. An information-driven assignment module allocates targets and selects UAV hover viewpoints by trading expected uncertainty reduction against travel effort and safety separation. Simulation results in a maritime scenario demonstrate improved coverage, localization accuracy, and tracking consistency while maintaining modest communication requirements.

  </details>



- **When would Vision-Proprioception Policies Fail in Robotic Manipulation?**  
  Jingxian Lu, Wenke Xia, Yuxuan Wu, Zhiwu Lu, Di Hu  
  _2026-02-12_ · https://arxiv.org/abs/2602.12032v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Proprioceptive information is critical for precise servo control by providing real-time robotic states. Its collaboration with vision is highly expected to enhance performances of the manipulation policy in complex tasks. However, recent studies have reported inconsistent observations on the generalization of vision-proprioception policies. In this work, we investigate this by conducting temporally controlled experiments. We found that during task sub-phases that robot's motion transitions, which require target localization, the vision modality of the vision-proprioception policy plays a limited role. Further analysis reveals that the policy naturally gravitates toward concise proprioceptive signals that offer faster loss reduction when training, thereby dominating the optimization and suppressing the learning of the visual modality during motion-transition phases. To alleviate this, we propose the Gradient Adjustment with Phase-guidance (GAP) algorithm that adaptively modulates the optimization of proprioception, enabling dynamic collaboration within the vision-proprioception policy. Specifically, we leverage proprioception to capture robotic states and estimate the probability of each timestep in the trajectory belonging to motion-transition phases. During policy learning, we apply fine-grained adjustment that reduces the magnitude of proprioception's gradient based on estimated probabilities, leading to robust and generalizable vision-proprioception policies. The comprehensive experiments demonstrate GAP is applicable in both simulated and real-world environments, across one-arm and dual-arm setups, and compatible with both conventional and Vision-Language-Action models. We believe this work can offer valuable insights into the development of vision-proprioception policies in robotic manipulation.

  </details>



- **Creative Ownership in the Age of AI**  
  Annie Liang, Jay Lu  
  _2026-02-12_ · https://arxiv.org/abs/2602.12270v1 · `econ.TH`  
  <details><summary>Abstract</summary>

  Copyright law focuses on whether a new work is "substantially similar" to an existing one, but generative AI can closely imitate style without copying content, a capability now central to ongoing litigation. We argue that existing definitions of infringement are ill-suited to this setting and propose a new criterion: a generative AI output infringes on an existing work if it could not have been generated without that work in its training corpus. To operationalize this definition, we model generative systems as closure operators mapping a corpus of existing works to an output of new works. AI generated outputs are \emph{permissible} if they do not infringe on any existing work according to our criterion. Our results characterize structural properties of permissible generation and reveal a sharp asymptotic dichotomy: when the process of organic creations is light-tailed, dependence on individual works eventually vanishes, so that regulation imposes no limits on AI generation; with heavy-tailed creations, regulation can be persistently constraining.

  </details>



- **ModelWisdom: An Integrated Toolkit for TLA+ Model Visualization, Digest and Repair**  
  Zhiyong Chen, Jialun Cao, Chang Xu, Shing-Chi Cheung  
  _2026-02-12_ · https://arxiv.org/abs/2602.12058v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Model checking in TLA+ provides strong correctness guarantees, yet practitioners continue to face significant challenges in interpreting counterexamples, understanding large state-transition graphs, and repairing faulty models. These difficulties stem from the limited explainability of raw model-checker output and the substantial manual effort required to trace violations back to source specifications. Although the TLA+ Toolbox includes a state diagram viewer, it offers only a static, fully expanded graph without folding, color highlighting, or semantic explanations, which limits its scalability and interpretability. We present ModelWisdom, an interactive environment that uses visualization and large language models to make TLA+ model checking more interpretable and actionable. ModelWisdom offers: (i) Model Visualization, with colorized violation highlighting, click-through links from transitions to TLA+ code, and mapping between violating states and broken properties; (ii) Graph Optimization, including tree-based structuring and node/edge folding to manage large models; (iii) Model Digest, which summarizes and explains subgraphs via large language models (LLMs) and performs preprocessing and partial explanations; and (iv) Model Repair, which extracts error information and supports iterative debugging. Together, these capabilities turn raw model-checker output into an interactive, explainable workflow, improving understanding and reducing debugging effort for nontrivial TLA+ specifications. The website to ModelWisdom is available: https://model-wisdom.pages.dev. A demonstrative video can be found at https://www.youtube.com/watch?v=plyZo30VShA.

  </details>



- **Decomposition of Spillover Effects Under Misspecification:Pseudo-true Estimands and a Local--Global Extension**  
  Yechan Park, Xiaodong Yang  
  _2026-02-12_ · https://arxiv.org/abs/2602.12023v1 · `econ.EM`  
  <details><summary>Abstract</summary>

  Applied work with interference typically models outcomes as functions of own treatment and a low-dimensional exposure mapping of others' treatments, even when that mapping may be misspecified. This raises a basic question: what policy object are exposure-based estimands implicitly targeting, and how should we interpret their direct and spillover components relative to the underlying policy question? We take as primitive the marginal policy effect, defined as the effect of a small change in the treatment probability under the actual experimental design, and show that any researcher-chosen exposure mapping induces a unique pseudo-true outcome model. This model is the best approximation to the underlying potential outcomes that depends only on the user-chosen exposure. Utilizing that representation, the marginal policy effect admits a canonical decomposition into exposure-based direct and spillover effects, and each component provides its optimal approximation to the corresponding oracle objects that would be available if interference were fully known. We then focus on a setting that nests important empirical and theoretical applications in which both local network spillovers and global spillovers, such as market equilibrium, operate. There, the marginal policy effect further decomposes asymptotically into direct, local, and global channels. An important implication is that many existing methods are more robust than previously understood once we reinterpret their targets as channel-specific components of this pseudo-true policy estimand. Simulations and a semi-synthetic experiment calibrated to a large cash-transfer experiment show that these components can be recovered in realistic experimental designs.

  </details>



- **Radio Map Prediction from Noisy Environment Information and Sparse Observations**  
  Fabian Jaensch, Çağkan Yapar, Giuseppe Caire, Begüm Demir  
  _2026-02-12_ · https://arxiv.org/abs/2602.11950v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Many works have investigated radio map and path loss prediction in wireless networks using deep learning, in particular using convolutional neural networks. However, most assume perfect environment information, which is unrealistic in practice due to sensor limitations, mapping errors, and temporal changes. We demonstrate that convolutional neural networks trained with task-specific perturbations of geometry, materials, and Tx positions can implicitly compensate for prediction errors caused by inaccurate environment inputs. When tested with noisy inputs on synthetic indoor scenarios, models trained with perturbed environment data reduce error by up to 25\% compared to models trained on clean data. We verify our approach on real-world measurements, achieving 2.1 dB RMSE with noisy input data and 1.3 dB with complete information, compared to 2.3-3.1 dB for classical methods such as ray-tracing and radial basis function interpolation. Additionally, we compare different ways of encoding environment information at varying levels of detail and we find that, in the considered single-room indoor scenarios, binary occupancy encoding performs at least as well as detailed material property information, simplifying practical deployment.

  </details>



- **DiffPlace: Street View Generation via Place-Controllable Diffusion Model Enhancing Place Recognition**  
  Ji Li, Zhiwei Li, Shihao Li, Zhenjiang Yu, Boyang Wang, Haiou Liu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11875v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Generative models have advanced significantly in realistic image synthesis, with diffusion models excelling in quality and stability. Recent multi-view diffusion models improve 3D-aware street view generation, but they struggle to produce place-aware and background-consistent urban scenes from text, BEV maps, and object bounding boxes. This limits their effectiveness in generating realistic samples for place recognition tasks. To address these challenges, we propose DiffPlace, a novel framework that introduces a place-ID controller to enable place-controllable multi-view image generation. The place-ID controller employs linear projection, perceiver transformer, and contrastive learning to map place-ID embeddings into a fixed CLIP space, allowing the model to synthesize images with consistent background buildings while flexibly modifying foreground objects and weather conditions. Extensive experiments, including quantitative comparisons and augmented training evaluations, demonstrate that DiffPlace outperforms existing methods in both generation quality and training support for visual place recognition. Our results highlight the potential of generative models in enhancing scene-level and place-aware synthesis, providing a valuable approach for improving place recognition in autonomous driving

  </details>



- **CAAL: Confidence-Aware Active Learning for Heteroscedastic Atmospheric Regression**  
  Fei Jiang, Jiyang Xia, Junjie Yu, Mingfei Sun, Hugh Coe, David Topping, Dantong Liu, Zhenhui Jessie Li, Zhonghua Zheng  
  _2026-02-12_ · https://arxiv.org/abs/2602.11825v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Quantifying the impacts of air pollution on health and climate relies on key atmospheric particle properties such as toxicity and hygroscopicity. However, these properties typically require complex observational techniques or expensive particle-resolved numerical simulations, limiting the availability of labeled data. We therefore estimate these hard-to-measure particle properties from routinely available observations (e.g., air pollutant concentrations and meteorological conditions). Because routine observations only indirectly reflect particle composition and structure, the mapping from routine observations to particle properties is noisy and input-dependent, yielding a heteroscedastic regression setting. With a limited and costly labeling budget, the central challenge is to select which samples to measure or simulate. While active learning is a natural approach, most acquisition strategies rely on predictive uncertainty. Under heteroscedastic noise, this signal conflates reducible epistemic uncertainty with irreducible aleatoric uncertainty, causing limited budgets to be wasted in noise-dominated regions. To address this challenge, we propose a confidence-aware active learning framework (CAAL) for efficient and robust sample selection in heteroscedastic settings. CAAL consists of two components: a decoupled uncertainty-aware training objective that separately optimises the predictive mean and noise level to stabilise uncertainty estimation, and a confidence-aware acquisition function that dynamically weights epistemic uncertainty using predicted aleatoric uncertainty as a reliability signal. Experiments on particle-resolved numerical simulations and real atmospheric observations show that CAAL consistently outperforms standard AL baselines. The proposed framework provides a practical and general solution for the efficient expansion of high-cost atmospheric particle property databases.

  </details>



- **SpaTeoGL: Spatiotemporal Graph Learning for Interpretable Seizure Onset Zone Analysis from Intracranial EEG**  
  Elham Rostami, Aref Einizade, Taous-Meriem Laleg-Kirati  
  _2026-02-12_ · https://arxiv.org/abs/2602.11801v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Accurate localization of the seizure onset zone (SOZ) from intracranial EEG (iEEG) is essential for epilepsy surgery but is challenged by complex spatiotemporal seizure dynamics. We propose SpaTeoGL, a spatiotemporal graph learning framework for interpretable seizure network analysis. SpaTeoGL jointly learns window-level spatial graphs capturing interactions among iEEG electrodes and a temporal graph linking time windows based on similarity of their spatial structure. The method is formulated within a smooth graph signal processing framework and solved via an alternating block coordinate descent algorithm with convergence guarantees. Experiments on a multicenter iEEG dataset with successful surgical outcomes show that SpaTeoGL is competitive with a baseline based on horizontal visibility graphs and logistic regression, while improving non-SOZ identification and providing interpretable insights into seizure onset and propagation dynamics.

  </details>



- **Safe Fairness Guarantees Without Demographics in Classification: Spectral Uncertainty Set Perspective**  
  Ainhize Barrainkua, Santiago Mazuelas, Novi Quadrianto, Jose A. Lozano  
  _2026-02-12_ · https://arxiv.org/abs/2602.11785v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  As automated classification systems become increasingly prevalent, concerns have emerged over their potential to reinforce and amplify existing societal biases. In the light of this issue, many methods have been proposed to enhance the fairness guarantees of classifiers. Most of the existing interventions assume access to group information for all instances, a requirement rarely met in practice. Fairness without access to demographic information has often been approached through robust optimization techniques,which target worst-case outcomes over a set of plausible distributions known as the uncertainty set. However, their effectiveness is strongly influenced by the chosen uncertainty set. In fact, existing approaches often overemphasize outliers or overly pessimistic scenarios, compromising both overall performance and fairness. To overcome these limitations, we introduce SPECTRE, a minimax-fair method that adjusts the spectrum of a simple Fourier feature mapping and constrains the extent to which the worst-case distribution can deviate from the empirical distribution. We perform extensive experiments on the American Community Survey datasets involving 20 states. The safeness of SPECTRE comes as it provides the highest average values on fairness guarantees together with the smallest interquartile range in comparison to state-of-the-art approaches, even compared to those with access to demographic group information. In addition, we provide a theoretical analysis that derives computable bounds on the worst-case error for both individual groups and the overall population, as well as characterizes the worst-case distributions responsible for these extremal performances

  </details>


