# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-05 07:13 UTC_

Total papers shown: **16**


---

- **Safe Urban Traffic Control via Uncertainty-Aware Conformal Prediction and World-Model Reinforcement Learning**  
  Joydeep Chandra, Satyam Kumar Navneet, Aleksandr Algazinov, Yong Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04821v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Urban traffic management demands systems that simultaneously predict future conditions, detect anomalies, and take safe corrective actions -- all while providing reliability guarantees. We present STREAM-RL, a unified framework that introduces three novel algorithmic contributions: (1) PU-GAT+, an Uncertainty-Guided Adaptive Conformal Forecaster that uses prediction uncertainty to dynamically reweight graph attention via confidence-monotonic attention, achieving distribution-free coverage guarantees; (2) CRFN-BY, a Conformal Residual Flow Network that models uncertainty-normalized residuals via normalizing flows with Benjamini-Yekutieli FDR control under arbitrary dependence; and (3) LyCon-WRL+, an Uncertainty-Guided Safe World-Model RL agent with Lyapunov stability certificates, certified Lipschitz bounds, and uncertainty-propagated imagination rollouts. To our knowledge, this is the first framework to propagate calibrated uncertainty from forecasting through anomaly detection to safe policy learning with end-to-end theoretical guarantees. Experiments on multiple real-world traffic trajectory data demonstrate that STREAM-RL achieves 91.4\% coverage efficiency, controls FDR at 4.1\% under verified dependence, and improves safety rate to 95.2\% compared to 69\% for standard PPO while achieving higher reward, with 23ms end-to-end inference latency.

  </details>



- **Safe-NEureka: a Hybrid Modular Redundant DNN Accelerator for On-board Satellite AI Processing**  
  Riccardo Tedeschi, Luigi Ghionda, Alessandro Nadalini, Yvan Tortorella, Arpan Suravi Prasad, Luca Benini, Davide Rossi, Francesco Conti  
  _2026-02-04_ · https://arxiv.org/abs/2602.04803v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Low Earth Orbit (LEO) constellations are revolutionizing the space sector, with on-board Artificial Intelligence (AI) becoming pivotal for next-generation satellites. AI acceleration is essential for safety-critical functions such as autonomous Guidance, Navigation, and Control (GNC), where errors cannot be tolerated, and performance-critical processing of high-bandwidth sensor data, where occasional errors are tolerable. Consequently, AI accelerators for satellites must combine robust protection against radiation-induced faults with high throughput. This paper presents Safe-NEureka, a Hybrid Modular Redundant Deep Neural Network (DNN) accelerator for heterogeneous RISC-V systems. It operates in two modes: a redundancy mode utilizing Dual Modular Redundancy (DMR) with hardware-based recovery, and a performance mode repurposing redundant datapaths to maximize parallel throughput. Furthermore, its memory interface is protected by Error Correction Codes (ECCs), and the controller by Triple Modular Redundancy (TMR). Implementation in GlobalFoundries 12nm technology shows a 96 reduction in faulty executions in redundancy mode, with a manageable 15 area overhead. In performance mode, the architecture achieves near-baseline speeds on 3x3 dense convolutions with a 5 throughput and 11 efficiency reduction, compared to 48 and 53 in redundancy mode. This flexibility ensures high overheads are limited to critical tasks, establishing Safe-NEureka as a versatile solution for space applications.

  </details>



- **Act, Sense, Act: Learning Non-Markovian Active Perception Strategies from Large-Scale Egocentric Human Data**  
  Jialiang Li, Yi Qiao, Yunhan Guo, Changwen Chen, Wenzhao Lian  
  _2026-02-04_ · https://arxiv.org/abs/2602.04600v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Achieving generalizable manipulation in unconstrained environments requires the robot to proactively resolve information uncertainty, i.e., the capability of active perception. However, existing methods are often confined in limited types of sensing behaviors, restricting their applicability to complex environments. In this work, we formalize active perception as a non-Markovian process driven by information gain and decision branching, providing a structured categorization of visual active perception paradigms. Building on this perspective, we introduce CoMe-VLA, a cognitive and memory-aware vision-language-action (VLA) framework that leverages large-scale human egocentric data to learn versatile exploration and manipulation priors. Our framework integrates a cognitive auxiliary head for autonomous sub-task transitions and a dual-track memory system to maintain consistent self and environmental awareness by fusing proprioceptive and visual temporal contexts. By aligning human and robot hand-eye coordination behaviors in a unified egocentric action space, we train the model progressively in three stages. Extensive experiments on a wheel-based humanoid have demonstrated strong robustness and adaptability of our proposed method across diverse long-horizon tasks spanning multiple active perception scenarios.

  </details>



- **Active Asymmetric Multi-Agent Multimodal Learning under Uncertainty**  
  Rui Liu, Pratap Tokekar, Ming Lin  
  _2026-02-04_ · https://arxiv.org/abs/2602.04763v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Multi-agent systems are increasingly equipped with heterogeneous multimodal sensors, enabling richer perception but introducing modality-specific and agent-dependent uncertainty. Existing multi-agent collaboration frameworks typically reason at the agent level, assume homogeneous sensing, and handle uncertainty implicitly, limiting robustness under sensor corruption. We propose Active Asymmetric Multi-Agent Multimodal Learning under Uncertainty (A2MAML), a principled approach for uncertainty-aware, modality-level collaboration. A2MAML models each modality-specific feature as a stochastic estimate with uncertainty prediction, actively selects reliable agent-modality pairs, and aggregates information via Bayesian inverse-variance weighting. This formulation enables fine-grained, modality-level fusion, supports asymmetric modality availability, and provides a principled mechanism to suppress corrupted or noisy modalities. Extensive experiments on connected autonomous driving scenarios for collaborative accident detection demonstrate that A2MAML consistently outperforms both single-agent and collaborative baselines, achieving up to 18.7% higher accident detection rate.

  </details>



- **From Vision to Assistance: Gaze and Vision-Enabled Adaptive Control for a Back-Support Exoskeleton**  
  Alessandro Leanza, Paolo Franceschi, Blerina Spahiu, Loris Roveda  
  _2026-02-04_ · https://arxiv.org/abs/2602.04648v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Back-support exoskeletons have been proposed to mitigate spinal loading in industrial handling, yet their effectiveness critically depends on timely and context-aware assistance. Most existing approaches rely either on load-estimation techniques (e.g., EMG, IMU) or on vision systems that do not directly inform control. In this work, we present a vision-gated control framework for an active lumbar occupational exoskeleton that leverages egocentric vision with wearable gaze tracking. The proposed system integrates real-time grasp detection from a first-person YOLO-based perception system, a finite-state machine (FSM) for task progression, and a variable admittance controller to adapt torque delivery to both posture and object state. A user study with 15 participants performing stooping load lifting trials under three conditions (no exoskeleton, exoskeleton without vision, exoskeleton with vision) shows that vision-gated assistance significantly reduces perceived physical demand and improves fluency, trust, and comfort. Quantitative analysis reveals earlier and stronger assistance when vision is enabled, while questionnaire results confirm user preference for the vision-gated mode. These findings highlight the potential of egocentric vision to enhance the responsiveness, ergonomics, safety, and acceptance of back-support exoskeletons.

  </details>



- **Probabilistic Label Spreading: Efficient and Consistent Estimation of Soft Labels with Epistemic Uncertainty on Graphs**  
  Jonathan Klees, Tobias Riedlinger, Peter Stehr, Bennet Böddecker, Daniel Kondermann, Matthias Rottmann  
  _2026-02-04_ · https://arxiv.org/abs/2602.04574v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Safe artificial intelligence for perception tasks remains a major challenge, partly due to the lack of data with high-quality labels. Annotations themselves are subject to aleatoric and epistemic uncertainty, which is typically ignored during annotation and evaluation. While crowdsourcing enables collecting multiple annotations per image to estimate these uncertainties, this approach is impractical at scale due to the required annotation effort. We introduce a probabilistic label spreading method that provides reliable estimates of aleatoric and epistemic uncertainty of labels. Assuming label smoothness over the feature space, we propagate single annotations using a graph-based diffusion method. We prove that label spreading yields consistent probability estimators even when the number of annotations per data point converges to zero. We present and analyze a scalable implementation of our method. Experimental results indicate that, compared to baselines, our approach substantially reduces the annotation budget required to achieve a desired label quality on common image datasets and achieves a new state of the art on the Data-Centric Image Classification benchmark.

  </details>



- **Anytime-Valid Conformal Risk Control**  
  Bror Hultberg, Dave Zachariah, Antônio H. Ribeiro  
  _2026-02-04_ · https://arxiv.org/abs/2602.04364v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  Prediction sets provide a means of quantifying the uncertainty in predictive tasks. Using held out calibration data, conformal prediction and risk control can produce prediction sets that exhibit statistically valid error control in a computationally efficient manner. However, in the standard formulations, the error is only controlled on average over many possible calibration datasets of fixed size. In this paper, we extend the control to remain valid with high probability over a cumulatively growing calibration dataset at any time point. We derive such guarantees using quantile-based arguments and illustrate the applicability of the proposed framework to settings involving distribution shift. We further establish a matching lower bound and show that our guarantees are asymptotically tight. Finally, we demonstrate the practical performance of our methods through both simulations and real-world numerical examples.

  </details>



- **Beyond the Control Equations: An Artifact Study of Implementation Quality in Robot Control Software**  
  Nils Chur, Thorsten Berger, Einar Broch Johnsen, Andrzej Wąsowski  
  _2026-02-04_ · https://arxiv.org/abs/2602.04799v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  A controller -- a software module managing hardware behavior -- is a key component of a typical robot system. While control theory gives safety guarantees for standard controller designs, the practical implementation of controllers in software introduces complexities that are often overlooked. Controllers are often designed in continuous space, while the software is executed in discrete space, undermining some of the theoretical guarantees. Despite extensive research on control theory and control modeling, little attention has been paid to the implementations of controllers and how their theoretical guarantees are ensured in real-world software systems. We investigate 184 real-world controller implementations in open-source robot software. We examine their application context, the implementation characteristics, and the testing methods employed to ensure correctness. We find that the implementations often handle discretization in an ad hoc manner, leading to potential issues with real-time reliability. Challenges such as timing inconsistencies, lack of proper error handling, and inadequate consideration of real-time constraints further complicate matters. Testing practices are superficial, no systematic verification of theoretical guarantees is used, leaving possible inconsistencies between expected and actual behavior. Our findings highlight the need for improved implementation guidelines and rigorous verification techniques to ensure the reliability and safety of robotic controllers in practice.

  </details>



- **Stochastic Decision Horizons for Constrained Reinforcement Learning**  
  Nikola Milosevic, Leonard Franz, Daniel Haeufle, Georg Martius, Nico Scherf, Pavel Kolev  
  _2026-02-04_ · https://arxiv.org/abs/2602.04599v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Constrained Markov decision processes (CMDPs) provide a principled model for handling constraints, such as safety and other auxiliary objectives, in reinforcement learning. The common approach of using additive-cost constraints and dual variables often hinders off-policy scalability. We propose a Control as Inference formulation based on stochastic decision horizons, where constraint violations attenuate reward contributions and shorten the effective planning horizon via state-action-dependent continuation. This yields survival-weighted objectives that remain replay-compatible for off-policy actor-critic learning. We propose two violation semantics, absorbing and virtual termination, that share the same survival-weighted return but result in distinct optimization structures that lead to SAC/MPO-style policy improvement. Experiments demonstrate improved sample efficiency and favorable return-violation trade-offs on standard benchmarks. Moreover, MPO with virtual termination (VT-MPO) scales effectively to our high-dimensional musculoskeletal Hyfydy setup.

  </details>



- **Agentic AI in Healthcare & Medicine: A Seven-Dimensional Taxonomy for Empirical Evaluation of LLM-based Agents**  
  Shubham Vatsal, Harsh Dubey, Aditi Singh  
  _2026-02-04_ · https://arxiv.org/abs/2602.04813v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Model (LLM)-based agents that plan, use tools and act has begun to shape healthcare and medicine. Reported studies demonstrate competence on various tasks ranging from EHR analysis and differential diagnosis to treatment planning and research workflows. Yet the literature largely consists of overviews which are either broad surveys or narrow dives into a single capability (e.g., memory, planning, reasoning), leaving healthcare work without a common frame. We address this by reviewing 49 studies using a seven-dimensional taxonomy: Cognitive Capabilities, Knowledge Management, Interaction Patterns, Adaptation & Learning, Safety & Ethics, Framework Typology and Core Tasks & Subtasks with 29 operational sub-dimensions. Using explicit inclusion and exclusion criteria and a labeling rubric (Fully Implemented, Partially Implemented, Not Implemented), we map each study to the taxonomy and report quantitative summaries of capability prevalence and co-occurrence patterns. Our empirical analysis surfaces clear asymmetries. For instance, the External Knowledge Integration sub-dimension under Knowledge Management is commonly realized (~76% Fully Implemented) whereas Event-Triggered Activation sub-dimenison under Interaction Patterns is largely absent (~92% Not Implemented) and Drift Detection & Mitigation sub-dimension under Adaptation & Learning is rare (~98% Not Implemented). Architecturally, Multi-Agent Design sub-dimension under Framework Typology is the dominant pattern (~82% Fully Implemented) while orchestration layers remain mostly partial. Across Core Tasks & Subtasks, information centric capabilities lead e.g., Medical Question Answering & Decision Support and Benchmarking & Simulation, while action and discovery oriented areas such as Treatment Planning & Prescription still show substantial gaps (~59% Not Implemented).

  </details>



- **AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation**  
  Jin-Chuan Shi, Binhong Ye, Tao Liu, Junzhe He, Yangjinhui Xu, Xiaoyang Liu, Zeju Li, Hao Chen, Chunhua Shen  
  _2026-02-04_ · https://arxiv.org/abs/2602.04672v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior art frequently collapses. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.

  </details>



- **SAFE: Stable Alignment Finetuning with Entropy-Aware Predictive Control for RLHF**  
  Dipan Maity  
  _2026-02-04_ · https://arxiv.org/abs/2602.04651v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Optimization (PPO) has been positioned by recent literature as the canonical method for the RL part of RLHF. PPO performs well empirically but has a heuristic motivation and handles the KL-divergence constraint used in LM-RLHF in an ad-hoc manner and suffers form reward oscillations, entropy collapse, value function drift, and sudden policy divergence that require frequent restarts and extensive hyperparameter tuning. In this paper, we develop a new pure on policy actor-critic RL method for the LM-RLHF setting. We present SAFE (Stable Alignment Finetuning with Entropy-aware control),a novel RLHF algorithm that combines a Double Soft-Min Critic for pessimistic value estimation with a new multi-layer stabilization framework combining entropy-gated KL regulation, and PID-controlled adaptive thresholds. Unlike standard PPO's symmetric KL penalties, SAFE distinguishes high-entropy exploration from low-entropy mode collapse and adjusts penalties dynamically based on reward velocity. Experiments on a 3B parameter model show SAFE achieves +5.15\% training-average reward than PPO (0.725 vs 0.689), negligible reward crashes, and superior KL control than ppo . Our method adds minimal computational overhead and provides an interpretable, crash-resistant RLHF framework that maintains aggressive learning speed while ensuring stable long-horizon optimization suitable for production deployment. Code is available at https://github.com/ryyzn9/SAFE

  </details>



- **PEPR: Privileged Event-based Predictive Regularization for Domain Generalization**  
  Gabriele Magrini, Federico Becattini, Niccolò Biondi, Pietro Pala  
  _2026-02-04_ · https://arxiv.org/abs/2602.04583v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Deep neural networks for visual perception are highly susceptible to domain shift, which poses a critical challenge for real-world deployment under conditions that differ from the training data. To address this domain generalization challenge, we propose a cross-modal framework under the learning using privileged information (LUPI) paradigm for training a robust, single-modality RGB model. We leverage event cameras as a source of privileged information, available only during training. The two modalities exhibit complementary characteristics: the RGB stream is semantically dense but domain-dependent, whereas the event stream is sparse yet more domain-invariant. Direct feature alignment between them is therefore suboptimal, as it forces the RGB encoder to mimic the sparse event representation, thereby losing semantic detail. To overcome this, we introduce Privileged Event-based Predictive Regularization (PEPR), which reframes LUPI as a predictive problem in a shared latent space. Instead of enforcing direct cross-modal alignment, we train the RGB encoder with PEPR to predict event-based latent features, distilling robustness without sacrificing semantic richness. The resulting standalone RGB model consistently improves robustness to day-to-night and other domain shifts, outperforming alignment-based baselines across object detection and semantic segmentation.

  </details>



- **SLUM-i: Semi-supervised Learning for Urban Mapping of Informal Settlements and Data Quality Benchmarking**  
  Muhammad Taha Mukhtar, Syed Musa Ali Kazmi, Khola Naseem, Muhammad Ali Chattha, Andreas Dengel, Sheraz Ahmed, Muhammad Naseer Bajwa, Muhammad Imran Malik  
  _2026-02-04_ · https://arxiv.org/abs/2602.04525v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Rapid urban expansion has fueled the growth of informal settlements in major cities of low- and middle-income countries, with Lahore and Karachi in Pakistan and Mumbai in India serving as prominent examples. However, large-scale mapping of these settlements is severely constrained not only by the scarcity of annotations but by inherent data quality challenges, specifically high spectral ambiguity between formal and informal structures and significant annotation noise. We address this by introducing a benchmark dataset for Lahore, constructed from scratch, along with companion datasets for Karachi and Mumbai, which were derived from verified administrative boundaries, totaling 1,869 $\text{km}^2$ of area. To evaluate the global robustness of our framework, we extend our experiments to five additional established benchmarks, encompassing eight cities across three continents, and provide comprehensive data quality assessments of all datasets. We also propose a new semi-supervised segmentation framework designed to mitigate the class imbalance and feature degradation inherent in standard semi-supervised learning pipelines. Our method integrates a Class-Aware Adaptive Thresholding mechanism that dynamically adjusts confidence thresholds to prevent minority class suppression and a Prototype Bank System that enforces semantic consistency by anchoring predictions to historically learned high-fidelity feature representations. Extensive experiments across a total of eight cities spanning three continents demonstrate that our approach outperforms state-of-the-art semi-supervised baselines. Most notably, our method demonstrates superior domain transfer capability whereby a model trained on only 10% of source labels reaches a 0.461 mIoU on unseen geographies and outperforms the zero-shot generalization of fully supervised models.

  </details>



- **Robot-Assisted Group Tours for Blind People**  
  Yaxin Hu, Masaki Kuribayashi, Allan Wang, Seita Kayukawa, Daisuke Sato, Bilge Mutlu, Hironobu Takagi, Chieko Asakawa  
  _2026-02-04_ · https://arxiv.org/abs/2602.04458v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Group interactions are essential to social functioning, yet effective engagement relies on the ability to recognize and interpret visual cues, making such engagement a significant challenge for blind people. In this paper, we investigate how a mobile robot can support group interactions for blind people. We used the scenario of a guided tour with mixed-visual groups involving blind and sighted visitors. Based on insights from an interview study with blind people (n=5) and museum experts (n=5), we designed and prototyped a robotic system that supported blind visitors to join group tours. We conducted a field study in a science museum where each blind participant (n=8) joined a group tour with one guide and two sighted participants (n=8). Findings indicated users' sense of safety from the robot's navigational support, concerns in the group participation, and preferences for obtaining environmental information. We present design implications for future robotic systems to support blind people's mixed-visual group participation.

  </details>



- **HoRD: Robust Humanoid Control via History-Conditioned Reinforcement Learning and Online Distillation**  
  Puyue Wang, Jiawei Hu, Yan Gao, Junyan Wang, Yu Zhang, Gillian Dobbie, Tao Gu, Wafa Johal, Ting Dang, Hong Jia  
  _2026-02-04_ · https://arxiv.org/abs/2602.04412v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Humanoid robots can suffer significant performance drops under small changes in dynamics, task specifications, or environment setup. We propose HoRD, a two-stage learning framework for robust humanoid control under domain shift. First, we train a high-performance teacher policy via history-conditioned reinforcement learning, where the policy infers latent dynamics context from recent state--action trajectories to adapt online to diverse randomized dynamics. Second, we perform online distillation to transfer the teacher's robust control capabilities into a transformer-based student policy that operates on sparse root-relative 3D joint keypoint trajectories. By combining history-conditioned adaptation with online distillation, HoRD enables a single policy to adapt zero-shot to unseen domains without per-domain retraining. Extensive experiments show HoRD outperforms strong baselines in robustness and transfer, especially under unseen domains and external perturbations. Code and project page are available at \href{https://tonywang-0517.github.io/hord/}{https://tonywang-0517.github.io/hord/}.

  </details>


