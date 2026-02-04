# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-04 07:06 UTC_

Total papers shown: **15**


---

- **Conformal Reachability for Safe Control in Unknown Environments**  
  Xinhang Ma, Junlin Wu, Yiannis Kantaros, Yevgeniy Vorobeychik  
  _2026-02-03_ · https://arxiv.org/abs/2602.03799v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Designing provably safe control is a core problem in trustworthy autonomy. However, most prior work in this regard assumes either that the system dynamics are known or deterministic, or that the state and action space are finite, significantly limiting application scope. We address this limitation by developing a probabilistic verification framework for unknown dynamical systems which combines conformal prediction with reachability analysis. In particular, we use conformal prediction to obtain valid uncertainty intervals for the unknown dynamics at each time step, with reachability then verifying whether safety is maintained within the conformal uncertainty bounds. Next, we develop an algorithmic approach for training control policies that optimize nominal reward while also maximizing the planning horizon with sound probabilistic safety guarantees. We evaluate the proposed approach in seven safe control settings spanning four domains -- cartpole, lane following, drone control, and safe navigation -- for both affine and nonlinear safety specifications. Our experiments show that the policies we learn achieve the strongest provable safety guarantees while still maintaining high average reward.

  </details>



- **SAGE-5GC: Security-Aware Guidelines for Evaluating Anomaly Detection in the 5G Core Network**  
  Cristian Manca, Christian Scano, Giorgio Piras, Fabio Brau, Maura Pintor, Battista Biggio  
  _2026-02-03_ · https://arxiv.org/abs/2602.03596v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Machine learning-based anomaly detection systems are increasingly being adopted in 5G Core networks to monitor complex, high-volume traffic. However, most existing approaches are evaluated under strong assumptions that rarely hold in operational environments, notably the availability of independent and identically distributed (IID) data and the absence of adaptive attackers.In this work, we study the problem of detecting 5G attacks \textit{in the wild}, focusing on realistic deployment settings. We propose a set of Security-Aware Guidelines for Evaluating anomaly detectors in 5G Core Network (SAGE-5GC), driven by domain knowledge and consideration of potential adversarial threats. Using a realistic 5G Core dataset, we first train several anomaly detectors and assess their baseline performance against standard 5GC control-plane cyberattacks targeting PFCP-based network services.We then extend the evaluation to adversarial settings, where an attacker tries to manipulate the observable features of the network traffic to evade detection, under the constraint that the intended functionality of the malicious traffic is preserved. Starting from a selected set of controllable features, we analyze model sensitivity and adversarial robustness through randomized perturbations. Finally, we introduce a practical optimization strategy based on genetic algorithms that operates exclusively on attacker-controllable features and does not require prior knowledge of the underlying detection model. Our experimental results show that adversarially crafted attacks can substantially degrade detection performance, underscoring the need for robust, security-aware evaluation methodologies for anomaly detection in 5G networks deployed in the wild.

  </details>



- **Input-to-State Safe Backstepping: Robust Safety-Critical Control with Unmatched Uncertainties**  
  Max H. Cohen, Pio Ong, Aaron D. Ames  
  _2026-02-03_ · https://arxiv.org/abs/2602.03691v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Guaranteeing safety in the presence of unmatched disturbances -- uncertainties that cannot be directly canceled by the control input -- remains a key challenge in nonlinear control. This paper presents a constructive approach to safety-critical control of nonlinear systems with unmatched disturbances. We first present a generalization of the input-to-state safety (ISSf) framework for systems with these uncertainties using the recently developed notion of an Optimal Decay CBF, which provides more flexibility for satisfying the associated Lyapunov-like conditions for safety. From there, we outline a procedure for constructing ISSf-CBFs for two relevant classes of systems with unmatched uncertainties: i) strict-feedback systems; ii) dual-relative-degree systems, which are similar to differentially flat systems. Our theoretical results are illustrated via numerical simulations of an inverted pendulum and planar quadrotor.

  </details>



- **TIPS Over Tricks: Simple Prompts for Effective Zero-shot Anomaly Detection**  
  Alireza Salehi, Ehsan Karami, Sepehr Noey, Sahand Noey, Makoto Yamada, Reshad Hosseini, Mohammad Sabokrou  
  _2026-02-03_ · https://arxiv.org/abs/2602.03594v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Anomaly detection identifies departures from expected behavior in safety-critical settings. When target-domain normal data are unavailable, zero-shot anomaly detection (ZSAD) leverages vision-language models (VLMs). However, CLIP's coarse image-text alignment limits both localization and detection due to (i) spatial misalignment and (ii) weak sensitivity to fine-grained anomalies; prior work compensates with complex auxiliary modules yet largely overlooks the choice of backbone. We revisit the backbone and use TIPS-a VLM trained with spatially aware objectives. While TIPS alleviates CLIP's issues, it exposes a distributional gap between global and local features. We address this with decoupled prompts-fixed for image-level detection and learnable for pixel-level localization-and by injecting local evidence into the global score. Without CLIP-specific tricks, our TIPS-based pipeline improves image-level performance by 1.1-3.9% and pixel-level by 1.5-6.9% across seven industrial datasets, delivering strong generalization with a lean architecture. Code is available at github.com/AlirezaSalehy/Tipsomaly.

  </details>



- **Self-supervised Physics-Informed Manipulation of Deformable Linear Objects with Non-negligible Dynamics**  
  Youyuan Long, Gokhan Solak, Sara Zeynalpour, Heng Zhang, Arash Ajoudani  
  _2026-02-03_ · https://arxiv.org/abs/2602.03623v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We address dynamic manipulation of deformable linear objects by presenting SPiD, a physics-informed self-supervised learning framework that couples an accurate deformable object model with an augmented self-supervised training strategy. On the modeling side, we extend a mass-spring model to more accurately capture object dynamics while remaining lightweight enough for high-throughput rollouts during self-supervised learning. On the learning side, we train a neural controller using a task-oriented cost, enabling end-to-end optimization through interaction with the differentiable object model. In addition, we propose a self-supervised DAgger variant that detects distribution shift during deployment and performs offline self-correction to further enhance robustness without expert supervision. We evaluate our method primarily on the rope stabilization task, where a robot must bring a swinging rope to rest as quickly and smoothly as possible. Extensive experiments in both simulation and the real world demonstrate that the proposed controller achieves fast and smooth rope stabilization, generalizing across unseen initial states, rope lengths, masses, non-uniform mass distributions, and external disturbances. Additionally, we develop an affordable markerless rope perception method and demonstrate that our controller maintains performance with noisy and low-frequency state updates. Furthermore, we demonstrate the generality of the framework by extending it to the rope trajectory tracking task. Overall, SPiD offers a data-efficient, robust, and physically grounded framework for dynamic manipulation of deformable linear objects, featuring strong sim-to-real generalization.

  </details>



- **Formal Evidence Generation for Assurance Cases for Robotic Software Models**  
  Fang Yan, Simon Foster, Ana Cavalcanti, Ibrahim Habli, James Baxter  
  _2026-02-03_ · https://arxiv.org/abs/2602.03550v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Robotics and Autonomous Systems are increasingly deployed in safety-critical domains, so that demonstrating their safety is essential. Assurance Cases (ACs) provide structured arguments supported by evidence, but generating and maintaining this evidence is labour-intensive, error-prone, and difficult to keep consistent as systems evolve. We present a model-based approach to systematically generating AC evidence by embedding formal verification into the assurance workflow. The approach addresses three challenges: systematically deriving formal assertions from natural language requirements using templates, orchestrating multiple formal verification tools to handle diverse property types, and integrating formal evidence production into the workflow. Leveraging RoboChart, a domain-specific modelling language with formal semantics, we combine model checking and theorem proving in our approach. Structured requirements are automatically transformed into formal assertions using predefined templates, and verification results are automatically integrated as evidence. Case studies demonstrate the effectiveness of our approach.

  </details>



- **CMR: Contractive Mapping Embeddings for Robust Humanoid Locomotion on Unstructured Terrains**  
  Qixin Zeng, Hongyin Zhang, Shangke Lyu, Junxi Jin, Donglin Wang, Chao Huang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03511v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robust disturbance rejection remains a longstanding challenge in humanoid locomotion, particularly on unstructured terrains where sensing is unreliable and model mismatch is pronounced. While perception information, such as height map, enhances terrain awareness, sensor noise and sim-to-real gaps can destabilize policies in practice. In this work, we provide theoretical analysis that bounds the return gap under observation noise, when the induced latent dynamics are contractive. Furthermore, we present Contractive Mapping for Robustness (CMR) framework that maps high-dimensional, disturbance-prone observations into a latent space, where local perturbations are attenuated over time. Specifically, this approach couples contrastive representation learning with Lipschitz regularization to preserve task-relevant geometry while explicitly controlling sensitivity. Notably, the formulation can be incorporated into modern deep reinforcement learning pipelines as an auxiliary loss term with minimal additional technical effort required. Further, our extensive humanoid experiments show that CMR potently outperforms other locomotion algorithms under increased noise.

  </details>



- **Edge-Optimized Vision-Language Models for Underground Infrastructure Assessment**  
  Johny J. Lopez, Md Meftahul Ferdaus, Mahdi Abdelguerfi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03742v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous inspection of underground infrastructure, such as sewer and culvert systems, is critical to public safety and urban sustainability. Although robotic platforms equipped with visual sensors can efficiently detect structural deficiencies, the automated generation of human-readable summaries from these detections remains a significant challenge, especially on resource-constrained edge devices. This paper presents a novel two-stage pipeline for end-to-end summarization of underground deficiencies, combining our lightweight RAPID-SCAN segmentation model with a fine-tuned Vision-Language Model (VLM) deployed on an edge computing platform. The first stage employs RAPID-SCAN (Resource-Aware Pipeline Inspection and Defect Segmentation using Compact Adaptive Network), achieving 0.834 F1-score with only 0.64M parameters for efficient defect segmentation. The second stage utilizes a fine-tuned Phi-3.5 VLM that generates concise, domain-specific summaries in natural language from the segmentation outputs. We introduce a curated dataset of inspection images with manually verified descriptions for VLM fine-tuning and evaluation. To enable real-time performance, we employ post-training quantization with hardware-specific optimization, achieving significant reductions in model size and inference latency without compromising summarization quality. We deploy and evaluate our complete pipeline on a mobile robotic platform, demonstrating its effectiveness in real-world inspection scenarios. Our results show the potential of edge-deployable integrated AI systems to bridge the gap between automated defect detection and actionable insights for infrastructure maintenance, paving the way for more scalable and autonomous inspection solutions.

  </details>



- **Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems**  
  Haibo Jin, Kuang Peng, Ye Yu, Xiaopeng Yuan, Haohan Wang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03695v1 · `cs.MA`  
  <details><summary>Abstract</summary>

  While existing multi-agent systems (MAS) can handle complex problems by enabling collaboration among multiple agents, they are often highly task-specific, relying on manually crafted agent roles and interaction prompts, which leads to increased architectural complexity and limited reusability across tasks. Moreover, most MAS communicate primarily through natural language, making them vulnerable to error accumulation and instability in long-context, multi-stage interactions within internal agent histories. In this work, we propose \textbf{Agent Primitives}, a set of reusable latent building blocks for LLM-based MAS. Inspired by neural network design, where complex models are built from reusable components, we observe that many existing MAS architectures can be decomposed into a small number of recurring internal computation patterns. Based on this observation, we instantiate three primitives: Review, Voting and Selection, and Planning and Execution. All primitives communicate internally via key-value (KV) cache, which improves both robustness and efficiency by mitigating information degradation across multi-stage interactions. To enable automatic system construction, an Organizer agent selects and composes primitives for each query, guided by a lightweight knowledge pool of previously successful configurations, forming a primitive-based MAS. Experiments show that primitives-based MAS improve average accuracy by 12.0-16.5\% over single-agent baselines, reduce token usage and inference latency by approximately 3$\times$-4$\times$ compared to text-based MAS, while incurring only 1.3$\times$-1.6$\times$ overhead relative to single-agent inference and providing more stable performance across model backbones.

  </details>



- **Efficient Sequential Neural Network with Spatial-Temporal Attention and Linear LSTM for Robust Lane Detection Using Multi-Frame Images**  
  Sandeep Patil, Yongqi Dong, Haneen Farah, Hans Hellendoorn  
  _2026-02-03_ · https://arxiv.org/abs/2602.03669v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Lane detection is a crucial perception task for all levels of automated vehicles (AVs) and Advanced Driver Assistance Systems, particularly in mixed-traffic environments where AVs must interact with human-driven vehicles (HDVs) and challenging traffic scenarios. Current methods lack versatility in delivering accurate, robust, and real-time compatible lane detection, especially vision-based methods often neglect critical regions of the image and their spatial-temporal (ST) salience, leading to poor performance in difficult circumstances such as serious occlusion and dazzle lighting. This study introduces a novel sequential neural network model with a spatial-temporal attention mechanism to focus on key features of lane lines and exploit salient ST correlations among continuous image frames. The proposed model, built on a standard encoder-decoder structure and common neural network backbones, is trained and evaluated on three large-scale open-source datasets. Extensive experiments demonstrate the strength and robustness of the proposed model, outperforming state-of-the-art methods in various testing scenarios. Furthermore, with the ST attention mechanism, the developed sequential neural network models exhibit fewer parameters and reduced Multiply-Accumulate Operations (MACs) compared to baseline sequential models, highlighting their computational efficiency. Relevant data, code, and models are released at https://doi.org/10.4121/4619cab6-ae4a-40d5-af77-582a77f3d821.

  </details>



- **MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction**  
  Jung Min Lee, Dohyeok Lee, Seokhun Ju, Taehyun Cho, Jin Woo Koo, Li Zhao, Sangwoo Hong, Jungwoo Lee  
  _2026-02-03_ · https://arxiv.org/abs/2602.03668v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning \emph{latent actions} from diverse human videos enables scaling robot learning beyond embodiment-specific robot datasets, and these latent actions have recently been used as pseudo-action labels for vision-language-action (VLA) model pretraining. To make VLA pretraining effective, latent actions should contain information about the underlying agent's actions despite the absence of ground-truth labels. We propose \textbf{M}ulti-\textbf{V}iew\textbf{P}oint \textbf{L}atent \textbf{A}ction \textbf{M}odel (\textbf{MVP-LAM}), which learns discrete latent actions that are highly informative about ground-truth actions from time-synchronized multi-view videos. MVP-LAM trains latent actions with a \emph{cross-viewpoint reconstruction} objective, so that a latent action inferred from one view must explain the future in another view, reducing reliance on viewpoint-specific cues. On Bridge V2, MVP-LAM produces more action-centric latent actions, achieving higher mutual information with ground-truth actions and improved action prediction, including under out-of-distribution evaluation. Finally, pretraining VLAs with MVP-LAM latent actions improves downstream manipulation performance on the SIMPLER and LIBERO-Long benchmarks.

  </details>



- **Inlier-Centric Post-Training Quantization for Object Detection Models**  
  Minsu Kim, Dongyeun Lee, Jaemyung Yu, Jiwan Hur, Giseop Kim, Junmo Kim  
  _2026-02-03_ · https://arxiv.org/abs/2602.03472v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Object detection is pivotal in computer vision, yet its immense computational demands make deployment slow and power-hungry, motivating quantization. However, task-irrelevant morphologies such as background clutter and sensor noise induce redundant activations (or anomalies). These anomalies expand activation ranges and skew activation distributions toward task-irrelevant responses, complicating bit allocation and weakening the preservation of informative features. Without a clear criterion to distinguish anomalies, suppressing them can inadvertently discard useful information. To address this, we present InlierQ, an inlier-centric post-training quantization approach that separates anomalies from informative inliers. InlierQ computes gradient-aware volume saliency scores, classifies each volume as an inlier or anomaly, and fits a posterior distribution over these scores using the Expectation-Maximization (EM) algorithm. This design suppresses anomalies while preserving informative features. InlierQ is label-free, drop-in, and requires only 64 calibration samples. Experiments on the COCO and nuScenes benchmarks show consistent reductions in quantization error for camera-based (2D and 3D) and LiDAR-based (3D) object detection.

  </details>



- **Soft-Radial Projection for Constrained End-to-End Learning**  
  Philipp J. Schneider, Daniel Kuhn  
  _2026-02-03_ · https://arxiv.org/abs/2602.03461v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Integrating hard constraints into deep learning is essential for safety-critical systems. Yet existing constructive layers that project predictions onto constraint boundaries face a fundamental bottleneck: gradient saturation. By collapsing exterior points onto lower-dimensional surfaces, standard orthogonal projections induce rank-deficient Jacobians, which nullify gradients orthogonal to active constraints and hinder optimization. We introduce Soft-Radial Projection, a differentiable reparameterization layer that circumvents this issue through a radial mapping from Euclidean space into the interior of the feasible set. This construction guarantees strict feasibility while preserving a full-rank Jacobian almost everywhere, thereby preventing the optimization stalls typical of boundary-based methods. We theoretically prove that the architecture retains the universal approximation property and empirically show improved convergence behavior and solution quality over state-of-the-art optimization- and projection-based baselines.

  </details>



- **Causal Inference on Networks under Misspecified Exposure Mappings: A Partial Identification Framework**  
  Maresa Schröder, Miruna Oprescu, Stefan Feuerriegel, Nathan Kallus  
  _2026-02-03_ · https://arxiv.org/abs/2602.03459v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Estimating treatment effects in networks is challenging, as each potential outcome depends on the treatments of all other nodes in the network. To overcome this difficulty, existing methods typically impose an exposure mapping that compresses the treatment assignments in the network into a low-dimensional summary. However, if this mapping is misspecified, standard estimators for direct and spillover effects can be severely biased. We propose a novel partial identification framework for causal inference on networks to assess the robustness of treatment effects under misspecifications of the exposure mapping. Specifically, we derive sharp upper and lower bounds on direct and spillover effects under such misspecifications. As such, our framework presents a novel application of causal sensitivity analysis to exposure mappings. We instantiate our framework for three canonical exposure settings widely used in practice: (i) weighted means of the neighborhood treatments, (ii) threshold-based exposure mappings, and (iii) truncated neighborhood interference in the presence of higher-order spillovers. Furthermore, we develop orthogonal estimators for these bounds and prove that the resulting bound estimates are valid, sharp, and efficient. Our experiments show the bounds remain informative and provide reliable conclusions under misspecification of exposure mappings.

  </details>



- **Model-based Optimal Control for Rigid-Soft Underactuated Systems**  
  Daniele Caradonna, Nikhil Nair, Anup Teejo Mathew, Daniel Feliu Talegón, Imran Afgan, Egidio Falotico, Cosimo Della Santina, Federico Renda  
  _2026-02-03_ · https://arxiv.org/abs/2602.03435v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Continuum soft robots are inherently underactuated and subject to intrinsic input constraints, making dynamic control particularly challenging, especially in hybrid rigid-soft robots. While most existing methods focus on quasi-static behaviors, dynamic tasks such as swing-up require accurate exploitation of continuum dynamics. This has led to studies on simple low-order template systems that often fail to capture the complexity of real continuum deformations. Model-based optimal control offers a systematic solution; however, its application to rigid-soft robots is often limited by the computational cost and inaccuracy of numerical differentiation for high-dimensional models. Building on recent advances in the Geometric Variable Strain model that enable analytical derivatives, this work investigates three optimal control strategies for underactuated soft systems-Direct Collocation, Differential Dynamic Programming, and Nonlinear Model Predictive Control-to perform dynamic swing-up tasks. To address stiff continuum dynamics and constrained actuation, implicit integration schemes and warm-start strategies are employed to improve numerical robustness and computational efficiency. The methods are evaluated in simulation on three Rigid-Soft and high-order soft benchmark systems-the Soft Cart-Pole, the Soft Pendubot, and the Soft Furuta Pendulum- highlighting their performance and computational trade-offs.

  </details>


