# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-12 07:15 UTC_

Total papers shown: **10**


---

- **Safe mobility support system using crowd mapping and avoidance route planning using VLM**  
  Sena Saito, Kenta Tabata, Renato Miyagusuku, Koichi Ozaki  
  _2026-02-11_ · https://arxiv.org/abs/2602.10910v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous mobile robots offer promising solutions for labor shortages and increased operational efficiency. However, navigating safely and effectively in dynamic environments, particularly crowded areas, remains challenging. This paper proposes a novel framework that integrates Vision-Language Models (VLM) and Gaussian Process Regression (GPR) to generate dynamic crowd-density maps (``Abstraction Maps'') for autonomous robot navigation. Our approach utilizes VLM's capability to recognize abstract environmental concepts, such as crowd densities, and represents them probabilistically via GPR. Experimental results from real-world trials on a university campus demonstrated that robots successfully generated routes avoiding both static obstacles and dynamic crowds, enhancing navigation safety and adaptability.

  </details>



- **APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots**  
  Yikai Wang, Tingxuan Leng, Changyi Lin, Shiqi Liu, Shir Simon, Bingqing Chen, Jonathan Francis, Ding Zhao  
  _2026-02-11_ · https://arxiv.org/abs/2602.11143v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Humanoid locomotion has advanced rapidly with deep reinforcement learning (DRL), enabling robust feet-based traversal over uneven terrain. Yet platforms beyond leg length remain largely out of reach because current RL training paradigms often converge to jumping-like solutions that are high-impact, torque-limited, and unsafe for real-world deployment. To address this gap, we propose APEX, a system for perceptive, climbing-based high-platform traversal that composes terrain-conditioned behaviors: climb-up and climb-down at vertical edges, walking or crawling on the platform, and stand-up and lie-down for posture reconfiguration. Central to our approach is a generalized ratchet progress reward for learning contact-rich, goal-reaching maneuvers. It tracks the best-so-far task progress and penalizes non-improving steps, providing dense yet velocity-free supervision that enables efficient exploration under strong safety regularization. Based on this formulation, we train LiDAR-based full-body maneuver policies and reduce the sim-to-real perception gap through a dual strategy: modeling mapping artifacts during training and applying filtering and inpainting to elevation maps during deployment. Finally, we distill all six skills into a single policy that autonomously selects behaviors and transitions based on local geometry and commands. Experiments on a 29-DoF Unitree G1 humanoid demonstrate zero-shot sim-to-real traversal of 0.8 meter platforms (approximately 114% of leg length), with robust adaptation to platform height and initial pose, as well as smooth and stable multi-skill transitions.

  </details>



- **(MGS)$^2$-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization**  
  Minglei Li, Mengfan He, Chao Chen, Ziyang Meng  
  _2026-02-11_ · https://arxiv.org/abs/2602.10704v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Cross-view geo-localization (CVGL) is pivotal for GNSS-denied UAV navigation but remains brittle under the drastic geometric misalignment between oblique aerial views and orthographic satellite references. Existing methods predominantly operate within a 2D manifold, neglecting the underlying 3D geometry where view-dependent vertical facades (macro-structure) and scale variations (micro-scale) severely corrupt feature alignment. To bridge this gap, we propose (MGS)$^2$, a geometry-grounded framework. The core of our innovation is the Macro-Geometric Structure Filtering (MGSF) module. Unlike pixel-wise matching sensitive to noise, MGSF leverages dilated geometric gradients to physically filter out high-frequency facade artifacts while enhancing the view-invariant horizontal plane, directly addressing the domain shift. To guarantee robust input for this structural filtering, we explicitly incorporate a Micro-Geometric Scale Adaptation (MGSA) module. MGSA utilizes depth priors to dynamically rectify scale discrepancies via multi-branch feature fusion. Furthermore, a Geometric-Appearance Contrastive Distillation (GACD) loss is designed to strictly discriminate against oblique occlusions. Extensive experiments demonstrate that (MGS)$^2$ achieves state-of-the-art performance, recording a Recall@1 of 97.5\% on University-1652 and 97.02\% on SUES-200. Furthermore, the framework exhibits superior cross-dataset generalization against geometric ambiguity. The code is available at: \href{https://github.com/GabrielLi1473/MGS-Net}{https://github.com/GabrielLi1473/MGS-Net}.

  </details>



- **Multi-UAV Trajectory Optimization for Bearing-Only Localization in GPS Denied Environments**  
  Alfonso Sciacchitano, Liraz Mudrik, Sean Kragelund, Isaac Kaminer  
  _2026-02-11_ · https://arxiv.org/abs/2602.11116v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Accurate localization of maritime targets by unmanned aerial vehicles (UAVs) remains challenging in GPS-denied environments. UAVs equipped with gimballed electro-optical sensors are typically used to localize targets, however, reliance on these sensors increases mechanical complexity, cost, and susceptibility to single-point failures, limiting scalability and robustness in multi-UAV operations. This work presents a new trajectory optimization framework that enables cooperative target localization using UAVs with fixed, non-gimballed cameras operating in coordination with a surface vessel. This estimation-aware optimization generates dynamically feasible trajectories that explicitly account for mission constraints, platform dynamics, and out-of-frame events. Estimation-aware trajectories outperform heuristic paths by reducing localization error by more than a factor of two, motivating their use in cooperative operations. Results further demonstrate that coordinated UAVs with fixed, non-gimballed cameras achieve localization accuracy that meets or exceeds that of single gimballed systems, while substantially lowering system complexity and cost, enabling scalability, and enhancing mission resilience.

  </details>



- **Deep Learning-based Method for Expressing Knowledge Boundary of Black-Box LLM**  
  Haotian Sheng, Heyong Wang, Ming Hong, Hongman He, Junqiu Liu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10801v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have achieved remarkable success, however, the emergence of content generation distortion (hallucination) limits their practical applications. The core cause of hallucination lies in LLMs' lack of awareness regarding their stored internal knowledge, preventing them from expressing their knowledge state on questions beyond their internal knowledge boundaries, as humans do. However, existing research on knowledge boundary expression primarily focuses on white-box LLMs, leaving methods suitable for black-box LLMs which offer only API access without revealing internal parameters-largely unexplored. Against this backdrop, this paper proposes LSCL (LLM-Supervised Confidence Learning), a deep learning-based method for expressing the knowledge boundaries of black-box LLMs. Based on the knowledge distillation framework, this method designs a deep learning model. Taking the input question, output answer, and token probability from a black-box LLM as inputs, it constructs a mapping between the inputs and the model' internal knowledge state, enabling the quantification and expression of the black-box LLM' knowledge boundaries. Experiments conducted on diverse public datasets and with multiple prominent black-box LLMs demonstrate that LSCL effectively assists black-box LLMs in accurately expressing their knowledge boundaries. It significantly outperforms existing baseline models on metrics such as accuracy and recall rate. Furthermore, considering scenarios where some black-box LLMs do not support access to token probability, an adaptive alternative method is proposed. The performance of this alternative approach is close to that of LSCL and surpasses baseline models.

  </details>



- **Dual-End Consistency Model**  
  Linwei Dong, Ruoyu Guo, Ge Bai, Zehuan Yuan, Yawei Luo, Changqing Zou  
  _2026-02-11_ · https://arxiv.org/abs/2602.10764v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The slow iterative sampling nature remains a major bottleneck for the practical deployment of diffusion and flow-based generative models. While consistency models (CMs) represent a state-of-the-art distillation-based approach for efficient generation, their large-scale application is still limited by two key issues: training instability and inflexible sampling. Existing methods seek to mitigate these problems through architectural adjustments or regularized objectives, yet overlook the critical reliance on trajectory selection. In this work, we first conduct an analysis on these two limitations: training instability originates from loss divergence induced by unstable self-supervised term, whereas sampling inflexibility arises from error accumulation. Based on these insights and analysis, we propose the Dual-End Consistency Model (DE-CM) that selects vital sub-trajectory clusters to achieve stable and effective training. DE-CM decomposes the PF-ODE trajectory and selects three critical sub-trajectories as optimization targets. Specifically, our approach leverages continuous-time CMs objectives to achieve few-step distillation and utilizes flow matching as a boundary regularizer to stabilize the training process. Furthermore, we propose a novel noise-to-noisy (N2N) mapping that can map noise to any point, thereby alleviating the error accumulation in the first step. Extensive experimental results show the effectiveness of our method: it achieves a state-of-the-art FID score of 1.70 in one-step generation on the ImageNet 256x256 dataset, outperforming existing CM-based one-step approaches.

  </details>



- **Ecological mapping with geospatial foundation models**  
  Craig Mahlasi, Gciniwe S. Baloyi, Zaheed Gaffoor, Levente Klein, Anne Jones, Etienne Vos, Michal Muszynski, Geoffrey Dawson, Campbell Watson  
  _2026-02-11_ · https://arxiv.org/abs/2602.10720v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Geospatial foundation models (GFMs) are a fast-emerging paradigm for various geospatial tasks, such as ecological mapping. However, the utility of GFMs has not been fully explored for high-value use cases. This study aims to explore the utility, challenges and opportunities associated with the application of GFMs for ecological uses. In this regard, we fine-tune several pretrained AI models, namely, Prithvi-E0-2.0 and TerraMind, across three use cases, and compare this with a baseline ResNet-101 model. Firstly, we demonstrate TerraMind's LULC generation capabilities. Lastly, we explore the utility of the GFMs in forest functional trait mapping and peatlands detection. In all experiments, the GFMs outperform the baseline ResNet models. In general TerraMind marginally outperforms Prithvi. However, with additional modalities TerraMind significantly outperforms the baseline ResNet and Prithvi models. Nonetheless, consideration should be given to the divergence of input data from pretrained modalities. We note that these models would benefit from higher resolution and more accurate labels, especially for use cases where pixel-level dynamics need to be mapped.

  </details>



- **Omnidirectional Dual-Arm Aerial Manipulator with Proprioceptive Contact Localization for Landing on Slanted Roofs**  
  Martijn B. J. Brummelhuis, Nathan F. Lepora, Salua Hamaza  
  _2026-02-11_ · https://arxiv.org/abs/2602.10703v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Operating drones in urban environments often means they need to land on rooftops, which can have different geometries and surface irregularities. Accurately detecting roof inclination using conventional sensing methods, such as vision-based or acoustic techniques, can be unreliable, as measurement quality is strongly influenced by external factors including weather conditions and surface materials. To overcome these challenges, we propose a novel unmanned aerial manipulator morphology featuring a dual-arm aerial manipulator with an omnidirectional 3D workspace and extended reach. Building on this design, we develop a proprioceptive contact detection and contact localization strategy based on a momentum-based torque observer. This enables the UAM to infer the inclination of slanted surfaces blindly - through physical interaction - prior to touchdown. We validate the approach in flight experiments, demonstrating robust landings on surfaces with inclinations of up to 30.5 degrees and achieving an average surface inclination estimation error of 2.87 degrees over 9 experiments at different incline angles.

  </details>



- **OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL**  
  Jinjie Shen, Jing Wu, Yaxiong Wang, Lechao Cheng, Shengeng Tang, Tianrui Hui, Nan Pu, Zhun Zhong  
  _2026-02-11_ · https://arxiv.org/abs/2602.10687v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Existing forgery detection methods are often limited to uni-modal or bi-modal settings, failing to handle the interleaved text, images, and videos prevalent in real-world misinformation. To bridge this gap, this paper targets to develop a unified framework for omnibus vision-language forgery detection and grounding. In this unified setting, the {interplay} between diverse modalities and the dual requirements of simultaneous detection and localization pose a critical ``difficulty bias`` problem: the simpler veracity classification task tends to dominate the gradients, leading to suboptimal performance in fine-grained grounding during multi-task optimization. To address this challenge, we propose \textbf{OmniVL-Guard}, a balanced reinforcement learning framework for omnibus vision-language forgery detection and grounding. Particularly, OmniVL-Guard comprises two core designs: Self-Evolving CoT Generatio and Adaptive Reward Scaling Policy Optimization (ARSPO). {Self-Evolving CoT Generation} synthesizes high-quality reasoning paths, effectively overcoming the cold-start challenge. Building upon this, {Adaptive Reward Scaling Policy Optimization (ARSPO)} dynamically modulates reward scales and task weights, ensuring a balanced joint optimization. Extensive experiments demonstrate that OmniVL-Guard significantly outperforms state-of-the-art methods and exhibits zero-shot robust generalization across out-of-domain scenarios.

  </details>



- **Multimodal Priors-Augmented Text-Driven 3D Human-Object Interaction Generation**  
  Yin Wang, Ziyao Zhang, Zhiying Leng, Haitian Liu, Frederick W. B. Li, Mu Li, Xiaohui Liang  
  _2026-02-11_ · https://arxiv.org/abs/2602.10659v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We address the challenging task of text-driven 3D human-object interaction (HOI) motion generation. Existing methods primarily rely on a direct text-to-HOI mapping, which suffers from three key limitations due to the significant cross-modality gap: (Q1) sub-optimal human motion, (Q2) unnatural object motion, and (Q3) weak interaction between humans and objects. To address these challenges, we propose MP-HOI, a novel framework grounded in four core insights: (1) Multimodal Data Priors: We leverage multimodal data (text, image, pose/object) from large multimodal models as priors to guide HOI generation, which tackles Q1 and Q2 in data modeling. (2) Enhanced Object Representation: We improve existing object representations by incorporating geometric keypoints, contact features, and dynamic properties, enabling expressive object representations, which tackles Q2 in data representation. (3) Multimodal-Aware Mixture-of-Experts (MoE) Model: We propose a modality-aware MoE model for effective multimodal feature fusion paradigm, which tackles Q1 and Q2 in feature fusion. (4) Cascaded Diffusion with Interaction Supervision: We design a cascaded diffusion framework that progressively refines human-object interaction features under dedicated supervision, which tackles Q3 in interaction refinement. Comprehensive experiments demonstrate that MP-HOI outperforms existing approaches in generating high-fidelity and fine-grained HOI motions.

  </details>


