# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-12 07:15 UTC_

Total papers shown: **12**


---

- **Safe mobility support system using crowd mapping and avoidance route planning using VLM**  
  Sena Saito, Kenta Tabata, Renato Miyagusuku, Koichi Ozaki  
  _2026-02-11_ · https://arxiv.org/abs/2602.10910v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous mobile robots offer promising solutions for labor shortages and increased operational efficiency. However, navigating safely and effectively in dynamic environments, particularly crowded areas, remains challenging. This paper proposes a novel framework that integrates Vision-Language Models (VLM) and Gaussian Process Regression (GPR) to generate dynamic crowd-density maps (``Abstraction Maps'') for autonomous robot navigation. Our approach utilizes VLM's capability to recognize abstract environmental concepts, such as crowd densities, and represents them probabilistically via GPR. Experimental results from real-world trials on a university campus demonstrated that robots successfully generated routes avoiding both static obstacles and dynamic crowds, enhancing navigation safety and adaptability.

  </details>



- **From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?**  
  Krishna Kanth Nakka, Vedasri Nakka  
  _2026-02-11_ · https://arxiv.org/abs/2602.10771v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems.

  </details>



- **SQ-CBF: Signed Distance Functions for Numerically Stable Superquadric-Based Safety Filtering**  
  Haocheng Zhao, Lukas Brunke, Oliver Lagerquist, Siqi Zhou, Angela P. Schoellig  
  _2026-02-11_ · https://arxiv.org/abs/2602.11049v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Ensuring safe robot operation in cluttered and dynamic environments remains a fundamental challenge. While control barrier functions provide an effective framework for real-time safety filtering, their performance critically depends on the underlying geometric representation, which is often simplified, leading to either overly conservative behavior or insufficient collision coverage. Superquadrics offer an expressive way to model complex shapes using a few primitives and are increasingly used for robot safety. To integrate this representation into collision avoidance, most existing approaches directly use their implicit functions as barrier candidates. However, we identify a critical but overlooked issue in this practice: the gradients of the implicit SQ function can become severely ill-conditioned, potentially rendering the optimization infeasible and undermining reliable real-time safety filtering. To address this issue, we formulate an SQ-based safety filtering framework that uses signed distance functions as barrier candidates. Since analytical SDFs are unavailable for general SQs, we compute distances using the efficient Gilbert-Johnson-Keerthi algorithm and obtain gradients via randomized smoothing. Extensive simulation and real-world experiments demonstrate consistent collision-free manipulation in cluttered and unstructured scenes, showing robustness to challenging geometries, sensing noise, and dynamic disturbances, while improving task efficiency in teleoperation tasks. These results highlight a pathway toward safety filters that remain precise and reliable under the geometric complexity of real-world environments.

  </details>



- **APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots**  
  Yikai Wang, Tingxuan Leng, Changyi Lin, Shiqi Liu, Shir Simon, Bingqing Chen, Jonathan Francis, Ding Zhao  
  _2026-02-11_ · https://arxiv.org/abs/2602.11143v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Humanoid locomotion has advanced rapidly with deep reinforcement learning (DRL), enabling robust feet-based traversal over uneven terrain. Yet platforms beyond leg length remain largely out of reach because current RL training paradigms often converge to jumping-like solutions that are high-impact, torque-limited, and unsafe for real-world deployment. To address this gap, we propose APEX, a system for perceptive, climbing-based high-platform traversal that composes terrain-conditioned behaviors: climb-up and climb-down at vertical edges, walking or crawling on the platform, and stand-up and lie-down for posture reconfiguration. Central to our approach is a generalized ratchet progress reward for learning contact-rich, goal-reaching maneuvers. It tracks the best-so-far task progress and penalizes non-improving steps, providing dense yet velocity-free supervision that enables efficient exploration under strong safety regularization. Based on this formulation, we train LiDAR-based full-body maneuver policies and reduce the sim-to-real perception gap through a dual strategy: modeling mapping artifacts during training and applying filtering and inpainting to elevation maps during deployment. Finally, we distill all six skills into a single policy that autonomously selects behaviors and transitions based on local geometry and commands. Experiments on a 29-DoF Unitree G1 humanoid demonstrate zero-shot sim-to-real traversal of 0.8 meter platforms (approximately 114% of leg length), with robust adaptation to platform height and initial pose, as well as smooth and stable multi-skill transitions.

  </details>



- **RISE: Self-Improving Robot Policy with Compositional World Model**  
  Jiazhi Yang, Kunyang Lin, Jinwei Li, Wencong Zhang, Tianwei Lin, Longyan Wu, Zhizhong Su, Hao Zhao, Ya-Qin Zhang, Li Chen, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11075v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Despite the sustained scaling on model capacity and data acquisition, Vision-Language-Action (VLA) models remain brittle in contact-rich and dynamic manipulation tasks, where minor execution deviations can compound into failures. While reinforcement learning (RL) offers a principled path to robustness, on-policy RL in the physical world is constrained by safety risk, hardware cost, and environment reset. To bridge this gap, we present RISE, a scalable framework of robotic reinforcement learning via imagination. At its core is a Compositional World Model that (i) predicts multi-view future via a controllable dynamics model, and (ii) evaluates imagined outcomes with a progress value model, producing informative advantages for the policy improvement. Such compositional design allows state and value to be tailored by best-suited yet distinct architectures and objectives. These components are integrated into a closed-loop self-improving pipeline that continuously generates imaginary rollouts, estimates advantages, and updates the policy in imaginary space without costly physical interaction. Across three challenging real-world tasks, RISE yields significant improvement over prior art, with more than +35% absolute performance increase in dynamic brick sorting, +45% for backpack packing, and +35% for box closing, respectively.

  </details>



- **Robust Assortment Optimization from Observational Data**  
  Miao Lu, Yuxuan Han, Han Zhong, Zhengyuan Zhou, Jose Blanchet  
  _2026-02-11_ · https://arxiv.org/abs/2602.10696v1 · `stat.ML`  
  <details><summary>Abstract</summary>

  Assortment optimization is a fundamental challenge in modern retail and recommendation systems, where the goal is to select a subset of products that maximizes expected revenue under complex customer choice behaviors. While recent advances in data-driven methods have leveraged historical data to learn and optimize assortments, these approaches typically rely on strong assumptions -- namely, the stability of customer preferences and the correctness of the underlying choice models. However, such assumptions frequently break in real-world scenarios due to preference shifts and model misspecification, leading to poor generalization and revenue loss. Motivated by this limitation, we propose a robust framework for data-driven assortment optimization that accounts for potential distributional shifts in customer choice behavior. Our approach models potential preference shift from a nominal choice model that generates data and seeks to maximize worst-case expected revenue. We first establish the computational tractability of robust assortment planning when the nominal model is known, then advance to the data-driven setting, where we design statistically optimal algorithms that minimize the data requirements while maintaining robustness. Our theoretical analysis provides both upper bounds and matching lower bounds on the sample complexity, offering theoretical guarantees for robust generalization. Notably, we uncover and identify the notion of ``robust item-wise coverage'' as the minimal data requirement to enable sample-efficient robust assortment learning. Our work bridges the gap between robustness and statistical efficiency in assortment learning, contributing new insights and tools for reliable assortment optimization under uncertainty.

  </details>



- **Multi-UAV Trajectory Optimization for Bearing-Only Localization in GPS Denied Environments**  
  Alfonso Sciacchitano, Liraz Mudrik, Sean Kragelund, Isaac Kaminer  
  _2026-02-11_ · https://arxiv.org/abs/2602.11116v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Accurate localization of maritime targets by unmanned aerial vehicles (UAVs) remains challenging in GPS-denied environments. UAVs equipped with gimballed electro-optical sensors are typically used to localize targets, however, reliance on these sensors increases mechanical complexity, cost, and susceptibility to single-point failures, limiting scalability and robustness in multi-UAV operations. This work presents a new trajectory optimization framework that enables cooperative target localization using UAVs with fixed, non-gimballed cameras operating in coordination with a surface vessel. This estimation-aware optimization generates dynamically feasible trajectories that explicitly account for mission constraints, platform dynamics, and out-of-frame events. Estimation-aware trajectories outperform heuristic paths by reducing localization error by more than a factor of two, motivating their use in cooperative operations. Results further demonstrate that coordinated UAVs with fixed, non-gimballed cameras achieve localization accuracy that meets or exceeds that of single gimballed systems, while substantially lowering system complexity and cost, enabling scalability, and enhancing mission resilience.

  </details>



- **Interpretable Vision Transformers in Monocular Depth Estimation via SVDA**  
  Vasileios Arampatzakis, George Pavlidis, Nikolaos Mitianoudis, Nikos Papamarkos  
  _2026-02-11_ · https://arxiv.org/abs/2602.11005v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Monocular depth estimation is a central problem in computer vision with applications in robotics, AR, and autonomous driving, yet the self-attention mechanisms that drive modern Transformer architectures remain opaque. We introduce SVD-Inspired Attention (SVDA) into the Dense Prediction Transformer (DPT), providing the first spectrally structured formulation of attention for dense prediction tasks. SVDA decouples directional alignment from spectral modulation by embedding a learnable diagonal matrix into normalized query-key interactions, enabling attention maps that are intrinsically interpretable rather than post-hoc approximations. Experiments on KITTI and NYU-v2 show that SVDA preserves or slightly improves predictive accuracy while adding only minor computational overhead. More importantly, SVDA unlocks six spectral indicators that quantify entropy, rank, sparsity, alignment, selectivity, and robustness. These reveal consistent cross-dataset and depth-wise patterns in how attention organizes during training, insights that remain inaccessible in standard Transformers. By shifting the role of attention from opaque mechanism to quantifiable descriptor, SVDA redefines interpretability in monocular depth estimation and opens a principled avenue toward transparent dense prediction models.

  </details>



- **Transfer to Sky: Unveil Low-Altitude Route-Level Radio Maps via Ground Crowdsourced Data**  
  Wenlihan Lu, Huacong Chen, Ruiyang Duan, Weijie Yuan, Shijian Gao  
  _2026-02-11_ · https://arxiv.org/abs/2602.10736v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  The expansion of the low-altitude economy is contingent on reliable cellular connectivity for unmanned aerial vehicles (UAVs). A key challenge in pre-flight planning is predicting communication link quality along proposed and pre-defined routes, a task hampered by sparse measurements that render existing radio map methods ineffective. This paper introduces a transfer learning framework for high-fidelity route-level radio map prediction. Our key insight is to leverage abundant crowdsourced ground signals as auxiliary supervision. To bridge the significant domain gap between ground and aerial data and address spatial sparsity, our framework learns general propagation priors from simulation, performs adversarial alignment of the feature spaces, and is fine-tuned on limited real UAV measurements. Extensive experiments on a real-world dataset from Meituan show that our method achieves over 50% higher accuracy in predicting Route RSRP compared to state-of-the-art baselines.

  </details>



- **AugVLA-3D: Depth-Driven Feature Augmentation for Vision-Language-Action Models**  
  Zhifeng Rao, Wenlong Chen, Lei Xie, Xia Hua, Dongfu Yin, Zhen Tian, F. Richard Yu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10698v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic perception and control, yet most existing approaches primarily rely on VLM trained using 2D images, which limits their spatial understanding and action grounding in complex 3D environments. To address this limitation, we propose a novel framework that integrates depth estimation into VLA models to enrich 3D feature representations. Specifically, we employ a depth estimation baseline called VGGT to extract geometry-aware 3D cues from standard RGB inputs, enabling efficient utilization of existing large-scale 2D datasets while implicitly recovering 3D structural information. To further enhance the reliability of these depth-derived features, we introduce a new module called action assistant, which constrains the learned 3D representations with action priors and ensures their consistency with downstream control tasks. By fusing the enhanced 3D features with conventional 2D visual tokens, our approach significantly improves the generalization ability and robustness of VLA models. Experimental results demonstrate that the proposed method not only strengthens perception in geometrically ambiguous scenarios but also leads to superior action prediction accuracy. This work highlights the potential of depth-driven data augmentation and auxiliary expert supervision for bridging the gap between 2D observations and 3D-aware decision-making in robotic systems.

  </details>



- **Scaling World Model for Hierarchical Manipulation Policies**  
  Qian Long, Yueze Wang, Jiaxi Song, Junbo Zhang, Peiyan Li, Wenxuan Wang, Yuqi Wang, Haoyang Li, Shaoxuan Xie, Guocai Yao, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10983v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models are promising for generalist robot manipulation but remain brittle in out-of-distribution (OOD) settings, especially with limited real-robot data. To resolve the generalization bottleneck, we introduce a hierarchical Vision-Language-Action framework \our{} that leverages the generalization of large-scale pre-trained world model for robust and generalizable VIsual Subgoal TAsk decomposition VISTA. Our hierarchical framework \our{} consists of a world model as the high-level planner and a VLA as the low-level executor. The high-level world model first divides manipulation tasks into subtask sequences with goal images, and the low-level policy follows the textual and visual guidance to generate action sequences. Compared to raw textual goal specification, these synthesized goal images provide visually and physically grounded details for low-level policies, making it feasible to generalize across unseen objects and novel scenarios. We validate both visual goal synthesis and our hierarchical VLA policies in massive out-of-distribution scenarios, and the performance of the same-structured VLA in novel scenarios could boost from 14% to 69% with the guidance generated by the world model. Results demonstrate that our method outperforms previous baselines with a clear margin, particularly in out-of-distribution scenarios. Project page: \href{https://vista-wm.github.io/}{https://vista-wm.github.io}

  </details>



- **Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation**  
  Songen Gu, Yunuo Cai, Tianyu Wang, Simo Wu, Yanwei Fu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10717v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic manipulation requires anticipating how the environment evolves in response to actions, yet most existing systems lack this predictive capability, often resulting in errors and inefficiency. While Vision-Language Models (VLMs) provide high-level guidance, they cannot explicitly forecast future states, and existing world models either predict only short horizons or produce spatially inconsistent frames. To address these challenges, we propose a framework for fast and predictive video-conditioned action. Our approach first selects and adapts a robust video generation model to ensure reliable future predictions, then applies adversarial distillation for fast, few-step video generation, and finally trains an action model that leverages both generated videos and real observations to correct spatial errors. Extensive experiments show that our method produces temporally coherent, spatially accurate video predictions that directly support precise manipulation, achieving significant improvements in embodiment consistency, spatial referring ability, and task completion over existing baselines. Codes & Models will be released.

  </details>


