# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-01-24 06:46 UTC_

Total papers shown: **15**


---

- **DualShield: Safe Model Predictive Diffusion via Reachability Analysis for Interactive Autonomous Driving**  
  Rui Yang, Lei Zheng, Ruoyu Yao, Jun Ma  
  _2026-01-22_ · https://arxiv.org/abs/2601.15729v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Diffusion models have emerged as a powerful approach for multimodal motion planning in autonomous driving. However, their practical deployment is typically hindered by the inherent difficulty in enforcing vehicle dynamics and a critical reliance on accurate predictions of other agents, making them prone to safety issues under uncertain interactions. To address these limitations, we introduce DualShield, a planning and control framework that leverages Hamilton-Jacobi (HJ) reachability value functions in a dual capacity. First, the value functions act as proactive guidance, steering the diffusion denoising process towards safe and dynamically feasible regions. Second, they form a reactive safety shield using control barrier-value functions (CBVFs) to modify the executed actions and ensure safety. This dual mechanism preserves the rich exploration capabilities of diffusion models while providing principled safety assurance under uncertain and even adversarial interactions. Simulations in challenging unprotected U-turn scenarios demonstrate that DualShield significantly improves both safety and task efficiency compared to leading methods from different planning paradigms under uncertainty.

  </details>



- **Accurate Calibration and Robust LiDAR-Inertial Odometry for Spinning Actuated LiDAR Systems**  
  Zijie Chen, Xiaowei Liu, Yong Xu, Shenghai Yuan, Jianping Li, Lihua Xie  
  _2026-01-22_ · https://arxiv.org/abs/2601.15946v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate calibration and robust localization are fundamental for downstream tasks in spinning actuated LiDAR applications. Existing methods, however, require parameterizing extrinsic parameters based on different mounting configurations, limiting their generalizability. Additionally, spinning actuated LiDAR inevitably scans featureless regions, which complicates the balance between scanning coverage and localization robustness. To address these challenges, this letter presents a targetless LiDAR-motor calibration (LM-Calibr) on the basis of the Denavit-Hartenberg convention and an environmental adaptive LiDAR-inertial odometry (EVA-LIO). LM-Calibr supports calibration of LiDAR-motor systems with various mounting configurations. Extensive experiments demonstrate its accuracy and convergence across different scenarios, mounting angles, and initial values. Additionally, EVA-LIO adaptively selects downsample rates and map resolutions according to spatial scale. This adaptivity enables the actuator to operate at maximum speed, thereby enhancing scanning completeness while ensuring robust localization, even when LiDAR briefly scans featureless areas. The source code and hardware design are available on GitHub: \textcolor{blue}{\href{https://github.com/zijiechenrobotics/lm_calibr}{github.com/zijiechenrobotics/lm\_calibr}}. The video is available at \textcolor{blue}{\href{https://youtu.be/cZyyrkmeoSk}{youtu.be/cZyyrkmeoSk}}

  </details>



- **Data-Driven Conditional Flexibility Index**  
  Moritz Wedemeyer, Eike Cramer, Alexander Mitsos, Manuel Dahmen  
  _2026-01-22_ · https://arxiv.org/abs/2601.16028v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  With the increasing flexibilization of processes, determining robust scheduling decisions has become an important goal. Traditionally, the flexibility index has been used to identify safe operating schedules by approximating the admissible uncertainty region using simple admissible uncertainty sets, such as hypercubes. Presently, available contextual information, such as forecasts, has not been considered to define the admissible uncertainty set when determining the flexibility index. We propose the conditional flexibility index (CFI), which extends the traditional flexibility index in two ways: by learning the parametrized admissible uncertainty set from historical data and by using contextual information to make the admissible uncertainty set conditional. This is achieved using a normalizing flow that learns a bijective mapping from a Gaussian base distribution to the data distribution. The admissible latent uncertainty set is constructed as a hypersphere in the latent space and mapped to the data space. By incorporating contextual information, the CFI provides a more informative estimate of flexibility by defining admissible uncertainty sets in regions that are more likely to be relevant under given conditions. Using an illustrative example, we show that no general statement can be made about data-driven admissible uncertainty sets outperforming simple sets, or conditional sets outperforming unconditional ones. However, both data-driven and conditional admissible uncertainty sets ensure that only regions of the uncertain parameter space containing realizations are considered. We apply the CFI to a security-constrained unit commitment example and demonstrate that the CFI can improve scheduling quality by incorporating temporal information.

  </details>



- **Agentic Confidence Calibration**  
  Jiaxin Zhang, Caiming Xiong, Chien-Sheng Wu  
  _2026-01-22_ · https://arxiv.org/abs/2601.15778v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  AI agents are rapidly advancing from passive language models to autonomous systems executing complex, multi-step tasks. Yet their overconfidence in failure remains a fundamental barrier to deployment in high-stakes settings. Existing calibration methods, built for static single-turn outputs, cannot address the unique challenges of agentic systems, such as compounding errors along trajectories, uncertainty from external tools, and opaque failure modes. To address these challenges, we introduce, for the first time, the problem of Agentic Confidence Calibration and propose Holistic Trajectory Calibration (HTC), a novel diagnostic framework that extracts rich process-level features ranging from macro dynamics to micro stability across an agent's entire trajectory. Powered by a simple, interpretable model, HTC consistently surpasses strong baselines in both calibration and discrimination, across eight benchmarks, multiple LLMs, and diverse agent frameworks. Beyond performance, HTC delivers three essential advances: it provides interpretability by revealing the signals behind failure, enables transferability by applying across domains without retraining, and achieves generalization through a General Agent Calibrator (GAC) that achieves the best calibration (lowest ECE) on the out-of-domain GAIA benchmark. Together, these contributions establish a new process-centric paradigm for confidence calibration, providing a framework for diagnosing and enhancing the reliability of AI agents.

  </details>



- **Off-Policy Actor-Critic with Sigmoid-Bounded Entropy for Real-World Robot Learning**  
  Xiefeng Wu, Mingyu Hu, Shu Zhang  
  _2026-01-22_ · https://arxiv.org/abs/2601.15761v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Deploying reinforcement learning in the real world remains challenging due to sample inefficiency, sparse rewards, and noisy visual observations. Prior work leverages demonstrations and human feedback to improve learning efficiency and robustness. However, offline-to-online methods need large datasets and can be unstable, while VLA-assisted RL relies on large-scale pretraining and fine-tuning. As a result, a low-cost real-world RL method with minimal data requirements has yet to emerge. We introduce \textbf{SigEnt-SAC}, an off-policy actor-critic method that learns from scratch using a single expert trajectory. Our key design is a sigmoid-bounded entropy term that prevents negative-entropy-driven optimization toward out-of-distribution actions and reduces Q-function oscillations. We benchmark SigEnt-SAC on D4RL tasks against representative baselines. Experiments show that SigEnt-SAC substantially alleviates Q-function oscillations and reaches a 100\% success rate faster than prior methods. Finally, we validate SigEnt-SAC on four real-world robotic tasks across multiple embodiments, where agents learn from raw images and sparse rewards; results demonstrate that SigEnt-SAC can learn successful policies with only a small number of real-world interactions, suggesting a low-cost and practical pathway for real-world RL deployment.

  </details>



- **D-Optimality-Guided Reinforcement Learning for Efficient Open-Loop Calibration of a 3-DOF Ankle Rehabilitation Robot**  
  Qifan Hu, Branko Celler, Weidong Mu, Steven W. Su  
  _2026-01-22_ · https://arxiv.org/abs/2601.15707v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate alignment of multi-degree-of-freedom rehabilitation robots is essential for safe and effective patient training. This paper proposes a two-stage calibration framework for a self-designed three-degree-of-freedom (3-DOF) ankle rehabilitation robot. First, a Kronecker-product-based open-loop calibration method is developed to cast the input-output alignment into a linear parameter identification problem, which in turn defines the associated experimental design objective through the resulting information matrix. Building on this formulation, calibration posture selection is posed as a combinatorial design-of-experiments problem guided by a D-optimality criterion, i.e., selecting a small subset of postures that maximises the determinant of the information matrix. To enable practical selection under constraints, a Proximal Policy Optimization (PPO) agent is trained in simulation to choose 4 informative postures from a candidate set of 50. Across simulation and real-robot evaluations, the learned policy consistently yields substantially more informative posture combinations than random selection: the mean determinant of the information matrix achieved by PPO is reported to be more than two orders of magnitude higher with reduced variance. In addition, real-world results indicate that a parameter vector identified from only four D-optimality-guided postures provides stronger cross-episode prediction consistency than estimates obtained from a larger but unstructured set of 50 postures. The proposed framework therefore improves calibration efficiency while maintaining robust parameter estimation, offering practical guidance for high-precision alignment of multi-DOF rehabilitation robots.

  </details>



- **Agentic Uncertainty Quantification**  
  Jiaxin Zhang, Prafulla Kumar Choubey, Kung-Hsiang Huang, Caiming Xiong, Chien-Sheng Wu  
  _2026-01-22_ · https://arxiv.org/abs/2601.15703v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Although AI agents have demonstrated impressive capabilities in long-horizon reasoning, their reliability is severely hampered by the ``Spiral of Hallucination,'' where early epistemic errors propagate irreversibly. Existing methods face a dilemma: uncertainty quantification (UQ) methods typically act as passive sensors, only diagnosing risks without addressing them, while self-reflection mechanisms suffer from continuous or aimless corrections. To bridge this gap, we propose a unified Dual-Process Agentic UQ (AUQ) framework that transforms verbalized uncertainty into active, bi-directional control signals. Our architecture comprises two complementary mechanisms: System 1 (Uncertainty-Aware Memory, UAM), which implicitly propagates verbalized confidence and semantic explanations to prevent blind decision-making; and System 2 (Uncertainty-Aware Reflection, UAR), which utilizes these explanations as rational cues to trigger targeted inference-time resolution only when necessary. This enables the agent to balance efficient execution and deep deliberation dynamically. Extensive experiments on closed-loop benchmarks and open-ended deep research tasks demonstrate that our training-free approach achieves superior performance and trajectory-level calibration. We believe this principled framework AUQ represents a significant step towards reliable agents.

  </details>



- **PUMA: Perception-driven Unified Foothold Prior for Mobility Augmented Quadruped Parkour**  
  Liang Wang, Kanzhong Yao, Yang Liu, Weikai Qin, Jun Wu, Zhe Sun, Qiuguo Zhu  
  _2026-01-22_ · https://arxiv.org/abs/2601.15995v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Parkour tasks for quadrupeds have emerged as a promising benchmark for agile locomotion. While human athletes can effectively perceive environmental characteristics to select appropriate footholds for obstacle traversal, endowing legged robots with similar perceptual reasoning remains a significant challenge. Existing methods often rely on hierarchical controllers that follow pre-computed footholds, thereby constraining the robot's real-time adaptability and the exploratory potential of reinforcement learning. To overcome these challenges, we present PUMA, an end-to-end learning framework that integrates visual perception and foothold priors into a single-stage training process. This approach leverages terrain features to estimate egocentric polar foothold priors, composed of relative distance and heading, guiding the robot in active posture adaptation for parkour tasks. Extensive experiments conducted in simulation and real-world environments across various discrete complex terrains, demonstrate PUMA's exceptional agility and robustness in challenging scenarios.

  </details>



- **ICON: Invariant Counterfactual Optimization with Neuro-Symbolic Priors for Text-Based Person Search**  
  Xiangyu Wang, Zhixin Lv, Yongjiao Sun, Anrui Han, Ye Yuan, Hangxu Ji  
  _2026-01-22_ · https://arxiv.org/abs/2601.15931v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Text-Based Person Search (TBPS) holds unique value in real-world surveillance bridging visual perception and language understanding, yet current paradigms utilizing pre-training models often fail to transfer effectively to complex open-world scenarios. The reliance on "Passive Observation" leads to multifaceted spurious correlations and spatial semantic misalignment, causing a lack of robustness against distribution shifts. To fundamentally resolve these defects, this paper proposes ICON (Invariant Counterfactual Optimization with Neuro-symbolic priors), a framework integrating causal and topological priors. First, we introduce Rule-Guided Spatial Intervention to strictly penalize sensitivity to bounding box noise, forcibly severing location shortcuts to achieve geometric invariance. Second, Counterfactual Context Disentanglement is implemented via semantic-driven background transplantation, compelling the model to ignore background interference for environmental independence. Then, we employ Saliency-Driven Semantic Regularization with adaptive masking to resolve local saliency bias and guarantee holistic completeness. Finally, Neuro-Symbolic Topological Alignment utilizes neuro-symbolic priors to constrain feature matching, ensuring activated regions are topologically consistent with human structural logic. Experimental results demonstrate that ICON not only maintains leading performance on standard benchmarks but also exhibits exceptional robustness against occlusion, background interference, and localization noise. This approach effectively advances the field by shifting from fitting statistical co-occurrences to learning causal invariance.

  </details>



- **TeNet: Text-to-Network for Compact Policy Synthesis**  
  Ariyan Bighashdel, Kevin Sebastian Luck  
  _2026-01-22_ · https://arxiv.org/abs/2601.15912v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots that follow natural-language instructions often either plan at a high level using hand-designed interfaces or rely on large end-to-end models that are difficult to deploy for real-time control. We propose TeNet (Text-to-Network), a framework for instantiating compact, task-specific robot policies directly from natural language descriptions. TeNet conditions a hypernetwork on text embeddings produced by a pretrained large language model (LLM) to generate a fully executable policy, which then operates solely on low-dimensional state inputs at high control frequencies. By using the language only once at the policy instantiation time, TeNet inherits the general knowledge and paraphrasing robustness of pretrained LLMs while remaining lightweight and efficient at execution time. To improve generalization, we optionally ground language in behavior during training by aligning text embeddings with demonstrated actions, while requiring no demonstrations at inference time. Experiments on MuJoCo and Meta-World benchmarks show that TeNet produces policies that are orders of magnitude smaller than sequence-based baselines, while achieving strong performance in both multi-task and meta-learning settings and supporting high-frequency control. These results show that text-conditioned hypernetworks offer a practical way to build compact, language-driven controllers for ressource-constrained robot control tasks with real-time requirements.

  </details>



- **360Anything: Geometry-Free Lifting of Images and Videos to 360°**  
  Ziyi Wu, Daniel Watson, Andrea Tagliasacchi, David J. Fleet, Marcus A. Brubaker, Saurabh Saxena  
  _2026-01-22_ · https://arxiv.org/abs/2601.16192v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Lifting perspective images and videos to 360° panoramas enables immersive 3D world generation. Existing approaches often rely on explicit geometric alignment between the perspective and the equirectangular projection (ERP) space. Yet, this requires known camera metadata, obscuring the application to in-the-wild data where such calibration is typically absent or noisy. We propose 360Anything, a geometry-free framework built upon pre-trained diffusion transformers. By treating the perspective input and the panorama target simply as token sequences, 360Anything learns the perspective-to-equirectangular mapping in a purely data-driven way, eliminating the need for camera information. Our approach achieves state-of-the-art performance on both image and video perspective-to-360° generation, outperforming prior works that use ground-truth camera information. We also trace the root cause of the seam artifacts at ERP boundaries to zero-padding in the VAE encoder, and introduce Circular Latent Encoding to facilitate seamless generation. Finally, we show competitive results in zero-shot camera FoV and orientation estimation benchmarks, demonstrating 360Anything's deep geometric understanding and broader utility in computer vision tasks. Additional results are available at https://360anything.github.io/.

  </details>



- **Efficiently Learning Robust Torque-based Locomotion Through Reinforcement with Model-Based Supervision**  
  Yashuai Yan, Tobias Egle, Christian Ott, Dongheui Lee  
  _2026-01-22_ · https://arxiv.org/abs/2601.16109v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We propose a control framework that integrates model-based bipedal locomotion with residual reinforcement learning (RL) to achieve robust and adaptive walking in the presence of real-world uncertainties. Our approach leverages a model-based controller, comprising a Divergent Component of Motion (DCM) trajectory planner and a whole-body controller, as a reliable base policy. To address the uncertainties of inaccurate dynamics modeling and sensor noise, we introduce a residual policy trained through RL with domain randomization. Crucially, we employ a model-based oracle policy, which has privileged access to ground-truth dynamics during training, to supervise the residual policy via a novel supervised loss. This supervision enables the policy to efficiently learn corrective behaviors that compensate for unmodeled effects without extensive reward shaping. Our method demonstrates improved robustness and generalization across a range of randomized conditions, offering a scalable solution for sim-to-real transfer in bipedal locomotion.

  </details>



- **Neural Particle Automata: Learning Self-Organizing Particle Dynamics**  
  Hyunsoo Kim, Ehsan Pajouheshgar, Sabine Süsstrunk, Wenzel Jakob, Jinah Park  
  _2026-01-22_ · https://arxiv.org/abs/2601.16096v1 · `cs.NE`  
  <details><summary>Abstract</summary>

  We introduce Neural Particle Automata (NPA), a Lagrangian generalization of Neural Cellular Automata (NCA) from static lattices to dynamic particle systems. Unlike classical Eulerian NCA where cells are pinned to pixels or voxels, NPA model each cell as a particle with a continuous position and internal state, both updated by a shared, learnable neural rule. This particle-based formulation yields clear individuation of cells, allows heterogeneous dynamics, and concentrates computation only on regions where activity is present. At the same time, particle systems pose challenges: neighborhoods are dynamic, and a naive implementation of local interactions scale quadratically with the number of particles. We address these challenges by replacing grid-based neighborhood perception with differentiable Smoothed Particle Hydrodynamics (SPH) operators backed by memory-efficient, CUDA-accelerated kernels, enabling scalable end-to-end training. Across tasks including morphogenesis, point-cloud classification, and particle-based texture synthesis, we show that NPA retain key NCA behaviors such as robustness and self-regeneration, while enabling new behaviors specific to particle systems. Together, these results position NPA as a compact neural model for learning self-organizing particle dynamics.

  </details>



- **Masked Modeling for Human Motion Recovery Under Occlusions**  
  Zhiyin Qian, Siwei Zhang, Bharat Lal Bhatnagar, Federica Bogo, Siyu Tang  
  _2026-01-22_ · https://arxiv.org/abs/2601.16079v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Human motion reconstruction from monocular videos is a fundamental challenge in computer vision, with broad applications in AR/VR, robotics, and digital content creation, but remains challenging under frequent occlusions in real-world settings.Existing regression-based methods are efficient but fragile to missing observations, while optimization- and diffusion-based approaches improve robustness at the cost of slow inference speed and heavy preprocessing steps. To address these limitations, we leverage recent advances in generative masked modeling and present MoRo: Masked Modeling for human motion Recovery under Occlusions. MoRo is an occlusion-robust, end-to-end generative framework that formulates motion reconstruction as a video-conditioned task, and efficiently recover human motion in a consistent global coordinate system from RGB videos. By masked modeling, MoRo naturally handles occlusions while enabling efficient, end-to-end inference. To overcome the scarcity of paired video-motion data, we design a cross-modality learning scheme that learns multi-modal priors from a set of heterogeneous datasets: (i) a trajectory-aware motion prior trained on MoCap datasets, (ii) an image-conditioned pose prior trained on image-pose datasets, capturing diverse per-frame poses, and (iii) a video-conditioned masked transformer that fuses motion and pose priors, finetuned on video-motion datasets to integrate visual cues with motion dynamics for robust inference. Extensive experiments on EgoBody and RICH demonstrate that MoRo substantially outperforms state-of-the-art methods in accuracy and motion realism under occlusions, while performing on-par in non-occluded scenarios. MoRo achieves real-time inference at 70 FPS on a single H200 GPU.

  </details>



- **AgriPINN: A Process-Informed Neural Network for Interpretable and Scalable Crop Biomass Prediction Under Water Stress**  
  Yue Shi, Liangxiu Han, Xin Zhang, Tam Sobeih, Thomas Gaiser, Nguyen Huu Thuy, Dominik Behrend, Amit Kumar Srivastava, Krishnagopal Halder, Frank Ewert  
  _2026-01-22_ · https://arxiv.org/abs/2601.16045v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Accurate prediction of crop above-ground biomass (AGB) under water stress is critical for monitoring crop productivity, guiding irrigation, and supporting climate-resilient agriculture. Data-driven models scale well but often lack interpretability and degrade under distribution shift, whereas process-based crop models (e.g. DSSAT, APSIM, LINTUL5) require extensive calibration and are difficult to deploy over large spatial domains. To address these limitations, we propose AgriPINN, a process-informed neural network that integrates a biophysical crop-growth differential equation as a differentiable constraint within a deep learning backbone. This design encourages physiologically consistent biomass dynamics under water-stress conditions while preserving model scalability for spatially distributed AGB prediction. AgriPINN recovers latent physiological variables, including leaf area index (LAI), absorbed photosynthetically active radiation (PAR), radiation use efficiency (RUE), and water-stress factors, without requiring direct supervision. We pretrain AgriPINN on 60 years of historical data across 397 regions in Germany and fine-tune it on three years of field experiments under controlled water treatments. Results show that AgriPINN consistently outperforms state-of-the-art deep-learning baselines (ConvLSTM-ViT, SLTF, CNN-Transformer) and the process-based LINTUL5 model in terms of accuracy (RMSE reductions up to $43\%$) and computational efficiency. By combining the scalability of deep learning with the biophysical rigor of process-based modeling, AgriPINN provides a robust and interpretable framework for spatio-temporal AGB prediction, offering practical value for planning of irrigation infrastructure, yield forecasting, and climate-adaptation planning.

  </details>


