# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-25 07:14 UTC_

Total papers shown: **14**


---

- **Robot Local Planner: A Periodic Sampling-Based Motion Planner with Minimal Waypoints for Home Environments**  
  Keisuke Takeshita, Takahiro Yamazaki, Tomohiro Ono, Takashi Yamamoto  
  _2026-02-24_ · https://arxiv.org/abs/2602.20645v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The objective of this study is to enable fast and safe manipulation tasks in home environments. Specifically, we aim to develop a system that can recognize its surroundings and identify target objects while in motion, enabling it to plan and execute actions accordingly. We propose a periodic sampling-based whole-body trajectory planning method, called the "Robot Local Planner (RLP)." This method leverages unique features of home environments to enhance computational efficiency, motion optimality, and robustness against recognition and control errors, all while ensuring safety. The RLP minimizes computation time by planning with minimal waypoints and generating safe trajectories. Furthermore, overall motion optimality is improved by periodically executing trajectory planning to select more optimal motions. This approach incorporates inverse kinematics that are robust to base position errors, further enhancing robustness. Evaluation experiments demonstrated that the RLP outperformed existing methods in terms of motion planning time, motion duration, and robustness, confirming its effectiveness in home environments. Moreover, application experiments using a tidy-up task achieved high success rates and short operation times, thereby underscoring its practical feasibility.

  </details>



- **LST-SLAM: A Stereo Thermal SLAM System for Kilometer-Scale Dynamic Environments**  
  Zeyu Jiang, Kuan Xu, Changhao Chen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20925v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Thermal cameras offer strong potential for robot perception under challenging illumination and weather conditions. However, thermal Simultaneous Localization and Mapping (SLAM) remains difficult due to unreliable feature extraction, unstable motion tracking, and inconsistent global pose and map construction, particularly in dynamic large-scale outdoor environments. To address these challenges, we propose LST-SLAM, a novel large-scale stereo thermal SLAM system that achieves robust performance in complex, dynamic scenes. Our approach combines self-supervised thermal feature learning, stereo dual-level motion tracking, and geometric pose optimization. We also introduce a semantic-geometric hybrid constraint that suppresses potentially dynamic features lacking strong inter-frame geometric consistency. Furthermore, we develop an online incremental bag-of-words model for loop closure detection, coupled with global pose optimization to mitigate accumulated drift. Extensive experiments on kilometer-scale dynamic thermal datasets show that LST-SLAM significantly outperforms recent representative SLAM systems, including AirSLAM and DROID-SLAM, in both robustness and accuracy.

  </details>



- **EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations**  
  Luka Šiktar, Branimir Ćaran, Bojan Šekoranja, Marko Švaco  
  _2026-02-24_ · https://arxiv.org/abs/2602.20958v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Search and rescue (SAR) operations require rapid responses to save lives or property. Unmanned Aerial Vehicles (UAVs) equipped with vision-based systems support these missions through prior terrain investigation or real-time assistance during the mission itself. Vision-based UAV frameworks aid human search tasks by detecting and recognizing specific individuals, then tracking and following them while maintaining a safe distance. A key safety requirement for UAV following is the accurate estimation of the distance between camera and target object under real-world conditions, achieved by fusing multiple image modalities. UAVs with deep learning-based vision systems offer a new approach to the planning and execution of SAR operations. As part of the system for automatic people detection and face recognition using deep learning, in this paper we present the fusion of depth camera measurements and monocular camera-to-body distance estimation for robust tracking and following. Deep learning-based filtering of depth camera data and estimation of camera-to-body distance from a monocular camera are achieved with YOLO-pose, enabling real-time fusion of depth information using the Extended Kalman Filter (EKF) algorithm. The proposed subsystem, designed for use in drones, estimates and measures the distance between the depth camera and the human body keypoints, to maintain the safe distance between the drone and the human target. Our system provides an accurate estimated distance, which has been validated against motion capture ground truth data. The system has been tested in real time indoors, where it reduces the average errors, root mean square error (RMSE) and standard deviations of distance estimation up to 15,3\% in three tested scenarios.

  </details>



- **RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction**  
  Yangfan Zhao, Hanwei Zhang, Ke Huang, Qiufeng Wang, Zhenzhou Shao, Dengyu Wu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20807v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: https://ru4d-slam.github.io

  </details>



- **Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty**  
  Xu Wan, Chao Yang, Cheng Yang, Jie Song, Mingyang Sun  
  _2026-02-24_ · https://arxiv.org/abs/2602.20729v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Safe Reinforcement Learning (RL) is crucial for achieving high performance while ensuring safety in real-world applications. However, the complex interplay of multiple uncertainty sources in real environments poses significant challenges for interpretable risk assessment and robust decision-making. To address these challenges, we propose Fuz-RL, a fuzzy measure-guided robust framework for safe RL. Specifically, our framework develops a novel fuzzy Bellman operator for estimating robust value functions using Choquet integrals. Theoretically, we prove that solving the Fuz-RL problem (in Constrained Markov Decision Process (CMDP) form) is equivalent to solving distributionally robust safe RL problems (in robust CMDP form), effectively avoiding min-max optimization. Empirical analyses on safe-control-gym and safety-gymnasium scenarios demonstrate that Fuz-RL effectively integrates with existing safe RL baselines in a model-free manner, significantly improving both safety and control performance under various types of uncertainties in observation, action, and dynamics.

  </details>



- **POMDPPlanners: Open-Source Package for POMDP Planning**  
  Yaacov Pariente, Vadim Indelman  
  _2026-02-24_ · https://arxiv.org/abs/2602.20810v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We present POMDPPlanners, an open-source Python package for empirical evaluation of Partially Observable Markov Decision Process (POMDP) planning algorithms. The package integrates state-of-the-art planning algorithms, a suite of benchmark environments with safety-critical variants, automated hyperparameter optimization via Optuna, persistent caching with failure recovery, and configurable parallel simulation -- reducing the overhead of extensive simulation studies. POMDPPlanners is designed to enable scalable, reproducible research on decision-making under uncertainty, with particular emphasis on risk-sensitive settings where standard toolkits fall short.

  </details>



- **Empirically Calibrated Conditional Independence Tests**  
  Milleno Pan, Antoine de Mathelin, Wesley Tansey  
  _2026-02-24_ · https://arxiv.org/abs/2602.21036v1 · `stat.ME`  
  <details><summary>Abstract</summary>

  Conditional independence tests (CIT) are widely used for causal discovery and feature selection. Even with false discovery rate (FDR) control procedures, they often fail to provide frequentist guarantees in practice. We highlight two common failure modes: (i) in small samples, asymptotic guarantees for many CITs can be inaccurate and even correctly specified models fail to estimate the noise levels and control the error, and (ii) when sample sizes are large but models are misspecified, unaccounted dependencies skew the test's behavior and fail to return uniform p-values under the null. We propose Empirically Calibrated Conditional Independence Tests (ECCIT), a method that measures and corrects for miscalibration. For a chosen base CIT (e.g., GCM, HRT), ECCIT optimizes an adversary that selects features and response functions to maximize a miscalibration metric. ECCIT then fits a monotone calibration map that adjusts the base-test p-values in proportion to the observed miscalibration. Across empirical benchmarks on synthetic and real data, ECCIT achieves valid FDR with higher power than existing calibration strategies while remaining test agnostic.

  </details>



- **Exchangeable Gaussian Processes for Staggered-Adoption Policy Evaluation**  
  Hayk Gevorgyan, Konstantinos Kalogeropoulos, Angelos Alexopoulos  
  _2026-02-24_ · https://arxiv.org/abs/2602.21031v1 · `stat.ME`  
  <details><summary>Abstract</summary>

  We study the use of exchangeable multi-task Gaussian processes (GPs) for causal inference in panel data, applying the framework to two settings: one with a single treated unit subject to a once-and-for-all treatment and another with multiple treated units and staggered treatment adoption. Our approach models the joint evolution of outcomes for treated and control units through a GP prior that ensures exchangeability across units while allowing for flexible nonlinear trends over time. The resulting posterior predictive distribution for the untreated potential outcomes of the treated unit provides a counterfactual path, from which we derive pointwise and cumulative treatment effects, along with credible intervals to quantify uncertainty. We implement several variations of the exchangeable GP model using different kernel functions. To assess prediction accuracy, we conduct a placebo-style validation within the pre-intervention window by selecting a ``fake'' intervention date. Ultimately, this study illustrates how exchangeable GPs serve as a flexible tool for policy evaluation in panel data settings and proposes a novel approach to staggered-adoption designs with a large number of treated and control units.

  </details>



- **VII: Visual Instruction Injection for Jailbreaking Image-to-Video Generation Models**  
  Bowen Zheng, Yongli Xiang, Ziming Hong, Zerong Lin, Chaojian Yu, Tongliang Liu, Xinge You  
  _2026-02-24_ · https://arxiv.org/abs/2602.20999v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Image-to-Video (I2V) generation models, which condition video generation on reference images, have shown emerging visual instruction-following capability, allowing certain visual cues in reference images to act as implicit control signals for video generation. However, this capability also introduces a previously overlooked risk: adversaries may exploit visual instructions to inject malicious intent through the image modality. In this work, we uncover this risk by proposing Visual Instruction Injection (VII), a training-free and transferable jailbreaking framework that intentionally disguises the malicious intent of unsafe text prompts as benign visual instructions in the safe reference image. Specifically, VII coordinates a Malicious Intent Reprogramming module to distill malicious intent from unsafe text prompts while minimizing their static harmfulness, and a Visual Instruction Grounding module to ground the distilled intent onto a safe input image by rendering visual instructions that preserve semantic consistency with the original unsafe text prompt, thereby inducing harmful content during I2V generation. Empirically, our extensive experiments on four state-of-the-art commercial I2V models (Kling-v2.5-turbo, Gemini Veo-3.1, Seedance-1.5-pro, and PixVerse-V5) demonstrate that VII achieves Attack Success Rates of up to 83.5% while reducing Refusal Rates to near zero, significantly outperforming existing baselines.

  </details>



- **ParkDiffusion++: Ego Intention Conditioned Joint Multi-Agent Trajectory Prediction for Automated Parking using Diffusion Models**  
  Jiarong Wei, Anna Rehr, Christian Feist, Abhinav Valada  
  _2026-02-24_ · https://arxiv.org/abs/2602.20923v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Automated parking is a challenging operational domain for advanced driver assistance systems, requiring robust scene understanding and interaction reasoning. The key challenge is twofold: (i) predict multiple plausible ego intentions according to context and (ii) for each intention, predict the joint responses of surrounding agents, enabling effective what-if decision-making. However, existing methods often fall short, typically treating these interdependent problems in isolation. We propose ParkDiffusion++, which jointly learns a multi-modal ego intention predictor and an ego-conditioned multi-agent joint trajectory predictor for automated parking. Our approach makes several key contributions. First, we introduce an ego intention tokenizer that predicts a small set of discrete endpoint intentions from agent histories and vectorized map polylines. Second, we perform ego-intention-conditioned joint prediction, yielding socially consistent predictions of the surrounding agents for each possible ego intention. Third, we employ a lightweight safety-guided denoiser with different constraints to refine joint scenes during training, thus improving accuracy and safety. Fourth, we propose counterfactual knowledge distillation, where an EMA teacher refined by a frozen safety-guided denoiser provides pseudo-targets that capture how agents react to alternative ego intentions. Extensive evaluations demonstrate that ParkDiffusion++ achieves state-of-the-art performance on the Dragon Lake Parking (DLP) dataset and the Intersections Drone (inD) dataset. Importantly, qualitative what-if visualizations show that other agents react appropriately to different ego intentions.

  </details>



- **KCFRC: Kinematic Collision-Aware Foothold Reachability Criteria for Legged Locomotion**  
  Lei Ye, Haibo Gao, Huaiguang Yang, Peng Xu, Haoyu Wang, Tie Liu, Junqi Shan, Zongquan Deng, Liang Ding  
  _2026-02-24_ · https://arxiv.org/abs/2602.20850v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Legged robots face significant challenges in navigating complex environments, as they require precise real-time decisions for foothold selection and contact planning. While existing research has explored methods to select footholds based on terrain geometry or kinematics, a critical gap remains: few existing methods efficiently validate the existence of a non-collision swing trajectory. This paper addresses this gap by introducing KCFRC, a novel approach for efficient foothold reachability analysis. We first formally define the foothold reachability problem and establish a sufficient condition for foothold reachability. Based on this condition, we develop the KCFRC algorithm, which enables robots to validate foothold reachability in real time. Our experimental results demonstrate that KCFRC achieves remarkable time efficiency, completing foothold reachability checks for a single leg across 900 potential footholds in an average of 2 ms. Furthermore, we show that KCFRC can accelerate trajectory optimization and is particularly beneficial for contact planning in confined spaces, enhancing the adaptability and robustness of legged robots in challenging environments.

  </details>



- **Regret-Guided Search Control for Efficient Learning in AlphaZero**  
  Yun-Jui Tsai, Wei-Yu Chen, Yan-Ru Ju, Yu-Hung Chang, Ti-Rong Wu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20809v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) agents achieve remarkable performance but remain far less learning-efficient than humans. While RL agents require extensive self-play games to extract useful signals, humans often need only a few games, improving rapidly by repeatedly revisiting states where mistakes occurred. This idea, known as search control, aims to restart from valuable states rather than always from the initial state. In AlphaZero, prior work Go-Exploit applies this idea by sampling past states from self-play or search trees, but it treats all states equally, regardless of their learning potential. We propose Regret-Guided Search Control (RGSC), which extends AlphaZero with a regret network that learns to identify high-regret states, where the agent's evaluation diverges most from the actual outcome. These states are collected from both self-play trajectories and MCTS nodes, stored in a prioritized regret buffer, and reused as new starting positions. Across 9x9 Go, 10x10 Othello, and 11x11 Hex, RGSC outperforms AlphaZero and Go-Exploit by an average of 77 and 89 Elo, respectively. When training on a well-trained 9x9 Go model, RGSC further improves the win rate against KataGo from 69.3% to 78.2%, while both baselines show no improvement. These results demonstrate that RGSC provides an effective mechanism for search control, improving both efficiency and robustness of AlphaZero training. Our code is available at https://rlg.iis.sinica.edu.tw/papers/rgsc.

  </details>



- **WeirNet: A Large-Scale 3D CFD Benchmark for Geometric Surrogate Modeling of Piano Key Weirs**  
  Lisa Lüddecke, Michael Hohmann, Sebastian Eilermann, Jan Tillmann-Mumm, Pezhman Pourabdollah, Mario Oertel, Oliver Niggemann  
  _2026-02-24_ · https://arxiv.org/abs/2602.20714v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reliable prediction of hydraulic performance is challenging for Piano Key Weir (PKW) design because discharge capacity depends on three-dimensional geometry and operating conditions. Surrogate models can accelerate hydraulic-structure design, but progress is limited by scarce large, well-documented datasets that jointly capture geometric variation, operating conditions, and functional performance. This study presents WeirNet, a large 3D CFD benchmark dataset for geometric surrogate modeling of PKWs. WeirNet contains 3,794 parametric, feasibility-constrained rectangular and trapezoidal PKW geometries, each scheduled at 19 discharge conditions using a consistent free-surface OpenFOAM workflow, resulting in 71,387 completed simulations that form the benchmark and with complete discharge coefficient labels. The dataset is released as multiple modalities compact parametric descriptors, watertight surface meshes and high-resolution point clouds together with standardized tasks and in-distribution and out-of-distribution splits. Representative surrogate families are benchmarked for discharge coefficient prediction. Tree-based regressors on parametric descriptors achieve the best overall accuracy, while point- and mesh-based models remain competitive and offer parameterization-agnostic inference. All surrogates evaluate in milliseconds per sample, providing orders-of-magnitude speedups over CFD runtimes. Out-of-distribution results identify geometry shift as the dominant failure mode compared to unseen discharge values, and data-efficiency experiments show diminishing returns beyond roughly 60% of the training data. By publicly releasing the dataset together with simulation setups and evaluation pipelines, WeirNet establishes a reproducible framework for data-driven hydraulic modeling and enables faster exploration of PKW designs during the early stages of hydraulic planning.

  </details>



- **SD4R: Sparse-to-Dense Learning for 3D Object Detection with 4D Radar**  
  Xiaokai Bai, Jiahao Cheng, Songkai Wang, Yixuan Luo, Lianqing Zheng, Xiaohan Zhang, Si-Yuan Cao, Hui-Liang Shen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20653v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  4D radar measurements offer an affordable and weather-robust solution for 3D perception. However, the inherent sparsity and noise of radar point clouds present significant challenges for accurate 3D object detection, underscoring the need for effective and robust point clouds densification. Despite recent progress, existing densification methods often fail to address the extreme sparsity of 4D radar point clouds and exhibit limited robustness when processing scenes with a small number of points. In this paper, we propose SD4R, a novel framework that transforms sparse radar point clouds into dense representations. SD4R begins by utilizing a foreground point generator (FPG) to mitigate noise propagation and produce densified point clouds. Subsequently, a logit-query encoder (LQE) enhances conventional pillarization, resulting in robust feature representations. Through these innovations, our SD4R demonstrates strong capability in both noise reduction and foreground point densification. Extensive experiments conducted on the publicly available View-of-Delft dataset demonstrate that SD4R achieves state-of-the-art performance. Source code is available at https://github.com/lancelot0805/SD4R.

  </details>


