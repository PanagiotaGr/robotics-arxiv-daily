# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-03-12 07:09 UTC_

Total papers shown: **10**


---

- **RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion**  
  Andrea Patrizi, Carlo Rizzardo, Arturo Laurenzi, Francesco Ruscelli, Luca Rossini, Nikos G. Tsagarakis  
  _2026-03-11_ · https://arxiv.org/abs/2603.10878v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We propose a contact-explicit hierarchical architecture coupling Reinforcement Learning (RL) and Model Predictive Control (MPC), where a high-level RL agent provides gait and navigation commands to a low-level locomotion MPC. This offloads the combinatorial burden of contact timing from the MPC by learning acyclic gaits through trial and error in simulation. We show that only a minimal set of rewards and limited tuning are required to obtain effective policies. We validate the architecture in simulation across robotic platforms spanning 50 kg to 120 kg and different MPC implementations, observing the emergence of acyclic gaits and timing adaptations in flat-terrain legged and hybrid locomotion, and further demonstrating extensibility to non-flat terrains. Across all platforms, we achieve zero-shot sim-to-sim transfer without domain randomization, and we further demonstrate zero-shot sim-to-real transfer without domain randomization on Centauro, our 120 kg wheeled-legged humanoid robot. We make our software framework and evaluation results publicly available at https://github.com/AndrePatri/AugMPC.

  </details>



- **Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming**  
  Yilin Zou, Zhong Zhang, Fanghua Jiang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10711v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Real-time trajectory optimization for nonlinear constrained autonomous systems is critical and typically performed by CPU-based sequential solvers. Specifically, reliance on global sparse linear algebra or the serial nature of dynamic programming algorithms restricts the utilization of massively parallel computing architectures like GPUs. To bridge this gap, we introduce a fully GPU-native trajectory optimization framework that combines sequential convex programming with a consensus-based alternating direction method of multipliers. By applying a temporal splitting strategy, our algorithm decouples the optimization horizon into independent, per-node subproblems that execute massively in parallel. The entire process runs fully on the GPU, eliminating costly memory transfers and large-scale sparse factorizations. This architecture naturally scales to multi-trajectory optimization. We validate the solver on a quadrotor agile flight task and a Mars powered descent problem using an on-board edge computing platform. Benchmarks reveal a sustained 4x throughput speedup and a 51% reduction in energy consumption over a heavily optimized 12-core CPU baseline. Crucially, the framework saturates the hardware, maintaining over 96% active GPU utilization to achieve planning rates exceeding 100 Hz. Furthermore, we demonstrate the solver's extensibility to robust Model Predictive Control by jointly optimizing dynamically coupled scenarios under stochastic disturbances, enabling scalable and safe autonomy.

  </details>



- **Path Planning for Sound Speed Profile Estimation**  
  Ludvig Lindström, Tadas Paskevicius, Andreas Jakobsson, Isaac Skog  
  _2026-03-11_ · https://arxiv.org/abs/2603.10585v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Accurate estimation of the sound speed profile (SSP) is essential for underwater acoustic communication, sonar performance, and navigation, as the acoustic wave propagation depends strongly on the SSP. This work considers SSP estimation in a region of interest using an autonomous underwater vehicle (AUV) equipped with a conductivity-temperature-depth (CTD) sensor and an acoustic receiver measuring transmission loss (TL) from a sonar transmitter. The SSP is modeled using a linear basis-function expansion and is sequentially estimated with an unscented Kalman filter that fuses local CTD measurements with TL measurements. A receding-horizon path planning scheme is also employed to select future AUV positions by minimizing the predicted total sound speed variance. Simulations using the Bellhop acoustic wave propagation solver show that CTD measurements provide accurate local SSP estimates, whereas TL measurements are seen to capture the global characteristics of the SSP, with their joint use improving the reconstruction of both local variations and large-scale SSP behavior. The results also indicate that the proposed path planning strategy reduces the estimation uncertainty compared to constant-velocity motion, thereby enabling improved environmental characterization for underwater acoustic systems.

  </details>



- **GRACE: A Unified 2D Multi-Robot Path Planning Simulator & Benchmark for Grid, Roadmap, And Continuous Environments**  
  Chuanlong Zang, Anna Mannucci, Isabelle Barz, Philipp Schillinger, Florian Lier, Wolfgang Hönig  
  _2026-03-11_ · https://arxiv.org/abs/2603.10858v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Advancing Multi-Agent Pathfinding (MAPF) and Multi-Robot Motion Planning (MRMP) requires platforms that enable transparent, reproducible comparisons across modeling choices. Existing tools either scale under simplifying assumptions (grids, homogeneous agents) or offer higher fidelity with less comparable instrumentation. We present GRACE, a unified 2D simulator+benchmark that instantiates the same task at multiple abstraction levels (grid, roadmap, continuous) via explicit, reproducible operators and a common evaluation protocol. Our empirical results on public maps and representative planners enable commensurate comparisons on a shared instance set. Furthermore, we quantify the expected representation-fidelity trade-offs (MRMP solves instances at higher fidelity but lower speed, while grid/roadmap planners scale farther). By consolidating representation, execution, and evaluation, GRACE thereby aims to make cross-representation studies more comparable and provides a means to advance multi-robot planning research and its translation to practice.

  </details>



- **Interleaving Scheduling and Motion Planning with Incremental Learning of Symbolic Space-Time Motion Abstractions**  
  Elisa Tosello, Arthur Bit-Monnot, Davide Lusuardi, Alessandro Valentini, Andrea Micheli  
  _2026-03-11_ · https://arxiv.org/abs/2603.10651v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Task and Motion Planning combines high-level task sequencing (what to do) with low-level motion planning (how to do it) to generate feasible, collision-free execution plans. However, in many real-world domains, such as automated warehouses, tasks are predefined, shifting the challenge to if, when, and how to execute them safely and efficiently under resource, time and motion constraints. In this paper, we formalize this as the Scheduling and Motion Planning problem for multi-object navigation in shared workspaces. We propose a novel solution framework that interleaves off-the-shelf schedulers and motion planners in an incremental learning loop. The scheduler generates candidate plans, while the motion planner checks feasibility and returns symbolic feedback, i.e., spatial conflicts and timing adjustments, to guide the scheduler towards motion-feasible solutions. We validate our proposal on logistics and job-shop scheduling benchmarks augmented with motion tasks, using state-of-the-art schedulers and sampling-based motion planners. Our results show the effectiveness of our framework in generating valid plans under complex temporal and spatial constraints, where synchronized motion is critical.

  </details>



- **OnFly: Onboard Zero-Shot Aerial Vision-Language Navigation toward Safety and Efficiency**  
  Guiyong Zheng, Yueting Ban, Mingjie Zhang, Juepeng Zheng, Boyu Zhou  
  _2026-03-11_ · https://arxiv.org/abs/2603.10682v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Aerial vision-language navigation (AVLN) enables UAVs to follow natural-language instructions in complex 3D environments. However, existing zero-shot AVLN methods often suffer from unstable single-stream Vision-Language Model decision-making, unreliable long-horizon progress monitoring, and a trade-off between safety and efficiency. We propose OnFly, a fully onboard, real-time framework for zero-shot AVLN. OnFly adopts a shared-perception dual-agent architecture that decouples high-frequency target generation from low-frequency progress monitoring, thereby stabilizing decision-making. It further employs a hybrid keyframe-recent-frame memory to preserve global trajectory context while maintaining KV-cache prefix stability, enabling reliable long-horizon monitoring with termination and recovery signals. In addition, a semantic-geometric verifier refines VLM-predicted targets for instruction consistency and geometric safety using VLM features and depth cues, while a receding-horizon planner generates optimized collision-free trajectories under geometric safety constraints, improving both safety and efficiency. In simulation, OnFly improves task success from 26.4% to 67.8%, compared with the strongest state-of-the-art baseline, while fully onboard real-world flights validate its feasibility for real-time deployment. The code will be released at https://github.com/Robotics-STAR-Lab/OnFly

  </details>



- **CUPID: A Plug-in Framework for Joint Aleatoric and Epistemic Uncertainty Estimation with a Single Model**  
  Xinran Xu, Xiuyi Fan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10745v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Accurate estimation of uncertainty in deep learning is critical for deploying models in high-stakes domains such as medical diagnosis and autonomous decision-making, where overconfident predictions can lead to harmful outcomes. In practice, understanding the reason behind a model's uncertainty and the type of uncertainty it represents can support risk-aware decisions, enhance user trust, and guide additional data collection. However, many existing methods only address a single type of uncertainty or require modifications and retraining of the base model, making them difficult to adopt in real-world systems. We introduce CUPID (Comprehensive Uncertainty Plug-in estImation moDel), a general-purpose module that jointly estimates aleatoric and epistemic uncertainty without modifying or retraining the base model. CUPID can be flexibly inserted into any layer of a pretrained network. It models aleatoric uncertainty through a learned Bayesian identity mapping and captures epistemic uncertainty by analyzing the model's internal responses to structured perturbations. We evaluate CUPID across a range of tasks, including classification, regression, and out-of-distribution detection. The results show that it consistently delivers competitive performance while offering layer-wise insights into the origins of uncertainty. By making uncertainty estimation modular, interpretable, and model-agnostic, CUPID supports more transparent and trustworthy AI. Related code and data are available at https://github.com/a-Fomalhaut-a/CUPID.

  </details>



- **MAVEN: A Meta-Reinforcement Learning Framework for Varying-Dynamics Expertise in Agile Quadrotor Maneuvers**  
  Jin Zhou, Dongcheng Cao, Xian Wang, Shuo Li  
  _2026-03-11_ · https://arxiv.org/abs/2603.10714v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has emerged as a powerful paradigm for achieving online agile navigation with quadrotors. Despite this success, policies trained via standard RL typically fail to generalize across significant dynamic variations, exhibiting a critical lack of adaptability. This work introduces MAVEN, a meta-RL framework that enables a single policy to achieve robust end-to-end navigation across a wide range of quadrotor dynamics. Our approach features a novel predictive context encoder, which learns to infer a latent representation of the system dynamics from interaction history. We demonstrate our method in agile waypoint traversal tasks under two challenging scenarios: large variations in quadrotor mass and severe single-rotor thrust loss. We leverage a GPU-vectorized simulator to distribute tasks across thousands of parallel environments, overcoming the long training times of meta-RL to converge in less than an hour. Through extensive experiments in both simulation and the real world, we validate that MAVEN achieves superior adaptation and agility. The policy successfully executes zero-shot sim-to-real transfer, demonstrating robust online adaptation by performing high-speed maneuvers despite mass variations of up to 66.7% and single-rotor thrust losses as severe as 70%.

  </details>



- **WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation**  
  Rafi Ibn Sultan, Hui Zhu, Xiangyu Zhou, Chengyin Li, Prashant Khanduri, Marco Brocanelli, Dongxiao Zhu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10703v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Ensuring accessible pedestrian navigation requires reasoning about both semantic and spatial aspects of complex urban scenes, a challenge that existing Large Vision-Language Models (LVLMs) struggle to meet. Although these models can describe visual content, their lack of explicit grounding leads to object hallucinations and unreliable depth reasoning, limiting their usefulness for accessibility guidance. We introduce WalkGPT, a pixel-grounded LVLM for the new task of Grounded Navigation Guide, unifying language reasoning and segmentation within a single architecture for depth-aware accessibility guidance. Given a pedestrian-view image and a navigation query, WalkGPT generates a conversational response with segmentation masks that delineate accessible and harmful features, along with relative depth estimation. The model incorporates a Multi-Scale Query Projector (MSQP) that shapes the final image tokens by aggregating them along text tokens across spatial hierarchies, and a Calibrated Text Projector (CTP), guided by a proposed Region Alignment Loss, that maps language embeddings into segmentation-aware representations. These components enable fine-grained grounding and depth inference without user-provided cues or anchor points, allowing the model to generate complete and realistic navigation guidance. We also introduce PAVE, a large-scale benchmark of 41k pedestrian-view images paired with accessibility-aware questions and depth-grounded answers. Experiments show that WalkGPT achieves strong grounded reasoning and segmentation performance. The source code and dataset are available on the \href{https://sites.google.com/view/walkgpt-26/home}{project website}.

  </details>



- **AdaClearGrasp: Learning Adaptive Clearing for Zero-Shot Robust Dexterous Grasping in Densely Cluttered Environments**  
  Zixuan Chen, Wenquan Zhang, Jing Fang, Ruiming Zeng, Zhixuan Xu, Yiwen Hou, Xinke Wang, Jieqi Shi, Jing Huo, Yang Gao  
  _2026-03-11_ · https://arxiv.org/abs/2603.10616v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In densely cluttered environments, physical interference, visual occlusions, and unstable contacts often cause direct dexterous grasping to fail, while aggressive singulation strategies may compromise safety. Enabling robots to adaptively decide whether to clear surrounding objects or directly grasp the target is therefore crucial for robust manipulation. We propose AdaClearGrasp, a closed-loop decision-execution framework for adaptive clearing and zero-shot dexterous grasping in densely cluttered environments. The framework formulates manipulation as a controllable high-level decision process that determines whether to directly grasp the target or first clear surrounding objects. A pretrained vision-language model (VLM) interprets visual observations and language task descriptions to reason about grasp interference and generate a high-level planning skeleton, which invokes structured atomic skills through a unified action interface. For dexterous grasping, we train a reinforcement learning policy with a relative hand-object distance representation, enabling zero-shot generalization across diverse object geometries and physical properties. During execution, visual feedback monitors outcomes and triggers replanning upon failures, forming a closed-loop correction mechanism. To evaluate language-conditioned dexterous grasping in clutter, we introduce Clutter-Bench, the first simulation benchmark with graded clutter complexity. It includes seven target objects across three clutter levels, yielding 210 task scenarios. We further perform sim-to-real experiments on three objects under three clutter levels (18 scenarios). Results demonstrate that AdaClearGrasp significantly improves grasp success rates in densely cluttered environments. For more videos and code, please visit our project website: https://chenzixuan99.github.io/adaclear-grasp.github.io/.

  </details>


