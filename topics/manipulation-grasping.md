# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-03-12 07:09 UTC_

Total papers shown: **8**


---

- **Learning Bimanual Cloth Manipulation with Vision-based Tactile Sensing via Single Robotic Arm**  
  Dongmyoung Lee, Wei Chen, Xiaoshuai Chen, Rui Zong, Petar Kormushev  
  _2026-03-11_ · https://arxiv.org/abs/2603.10609v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic cloth manipulation remains challenging due to the high-dimensional state space of fabrics, their deformable nature, and frequent occlusions that limit vision-based sensing. Although dual-arm systems can mitigate some of these issues, they increase hardware and control complexity. This paper presents Touch G.O.G., a compact vision-based tactile gripper and perception/control framework for single-arm bimanual cloth manipulation. The proposed framework combines three key components: (1) a novel gripper design and control strategy for in-gripper cloth sliding with a single robot arm, (2) a Vision Foundation Model-backboned Vision Transformer pipeline for cloth part classification (PC-Net) and edge pose estimation (PE-Net) using real and synthetic tactile images, and (3) an encoder-decoder synthetic data generator (SD-Net) that reduces manual annotation by producing high-fidelity tactile images. Experiments show 96% accuracy in distinguishing edges, corners, interior regions, and grasp failures, together with sub-millimeter edge localization and 4.5° orientation error. Real-world results demonstrate reliable cloth unfolding, even for crumpled fabrics, using only a single robotic arm. These results highlight Touch G.O.G. as a compact and cost-effective solution for deformable object manipulation.

  </details>



- **Learning Adaptive Force Control for Contact-Rich Sample Scraping with Heterogeneous Materials**  
  Cenk Cetin, Shreyas Pouli, Gabriella Pizzuto  
  _2026-03-11_ · https://arxiv.org/abs/2603.10979v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The increasing demand for accelerated scientific discovery, driven by global challenges, highlights the need for advanced AI-driven robotics. Deploying robotic chemists in human-centric labs is key for the next horizon of autonomous discovery, as complex tasks still demand the dexterity of human scientists. Robotic manipulation in this context is uniquely challenged by handling diverse chemicals (granular, powdery, or viscous liquids), under varying lab conditions. For example, humans use spatulas for scraping materials from vial walls. Automating this process is challenging because it goes beyond simple robotic insertion tasks and traditional lab automation, requiring the execution of fine-granular movements within a constrained environment (the sample vial). Our work proposes an adaptive control framework to address this, relying on a low-level Cartesian impedance controller for stable and compliant physical interaction and a high-level reinforcement learning agent that learns to dynamically adjust interaction forces at the end-effector. The agent is guided by perception feedback, which provides the material's location. We first created a task-representative simulation environment with a Franka Research 3 robot, a scraping tool, and a sample vial containing heterogeneous materials. To facilitate the learning of an adaptive policy and model diverse characteristics, the sample is modelled as a collection of spheres, where each sphere is assigned a unique dislodgement force threshold, which is procedurally generated using Perlin noise. We train an agent to autonomously learn and adapt the optimal contact wrench for a sample scraping task in simulation and then successfully transfer this policy to a real robotic setup. Our method was evaluated across five different material setups, outperforming a fixed-wrench baseline by an average of 10.9%.

  </details>



- **AdaClearGrasp: Learning Adaptive Clearing for Zero-Shot Robust Dexterous Grasping in Densely Cluttered Environments**  
  Zixuan Chen, Wenquan Zhang, Jing Fang, Ruiming Zeng, Zhixuan Xu, Yiwen Hou, Xinke Wang, Jieqi Shi, Jing Huo, Yang Gao  
  _2026-03-11_ · https://arxiv.org/abs/2603.10616v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In densely cluttered environments, physical interference, visual occlusions, and unstable contacts often cause direct dexterous grasping to fail, while aggressive singulation strategies may compromise safety. Enabling robots to adaptively decide whether to clear surrounding objects or directly grasp the target is therefore crucial for robust manipulation. We propose AdaClearGrasp, a closed-loop decision-execution framework for adaptive clearing and zero-shot dexterous grasping in densely cluttered environments. The framework formulates manipulation as a controllable high-level decision process that determines whether to directly grasp the target or first clear surrounding objects. A pretrained vision-language model (VLM) interprets visual observations and language task descriptions to reason about grasp interference and generate a high-level planning skeleton, which invokes structured atomic skills through a unified action interface. For dexterous grasping, we train a reinforcement learning policy with a relative hand-object distance representation, enabling zero-shot generalization across diverse object geometries and physical properties. During execution, visual feedback monitors outcomes and triggers replanning upon failures, forming a closed-loop correction mechanism. To evaluate language-conditioned dexterous grasping in clutter, we introduce Clutter-Bench, the first simulation benchmark with graded clutter complexity. It includes seven target objects across three clutter levels, yielding 210 task scenarios. We further perform sim-to-real experiments on three objects under three clutter levels (18 scenarios). Results demonstrate that AdaClearGrasp significantly improves grasp success rates in densely cluttered environments. For more videos and code, please visit our project website: https://chenzixuan99.github.io/adaclear-grasp.github.io/.

  </details>



- **BinWalker: Development and Field Evaluation of a Quadruped Manipulator Platform for Sustainable Litter Collection**  
  Giulio Turrisi, Angelo Bratta, Giovanni Minelli, Gabriel Fischer Abati, Amir H. Rad, João Carlos Virgolino Soares, Claudio Semini  
  _2026-03-11_ · https://arxiv.org/abs/2603.10529v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Litter pollution represents a growing environmental problem affecting natural and urban ecosystems worldwide. Waste discarded in public spaces often accumulates in areas that are difficult to access, such as uneven terrains, coastal environments, parks, and roadside vegetation. Over time, these materials degrade and release harmful substances, including toxic chemicals and microplastics, which can contaminate soil and water and pose serious threats to wildlife and human health. Despite increasing awareness of the problem, litter collection is still largely performed manually by human operators, making large-scale cleanup operations labor-intensive, time-consuming, and costly. Robotic solutions have the potential to support and partially automate environmental cleanup tasks. In this work, we present a quadruped robotic system designed for autonomous litter collection in challenging outdoor scenarios. The robot combines the mobility advantages of legged locomotion with a manipulation system consisting of a robotic arm and an onboard litter container. This configuration enables the robot to detect, grasp, and store litter items while navigating through uneven terrains. The proposed system aims to demonstrate the feasibility of integrating perception, locomotion, and manipulation on a legged robotic platform for environmental cleanup tasks. Experimental evaluations conducted in outdoor scenarios highlight the effectiveness of the approach and its potential for assisting large-scale litter removal operations in environments that are difficult to reach with traditional robotic platforms. The code associated with this work can be found at: https://github.com/iit-DLSLab/trash-collection-isaaclab.

  </details>



- **FG-CLTP: Fine-Grained Contrastive Language Tactile Pretraining for Robotic Manipulation**  
  Wenxuan Ma, Chaofan Zhang, Yinghao Cai, Guocai Yao, Shaowei Cui, Shuo Wang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10871v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advancements in integrating tactile sensing into vision-language-action (VLA) models have demonstrated transformative potential for robotic perception. However, existing tactile representations predominantly rely on qualitative descriptors (e.g., texture), neglecting quantitative contact states such as force magnitude, contact geometry, and principal axis orientation, which are indispensable for fine-grained manipulation. To bridge this gap, we propose FG-CLTP, a fine-grained contrastive language tactile pretraining framework. We first introduce a novel dataset comprising over 100k tactile 3D point cloud-language pairs that explicitly capture multidimensional contact states from the sensor's perspective. We then implement a discretized numerical tokenization mechanism to achieve quantitative-semantic alignment, effectively injecting explicit physical metrics into the multimodal feature space. The proposed FG-CLTP model yields a 95.9% classification accuracy and reduces the regression error (MAE) by 52.6% compared to state-of-the-art methods. Furthermore, the integration of 3D point cloud representations establishes a sensor-agnostic foundation with a minimal sim-to-real gap of 3.5%. Building upon this fine-grained representation, we develop a 3D tactile-language-action (3D-TLA) architecture driven by a flow matching policy to enable multimodal reasoning and control. Extensive experiments demonstrate that our framework significantly outperforms strong baselines in contact-rich manipulation tasks, providing a robust and generalizable foundation for tactile-language-action models.

  </details>



- **TacLoc: Global Tactile Localization on Objects from a Registration Perspective**  
  Zirui Zhang, Boyang Zhang, Fumin Zhang, Huan Yin  
  _2026-03-11_ · https://arxiv.org/abs/2603.10565v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Pose estimation is essential for robotic manipulation, particularly when visual perception is occluded during gripper-object interactions. Existing tactile-based methods generally rely on tactile simulation or pre-trained models, which limits their generalizability and efficiency. In this study, we propose TacLoc, a novel tactile localization framework that formulates the problem as a one-shot point cloud registration task. TacLoc introduces a graph-theoretic partial-to-full registration method, leveraging dense point clouds and surface normals from tactile sensing for efficient and accurate pose estimation. Without requiring rendered data or pre-trained models, TacLoc achieves improved performance through normal-guided graph pruning and a hypothesis-and-verification pipeline. TacLoc is evaluated extensively on the YCB dataset. We further demonstrate TacLoc on real-world objects across two different visual-tactile sensors.

  </details>



- **FutureVLA: Joint Visuomotor Prediction for Vision-Language-Action Model**  
  Xiaoxu Xu, Hao Li, Jinhui Ye, Yilun Chen, Jia Zeng, Xinyi Chen, Linning Xu, Dahua Lin, Weixin Li, Jiangmiao Pang  
  _2026-03-11_ · https://arxiv.org/abs/2603.10712v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Predictive foresight is important to intelligent embodied agents. Since the motor execution of a robot is intrinsically constrained by its visual perception of environmental geometry, effectively anticipating the future requires capturing this tightly coupled visuomotor interplay. While recent vision-language-action models attempt to incorporate future guidance, they struggle with this joint modeling. Existing explicit methods divert capacity to task-irrelevant visual details, whereas implicit methods relying on sparse frame pairs disrupt temporal continuity. By heavily relying on visual reconstruction, these methods become visually dominated, entangling static scene context with dynamic action intent. We argue that effective joint visuomotor predictive modeling requires both temporal continuity and visually-conditioned supervision decoupling. To this end, we propose FutureVLA, featuring a novel Joint Visuomotor Predictive Architecture. FutureVLA is designed to extract joint visuomotor embeddings by first decoupling visual and motor information, and then jointly encoding generalized physical priors. Specifically, in the pretraining stage, we leverage heterogeneous manipulation datasets and introduce a Joint Visuomotor Gating mechanism to structurally separate visual state preservation from temporal action modeling. It allows the motor stream to focus on continuous physical dynamics while explicitly querying visual tokens for environmental constraints, yielding highly generalizable joint visuomotor embeddings. Subsequently, in the post-training stage, we employ a latent embeddings alignment strategy, enabling diverse downstream VLA models to internalize these temporal priors without modifying their inference architectures. Extensive experiments demonstrate that FutureVLA consistently improves VLA frameworks.

  </details>



- **DepthCache: Depth-Guided Training-Free Visual Token Merging for Vision-Language-Action Model Inference**  
  Yuquan Li, Lianjie Ma, Han Ding, Lijun Zhu  
  _2026-03-11_ · https://arxiv.org/abs/2603.10469v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models enable generalist robotic manipulation but suffer from high inference latency. This bottleneck stems from the massive number of visual tokens processed by large language backbones. Existing methods either prune or merge tokens uniformly, degrading the spatial reasoning essential for robotic control. We present DepthCache, a training-free framework that leverages depth as a structural prior for visual token compression. It partitions observations into depth-based regions and applies spatially differentiated merge ratios, preserving the near-field workspace while compressing the distant background. To exploit temporal redundancy, DepthCache distributes the merging process across consecutive frames, ensuring consistent representations while reducing per-step computation. A motion-adaptive pipeline further optimizes auxiliary view compression based on end-effector dynamics. The framework requires no model modification, generalizing across diverse VLA architectures. On the LIBERO benchmark, DepthCache achieves up to 1.28x inference speedup with less than 1% average success rate degradation across three VLA models (pi_0.5, OpenVLA, GR00T), whereas pruning and merging baselines incur 4--24% degradation at comparable compression. Real-world experiments on a physical manipulator demonstrate that DepthCache enables faster task throughput and more responsive closed-loop control in latency-sensitive scenarios.

  </details>


