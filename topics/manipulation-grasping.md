# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-03-07 06:56 UTC_

Total papers shown: **10**


---

- **UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data**  
  Sizhe Yang, Yiman Xie, Zhixuan Liang, Yang Tian, Jia Zeng, Dahua Lin, Jiangmiao Pang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05312v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Grasping is a fundamental capability for robots to interact with the physical world. Humans, equipped with two hands, autonomously select appropriate grasp strategies based on the shape, size, and weight of objects, enabling robust grasping and subsequent manipulation. In contrast, current robotic grasping remains limited, particularly in multi-strategy settings. Although substantial efforts have targeted parallel-gripper and single-hand grasping, dexterous grasping for bimanual robots remains underexplored, with data being a primary bottleneck. Achieving physically plausible and geometrically conforming grasps that can withstand external wrenches poses significant challenges. To address these issues, we introduce UltraDexGrasp, a framework for universal dexterous grasping with bimanual robots. The proposed data-generation pipeline integrates optimization-based grasp synthesis with planning-based demonstration generation, yielding high-quality and diverse trajectories across multiple grasp strategies. With this framework, we curate UltraDexGrasp-20M, a large-scale, multi-strategy grasp dataset comprising 20 million frames across 1,000 objects. Based on UltraDexGrasp-20M, we further develop a simple yet effective grasp policy that takes point clouds as input, aggregates scene features via unidirectional attention, and predicts control commands. Trained exclusively on synthetic data, the policy achieves robust zero-shot sim-to-real transfer and consistently succeeds on novel objects with varied shapes, sizes, and weights, attaining an average success rate of 81.2% in real-world universal dexterous grasping. To facilitate future research on grasping with bimanual robots, we open-source the data generation pipeline at https://github.com/InternRobotics/UltraDexGrasp.

  </details>



- **Digital Twin Driven Textile Classification and Foreign Object Recognition in Automated Sorting Systems**  
  Serkan Ergun, Tobias Mitterer, Hubert Zangl  
  _2026-03-05_ · https://arxiv.org/abs/2603.05230v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The increasing demand for sustainable textile recycling requires robust automation solutions capable of handling deformable garments and detecting foreign objects in cluttered environments. This work presents a digital twin driven robotic sorting system that integrates grasp prediction, multi modal perception, and semantic reasoning for real world textile classification. A dual arm robotic cell equipped with RGBD sensing, capacitive tactile feedback, and collision-aware motion planning autonomously separates garments from an unsorted basket, transfers them to an inspection zone, and classifies them using state of the art Visual Language Models (VLMs). We benchmark nine VLM s from five model families on a dataset of 223 inspection scenarios comprising shirts, socks, trousers, underwear, foreign objects (including garments outside of the aforementioned classes), and empty scenes. The evaluation assesses per class accuracy, hallucination behavior, and computational performance under practical hardware constraints. Results show that the Qwen model family achieves the highest overall accuracy (up to 87.9 %), with strong foreign object detection performance, while lighter models such as Gemma3 offer competitive speed accuracy trade offs for edge deployment. A digital twin combined with MoveIt enables collision aware path planning and integrates segmented 3D point clouds of inspected garments into the virtual environment for improved manipulation reliability. The presented system demonstrates the feasibility of combining semantic VLM reasoning with conventional grasp detection and digital twin technology for scalable, autonomous textile sorting in realistic industrial settings.

  </details>



- **Omni-Manip: Beyond-FOV Large-Workspace Humanoid Manipulation with Omnidirectional 3D Perception**  
  Pei Qu, Zheng Li, Yufei Jia, Ziyun Liu, Liang Zhu, Haoang Li, Jinni Zhou, Jun Ma  
  _2026-03-05_ · https://arxiv.org/abs/2603.05355v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The deployment of humanoid robots for dexterous manipulation in unstructured environments remains challenging due to perceptual limitations that constrain the effective workspace. In scenarios where physical constraints prevent the robot from repositioning itself, maintaining omnidirectional awareness becomes far more critical than color or semantic information. While recent advances in visuomotor policy learning have improved manipulation capabilities, conventional RGB-D solutions suffer from narrow fields of view (FOV) and self-occlusion, requiring frequent base movements that introduce motion uncertainty and safety risks. Existing approaches to expanding perception, including active vision systems and third-view cameras, introduce mechanical complexity, calibration dependencies, and latency that hinder reliable real-time performance. In this work, We propose Omni-Manip, an end-to-end LiDAR-driven 3D visuomotor policy that enables robust manipulation in large workspaces. Our method processes panoramic point clouds through a Time-Aware Attention Pooling mechanism, efficiently encoding sparse 3D data while capturing temporal dependencies. This 360° perception allows the robot to interact with objects across wide areas without frequent repositioning. To support policy learning, we develop a whole-body teleoperation system for efficient data collection on full-body coordination. Extensive experiments in simulation and real-world environments show that Omni-Manip achieves robust performance in large-workspace and cluttered scenarios, outperforming baselines that rely on egocentric depth cameras.

  </details>



- **SPIRIT: Perceptive Shared Autonomy for Robust Robotic Manipulation under Deep Learning Uncertainty**  
  Jongseok Lee, Ribin Balachandran, Harsimran Singh, Jianxiang Feng, Hrishik Mishra, Marco De Stefano, Rudolph Triebel, Alin Albu-Schaeffer, Konstantin Kondak  
  _2026-03-05_ · https://arxiv.org/abs/2603.05111v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deep learning (DL) has enabled impressive advances in robotic perception, yet its limited robustness and lack of interpretability hinder reliable deployment in safety critical applications. We propose a concept termed perceptive shared autonomy, in which uncertainty estimates from DL based perception are used to regulate the level of autonomy. Specifically, when the robot's perception is confident, semi-autonomous manipulation is enabled to improve performance; when uncertainty increases, control transitions to haptic teleoperation for maintaining robustness. In this way, high-performing but uninterpretable DL methods can be integrated safely into robotic systems. A key technical enabler is an uncertainty aware DL based point cloud registration approach based on the so called Neural Tangent Kernels (NTK). We evaluate perceptive shared autonomy on challenging aerial manipulation tasks through a user study of 15 participants and realization of mock-up industrial scenarios, demonstrating reliable robotic manipulation despite failures in DL based perception. The resulting system, named SPIRIT, improves both manipulation performance and system reliability. SPIRIT was selected as a finalist of a major industrial innovation award.

  </details>



- **cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots**  
  Balakumar Sundaralingam, Adithyavairavan Murali, Stan Birchfield  
  _2026-03-05_ · https://arxiv.org/abs/2603.05493v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Effective robot autonomy requires motion generation that is safe, feasible, and reactive. Current methods are fragmented: fast planners output physically unexecutable trajectories, reactive controllers struggle with high-fidelity perception, and existing solvers fail on high-DoF systems. We present cuRoboV2, a unified framework with three key innovations: (1) B-spline trajectory optimization that enforces smoothness and torque limits; (2) a GPU-native TSDF/ESDF perception pipeline that generates dense signed distance fields covering the full workspace, unlike existing methods that only provide distances within sparsely allocated blocks, up to 10x faster and in 8x less memory than the state-of-the-art at manipulation scale, with up to 99% collision recall; and (3) scalable GPU-native whole-body computation, namely topology-aware kinematics, differentiable inverse dynamics, and map-reduce self-collision, that achieves up to 61x speedup while also extending to high-DoF humanoids (where previous GPU implementations fail). On benchmarks, cuRoboV2 achieves 99.7% success under 3kg payload (where baselines achieve only 72--77%), 99.6% collision-free IK on a 48-DoF humanoid (where prior methods fail entirely), and 89.5% retargeting constraint satisfaction (vs. 61% for PyRoki); these collision-free motions yield locomotion policies with 21% lower tracking error than PyRoki and 12x lower cross-seed variance than mink. A ground-up codebase redesign for discoverability enabled LLM coding assistants to author up to 73% of new modules, including hand-optimized CUDA kernels, demonstrating that well-structured robotics code can unlock productive human--LLM collaboration. Together, these advances provide a unified, dynamics-aware motion generation stack that scales from single-arm manipulators to full humanoids.

  </details>



- **GaussTwin: Unified Simulation and Correction with Gaussian Splatting for Robotic Digital Twins**  
  Yichen Cai, Paul Jansonnie, Cristiana de Farias, Oleg Arenz, Jan Peters  
  _2026-03-05_ · https://arxiv.org/abs/2603.05108v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Digital twins promise to enhance robotic manipulation by maintaining a consistent link between real-world perception and simulation. However, most existing systems struggle with the lack of a unified model, complex dynamic interactions, and the real-to-sim gap, which limits downstream applications such as model predictive control. Thus, we propose GaussTwin, a real-time digital twin that combines position-based dynamics with discrete Cosserat rod formulations for physically grounded simulation, and Gaussian splatting for efficient rendering and visual correction. By anchoring Gaussians to physical primitives and enforcing coherent SE(3) updates driven by photometric error and segmentation masks, GaussTwin achieves stable prediction-correction while preserving physical fidelity. Through experiments in both simulation and on a Franka Research 3 platform, we show that GaussTwin consistently improves tracking accuracy and robustness compared to shape-matching and rigid-only baselines, while also enabling downstream tasks such as push-based planning. These results highlight GaussTwin as a step toward unified, physically meaningful digital twins that can support closed-loop robotic interaction and learning.

  </details>



- **The Spatial and Temporal Resolution of Motor Intention in Multi-Target Prediction**  
  Marie Dominique Schmidt, Ioannis Iossifidis  
  _2026-03-05_ · https://arxiv.org/abs/2603.05418v1 · `q-bio.NC`  
  <details><summary>Abstract</summary>

  Reaching for grasping, and manipulating objects are essential motor functions in everyday life. Decoding human motor intentions is a central challenge for rehabilitation and assistive technologies. This study focuses on predicting intentions by inferring movement direction and target location from multichannel electromyography (EMG) signals, and investigating how spatially and temporally accurate such information can be detected relative to movement onset. We present a computational pipeline that combines data-driven temporal segmentation with classical and deep learning classifiers in order to analyse EMG data recorded during the planning, early execution, and target contact phases of a delayed reaching task. Early intention prediction enables devices to anticipate user actions, improving responsiveness and supporting active motor recovery in adaptive rehabilitation systems. Random Forest achieves $80\%$ accuracy and Convolutional Neural Network $75\%$ accuracy across $25$ spatial targets, each separated by $14^\circ$ azimuth/altitude. Furthermore, a systematic evaluation of EMG channels, feature sets, and temporal windows demonstrates that motor intention can be efficiently decoded even with drastically reduced data. This work sheds light on the temporal and spatial evolution of motor intention, paving the way for anticipatory control in adaptive rehabilitation systems and driving advancements in computational approaches to motor neuroscience.

  </details>



- **Critic in the Loop: A Tri-System VLA Framework for Robust Long-Horizon Manipulation**  
  Pengfei Yi, Yingjie Ma, Wenjiang Xu, Yanan Hao, Shuai Gan, Wanting Li, Shanlin Zhong  
  _2026-03-05_ · https://arxiv.org/abs/2603.05185v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Balancing high-level semantic reasoning with low-level reactive control remains a core challenge in visual robotic manipulation. While Vision-Language Models (VLMs) excel at cognitive planning, their inference latency precludes real-time execution. Conversely, fast Vision-Language-Action (VLA) models often lack the semantic depth required for complex, long-horizon tasks. To bridge this gap, we introduce Critic in the Loop, an adaptive hierarchical framework driven by dynamic VLM-Expert scheduling. At its core is a bionic Tri-System architecture comprising a VLM brain for global reasoning, a VLA cerebellum for reactive execution, and a lightweight visual Critic. By continuously monitoring the workspace, the Critic dynamically routes control authority. It sustains rapid closed-loop execution via the VLA for routine subtasks, and adaptively triggers the VLM for replanning upon detecting execution anomalies such as task stagnation or failures. Furthermore, our architecture seamlessly integrates human-inspired rules to intuitively break infinite retry loops. This visually-grounded scheduling minimizes expensive VLM queries, while substantially enhancing system robustness and autonomy in out-of-distribution (OOD) scenarios. Comprehensive experiments on challenging, long-horizon manipulation benchmarks reveal that our approach achieves state-of-the-art performance.

  </details>



- **Iterative On-Policy Refinement of Hierarchical Diffusion Policies for Language-Conditioned Manipulation**  
  Clemence Grislain, Olivier Sigaud, Mohamed Chetouani  
  _2026-03-05_ · https://arxiv.org/abs/2603.05291v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Hierarchical policies for language-conditioned manipulation decompose tasks into subgoals, where a high-level planner guides a low-level controller. However, these hierarchical agents often fail because the planner generates subgoals without considering the actual limitations of the controller. Existing solutions attempt to bridge this gap via intermediate modules or shared representations, but they remain limited by their reliance on fixed offline datasets. We propose HD-ExpIt, a framework for iterative fine-tuning of hierarchical diffusion policies via environment feedback. HD-ExpIt organizes training into a self-reinforcing cycle: it utilizes diffusion-based planning to autonomously discover successful behaviors, which are then distilled back into the hierarchical policy. This loop enables both components to improve while implicitly grounding the planner in the controller's actual capabilities without requiring explicit proxy models. Empirically, HD-ExpIt significantly improves hierarchical policies trained solely on offline data, achieving state-of-the-art performance on the long-horizon CALVIN benchmark among methods trained from scratch.

  </details>



- **SeedPolicy: Horizon Scaling via Self-Evolving Diffusion Policy for Robot Manipulation**  
  Youqiang Gui, Yuxuan Zhou, Shen Cheng, Xinyang Yuan, Haoqiang Fan, Peng Cheng, Shuaicheng Liu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05117v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Imitation Learning (IL) enables robots to acquire manipulation skills from expert demonstrations. Diffusion Policy (DP) models multi-modal expert behaviors but suffers performance degradation as observation horizons increase, limiting long-horizon manipulation. We propose Self-Evolving Gated Attention (SEGA), a temporal module that maintains a time-evolving latent state via gated attention, enabling efficient recurrent updates that compress long-horizon observations into a fixed-size representation while filtering irrelevant temporal information. Integrating SEGA into DP yields Self-Evolving Diffusion Policy (SeedPolicy), which resolves the temporal modeling bottleneck and enables scalable horizon extension with moderate overhead. On the RoboTwin 2.0 benchmark with 50 manipulation tasks, SeedPolicy outperforms DP and other IL baselines. Averaged across both CNN and Transformer backbones, SeedPolicy achieves 36.8% relative improvement in clean settings and 169% relative improvement in randomized challenging settings over the DP. Compared to vision-language-action models such as RDT with 1.2B parameters, SeedPolicy achieves competitive performance with one to two orders of magnitude fewer parameters, demonstrating strong efficiency and scalability. These results establish SeedPolicy as a state-of-the-art imitation learning method for long-horizon robotic manipulation. Code is available at: https://github.com/Youqiang-Gui/SeedPolicy.

  </details>


