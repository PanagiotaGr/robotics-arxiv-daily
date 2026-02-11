# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-11 07:16 UTC_

Total papers shown: **17**


---

- **Learning Force-Regulated Manipulation with a Low-Cost Tactile-Force-Controlled Gripper**  
  Xuhui Kang, Tongxuan Tian, Sung-Wook Lee, Binghao Huang, Yunzhu Li, Yen-Ling Kuo  
  _2026-02-10_ · https://arxiv.org/abs/2602.10013v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Successfully manipulating many everyday objects, such as potato chips, requires precise force regulation. Failure to modulate force can lead to task failure or irreversible damage to the objects. Humans can precisely achieve this by adapting force from tactile feedback, even within a short period of physical contact. We aim to give robots this capability. However, commercial grippers exhibit high cost or high minimum force, making them unsuitable for studying force-controlled policy learning with everyday force-sensitive objects. We introduce TF-Gripper, a low-cost (~$150) force-controlled parallel-jaw gripper that integrates tactile sensing as feedback. It has an effective force range of 0.45-45N and is compatible with different robot arms. Additionally, we designed a teleoperation device paired with TF-Gripper to record human-applied grasping forces. While standard low-frequency policies can be trained on this data, they struggle with the reactive, contact-dependent nature of force regulation. To overcome this, we propose RETAF (REactive Tactile Adaptation of Force), a framework that decouples grasping force control from arm pose prediction. RETAF regulates force at high frequency using wrist images and tactile feedback, while a base policy predicts end-effector pose and gripper open/close action. We evaluate TF-Gripper and RETAF across five real-world tasks requiring precise force regulation. Results show that compared to position control, direct force control significantly improves grasp stability and task performance. We further show that tactile feedback is essential for force regulation, and that RETAF consistently outperforms baselines and can be integrated with various base policies. We hope this work opens a path for scaling the learning of force-controlled policies in robotic manipulation. Project page: https://force-gripper.github.io .

  </details>



- **TaCo: A Benchmark for Lossless and Lossy Codecs of Heterogeneous Tactile Data**  
  Zhengxue Cheng, Yan Zhao, Keyu Wang, Hengdi Zhang, Li Song  
  _2026-02-10_ · https://arxiv.org/abs/2602.09893v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Tactile sensing is crucial for embodied intelligence, providing fine-grained perception and control in complex environments. However, efficient tactile data compression, which is essential for real-time robotic applications under strict bandwidth constraints, remains underexplored. The inherent heterogeneity and spatiotemporal complexity of tactile data further complicate this challenge. To bridge this gap, we introduce TaCo, the first comprehensive benchmark for Tactile data Codecs. TaCo evaluates 30 compression methods, including off-the-shelf compression algorithms and neural codecs, across five diverse datasets from various sensor types. We systematically assess both lossless and lossy compression schemes on four key tasks: lossless storage, human visualization, material and object classification, and dexterous robotic grasping. Notably, we pioneer the development of data-driven codecs explicitly trained on tactile data, TaCo-LL (lossless) and TaCo-L (lossy). Results have validated the superior performance of our TaCo-LL and TaCo-L. This benchmark provides a foundational framework for understanding the critical trade-offs between compression efficiency and task performance, paving the way for future advances in tactile perception.

  </details>



- **Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction**  
  Sizhe Yang, Linning Xu, Hao Li, Juncheng Mu, Jia Zeng, Dahua Lin, Jiangmiao Pang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10101v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  3D spatial perception is fundamental to generalizable robotic manipulation, yet obtaining reliable, high-quality 3D geometry remains challenging. Depth sensors suffer from noise and material sensitivity, while existing reconstruction models lack the precision and metric consistency required for physical interaction. We introduce Robo3R, a feed-forward, manipulation-ready 3D reconstruction model that predicts accurate, metric-scale scene geometry directly from RGB images and robot states in real time. Robo3R jointly infers scale-invariant local geometry and relative camera poses, which are unified into the scene representation in the canonical robot frame via a learned global similarity transformation. To meet the precision demands of manipulation, Robo3R employs a masked point head for sharp, fine-grained point clouds, and a keypoint-based Perspective-n-Point (PnP) formulation to refine camera extrinsics and global alignment. Trained on Robo3R-4M, a curated large-scale synthetic dataset with four million high-fidelity annotated frames, Robo3R consistently outperforms state-of-the-art reconstruction methods and depth sensors. Across downstream tasks including imitation learning, sim-to-real transfer, grasp synthesis, and collision-free motion planning, we observe consistent gains in performance, suggesting the promise of this alternative 3D sensing module for robotic manipulation.

  </details>



- **RoboSubtaskNet: Temporal Sub-task Segmentation for Human-to-Robot Skill Transfer in Real-World Environments**  
  Dharmendra Sharma, Archit Sharma, John Reberio, Vaibhav Kesharwani, Peeyush Thakur, Narendra Kumar Dhar, Laxmidhar Behera  
  _2026-02-10_ · https://arxiv.org/abs/2602.10015v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Temporally locating and classifying fine-grained sub-task segments in long, untrimmed videos is crucial to safe human-robot collaboration. Unlike generic activity recognition, collaborative manipulation requires sub-task labels that are directly robot-executable. We present RoboSubtaskNet, a multi-stage human-to-robot sub-task segmentation framework that couples attention-enhanced I3D features (RGB plus optical flow) with a modified MS-TCN employing a Fibonacci dilation schedule to capture better short-horizon transitions such as reach-pick-place. The network is trained with a composite objective comprising cross-entropy and temporal regularizers (truncated MSE and a transition-aware term) to reduce over-segmentation and to encourage valid sub-task progressions. To close the gap between vision benchmarks and control, we introduce RoboSubtask, a dataset of healthcare and industrial demonstrations annotated at the sub-task level and designed for deterministic mapping to manipulator primitives. Empirically, RoboSubtaskNet outperforms MS-TCN and MS-TCN++ on GTEA and our RoboSubtask benchmark (boundary-sensitive and sequence metrics), while remaining competitive on the long-horizon Breakfast benchmark. Specifically, RoboSubtaskNet attains F1 @ 50 = 79.5%, Edit = 88.6%, Acc = 78.9% on GTEA; F1 @ 50 = 30.4%, Edit = 52.0%, Acc = 53.5% on Breakfast; and F1 @ 50 = 94.2%, Edit = 95.6%, Acc = 92.2% on RoboSubtask. We further validate the full perception-to-execution pipeline on a 7-DoF Kinova Gen3 manipulator, achieving reliable end-to-end behavior in physical trials (overall task success approx 91.25%). These results demonstrate a practical path from sub-task level video understanding to deployed robotic manipulation in real-world settings.

  </details>



- **AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception**  
  Ruoxuan Feng, Yuxuan Zhou, Siyu Mei, Dongzhan Zhou, Pengwei Wang, Shaowei Cui, Bin Fang, Guocai Yao, Di Hu  
  _2026-02-10_ · https://arxiv.org/abs/2602.09617v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Real-world contact-rich manipulation demands robots to perceive temporal tactile feedback, capture subtle surface deformations, and reason about object properties as well as force dynamics. Although optical tactile sensors are uniquely capable of providing such rich information, existing tactile datasets and models remain limited. These resources primarily focus on object-level attributes (e.g., material) while largely overlooking fine-grained tactile temporal dynamics during physical interactions. We consider that advancing dynamic tactile perception requires a systematic hierarchy of dynamic perception capabilities to guide both data collection and model design. To address the lack of tactile data with rich dynamic information, we present ToucHD, a large-scale hierarchical tactile dataset spanning tactile atomic actions, real-world manipulations, and touch-force paired data. Beyond scale, ToucHD establishes a comprehensive tactile dynamic data ecosystem that explicitly supports hierarchical perception capabilities from the data perspective. Building on it, we propose AnyTouch 2, a general tactile representation learning framework for diverse optical tactile sensors that unifies object-level understanding with fine-grained, force-aware dynamic perception. The framework captures both pixel-level and action-specific deformations across frames, while explicitly modeling physical force dynamics, thereby learning multi-level dynamic perception capabilities from the model perspective. We evaluate our model on benchmarks that covers static object properties and dynamic physical attributes, as well as real-world manipulation tasks spanning multiple tiers of dynamic perception capabilities-from basic object-level understanding to force-aware dexterous manipulation. Experimental results demonstrate consistent and strong performance across sensors and tasks.

  </details>



- **Sample-Efficient Real-World Dexterous Policy Fine-Tuning via Action-Chunked Critics and Normalizing Flows**  
  Chenyu Yang, Denis Tarasov, Davide Liconti, Hehui Zheng, Robert K. Katzschmann  
  _2026-02-10_ · https://arxiv.org/abs/2602.09580v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Real-world fine-tuning of dexterous manipulation policies remains challenging due to limited real-world interaction budgets and highly multimodal action distributions. Diffusion-based policies, while expressive, do not permit conservative likelihood-based updates during fine-tuning because action probabilities are intractable. In contrast, conventional Gaussian policies collapse under multimodality, particularly when actions are executed in chunks, and standard per-step critics fail to align with chunked execution, leading to poor credit assignment. We present SOFT-FLOW, a sample-efficient off-policy fine-tuning framework with normalizing flow (NF) to address these challenges. The normalizing flow policy yields exact likelihoods for multimodal action chunks, allowing conservative, stable policy updates through likelihood regularization and thereby improving sample efficiency. An action-chunked critic evaluates entire action sequences, aligning value estimation with the policy's temporal structure and improving long-horizon credit assignment. To our knowledge, this is the first demonstration of a likelihood-based, multimodal generative policy combined with chunk-level value learning on real robotic hardware. We evaluate SOFT-FLOW on two challenging dexterous manipulation tasks in the real world: cutting tape with scissors retrieved from a case, and in-hand cube rotation with a palm-down grasp -- both of which require precise, dexterous control over long horizons. On these tasks, SOFT-FLOW achieves stable, sample-efficient adaptation where standard methods struggle.

  </details>



- **Instruct2Act: From Human Instruction to Actions Sequencing and Execution via Robot Action Network for Robotic Manipulation**  
  Archit Sharma, Dharmendra Sharma, John Rebeiro, Peeyush Thakur, Narendra Dhar, Laxmidhar Behera  
  _2026-02-10_ · https://arxiv.org/abs/2602.09940v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots often struggle to follow free-form human instructions in real-world settings due to computational and sensing limitations. We address this gap with a lightweight, fully on-device pipeline that converts natural-language commands into reliable manipulation. Our approach has two stages: (i) the instruction to actions module (Instruct2Act), a compact BiLSTM with a multi-head-attention autoencoder that parses an instruction into an ordered sequence of atomic actions (e.g., reach, grasp, move, place); and (ii) the robot action network (RAN), which uses the dynamic adaptive trajectory radial network (DATRN) together with a vision-based environment analyzer (YOLOv8) to generate precise control trajectories for each sub-action. The entire system runs on a modest system with no cloud services. On our custom proprietary dataset, Instruct2Act attains 91.5% sub-actions prediction accuracy while retaining a small footprint. Real-robot evaluations across four tasks (pick-place, pick-pour, wipe, and pick-give) yield an overall 90% success; sub-action inference completes in < 3.8s, with end-to-end executions in 30-60s depending on task complexity. These results demonstrate that fine-grained instruction-to-action parsing, coupled with DATRN-based trajectory generation and vision-guided grounding, provides a practical path to deterministic, real-time manipulation in resource-constrained, single-camera settings.

  </details>



- **TriPilot-FF: Coordinated Whole-Body Teleoperation with Force Feedback**  
  Zihao Li, Yanan Zhou, Ranpeng Qiu, Hangyu Wu, Guoqiang Ren, Weiming Zhi  
  _2026-02-10_ · https://arxiv.org/abs/2602.09888v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mobile manipulators broaden the operational envelope for robot manipulation. However, the whole-body teleoperation of such robots remains a problem: operators must coordinate a wheeled base and two arms while reasoning about obstacles and contact. Existing interfaces are predominantly hand-centric (e.g., VR controllers and joysticks), leaving foot-operated channels underexplored for continuous base control. We present TriPilot-FF, an open-source whole-body teleoperation system for a custom bimanual mobile manipulator that introduces a foot-operated pedal with lidar-driven pedal haptics, coupled with upper-body bimanual leader-follower teleoperation. Using only a low-cost base-mounted lidar, TriPilot-FF renders a resistive pedal cue from proximity-to-obstacle signals in the commanded direction, shaping operator commands toward collision-averse behaviour without an explicit collision-avoidance controller. The system also supports arm-side force reflection for contact awareness and provides real-time force and visual guidance of bimanual manipulability to prompt mobile base repositioning, thereby improving reach. We demonstrate the capability of TriPilot-FF to effectively ``co-pilot'' the human operator over long time-horizons and tasks requiring precise mobile base movement and coordination. Finally, we incorporate teleoperation feedback signals into an Action Chunking with Transformers (ACT) policy and demonstrate improved performance when the additional information is available. We release the pedal device design, full software stack, and conduct extensive real-world evaluations on a bimanual wheeled platform. The project page of TriPilot-FF is http://bit.ly/46H3ZJT.

  </details>



- **DexImit: Learning Bimanual Dexterous Manipulation from Monocular Human Videos**  
  Juncheng Mu, Sizhe Yang, Yiming Bao, Hojin Bae, Tianming Wei, Linning Xu, Boyi Li, Huazhe Xu, Jiangmiao Pang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10105v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Data scarcity fundamentally limits the generalization of bimanual dexterous manipulation, as real-world data collection for dexterous hands is expensive and labor-intensive. Human manipulation videos, as a direct carrier of manipulation knowledge, offer significant potential for scaling up robot learning. However, the substantial embodiment gap between human hands and robotic dexterous hands makes direct pretraining from human videos extremely challenging. To bridge this gap and unleash the potential of large-scale human manipulation video data, we propose DexImit, an automated framework that converts monocular human manipulation videos into physically plausible robot data, without any additional information. DexImit employs a four-stage generation pipeline: (1) reconstructing hand-object interactions from arbitrary viewpoints with near-metric scale; (2) performing subtask decomposition and bimanual scheduling; (3) synthesizing robot trajectories consistent with the demonstrated interactions; (4) comprehensive data augmentation for zero-shot real-world deployment. Building on these designs, DexImit can generate large-scale robot data based on human videos, either from the Internet or video generation models. DexImit is capable of handling diverse manipulation tasks, including tool use (e.g., cutting an apple), long-horizon tasks (e.g., making a beverage), and fine-grained manipulations (e.g., stacking cups).

  </details>



- **UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking**  
  Baijun Chen, Weijie Wan, Tianxing Chen, Xianda Guo, Congsheng Xu, Yuanyang Qi, Haojie Zhang, Longyan Wu, Tianling Xu, Zixuan Li, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.10093v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic manipulation has seen rapid progress with vision-language-action (VLA) policies. However, visuo-tactile perception is critical for contact-rich manipulation, as tasks such as insertion are difficult to complete robustly using vision alone. At the same time, acquiring large-scale and reliable tactile data in the physical world remains costly and challenging, and the lack of a unified evaluation platform further limits policy learning and systematic analysis. To address these challenges, we propose UniVTAC, a simulation-based visuo-tactile data synthesis platform that supports three commonly used visuo-tactile sensors and enables scalable and controllable generation of informative contact interactions. Based on this platform, we introduce the UniVTAC Encoder, a visuo-tactile encoder trained on large-scale simulation-synthesized data with designed supervisory signals, providing tactile-centric visuo-tactile representations for downstream manipulation tasks. In addition, we present the UniVTAC Benchmark, which consists of eight representative visuo-tactile manipulation tasks for evaluating tactile-driven policies. Experimental results show that integrating the UniVTAC Encoder improves average success rates by 17.1% on the UniVTAC Benchmark, while real-world robotic experiments further demonstrate a 25% improvement in task success. Our webpage is available at https://univtac.github.io/.

  </details>



- **EgoHumanoid: Unlocking In-the-Wild Loco-Manipulation with Robot-Free Egocentric Demonstration**  
  Modi Shi, Shijia Peng, Jin Chen, Haoran Jiang, Yinghui Li, Di Huang, Ping Luo, Hongyang Li, Li Chen  
  _2026-02-10_ · https://arxiv.org/abs/2602.10106v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Human demonstrations offer rich environmental diversity and scale naturally, making them an appealing alternative to robot teleoperation. While this paradigm has advanced robot-arm manipulation, its potential for the more challenging, data-hungry problem of humanoid loco-manipulation remains largely unexplored. We present EgoHumanoid, the first framework to co-train a vision-language-action policy using abundant egocentric human demonstrations together with a limited amount of robot data, enabling humanoids to perform loco-manipulation across diverse real-world environments. To bridge the embodiment gap between humans and robots, including discrepancies in physical morphology and viewpoint, we introduce a systematic alignment pipeline spanning from hardware design to data processing. A portable system for scalable human data collection is developed, and we establish practical collection protocols to improve transferability. At the core of our human-to-humanoid alignment pipeline lies two key components. The view alignment reduces visual domain discrepancies caused by camera height and perspective variation. The action alignment maps human motions into a unified, kinematically feasible action space for humanoid control. Extensive real-world experiments demonstrate that incorporating robot-free egocentric data significantly outperforms robot-only baselines by 51\%, particularly in unseen environments. Our analysis further reveals which behaviors transfer effectively and the potential for scaling human data.

  </details>



- **RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation**  
  Hao Li, Ziqin Wang, Zi-han Ding, Shuai Yang, Yilun Chen, Yang Tian, Xiaolin Hu, Tai Wang, Dahua Lin, Feng Zhao, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09973v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Advances in large vision-language models (VLMs) have stimulated growing interest in vision-language-action (VLA) systems for robot manipulation. However, existing manipulation datasets remain costly to curate, highly embodiment-specific, and insufficient in coverage and diversity, thereby hindering the generalization of VLA models. Recent approaches attempt to mitigate these limitations via a plan-then-execute paradigm, where high-level plans (e.g., subtasks, trace) are first generated and subsequently translated into low-level actions, but they critically rely on extra intermediate supervision, which is largely absent from existing datasets. To bridge this gap, we introduce the RoboInter Manipulation Suite, a unified resource including data, benchmarks, and models of intermediate representations for manipulation. It comprises RoboInter-Tool, a lightweight GUI that enables semi-automatic annotation of diverse representations, and RoboInter-Data, a large-scale dataset containing over 230k episodes across 571 diverse scenes, which provides dense per-frame annotations over more than 10 categories of intermediate representations, substantially exceeding prior work in scale and annotation quality. Building upon this foundation, RoboInter-VQA introduces 9 spatial and 20 temporal embodied VQA categories to systematically benchmark and enhance the embodied reasoning capabilities of VLMs. Meanwhile, RoboInter-VLA offers an integrated plan-then-execute framework, supporting modular and end-to-end VLA variants that bridge high-level planning with low-level execution via intermediate supervision. In total, RoboInter establishes a practical foundation for advancing robust and generalizable robotic learning via fine-grained and diverse intermediate representations.

  </details>



- **VideoWorld 2: Learning Transferable Knowledge from Real-world Videos**  
  Zhongwei Ren, Yunchao Wei, Xiao Yu, Guixun Luo, Yao Zhao, Bingyi Kang, Jiashi Feng, Xiaojie Jin  
  _2026-02-10_ · https://arxiv.org/abs/2602.10102v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Learning transferable knowledge from unlabeled video data and applying it in new environments is a fundamental capability of intelligent agents. This work presents VideoWorld 2, which extends VideoWorld and offers the first investigation into learning transferable knowledge directly from raw real-world videos. At its core, VideoWorld 2 introduces a dynamic-enhanced Latent Dynamics Model (dLDM) that decouples action dynamics from visual appearance: a pretrained video diffusion model handles visual appearance modeling, enabling the dLDM to learn latent codes that focus on compact and meaningful task-related dynamics. These latent codes are then modeled autoregressively to learn task policies and support long-horizon reasoning. We evaluate VideoWorld 2 on challenging real-world handcraft making tasks, where prior video generation and latent-dynamics models struggle to operate reliably. Remarkably, VideoWorld 2 achieves up to 70% improvement in task success rate and produces coherent long execution videos. In robotics, we show that VideoWorld 2 can acquire effective manipulation knowledge from the Open-X dataset, which substantially improves task performance on CALVIN. This study reveals the potential of learning transferable world knowledge directly from raw videos, with all code, data, and models to be open-sourced for further research.

  </details>



- **BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation**  
  Yucheng Hu, Jianke Zhang, Yuanfei Luo, Yanjiang Guo, Xiaoyu Chen, Xinshu Sun, Kun Feng, Qingzhou Lu, Sheng Chen, Yangang Zhang, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09849v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Equipping embodied agents with the ability to reason about tasks, foresee physical outcomes, and generate precise actions is essential for general-purpose manipulation. While recent Vision-Language-Action (VLA) models have leveraged pre-trained foundation models, they typically focus on either linguistic planning or visual forecasting in isolation. These methods rarely integrate both capabilities simultaneously to guide action generation, leading to suboptimal performance in complex, long-horizon manipulation tasks. To bridge this gap, we propose BagelVLA, a unified model that integrates linguistic planning, visual forecasting, and action generation within a single framework. Initialized from a pretrained unified understanding and generative model, BagelVLA is trained to interleave textual reasoning and visual prediction directly into the action execution loop. To efficiently couple these modalities, we introduce Residual Flow Guidance (RFG), which initializes from current observation and leverages single-step denoising to extract predictive visual features, guiding action generation with minimal latency. Extensive experiments demonstrate that BagelVLA outperforms existing baselines by a significant margin on multiple simulated and real-world benchmarks, particularly in tasks requiring multi-stage reasoning.

  </details>



- **A Survey on STAR-RIS Enabled Joint Communications and Sensing: Fundamentals, Recent Advances and Research Challenges**  
  Wali Ullah Khan, Chandan Kumar Sheemar, Syed Tariq Shah, Manzoor Ahmed, Symeon Chatzinotas  
  _2026-02-10_ · https://arxiv.org/abs/2602.09589v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  The joint communications and sensing (JCAS) paradigm is envisioned as a core capability of sixth-generation (6G) wireless networks, enabling the integration of data communication and environmental sensing within a unified system. By reusing spectrum, waveforms, and hardware resources, JCAS improves spectral efficiency, reduces system complexity, and hardware cost, while enabling new use cases. Nevertheless, the realization of JCAS is hindered by inherent trade-offs between communication and sensing objectives, limited controllability of wireless propagation, and stringent hardware and design constraints. Simultaneously transmitting and reflecting reconfigurable intelligent surfaces (STAR-RIS) have recently emerged as a promising technology to address these challenges by enabling full-space programmable manipulation of electromagnetic waves. This survey provides a systematic and in-depth review of STAR-RIS-enabled JCAS systems. Specifically, we first introduce the fundamental principles of JCAS and STAR-RIS. We then classify and review the state-of-the-art research on STAR-RIS-assisted JCAS from multiple perspectives, encompassing system architectures, waveform and beamforming design, resource allocation, optimization frameworks, and learning-based control. Finally, we identify key open challenges that remain unsolved and outline promising future research directions toward intelligent, flexible, and perceptive 6G wireless networks.

  </details>



- **Preference Aligned Visuomotor Diffusion Policies for Deformable Object Manipulation**  
  Marco Moletta, Michael C. Welle, Danica Kragic  
  _2026-02-10_ · https://arxiv.org/abs/2602.09583v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Humans naturally develop preferences for how manipulation tasks should be performed, which are often subtle, personal, and difficult to articulate. Although it is important for robots to account for these preferences to increase personalization and user satisfaction, they remain largely underexplored in robotic manipulation, particularly in the context of deformable objects like garments and fabrics. In this work, we study how to adapt pretrained visuomotor diffusion policies to reflect preferred behaviors using limited demonstrations. We introduce RKO, a novel preference-alignment method that combines the benefits of two recent frameworks: RPO and KTO. We evaluate RKO against common preference learning frameworks, including these two, as well as a baseline vanilla diffusion policy, on real-world cloth-folding tasks spanning multiple garments and preference settings. We show that preference-aligned policies (particularly RKO) achieve superior performance and sample efficiency compared to standard diffusion policy fine-tuning. These results highlight the importance and feasibility of structured preference learning for scaling personalized robot behavior in complex deformable object manipulation tasks.

  </details>



- **Mitigating the Likelihood Paradox in Flow-based OOD Detection via Entropy Manipulation**  
  Donghwan Kim, Hyunsoo Yoon  
  _2026-02-10_ · https://arxiv.org/abs/2602.09581v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Deep generative models that can tractably compute input likelihoods, including normalizing flows, often assign unexpectedly high likelihoods to out-of-distribution (OOD) inputs. We mitigate this likelihood paradox by manipulating input entropy based on semantic similarity, applying stronger perturbations to inputs that are less similar to an in-distribution memory bank. We provide a theoretical analysis showing that entropy control increases the expected log-likelihood gap between in-distribution and OOD samples in favor of the in-distribution, and we explain why the procedure works without any additional training of the density model. We then evaluate our method against likelihood-based OOD detectors on standard benchmarks and find consistent AUROC improvements over baselines, supporting our explanation.

  </details>


