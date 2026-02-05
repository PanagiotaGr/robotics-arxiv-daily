# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-05 07:13 UTC_

Total papers shown: **10**


---

- **From Vision to Assistance: Gaze and Vision-Enabled Adaptive Control for a Back-Support Exoskeleton**  
  Alessandro Leanza, Paolo Franceschi, Blerina Spahiu, Loris Roveda  
  _2026-02-04_ · https://arxiv.org/abs/2602.04648v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Back-support exoskeletons have been proposed to mitigate spinal loading in industrial handling, yet their effectiveness critically depends on timely and context-aware assistance. Most existing approaches rely either on load-estimation techniques (e.g., EMG, IMU) or on vision systems that do not directly inform control. In this work, we present a vision-gated control framework for an active lumbar occupational exoskeleton that leverages egocentric vision with wearable gaze tracking. The proposed system integrates real-time grasp detection from a first-person YOLO-based perception system, a finite-state machine (FSM) for task progression, and a variable admittance controller to adapt torque delivery to both posture and object state. A user study with 15 participants performing stooping load lifting trials under three conditions (no exoskeleton, exoskeleton without vision, exoskeleton with vision) shows that vision-gated assistance significantly reduces perceived physical demand and improves fluency, trust, and comfort. Quantitative analysis reveals earlier and stronger assistance when vision is enabled, while questionnaire results confirm user preference for the vision-gated mode. These findings highlight the potential of egocentric vision to enhance the responsiveness, ergonomics, safety, and acceptance of back-support exoskeletons.

  </details>



- **Act, Sense, Act: Learning Non-Markovian Active Perception Strategies from Large-Scale Egocentric Human Data**  
  Jialiang Li, Yi Qiao, Yunhan Guo, Changwen Chen, Wenzhao Lian  
  _2026-02-04_ · https://arxiv.org/abs/2602.04600v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Achieving generalizable manipulation in unconstrained environments requires the robot to proactively resolve information uncertainty, i.e., the capability of active perception. However, existing methods are often confined in limited types of sensing behaviors, restricting their applicability to complex environments. In this work, we formalize active perception as a non-Markovian process driven by information gain and decision branching, providing a structured categorization of visual active perception paradigms. Building on this perspective, we introduce CoMe-VLA, a cognitive and memory-aware vision-language-action (VLA) framework that leverages large-scale human egocentric data to learn versatile exploration and manipulation priors. Our framework integrates a cognitive auxiliary head for autonomous sub-task transitions and a dual-track memory system to maintain consistent self and environmental awareness by fusing proprioceptive and visual temporal contexts. By aligning human and robot hand-eye coordination behaviors in a unified egocentric action space, we train the model progressively in three stages. Extensive experiments on a wheel-based humanoid have demonstrated strong robustness and adaptability of our proposed method across diverse long-horizon tasks spanning multiple active perception scenarios.

  </details>



- **EgoActor: Grounding Task Planning into Spatial-aware Egocentric Actions for Humanoid Robots via Visual-Language Models**  
  Yu Bai, MingMing Yu, Chaojie Li, Ziyi Bai, Xinlong Wang, Börje F. Karlsson  
  _2026-02-04_ · https://arxiv.org/abs/2602.04515v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deploying humanoid robots in real-world settings is fundamentally challenging, as it demands tight integration of perception, locomotion, and manipulation under partial-information observations and dynamically changing environments. As well as transitioning robustly between sub-tasks of different types. Towards addressing these challenges, we propose a novel task - EgoActing, which requires directly grounding high-level instructions into various, precise, spatially aware humanoid actions. We further instantiate this task by introducing EgoActor, a unified and scalable vision-language model (VLM) that can predict locomotion primitives (e.g., walk, turn, move sideways, change height), head movements, manipulation commands, and human-robot interactions to coordinate perception and execution in real-time. We leverage broad supervision over egocentric RGB-only data from real-world demonstrations, spatial reasoning question-answering, and simulated environment demonstrations, enabling EgoActor to make robust, context-aware decisions and perform fluent action inference (under 1s) with both 8B and 4B parameter models. Extensive evaluations in both simulated and real-world environments demonstrate that EgoActor effectively bridges abstract task planning and concrete motor execution, while generalizing across diverse tasks and unseen environments.

  </details>



- **AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation**  
  Jin-Chuan Shi, Binhong Ye, Tao Liu, Junzhe He, Yangjinhui Xu, Xiaoyang Liu, Zeju Li, Hao Chen, Chunhua Shen  
  _2026-02-04_ · https://arxiv.org/abs/2602.04672v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior art frequently collapses. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.

  </details>



- **Capturing Visual Environment Structure Correlates with Control Performance**  
  Jiahua Dong, Yunze Man, Pavel Tokmakov, Yu-Xiong Wang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04880v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The choice of visual representation is key to scaling generalist robot policies. However, direct evaluation via policy rollouts is expensive, even in simulation. Existing proxy metrics focus on the representation's capacity to capture narrow aspects of the visual world, like object shape, limiting generalization across environments. In this paper, we take an analytical perspective: we probe pretrained visual encoders by measuring how well they support decoding of environment state -- including geometry, object structure, and physical attributes -- from images. Leveraging simulation environments with access to ground-truth state, we show that this probing accuracy strongly correlates with downstream policy performance across diverse environments and learning settings, significantly outperforming prior metrics and enabling efficient representation selection. More broadly, our study provides insight into the representational properties that support generalizable manipulation, suggesting that learning to encode the latent physical state of the environment is a promising objective for control.

  </details>



- **El Agente Estructural: An Artificially Intelligent Molecular Editor**  
  Changhyeok Choi, Yunheng Zou, Marcel Müller, Han Hao, Yeonghun Kang, Juan B. Pérez-Sánchez, Ignacio Gustin, Hanyong Xu, Mohammad Ghazi Vakili, Chris Crebolder, et al.  
  _2026-02-04_ · https://arxiv.org/abs/2602.04849v1 · `physics.chem-ph`  
  <details><summary>Abstract</summary>

  We present El Agente Estructural, a multimodal, natural-language-driven geometry-generation and manipulation agent for autonomous chemistry and molecular modelling. Unlike molecular generation or editing via generative models, Estructural mimics how human experts directly manipulate molecular systems in three dimensions by integrating a comprehensive set of domain-informed tools and vision-language models. This design enables precise control over atomic or functional group replacements, atomic connectivity, and stereochemistry without the need to rebuild extensive core molecular frameworks. Through a series of representative case studies, we demonstrate that Estructural enables chemically meaningful geometry manipulation across a wide range of real-world scenarios. These include site-selective functionalization, ligand binding, ligand exchange, stereochemically controlled structure construction, isomer interconversion, fragment-level structural analysis, image-guided generation of structures from schematic reaction mechanisms, and mechanism-driven geometry generation and modification. These examples illustrate how multimodal reasoning, when combined with specialized geometry-aware tools, supports interactive and context-aware molecular modelling beyond structure generation. Looking forward, the integration of Estructural into El Agente Quntur, an autonomous multi-agent quantum chemistry platform, enhances its capabilities by adding sophisticated tools for the generation and editing of three-dimensional structures.

  </details>



- **PuppetAI: A Customizable Platform for Designing Tactile-Rich Affective Robot Interaction**  
  Jiaye Li, Tongshun Chen, Siyi Ma, Elizabeth Churchill, Ke Wu  
  _2026-02-04_ · https://arxiv.org/abs/2602.04787v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  We introduce PuppetAI, a modular soft robot interaction platform. This platform offers a scalable cable-driven actuation system and a customizable, puppet-inspired robot gesture framework, supporting a multitude of interaction gesture robot design formats. The platform comprises a four-layer decoupled software architecture that includes perceptual processing, affective modeling, motion scheduling, and low-level actuation. We also implemented an affective expression loop that connects human input to the robot platform by producing real-time emotional gestural responses to human vocal input. For our own designs, we have worked with nuanced gestures enacted by "soft robots" with enhanced dexterity and "pleasant-to-touch" plush exteriors. By reducing operational complexity and production costs while enhancing customizability, our work creates an adaptable and accessible foundation for future tactile-based expressive robot research. Our goal is to provide a platform that allows researchers to independently construct or refine highly specific gestures and movements performed by social robots.

  </details>



- **A Unified Complementarity-based Approach for Rigid-Body Manipulation and Motion Prediction**  
  Bingkun Huang, Xin Ma, Nilanjan Chakraborty, Riddhiman Laha  
  _2026-02-04_ · https://arxiv.org/abs/2602.04522v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic manipulation in unstructured environments requires planners to reason jointly about free-space motion and sustained, frictional contact with the environment. Existing (local) planning and simulation frameworks typically separate these regimes or rely on simplified contact representations, particularly when modeling non-convex or distributed contact patches. Such approximations limit the fidelity of contact-mode transitions and hinder the robust execution of contact-rich behaviors in real time. This paper presents a unified discrete-time modeling framework for robotic manipulation that consistently captures both free motion and frictional contact within a single mathematical formalism (Unicomp). Building on complementarity-based rigid-body dynamics, we formulate free-space motion and contact interactions as coupled linear and nonlinear complementarity problems, enabling principled transitions between contact modes without enforcing fixed-contact assumptions. For planar patch contact, we derive a frictional contact model from the maximum power dissipation principle in which the set of admissible contact wrenches is represented by an ellipsoidal limit surface. This representation captures coupled force-moment effects, including torsional friction, while remaining agnostic to the underlying pressure distribution across the contact patch. The resulting formulation yields a discrete-time predictive model that relates generalized velocities and contact wrenches through quadratic constraints and is suitable for real-time optimization-based planning. Experimental results show that the proposed approach enables stable, physically consistent behavior at interactive speeds across tasks, from planar pushing to contact-rich whole-body maneuvers.

  </details>



- **SynthVerse: A Large-Scale Diverse Synthetic Dataset for Point Tracking**  
  Weiguang Zhao, Haoran Xu, Xingyu Miao, Qin Zhao, Rui Zhang, Kaizhu Huang, Ning Gao, Peizhou Cao, Mingze Sun, Mulin Yu, et al.  
  _2026-02-04_ · https://arxiv.org/abs/2602.04441v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Point tracking aims to follow visual points through complex motion, occlusion, and viewpoint changes, and has advanced rapidly with modern foundation models. Yet progress toward general point tracking remains constrained by limited high-quality data, as existing datasets often provide insufficient diversity and imperfect trajectory annotations. To this end, we introduce SynthVerse, a large-scale, diverse synthetic dataset specifically designed for point tracking. SynthVerse includes several new domains and object types missing from existing synthetic datasets, such as animated-film-style content, embodied manipulation, scene navigation, and articulated objects. SynthVerse substantially expands dataset diversity by covering a broader range of object categories and providing high-quality dynamic motions and interactions, enabling more robust training and evaluation for general point tracking. In addition, we establish a highly diverse point tracking benchmark to systematically evaluate state-of-the-art methods under broader domain shifts. Extensive experiments and analyses demonstrate that training with SynthVerse yields consistent improvements in generalization and reveal limitations of existing trackers under diverse settings.

  </details>



- **Integrated Exploration and Sequential Manipulation on Scene Graph with LLM-based Situated Replanning**  
  Heqing Yang, Ziyuan Jiao, Shu Wang, Yida Niu, Si Liu, Hangxin Liu  
  _2026-02-04_ · https://arxiv.org/abs/2602.04419v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In partially known environments, robots must combine exploration to gather information with task planning for efficient execution. To address this challenge, we propose EPoG, an Exploration-based sequential manipulation Planning framework on Scene Graphs. EPoG integrates a graph-based global planner with a Large Language Model (LLM)-based situated local planner, continuously updating a belief graph using observations and LLM predictions to represent known and unknown objects. Action sequences are generated by computing graph edit operations between the goal and belief graphs, ordered by temporal dependencies and movement costs. This approach seamlessly combines exploration and sequential manipulation planning. In ablation studies across 46 realistic household scenes and 5 long-horizon daily object transportation tasks, EPoG achieved a success rate of 91.3%, reducing travel distance by 36.1% on average. Furthermore, a physical mobile manipulator successfully executed complex tasks in unknown and dynamic environments, demonstrating EPoG's potential for real-world applications.

  </details>


