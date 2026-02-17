# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-17 07:12 UTC_

Total papers shown: **8**


---

- **Neurosim: A Fast Simulator for Neuromorphic Robot Perception**  
  Richeek Das, Pratik Chaudhari  
  _2026-02-16_ · https://arxiv.org/abs/2602.15018v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Neurosim is a fast, real-time, high-performance library for simulating sensors such as dynamic vision sensors, RGB cameras, depth sensors, and inertial sensors. It can also simulate agile dynamics of multi-rotor vehicles in complex and dynamic environments. Neurosim can achieve frame rates as high as ~2700 FPS on a desktop GPU. Neurosim integrates with a ZeroMQ-based communication library called Cortex to facilitate seamless integration with machine learning and robotics workflows. Cortex provides a high-throughput, low-latency message-passing system for Python and C++ applications, with native support for NumPy arrays and PyTorch tensors. This paper discusses the design philosophy behind Neurosim and Cortex. It demonstrates how they can be used to (i) train neuromorphic perception and control algorithms, e.g., using self-supervised learning on time-synchronized multi-modal data, and (ii) test real-time implementations of these algorithms in closed-loop. Neurosim and Cortex are available at https://github.com/grasp-lyrl/neurosim .

  </details>



- **DM0: An Embodied-Native Vision-Language-Action Model towards Physical AI**  
  En Yu, Haoran Lv, Jianjian Sun, Kangheng Lin, Ruitao Zhang, Yukang Shi, Yuyang Chen, Ze Chen, Ziheng Zhang, Fan Jia, et al.  
  _2026-02-16_ · https://arxiv.org/abs/2602.14974v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Moving beyond the traditional paradigm of adapting internet-pretrained models to physical tasks, we present DM0, an Embodied-Native Vision-Language-Action (VLA) framework designed for Physical AI. Unlike approaches that treat physical grounding as a fine-tuning afterthought, DM0 unifies embodied manipulation and navigation by learning from heterogeneous data sources from the onset. Our methodology follows a comprehensive three-stage pipeline: Pretraining, Mid-Training, and Post-Training. First, we conduct large-scale unified pretraining on the Vision-Language Model (VLM) using diverse corpora--seamlessly integrating web text, autonomous driving scenarios, and embodied interaction logs-to jointly acquire semantic knowledge and physical priors. Subsequently, we build a flow-matching action expert atop the VLM. To reconcile high-level reasoning with low-level control, DM0 employs a hybrid training strategy: for embodied data, gradients from the action expert are not backpropagated to the VLM to preserve generalized representations, while the VLM remains trainable on non-embodied data. Furthermore, we introduce an Embodied Spatial Scaffolding strategy to construct spatial Chain-of-Thought (CoT) reasoning, effectively constraining the action solution space. Experiments on the RoboChallenge benchmark demonstrate that DM0 achieves state-of-the-art performance in both Specialist and Generalist settings on Table30.

  </details>



- **Real-time Monocular 2D and 3D Perception of Endoluminal Scenes for Controlling Flexible Robotic Endoscopic Instruments**  
  Ruofeng Wei, Kai Chen, Yui Lun Ng, Yiyao Ma, Justin Di-Lang Ho, Hon Sing Tong, Xiaomei Wang, Jing Dai, Ka-Wai Kwok, Qi Dou  
  _2026-02-16_ · https://arxiv.org/abs/2602.14666v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Endoluminal surgery offers a minimally invasive option for early-stage gastrointestinal and urinary tract cancers but is limited by surgical tools and a steep learning curve. Robotic systems, particularly continuum robots, provide flexible instruments that enable precise tissue resection, potentially improving outcomes. This paper presents a visual perception platform for a continuum robotic system in endoluminal surgery. Our goal is to utilize monocular endoscopic image-based perception algorithms to identify position and orientation of flexible instruments and measure their distances from tissues. We introduce 2D and 3D learning-based perception algorithms and develop a physically-realistic simulator that models flexible instruments dynamics. This simulator generates realistic endoluminal scenes, enabling control of flexible robots and substantial data collection. Using a continuum robot prototype, we conducted module and system-level evaluations. Results show that our algorithms improve control of flexible instruments, reducing manipulation time by over 70% for trajectory-following tasks and enhancing understanding of surgical scenarios, leading to robust endoluminal surgeries.

  </details>



- **Affordance Transfer Across Object Instances via Semantically Anchored Functional Map**  
  Xiaoxiang Dong, Weiming Zhi  
  _2026-02-16_ · https://arxiv.org/abs/2602.14874v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Traditional learning from demonstration (LfD) generally demands a cumbersome collection of physical demonstrations, which can be time-consuming and challenging to scale. Recent advances show that robots can instead learn from human videos by extracting interaction cues without direct robot involvement. However, a fundamental challenge remains: how to generalize demonstrated interactions across different object instances that share similar functionality but vary significantly in geometry. In this work, we propose \emph{Semantic Anchored Functional Maps} (SemFM), a framework for transferring affordances across objects from a single visual demonstration. Starting from a coarse mesh reconstructed from an image, our method identifies semantically corresponding functional regions between objects, selects mutually exclusive semantic anchors, and propagates these constraints over the surface using a functional map to obtain a dense, semantically consistent correspondence. This enables demonstrated interaction regions to be transferred across geometrically diverse objects in a lightweight and interpretable manner. Experiments on synthetic object categories and real-world robotic manipulation tasks show that our approach enables accurate affordance transfer with modest computational cost, making it well-suited for practical robotic perception-to-action pipelines.

  </details>



- **EmbeWebAgent: Embedding Web Agents into Any Customized UI**  
  Chenyang Ma, Clyde Fare, Matthew Wilson, Dave Braines  
  _2026-02-16_ · https://arxiv.org/abs/2602.14865v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Most web agents operate at the human interface level, observing screenshots or raw DOM trees without application-level access, which limits robustness and action expressiveness. In enterprise settings, however, explicit control of both the frontend and backend is available. We present EmbeWebAgent, a framework for embedding agents directly into existing UIs using lightweight frontend hooks (curated ARIA and URL-based observations, and a per-page function registry exposed via a WebSocket) and a reusable backend workflow that performs reasoning and takes actions. EmbeWebAgent is stack-agnostic (e.g., React or Angular), supports mixed-granularity actions ranging from GUI primitives to higher-level composites, and orchestrates navigation, manipulation, and domain-specific analytics via MCP tools. Our demo shows minimal retrofitting effort and robust multi-step behaviors grounded in a live UI setting. Live Demo: https://youtu.be/Cy06Ljee1JQ

  </details>



- **BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames**  
  Max Sobol Mark, Jacky Liang, Maria Attarian, Chuyuan Fu, Debidatta Dwibedi, Dhruv Shah, Aviral Kumar  
  _2026-02-16_ · https://arxiv.org/abs/2602.15010v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many robot tasks require attending to the history of past observations. For example, finding an item in a room requires remembering which places have already been searched. However, the best-performing robot policies typically condition only on the current observation, limiting their applicability to such tasks. Naively conditioning on past observations often fails due to spurious correlations: policies latch onto incidental features of training histories that do not generalize to out-of-distribution trajectories upon deployment. We analyze why policies latch onto these spurious correlations and find that this problem stems from limited coverage over the space of possible histories during training, which grows exponentially with horizon. Existing regularization techniques provide inconsistent benefits across tasks, as they do not fundamentally address this coverage problem. Motivated by these findings, we propose Big Picture Policies (BPP), an approach that conditions on a minimal set of meaningful keyframes detected by a vision-language model. By projecting diverse rollouts onto a compact set of task-relevant events, BPP substantially reduces distribution shift between training and deployment, without sacrificing expressivity. We evaluate BPP on four challenging real-world manipulation tasks and three simulation tasks, all requiring history conditioning. BPP achieves 70% higher success rates than the best comparison on real-world evaluations.

  </details>



- **Towards Selection as Power: Bounding Decision Authority in Autonomous Agents**  
  Jose Manuel de la Chica Rodriguez, Juan Manuel Vera Díaz  
  _2026-02-16_ · https://arxiv.org/abs/2602.14606v1 · `cs.MA`  
  <details><summary>Abstract</summary>

  Autonomous agentic systems are increasingly deployed in regulated, high-stakes domains where decisions may be irreversible and institutionally constrained. Existing safety approaches emphasize alignment, interpretability, or action-level filtering. We argue that these mechanisms are necessary but insufficient because they do not directly govern selection power: the authority to determine which options are generated, surfaced, and framed for decision. We propose a governance architecture that separates cognition, selection, and action into distinct domains and models autonomy as a vector of sovereignty. Cognitive autonomy remains unconstrained, while selection and action autonomy are bounded through mechanically enforced primitives operating outside the agent's optimization space. The architecture integrates external candidate generation (CEFL), a governed reducer, commit-reveal entropy isolation, rationale validation, and fail-loud circuit breakers. We evaluate the system across multiple regulated financial scenarios under adversarial stress targeting variance manipulation, threshold gaming, framing skew, ordering effects, and entropy probing. Metrics quantify selection concentration, narrative diversity, governance activation cost, and failure visibility. Results show that mechanical selection governance is implementable, auditable, and prevents deterministic outcome capture while preserving reasoning capacity. Although probabilistic concentration remains, the architecture measurably bounds selection authority relative to conventional scalar pipelines. This work reframes governance as bounded causal power rather than internal intent alignment, offering a foundation for deploying autonomous agents where silent failure is unacceptable.

  </details>



- **Controlling Your Image via Simplified Vector Graphics**  
  Lanqing Guo, Xi Liu, Yufei Wang, Zhihao Li, Siyu Huang  
  _2026-02-16_ · https://arxiv.org/abs/2602.14443v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advances in image generation have achieved remarkable visual quality, while a fundamental challenge remains: Can image generation be controlled at the element level, enabling intuitive modifications such as adjusting shapes, altering colors, or adding and removing objects? In this work, we address this challenge by introducing layer-wise controllable generation through simplified vector graphics (VGs). Our approach first efficiently parses images into hierarchical VG representations that are semantic-aligned and structurally coherent. Building on this representation, we design a novel image synthesis framework guided by VGs, allowing users to freely modify elements and seamlessly translate these edits into photorealistic outputs. By leveraging the structural and semantic features of VGs in conjunction with noise prediction, our method provides precise control over geometry, color, and object semantics. Extensive experiments demonstrate the effectiveness of our approach in diverse applications, including image editing, object-level manipulation, and fine-grained content creation, establishing a new paradigm for controllable image generation. Project page: https://guolanqing.github.io/Vec2Pix/

  </details>


