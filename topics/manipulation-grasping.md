# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-03-04 07:02 UTC_

Total papers shown: **9**


---

- **From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?**  
  Shinas Shaji, Fabian Huppertz, Alex Mitrevski, Sebastian Houben  
  _2026-03-03_ · https://arxiv.org/abs/2603.03148v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In order to flexibly act in an everyday environment, a robotic agent needs a variety of cognitive capabilities that enable it to reason about plans and perform execution recovery. Large language models (LLMs) have been shown to demonstrate emergent cognitive aspects, such as reasoning and language understanding; however, the ability to control embodied robotic agents requires reliably bridging high-level language to low-level functionalities for perception and control. In this paper, we investigate the extent to which an LLM can serve as a core component for planning and execution reasoning in a cognitive robot architecture. For this purpose, we propose a cognitive architecture in which an agentic LLM serves as the core component for planning and reasoning, while components for working and episodic memories support learning from experience and adaptation. An instance of the architecture is then used to control a mobile manipulator in a simulated household environment, where environment interaction is done through a set of high-level tools for perception, reasoning, navigation, grasping, and placement, all of which are made available to the LLM-based agent. We evaluate our proposed system on two household tasks (object placement and object swapping), which evaluate the agent's reasoning, planning, and memory utilisation. The results demonstrate that the LLM-driven agent can complete structured tasks and exhibits emergent adaptation and memory-guided planning, but also reveal significant limitations, such as hallucinations about the task success and poor instruction following by refusing to acknowledge and complete sequential tasks. These findings highlight both the potential and challenges of employing LLMs as embodied cognitive controllers for autonomous robots.

  </details>



- **Robotic Grasping and Placement Controlled by EEG-Based Hybrid Visual and Motor Imagery**  
  Yichang Liu, Tianyu Wang, Ziyi Ye, Yawei Li, Yu-Gang Jiang, Shouyan Wang, Yanwei Fu  
  _2026-03-03_ · https://arxiv.org/abs/2603.03181v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a framework that integrates EEG-based visual and motor imagery (VI/MI) with robotic control to enable real-time, intention-driven grasping and placement. Motivated by the promise of BCI-driven robotics to enhance human-robot interaction, this system bridges neural signals with physical control by deploying offline-pretrained decoders in a zero-shot manner within an online streaming pipeline. This establishes a dual-channel intent interface that translates visual intent into robotic actions, with VI identifying objects for grasping and MI determining placement poses, enabling intuitive control over both what to grasp and where to place. The system operates solely on EEG via a cue-free imagery protocol, achieving integration and online validation. Implemented on a Base robotic platform and evaluated across diverse scenarios, including occluded targets or varying participant postures, the system achieves online decoding accuracies of 40.23% (VI) and 62.59% (MI), with an end-to-end task success rate of 20.88%. These results demonstrate that high-level visual cognition can be decoded in real time and translated into executable robot commands, bridging the gap between neural signals and physical interaction, and validating the flexibility of a purely imagery-based BCI paradigm for practical human-robot collaboration.

  </details>



- **RL-Based Coverage Path Planning for Deformable Objects on 3D Surfaces**  
  Yuhang Zhang, Jinming Ma, Feng Wu  
  _2026-03-03_ · https://arxiv.org/abs/2603.03137v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Currently, manipulation tasks for deformable objects often focus on activities like folding clothes, handling ropes, and manipulating bags. However, research on contact-rich tasks involving deformable objects remains relatively underdeveloped. When humans use cloth or sponges to wipe surfaces, they rely on both vision and tactile feedback. Yet, current algorithms still face challenges with issues like occlusion, while research on tactile perception for manipulation is still evolving. Tasks such as covering surfaces with deformable objects demand not only perception but also precise robotic manipulation. To address this, we propose a method that leverages efficient and accessible simulators for task execution. Specifically, we train a reinforcement learning agent in a simulator to manipulate deformable objects for surface wiping tasks. We simplify the state representation of object surfaces using harmonic UV mapping, process contact feedback from the simulator on 2D feature maps, and use scaled grouped convolutions (SGCNN) to extract features efficiently. The agent then outputs actions in a reduced-dimensional action space to generate coverage paths. Experiments demonstrate that our method outperforms previous approaches in key metrics, including total path length and coverage area. We deploy these paths on a Kinova Gen3 manipulator to perform wiping experiments on the back of a torso model, validating the feasibility of our approach.

  </details>



- **Utonia: Toward One Encoder for All Point Clouds**  
  Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, Hengshuang Zhao  
  _2026-03-03_ · https://arxiv.org/abs/2603.03283v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We dream of a future where point clouds from all domains can come together to shape a single model that benefits them all. Toward this goal, we present Utonia, a first step toward training a single self-supervised point transformer encoder across diverse domains, spanning remote sensing, outdoor LiDAR, indoor RGB-D sequences, object-centric CAD models, and point clouds lifted from RGB-only videos. Despite their distinct sensing geometries, densities, and priors, Utonia learns a consistent representation space that transfers across domains. This unification improves perception capability while revealing intriguing emergent behaviors that arise only when domains are trained jointly. Beyond perception, we observe that Utonia representations can also benefit embodied and multimodal reasoning: conditioning vision-language-action policies on Utonia features improves robotic manipulation, and integrating them into vision-language models yields gains on spatial reasoning. We hope Utonia can serve as a step toward foundation models for sparse 3D data, and support downstream applications in AR/VR, robotics, and autonomous driving.

  </details>



- **ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation**  
  Xialin He, Sirui Xu, Xinyao Li, Runpei Dong, Liuyu Bian, Yu-Xiong Wang, Liang-Yan Gui  
  _2026-03-03_ · https://arxiv.org/abs/2603.03279v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Achieving autonomous and versatile whole-body loco-manipulation remains a central barrier to making humanoids practically useful. Yet existing approaches are fundamentally constrained: retargeted data are often scarce or low-quality; methods struggle to scale to large skill repertoires; and, most importantly, they rely on tracking predefined motion references rather than generating behavior from perception and high-level task specifications. To address these limitations, we propose ULTRA, a unified framework with two key components. First, we introduce a physics-driven neural retargeting algorithm that translates large-scale motion capture to humanoid embodiments while preserving physical plausibility for contact-rich interactions. Second, we learn a unified multimodal controller that supports both dense references and sparse task specifications, under sensing ranging from accurate motion-capture state to noisy egocentric visual inputs. We distill a universal tracking policy into this controller, compress motor skills into a compact latent space, and apply reinforcement learning finetuning to expand coverage and improve robustness under out-of-distribution scenarios. This enables coordinated whole-body behavior from sparse intent without test-time reference motions. We evaluate ULTRA in simulation and on a real Unitree G1 humanoid. Results show that ULTRA generalizes to autonomous, goal-conditioned whole-body loco-manipulation from egocentric perception, consistently outperforming tracking-only baselines with limited skills.

  </details>



- **HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations**  
  Xiaomeng Xu, Jisang Park, Han Zhang, Eric Cousineau, Aditya Bhat, Jose Barreiros, Dian Wang, Shuran Song  
  _2026-03-03_ · https://arxiv.org/abs/2603.03243v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present Whole-Body Mobile Manipulation Interface (HoMMI), a data collection and policy learning framework that learns whole-body mobile manipulation directly from robot-free human demonstrations. We augment UMI interfaces with egocentric sensing to capture the global context required for mobile manipulation, enabling portable, robot-free, and scalable data collection. However, naively incorporating egocentric sensing introduces a larger human-to-robot embodiment gap in both observation and action spaces, making policy transfer difficult. We explicitly bridge this gap with a cross-embodiment hand-eye policy design, including an embodiment agnostic visual representation; a relaxed head action representation; and a whole-body controller that realizes hand-eye trajectories through coordinated whole-body motion under robot-specific physical constraints. Together, these enable long-horizon mobile manipulation tasks requiring bimanual and whole-body coordination, navigation, and active perception. Results are best viewed on: https://hommi-robot.github.io

  </details>



- **ACE-Brain-0: Spatial Intelligence as a Shared Scaffold for Universal Embodiments**  
  Ziyang Gong, Zehang Luo, Anke Tang, Zhe Liu, Shi Fu, Zhi Hou, Ganlin Yang, Weiyun Wang, Xiaofeng Wang, Jianbo Liu, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.03198v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Universal embodied intelligence demands robust generalization across heterogeneous embodiments, such as autonomous driving, robotics, and unmanned aerial vehicles (UAVs). However, existing embodied brain in training a unified model over diverse embodiments frequently triggers long-tail data, gradient interference, and catastrophic forgetting, making it notoriously difficult to balance universal generalization with domain-specific proficiency. In this report, we introduce ACE-Brain-0, a generalist foundation brain that unifies spatial reasoning, autonomous driving, and embodied manipulation within a single multimodal large language model~(MLLM). Our key insight is that spatial intelligence serves as a universal scaffold across diverse physical embodiments: although vehicles, robots, and UAVs differ drastically in morphology, they share a common need for modeling 3D mental space, making spatial cognition a natural, domain-agnostic foundation for cross-embodiment transfer. Building on this insight, we propose the Scaffold-Specialize-Reconcile~(SSR) paradigm, which first establishes a shared spatial foundation, then cultivates domain-specialized experts, and finally harmonizes them through data-free model merging. Furthermore, we adopt Group Relative Policy Optimization~(GRPO) to strengthen the model's comprehensive capability. Extensive experiments demonstrate that ACE-Brain-0 achieves competitive and even state-of-the-art performance across 24 spatial and embodiment-related benchmarks.

  </details>



- **How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference**  
  Toru Lin, Shuying Deng, Zhao-Heng Yin, Pieter Abbeel, Jitendra Malik  
  _2026-03-03_ · https://arxiv.org/abs/2603.03280v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many essential manipulation tasks - such as food preparation, surgery, and craftsmanship - remain intractable for autonomous robots. These tasks are characterized not only by contact-rich, force-sensitive dynamics, but also by their "implicit" success criteria: unlike pick-and-place, task quality in these domains is continuous and subjective (e.g. how well a potato is peeled), making quantitative evaluation and reward engineering difficult. We present a learning framework for such tasks, using peeling with a knife as a representative example. Our approach follows a two-stage pipeline: first, we learn a robust initial policy via force-aware data collection and imitation learning, enabling generalization across object variations; second, we refine the policy through preference-based finetuning using a learned reward model that combines quantitative task metrics with qualitative human feedback, aligning policy behavior with human notions of task quality. Using only 50-200 peeling trajectories, our system achieves over 90% average success rates on challenging produce including cucumbers, apples, and potatoes, with performance improving by up to 40% through preference-based finetuning. Remarkably, policies trained on a single produce category exhibit strong zero-shot generalization to unseen in-category instances and to out-of-distribution produce from different categories while maintaining over 90% success rates.

  </details>



- **Design Generative AI for Practitioners: Exploring Interaction Approaches Aligned with Creative Practice**  
  Xiaohan Peng, Wendy E. Mackay, Janin Koch  
  _2026-03-03_ · https://arxiv.org/abs/2603.03074v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Design is a non-linear, reflective process in which practitioners engage with visual, semantic, and other expressive materials to explore, iterate, and refine ideas. As Generative AI (GenAI) becomes integrated into professional design practice, traditional interaction approaches focusing on prompts or whole-image manipulation can misalign AI output with designers' intent, forcing visual thinkers into verbal reasoning or post-hoc adjustments. We present three interaction approaches from DesignPrompt, FusAIn, and DesignTrace that distribute control across intent, input, and process, enabling designers to guide AI alignment at different stages of interaction. We further argue that alignment is a dynamic negotiation, with AI adopting proactive or reactive roles according to designers' instrumental and inspirational needs and the creative stage.

  </details>


