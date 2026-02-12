# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-12 07:15 UTC_

Total papers shown: **9**


---

- **YOR: Your Own Mobile Manipulator for Generalizable Robotics**  
  Manan H Anjaria, Mehmet Enes Erciyes, Vedant Ghatnekar, Neha Navarkar, Haritheja Etukuru, Xiaole Jiang, Kanad Patel, Dhawal Kabra, Nicholas Wojno, Radhika Ajay Prayage, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11150v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advances in robot learning have generated significant interest in capable platforms that may eventually approach human-level competence. This interest, combined with the commoditization of actuators, has propelled growth in low-cost robotic platforms. However, the optimal form factor for mobile manipulation, especially on a budget, remains an open question. We introduce YOR, an open-source, low-cost mobile manipulator that integrates an omnidirectional base, a telescopic vertical lift, and two arms with grippers to achieve whole-body mobility and manipulation. Our design emphasizes modularity, ease of assembly using off-the-shelf components, and affordability, with a bill-of-materials cost under 10,000 USD. We demonstrate YOR's capability by completing tasks that require coordinated whole-body control, bimanual manipulation, and autonomous navigation. Overall, YOR offers competitive functionality for mobile manipulation research at a fraction of the cost of existing platforms. Project website: https://www.yourownrobot.ai/

  </details>



- **SQ-CBF: Signed Distance Functions for Numerically Stable Superquadric-Based Safety Filtering**  
  Haocheng Zhao, Lukas Brunke, Oliver Lagerquist, Siqi Zhou, Angela P. Schoellig  
  _2026-02-11_ · https://arxiv.org/abs/2602.11049v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Ensuring safe robot operation in cluttered and dynamic environments remains a fundamental challenge. While control barrier functions provide an effective framework for real-time safety filtering, their performance critically depends on the underlying geometric representation, which is often simplified, leading to either overly conservative behavior or insufficient collision coverage. Superquadrics offer an expressive way to model complex shapes using a few primitives and are increasingly used for robot safety. To integrate this representation into collision avoidance, most existing approaches directly use their implicit functions as barrier candidates. However, we identify a critical but overlooked issue in this practice: the gradients of the implicit SQ function can become severely ill-conditioned, potentially rendering the optimization infeasible and undermining reliable real-time safety filtering. To address this issue, we formulate an SQ-based safety filtering framework that uses signed distance functions as barrier candidates. Since analytical SDFs are unavailable for general SQs, we compute distances using the efficient Gilbert-Johnson-Keerthi algorithm and obtain gradients via randomized smoothing. Extensive simulation and real-world experiments demonstrate consistent collision-free manipulation in cluttered and unstructured scenes, showing robustness to challenging geometries, sensing noise, and dynamic disturbances, while improving task efficiency in teleoperation tasks. These results highlight a pathway toward safety filters that remain precise and reliable under the geometric complexity of real-world environments.

  </details>



- **RADAR: Benchmarking Vision-Language-Action Generalization via Real-World Dynamics, Spatial-Physical Intelligence, and Autonomous Evaluation**  
  Yuhao Chen, Zhihao Zhan, Xiaoxin Lin, Zijian Song, Hao Liu, Qinhan Lyu, Yubo Zu, Xiao Chen, Zhiyuan Liu, Tao Pu, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10980v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  VLA models have achieved remarkable progress in embodied intelligence; however, their evaluation remains largely confined to simulations or highly constrained real-world settings. This mismatch creates a substantial reality gap, where strong benchmark performance often masks poor generalization in diverse physical environments. We identify three systemic shortcomings in current benchmarking practices that hinder fair and reliable model comparison. (1) Existing benchmarks fail to model real-world dynamics, overlooking critical factors such as dynamic object configurations, robot initial states, lighting changes, and sensor noise. (2) Current protocols neglect spatial--physical intelligence, reducing evaluation to rote manipulation tasks that do not probe geometric reasoning. (3) The field lacks scalable fully autonomous evaluation, instead relying on simplistic 2D metrics that miss 3D spatial structure or on human-in-the-loop systems that are costly, biased, and unscalable. To address these limitations, we introduce RADAR (Real-world Autonomous Dynamics And Reasoning), a benchmark designed to systematically evaluate VLA generalization under realistic conditions. RADAR integrates three core components: (1) a principled suite of physical dynamics; (2) dedicated tasks that explicitly test spatial reasoning and physical understanding; and (3) a fully autonomous evaluation pipeline based on 3D metrics, eliminating the need for human supervision. We apply RADAR to audit multiple state-of-the-art VLA models and uncover severe fragility beneath their apparent competence. Performance drops precipitously under modest physical dynamics, with the expectation of 3D IoU declining from 0.261 to 0.068 under sensor noise. Moreover, models exhibit limited spatial reasoning capability. These findings position RADAR as a necessary bench toward reliable and generalizable real-world evaluation of VLA models.

  </details>



- **See, Plan, Snap: Evaluating Multimodal GUI Agents in Scratch**  
  Xingyi Zhang, Yulei Ye, Kaifeng Huang, Wenhao Li, Xiangfeng Wang  
  _2026-02-11_ · https://arxiv.org/abs/2602.10814v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Block-based programming environments such as Scratch play a central role in low-code education, yet evaluating the capabilities of AI agents to construct programs through Graphical User Interfaces (GUIs) remains underexplored. We introduce ScratchWorld, a benchmark for evaluating multimodal GUI agents on program-by-construction tasks in Scratch. Grounded in the Use-Modify-Create pedagogical framework, ScratchWorld comprises 83 curated tasks spanning four distinct problem categories: Create, Debug, Extend, and Compute. To rigorously diagnose the source of agent failures, the benchmark employs two complementary interaction modes: primitive mode requires fine-grained drag-and-drop manipulation to directly assess visuomotor control, while composite mode uses high-level semantic APIs to disentangle program reasoning from GUI execution. To ensure reliable assessment, we propose an execution-based evaluation protocol that validates the functional correctness of the constructed Scratch programs through runtime tests within the browser environment. Extensive experiments across state-of-the-art multimodal language models and GUI agents reveal a substantial reasoning--acting gap, highlighting persistent challenges in fine-grained GUI manipulation despite strong planning capabilities.

  </details>



- **LCIP: Loss-Controlled Inverse Projection of High-Dimensional Image Data**  
  Yu Wang, Frederik L. Dennig, Michael Behrisch, Alexandru Telea  
  _2026-02-11_ · https://arxiv.org/abs/2602.11141v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Projections (or dimensionality reduction) methods $P$ aim to map high-dimensional data to typically 2D scatterplots for visual exploration. Inverse projection methods $P^{-1}$ aim to map this 2D space to the data space to support tasks such as data augmentation, classifier analysis, and data imputation. Current $P^{-1}$ methods suffer from a fundamental limitation -- they can only generate a fixed surface-like structure in data space, which poorly covers the richness of this space. We address this by a new method that can `sweep' the data space under user control. Our method works generically for any $P$ technique and dataset, is controlled by two intuitive user-set parameters, and is simple to implement. We demonstrate it by an extensive application involving image manipulation for style transfer.

  </details>



- **RISE: Self-Improving Robot Policy with Compositional World Model**  
  Jiazhi Yang, Kunyang Lin, Jinwei Li, Wencong Zhang, Tianwei Lin, Longyan Wu, Zhizhong Su, Hao Zhao, Ya-Qin Zhang, Li Chen, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11075v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Despite the sustained scaling on model capacity and data acquisition, Vision-Language-Action (VLA) models remain brittle in contact-rich and dynamic manipulation tasks, where minor execution deviations can compound into failures. While reinforcement learning (RL) offers a principled path to robustness, on-policy RL in the physical world is constrained by safety risk, hardware cost, and environment reset. To bridge this gap, we present RISE, a scalable framework of robotic reinforcement learning via imagination. At its core is a Compositional World Model that (i) predicts multi-view future via a controllable dynamics model, and (ii) evaluates imagined outcomes with a progress value model, producing informative advantages for the policy improvement. Such compositional design allows state and value to be tailored by best-suited yet distinct architectures and objectives. These components are integrated into a closed-loop self-improving pipeline that continuously generates imaginary rollouts, estimates advantages, and updates the policy in imaginary space without costly physical interaction. Across three challenging real-world tasks, RISE yields significant improvement over prior art, with more than +35% absolute performance increase in dynamic brick sorting, +45% for backpack packing, and +35% for box closing, respectively.

  </details>



- **Scaling World Model for Hierarchical Manipulation Policies**  
  Qian Long, Yueze Wang, Jiaxi Song, Junbo Zhang, Peiyan Li, Wenxuan Wang, Yuqi Wang, Haoyang Li, Shaoxuan Xie, Guocai Yao, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10983v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models are promising for generalist robot manipulation but remain brittle in out-of-distribution (OOD) settings, especially with limited real-robot data. To resolve the generalization bottleneck, we introduce a hierarchical Vision-Language-Action framework \our{} that leverages the generalization of large-scale pre-trained world model for robust and generalizable VIsual Subgoal TAsk decomposition VISTA. Our hierarchical framework \our{} consists of a world model as the high-level planner and a VLA as the low-level executor. The high-level world model first divides manipulation tasks into subtask sequences with goal images, and the low-level policy follows the textual and visual guidance to generate action sequences. Compared to raw textual goal specification, these synthesized goal images provide visually and physically grounded details for low-level policies, making it feasible to generalize across unseen objects and novel scenarios. We validate both visual goal synthesis and our hierarchical VLA policies in massive out-of-distribution scenarios, and the performance of the same-structured VLA in novel scenarios could boost from 14% to 69% with the guidance generated by the world model. Results demonstrate that our method outperforms previous baselines with a clear margin, particularly in out-of-distribution scenarios. Project page: \href{https://vista-wm.github.io/}{https://vista-wm.github.io}

  </details>



- **Towards Learning a Generalizable 3D Scene Representation from 2D Observations**  
  Martin Gromniak, Jan-Gerrit Habekost, Sebastian Kamp, Sven Magg, Stefan Wermter  
  _2026-02-11_ · https://arxiv.org/abs/2602.10943v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We introduce a Generalizable Neural Radiance Field approach for predicting 3D workspace occupancy from egocentric robot observations. Unlike prior methods operating in camera-centric coordinates, our model constructs occupancy representations in a global workspace frame, making it directly applicable to robotic manipulation. The model integrates flexible source views and generalizes to unseen object arrangements without scene-specific finetuning. We demonstrate the approach on a humanoid robot and evaluate predicted geometry against 3D sensor ground truth. Trained on 40 real scenes, our model achieves 26mm reconstruction error, including occluded regions, validating its ability to infer complete 3D occupancy beyond traditional stereo vision methods.

  </details>



- **Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation**  
  Songen Gu, Yunuo Cai, Tianyu Wang, Simo Wu, Yanwei Fu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10717v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic manipulation requires anticipating how the environment evolves in response to actions, yet most existing systems lack this predictive capability, often resulting in errors and inefficiency. While Vision-Language Models (VLMs) provide high-level guidance, they cannot explicitly forecast future states, and existing world models either predict only short horizons or produce spatially inconsistent frames. To address these challenges, we propose a framework for fast and predictive video-conditioned action. Our approach first selects and adapts a robust video generation model to ensure reliable future predictions, then applies adversarial distillation for fast, few-step video generation, and finally trains an action model that leverages both generated videos and real observations to correct spatial errors. Extensive experiments show that our method produces temporally coherent, spatially accurate video predictions that directly support precise manipulation, achieving significant improvements in embodiment consistency, spatial referring ability, and task completion over existing baselines. Codes & Models will be released.

  </details>


