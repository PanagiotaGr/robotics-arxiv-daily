# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-17 07:12 UTC_

Total papers shown: **8**


---

- **DM0: An Embodied-Native Vision-Language-Action Model towards Physical AI**  
  En Yu, Haoran Lv, Jianjian Sun, Kangheng Lin, Ruitao Zhang, Yukang Shi, Yuyang Chen, Ze Chen, Ziheng Zhang, Fan Jia, et al.  
  _2026-02-16_ · https://arxiv.org/abs/2602.14974v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Moving beyond the traditional paradigm of adapting internet-pretrained models to physical tasks, we present DM0, an Embodied-Native Vision-Language-Action (VLA) framework designed for Physical AI. Unlike approaches that treat physical grounding as a fine-tuning afterthought, DM0 unifies embodied manipulation and navigation by learning from heterogeneous data sources from the onset. Our methodology follows a comprehensive three-stage pipeline: Pretraining, Mid-Training, and Post-Training. First, we conduct large-scale unified pretraining on the Vision-Language Model (VLM) using diverse corpora--seamlessly integrating web text, autonomous driving scenarios, and embodied interaction logs-to jointly acquire semantic knowledge and physical priors. Subsequently, we build a flow-matching action expert atop the VLM. To reconcile high-level reasoning with low-level control, DM0 employs a hybrid training strategy: for embodied data, gradients from the action expert are not backpropagated to the VLM to preserve generalized representations, while the VLM remains trainable on non-embodied data. Furthermore, we introduce an Embodied Spatial Scaffolding strategy to construct spatial Chain-of-Thought (CoT) reasoning, effectively constraining the action solution space. Experiments on the RoboChallenge benchmark demonstrate that DM0 achieves state-of-the-art performance in both Specialist and Generalist settings on Table30.

  </details>



- **Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery**  
  Chandrakanth Gudavalli, Tajuddin Manhar Mohammed, Abhay Yadav, Ananth Vishnu Bhaskar, Hardik Prajapati, Cheng Peng, Rama Chellappa, Shivkumar Chandrasekaran, B. S. Manjunath  
  _2026-02-16_ · https://arxiv.org/abs/2602.14929v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Aligning ground-level imagery with geo-registered satellite maps is crucial for mapping, navigation, and situational awareness, yet remains challenging under large viewpoint gaps or when GPS is unreliable. We introduce Wrivinder, a zero-shot, geometry-driven framework that aggregates multiple ground photographs to reconstruct a consistent 3D scene and align it with overhead satellite imagery. Wrivinder combines SfM reconstruction, 3D Gaussian Splatting, semantic grounding, and monocular depth--based metric cues to produce a stable zenith-view rendering that can be directly matched to satellite context for metrically accurate camera geo-localization. To support systematic evaluation of this task, which lacks suitable benchmarks, we also release MC-Sat, a curated dataset linking multi-view ground imagery with geo-registered satellite tiles across diverse outdoor environments. Together, Wrivinder and MC-Sat provide a first comprehensive baseline and testbed for studying geometry-centered cross-view alignment without paired supervision. In zero-shot experiments, Wrivinder achieves sub-30\,m geolocation accuracy across both dense and large-area scenes, highlighting the promise of geometry-based aggregation for robust ground-to-satellite localization.

  </details>



- **Scalable Multi-Robot Path Planning via Quadratic Unconstrained Binary Optimization**  
  Javier González Villasmil  
  _2026-02-16_ · https://arxiv.org/abs/2602.14799v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-Agent Path Finding (MAPF) remains a fundamental challenge in robotics, where classical centralized approaches exhibit exponential growth in joint-state complexity as the number of agents increases. This paper investigates Quadratic Unconstrained Binary Optimization (QUBO) as a structurally scalable alternative for simultaneous multi-robot path planning. This approach is a robotics-oriented QUBO formulation incorporating BFS-based logical pre-processing (achieving over 95% variable reduction), adaptive penalty design for collision and constraint enforcement, and a time-windowed decomposition strategy that enables execution within current hardware limitations. An experimental evaluation in grid environments with up to four robots demonstrated near-optimal solutions in dense scenarios and favorable scaling behavior compared to sequential classical planning. These results establish a practical and reproducible baseline for future quantum and quantum-inspired multi-robot coordinations.

  </details>



- **Multimodal Covariance Steering in Belief Space with Active Probing and Influence for Autonomous Driving**  
  Devodita Chakravarty, John Dolan, Yiwei Lyu  
  _2026-02-16_ · https://arxiv.org/abs/2602.14540v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous driving in complex traffic requires reasoning under uncertainty. Common approaches rely on prediction-based planning or risk-aware control, but these are typically treated in isolation, limiting their ability to capture the coupled nature of action and inference in interactive settings. This gap becomes especially critical in uncertain scenarios, where simply reacting to predictions can lead to unsafe maneuvers or overly conservative behavior. Our central insight is that safe interaction requires not only estimating human behavior but also shaping it when ambiguity poses risks. To this end, we introduce a hierarchical belief model that structures human behavior across coarse discrete intents and fine motion modes, updated via Bayesian inference for interpretable multi-resolution reasoning. On top of this, we develop an active probing strategy that identifies when multimodal ambiguity in human predictions may compromise safety and plans disambiguating actions that both reveal intent and gently steer human decisions toward safer outcomes. Finally, a runtime risk-evaluation layer based on Conditional Value-at-Risk (CVaR) ensures that all probing actions remain within human risk tolerance during influence. Our simulations in lane-merging and unsignaled intersection scenarios demonstrate that our approach achieves higher success rates and shorter completion times compared to existing methods. These results highlight the benefit of coupling belief inference, probing, and risk monitoring, yielding a principled and interpretable framework for planning under uncertainty.

  </details>



- **EmbeWebAgent: Embedding Web Agents into Any Customized UI**  
  Chenyang Ma, Clyde Fare, Matthew Wilson, Dave Braines  
  _2026-02-16_ · https://arxiv.org/abs/2602.14865v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Most web agents operate at the human interface level, observing screenshots or raw DOM trees without application-level access, which limits robustness and action expressiveness. In enterprise settings, however, explicit control of both the frontend and backend is available. We present EmbeWebAgent, a framework for embedding agents directly into existing UIs using lightweight frontend hooks (curated ARIA and URL-based observations, and a per-page function registry exposed via a WebSocket) and a reusable backend workflow that performs reasoning and takes actions. EmbeWebAgent is stack-agnostic (e.g., React or Angular), supports mixed-granularity actions ranging from GUI primitives to higher-level composites, and orchestrates navigation, manipulation, and domain-specific analytics via MCP tools. Our demo shows minimal retrofitting effort and robust multi-step behaviors grounded in a live UI setting. Live Demo: https://youtu.be/Cy06Ljee1JQ

  </details>



- **Morphing of and writing with a scissor linkage mechanism**  
  Mohanraj A, S Ganga Prasath  
  _2026-02-16_ · https://arxiv.org/abs/2602.14958v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Kinematics of mechanisms is intricately coupled to their geometry and their utility often arises out of the ability to perform reproducible motion with fewer actuating degrees of freedom. In this article, we explore the assembly of scissor-units, each made of two rigid linear members connected by a pin joint. The assembly has a single degree of freedom, where actuating any single unit results in a shape change of the entire assembly. We derive expressions for the effective curvature of the unit and the trajectory of the mechanism's tip as a function of the geometric variables which we then use as the basis to program two tasks in the mechanism: shape morphing and writing. By phrasing these tasks as optimization problems and utilizing the differentiable simulation framework, we arrive at solutions that are then tested in table-top experiments. Our results show that the geometry of scissor assemblies can be leveraged for automated navigation and inspection in complex domains, in light of the optimization framework. However, we highlight that the challenges associated with rapid programming and error-free implementation in experiments without feedback still remain.

  </details>



- **Arbor: A Framework for Reliable Navigation of Critical Conversation Flows**  
  Luís Silva, Diogo Gonçalves, Catarina Farinha, Clara Matos, Luís Ungaro  
  _2026-02-16_ · https://arxiv.org/abs/2602.14643v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large language models struggle to maintain strict adherence to structured workflows in high-stakes domains such as healthcare triage. Monolithic approaches that encode entire decision structures within a single prompt are prone to instruction-following degradation as prompt length increases, including lost-in-the-middle effects and context window overflow. To address this gap, we present Arbor, a framework that decomposes decision tree navigation into specialized, node-level tasks. Decision trees are standardized into an edge-list representation and stored for dynamic retrieval. At runtime, a directed acyclic graph (DAG)-based orchestration mechanism iteratively retrieves only the outgoing edges of the current node, evaluates valid transitions via a dedicated LLM call, and delegates response generation to a separate inference step. The framework is agnostic to the underlying decision logic and model provider. Evaluated against single-prompt baselines across 10 foundation models using annotated turns from real clinical triage conversations. Arbor improves mean turn accuracy by 29.4 percentage points, reduces per-turn latency by 57.1%, and achieves an average 14.4x reduction in per-turn cost. These results indicate that architectural decomposition reduces dependence on intrinsic model capability, enabling smaller models to match or exceed larger models operating under single-prompt baselines.

  </details>



- **Replanning Human-Robot Collaborative Tasks with Vision-Language Models via Semantic and Physical Dual-Correction**  
  Taichi Kato, Takuya Kiyokawa, Namiko Saito, Kensuke Harada  
  _2026-02-16_ · https://arxiv.org/abs/2602.14551v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Human-Robot Collaboration (HRC) plays an important role in assembly tasks by enabling robots to plan and adjust their motions based on interactive, real-time human instructions. However, such instructions are often linguistically ambiguous and underspecified, making it difficult to generate physically feasible and cooperative robot behaviors. To address this challenge, many studies have applied Vision-Language Models (VLMs) to interpret high-level instructions and generate corresponding actions. Nevertheless, VLM-based approaches still suffer from hallucinated reasoning and an inability to anticipate physical execution failures. To address these challenges, we propose an HRC framework that augments a VLM-based reasoning with a dual-correction mechanism: an internal correction model that verifies logical consistency and task feasibility prior to action execution, and an external correction model that detects and rectifies physical failures through post-execution feedback. Simulation ablation studies demonstrate that the proposed method improves the success rate compared to baselines without correction models. Our real-world experiments in collaborative assembly tasks supported by object fixation or tool preparation by an upper body humanoid robot further confirm the framewor's effectiveness in enabling interactive replanning across different collaborative tasks in response to human instructions, validating its practical feasibility.

  </details>


