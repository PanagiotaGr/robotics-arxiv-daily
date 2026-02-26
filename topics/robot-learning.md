# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-26 07:13 UTC_

Total papers shown: **18**


---

- **Self-Correcting VLA: Online Action Refinement via Sparse World Imagination**  
  Chenyv Liu, Wentao Tan, Lei Zhu, Fengling Li, Jingjing Li, Guoli Yang, Heng Tao Shen  
  _2026-02-25_ · https://arxiv.org/abs/2602.21633v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Standard vision-language-action (VLA) models rely on fitting statistical data priors, limiting their robust understanding of underlying physical dynamics. Reinforcement learning enhances physical grounding through exploration yet typically relies on external reward signals that remain isolated from the agent's internal states. World action models have emerged as a promising paradigm that integrates imagination and control to enable predictive planning. However, they rely on implicit context modeling, lacking explicit mechanisms for self-improvement. To solve these problems, we propose Self-Correcting VLA (SC-VLA), which achieve self-improvement by intrinsically guiding action refinement through sparse imagination. We first design sparse world imagination by integrating auxiliary predictive heads to forecast current task progress and future trajectory trends, thereby constraining the policy to encode short-term physical evolution. Then we introduce the online action refinement module to reshape progress-dependent dense rewards, adjusting trajectory orientation based on the predicted sparse future states. Evaluations on challenging robot manipulation tasks from simulation benchmarks and real-world settings demonstrate that SC-VLA achieve state-of-the-art performance, yielding the highest task throughput with 16% fewer steps and a 9% higher success rate than the best-performing baselines, alongside a 14% gain in real-world experiments. Code is available at https://github.com/Kisaragi0/SC-VLA.

  </details>



- **Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos**  
  Matthew Strong, Wei-Jer Chang, Quentin Herau, Jiezhi Yang, Yihan Hu, Chensheng Peng, Wei Zhan  
  _2026-02-25_ · https://arxiv.org/abs/2602.22091v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Ego-centric driving videos available online provide an abundant source of visual data for autonomous driving, yet their lack of annotations makes it difficult to learn representations that capture both semantic structure and 3D geometry. Recent advances in large feedforward spatial models demonstrate that point maps and ego-motion can be inferred in a single forward pass, suggesting a promising direction for scalable driving perception. We therefore propose a label-free, teacher-guided framework for learning autonomous driving representations directly from unposed videos. Unlike prior self-supervised approaches that focus primarily on frame-to-frame consistency, we posit that safe and reactive driving depends critically on temporal context. To this end, we leverage a feedforward architecture equipped with a lightweight autoregressive module, trained using multi-modal supervisory signals that guide the model to jointly predict current and future point maps, camera poses, semantic segmentation, and motion masks. Multi-modal teachers provide sequence-level pseudo-supervision, enabling LFG to learn a unified pseudo-4D representation from raw YouTube videos without poses, labels, or LiDAR. The resulting encoder not only transfers effectively to downstream autonomous driving planning on the NAVSIM benchmark, surpassing multi-camera and LiDAR baselines with only a single monocular camera, but also yields strong performance when evaluated on a range of semantic, geometric, and qualitative motion prediction tasks. These geometry and motion-aware features position LFG as a compelling video-centric foundation model for autonomous driving.

  </details>



- **RGB-Event HyperGraph Prompt for Kilometer Marker Recognition based on Pre-trained Foundation Models**  
  Xiaoyu Xian, Shiao Wang, Xiao Wang, Daxin Tian, Yan Tian  
  _2026-02-25_ · https://arxiv.org/abs/2602.22026v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Metro trains often operate in highly complex environments, characterized by illumination variations, high-speed motion, and adverse weather conditions. These factors pose significant challenges for visual perception systems, especially those relying solely on conventional RGB cameras. To tackle these difficulties, we explore the integration of event cameras into the perception system, leveraging their advantages in low-light conditions, high-speed scenarios, and low power consumption. Specifically, we focus on Kilometer Marker Recognition (KMR), a critical task for autonomous metro localization under GNSS-denied conditions. In this context, we propose a robust baseline method based on a pre-trained RGB OCR foundation model, enhanced through multi-modal adaptation. Furthermore, we construct the first large-scale RGB-Event dataset, EvMetro5K, containing 5,599 pairs of synchronized RGB-Event samples, split into 4,479 training and 1,120 testing samples. Extensive experiments on EvMetro5K and other widely used benchmarks demonstrate the effectiveness of our approach for KMR. Both the dataset and source code will be released on https://github.com/Event-AHU/EvMetro5K_benchmark

  </details>



- **PanoEnv: Exploring 3D Spatial Intelligence in Panoramic Environments with Reinforcement Learning**  
  Zekai Lin, Xu Zheng  
  _2026-02-25_ · https://arxiv.org/abs/2602.21992v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  360 panoramic images are increasingly used in virtual reality, autonomous driving, and robotics for holistic scene understanding. However, current Vision-Language Models (VLMs) struggle with 3D spatial reasoning on Equirectangular Projection (ERP) images due to geometric distortion and limited 3D supervision. We introduce PanoEnv, a large-scale VQA benchmark built from synthetic 3D environments, containing 14.8K questions across five categories (e.g., relative position, volume comparison) grounded in accurate 3D annotations including depth, segmentation, and bounding boxes. Benchmarking 14 state-of-the-art VLMs reveals limited 3D understanding, achieving only 49.34% overall accuracy and 8.36% on open-ended (OE) questions. To enhance 3D reasoning, we propose a reinforcement learning post-training framework based on Group Relative Policy Optimization (GRPO) with a ground-truth-guided reward that incorporates five geometry-aware strategies such as distance tolerance and spatial consistency. A two-stage curriculum further mitigates catastrophic forgetting: Stage 1 trains on structured tasks (true/false and multiple choice), and Stage 2 fine-tunes on mixed open-ended data to improve generalization. Our 7B model achieves new state-of-the-art performance, improving overall accuracy to 52.93% (+3.59%) and open-ended accuracy to 14.83% while maintaining structured-task performance. It also achieves top semantic evaluation scores (Q-Score 6.24, P-Score 5.95), surpassing 32B models. These results demonstrate that PanoEnv-QA and our curriculum-based RL framework effectively instill 3D spatial intelligence in VLMs for omnidirectional perception.

  </details>



- **Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild**  
  Hao Luo, Ye Wang, Wanpeng Zhang, Haoqi Yuan, Yicheng Feng, Haiweng Xu, Sipeng Zheng, Zongqing Lu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21736v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Despite progress, Vision-Language-Action models (VLAs) are limited by a scarcity of large-scale, diverse robot data. While human manipulation videos offer a rich alternative, existing methods are forced to choose between small, precisely-labeled datasets and vast in-the-wild footage with unreliable hand tracking labels. We present JALA, a pretraining framework that learns Jointly-Aligned Latent Actions. JALA bypasses full visual dynamic reconstruction, instead learns a predictive action embedding aligned with both inverse dynamics and real actions. This yields a transition-aware, behavior-centric latent space for learning from heterogeneous human data. We scale this approach with UniHand-Mix, a 7.5M video corpus (>2,000 hours) blending laboratory and in-the-wild footage. Experiments demonstrate that JALA generates more realistic hand motions in both controlled and unconstrained scenarios, significantly improving downstream robot manipulation performance in both simulation and real-world tasks. These results indicate that jointly-aligned latent actions offer a scalable pathway for VLA pretraining from human data.

  </details>



- **Two-Stage Active Distribution Network Voltage Control via LLM-RL Collaboration: A Hybrid Knowledge-Data-Driven Approach**  
  Xu Yang, Chenhui Lin, Xiang Ma, Dong Liu, Ran Zheng, Haotian Liu, Wenchuan Wu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21715v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  The growing integration of distributed photovoltaics (PVs) into active distribution networks (ADNs) has exacerbated operational challenges, making it imperative to coordinate diverse equipment to mitigate voltage violations and enhance power quality. Although existing data-driven approaches have demonstrated effectiveness in the voltage control problem, they often require extensive trial-and-error exploration and struggle to incorporate heterogeneous information, such as day-ahead forecasts and semantic-based grid codes. Considering the operational scenarios and requirements in real-world ADNs, in this paper, we propose a hybrid knowledge-data-driven approach that leverages dynamic collaboration between a large language model (LLM) agent and a reinforcement learning (RL) agent to achieve two-stage voltage control. In the day-ahead stage, the LLM agent receives coarse region-level forecasts and generates scheduling strategies for on-load tap changer (OLTC) and shunt capacitors (SCs) to regulate the overall voltage profile. Then in the intra-day stage, based on accurate node-level measurements, the RL agent refines terminal voltages by deriving reactive power generation strategies for PV inverters. On top of the LLM-RL collaboration framework, we further propose a self-evolution mechanism for the LLM agent and a pretrain-finetune pipeline for the RL agent, effectively enhancing and coordinating the policies for both agents. The proposed approach not only aligns more closely with practical operational characteristics but also effectively utilizes the inherent knowledge and reasoning capabilities of the LLM agent, significantly improving training efficiency and voltage control performance. Comprehensive comparisons and ablation studies demonstrate the effectiveness of the proposed method.

  </details>



- **System Design of the Ultra Mobility Vehicle: A Driving, Balancing, and Jumping Bicycle Robot**  
  Benjamin Bokser, Daniel Gonzalez, Surya Singh, Aaron Preston, Alex Bahner, Annika Wollschläger, Arianna Ilvonen, Asa Eckert-Erdheim, Ashwin Khadke, Bilal Hammoud, et al.  
  _2026-02-25_ · https://arxiv.org/abs/2602.22118v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Trials cyclists and mountain bike riders can hop, jump, balance, and drive on one or both wheels. This versatility allows them to achieve speed and energy-efficiency on smooth terrain and agility over rough terrain. Inspired by these athletes, we present the design and control of a robotic platform, Ultra Mobility Vehicle (UMV), which combines a bicycle and a reaction mass to move dynamically with minimal actuated degrees of freedom. We employ a simulation-driven design optimization process to synthesize a spatial linkage topology with a focus on vertical jump height and momentum-based balancing on a single wheel contact. Using a constrained Reinforcement Learning (RL) framework, we demonstrate zero-shot transfer of diverse athletic behaviors, including track-stands, jumps, wheelies, rear wheel hopping, and front flips. This 23.5 kg robot is capable of high speeds (8 m/s) and jumping on and over large obstacles (1 m tall, or 130% of the robot's nominal height).

  </details>



- **Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning**  
  Tomoya Kawabe, Rin Takano  
  _2026-02-25_ · https://arxiv.org/abs/2602.21670v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-robot task planning requires decomposing natural-language instructions into executable actions for heterogeneous robot teams. Conventional Planning Domain Definition Language (PDDL) planners provide rigorous guarantees but struggle to handle ambiguous or long-horizon missions, while large language models (LLMs) can interpret instructions and propose plans but may hallucinate or produce infeasible actions. We present a hierarchical multi-agent LLM-based planner with prompt optimization: an upper layer decomposes tasks and assigns them to lower-layer agents, which generate PDDL problems solved by a classical planner. When plans fail, the system applies TextGrad-inspired textual-gradient updates to optimize each agent's prompt and thereby improve planning accuracy. In addition, meta-prompts are learned and shared across agents within the same layer, enabling efficient prompt optimization in multi-agent settings. On the MAT-THOR benchmark, our planner achieves success rates of 0.95 on compound tasks, 0.84 on complex tasks, and 0.60 on vague tasks, improving over the previous state-of-the-art LaMMA-P by 2, 7, and 15 percentage points respectively. An ablation study shows that the hierarchical structure, prompt optimization, and meta-prompt sharing contribute roughly +59, +37, and +4 percentage points to the overall success rate.

  </details>



- **Tacmap: Bridging the Tactile Sim-to-Real Gap via Geometry-Consistent Penetration Depth Map**  
  Lei Su, Zhijie Peng, Renyuan Ren, Shengping Mao, Juan Du, Kaifeng Zhang, Xuezhou Zhu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21625v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Based Tactile Sensors (VBTS) are essential for achieving dexterous robotic manipulation, yet the tactile sim-to-real gap remains a fundamental bottleneck. Current tactile simulations suffer from a persistent dilemma: simplified geometric projections lack physical authenticity, while high-fidelity Finite Element Methods (FEM) are too computationally prohibitive for large-scale reinforcement learning. In this work, we present Tacmap, a high-fidelity, computationally efficient tactile simulation framework anchored in volumetric penetration depth. Our key insight is to bridge the tactile sim-to-real gap by unifying both domains through a shared deform map representation. Specifically, we compute 3D intersection volumes as depth maps in simulation, while in the real world, we employ an automated data-collection rig to learn a robust mapping from raw tactile images to ground-truth depth maps. By aligning simulation and real-world in this unified geometric space, Tacmap minimizes domain shift while maintaining physical consistency. Quantitative evaluations across diverse contact scenarios demonstrate that Tacmap's deform maps closely mirror real-world measurements. Moreover, we validate the utility of Tacmap through an in-hand rotation task, where a policy trained exclusively in simulation achieves zero-shot transfer to a physical robot.

  </details>



- **ADM-DP: Adaptive Dynamic Modality Diffusion Policy through Vision-Tactile-Graph Fusion for Multi-Agent Manipulation**  
  Enyi Wang, Wen Fan, Dandan Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21622v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-agent robotic manipulation remains challenging due to the combined demands of coordination, grasp stability, and collision avoidance in shared workspaces. To address these challenges, we propose the Adaptive Dynamic Modality Diffusion Policy (ADM-DP), a framework that integrates vision, tactile, and graph-based (multi-agent pose) modalities for coordinated control. ADM-DP introduces four key innovations. First, an enhanced visual encoder merges RGB and point-cloud features via Feature-wise Linear Modulation (FiLM) modulation to enrich perception. Second, a tactile-guided grasping strategy uses Force-Sensitive Resistor (FSR) feedback to detect insufficient contact and trigger corrective grasp refinement, improving grasp stability. Third, a graph-based collision encoder leverages shared tool center point (TCP) positions of multiple agents as structured kinematic context to maintain spatial awareness and reduce inter-agent interference. Fourth, an Adaptive Modality Attention Mechanism (AMAM) dynamically re-weights modalities according to task context, enabling flexible fusion. For scalability and modularity, a decoupled training paradigm is employed in which agents learn independent policies while sharing spatial information. This maintains low interdependence between agents while retaining collective awareness. Across seven multi-agent tasks, ADM-DP achieves 12-25% performance gains over state-of-the-art baselines. Ablation studies show the greatest improvements in tasks requiring multiple sensory modalities, validating our adaptive fusion strategy and demonstrating its robustness for diverse manipulation scenarios.

  </details>



- **Recovered in Translation: Efficient Pipeline for Automated Translation of Benchmarks and Datasets**  
  Hanna Yukhymenko, Anton Alexandrov, Martin Vechev  
  _2026-02-25_ · https://arxiv.org/abs/2602.22207v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  The reliability of multilingual Large Language Model (LLM) evaluation is currently compromised by the inconsistent quality of translated benchmarks. Existing resources often suffer from semantic drift and context loss, which can lead to misleading performance metrics. In this work, we present a fully automated framework designed to address these challenges by enabling scalable, high-quality translation of datasets and benchmarks. We demonstrate that adapting test-time compute scaling strategies, specifically Universal Self-Improvement (USI) and our proposed multi-round ranking method, T-RANK, allows for significantly higher quality outputs compared to traditional pipelines. Our framework ensures that benchmarks preserve their original task structure and linguistic nuances during localization. We apply this approach to translate popular benchmarks and datasets into eight Eastern and Southern European languages (Ukrainian, Bulgarian, Slovak, Romanian, Lithuanian, Estonian, Turkish, Greek). Evaluations using both reference-based metrics and LLM-as-a-judge show that our translations surpass existing resources, resulting in more accurate downstream model assessment. We release both the framework and the improved benchmarks to facilitate robust and reproducible multilingual AI development.

  </details>



- **WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs**  
  Yulin Zhang, Cheng Shi, Sibei Yang  
  _2026-02-25_ · https://arxiv.org/abs/2602.22142v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advances in Multimodal Large Language Models have greatly improved visual understanding and reasoning, yet their quadratic attention and offline training protocols make them ill-suited for streaming settings where frames arrive sequentially and future observations are inaccessible. We diagnose a core limitation of current Video-LLMs, namely Time-Agnosticism, in which videos are treated as an unordered bag of evidence rather than a causally ordered sequence, yielding two failures in streams: temporal order ambiguity, in which the model cannot follow or reason over the correct chronological order, and past-current focus blindness where it fails to distinguish present observations from accumulated history. We present WeaveTime, a simple, efficient, and model agnostic framework that first teaches order and then uses order. We introduce a lightweight Temporal Reconstruction objective-our Streaming Order Perception enhancement-that instills order aware representations with minimal finetuning and no specialized streaming data. At inference, a Past-Current Dynamic Focus Cache performs uncertainty triggered, coarse-to-fine retrieval, expanding history only when needed. Plugged into exsiting Video-LLM without architectural changes, WeaveTime delivers consistent gains on representative streaming benchmarks, improving accuracy while reducing latency. These results establish WeaveTime as a practical path toward time aware stream Video-LLMs under strict online, time causal constraints. Code and weights will be made publicly available. Project Page: https://zhangyl4.github.io/publications/weavetime/

  </details>



- **Semantic Partial Grounding via LLMs**  
  Giuseppe Canonaco, Alberto Pozanco, Daniel Borrajo  
  _2026-02-25_ · https://arxiv.org/abs/2602.22067v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Grounding is a critical step in classical planning, yet it often becomes a computational bottleneck due to the exponential growth in grounded actions and atoms as task size increases. Recent advances in partial grounding have addressed this challenge by incrementally grounding only the most promising operators, guided by predictive models. However, these approaches primarily rely on relational features or learned embeddings and do not leverage the textual and structural cues present in PDDL descriptions. We propose SPG-LLM, which uses LLMs to analyze the domain and problem files to heuristically identify potentially irrelevant objects, actions, and predicates prior to grounding, significantly reducing the size of the grounded task. Across seven hard-to-ground benchmarks, SPG-LLM achieves faster grounding-often by orders of magnitude-while delivering comparable or better plan costs in some domains.

  </details>



- **Are Foundation Models the Route to Full-Stack Transfer in Robotics?**  
  Freek Stulp, Samuel Bustamante, João Silvério, Alin Albu-Schäffer, Jeannette Bohg, Shuran Song  
  _2026-02-25_ · https://arxiv.org/abs/2602.22001v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In humans and robots alike, transfer learning occurs at different levels of abstraction, from high-level linguistic transfer to low-level transfer of motor skills. In this article, we provide an overview of the impact that foundation models and transformer networks have had on these different levels, bringing robots closer than ever to "full-stack transfer". Considering LLMs, VLMs and VLAs from a robotic transfer learning perspective allows us to highlight recurring concepts for transfer, beyond specific implementations. We also consider the challenges of data collection and transfer benchmarks for robotics in the age of foundation models. Are foundation models the route to full-stack transfer in robotics? Our expectation is that they will certainly stay on this route as a key technology.

  </details>



- **Self-Curriculum Model-based Reinforcement Learning for Shape Control of Deformable Linear Objects**  
  Zhaowei Liang, Song Wang, Zhao Jin, Shirui Wu, Dan Wu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21816v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Precise shape control of Deformable Linear Objects (DLOs) is crucial in robotic applications such as industrial and medical fields. However, existing methods face challenges in handling complex large deformation tasks, especially those involving opposite curvatures, and lack efficiency and precision. To address this, we propose a two-stage framework combining Reinforcement Learning (RL) and online visual servoing. In the large-deformation stage, a model-based reinforcement learning approach using an ensemble of dynamics models is introduced to significantly improve sample efficiency. Additionally, we design a self-curriculum goal generation mechanism that dynamically selects intermediate-difficulty goals with high diversity through imagined evaluations, thereby optimizing the policy learning process. In the small-deformation stage, a Jacobian-based visual servo controller is deployed to ensure high-precision convergence. Simulation results show that the proposed method enables efficient policy learning and significantly outperforms mainstream baselines in shape control success rate and precision. Furthermore, the framework effectively transfers the policy trained in simulation to real-world tasks with zero-shot adaptation. It successfully completes all 30 cases with diverse initial and target shapes across DLOs of different sizes and materials. The project website is available at: https://anonymous.4open.science/w/sc-mbrl-dlo-EB48/

  </details>



- **Generalisation of RLHF under Reward Shift and Clipped KL Regularisation**  
  Kenton Tang, Yuzhu Chen, Fengxiang He  
  _2026-02-25_ · https://arxiv.org/abs/2602.21765v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Alignment and adaptation in large language models heavily rely on reinforcement learning from human feedback (RLHF); yet, theoretical understanding of its generalisability remains premature, especially when the learned reward could shift, and the KL control is estimated and clipped. To address this issue, we develop generalisation theory for RLHF that explicitly accounts for (1) \emph{reward shift}: reward models are trained on preference data from earlier or mixed behaviour policies while RLHF optimises the current policy on its own rollouts; and (2) \emph{clipped KL regularisation}: the KL regulariser is estimated from sampled log-probability ratios and then clipped for stabilisation, resulting in an error to RLHF. We present generalisation bounds for RLHF, suggesting that the generalisation error stems from a sampling error from prompts and rollouts, a reward shift error, and a KL clipping error. We also discuss special cases of (1) initialising RLHF parameters with a uniform prior over a finite space, and (2) training RLHF by stochastic gradient descent, as an Ornstein-Uhlenbeck process. The theory yields practical implications in (1) optimal KL clipping threshold, and (2) budget allocation in prompts, rollouts, and preference data.

  </details>



- **Primary-Fine Decoupling for Action Generation in Robotic Imitation**  
  Xiaohan Lei, Min Wang, Wengang Zhou, Xingyu Lu, Houqiang Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21684v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-modal distribution in robotic manipulation action sequences poses critical challenges for imitation learning. To this end, existing approaches often model the action space as either a discrete set of tokens or a continuous, latent-variable distribution. However, both approaches present trade-offs: some methods discretize actions into tokens and therefore lose fine-grained action variations, while others generate continuous actions in a single stage tend to produce unstable mode transitions. To address these limitations, we propose Primary-Fine Decoupling for Action Generation (PF-DAG), a two-stage framework that decouples coarse action consistency from fine-grained variations. First, we compress action chunks into a small set of discrete modes, enabling a lightweight policy to select consistent coarse modes and avoid mode bouncing. Second, a mode conditioned MeanFlow policy is learned to generate high-fidelity continuous actions. Theoretically, we prove PF-DAG's two-stage design achieves a strictly lower MSE bound than single-stage generative policies. Empirically, PF-DAG outperforms state-of-the-art baselines across 56 tasks from Adroit, DexArt, and MetaWorld benchmarks. It further generalizes to real-world tactile dexterous manipulation tasks. Our work demonstrates that explicit mode-level decoupling enables both robust multi-modal modeling and reactive closed-loop control for robotic manipulation.

  </details>



- **PPCR-IM: A System for Multi-layer DAG-based Public Policy Consequence Reasoning and Social Indicator Mapping**  
  Zichen Song, Weijia Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21650v1 · `cs.SI`  
  <details><summary>Abstract</summary>

  Public policy decisions are typically justified using a narrow set of headline indicators, leaving many downstream social impacts unstructured and difficult to compare across policies. We propose PPCR-IM, a system for multi-layer DAG-based consequence reasoning and social indicator mapping that addresses this gap. Given a policy description and its context, PPCR-IM uses an LLM-driven, layer-wise generator to construct a directed acyclic graph of intermediate consequences, allowing child nodes to have multiple parents to capture joint influences. A mapping module then aligns these nodes to a fixed indicator set and assigns one of three qualitative impact directions: increase, decrease, or ambiguous change. For each policy episode, the system outputs a structured record containing the DAG, indicator mappings, and three evaluation measures: an expected-indicator coverage score, a discovery rate for overlooked but relevant indicators, and a relative focus ratio comparing the systems coverage to that of the government. PPCR-IM is available both as an online demo and as a configurable XLSX-to-JSON batch pipeline.

  </details>


