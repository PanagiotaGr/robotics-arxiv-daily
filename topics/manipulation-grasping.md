# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-26 07:13 UTC_

Total papers shown: **8**


---

- **ADM-DP: Adaptive Dynamic Modality Diffusion Policy through Vision-Tactile-Graph Fusion for Multi-Agent Manipulation**  
  Enyi Wang, Wen Fan, Dandan Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21622v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-agent robotic manipulation remains challenging due to the combined demands of coordination, grasp stability, and collision avoidance in shared workspaces. To address these challenges, we propose the Adaptive Dynamic Modality Diffusion Policy (ADM-DP), a framework that integrates vision, tactile, and graph-based (multi-agent pose) modalities for coordinated control. ADM-DP introduces four key innovations. First, an enhanced visual encoder merges RGB and point-cloud features via Feature-wise Linear Modulation (FiLM) modulation to enrich perception. Second, a tactile-guided grasping strategy uses Force-Sensitive Resistor (FSR) feedback to detect insufficient contact and trigger corrective grasp refinement, improving grasp stability. Third, a graph-based collision encoder leverages shared tool center point (TCP) positions of multiple agents as structured kinematic context to maintain spatial awareness and reduce inter-agent interference. Fourth, an Adaptive Modality Attention Mechanism (AMAM) dynamically re-weights modalities according to task context, enabling flexible fusion. For scalability and modularity, a decoupled training paradigm is employed in which agents learn independent policies while sharing spatial information. This maintains low interdependence between agents while retaining collective awareness. Across seven multi-agent tasks, ADM-DP achieves 12-25% performance gains over state-of-the-art baselines. Ablation studies show the greatest improvements in tasks requiring multiple sensory modalities, validating our adaptive fusion strategy and demonstrating its robustness for diverse manipulation scenarios.

  </details>



- **Tacmap: Bridging the Tactile Sim-to-Real Gap via Geometry-Consistent Penetration Depth Map**  
  Lei Su, Zhijie Peng, Renyuan Ren, Shengping Mao, Juan Du, Kaifeng Zhang, Xuezhou Zhu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21625v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Based Tactile Sensors (VBTS) are essential for achieving dexterous robotic manipulation, yet the tactile sim-to-real gap remains a fundamental bottleneck. Current tactile simulations suffer from a persistent dilemma: simplified geometric projections lack physical authenticity, while high-fidelity Finite Element Methods (FEM) are too computationally prohibitive for large-scale reinforcement learning. In this work, we present Tacmap, a high-fidelity, computationally efficient tactile simulation framework anchored in volumetric penetration depth. Our key insight is to bridge the tactile sim-to-real gap by unifying both domains through a shared deform map representation. Specifically, we compute 3D intersection volumes as depth maps in simulation, while in the real world, we employ an automated data-collection rig to learn a robust mapping from raw tactile images to ground-truth depth maps. By aligning simulation and real-world in this unified geometric space, Tacmap minimizes domain shift while maintaining physical consistency. Quantitative evaluations across diverse contact scenarios demonstrate that Tacmap's deform maps closely mirror real-world measurements. Moreover, we validate the utility of Tacmap through an in-hand rotation task, where a policy trained exclusively in simulation achieves zero-shot transfer to a physical robot.

  </details>



- **Primary-Fine Decoupling for Action Generation in Robotic Imitation**  
  Xiaohan Lei, Min Wang, Wengang Zhou, Xingyu Lu, Houqiang Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21684v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-modal distribution in robotic manipulation action sequences poses critical challenges for imitation learning. To this end, existing approaches often model the action space as either a discrete set of tokens or a continuous, latent-variable distribution. However, both approaches present trade-offs: some methods discretize actions into tokens and therefore lose fine-grained action variations, while others generate continuous actions in a single stage tend to produce unstable mode transitions. To address these limitations, we propose Primary-Fine Decoupling for Action Generation (PF-DAG), a two-stage framework that decouples coarse action consistency from fine-grained variations. First, we compress action chunks into a small set of discrete modes, enabling a lightweight policy to select consistent coarse modes and avoid mode bouncing. Second, a mode conditioned MeanFlow policy is learned to generate high-fidelity continuous actions. Theoretically, we prove PF-DAG's two-stage design achieves a strictly lower MSE bound than single-stage generative policies. Empirically, PF-DAG outperforms state-of-the-art baselines across 56 tasks from Adroit, DexArt, and MetaWorld benchmarks. It further generalizes to real-world tactile dexterous manipulation tasks. Our work demonstrates that explicit mode-level decoupling enables both robust multi-modal modeling and reactive closed-loop control for robotic manipulation.

  </details>



- **Self-Correcting VLA: Online Action Refinement via Sparse World Imagination**  
  Chenyv Liu, Wentao Tan, Lei Zhu, Fengling Li, Jingjing Li, Guoli Yang, Heng Tao Shen  
  _2026-02-25_ · https://arxiv.org/abs/2602.21633v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Standard vision-language-action (VLA) models rely on fitting statistical data priors, limiting their robust understanding of underlying physical dynamics. Reinforcement learning enhances physical grounding through exploration yet typically relies on external reward signals that remain isolated from the agent's internal states. World action models have emerged as a promising paradigm that integrates imagination and control to enable predictive planning. However, they rely on implicit context modeling, lacking explicit mechanisms for self-improvement. To solve these problems, we propose Self-Correcting VLA (SC-VLA), which achieve self-improvement by intrinsically guiding action refinement through sparse imagination. We first design sparse world imagination by integrating auxiliary predictive heads to forecast current task progress and future trajectory trends, thereby constraining the policy to encode short-term physical evolution. Then we introduce the online action refinement module to reshape progress-dependent dense rewards, adjusting trajectory orientation based on the predicted sparse future states. Evaluations on challenging robot manipulation tasks from simulation benchmarks and real-world settings demonstrate that SC-VLA achieve state-of-the-art performance, yielding the highest task throughput with 16% fewer steps and a 9% higher success rate than the best-performing baselines, alongside a 14% gain in real-world experiments. Code is available at https://github.com/Kisaragi0/SC-VLA.

  </details>



- **WeatherCity: Urban Scene Reconstruction with Controllable Multi-Weather Transformation**  
  Wenhua Wu, Huai Guan, Zhe Liu, Hesheng Wang  
  _2026-02-25_ · https://arxiv.org/abs/2602.22096v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Editable high-fidelity 4D scenes are crucial for autonomous driving, as they can be applied to end-to-end training and closed-loop simulation. However, existing reconstruction methods are primarily limited to replicating observed scenes and lack the capability for diverse weather simulation. While image-level weather editing methods tend to introduce scene artifacts and offer poor controllability over the weather effects. To address these limitations, we propose WeatherCity, a novel framework for 4D urban scene reconstruction and weather editing. Specifically, we leverage a text-guided image editing model to achieve flexible editing of image weather backgrounds. To tackle the challenge of multi-weather modeling, we introduce a novel weather Gaussian representation based on shared scene features and dedicated weather-specific decoders. This representation is further enhanced with a content consistency optimization, ensuring coherent modeling across different weather conditions. Additionally, we design a physics-driven model that simulates dynamic weather effects through particles and motion patterns. Extensive experiments on multiple datasets and various scenes demonstrate that WeatherCity achieves flexible controllability, high fidelity, and temporal consistency in 4D reconstruction and weather editing. Our framework not only enables fine-grained control over weather conditions (e.g., light rain and heavy snow) but also supports object-level manipulation within the scene.

  </details>



- **Force Policy: Learning Hybrid Force-Position Control Policy under Interaction Frame for Contact-Rich Manipulation**  
  Hongjie Fang, Shirun Tang, Mingyu Mei, Haoxiang Qin, Zihao He, Jingjing Chen, Ying Feng, Chenxi Wang, Wanxi Liu, Zaixing He, et al.  
  _2026-02-25_ · https://arxiv.org/abs/2602.22088v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Contact-rich manipulation demands human-like integration of perception and force feedback: vision should guide task progress, while high-frequency interaction control must stabilize contact under uncertainty. Existing learning-based policies often entangle these roles in a monolithic network, trading off global generalization against stable local refinement, while control-centric approaches typically assume a known task structure or learn only controller parameters rather than the structure itself. In this paper, we formalize a physically grounded interaction frame, an instantaneous local basis that decouples force regulation from motion execution, and propose a method to recover it from demonstrations. Based on this, we address both issues by proposing Force Policy, a global-local vision-force policy in which a global policy guides free-space actions using vision, and upon contact, a high-frequency local policy with force feedback estimates the interaction frame and executes hybrid force-position control for stable interaction. Real-world experiments across diverse contact-rich tasks show consistent gains over strong baselines, with more robust contact establishment, more accurate force regulation, and reliable generalization to novel objects with varied geometries and physical properties, ultimately improving both contact stability and execution quality. Project page: https://force-policy.github.io/

  </details>



- **FlowCorrect: Efficient Interactive Correction of Generative Flow Policies for Robotic Manipulation**  
  Edgar Welte, Yitian Shi, Rosa Wolf, Maximillian Gilles, Rania Rayyes  
  _2026-02-25_ · https://arxiv.org/abs/2602.22056v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Generative manipulation policies can fail catastrophically under deployment-time distribution shift, yet many failures are near-misses: the robot reaches almost-correct poses and would succeed with a small corrective motion. We present FlowCorrect, a deployment-time correction framework that converts near-miss failures into successes using sparse human nudges, without full policy retraining. During execution, a human provides brief corrective pose nudges via a lightweight VR interface. FlowCorrect uses these sparse corrections to locally adapt the policy, improving actions without retraining the backbone while preserving the model performance on previously learned scenarios. We evaluate on a real-world robot across three tabletop tasks: pick-and-place, pouring, and cup uprighting. With a low correction budget, FlowCorrect improves success on hard cases by 85\% while preserving performance on previously solved scenarios. The results demonstrate clearly that FlowCorrect learns only with very few demonstrations and enables fast and sample-efficient incremental, human-in-the-loop corrections of generative visuomotor policies at deployment time in real-world robotics.

  </details>



- **Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild**  
  Hao Luo, Ye Wang, Wanpeng Zhang, Haoqi Yuan, Yicheng Feng, Haiweng Xu, Sipeng Zheng, Zongqing Lu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21736v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Despite progress, Vision-Language-Action models (VLAs) are limited by a scarcity of large-scale, diverse robot data. While human manipulation videos offer a rich alternative, existing methods are forced to choose between small, precisely-labeled datasets and vast in-the-wild footage with unreliable hand tracking labels. We present JALA, a pretraining framework that learns Jointly-Aligned Latent Actions. JALA bypasses full visual dynamic reconstruction, instead learns a predictive action embedding aligned with both inverse dynamics and real actions. This yields a transition-aware, behavior-centric latent space for learning from heterogeneous human data. We scale this approach with UniHand-Mix, a 7.5M video corpus (>2,000 hours) blending laboratory and in-the-wild footage. Experiments demonstrate that JALA generates more realistic hand motions in both controlled and unconstrained scenarios, significantly improving downstream robot manipulation performance in both simulation and real-world tasks. These results indicate that jointly-aligned latent actions offer a scalable pathway for VLA pretraining from human data.

  </details>


