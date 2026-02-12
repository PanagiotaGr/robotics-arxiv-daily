# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-12 07:15 UTC_

Total papers shown: **15**


---

- **RISE: Self-Improving Robot Policy with Compositional World Model**  
  Jiazhi Yang, Kunyang Lin, Jinwei Li, Wencong Zhang, Tianwei Lin, Longyan Wu, Zhizhong Su, Hao Zhao, Ya-Qin Zhang, Li Chen, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11075v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Despite the sustained scaling on model capacity and data acquisition, Vision-Language-Action (VLA) models remain brittle in contact-rich and dynamic manipulation tasks, where minor execution deviations can compound into failures. While reinforcement learning (RL) offers a principled path to robustness, on-policy RL in the physical world is constrained by safety risk, hardware cost, and environment reset. To bridge this gap, we present RISE, a scalable framework of robotic reinforcement learning via imagination. At its core is a Compositional World Model that (i) predicts multi-view future via a controllable dynamics model, and (ii) evaluates imagined outcomes with a progress value model, producing informative advantages for the policy improvement. Such compositional design allows state and value to be tailored by best-suited yet distinct architectures and objectives. These components are integrated into a closed-loop self-improving pipeline that continuously generates imaginary rollouts, estimates advantages, and updates the policy in imaginary space without costly physical interaction. Across three challenging real-world tasks, RISE yields significant improvement over prior art, with more than +35% absolute performance increase in dynamic brick sorting, +45% for backpack packing, and +35% for box closing, respectively.

  </details>



- **RADAR: Benchmarking Vision-Language-Action Generalization via Real-World Dynamics, Spatial-Physical Intelligence, and Autonomous Evaluation**  
  Yuhao Chen, Zhihao Zhan, Xiaoxin Lin, Zijian Song, Hao Liu, Qinhan Lyu, Yubo Zu, Xiao Chen, Zhiyuan Liu, Tao Pu, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10980v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  VLA models have achieved remarkable progress in embodied intelligence; however, their evaluation remains largely confined to simulations or highly constrained real-world settings. This mismatch creates a substantial reality gap, where strong benchmark performance often masks poor generalization in diverse physical environments. We identify three systemic shortcomings in current benchmarking practices that hinder fair and reliable model comparison. (1) Existing benchmarks fail to model real-world dynamics, overlooking critical factors such as dynamic object configurations, robot initial states, lighting changes, and sensor noise. (2) Current protocols neglect spatial--physical intelligence, reducing evaluation to rote manipulation tasks that do not probe geometric reasoning. (3) The field lacks scalable fully autonomous evaluation, instead relying on simplistic 2D metrics that miss 3D spatial structure or on human-in-the-loop systems that are costly, biased, and unscalable. To address these limitations, we introduce RADAR (Real-world Autonomous Dynamics And Reasoning), a benchmark designed to systematically evaluate VLA generalization under realistic conditions. RADAR integrates three core components: (1) a principled suite of physical dynamics; (2) dedicated tasks that explicitly test spatial reasoning and physical understanding; and (3) a fully autonomous evaluation pipeline based on 3D metrics, eliminating the need for human supervision. We apply RADAR to audit multiple state-of-the-art VLA models and uncover severe fragility beneath their apparent competence. Performance drops precipitously under modest physical dynamics, with the expectation of 3D IoU declining from 0.261 to 0.068 under sensor noise. Moreover, models exhibit limited spatial reasoning capability. These findings position RADAR as a necessary bench toward reliable and generalizable real-world evaluation of VLA models.

  </details>



- **AugVLA-3D: Depth-Driven Feature Augmentation for Vision-Language-Action Models**  
  Zhifeng Rao, Wenlong Chen, Lei Xie, Xia Hua, Dongfu Yin, Zhen Tian, F. Richard Yu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10698v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic perception and control, yet most existing approaches primarily rely on VLM trained using 2D images, which limits their spatial understanding and action grounding in complex 3D environments. To address this limitation, we propose a novel framework that integrates depth estimation into VLA models to enrich 3D feature representations. Specifically, we employ a depth estimation baseline called VGGT to extract geometry-aware 3D cues from standard RGB inputs, enabling efficient utilization of existing large-scale 2D datasets while implicitly recovering 3D structural information. To further enhance the reliability of these depth-derived features, we introduce a new module called action assistant, which constrains the learned 3D representations with action priors and ensures their consistency with downstream control tasks. By fusing the enhanced 3D features with conventional 2D visual tokens, our approach significantly improves the generalization ability and robustness of VLA models. Experimental results demonstrate that the proposed method not only strengthens perception in geometrically ambiguous scenarios but also leads to superior action prediction accuracy. This work highlights the potential of depth-driven data augmentation and auxiliary expert supervision for bridging the gap between 2D observations and 3D-aware decision-making in robotic systems.

  </details>



- **APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots**  
  Yikai Wang, Tingxuan Leng, Changyi Lin, Shiqi Liu, Shir Simon, Bingqing Chen, Jonathan Francis, Ding Zhao  
  _2026-02-11_ · https://arxiv.org/abs/2602.11143v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Humanoid locomotion has advanced rapidly with deep reinforcement learning (DRL), enabling robust feet-based traversal over uneven terrain. Yet platforms beyond leg length remain largely out of reach because current RL training paradigms often converge to jumping-like solutions that are high-impact, torque-limited, and unsafe for real-world deployment. To address this gap, we propose APEX, a system for perceptive, climbing-based high-platform traversal that composes terrain-conditioned behaviors: climb-up and climb-down at vertical edges, walking or crawling on the platform, and stand-up and lie-down for posture reconfiguration. Central to our approach is a generalized ratchet progress reward for learning contact-rich, goal-reaching maneuvers. It tracks the best-so-far task progress and penalizes non-improving steps, providing dense yet velocity-free supervision that enables efficient exploration under strong safety regularization. Based on this formulation, we train LiDAR-based full-body maneuver policies and reduce the sim-to-real perception gap through a dual strategy: modeling mapping artifacts during training and applying filtering and inpainting to elevation maps during deployment. Finally, we distill all six skills into a single policy that autonomously selects behaviors and transitions based on local geometry and commands. Experiments on a 29-DoF Unitree G1 humanoid demonstrate zero-shot sim-to-real traversal of 0.8 meter platforms (approximately 114% of leg length), with robust adaptation to platform height and initial pose, as well as smooth and stable multi-skill transitions.

  </details>



- **Multi-Task Reinforcement Learning of Drone Aerobatics by Exploiting Geometric Symmetries**  
  Zhanyu Guo, Zikang Yin, Guobin Zhu, Shiliang Guo, Shiyu Zhao  
  _2026-02-11_ · https://arxiv.org/abs/2602.10997v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Flight control for autonomous micro aerial vehicles (MAVs) is evolving from steady flight near equilibrium points toward more aggressive aerobatic maneuvers, such as flips, rolls, and Power Loop. Although reinforcement learning (RL) has shown great potential in these tasks, conventional RL methods often suffer from low data efficiency and limited generalization. This challenge becomes more pronounced in multi-task scenarios where a single policy is required to master multiple maneuvers. In this paper, we propose a novel end-to-end multi-task reinforcement learning framework, called GEAR (Geometric Equivariant Aerobatics Reinforcement), which fully exploits the inherent SO(2) rotational symmetry in MAV dynamics and explicitly incorporates this property into the policy network architecture. By integrating an equivariant actor network, FiLM-based task modulation, and a multi-head critic, GEAR achieves both efficiency and flexibility in learning diverse aerobatic maneuvers, enabling a data-efficient, robust, and unified framework for aerobatic control. GEAR attains a 98.85\% success rate across various aerobatic tasks, significantly outperforming baseline methods. In real-world experiments, GEAR demonstrates stable execution of multiple maneuvers and the capability to combine basic motion primitives to complete complex aerobatics.

  </details>



- **Interactive LLM-assisted Curriculum Learning for Multi-Task Evolutionary Policy Search**  
  Berfin Sakallioglu, Giorgia Nadizar, Eric Medvet  
  _2026-02-11_ · https://arxiv.org/abs/2602.10891v1 · `cs.NE`  
  <details><summary>Abstract</summary>

  Multi-task policy search is a challenging problem because policies are required to generalize beyond training cases. Curriculum learning has proven to be effective in this setting, as it introduces complexity progressively. However, designing effective curricula is labor-intensive and requires extensive domain expertise. LLM-based curriculum generation has only recently emerged as a potential solution, but was limited to operate in static, offline modes without leveraging real-time feedback from the optimizer. Here we propose an interactive LLM-assisted framework for online curriculum generation, where the LLM adaptively designs training cases based on real-time feedback from the evolutionary optimization process. We investigate how different feedback modalities, ranging from numeric metrics alone to combinations with plots and behavior visualizations, influence the LLM ability to generate meaningful curricula. Through a 2D robot navigation case study, tackled with genetic programming as optimizer, we evaluate our approach against static LLM-generated curricula and expert-designed baselines. We show that interactive curriculum generation outperforms static approaches, with multimodal feedback incorporating both progression plots and behavior visualizations yielding performance competitive with expert-designed curricula. This work contributes to understanding how LLMs can serve as interactive curriculum designers for embodied AI systems, with potential extensions to broader evolutionary robotics applications.

  </details>



- **Scaling World Model for Hierarchical Manipulation Policies**  
  Qian Long, Yueze Wang, Jiaxi Song, Junbo Zhang, Peiyan Li, Wenxuan Wang, Yuqi Wang, Haoyang Li, Shaoxuan Xie, Guocai Yao, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10983v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models are promising for generalist robot manipulation but remain brittle in out-of-distribution (OOD) settings, especially with limited real-robot data. To resolve the generalization bottleneck, we introduce a hierarchical Vision-Language-Action framework \our{} that leverages the generalization of large-scale pre-trained world model for robust and generalizable VIsual Subgoal TAsk decomposition VISTA. Our hierarchical framework \our{} consists of a world model as the high-level planner and a VLA as the low-level executor. The high-level world model first divides manipulation tasks into subtask sequences with goal images, and the low-level policy follows the textual and visual guidance to generate action sequences. Compared to raw textual goal specification, these synthesized goal images provide visually and physically grounded details for low-level policies, making it feasible to generalize across unseen objects and novel scenarios. We validate both visual goal synthesis and our hierarchical VLA policies in massive out-of-distribution scenarios, and the performance of the same-structured VLA in novel scenarios could boost from 14% to 69% with the guidance generated by the world model. Results demonstrate that our method outperforms previous baselines with a clear margin, particularly in out-of-distribution scenarios. Project page: \href{https://vista-wm.github.io/}{https://vista-wm.github.io}

  </details>



- **From Representational Complementarity to Dual Systems: Synergizing VLM and Vision-Only Backbones for End-to-End Driving**  
  Sining Ang, Yuguang Yang, Chenxu Dang, Canyu Chen, Cheng Chi, Haiyan Liu, Xuanyao Mao, Jason Bao, Xuliang, Bingchuan Sun, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10719v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) driving augments end-to-end (E2E) planning with language-enabled backbones, yet it remains unclear what changes beyond the usual accuracy--cost trade-off. We revisit this question with 3--RQ analysis in RecogDrive by instantiating the system with a full VLM and vision-only backbones, all under an identical diffusion Transformer planner. RQ1: At the backbone level, the VLM can introduce additional subspaces upon the vision-only backbones. RQ2: This unique subspace leads to a different behavioral in some long-tail scenario: the VLM tends to be more aggressive whereas ViT is more conservative, and each decisively wins on about 2--3% of test scenarios; With an oracle that selects, per scenario, the better trajectory between the VLM and ViT branches, we obtain an upper bound of 93.58 PDMS. RQ3: To fully harness this observation, we propose HybridDriveVLA, which runs both ViT and VLM branches and selects between their endpoint trajectories using a learned scorer, improving PDMS to 92.10. Finally, DualDriveVLA implements a practical fast--slow policy: it runs ViT by default and invokes the VLM only when the scorer's confidence falls below a threshold; calling the VLM on 15% of scenarios achieves 91.00 PDMS while improving throughput by 3.2x. Code will be released.

  </details>



- **Interpretable Attention-Based Multi-Agent PPO for Latency Spike Resolution in 6G RAN Slicing**  
  Kavan Fatehi, Mostafa Rahmani Ghourtani, Amir Sonee, Poonam Yadav, Alessandra M Russo, Hamed Ahmadi, Radu Calinescu  
  _2026-02-11_ · https://arxiv.org/abs/2602.11076v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Sixth-generation (6G) radio access networks (RANs) must enforce strict service-level agreements (SLAs) for heterogeneous slices, yet sudden latency spikes remain difficult to diagnose and resolve with conventional deep reinforcement learning (DRL) or explainable RL (XRL). We propose \emph{Attention-Enhanced Multi-Agent Proximal Policy Optimization (AE-MAPPO)}, which integrates six specialized attention mechanisms into multi-agent slice control and surfaces them as zero-cost, faithful explanations. The framework operates across O-RAN timescales with a three-phase strategy: predictive, reactive, and inter-slice optimization. A URLLC case study shows AE-MAPPO resolves a latency spike in $18$ms, restores latency to $0.98$ms with $99.9999\%$ reliability, and reduces troubleshooting time by $93\%$ while maintaining eMBB and mMTC continuity. These results confirm AE-MAPPO's ability to combine SLA compliance with inherent interpretability, enabling trustworthy and real-time automation for 6G RAN slicing.

  </details>



- **Conversational Behavior Modeling Foundation Model With Multi-Level Perception**  
  Dingkun Zhou, Shuchang Pan, Jiachen Lian, Siddharth Banerjee, Sarika Pasumarthy, Dhruv Hebbar, Siddhant Patel, Zeyi Austin Li, Kan Jen Cheng, Sanay Bordia, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11065v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Human conversation is organized by an implicit chain of thoughts that manifests as timed speech acts. Capturing this perceptual pathway is key to building natural full-duplex interactive systems. We introduce a framework that models this process as multi-level perception, and then reasons over conversational behaviors via a Graph-of-Thoughts (GoT). Our approach formalizes the intent-to-action pathway with a hierarchical labeling scheme, predicting high-level communicative intents and low-level speech acts to learn their causal and temporal dependencies. To train this system, we develop a high quality corpus that pairs controllable, event-rich dialogue data with human-annotated labels. The GoT framework structures streaming predictions as an evolving graph, enabling a transformer to forecast the next speech act, generate concise justifications for its decisions, and dynamically refine its reasoning. Experiments on both synthetic and real duplex dialogues show that the framework delivers robust behavior detection, produces interpretable reasoning chains, and establishes a foundation for benchmarking conversational reasoning in full duplex spoken dialogue systems.

  </details>



- **GraphSeek: Next-Generation Graph Analytics with LLMs**  
  Maciej Besta, Łukasz Jarmocik, Orest Hrycyna, Shachar Klaiman, Konrad Mączka, Robert Gerstenberger, Jürgen Müller, Piotr Nyczyk, Hubert Niewiadomski, Torsten Hoefler  
  _2026-02-11_ · https://arxiv.org/abs/2602.11052v1 · `cs.DB`  
  <details><summary>Abstract</summary>

  Graphs are foundational across domains but remain hard to use without deep expertise. LLMs promise accessible natural language (NL) graph analytics, yet they fail to process industry-scale property graphs effectively and efficiently: such datasets are large, highly heterogeneous, structurally complex, and evolve dynamically. To address this, we devise a novel abstraction for complex multi-query analytics over such graphs. Its key idea is to replace brittle generation of graph queries directly from NL with planning over a Semantic Catalog that describes both the graph schema and the graph operations. Concretely, this induces a clean separation between a Semantic Plane for LLM planning and broader reasoning, and an Execution Plane for deterministic, database-grade query execution over the full dataset and tool implementations. This design yields substantial gains in both token efficiency and task effectiveness even with small-context LLMs. We use this abstraction as the basis of the first LLM-enhanced graph analytics framework called GraphSeek. GraphSeek achieves substantially higher success rates (e.g., 86% over enhanced LangChain) and points toward the next generation of affordable and accessible graph analytics that unify LLM reasoning with database-grade execution over large and complex property graphs.

  </details>



- **Reinforcing Chain-of-Thought Reasoning with Self-Evolving Rubrics**  
  Leheng Sheng, Wenchang Ma, Ruixin Hong, Xiang Wang, An Zhang, Tat-Seng Chua  
  _2026-02-11_ · https://arxiv.org/abs/2602.10885v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Despite chain-of-thought (CoT) playing crucial roles in LLM reasoning, directly rewarding it is difficult: training a reward model demands heavy human labeling efforts, and static RMs struggle with evolving CoT distributions and reward hacking. These challenges motivate us to seek an autonomous CoT rewarding approach that requires no human annotation efforts and can evolve gradually. Inspired by recent self-evolving training methods, we propose \textbf{RLCER} (\textbf{R}einforcement \textbf{L}earning with \textbf{C}oT Supervision via Self-\textbf{E}volving \textbf{R}ubrics), which enhances the outcome-centric RLVR by rewarding CoTs with self-proposed and self-evolving rubrics. We show that self-proposed and self-evolving rubrics provide reliable CoT supervision signals even without outcome rewards, enabling RLCER to outperform outcome-centric RLVR. Moreover, when used as in-prompt hints, these self-proposed rubrics further improve inference-time performance.

  </details>



- **Deep Learning-based Method for Expressing Knowledge Boundary of Black-Box LLM**  
  Haotian Sheng, Heyong Wang, Ming Hong, Hongman He, Junqiu Liu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10801v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have achieved remarkable success, however, the emergence of content generation distortion (hallucination) limits their practical applications. The core cause of hallucination lies in LLMs' lack of awareness regarding their stored internal knowledge, preventing them from expressing their knowledge state on questions beyond their internal knowledge boundaries, as humans do. However, existing research on knowledge boundary expression primarily focuses on white-box LLMs, leaving methods suitable for black-box LLMs which offer only API access without revealing internal parameters-largely unexplored. Against this backdrop, this paper proposes LSCL (LLM-Supervised Confidence Learning), a deep learning-based method for expressing the knowledge boundaries of black-box LLMs. Based on the knowledge distillation framework, this method designs a deep learning model. Taking the input question, output answer, and token probability from a black-box LLM as inputs, it constructs a mapping between the inputs and the model' internal knowledge state, enabling the quantification and expression of the black-box LLM' knowledge boundaries. Experiments conducted on diverse public datasets and with multiple prominent black-box LLMs demonstrate that LSCL effectively assists black-box LLMs in accurately expressing their knowledge boundaries. It significantly outperforms existing baseline models on metrics such as accuracy and recall rate. Furthermore, considering scenarios where some black-box LLMs do not support access to token probability, an adaptive alternative method is proposed. The performance of this alternative approach is close to that of LSCL and surpasses baseline models.

  </details>



- **Ecological mapping with geospatial foundation models**  
  Craig Mahlasi, Gciniwe S. Baloyi, Zaheed Gaffoor, Levente Klein, Anne Jones, Etienne Vos, Michal Muszynski, Geoffrey Dawson, Campbell Watson  
  _2026-02-11_ · https://arxiv.org/abs/2602.10720v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Geospatial foundation models (GFMs) are a fast-emerging paradigm for various geospatial tasks, such as ecological mapping. However, the utility of GFMs has not been fully explored for high-value use cases. This study aims to explore the utility, challenges and opportunities associated with the application of GFMs for ecological uses. In this regard, we fine-tune several pretrained AI models, namely, Prithvi-E0-2.0 and TerraMind, across three use cases, and compare this with a baseline ResNet-101 model. Firstly, we demonstrate TerraMind's LULC generation capabilities. Lastly, we explore the utility of the GFMs in forest functional trait mapping and peatlands detection. In all experiments, the GFMs outperform the baseline ResNet models. In general TerraMind marginally outperforms Prithvi. However, with additional modalities TerraMind significantly outperforms the baseline ResNet and Prithvi models. Nonetheless, consideration should be given to the divergence of input data from pretrained modalities. We note that these models would benefit from higher resolution and more accurate labels, especially for use cases where pixel-level dynamics need to be mapped.

  </details>



- **OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL**  
  Jinjie Shen, Jing Wu, Yaxiong Wang, Lechao Cheng, Shengeng Tang, Tianrui Hui, Nan Pu, Zhun Zhong  
  _2026-02-11_ · https://arxiv.org/abs/2602.10687v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Existing forgery detection methods are often limited to uni-modal or bi-modal settings, failing to handle the interleaved text, images, and videos prevalent in real-world misinformation. To bridge this gap, this paper targets to develop a unified framework for omnibus vision-language forgery detection and grounding. In this unified setting, the {interplay} between diverse modalities and the dual requirements of simultaneous detection and localization pose a critical ``difficulty bias`` problem: the simpler veracity classification task tends to dominate the gradients, leading to suboptimal performance in fine-grained grounding during multi-task optimization. To address this challenge, we propose \textbf{OmniVL-Guard}, a balanced reinforcement learning framework for omnibus vision-language forgery detection and grounding. Particularly, OmniVL-Guard comprises two core designs: Self-Evolving CoT Generatio and Adaptive Reward Scaling Policy Optimization (ARSPO). {Self-Evolving CoT Generation} synthesizes high-quality reasoning paths, effectively overcoming the cold-start challenge. Building upon this, {Adaptive Reward Scaling Policy Optimization (ARSPO)} dynamically modulates reward scales and task weights, ensuring a balanced joint optimization. Extensive experiments demonstrate that OmniVL-Guard significantly outperforms state-of-the-art methods and exhibits zero-shot robust generalization across out-of-domain scenarios.

  </details>


