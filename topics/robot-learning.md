# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-03-13 07:08 UTC_

Total papers shown: **20**


---

- **RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset**  
  Yongzhong Wang, Keyu Zhu, Yong Zhong, Liqiong Wang, Jinyu Yang, Feng Zheng  
  _2026-03-12_ · https://arxiv.org/abs/2603.11811v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The acquisition of large-scale physical interaction data, a critical prerequisite for modern robot learning, is severely bottlenecked by the prohibitive cost and scalability limits of human-in-the-loop collection paradigms. To break this barrier, we introduce Robust Autonomous Data Acquisition for Robotics (RADAR), a fully autonomous, closed-loop data generation engine that completely removes human intervention from the collection cycle. RADAR elegantly divides the cognitive load into a four-module pipeline. Anchored by 2-5 3D human demonstrations as geometric priors, a Vision-Language Model first orchestrates scene-relevant task generation via precise semantic object grounding and skill retrieval. Next, a Graph Neural Network policy translates these subtasks into physical actions via in-context imitation learning. Following execution, the VLM performs automated success evaluation using a structured Visual Question Answering pipeline. Finally, to shatter the bottleneck of manual resets, a Finite State Machine orchestrates an autonomous environment reset and asymmetric data routing mechanism. Driven by simultaneous forward-reverse planning with a strict Last-In, First-Out causal sequence, the system seamlessly restores unstructured workspaces and robustly recovers from execution failures. This continuous brain-cerebellum synergy transforms data collection into a self-sustaining process. Extensive evaluations highlight RADAR's exceptional versatility. In simulation, our framework achieves up to 90% success rates on complex, long-horizon tasks, effortlessly solving challenges where traditional baselines plummet to near-zero performance. In real-world deployments, the system reliably executes diverse, contact-rich skills (e.g., deformable object manipulation) via few-shot adaptation without domain-specific fine-tuning, providing a highly scalable paradigm for robotic data acquisition.

  </details>



- **HumDex:Humanoid Dexterous Manipulation Made Easy**  
  Liang Heng, Yihe Tang, Jiajun Xu, Henghui Bao, Di Huang, Yue Wang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12260v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper investigates humanoid whole-body dexterous manipulation, where the efficient collection of high-quality demonstration data remains a central bottleneck. Existing teleoperation systems often suffer from limited portability, occlusion, or insufficient precision, which hinders their applicability to complex whole-body tasks. To address these challenges, we introduce HumDex, a portable teleoperation system designed for humanoid whole-body dexterous manipulation. Our system leverages IMU-based motion tracking to address the portability-precision trade-off, enabling accurate full-body tracking while remaining easy to deploy. For dexterous hand control, we further introduce a learning-based retargeting method that generates smooth and natural hand motions without manual parameter tuning. Beyond teleoperation, HumDex enables efficient collection of human motion data. Building on this capability, we propose a two-stage imitation learning framework that first pre-trains on diverse human motion data to learn generalizable priors, and then fine-tunes on robot data to bridge the embodiment gap for precise execution. We demonstrate that this approach significantly improves generalization to new configurations, objects, and backgrounds with minimal data acquisition costs. The entire system is fully reproducible and open-sourced at https://github.com/physical-superintelligence-lab/HumDex.

  </details>



- **Separable neural architectures as a primitive for unified predictive and generative intelligence**  
  Reza T. Batley, Apurba Sarker, Rajib Mostakim, Andrew Klichine, Sourav Saha  
  _2026-03-12_ · https://arxiv.org/abs/2603.12244v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Intelligent systems across physics, language and perception often exhibit factorisable structure, yet are typically modelled by monolithic neural architectures that do not explicitly exploit this structure. The separable neural architecture (SNA) addresses this by formalising a representational class that unifies additive, quadratic and tensor-decomposed neural models. By constraining interaction order and tensor rank, SNAs impose a structural inductive bias that factorises high-dimensional mappings into low-arity components. Separability need not be a property of the system itself: it often emerges in the coordinates or representations through which the system is expressed. Crucially, this coordinate-aware formulation reveals a structural analogy between chaotic spatiotemporal dynamics and linguistic autoregression. By treating continuous physical states as smooth, separable embeddings, SNAs enable distributional modelling of chaotic systems. This approach mitigates the nonphysical drift characteristics of deterministic operators whilst remaining applicable to discrete sequences. The compositional versatility of this approach is demonstrated across four domains: autonomous waypoint navigation via reinforcement learning, inverse generation of multifunctional microstructures, distributional modelling of turbulent flow and neural language modelling. These results establish the separable neural architecture as a domain-agnostic primitive for predictive and generative intelligence, capable of unifying both deterministic and distributional representations.

  </details>



- **SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics**  
  Mengzhen Liu, Enshen Zhou, Cheng Chi, Yi Han, Shanyu Rong, Liming Chen, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12193v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Active perception and manipulation are crucial for robots to interact with complex scenes. Existing methods struggle to unify semantic-driven active perception with robust, viewpoint-invariant execution. We propose SaPaVe, an end-to-end framework that jointly learns these capabilities in a data-efficient manner. Our approach decouples camera and manipulation actions rather than placing them in a shared action space, and follows a bottom-up training strategy: we first train semantic camera control on a large-scale dataset, then jointly optimize both action types using hybrid data. To support this framework, we introduce ActiveViewPose-200K, a dataset of 200k image-language-camera movement pairs for semantic camera movement learning, and a 3D geometry-aware module that improves execution robustness under dynamic viewpoints. We also present ActiveManip-Bench, the first benchmark for evaluating active manipulation beyond fixed-view settings. Extensive experiments in both simulation and real-world environments show that SaPaVe outperforms recent vision-language-action models such as GR00T N1 and \(π_0\), achieving up to 31.25\% higher success rates in real-world tasks. These results show that tightly coupled perception and execution, when trained with decoupled yet coordinated strategies, enable efficient and generalizable active manipulation. Project page: https://lmzpai.github.io/SaPaVe

  </details>



- **$Ψ_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation**  
  Songlin Wei, Hongyi Jing, Boqian Li, Zhenyu Zhao, Jiageng Mao, Zhenhao Ni, Sicheng He, Jie Liu, Xiawei Liu, Kaidi Kang, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12263v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We introduce $Ψ_0$ (Psi-Zero), an open foundation model to address challenging humanoid loco-manipulation tasks. While existing approaches often attempt to address this fundamental problem by co-training on large and diverse human and humanoid data, we argue that this strategy is suboptimal due to the fundamental kinematic and motion disparities between humans and humanoid robots. Therefore, data efficiency and model performance remain unsatisfactory despite the considerable data volume. To address this challenge, \ours\;decouples the learning process to maximize the utility of heterogeneous data sources. Specifically, we propose a staged training paradigm with different learning objectives: First, we autoregressively pre-train a VLM backbone on large-scale egocentric human videos to acquire generalizable visual-action representations. Then, we post-train a flow-based action expert on high-quality humanoid robot data to learn precise robot joint control. Our research further identifies a critical yet often overlooked data recipe: in contrast to approaches that scale with noisy Internet clips or heterogeneous cross-embodiment robot datasets, we demonstrate that pre-training on high-quality egocentric human manipulation data followed by post-training on domain-specific real-world humanoid trajectories yields superior performance. Extensive real-world experiments demonstrate that \ours\ achieves the best performance using only about 800 hours of human video data and 30 hours of real-world robot data, outperforming baselines pre-trained on more than 10$\times$ as much data by over 40\% in overall success rate across multiple tasks. We will open-source the entire ecosystem to the community, including a data processing and training pipeline, a humanoid foundation model, and a real-time action inference engine.

  </details>



- **HandelBot: Real-World Piano Playing via Fast Adaptation of Dexterous Robot Policies**  
  Amber Xie, Haozhi Qi, Dorsa Sadigh  
  _2026-03-12_ · https://arxiv.org/abs/2603.12243v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mastering dexterous manipulation with multi-fingered hands has been a grand challenge in robotics for decades. Despite its potential, the difficulty of collecting high-quality data remains a primary bottleneck for high-precision tasks. While reinforcement learning and simulation-to-real-world transfer offer a promising alternative, the transferred policies often fail for tasks demanding millimeter-scale precision, such as bimanual piano playing. In this work, we introduce HandelBot, a framework that combines a simulation policy and rapid adaptation through a two-stage pipeline. Starting from a simulation-trained policy, we first apply a structured refinement stage to correct spatial alignments by adjusting lateral finger joints based on physical rollouts. Next, we use residual reinforcement learning to autonomously learn fine-grained corrective actions. Through extensive hardware experiments across five recognized songs, we demonstrate that HandelBot can successfully perform precise bimanual piano playing. Our system outperforms direct simulation deployment by a factor of 1.8x and requires only 30 minutes of physical interaction data.

  </details>



- **Sim-to-reality adaptation for Deep Reinforcement Learning applied to an underwater docking application**  
  Alaaeddine Chaarani, Narcis Palomeras, Pere Ridao  
  _2026-03-12_ · https://arxiv.org/abs/2603.12020v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deep Reinforcement Learning (DRL) offers a robust alternative to traditional control methods for autonomous underwater docking, particularly in adapting to unpredictable environmental conditions. However, bridging the "sim-to-real" gap and managing high training latencies remain significant bottlenecks for practical deployment. This paper presents a systematic approach for autonomous docking using the Girona Autonomous Underwater Vehicle (AUV) by leveraging a high-fidelity digital twin environment. We adapted the Stonefish simulator into a multiprocessing RL framework to significantly accelerate the learning process while incorporating realistic AUV dynamics, collision models, and sensor noise. Using the Proximal Policy Optimization (PPO) algorithm, we developed a 6-DoF control policy trained in a headless environment with randomized starting positions to ensure generalized performance. Our reward structure accounts for distance, orientation, action smoothness, and adaptive collision penalties to facilitate soft docking. Experimental results demonstrate that the agent achieved a success rate of over 90% in simulation. Furthermore, successful validation in a physical test tank confirmed the efficacy of the sim-to-reality adaptation, with the DRL controller exhibiting emergent behaviors such as pitch-based braking and yaw oscillations to assist in mechanical alignment.

  </details>



- **Learning Visuomotor Policy for Multi-Robot Laser Tag Game**  
  Kai Li, Shiyu Zhao  
  _2026-03-12_ · https://arxiv.org/abs/2603.11980v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In this paper, we study multi robot laser tag, a simplified yet practical shooting-game-style task. Classic modular approaches on these tasks face challenges such as limited observability and reliance on depth mapping and inter robot communication. To overcome these issues, we present an end-to-end visuomotor policy that maps images directly to robot actions. We train a high performing teacher policy with multi agent reinforcement learning and distill its knowledge into a vision-based student policy. Technical designs, including a permutation-invariant feature extractor and depth heatmap input, improve performance over standard architectures. Our policy outperforms classic methods by 16.7% in hitting accuracy and 6% in collision avoidance, and is successfully deployed on real robots. Code will be released publicly.

  </details>



- **Exhaustive Circuit Mapping of a Single-Cell Foundation Model Reveals Massive Redundancy, Heavy-Tailed Hub Architecture, and Layer-Dependent Differentiation Control**  
  Ihor Kendiukhov  
  _2026-03-12_ · https://arxiv.org/abs/2603.11940v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Mechanistic interpretability of biological foundation models has relied on selective feature sampling, pairwise interaction testing, and observational trajectory analysis. Each of these can introduce systematic bias. Here we present three experiments that address these limitations through exhaustive circuit tracing, higher order combinatorial ablation, and causal trajectory steering in Geneformer, a transformer based single cell foundation model. First, exhaustive tracing of all 4065 active sparse autoencoder features at layer 5 yields 1393850 significant downstream edges, a 27 fold expansion over selective sampling. This reveals a heavy tailed hub distribution in which 1.8 percent of features account for disproportionate connectivity and 40 percent of the top 20 hubs lack biological annotation. These results indicate systematic annotation bias in prior selective analyses. Second, three way combinatorial ablation across 8 feature triplets shows that redundancy deepens monotonically with interaction order, with a three way ratio of 0.59 versus a pairwise ratio of 0.74, and with zero synergy. This confirms that the model architecture is subadditive at all tested orders. Third, trajectory guided feature steering establishes a causal link between layer position and differentiation directionality. Late layer features at L17 consistently push cell states toward maturity, with fraction positive equal to 1.0. Early and mid layer features at L0 and L11 mostly push away from maturity, with fraction positive ranging from 0.00 to 0.58. Together these results move from correlation toward causal evidence for layer dependent control of cell state.

  </details>



- **Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework**  
  Chingkwun Lam, Jiaxin Li, Lingfei Zhang, Kuo Zhao  
  _2026-03-12_ · https://arxiv.org/abs/2603.11768v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Long-term memory has emerged as a foundational component of autonomous Large Language Model (LLM) agents, enabling continuous adaptation, lifelong multimodal learning, and sophisticated reasoning. However, as memory systems transition from static retrieval databases to dynamic, agentic mechanisms, critical concerns regarding memory governance, semantic drift, and privacy vulnerabilities have surfaced. While recent surveys have focused extensively on memory retrieval efficiency, they largely overlook the emergent risks of memory corruption in highly dynamic environments. To address these emerging challenges, we propose the Stability and Safety-Governed Memory (SSGM) framework, a conceptual governance architecture. SSGM decouples memory evolution from execution by enforcing consistency verification, temporal decay modeling, and dynamic access control prior to any memory consolidation. Through formal analysis and architectural decomposition, we show how SSGM can mitigate topology-induced knowledge leakage where sensitive contexts are solidified into long-term storage, and help prevent semantic drift where knowledge degrades through iterative summarization. Ultimately, this work provides a comprehensive taxonomy of memory corruption risks and establishes a robust governance paradigm for deploying safe, persistent, and reliable agentic memory systems.

  </details>



- **OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams**  
  Yibin Yan, Jilan Xu, Shangzhe Di, Haoning Wu, Weidi Xie  
  _2026-03-12_ · https://arxiv.org/abs/2603.12265v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Modern visual agents require representations that are general, causal, and physically structured to operate in real-time streaming environments. However, current vision foundation models remain fragmented, specializing narrowly in image semantic perception, offline temporal modeling, or spatial geometry. This paper introduces OmniStream, a unified streaming visual backbone that effectively perceives, reconstructs, and acts from diverse visual inputs. By incorporating causal spatiotemporal attention and 3D rotary positional embeddings (3D-RoPE), our model supports efficient, frame-by-frame online processing of video streams via a persistent KV-cache. We pre-train OmniStream using a synergistic multi-task framework coupling static and temporal representation learning, streaming geometric reconstruction, and vision-language alignment on 29 datasets. Extensive evaluations show that, even with a strictly frozen backbone, OmniStream achieves consistently competitive performance with specialized experts across image and video probing, streaming geometric reconstruction, complex video and spatial reasoning, as well as robotic manipulation (unseen at training). Rather than pursuing benchmark-specific dominance, our work demonstrates the viability of training a single, versatile vision backbone that generalizes across semantic, spatial, and temporal reasoning, i.e., a more meaningful step toward general-purpose visual understanding for interactive and embodied agents.

  </details>



- **Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously**  
  Yiran Guan, Liang Yin, Dingkang Liang, Jianzhong Ju, Zhenbo Luo, Jian Luan, Yuliang Liu, Xiang Bai  
  _2026-03-12_ · https://arxiv.org/abs/2603.12262v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Online Video Large Language Models (VideoLLMs) play a critical role in supporting responsive, real-time interaction. Existing methods focus on streaming perception, lacking a synchronized logical reasoning stream. However, directly applying test-time scaling methods incurs unacceptable response latency. To address this trade-off, we propose Video Streaming Thinking (VST), a novel paradigm for streaming video understanding. It supports a thinking while watching mechanism, which activates reasoning over incoming video clips during streaming. This design improves timely comprehension and coherent cognition while preserving real-time responsiveness by amortizing LLM reasoning latency over video playback. Furthermore, we introduce a comprehensive post-training pipeline that integrates VST-SFT, which structurally adapts the offline VideoLLM to causal streaming reasoning, and VST-RL, which provides end-to-end improvement through self-exploration in a multi-turn video interaction environment. Additionally, we devise an automated training-data synthesis pipeline that uses video knowledge graphs to generate high-quality streaming QA pairs, with an entity-relation grounded streaming Chain-of-Thought to enforce multi-evidence reasoning and sustained attention to the video stream. Extensive evaluations show that VST-7B performs strongly on online benchmarks, e.g. 79.5% on StreamingBench and 59.3% on OVO-Bench. Meanwhile, VST remains competitive on offline long-form or reasoning benchmarks. Compared with Video-R1, VST responds 15.7 times faster and achieves +5.4% improvement on VideoHolmes, demonstrating higher efficiency and strong generalization across diverse video understanding tasks. Code, data, and models will be released at https://github.com/1ranGuan/VST.

  </details>



- **DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning**  
  Yujie Wei, Xinyu Liu, Shiwei Zhang, Hangjie Yuan, Jinbo Xing, Zhekai Chen, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12257v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  While large-scale diffusion models have revolutionized video synthesis, achieving precise control over both multi-subject identity and multi-granularity motion remains a significant challenge. Recent attempts to bridge this gap often suffer from limited motion granularity, control ambiguity, and identity degradation, leading to suboptimal performance on identity preservation and motion control. In this work, we present DreamVideo-Omni, a unified framework enabling harmonious multi-subject customization with omni-motion control via a progressive two-stage training paradigm. In the first stage, we integrate comprehensive control signals for joint training, encompassing subject appearances, global motion, local dynamics, and camera movements. To ensure robust and precise controllability, we introduce a condition-aware 3D rotary positional embedding to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance. Furthermore, to resolve multi-subject ambiguity, we introduce group and role embeddings to explicitly anchor motion signals to specific identities, effectively disentangling complex scenes into independent controllable instances. In the second stage, to mitigate identity degradation, we design a latent identity reward feedback learning paradigm by training a latent identity reward model upon a pretrained video diffusion backbone. This provides motion-aware identity rewards in the latent space, prioritizing identity preservation aligned with human preferences. Supported by our curated large-scale dataset and the comprehensive DreamOmni Bench for multi-subject and omni-motion control evaluation, DreamVideo-Omni demonstrates superior performance in generating high-quality videos with precise controllability.

  </details>



- **Linking Perception, Confidence and Accuracy in MLLMs**  
  Yuetian Du, Yucheng Wang, Rongyu Zhang, Zhijie Xu, Boyu Yang, Ming Kong, Jie Liu, Qiang Zhu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12149v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advances in Multi-modal Large Language Models (MLLMs) have predominantly focused on enhancing visual perception to improve accuracy. However, a critical question remains unexplored: Do models know when they do not know? Through a probing experiment, we reveal a severe confidence miscalibration problem in MLLMs. To address this, we propose Confidence-Driven Reinforcement Learning (CDRL), which uses original-noise image pairs and a novel confidence-based reward to enhance perceptual sensitivity and robustly calibrate the model's confidence. Beyond training benefits, calibrated confidence enables more effective test-time scaling as a free lunch. We further propose Confidence-Aware Test-Time Scaling (CA-TTS), which dynamically coordinates Self-Consistency, Self-Reflection, and Visual Self-Check modules guided by confidence signals. An Expert Model acts in multiple roles (e.g., Planner, Critic, Voter) to schedule these modules and provide external verification. Our integrated framework establishes new state-of-the-art results with consistent 8.8% gains across four benchmarks. More ablation studies demonstrate the effectiveness of each module and scaling superiority.

  </details>



- **EgoIntent: An Egocentric Step-level Benchmark for Understanding What, Why, and Next**  
  Ye Pan, Chi Kit Wong, Yuanhuiyi Lyu, Hanqian Li, Jiahao Huo, Jiacheng Chen, Lutao Jiang, Xu Zheng, Xuming Hu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12147v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have demonstrated remarkable video reasoning capabilities across diverse tasks. However, their ability to understand human intent at a fine-grained level in egocentric videos remains largely unexplored. Existing benchmarks focus primarily on episode-level intent reasoning, overlooking the finer granularity of step-level intent understanding. Yet applications such as intelligent assistants, robotic imitation learning, and augmented reality guidance require understanding not only what a person is doing at each step, but also why and what comes next, in order to provide timely and context-aware support. To this end, we introduce EgoIntent, a step-level intent understanding benchmark for egocentric videos. It comprises 3,014 steps spanning 15 diverse indoor and outdoor daily-life scenarios, and evaluates models on three complementary dimensions: local intent (What), global intent (Why), and next-step plan (Next). Crucially, each clip is truncated immediately before the key outcome of the queried step (e.g., contact or grasp) occurs and contains no frames from subsequent steps, preventing future-frame leakage and enabling a clean evaluation of anticipatory step understanding and next-step planning. We evaluate 15 MLLMs, including both state-of-the-art closed-source and open-source models. Even the best-performing model achieves an average score of only 33.31 across the three intent dimensions, underscoring that step-level intent understanding in egocentric videos remains a highly challenging problem that calls for further investigation.

  </details>



- **Automatic Generation of High-Performance RL Environments**  
  Seth Karten, Rahul Dev Appapogu, Chi Jin  
  _2026-03-12_ · https://arxiv.org/abs/2603.12145v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Translating complex reinforcement learning (RL) environments into high-performance implementations has traditionally required months of specialized engineering. We present a reusable recipe - a generic prompt template, hierarchical verification, and iterative agent-assisted repair - that produces semantically equivalent high-performance environments for <$10 in compute cost. We demonstrate three distinct workflows across five environments. Direct translation (no prior performance implementation exists): EmuRust (1.5x PPO speedup via Rust parallelism for a Game Boy emulator) and PokeJAX, the first GPU-parallel Pokemon battle simulator (500M SPS random action, 15.2M SPS PPO; 22,320x over the TypeScript reference). Translation verified against existing performance implementations: throughput parity with MJX (1.04x) and 5x over Brax at matched GPU batch sizes (HalfCheetah JAX); 42x PPO (Puffer Pong). New environment creation: TCGJax, the first deployable JAX Pokemon TCG engine (717K SPS random action, 153K SPS PPO; 6.6x over the Python reference), synthesized from a web-extracted specification. At 200M parameters, the environment overhead drops below 4% of training time. Hierarchical verification (property, interaction, and rollout tests) confirms semantic equivalence for all five environments; cross-backend policy transfer confirms zero sim-to-sim gap for all five environments. TCGJax, synthesized from a private reference absent from public repositories, serves as a contamination control for agent pretraining data concerns. The paper contains sufficient detail - including representative prompts, verification methodology, and complete results - that a coding agent could reproduce the translations directly from the manuscript.

  </details>



- **Taming the Adversary: Stable Minimax Deep Deterministic Policy Gradient via Fractional Objectives**  
  Taeho Lee, Donghwan Lee  
  _2026-03-12_ · https://arxiv.org/abs/2603.12110v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has achieved remarkable success in a wide range of control and decision-making tasks. However, RL agents often exhibit unstable or degraded performance when deployed in environments subject to unexpected external disturbances and model uncertainties. Consequently, ensuring reliable performance under such conditions remains a critical challenge. In this paper, we propose minimax deep deterministic policy gradient (MMDDPG), a framework for learning disturbance-resilient policies in continuous control tasks. The training process is formulated as a minimax optimization problem between a user policy and an adversarial disturbance policy. In this problem, the user learns a robust policy that minimizes the objective function, while the adversary generates disturbances that maximize it. To stabilize this interaction, we introduce a fractional objective that balances task performance and disturbance magnitude. This objective prevents excessively aggressive disturbances and promotes robust learning. Experimental evaluations in MuJoCo environments demonstrate that the proposed MMDDPG achieves significantly improved robustness against both external force perturbations and model parameter variations.

  </details>



- **A Robust and Efficient Multi-Agent Reinforcement Learning Framework for Traffic Signal Control**  
  Sheng-You Huang, Hsiao-Chuan Chang, Yen-Chi Chen, Ting-Han Wei, I-Hau Yeh, Sheng-Yao Kuan, Chien-Yao Wang, Hsuan-Han Lee, I-Chen Wu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12096v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Reinforcement Learning (RL) in Traffic Signal Control (TSC) faces significant hurdles in real-world deployment due to limited generalization to dynamic traffic flow variations. Existing approaches often overfit static patterns and use action spaces incompatible with driver expectations. This paper proposes a robust Multi-Agent Reinforcement Learning (MARL) framework validated in the Vissim traffic simulator. The framework integrates three mechanisms: (1) Turning Ratio Randomization, a training strategy that exposes agents to dynamic turning probabilities to enhance robustness against unseen scenarios; (2) a stability-oriented Exponential Phase Duration Adjustment action space, which balances responsiveness and precision through cyclical, exponential phase adjustments; and (3) a Neighbor-Based Observation scheme utilizing the MAPPO algorithm with Centralized Training with Decentralized Execution (CTDE). By leveraging centralized updates, this approach approximates the efficacy of global observations while maintaining scalable local communication. Experimental results demonstrate that our framework outperforms standard RL baselines, reducing average waiting time by over 10%. The proposed model exhibits superior generalization in unseen traffic scenarios and maintains high control stability, offering a practical solution for adaptive signal control.

  </details>



- **Human-Centred LLM Privacy Audits: Findings and Frictions**  
  Dimitri Staufer, Kirsten Morehouse, David Hartmann, Bettina Berendt  
  _2026-03-12_ · https://arxiv.org/abs/2603.12094v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) learn statistical associations from massive training corpora and user interactions, and deployed systems can surface or infer information about individuals. Yet people lack practical ways to inspect what a model associates with their name. We report interim findings from an ongoing study and introduce LMP2, a browser-based self-audit tool. In two user studies ($N_{total}{=}458$), GPT-4o predicts 11 of 50 features for everyday people with $\ge$60\% accuracy, and participants report wanting control over LLM-generated associations despite not considering all outputs privacy violations. To validate our probing method, we evaluate eight LLMs on public figures and non-existent names, observing clear separation between stable name-conditioned associations and model defaults. Our findings also contribute to exposing a broader generative AI evaluation crisis: when outputs are probabilistic, context-dependent, and user-mediated through elicitation, what model--individual associations even include is under-specified and operationalisation relies on crafting probes and metrics that are hard to validate or compare. To move towards reliable, actionable human-centred LLM privacy audits, we identify nine frictions that emerged in our study and offer recommendations for future work and the design of human-centred LLM privacy audits.

  </details>



- **Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics**  
  Ming-Hong Chen, Kuan-Chen Pan, You-De Huang, Xi Liu, Ping-Chun Hsieh  
  _2026-03-12_ · https://arxiv.org/abs/2603.12087v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Cross-domain reinforcement learning (CDRL) is meant to improve the data efficiency of RL by leveraging the data samples collected from a source domain to facilitate the learning in a similar target domain. Despite its potential, cross-domain transfer in RL is known to have two fundamental and intertwined challenges: (i) The source and target domains can have distinct state space or action space, and this makes direct transfer infeasible and thereby requires more sophisticated inter-domain mappings; (ii) The transferability of a source-domain model in RL is not easily identifiable a priori, and hence CDRL can be prone to negative effect during transfer. In this paper, we propose to jointly tackle these two challenges through the lens of \textit{cross-domain Bellman consistency} and \textit{hybrid critic}. Specifically, we first introduce the notion of cross-domain Bellman consistency as a way to measure transferability of a source-domain model. Then, we propose $Q$Avatar, which combines the Q functions from both the source and target domains with an adaptive hyperparameter-free weight function. Through this design, we characterize the convergence behavior of $Q$Avatar and show that $Q$Avatar achieves reliable transfer in the sense that it effectively leverages a source-domain Q function for knowledge transfer to the target domain. Through experiments, we demonstrate that $Q$Avatar achieves favorable transferability across various RL benchmark tasks, including locomotion and robot arm manipulation. Our code is available at https://rl-bandits-lab.github.io/Cross-Domain-RL/.

  </details>


