# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-03-13 07:08 UTC_

Total papers shown: **13**


---

- **HumDex:Humanoid Dexterous Manipulation Made Easy**  
  Liang Heng, Yihe Tang, Jiajun Xu, Henghui Bao, Di Huang, Yue Wang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12260v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper investigates humanoid whole-body dexterous manipulation, where the efficient collection of high-quality demonstration data remains a central bottleneck. Existing teleoperation systems often suffer from limited portability, occlusion, or insufficient precision, which hinders their applicability to complex whole-body tasks. To address these challenges, we introduce HumDex, a portable teleoperation system designed for humanoid whole-body dexterous manipulation. Our system leverages IMU-based motion tracking to address the portability-precision trade-off, enabling accurate full-body tracking while remaining easy to deploy. For dexterous hand control, we further introduce a learning-based retargeting method that generates smooth and natural hand motions without manual parameter tuning. Beyond teleoperation, HumDex enables efficient collection of human motion data. Building on this capability, we propose a two-stage imitation learning framework that first pre-trains on diverse human motion data to learn generalizable priors, and then fine-tunes on robot data to bridge the embodiment gap for precise execution. We demonstrate that this approach significantly improves generalization to new configurations, objects, and backgrounds with minimal data acquisition costs. The entire system is fully reproducible and open-sourced at https://github.com/physical-superintelligence-lab/HumDex.

  </details>



- **ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control**  
  Chetan Borse, Zhixian Xie, Wei-Cheng Huang, Wanxin Jin  
  _2026-03-12_ · https://arxiv.org/abs/2603.12185v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Physics simulation for contact-rich robotics is often bottlenecked by contact resolution: mainstream engines enforce non-penetration and Coulomb friction via complementarity constraints or constrained optimization, requiring per-step iterative solves whose cost grows superlinearly with contact density. We present ComFree-Sim, a GPU-parallelized analytical contact physics engine built on complementarity-free contact modeling. ComFree-Sim computes contact impulses in closed form via an impedance-style prediction--correction update in the dual cone of Coulomb friction. Contact computation decouples across contact pairs and becomes separable across cone facets, mapping naturally to GPU kernels and yielding near-linear runtime scaling with the number of contacts. We further extend the formulation to a unified 6D contact model capturing tangential, torsional, and rolling friction, and introduce a practical dual-cone impedance heuristic. ComFree-Sim is implemented in Warp and exposed through a MuJoCo-compatible interface as a drop-in backend alternative to MuJoCo Warp (MJWarp). Experiments benchmark penetration, friction behaviors, stability, and simulation runtime scaling against MJWarp, demonstrating near-linear scaling and 2--3 times higher throughput in dense contact scenes with comparable physical fidelity. We deploy ComFree-Sim in real-time MPC for in-hand dexterous manipulation on a real-world multi-fingered LEAP hand and in dynamics-aware motion retargeting, demonstrating that low-latency simulation yields higher closed-loop success rates and enables practical high-frequency control in contact-rich tasks.

  </details>



- **RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset**  
  Yongzhong Wang, Keyu Zhu, Yong Zhong, Liqiong Wang, Jinyu Yang, Feng Zheng  
  _2026-03-12_ · https://arxiv.org/abs/2603.11811v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The acquisition of large-scale physical interaction data, a critical prerequisite for modern robot learning, is severely bottlenecked by the prohibitive cost and scalability limits of human-in-the-loop collection paradigms. To break this barrier, we introduce Robust Autonomous Data Acquisition for Robotics (RADAR), a fully autonomous, closed-loop data generation engine that completely removes human intervention from the collection cycle. RADAR elegantly divides the cognitive load into a four-module pipeline. Anchored by 2-5 3D human demonstrations as geometric priors, a Vision-Language Model first orchestrates scene-relevant task generation via precise semantic object grounding and skill retrieval. Next, a Graph Neural Network policy translates these subtasks into physical actions via in-context imitation learning. Following execution, the VLM performs automated success evaluation using a structured Visual Question Answering pipeline. Finally, to shatter the bottleneck of manual resets, a Finite State Machine orchestrates an autonomous environment reset and asymmetric data routing mechanism. Driven by simultaneous forward-reverse planning with a strict Last-In, First-Out causal sequence, the system seamlessly restores unstructured workspaces and robustly recovers from execution failures. This continuous brain-cerebellum synergy transforms data collection into a self-sustaining process. Extensive evaluations highlight RADAR's exceptional versatility. In simulation, our framework achieves up to 90% success rates on complex, long-horizon tasks, effortlessly solving challenges where traditional baselines plummet to near-zero performance. In real-world deployments, the system reliably executes diverse, contact-rich skills (e.g., deformable object manipulation) via few-shot adaptation without domain-specific fine-tuning, providing a highly scalable paradigm for robotic data acquisition.

  </details>



- **HandelBot: Real-World Piano Playing via Fast Adaptation of Dexterous Robot Policies**  
  Amber Xie, Haozhi Qi, Dorsa Sadigh  
  _2026-03-12_ · https://arxiv.org/abs/2603.12243v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mastering dexterous manipulation with multi-fingered hands has been a grand challenge in robotics for decades. Despite its potential, the difficulty of collecting high-quality data remains a primary bottleneck for high-precision tasks. While reinforcement learning and simulation-to-real-world transfer offer a promising alternative, the transferred policies often fail for tasks demanding millimeter-scale precision, such as bimanual piano playing. In this work, we introduce HandelBot, a framework that combines a simulation policy and rapid adaptation through a two-stage pipeline. Starting from a simulation-trained policy, we first apply a structured refinement stage to correct spatial alignments by adjusting lateral finger joints based on physical rollouts. Next, we use residual reinforcement learning to autonomously learn fine-grained corrective actions. Through extensive hardware experiments across five recognized songs, we demonstrate that HandelBot can successfully perform precise bimanual piano playing. Our system outperforms direct simulation deployment by a factor of 1.8x and requires only 30 minutes of physical interaction data.

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



- **LABSHIELD: A Multimodal Benchmark for Safety-Critical Reasoning and Planning in Scientific Laboratories**  
  Qianpu Sun, Xiaowei Chi, Yuhan Rui, Ying Li, Kuangzhi Ge, Jiajun Li, Sirui Han, Shanghang Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11987v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Artificial intelligence is increasingly catalyzing scientific automation, with multimodal large language model (MLLM) agents evolving from lab assistants into self-driving lab operators. This transition imposes stringent safety requirements on laboratory environments, where fragile glassware, hazardous substances, and high-precision laboratory equipment render planning errors or misinterpreted risks potentially irreversible. However, the safety awareness and decision-making reliability of embodied agents in such high-stakes settings remain insufficiently defined and evaluated. To bridge this gap, we introduce LABSHIELD, a realistic multi-view benchmark designed to assess MLLMs in hazard identification and safety-critical reasoning. Grounded in U.S. Occupational Safety and Health Administration (OSHA) standards and the Globally Harmonized System (GHS), LABSHIELD establishes a rigorous safety taxonomy spanning 164 operational tasks with diverse manipulation complexities and risk profiles. We evaluate 20 proprietary models, 9 open-source models, and 3 embodied models under a dual-track evaluation framework. Our results reveal a systematic gap between general-domain MCQ accuracy and Semi-open QA safety performance, with models exhibiting an average drop of 32.0% in professional laboratory scenarios, particularly in hazard interpretation and safety-aware planning. These findings underscore the urgent necessity for safety-centric reasoning frameworks to ensure reliable autonomous scientific experimentation in embodied laboratory contexts. The full dataset will be released soon.

  </details>



- **Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation**  
  Chongyang Xu, Yixian Zou, Ziliang Feng, Fanman Meng, Shuaicheng Liu  
  _2026-03-12_ · https://arxiv.org/abs/2603.11984v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Diffusion-based visuomotor policies effectively capture multimodal action distributions through iterative denoising, but their high inference latency limits real-time robotic control. Recent flow matching and consistency-based methods achieve single-step generation, yet sacrifice the ability to preserve distinct action modes, collapsing multimodal behaviors into averaged, often physically infeasible trajectories. We observe that the compute budget asymmetry in robotics (offline training vs.\ real-time inference) naturally motivates recovering this multimodal fidelity by shifting iterative refinement from inference time to training time. Building on this insight, we propose Ada3Drift, which learns a training-time drifting field that attracts predicted actions toward expert demonstration modes while repelling them from other generated samples, enabling high-fidelity single-step generation (1 NFE) from 3D point cloud observations. To handle the few-shot robotic regime, Ada3Drift further introduces a sigmoid-scheduled loss transition from coarse distribution learning to mode-sharpening refinement, and multi-scale field aggregation that captures action modes at varying spatial granularities. Experiments on three simulation benchmarks (Adroit, Meta-World, and RoboTwin) and real-world robotic manipulation tasks demonstrate that Ada3Drift achieves state-of-the-art performance while requiring $10\times$ fewer function evaluations than diffusion-based alternatives.

  </details>



- **OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams**  
  Yibin Yan, Jilan Xu, Shangzhe Di, Haoning Wu, Weidi Xie  
  _2026-03-12_ · https://arxiv.org/abs/2603.12265v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Modern visual agents require representations that are general, causal, and physically structured to operate in real-time streaming environments. However, current vision foundation models remain fragmented, specializing narrowly in image semantic perception, offline temporal modeling, or spatial geometry. This paper introduces OmniStream, a unified streaming visual backbone that effectively perceives, reconstructs, and acts from diverse visual inputs. By incorporating causal spatiotemporal attention and 3D rotary positional embeddings (3D-RoPE), our model supports efficient, frame-by-frame online processing of video streams via a persistent KV-cache. We pre-train OmniStream using a synergistic multi-task framework coupling static and temporal representation learning, streaming geometric reconstruction, and vision-language alignment on 29 datasets. Extensive evaluations show that, even with a strictly frozen backbone, OmniStream achieves consistently competitive performance with specialized experts across image and video probing, streaming geometric reconstruction, complex video and spatial reasoning, as well as robotic manipulation (unseen at training). Rather than pursuing benchmark-specific dominance, our work demonstrates the viability of training a single, versatile vision backbone that generalizes across semantic, spatial, and temporal reasoning, i.e., a more meaningful step toward general-purpose visual understanding for interactive and embodied agents.

  </details>



- **The Latent Color Subspace: Emergent Order in High-Dimensional Chaos**  
  Mateusz Pach, Jessica Bader, Quentin Bouniot, Serge Belongie, Zeynep Akata  
  _2026-03-12_ · https://arxiv.org/abs/2603.12261v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Text-to-image generation models have advanced rapidly, yet achieving fine-grained control over generated images remains difficult, largely due to limited understanding of how semantic information is encoded. We develop an interpretation of the color representation in the Variational Autoencoder latent space of FLUX.1 [Dev], revealing a structure reflecting Hue, Saturation, and Lightness. We verify our Latent Color Subspace (LCS) interpretation by demonstrating that it can both predict and explicitly control color, introducing a fully training-free method in FLUX based solely on closed-form latent-space manipulation. Code is available at https://github.com/ExplainableML/LCS.

  </details>



- **EgoIntent: An Egocentric Step-level Benchmark for Understanding What, Why, and Next**  
  Ye Pan, Chi Kit Wong, Yuanhuiyi Lyu, Hanqian Li, Jiahao Huo, Jiacheng Chen, Lutao Jiang, Xu Zheng, Xuming Hu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12147v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have demonstrated remarkable video reasoning capabilities across diverse tasks. However, their ability to understand human intent at a fine-grained level in egocentric videos remains largely unexplored. Existing benchmarks focus primarily on episode-level intent reasoning, overlooking the finer granularity of step-level intent understanding. Yet applications such as intelligent assistants, robotic imitation learning, and augmented reality guidance require understanding not only what a person is doing at each step, but also why and what comes next, in order to provide timely and context-aware support. To this end, we introduce EgoIntent, a step-level intent understanding benchmark for egocentric videos. It comprises 3,014 steps spanning 15 diverse indoor and outdoor daily-life scenarios, and evaluates models on three complementary dimensions: local intent (What), global intent (Why), and next-step plan (Next). Crucially, each clip is truncated immediately before the key outcome of the queried step (e.g., contact or grasp) occurs and contains no frames from subsequent steps, preventing future-frame leakage and enabling a clean evaluation of anticipatory step understanding and next-step planning. We evaluate 15 MLLMs, including both state-of-the-art closed-source and open-source models. Even the best-performing model achieves an average score of only 33.31 across the three intent dimensions, underscoring that step-level intent understanding in egocentric videos remains a highly challenging problem that calls for further investigation.

  </details>



- **Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics**  
  Ming-Hong Chen, Kuan-Chen Pan, You-De Huang, Xi Liu, Ping-Chun Hsieh  
  _2026-03-12_ · https://arxiv.org/abs/2603.12087v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Cross-domain reinforcement learning (CDRL) is meant to improve the data efficiency of RL by leveraging the data samples collected from a source domain to facilitate the learning in a similar target domain. Despite its potential, cross-domain transfer in RL is known to have two fundamental and intertwined challenges: (i) The source and target domains can have distinct state space or action space, and this makes direct transfer infeasible and thereby requires more sophisticated inter-domain mappings; (ii) The transferability of a source-domain model in RL is not easily identifiable a priori, and hence CDRL can be prone to negative effect during transfer. In this paper, we propose to jointly tackle these two challenges through the lens of \textit{cross-domain Bellman consistency} and \textit{hybrid critic}. Specifically, we first introduce the notion of cross-domain Bellman consistency as a way to measure transferability of a source-domain model. Then, we propose $Q$Avatar, which combines the Q functions from both the source and target domains with an adaptive hyperparameter-free weight function. Through this design, we characterize the convergence behavior of $Q$Avatar and show that $Q$Avatar achieves reliable transfer in the sense that it effectively leverages a source-domain Q function for knowledge transfer to the target domain. Through experiments, we demonstrate that $Q$Avatar achieves favorable transferability across various RL benchmark tasks, including locomotion and robot arm manipulation. Our code is available at https://rl-bandits-lab.github.io/Cross-Domain-RL/.

  </details>



- **COTONET: A custom cotton detection algorithm based on YOLO11 for stage of growth cotton boll detection**  
  Guillem González, Guillem Alenyà, Sergi Foix  
  _2026-03-12_ · https://arxiv.org/abs/2603.11717v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Cotton harvesting is a critical phase where cotton capsules are physically manipulated and can lead to fibre degradation. To maintain the highest quality, harvesting methods must emulate delicate manual grasping, to preserve cotton's intrinsic properties. Automating this process requires systems capable of recognising cotton capsules across various phenological stages. To address this challenge, we propose COTONET, an enhanced custom YOLO11 model tailored with attention mechanisms to improve the detection of difficult instances. The architecture incorporates gradients in non-learnable operations to enhance shape and feature extraction. Key architectural modifications include: the replacement of convolutional blocks with Squeeze-and-Exitation blocks, a redesigned backbone integrating attention mechanisms, and the substitution of standard upsampling operations for Content Aware Reassembly of Features (CARAFE). Additionally, we integrate Simple Attention Modules (SimAM) for primary feature aggregation and Parallel Hybrid Attention Mechanisms (PHAM) for channel-wise, spatial-wise and coordinate-wise attention in the downward neck path. This configuration offers increased flexibility and robustness for interpreting the complexity of cotton crop growth. COTONET aligns with small-to-medium YOLO models utilizing 7.6M parameters and 27.8 GFLOPS, making it suitable for low-resource edge computing and mobile robotics. COTONET outperforms the standard YOLO baselines, achieving a mAP50 of 81.1% and a mAP50-95 of 60.6%.

  </details>


