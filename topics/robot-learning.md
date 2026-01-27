# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-01-27 06:52 UTC_

Total papers shown: **20**


---

- **$α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks**  
  Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah  
  _2026-01-26_ · https://arxiv.org/abs/2601.18754v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Autonomous unmanned aerial vehicle (UAV) systems are increasingly deployed in safety-critical, networked environments where they must operate reliably in the presence of malicious adversaries. While recent benchmarks have evaluated large language model (LLM)-based UAV agents in reasoning, navigation, and efficiency, systematic assessment of security, resilience, and trust under adversarial conditions remains largely unexplored, particularly in emerging 6G-enabled settings. We introduce $α^{3}$-SecBench, the first large-scale evaluation suite for assessing the security-aware autonomy of LLM-based UAV agents under realistic adversarial interference. Building on multi-turn conversational UAV missions from $α^{3}$-Bench, the framework augments benign episodes with 20,000 validated security overlay attack scenarios targeting seven autonomy layers, including sensing, perception, planning, control, communication, edge/cloud infrastructure, and LLM reasoning. $α^{3}$-SecBench evaluates agents across three orthogonal dimensions: security (attack detection and vulnerability attribution), resilience (safe degradation behavior), and trust (policy-compliant tool usage). We evaluate 23 state-of-the-art LLMs from major industrial providers and leading AI labs using thousands of adversarially augmented UAV episodes sampled from a corpus of 113,475 missions spanning 175 threat types. While many models reliably detect anomalous behavior, effective mitigation, vulnerability attribution, and trustworthy control actions remain inconsistent. Normalized overall scores range from 12.9% to 57.1%, highlighting a significant gap between anomaly detection and security-aware autonomous decision-making. We release $α^{3}$-SecBench on GitHub: https://github.com/maferrag/AlphaSecBench

  </details>



- **TC-IDM: Grounding Video Generation for Executable Zero-shot Robot Motion**  
  Weishi Mi, Yong Bao, Xiaowei Chi, Xiaozhu Ju, Zhiyuan Qin, Kuangzhi Ge, Kai Tang, Peidong Jia, Shanghang Zhang, Jian Tang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18323v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The vision-language-action (VLA) paradigm has enabled powerful robotic control by leveraging vision-language models, but its reliance on large-scale, high-quality robot data limits its generalization. Generative world models offer a promising alternative for general-purpose embodied AI, yet a critical gap remains between their pixel-level plans and physically executable actions. To this end, we propose the Tool-Centric Inverse Dynamics Model (TC-IDM). By focusing on the tool's imagined trajectory as synthesized by the world model, TC-IDM establishes a robust intermediate representation that bridges the gap between visual planning and physical control. TC-IDM extracts the tool's point cloud trajectories via segmentation and 3D motion estimation from generated videos. Considering diverse tool attributes, our architecture employs decoupled action heads to project these planned trajectories into 6-DoF end-effector motions and corresponding control signals. This plan-and-translate paradigm not only supports a wide range of end-effectors but also significantly improves viewpoint invariance. Furthermore, it exhibits strong generalization capabilities across long-horizon and out-of-distribution tasks, including interacting with deformable objects. In real-world evaluations, the world model with TC-IDM achieves an average success rate of 61.11 percent, with 77.7 percent on simple tasks and 38.46 percent on zero-shot deformable object tasks. It substantially outperforms end-to-end VLA-style baselines and other inverse dynamics models.

  </details>



- **A Pragmatic VLA Foundation Model**  
  Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18692v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Offering great potential in robotic manipulation, a capable Vision-Language-Action (VLA) foundation model is expected to faithfully generalize across tasks and platforms while ensuring cost efficiency (e.g., data and GPU hours required for adaptation). To this end, we develop LingBot-VLA with around 20,000 hours of real-world data from 9 popular dual-arm robot configurations. Through a systematic assessment on 3 robotic platforms, each completing 100 tasks with 130 post-training episodes per task, our model achieves clear superiority over competitors, showcasing its strong performance and broad generalizability. We have also built an efficient codebase, which delivers a throughput of 261 samples per second per GPU with an 8-GPU training setup, representing a 1.5~2.8$\times$ (depending on the relied VLM base model) speedup over existing VLA-oriented codebases. The above features ensure that our model is well-suited for real-world deployment. To advance the field of robot learning, we provide open access to the code, base model, and benchmark data, with a focus on enabling more challenging tasks and promoting sound evaluation standards.

  </details>



- **Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents**  
  Zhihan Liu, Lin Guan, Yixin Nie, Kai Zhang, Zhuoqun Hao, Lin Chen, Asli Celikyilmaz, Zhaoran Wang, Na Zhang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18217v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Generalist LLM agents are often post-trained on a narrow set of environments but deployed across far broader, unseen domains. In this work, we investigate the challenge of agentic post-training when the eventual test domains are unknown. Specifically, we analyze which properties of reinforcement learning (RL) environments and modeling choices have the greatest influence on out-of-domain performance. First, we identify two environment axes that strongly correlate with cross-domain generalization: (i) state information richness, i.e., the amount of information for the agent to process from the state, and (ii) planning complexity, estimated via goal reachability and trajectory length under a base policy. Notably, domain realism and text-level similarity are not the primary factors; for instance, the simple grid-world domain Sokoban leads to even stronger generalization in SciWorld than the more realistic ALFWorld. Motivated by these findings, we further show that increasing state information richness alone can already effectively improve cross-domain robustness. We propose a randomization technique, which is low-overhead and broadly applicable: add small amounts of distractive goal-irrelevant features to the state to make it richer without altering the task. Beyond environment-side properties, we also examine several modeling choices: (a) SFT warmup or mid-training helps prevent catastrophic forgetting during RL but undermines generalization to domains that are not included in the mid-training datamix; and (b) turning on step-by-step thinking during RL, while not always improving in-domain performance, plays a crucial role in preserving generalization.

  </details>



- **3DGesPolicy: Phoneme-Aware Holistic Co-Speech Gesture Generation Based on Action Control**  
  Xuanmeng Sha, Liyun Zhang, Tomohiro Mashita, Naoya Chiba, Yuki Uranishi  
  _2026-01-26_ · https://arxiv.org/abs/2601.18451v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Generating holistic co-speech gestures that integrate full-body motion with facial expressions suffers from semantically incoherent coordination on body motion and spatially unstable meaningless movements due to existing part-decomposed or frame-level regression methods, We introduce 3DGesPolicy, a novel action-based framework that reformulates holistic gesture generation as a continuous trajectory control problem through diffusion policy from robotics. By modeling frame-to-frame variations as unified holistic actions, our method effectively learns inter-frame holistic gesture motion patterns and ensures both spatially and semantically coherent movement trajectories that adhere to realistic motion manifolds. To further bridge the gap in expressive alignment, we propose a Gesture-Audio-Phoneme (GAP) fusion module that can deeply integrate and refine multi-modal signals, ensuring structured and fine-grained alignment between speech semantics, body motion, and facial expressions. Extensive quantitative and qualitative experiments on the BEAT2 dataset demonstrate the effectiveness of our 3DGesPolicy across other state-of-the-art methods in generating natural, expressive, and highly speech-aligned holistic gestures.

  </details>



- **Unsupervised Text Segmentation via Kernel Change-Point Detection on Sentence Embeddings**  
  Mumin Jia, Jairo Diaz-Rodriguez  
  _2026-01-26_ · https://arxiv.org/abs/2601.18788v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Unsupervised text segmentation is crucial because boundary labels are expensive, subjective, and often fail to transfer across domains and granularity choices. We propose Embed-KCPD, a training-free method that represents sentences as embedding vectors and estimates boundaries by minimizing a penalized KCPD objective. Beyond the algorithmic instantiation, we develop, to our knowledge, the first dependence-aware theory for KCPD under $m$-dependent sequences, a finite-memory abstraction of short-range dependence common in language. We prove an oracle inequality for the population penalized risk and a localization guarantee showing that each true change point is recovered within a window that is small relative to segment length. To connect theory to practice, we introduce an LLM-based simulation framework that generates synthetic documents with controlled finite-memory dependence and known boundaries, validating the predicted scaling behavior. Across standard segmentation benchmarks, Embed-KCPD often outperforms strong unsupervised baselines. A case study on Taylor Swift's tweets illustrates that Embed-KCPD combines strong theoretical guarantees, simulated reliability, and practical effectiveness for text segmentation.

  </details>



- **Multi-Objective Reinforcement Learning for Efficient Tactical Decision Making for Trucks in Highway Traffic**  
  Deepthi Pathare, Leo Laine, Morteza Haghir Chehreghani  
  _2026-01-26_ · https://arxiv.org/abs/2601.18783v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Balancing safety, efficiency, and operational costs in highway driving poses a challenging decision-making problem for heavy-duty vehicles. A central difficulty is that conventional scalar reward formulations, obtained by aggregating these competing objectives, often obscure the structure of their trade-offs. We present a Proximal Policy Optimization based multi-objective reinforcement learning framework that learns a continuous set of policies explicitly representing these trade-offs and evaluates it on a scalable simulation platform for tactical decision making in trucks. The proposed approach learns a continuous set of Pareto-optimal policies that capture the trade-offs among three conflicting objectives: safety, quantified in terms of collisions and successful completion; energy efficiency and time efficiency, quantified using energy cost and driver cost, respectively. The resulting Pareto frontier is smooth and interpretable, enabling flexibility in choosing driving behavior along different conflicting objectives. This framework allows seamless transitions between different driving policies without retraining, yielding a robust and adaptive decision-making strategy for autonomous trucking applications.

  </details>



- **Dep-Search: Learning Dependency-Aware Reasoning Traces with Persistent Memory**  
  Yanming Liu, Xinyue Peng, Zixuan Yan, Yanxin Shen, Wenjie Xu, Yuefeng Huang, Xinyi Wang, Jiannan Cao, Jianwei Yin, Xuhong Zhang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18771v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, particularly when augmented with search mechanisms that enable systematic exploration of external knowledge bases. The field has evolved from traditional retrieval-augmented generation (RAG) frameworks to more sophisticated search-based frameworks that orchestrate multi-step reasoning through explicit search strategies. However, existing search frameworks still rely heavily on implicit natural language reasoning to determine search strategies and how to leverage retrieved information across reasoning steps. This reliance on implicit reasoning creates fundamental challenges for managing dependencies between sub-questions, efficiently reusing previously retrieved knowledge, and learning optimal search strategies through reinforcement learning. To address these limitations, we propose Dep-Search, a dependency-aware search framework that advances beyond existing search frameworks by integrating structured reasoning, retrieval, and persistent memory through GRPO. Dep-Search introduces explicit control mechanisms that enable the model to decompose questions with dependency relationships, retrieve information when needed, access previously stored knowledge from memory, and summarize long reasoning contexts into reusable memory entries. Through extensive experiments on seven diverse question answering datasets, we demonstrate that Dep-Search significantly enhances LLMs' ability to tackle complex multi-hop reasoning tasks, achieving substantial improvements over strong baselines across different model scales.

  </details>



- **Trust, Don't Trust, or Flip: Robust Preference-Based Reinforcement Learning with Multi-Expert Feedback**  
  Seyed Amir Hosseini, Maryam Abdolali, Amirhosein Tavakkoli, Fardin Ayar, Ehsan Javanmardi, Manabu Tsukada, Mahdi Javanmardi  
  _2026-01-26_ · https://arxiv.org/abs/2601.18751v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Preference-based reinforcement learning (PBRL) offers a promising alternative to explicit reward engineering by learning from pairwise trajectory comparisons. However, real-world preference data often comes from heterogeneous annotators with varying reliability; some accurate, some noisy, and some systematically adversarial. Existing PBRL methods either treat all feedback equally or attempt to filter out unreliable sources, but both approaches fail when faced with adversarial annotators who systematically provide incorrect preferences. We introduce TriTrust-PBRL (TTP), a unified framework that jointly learns a shared reward model and expert-specific trust parameters from multi-expert preference feedback. The key insight is that trust parameters naturally evolve during gradient-based optimization to be positive (trust), near zero (ignore), or negative (flip), enabling the model to automatically invert adversarial preferences and recover useful signal rather than merely discarding corrupted feedback. We provide theoretical analysis establishing identifiability guarantees and detailed gradient analysis that explains how expert separation emerges naturally during training without explicit supervision. Empirically, we evaluate TTP on four diverse domains spanning manipulation tasks (MetaWorld) and locomotion (DM Control) under various corruption scenarios. TTP achieves state-of-the-art robustness, maintaining near-oracle performance under adversarial corruption while standard PBRL methods fail catastrophically. Notably, TTP outperforms existing baselines by successfully learning from mixed expert pools containing both reliable and adversarial annotators, all while requiring no expert features beyond identification indices and integrating seamlessly with existing PBRL pipelines.

  </details>



- **ART for Diffusion Sampling: A Reinforcement Learning Approach to Timestep Schedule**  
  Yilie Huang, Wenpin Tang, Xunyu Zhou  
  _2026-01-26_ · https://arxiv.org/abs/2601.18681v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We consider time discretization for score-based diffusion models to generate samples from a learned reverse-time dynamic on a finite grid. Uniform and hand-crafted grids can be suboptimal given a budget on the number of time steps. We introduce Adaptive Reparameterized Time (ART) that controls the clock speed of a reparameterized time variable, leading to a time change and uneven timesteps along the sampling trajectory while preserving the terminal time. The objective is to minimize the aggregate error arising from the discretized Euler scheme. We derive a randomized control companion, ART-RL, and formulate time change as a continuous-time reinforcement learning (RL) problem with Gaussian policies. We then prove that solving ART-RL recovers the optimal ART schedule, which in turn enables practical actor--critic updates to learn the latter in a data-driven way. Empirically, based on the official EDM pipeline, ART-RL improves Fréchet Inception Distance on CIFAR-10 over a wide range of budgets and transfers to AFHQv2, FFHQ, and ImageNet without the need of retraining.

  </details>



- **FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory**  
  Lei Wei, Xu Dong, Xiao Peng, Niantao Xie, Bin Wang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18642v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large language models deployed as autonomous agents face critical memory limitations, lacking selective forgetting mechanisms that lead to either catastrophic forgetting at context boundaries or information overload within them. While human memory naturally balances retention and forgetting through adaptive decay processes, current AI systems employ binary retention strategies that preserve everything or lose it entirely. We propose FadeMem, a biologically-inspired agent memory architecture that incorporates active forgetting mechanisms mirroring human cognitive efficiency. FadeMem implements differential decay rates across a dual-layer memory hierarchy, where retention is governed by adaptive exponential decay functions modulated by semantic relevance, access frequency, and temporal patterns. Through LLM-guided conflict resolution and intelligent memory fusion, our system consolidates related information while allowing irrelevant details to fade. Experiments on Multi-Session Chat, LoCoMo, and LTI-Bench demonstrate superior multi-hop reasoning and retrieval with 45\% storage reduction, validating the effectiveness of biologically-inspired forgetting in agent memory systems.

  </details>



- **CASSANDRA: Programmatic and Probabilistic Learning and Inference for Stochastic World Modeling**  
  Panagiotis Lymperopoulos, Abhiramon Rajasekharan, Ian Berlot-Attwell, Stéphane Aroca-Ouellette, Kaheer Suleman  
  _2026-01-26_ · https://arxiv.org/abs/2601.18620v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Building world models is essential for planning in real-world domains such as businesses. Since such domains have rich semantics, we can leverage world knowledge to effectively model complex action effects and causal relationships from limited data. In this work, we propose CASSANDRA, a neurosymbolic world modeling approach that leverages an LLM as a knowledge prior to construct lightweight transition models for planning. CASSANDRA integrates two components: (1) LLM-synthesized code to model deterministic features, and (2) LLM-guided structure learning of a probabilistic graphical model to capture causal relationships among stochastic variables. We evaluate CASSANDRA in (i) a small-scale coffee-shop simulator and (ii) a complex theme park business simulator, where we demonstrate significant improvements in transition prediction and planning over baselines.

  </details>



- **A Balanced Neuro-Symbolic Approach for Commonsense Abductive Logic**  
  Joseph Cotnareanu, Didier Chetelat, Yingxue Zhang, Mark Coates  
  _2026-01-26_ · https://arxiv.org/abs/2601.18595v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Although Large Language Models (LLMs) have demonstrated impressive formal reasoning abilities, they often break down when problems require complex proof planning. One promising approach for improving LLM reasoning abilities involves translating problems into formal logic and using a logic solver. Although off-the-shelf logic solvers are in principle substantially more efficient than LLMs at logical reasoning, they assume that all relevant facts are provided in a question and are unable to deal with missing commonsense relations. In this work, we propose a novel method that uses feedback from the logic solver to augment a logic problem with commonsense relations provided by the LLM, in an iterative manner. This involves a search procedure through potential commonsense assumptions to maximize the chance of finding useful facts while keeping cost tractable. On a collection of pure-logical reasoning datasets, from which some commonsense information has been removed, our method consistently achieves considerable improvements over existing techniques, demonstrating the value in balancing neural and symbolic elements when working in human contexts.

  </details>



- **K-Myriad: Jump-starting reinforcement learning with unsupervised parallel agents**  
  Vincenzo De Paola, Mirco Mutti, Riccardo Zamboni, Marcello Restelli  
  _2026-01-26_ · https://arxiv.org/abs/2601.18580v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Parallelization in Reinforcement Learning is typically employed to speed up the training of a single policy, where multiple workers collect experience from an identical sampling distribution. This common design limits the potential of parallelization by neglecting the advantages of diverse exploration strategies. We propose K-Myriad, a scalable and unsupervised method that maximizes the collective state entropy induced by a population of parallel policies. By cultivating a portfolio of specialized exploration strategies, K-Myriad provides a robust initialization for Reinforcement Learning, leading to both higher training efficiency and the discovery of heterogeneous solutions. Experiments on high-dimensional continuous control tasks, with large-scale parallelization, demonstrate that K-Myriad can learn a broad set of distinct policies, highlighting its effectiveness for collective exploration and paving the way towards novel parallelization strategies.

  </details>



- **GenAgent: Scaling Text-to-Image Generation via Agentic Multimodal Reasoning**  
  Kaixun Jiang, Yuzheng Wang, Junjie Zhou, Pandeng Li, Zhihang Liu, Chen-Wei Xie, Zhaoyu Chen, Yun Zheng, Wenqiang Zhang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18543v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We introduce GenAgent, unifying visual understanding and generation through an agentic multimodal model. Unlike unified models that face expensive training costs and understanding-generation trade-offs, GenAgent decouples these capabilities through an agentic framework: understanding is handled by the multimodal model itself, while generation is achieved by treating image generation models as invokable tools. Crucially, unlike existing modular systems constrained by static pipelines, this design enables autonomous multi-turn interactions where the agent generates multimodal chains-of-thought encompassing reasoning, tool invocation, judgment, and reflection to iteratively refine outputs. We employ a two-stage training strategy: first, cold-start with supervised fine-tuning on high-quality tool invocation and reflection data to bootstrap agent behaviors; second, end-to-end agentic reinforcement learning combining pointwise rewards (final image quality) and pairwise rewards (reflection accuracy), with trajectory resampling for enhanced multi-turn exploration. GenAgent significantly boosts base generator(FLUX.1-dev) performance on GenEval++ (+23.6\%) and WISE (+14\%). Beyond performance gains, our framework demonstrates three key properties: 1) cross-tool generalization to generators with varying capabilities, 2) test-time scaling with consistent improvements across interaction rounds, and 3) task-adaptive reasoning that automatically adjusts to different tasks. Our code will be available at \href{https://github.com/deep-kaixun/GenAgent}{this url}.

  </details>



- **DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation**  
  Zijun Li, Shijie Li, Zhenxi Zhang, Bin Li, Shoujun Zhou  
  _2026-01-26_ · https://arxiv.org/abs/2601.18492v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-and-Language Navigation (VLN) requires an embodied agent to navigate in a complex 3D environment according to natural language instructions. Recent progress in large language models (LLMs) has enabled language-driven navigation with improved interpretability. However, most LLM-based agents still rely on single-shot action decisions, where the model must choose one option from noisy, textualized multi-perspective observations. Due to local mismatches and imperfect intermediate reasoning, such decisions can easily deviate from the correct path, leading to error accumulation and reduced reliability in unseen environments. In this paper, we propose DV-VLN, a new VLN framework that follows a generate-then-verify paradigm. DV-VLN first performs parameter-efficient in-domain adaptation of an open-source LLaMA-2 backbone to produce a structured navigational chain-of-thought, and then verifies candidate actions with two complementary channels: True-False Verification (TFV) and Masked-Entity Verification (MEV). DV-VLN selects actions by aggregating verification successes across multiple samples, yielding interpretable scores for reranking. Experiments on R2R, RxR (English subset), and REVERIE show that DV-VLN consistently improves over direct prediction and sampling-only baselines, achieving competitive performance among language-only VLN agents and promising results compared with several cross-modal systems.Code is available at https://github.com/PlumJun/DV-VLN.

  </details>



- **Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States**  
  Kyoleen Kwak, Hyoseok Hwang  
  _2026-01-26_ · https://arxiv.org/abs/2601.18479v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Deep reinforcement learning has proven to be a powerful approach to solving control tasks, but its characteristic high-frequency oscillations make it difficult to apply in real-world environments. While prior methods have addressed action oscillations via architectural or loss-based methods, the latter typically depend on heuristic or synthetic definitions of state similarity to promote action consistency, which often fail to accurately reflect the underlying system dynamics. In this paper, we propose a novel loss-based method by introducing a transition-induced similar state. The transition-induced similar state is defined as the distribution of next states transitioned from the previous state. Since it utilizes only environmental feedback and actually collected data, it better captures system dynamics. Building upon this foundation, we introduce Action Smoothing by Aligning Actions with Predictions from Preceding States (ASAP), an action smoothing method that effectively mitigates action oscillations. ASAP enforces action smoothness by aligning the actions with those taken in transition-induced similar states and by penalizing second-order differences to suppress high-frequency oscillations. Experiments in Gymnasium and Isaac-Lab environments demonstrate that ASAP yields smoother control and improved policy performance over existing methods.

  </details>



- **Deep Reinforcement Learning for Hybrid RIS Assisted MIMO Communications**  
  Phuong Nam Tran, Nhan Thanh Nguyen, Markku Juntti  
  _2026-01-26_ · https://arxiv.org/abs/2601.18453v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Hybrid reconfigurable intelligent surfaces (HRIS) enhance wireless systems by combining passive reflection with active signal amplification. However, jointly optimizing the transmit beamforming with the HRIS reflection and amplification coefficients to maximize spectral efficiency (SE) is a non-convex problem, and conventional iterative solutions are computationally intensive. To address this, we propose a deep reinforcement learning (DRL) framework that learns a direct mapping from channel state information to the near-optimal transmit beamforming and HRIS configurations. The DRL model is trained offline, after which it can compute the beamforming and HRIS configurations with low complexity and latency. Simulation results demonstrate that our DRL-based method achieves 95% of the SE obtained by the alternating optimization benchmark, while significantly lowering the computational complexity.

  </details>



- **SG-CADVLM: A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation**  
  Hongyi Zhao, Shuo Wang, Qijie He, Ziyuan Pu  
  _2026-01-26_ · https://arxiv.org/abs/2601.18442v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous vehicle safety validation requires testing on safety-critical scenarios, but these events are rare in real-world driving and costly to test due to collision risks. Crash reports provide authentic specifications of safety-critical events, offering a vital alternative to scarce real-world collision trajectory data. This makes them valuable sources for generating realistic high-risk scenarios through simulation. Existing approaches face significant limitations because data-driven methods lack diversity due to their reliance on existing latent distributions, whereas adversarial methods often produce unrealistic scenarios lacking physical fidelity. Large Language Model (LLM) and Vision Language Model (VLM)-based methods show significant promise. However, they suffer from context suppression issues where internal parametric knowledge overrides crash specifications, producing scenarios that deviate from actual accident characteristics. This paper presents SG-CADVLM (A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation), a framework that integrates Context-Aware Decoding with multi-modal input processing to generate safety-critical scenarios from crash reports and road network diagrams. The framework mitigates VLM hallucination issues while enabling the simultaneous generation of road geometry and vehicle trajectories. The experimental results demonstrate that SG-CADVLM generates critical risk scenarios at a rate of 84.4% compared to 12.5% for the baseline methods, representing an improvement of 469%, while producing executable simulations for autonomous vehicle testing.

  </details>



- **Integrating Fine-Grained Audio-Visual Evidence for Robust Multimodal Emotion Reasoning**  
  Zhixian Zhao, Wenjie Tian, Xiaohai Tian, Jun Zhang, Lei Xie  
  _2026-01-26_ · https://arxiv.org/abs/2601.18321v1 · `cs.MM`  
  <details><summary>Abstract</summary>

  Multimodal emotion analysis is shifting from static classification to generative reasoning. Beyond simple label prediction, robust affective reasoning must synthesize fine-grained signals such as facial micro-expressions and prosodic which shifts to decode the latent causality within complex social contexts. However, current Multimodal Large Language Models (MLLMs) face significant limitations in fine-grained perception, primarily due to data scarcity and insufficient cross-modal fusion. As a result, these models often exhibit unimodal dominance which leads to hallucinations in complex multimodal interactions, particularly when visual and acoustic cues are subtle, ambiguous, or even contradictory (e.g., in sarcastic scenery). To address this, we introduce SABER-LLM, a framework designed for robust multimodal reasoning. First, we construct SABER, a large-scale emotion reasoning dataset comprising 600K video clips, annotated with a novel six-dimensional schema that jointly captures audiovisual cues and causal logic. Second, we propose the structured evidence decomposition paradigm, which enforces a "perceive-then-reason" separation between evidence extraction and reasoning to alleviate unimodal dominance. The ability to perceive complex scenes is further reinforced by consistency-aware direct preference optimization, which explicitly encourages alignment among modalities under ambiguous or conflicting perceptual conditions. Experiments on EMER, EmoBench-M, and SABER-Test demonstrate that SABER-LLM significantly outperforms open-source baselines and achieves robustness competitive with closed-source models in decoding complex emotional dynamics. The dataset and model are available at https://github.com/zxzhao0/SABER-LLM.

  </details>


