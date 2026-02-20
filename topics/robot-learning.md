# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-20 07:10 UTC_

Total papers shown: **13**


---

- **When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs**  
  Yu Fang, Yuchun Feng, Dong Jing, Jiaqi Liu, Yue Yang, Zhenyu Wei, Daniel Szafir, Mingyu Ding  
  _2026-02-19_ · https://arxiv.org/abs/2602.17659v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language-Action models (VLAs) promise to ground language instructions in robot control, yet in practice often fail to faithfully follow language. When presented with instructions that lack strong scene-specific supervision, VLAs suffer from counterfactual failures: they act based on vision shortcuts induced by dataset biases, repeatedly executing well-learned behaviors and selecting objects frequently seen during training regardless of language intent. To systematically study it, we introduce LIBERO-CF, the first counterfactual benchmark for VLAs that evaluates language following capability by assigning alternative instructions under visually plausible LIBERO layouts. Our evaluation reveals that counterfactual failures are prevalent yet underexplored across state-of-the-art VLAs. We propose Counterfactual Action Guidance (CAG), a simple yet effective dual-branch inference scheme that explicitly regularizes language conditioning in VLAs. CAG combines a standard VLA policy with a language-unconditioned Vision-Action (VA) module, enabling counterfactual comparison during action selection. This design reduces reliance on visual shortcuts, improves robustness on under-observed tasks, and requires neither additional demonstrations nor modifications to existing architectures or pretrained models. Extensive experiments demonstrate its plug-and-play integration across diverse VLAs and consistent improvements. For example, on LIBERO-CF, CAG improves $π_{0.5}$ by 9.7% in language following accuracy and 3.6% in task success on under-observed tasks using a training-free strategy, with further gains of 15.5% and 8.5%, respectively, when paired with a VA model. In real-world evaluations, CAG reduces counterfactual failures of 9.4% and improves task success by 17.2% on average.

  </details>



- **IRIS: Learning-Driven Task-Specific Cinema Robot Arm for Visuomotor Motion Control**  
  Qilong Cheng, Matthew Mackay, Ali Bereyhi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17537v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic camera systems enable dynamic, repeatable motion beyond human capabilities, yet their adoption remains limited by the high cost and operational complexity of industrial-grade platforms. We present the Intelligent Robotic Imaging System (IRIS), a task-specific 6-DOF manipulator designed for autonomous, learning-driven cinematic motion control. IRIS integrates a lightweight, fully 3D-printed hardware design with a goal-conditioned visuomotor imitation learning framework based on Action Chunking with Transformers (ACT). The system learns object-aware and perceptually smooth camera trajectories directly from human demonstrations, eliminating the need for explicit geometric programming. The complete platform costs under $1,000 USD, supports a 1.5 kg payload, and achieves approximately 1 mm repeatability. Real-world experiments demonstrate accurate trajectory tracking, reliable autonomous execution, and generalization across diverse cinematic motions.

  </details>



- **Agentic Wireless Communication for 6G: Intent-Aware and Continuously Evolving Physical-Layer Intelligence**  
  Zhaoyang Li, Xingzhi Jin, Junyu Pan, Qianqian Yang, Zhiguo Shi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17096v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  As 6G wireless systems evolve, growing functional complexity and diverse service demands are driving a shift from rule-based control to intent-driven autonomous intelligence. User requirements are no longer captured by a single metric (e.g., throughput or reliability), but by multi-dimensional objectives such as latency sensitivity, energy preference, computational constraints, and service-level requirements. These objectives may also change over time due to environmental dynamics and user-network interactions. Therefore, accurate understanding of both the communication environment and user intent is critical for autonomous and sustainably evolving 6G communications. Large language models (LLMs), with strong contextual understanding and cross-modal reasoning, provide a promising foundation for intent-aware network agents. Compared with rule-driven or centrally optimized designs, LLM-based agents can integrate heterogeneous information and translate natural-language intents into executable control and configuration decisions. Focusing on a closed-loop pipeline of intent perception, autonomous decision making, and network execution, this paper investigates agentic AI for the 6G physical layer and its realization pathways. We review representative physical-layer tasks and their limitations in supporting intent awareness and autonomy, identify application scenarios where agentic AI is advantageous, and discuss key challenges and enabling technologies in multimodal perception, cross-layer decision making, and sustainable optimization. Finally, we present a case study of an intent-driven link decision agent, termed AgenCom, which adaptively constructs communication links under diverse user preferences and channel conditions.

  </details>



- **Voice-Driven Semantic Perception for UAV-Assisted Emergency Networks**  
  Nuno Saavedra, Pedro Ribeiro, André Coelho, Rui Campos  
  _2026-02-19_ · https://arxiv.org/abs/2602.17394v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  Unmanned Aerial Vehicle (UAV)-assisted networks are increasingly foreseen as a promising approach for emergency response, providing rapid, flexible, and resilient communications in environments where terrestrial infrastructure is degraded or unavailable. In such scenarios, voice radio communications remain essential for first responders due to their robustness; however, their unstructured nature prevents direct integration with automated UAV-assisted network management. This paper proposes SIREN, an AI-driven framework that enables voice-driven perception for UAV-assisted networks. By integrating Automatic Speech Recognition (ASR) with Large Language Model (LLM)-based semantic extraction and Natural Language Processing (NLP) validation, SIREN converts emergency voice traffic into structured, machine-readable information, including responding units, location references, emergency severity, and Quality-of-Service (QoS) requirements. SIREN is evaluated using synthetic emergency scenarios with controlled variations in language, speaker count, background noise, and message complexity. The results demonstrate robust transcription and reliable semantic extraction across diverse operating conditions, while highlighting speaker diarization and geographic ambiguity as the main limiting factors. These findings establish the feasibility of voice-driven situational awareness for UAV-assisted networks and show a practical foundation for human-in-the-loop decision support and adaptive network management in emergency response operations.

  </details>



- **What Breaks Embodied AI Security:LLM Vulnerabilities, CPS Flaws,or Something Else?**  
  Boyang Ma, Hechuan Guo, Peizhuo Lv, Minghui Xu, Xuelong Dai, YeChao Zhang, Yijun Yang, Yue Zhang  
  _2026-02-19_ · https://arxiv.org/abs/2602.17345v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Embodied AI systems (e.g., autonomous vehicles, service robots, and LLM-driven interactive agents) are rapidly transitioning from controlled environments to safety critical real-world deployments. Unlike disembodied AI, failures in embodied intelligence lead to irreversible physical consequences, raising fundamental questions about security, safety, and reliability. While existing research predominantly analyzes embodied AI through the lenses of Large Language Model (LLM) vulnerabilities or classical Cyber-Physical System (CPS) failures, this survey argues that these perspectives are individually insufficient to explain many observed breakdowns in modern embodied systems. We posit that a significant class of failures arises from embodiment-induced system-level mismatches, rather than from isolated model flaws or traditional CPS attacks. Specifically, we identify four core insights that explain why embodied AI is fundamentally harder to secure: (i) semantic correctness does not imply physical safety, as language-level reasoning abstracts away geometry, dynamics, and contact constraints; (ii) identical actions can lead to drastically different outcomes across physical states due to nonlinear dynamics and state uncertainty; (iii) small errors propagate and amplify across tightly coupled perception-decision-action loops; and (iv) safety is not compositional across time or system layers, enabling locally safe decisions to accumulate into globally unsafe behavior. These insights suggest that securing embodied AI requires moving beyond component-level defenses toward system-level reasoning about physical risk, uncertainty, and failure propagation.

  </details>



- **Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs**  
  Luke Huang, Zhuoyang Zhang, Qinghao Hu, Shang Yang, Song Han  
  _2026-02-19_ · https://arxiv.org/abs/2602.17616v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) is widely used to improve large language models on reasoning tasks, and asynchronous RL training is attractive because it increases end-to-end throughput. However, for widely adopted critic-free policy-gradient methods such as REINFORCE and GRPO, high asynchrony makes the policy-gradient estimator markedly $\textbf{higher variance}$: training on stale rollouts creates heavy-tailed importance ratios, causing a small fraction of samples to dominate updates. This amplification makes gradients noisy and learning unstable relative to matched on-policy training. Across math and general reasoning benchmarks, we find collapse is reliably predicted by effective sample size (ESS) and unstable gradient norms. Motivated by this diagnosis, we propose $\textbf{V}$ariance $\textbf{C}$ontrolled $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{VCPO}$), a general stabilization method for REINFORCE/GRPO-style algorithms that (i) scales learning rate based on effective sample size to dampen unreliable updates, and (ii) applies a closed-form minimum-variance baseline for the off-policy setting, avoiding an auxiliary value model and adding minimal overhead. Empirically, VCPO substantially improves robustness for asynchronous training across math, general reasoning, and tool-use tasks, outperforming a broad suite of baselines spanning masking/clipping stabilizers and algorithmic variants. This reduces long-context, multi-turn training time by 2.5$\times$ while matching synchronous performance, demonstrating that explicit control of policy-gradient variance is key for reliable asynchronous RL at scale.

  </details>



- **AutoNumerics: An Autonomous, PDE-Agnostic Multi-Agent Pipeline for Scientific Computing**  
  Jianda Du, Youran Sun, Haizhao Yang  
  _2026-02-19_ · https://arxiv.org/abs/2602.17607v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  PDEs are central to scientific and engineering modeling, yet designing accurate numerical solvers typically requires substantial mathematical expertise and manual tuning. Recent neural network-based approaches improve flexibility but often demand high computational cost and suffer from limited interpretability. We introduce \texttt{AutoNumerics}, a multi-agent framework that autonomously designs, implements, debugs, and verifies numerical solvers for general PDEs directly from natural language descriptions. Unlike black-box neural solvers, our framework generates transparent solvers grounded in classical numerical analysis. We introduce a coarse-to-fine execution strategy and a residual-based self-verification mechanism. Experiments on 24 canonical and real-world PDE problems demonstrate that \texttt{AutoNumerics} achieves competitive or superior accuracy compared to existing neural and LLM-based baselines, and correctly selects numerical schemes based on PDE structural properties, suggesting its viability as an accessible paradigm for automated PDE solving.

  </details>



- **ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment**  
  Hongjue Zhao, Haosen Sun, Jiangtao Kong, Xiaochang Li, Qineng Wang, Liwei Jiang, Qi Zhu, Tarek Abdelzaher, Yejin Choi, Manling Li, et al.  
  _2026-02-19_ · https://arxiv.org/abs/2602.17560v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Activation steering, or representation engineering, offers a lightweight approach to align large language models (LLMs) by manipulating their internal activations at inference time. However, current methods suffer from two key limitations: \textit{(i)} the lack of a unified theoretical framework for guiding the design of steering directions, and \textit{(ii)} an over-reliance on \textit{one-step steering} that fail to capture complex patterns of activation distributions. In this work, we propose a unified ordinary differential equations (ODEs)-based \textit{theoretical} framework for activation steering in LLM alignment. We show that conventional activation addition can be interpreted as a first-order approximation to the solution of an ODE. Based on this ODE perspective, identifying a steering direction becomes equivalent to designing a \textit{barrier function} from control theory. Derived from this framework, we introduce ODESteer, a kind of ODE-based steering guided by barrier functions, which shows \textit{empirical} advancement in LLM alignment. ODESteer identifies steering directions by defining the barrier function as the log-density ratio between positive and negative activations, and employs it to construct an ODE for \textit{multi-step and adaptive} steering. Compared to state-of-the-art activation steering methods, ODESteer achieves consistent empirical improvements on diverse LLM alignment benchmarks, a notable $5.7\%$ improvement over TruthfulQA, $2.5\%$ over UltraFeedback, and $2.4\%$ over RealToxicityPrompts. Our work establishes a principled new view of activation steering in LLM alignment by unifying its theoretical foundations via ODEs, and validating it empirically through the proposed ODESteer method.

  </details>



- **RetouchIQ: MLLM Agents for Instruction-Based Image Retouching with Generalist Reward**  
  Qiucheng Wu, Jing Shi, Simon Jenni, Kushal Kafle, Tianyu Wang, Shiyu Chang, Handong Zhao  
  _2026-02-19_ · https://arxiv.org/abs/2602.17558v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advances in multimodal large language models (MLLMs) have shown great potential for extending vision-language reasoning to professional tool-based image editing, enabling intuitive and creative editing. A promising direction is to use reinforcement learning (RL) to enable MLLMs to reason about and execute optimal tool-use plans within professional image-editing software. However, training remains challenging due to the lack of reliable, verifiable reward signals that can reflect the inherently subjective nature of creative editing. In this work, we introduce RetouchIQ, a framework that performs instruction-based executable image editing through MLLM agents guided by a generalist reward model. RetouchIQ interprets user-specified editing intentions and generates corresponding, executable image adjustments, bridging high-level aesthetic goals with precise parameter control. To move beyond conventional, rule-based rewards that compute similarity against a fixed reference image using handcrafted metrics, we propose a generalist reward model, an RL fine-tuned MLLM that evaluates edited results through a set of generated metrics on a case-by-case basis. Then, the reward model provides scalar feedback through multimodal reasoning, enabling reinforcement learning with high-quality, instruction-consistent gradients. We curate an extended dataset with 190k instruction-reasoning pairs and establish a new benchmark for instruction-based image editing. Experiments show that RetouchIQ substantially improves both semantic consistency and perceptual quality over previous MLLM-based and diffusion-based editing systems. Our findings demonstrate the potential of generalist reward-driven MLLM agents as flexible, explainable, and executable assistants for professional image editing.

  </details>



- **Systematic Evaluation of Single-Cell Foundation Model Interpretability Reveals Attention Captures Co-Expression Rather Than Unique Regulatory Signal**  
  Ihor Kendiukhov  
  _2026-02-19_ · https://arxiv.org/abs/2602.17532v1 · `q-bio.GN`  
  <details><summary>Abstract</summary>

  We present a systematic evaluation framework - thirty-seven analyses, 153 statistical tests, four cell types, two perturbation modalities - for assessing mechanistic interpretability in single-cell foundation models. Applying this framework to scGPT and Geneformer, we find that attention patterns encode structured biological information with layer-specific organisation - protein-protein interactions in early layers, transcriptional regulation in late layers - but this structure provides no incremental value for perturbation prediction: trivial gene-level baselines outperform both attention and correlation edges (AUROC 0.81-0.88 versus 0.70), pairwise edge scores add zero predictive contribution, and causal ablation of regulatory heads produces no degradation. These findings generalise from K562 to RPE1 cells; the attention-correlation relationship is context-dependent, but gene-level dominance is universal. Cell-State Stratified Interpretability (CSSI) addresses an attention-specific scaling failure, improving GRN recovery up to 1.85x. The framework establishes reusable quality-control standards for the field.

  </details>



- **What Do LLMs Associate with Your Name? A Human-Centered Black-Box Audit of Personal Data**  
  Dimitri Staufer, Kirsten Morehouse  
  _2026-02-19_ · https://arxiv.org/abs/2602.17483v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Large language models (LLMs), and conversational agents based on them, are exposed to personal data (PD) during pre-training and during user interactions. Prior work shows that PD can resurface, yet users lack insight into how strongly models associate specific information to their identity. We audit PD across eight LLMs (3 open-source; 5 API-based, including GPT-4o), introduce LMP2 (Language Model Privacy Probe), a human-centered, privacy-preserving audit tool refined through two formative studies (N=20), and run two studies with EU residents to capture (i) intuitions about LLM-generated PD (N1=155) and (ii) reactions to tool output (N2=303). We show empirically that models confidently generate multiple PD categories for well-known individuals. For everyday users, GPT-4o generates 11 features with 60% or more accuracy (e.g., gender, hair color, languages). Finally, 72% of participants sought control over model-generated associations with their name, raising questions about what counts as PD and whether data privacy rights should extend to LLMs.

  </details>



- **Continual uncertainty learning**  
  Heisei Yonezawa, Ansei Yonezawa, Itsuro Kajiwara  
  _2026-02-19_ · https://arxiv.org/abs/2602.17174v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Robust control of mechanical systems with multiple uncertainties remains a fundamental challenge, particularly when nonlinear dynamics and operating-condition variations are intricately intertwined. While deep reinforcement learning (DRL) combined with domain randomization has shown promise in mitigating the sim-to-real gap, simultaneously handling all sources of uncertainty often leads to sub-optimal policies and poor learning efficiency. This study formulates a new curriculum-based continual learning framework for robust control problems involving nonlinear dynamical systems in which multiple sources of uncertainty are simultaneously superimposed. The key idea is to decompose a complex control problem with multiple uncertainties into a sequence of continual learning tasks, in which strategies for handling each uncertainty are acquired sequentially. The original system is extended into a finite set of plants whose dynamic uncertainties are gradually expanded and diversified as learning progresses. The policy is stably updated across the entire plant sets associated with tasks defined by different uncertainty configurations without catastrophic forgetting. To ensure learning efficiency, we jointly incorporate a model-based controller (MBC), which guarantees a shared baseline performance across the plant sets, into the learning process to accelerate the convergence. This residual learning scheme facilitates task-specific optimization of the DRL agent for each uncertainty, thereby enhancing sample efficiency. As a practical industrial application, this study applies the proposed method to designing an active vibration controller for automotive powertrains. We verified that the resulting controller is robust against structural nonlinearities and dynamic variations, realizing successful sim-to-real transfer.

  </details>



- **Dynamic Decision-Making under Model Misspecification: A Stochastic Stability Approach**  
  Xinyu Dai, Daniel Chen, Yian Qian  
  _2026-02-19_ · https://arxiv.org/abs/2602.17086v1 · `econ.TH`  
  <details><summary>Abstract</summary>

  Dynamic decision-making under model uncertainty is central to many economic environments, yet existing bandit and reinforcement learning algorithms rely on the assumption of correct model specification. This paper studies the behavior and performance of one of the most commonly used Bayesian reinforcement learning algorithms, Thompson Sampling (TS), when the model class is misspecified. We first provide a complete dynamic classification of posterior evolution in a misspecified two-armed Gaussian bandit, identifying distinct regimes: correct model concentration, incorrect model concentration, and persistent belief mixing, characterized by the direction of statistical evidence and the model-action mapping. These regimes yield sharp predictions for limiting beliefs, action frequencies, and asymptotic regret. We then extend the analysis to a general finite model class and develop a unified stochastic stability framework that represents posterior evolution as a Markov process on the belief simplex. This approach characterizes two sufficient conditions to classify the ergodic and transient behaviors and provides inductive dimensional reductions of the posterior dynamics. Our results offer the first qualitative and geometric classification of TS under misspecification, bridging Bayesian learning with evolutionary dynamics, and also build the foundations of robust decision-making in structured bandits.

  </details>


