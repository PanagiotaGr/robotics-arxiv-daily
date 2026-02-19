# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-19 07:13 UTC_

Total papers shown: **11**


---

- **Learning Humanoid End-Effector Control for Open-Vocabulary Visual Loco-Manipulation**  
  Runpei Dong, Ziyan Li, Xialin He, Saurabh Gupta  
  _2026-02-18_ · https://arxiv.org/abs/2602.16705v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual loco-manipulation of arbitrary objects in the wild with humanoid robots requires accurate end-effector (EE) control and a generalizable understanding of the scene via visual inputs (e.g., RGB-D images). Existing approaches are based on real-world imitation learning and exhibit limited generalization due to the difficulty in collecting large-scale training datasets. This paper presents a new paradigm, HERO, for object loco-manipulation with humanoid robots that combines the strong generalization and open-vocabulary understanding of large vision models with strong control performance from simulated training. We achieve this by designing an accurate residual-aware EE tracking policy. This EE tracking policy combines classical robotics with machine learning. It uses a) inverse kinematics to convert residual end-effector targets into reference trajectories, b) a learned neural forward model for accurate forward kinematics, c) goal adjustment, and d) replanning. Together, these innovations help us cut down the end-effector tracking error by 3.2x. We use this accurate end-effector tracker to build a modular system for loco-manipulation, where we use open-vocabulary large vision models for strong visual generalization. Our system is able to operate in diverse real-world environments, from offices to coffee shops, where the robot is able to reliably manipulate various everyday objects (e.g., mugs, apples, toys) on surfaces ranging from 43cm to 92cm in height. Systematic modular and end-to-end tests in simulation and the real world demonstrate the effectiveness of our proposed design. We believe the advances in this paper can open up new ways of training humanoid robots to interact with daily objects.

  </details>



- **HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents**  
  Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, Charles Fleming, Zhaoran Wang, Alfredo Garcia, Mingyi Hong  
  _2026-02-18_ · https://arxiv.org/abs/2602.16165v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Training LLMs as interactive agents for multi-turn decision-making remains challenging, particularly in long-horizon tasks with sparse and delayed rewards, where agents must execute extended sequences of actions before receiving meaningful feedback. Most existing reinforcement learning (RL) approaches model LLM agents as flat policies operating at a single time scale, selecting one action at each turn. In sparse-reward settings, such flat policies must propagate credit across the entire trajectory without explicit temporal abstraction, which often leads to unstable optimization and inefficient credit assignment. We propose HiPER, a novel Hierarchical Plan-Execute RL framework that explicitly separates high-level planning from low-level execution. HiPER factorizes the policy into a high-level planner that proposes subgoals and a low-level executor that carries them out over multiple action steps. To align optimization with this structure, we introduce a key technique called hierarchical advantage estimation (HAE), which carefully assigns credit at both the planning and execution levels. By aggregating returns over the execution of each subgoal and coordinating updates across the two levels, HAE provides an unbiased gradient estimator and provably reduces variance compared to flat generalized advantage estimation. Empirically, HiPER achieves state-of-the-art performance on challenging interactive benchmarks, reaching 97.4\% success on ALFWorld and 83.3\% on WebShop with Qwen2.5-7B-Instruct (+6.6\% and +8.3\% over the best prior method), with especially large gains on long-horizon tasks requiring multiple dependent subtasks. These results highlight the importance of explicit hierarchical decomposition for scalable RL training of multi-turn LLM agents.

  </details>



- **SPARC: Scenario Planning and Reasoning for Automated C Unit Test Generation**  
  Jaid Monwar Chowdhury, Chi-An Fu, Reyhaneh Jabbarvand  
  _2026-02-18_ · https://arxiv.org/abs/2602.16671v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Automated unit test generation for C remains a formidable challenge due to the semantic gap between high-level program intent and the rigid syntactic constraints of pointer arithmetic and manual memory management. While Large Language Models (LLMs) exhibit strong generative capabilities, direct intent-to-code synthesis frequently suffers from the leap-to-code failure mode, where models prematurely emit code without grounding in program structure, constraints, and semantics. This will result in non-compilable tests, hallucinated function signatures, low branch coverage, and semantically irrelevant assertions that cannot properly capture bugs. We introduce SPARC, a neuro-symbolic, scenario-based framework that bridges this gap through four stages: (1) Control Flow Graph (CFG) analysis, (2) an Operation Map that grounds LLM reasoning in validated utility helpers, (3) Path-targeted test synthesis, and (4) an iterative, self-correction validation loop using compiler and runtime feedback. We evaluate SPARC on 59 real-world and algorithmic subjects, where it outperforms the vanilla prompt generation baseline by 31.36% in line coverage, 26.01% in branch coverage, and 20.78% in mutation score, matching or exceeding the symbolic execution tool KLEE on complex subjects. SPARC retains 94.3% of tests through iterative repair and produces code with significantly higher developer-rated readability and maintainability. By aligning LLM reasoning with program structure, SPARC provides a scalable path for industrial-grade testing of legacy C codebases.

  </details>



- **Dual-Quadruped Collaborative Transportation in Narrow Environments via Safe Reinforcement Learning**  
  Zhezhi Lei, Zhihai Bi, Wenxin Wang, Jun Ma  
  _2026-02-18_ · https://arxiv.org/abs/2602.16353v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Collaborative transportation, where multiple robots collaboratively transport a payload, has garnered significant attention in recent years. While ensuring safe and high-performance inter-robot collaboration is critical for effective task execution, it is difficult to pursue in narrow environments where the feasible region is extremely limited. To address this challenge, we propose a novel approach for dual-quadruped collaborative transportation via safe reinforcement learning (RL). Specifically, we model the task as a fully cooperative constrained Markov game, where collision avoidance is formulated as constraints. We introduce a cost-advantage decomposition method that enforces the sum of team constraints to remain below an upper bound, thereby guaranteeing task safety within an RL framework. Furthermore, we propose a constraint allocation method that assigns shared constraints to individual robots to maximize the overall task reward, encouraging autonomous task-assignment among robots, thereby improving collaborative task performance. Simulation and real-time experimental results demonstrate that the proposed approach achieves superior performance and a higher success rate in dual-quadruped collaborative transportation compared to existing methods.

  </details>



- **EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data**  
  Ruijie Zheng, Dantong Niu, Yuqi Xie, Jing Wang, Mengda Xu, Yunfan Jiang, Fernando Castañeda, Fengyuan Hu, You Liang Tan, Letian Fu, et al.  
  _2026-02-18_ · https://arxiv.org/abs/2602.16710v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Human behavior is among the most scalable sources of data for learning physical intelligence, yet how to effectively leverage it for dexterous manipulation remains unclear. While prior work demonstrates human to robot transfer in constrained settings, it is unclear whether large scale human data can support fine grained, high degree of freedom dexterous manipulation. We present EgoScale, a human to dexterous manipulation transfer framework built on large scale egocentric human data. We train a Vision Language Action (VLA) model on over 20,854 hours of action labeled egocentric human video, more than 20 times larger than prior efforts, and uncover a log linear scaling law between human data scale and validation loss. This validation loss strongly correlates with downstream real robot performance, establishing large scale human data as a predictable supervision source. Beyond scale, we introduce a simple two stage transfer recipe: large scale human pretraining followed by lightweight aligned human robot mid training. This enables strong long horizon dexterous manipulation and one shot task adaptation with minimal robot supervision. Our final policy improves average success rate by 54% over a no pretraining baseline using a 22 DoF dexterous robotic hand, and transfers effectively to robots with lower DoF hands, indicating that large scale human motion provides a reusable, embodiment agnostic motor prior.

  </details>



- **Retrieval-Augmented Foundation Models for Matched Molecular Pair Transformations to Recapitulate Medicinal Chemistry Intuition**  
  Bo Pan, Peter Zhiping Zhang, Hao-Wei Pang, Alex Zhu, Xiang Yu, Liying Zhang, Liang Zhao  
  _2026-02-18_ · https://arxiv.org/abs/2602.16684v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Matched molecular pairs (MMPs) capture the local chemical edits that medicinal chemists routinely use to design analogs, but existing ML approaches either operate at the whole-molecule level with limited edit controllability or learn MMP-style edits from restricted settings and small models. We propose a variable-to-variable formulation of analog generation and train a foundation model on large-scale MMP transformations (MMPTs) to generate diverse variables conditioned on an input variable. To enable practical control, we develop prompting mechanisms that let the users specify preferred transformation patterns during generation. We further introduce MMPT-RAG, a retrieval-augmented framework that uses external reference analogs as contextual guidance to steer generation and generalize from project-specific series. Experiments on general chemical corpora and patent-specific datasets demonstrate improved diversity, novelty, and controllability, and show that our method recovers realistic analog structures in practical discovery scenarios.

  </details>



- **Learning Situated Awareness in the Real World**  
  Chuhan Li, Ruilin Han, Joy Hsu, Yongyuan Liang, Rajiv Dhawan, Jiajun Wu, Ming-Hsuan Yang, Xin Eric Wang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16682v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  A core aspect of human perception is situated awareness, the ability to relate ourselves to the surrounding physical environment and reason over possible actions in context. However, most existing benchmarks for multimodal foundation models (MFMs) emphasize environment-centric spatial relations (relations among objects in a scene), while largely overlooking observer-centric relationships that require reasoning relative to agent's viewpoint, pose, and motion. To bridge this gap, we introduce SAW-Bench (Situated Awareness in the Real World), a novel benchmark for evaluating egocentric situated awareness using real-world videos. SAW-Bench comprises 786 self-recorded videos captured with Ray-Ban Meta (Gen 2) smart glasses spanning diverse indoor and outdoor environments, and over 2,071 human-annotated question-answer pairs. It probes a model's observer-centric understanding with six different awareness tasks. Our comprehensive evaluation reveals a human-model performance gap of 37.66%, even with the best-performing MFM, Gemini 3 Flash. Beyond this gap, our in-depth analysis uncovers several notable findings; for example, while models can exploit partial geometric cues in egocentric videos, they often fail to infer a coherent camera geometry, leading to systematic spatial reasoning errors. We position SAW-Bench as a benchmark for situated spatial intelligence, moving beyond passive observation to understanding physically grounded, observer-centric dynamics.

  </details>



- **VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection**  
  Yingyuan Yang, Tian Lan, Yifei Gao, Yimeng Lu, Wenjun He, Meng Wang, Chenghao Liu, Chen Zhang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16681v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Time-series anomaly detection (TSAD) requires identifying both immediate Point Anomalies and long-range Context Anomalies. However, existing foundation models face a fundamental trade-off: 1D temporal models provide fine-grained pointwise localization but lack a global contextual perspective, while 2D vision-based models capture global patterns but suffer from information bottlenecks due to a lack of temporal alignment and coarse-grained pointwise detection. To resolve this dilemma, we propose VETime, the first TSAD framework that unifies temporal and visual modalities through fine-grained visual-temporal alignment and dynamic fusion. VETime introduces a Reversible Image Conversion and a Patch-Level Temporal Alignment module to establish a shared visual-temporal timeline, preserving discriminative details while maintaining temporal sensitivity. Furthermore, we design an Anomaly Window Contrastive Learning mechanism and a Task-Adaptive Multi-Modal Fusion to adaptively integrate the complementary perceptual strengths of both modalities. Extensive experiments demonstrate that VETime significantly outperforms state-of-the-art models in zero-shot scenarios, achieving superior localization precision with lower computational overhead than current vision-based approaches. Code available at: https://github.com/yyyangcoder/VETime.

  </details>



- **Learning to unfold cloth: Scaling up world models to deformable object manipulation**  
  Jack Rome, Stephen James, Subramanian Ramamoorthy  
  _2026-02-18_ · https://arxiv.org/abs/2602.16675v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning to manipulate cloth is both a paradigmatic problem for robotic research and a problem of immediate relevance to a variety of applications ranging from assistive care to the service industry. The complex physics of the deformable object makes this problem of cloth manipulation nontrivial. In order to create a general manipulation strategy that addresses a variety of shapes, sizes, fold and wrinkle patterns, in addition to the usual problems of appearance variations, it becomes important to carefully consider model structure and their implications for generalisation performance. In this paper, we present an approach to in-air cloth manipulation that uses a variation of a recently proposed reinforcement learning architecture, DreamerV2. Our implementation modifies this architecture to utilise surface normals input, in addition to modiying the replay buffer and data augmentation procedures. Taken together these modifications represent an enhancement to the world model used by the robot, addressing the physical complexity of the object being manipulated by the robot. We present evaluations both in simulation and in a zero-shot deployment of the trained policies in a physical robot setup, performing in-air unfolding of a variety of different cloth types, demonstrating the generalisation benefits of our proposed architecture.

  </details>



- **FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM Serving**  
  Chia-chi Hsieh, Zan Zong, Xinyang Chen, Jianjiang Li, Jidong Zhai, Lijie Wen  
  _2026-02-18_ · https://arxiv.org/abs/2602.16603v1 · `cs.DC`  
  <details><summary>Abstract</summary>

  The growing demand for large language models (LLMs) requires serving systems to handle many concurrent requests with diverse service level objectives (SLOs). This exacerbates head-of-line (HoL) blocking during the compute-intensive prefill phase, where long-running requests monopolize resources and delay higher-priority ones, leading to widespread time-to-first-token (TTFT) SLO violations. While chunked prefill enables interruptibility, it introduces an inherent trade-off between responsiveness and throughput: reducing chunk size improves response latency but degrades computational efficiency, whereas increasing chunk size maximizes throughput but exacerbates blocking. This necessitates an adaptive preemption mechanism. However, dynamically balancing execution granularity against scheduling overheads remains a key challenge. In this paper, we propose FlowPrefill, a TTFT-goodput-optimized serving system that resolves this conflict by decoupling preemption granularity from scheduling frequency. To achieve adaptive prefill scheduling, FlowPrefill introduces two key innovations: 1) Operator-Level Preemption, which leverages operator boundaries to enable fine-grained execution interruption without the efficiency loss associated with fixed small chunking; and 2) Event-Driven Scheduling, which triggers scheduling decisions only upon request arrival or completion events, thereby supporting efficient preemption responsiveness while minimizing control-plane overhead. Evaluation on real-world production traces shows that FlowPrefill improves maximum goodput by up to 5.6$\times$ compared to state-of-the-art systems while satisfying heterogeneous SLOs.

  </details>



- **EasyControlEdge: A Foundation-Model Fine-Tuning for Edge Detection**  
  Hiroki Nakamura, Hiroto Iino, Masashi Okada, Tadahiro Taniguchi  
  _2026-02-18_ · https://arxiv.org/abs/2602.16238v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We propose EasyControlEdge, adapting an image-generation foundation model to edge detection. In real-world edge detection (e.g., floor-plan walls, satellite roads/buildings, and medical organ boundaries), crispness and data efficiency are crucial, yet producing crisp raw edge maps with limited training samples remains challenging. Although image-generation foundation models perform well on many downstream tasks, their pretrained priors for data-efficient transfer and iterative refinement for high-frequency detail preservation remain underexploited for edge detection. To enable crisp and data-efficient edge detection using these capabilities, we introduce an edge-specialized adaptation of image-generation foundation models. To better specialize the foundation model for edge detection, we incorporate an edge-oriented objective with an efficient pixel-space loss. At inference, we introduce guidance based on unconditional dynamics, enabling a single model to control the edge density through a guidance scale. Experiments on BSDS500, NYUDv2, BIPED, and CubiCasa compare against state-of-the-art methods and show consistent gains, particularly under no-post-processing crispness evaluation and with limited training data.

  </details>


