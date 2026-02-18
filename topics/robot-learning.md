# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-18 07:14 UTC_

Total papers shown: **16**


---

- **Selective Perception for Robot: Task-Aware Attention in Multimodal VLA**  
  Young-Chae Son, Jung-Woo Lee, Yoon-Ji Choi, Dae-Kwan Ko, Soo-Chul Lim  
  _2026-02-17_ · https://arxiv.org/abs/2602.15543v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In robotics, Vision-Language-Action (VLA) models that integrate diverse multimodal signals from multi-view inputs have emerged as an effective approach. However, most prior work adopts static fusion that processes all visual inputs uniformly, which incurs unnecessary computational overhead and allows task-irrelevant background information to act as noise. Inspired by the principles of human active perception, we propose a dynamic information fusion framework designed to maximize the efficiency and robustness of VLA models. Our approach introduces a lightweight adaptive routing architecture that analyzes the current text prompt and observations from a wrist-mounted camera in real-time to predict the task-relevance of multiple camera views. By conditionally attenuating computations for views with low informational utility and selectively providing only essential visual features to the policy network, Our framework achieves computation efficiency proportional to task relevance. Furthermore, to efficiently secure large-scale annotation data for router training, we established an automated labeling pipeline utilizing Vision-Language Models (VLMs) to minimize data collection and annotation costs. Experimental results in real-world robotic manipulation scenarios demonstrate that the proposed approach achieves significant improvements in both inference efficiency and control performance compared to existing VLA models, validating the effectiveness and practicality of dynamic information fusion in resource-constrained, real-time robot control environments.

  </details>



- **Feasibility-aware Imitation Learning from Observation with Multimodal Feedback**  
  Kei Takahashi, Hikaru Sasaki, Takamitsu Matsubara  
  _2026-02-17_ · https://arxiv.org/abs/2602.15351v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Imitation learning frameworks that learn robot control policies from demonstrators' motions via hand-mounted demonstration interfaces have attracted increasing attention. However, due to differences in physical characteristics between demonstrators and robots, this approach faces two limitations: i) the demonstration data do not include robot actions, and ii) the demonstrated motions may be infeasible for robots. These limitations make policy learning difficult. To address them, we propose Feasibility-Aware Behavior Cloning from Observation (FABCO). FABCO integrates behavior cloning from observation, which complements robot actions using robot dynamics models, with feasibility estimation. In feasibility estimation, the demonstrated motions are evaluated using a robot-dynamics model, learned from the robot's execution data, to assess reproducibility under the robot's dynamics. The estimated feasibility is used for multimodal feedback and feasibility-aware policy learning to improve the demonstrator's motions and learn robust policies. Multimodal feedback provides feasibility through the demonstrator's visual and haptic senses to promote feasible demonstrated motions. Feasibility-aware policy learning reduces the influence of demonstrated motions that are infeasible for robots, enabling the learning of policies that robots can execute stably. We conducted experiments with 15 participants on two tasks and confirmed that FABCO improves imitation learning performance by more than 3.2 times compared to the case without feasibility feedback.

  </details>



- **Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching**  
  Zhen Wu, Xiaoyu Huang, Lujie Yang, Yuanhang Zhang, Koushil Sreenath, Xi Chen, Pieter Abbeel, Rocky Duan, Angjoo Kanazawa, Carmelo Sferrazza, et al.  
  _2026-02-17_ · https://arxiv.org/abs/2602.15827v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While recent advances in humanoid locomotion have achieved stable walking on varied terrains, capturing the agility and adaptivity of highly dynamic human motions remains an open challenge. In particular, agile parkour in complex environments demands not only low-level robustness, but also human-like motion expressiveness, long-horizon skill composition, and perception-driven decision-making. In this paper, we present Perceptive Humanoid Parkour (PHP), a modular framework that enables humanoid robots to autonomously perform long-horizon, vision-based parkour across challenging obstacle courses. Our approach first leverages motion matching, formulated as nearest-neighbor search in a feature space, to compose retargeted atomic human skills into long-horizon kinematic trajectories. This framework enables the flexible composition and smooth transition of complex skill chains while preserving the elegance and fluidity of dynamic human motions. Next, we train motion-tracking reinforcement learning (RL) expert policies for these composed motions, and distill them into a single depth-based, multi-skill student policy, using a combination of DAgger and RL. Crucially, the combination of perception and skill composition enables autonomous, context-aware decision-making: using only onboard depth sensing and a discrete 2D velocity command, the robot selects and executes whether to step over, climb onto, vault or roll off obstacles of varying geometries and heights. We validate our framework with extensive real-world experiments on a Unitree G1 humanoid robot, demonstrating highly dynamic parkour skills such as climbing tall obstacles up to 1.25m (96% robot height), as well as long-horizon multi-obstacle traversal with closed-loop adaptation to real-time obstacle perturbations.

  </details>



- **ActionCodec: What Makes for Good Action Tokenizers**  
  Zibin Dong, Yicheng Liu, Shiduo Zhang, Baijun Ye, Yifu Yuan, Fei Ni, Jingjing Gong, Xipeng Qiu, Hang Zhao, Yinchuan Li, et al.  
  _2026-02-17_ · https://arxiv.org/abs/2602.15397v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models leveraging the native autoregressive paradigm of Vision-Language Models (VLMs) have demonstrated superior instruction-following and training efficiency. Central to this paradigm is action tokenization, yet its design has primarily focused on reconstruction fidelity, failing to address its direct impact on VLA optimization. Consequently, the fundamental question of \textit{what makes for good action tokenizers} remains unanswered. In this paper, we bridge this gap by establishing design principles specifically from the perspective of VLA optimization. We identify a set of best practices based on information-theoretic insights, including maximized temporal token overlap, minimized vocabulary redundancy, enhanced multimodal mutual information, and token independence. Guided by these principles, we introduce \textbf{ActionCodec}, a high-performance action tokenizer that significantly enhances both training efficiency and VLA performance across diverse simulation and real-world benchmarks. Notably, on LIBERO, a SmolVLM2-2.2B fine-tuned with ActionCodec achieves a 95.5\% success rate without any robotics pre-training. With advanced architectural enhancements, this reaches 97.4\%, representing a new SOTA for VLA models without robotics pre-training. We believe our established design principles, alongside the released model, will provide a clear roadmap for the community to develop more effective action tokenizers.

  </details>



- **MeshMimic: Geometry-Aware Humanoid Motion Learning through 3D Scene Reconstruction**  
  Qiang Zhang, Jiahao Ma, Peiran Liu, Shuai Shi, Zeran Su, Zifan Wang, Jingkai Sun, Wei Cui, Jialin Yu, Gang Han, et al.  
  _2026-02-17_ · https://arxiv.org/abs/2602.15733v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Humanoid motion control has witnessed significant breakthroughs in recent years, with deep reinforcement learning (RL) emerging as a primary catalyst for achieving complex, human-like behaviors. However, the high dimensionality and intricate dynamics of humanoid robots make manual motion design impractical, leading to a heavy reliance on expensive motion capture (MoCap) data. These datasets are not only costly to acquire but also frequently lack the necessary geometric context of the surrounding physical environment. Consequently, existing motion synthesis frameworks often suffer from a decoupling of motion and scene, resulting in physical inconsistencies such as contact slippage or mesh penetration during terrain-aware tasks. In this work, we present MeshMimic, an innovative framework that bridges 3D scene reconstruction and embodied intelligence to enable humanoid robots to learn coupled "motion-terrain" interactions directly from video. By leveraging state-of-the-art 3D vision models, our framework precisely segments and reconstructs both human trajectories and the underlying 3D geometry of terrains and objects. We introduce an optimization algorithm based on kinematic consistency to extract high-quality motion data from noisy visual reconstructions, alongside a contact-invariant retargeting method that transfers human-environment interaction features to the humanoid agent. Experimental results demonstrate that MeshMimic achieves robust, highly dynamic performance across diverse and challenging terrains. Our approach proves that a low-cost pipeline utilizing only consumer-grade monocular sensors can facilitate the training of complex physical interactions, offering a scalable path toward the autonomous evolution of humanoid robots in unstructured environments.

  </details>



- **EAA: Automating materials characterization with vision language model agents**  
  Ming Du, Yanqi Luo, Srutarshi Banerjee, Michael Wojcik, Jelena Popovic, Mathew J. Cherukara  
  _2026-02-17_ · https://arxiv.org/abs/2602.15294v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We present Experiment Automation Agents (EAA), a vision-language-model-driven agentic system designed to automate complex experimental microscopy workflows. EAA integrates multimodal reasoning, tool-augmented action, and optional long-term memory to support both autonomous procedures and interactive user-guided measurements. Built on a flexible task-manager architecture, the system enables workflows ranging from fully agent-driven automation to logic-defined routines that embed localized LLM queries. EAA further provides a modern tool ecosystem with two-way compatibility for Model Context Protocol (MCP), allowing instrument-control tools to be consumed or served across applications. We demonstrate EAA at an imaging beamline at the Advanced Photon Source, including automated zone plate focusing, natural language-described feature search, and interactive data acquisition. These results illustrate how vision-capable agents can enhance beamline efficiency, reduce operational burden, and lower the expertise barrier for users.

  </details>



- **Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning**  
  Oswin So, Eric Yang Yu, Songyuan Zhang, Matthew Cleaveland, Mitchell Black, Chuchu Fan  
  _2026-02-17_ · https://arxiv.org/abs/2602.15817v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Recent advances in deep reinforcement learning (RL) have achieved strong results on high-dimensional control tasks, but applying RL to reachability problems raises a fundamental mismatch: reachability seeks to maximize the set of states from which a system remains safe indefinitely, while RL optimizes expected returns over a user-specified distribution. This mismatch can result in policies that perform poorly on low-probability states that are still within the safe set. A natural alternative is to frame the problem as a robust optimization over a set of initial conditions that specify the initial state, dynamics and safe set, but whether this problem has a solution depends on the feasibility of the specified set, which is unknown a priori. We propose Feasibility-Guided Exploration (FGE), a method that simultaneously identifies a subset of feasible initial conditions under which a safe policy exists, and learns a policy to solve the reachability problem over this set of initial conditions. Empirical results demonstrate that FGE learns policies with over 50% more coverage than the best existing method for challenging initial conditions across tasks in the MuJoCo simulator and the Kinetix simulator with pixel observations.

  </details>



- **Learning to Retrieve Navigable Candidates for Efficient Vision-and-Language Navigation**  
  Shutian Gu, Chengkai Huang, Ruoyu Wang, Lina Yao  
  _2026-02-17_ · https://arxiv.org/abs/2602.15724v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-and-Language Navigation (VLN) requires an agent to follow natural-language instructions and navigate through previously unseen environments. Recent approaches increasingly employ large language models (LLMs) as high-level navigators due to their flexibility and reasoning capability. However, prompt-based LLM navigation often suffers from inefficient decision-making, as the model must repeatedly interpret instructions from scratch and reason over noisy and verbose navigable candidates at each step. In this paper, we propose a retrieval-augmented framework to improve the efficiency and stability of LLM-based VLN without modifying or fine-tuning the underlying language model. Our approach introduces retrieval at two complementary levels. At the episode level, an instruction-level embedding retriever selects semantically similar successful navigation trajectories as in-context exemplars, providing task-specific priors for instruction grounding. At the step level, an imitation-learned candidate retriever prunes irrelevant navigable directions before LLM inference, reducing action ambiguity and prompt complexity. Both retrieval modules are lightweight, modular, and trained independently of the LLM. We evaluate our method on the Room-to-Room (R2R) benchmark. Experimental results demonstrate consistent improvements in Success Rate, Oracle Success Rate, and SPL on both seen and unseen environments. Ablation studies further show that instruction-level exemplar retrieval and candidate pruning contribute complementary benefits to global guidance and step-wise decision efficiency. These results indicate that retrieval-augmented decision support is an effective and scalable strategy for enhancing LLM-based vision-and-language navigation.

  </details>



- **CAMEL: An ECG Language Model for Forecasting Cardiac Events**  
  Neelay Velingker, Alaia Solko-Breslin, Mayank Keoliya, Seewon Choi, Jiayi Xin, Anika Marathe, Alireza Oraii, Rajat Deo, Sameed Khatana, Rajeev Alur, et al.  
  _2026-02-17_ · https://arxiv.org/abs/2602.15677v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Electrocardiograms (ECG) are electrical recordings of the heart that are critical for diagnosing cardiovascular conditions. ECG language models (ELMs) have recently emerged as a promising framework for ECG classification accompanied by report generation. However, current models cannot forecast future cardiac events despite the immense clinical value for planning earlier intervention. To address this gap, we propose CAMEL, the first ELM that is capable of inference over longer signal durations which enables its forecasting capability. Our key insight is a specialized ECG encoder which enables cross-understanding of ECG signals with text. We train CAMEL using established LLM training procedures, combining LoRA adaptation with a curriculum learning pipeline. Our curriculum includes ECG classification, metrics calculations, and multi-turn conversations to elicit reasoning. CAMEL demonstrates strong zero-shot performance across 6 tasks and 9 datasets, including ECGForecastBench, a new benchmark that we introduce for forecasting arrhythmias. CAMEL is on par with or surpasses ELMs and fully supervised baselines both in- and out-of-distribution, achieving SOTA results on ECGBench (+7.0% absolute average gain) as well as ECGForecastBench (+12.4% over fully supervised models and +21.1% over zero-shot ELMs).

  </details>



- **Revisiting Northrop Frye's Four Myths Theory with Large Language Models**  
  Edirlei Soares de Lima, Marco A. Casanova, Antonio L. Furtado  
  _2026-02-17_ · https://arxiv.org/abs/2602.15678v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Northrop Frye's theory of four fundamental narrative genres (comedy, romance, tragedy, satire) has profoundly influenced literary criticism, yet computational approaches to his framework have focused primarily on narrative patterns rather than character functions. In this paper, we present a new character function framework that complements pattern-based analysis by examining how archetypal roles manifest differently across Frye's genres. Drawing on Jungian archetype theory, we derive four universal character functions (protagonist, mentor, antagonist, companion) by mapping them to Jung's psychic structure components. These functions are then specialized into sixteen genre-specific roles based on prototypical works. To validate this framework, we conducted a multi-model study using six state-of-the-art Large Language Models (LLMs) to evaluate character-role correspondences across 40 narrative works. The validation employed both positive samples (160 valid correspondences) and negative samples (30 invalid correspondences) to evaluate whether models both recognize valid correspondences and reject invalid ones. LLMs achieved substantial performance (mean balanced accuracy of 82.5%) with strong inter-model agreement (Fleiss' $κ$ = 0.600), demonstrating that the proposed correspondences capture systematic structural patterns. Performance varied by genre (ranging from 72.7% to 89.9%) and role (52.5% to 99.2%), with qualitative analysis revealing that variations reflect genuine narrative properties, including functional distribution in romance and deliberate archetypal subversion in satire. This character-based approach demonstrates the potential of LLM-supported methods for computational narratology and provides a foundation for future development of narrative generation methods and interactive storytelling applications.

  </details>



- **PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra**  
  Xiachong Feng, Liang Zhao, Weihong Zhong, Yichong Huang, Yuxuan Gu, Lingpeng Kong, Xiaocheng Feng, Bing Qin  
  _2026-02-17_ · https://arxiv.org/abs/2602.15669v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Current methods for personality control in Large Language Models rely on static prompting or expensive fine-tuning, failing to capture the dynamic and compositional nature of human traits. We introduce PERSONA, a training-free framework that achieves fine-tuning level performance through direct manipulation of personality vectors in activation space. Our key insight is that personality traits appear as extractable, approximately orthogonal directions in the model's representation space that support algebraic operations. The framework operates through three stages: Persona-Base extracts orthogonal trait vectors via contrastive activation analysis; Persona-Algebra enables precise control through vector arithmetic (scalar multiplication for intensity, addition for composition, subtraction for suppression); and Persona-Flow achieves context-aware adaptation by dynamically composing these vectors during inference. On PersonalityBench, our approach achieves a mean score of 9.60, nearly matching the supervised fine-tuning upper bound of 9.61 without any gradient updates. On our proposed Persona-Evolve benchmark for dynamic personality adaptation, we achieve up to 91% win rates across diverse model families. These results provide evidence that aspects of LLM personality are mathematically tractable, opening new directions for interpretable and efficient behavioral control.

  </details>



- **Zombie Agents: Persistent Control of Self-Evolving LLM Agents via Self-Reinforcing Injections**  
  Xianglin Yang, Yufei He, Shuo Ji, Bryan Hooi, Jin Song Dong  
  _2026-02-17_ · https://arxiv.org/abs/2602.15654v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Self-evolving LLM agents update their internal state across sessions, often by writing and reusing long-term memory. This design improves performance on long-horizon tasks but creates a security risk: untrusted external content observed during a benign session can be stored as memory and later treated as instruction. We study this risk and formalize a persistent attack we call a Zombie Agent, where an attacker covertly implants a payload that survives across sessions, effectively turning the agent into a puppet of the attacker. We present a black-box attack framework that uses only indirect exposure through attacker-controlled web content. The attack has two phases. During infection, the agent reads a poisoned source while completing a benign task and writes the payload into long-term memory through its normal update process. During trigger, the payload is retrieved or carried forward and causes unauthorized tool behavior. We design mechanism-specific persistence strategies for common memory implementations, including sliding-window and retrieval-augmented memory, to resist truncation and relevance filtering. We evaluate the attack on representative agent setups and tasks, measuring both persistence over time and the ability to induce unauthorized actions while preserving benign task quality. Our results show that memory evolution can convert one-time indirect injection into persistent compromise, which suggests that defenses focused only on per-session prompt filtering are not sufficient for self-evolving agents.

  </details>



- **Latency-aware Human-in-the-Loop Reinforcement Learning for Semantic Communications**  
  Peizheng Li, Xinyi Lin, Adnan Aijaz  
  _2026-02-17_ · https://arxiv.org/abs/2602.15640v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Semantic communication promises task-aligned transmission but must reconcile semantic fidelity with stringent latency guarantees in immersive and safety-critical services. This paper introduces a time-constrained human-in-the-loop reinforcement learning (TC-HITL-RL) framework that embeds human feedback, semantic utility, and latency control within a semantic-aware Open radio access network (RAN) architecture. We formulate semantic adaptation driven by human feedback as a constrained Markov decision process (CMDP) whose state captures semantic quality, human preferences, queue slack, and channel dynamics, and solve it via a primal--dual proximal policy optimization algorithm with action shielding and latency-aware reward shaping. The resulting policy preserves PPO-level semantic rewards while tightening the variability of both air-interface and near-real-time RAN intelligent controller processing budgets. Simulations over point-to-multipoint links with heterogeneous deadlines show that TC-HITL-RL consistently meets per-user timing constraints, outperforms baseline schedulers in reward, and stabilizes resource consumption, providing a practical blueprint for latency-aware semantic adaptation.

  </details>



- **Efficient Knowledge Transfer for Jump-Starting Control Policy Learning of Multirotors through Physics-Aware Neural Architectures**  
  Welf Rehberg, Mihir Kulkarni, Philipp Weiss, Kostas Alexis  
  _2026-02-17_ · https://arxiv.org/abs/2602.15533v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Efficiently training control policies for robots is a major challenge that can greatly benefit from utilizing knowledge gained from training similar systems through cross-embodiment knowledge transfer. In this work, we focus on accelerating policy training using a library-based initialization scheme that enables effective knowledge transfer across multirotor configurations. By leveraging a physics-aware neural control architecture that combines a reinforcement learning-based controller and a supervised control allocation network, we enable the reuse of previously trained policies. To this end, we utilize a policy evaluation-based similarity measure that identifies suitable policies for initialization from a library. We demonstrate that this measure correlates with the reduction in environment interactions needed to reach target performance and is therefore suited for initialization. Extensive simulation and real-world experiments confirm that our control architecture achieves state-of-the-art control performance, and that our initialization scheme saves on average up to $73.5\%$ of environment interactions (compared to training a policy from scratch) across diverse quadrotor and hexarotor designs, paving the way for efficient cross-embodiment transfer in reinforcement learning.

  </details>



- **EventMemAgent: Hierarchical Event-Centric Memory for Online Video Understanding with Adaptive Tool Use**  
  Siwei Wen, Zhangcheng Wang, Xingjian Zhang, Lei Huang, Wenjun Wu  
  _2026-02-17_ · https://arxiv.org/abs/2602.15329v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Online video understanding requires models to perform continuous perception and long-range reasoning within potentially infinite visual streams. Its fundamental challenge lies in the conflict between the unbounded nature of streaming media input and the limited context window of Multimodal Large Language Models (MLLMs). Current methods primarily rely on passive processing, which often face a trade-off between maintaining long-range context and capturing the fine-grained details necessary for complex tasks. To address this, we introduce EventMemAgent, an active online video agent framework based on a hierarchical memory module. Our framework employs a dual-layer strategy for online videos: short-term memory detects event boundaries and utilizes event-granular reservoir sampling to process streaming video frames within a fixed-length buffer dynamically; long-term memory structuredly archives past observations on an event-by-event basis. Furthermore, we integrate a multi-granular perception toolkit for active, iterative evidence capture and employ Agentic Reinforcement Learning (Agentic RL) to end-to-end internalize reasoning and tool-use strategies into the agent's intrinsic capabilities. Experiments show that EventMemAgent achieves competitive results on online video benchmarks. The code will be released here: https://github.com/lingcco/EventMemAgent.

  </details>



- **Prescriptive Scaling Reveals the Evolution of Language Model Capabilities**  
  Hanlin Zhang, Jikai Jin, Vasilis Syrgkanis, Sham Kakade  
  _2026-02-17_ · https://arxiv.org/abs/2602.15327v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  For deploying foundation models, practitioners increasingly need prescriptive scaling laws: given a pre training compute budget, what downstream accuracy is attainable with contemporary post training practice, and how stable is that mapping as the field evolves? Using large scale observational evaluations with 5k observational and 2k newly sampled data on model performance, we estimate capability boundaries, high conditional quantiles of benchmark scores as a function of log pre training FLOPs, via smoothed quantile regression with a monotone, saturating sigmoid parameterization. We validate the temporal reliability by fitting on earlier model generations and evaluating on later releases. Across various tasks, the estimated boundaries are mostly stable, with the exception of math reasoning that exhibits a consistently advancing boundary over time. We then extend our approach to analyze task dependent saturation and to probe contamination related shifts on math reasoning tasks. Finally, we introduce an efficient algorithm that recovers near full data frontiers using roughly 20% of evaluation budget. Together, our work releases the Proteus 2k, the latest model performance evaluation dataset, and introduces a practical methodology for translating compute budgets into reliable performance expectations and for monitoring when capability boundaries shift across time.

  </details>


