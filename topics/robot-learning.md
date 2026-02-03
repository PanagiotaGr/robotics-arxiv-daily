# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-03 07:06 UTC_

Total papers shown: **20**


---

- **TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments**  
  Zhiyu Huang, Yun Zhang, Johnson Liu, Rui Song, Chen Tang, Jiaqi Ma  
  _2026-02-02_ · https://arxiv.org/abs/2602.02459v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots in dynamic, human-centric environments must follow language instructions while maintaining real-time reactive control. Vision-language-action (VLA) models offer a promising framework, but they assume temporally aligned reasoning and control, despite semantic inference being inherently delayed relative to real-time action. We introduce Think-in-Control (TIC)-VLA, a latency-aware framework that explicitly models delayed semantic reasoning during action generation. TIC-VLA defines a delayed semantic-control interface that conditions action generation on delayed vision-language semantic states and explicit latency metadata, in addition to current observations, enabling policies to compensate for asynchronous reasoning. We further propose a latency-consistent training pipeline that injects reasoning inference delays during imitation learning and online reinforcement learning, aligning training with asynchronous deployment. To support realistic evaluation, we present DynaNav, a physics-accurate, photo-realistic simulation suite for language-guided navigation in dynamic environments. Extensive experiments in simulation and on a real robot show that TIC-VLA consistently outperforms prior VLA models while maintaining robust real-time control under multi-second reasoning latency. Project website: https://ucla-mobility.github.io/TIC-VLA/

  </details>



- **Relationship-Aware Hierarchical 3D Scene Graph for Task Reasoning**  
  Albert Gassol Puigjaner, Angelos Zacharia, Kostas Alexis  
  _2026-02-02_ · https://arxiv.org/abs/2602.02456v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Representing and understanding 3D environments in a structured manner is crucial for autonomous agents to navigate and reason about their surroundings. While traditional Simultaneous Localization and Mapping (SLAM) methods generate metric reconstructions and can be extended to metric-semantic mapping, they lack a higher level of abstraction and relational reasoning. To address this gap, 3D scene graphs have emerged as a powerful representation for capturing hierarchical structures and object relationships. In this work, we propose an enhanced hierarchical 3D scene graph that integrates open-vocabulary features across multiple abstraction levels and supports object-relational reasoning. Our approach leverages a Vision Language Model (VLM) to infer semantic relationships. Notably, we introduce a task reasoning module that combines Large Language Models (LLM) and a VLM to interpret the scene graph's semantic and relational information, enabling agents to reason about tasks and interact with their environment more intelligently. We validate our method by deploying it on a quadruped robot in multiple environments and tasks, highlighting its ability to reason about them.

  </details>



- **FD-VLA: Force-Distilled Vision-Language-Action Model for Contact-Rich Manipulation**  
  Ruiteng Zhao, Wenshuo Wang, Yicheng Ma, Xiaocong Li, Francis E. H. Tay, Marcelo H. Ang, Haiyue Zhu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02142v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Force sensing is a crucial modality for Vision-Language-Action (VLA) frameworks, as it enables fine-grained perception and dexterous manipulation in contact-rich tasks. We present Force-Distilled VLA (FD-VLA), a novel framework that integrates force awareness into contact-rich manipulation without relying on physical force sensors. The core of our approach is a Force Distillation Module (FDM), which distills force by mapping a learnable query token, conditioned on visual observations and robot states, into a predicted force token aligned with the latent representation of actual force signals. During inference, this distilled force token is injected into the pretrained VLM, enabling force-aware reasoning while preserving the integrity of its vision-language semantics. This design provides two key benefits: first, it allows practical deployment across a wide range of robots that lack expensive or fragile force-torque sensors, thereby reducing hardware cost and complexity; second, the FDM introduces an additional force-vision-state fusion prior to the VLM, which improves cross-modal alignment and enhances perception-action robustness in contact-rich scenarios. Surprisingly, our physical experiments show that the distilled force token outperforms direct sensor force measurements as well as other baselines, which highlights the effectiveness of this force-distilled VLA approach.

  </details>



- **3D Foundation Model-Based Loop Closing for Decentralized Collaborative SLAM**  
  Pierre-Yves Lajoie, Benjamin Ramtoula, Daniele De Martini, Giovanni Beltrame  
  _2026-02-02_ · https://arxiv.org/abs/2602.02430v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Decentralized Collaborative Simultaneous Localization And Mapping (C-SLAM) techniques often struggle to identify map overlaps due to significant viewpoint variations among robots. Motivated by recent advancements in 3D foundation models, which can register images despite large viewpoint differences, we propose a robust loop closing approach that leverages these models to establish inter-robot measurements. In contrast to resource-intensive methods requiring full 3D reconstruction within a centralized map, our approach integrates foundation models into existing SLAM pipelines, yielding scalable and robust multi-robot mapping. Our contributions include: (1) integrating 3D foundation models to reliably estimate relative poses from monocular image pairs within decentralized C-SLAM; (2) introducing robust outlier mitigation techniques critical to the use of these relative poses; and (3) developing specialized pose graph optimization formulations that efficiently resolve scale ambiguities. We evaluate our method against state-of-the-art approaches, demonstrating improvements in localization and mapping accuracy, alongside significant gains in computational and memory efficiency. These results highlight the potential of our approach for deployment in large-scale multi-robot scenarios.

  </details>



- **World-Gymnast: Training Robots with Reinforcement Learning in a World Model**  
  Ansh Kumar Sharma, Yixiang Sun, Ninghao Lu, Yunzhe Zhang, Jiarao Liu, Sherry Yang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02454v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robot learning from interacting with the physical world is fundamentally bottlenecked by the cost of physical interaction. The two alternatives, supervised finetuning (SFT) from expert demonstrations and reinforcement learning (RL) in a software-based simulator, are limited by the amount of expert data available and the sim-to-real gap for manipulation. With the recent emergence of world models learned from real-world video-action data, we ask the question of whether training a policy in a world model can be more effective than supervised learning or software simulation in achieving better real-robot performance. We propose World-Gymnast, which performs RL finetuning of a vision-language-action (VLA) policy by rolling out the policy in an action-conditioned video world model and rewarding the rollouts with a vision-language model (VLM). On the Bridge robot setup, World-Gymnast outperforms SFT by as much as 18x and outperforms software simulator by as much as 2x. More importantly, World-Gymnast demonstrates intriguing capabilities of RL with a world model, including training on diverse language instructions and novel scenes from the world model, test-time training in a novel scene, and online iterative world model and policy improvement. Our results suggest learning a world model and training robot policies in the cloud could be the key to bridging the gap between robots that work in demonstrations and robots that can work in anyone's household.

  </details>



- **HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos**  
  Yinhuai Wang, Qihan Zhao, Yuen Fui Lau, Runyi Yu, Hok Wai Tsui, Qifeng Chen, Jingbo Wang, Jiangmiao Pang, Ping Tan  
  _2026-02-02_ · https://arxiv.org/abs/2602.02473v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Enabling humanoid robots to perform agile and adaptive interactive tasks has long been a core challenge in robotics. Current approaches are bottlenecked by either the scarcity of realistic interaction data or the need for meticulous, task-specific reward engineering, which limits their scalability. To narrow this gap, we present HumanX, a full-stack framework that compiles human video into generalizable, real-world interaction skills for humanoids, without task-specific rewards. HumanX integrates two co-designed components: XGen, a data generation pipeline that synthesizes diverse and physically plausible robot interaction data from video while supporting scalable data augmentation; and XMimic, a unified imitation learning framework that learns generalizable interaction skills. Evaluated across five distinct domains--basketball, football, badminton, cargo pickup, and reactive fighting--HumanX successfully acquires 10 different skills and transfers them zero-shot to a physical Unitree G1 humanoid. The learned capabilities include complex maneuvers such as pump-fake turnaround fadeaway jumpshots without any external perception, as well as interactive tasks like sustained human-robot passing sequences over 10 consecutive cycles--learned from a single video demonstration. Our experiments show that HumanX achieves over 8 times higher generalization success than prior methods, demonstrating a scalable and task-agnostic pathway for learning versatile, real-world robot interactive skills.

  </details>



- **Segment to Focus: Guiding Latent Action Models in the Presence of Distractors**  
  Hamza Adnan, Matthew T. Jackson, Alexey Zakharov  
  _2026-02-02_ · https://arxiv.org/abs/2602.02259v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Latent Action Models (LAMs) learn to extract action-relevant representations solely from raw observations, enabling reinforcement learning from unlabelled videos and significantly scaling available training data. However, LAMs face a critical challenge in disentangling action-relevant features from action-correlated noise (e.g., background motion). Failing to filter these distractors causes LAMs to capture spurious correlations and build sub-optimal latent action spaces. In this paper, we introduce MaskLAM -- a lightweight modification to LAM training to mitigate this issue by incorporating visual agent segmentation. MaskLAM utilises segmentation masks from pretrained foundation models to weight the LAM reconstruction loss, thereby prioritising salient information over background elements while requiring no architectural modifications. We demonstrate the effectiveness of our method on continuous-control MuJoCo tasks, modified with action-correlated background noise. Our approach yields up to a 4x increase in accrued rewards compared to standard baselines and a 3x improvement in the latent action quality, as evidenced by linear probe evaluation.

  </details>



- **DCoPilot: Generative AI-Empowered Policy Adaptation for Dynamic Data Center Operations**  
  Minghao Li, Ruihang Wang, Rui Tan, Yonggang Wen  
  _2026-02-02_ · https://arxiv.org/abs/2602.02137v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Modern data centers (DCs) hosting artificial intelligence (AI)-dedicated devices operate at high power densities with rapidly varying workloads, making minute-level adaptation essential for safe and energy-efficient operation. However, manually designing piecewise deep reinforcement learning (DRL) agents cannot keep pace with frequent dynamics shifts and service-level agreement (SLA) changes of an evolving DC. This specification-to-policy lag causes a lack of timely, effective control policies, which may lead to service outages. To bridge the gap, we present DCoPilot, a hybrid framework for generative control policies in dynamic DC operation. DCoPilot synergizes two distinct generative paradigms, i.e., a large language model (LLM) that performs symbolic generation of structured reward forms, and a hypernetwork that conducts parametric generation of policy weights. DCoPilot operates through three coordinated phases: (i) simulation scale-up, which stress-tests reward candidates across diverse simulation-ready (SimReady) scenes; (ii) meta policy distillation, where a hypernetwork is trained to output policy weights conditioned on SLA and scene embeddings; and (iii) online adaptation, enabling zero-shot policy generation in response to updated specifications. Evaluated across five control task families spanning diverse DC components, DCoPilot achieves near-zero constraint violations and outperforms all baselines across specification variations. Ablation studies validate the effectiveness of LLM-based unified reward generation in enabling stable hypernetwork convergence.

  </details>



- **David vs. Goliath: Verifiable Agent-to-Agent Jailbreaking via Reinforcement Learning**  
  Samuel Nellessen, Tal Kachman  
  _2026-02-02_ · https://arxiv.org/abs/2602.02395v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The evolution of large language models into autonomous agents introduces adversarial failures that exploit legitimate tool privileges, transforming safety evaluation in tool-augmented environments from a subjective NLP task into an objective control problem. We formalize this threat model as Tag-Along Attacks: a scenario where a tool-less adversary "tags along" on the trusted privileges of a safety-aligned Operator to induce prohibited tool use through conversation alone. To validate this threat, we present Slingshot, a 'cold-start' reinforcement learning framework that autonomously discovers emergent attack vectors, revealing a critical insight: in our setting, learned attacks tend to converge to short, instruction-like syntactic patterns rather than multi-turn persuasion. On held-out extreme-difficulty tasks, Slingshot achieves a 67.0% success rate against a Qwen2.5-32B-Instruct-AWQ Operator (vs. 1.7% baseline), reducing the expected attempts to first success (on solved tasks) from 52.3 to 1.3. Crucially, Slingshot transfers zero-shot to several model families, including closed-source models like Gemini 2.5 Flash (56.0% attack success rate) and defensive-fine-tuned open-source models like Meta-SecAlign-8B (39.2% attack success rate). Our work establishes Tag-Along Attacks as a first-class, verifiable threat model and shows that effective agentic attacks can be elicited from off-the-shelf open-weight models through environment interaction alone.

  </details>



- **Interpreting and Controlling LLM Reasoning through Integrated Policy Gradient**  
  Changming Li, Kaixing Zhang, Haoyun Xu, Yingdong Shi, Zheng Zhang, Kaitao Song, Kan Ren  
  _2026-02-02_ · https://arxiv.org/abs/2602.02313v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) demonstrate strong reasoning abilities in solving complex real-world problems. Yet, the internal mechanisms driving these complex reasoning behaviors remain opaque. Existing interpretability approaches targeting reasoning either identify components (e.g., neurons) correlated with special textual patterns, or rely on human-annotated contrastive pairs to derive control vectors. Consequently, current methods struggle to precisely localize complex reasoning mechanisms or capture sequential influence from model internal workings to the reasoning outputs. In this paper, built on outcome-oriented and sequential-influence-aware principles, we focus on identifying components that have sequential contribution to reasoning behavior where outcomes are cumulated by long-range effects. We propose Integrated Policy Gradient (IPG), a novel framework that attributes reasoning behaviors to model's inner components by propagating compound outcome-based signals such as post reasoning accuracy backward through model inference trajectories. Empirical evaluations demonstrate that our approach achieves more precise localization and enables reliable modulation of reasoning behaviors (e.g., reasoning capability, reasoning strength) across diverse reasoning models.

  </details>



- **Online Fine-Tuning of Pretrained Controllers for Autonomous Driving via Real-Time Recurrent RL**  
  Julian Lemmel, Felix Resch, Mónika Farsang, Ramin Hasani, Daniela Rus, Radu Grosu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02236v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Deploying pretrained policies in real-world applications presents substantial challenges that fundamentally limit the practical applicability of learning-based control systems. When autonomous systems encounter environmental changes in system dynamics, sensor drift, or task objectives, fixed policies rapidly degrade in performance. We show that employing Real-Time Recurrent Reinforcement Learning (RTRRL), a biologically plausible algorithm for online adaptation, can effectively fine-tune a pretrained policy to improve autonomous agents' performance on driving tasks. We further show that RTRRL synergizes with a recent biologically inspired recurrent network model, the Liquid-Resistance Liquid-Capacitance RNN. We demonstrate the effectiveness of this closed-loop approach in a simulated CarRacing environment and in a real-world line-following task with a RoboRacer car equipped with an event camera.

  </details>



- **RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents**  
  Jialiang Zhu, Gongrui Zhang, Xiaolong Ma, Lin Xu, Miaosen Zhang, Ruiqi Yang, Song Wang, Kai Qiu, Zhirong Wu, Qi Dai, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02486v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  LLM-based deep research agents are largely built on the ReAct framework. This linear design makes it difficult to revisit earlier states, branch into alternative search directions, or maintain global awareness under long contexts, often leading to local optima, redundant exploration, and inefficient search. We propose Re-TRAC, an agentic framework that performs cross-trajectory exploration by generating a structured state representation after each trajectory to summarize evidence, uncertainties, failures, and future plans, and conditioning subsequent trajectories on this state representation. This enables iterative reflection and globally informed planning, reframing research as a progressive process. Empirical results show that Re-TRAC consistently outperforms ReAct by 15-20% on BrowseComp with frontier LLMs. For smaller models, we introduce Re-TRAC-aware supervised fine-tuning, achieving state-of-the-art performance at comparable scales. Notably, Re-TRAC shows a monotonic reduction in tool calls and token usage across rounds, indicating progressively targeted exploration driven by cross-trajectory reflection rather than redundant search.

  </details>



- **AgentRx: Diagnosing AI Agent Failures from Execution Trajectories**  
  Shraddha Barke, Arnav Goyal, Alind Khare, Avaljot Singh, Suman Nath, Chetan Bansal  
  _2026-02-02_ · https://arxiv.org/abs/2602.02475v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  AI agents often fail in ways that are difficult to localize because executions are probabilistic, long-horizon, multi-agent, and mediated by noisy tool outputs. We address this gap by manually annotating failed agent runs and release a novel benchmark of 115 failed trajectories spanning structured API workflows, incident management, and open-ended web/file tasks. Each trajectory is annotated with a critical failure step and a category from a grounded-theory derived, cross-domain failure taxonomy. To mitigate the human cost of failure attribution, we present AGENTRX, an automated domain-agnostic diagnostic framework that pinpoints the critical failure step in a failed agent trajectory. It synthesizes constraints, evaluates them step-by-step, and produces an auditable validation log of constraint violations with associated evidence; an LLM-based judge uses this log to localize the critical step and category. Our framework improves step localization and failure attribution over existing baselines across three domains.

  </details>



- **Drift-Bench: Diagnosing Cooperative Breakdowns in LLM Agents under Input Faults via Multi-Turn Interaction**  
  Han Bao, Zheyuan Zhang, Pengcheng Jing, Zhengqing Yuan, Kaiwen Shi, Yanfang Ye  
  _2026-02-02_ · https://arxiv.org/abs/2602.02455v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  As Large Language Models transition to autonomous agents, user inputs frequently violate cooperative assumptions (e.g., implicit intent, missing parameters, false presuppositions, or ambiguous expressions), creating execution risks that text-only evaluations do not capture. Existing benchmarks typically assume well-specified instructions or restrict evaluation to text-only, single-turn clarification, and thus do not measure multi-turn disambiguation under grounded execution risk. We introduce \textbf{Drift-Bench}, the first diagnostic benchmark that evaluates agentic pragmatics under input faults through multi-turn clarification across state-oriented and service-oriented execution environments. Grounded in classical theories of communication, \textbf{Drift-Bench} provides a unified taxonomy of cooperative breakdowns and employs a persona-driven user simulator with the \textbf{Rise} evaluation protocol. Experiments show substantial performance drops under these faults, with clarification effectiveness varying across user personas and fault types. \MethodName bridges clarification research and agent safety evaluation, enabling systematic diagnosis of failures that can lead to unsafe executions.

  </details>



- **PRISM: Performer RS-IMLE for Single-pass Multisensory Imitation Learning**  
  Amisha Bhaskar, Pratap Tokekar, Stefano Di Cairano, Alexander Schperberg  
  _2026-02-02_ · https://arxiv.org/abs/2602.02396v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic imitation learning typically requires models that capture multimodal action distributions while operating at real-time control rates and accommodating multiple sensing modalities. Although recent generative approaches such as diffusion models, flow matching, and Implicit Maximum Likelihood Estimation (IMLE) have achieved promising results, they often satisfy only a subset of these requirements. To address this, we introduce PRISM, a single-pass policy based on a batch-global rejection-sampling variant of IMLE. PRISM couples a temporal multisensory encoder (integrating RGB, depth, tactile, audio, and proprioception) with a linear-attention generator using a Performer architecture. We demonstrate the efficacy of PRISM on a diverse real-world hardware suite, including loco-manipulation using a Unitree Go2 with a 7-DoF arm D1 and tabletop manipulation with a UR5 manipulator. Across challenging physical tasks such as pre-manipulation parking, high-precision insertion, and multi-object pick-and-place, PRISM outperforms state-of-the-art diffusion policies by 10-25% in success rate while maintaining high-frequency (30-50 Hz) closed-loop control. We further validate our approach on large-scale simulation benchmarks, including CALVIN, MetaWorld, and Robomimic. In CALVIN (10% data split), PRISM improves success rates by approximately 25% over diffusion and approximately 20% over flow matching, while simultaneously reducing trajectory jerk by 20x-50x. These results position PRISM as a fast, accurate, and multisensory imitation policy that retains multimodal action coverage without the latency of iterative sampling.

  </details>



- **Context Learning for Multi-Agent Discussion**  
  Xingyuan Hua, Sheng Yue, Xinyi Li, Yizhe Zhao, Jinrui Zhang, Ju Ren  
  _2026-02-02_ · https://arxiv.org/abs/2602.02350v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Multi-Agent Discussion (MAD) has garnered increasing attention very recently, where multiple LLM instances collaboratively solve problems via structured discussion. However, we find that current MAD methods easily suffer from discussion inconsistency, LLMs fail to reach a coherent solution, due to the misalignment between their individual contexts.In this paper, we introduce a multi-LLM context learning method (M2CL) that learns a context generator for each agent, capable of dynamically generating context instructions per discussion round via automatic information organization and refinement. Specifically, inspired by our theoretical insights on the context instruction, M2CL train the generators to control context coherence and output discrepancies via a carefully crafted self-adaptive mechanism.It enables LLMs to avoid premature convergence on majority noise and progressively reach the correct consensus. We evaluate M2CL on challenging tasks, including academic reasoning, embodied tasks, and mobile control. The results show that the performance of M2CL significantly surpasses existing methods by 20%--50%, while enjoying favorable transferability and computational efficiency.

  </details>



- **Well-Posed KL-Regularized Control via Wasserstein and Kalman-Wasserstein KL Divergences**  
  Viktor Stein, Adwait Datar, Nihat Ay  
  _2026-02-02_ · https://arxiv.org/abs/2602.02250v1 · `math.OC`  
  <details><summary>Abstract</summary>

  Kullback-Leibler divergence (KL) regularization is widely used in reinforcement learning, but it becomes infinite under support mismatch and can degenerate in low-noise limits. Utilizing a unified information-geometric framework, we introduce (Kalman)-Wasserstein-based KL analogues by replacing the Fisher-Rao geometry in the dynamical formulation of the KL with transport-based geometries, and we derive closed-form values for common distribution families. These divergences remain finite under support mismatch and yield a geometric interpretation of regularization heuristics used in Kalman ensemble methods. We demonstrate the utility of these divergences in KL-regularized optimal control. In the fully tractable setting of linear time-invariant systems with Gaussian process noise, the classical KL reduces to a quadratic control penalty that becomes singular as process noise vanishes. Our variants remove this singularity, yielding well-posed problems. On a double integrator and a cart-pole example, the resulting controls outperform KL-based regularization.

  </details>



- **TIDE: Trajectory-based Diagnostic Evaluation of Test-Time Improvement in LLM Agents**  
  Hang Yan, Xinyu Che, Fangzhi Xu, Qiushi Sun, Zichen Ding, Kanzhi Cheng, Jian Zhang, Tao Qin, Jun Liu, Qika Lin  
  _2026-02-02_ · https://arxiv.org/abs/2602.02196v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Recent advances in autonomous LLM agents demonstrate their ability to improve performance through iterative interaction with the environment. We define this paradigm as Test-Time Improvement (TTI). However, the mechanisms under how and why TTI succeed or fail remain poorly understood, and existing evaluation metrics fail to capture their task optimization efficiency, behavior adaptation after erroneous actions, and the specific utility of working memory for task completion. To address these gaps, we propose Test-time Improvement Diagnostic Evaluation (TIDE), an agent-agnostic and environment-agnostic framework that decomposes TTI into three comprehensive and interconnected dimensions. The framework measures (1) the overall temporal dynamics of task completion and (2) identifies whether performance is primarily constrained by recursive looping behaviors or (3) by burdensome accumulated memory. Through extensive experiments across diverse agents and environments, TIDE highlights that improving agent performance requires more than scaling internal reasoning, calling for explicitly optimizing the interaction dynamics between the agent and the environment.

  </details>



- **ECHO: Entropy-Confidence Hybrid Optimization for Test-Time Reinforcement Learning**  
  Chu Zhao, Enneng Yang, Yuting Liu, Jianzhe Zhao, Guibing Guo  
  _2026-02-02_ · https://arxiv.org/abs/2602.02150v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Test-time reinforcement learning generates multiple candidate answers via repeated rollouts and performs online updates using pseudo-labels constructed by majority voting. To reduce overhead and improve exploration, prior work introduces tree structured rollouts, which share reasoning prefixes and branch at key nodes to improve sampling efficiency. However, this paradigm still faces two challenges: (1) high entropy branching can trigger rollout collapse, where the branching budget concentrates on a few trajectories with consecutive high-entropy segments, rapidly reducing the number of effective branches; (2) early pseudo-labels are noisy and biased, which can induce self-reinforcing overfitting, causing the policy to sharpen prematurely and suppress exploration. To address these issues, we propose Entropy Confidence Hybrid Group Relative Policy Optimization (ECHO). During rollout, ECHO jointly leverages local entropy and group level confidence to adaptively control branch width, and further introduces online confidence-based pruning to terminate persistently low confidence branches, avoiding high entropy traps and mitigating collapse. During policy updates, ECHO employs confidence adaptive clipping and an entropy confidence hybrid advantage shaping approach to enhance training robustness and mitigate early stage bias. Experiments demonstrate that ECHO achieves consistent gains on multiple mathematical and visual reasoning benchmarks, and generalizes more effectively under a limited rollout budget.

  </details>



- **No Global Plan in Chain-of-Thought: Uncover the Latent Planning Horizon of LLMs**  
  Liyan Xu, Mo Yu, Fandong Meng, Jie Zhou  
  _2026-02-02_ · https://arxiv.org/abs/2602.02103v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  This work stems from prior complementary observations on the dynamics of Chain-of-Thought (CoT): Large Language Models (LLMs) is shown latent planning of subsequent reasoning prior to CoT emergence, thereby diminishing the significance of explicit CoT; whereas CoT remains critical for tasks requiring multi-step reasoning. To deepen the understanding between LLM's internal states and its verbalized reasoning trajectories, we investigate the latent planning strength of LLMs, through our probing method, Tele-Lens, applying to hidden states across diverse task domains. Our empirical results indicate that LLMs exhibit a myopic horizon, primarily conducting incremental transitions without precise global planning. Leveraging this characteristic, we propose a hypothesis on enhancing uncertainty estimation of CoT, which we validate that a small subset of CoT positions can effectively represent the uncertainty of the entire path. We further underscore the significance of exploiting CoT dynamics, and demonstrate that automatic recognition of CoT bypass can be achieved without performance degradation. Our code, data and models are released at https://github.com/lxucs/tele-lens.

  </details>


