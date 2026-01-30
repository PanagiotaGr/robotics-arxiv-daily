# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-01-30 07:03 UTC_

Total papers shown: **20**


---

- **MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts**  
  Lorenzo Mazza, Ariel Rodriguez, Rayan Younis, Martin Lelis, Ortrun Hellig, Chenpan Li, Sebastian Bodenstedt, Martin Wagner, Stefanie Speidel  
  _2026-01-29_ · https://arxiv.org/abs/2601.21971v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Imitation learning has achieved remarkable success in robotic manipulation, yet its application to surgical robotics remains challenging due to data scarcity, constrained workspaces, and the need for an exceptional level of safety and predictability. We present a supervised Mixture-of-Experts (MoE) architecture designed for phase-structured surgical manipulation tasks, which can be added on top of any autonomous policy. Unlike prior surgical robot learning approaches that rely on multi-camera setups or thousands of demonstrations, we show that a lightweight action decoder policy like Action Chunking Transformer (ACT) can learn complex, long-horizon manipulation from less than 150 demonstrations using solely stereo endoscopic images, when equipped with our architecture. We evaluate our approach on the collaborative surgical task of bowel grasping and retraction, where a robot assistant interprets visual cues from a human surgeon, executes targeted grasping on deformable tissue, and performs sustained retraction. We benchmark our method against state-of-the-art Vision-Language-Action (VLA) models and the standard ACT baseline. Our results show that generalist VLAs fail to acquire the task entirely, even under standard in-distribution conditions. Furthermore, while standard ACT achieves moderate success in-distribution, adopting a supervised MoE architecture significantly boosts its performance, yielding higher success rates in-distribution and demonstrating superior robustness in out-of-distribution scenarios, including novel grasp locations, reduced illumination, and partial occlusions. Notably, it generalizes to unseen testing viewpoints and also transfers zero-shot to ex vivo porcine tissue without additional training, offering a promising pathway toward in vivo deployment. To support this, we present qualitative preliminary results of policy roll-outs during in vivo porcine surgery.

  </details>



- **LLM-Driven Scenario-Aware Planning for Autonomous Driving**  
  He Li, Zhaowei Chen, Rui Gao, Guoliang Li, Qi Hao, Shuai Wang, Chengzhong Xu  
  _2026-01-29_ · https://arxiv.org/abs/2601.21876v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Hybrid planner switching framework (HPSF) for autonomous driving needs to reconcile high-speed driving efficiency with safe maneuvering in dense traffic. Existing HPSF methods often fail to make reliable mode transitions or sustain efficient driving in congested environments, owing to heuristic scene recognition and low-frequency control updates. To address the limitation, this paper proposes LAP, a large language model (LLM) driven, adaptive planning method, which switches between high-speed driving in low-complexity scenes and precise driving in high-complexity scenes, enabling high qualities of trajectory generation through confined gaps. This is achieved by leveraging LLM for scene understanding and integrating its inference into the joint optimization of mode configuration and motion planning. The joint optimization is solved using tree-search model predictive control and alternating minimization. We implement LAP by Python in Robot Operating System (ROS). High-fidelity simulation results show that the proposed LAP outperforms other benchmarks in terms of both driving time and success rate.

  </details>



- **DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation**  
  Haozhe Xie, Beichen Wen, Jiarui Zheng, Zhaoxi Chen, Fangzhou Hong, Haiwen Diao, Ziwei Liu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22153v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Manipulating dynamic objects remains an open challenge for Vision-Language-Action (VLA) models, which, despite strong generalization in static manipulation, struggle in dynamic scenarios requiring rapid perception, temporal anticipation, and continuous control. We present DynamicVLA, a framework for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptation through three key designs: 1) a compact 0.4B VLA using a convolutional vision encoder for spatially efficient, structurally faithful encoding, enabling fast multimodal inference; 2) Continuous Inference, enabling overlapping reasoning and execution for lower latency and timely adaptation to object motion; and 3) Latent-aware Action Streaming, which bridges the perception-execution gap by enforcing temporally aligned action execution. To fill the missing foundation of dynamic manipulation data, we introduce the Dynamic Object Manipulation (DOM) benchmark, built from scratch with an auto data collection pipeline that efficiently gathers 200K synthetic episodes across 2.8K scenes and 206 objects, and enables fast collection of 2K real-world episodes without teleoperation. Extensive evaluations demonstrate remarkable improvements in response speed, perception, and generalization, positioning DynamicVLA as a unified framework for general dynamic object manipulation across embodiments.

  </details>



- **MetricAnything: Scaling Metric Depth Pretraining with Noisy Heterogeneous Sources**  
  Baorui Ma, Jiahui Yang, Donglin Di, Xuancheng Zhang, Jianxun Cui, Hao Li, Yan Xie, Wei Chen  
  _2026-01-29_ · https://arxiv.org/abs/2601.22054v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Scaling has powered recent advances in vision foundation models, yet extending this paradigm to metric depth estimation remains challenging due to heterogeneous sensor noise, camera-dependent biases, and metric ambiguity in noisy cross-source 3D data. We introduce Metric Anything, a simple and scalable pretraining framework that learns metric depth from noisy, diverse 3D sources without manually engineered prompts, camera-specific modeling, or task-specific architectures. Central to our approach is the Sparse Metric Prompt, created by randomly masking depth maps, which serves as a universal interface that decouples spatial reasoning from sensor and camera biases. Using about 20M image-depth pairs spanning reconstructed, captured, and rendered 3D data across 10000 camera models, we demonstrate-for the first time-a clear scaling trend in the metric depth track. The pretrained model excels at prompt-driven tasks such as depth completion, super-resolution and Radar-camera fusion, while its distilled prompt-free student achieves state-of-the-art results on monocular depth estimation, camera intrinsics recovery, single/multi-view metric 3D reconstruction, and VLA planning. We also show that using pretrained ViT of Metric Anything as a visual encoder significantly boosts Multimodal Large Language Model capabilities in spatial intelligence. These results show that metric depth estimation can benefit from the same scaling laws that drive modern foundation models, establishing a new path toward scalable and efficient real-world metric perception. We open-source MetricAnything at http://metric-anything.github.io/metric-anything-io/ to support community research.

  </details>



- **Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding**  
  Yifan Zhu, Huiqiang Rong, Haoran Luo  
  _2026-01-29_ · https://arxiv.org/abs/2601.21969v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) often hallucinate, generating content inconsistent with the input. Retrieval-Augmented Generation (RAG) and Reinforcement Learning with Human Feedback (RLHF) can mitigate hallucinations but require resource-intensive retrieval or large-scale fine-tuning. Decoding-based methods are lighter yet lack explicit hallucination control. To address this, we present Token-Guard, a token-level hallucination control method based on self-checking decoding. Token-Guard performs internal verification at each reasoning step to detect hallucinated tokens before they propagate. Candidate fragments are further evaluated in a latent space with explicit hallucination risk scoring, while iterative pruning and regeneration dynamically correct detected errors. Experiments on HALU datasets show Token-Guard substantially reduces hallucinations and improves generation accuracy, offering a scalable, modular solution for reliable LLM outputs. Our code is publicly available.

  </details>



- **CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty**  
  Johannes Kirmayr, Lukas Stappen, Elisabeth André  
  _2026-01-29_ · https://arxiv.org/abs/2601.22027v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Existing benchmarks for Large Language Model (LLM) agents focus on task completion under idealistic settings but overlook reliability in real-world, user-facing applications. In domains, such as in-car voice assistants, users often issue incomplete or ambiguous requests, creating intrinsic uncertainty that agents must manage through dialogue, tool use, and policy adherence. We introduce CAR-bench, a benchmark for evaluating consistency, uncertainty handling, and capability awareness in multi-turn, tool-using LLM agents in an in-car assistant domain. The environment features an LLM-simulated user, domain policies, and 58 interconnected tools spanning navigation, productivity, charging, and vehicle control. Beyond standard task completion, CAR-bench introduces Hallucination tasks that test agents' limit-awareness under missing tools or information, and Disambiguation tasks that require resolving uncertainty through clarification or internal information gathering. Baseline results reveal large gaps between occasional and consistent success on all task types. Even frontier reasoning LLMs achieve less than 50% consistent pass rate on Disambiguation tasks due to premature actions, and frequently violate policies or fabricate information to satisfy user requests in Hallucination tasks, underscoring the need for more reliable and self-aware LLM agents in real-world settings.

  </details>



- **DynaWeb: Model-Based Reinforcement Learning of Web Agents**  
  Hang Ding, Peidong Liu, Junqiao Wang, Ziwei Ji, Meng Cao, Rongzhao Zhang, Lynn Ai, Eric Yang, Tianyu Shi, Lei Yu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22149v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  The development of autonomous web agents, powered by Large Language Models (LLMs) and reinforcement learning (RL), represents a significant step towards general-purpose AI assistants. However, training these agents is severely hampered by the challenges of interacting with the live internet, which is inefficient, costly, and fraught with risks. Model-based reinforcement learning (MBRL) offers a promising solution by learning a world model of the environment to enable simulated interaction. This paper introduces DynaWeb, a novel MBRL framework that trains web agents through interacting with a web world model trained to predict naturalistic web page representations given agent actions. This model serves as a synthetic web environment where an agent policy can dream by generating vast quantities of rollout action trajectories for efficient online reinforcement learning. Beyond free policy rollouts, DynaWeb incorporates real expert trajectories from training data, which are randomly interleaved with on-policy rollouts during training to improve stability and sample efficiency. Experiments conducted on the challenging WebArena and WebVoyager benchmarks demonstrate that DynaWeb consistently and significantly improves the performance of state-of-the-art open-source web agent models. Our findings establish the viability of training web agents through imagination, offering a scalable and efficient way to scale up online agentic RL.

  </details>



- **Pay for Hints, Not Answers: LLM Shepherding for Cost-Efficient Inference**  
  Ziming Dong, Hardik Sharma, Evan O'Toole, Jaya Prakash Champati, Kui Wu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22132v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) deliver state-of-the-art performance on complex reasoning tasks, but their inference costs limit deployment at scale. Small Language Models (SLMs) offer dramatic cost savings yet lag substantially in accuracy. Existing approaches - routing and cascading - treat the LLM as an all-or-nothing resource: either the query bypasses the LLM entirely, or the LLM generates a complete response at full cost. We introduce LLM Shepherding, a framework that requests only a short prefix (a hint) from the LLM and provides it to SLM. This simple mechanism is surprisingly effective for math and coding tasks: even hints comprising 10-30% of the full LLM response improve SLM accuracy significantly. Shepherding generalizes both routing and cascading, and it achieves lower cost under oracle decision-making. We develop a two-stage predictor that jointly determines whether a hint is needed and how many tokens to request. On the widely-used mathematical reasoning (GSM8K, CNK12) and code generation (HumanEval, MBPP) benchmarks, Shepherding reduces costs by 42-94% relative to LLM-only inference. Compared to state-of-the-art routing and cascading baselines, shepherding delivers up to 2.8x cost reduction while matching accuracy. To our knowledge, this is the first work to exploit token-level budget control for SLM-LLM collaboration.

  </details>



- **SIA: Symbolic Interpretability for Anticipatory Deep Reinforcement Learning in Network Control**  
  MohammadErfan Jabbari, Abhishek Duttagupta, Claudio Fiandrino, Leonardo Bonati, Salvatore D'Oro, Michele Polese, Marco Fiore, Tommaso Melodia  
  _2026-01-29_ · https://arxiv.org/abs/2601.22044v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  Deep reinforcement learning (DRL) promises adaptive control for future mobile networks but conventional agents remain reactive: they act on past and current measurements and cannot leverage short-term forecasts of exogenous KPIs such as bandwidth. Augmenting agents with predictions can overcome this temporal myopia, yet uptake in networking is scarce because forecast-aware agents act as closed-boxes; operators cannot tell whether predictions guide decisions or justify the added complexity. We propose SIA, the first interpreter that exposes in real time how forecast-augmented DRL agents operate. SIA fuses Symbolic AI abstractions with per-KPI Knowledge Graphs to produce explanations, and includes a new Influence Score metric. SIA achieves sub-millisecond speed, over 200x faster than existing XAI methods. We evaluate SIA on three diverse networking use cases, uncovering hidden issues, including temporal misalignment in forecast integration and reward-design biases that trigger counter-productive policies. These insights enable targeted fixes: a redesigned agent achieves a 9% higher average bitrate in video streaming, and SIA's online Action-Refinement module improves RAN-slicing reward by 25% without retraining. By making anticipatory DRL transparent and tunable, SIA lowers the barrier to proactive control in next-generation mobile networks.

  </details>



- **SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks**  
  Abhishek Duttagupta, MohammadErfan Jabbari, Claudio Fiandrino, Marco Fiore, Joerg Widmer  
  _2026-01-29_ · https://arxiv.org/abs/2601.22024v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  The operation of future 6th-generation (6G) mobile networks will increasingly rely on the ability of deep reinforcement learning (DRL) to optimize network decisions in real-time. DRL yields demonstrated efficacy in various resource allocation problems, such as joint decisions on user scheduling and antenna allocation or simultaneous control of computing resources and modulation. However, trained DRL agents are closed-boxes and inherently difficult to explain, which hinders their adoption in production settings. In this paper, we make a step towards removing this critical barrier by presenting SymbXRL, a novel technique for explainable reinforcement learning (XRL) that synthesizes human-interpretable explanations for DRL agents. SymbXRL leverages symbolic AI to produce explanations where key concepts and their relationships are described via intuitive symbols and rules; coupling such a representation with logical reasoning exposes the decision process of DRL agents and offers more comprehensible descriptions of their behaviors compared to existing approaches. We validate SymbXRL in practical network management use cases supported by DRL, proving that it not only improves the semantics of the explanations but also paves the way for explicit agent control: for instance, it enables intent-based programmatic action steering that improves by 12% the median cumulative reward over a pure DRL solution.

  </details>



- **Geometry of Drifting MDPs with Path-Integral Stability Certificates**  
  Zuyuan Zhang, Mahdi Imani, Tian Lan  
  _2026-01-29_ · https://arxiv.org/abs/2601.21991v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Real-world reinforcement learning is often \emph{nonstationary}: rewards and dynamics drift, accelerate, oscillate, and trigger abrupt switches in the optimal action. Existing theory often represents nonstationarity with coarse-scale models that measure \emph{how much} the environment changes, not \emph{how} it changes locally -- even though acceleration and near-ties drive tracking error and policy chattering. We take a geometric view of nonstationary discounted Markov Decision Processes (MDPs) by modeling the environment as a differentiable homotopy path and tracking the induced motion of the optimal Bellman fixed point. This yields a length-curvature-kink signature of intrinsic complexity: cumulative drift, acceleration/oscillation, and action-gap-induced nonsmoothness. We prove a solver-agnostic path-integral stability bound and derive gap-safe feasible regions that certify local stability away from switch regimes. Building on these results, we introduce \textit{Homotopy-Tracking RL (HT-RL)} and \textit{HT-MCTS}, lightweight wrappers that estimate replay-based proxies of length, curvature, and near-tie proximity online and adapt learning or planning intensity accordingly. Experiments show improved tracking and dynamic regret over matched static baselines, with the largest gains in oscillatory and switch-prone regimes.

  </details>



- **Industrialized Deception: The Collateral Effects of LLM-Generated Misinformation on Digital Ecosystems**  
  Alexander Loth, Martin Kappes, Marc-Oliver Pahl  
  _2026-01-29_ · https://arxiv.org/abs/2601.21963v1 · `cs.CY`  
  <details><summary>Abstract</summary>

  Generative AI and misinformation research has evolved since our 2024 survey. This paper presents an updated perspective, transitioning from literature review to practical countermeasures. We report on changes in the threat landscape, including improved AI-generated content through Large Language Models (LLMs) and multimodal systems. Central to this work are our practical contributions: JudgeGPT, a platform for evaluating human perception of AI-generated news, and RogueGPT, a controlled stimulus generation engine for research. Together, these tools form an experimental pipeline for studying how humans perceive and detect AI-generated misinformation. Our findings show that detection capabilities have improved, but the competition between generation and detection continues. We discuss mitigation strategies including LLM-based detection, inoculation approaches, and the dual-use nature of generative AI. This work contributes to research addressing the adverse impacts of AI on information quality.

  </details>



- **ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models**  
  Bowen Fang, Wen Ye, Yunyue Su, Jinghao Zhang, Qiang Liu, Yesheng Liu, Xin Sun, Shu Wu, Jiabing Yang, Baole Wei, et al.  
  _2026-01-29_ · https://arxiv.org/abs/2601.21947v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Prevalent retrieval-based tool-use pipelines struggle with a dual semantic challenge: their retrievers often employ encoders that fail to capture complex semantics, while the Large Language Model (LLM) itself lacks intrinsic tool knowledge from its natural language pretraining. Generative methods offer a powerful alternative by unifying selection and execution, tasking the LLM to directly learn and generate tool identifiers. However, the common practice of mapping each tool to a unique new token introduces substantial limitations: it creates a scalability and generalization crisis, as the vocabulary size explodes and each tool is assigned a semantically isolated token. This approach also creates a semantic bottleneck that hinders the learning of collaborative tool relationships, as the model must infer them from sparse co-occurrences of monolithic tool IDs within a vast library. To address these limitations, we propose ToolWeaver, a novel generative tool learning framework that encodes tools into hierarchical sequences. This approach makes vocabulary expansion logarithmic to the number of tools. Crucially, it enables the model to learn collaborative patterns from the dense co-occurrence of shared codes, rather than the sparse co-occurrence of monolithic tool IDs. We generate these structured codes through a novel tokenization process designed to weave together a tool's intrinsic semantics with its extrinsic co-usage patterns. These structured codes are then integrated into the LLM through a generative alignment stage, where the model is fine-tuned to produce the hierarchical code sequences. Evaluation results with nearly 47,000 tools show that ToolWeaver significantly outperforms state-of-the-art methods, establishing a more scalable, generalizable, and semantically-aware foundation for advanced tool-augmented agents.

  </details>



- **AgenticSimLaw: A Juvenile Courtroom Multi-Agent Debate Simulation for Explainable High-Stakes Tabular Decision Making**  
  Jon Chun, Kathrine Elkins, Yong Suk Lee  
  _2026-01-29_ · https://arxiv.org/abs/2601.21936v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We introduce AgenticSimLaw, a role-structured, multi-agent debate framework that provides transparent and controllable test-time reasoning for high-stakes tabular decision-making tasks. Unlike black-box approaches, our courtroom-style orchestration explicitly defines agent roles (prosecutor, defense, judge), interaction protocols (7-turn structured debate), and private reasoning strategies, creating a fully auditable decision-making process. We benchmark this framework on young adult recidivism prediction using the NLSY97 dataset, comparing it against traditional chain-of-thought (CoT) prompting across almost 90 unique combinations of models and strategies. Our results demonstrate that structured multi-agent debate provides more stable and generalizable performance compared to single-agent reasoning, with stronger correlation between accuracy and F1-score metrics. Beyond performance improvements, AgenticSimLaw offers fine-grained control over reasoning steps, generates complete interaction transcripts for explainability, and enables systematic profiling of agent behaviors. While we instantiate this framework in the criminal justice domain to stress-test reasoning under ethical complexity, the approach generalizes to any deliberative, high-stakes decision task requiring transparency and human oversight. This work addresses key LLM-based multi-agent system challenges: organization through structured roles, observability through logged interactions, and responsibility through explicit non-deployment constraints for sensitive domains. Data, results, and code will be available on github.com under the MIT license.

  </details>



- **Not All Code Is Equal: A Data-Centric Study of Code Complexity and LLM Reasoning**  
  Lukas Twist, Shu Yang, Hanqi Yan, Jingzhi Gong, Di Wang, Helen Yannakoudakis, Jie M. Zhang  
  _2026-01-29_ · https://arxiv.org/abs/2601.21894v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) increasingly exhibit strong reasoning abilities, often attributed to their capacity to generate chain-of-thought-style intermediate reasoning. Recent work suggests that exposure to code can further enhance these skills, but existing studies largely treat code as a generic training signal, leaving open the question of which properties of code actually contribute to improved reasoning. To address this gap, we study the structural complexity of code, which captures control flow and compositional structure that may shape how models internalise multi-step reasoning during fine-tuning. We examine two complementary settings: solution-driven complexity, where complexity varies across multiple solutions to the same problem, and problem-driven complexity, where complexity reflects variation in the underlying tasks. Using cyclomatic complexity and logical lines of code to construct controlled fine-tuning datasets, we evaluate a range of open-weight LLMs on diverse reasoning benchmarks. Our findings show that although code can improve reasoning, structural properties strongly determine its usefulness. In 83% of experiments, restricting fine-tuning data to a specific structural complexity range outperforms training on structurally diverse code, pointing to a data-centric path for improving reasoning beyond scaling.

  </details>



- **WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents**  
  Yao Zhang, Shijie Tang, Zeyu Li, Zhen Han, Volker Tresp  
  _2026-01-29_ · https://arxiv.org/abs/2601.21872v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Web agents hold great potential for automating complex computer tasks, yet their interactions involve long-horizon, sequential decision-making with irreversible actions. In such settings, outcome-based supervision is sparse and delayed, often rewarding incorrect trajectories and failing to support inference-time scaling. This motivates the use of Process Reward Models (WebPRMs) for web navigation, but existing approaches remain limited: scalar WebPRMs collapse progress into coarse, weakly grounded signals, while checklist-based WebPRMs rely on brittle template matching that fails under layout or semantic changes and often mislabels superficially correct actions as successful, providing little insight or interpretability. To address these challenges, we introduce WebArbiter, a reasoning-first, principle-inducing WebPRM that formulates reward modeling as text generation, producing structured justifications that conclude with a preference verdict and identify the action most conducive to task completion under the current context. Training follows a two-stage pipeline: reasoning distillation equips the model with coherent principle-guided reasoning, and reinforcement learning corrects teacher biases by directly aligning verdicts with correctness, enabling stronger generalization. To support systematic evaluation, we release WebPRMBench, a comprehensive benchmark spanning four diverse web environments with rich tasks and high-quality preference annotations. On WebPRMBench, WebArbiter-7B outperforms the strongest baseline, GPT-5, by 9.1 points. In reward-guided trajectory search on WebArena-Lite, it surpasses the best prior WebPRM by up to 7.2 points, underscoring its robustness and practical value in real-world complex web tasks.

  </details>



- **Constrained Meta Reinforcement Learning with Provable Test-Time Safety**  
  Tingting Ni, Maryam Kamgarpour  
  _2026-01-29_ · https://arxiv.org/abs/2601.21845v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Meta reinforcement learning (RL) allows agents to leverage experience across a distribution of tasks on which the agent can train at will, enabling faster learning of optimal policies on new test tasks. Despite its success in improving sample complexity on test tasks, many real-world applications, such as robotics and healthcare, impose safety constraints during testing. Constrained meta RL provides a promising framework for integrating safety into meta RL. An open question in constrained meta RL is how to ensure the safety of the policy on the real-world test task, while reducing the sample complexity and thus, enabling faster learning of optimal policies. To address this gap, we propose an algorithm that refines policies learned during training, with provable safety and sample complexity guarantees for learning a near optimal policy on the test tasks. We further derive a matching lower bound, showing that this sample complexity is tight.

  </details>



- **CORE:Toward Ubiquitous 6G Intelligence Through Collaborative Orchestration of Large Language Model Agents Over Hierarchical Edge**  
  Zitong Yu, Boquan Sun, Yang Li, Zheyan Qu, Xing Zhang  
  _2026-01-29_ · https://arxiv.org/abs/2601.21822v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Rapid advancements in sixth-generation (6G) networks and large language models (LLMs) have paved the way for ubiquitous intelligence, wherein seamless connectivity and distributed artificial intelligence (AI) have revolutionized various aspects of our lives.However, realizing this vision faces significant challenges owing to the fragmented and heterogeneous computing resources across hierarchical networks, which are insufficient for individual LLM agents to perform complex reasoning tasks.To address this issue, we propose Collaborative Orchestration Role at Edge (CORE), an innovative framework that employs a collaborative learning system in which multiple LLMs, each assigned a distinct functional role, are distributed across mobile devices and tiered edge servers. The system integrates three optimization modules, encompassing real-time perception,dynamic role orchestration, and pipeline-parallel execution, to facilitate efficient and rapid collaboration among distributed agents. Furthermore, we introduce a novel role affinity scheduling algorithm for dynamically orchestrating LLM role assignments across the hierarchical edge infrastructure, intelligently matching computational demands with available dispersed resources.Finally, comprehensive case studies and performance evaluations across various 6G application scenarios demonstrated the efficacy of CORE, revealing significant enhancements in the system efficiency and task completion rates. Building on these promising outcomes, we further validated the practical applicability of CORE by deploying it on a real-world edge-computing platform,that exhibits robust performance in operational environments.

  </details>



- **CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models**  
  Junming Huang, Weiwei Xu  
  _2026-01-29_ · https://arxiv.org/abs/2601.21798v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Large Language Models(LLMs) have revolutionized text generation and multimodal perception, but their capabilities in 3D content generation remain underexplored. Existing methods compromise by producing either low-resolution meshes or coarse structural proxies, failing to capture fine-grained geometry natively. In this paper, we propose CG-MLLM, a novel Multi-modal Large Language Model (MLLM) capable of 3D captioning and high-resolution 3D generation in a single framework. Leveraging the Mixture-of-Transformer architecture, CG-MLLM decouples disparate modeling needs, where the Token-level Autoregressive (TokenAR) Transformer handles token-level content, and the Block-level Autoregressive (BlockAR) Transformer handles block-level content. By integrating a pre-trained vision-language backbone with a specialized 3D VAE latent space, CG-MLLM facilitates long-context interactions between standard tokens and spatial blocks within a single integrated architecture. Experimental results show that CG-MLLM significantly outperforms existing MLLMs in generating high-fidelity 3D objects, effectively bringing high-resolution 3D content creation into the mainstream LLM paradigm.

  </details>



- **Error Amplification Limits ANN-to-SNN Conversion in Continuous Control**  
  Zijie Xu, Zihan Huang, Yiting Dong, Kang Chen, Wenxuan Liu, Zhaofei Yu  
  _2026-01-29_ · https://arxiv.org/abs/2601.21778v1 · `cs.NE`  
  <details><summary>Abstract</summary>

  Spiking Neural Networks (SNNs) can achieve competitive performance by converting already existing well-trained Artificial Neural Networks (ANNs), avoiding further costly training. This property is particularly attractive in Reinforcement Learning (RL), where training through environment interaction is expensive and potentially unsafe. However, existing conversion methods perform poorly in continuous control, where suitable baselines are largely absent. We identify error amplification as the key cause: small action approximation errors become temporally correlated across decision steps, inducing cumulative state distribution shift and severe performance degradation. To address this issue, we propose Cross-Step Residual Potential Initialization (CRPI), a lightweight training-free mechanism that carries over residual membrane potentials across decision steps to suppress temporally correlated errors. Experiments on continuous control benchmarks with both vector and visual observations demonstrate that CRPI can be integrated into existing conversion pipelines and substantially recovers lost performance. Our results highlight continuous control as a critical and challenging benchmark for ANN-to-SNN conversion, where small errors can be strongly amplified and impact performance.

  </details>


