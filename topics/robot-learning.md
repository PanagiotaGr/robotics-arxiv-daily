# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-01-24 06:46 UTC_

Total papers shown: **15**


---

- **Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning**  
  Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al.  
  _2026-01-22_ · https://arxiv.org/abs/2601.16163v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Recent video generation models demonstrate remarkable ability to capture complex physical interactions and scene evolution over time. To leverage their spatiotemporal priors, robotics works have adapted video models for policy learning but introduce complexity by requiring multiple stages of post-training and new architectural components for action generation. In this work, we introduce Cosmos Policy, a simple approach for adapting a large pretrained video model (Cosmos-Predict2) into an effective robot policy through a single stage of post-training on the robot demonstration data collected on the target platform, with no architectural modifications. Cosmos Policy learns to directly generate robot actions encoded as latent frames within the video model's latent diffusion process, harnessing the model's pretrained priors and core learning algorithm to capture complex action distributions. Additionally, Cosmos Policy generates future state images and values (expected cumulative rewards), which are similarly encoded as latent frames, enabling test-time planning of action trajectories with higher likelihood of success. In our evaluations, Cosmos Policy achieves state-of-the-art performance on the LIBERO and RoboCasa simulation benchmarks (98.5% and 67.1% average success rates, respectively) and the highest average score in challenging real-world bimanual manipulation tasks, outperforming strong diffusion policies trained from scratch, video model-based policies, and state-of-the-art vision-language-action models fine-tuned on the same robot demonstrations. Furthermore, given policy rollout data, Cosmos Policy can learn from experience to refine its world model and value function and leverage model-based planning to achieve even higher success rates in challenging tasks. We release code, models, and training data at https://research.nvidia.com/labs/dir/cosmos-policy/

  </details>



- **Keyframe-Based Feed-Forward Visual Odometry**  
  Weichen Dai, Wenhan Su, Da Kong, Yuhang Ming, Wanzeng Kong  
  _2026-01-22_ · https://arxiv.org/abs/2601.16020v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The emergence of visual foundation models has revolutionized visual odometry~(VO) and SLAM, enabling pose estimation and dense reconstruction within a single feed-forward network. However, unlike traditional pipelines that leverage keyframe methods to enhance efficiency and accuracy, current foundation model based methods, such as VGGT-Long, typically process raw image sequences indiscriminately. This leads to computational redundancy and degraded performance caused by low inter-frame parallax, which provides limited contextual stereo information. Integrating traditional geometric heuristics into these methods is non-trivial, as their performance depends on high-dimensional latent representations rather than explicit geometric metrics. To bridge this gap, we propose a novel keyframe-based feed-forward VO. Instead of relying on hand-crafted rules, our approach employs reinforcement learning to derive an adaptive keyframe policy in a data-driven manner, aligning selection with the intrinsic characteristics of the underlying foundation model. We train our agent on TartanAir dataset and conduct extensive evaluations across several real-world datasets. Experimental results demonstrate that the proposed method achieves consistent and substantial improvements over state-of-the-art feed-forward VO methods.

  </details>



- **IVRA: Improving Visual-Token Relations for Robot Action Policy with Training-Free Hint-Based Guidance**  
  Jongwoo Park, Kanchana Ranasinghe, Jinhyeok Jang, Cristina Mata, Yoo Sung Jang, Michael S Ryoo  
  _2026-01-22_ · https://arxiv.org/abs/2601.16207v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many Vision-Language-Action (VLA) models flatten image patches into a 1D token sequence, weakening the 2D spatial cues needed for precise manipulation. We introduce IVRA, a lightweight, training-free method that improves spatial understanding by exploiting affinity hints already available in the model's built-in vision encoder, without requiring any external encoder or retraining. IVRA selectively injects these affinity signals into a language-model layer in which instance-level features reside. This inference-time intervention realigns visual-token interactions and better preserves geometric structure while keeping all model parameters fixed. We demonstrate the generality of IVRA by applying it to diverse VLA architectures (LLaRA, OpenVLA, and FLOWER) across simulated benchmarks spanning both 2D and 3D manipulation (VIMA and LIBERO) and on various real-robot tasks. On 2D VIMA, IVRA improves average success by +4.2% over the baseline LLaRA in a low-data regime. On 3D LIBERO, it yields consistent gains over the OpenVLA and FLOWER baselines, including improvements when baseline accuracy is near saturation (96.3% to 97.1%). All code and models will be released publicly. Visualizations are available at: jongwoopark7978.github.io/IVRA

  </details>



- **Off-Policy Actor-Critic with Sigmoid-Bounded Entropy for Real-World Robot Learning**  
  Xiefeng Wu, Mingyu Hu, Shu Zhang  
  _2026-01-22_ · https://arxiv.org/abs/2601.15761v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Deploying reinforcement learning in the real world remains challenging due to sample inefficiency, sparse rewards, and noisy visual observations. Prior work leverages demonstrations and human feedback to improve learning efficiency and robustness. However, offline-to-online methods need large datasets and can be unstable, while VLA-assisted RL relies on large-scale pretraining and fine-tuning. As a result, a low-cost real-world RL method with minimal data requirements has yet to emerge. We introduce \textbf{SigEnt-SAC}, an off-policy actor-critic method that learns from scratch using a single expert trajectory. Our key design is a sigmoid-bounded entropy term that prevents negative-entropy-driven optimization toward out-of-distribution actions and reduces Q-function oscillations. We benchmark SigEnt-SAC on D4RL tasks against representative baselines. Experiments show that SigEnt-SAC substantially alleviates Q-function oscillations and reaches a 100\% success rate faster than prior methods. Finally, we validate SigEnt-SAC on four real-world robotic tasks across multiple embodiments, where agents learn from raw images and sparse rewards; results demonstrate that SigEnt-SAC can learn successful policies with only a small number of real-world interactions, suggesting a low-cost and practical pathway for real-world RL deployment.

  </details>



- **PUMA: Perception-driven Unified Foothold Prior for Mobility Augmented Quadruped Parkour**  
  Liang Wang, Kanzhong Yao, Yang Liu, Weikai Qin, Jun Wu, Zhe Sun, Qiuguo Zhu  
  _2026-01-22_ · https://arxiv.org/abs/2601.15995v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Parkour tasks for quadrupeds have emerged as a promising benchmark for agile locomotion. While human athletes can effectively perceive environmental characteristics to select appropriate footholds for obstacle traversal, endowing legged robots with similar perceptual reasoning remains a significant challenge. Existing methods often rely on hierarchical controllers that follow pre-computed footholds, thereby constraining the robot's real-time adaptability and the exploratory potential of reinforcement learning. To overcome these challenges, we present PUMA, an end-to-end learning framework that integrates visual perception and foothold priors into a single-stage training process. This approach leverages terrain features to estimate egocentric polar foothold priors, composed of relative distance and heading, guiding the robot in active posture adaptation for parkour tasks. Extensive experiments conducted in simulation and real-world environments across various discrete complex terrains, demonstrate PUMA's exceptional agility and robustness in challenging scenarios.

  </details>



- **TeNet: Text-to-Network for Compact Policy Synthesis**  
  Ariyan Bighashdel, Kevin Sebastian Luck  
  _2026-01-22_ · https://arxiv.org/abs/2601.15912v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots that follow natural-language instructions often either plan at a high level using hand-designed interfaces or rely on large end-to-end models that are difficult to deploy for real-time control. We propose TeNet (Text-to-Network), a framework for instantiating compact, task-specific robot policies directly from natural language descriptions. TeNet conditions a hypernetwork on text embeddings produced by a pretrained large language model (LLM) to generate a fully executable policy, which then operates solely on low-dimensional state inputs at high control frequencies. By using the language only once at the policy instantiation time, TeNet inherits the general knowledge and paraphrasing robustness of pretrained LLMs while remaining lightweight and efficient at execution time. To improve generalization, we optionally ground language in behavior during training by aligning text embeddings with demonstrated actions, while requiring no demonstrations at inference time. Experiments on MuJoCo and Meta-World benchmarks show that TeNet produces policies that are orders of magnitude smaller than sequence-based baselines, while achieving strong performance in both multi-task and meta-learning settings and supporting high-frequency control. These results show that text-conditioned hypernetworks offer a practical way to build compact, language-driven controllers for ressource-constrained robot control tasks with real-time requirements.

  </details>



- **Point Bridge: 3D Representations for Cross Domain Policy Learning**  
  Siddhant Haldar, Lars Johannsmeier, Lerrel Pinto, Abhishek Gupta, Dieter Fox, Yashraj Narang, Ajay Mandlekar  
  _2026-01-22_ · https://arxiv.org/abs/2601.16212v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robot foundation models are beginning to deliver on the promise of generalist robotic agents, yet progress remains constrained by the scarcity of large-scale real-world manipulation datasets. Simulation and synthetic data generation offer a scalable alternative, but their usefulness is limited by the visual domain gap between simulation and reality. In this work, we present Point Bridge, a framework that leverages unified, domain-agnostic point-based representations to unlock synthetic datasets for zero-shot sim-to-real policy transfer, without explicit visual or object-level alignment. Point Bridge combines automated point-based representation extraction via Vision-Language Models (VLMs), transformer-based policy learning, and efficient inference-time pipelines to train capable real-world manipulation agents using only synthetic data. With additional co-training on small sets of real demonstrations, Point Bridge further improves performance, substantially outperforming prior vision-based sim-and-real co-training methods. It achieves up to 44% gains in zero-shot sim-to-real transfer and up to 66% with limited real data across both single-task and multitask settings. Videos of the robot are best viewed at: https://pointbridge3d.github.io/

  </details>



- **Efficiently Learning Robust Torque-based Locomotion Through Reinforcement with Model-Based Supervision**  
  Yashuai Yan, Tobias Egle, Christian Ott, Dongheui Lee  
  _2026-01-22_ · https://arxiv.org/abs/2601.16109v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We propose a control framework that integrates model-based bipedal locomotion with residual reinforcement learning (RL) to achieve robust and adaptive walking in the presence of real-world uncertainties. Our approach leverages a model-based controller, comprising a Divergent Component of Motion (DCM) trajectory planner and a whole-body controller, as a reliable base policy. To address the uncertainties of inaccurate dynamics modeling and sensor noise, we introduce a residual policy trained through RL with domain randomization. Crucially, we employ a model-based oracle policy, which has privileged access to ground-truth dynamics during training, to supervise the residual policy via a novel supervised loss. This supervision enables the policy to efficiently learn corrective behaviors that compensate for unmodeled effects without extensive reward shaping. Our method demonstrates improved robustness and generalization across a range of randomized conditions, offering a scalable solution for sim-to-real transfer in bipedal locomotion.

  </details>



- **DTP: A Simple yet Effective Distracting Token Pruning Framework for Vision-Language Action Models**  
  Chenyang Li, Jieyuan Liu, Bin Li, Bo Gao, Yilin Yuan, Yangfan He, Yuchen Li, Jingqun Tang  
  _2026-01-22_ · https://arxiv.org/abs/2601.16065v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language Action (VLA) models have shown remarkable progress in robotic manipulation by leveraging the powerful perception abilities of Vision-Language Models (VLMs) to understand environments and directly output actions. However, by default, VLA models may overly attend to image tokens in the task-irrelevant region, which we describe as 'distracting tokens'. This behavior can disturb the model from the generation of the desired action tokens in each step, affecting the success rate of tasks. In this paper, we introduce a simple yet effective plug-and-play Distracting Token Pruning (DTP) framework, which dynamically detects and prunes these distracting image tokens. By correcting the model's visual attention patterns, we aim to improve the task success rate, as well as exploring the performance upper boundaries of the model without altering its original architecture or adding additional inputs. Experiments on the SIMPLER Benchmark (Li et al., 2024) show that our method consistently achieving relative improvements in task success rates across different types of novel VLA models, demonstrating generalizability to transformer-based VLAs. Further analysis reveals a negative correlation between the task success rate and the amount of attentions in the task-irrelevant region for all models tested, highlighting a common phenomenon of VLA models that could guide future research. We also publish our code at: https://anonymous.4open.science/r/CBD3.

  </details>



- **Grounding Large Language Models in Reaction Knowledge Graphs for Synthesis Retrieval**  
  Olga Bunkova, Lorenzo Di Fruscia, Sophia Rupprecht, Artur M. Schweidtmann, Marcel J. T. Reinders, Jana M. Weber  
  _2026-01-22_ · https://arxiv.org/abs/2601.16038v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) can aid synthesis planning in chemistry, but standard prompting methods often yield hallucinated or outdated suggestions. We study LLM interactions with a reaction knowledge graph by casting reaction path retrieval as a Text2Cypher (natural language to graph query) generation problem, and define single- and multi-step retrieval tasks. We compare zero-shot prompting to one-shot variants using static, random, and embedding-based exemplar selection, and assess a checklist-driven validator/corrector loop. To evaluate our framework, we consider query validity and retrieval accuracy. We find that one-shot prompting with aligned exemplars consistently performs best. Our checklist-style self-correction loop mainly improves executability in zero-shot settings and offers limited additional retrieval gains once a good exemplar is present. We provide a reproducible Text2Cypher evaluation setup to facilitate further work on KG-grounded LLMs for synthesis planning. Code is available at https://github.com/Intelligent-molecular-systems/KG-LLM-Synthesis-Retrieval.

  </details>



- **Introducing the Generative Application Firewall (GAF)**  
  Joan Vendrell Farreny, Martí Jordà Roca, Miquel Cornudella Gaya, Rodrigo Fernández Baón, Víctor García Martínez, Eduard Camacho Sucarrat, Alessandro Pignati  
  _2026-01-22_ · https://arxiv.org/abs/2601.15824v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  This paper introduces the Generative Application Firewall (GAF), a new architectural layer for securing LLM applications. Existing defenses -- prompt filters, guardrails, and data-masking -- remain fragmented; GAF unifies them into a single enforcement point, much like a WAF coordinates defenses for web traffic, while also covering autonomous agents and their tool interactions.

  </details>



- **Virtual Traffic Police: Large Language Model-Augmented Traffic Signal Control for Unforeseen Incidents**  
  Shiqi Wei, Qiqing Wang, Kaidi Yang  
  _2026-01-22_ · https://arxiv.org/abs/2601.15816v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Adaptive traffic signal control (TSC) has demonstrated strong effectiveness in managing dynamic traffic flows. However, conventional methods often struggle when unforeseen traffic incidents occur (e.g., accidents and road maintenance), which typically require labor-intensive and inefficient manual interventions by traffic police officers. Large Language Models (LLMs) appear to be a promising solution thanks to their remarkable reasoning and generalization capabilities. Nevertheless, existing works often propose to replace existing TSC systems with LLM-based systems, which can be (i) unreliable due to the inherent hallucinations of LLMs and (ii) costly due to the need for system replacement. To address the issues of existing works, we propose a hierarchical framework that augments existing TSC systems with LLMs, whereby a virtual traffic police agent at the upper level dynamically fine-tunes selected parameters of signal controllers at the lower level in response to real-time traffic incidents. To enhance domain-specific reliability in response to unforeseen traffic incidents, we devise a self-refined traffic language retrieval system (TLRS), whereby retrieval-augmented generation is employed to draw knowledge from a tailored traffic language database that encompasses traffic conditions and controller operation principles. Moreover, we devise an LLM-based verifier to update the TLRS continuously over the reasoning process. Our results show that LLMs can serve as trustworthy virtual traffic police officers that can adapt conventional TSC methods to unforeseen traffic incidents with significantly improved operational efficiency and reliability.

  </details>



- **VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning**  
  Chenglin Li, Qianglong Chen, Feng Han, Yikun Wang, Xingxi Yin, Yan Gong, Ruilin Li, Yin Zhang, Jiaqi Wang  
  _2026-01-22_ · https://arxiv.org/abs/2601.15724v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Long-form video understanding remains a fundamental challenge for current Video Large Language Models. Most existing models rely on static reasoning over uniformly sampled frames, which weakens temporal localization and leads to substantial information loss in long videos. Agentic tools such as temporal retrieval, spatial zoom, and temporal zoom offer a natural way to overcome these limitations by enabling adaptive exploration of key moments. However, constructing agentic video understanding data requires models that already possess strong long-form video comprehension, creating a circular dependency. We address this challenge with VideoThinker, an agentic Video Large Language Model trained entirely on synthetic tool interaction trajectories. Our key idea is to convert videos into rich captions and employ a powerful agentic language model to generate multi-step tool use sequences in caption space. These trajectories are subsequently grounded back to video by replacing captions with the corresponding frames, yielding a large-scale interleaved video and tool reasoning dataset without requiring any long-form understanding from the underlying model. Training on this synthetic agentic dataset equips VideoThinker with dynamic reasoning capabilities, adaptive temporal exploration, and multi-step tool use. Remarkably, VideoThinker significantly outperforms both caption-only language model agents and strong video model baselines across long-video benchmarks, demonstrating the effectiveness of tool augmented synthetic data and adaptive retrieval and zoom reasoning for long-form video understanding.

  </details>



- **Dancing in Chains: Strategic Persuasion in Academic Rebuttal via Theory of Mind**  
  Zhitao He, Zongwei Lyu, Yi R Fung  
  _2026-01-22_ · https://arxiv.org/abs/2601.15715v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Although artificial intelligence (AI) has become deeply integrated into various stages of the research workflow and achieved remarkable advancements, academic rebuttal remains a significant and underexplored challenge. This is because rebuttal is a complex process of strategic communication under severe information asymmetry rather than a simple technical debate. Consequently, current approaches struggle as they largely imitate surface-level linguistics, missing the essential element of perspective-taking required for effective persuasion. In this paper, we introduce RebuttalAgent, the first framework to ground academic rebuttal in Theory of Mind (ToM), operationalized through a ToM-Strategy-Response (TSR) pipeline that models reviewer mental state, formulates persuasion strategy, and generates strategy-grounded response. To train our agent, we construct RebuttalBench, a large-scale dataset synthesized via a novel critique-and-refine approach. Our training process consists of two stages, beginning with a supervised fine-tuning phase to equip the agent with ToM-based analysis and strategic planning capabilities, followed by a reinforcement learning phase leveraging the self-reward mechanism for scalable self-improvement. For reliable and efficient automated evaluation, we further develop Rebuttal-RM, a specialized evaluator trained on over 100K samples of multi-source rebuttal data, which achieves scoring consistency with human preferences surpassing powerful judge GPT-4.1. Extensive experiments show RebuttalAgent significantly outperforms the base model by an average of 18.3% on automated metrics, while also outperforming advanced proprietary models across both automated and human evaluations. Disclaimer: the generated rebuttal content is for reference only to inspire authors and assist in drafting. It is not intended to replace the author's own critical analysis and response.

  </details>



- **D-Optimality-Guided Reinforcement Learning for Efficient Open-Loop Calibration of a 3-DOF Ankle Rehabilitation Robot**  
  Qifan Hu, Branko Celler, Weidong Mu, Steven W. Su  
  _2026-01-22_ · https://arxiv.org/abs/2601.15707v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate alignment of multi-degree-of-freedom rehabilitation robots is essential for safe and effective patient training. This paper proposes a two-stage calibration framework for a self-designed three-degree-of-freedom (3-DOF) ankle rehabilitation robot. First, a Kronecker-product-based open-loop calibration method is developed to cast the input-output alignment into a linear parameter identification problem, which in turn defines the associated experimental design objective through the resulting information matrix. Building on this formulation, calibration posture selection is posed as a combinatorial design-of-experiments problem guided by a D-optimality criterion, i.e., selecting a small subset of postures that maximises the determinant of the information matrix. To enable practical selection under constraints, a Proximal Policy Optimization (PPO) agent is trained in simulation to choose 4 informative postures from a candidate set of 50. Across simulation and real-robot evaluations, the learned policy consistently yields substantially more informative posture combinations than random selection: the mean determinant of the information matrix achieved by PPO is reported to be more than two orders of magnitude higher with reduced variance. In addition, real-world results indicate that a parameter vector identified from only four D-optimality-guided postures provides stronger cross-episode prediction consistency than estimates obtained from a larger but unstructured set of 50 postures. The proposed framework therefore improves calibration efficiency while maintaining robust parameter estimation, offering practical guidance for high-precision alignment of multi-DOF rehabilitation robots.

  </details>


