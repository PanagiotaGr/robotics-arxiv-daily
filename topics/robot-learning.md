# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-13 07:12 UTC_

Total papers shown: **17**


---

- **HoloBrain-0 Technical Report**  
  Xuewu Lin, Tianwei Lin, Yun Du, Hongyu Xie, Yiwei Jin, Jiawei Li, Shijie Wu, Qingze Wang, Mengdi Li, Mengao Zhao, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.12062v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In this work, we introduce HoloBrain-0, a comprehensive Vision-Language-Action (VLA) framework that bridges the gap between foundation model research and reliable real-world robot deployment. The core of our system is a novel VLA architecture that explicitly incorporates robot embodiment priors, including multi-view camera parameters and kinematic descriptions (URDF), to enhance 3D spatial reasoning and support diverse embodiments. We validate this design through a scalable ``pre-train then post-train" paradigm, achieving state-of-the-art results on simulation benchmarks such as RoboTwin 2.0, LIBERO, and GenieSim, as well as strong results on challenging long-horizon real-world manipulation tasks. Notably, our efficient 0.2B-parameter variant rivals significantly larger baselines, enabling low-latency on-device deployment. To further accelerate research and practical adoption, we fully open-source the entire HoloBrain ecosystem, which includes: (1) powerful pre-trained VLA foundations; (2) post-trained checkpoints for multiple simulation suites and real-world tasks; and (3) RoboOrchard, a full-stack VLA infrastructure for data curation, model training and deployment. Together with standardized data collection protocols, this release provides the community with a complete, reproducible path toward high-performance robotic manipulation.

  </details>



- **When would Vision-Proprioception Policies Fail in Robotic Manipulation?**  
  Jingxian Lu, Wenke Xia, Yuxuan Wu, Zhiwu Lu, Di Hu  
  _2026-02-12_ · https://arxiv.org/abs/2602.12032v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Proprioceptive information is critical for precise servo control by providing real-time robotic states. Its collaboration with vision is highly expected to enhance performances of the manipulation policy in complex tasks. However, recent studies have reported inconsistent observations on the generalization of vision-proprioception policies. In this work, we investigate this by conducting temporally controlled experiments. We found that during task sub-phases that robot's motion transitions, which require target localization, the vision modality of the vision-proprioception policy plays a limited role. Further analysis reveals that the policy naturally gravitates toward concise proprioceptive signals that offer faster loss reduction when training, thereby dominating the optimization and suppressing the learning of the visual modality during motion-transition phases. To alleviate this, we propose the Gradient Adjustment with Phase-guidance (GAP) algorithm that adaptively modulates the optimization of proprioception, enabling dynamic collaboration within the vision-proprioception policy. Specifically, we leverage proprioception to capture robotic states and estimate the probability of each timestep in the trajectory belonging to motion-transition phases. During policy learning, we apply fine-grained adjustment that reduces the magnitude of proprioception's gradient based on estimated probabilities, leading to robust and generalizable vision-proprioception policies. The comprehensive experiments demonstrate GAP is applicable in both simulated and real-world environments, across one-arm and dual-arm setups, and compatible with both conventional and Vision-Language-Action models. We believe this work can offer valuable insights into the development of vision-proprioception policies in robotic manipulation.

  </details>



- **Any House Any Task: Scalable Long-Horizon Planning for Abstract Human Tasks**  
  Zhihong Liu, Yang Li, Rengming Huang, Cewu Lu, Panpan Cai  
  _2026-02-12_ · https://arxiv.org/abs/2602.12244v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Open world language conditioned task planning is crucial for robots operating in large-scale household environments. While many recent works attempt to address this problem using Large Language Models (LLMs) via prompting or training, a key challenge remains scalability. Performance often degrades rapidly with increasing environment size, plan length, instruction ambiguity, and constraint complexity. In this work, we propose Any House Any Task (AHAT), a household task planner optimized for long-horizon planning in large environments given ambiguous human instructions. At its core, AHAT utilizes an LLM trained to map task instructions and textual scene graphs into grounded subgoals defined in the Planning Domain Definition Language (PDDL). These subgoals are subsequently solved to generate feasible and optimal long-horizon plans through explicit symbolic reasoning. To enhance the model's ability to decompose complex and ambiguous intentions, we introduce TGPO, a novel reinforcement learning algorithm that integrates external correction of intermediate reasoning traces into Group Relative Policy Optimization (GRPO). Experiments demonstrate that AHAT achieves significant performance gains over state-of-the-art prompting, planning, and learning methods, particularly in human-style household tasks characterized by brief instructions but requiring complex execution plans.

  </details>



- **LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion**  
  Jiangran Lyu, Kai Liu, Xuheng Zhang, Haoran Liao, Yusen Feng, Wenxuan Zhu, Tingrui Shen, Jiayi Chen, Jiazhao Zhang, Yifei Dong, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.12215v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent robot foundation models largely rely on large-scale behavior cloning, which imitates expert actions but discards transferable dynamics knowledge embedded in heterogeneous embodied data. While the Unified World Model (UWM) formulation has the potential to leverage such diverse data, existing instantiations struggle to scale to foundation-level due to coarse data usage and fragmented datasets. We introduce LDA-1B, a robot foundation model that scales through universal embodied data ingestion by jointly learning dynamics, policy, and visual forecasting, assigning distinct roles to data of varying quality. To support this regime at scale, we assemble and standardize EI-30k, an embodied interaction dataset comprising over 30k hours of human and robot trajectories in a unified format. Scalable dynamics learning over such heterogeneous data is enabled by prediction in a structured DINO latent space, which avoids redundant pixel-space appearance modeling. Complementing this representation, LDA-1B employs a multi-modal diffusion transformer to handle asynchronous vision and action streams, enabling stable training at the 1B-parameter scale. Experiments in simulation and the real world show LDA-1B outperforms prior methods (e.g., $π_{0.5}$) by up to 21\%, 48\%, and 23\% on contact-rich, dexterous, and long-horizon tasks, respectively. Notably, LDA-1B enables data-efficient fine-tuning, gaining 10\% by leveraging 30\% low-quality trajectories typically harmful and discarded.

  </details>



- **VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model**  
  Yanjiang Guo, Tony Lee, Lucy Xiaoyang Shi, Jianyu Chen, Percy Liang, Chelsea Finn  
  _2026-02-12_ · https://arxiv.org/abs/2602.12063v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The goal of this paper is to improve the performance and reliability of vision-language-action (VLA) models through iterative online interaction. Since collecting policy rollouts in the real world is expensive, we investigate whether a learned simulator-specifically, an action-conditioned video generation model-can be used to generate additional rollout data. Unfortunately, existing world models lack the physical fidelity necessary for policy improvement: they are predominantly trained on demonstration datasets that lack coverage of many different physical interactions (particularly failure cases) and struggle to accurately model small yet critical physical details in contact-rich object manipulation. We propose a simple iterative improvement algorithm that uses real-world roll-out data to improve the fidelity of the world model, which can then, in turn, be used to generate supplemental synthetic data for improving the VLA model. In our experiments on a real robot, we use this approach to improve the performance of a state-of-the-art VLA model on multiple downstream tasks. We achieve a 39.2% absolute success rate improvement over the base policy and 11.6% improvement from training with the generated synthetic rollouts. Videos can be found at this anonymous website: https://sites.google.com/view/vla-w

  </details>



- **JEPA-VLA: Video Predictive Embedding is Needed for VLA Models**  
  Shangchen Miao, Ningya Feng, Jialong Wu, Ye Lin, Xu He, Dong Li, Mingsheng Long  
  _2026-02-12_ · https://arxiv.org/abs/2602.11832v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent vision-language-action (VLA) models built upon pretrained vision-language models (VLMs) have achieved significant improvements in robotic manipulation. However, current VLAs still suffer from low sample efficiency and limited generalization. This paper argues that these limitations are closely tied to an overlooked component, pretrained visual representation, which offers insufficient knowledge on both aspects of environment understanding and policy prior. Through an in-depth analysis, we find that commonly used visual representations in VLAs, whether pretrained via language-image contrastive learning or image-based self-supervised learning, remain inadequate at capturing crucial, task-relevant environment information and at inducing effective policy priors, i.e., anticipatory knowledge of how the environment evolves under successful task execution. In contrast, we discover that predictive embeddings pretrained on videos, in particular V-JEPA 2, are adept at flexibly discarding unpredictable environment factors and encoding task-relevant temporal dynamics, thereby effectively compensating for key shortcomings of existing visual representations in VLAs. Building on these observations, we introduce JEPA-VLA, a simple yet effective approach that adaptively integrates predictive embeddings into existing VLAs. Our experiments demonstrate that JEPA-VLA yields substantial performance gains across a range of benchmarks, including LIBERO, LIBERO-plus, RoboTwin2.0, and real-robot tasks.

  </details>



- **Differentiable Modal Logic for Multi-Agent Diagnosis, Orchestration and Communication**  
  Antonin Sulc  
  _2026-02-12_ · https://arxiv.org/abs/2602.12083v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  As multi-agent AI systems evolve from simple chatbots to autonomous swarms, debugging semantic failures requires reasoning about knowledge, belief, causality, and obligation, precisely what modal logic was designed to formalize. However, traditional modal logic requires manual specification of relationship structures that are unknown or dynamic in real systems. This tutorial demonstrates differentiable modal logic (DML), implemented via Modal Logical Neural Networks (MLNNs), enabling systems to learn trust networks, causal chains, and regulatory boundaries from behavioral data alone. We present a unified neurosymbolic debugging framework through four modalities: epistemic (who to trust), temporal (when events cause failures), deontic (what actions are permitted), and doxastic (how to interpret agent confidence). Each modality is demonstrated on concrete multi-agent scenarios, from discovering deceptive alliances in diplomacy games to detecting LLM hallucinations, with complete implementations showing how logical contradictions become learnable optimization objectives. Key contributions for the neurosymbolic community: (1) interpretable learned structures where trust and causality are explicit parameters, not opaque embeddings; (2) knowledge injection via differentiable axioms that guide learning with sparse data (3) compositional multi-modal reasoning that combines epistemic, temporal, and deontic constraints; and (4) practical deployment patterns for monitoring, active control and communication of multi-agent systems. All code provided as executable Jupyter notebooks.

  </details>



- **Accelerating Robotic Reinforcement Learning with Agent Guidance**  
  Haojun Chen, Zili Zou, Chengdong Ma, Yaoxiang Pu, Haotong Zhang, Yuanpei Chen, Yaodong Yang  
  _2026-02-12_ · https://arxiv.org/abs/2602.11978v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reinforcement Learning (RL) offers a powerful paradigm for autonomous robots to master generalist manipulation skills through trial-and-error. However, its real-world application is stifled by severe sample inefficiency. Recent Human-in-the-Loop (HIL) methods accelerate training by using human corrections, yet this approach faces a scalability barrier. Reliance on human supervisors imposes a 1:1 supervision ratio that limits fleet expansion, suffers from operator fatigue over extended sessions, and introduces high variance due to inconsistent human proficiency. We present Agent-guided Policy Search (AGPS), a framework that automates the training pipeline by replacing human supervisors with a multimodal agent. Our key insight is that the agent can be viewed as a semantic world model, injecting intrinsic value priors to structure physical exploration. By using executable tools, the agent provides precise guidance via corrective waypoints and spatial constraints for exploration pruning. We validate our approach on two tasks, ranging from precision insertion to deformable object manipulation. Results demonstrate that AGPS outperforms HIL methods in sample efficiency. This automates the supervision pipeline, unlocking the path to labor-free and scalable robot learning. Project website: https://agps-rl.github.io/agps.

  </details>



- **Intrinsic-Energy Joint Embedding Predictive Architectures Induce Quasimetric Spaces**  
  Anthony Kobanda, Waris Radji  
  _2026-02-12_ · https://arxiv.org/abs/2602.12245v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Joint-Embedding Predictive Architectures (JEPAs) aim to learn representations by predicting target embeddings from context embeddings, inducing a scalar compatibility energy in a latent space. In contrast, Quasimetric Reinforcement Learning (QRL) studies goal-conditioned control through directed distance values (cost-to-go) that support reaching goals under asymmetric dynamics. In this short article, we connect these viewpoints by restricting attention to a principled class of JEPA energy functions : intrinsic (least-action) energies, defined as infima of accumulated local effort over admissible trajectories between two states. Under mild closure and additivity assumptions, any intrinsic energy is a quasimetric. In goal-reaching control, optimal cost-to-go functions admit exactly this intrinsic form ; inversely, JEPAs trained to model intrinsic energies lie in the quasimetric value class targeted by QRL. Moreover, we observe why symmetric finite energies are structurally mismatched with one-way reachability, motivating asymmetric (quasimetric) energies when directionality matters.

  </details>



- **DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing**  
  Dianyi Wang, Ruihang Li, Feng Han, Chaofan Ma, Wei Song, Siyuan Wang, Yibin Wang, Yi Xin, Hongjian Liu, Zhixiong Zhang, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.12205v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Current unified multimodal models for image generation and editing typically rely on massive parameter scales (e.g., >10B), entailing prohibitive training costs and deployment footprints. In this work, we present DeepGen 1.0, a lightweight 5B unified model that achieves comprehensive capabilities competitive with or surpassing much larger counterparts. To overcome the limitations of compact models in semantic understanding and fine-grained control, we introduce Stacked Channel Bridging (SCB), a deep alignment framework that extracts hierarchical features from multiple VLM layers and fuses them with learnable 'think tokens' to provide the generative backbone with structured, reasoning-rich guidance. We further design a data-centric training strategy spanning three progressive stages: (1) Alignment Pre-training on large-scale image-text pairs and editing triplets to synchronize VLM and DiT representations, (2) Joint Supervised Fine-tuning on a high-quality mixture of generation, editing, and reasoning tasks to foster omni-capabilities, and (3) Reinforcement Learning with MR-GRPO, which leverages a mixture of reward functions and supervision signals, resulting in substantial gains in generation quality and alignment with human preferences, while maintaining stable training progress and avoiding visual artifacts. Despite being trained on only ~50M samples, DeepGen 1.0 achieves leading performance across diverse benchmarks, surpassing the 80B HunyuanImage by 28% on WISE and the 27B Qwen-Image-Edit by 37% on UniREditBench. By open-sourcing our training code, weights, and datasets, we provide an efficient, high-performance alternative to democratize unified multimodal research.

  </details>



- **DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation**  
  Xu Guo, Fulong Ye, Qichao Sun, Liyang Chen, Bingchuan Li, Pengze Zhang, Jiawei Liu, Songtao Zhao, Qian He, Xiangwang Hou  
  _2026-02-12_ · https://arxiv.org/abs/2602.12160v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advancements in foundation models have revolutionized joint audio-video generation. However, existing approaches typically treat human-centric tasks including reference-based audio-video generation (R2AV), video editing (RV2AV) and audio-driven video animation (RA2V) as isolated objectives. Furthermore, achieving precise, disentangled control over multiple character identities and voice timbres within a single framework remains an open challenge. In this paper, we propose DreamID-Omni, a unified framework for controllable human-centric audio-video generation. Specifically, we design a Symmetric Conditional Diffusion Transformer that integrates heterogeneous conditioning signals via a symmetric conditional injection scheme. To resolve the pervasive identity-timbre binding failures and speaker confusion in multi-person scenarios, we introduce a Dual-Level Disentanglement strategy: Synchronized RoPE at the signal level to ensure rigid attention-space binding, and Structured Captions at the semantic level to establish explicit attribute-subject mappings. Furthermore, we devise a Multi-Task Progressive Training scheme that leverages weakly-constrained generative priors to regularize strongly-constrained tasks, preventing overfitting and harmonizing disparate objectives. Extensive experiments demonstrate that DreamID-Omni achieves comprehensive state-of-the-art performance across video, audio, and audio-visual consistency, even outperforming leading proprietary commercial models. We will release our code to bridge the gap between academic research and commercial-grade applications.

  </details>



- **Choose Your Agent: Tradeoffs in Adopting AI Advisors, Coaches, and Delegates in Multi-Party Negotiation**  
  Kehang Zhu, Lithium Thain, Vivian Tsai, James Wexler, Crystal Qian  
  _2026-02-12_ · https://arxiv.org/abs/2602.12089v1 · `cs.GT`  
  <details><summary>Abstract</summary>

  As AI usage becomes more prevalent in social contexts, understanding agent-user interaction is critical to designing systems that improve both individual and group outcomes. We present an online behavioral experiment (N = 243) in which participants play three multi-turn bargaining games in groups of three. Each game, presented in randomized order, grants \textit{access to} a single LLM assistance modality: proactive recommendations from an \textit{Advisor}, reactive feedback from a \textit{Coach}, or autonomous execution by a \textit{Delegate}; all modalities are powered by an underlying LLM that achieves superhuman performance in an all-agent environment. On each turn, participants privately decide whether to act manually or use the AI modality available in that game. Despite preferring the \textit{Advisor} modality, participants achieve the highest mean individual gains with the \textit{Delegate}, demonstrating a preference-performance misalignment. Moreover, delegation generates positive externalities; even non-adopting users in \textit{access-to-delegate} treatment groups benefit by receiving higher-quality offers. Mechanism analysis reveals that the \textit{Delegate} agent acts as a market maker, injecting rational, Pareto-improving proposals that restructure the trading environment. Our research reveals a gap between agent capabilities and realized group welfare. While autonomous agents can exhibit super-human strategic performance, their impact on realized welfare gains can be constrained by interfaces, user perceptions, and adoption barriers. Assistance modalities should be designed as mechanisms with endogenous participation; adoption-compatible interaction rules are a prerequisite to improving human welfare with automated assistance.

  </details>



- **Are Two LLMs Better Than One? A Student-Teacher Dual-Head LLMs Architecture for Pharmaceutical Content Optimization**  
  Suyash Mishra, Qiang Li, Anubhav Girdhar  
  _2026-02-12_ · https://arxiv.org/abs/2602.11957v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) are increasingly used to create content in regulated domains such as pharmaceuticals, where outputs must be scientifically accurate and legally compliant. Manual quality control (QC) is slow, error prone, and can become a publication bottleneck. We introduce LRBTC, a modular LLM and vision language model (VLM) driven QC architecture covering Language, Regulatory, Brand, Technical, and Content Structure checks. LRBTC combines a Student-Teacher dual model architecture, human in the loop (HITL) workflow with waterfall rule filtering to enable scalable, verifiable content validation and optimization. On AIReg-Bench, our approach achieves 83.0% F1 and 97.5% recall, reducing missed violations by 5x compared with Gemini 2.5 Pro. On CSpelling, it improves mean accuracy by 26.7%. Error analysis further reveals that while current models are strong at detecting misspellings (92.5 recall), they fail to identify complex medical grammatical (25.0 recall) and punctuation (41.7 recall) errors, highlighting a key area for future work. This work provides a practical, plug and play solution for reliable, transparent quality control of content in high stakes, compliance critical industries. We also provide access to our Demo under MIT Licenses.

  </details>



- **Who Does What? Archetypes of Roles Assigned to LLMs During Human-AI Decision-Making**  
  Shreya Chappidi, Jatinder Singh, Andra V. Krauze  
  _2026-02-12_ · https://arxiv.org/abs/2602.11924v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  LLMs are increasingly supporting decision-making across high-stakes domains, requiring critical reflection on the socio-technical factors that shape how humans and LLMs are assigned roles and interact during human-in-the-loop decision-making. This paper introduces the concept of human-LLM archetypes -- defined as re-curring socio-technical interaction patterns that structure the roles of humans and LLMs in collaborative decision-making. We describe 17 human-LLM archetypes derived from a scoping literature review and thematic analysis of 113 LLM-supported decision-making papers. Then, we evaluate these diverse archetypes across real-world clinical diagnostic cases to examine the potential effects of adopting distinct human-LLM archetypes on LLM outputs and decision outcomes. Finally, we present relevant tradeoffs and design choices across human-LLM archetypes, including decision control, social hierarchies, cognitive forcing strategies, and information requirements. Through our analysis, we show that selection of human-LLM interaction archetype can influence LLM outputs and decisions, bringing important risks and considerations for the designers of human-AI decision-making systems

  </details>



- **Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning**  
  Daiqing Wu, Xuan Zhang, Dongbao Yang, Jiashu Yao, Longfei Chen, Qingsong Liu, Sicheng Zhao, Can Ma, Yangyang Kang, Yu Zhou  
  _2026-02-12_ · https://arxiv.org/abs/2602.11909v1 · `cs.SD`  
  <details><summary>Abstract</summary>

  The maturation of Large Audio Language Models (LALMs) has raised growing expectations for them to comprehend complex audio much like humans. Current efforts primarily replicate text-based reasoning by contextualizing audio content through a one-time encoding, which introduces a critical information bottleneck. Drawing inspiration from human cognition, we propose audio-interleaved reasoning to break through this bottleneck. It treats audio as an active reasoning component, enabling sustained audio engagement and perception-grounded analysis. To instantiate it, we introduce a two-stage training framework, first teaching LALMs to localize salient audio segments through supervised fine-tuning, and then incentivizing proficient re-listening via reinforcement learning. In parallel, a structured data generation pipeline is developed to produce high-quality training data. Consequently, we present Echo, a LALM capable of dynamically re-listening to audio in demand during reasoning. On audio comprehension benchmarks, Echo achieves overall superiority in both challenging expert-level and general-purpose tasks. Comprehensive analysis further confirms the efficiency and generalizability of audio-interleaved reasoning, establishing it as a promising direction for advancing audio comprehension. Project page: https://github.com/wdqqdw/Echo.

  </details>



- **From Path Signatures to Sequential Modeling: Incremental Signature Contributions for Offline RL**  
  Ziyi Zhao, Qingchuan Li, Yuxuan Xu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11805v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Path signatures embed trajectories into tensor algebra and constitute a universal, non-parametric representation of paths; however, in the standard form, they collapse temporal structure into a single global object, which limits their suitability for decision-making problems that require step-wise reactivity. We propose the Incremental Signature Contribution (ISC) method, which decomposes truncated path signatures into a temporally ordered sequence of elements in the tensor-algebra space, corresponding to incremental contributions induced by last path increments. This reconstruction preserves the algebraic structure and expressivity of signatures, while making their internal temporal evolution explicit, enabling processing signature-based representations via sequential modeling approaches. In contrast to full signatures, ISC is inherently sensitive to instantaneous trajectory updates, which is critical for sensitive and stability-requiring control dynamics. Building on this representation, we introduce ISC-Transformer (ISCT), an offline reinforcement learning model that integrates ISC into a standard Transformer architecture without further architectural modification. We evaluate ISCT on HalfCheetah, Walker2d, Hopper, and Maze2d, including settings with delayed rewards and downgraded datasets. The results demonstrate that ISC method provides a theoretically grounded and practically effective alternative to path processing for temporally sensitive control tasks.

  </details>



- **Temporal Difference Learning with Constrained Initial Representations**  
  Jiafei Lyu, Jingwen Yang, Zhongjian Qiao, Runze Liu, Zeyuan Liu, Deheng Ye, Zongqing Lu, Xiu Li  
  _2026-02-12_ · https://arxiv.org/abs/2602.11800v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Recently, there have been numerous attempts to enhance the sample efficiency of off-policy reinforcement learning (RL) agents when interacting with the environment, including architecture improvements and new algorithms. Despite these advances, they overlook the potential of directly constraining the initial representations of the input data, which can intuitively alleviate the distribution shift issue and stabilize training. In this paper, we introduce the Tanh function into the initial layer to fulfill such a constraint. We theoretically unpack the convergence property of the temporal difference learning with the Tanh function under linear function approximation. Motivated by theoretical insights, we present our Constrained Initial Representations framework, tagged CIR, which is made up of three components: (i) the Tanh activation along with normalization methods to stabilize representations; (ii) the skip connection module to provide a linear pathway from the shallow layer to the deep layer; (iii) the convex Q-learning that allows a more flexible value estimate and mitigates potential conservatism. Empirical results show that CIR exhibits strong performance on numerous continuous control tasks, even being competitive or surpassing existing strong baseline methods.

  </details>


