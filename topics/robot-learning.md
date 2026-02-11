# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-11 07:16 UTC_

Total papers shown: **20**


---

- **AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild**  
  Xiaolou Sun, Wufei Si, Wenhui Ni, Yuntian Li, Dongming Wu, Fei Xie, Runwei Guan, He-Yang Xu, Henghui Ding, Yuan Wu, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09657v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-language navigation (VLN) requires intelligent agents to navigate environments by interpreting linguistic instructions alongside visual observations, serving as a cornerstone task in Embodied AI. Current VLN research for unmanned aerial vehicles (UAVs) relies on detailed, pre-specified instructions to guide the UAV along predetermined routes. However, real-world outdoor exploration typically occurs in unknown environments where detailed navigation instructions are unavailable. Instead, only coarse-grained positional or directional guidance can be provided, requiring UAVs to autonomously navigate through continuous planning and obstacle avoidance. To bridge this gap, we propose AutoFly, an end-to-end Vision-Language-Action (VLA) model for autonomous UAV navigation. AutoFly incorporates a pseudo-depth encoder that derives depth-aware features from RGB inputs to enhance spatial reasoning, coupled with a progressive two-stage training strategy that effectively aligns visual, depth, and linguistic representations with action policies. Moreover, existing VLN datasets have fundamental limitations for real-world autonomous navigation, stemming from their heavy reliance on explicit instruction-following over autonomous decision-making and insufficient real-world data. To address these issues, we construct a novel autonomous navigation dataset that shifts the paradigm from instruction-following to autonomous behavior modeling through: (1) trajectory collection emphasizing continuous obstacle avoidance, autonomous planning, and recognition workflows; (2) comprehensive real-world data integration. Experimental results demonstrate that AutoFly achieves a 3.9% higher success rate compared to state-of-the-art VLA baselines, with consistent performance across simulated and real environments.

  </details>



- **Rethinking Visual-Language-Action Model Scaling: Alignment, Mixture, and Regularization**  
  Ye Wang, Sipeng Zheng, Hao Luo, Wanpeng Zhang, Haoqi Yuan, Chaoyi Xu, Haiweng Xu, Yicheng Feng, Mingyang Yu, Zhiyu Kang, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09722v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While Vision-Language-Action (VLA) models show strong promise for generalist robot control, it remains unclear whether -- and under what conditions -- the standard "scale data" recipe translates to robotics, where training data is inherently heterogeneous across embodiments, sensors, and action spaces. We present a systematic, controlled study of VLA scaling that revisits core training choices for pretraining across diverse robots. Using a representative VLA framework that combines a vision-language backbone with flow-matching, we ablate key design decisions under matched conditions and evaluate in extensive simulation and real-robot experiments. To improve the reliability of real-world results, we introduce a Grouped Blind Ensemble protocol that blinds operators to model identity and separates policy execution from outcome judgment, reducing experimenter bias. Our analysis targets three dimensions of VLA scaling. (1) Physical alignment: we show that a unified end-effector (EEF)-relative action representation is critical for robust cross-embodiment transfer. (2) Embodiment mixture: we find that naively pooling heterogeneous robot datasets often induces negative transfer rather than gains, underscoring the fragility of indiscriminate data scaling. (3) Training regularization: we observe that intuitive strategies, such as sensory dropout and multi-stage fine-tuning, do not consistently improve performance at scale. Together, this study challenge some common assumptions about embodied scaling and provide practical guidance for training large-scale VLA policies from diverse robotic data. Project website: https://research.beingbeyond.com/rethink_vla

  </details>



- **BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation**  
  Yucheng Hu, Jianke Zhang, Yuanfei Luo, Yanjiang Guo, Xiaoyu Chen, Xinshu Sun, Kun Feng, Qingzhou Lu, Sheng Chen, Yangang Zhang, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09849v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Equipping embodied agents with the ability to reason about tasks, foresee physical outcomes, and generate precise actions is essential for general-purpose manipulation. While recent Vision-Language-Action (VLA) models have leveraged pre-trained foundation models, they typically focus on either linguistic planning or visual forecasting in isolation. These methods rarely integrate both capabilities simultaneously to guide action generation, leading to suboptimal performance in complex, long-horizon manipulation tasks. To bridge this gap, we propose BagelVLA, a unified model that integrates linguistic planning, visual forecasting, and action generation within a single framework. Initialized from a pretrained unified understanding and generative model, BagelVLA is trained to interleave textual reasoning and visual prediction directly into the action execution loop. To efficiently couple these modalities, we introduce Residual Flow Guidance (RFG), which initializes from current observation and leverages single-step denoising to extract predictive visual features, guiding action generation with minimal latency. Extensive experiments demonstrate that BagelVLA outperforms existing baselines by a significant margin on multiple simulated and real-world benchmarks, particularly in tasks requiring multi-stage reasoning.

  </details>



- **RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation**  
  Hao Li, Ziqin Wang, Zi-han Ding, Shuai Yang, Yilun Chen, Yang Tian, Xiaolin Hu, Tai Wang, Dahua Lin, Feng Zhao, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09973v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Advances in large vision-language models (VLMs) have stimulated growing interest in vision-language-action (VLA) systems for robot manipulation. However, existing manipulation datasets remain costly to curate, highly embodiment-specific, and insufficient in coverage and diversity, thereby hindering the generalization of VLA models. Recent approaches attempt to mitigate these limitations via a plan-then-execute paradigm, where high-level plans (e.g., subtasks, trace) are first generated and subsequently translated into low-level actions, but they critically rely on extra intermediate supervision, which is largely absent from existing datasets. To bridge this gap, we introduce the RoboInter Manipulation Suite, a unified resource including data, benchmarks, and models of intermediate representations for manipulation. It comprises RoboInter-Tool, a lightweight GUI that enables semi-automatic annotation of diverse representations, and RoboInter-Data, a large-scale dataset containing over 230k episodes across 571 diverse scenes, which provides dense per-frame annotations over more than 10 categories of intermediate representations, substantially exceeding prior work in scale and annotation quality. Building upon this foundation, RoboInter-VQA introduces 9 spatial and 20 temporal embodied VQA categories to systematically benchmark and enhance the embodied reasoning capabilities of VLMs. Meanwhile, RoboInter-VLA offers an integrated plan-then-execute framework, supporting modular and end-to-end VLA variants that bridge high-level planning with low-level execution via intermediate supervision. In total, RoboInter establishes a practical foundation for advancing robust and generalizable robotic learning via fine-grained and diverse intermediate representations.

  </details>



- **Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction**  
  Sizhe Yang, Linning Xu, Hao Li, Juncheng Mu, Jia Zeng, Dahua Lin, Jiangmiao Pang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10101v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  3D spatial perception is fundamental to generalizable robotic manipulation, yet obtaining reliable, high-quality 3D geometry remains challenging. Depth sensors suffer from noise and material sensitivity, while existing reconstruction models lack the precision and metric consistency required for physical interaction. We introduce Robo3R, a feed-forward, manipulation-ready 3D reconstruction model that predicts accurate, metric-scale scene geometry directly from RGB images and robot states in real time. Robo3R jointly infers scale-invariant local geometry and relative camera poses, which are unified into the scene representation in the canonical robot frame via a learned global similarity transformation. To meet the precision demands of manipulation, Robo3R employs a masked point head for sharp, fine-grained point clouds, and a keypoint-based Perspective-n-Point (PnP) formulation to refine camera extrinsics and global alignment. Trained on Robo3R-4M, a curated large-scale synthetic dataset with four million high-fidelity annotated frames, Robo3R consistently outperforms state-of-the-art reconstruction methods and depth sensors. Across downstream tasks including imitation learning, sim-to-real transfer, grasp synthesis, and collision-free motion planning, we observe consistent gains in performance, suggesting the promise of this alternative 3D sensing module for robotic manipulation.

  </details>



- **NavDreamer: Video Models as Zero-Shot 3D Navigators**  
  Xijie Huang, Weiqi Gai, Tianyue Wu, Congyu Wang, Zhiyang Liu, Xin Zhou, Yuze Wu, Fei Gao  
  _2026-02-10_ · https://arxiv.org/abs/2602.09765v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Previous Vision-Language-Action models face critical limitations in navigation: scarce, diverse data from labor-intensive collection and static representations that fail to capture temporal dynamics and physical laws. We propose NavDreamer, a video-based framework for 3D navigation that leverages generative video models as a universal interface between language instructions and navigation trajectories. Our main hypothesis is that video's ability to encode spatiotemporal information and physical dynamics, combined with internet-scale availability, enables strong zero-shot generalization in navigation. To mitigate the stochasticity of generative predictions, we introduce a sampling-based optimization method that utilizes a VLM for trajectory scoring and selection. An inverse dynamics model is employed to decode executable waypoints from generated video plans for navigation. To systematically evaluate this paradigm in several video model backbones, we introduce a comprehensive benchmark covering object navigation, precise navigation, spatial grounding, language control, and scene reasoning. Extensive experiments demonstrate robust generalization across novel objects and unseen environments, with ablation studies revealing that navigation's high-level decision-making nature makes it particularly suited for video-based planning.

  </details>



- **ST4VLA: Spatially Guided Training for Vision-Language-Action Models**  
  Jinhui Ye, Fangjing Wang, Ning Gao, Junqiu Yu, Yangkun Zhu, Bin Wang, Jinyu Zhang, Weiyang Jin, Yanwei Fu, Feng Zheng, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.10109v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Large vision-language models (VLMs) excel at multimodal understanding but fall short when extended to embodied tasks, where instructions must be transformed into low-level motor actions. We introduce ST4VLA, a dual-system Vision-Language-Action framework that leverages Spatial Guided Training to align action learning with spatial priors in VLMs. ST4VLA includes two stages: (i) spatial grounding pre-training, which equips the VLM with transferable priors via scalable point, box, and trajectory prediction from both web-scale and robot-specific data, and (ii) spatially guided action post-training, which encourages the model to produce richer spatial priors to guide action generation via spatial prompting. This design preserves spatial grounding during policy learning and promotes consistent optimization across spatial and action objectives. Empirically, ST4VLA achieves substantial improvements over vanilla VLA, with performance increasing from 66.1 -> 84.6 on Google Robot and from 54.7 -> 73.2 on WidowX Robot, establishing new state-of-the-art results on SimplerEnv. It also demonstrates stronger generalization to unseen objects and paraphrased instructions, as well as robustness to long-horizon perturbations in real-world settings. These results highlight scalable spatially guided training as a promising direction for robust, generalizable robot learning. Source code, data and models are released at https://internrobotics.github.io/internvla-m1.github.io/

  </details>



- **UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking**  
  Baijun Chen, Weijie Wan, Tianxing Chen, Xianda Guo, Congsheng Xu, Yuanyang Qi, Haojie Zhang, Longyan Wu, Tianling Xu, Zixuan Li, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.10093v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic manipulation has seen rapid progress with vision-language-action (VLA) policies. However, visuo-tactile perception is critical for contact-rich manipulation, as tasks such as insertion are difficult to complete robustly using vision alone. At the same time, acquiring large-scale and reliable tactile data in the physical world remains costly and challenging, and the lack of a unified evaluation platform further limits policy learning and systematic analysis. To address these challenges, we propose UniVTAC, a simulation-based visuo-tactile data synthesis platform that supports three commonly used visuo-tactile sensors and enables scalable and controllable generation of informative contact interactions. Based on this platform, we introduce the UniVTAC Encoder, a visuo-tactile encoder trained on large-scale simulation-synthesized data with designed supervisory signals, providing tactile-centric visuo-tactile representations for downstream manipulation tasks. In addition, we present the UniVTAC Benchmark, which consists of eight representative visuo-tactile manipulation tasks for evaluating tactile-driven policies. Experimental results show that integrating the UniVTAC Encoder improves average success rates by 17.1% on the UniVTAC Benchmark, while real-world robotic experiments further demonstrate a 25% improvement in task success. Our webpage is available at https://univtac.github.io/.

  </details>



- **Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning**  
  Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, Yuxiong He  
  _2026-02-10_ · https://arxiv.org/abs/2602.10090v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Recent advances in large language model (LLM) have empowered autonomous agents to perform complex tasks that require multi-turn interactions with tools and environments. However, scaling such agent training is limited by the lack of diverse and reliable environments. In this paper, we propose Agent World Model (AWM), a fully synthetic environment generation pipeline. Using this pipeline, we scale to 1,000 environments covering everyday scenarios, in which agents can interact with rich toolsets (35 tools per environment on average) and obtain high-quality observations. Notably, these environments are code-driven and backed by databases, providing more reliable and consistent state transitions than environments simulated by LLMs. Moreover, they enable more efficient agent interaction compared with collecting trajectories from realistic environments. To demonstrate the effectiveness of this resource, we perform large-scale reinforcement learning for multi-turn tool-use agents. Thanks to the fully executable environments and accessible database states, we can also design reliable reward functions. Experiments on three benchmarks show that training exclusively in synthetic environments, rather than benchmark-specific ones, yields strong out-of-distribution generalization. The code is available at https://github.com/Snowflake-Labs/agent-world-model.

  </details>



- **Diverse Skill Discovery for Quadruped Robots via Unsupervised Learning**  
  Ruopeng Cui, Yifei Bi, Haojie Luo, Wei Li  
  _2026-02-10_ · https://arxiv.org/abs/2602.09767v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reinforcement learning necessitates meticulous reward shaping by specialists to elicit target behaviors, while imitation learning relies on costly task-specific data. In contrast, unsupervised skill discovery can potentially reduce these burdens by learning a diverse repertoire of useful skills driven by intrinsic motivation. However, existing methods exhibit two key limitations: they typically rely on a single policy to master a versatile repertoire of behaviors without modeling the shared structure or distinctions among them, which results in low learning efficiency; moreover, they are susceptible to reward hacking, where the reward signal increases and converges rapidly while the learned skills display insufficient actual diversity. In this work, we introduce an Orthogonal Mixture-of-Experts (OMoE) architecture that prevents diverse behaviors from collapsing into overlapping representations, enabling a single policy to master a wide spectrum of locomotion skills. In addition, we design a multi-discriminator framework in which different discriminators operate on distinct observation spaces, effectively mitigating reward hacking. We evaluated our method on the 12-DOF Unitree A1 quadruped robot, demonstrating a diverse set of locomotion skills. Our experiments demonstrate that the proposed framework boosts training efficiency and yields an 18.3\% expansion in state-space coverage compared to the baseline.

  </details>



- **EgoHumanoid: Unlocking In-the-Wild Loco-Manipulation with Robot-Free Egocentric Demonstration**  
  Modi Shi, Shijia Peng, Jin Chen, Haoran Jiang, Yinghui Li, Di Huang, Ping Luo, Hongyang Li, Li Chen  
  _2026-02-10_ · https://arxiv.org/abs/2602.10106v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Human demonstrations offer rich environmental diversity and scale naturally, making them an appealing alternative to robot teleoperation. While this paradigm has advanced robot-arm manipulation, its potential for the more challenging, data-hungry problem of humanoid loco-manipulation remains largely unexplored. We present EgoHumanoid, the first framework to co-train a vision-language-action policy using abundant egocentric human demonstrations together with a limited amount of robot data, enabling humanoids to perform loco-manipulation across diverse real-world environments. To bridge the embodiment gap between humans and robots, including discrepancies in physical morphology and viewpoint, we introduce a systematic alignment pipeline spanning from hardware design to data processing. A portable system for scalable human data collection is developed, and we establish practical collection protocols to improve transferability. At the core of our human-to-humanoid alignment pipeline lies two key components. The view alignment reduces visual domain discrepancies caused by camera height and perspective variation. The action alignment maps human motions into a unified, kinematically feasible action space for humanoid control. Extensive real-world experiments demonstrate that incorporating robot-free egocentric data significantly outperforms robot-only baselines by 51\%, particularly in unseen environments. Our analysis further reveals which behaviors transfer effectively and the potential for scaling human data.

  </details>



- **Humanoid Factors: Design Principles for AI Humanoids in Human Worlds**  
  Xinyuan Liu, Eren Sadikoglu, Ransalu Senanayake, Lixiao Huang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10069v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Human factors research has long focused on optimizing environments, tools, and systems to account for human performance. Yet, as humanoid robots begin to share our workplaces, homes, and public spaces, the design challenge expands. We must now consider not only factors for humans but also factors for humanoids, since both will coexist and interact within the same environments. Unlike conventional machines, humanoids introduce expectations of human-like behavior, communication, and social presence, which reshape usability, trust, and safety considerations. In this article, we introduce the concept of humanoid factors as a framework structured around four pillars - physical, cognitive, social, and ethical - that shape the development of humanoids to help them effectively coexist and collaborate with humans. This framework characterizes the overlap and divergence between human capabilities and those of general-purpose humanoids powered by AI foundation models. To demonstrate our framework's practical utility, we then apply the framework to evaluate a real-world humanoid control algorithm, illustrating how conventional task completion metrics in robotics overlook key human cognitive and interaction principles. We thus position humanoid factors as a foundational framework for designing, evaluating, and governing sustained human-humanoid coexistence.

  </details>



- **A Collaborative Safety Shield for Safe and Efficient CAV Lane Changes in Congested On-Ramp Merging**  
  Bharathkumar Hegde, Melanie Bouroche  
  _2026-02-10_ · https://arxiv.org/abs/2602.10007v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Lane changing in dense traffic is a significant challenge for Connected and Autonomous Vehicles (CAVs). Existing lane change controllers primarily either ensure safety or collaboratively improve traffic efficiency, but do not consider these conflicting objectives together. To address this, we propose the Multi-Agent Safety Shield (MASS), designed using Control Barrier Functions (CBFs) to enable safe and collaborative lane changes. The MASS enables collaboration by capturing multi-agent interactions among CAVs through interaction topologies constructed as a graph using a simple algorithm. Further, a state-of-the-art Multi-Agent Reinforcement Learning (MARL) lane change controller is extended by integrating MASS to ensure safety and defining a customised reward function to prioritise efficiency improvements. As a result, we propose a lane change controller, known as MARL-MASS, and evaluate it in a congested on-ramp merging simulation. The results demonstrate that MASS enables collaborative lane changes with safety guarantees by strictly respecting the safety constraints. Moreover, the proposed custom reward function improves the stability of MARL policies trained with a safety shield. Overall, by encouraging the exploration of a collaborative lane change policy while respecting safety constraints, MARL-MASS effectively balances the trade-off between ensuring safety and improving traffic efficiency in congested traffic. The code for MARL-MASS is available with an open-source licence at https://github.com/hkbharath/MARL-MASS

  </details>



- **Code2World: A GUI World Model via Renderable Code Generation**  
  Yuhao Zheng, Li'an Zhong, Yi Wang, Rui Dai, Kaikui Liu, Xiangxiang Chu, Linyuan Lv, Philip Torr, Kevin Qinghong Lin  
  _2026-02-10_ · https://arxiv.org/abs/2602.09856v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous GUI agents interact with environments by perceiving interfaces and executing actions. As a virtual sandbox, the GUI World model empowers agents with human-like foresight by enabling action-conditioned prediction. However, existing text- and pixel-based approaches struggle to simultaneously achieve high visual fidelity and fine-grained structural controllability. To this end, we propose Code2World, a vision-language coder that simulates the next visual state via renderable code generation. Specifically, to address the data scarcity problem, we construct AndroidCode by translating GUI trajectories into high-fidelity HTML and refining synthesized code through a visual-feedback revision mechanism, yielding a corpus of over 80K high-quality screen-action pairs. To adapt existing VLMs into code prediction, we first perform SFT as a cold start for format layout following, then further apply Render-Aware Reinforcement Learning which uses rendered outcome as the reward signal by enforcing visual semantic fidelity and action consistency. Extensive experiments demonstrate that Code2World-8B achieves the top-performing next UI prediction, rivaling the competitive GPT-5 and Gemini-3-Pro-Image. Notably, Code2World significantly enhances downstream navigation success rates in a flexible manner, boosting Gemini-2.5-Flash by +9.5% on AndroidWorld navigation. The code is available at https://github.com/AMAP-ML/Code2World.

  </details>



- **Optimistic World Models: Efficient Exploration in Model-Based Deep Reinforcement Learning**  
  Akshay Mete, Shahid Aamir Sheikh, Tzu-Hsiang Lin, Dileep Kalathil, P. R. Kumar  
  _2026-02-10_ · https://arxiv.org/abs/2602.10044v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Efficient exploration remains a central challenge in reinforcement learning (RL), particularly in sparse-reward environments. We introduce Optimistic World Models (OWMs), a principled and scalable framework for optimistic exploration that brings classical reward-biased maximum likelihood estimation (RBMLE) from adaptive control into deep RL. In contrast to upper confidence bound (UCB)-style exploration methods, OWMs incorporate optimism directly into model learning by augmentation with an optimistic dynamics loss that biases imagined transitions toward higher-reward outcomes. This fully gradient-based loss requires neither uncertainty estimates nor constrained optimization. Our approach is plug-and-play with existing world model frameworks, preserving scalability while requiring only minimal modifications to standard training procedures. We instantiate OWMs within two state-of-the-art world model architectures, leading to Optimistic DreamerV3 and Optimistic STORM, which demonstrate significant improvements in sample efficiency and cumulative return compared to their baseline counterparts.

  </details>



- **Why Do AI Agents Systematically Fail at Cloud Root Cause Analysis?**  
  Taeyoon Kim, Woohyeok Park, Hoyeong Yun, Kyungyong Lee  
  _2026-02-10_ · https://arxiv.org/abs/2602.09937v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Failures in large-scale cloud systems incur substantial financial losses, making automated Root Cause Analysis (RCA) essential for operational stability. Recent efforts leverage Large Language Model (LLM) agents to automate this task, yet existing systems exhibit low detection accuracy even with capable models, and current evaluation frameworks assess only final answer correctness without revealing why the agent's reasoning failed. This paper presents a process level failure analysis of LLM-based RCA agents. We execute the full OpenRCA benchmark across five LLM models, producing 1,675 agent runs, and classify observed failures into 12 pitfall types across intra-agent reasoning, inter-agent communication, and agent-environment interaction. Our analysis reveals that the most prevalent pitfalls, notably hallucinated data interpretation and incomplete exploration, persist across all models regardless of capability tier, indicating that these failures originate from the shared agent architecture rather than from individual model limitations. Controlled mitigation experiments further show that prompt engineering alone cannot resolve the dominant pitfalls, whereas enriching the inter-agent communication protocol reduces communication-related failures by up to 15 percentage points. The pitfall taxonomy and diagnostic methodology developed in this work provide a foundation for designing more reliable autonomous agents for cloud RCA.

  </details>



- **Flexible Entropy Control in RLVR with Gradient-Preserving Perspective**  
  Kun Chen, Peng Shi, Fanfan Liu, Haibo Qiu, Zhixiong Zeng, Siqi Yang, Wenji Mao  
  _2026-02-10_ · https://arxiv.org/abs/2602.09782v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a critical method for enhancing the reasoning capabilities of Large Language Models (LLMs). However, continuous training often leads to policy entropy collapse, characterized by a rapid decay in entropy that results in premature overconfidence, reduced output diversity, and vanishing gradient norms that inhibit learning. Gradient-Preserving Clipping is a primary factor influencing these dynamics, but existing mitigation strategies are largely static and lack a framework connecting clipping mechanisms to precise entropy control. This paper proposes reshaping entropy control in RL from the perspective of Gradient-Preserving Clipping. We first theoretically and empirically verify the contributions of specific importance sampling ratio regions to entropy growth and reduction. Leveraging these findings, we introduce a novel regulation mechanism using dynamic clipping threshold to precisely manage entropy. Furthermore, we design and evaluate dynamic entropy control strategies, including increase-then-decrease, decrease-increase-decrease, and oscillatory decay. Experimental results demonstrate that these strategies effectively mitigate entropy collapse, and achieve superior performance across multiple benchmarks.

  </details>



- **Grounding LTL Tasks in Sub-Symbolic RL Environments for Zero-Shot Generalization**  
  Matteo Pannacci, Andrea Fanti, Elena Umili, Roberto Capobianco  
  _2026-02-10_ · https://arxiv.org/abs/2602.09761v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  In this work we address the problem of training a Reinforcement Learning agent to follow multiple temporally-extended instructions expressed in Linear Temporal Logic in sub-symbolic environments. Previous multi-task work has mostly relied on knowledge of the mapping between raw observations and symbols appearing in the formulae. We drop this unrealistic assumption by jointly training a multi-task policy and a symbol grounder with the same experience. The symbol grounder is trained only from raw observations and sparse rewards via Neural Reward Machines in a semi-supervised fashion. Experiments on vision-based environments show that our method achieves performance comparable to using the true symbol grounding and significantly outperforms state-of-the-art methods for sub-symbolic environments.

  </details>



- **Towards Training-free Multimodal Hate Localisation with Large Language Models**  
  Yueming Sun, Long Yang, Jianbo Jiao, Zeyu Fu  
  _2026-02-10_ · https://arxiv.org/abs/2602.09637v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The proliferation of hateful content in online videos poses severe threats to individual well-being and societal harmony. However, existing solutions for video hate detection either rely heavily on large-scale human annotations or lack fine-grained temporal precision. In this work, we propose LELA, the first training-free Large Language Model (LLM) based framework for hate video localization. Distinct from state-of-the-art models that depend on supervised pipelines, LELA leverages LLMs and modality-specific captioning to detect and temporally localize hateful content in a training-free manner. Our method decomposes a video into five modalities, including image, speech, OCR, music, and video context, and uses a multi-stage prompting scheme to compute fine-grained hateful scores for each frame. We further introduce a composition matching mechanism to enhance cross-modal reasoning. Experiments on two challenging benchmarks, HateMM and MultiHateClip, demonstrate that LELA outperforms all existing training-free baselines by a large margin. We also provide extensive ablations and qualitative visualizations, establishing LELA as a strong foundation for scalable and interpretable hate video localization.

  </details>



- **On the Optimal Reasoning Length for RL-Trained Language Models**  
  Daisuke Nohara, Taishi Nakamura, Rio Yokota  
  _2026-02-10_ · https://arxiv.org/abs/2602.09591v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Reinforcement learning substantially improves reasoning in large language models, but it also tends to lengthen chain of thought outputs and increase computational cost during both training and inference. Though length control methods have been proposed, it remains unclear what the optimal output length is for balancing efficiency and performance. In this work, we compare several length control methods on two models, Qwen3-1.7B Base and DeepSeek-R1-Distill-Qwen-1.5B. Our results indicate that length penalties may hinder reasoning acquisition, while properly tuned length control can improve efficiency for models with strong prior reasoning. By extending prior work to RL trained policies, we identify two failure modes, 1) long outputs increase dispersion, and 2) short outputs lead to under-thinking.

  </details>


