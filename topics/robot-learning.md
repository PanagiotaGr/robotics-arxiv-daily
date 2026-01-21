# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-01-21 06:53 UTC_

Total papers shown: **19**


---

- **DroneVLA: VLA based Aerial Manipulation**  
  Fawad Mehboob, Monijesu James, Amir Habel, Jeffrin Sam, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13809v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  As aerial platforms evolve from passive observers to active manipulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally. This work introduces a novel concept of autonomous aerial manipulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user. The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom-built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera. VLA performs semantic reasoning to interpret the intent of a user prompt and generates a prioritized task queue for grasping of relevant objects in the scene. Grounding DINO and dynamic A* planning algorithm are used to navigate and safely relocate the object. To ensure safe and natural interaction during the handover phase, the system employs a human-centric controller driven by MediaPipe. This module provides real-time human pose estimation, allowing the drone to employ visual servoing to maintain a stable, distinct position directly in front of the user, facilitating a comfortable handover. We demonstrate the system's efficacy through real-world experiments for localization and navigation, which resulted in a 0.164m, 0.070m, and 0.084m of max, mean euclidean, and root-mean squared errors, respectively, highlighting the feasibility of VLA for aerial manipulation operations.

  </details>



- **Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep Reinforcement Learning**  
  Myong-Yol Choi, Hankyoul Ko, Hanse Cho, Changseung Kim, Seunghwan Kim, Jaemin Seo, Hyondong Oh  
  _2026-01-20_ · https://arxiv.org/abs/2601.13657v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a deep reinforcement learning (DRL) based controller for collective navigation of unmanned aerial vehicle (UAV) swarms in communication-denied environments, enabling robust operation in complex, obstacle-rich environments. Inspired by biological swarms where informed individuals guide groups without explicit communication, we employ an implicit leader-follower framework. In this paradigm, only the leader possesses goal information, while follower UAVs learn robust policies using only onboard LiDAR sensing, without requiring any inter-agent communication or leader identification. Our system utilizes LiDAR point clustering and an extended Kalman filter for stable neighbor tracking, providing reliable perception independent of external positioning systems. The core of our approach is a DRL controller, trained in GPU-accelerated Nvidia Isaac Sim, that enables followers to learn complex emergent behaviors - balancing flocking and obstacle avoidance - using only local perception. This allows the swarm to implicitly follow the leader while robustly addressing perceptual challenges such as occlusion and limited field-of-view. The robustness and sim-to-real transfer of our approach are confirmed through extensive simulations and challenging real-world experiments with a swarm of five UAVs, which successfully demonstrated collective navigation across diverse indoor and outdoor environments without any communication or external localization.

  </details>



- **TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers**  
  Bin Yu, Shijie Lian, Xiaopeng Lin, Yuliang Wei, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Xinming Wang, Bailing Wang, Cong Huang, et al.  
  _2026-01-20_ · https://arxiv.org/abs/2601.14133v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Standard Vision-Language-Action (VLA) models typically fine-tune a monolithic Vision-Language Model (VLM) backbone explicitly for robotic control. However, this approach creates a critical tension between maintaining high-level general semantic understanding and learning low-level, fine-grained sensorimotor skills, often leading to "catastrophic forgetting" of the model's open-world capabilities. To resolve this conflict, we introduce TwinBrainVLA, a novel architecture that coordinates a generalist VLM retaining universal semantic understanding and a specialist VLM dedicated to embodied proprioception for joint robotic control. TwinBrainVLA synergizes a frozen "Left Brain", which retains robust general visual reasoning, with a trainable "Right Brain", specialized for embodied perception, via a novel Asymmetric Mixture-of-Transformers (AsyMoT) mechanism. This design allows the Right Brain to dynamically query semantic knowledge from the frozen Left Brain and fuse it with proprioceptive states, providing rich conditioning for a Flow-Matching Action Expert to generate precise continuous controls. Extensive experiments on SimplerEnv and RoboCasa benchmarks demonstrate that TwinBrainVLA achieves superior manipulation performance compared to state-of-the-art baselines while explicitly preserving the comprehensive visual understanding capabilities of the pre-trained VLM, offering a promising direction for building general-purpose robots that simultaneously achieve high-level semantic understanding and low-level physical dexterity.

  </details>



- **Zero-shot adaptable task planning for autonomous construction robots: a comparative study of lightweight single and multi-AI agent systems**  
  Hossein Naderi, Alireza Shojaei, Lifu Huang, Philip Agee, Kereshmeh Afsari, Abiola Akanmu  
  _2026-01-20_ · https://arxiv.org/abs/2601.14091v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots are expected to play a major role in the future construction industry but face challenges due to high costs and difficulty adapting to dynamic tasks. This study explores the potential of foundation models to enhance the adaptability and generalizability of task planning in construction robots. Four models are proposed and implemented using lightweight, open-source large language models (LLMs) and vision language models (VLMs). These models include one single agent and three multi-agent teams that collaborate to create robot action plans. The models are evaluated across three construction roles: Painter, Safety Inspector, and Floor Tiling. Results show that the four-agent team outperforms the state-of-the-art GPT-4o in most metrics while being ten times more cost-effective. Additionally, teams with three and four agents demonstrate the improved generalizability. By discussing how agent behaviors influence outputs, this study enhances the understanding of AI teams and supports future research in diverse unstructured environments beyond construction.

  </details>



- **FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation**  
  Jing Zuo, Lingzhou Mu, Fan Jiang, Chengcheng Ma, Mu Xu, Yonggang Qi  
  _2026-01-20_ · https://arxiv.org/abs/2601.13976v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Achieving human-level performance in Vision-and-Language Navigation (VLN) requires an embodied agent to jointly understand multimodal instructions and visual-spatial context while reasoning over long action sequences. Recent works, such as NavCoT and NavGPT-2, demonstrate the potential of Chain-of-Thought (CoT) reasoning for improving interpretability and long-horizon planning. Moreover, multimodal extensions like OctoNav-R1 and CoT-VLA further validate CoT as a promising pathway toward human-like navigation reasoning. However, existing approaches face critical drawbacks: purely textual CoTs lack spatial grounding and easily overfit to sparse annotated reasoning steps, while multimodal CoTs incur severe token inflation by generating imagined visual observations, making real-time navigation impractical. In this work, we propose FantasyVLN, a unified implicit reasoning framework that preserves the benefits of CoT reasoning without explicit token overhead. Specifically, imagined visual tokens are encoded into a compact latent space using a pretrained Visual AutoRegressor (VAR) during CoT reasoning training, and the model jointly learns from textual, visual, and multimodal CoT modes under a unified multi-CoT strategy. At inference, our model performs direct instruction-to-action mapping while still enjoying reasoning-aware representations. Extensive experiments on LH-VLN show that our approach achieves reasoning-aware yet real-time navigation, improving success rates and efficiency while reducing inference latency by an order of magnitude compared to explicit CoT methods.

  </details>



- **Glance-or-Gaze: Incentivizing LMMs to Adaptively Focus Search via Reinforcement Learning**  
  Hongbo Bai, Yujin Zhou, Yile Wu, Chi-Min Chan, Pengcheng Wen, Kunhao Pan, Sirui Han, Yike Guo  
  _2026-01-20_ · https://arxiv.org/abs/2601.13942v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Large Multimodal Models (LMMs) have achieved remarkable success in visual understanding, yet they struggle with knowledge-intensive queries involving long-tail entities or evolving information due to static parametric knowledge. Recent search-augmented approaches attempt to address this limitation, but existing methods rely on indiscriminate whole-image retrieval that introduces substantial visual redundancy and noise, and lack deep iterative reflection, limiting their effectiveness on complex visual queries. To overcome these challenges, we propose Glance-or-Gaze (GoG), a fully autonomous framework that shifts from passive perception to active visual planning. GoG introduces a Selective Gaze mechanism that dynamically chooses whether to glance at global context or gaze into high-value regions, filtering irrelevant information before retrieval. We design a dual-stage training strategy: Reflective GoG Behavior Alignment via supervised fine-tuning instills the fundamental GoG paradigm, while Complexity-Adaptive Reinforcement Learning further enhances the model's capability to handle complex queries through iterative reasoning. Experiments across six benchmarks demonstrate state-of-the-art performance. Ablation studies confirm that both Selective Gaze and complexity-adaptive RL are essential for effective visual search. We will release our data and models for further exploration soon.

  </details>



- **Diffusion-Guided Backdoor Attacks in Real-World Reinforcement Learning**  
  Tairan Huang, Qingqing Ye, Yulin Jin, Jiawei Lian, Yi Wang, Haibo Hu  
  _2026-01-20_ · https://arxiv.org/abs/2601.14104v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Backdoor attacks embed hidden malicious behaviors in reinforcement learning (RL) policies and activate them using triggers at test time. Most existing attacks are validated only in simulation, while their effectiveness in real-world robotic systems remains unclear. In physical deployment, safety-constrained control pipelines such as velocity limiting, action smoothing, and collision avoidance suppress abnormal actions, causing strong attenuation of conventional backdoor attacks. We study this previously overlooked problem and propose a diffusion-guided backdoor attack framework (DGBA) for real-world RL. We design small printable visual patch triggers placed on the floor and generate them using a conditional diffusion model that produces diverse patch appearances under real-world visual variations. We treat the robot control stack as a black-box system. We further introduce an advantage-based poisoning strategy that injects triggers only at decision-critical training states. We evaluate our method on a TurtleBot3 mobile robot and demonstrate reliable activation of targeted attacks while preserving normal task performance. Demo videos and code are available in the supplementary material.

  </details>



- **Optimizing Energy and Data Collection in UAV-aided IoT Networks using Attention-based Multi-Objective Reinforcement Learning**  
  Babacar Toure, Dimitrios Tsilimantos, Omid Esrafilian, Marios Kountouris  
  _2026-01-20_ · https://arxiv.org/abs/2601.14092v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Due to their adaptability and mobility, Unmanned Aerial Vehicles (UAVs) are becoming increasingly essential for wireless network services, particularly for data harvesting tasks. In this context, Artificial Intelligence (AI)-based approaches have gained significant attention for addressing UAV path planning tasks in large and complex environments, bridging the gap with real-world deployments. However, many existing algorithms suffer from limited training data, which hampers their performance in highly dynamic environments. Moreover, they often overlook the inherently multi-objective nature of the task, treating it in an overly simplistic manner. To address these limitations, we propose an attention-based Multi-Objective Reinforcement Learning (MORL) architecture that explicitly handles the trade-off between data collection and energy consumption in urban environments, even without prior knowledge of wireless channel conditions. Our method develops a single model capable of adapting to varying trade-off preferences and dynamic scenario parameters without the need for fine-tuning or retraining. Extensive simulations show that our approach achieves substantial improvements in performance, model compactness, sample efficiency, and most importantly, generalization to previously unseen scenarios, outperforming existing RL solutions.

  </details>



- **HoverAI: An Embodied Aerial Agent for Natural Human-Drone Interaction**  
  Yuhua Jin, Nikita Kuzmin, Georgii Demianchuk, Mariya Lezina, Fawad Mehboob, Issatay Tokmurziyev, Miguel Altamirano Cabrera, Muhammad Ahsan Mustafa, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13801v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Drones operating in human-occupied spaces suffer from insufficient communication mechanisms that create uncertainty about their intentions. We present HoverAI, an embodied aerial agent that integrates drone mobility, infrastructure-independent visual projection, and real-time conversational AI into a unified platform. Equipped with a MEMS laser projector, onboard semi-rigid screen, and RGB camera, HoverAI perceives users through vision and voice, responding via lip-synced avatars that adapt appearance to user demographics. The system employs a multimodal pipeline combining VAD, ASR (Whisper), LLM-based intent classification, RAG for dialogue, face analysis for personalization, and voice synthesis (XTTS v2). Evaluation demonstrates high accuracy in command recognition (F1: 0.90), demographic estimation (gender F1: 0.89, age MAE: 5.14 years), and speech transcription (WER: 0.181). By uniting aerial robotics with adaptive conversational AI and self-contained visual output, HoverAI introduces a new class of spatially-aware, socially responsive embodied agents for applications in guidance, assistance, and human-centered interaction.

  </details>



- **Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression**  
  Shaurya Mathur, Shreyas Bellary Manjunath, Nitin Kulkarni, Alina Vereshchaka  
  _2026-01-20_ · https://arxiv.org/abs/2601.14238v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Wildfires are growing in frequency and intensity, devastating ecosystems and communities while causing billions of dollars in suppression costs and economic damage annually in the U.S. Traditional wildfire management is mostly reactive, addressing fires only after they are detected. We introduce \textit{FireCastRL}, a proactive artificial intelligence (AI) framework that combines wildfire forecasting with intelligent suppression strategies. Our framework first uses a deep spatiotemporal model to predict wildfire ignition. For high-risk predictions, we deploy a pre-trained reinforcement learning (RL) agent to execute real-time suppression tactics with helitack units inside a physics-informed 3D simulation. The framework generates a threat assessment report to help emergency responders optimize resource allocation and planning. In addition, we are publicly releasing a large-scale, spatiotemporal dataset containing $\mathbf{9.5}$ million samples of environmental variables for wildfire prediction. Our work demonstrates how deep learning and RL can be combined to support both forecasting and tactical wildfire response. More details can be found at https://sites.google.com/view/firecastrl.

  </details>



- **KAGE-Bench: Fast Known-Axis Visual Generalization Evaluation for Reinforcement Learning**  
  Egor Cherepanov, Daniil Zelezetsky, Alexey K. Kovalev, Aleksandr I. Panov  
  _2026-01-20_ · https://arxiv.org/abs/2601.14232v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Pixel-based reinforcement learning agents often fail under purely visual distribution shift even when latent dynamics and rewards are unchanged, but existing benchmarks entangle multiple sources of shift and hinder systematic analysis. We introduce KAGE-Env, a JAX-native 2D platformer that factorizes the observation process into independently controllable visual axes while keeping the underlying control problem fixed. By construction, varying a visual axis affects performance only through the induced state-conditional action distribution of a pixel policy, providing a clean abstraction for visual generalization. Building on this environment, we define KAGE-Bench, a benchmark of six known-axis suites comprising 34 train-evaluation configuration pairs that isolate individual visual shifts. Using a standard PPO-CNN baseline, we observe strong axis-dependent failures, with background and photometric shifts often collapsing success, while agent-appearance shifts are comparatively benign. Several shifts preserve forward motion while breaking task completion, showing that return alone can obscure generalization failures. Finally, the fully vectorized JAX implementation enables up to 33M environment steps per second on a single GPU, enabling fast and reproducible sweeps over visual factors. Code: https://avanturist322.github.io/KAGEBench/.

  </details>



- **Toward Efficient Agents: Memory, Tool learning, and Planning**  
  Xiaofang Yang, Lijun Li, Heng Zhou, Tong Zhu, Xiaoye Qu, Yuchen Fan, Qianshan Wei, Rui Ye, Li Kang, Yiran Qin, et al.  
  _2026-01-20_ · https://arxiv.org/abs/2601.14192v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Recent years have witnessed increasing interest in extending large language models into agentic systems. While the effectiveness of agents has continued to improve, efficiency, which is crucial for real-world deployment, has often been overlooked. This paper therefore investigates efficiency from three core components of agents: memory, tool learning, and planning, considering costs such as latency, tokens, steps, etc. Aimed at conducting comprehensive research addressing the efficiency of the agentic system itself, we review a broad range of recent approaches that differ in implementation yet frequently converge on shared high-level principles including but not limited to bounding context via compression and management, designing reinforcement learning rewards to minimize tool invocation, and employing controlled search mechanisms to enhance efficiency, which we discuss in detail. Accordingly, we characterize efficiency in two complementary ways: comparing effectiveness under a fixed cost budget, and comparing cost at a comparable level of effectiveness. This trade-off can also be viewed through the Pareto frontier between effectiveness and cost. From this perspective, we also examine efficiency oriented benchmarks by summarizing evaluation protocols for these components and consolidating commonly reported efficiency metrics from both benchmark and methodological studies. Moreover, we discuss the key challenges and future directions, with the goal of providing promising insights.

  </details>



- **Fine-Grained Zero-Shot Composed Image Retrieval with Complementary Visual-Semantic Integration**  
  Yongcong Ye, Kai Zhang, Yanghai Zhang, Enhong Chen, Longfei Li, Jun Zhou  
  _2026-01-20_ · https://arxiv.org/abs/2601.14060v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Zero-shot composed image retrieval (ZS-CIR) is a rapidly growing area with significant practical applications, allowing users to retrieve a target image by providing a reference image and a relative caption describing the desired modifications. Existing ZS-CIR methods often struggle to capture fine-grained changes and integrate visual and semantic information effectively. They primarily rely on either transforming the multimodal query into a single text using image-to-text models or employing large language models for target image description generation, approaches that often fail to capture complementary visual information and complete semantic context. To address these limitations, we propose a novel Fine-Grained Zero-Shot Composed Image Retrieval method with Complementary Visual-Semantic Integration (CVSI). Specifically, CVSI leverages three key components: (1) Visual Information Extraction, which not only extracts global image features but also uses a pre-trained mapping network to convert the image into a pseudo token, combining it with the modification text and the objects most likely to be added. (2) Semantic Information Extraction, which involves using a pre-trained captioning model to generate multiple captions for the reference image, followed by leveraging an LLM to generate the modified captions and the objects most likely to be added. (3) Complementary Information Retrieval, which integrates information extracted from both the query and database images to retrieve the target image, enabling the system to efficiently handle retrieval queries in a variety of situations. Extensive experiments on three public datasets (e.g., CIRR, CIRCO, and FashionIQ) demonstrate that CVSI significantly outperforms existing state-of-the-art methods. Our code is available at https://github.com/yyc6631/CVSI.

  </details>



- **RL-BioAug: Label-Efficient Reinforcement Learning for Self-Supervised EEG Representation Learning**  
  Cheol-Hui Lee, Hwa-Yeon Lee, Dong-Joo Kim  
  _2026-01-20_ · https://arxiv.org/abs/2601.13964v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The quality of data augmentation serves as a critical determinant for the performance of contrastive learning in EEG tasks. Although this paradigm is promising for utilizing unlabeled data, static or random augmentation strategies often fail to preserve intrinsic information due to the non-stationarity of EEG signals where statistical properties change over time. To address this, we propose RL-BioAug, a framework that leverages a label-efficient reinforcement learning (RL) agent to autonomously determine optimal augmentation policies. While utilizing only a minimal fraction (10\%) of labeled data to guide the agent's policy, our method enables the encoder to learn robust representations in a strictly self-supervised manner. Experimental results demonstrate that RL-BioAug significantly outperforms the random selection strategy, achieving substantial improvements of 9.69\% and 8.80\% in Macro-F1 score on the Sleep-EDFX and CHB-MIT datasets, respectively. Notably, this agent mainly chose optimal strategies for each task -- for example, Time Masking with a 62\% probability for sleep stage classification and Crop \& Resize with a 77\% probability for seizure detection. Our framework suggests its potential to replace conventional heuristic-based augmentations and establish a new autonomous paradigm for data augmentation. The source code is available at \href{https://github.com/dlcjfgmlnasa/RL-BioAug}{https://github.com/dlcjfgmlnasa/RL-BioAug}.

  </details>



- **Deep Reinforcement Learning-Based Dynamic Resource Allocation in Cell-Free Massive MIMO**  
  Phuong Nam Tran, Nhan Thanh Nguyen, Hien Quoc Ngo, Markku Juntti  
  _2026-01-20_ · https://arxiv.org/abs/2601.13934v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  In this paper, we consider power allocation and antenna activation of cell-free massive multiple-input multiple-output (CFmMIMO) systems. We first derive closed-form expressions for the system spectral efficiency (SE) and energy efficiency (EE) as functions of the power allocation coefficients and the number of active antennas at the access points (APs). Then, we aim to enhance the EE through jointly optimizing antenna activation and power control. This task leads to a non-convex and mixed-integer design problem with high-dimensional design variables. To address this, we propose a novel DRL-based framework, in which the agent learns to map large-scale fading coefficients to AP activation ratio, antenna coefficient, and power coefficient. These coefficients are then employed to determine the number of active antennas per AP and the power factors assigned to users based on closed-form expressions. By optimizing these parameters instead of directly controlling antenna selection and power allocation, the proposed method transforms the intractable optimization into a low-dimensional learning task. Our extensive simulations demonstrate the efficiency and scalability of the proposed scheme. Specifically, in a CFmMIMO system with 40 APs and 20 users, it achieves a 50% EE improvement and 3350 times run time reduction compared to the conventional sequential convex approximation method.

  </details>



- **TractRLFusion: A GPT-Based Multi-Critic Policy Fusion Framework for Fiber Tractography**  
  Ankita Joshi, Ashutosh Sharma, Anoushkrit Goel, Ranjeet Ranjan Jha, Chirag Ahuja, Arnav Bhavsar, Aditya Nigam  
  _2026-01-20_ · https://arxiv.org/abs/2601.13897v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Tractography plays a pivotal role in the non-invasive reconstruction of white matter fiber pathways, providing vital information on brain connectivity and supporting precise neurosurgical planning. Although traditional methods relied mainly on classical deterministic and probabilistic approaches, recent progress has benefited from supervised deep learning (DL) and deep reinforcement learning (DRL) to improve tract reconstruction. A persistent challenge in tractography is accurately reconstructing white matter tracts while minimizing spurious connections. To address this, we propose TractRLFusion, a novel GPT-based policy fusion framework that integrates multiple RL policies through a data-driven fusion strategy. Our method employs a two-stage training data selection process for effective policy fusion, followed by a multi-critic fine-tuning phase to enhance robustness and generalization. Experiments on HCP, ISMRM, and TractoInferno datasets demonstrate that TractRLFusion outperforms individual RL policies as well as state-of-the-art classical and DRL methods in accuracy and anatomical reliability.

  </details>



- **FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs**  
  Qian Chen, Jinlan Fu, Changsong Li, See-Kiong Ng, Xipeng Qiu  
  _2026-01-20_ · https://arxiv.org/abs/2601.13836v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Although Multimodal Large Language Models (MLLMs) demonstrate strong omni-modal perception, their ability to forecast future events from audio-visual cues remains largely unexplored, as existing benchmarks focus mainly on retrospective understanding. To bridge this gap, we introduce FutureOmni, the first benchmark designed to evaluate omni-modal future forecasting from audio-visual environments. The evaluated models are required to perform cross-modal causal and temporal reasoning, as well as effectively leverage internal knowledge to predict future events. FutureOmni is constructed via a scalable LLM-assisted, human-in-the-loop pipeline and contains 919 videos and 1,034 multiple-choice QA pairs across 8 primary domains. Evaluations on 13 omni-modal and 7 video-only models show that current systems struggle with audio-visual future prediction, particularly in speech-heavy scenarios, with the best accuracy of 64.8% achieved by Gemini 3 Flash. To mitigate this limitation, we curate a 7K-sample instruction-tuning dataset and propose an Omni-Modal Future Forecasting (OFF) training strategy. Evaluations on FutureOmni and popular audio-visual and video-only benchmarks demonstrate that OFF enhances future forecasting and generalization. We publicly release all code (https://github.com/OpenMOSS/FutureOmni) and datasets (https://huggingface.co/datasets/OpenMOSS-Team/FutureOmni).

  </details>



- **Reinforcement Learning for Opportunistic Routing in Software-Defined LEO-Terrestrial Systems**  
  Sivaram Krishnan, Zhouyou Gu, Jihong Park, Sung-Min Oh, Jinho Choi  
  _2026-01-20_ · https://arxiv.org/abs/2601.13662v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  The proliferation of large-scale low Earth orbit (LEO) satellite constellations is driving the need for intelligent routing strategies that can effectively deliver data to terrestrial networks under rapidly time-varying topologies and intermittent gateway visibility. Leveraging the global control capabilities of a geostationary (GEO)-resident software-defined networking (SDN) controller, we introduce opportunistic routing, which aims to minimize delivery delay by forwarding packets to any currently available ground gateways rather than fixed destinations. This makes it a promising approach for achieving low-latency and robust data delivery in highly dynamic LEO networks. Specifically, we formulate a constrained stochastic optimization problem and employ a residual reinforcement learning framework to optimize opportunistic routing for reducing transmission delay. Simulation results over multiple days of orbital data demonstrate that our method achieves significant improvements in queue length reduction compared to classical backpressure and other well-known queueing algorithms.

  </details>



- **TimeART: Towards Agentic Time Series Reasoning via Tool-Augmentation**  
  Xingjian Wu, Junkai Lu, Zhengyu Li, Xiangfei Qiu, Jilin Hu, Chenjuan Guo, Christian S. Jensen, Bin Yang  
  _2026-01-20_ · https://arxiv.org/abs/2601.13653v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Time series data widely exist in real-world cyber-physical systems. Though analyzing and interpreting them contributes to significant values, e.g, disaster prediction and financial risk control, current workflows mainly rely on human data scientists, which requires significant labor costs and lacks automation. To tackle this, we introduce TimeART, a framework fusing the analytical capability of strong out-of-the-box tools and the reasoning capability of Large Language Models (LLMs), which serves as a fully agentic data scientist for Time Series Question Answering (TSQA). To teach the LLM-based Time Series Reasoning Models (TSRMs) strategic tool-use, we also collect a 100k expert trajectory corpus called TimeToolBench. To enhance TSRMs' generalization capability, we then devise a four-stage training strategy, which boosts TSRMs through learning from their own early experiences and self-reflections. Experimentally, we train an 8B TSRM on TimeToolBench and equip it with the TimeART framework, and it achieves consistent state-of-the-art performance on multiple TSQA tasks, which pioneers a novel approach towards agentic time series reasoning.

  </details>


