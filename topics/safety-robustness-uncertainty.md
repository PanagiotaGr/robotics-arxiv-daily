# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-03-14 07:02 UTC_

Total papers shown: **18**


---

- **Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework**  
  Chingkwun Lam, Jiaxin Li, Lingfei Zhang, Kuo Zhao  
  _2026-03-12_ · https://arxiv.org/abs/2603.11768v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Long-term memory has emerged as a foundational component of autonomous Large Language Model (LLM) agents, enabling continuous adaptation, lifelong multimodal learning, and sophisticated reasoning. However, as memory systems transition from static retrieval databases to dynamic, agentic mechanisms, critical concerns regarding memory governance, semantic drift, and privacy vulnerabilities have surfaced. While recent surveys have focused extensively on memory retrieval efficiency, they largely overlook the emergent risks of memory corruption in highly dynamic environments. To address these emerging challenges, we propose the Stability and Safety-Governed Memory (SSGM) framework, a conceptual governance architecture. SSGM decouples memory evolution from execution by enforcing consistency verification, temporal decay modeling, and dynamic access control prior to any memory consolidation. Through formal analysis and architectural decomposition, we show how SSGM can mitigate topology-induced knowledge leakage where sensitive contexts are solidified into long-term storage, and help prevent semantic drift where knowledge degrades through iterative summarization. Ultimately, this work provides a comprehensive taxonomy of memory corruption risks and establishes a robust governance paradigm for deploying safe, persistent, and reliable agentic memory systems.

  </details>



- **SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics**  
  Mengzhen Liu, Enshen Zhou, Cheng Chi, Yi Han, Shanyu Rong, Liming Chen, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12193v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Active perception and manipulation are crucial for robots to interact with complex scenes. Existing methods struggle to unify semantic-driven active perception with robust, viewpoint-invariant execution. We propose SaPaVe, an end-to-end framework that jointly learns these capabilities in a data-efficient manner. Our approach decouples camera and manipulation actions rather than placing them in a shared action space, and follows a bottom-up training strategy: we first train semantic camera control on a large-scale dataset, then jointly optimize both action types using hybrid data. To support this framework, we introduce ActiveViewPose-200K, a dataset of 200k image-language-camera movement pairs for semantic camera movement learning, and a 3D geometry-aware module that improves execution robustness under dynamic viewpoints. We also present ActiveManip-Bench, the first benchmark for evaluating active manipulation beyond fixed-view settings. Extensive experiments in both simulation and real-world environments show that SaPaVe outperforms recent vision-language-action models such as GR00T N1 and \(π_0\), achieving up to 31.25\% higher success rates in real-world tasks. These results show that tightly coupled perception and execution, when trained with decoupled yet coordinated strategies, enable efficient and generalizable active manipulation. Project page: https://lmzpai.github.io/SaPaVe

  </details>



- **Decentralized Cooperative Localization for Multi-Robot Systems with Asynchronous Sensor Fusion**  
  Nivand Khosravi, Niusha Khosravi, Mohammad Bozorg, Masoud S. Bahraini  
  _2026-03-12_ · https://arxiv.org/abs/2603.12075v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Decentralized cooperative localization (DCL) is a promising approach for nonholonomic mobile robots operating in GPS-denied environments with limited communication infrastructure. This paper presents a DCL framework in which each robot performs localization locally using an Extended Kalman Filter, while sharing measurement information during update stages only when communication links are available and companion robots are successfully detected by LiDAR. The framework preserves cross-correlation consistency among robot state estimates while handling asynchronous sensor data with heterogeneous sampling rates and accommodating accelerations during dynamic maneuvers. Unlike methods that require pre-aligned coordinate systems, the proposed approach allows robots to initialize with arbitrary reference-frame orientations and achieves automatic alignment through transformation matrices in both the prediction and update stages. To improve robustness in feature-sparse environments, we introduce a dual-landmark evaluation framework that exploits both static environmental features and mobile robots as dynamic landmarks. The proposed framework enables reliable detection and feature extraction during sharp turns, while prediction accuracy is improved through information sharing from mutual observations. Experimental results in both Gazebo simulation and real-world basement environments show that DCL outperforms centralized cooperative localization (CCL), achieving a 34% reduction in RMSE, while the dual-landmark variant yields an improvement of 56%. These results demonstrate the applicability of DCL to challenging domains such as enclosed spaces, underwater environments, and feature-sparse terrains where conventional localization methods are ineffective.

  </details>



- **Taming the Adversary: Stable Minimax Deep Deterministic Policy Gradient via Fractional Objectives**  
  Taeho Lee, Donghwan Lee  
  _2026-03-12_ · https://arxiv.org/abs/2603.12110v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has achieved remarkable success in a wide range of control and decision-making tasks. However, RL agents often exhibit unstable or degraded performance when deployed in environments subject to unexpected external disturbances and model uncertainties. Consequently, ensuring reliable performance under such conditions remains a critical challenge. In this paper, we propose minimax deep deterministic policy gradient (MMDDPG), a framework for learning disturbance-resilient policies in continuous control tasks. The training process is formulated as a minimax optimization problem between a user policy and an adversarial disturbance policy. In this problem, the user learns a robust policy that minimizes the objective function, while the adversary generates disturbances that maximize it. To stabilize this interaction, we introduce a fractional objective that balances task performance and disturbance magnitude. This objective prevents excessively aggressive disturbances and promotes robust learning. Experimental evaluations in MuJoCo environments demonstrate that the proposed MMDDPG achieves significantly improved robustness against both external force perturbations and model parameter variations.

  </details>



- **Cascade: Composing Software-Hardware Attack Gadgets for Adversarial Threat Amplification in Compound AI Systems**  
  Sarbartha Banerjee, Prateek Sahu, Anjo Vahldiek-Oberwagner, Jose Sanchez Vicarte, Mohit Tiwari  
  _2026-03-12_ · https://arxiv.org/abs/2603.12023v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Rapid progress in generative AI has given rise to Compound AI systems - pipelines comprised of multiple large language models (LLM), software tools and database systems. Compound AI systems are constructed on a layered traditional software stack running on a distributed hardware infrastructure. Many of the diverse software components are vulnerable to traditional security flaws documented in the Common Vulnerabilities and Exposures (CVE) database, while the underlying distributed hardware infrastructure remains exposed to timing attacks, bit-flip faults, and power-based side channels. Today, research targets LLM-specific risks like model extraction, training data leakage, and unsafe generation -- overlooking the impact of traditional system vulnerabilities. This work investigates how traditional software and hardware vulnerabilities can complement LLM-specific algorithmic attacks to compromise the integrity of a compound AI pipeline. We demonstrate two novel attacks that combine system-level vulnerabilities with algorithmic weaknesses: (1) Exploiting a software code injection flaw along with a guardrail Rowhammer attack to inject an unaltered jailbreak prompt into an LLM, resulting in an AI safety violation, and (2) Manipulating a knowledge database to redirect an LLM agent to transmit sensitive user data to a malicious application, thus breaching confidentiality. These attacks highlight the need to address traditional vulnerabilities; we systematize the attack primitives and analyze their composition by grouping vulnerabilities by their objective and mapping them to distinct stages of an attack lifecycle. This approach enables a rigorous red-teaming exercise and lays the groundwork for future defense strategies.

  </details>



- **You Told Me to Do It: Measuring Instructional Text-induced Private Data Leakage in LLM Agents**  
  Ching-Yu Kao, Xinfeng Li, Shenyu Dai, Tianze Qiu, Pengcheng Zhou, Eric Hanchen Jiang, Philip Sperl  
  _2026-03-12_ · https://arxiv.org/abs/2603.11862v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  High-privilege LLM agents that autonomously process external documentation are increasingly trusted to automate tasks by reading and executing project instructions, yet they are granted terminal access, filesystem control, and outbound network connectivity with minimal security oversight. We identify and systematically measure a fundamental vulnerability in this trust model, which we term the \emph{Trusted Executor Dilemma}: agents execute documentation-embedded instructions, including adversarial ones, at high rates because they cannot distinguish malicious directives from legitimate setup guidance. This vulnerability is a structural consequence of the instruction-following design paradigm, not an implementation bug. To structure our measurement, we formalize a three-dimensional taxonomy covering linguistic disguise, structural obfuscation, and semantic abstraction, and construct \textbf{ReadSecBench}, a benchmark of 500 real-world README files enabling reproducible evaluation. Experiments on the commercially deployed computer-use agent show end-to-end exfiltration success rates up to 85\%, consistent across five programming languages and three injection positions. Cross-model evaluation on four LLM families in a simulation environment confirms that semantic compliance with injected instructions is consistent across model families. A 15-participant user study yields a 0\% detection rate across all participants, and evaluation of 12 rule-based and 6 LLM-based defenses shows neither category achieves reliable detection without unacceptable false-positive rates. Together, these results quantify a persistent \emph{Semantic-Safety Gap} between agents' functional compliance and their security awareness, establishing that documentation-embedded instruction injection is a persistent and currently unmitigated threat to high-privilege LLM agent deployments.

  </details>



- **When OpenClaw Meets Hospital: Toward an Agentic Operating System for Dynamic Clinical Workflows**  
  Wenxian Yang, Hanzheng Qiu, Bangqun Zhang, Chengquan Li, Zhiyong Huang, Xiaobin Feng, Rongshan Yu, Jiahong Dong  
  _2026-03-12_ · https://arxiv.org/abs/2603.11721v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large language model (LLM) agents extend conventional generative models by integrating reasoning, tool invocation, and persistent memory. Recent studies suggest that such agents may significantly improve clinical workflows by automating documentation, coordinating care processes, and assisting medical decision making. However, despite rapid progress, deploying autonomous agents in healthcare environments remains difficult due to reliability limitations, security risks, and insufficient long-term memory mechanisms. This work proposes an architecture that adapts LLM agents for hospital environments. The design introduces four core components: a restricted execution environment inspired by Linux multi-user systems, a document-centric interaction paradigm connecting patient and clinician agents, a page-indexed memory architecture designed for long-term clinical context management, and a curated medical skills library enabling ad-hoc composition of clinical task sequences. Rather than granting agents unrestricted system access, the architecture constrains actions through predefined skill interfaces and resource isolation. We argue that such a system forms the basis of an Agentic Operating System for Hospital, a computing layer capable of coordinating clinical workflows while maintaining safety, transparency, and auditability. This work grounds the design in OpenClaw, an open-source autonomous agent framework that structures agent capabilities as a curated library of discrete skills, and extends it with the infrastructure-level constraints required for safe clinical deployment.

  </details>



- **O3N: Omnidirectional Open-Vocabulary Occupancy Prediction**  
  Mengfei Duan, Hao Shi, Fei Teng, Guoqiang Zhao, Yuheng Zhang, Zhiyong Li, Kailun Yang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12144v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Understanding and reconstructing the 3D world through omnidirectional perception is an inevitable trend in the development of autonomous agents and embodied intelligence. However, existing 3D occupancy prediction methods are constrained by limited perspective inputs and predefined training distribution, making them difficult to apply to embodied agents that require comprehensive and safe perception of scenes in open world exploration. To address this, we present O3N, the first purely visual, end-to-end Omnidirectional Open-vocabulary Occupancy predictioN framework. O3N embeds omnidirectional voxels in a polar-spiral topology via the Polar-spiral Mamba (PsM) module, enabling continuous spatial representation and long-range context modeling across 360°. The Occupancy Cost Aggregation (OCA) module introduces a principled mechanism for unifying geometric and semantic supervision within the voxel space, ensuring consistency between the reconstructed geometry and the underlying semantic structure. Moreover, Natural Modality Alignment (NMA) establishes a gradient-free alignment pathway that harmonizes visual features, voxel embeddings, and text semantics, forming a consistent "pixel-voxel-text" representation triad. Extensive experiments on multiple models demonstrate that our method not only achieves state-of-the-art performance on QuadOcc and Human360Occ benchmarks but also exhibits remarkable cross-scene generalization and semantic scalability, paving the way toward universal 3D world modeling. The source code will be made publicly available at https://github.com/MengfeiD/O3N.

  </details>



- **LABSHIELD: A Multimodal Benchmark for Safety-Critical Reasoning and Planning in Scientific Laboratories**  
  Qianpu Sun, Xiaowei Chi, Yuhan Rui, Ying Li, Kuangzhi Ge, Jiajun Li, Sirui Han, Shanghang Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11987v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Artificial intelligence is increasingly catalyzing scientific automation, with multimodal large language model (MLLM) agents evolving from lab assistants into self-driving lab operators. This transition imposes stringent safety requirements on laboratory environments, where fragile glassware, hazardous substances, and high-precision laboratory equipment render planning errors or misinterpreted risks potentially irreversible. However, the safety awareness and decision-making reliability of embodied agents in such high-stakes settings remain insufficiently defined and evaluated. To bridge this gap, we introduce LABSHIELD, a realistic multi-view benchmark designed to assess MLLMs in hazard identification and safety-critical reasoning. Grounded in U.S. Occupational Safety and Health Administration (OSHA) standards and the Globally Harmonized System (GHS), LABSHIELD establishes a rigorous safety taxonomy spanning 164 operational tasks with diverse manipulation complexities and risk profiles. We evaluate 20 proprietary models, 9 open-source models, and 3 embodied models under a dual-track evaluation framework. Our results reveal a systematic gap between general-domain MCQ accuracy and Semi-open QA safety performance, with models exhibiting an average drop of 32.0% in professional laboratory scenarios, particularly in hazard interpretation and safety-aware planning. These findings underscore the urgent necessity for safety-centric reasoning frameworks to ensure reliable autonomous scientific experimentation in embodied laboratory contexts. The full dataset will be released soon.

  </details>



- **RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images**  
  Bin Wan, Runmin Cong, Xiaofei Zhou, Hao Fang, Yaoqi Sun, Sam Kwong  
  _2026-03-12_ · https://arxiv.org/abs/2603.12215v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Salient object detection (SOD) in remote sensing images faces significant challenges due to large variations in object sizes, the computational cost of self-attention mechanisms, and the limitations of CNN-based extractors in capturing global context and long-range dependencies. Existing methods that rely on fixed convolution kernels often struggle to adapt to diverse object scales, leading to detail loss or irrelevant feature aggregation. To address these issues, this work aims to enhance robustness to scale variations and achieve precise object localization. We propose the Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network (RDNet), which replaces the CNN backbone with the SwinTransformer for global context modeling and introduces three key modules: (1) the Dynamic Adaptive Detail-aware (DAD) module, which applies varied convolution kernels guided by object region proportions; (2) the Frequency-matching Context Enhancement (FCE) module, which enriches contextual information through wavelet interactions and attention; and (3) the Region Proportion-aware Localization (RPL) module, which employs cross-attention to highlight semantic details and integrates a Proportion Guidance (PG) block to assist the DAD module. By combining these modules, RDNet achieves robustness against scale variations and accurate localization, delivering superior detection performance compared with state-of-the-art methods.

  </details>



- **FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance**  
  Quanhao Li, Zhen Xing, Rui Wang, Haidong Cao, Qi Dai, Daoguo Dong, Zuxuan Wu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12146v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advances in trajectory-controllable video generation have achieved remarkable progress. Previous methods mainly use adapter-based architectures for precise motion control along predefined trajectories. However, all these methods rely on a multi-step denoising process, leading to substantial time redundancy and computational overhead. While existing video distillation methods successfully distill multi-step generators into few-step, directly applying these approaches to trajectory-controllable video generation results in noticeable degradation in both video quality and trajectory accuracy. To bridge this gap, we introduce FlashMotion, a novel training framework designed for few-step trajectory-controllable video generation. We first train a trajectory adapter on a multi-step video generator for precise trajectory control. Then, we distill the generator into a few-step version to accelerate video generation. Finally, we finetune the adapter using a hybrid strategy that combines diffusion and adversarial objectives, aligning it with the few-step generator to produce high-quality, trajectory-accurate videos. For evaluation, we introduce FlashBench, a benchmark for long-sequence trajectory-controllable video generation that measures both video quality and trajectory accuracy across varying numbers of foreground objects. Experiments on two adapter architectures show that FlashMotion surpasses existing video distillation methods and previous multi-step models in both visual quality and trajectory consistency.

  </details>



- **A Robust and Efficient Multi-Agent Reinforcement Learning Framework for Traffic Signal Control**  
  Sheng-You Huang, Hsiao-Chuan Chang, Yen-Chi Chen, Ting-Han Wei, I-Hau Yeh, Sheng-Yao Kuan, Chien-Yao Wang, Hsuan-Han Lee, I-Chen Wu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12096v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Reinforcement Learning (RL) in Traffic Signal Control (TSC) faces significant hurdles in real-world deployment due to limited generalization to dynamic traffic flow variations. Existing approaches often overfit static patterns and use action spaces incompatible with driver expectations. This paper proposes a robust Multi-Agent Reinforcement Learning (MARL) framework validated in the Vissim traffic simulator. The framework integrates three mechanisms: (1) Turning Ratio Randomization, a training strategy that exposes agents to dynamic turning probabilities to enhance robustness against unseen scenarios; (2) a stability-oriented Exponential Phase Duration Adjustment action space, which balances responsiveness and precision through cyclical, exponential phase adjustments; and (3) a Neighbor-Based Observation scheme utilizing the MAPPO algorithm with Centralized Training with Decentralized Execution (CTDE). By leveraging centralized updates, this approach approximates the efficacy of global observations while maintaining scalable local communication. Experimental results demonstrate that our framework outperforms standard RL baselines, reducing average waiting time by over 10%. The proposed model exhibits superior generalization in unseen traffic scenarios and maintains high control stability, offering a practical solution for adaptive signal control.

  </details>



- **Dense Dynamic Scene Reconstruction and Camera Pose Estimation from Multi-View Videos**  
  Shuo Sun, Unal Artan, Malcolm Mielle, Achim J. Lilienthaland, Martin Magnusson  
  _2026-03-12_ · https://arxiv.org/abs/2603.12064v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We address the challenging problem of dense dynamic scene reconstruction and camera pose estimation from multiple freely moving cameras -- a setting that arises naturally when multiple observers capture a shared event. Prior approaches either handle only single-camera input or require rigidly mounted, pre-calibrated camera rigs, limiting their practical applicability. We propose a two-stage optimization framework that decouples the task into robust camera tracking and dense depth refinement. In the first stage, we extend single-camera visual SLAM to the multi-camera setting by constructing a spatiotemporal connection graph that exploits both intra-camera temporal continuity and inter-camera spatial overlap, enabling consistent scale and robust tracking. To ensure robustness under limited overlap, we introduce a wide-baseline initialization strategy using feed-forward reconstruction models. In the second stage, we refine depth and camera poses by optimizing dense inter- and intra-camera consistency using wide-baseline optical flow. Additionally, we introduce MultiCamRobolab, a new real-world dataset with ground-truth poses from a motion capture system. Finally, we demonstrate that our method significantly outperforms state-of-the-art feed-forward models on both synthetic and real-world benchmarks, while requiring less memory.

  </details>



- **Single Pixel Image Classification using an Ultrafast Digital Light Projector**  
  Aisha Kanwal, Graeme E. Johnstone, Fahimeh Dehkhoda, Johannes H. Herrnsdorf, Robert K. Henderson, Martin D. Dawson, Xavier Porte, Michael J. Strain  
  _2026-03-12_ · https://arxiv.org/abs/2603.12036v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Pattern recognition and image classification are essential tasks in machine vision. Autonomous vehicles, for example, require being able to collect the complex information contained in a changing environment and classify it in real time. Here, we experimentally demonstrate image classification at multi-kHz frame rates combining the technique of single pixel imaging (SPI) with a low complexity machine learning model. The use of a microLED-on-CMOS digital light projector for SPI enables ultrafast pattern generation for sub-ms image encoding. We investigate the classification accuracy of our experimental system against the broadly accepted benchmarking task of the MNIST digits classification. We compare the classification performance of two machine learning models: An extreme learning machine (ELM) and a backpropagation trained deep neural network. The complexity of both models is kept low so the overhead added to the inference time is comparable to the image generation time. Crucially, our single pixel image classification approach is based on a spatiotemporal transformation of the information, entirely bypassing the need for image reconstruction. By exploring the performance of our SPI based ELM as binary classifier we demonstrate its potential for efficient anomaly detection in ultrafast imaging scenarios.

  </details>



- **Decentralized Orchestration Architecture for Fluid Computing: A Secure Distributed AI Use Case**  
  Diego Cajaraville-Aboy, Ana Fernández-Vilas, Rebeca P. Díaz-Redondo, Manuel Fernández-Veiga, Pablo Picallo-López  
  _2026-03-12_ · https://arxiv.org/abs/2603.12001v1 · `cs.DC`  
  <details><summary>Abstract</summary>

  Distributed AI and IoT applications increasingly execute across heterogeneous resources spanning end devices, edge/fog infrastructure, and cloud platforms, often under different administrative domains. Fluid Computing has emerged as a promising paradigm for enhancing massive resource management across the computing continuum by treating such resources as a unified fabric, enabling optimal service-agnostic deployments driven by application requirements. However, existing solutions remain largely centralized and often do not explicitly address multi-domain considerations. This paper proposes an agnostic multi-domain orchestration architecture for fluid computing environments. The orchestration plane enables decentralized coordination among domains that maintain local autonomy while jointly realizing intent-based deployment requests from tenants, ensuring end-to-end placement and execution. To this end, the architecture elevates domain-side control services as first-class capabilities to support application-level enhancement at runtime. As a representative use case, we consider a multi-domain Decentralized Federated Learning (DFL) deployment under Byzantine threats. We leverage domain-side capabilities to enhance Byzantine security by introducing FU-HST, an SDN-enabled multi-domain anomaly detection mechanism that complements Byzantine-robust aggregation. We validate the approach via simulation in single- and multi-domain settings, evaluating anomaly detection, DFL performance, and computation/communication overhead.

  </details>



- **HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios**  
  Jiayue Pu, Zhongxiang Sun, Zilu Zhang, Xiao Zhang, Jun Xu  
  _2026-03-12_ · https://arxiv.org/abs/2603.11975v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The rapid evolution of embodied agents has accelerated the deployment of household robots in real-world environments. However, unlike structured industrial settings, household spaces introduce unpredictable safety risks, where system limitations such as perception latency and lack of common sense knowledge can lead to dangerous errors. Current safety evaluations, often restricted to static images, text, or general hazards, fail to adequately benchmark dynamic unsafe action detection in these specific contexts. To bridge this gap, we introduce \textbf{HomeSafe-Bench}, a challenging benchmark designed to evaluate Vision-Language Models (VLMs) on unsafe action detection in household scenarios. HomeSafe-Bench is contrusted via a hybrid pipeline combining physical simulation with advanced video generation and features 438 diverse cases across six functional areas with fine-grained multidimensional annotations. Beyond benchmarking, we propose \textbf{Hierarchical Dual-Brain Guard for Household Safety (HD-Guard)}, a hierarchical streaming architecture for real-time safety monitoring. HD-Guard coordinates a lightweight FastBrain for continuous high-frequency screening with an asynchronous large-scale SlowBrain for deep multimodal reasoning, effectively balancing inference efficiency with detection accuracy. Evaluations demonstrate that HD-Guard achieves a superior trade-off between latency and performance, while our analysis identifies critical bottlenecks in current VLM-based safety detection.

  </details>



- **A Semi-Decentralized Approach to Multiagent Control**  
  Mahdi Al-Husseini, Mykel J. Kochenderfer, Kyle H. Wray  
  _2026-03-12_ · https://arxiv.org/abs/2603.11802v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We introduce an expressive framework and algorithms for the semi-decentralized control of cooperative agents in environments with communication uncertainty. Whereas semi-Markov control admits a distribution over time for agent actions, semi-Markov communication, or what we refer to as semi-decentralization, gives a distribution over time for what actions and observations agents can store in their histories. We extend semi-decentralization to the partially observable Markov decision process (POMDP). The resulting SDec-POMDP unifies decentralized and multiagent POMDPs and several existing explicit communication mechanisms. We present recursive small-step semi-decentralized A* (RS-SDA*), an exact algorithm for generating optimal SDec-POMDP policies. RS-SDA* is evaluated on semi-decentralized versions of several standard benchmarks and a maritime medical evacuation scenario. This paper provides a well-defined theoretical foundation for exploring many classes of multiagent communication problems through the lens of semi-decentralization.

  </details>



- **COTONET: A custom cotton detection algorithm based on YOLO11 for stage of growth cotton boll detection**  
  Guillem González, Guillem Alenyà, Sergi Foix  
  _2026-03-12_ · https://arxiv.org/abs/2603.11717v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Cotton harvesting is a critical phase where cotton capsules are physically manipulated and can lead to fibre degradation. To maintain the highest quality, harvesting methods must emulate delicate manual grasping, to preserve cotton's intrinsic properties. Automating this process requires systems capable of recognising cotton capsules across various phenological stages. To address this challenge, we propose COTONET, an enhanced custom YOLO11 model tailored with attention mechanisms to improve the detection of difficult instances. The architecture incorporates gradients in non-learnable operations to enhance shape and feature extraction. Key architectural modifications include: the replacement of convolutional blocks with Squeeze-and-Exitation blocks, a redesigned backbone integrating attention mechanisms, and the substitution of standard upsampling operations for Content Aware Reassembly of Features (CARAFE). Additionally, we integrate Simple Attention Modules (SimAM) for primary feature aggregation and Parallel Hybrid Attention Mechanisms (PHAM) for channel-wise, spatial-wise and coordinate-wise attention in the downward neck path. This configuration offers increased flexibility and robustness for interpreting the complexity of cotton crop growth. COTONET aligns with small-to-medium YOLO models utilizing 7.6M parameters and 27.8 GFLOPS, making it suitable for low-resource edge computing and mobile robotics. COTONET outperforms the standard YOLO baselines, achieving a mAP50 of 81.1% and a mAP50-95 of 60.6%.

  </details>


