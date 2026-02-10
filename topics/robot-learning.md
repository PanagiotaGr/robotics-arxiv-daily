# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-02-10 07:19 UTC_

Total papers shown: **20**


---

- **TwinRL-VLA: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation**  
  Qinwen Xu, Jiaming Liu, Rui Zhou, Shaojun Shi, Nuowei Han, Zhuoyang Liu, Chenyang Gu, Shuo Gu, Yang Yue, Gao Huang, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.09023v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Despite strong generalization capabilities, Vision-Language-Action (VLA) models remain constrained by the high cost of expert demonstrations and insufficient real-world interaction. While online reinforcement learning (RL) has shown promise in improving general foundation models, applying RL to VLA manipulation in real-world settings is still hindered by low exploration efficiency and a restricted exploration space. Through systematic real-world experiments, we observe that the effective exploration space of online RL is closely tied to the data distribution of supervised fine-tuning (SFT). Motivated by this observation, we propose TwinRL, a digital twin-real-world collaborative RL framework designed to scale and guide exploration for VLA models. First, a high-fidelity digital twin is efficiently reconstructed from smartphone-captured scenes, enabling realistic bidirectional transfer between real and simulated environments. During the SFT warm-up stage, we introduce an exploration space expansion strategy using digital twins to broaden the support of the data trajectory distribution. Building on this enhanced initialization, we propose a sim-to-real guided exploration strategy to further accelerate online RL. Specifically, TwinRL performs efficient and parallel online RL in the digital twin prior to deployment, effectively bridging the gap between offline and online training stages. Subsequently, we exploit efficient digital twin sampling to identify failure-prone yet informative configurations, which are used to guide targeted human-in-the-loop rollouts on the real robot. In our experiments, TwinRL approaches 100% success in both in-distribution regions covered by real-world demonstrations and out-of-distribution regions, delivering at least a 30% speedup over prior real-world RL methods and requiring only about 20 minutes on average across four tasks.

  </details>



- **Mimic Intent, Not Just Trajectories**  
  Renming Huang, Chendong Zeng, Wenjing Tang, Jingtian Cai, Cewu Lu, Panpan Cai  
  _2026-02-09_ · https://arxiv.org/abs/2602.08602v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While imitation learning (IL) has achieved impressive success in dexterous manipulation through generative modeling and pretraining, state-of-the-art approaches like Vision-Language-Action (VLA) models still struggle with adaptation to environmental changes and skill transfer. We argue this stems from mimicking raw trajectories without understanding the underlying intent. To address this, we propose explicitly disentangling behavior intent from execution details in end-2-end IL: \textit{``Mimic Intent, Not just Trajectories'' (MINT)}. We achieve this via \textit{multi-scale frequency-space tokenization}, which enforces a spectral decomposition of action chunk representation. We learn action tokens with a multi-scale coarse-to-fine structure, and force the coarsest token to capture low-frequency global structure and finer tokens to encode high-frequency details. This yields an abstract \textit{Intent token} that facilitates planning and transfer, and multi-scale \textit{Execution tokens} that enable precise adaptation to environmental dynamics. Building on this hierarchy, our policy generates trajectories through \textit{next-scale autoregression}, performing progressive \textit{intent-to-execution reasoning}, thus boosting learning efficiency and generalization. Crucially, this disentanglement enables \textit{one-shot transfer} of skills, by simply injecting the Intent token from a demonstration into the autoregressive generation process. Experiments on several manipulation benchmarks and on a real robot demonstrate state-of-the-art success rates, superior inference efficiency, robust generalization against disturbances, and effective one-shot transfer.

  </details>



- **From Obstacles to Etiquette: Robot Social Navigation with VLM-Informed Path Selection**  
  Zilin Fang, Anxing Xiao, David Hsu, Gim Hee Lee  
  _2026-02-09_ · https://arxiv.org/abs/2602.09002v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Navigating socially in human environments requires more than satisfying geometric constraints, as collision-free paths may still interfere with ongoing activities or conflict with social norms. Addressing this challenge calls for analyzing interactions between agents and incorporating common-sense reasoning into planning. This paper presents a social robot navigation framework that integrates geometric planning with contextual social reasoning. The system first extracts obstacles and human dynamics to generate geometrically feasible candidate paths, then leverages a fine-tuned vision-language model (VLM) to evaluate these paths, informed by contextually grounded social expectations, selecting a socially optimized path for the controller. This task-specific VLM distills social reasoning from large foundation models into a smaller and efficient model, allowing the framework to perform real-time adaptation in diverse human-robot interaction contexts. Experiments in four social navigation contexts demonstrate that our method achieves the best overall performance with the lowest personal space violation duration, the minimal pedestrian-facing time, and no social zone intrusions. Project page: https://path-etiquette.github.io

  </details>



- **Mind the Gap: Learning Implicit Impedance in Visuomotor Policies via Intent-Execution Mismatch**  
  Cuijie Xu, Shurui Zheng, Zihao Su, Yuanfan Xu, Tinghao Yi, Xudong Zhang, Jian Wang, Yu Wang, Jinchen Yu  
  _2026-02-09_ · https://arxiv.org/abs/2602.08776v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Teleoperation inherently relies on the human operator acting as a closed-loop controller to actively compensate for hardware imperfections, including latency, mechanical friction, and lack of explicit force feedback. Standard Behavior Cloning (BC), by mimicking the robot's executed trajectory, fundamentally ignores this compensatory mechanism. In this work, we propose a Dual-State Conditioning framework that shifts the learning objective to "Intent Cloning" (master command). We posit that the Intent-Execution Mismatch, the discrepancy between master command and slave response, is not noise, but a critical signal that physically encodes implicit interaction forces and algorithmically reveals the operator's strategy for overcoming system dynamics. By predicting the master intent, our policy learns to generate a "virtual equilibrium point", effectively realizing implicit impedance control. Furthermore, by explicitly conditioning on the history of this mismatch, the model performs implicit system identification, perceiving tracking errors as external forces to close the control loop. To bridge the temporal gap caused by inference latency, we further formulate the policy as a trajectory inpainter to ensure continuous control. We validate our approach on a sensorless, low-cost bi-manual setup. Empirical results across tasks requiring contact-rich manipulation and dynamic tracking reveal a decisive gap: while standard execution-cloning fails due to the inability to overcome contact stiffness and tracking lag, our mismatch-aware approach achieves robust success. This presents a minimalist behavior cloning framework for low-cost hardware, enabling force perception and dynamic compensation without relying on explicit force sensing. Videos are available on the \href{https://xucj98.github.io/mind-the-gap-page/}{project page}.

  </details>



- **Any-to-All MRI Synthesis: A Unified Foundation Model for Nasopharyngeal Carcinoma and Its Downstream Applications**  
  Yao Pu, Yiming Shi, Zhenxi Zhang, Peixin Yu, Yitao Zhuang, Xiang Wang, Hongzhao Chen, Jing Cai, Ge Ren  
  _2026-02-09_ · https://arxiv.org/abs/2602.08822v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Magnetic resonance imaging (MRI) is essential for nasopharyngeal carcinoma (NPC) radiotherapy (RT), but practical constraints, such as patient discomfort, long scan times, and high costs often lead to incomplete modalities in clinical practice, compromising RT planning accuracy. Traditional MRI synthesis methods are modality-specific, limited in anatomical adaptability, and lack clinical interpretability-failing to meet NPC's RT needs. Here, we developed a unified foundation model integrating contrastive visual representation learning and vision-language alignment (VLA) to enable any-to-all MRI synthesis. The model uses a contrastive encoder for modality-invariant representations and a CLIP-based text-informed decoder for semantically consistent synthesis, supporting any-to-all MRI synthesis via one unified foundation model. Trained on 40,825 images from 13 institutions, it achieves consistently high performance (average SSIM 0.90, PSNR 27) across 26 internal/external validation sites (15,748 images), with superior synthesis fidelity and robustness to noise and domain shifts. Meanwhile, its unified representation enhances downstream RT-relevant tasks (e.g., segmentation). This work advances digital medicine solutions for NPC care by leveraging foundation models to bridge technical synthesis and clinical utility.

  </details>



- **Constrained Sampling to Guide Universal Manipulation RL**  
  Marc Toussaint, Cornelius V. Braun, Eckart Cobo-Briesewitz, Sayantan Auddy, Armand Jordana, Justin Carpentier  
  _2026-02-09_ · https://arxiv.org/abs/2602.08557v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We consider how model-based solvers can be leveraged to guide training of a universal policy to control from any feasible start state to any feasible goal in a contact-rich manipulation setting. While Reinforcement Learning (RL) has demonstrated its strength in such settings, it may struggle to sufficiently explore and discover complex manipulation strategies, especially in sparse-reward settings. Our approach is based on the idea of a lower-dimensional manifold of feasible, likely-visited states during such manipulation and to guide RL with a sampler from this manifold. We propose Sample-Guided RL, which uses model-based constraint solvers to efficiently sample feasible configurations (satisfying differentiable collision, contact, and force constraints) and leverage them to guide RL for universal (goal-conditioned) manipulation policies. We study using this data directly to bias state visitation, as well as using black-box optimization of open-loop trajectories between random configurations to impose a state bias and optionally add a behavior cloning loss. In a minimalistic double sphere manipulation setting, Sample-Guided RL discovers complex manipulation strategies and achieves high success rates in reaching any statically stable state. In a more challenging panda arm setting, our approach achieves a significant success rate over a near-zero baseline, and demonstrates a breadth of complex whole-body-contact manipulation strategies.

  </details>



- **High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning**  
  Jiarui Zhang, Chengyong Lei, Chengjiang Dai, Lijie Wang, Zhichao Han, Fei Gao  
  _2026-02-09_ · https://arxiv.org/abs/2602.08653v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Quadrotor unmanned aerial vehicles (UAVs) are increasingly deployed in complex missions that demand reliable autonomous navigation and robust obstacle avoidance. However, traditional modular pipelines often incur cumulative latency, whereas purely reinforcement learning (RL) approaches typically provide limited formal safety guarantees. To bridge this gap, we propose an end-to-end RL framework augmented with model-based safety mechanisms. We incorporate physical priors in both training and deployment. During training, we design a physics-informed reward structure that provides global navigational guidance. During deployment, we integrate a real-time safety filter that projects the policy outputs onto a provably safe set to enforce strict collision-avoidance constraints. This hybrid architecture reconciles high-speed flight with robust safety assurances. Benchmark evaluations demonstrate that our method outperforms both traditional planners and recent end-to-end obstacle avoidance approaches based on differentiable physics. Extensive experiments demonstrate strong generalization, enabling reliable high-speed navigation in dense clutter and challenging outdoor forest environments at velocities up to 7.5m/s.

  </details>



- **SemiNFT: Learning to Transfer Presets from Imitation to Appreciation via Hybrid-Sample Reinforcement Learning**  
  Melany Yang, Yuhang Yu, Diwang Weng, Jinwei Chen, Wei Dong  
  _2026-02-09_ · https://arxiv.org/abs/2602.08582v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Photorealistic color retouching plays a vital role in visual content creation, yet manual retouching remains inaccessible to non-experts due to its reliance on specialized expertise. Reference-based methods offer a promising alternative by transferring the preset color of a reference image to a source image. However, these approaches often operate as novice learners, performing global color mappings derived from pixel-level statistics, without a true understanding of semantic context or human aesthetics. To address this issue, we propose SemiNFT, a Diffusion Transformer (DiT)-based retouching framework that mirrors the trajectory of human artistic training: beginning with rigid imitation and evolving into intuitive creation. Specifically, SemiNFT is first taught with paired triplets to acquire basic structural preservation and color mapping skills, and then advanced to reinforcement learning (RL) on unpaired data to cultivate nuanced aesthetic perception. Crucially, during the RL stage, to prevent catastrophic forgetting of old skills, we design a hybrid online-offline reward mechanism that anchors aesthetic exploration with structural review. % experiments Extensive experiments show that SemiNFT not only outperforms state-of-the-art methods on standard preset transfer benchmarks but also demonstrates remarkable intelligence in zero-shot tasks, such as black-and-white photo colorization and cross-domain (anime-to-photo) preset transfer. These results confirm that SemiNFT transcends simple statistical matching and achieves a sophisticated level of aesthetic comprehension. Our project can be found at https://melanyyang.github.io/SemiNFT/.

  </details>



- **CoRefine: Confidence-Guided Self-Refinement for Adaptive Test-Time Compute**  
  Chen Jin, Ryutaro Tanno, Tom Diethe, Philip Teare  
  _2026-02-09_ · https://arxiv.org/abs/2602.08948v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) often rely on test-time scaling via parallel decoding (for example, 512 samples) to boost reasoning accuracy, but this incurs substantial compute. We introduce CoRefine, a confidence-guided self-refinement method that achieves competitive accuracy using a fraction of the tokens via a lightweight 211k-parameter Conv1D controller atop a frozen LLM. The controller consumes full-trace confidence to decide whether to halt, re-examine, or try a different approach, enabling targeted self-correction with an average of 2.7 refinement steps per problem and roughly 190-fold token reduction relative to 512-sample baselines. Across diverse reasoning benchmarks and three open-source models, the controller achieves 92.6 percent precision when it confidently halts, indicating that confidence dynamics reliably signal correctness without ground-truth verification. We extend this to CoRefine-Tree, a hybrid sequential-parallel variant that adaptively balances exploration and exploitation, with easy serving integration and verifier compatibility. By treating confidence as a control signal rather than a correctness guarantee, CoRefine provides a modular primitive for scalable reasoning and agentic settings with imperfect verifiers.

  </details>



- **CausalT5K: Diagnosing and Informing Refusal for Trustworthy Causal Reasoning of Skepticism, Sycophancy, Detection-Correction, and Rung Collapse**  
  Longling Geng, Andy Ouyang, Theodore Wu, Daphne Barretto, Matthew John Hayes, Rachael Cooper, Yuqiao Zeng, Sameer Vijay, Gia Ancone, Ankit Rai, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08939v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  LLM failures in causal reasoning, including sycophancy, rung collapse, and miscalibrated refusal, are well-documented, yet progress on remediation is slow because no benchmark enables systematic diagnosis. We introduce CausalT5K, a diagnostic benchmark of over 5,000 cases across 10 domains that tests three critical capabilities: (1) detecting rung collapse, where models answer interventional queries with associational evidence; (2) resisting sycophantic drift under adversarial pressure; and (3) generating Wise Refusals that specify missing information when evidence is underdetermined. Unlike synthetic benchmarks, CausalT5K embeds causal traps in realistic narratives and decomposes performance into Utility (sensitivity) and Safety (specificity), revealing failure modes invisible to aggregate accuracy. Developed through a rigorous human-machine collaborative pipeline involving 40 domain experts, iterative cross-validation cycles, and composite verification via rule-based, LLM, and human scoring, CausalT5K implements Pearl's Ladder of Causation as research infrastructure. Preliminary experiments reveal a Four-Quadrant Control Landscape where static audit policies universally fail, a finding that demonstrates CausalT5K's value for advancing trustworthy reasoning systems. Repository: https://github.com/genglongling/CausalT5kBench

  </details>



- **Designing Multi-Robot Ground Video Sensemaking with Public Safety Professionals**  
  Puqi Zhou, Ali Asgarov, Aafiya Hussain, Wonjoon Park, Amit Paudyal, Sameep Shrestha, Chia-wei Tang, Michael F. Lighthiser, Michael R. Hieb, Xuesu Xiao, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08882v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Videos from fleets of ground robots can advance public safety by providing scalable situational awareness and reducing professionals' burden. Yet little is known about how to design and integrate multi-robot videos into public safety workflows. Collaborating with six police agencies, we examined how such videos could be made practical. In Study 1, we presented the first testbed for multi-robot ground video sensemaking. The testbed includes 38 events-of-interest (EoI) relevant to public safety, a dataset of 20 robot patrol videos (10 day/night pairs) covering EoI types, and 6 design requirements aimed at improving current video sensemaking practices. In Study 2, we built MRVS, a tool that augments multi-robot patrol video streams with a prompt-engineered video understanding model. Participants reported reduced manual workload and greater confidence with LLM-based explanations, while noting concerns about false alarms and privacy. We conclude with implications for designing future multi-robot video sensemaking tools. The testbed is available at https://github.com/Puqi7/MRVS\_VideoSensemaking

  </details>



- **AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection**  
  Junru Zhang, Lang Feng, Haoran Shi, Xu Guo, Han Yu, Yabo Dong, Duanqing Xu  
  _2026-02-09_ · https://arxiv.org/abs/2602.08868v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Time-series anomaly detection (TSAD) with multimodal large language models (MLLMs) is an emerging area, yet a persistent challenge remains: MLLMs rely on coarse time-series heuristics but struggle with multi-dimensional, detailed reasoning, which is vital for understanding complex time-series data. We present AnomSeer to address this by reinforcing the model to ground its reasoning in precise, structural details of time series, unifying anomaly classification, localization, and explanation. At its core, an expert chain-of-thought trace is generated to provide a verifiable, fine-grained reasoning from classical analyses (e.g., statistical measures, frequency transforms). Building on this, we propose a novel time-series grounded policy optimization (TimerPO) that incorporates two additional components beyond standard reinforcement learning: a time-series grounded advantage based on optimal transport and an orthogonal projection to ensure this auxiliary granular signal does not interfere with the primary detection objective. Across diverse anomaly scenarios, AnomSeer, with Qwen2.5-VL-3B/7B-Instruct, outperforms larger commercial baselines (e.g., GPT-4o) in classification and localization accuracy, particularly on point- and frequency-driven exceptions. Moreover, it produces plausible time-series reasoning traces that support its conclusions.

  </details>



- **VideoVeritas: AI-Generated Video Detection via Perception Pretext Reinforcement Learning**  
  Hao Tan, Jun Lan, Senyuan Shi, Zichang Tan, Zijian Yu, Huijia Zhu, Weiqiang Wang, Jun Wan, Zhen Lei  
  _2026-02-09_ · https://arxiv.org/abs/2602.08828v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The growing capability of video generation poses escalating security risks, making reliable detection increasingly essential. In this paper, we introduce VideoVeritas, a framework that integrates fine-grained perception and fact-based reasoning. We observe that while current multi-modal large language models (MLLMs) exhibit strong reasoning capacity, their granular perception ability remains limited. To mitigate this, we introduce Joint Preference Alignment and Perception Pretext Reinforcement Learning (PPRL). Specifically, rather than directly optimizing for detection task, we adopt general spatiotemporal grounding and self-supervised object counting in the RL stage, enhancing detection performance with simple perception pretext tasks. To facilitate robust evaluation, we further introduce MintVid, a light yet high-quality dataset containing 3K videos from 9 state-of-the-art generators, along with a real-world collected subset that has factual errors in content. Experimental results demonstrate that existing methods tend to bias towards either superficial reasoning or mechanical analysis, while VideoVeritas achieves more balanced performance across diverse benchmarks.

  </details>



- **Kirin: Improving ANN efficiency with SNN Hybridization**  
  Chenyu Wang, Zhanglu Yan, Zhi Zhou, Xu Chen, Weng-Fai Wong  
  _2026-02-09_ · https://arxiv.org/abs/2602.08817v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Artificial neural networks (ANNs), particularly large language models (LLMs), demonstrate powerful inference capabilities but consume substantial energy. Conversely, spiking neural networks (SNNs) exhibit exceptional energy efficiency due to their binary and event-driven characteristics, thus motivating the study of ANN-to-SNN conversion. In this process, quantization plays a pivotal role, mapping LLMs' floating-point parameters to discrete SNN parameters via the temporal dimension of the time window. However, several challenges remain in the conversion process: (i) converting high bit-width quantization values into binary spikes requires longer time windows, increasing system latency; and (ii) the inherent trade-off between the information loss of single-spike schemes and the energy costs of multi-spike ones in SNN. To address these challenges, we propose Kirin, a integer and spike hybrid based SNN to achieve accuracy lossless ANN-to-SNN conversion with time and energy efficiency. Specifically, we first propose a Spike Matrix Hybridization strategy that encoding low bit-width parameters that leading to small time window size into binary spikes while preserving the rest in integer format, thereby reducing the overall latency of SNN execution. Second, we introduce a silence threshold mechanism to regulate the timing of single-spike firing, ensuring the output is mathematically equivalent to the LLM's output and preserves accuracy. Experimental results demonstrate that Kirin, under a W4A4\&8 quantization setting, achieves near-FP16 accuracy while reducing energy consumption by up to 84.66\% and shortening time steps by 93.75\%.

  </details>



- **How2Everything: Mining the Web for How-To Procedures to Evaluate and Improve LLMs**  
  Yapei Chang, Kyle Lo, Mohit Iyyer, Luca Soldaini  
  _2026-02-09_ · https://arxiv.org/abs/2602.08808v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Generating step-by-step "how-to" procedures is a key LLM capability: how-to advice is commonly requested in chatbots, and step-by-step planning is critical for reasoning over complex tasks. Yet, measuring and improving procedural validity at scale on real-world tasks remains challenging and understudied. To address this, we introduce How2Everything, a scalable framework to evaluate and improve goal-conditioned procedure generation. Our framework includes How2Mine, which mines 351K procedures from 980K web pages across 14 topics and readily scales to larger corpora. From this pool we build How2Bench, a 7K-example evaluation set balanced across topics. To reliably score model outputs, we develop How2Score, an evaluation protocol that uses an LLM judge to detect whether a generation contains any critical failure that would prevent achieving the goal. For low-cost, reproducible evaluation, we distill a frontier model into an open 8B model, achieving 80.5% agreement with human annotators. How2Bench reveals clear scaling trends across model sizes and training stages, providing signal early in pretraining. Finally, RL using How2Score as a reward improves performance on How2Bench by >10 points across three models without systematic regressions on standard benchmarks, with gains robust to superficial source-document memorization or format compliance. Taken together, How2Everything shows how pretraining web data can support a closed loop of capability evaluation and improvement at scale.

  </details>



- **Root Cause Analysis Method Based on Large Language Models with Residual Connection Structures**  
  Liming Zhou, Ailing Liu, Hongwei Liu, Min He, Heng Zhang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08804v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Root cause localization remain challenging in complex and large-scale microservice architectures. The complex fault propagation among microservices and the high dimensionality of telemetry data, including metrics, logs, and traces, limit the effectiveness of existing root cause analysis (RCA) methods. In this paper, a residual-connection-based RCA method using large language model (LLM), named RC-LLM, is proposed. A residual-like hierarchical fusion structure is designed to integrate multi-source telemetry data, while the contextual reasoning capability of large language models is leveraged to model temporal and cross-microservice causal dependencies. Experimental results on CCF-AIOps microservice datasets demonstrate that RC-LLM achieves strong accuracy and efficiency in root cause analysis.

  </details>



- **SoK: The Pitfalls of Deep Reinforcement Learning for Cybersecurity**  
  Shae McFadden, Myles Foley, Elizabeth Bates, Ilias Tsingenopoulos, Sanyam Vyas, Vasilios Mavroudis, Chris Hicks, Fabio Pierazzi  
  _2026-02-09_ · https://arxiv.org/abs/2602.08690v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Deep Reinforcement Learning (DRL) has achieved remarkable success in domains requiring sequential decision-making, motivating its application to cybersecurity problems. However, transitioning DRL from laboratory simulations to bespoke cyber environments can introduce numerous issues. This is further exacerbated by the often adversarial, non-stationary, and partially-observable nature of most cybersecurity tasks. In this paper, we identify and systematize 11 methodological pitfalls that frequently occur in DRL for cybersecurity (DRL4Sec) literature across the stages of environment modeling, agent training, performance evaluation, and system deployment. By analyzing 66 significant DRL4Sec papers (2018-2025), we quantify the prevalence of each pitfall and find an average of over five pitfalls per paper. We demonstrate the practical impact of these pitfalls using controlled experiments in (i) autonomous cyber defense, (ii) adversarial malware creation, and (iii) web security testing environments. Finally, we provide actionable recommendations for each pitfall to support the development of more rigorous and deployable DRL-based security systems.

  </details>



- **ALIVE: Animate Your World with Lifelike Audio-Video Generation**  
  Ying Guo, Qijun Gan, Yifu Zhang, Jinlai Liu, Yifei Hu, Pan Xie, Dongjun Qian, Yu Zhang, Ruiqi Li, Yuqi Zhang, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08682v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Video generation is rapidly evolving towards unified audio-video generation. In this paper, we present ALIVE, a generation model that adapts a pretrained Text-to-Video (T2V) model to Sora-style audio-video generation and animation. In particular, the model unlocks the Text-to-Video&Audio (T2VA) and Reference-to-Video&Audio (animation) capabilities compared to the T2V foundation models. To support the audio-visual synchronization and reference animation, we augment the popular MMDiT architecture with a joint audio-video branch which includes TA-CrossAttn for temporally-aligned cross-modal fusion and UniTemp-RoPE for precise audio-visual alignment. Meanwhile, a comprehensive data pipeline consisting of audio-video captioning, quality control, etc., is carefully designed to collect high-quality finetuning data. Additionally, we introduce a new benchmark to perform a comprehensive model test and comparison. After continue pretraining and finetuning on million-level high-quality data, ALIVE demonstrates outstanding performance, consistently outperforming open-source models and matching or surpassing state-of-the-art commercial solutions. With detailed recipes and benchmarks, we hope ALIVE helps the community develop audio-video generation models more efficiently. Official page: https://github.com/FoundationVision/Alive.

  </details>



- **From Robotics to Sepsis Treatment: Offline RL via Geometric Pessimism**  
  Sarthak Wanjari  
  _2026-02-09_ · https://arxiv.org/abs/2602.08655v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Offline Reinforcement Learning (RL) promises the recovery of optimal policies from static datasets, yet it remains susceptible to the overestimation of out-of-distribution (OOD) actions, particularly in fractured and sparse data manifolds.Current solutions necessitates a trade off between computational efficiency and performance. Methods like CQL offers rigorous conservatism but require tremendous compute power while efficient expectile-based methods like IQL often fail to correct OOD errors on pathological datasets, collapsing to Behavioural Cloning. In this work, we propose Geometric Pessimism, a modular, compute-efficient framework that augments standard IQL with density-based penalty derived from k-nearest-neighbour distances in the state-action embedding space. By pre-computing the penalties applied to each state-action pair our method injects OOD conservatism via reward shaping with a O(1) training overhead. Evaluated on the D4Rl MuJoCo benchmark, our method, Geo-IQL outperforms standard IQL on sensitive and unstable medium-replay tasks by over 18 points, while reducing inter-seed variance by 4x. Furthermore, Geo-IQL does not degrade performance on stable manifolds. Crucially, we validate our algorithm on the MIMIC-III Sepsis critical care dataset. While standard IQL collapses to behaviour cloning, Geo-IQL demonstrates active policy improvement. Maintaining safety constraints, achieving 86.4% terminal agreement with clinicians compared to IQL's 75%. Our results suggest that geometric pessimism provides the necessary regularisation to safely overcome local optima in critical, real-world decision systems.

  </details>



- **Conditional Sequence Modeling for Safe Reinforcement Learning**  
  Wensong Bai, Chao Zhang, Qihang Xu, Chufan Chen, Chenhao Zhou, Hui Qian  
  _2026-02-09_ · https://arxiv.org/abs/2602.08584v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Offline safe reinforcement learning (RL) aims to learn policies from a fixed dataset while maximizing performance under cumulative cost constraints. In practice, deployment requirements often vary across scenarios, necessitating a single policy that can adapt zero-shot to different cost thresholds. However, most existing offline safe RL methods are trained under a pre-specified threshold, yielding policies with limited generalization and deployment flexibility across cost thresholds. Motivated by recent progress in conditional sequence modeling (CSM), which enables flexible goal-conditioned control by specifying target returns, we propose RCDT, a CSM-based method that supports zero-shot deployment across multiple cost thresholds within a single trained policy. RCDT is the first CSM-based offline safe RL algorithm that integrates a Lagrangian-style cost penalty with an auto-adaptive penalty coefficient. To avoid overly conservative behavior and achieve a more favorable return--cost trade-off, a reward--cost-aware trajectory reweighting mechanism and Q-value regularization are further incorporated. Extensive experiments on the DSRL benchmark demonstrate that RCDT consistently improves return--cost trade-offs over representative baselines, advancing the state-of-the-art in offline safe RL.

  </details>


