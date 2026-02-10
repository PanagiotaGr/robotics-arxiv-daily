# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-10 07:19 UTC_

Total papers shown: **20**


---

- **GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion**  
  Santiago Montiel-Marín, Miguel Antunes-García, Fabio Sánchez-García, Angel Llamazares, Holger Caesar, Luis M. Bergasa  
  _2026-02-09_ · https://arxiv.org/abs/2602.08784v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robust and accurate perception of dynamic objects and map elements is crucial for autonomous vehicles performing safe navigation in complex traffic scenarios. While vision-only methods have become the de facto standard due to their technical advances, they can benefit from effective and cost-efficient fusion with radar measurements. In this work, we advance fusion methods by repurposing Gaussian Splatting as an efficient universal view transformer that bridges the view disparity gap, mapping both image pixels and radar points into a common Bird's-Eye View (BEV) representation. Our main contribution is GaussianCaR, an end-to-end network for BEV segmentation that, unlike prior BEV fusion methods, leverages Gaussian Splatting to map raw sensor information into latent features for efficient camera-radar fusion. Our architecture combines multi-scale fusion with a transformer decoder to efficiently extract BEV features. Experimental results demonstrate that our approach achieves performance on par with, or even surpassing, the state of the art on BEV segmentation tasks (57.3%, 82.9%, and 50.1% IoU for vehicles, roads, and lane dividers) on the nuScenes dataset, while maintaining a 3.2x faster inference runtime. Code and project page are available online.

  </details>



- **High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning**  
  Jiarui Zhang, Chengyong Lei, Chengjiang Dai, Lijie Wang, Zhichao Han, Fei Gao  
  _2026-02-09_ · https://arxiv.org/abs/2602.08653v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Quadrotor unmanned aerial vehicles (UAVs) are increasingly deployed in complex missions that demand reliable autonomous navigation and robust obstacle avoidance. However, traditional modular pipelines often incur cumulative latency, whereas purely reinforcement learning (RL) approaches typically provide limited formal safety guarantees. To bridge this gap, we propose an end-to-end RL framework augmented with model-based safety mechanisms. We incorporate physical priors in both training and deployment. During training, we design a physics-informed reward structure that provides global navigational guidance. During deployment, we integrate a real-time safety filter that projects the policy outputs onto a provably safe set to enforce strict collision-avoidance constraints. This hybrid architecture reconciles high-speed flight with robust safety assurances. Benchmark evaluations demonstrate that our method outperforms both traditional planners and recent end-to-end obstacle avoidance approaches based on differentiable physics. Extensive experiments demonstrate strong generalization, enabling reliable high-speed navigation in dense clutter and challenging outdoor forest environments at velocities up to 7.5m/s.

  </details>



- **CausalT5K: Diagnosing and Informing Refusal for Trustworthy Causal Reasoning of Skepticism, Sycophancy, Detection-Correction, and Rung Collapse**  
  Longling Geng, Andy Ouyang, Theodore Wu, Daphne Barretto, Matthew John Hayes, Rachael Cooper, Yuqiao Zeng, Sameer Vijay, Gia Ancone, Ankit Rai, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08939v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  LLM failures in causal reasoning, including sycophancy, rung collapse, and miscalibrated refusal, are well-documented, yet progress on remediation is slow because no benchmark enables systematic diagnosis. We introduce CausalT5K, a diagnostic benchmark of over 5,000 cases across 10 domains that tests three critical capabilities: (1) detecting rung collapse, where models answer interventional queries with associational evidence; (2) resisting sycophantic drift under adversarial pressure; and (3) generating Wise Refusals that specify missing information when evidence is underdetermined. Unlike synthetic benchmarks, CausalT5K embeds causal traps in realistic narratives and decomposes performance into Utility (sensitivity) and Safety (specificity), revealing failure modes invisible to aggregate accuracy. Developed through a rigorous human-machine collaborative pipeline involving 40 domain experts, iterative cross-validation cycles, and composite verification via rule-based, LLM, and human scoring, CausalT5K implements Pearl's Ladder of Causation as research infrastructure. Preliminary experiments reveal a Four-Quadrant Control Landscape where static audit policies universally fail, a finding that demonstrates CausalT5K's value for advancing trustworthy reasoning systems. Repository: https://github.com/genglongling/CausalT5kBench

  </details>



- **Diffusion-Inspired Reconfiguration of Transformers for Uncertainty Calibration**  
  Manh Cuong Dao, Quang Hung Pham, Phi Le Nguyen, Thao Nguyen Truong, Bryan Kian Hsiang Low, Trong Nghia Hoang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08920v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Uncertainty calibration in pre-trained transformers is critical for their reliable deployment in risk-sensitive applications. Yet, most existing pre-trained transformers do not have a principled mechanism for uncertainty propagation through their feature transformation stack. In this work, we propose a diffusion-inspired reconfiguration of transformers in which each feature transformation block is modeled as a probabilistic mapping. Composing these probabilistic mappings reveals a probability path that mimics the structure of a diffusion process, transporting data mass from the input distribution to the pre-trained feature distribution. This probability path can then be recompiled on a diffusion process with a unified transition model to enable principled propagation of representation uncertainty throughout the pre-trained model's architecture while maintaining its original predictive performance. Empirical results across a variety of vision and language benchmarks demonstrate that our method achieves superior calibration and predictive accuracy compared to existing uncertainty-aware transformers.

  </details>



- **From Robotics to Sepsis Treatment: Offline RL via Geometric Pessimism**  
  Sarthak Wanjari  
  _2026-02-09_ · https://arxiv.org/abs/2602.08655v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Offline Reinforcement Learning (RL) promises the recovery of optimal policies from static datasets, yet it remains susceptible to the overestimation of out-of-distribution (OOD) actions, particularly in fractured and sparse data manifolds.Current solutions necessitates a trade off between computational efficiency and performance. Methods like CQL offers rigorous conservatism but require tremendous compute power while efficient expectile-based methods like IQL often fail to correct OOD errors on pathological datasets, collapsing to Behavioural Cloning. In this work, we propose Geometric Pessimism, a modular, compute-efficient framework that augments standard IQL with density-based penalty derived from k-nearest-neighbour distances in the state-action embedding space. By pre-computing the penalties applied to each state-action pair our method injects OOD conservatism via reward shaping with a O(1) training overhead. Evaluated on the D4Rl MuJoCo benchmark, our method, Geo-IQL outperforms standard IQL on sensitive and unstable medium-replay tasks by over 18 points, while reducing inter-seed variance by 4x. Furthermore, Geo-IQL does not degrade performance on stable manifolds. Crucially, we validate our algorithm on the MIMIC-III Sepsis critical care dataset. While standard IQL collapses to behaviour cloning, Geo-IQL demonstrates active policy improvement. Maintaining safety constraints, achieving 86.4% terminal agreement with clinicians compared to IQL's 75%. Our results suggest that geometric pessimism provides the necessary regularisation to safely overcome local optima in critical, real-world decision systems.

  </details>



- **A Precise Real-Time Force-Aware Grasping System for Robust Aerial Manipulation**  
  Kenghou Hoi, Yuze Wu, Annan Ding, Junjie Wang, Anke Zhao, Chengqian Zhang, Fei Gao  
  _2026-02-09_ · https://arxiv.org/abs/2602.08599v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Aerial manipulation requires force-aware capabilities to enable safe and effective grasping and physical interaction. Previous works often rely on heavy, expensive force sensors unsuitable for typical quadrotor platforms, or perform grasping without force feedback, risking damage to fragile objects. To address these limitations, we propose a novel force-aware grasping framework incorporating six low-cost, sensitive skin-like tactile sensors. We introduce a magnetic-based tactile sensing module that provides high-precision three-dimensional force measurements. We eliminate geomagnetic interference through a reference Hall sensor and simplify the calibration process compared to previous work. The proposed framework enables precise force-aware grasping control, allowing safe manipulation of fragile objects and real-time weight measurement of grasped items. The system is validated through comprehensive real-world experiments, including balloon grasping, dynamic load variation tests, and ablation studies, demonstrating its effectiveness in various aerial manipulation scenarios. Our approach achieves fully onboard operation without external motion capture systems, significantly enhancing the practicality of force-sensitive aerial manipulation. The supplementary video is available at: https://www.youtube.com/watch?v=mbcZkrJEf1I.

  </details>



- **Robustness Is a Function, Not a Number: A Factorized Comprehensive Study of OOD Robustness in Vision-Based Driving**  
  Amir Mallak, Alaa Maalouf  
  _2026-02-09_ · https://arxiv.org/abs/2602.09018v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Out of distribution (OOD) robustness in autonomous driving is often reduced to a single number, hiding what breaks a policy. We decompose environments along five axes: scene (rural/urban), season, weather, time (day/night), and agent mix; and measure performance under controlled $k$-factor perturbations ($k \in \{0,1,2,3\}$). Using closed loop control in VISTA, we benchmark FC, CNN, and ViT policies, train compact ViT heads on frozen foundation-model (FM) features, and vary ID support in scale, diversity, and temporal context. (1) ViT policies are markedly more OOD-robust than comparably sized CNN/FC, and FM features yield state-of-the-art success at a latency cost. (2) Naive temporal inputs (multi-frame) do not beat the best single-frame baseline. (3) The largest single factor drops are rural $\rightarrow$ urban and day $\rightarrow$ night ($\sim 31\%$ each); actor swaps $\sim 10\%$, moderate rain $\sim 7\%$; season shifts can be drastic, and combining a time flip with other changes further degrades performance. (4) FM-feature policies stay above $85\%$ under three simultaneous changes; non-FM single-frame policies take a large first-shift hit, and all no-FM models fall below $50\%$ by three changes. (5) Interactions are non-additive: some pairings partially offset, whereas season-time combinations are especially harmful. (6) Training on winter/snow is most robust to single-factor shifts, while a rural+summer baseline gives the best overall OOD performance. (7) Scaling traces/views improves robustness ($+11.8$ points from $5$ to $14$ traces), yet targeted exposure to hard conditions can substitute for scale. (8) Using multiple ID environments broadens coverage and strengthens weak cases (urban OOD $60.6\% \rightarrow 70.1\%$) with a small ID drop; single-ID preserves peak performance but in a narrow domain. These results yield actionable design rules for OOD-robust driving policies.

  </details>



- **TwinRL-VLA: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation**  
  Qinwen Xu, Jiaming Liu, Rui Zhou, Shaojun Shi, Nuowei Han, Zhuoyang Liu, Chenyang Gu, Shuo Gu, Yang Yue, Gao Huang, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.09023v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Despite strong generalization capabilities, Vision-Language-Action (VLA) models remain constrained by the high cost of expert demonstrations and insufficient real-world interaction. While online reinforcement learning (RL) has shown promise in improving general foundation models, applying RL to VLA manipulation in real-world settings is still hindered by low exploration efficiency and a restricted exploration space. Through systematic real-world experiments, we observe that the effective exploration space of online RL is closely tied to the data distribution of supervised fine-tuning (SFT). Motivated by this observation, we propose TwinRL, a digital twin-real-world collaborative RL framework designed to scale and guide exploration for VLA models. First, a high-fidelity digital twin is efficiently reconstructed from smartphone-captured scenes, enabling realistic bidirectional transfer between real and simulated environments. During the SFT warm-up stage, we introduce an exploration space expansion strategy using digital twins to broaden the support of the data trajectory distribution. Building on this enhanced initialization, we propose a sim-to-real guided exploration strategy to further accelerate online RL. Specifically, TwinRL performs efficient and parallel online RL in the digital twin prior to deployment, effectively bridging the gap between offline and online training stages. Subsequently, we exploit efficient digital twin sampling to identify failure-prone yet informative configurations, which are used to guide targeted human-in-the-loop rollouts on the real robot. In our experiments, TwinRL approaches 100% success in both in-distribution regions covered by real-world demonstrations and out-of-distribution regions, delivering at least a 30% speedup over prior real-world RL methods and requiring only about 20 minutes on average across four tasks.

  </details>



- **stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation**  
  Lucas Maes, Quentin Le Lidec, Dan Haramati, Nassim Massaudi, Damien Scieur, Yann LeCun, Randall Balestriero  
  _2026-02-09_ · https://arxiv.org/abs/2602.08968v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  World Models have emerged as a powerful paradigm for learning compact, predictive representations of environment dynamics, enabling agents to reason, plan, and generalize beyond direct experience. Despite recent interest in World Models, most available implementations remain publication-specific, severely limiting their reusability, increasing the risk of bugs, and reducing evaluation standardization. To mitigate these issues, we introduce stable-worldmodel (SWM), a modular, tested, and documented world-model research ecosystem that provides efficient data-collection tools, standardized environments, planning algorithms, and baseline implementations. In addition, each environment in SWM enables controllable factors of variation, including visual and physical properties, to support robustness and continual learning research. Finally, we demonstrate the utility of SWM by using it to study zero-shot robustness in DINO-WM.

  </details>



- **Modeling 3D Pedestrian-Vehicle Interactions for Vehicle-Conditioned Pose Forecasting**  
  Guangxun Zhu, Xuan Liu, Nicolas Pugeault, Chongfeng Wei, Edmond S. L. Ho  
  _2026-02-09_ · https://arxiv.org/abs/2602.08962v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurately predicting pedestrian motion is crucial for safe and reliable autonomous driving in complex urban environments. In this work, we present a 3D vehicle-conditioned pedestrian pose forecasting framework that explicitly incorporates surrounding vehicle information. To support this, we enhance the Waymo-3DSkelMo dataset with aligned 3D vehicle bounding boxes, enabling realistic modeling of multi-agent pedestrian-vehicle interactions. We introduce a sampling scheme to categorize scenes by pedestrian and vehicle count, facilitating training across varying interaction complexities. Our proposed network adapts the TBIFormer architecture with a dedicated vehicle encoder and pedestrian-vehicle interaction cross-attention module to fuse pedestrian and vehicle features, allowing predictions to be conditioned on both historical pedestrian motion and surrounding vehicles. Extensive experiments demonstrate substantial improvements in forecasting accuracy and validate different approaches for modeling pedestrian-vehicle interactions, highlighting the importance of vehicle-aware 3D pose prediction for autonomous driving. Code is available at: https://github.com/GuangxunZhu/VehCondPose3D

  </details>



- **GEMSS: A Variational Bayesian Method for Discovering Multiple Sparse Solutions in Classification and Regression Problems**  
  Kateřina Henclová, Václav Šmídl  
  _2026-02-09_ · https://arxiv.org/abs/2602.08913v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Selecting interpretable feature sets in underdetermined ($n \ll p$) and highly correlated regimes constitutes a fundamental challenge in data science, particularly when analyzing physical measurements. In such settings, multiple distinct sparse subsets may explain the response equally well. Identifying these alternatives is crucial for generating domain-specific insights into the underlying mechanisms, yet conventional methods typically isolate a single solution, obscuring the full spectrum of plausible explanations. We present GEMSS (Gaussian Ensemble for Multiple Sparse Solutions), a variational Bayesian framework specifically designed to simultaneously discover multiple, diverse sparse feature combinations. The method employs a structured spike-and-slab prior for sparsity, a mixture of Gaussians to approximate the intractable multimodal posterior, and a Jaccard-based penalty to further control solution diversity. Unlike sequential greedy approaches, GEMSS optimizes the entire ensemble of solutions within a single objective function via stochastic gradient descent. The method is validated on a comprehensive benchmark comprising 128 synthetic experiments across classification and regression tasks. Results demonstrate that GEMSS scales effectively to high-dimensional settings ($p=5000$) with sample size as small as $n = 50$, generalizes seamlessly to continuous targets, handles missing data natively, and exhibits remarkable robustness to class imbalance and Gaussian noise. GEMSS is available as a Python package 'gemss' at PyPI. The full GitHub repository at https://github.com/kat-er-ina/gemss/ also includes a free, easy-to-use application suitable for non-coders.

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



- **Any-to-All MRI Synthesis: A Unified Foundation Model for Nasopharyngeal Carcinoma and Its Downstream Applications**  
  Yao Pu, Yiming Shi, Zhenxi Zhang, Peixin Yu, Yitao Zhuang, Xiang Wang, Hongzhao Chen, Jing Cai, Ge Ren  
  _2026-02-09_ · https://arxiv.org/abs/2602.08822v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Magnetic resonance imaging (MRI) is essential for nasopharyngeal carcinoma (NPC) radiotherapy (RT), but practical constraints, such as patient discomfort, long scan times, and high costs often lead to incomplete modalities in clinical practice, compromising RT planning accuracy. Traditional MRI synthesis methods are modality-specific, limited in anatomical adaptability, and lack clinical interpretability-failing to meet NPC's RT needs. Here, we developed a unified foundation model integrating contrastive visual representation learning and vision-language alignment (VLA) to enable any-to-all MRI synthesis. The model uses a contrastive encoder for modality-invariant representations and a CLIP-based text-informed decoder for semantically consistent synthesis, supporting any-to-all MRI synthesis via one unified foundation model. Trained on 40,825 images from 13 institutions, it achieves consistently high performance (average SSIM 0.90, PSNR 27) across 26 internal/external validation sites (15,748 images), with superior synthesis fidelity and robustness to noise and domain shifts. Meanwhile, its unified representation enhances downstream RT-relevant tasks (e.g., segmentation). This work advances digital medicine solutions for NPC care by leveraging foundation models to bridge technical synthesis and clinical utility.

  </details>



- **Multi-Staged Framework for Safety Analysis of Offloaded Services in Distributed Intelligent Transportation Systems**  
  Robin Dehler, Oliver Schumann, Jona Ruof, Michael Buchholz  
  _2026-02-09_ · https://arxiv.org/abs/2602.08821v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The integration of service-oriented architectures (SOA) with function offloading for distributed, intelligent transportation systems (ITS) offers the opportunity for connected autonomous vehicles (CAVs) to extend their locally available services. One major goal of offloading a subset of functions in the processing chain of a CAV to remote devices is to reduce the overall computational complexity on the CAV. The extension of using remote services, however, requires careful safety analysis, since the remotely created data are corrupted more easily, e.g., through an attacker on the remote device or by intercepting the wireless transmission. To tackle this problem, we first analyze the concept of SOA for distributed environments. From this, we derive a safety framework that validates the reliability of remote services and the data received locally. Since it is possible for the autonomous driving task to offload multiple different services, we propose a specific multi-staged framework for safety analysis dependent on the service composition of local and remote services. For efficiency reasons, we directly include the multi-staged framework for safety analysis in our service-oriented function offloading framework (SOFOF) that we have proposed in earlier work. The evaluation compares the performance of the extended framework considering computational complexity, with energy savings being a major motivation for function offloading, and its capability to detect data from corrupted remote services.

  </details>



- **SynSacc: A Blender-to-V2E Pipeline for Synthetic Neuromorphic Eye-Movement Data and Sim-to-Real Spiking Model Training**  
  Khadija Iddrisu, Waseem Shariff, Suzanne Little, Noel OConnor  
  _2026-02-09_ · https://arxiv.org/abs/2602.08726v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The study of eye movements, particularly saccades and fixations, are fundamental to understanding the mechanisms of human cognition and perception. Accurate classification of these movements requires sensing technologies capable of capturing rapid dynamics without distortion. Event cameras, also known as Dynamic Vision Sensors (DVS), provide asynchronous recordings of changes in light intensity, thereby eliminating motion blur inherent in conventional frame-based cameras and offering superior temporal resolution and data efficiency. In this study, we introduce a synthetic dataset generated with Blender to simulate saccades and fixations under controlled conditions. Leveraging Spiking Neural Networks (SNNs), we evaluate its robustness by training two architectures and finetuning on real event data. The proposed models achieve up to 0.83 accuracy and maintain consistent performance across varying temporal resolutions, demonstrating stability in eye movement classification. Moreover, the use of SNNs with synthetic event streams yields substantial computational efficiency gains over artificial neural network (ANN) counterparts, underscoring the utility of synthetic data augmentation in advancing event-based vision. All code and datasets associated with this work is available at https: //github.com/Ikhadija-5/SynSacc-Dataset.

  </details>



- **Reasoning aligns language models to human cognition**  
  Gonçalo Guiomar, Elia Torre, Pehuen Moure, Victoria Shavina, Mario Giulianelli, Shih-Chii Liu, Valerio Mante  
  _2026-02-09_ · https://arxiv.org/abs/2602.08693v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Do language models make decisions under uncertainty like humans do, and what role does chain-of-thought (CoT) reasoning play in the underlying decision process? We introduce an active probabilistic reasoning task that cleanly separates sampling (actively acquiring evidence) from inference (integrating evidence toward a decision). Benchmarking humans and a broad set of contemporary large language models against near-optimal reference policies reveals a consistent pattern: extended reasoning is the key determinant of strong performance, driving large gains in inference and producing belief trajectories that become strikingly human-like, while yielding only modest improvements in active sampling. To explain these differences, we fit a mechanistic model that captures systematic deviations from optimal behavior via four interpretable latent variables: memory, strategy, choice bias, and occlusion awareness. This model places humans and models in a shared low-dimensional cognitive space, reproduces behavioral signatures across agents, and shows how chain-of-thought shifts language models toward human-like regimes of evidence accumulation and belief-to-choice mapping, tightening alignment in inference while leaving a persistent gap in information acquisition.

  </details>



- **SoK: The Pitfalls of Deep Reinforcement Learning for Cybersecurity**  
  Shae McFadden, Myles Foley, Elizabeth Bates, Ilias Tsingenopoulos, Sanyam Vyas, Vasilios Mavroudis, Chris Hicks, Fabio Pierazzi  
  _2026-02-09_ · https://arxiv.org/abs/2602.08690v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Deep Reinforcement Learning (DRL) has achieved remarkable success in domains requiring sequential decision-making, motivating its application to cybersecurity problems. However, transitioning DRL from laboratory simulations to bespoke cyber environments can introduce numerous issues. This is further exacerbated by the often adversarial, non-stationary, and partially-observable nature of most cybersecurity tasks. In this paper, we identify and systematize 11 methodological pitfalls that frequently occur in DRL for cybersecurity (DRL4Sec) literature across the stages of environment modeling, agent training, performance evaluation, and system deployment. By analyzing 66 significant DRL4Sec papers (2018-2025), we quantify the prevalence of each pitfall and find an average of over five pitfalls per paper. We demonstrate the practical impact of these pitfalls using controlled experiments in (i) autonomous cyber defense, (ii) adversarial malware creation, and (iii) web security testing environments. Finally, we provide actionable recommendations for each pitfall to support the development of more rigorous and deployable DRL-based security systems.

  </details>



- **MOSAIC: Bridging the Sim-to-Real Gap in Generalist Humanoid Motion Tracking and Teleoperation with Rapid Residual Adaptation**  
  Zhenguo Sun, Bo-Sheng Huang, Yibo Peng, Xukun Li, Jingyu Ma, Yu Sun, Zhe Li, Haojun Jiang, Biao Gao, Zhenshan Bing, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08594v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Generalist humanoid motion trackers have recently achieved strong simulation metrics by scaling data and training, yet often remain brittle on hardware during sustained teleoperation due to interface- and dynamics-induced errors. We present MOSAIC, an open-source, full-stack system for humanoid motion tracking and whole-body teleoperation across multiple interfaces. MOSAIC first learns a teleoperation-oriented general motion tracker via RL on a multi-source motion bank with adaptive resampling and rewards that emphasize world-frame motion consistency, which is critical for mobile teleoperation. To bridge the sim-to-real interface gap without sacrificing generality, MOSAIC then performs rapid residual adaptation: an interface-specific policy is trained using minimal interface-specific data, and then distilled into the general tracker through an additive residual module, outperforming naive fine-tuning or continual learning. We validate MOSAIC with systematic ablations, out-of-distribution benchmarking, and real-robot experiments demonstrating robust offline motion replay and online long-horizon teleoperation under realistic latency and noise.

  </details>



- **SDFed: Bridging Local Global Discrepancy via Subspace Refinement and Divergence Control in Federated Prompt Learning**  
  Yicheng Di, Wei Yuan, Tieke He, Zhanjie Zhang, Ao Ma, Yuan Liu, Hongzhi Yin  
  _2026-02-09_ · https://arxiv.org/abs/2602.08590v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Vision-language pretrained models offer strong transferable representations, yet adapting them in privacy-sensitive multi-party settings is challenging due to the high communication cost of federated optimization and the limited local data on clients. Federated prompt learning mitigates this issue by keeping the VLPM backbone frozen and collaboratively training lightweight prompt parameters. However, existing approaches typically enforce a unified prompt structure and length across clients, which is inadequate under practical client heterogeneity in both data distributions and system resources, and may further introduce conflicts between globally shared and locally optimal knowledge. To address these challenges, we propose \textbf{SDFed}, a heterogeneous federated prompt learning framework that bridges Local-Global Discrepancy via Subspace Refinement and Divergence Control. SDFed maintains a fixed-length global prompt for efficient aggregation while allowing each client to learn a variable-length local prompt to better match its data characteristics and capacity. To mitigate local-global conflicts and facilitate effective knowledge transfer, SDFed introduces a subspace refinement method for local prompts and an information retention and divergence control strategy that preserves key local information while maintaining appropriate separability between global and local representations. Extensive experiments on several datasets demonstrate that SDFed consistently improves performance and robustness in heterogeneous federated settings.

  </details>


