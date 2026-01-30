# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-01-30 07:03 UTC_

Total papers shown: **13**


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



- **Dynamic Topology Awareness: Breaking the Granularity Rigidity in Vision-Language Navigation**  
  Jiankun Peng, Jianyuan Guo, Ying Xu, Yue Liu, Jiashuang Yan, Xuanwei Ye, Houhua Li, Xiaoming Wang  
  _2026-01-29_ · https://arxiv.org/abs/2601.21751v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language Navigation in Continuous Environments (VLN-CE) presents a core challenge: grounding high-level linguistic instructions into precise, safe, and long-horizon spatial actions. Explicit topological maps have proven to be a vital solution for providing robust spatial memory in such tasks. However, existing topological planning methods suffer from a "Granularity Rigidity" problem. Specifically, these methods typically rely on fixed geometric thresholds to sample nodes, which fails to adapt to varying environmental complexities. This rigidity leads to a critical mismatch: the model tends to over-sample in simple areas, causing computational redundancy, while under-sampling in high-uncertainty regions, increasing collision risks and compromising precision. To address this, we propose DGNav, a framework for Dynamic Topological Navigation, introducing a context-aware mechanism to modulate map density and connectivity on-the-fly. Our approach comprises two core innovations: (1) A Scene-Aware Adaptive Strategy that dynamically modulates graph construction thresholds based on the dispersion of predicted waypoints, enabling "densification on demand" in challenging environments; (2) A Dynamic Graph Transformer that reconstructs graph connectivity by fusing visual, linguistic, and geometric cues into dynamic edge weights, enabling the agent to filter out topological noise and enhancing instruction adherence. Extensive experiments on the R2R-CE and RxR-CE benchmarks demonstrate DGNav exhibits superior navigation performance and strong generalization capabilities. Furthermore, ablation studies confirm that our framework achieves an optimal trade-off between navigation efficiency and safe exploration. The code is available at https://github.com/shannanshouyin/DGNav.

  </details>



- **Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving**  
  Linhan Wang, Zichong Yang, Chen Bai, Guoxiang Zhang, Xiaotong Liu, Xiaoyin Zheng, Xiao-Xiao Long, Chang-Tien Lu, Cheng Lu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22032v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  End-to-end autonomous driving increasingly leverages self-supervised video pretraining to learn transferable planning representations. However, pretraining video world models for scene understanding has so far brought only limited improvements. This limitation is compounded by the inherent ambiguity of driving: each scene typically provides only a single human trajectory, making it difficult to learn multimodal behaviors. In this work, we propose Drive-JEPA, a framework that integrates Video Joint-Embedding Predictive Architecture (V-JEPA) with multimodal trajectory distillation for end-to-end driving. First, we adapt V-JEPA for end-to-end driving, pretraining a ViT encoder on large-scale driving videos to produce predictive representations aligned with trajectory planning. Second, we introduce a proposal-centric planner that distills diverse simulator-generated trajectories alongside human trajectories, with a momentum-aware selection mechanism to promote stable and safe behavior. When evaluated on NAVSIM, the V-JEPA representation combined with a simple transformer-based decoder outperforms prior methods by 3 PDMS in the perception-free setting. The complete Drive-JEPA framework achieves 93.3 PDMS on v1 and 87.8 EPDMS on v2, setting a new state-of-the-art.

  </details>



- **ReactEMG Stroke: Healthy-to-Stroke Few-shot Adaptation for sEMG-Based Intent Detection**  
  Runsheng Wang, Katelyn Lee, Xinyue Zhu, Lauren Winterbottom, Dawn M. Nilsen, Joel Stein, Matei Ciocarlie  
  _2026-01-29_ · https://arxiv.org/abs/2601.22090v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Surface electromyography (sEMG) is a promising control signal for assist-as-needed hand rehabilitation after stroke, but detecting intent from paretic muscles often requires lengthy, subject-specific calibration and remains brittle to variability. We propose a healthy-to-stroke adaptation pipeline that initializes an intent detector from a model pretrained on large-scale able-bodied sEMG, then fine-tunes it for each stroke participant using only a small amount of subject-specific data. Using a newly collected dataset from three individuals with chronic stroke, we compare adaptation strategies (head-only tuning, parameter-efficient LoRA adapters, and full end-to-end fine-tuning) and evaluate on held-out test sets that include realistic distribution shifts such as within-session drift, posture changes, and armband repositioning. Across conditions, healthy-pretrained adaptation consistently improves stroke intent detection relative to both zero-shot transfer and stroke-only training under the same data budget; the best adaptation methods improve average transition accuracy from 0.42 to 0.61 and raw accuracy from 0.69 to 0.78. These results suggest that transferring a reusable healthy-domain EMG representation can reduce calibration burden while improving robustness for real-time post-stroke intent detection.

  </details>



- **CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty**  
  Johannes Kirmayr, Lukas Stappen, Elisabeth André  
  _2026-01-29_ · https://arxiv.org/abs/2601.22027v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Existing benchmarks for Large Language Model (LLM) agents focus on task completion under idealistic settings but overlook reliability in real-world, user-facing applications. In domains, such as in-car voice assistants, users often issue incomplete or ambiguous requests, creating intrinsic uncertainty that agents must manage through dialogue, tool use, and policy adherence. We introduce CAR-bench, a benchmark for evaluating consistency, uncertainty handling, and capability awareness in multi-turn, tool-using LLM agents in an in-car assistant domain. The environment features an LLM-simulated user, domain policies, and 58 interconnected tools spanning navigation, productivity, charging, and vehicle control. Beyond standard task completion, CAR-bench introduces Hallucination tasks that test agents' limit-awareness under missing tools or information, and Disambiguation tasks that require resolving uncertainty through clarification or internal information gathering. Baseline results reveal large gaps between occasional and consistent success on all task types. Even frontier reasoning LLMs achieve less than 50% consistent pass rate on Disambiguation tasks due to premature actions, and frequently violate policies or fabricate information to satisfy user requests in Hallucination tasks, underscoring the need for more reliable and self-aware LLM agents in real-world settings.

  </details>



- **Hybrid Foveated Path Tracing with Peripheral Gaussians for Immersive Anatomy**  
  Constantin Kleinbeck, Luisa Theelke, Hannah Schieber, Ulrich Eck, Rüdiger von Eisenhart-Rothe, Daniel Roth  
  _2026-01-29_ · https://arxiv.org/abs/2601.22026v1 · `cs.GR`  
  <details><summary>Abstract</summary>

  Volumetric medical imaging offers great potential for understanding complex pathologies. Yet, traditional 2D slices provide little support for interpreting spatial relationships, forcing users to mentally reconstruct anatomy into three dimensions. Direct volumetric path tracing and VR rendering can improve perception but are computationally expensive, while precomputed representations, like Gaussian Splatting, require planning ahead. Both approaches limit interactive use. We propose a hybrid rendering approach for high-quality, interactive, and immersive anatomical visualization. Our method combines streamed foveated path tracing with a lightweight Gaussian Splatting approximation of the periphery. The peripheral model generation is optimized with volume data and continuously refined using foveal renderings, enabling interactive updates. Depth-guided reprojection further improves robustness to latency and allows users to balance fidelity with refresh rate. We compare our method against direct path tracing and Gaussian Splatting. Our results highlight how their combination can preserve strengths in visual quality while re-generating the peripheral model in under a second, eliminating extensive preprocessing and approximations. This opens new options for interactive medical visualization.

  </details>



- **Learning Transient Convective Heat Transfer with Geometry Aware World Models**  
  Onur T. Doganay, Alexander Klawonn, Martin Eigel, Hanno Gottschalk  
  _2026-01-29_ · https://arxiv.org/abs/2601.22086v1 · `physics.flu-dyn`  
  <details><summary>Abstract</summary>

  Partial differential equation (PDE) simulations are fundamental to engineering and physics but are often computationally prohibitive for real-time applications. While generative AI offers a promising avenue for surrogate modeling, standard video generation architectures lack the specific control and data compatibility required for physical simulations. This paper introduces a geometry aware world model architecture, derived from a video generation architecture (LongVideoGAN), designed to learn transient physics. We introduce two key architecture elements: (1) a twofold conditioning mechanism incorporating global physical parameters and local geometric masks, and (2) an architectural adaptation to support arbitrary channel dimensions, moving beyond standard RGB constraints. We evaluate this approach on a 2D transient computational fluid dynamics (CFD) problem involving convective heat transfer from buoyancy-driven flow coupled to a heat flow in a solid structure. We demonstrate that the conditioned model successfully reproduces complex temporal dynamics and spatial correlations of the training data. Furthermore, we assess the model's generalization capabilities on unseen geometric configurations, highlighting both its potential for controlled simulation synthesis and current limitations in spatial precision for out-of-distribution samples.

  </details>



- **Liquid Interfaces: A Dynamic Ontology for the Interoperability of Autonomous Systems**  
  Dhiogo de Sá, Carlos Schmiedel, Carlos Pereira Lopes  
  _2026-01-29_ · https://arxiv.org/abs/2601.21993v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Contemporary software architectures struggle to support autonomous agents whose reasoning is adaptive, probabilistic, and context-dependent, while system integration remains dominated by static interfaces and deterministic contracts. This paper introduces Liquid Interfaces, a coordination paradigm in which interfaces are not persistent technical artifacts, but ephemeral relational events that emerge through intention articulation and semantic negotiation at runtime.We formalize this model and present the Liquid Interface Protocol (LIP),which governs intention-driven interaction, negotiated execution, and enforce ephemerality under semantic uncertainty. We further discuss the governance implications of this approach and describe a reference architecture that demonstrates practical feasibility. Liquid Interfaces provide a principled foundation for adaptive coordination in agent-based systems

  </details>



- **Geometry of Drifting MDPs with Path-Integral Stability Certificates**  
  Zuyuan Zhang, Mahdi Imani, Tian Lan  
  _2026-01-29_ · https://arxiv.org/abs/2601.21991v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Real-world reinforcement learning is often \emph{nonstationary}: rewards and dynamics drift, accelerate, oscillate, and trigger abrupt switches in the optimal action. Existing theory often represents nonstationarity with coarse-scale models that measure \emph{how much} the environment changes, not \emph{how} it changes locally -- even though acceleration and near-ties drive tracking error and policy chattering. We take a geometric view of nonstationary discounted Markov Decision Processes (MDPs) by modeling the environment as a differentiable homotopy path and tracking the induced motion of the optimal Bellman fixed point. This yields a length-curvature-kink signature of intrinsic complexity: cumulative drift, acceleration/oscillation, and action-gap-induced nonsmoothness. We prove a solver-agnostic path-integral stability bound and derive gap-safe feasible regions that certify local stability away from switch regimes. Building on these results, we introduce \textit{Homotopy-Tracking RL (HT-RL)} and \textit{HT-MCTS}, lightweight wrappers that estimate replay-based proxies of length, curvature, and near-tie proximity online and adapt learning or planning intensity accordingly. Experiments show improved tracking and dynamic regret over matched static baselines, with the largest gains in oscillatory and switch-prone regimes.

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



- **Bridging Forecast Accuracy and Inventory KPIs: A Simulation-Based Software Framework**  
  So Fukuhara, Abdallah Alabdallah, Nuwan Gunasekara, Slawomir Nowaczyk  
  _2026-01-29_ · https://arxiv.org/abs/2601.21844v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Efficient management of spare parts inventory is crucial in the automotive aftermarket, where demand is highly intermittent and uncertainty drives substantial cost and service risks. Forecasting is therefore central, but the quality of a forecasting model should be judged not by statistical accuracy (e.g., MAE, RMSE, IAE) but rather by its impact on key operational performance indicators (KPIs), such as total cost and service level. Yet most existing work evaluates models exclusively using accuracy metrics, and the relationship between these metrics and operational KPIs remains poorly understood. To address this gap, we propose a decision-centric simulation software framework that enables systematic evaluation of forecasting model in realistic inventory management setting. The framework comprises: (i) a synthetic demand generator tailored to spare-parts demand characteristics, (ii) a flexible forecasting module that can host arbitrary predictive models, and (iii) an inventory control simulator that consumes the forecasts and computes operational KPIs. This closed-loop setup enables researchers to evaluate models not only in terms of statistical error but also in terms of their downstream implications for inventory decisions. Using a wide range of simulation scenarios, we show that improvements in conventional accuracy metrics do not necessarily translate into better operational performance, and that models with similar statistical error profiles can induce markedly different cost-service trade-offs. We analyze these discrepancies to characterize how specific aspects of forecast performance affect inventory outcomes and derive guidance for model selection. Overall, the framework operationalizes the link between demand forecasting and inventory management, shifting evaluation from purely predictive accuracy toward operational relevance in the automotive aftermarket and related domains.

  </details>


