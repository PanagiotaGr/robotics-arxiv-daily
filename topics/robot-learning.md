# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-01-16 14:06 UTC_

Total papers shown: **14**


---

- **Service Provisioning and Path Planning with Obstacle Avoidance for Low-Altitude Wireless Networks**  
  Senning Wan, Bin Li, Hongbin Chen, Lei Liu  
  _2026-01-15_ · https://arxiv.org/abs/2601.10179v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  This paper investigates the three-dimensional (3D) deployment of uncrewed aerial vehicles (UAVs) as aerial base stations in heterogeneous communication networks under constraints imposed by diverse ground obstacles. Given the diverse data demands of user equipments (UEs), a user satisfaction model is developed to provide personalized services. In particular, when a UE is located within a ground obstacle, the UAV must approach the obstacle boundary to ensure reliable service quality. Considering constraints such as UAV failures due to battery depletion, heterogeneous UEs, and obstacles, we aim to maximize overall user satisfaction by jointly optimizing the 3D trajectories of UAVs, transmit beamforming vectors, and binary association indicators between UAVs and UEs. To address the complexity and dynamics of the problem, a block coordinate descent method is adopted to decompose it into two subproblems. The beamforming subproblem is efficiently addressed via a bisection-based water-filling algorithm. For the trajectory and association subproblem, we design a deep reinforcement learning algorithm based on proximal policy optimization to learn an adaptive control policy. Simulation results demonstrate that the proposed scheme outperforms baseline schemes in terms of convergence speed and overall system performance. Moreover, it achieves efficient association and accurate obstacle avoidance.

  </details>



- **HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning**  
  Ziang Cui, Mengran Yu, Tianjiao Li, Chenyu Shi, Yingxuan Shi, Lusheng Zhang, Hongwei Lin  
  _2026-01-15_ · https://arxiv.org/abs/2601.10187v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose HOMURA, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a KL-regularized objective with a novel dynamic syllable-ratio reward, HOMURA effectively "tames" the output length. Experimental results demonstrate that our method significantly outperforms strong LLM baselines, achieving precise length control that respects linguistic density hierarchies without compromising semantic adequacy.

  </details>



- **See Less, Drive Better: Generalizable End-to-End Autonomous Driving via Foundation Models Stochastic Patch Selection**  
  Amir Mallak, Erfan Aasi, Shiva Sreeram, Tsun-Hsuan Wang, Daniela Rus, Alaa Maalouf  
  _2026-01-15_ · https://arxiv.org/abs/2601.10707v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advances in end-to-end autonomous driving show that policies trained on patch-aligned features extracted from foundation models generalize better to Out-of-Distribution (OOD). We hypothesize that due to the self-attention mechanism, each patch feature implicitly embeds/contains information from all other patches, represented in a different way and intensity, making these descriptors highly redundant. We quantify redundancy in such (BLIP2) features via PCA and cross-patch similarity: $90$% of variance is captured by $17/64$ principal components, and strong inter-token correlations are pervasive. Training on such overlapping information leads the policy to overfit spurious correlations, hurting OOD robustness. We present Stochastic-Patch-Selection (SPS), a simple yet effective approach for learning policies that are more robust, generalizable, and efficient. For every frame, SPS randomly masks a fraction of patch descriptors, not feeding them to the policy model, while preserving the spatial layout of the remaining patches. Thus, the policy is provided with different stochastic but complete views of the (same) scene: every random subset of patches acts like a different, yet still sensible, coherent projection of the world. The policy thus bases its decisions on features that are invariant to which specific tokens survive. Extensive experiments confirm that across all OOD scenarios, our method outperforms the state of the art (SOTA), achieving a $6.2$% average improvement and up to $20.4$% in closed-loop simulations, while being $2.4\times$ faster. We conduct ablations over masking rates and patch-feature reorganization, training and evaluating 9 systems, with 8 of them surpassing prior SOTA. Finally, we show that the same learned policy transfers to a physical, real-world car without any tuning.

  </details>



- **Breaking Up with Normatively Monolithic Agency with GRACE: A Reason-Based Neuro-Symbolic Architecture for Safe and Ethical AI Alignment**  
  Felix Jahn, Yannic Muskalla, Lisa Dargasz, Patrick Schramowski, Kevin Baum  
  _2026-01-15_ · https://arxiv.org/abs/2601.10520v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  As AI agents become increasingly autonomous, widely deployed in consequential contexts, and efficacious in bringing about real-world impacts, ensuring that their decisions are not only instrumentally effective but also normatively aligned has become critical. We introduce a neuro-symbolic reason-based containment architecture, Governor for Reason-Aligned ContainmEnt (GRACE), that decouples normative reasoning from instrumental decision-making and can contain AI agents of virtually any design. GRACE restructures decision-making into three modules: a Moral Module (MM) that determines permissible macro actions via deontic logic-based reasoning; a Decision-Making Module (DMM) that encapsulates the target agent while selecting instrumentally optimal primitive actions in accordance with derived macro actions; and a Guard that monitors and enforces moral compliance. The MM uses a reason-based formalism providing a semantic foundation for deontic logic, enabling interpretability, contestability, and justifiability. Its symbolic representation enriches the DMM's informational context and supports formal verification and statistical guarantees of alignment enforced by the Guard. We demonstrate GRACE on an example of a LLM therapy assistant, showing how it enables stakeholders to understand, contest, and refine agent behavior.

  </details>



- **Projected Microbatch Accumulation yields reference-free proximal policy updates for reinforcement learning**  
  Nilin Abrahamsen  
  _2026-01-15_ · https://arxiv.org/abs/2601.10498v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  This note introduces Projected Microbatch Accumulation (PROMA), a proximal policy update method for large language model fine-tuning. PROMA accumulates policy gradients across microbatches by projecting out sequence-wise gradient components before microbatch aggregation. The projection is applied layer-wise during the backward pass, enabling efficient implementation without additional forward or backward passes. Empirically, PROMA enforces tighter control of local KL divergence than GRPO, resulting in more stable policy learning. Unlike PPO and GRPO, PROMA achieves proximal updates without inducing entropy collapse and does not rely on a reference policy or likelihood-ratio clipping.

  </details>



- **NSR-Boost: A Neuro-Symbolic Residual Boosting Framework for Industrial Legacy Models**  
  Ziming Dai, Dabiao Ma, Jinle Tong, Mengyuan Han, Jian Yang, Haojun Fei  
  _2026-01-15_ · https://arxiv.org/abs/2601.10457v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Although the Gradient Boosted Decision Trees (GBDTs) dominate industrial tabular applications, upgrading legacy models in high-concurrency production environments still faces prohibitive retraining costs and systemic risks. To address this problem, we present NSR-Boost, a neuro-symbolic residual boosting framework designed specifically for industrial scenarios. Its core advantage lies in being "non-intrusive". It treats the legacy model as a frozen model and performs targeted repairs on "hard regions" where predictions fail. The framework comprises three key stages: first, finding hard regions through residuals, then generating interpretable experts by generating symbolic code structures using Large Language Model (LLM) and fine-tuning parameters using Bayesian optimization, and finally dynamically integrating experts with legacy model output through a lightweight aggregator. We report on the successful deployment of NSR-Boost within the core financial risk control system at Qfin Holdings. This framework not only significantly outperforms state-of-the-art (SOTA) baselines across six public datasets and one private dataset, more importantly, shows excellent performance gains on real-world online data. In conclusion, it effectively captures long-tail risks missed by traditional models and offers a safe, low-cost evolutionary paradigm for industry.

  </details>



- **Reinforcement Learning with Multi-Step Lookahead Information Via Adaptive Batching**  
  Nadav Merlis  
  _2026-01-15_ · https://arxiv.org/abs/2601.10418v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We study tabular reinforcement learning problems with multiple steps of lookahead information. Before acting, the learner observes $\ell$ steps of future transition and reward realizations: the exact state the agent would reach and the rewards it would collect under any possible course of action. While it has been shown that such information can drastically boost the value, finding the optimal policy is NP-hard, and it is common to apply one of two tractable heuristics: processing the lookahead in chunks of predefined sizes ('fixed batching policies'), and model predictive control. We first illustrate the problems with these two approaches and propose utilizing the lookahead in adaptive (state-dependent) batches; we refer to such policies as adaptive batching policies (ABPs). We derive the optimal Bellman equations for these strategies and design an optimistic regret-minimizing algorithm that enables learning the optimal ABP when interacting with unknown environments. Our regret bounds are order-optimal up to a potential factor of the lookahead horizon $\ell$, which can usually be considered a small constant.

  </details>



- **An effective interactive brain cytoarchitectonic parcellation framework using pretrained foundation model**  
  Shiqi Zhang, Fang Xu, Pengcheng Zhou  
  _2026-01-15_ · https://arxiv.org/abs/2601.10412v1 · `eess.IV`  
  <details><summary>Abstract</summary>

  Cytoarchitectonic mapping provides anatomically grounded parcellations of brain structure and forms a foundation for integrative, multi-modal neuroscience analyses. These parcellations are defined based on the shape, density, and spatial arrangement of neuronal cell bodies observed in histological imaging. Recent works have demonstrated the potential of using deep learning models toward fully automatic segmentation of cytoarchitectonic areas in large-scale datasets, but performance is mainly constrained by the scarcity of training labels and the variability of staining and imaging conditions. To address these challenges, we propose an interactive cytoarchitectonic parcellation framework that leverages the strong transferability of the DINOv3 vision transformer. Our framework combines (i) multi-layer DINOv3 feature fusion, (ii) a lightweight segmentation decoder, and (iii) real-time user-guided training from sparse scribbles. This design enables rapid human-in-the-loop refinement while maintaining high segmentation accuracy. Compared with training an nnU-Net from scratch, transfer learning with DINOv3 yields markedly improved performance. We also show that features extracted by DINOv3 exhibit clear anatomical correspondence and demonstrate the method's practical utility for brain region segmentation using sparse labels. These results highlight the potential of foundation-model-driven interactive segmentation for scalable and efficient cytoarchitectonic mapping.

  </details>



- **FastStair: Learning to Run Up Stairs with Humanoid Robots**  
  Yan Liu, Tao Yu, Haolin Song, Hongbo Zhu, Nianzong Hu, Yuzhi Hao, Xiuyong Yao, Xizhe Zang, Hua Chen, Jie Zhao  
  _2026-01-15_ · https://arxiv.org/abs/2601.10365v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Running up stairs is effortless for humans but remains extremely challenging for humanoid robots due to the simultaneous requirements of high agility and strict stability. Model-free reinforcement learning (RL) can generate dynamic locomotion, yet implicit stability rewards and heavy reliance on task-specific reward shaping tend to result in unsafe behaviors, especially on stairs; conversely, model-based foothold planners encode contact feasibility and stability structure, but enforcing their hard constraints often induces conservative motion that limits speed. We present FastStair, a planner-guided, multi-stage learning framework that reconciles these complementary strengths to achieve fast and stable stair ascent. FastStair integrates a parallel model-based foothold planner into the RL training loop to bias exploration toward dynamically feasible contacts and to pretrain a safety-focused base policy. To mitigate planner-induced conservatism and the discrepancy between low- and high-speed action distributions, the base policy was fine-tuned into speed-specialized experts and then integrated via Low-Rank Adaptation (LoRA) to enable smooth operation across the full commanded-speed range. We deploy the resulting controller on the Oli humanoid robot, achieving stable stair ascent at commanded speeds up to 1.65 m/s and traversing a 33-step spiral staircase (17 cm rise per step) in 12 s, demonstrating robust high-speed performance on long staircases. Notably, the proposed approach served as the champion solution in the Canton Tower Robot Run Up Competition.

  </details>



- **The impact of tactile sensor configurations on grasp learning efficiency -- a comparative evaluation in simulation**  
  Eszter Birtalan, Miklós Koller  
  _2026-01-15_ · https://arxiv.org/abs/2601.10268v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Tactile sensors are breaking into the field of robotics to provide direct information related to contact surfaces, including contact events, slip events and even texture identification. These events are especially important for robotic hand designs, including prosthetics, as they can greatly improve grasp stability. Most presently published robotic hand designs, however, implement them in vastly different densities and layouts on the hand surface, often reserving the majority of the available space. We used simulations to evaluate 6 different tactile sensor configurations with different densities and layouts, based on their impact on reinforcement learning. Our two-setup system allows for robust results that are not dependent on the use of a given physics simulator, robotic hand model or machine learning algorithm. Our results show setup-specific, as well as generalized effects across the 6 sensorized simulations, and we identify one configuration as consistently yielding the best performance across both setups. These results could help future research aimed at robotic hand designs, including prostheses.

  </details>



- **DecisionLLM: Large Language Models for Long Sequence Decision Exploration**  
  Xiaowei Lv, Zhilin Zhang, Yijun Li, Yusen Huo, Siyuan Ju, Xuyan Li, Chunxiang Hong, Tianyu Wang, Yongcai Wang, Peng Sun, et al.  
  _2026-01-15_ · https://arxiv.org/abs/2601.10148v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Long-sequence decision-making, which is usually addressed through reinforcement learning (RL), is a critical component for optimizing strategic operations in dynamic environments, such as real-time bidding in computational advertising. The Decision Transformer (DT) introduced a powerful paradigm by framing RL as an autoregressive sequence modeling problem. Concurrently, Large Language Models (LLMs) have demonstrated remarkable success in complex reasoning and planning tasks. This inspires us whether LLMs, which share the same Transformer foundation, but operate at a much larger scale, can unlock new levels of performance in long-horizon sequential decision-making problem. This work investigates the application of LLMs to offline decision making tasks. A fundamental challenge in this domain is the LLMs' inherent inability to interpret continuous values, as they lack a native understanding of numerical magnitude and order when values are represented as text strings. To address this, we propose treating trajectories as a distinct modality. By learning to align trajectory data with natural language task descriptions, our model can autoregressively predict future decisions within a cohesive framework we term DecisionLLM. We establish a set of scaling laws governing this paradigm, demonstrating that performance hinges on three factors: model scale, data volume, and data quality. In offline experimental benchmarks and bidding scenarios, DecisionLLM achieves strong performance. Specifically, DecisionLLM-3B outperforms the traditional Decision Transformer (DT) by 69.4 on Maze2D umaze-v1 and by 0.085 on AuctionNet. It extends the AIGB paradigm and points to promising directions for future exploration in online bidding.

  </details>



- **History Is Not Enough: An Adaptive Dataflow System for Financial Time-Series Synthesis**  
  Haochong Xia, Yao Long Teng, Regan Tan, Molei Qin, Xinrun Wang, Bo An  
  _2026-01-15_ · https://arxiv.org/abs/2601.10143v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  In quantitative finance, the gap between training and real-world performance-driven by concept drift and distributional non-stationarity-remains a critical obstacle for building reliable data-driven systems. Models trained on static historical data often overfit, resulting in poor generalization in dynamic markets. The mantra "History Is Not Enough" underscores the need for adaptive data generation that learns to evolve with the market rather than relying solely on past observations. We present a drift-aware dataflow system that integrates machine learning-based adaptive control into the data curation process. The system couples a parameterized data manipulation module comprising single-stock transformations, multi-stock mix-ups, and curation operations, with an adaptive planner-scheduler that employs gradient-based bi-level optimization to control the system. This design unifies data augmentation, curriculum learning, and data workflow management under a single differentiable framework, enabling provenance-aware replay and continuous data quality monitoring. Extensive experiments on forecasting and reinforcement learning trading tasks demonstrate that our framework enhances model robustness and improves risk-adjusted returns. The system provides a generalizable approach to adaptive data management and learning-guided workflow automation for financial data.

  </details>



- **Role-Playing Agents Driven by Large Language Models: Current Status, Challenges, and Future Trends**  
  Ye Wang, Jiaxing Chen, Hongjiang Xiao  
  _2026-01-15_ · https://arxiv.org/abs/2601.10122v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  In recent years, with the rapid advancement of large language models (LLMs), role-playing language agents (RPLAs) have emerged as a prominent research focus at the intersection of natural language processing (NLP) and human-computer interaction. This paper systematically reviews the current development and key technologies of RPLAs, delineating the technological evolution from early rule-based template paradigms, through the language style imitation stage, to the cognitive simulation stage centered on personality modeling and memory mechanisms. It summarizes the critical technical pathways supporting high-quality role-playing, including psychological scale-driven character modeling, memory-augmented prompting mechanisms, and motivation-situation-based behavioral decision control. At the data level, the paper further analyzes the methods and challenges of constructing role-specific corpora, focusing on data sources, copyright constraints, and structured annotation processes. In terms of evaluation, it collates multi-dimensional assessment frameworks and benchmark datasets covering role knowledge, personality fidelity, value alignment, and interactive hallucination, while commenting on the advantages and disadvantages of methods such as human evaluation, reward models, and LLM-based scoring. Finally, the paper outlines future development directions of role-playing agents, including personality evolution modeling, multi-agent collaborative narrative, multimodal immersive interaction, and integration with cognitive neuroscience, aiming to provide a systematic perspective and methodological insights for subsequent research.

  </details>



- **MATRIX AS PLAN: Structured Logical Reasoning with Feedback-Driven Replanning**  
  Ke Chen, Jiandian Zeng, Zihao Peng, Guo Li, Guangxue Zhang, Tian Wang  
  _2026-01-15_ · https://arxiv.org/abs/2601.10101v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  As knowledge and semantics on the web grow increasingly complex, enhancing Large Language Models (LLMs) comprehension and reasoning capabilities has become particularly important. Chain-of-Thought (CoT) prompting has been shown to enhance the reasoning capabilities of LLMs. However, it still falls short on logical reasoning tasks that rely on symbolic expressions and strict deductive rules. Neuro-symbolic methods address this gap by enforcing formal correctness through external solvers. Yet these solvers are highly format-sensitive, and small instabilities in model outputs can lead to frequent processing failures. LLM-driven approaches avoid parsing brittleness, but they lack structured representations and process-level error-correction mechanisms. To further enhance the logical reasoning capabilities of LLMs, we propose MatrixCoT, a structured CoT framework with a matrix-based plan. Specifically, we normalize and type natural language expressions, attach explicit citation fields, and introduce a matrix-based planning method to preserve global relations among steps. The plan becomes a verifiable artifact, making execution more stable. For verification, we also add a feedback-driven replanning mechanism. Under semantic-equivalence constraints, it identifies omissions and defects, rewrites and compresses the dependency matrix, and produces a more trustworthy final answer. Experiments on five logical-reasoning benchmarks and five LLMs show that, without relying on external solvers, MatrixCoT enhances both robustness and interpretability when tackling complex symbolic reasoning tasks, while maintaining competitive performance.

  </details>


