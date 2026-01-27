# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-01-27 06:52 UTC_

Total papers shown: **7**


---

- **$α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks**  
  Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah  
  _2026-01-26_ · https://arxiv.org/abs/2601.18754v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Autonomous unmanned aerial vehicle (UAV) systems are increasingly deployed in safety-critical, networked environments where they must operate reliably in the presence of malicious adversaries. While recent benchmarks have evaluated large language model (LLM)-based UAV agents in reasoning, navigation, and efficiency, systematic assessment of security, resilience, and trust under adversarial conditions remains largely unexplored, particularly in emerging 6G-enabled settings. We introduce $α^{3}$-SecBench, the first large-scale evaluation suite for assessing the security-aware autonomy of LLM-based UAV agents under realistic adversarial interference. Building on multi-turn conversational UAV missions from $α^{3}$-Bench, the framework augments benign episodes with 20,000 validated security overlay attack scenarios targeting seven autonomy layers, including sensing, perception, planning, control, communication, edge/cloud infrastructure, and LLM reasoning. $α^{3}$-SecBench evaluates agents across three orthogonal dimensions: security (attack detection and vulnerability attribution), resilience (safe degradation behavior), and trust (policy-compliant tool usage). We evaluate 23 state-of-the-art LLMs from major industrial providers and leading AI labs using thousands of adversarially augmented UAV episodes sampled from a corpus of 113,475 missions spanning 175 threat types. While many models reliably detect anomalous behavior, effective mitigation, vulnerability attribution, and trustworthy control actions remain inconsistent. Normalized overall scores range from 12.9% to 57.1%, highlighting a significant gap between anomaly detection and security-aware autonomous decision-making. We release $α^{3}$-SecBench on GitHub: https://github.com/maferrag/AlphaSecBench

  </details>



- **Uncooled Poisson Bolometer for High-Speed Event-Based Long-wave Thermal Imaging**  
  Mohamed A. Mousa, Leif Bauer, Utkarsh Singh, Ziyi Yang, Angshuman Deka, Zubin Jacob  
  _2026-01-26_ · https://arxiv.org/abs/2601.18583v1 · `physics.optics`  
  <details><summary>Abstract</summary>

  Event-based vision provides high-speed, energy-efficient sensing for applications such as autonomous navigation and motion tracking. However, implementing this technology in the long-wave infrared remains a significant challenge. Traditional infrared sensors are hindered by slow thermal response times or the heavy power requirements of cryogenic cooling. Here, we introduce the first event-based infrared detector operating in a Poisson-counting regime. This is realized with a spintronic Poisson bolometer capable of broadband detection from 0.8-14$μ\text{m}$. In this regime, infrared signals are detected through statistically resolvable changes in stochastic switching events. This approach enables room-temperature operation with high timing resolution. Our device achieves a maximum event rate of 1,250 Hz, surpassing the temporal resolution of conventional uncooled microbolometers by a factor of 4. Power consumption is kept low at 0.2$μ$W per pixel. This work establishes an operating principle for infrared sensing and demonstrates a pathway toward high-speed, energy-efficient, event-driven thermal imaging.

  </details>



- **Data-Driven Qubit Characterization and Optimal Control using Deep Learning**  
  Paul Surrey, Julian D. Teske, Tobias Hangleiter, Hendrik Bluhm, Pascal Cerfontaine  
  _2026-01-26_ · https://arxiv.org/abs/2601.18704v1 · `quant-ph`  
  <details><summary>Abstract</summary>

  Quantum computing requires the optimization of control pulses to achieve high-fidelity quantum gates. We propose a machine learning-based protocol to address the challenges of evaluating gradients and modeling complex system dynamics. By training a recurrent neural network (RNN) to predict qubit behavior, our approach enables efficient gradient-based pulse optimization without the need for a detailed system model. First, we sample qubit dynamics using random control pulses with weak prior assumptions. We then train the RNN on the system's observed responses, and use the trained model to optimize high-fidelity control pulses. We demonstrate the effectiveness of this approach through simulations on a single $ST_0$ qubit.

  </details>



- **Fast and Safe Trajectory Optimization for Mobile Manipulators With Neural Configuration Space Distance Field**  
  Yulin Li, Zhiyuan Song, Yiming Li, Zhicheng Song, Kai Chen, Chunxin Zheng, Zhihai Bi, Jiahang Cao, Sylvain Calinon, Fan Shi, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18548v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mobile manipulators promise agile, long-horizon behavior by coordinating base and arm motion, yet whole-body trajectory optimization in cluttered, confined spaces remains difficult due to high-dimensional nonconvexity and the need for fast, accurate collision reasoning. Configuration Space Distance Fields (CDF) enable fixed-base manipulators to model collisions directly in configuration space via smooth, implicit distances. This representation holds strong potential to bypass the nonlinear configuration-to-workspace mapping while preserving accurate whole-body geometry and providing optimization-friendly collision costs. Yet, extending this capability to mobile manipulators is hindered by unbounded workspaces and tighter base-arm coupling. We lift this promise to mobile manipulation with Generalized Configuration Space Distance Fields (GCDF), extending CDF to robots with both translational and rotational joints in unbounded workspaces with tighter base-arm coupling. We prove that GCDF preserves Euclidean-like local distance structure and accurately encodes whole-body geometry in configuration space, and develop a data generation and training pipeline that yields continuous neural GCDFs with accurate values and gradients, supporting efficient GPU-batched queries. Building on this representation, we develop a high-performance sequential convex optimization framework centered on GCDF-based collision reasoning. The solver scales to large numbers of implicit constraints through (i) online specification of neural constraints, (ii) sparsity-aware active-set detection with parallel batched evaluation across thousands of constraints, and (iii) incremental constraint management for rapid replanning under scene changes.

  </details>



- **SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction**  
  Linyong Gan, Zimo Li, Wenxin Xu, Xingjian Li, Jianhua Z. Huang, Enmei Tu, Shuhang Chen  
  _2026-01-26_ · https://arxiv.org/abs/2601.18537v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate long-horizon vessel trajectory prediction remains challenging due to compounded uncertainty from complex navigation behaviors and environmental factors. Existing methods often struggle to maintain global directional consistency, leading to drifting or implausible trajectories when extrapolated over long time horizons. To address this issue, we propose a semantic-key-point-conditioned trajectory modeling framework, in which future trajectories are predicted by conditioning on a high-level Next Key Point (NKP) that captures navigational intent. This formulation decomposes long-horizon prediction into global semantic decision-making and local motion modeling, effectively restricting the support of future trajectories to semantically feasible subsets. To efficiently estimate the NKP prior from historical observations, we adopt a pretrain-finetune strategy. Extensive experiments on real-world AIS data demonstrate that the proposed method consistently outperforms state-of-the-art approaches, particularly for long travel durations, directional accuracy, and fine-grained trajectory prediction.

  </details>



- **DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation**  
  Zijun Li, Shijie Li, Zhenxi Zhang, Bin Li, Shoujun Zhou  
  _2026-01-26_ · https://arxiv.org/abs/2601.18492v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-and-Language Navigation (VLN) requires an embodied agent to navigate in a complex 3D environment according to natural language instructions. Recent progress in large language models (LLMs) has enabled language-driven navigation with improved interpretability. However, most LLM-based agents still rely on single-shot action decisions, where the model must choose one option from noisy, textualized multi-perspective observations. Due to local mismatches and imperfect intermediate reasoning, such decisions can easily deviate from the correct path, leading to error accumulation and reduced reliability in unseen environments. In this paper, we propose DV-VLN, a new VLN framework that follows a generate-then-verify paradigm. DV-VLN first performs parameter-efficient in-domain adaptation of an open-source LLaMA-2 backbone to produce a structured navigational chain-of-thought, and then verifies candidate actions with two complementary channels: True-False Verification (TFV) and Masked-Entity Verification (MEV). DV-VLN selects actions by aggregating verification successes across multiple samples, yielding interpretable scores for reranking. Experiments on R2R, RxR (English subset), and REVERIE show that DV-VLN consistently improves over direct prediction and sampling-only baselines, achieving competitive performance among language-only VLN agents and promising results compared with several cross-modal systems.Code is available at https://github.com/PlumJun/DV-VLN.

  </details>



- **Convex Chance-Constrained Stochastic Control under Uncertain Specifications with Application to Learning-Based Hybrid Powertrain Control**  
  Teruki Kato, Ryotaro Shima, Kenji Kashima  
  _2026-01-26_ · https://arxiv.org/abs/2601.18313v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  This paper presents a strictly convex chance-constrained stochastic control framework that accounts for uncertainty in control specifications such as reference trajectories and operational constraints. By jointly optimizing control inputs and risk allocation under general (possibly non-Gaussian) uncertainties, the proposed method guarantees probabilistic constraint satisfaction while ensuring strict convexity, leading to uniqueness and continuity of the optimal solution. The formulation is further extended to nonlinear model-based control using exactly linearizable models identified through machine learning. The effectiveness of the proposed approach is demonstrated through model predictive control applied to a hybrid powertrain system.

  </details>


