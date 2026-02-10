# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-10 07:19 UTC_

Total papers shown: **11**


---

- **GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion**  
  Santiago Montiel-Marín, Miguel Antunes-García, Fabio Sánchez-García, Angel Llamazares, Holger Caesar, Luis M. Bergasa  
  _2026-02-09_ · https://arxiv.org/abs/2602.08784v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robust and accurate perception of dynamic objects and map elements is crucial for autonomous vehicles performing safe navigation in complex traffic scenarios. While vision-only methods have become the de facto standard due to their technical advances, they can benefit from effective and cost-efficient fusion with radar measurements. In this work, we advance fusion methods by repurposing Gaussian Splatting as an efficient universal view transformer that bridges the view disparity gap, mapping both image pixels and radar points into a common Bird's-Eye View (BEV) representation. Our main contribution is GaussianCaR, an end-to-end network for BEV segmentation that, unlike prior BEV fusion methods, leverages Gaussian Splatting to map raw sensor information into latent features for efficient camera-radar fusion. Our architecture combines multi-scale fusion with a transformer decoder to efficiently extract BEV features. Experimental results demonstrate that our approach achieves performance on par with, or even surpassing, the state of the art on BEV segmentation tasks (57.3%, 82.9%, and 50.1% IoU for vehicles, roads, and lane dividers) on the nuScenes dataset, while maintaining a 3.2x faster inference runtime. Code and project page are available online.

  </details>



- **Intermediate Results on the Complexity of STRIPS$_{1}^{1}$**  
  Stefan Edelkamp, Jiří Fink, Petr Gregor, Anders Jonsson, Bernhard Nebel  
  _2026-02-09_ · https://arxiv.org/abs/2602.08708v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  This paper is based on Bylander's results on the computational complexity of propositional STRIPS planning. He showed that when only ground literals are permitted, determining plan existence is PSPACE-complete even if operators are limited to two preconditions and two postconditions. While NP-hardness is settled, it is unknown whether propositional STRIPS with operators that only have one precondition and one effect is NP-complete. We shed light on the question whether this small solution hypothesis for STRIPS$^1_1$ is true, calling a SAT solver for small instances, introducing the literal graph, and mapping it to Petri nets.

  </details>



- **SemiNFT: Learning to Transfer Presets from Imitation to Appreciation via Hybrid-Sample Reinforcement Learning**  
  Melany Yang, Yuhang Yu, Diwang Weng, Jinwei Chen, Wei Dong  
  _2026-02-09_ · https://arxiv.org/abs/2602.08582v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Photorealistic color retouching plays a vital role in visual content creation, yet manual retouching remains inaccessible to non-experts due to its reliance on specialized expertise. Reference-based methods offer a promising alternative by transferring the preset color of a reference image to a source image. However, these approaches often operate as novice learners, performing global color mappings derived from pixel-level statistics, without a true understanding of semantic context or human aesthetics. To address this issue, we propose SemiNFT, a Diffusion Transformer (DiT)-based retouching framework that mirrors the trajectory of human artistic training: beginning with rigid imitation and evolving into intuitive creation. Specifically, SemiNFT is first taught with paired triplets to acquire basic structural preservation and color mapping skills, and then advanced to reinforcement learning (RL) on unpaired data to cultivate nuanced aesthetic perception. Crucially, during the RL stage, to prevent catastrophic forgetting of old skills, we design a hybrid online-offline reward mechanism that anchors aesthetic exploration with structural review. % experiments Extensive experiments show that SemiNFT not only outperforms state-of-the-art methods on standard preset transfer benchmarks but also demonstrates remarkable intelligence in zero-shot tasks, such as black-and-white photo colorization and cross-domain (anime-to-photo) preset transfer. These results confirm that SemiNFT transcends simple statistical matching and achieves a sophisticated level of aesthetic comprehension. Our project can be found at https://melanyyang.github.io/SemiNFT/.

  </details>



- **GEBench: Benchmarking Image Generation Models as GUI Environments**  
  Haodong Li, Jingwei Wu, Quan Sun, Guopeng Li, Juanxi Tian, Huanyu Zhang, Yanlin Lai, Ruichuan An, Hongbo Peng, Yuhong Dai, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.09007v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Recent advancements in image generation models have enabled the prediction of future Graphical User Interface (GUI) states based on user instructions. However, existing benchmarks primarily focus on general domain visual fidelity, leaving the evaluation of state transitions and temporal coherence in GUI-specific contexts underexplored. To address this gap, we introduce GEBench, a comprehensive benchmark for evaluating dynamic interaction and temporal coherence in GUI generation. GEBench comprises 700 carefully curated samples spanning five task categories, covering both single-step interactions and multi-step trajectories across real-world and fictional scenarios, as well as grounding point localization. To support systematic evaluation, we propose GE-Score, a novel five-dimensional metric that assesses Goal Achievement, Interaction Logic, Content Consistency, UI Plausibility, and Visual Quality. Extensive evaluations on current models indicate that while they perform well on single-step transitions, they struggle significantly with maintaining temporal coherence and spatial grounding over longer interaction sequences. Our findings identify icon interpretation, text rendering, and localization precision as critical bottlenecks. This work provides a foundation for systematic assessment and suggests promising directions for future research toward building high-fidelity generative GUI environments. The code is available at: https://github.com/stepfun-ai/GEBench.

  </details>



- **Diffusion-Inspired Reconfiguration of Transformers for Uncertainty Calibration**  
  Manh Cuong Dao, Quang Hung Pham, Phi Le Nguyen, Thao Nguyen Truong, Bryan Kian Hsiang Low, Trong Nghia Hoang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08920v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Uncertainty calibration in pre-trained transformers is critical for their reliable deployment in risk-sensitive applications. Yet, most existing pre-trained transformers do not have a principled mechanism for uncertainty propagation through their feature transformation stack. In this work, we propose a diffusion-inspired reconfiguration of transformers in which each feature transformation block is modeled as a probabilistic mapping. Composing these probabilistic mappings reveals a probability path that mimics the structure of a diffusion process, transporting data mass from the input distribution to the pre-trained feature distribution. This probability path can then be recompiled on a diffusion process with a unified transition model to enable principled propagation of representation uncertainty throughout the pre-trained model's architecture while maintaining its original predictive performance. Empirical results across a variety of vision and language benchmarks demonstrate that our method achieves superior calibration and predictive accuracy compared to existing uncertainty-aware transformers.

  </details>



- **AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection**  
  Junru Zhang, Lang Feng, Haoran Shi, Xu Guo, Han Yu, Yabo Dong, Duanqing Xu  
  _2026-02-09_ · https://arxiv.org/abs/2602.08868v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Time-series anomaly detection (TSAD) with multimodal large language models (MLLMs) is an emerging area, yet a persistent challenge remains: MLLMs rely on coarse time-series heuristics but struggle with multi-dimensional, detailed reasoning, which is vital for understanding complex time-series data. We present AnomSeer to address this by reinforcing the model to ground its reasoning in precise, structural details of time series, unifying anomaly classification, localization, and explanation. At its core, an expert chain-of-thought trace is generated to provide a verifiable, fine-grained reasoning from classical analyses (e.g., statistical measures, frequency transforms). Building on this, we propose a novel time-series grounded policy optimization (TimerPO) that incorporates two additional components beyond standard reinforcement learning: a time-series grounded advantage based on optimal transport and an orthogonal projection to ensure this auxiliary granular signal does not interfere with the primary detection objective. Across diverse anomaly scenarios, AnomSeer, with Qwen2.5-VL-3B/7B-Instruct, outperforms larger commercial baselines (e.g., GPT-4o) in classification and localization accuracy, particularly on point- and frequency-driven exceptions. Moreover, it produces plausible time-series reasoning traces that support its conclusions.

  </details>



- **Kirin: Improving ANN efficiency with SNN Hybridization**  
  Chenyu Wang, Zhanglu Yan, Zhi Zhou, Xu Chen, Weng-Fai Wong  
  _2026-02-09_ · https://arxiv.org/abs/2602.08817v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Artificial neural networks (ANNs), particularly large language models (LLMs), demonstrate powerful inference capabilities but consume substantial energy. Conversely, spiking neural networks (SNNs) exhibit exceptional energy efficiency due to their binary and event-driven characteristics, thus motivating the study of ANN-to-SNN conversion. In this process, quantization plays a pivotal role, mapping LLMs' floating-point parameters to discrete SNN parameters via the temporal dimension of the time window. However, several challenges remain in the conversion process: (i) converting high bit-width quantization values into binary spikes requires longer time windows, increasing system latency; and (ii) the inherent trade-off between the information loss of single-spike schemes and the energy costs of multi-spike ones in SNN. To address these challenges, we propose Kirin, a integer and spike hybrid based SNN to achieve accuracy lossless ANN-to-SNN conversion with time and energy efficiency. Specifically, we first propose a Spike Matrix Hybridization strategy that encoding low bit-width parameters that leading to small time window size into binary spikes while preserving the rest in integer format, thereby reducing the overall latency of SNN execution. Second, we introduce a silence threshold mechanism to regulate the timing of single-spike firing, ensuring the output is mathematically equivalent to the LLM's output and preserves accuracy. Experimental results demonstrate that Kirin, under a W4A4\&8 quantization setting, achieves near-FP16 accuracy while reducing energy consumption by up to 84.66\% and shortening time steps by 93.75\%.

  </details>



- **Root Cause Analysis Method Based on Large Language Models with Residual Connection Structures**  
  Liming Zhou, Ailing Liu, Hongwei Liu, Min He, Heng Zhang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08804v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Root cause localization remain challenging in complex and large-scale microservice architectures. The complex fault propagation among microservices and the high dimensionality of telemetry data, including metrics, logs, and traces, limit the effectiveness of existing root cause analysis (RCA) methods. In this paper, a residual-connection-based RCA method using large language model (LLM), named RC-LLM, is proposed. A residual-like hierarchical fusion structure is designed to integrate multi-source telemetry data, while the contextual reasoning capability of large language models is leveraged to model temporal and cross-microservice causal dependencies. Experimental results on CCF-AIOps microservice datasets demonstrate that RC-LLM achieves strong accuracy and efficiency in root cause analysis.

  </details>



- **FreqLens: Interpretable Frequency Attribution for Time Series Forecasting**  
  Chi-Sheng Chen, Xinyu Zhang, En-Jui Kuo, Guan-Ying Chen, Qiuzhe Xie, Fan Zhang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08768v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Time series forecasting models often lack interpretability, limiting their adoption in domains requiring explainable predictions. We propose \textsc{FreqLens}, an interpretable forecasting framework that discovers and attributes predictions to learnable frequency components. \textsc{FreqLens} introduces two key innovations: (1) \emph{learnable frequency discovery} -- frequency bases are parameterized via sigmoid mapping and learned from data with diversity regularization, enabling automatic discovery of dominant periodic patterns without domain knowledge; and (2) \emph{axiomatic frequency attribution} -- a theoretically grounded framework that provably satisfies Completeness, Faithfulness, Null-Frequency, and Symmetry axioms, with per-frequency attributions equivalent to Shapley values. On Traffic and Weather datasets, \textsc{FreqLens} achieves competitive or superior performance while discovering physically meaningful frequencies: all 5 independent runs discover the 24-hour daily cycle ($24.6 \pm 0.1$h, 2.5\% error) and 12-hour half-daily cycle ($11.8 \pm 0.1$h, 1.6\% error) on Traffic, and weekly cycles ($10\times$ longer than the input window) on Weather. These results demonstrate genuine frequency-level knowledge discovery with formal theoretical guarantees on attribution quality.

  </details>



- **Reasoning aligns language models to human cognition**  
  Gonçalo Guiomar, Elia Torre, Pehuen Moure, Victoria Shavina, Mario Giulianelli, Shih-Chii Liu, Valerio Mante  
  _2026-02-09_ · https://arxiv.org/abs/2602.08693v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Do language models make decisions under uncertainty like humans do, and what role does chain-of-thought (CoT) reasoning play in the underlying decision process? We introduce an active probabilistic reasoning task that cleanly separates sampling (actively acquiring evidence) from inference (integrating evidence toward a decision). Benchmarking humans and a broad set of contemporary large language models against near-optimal reference policies reveals a consistent pattern: extended reasoning is the key determinant of strong performance, driving large gains in inference and producing belief trajectories that become strikingly human-like, while yielding only modest improvements in active sampling. To explain these differences, we fit a mechanistic model that captures systematic deviations from optimal behavior via four interpretable latent variables: memory, strategy, choice bias, and occlusion awareness. This model places humans and models in a shared low-dimensional cognitive space, reproduces behavioral signatures across agents, and shows how chain-of-thought shifts language models toward human-like regimes of evidence accumulation and belief-to-choice mapping, tightening alignment in inference while leaving a persistent gap in information acquisition.

  </details>



- **Ziv-Zakai Bound for Near-Field Localization and Sensing**  
  Nicolò Decarli, Davide Dardari  
  _2026-02-09_ · https://arxiv.org/abs/2602.08609v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  The increasing carrier frequencies and growing physical dimensions of antenna arrays in modern wireless systems are driving renewed interest in localization and sensing under near-field conditions. In this paper, we analyze the Ziv-Zakai Bound (ZZB) for near-field localization and sensing operated with large antenna arrays, which offers a tighter characterization of estimation accuracy compared to traditional bounds such as the Cramér-Rao Bound (CRB), especially in low signal-to-noise ratio or threshold regions. Leveraging spherical wavefront and array geometry in the signal model, we evaluate the ZZB for distance and angle estimation, investigating the dependence of the accuracy on key signal and system parameters such as array geometry, wavelength, and target position. Our analysis highlights the transition behavior of the ZZB and underscores the fundamental limitations and opportunities for accurate near-field sensing.

  </details>


