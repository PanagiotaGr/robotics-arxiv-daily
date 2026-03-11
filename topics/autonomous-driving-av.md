# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-03-11 07:08 UTC_

Total papers shown: **9**


---

- **Open-World Motion Forecasting**  
  Nicolas Schischka, Nikhil Gosala, B Ravi Kiran, Senthil Yogamani, Abhinav Valada  
  _2026-03-10_ · https://arxiv.org/abs/2603.09420v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Motion forecasting aims to predict the future trajectories of dynamic agents in the scene, enabling autonomous vehicles to effectively reason about scene evolution. Existing approaches operate under the closed-world regime and assume fixed object taxonomy as well as access to high-quality perception. Therefore, they struggle in real-world settings where perception is imperfect and object taxonomy evolves over time. In this work, we bridge this fundamental gap by introducing open-world motion forecasting, a novel setting in which new object classes are sequentially introduced over time and future object trajectories are estimated directly from camera images. We tackle this setting by proposing the first end-to-end class-incremental motion forecasting framework to mitigate catastrophic forgetting while simultaneously learning to forecast newly introduced classes. When a new class is introduced, our framework employs a pseudo-labeling strategy to first generate motion forecasting pseudo-labels for all known classes which are then processed by a vision-language model to filter inconsistent and over-confident predictions. Parallelly, our approach further mitigates catastrophic forgetting by using a novel replay sampling strategy that leverages query feature variance to sample previous sequences with informative motion patterns. Extensive evaluation on the nuScenes and Argoverse 2 datasets demonstrates that our approach successfully resists catastrophic forgetting and maintains performance on previously learned classes while improving adaptation to novel ones. Further, we demonstrate that our approach supports zero-shot transfer to real-world driving and naturally extends to end-to-end class-incremental planning, enabling continual adaptation of the full autonomous driving system. We provide the code at https://omen.cs.uni-freiburg.de .

  </details>



- **RESBev: Making BEV Perception More Robust**  
  Lifeng Zhuo, Kefan Jin, Zhe Liu, Hesheng Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09529v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Bird's-eye-view (BEV) perception has emerged as a cornerstone of autonomous driving systems, providing a structured, ego-centric representation critical for downstream planning and control. However, real-world deployment faces challenges from sensor degradation and adversarial attacks, which can cause severe perceptual anomalies and ultimately compromise the safety of autonomous driving systems. To address this, we propose a resilient and plug-and-play BEV perception method, RESBev, which can be easily applied to existing BEV perception methods to enhance their robustness to diverse disturbances. Specifically, we reframe perception robustness as a latent semantic prediction problem. A latent world model is constructed to extract spatiotemporal correlations across sequential BEV observations, thereby learning the underlying BEV state transitions to predict clean BEV features for reconstructing corrupted observations. The proposed framework operates at the semantic feature level of the Lift-Splat-Shoot pipeline, enabling recovery that generalizes across both natural disturbances and adversarial attacks without modifying the underlying backbone. Extensive experiments on the nuScenes dataset demonstrate that, with few-shot fine-tuning, RESBev significantly improves the robustness of existing BEV perception models against various external disturbances and adversarial attacks.

  </details>



- **$M^2$-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs**  
  Kaixin Lin, Kunyu Peng, Di Wen, Yufan Chen, Ruiping Liu, Kailun Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09737v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Semantic occupancy prediction enables dense 3D geometric and semantic understanding for autonomous driving. However, existing camera-based approaches implicitly assume complete surround-view observations, an assumption that rarely holds in real-world deployment due to occlusion, hardware malfunction, or communication failures. We study semantic occupancy prediction under incomplete multi-camera inputs and introduce $M^2$-Occ, a framework designed to preserve geometric structure and semantic coherence when views are missing. $M^2$-Occ addresses two complementary challenges. First, a Multi-view Masked Reconstruction (MMR) module leverages the spatial overlap among neighboring cameras to recover missing-view representations directly in the feature space. Second, a Feature Memory Module (FMM) introduces a learnable memory bank that stores class-level semantic prototypes. By retrieving and integrating these global priors, the FMM refines ambiguous voxel features, ensuring semantic consistency even when observational evidence is incomplete. We introduce a systematic missing-view evaluation protocol on the nuScenes-based SurroundOcc benchmark, encompassing both deterministic single-view failures and stochastic multi-view dropout scenarios. Under the safety-critical missing back-view setting, $M^2$-Occ improves the IoU by 4.93%. As the number of missing cameras increases, the robustness gap further widens; for instance, under the setting with five missing views, our method boosts the IoU by 5.01%. These gains are achieved without compromising full-view performance. The source code will be publicly released at https://github.com/qixi7up/M2-Occ.

  </details>



- **BEACON: Language-Conditioned Navigation Affordance Prediction under Occlusion**  
  Xinyu Gao, Gang Chen, Javier Alonso-Mora  
  _2026-03-10_ · https://arxiv.org/abs/2603.09961v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Language-conditioned local navigation requires a robot to infer a nearby traversable target location from its current observation and an open-vocabulary, relational instruction. Existing vision-language spatial grounding methods usually rely on vision-language models (VLMs) to reason in image space, producing 2D predictions tied to visible pixels. As a result, they struggle to infer target locations in occluded regions, typically caused by furniture or moving humans. To address this issue, we propose BEACON, which predicts an ego-centric Bird's-Eye View (BEV) affordance heatmap over a bounded local region including occluded areas. Given an instruction and surround-view RGB-D observations from four directions around the robot, BEACON predicts the BEV heatmap by injecting spatial cues into a VLM and fusing the VLM's output with depth-derived BEV features. Using an occlusion-aware dataset built in the Habitat simulator, we conduct detailed experimental analysis to validate both our BEV space formulation and the design choices of each module. Our method improves the accuracy averaged across geodesic thresholds by 22.74 percentage points over the state-of-the-art image-space baseline on the validation subset with occluded target locations. Our project page is: https://xin-yu-gao.github.io/beacon.

  </details>



- **StyleVLA: Driving Style-Aware Vision Language Action Model for Autonomous Driving**  
  Yuan Gao, Dengyuan Hua, Mattia Piccinini, Finn Rasmus Schäfer, Korbinian Moller, Lin Li, Johannes Betz  
  _2026-03-10_ · https://arxiv.org/abs/2603.09482v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision Language Models (VLMs) bridge visual perception and linguistic reasoning. In Autonomous Driving (AD), this synergy has enabled Vision Language Action (VLA) models, which translate high-level multimodal understanding into driving behaviors, typically represented as future trajectories. However, existing VLA models mainly generate generic collision-free trajectories. Beyond collision avoidance, adapting to diverse driving styles (e.g., sporty, comfortable) is essential for personalized driving. Moreover, many methods treat trajectory generation as naive token prediction, which can produce kinematically infeasible actions. To address these limitations, we present StyleVLA, a physics-informed VLA framework for generating diverse and physically plausible driving behaviors. We introduce a hybrid loss that combines a kinematic consistency constraint with a continuous regression head to improve trajectory feasibility. To train StyleVLA, built on Qwen3-VL-4B, we construct a large-scale instruction dataset with over 1.2k scenarios, 76k Bird's Eye View (BEV) samples, and 42k First Person View (FPV) samples, with ground-truth trajectories for five driving styles and natural-language instructions. Experiments show that our 4B-parameter StyleVLA significantly outperforms proprietary models (e.g., Gemini-3-Pro) and state-of-the-art VLA models. Using a composite driving score measuring success rate, physical feasibility, and style adherence, StyleVLA achieves 0.55 on BEV and 0.51 on FPV, versus 0.32 and 0.35 for Gemini-3-Pro. These results show that a specialized, physics-informed, lightweight model can surpass closed-source models on domain-specific tasks.

  </details>



- **EvoDriveVLA: Evolving Autonomous Driving Vision-Language-Action Model via Collaborative Perception-Planning Distillation**  
  Jiajun Cao, Xiaoan Zhang, Xiaobao Wei, Liyuqiu Huang, Wang Zijian, Hanzhen Zhang, Zhengyu Jia, Wei Mao, Hao Wang, Xianming Liu, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09465v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language-Action models have shown great promise for autonomous driving, yet they suffer from degraded perception after unfreezing the visual encoder and struggle with accumulated instability in long-term planning. To address these challenges, we propose EvoDriveVLA-a novel collaborative perception-planning distillation framework that integrates self-anchored perceptual constraints and oracle-guided trajectory optimization. Specifically, self-anchored visual distillation leverages self-anchor teacher to deliver visual anchoring constraints, regularizing student representations via trajectory-guided key-region awareness. In parallel, oracle-guided trajectory distillation employs a future-aware oracle teacher with coarse-to-fine trajectory refinement and Monte Carlo dropout sampling to produce high-quality trajectory candidates, thereby selecting the optimal trajectory to guide the student's prediction. EvoDriveVLA achieves SOTA performance in open-loop evaluation and significantly enhances performance in closed-loop evaluation. Our code is available at: https://github.com/hey-cjj/EvoDriveVLA.

  </details>



- **Declarative Scenario-based Testing with RoadLogic**  
  Ezio Bartocci, Alessio Gambi, Felix Gigler, Cristinel Mateis, Dejan Ničković  
  _2026-03-10_ · https://arxiv.org/abs/2603.09455v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Scenario-based testing is a key method for cost-effective and safe validation of autonomous vehicles (AVs). Existing approaches rely on imperative scenario definitions, requiring developers to manually enumerate numerous variants to achieve coverage. Declarative languages, such as OpenSCENARIO DSL (OS2), raise the abstraction level but lack systematic methods for instantiating concrete, specification-compliant scenarios as simulations. To our knowledge, currently, no open-source solution provides this capability. We present RoadLogic that bridges declarative OS2 specifications and executable simulations. It uses Answer Set Programming to generate abstract plans satisfying scenario constraints, motion planning to refine the plans into feasible trajectories, and specification-based monitoring to verify correctness. We evaluate RoadLogic on instantiating representative OS2 scenarios as simulations in the CommonRoad framework. Results show that RoadLogic consistently produces realistic, specification-satisfying simulations within minutes and captures diverse behavioral variants through parameter sampling, thus opening the door to systematic scenario-based testing for autonomous driving systems.

  </details>



- **VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models**  
  Shuhao Kang, Youqi Liao, Peijie Wang, Wenlong Liao, Qilin Zhang, Benjamin Busam, Xieyuanli Chen, Yun Liu  
  _2026-03-10_ · https://arxiv.org/abs/2603.09826v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Text-to-point-cloud (T2P) localization aims to infer precise spatial positions within 3D point cloud maps from natural language descriptions, reflecting how humans perceive and communicate spatial layouts through language. However, existing methods largely rely on shallow text-point cloud correspondence without effective spatial reasoning, limiting their accuracy in complex environments. To address this limitation, we propose VLM-Loc, a framework that leverages the spatial reasoning capability of large vision-language models (VLMs) for T2P localization. Specifically, we transform point clouds into bird's-eye-view (BEV) images and scene graphs that jointly encode geometric and semantic context, providing structured inputs for the VLM to learn cross-modal representations bridging linguistic and spatial semantics. On top of these representations, we introduce a partial node assignment mechanism that explicitly associates textual cues with scene graph nodes, enabling interpretable spatial reasoning for accurate localization. To facilitate systematic evaluation across diverse scenes, we present CityLoc, a benchmark built from multi-source point clouds for fine-grained T2P localization. Experiments on CityLoc demonstrate VLM-Loc achieves superior accuracy and robustness compared to state-of-the-art methods. Our code, model, and dataset are available at \href{https://github.com/MCG-NKU/nku-3d-vision}{repository}.

  </details>



- **Probing the Reliability of Driving VLMs: From Inconsistent Responses to Grounded Temporal Reasoning**  
  Chun-Peng Chang, Chen-Yu Wang, Holger Caesar, Alain Pagani  
  _2026-03-10_ · https://arxiv.org/abs/2603.09512v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  A reliable driving assistant should provide consistent responses based on temporally grounded reasoning derived from observed information. In this work, we investigate whether Vision-Language Models (VLMs), when applied as driving assistants, can response consistantly and understand how present observations shape future outcomes, or whether their outputs merely reflect patterns memorized during training without temporally grounded reasoning. While recent efforts have integrated VLMs into autonomous driving, prior studies typically emphasize scene understanding and instruction generation, implicitly assuming that strong visual interpretation naturally enables consistant future reasoning and thus ensures reliable decision-making, a claim we critically examine. We focus on two major challenges limiting VLM reliability in this setting: response inconsistency, where minor input perturbations yield different answers or, in some cases, responses degenerate toward near-random guessing, and limited temporal reasoning, in which models fail to reason and align sequential events from current observations, often resulting in incorrect or even contradictory responses. Moreover, we find that models with strong visual understanding do not necessarily perform best on tasks requiring temporal reasoning, indicating a tendency to over-rely on pretrained patterns rather than modeling temporal dynamics. To address these issues, we adopt existing evaluation methods and introduce FutureVQA, a human-annotated benchmark dataset specifically designed to assess future scene reasoning. In addition, we propose a simple yet effective self-supervised tuning approach with chain-of-thought reasoning that improves both consistency and temporal reasoning without requiring temporal labels.

  </details>


