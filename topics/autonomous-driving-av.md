# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-17 07:12 UTC_

Total papers shown: **5**


---

- **DM0: An Embodied-Native Vision-Language-Action Model towards Physical AI**  
  En Yu, Haoran Lv, Jianjian Sun, Kangheng Lin, Ruitao Zhang, Yukang Shi, Yuyang Chen, Ze Chen, Ziheng Zhang, Fan Jia, et al.  
  _2026-02-16_ · https://arxiv.org/abs/2602.14974v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Moving beyond the traditional paradigm of adapting internet-pretrained models to physical tasks, we present DM0, an Embodied-Native Vision-Language-Action (VLA) framework designed for Physical AI. Unlike approaches that treat physical grounding as a fine-tuning afterthought, DM0 unifies embodied manipulation and navigation by learning from heterogeneous data sources from the onset. Our methodology follows a comprehensive three-stage pipeline: Pretraining, Mid-Training, and Post-Training. First, we conduct large-scale unified pretraining on the Vision-Language Model (VLM) using diverse corpora--seamlessly integrating web text, autonomous driving scenarios, and embodied interaction logs-to jointly acquire semantic knowledge and physical priors. Subsequently, we build a flow-matching action expert atop the VLM. To reconcile high-level reasoning with low-level control, DM0 employs a hybrid training strategy: for embodied data, gradients from the action expert are not backpropagated to the VLM to preserve generalized representations, while the VLM remains trainable on non-embodied data. Furthermore, we introduce an Embodied Spatial Scaffolding strategy to construct spatial Chain-of-Thought (CoT) reasoning, effectively constraining the action solution space. Experiments on the RoboChallenge benchmark demonstrate that DM0 achieves state-of-the-art performance in both Specialist and Generalist settings on Table30.

  </details>



- **Multimodal Covariance Steering in Belief Space with Active Probing and Influence for Autonomous Driving**  
  Devodita Chakravarty, John Dolan, Yiwei Lyu  
  _2026-02-16_ · https://arxiv.org/abs/2602.14540v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous driving in complex traffic requires reasoning under uncertainty. Common approaches rely on prediction-based planning or risk-aware control, but these are typically treated in isolation, limiting their ability to capture the coupled nature of action and inference in interactive settings. This gap becomes especially critical in uncertain scenarios, where simply reacting to predictions can lead to unsafe maneuvers or overly conservative behavior. Our central insight is that safe interaction requires not only estimating human behavior but also shaping it when ambiguity poses risks. To this end, we introduce a hierarchical belief model that structures human behavior across coarse discrete intents and fine motion modes, updated via Bayesian inference for interpretable multi-resolution reasoning. On top of this, we develop an active probing strategy that identifies when multimodal ambiguity in human predictions may compromise safety and plans disambiguating actions that both reveal intent and gently steer human decisions toward safer outcomes. Finally, a runtime risk-evaluation layer based on Conditional Value-at-Risk (CVaR) ensures that all probing actions remain within human risk tolerance during influence. Our simulations in lane-merging and unsignaled intersection scenarios demonstrate that our approach achieves higher success rates and shorter completion times compared to existing methods. These results highlight the benefit of coupling belief inference, probing, and risk monitoring, yielding a principled and interpretable framework for planning under uncertainty.

  </details>



- **ThermEval: A Structured Benchmark for Evaluation of Vision-Language Models on Thermal Imagery**  
  Ayush Shrivastava, Kirtan Gangani, Laksh Jain, Mayank Goel, Nipun Batra  
  _2026-02-16_ · https://arxiv.org/abs/2602.14989v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision language models (VLMs) achieve strong performance on RGB imagery, but they do not generalize to thermal images. Thermal sensing plays a critical role in settings where visible light fails, including nighttime surveillance, search and rescue, autonomous driving, and medical screening. Unlike RGB imagery, thermal images encode physical temperature rather than color or texture, requiring perceptual and reasoning capabilities that existing RGB-centric benchmarks do not evaluate. We introduce ThermEval-B, a structured benchmark of approximately 55,000 thermal visual question answering pairs designed to assess the foundational primitives required for thermal vision language understanding. ThermEval-B integrates public datasets with our newly collected ThermEval-D, the first dataset to provide dense per-pixel temperature maps with semantic body-part annotations across diverse indoor and outdoor environments. Evaluating 25 open-source and closed-source VLMs, we find that models consistently fail at temperature-grounded reasoning, degrade under colormap transformations, and default to language priors or fixed responses, with only marginal gains from prompting or supervised fine-tuning. These results demonstrate that thermal understanding requires dedicated evaluation beyond RGB-centric assumptions, positioning ThermEval as a benchmark to drive progress in thermal vision language modeling.

  </details>



- **DriveFine: Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving**  
  Chenxu Dang, Sining Ang, Yongkang Li, Haochen Tian, Jie Wang, Guang Li, Hangjun Ye, Jie Ma, Long Chen, Yan Wang  
  _2026-02-16_ · https://arxiv.org/abs/2602.14577v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models for autonomous driving increasingly adopt generative planners trained with imitation learning followed by reinforcement learning. Diffusion-based planners suffer from modality alignment difficulties, low training efficiency, and limited generalization. Token-based planners are plagued by cumulative causal errors and irreversible decoding. In summary, the two dominant paradigms exhibit complementary strengths and weaknesses. In this paper, we propose DriveFine, a masked diffusion VLA model that combines flexible decoding with self-correction capabilities. In particular, we design a novel plug-and-play block-MoE, which seamlessly injects a refinement expert on top of the generation expert. By enabling explicit expert selection during inference and gradient blocking during training, the two experts are fully decoupled, preserving the foundational capabilities and generic patterns of the pretrained weights, which highlights the flexibility and extensibility of the block-MoE design. Furthermore, we design a hybrid reinforcement learning strategy that encourages effective exploration of refinement expert while maintaining training stability. Extensive experiments on NAVSIM v1, v2, and Navhard benchmarks demonstrate that DriveFine exhibits strong efficacy and robustness. The code will be released at https://github.com/MSunDYY/DriveFine.

  </details>



- **Cross-view Domain Generalization via Geometric Consistency for LiDAR Semantic Segmentation**  
  Jindong Zhao, Yuan Gao, Yang Xia, Sheng Nie, Jun Yue, Weiwei Sun, Shaobo Xia  
  _2026-02-16_ · https://arxiv.org/abs/2602.14525v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Domain-generalized LiDAR semantic segmentation (LSS) seeks to train models on source-domain point clouds that generalize reliably to multiple unseen target domains, which is essential for real-world LiDAR applications. However, existing approaches assume similar acquisition views (e.g., vehicle-mounted) and struggle in cross-view scenarios, where observations differ substantially due to viewpoint-dependent structural incompleteness and non-uniform point density. Accordingly, we formulate cross-view domain generalization for LiDAR semantic segmentation and propose a novel framework, termed CVGC (Cross-View Geometric Consistency). Specifically, we introduce a cross-view geometric augmentation module that models viewpoint-induced variations in visibility and sampling density, generating multiple cross-view observations of the same scene. Subsequently, a geometric consistency module enforces consistent semantic and occupancy predictions across geometrically augmented point clouds of the same scene. Extensive experiments on six public LiDAR datasets establish the first systematic evaluation of cross-view domain generalization for LiDAR semantic segmentation, demonstrating that CVGC consistently outperforms state-of-the-art methods when generalizing from a single source domain to multiple target domains with heterogeneous acquisition viewpoints. The source code will be publicly available at https://github.com/KintomZi/CVGC-DG

  </details>


