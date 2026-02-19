# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-19 07:13 UTC_

Total papers shown: **3**


---

- **PredMapNet: Future and Historical Reasoning for Consistent Online HD Vectorized Map Construction**  
  Bo Lang, Nirav Savaliya, Zhihao Zheng, Jinglun Feng, Zheng-Hang Yeh, Mooi Choo Chuah  
  _2026-02-18_ · https://arxiv.org/abs/2602.16669v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  High-definition (HD) maps are crucial to autonomous driving, providing structured representations of road elements to support navigation and planning. However, existing query-based methods often employ random query initialization and depend on implicit temporal modeling, which lead to temporal inconsistencies and instabilities during the construction of a global map. To overcome these challenges, we introduce a novel end-to-end framework for consistent online HD vectorized map construction, which jointly performs map instance tracking and short-term prediction. First, we propose a Semantic-Aware Query Generator that initializes queries with spatially aligned semantic masks to capture scene-level context globally. Next, we design a History Rasterized Map Memory to store fine-grained instance-level maps for each tracked instance, enabling explicit historical priors. A History-Map Guidance Module then integrates rasterized map information into track queries, improving temporal continuity. Finally, we propose a Short-Term Future Guidance module to forecast the immediate motion of map instances based on the stored history trajectories. These predicted future locations serve as hints for tracked instances to further avoid implausible predictions and keep temporal consistency. Extensive experiments on the nuScenes and Argoverse2 datasets demonstrate that our proposed method outperforms state-of-the-art (SOTA) methods with good efficiency.

  </details>



- **A Contrastive Learning Framework Empowered by Attention-based Feature Adaptation for Street-View Image Classification**  
  Qi You, Yitai Cheng, Zichao Zeng, James Haworth  
  _2026-02-18_ · https://arxiv.org/abs/2602.16590v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Street-view image attribute classification is a vital downstream task of image classification, enabling applications such as autonomous driving, urban analytics, and high-definition map construction. It remains computationally demanding whether training from scratch, initialising from pre-trained weights, or fine-tuning large models. Although pre-trained vision-language models such as CLIP offer rich image representations, existing adaptation or fine-tuning methods often rely on their global image embeddings, limiting their ability to capture fine-grained, localised attributes essential in complex, cluttered street scenes. To address this, we propose CLIP-MHAdapter, a variant of the current lightweight CLIP adaptation paradigm that appends a bottleneck MLP equipped with multi-head self-attention operating on patch tokens to model inter-patch dependencies. With approximately 1.4 million trainable parameters, CLIP-MHAdapter achieves superior or competitive accuracy across eight attribute classification tasks on the Global StreetScapes dataset, attaining new state-of-the-art results while maintaining low computational cost. The code is available at https://github.com/SpaceTimeLab/CLIP-MHAdapter.

  </details>



- **Parameter-Free Adaptive Multi-Scale Channel-Spatial Attention Aggregation framework for 3D Indoor Semantic Scene Completion Toward Assisting Visually Impaired**  
  Qi He, XiangXiang Wang, Jingtao Zhang, Yongbin Yu, Hongxiang Chu, Manping Fan, JingYe Cai, Zhenglin Yang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16385v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  In indoor assistive perception for visually impaired users, 3D Semantic Scene Completion (SSC) is expected to provide structurally coherent and semantically consistent occupancy under strictly monocular vision for safety-critical scene understanding. However, existing monocular SSC approaches often lack explicit modeling of voxel-feature reliability and regulated cross-scale information propagation during 2D-3D projection and multi-scale fusion, making them vulnerable to projection diffusion and feature entanglement and thus limiting structural stability.To address these challenges, this paper presents an Adaptive Multi-scale Attention Aggregation (AMAA) framework built upon the MonoScene pipeline. Rather than introducing a heavier backbone, AMAA focuses on reliability-oriented feature regulation within a monocular SSC framework. Specifically, lifted voxel features are jointly calibrated in semantic and spatial dimensions through parallel channel-spatial attention aggregation, while multi-scale encoder-decoder fusion is stabilized via a hierarchical adaptive feature-gating strategy that regulates information injection across scales.Experiments on the NYUv2 benchmark demonstrate consistent improvements over MonoScene without significantly increasing system complexity: AMAA achieves 27.25% SSC mIoU (+0.31) and 43.10% SC IoU (+0.59). In addition, system-level deployment on an NVIDIA Jetson platform verifies that the complete AMAA framework can be executed stably on embedded hardware. Overall, AMAA improves monocular SSC quality and provides a reliable and deployable perception framework for indoor assistive systems targeting visually impaired users.

  </details>


