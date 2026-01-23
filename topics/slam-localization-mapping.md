# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-01-23 06:51 UTC_

Total papers shown: **15**


---

- **AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning**  
  Zichen Yan, Yuchen Hou, Shenao Wang, Yichao Gao, Rui Huang, Lin Zhao  
  _2026-01-22_ · https://arxiv.org/abs/2601.15614v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Object-Goal Navigation (ObjectNav) requires an agent to autonomously explore an unknown environment and navigate toward target objects specified by a semantic label. While prior work has primarily studied zero-shot ObjectNav under 2D locomotion, extending it to aerial platforms with 3D locomotion capability remains underexplored. Aerial robots offer superior maneuverability and search efficiency, but they also introduce new challenges in spatial perception, dynamic control, and safety assurance. In this paper, we propose AION for vision-based aerial ObjectNav without relying on external localization or global maps. AION is an end-to-end dual-policy reinforcement learning (RL) framework that decouples exploration and goal-reaching behaviors into two specialized policies. We evaluate AION on the AI2-THOR benchmark and further assess its real-time performance in IsaacSim using high-fidelity drone models. Experimental results show that AION achieves superior performance across comprehensive evaluation metrics in exploration, navigation efficiency, and safety. The video can be found at https://youtu.be/TgsUm6bb7zg.

  </details>



- **Accurate Calibration and Robust LiDAR-Inertial Odometry for Spinning Actuated LiDAR Systems**  
  Zijie Chen, Xiaowei Liu, Yong Xu, Shenghai Yuan, Jianping Li, Lihua Xie  
  _2026-01-22_ · https://arxiv.org/abs/2601.15946v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Accurate calibration and robust localization are fundamental for downstream tasks in spinning actuated LiDAR applications. Existing methods, however, require parameterizing extrinsic parameters based on different mounting configurations, limiting their generalizability. Additionally, spinning actuated LiDAR inevitably scans featureless regions, which complicates the balance between scanning coverage and localization robustness. To address these challenges, this letter presents a targetless LiDAR-motor calibration (LM-Calibr) on the basis of the Denavit-Hartenberg convention and an environmental adaptive LiDAR-inertial odometry (EVA-LIO). LM-Calibr supports calibration of LiDAR-motor systems with various mounting configurations. Extensive experiments demonstrate its accuracy and convergence across different scenarios, mounting angles, and initial values. Additionally, EVA-LIO adaptively selects downsample rates and map resolutions according to spatial scale. This adaptivity enables the actuator to operate at maximum speed, thereby enhancing scanning completeness while ensuring robust localization, even when LiDAR briefly scans featureless areas. The source code and hardware design are available on GitHub: \textcolor{blue}{\href{https://github.com/zijiechenrobotics/lm_calibr}{github.com/zijiechenrobotics/lm\_calibr}}. The video is available at \textcolor{blue}{\href{https://youtu.be/cZyyrkmeoSk}{youtu.be/cZyyrkmeoSk}}

  </details>



- **Keyframe-Based Feed-Forward Visual Odometry**  
  Weichen Dai, Wenhan Su, Da Kong, Yuhang Ming, Wanzeng Kong  
  _2026-01-22_ · https://arxiv.org/abs/2601.16020v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The emergence of visual foundation models has revolutionized visual odometry~(VO) and SLAM, enabling pose estimation and dense reconstruction within a single feed-forward network. However, unlike traditional pipelines that leverage keyframe methods to enhance efficiency and accuracy, current foundation model based methods, such as VGGT-Long, typically process raw image sequences indiscriminately. This leads to computational redundancy and degraded performance caused by low inter-frame parallax, which provides limited contextual stereo information. Integrating traditional geometric heuristics into these methods is non-trivial, as their performance depends on high-dimensional latent representations rather than explicit geometric metrics. To bridge this gap, we propose a novel keyframe-based feed-forward VO. Instead of relying on hand-crafted rules, our approach employs reinforcement learning to derive an adaptive keyframe policy in a data-driven manner, aligning selection with the intrinsic characteristics of the underlying foundation model. We train our agent on TartanAir dataset and conduct extensive evaluations across several real-world datasets. Experimental results demonstrate that the proposed method achieves consistent and substantial improvements over state-of-the-art feed-forward VO methods.

  </details>



- **ICON: Invariant Counterfactual Optimization with Neuro-Symbolic Priors for Text-Based Person Search**  
  Xiangyu Wang, Zhixin Lv, Yongjiao Sun, Anrui Han, Ye Yuan, Hangxu Ji  
  _2026-01-22_ · https://arxiv.org/abs/2601.15931v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Text-Based Person Search (TBPS) holds unique value in real-world surveillance bridging visual perception and language understanding, yet current paradigms utilizing pre-training models often fail to transfer effectively to complex open-world scenarios. The reliance on "Passive Observation" leads to multifaceted spurious correlations and spatial semantic misalignment, causing a lack of robustness against distribution shifts. To fundamentally resolve these defects, this paper proposes ICON (Invariant Counterfactual Optimization with Neuro-symbolic priors), a framework integrating causal and topological priors. First, we introduce Rule-Guided Spatial Intervention to strictly penalize sensitivity to bounding box noise, forcibly severing location shortcuts to achieve geometric invariance. Second, Counterfactual Context Disentanglement is implemented via semantic-driven background transplantation, compelling the model to ignore background interference for environmental independence. Then, we employ Saliency-Driven Semantic Regularization with adaptive masking to resolve local saliency bias and guarantee holistic completeness. Finally, Neuro-Symbolic Topological Alignment utilizes neuro-symbolic priors to constrain feature matching, ensuring activated regions are topologically consistent with human structural logic. Experimental results demonstrate that ICON not only maintains leading performance on standard benchmarks but also exhibits exceptional robustness against occlusion, background interference, and localization noise. This approach effectively advances the field by shifting from fitting statistical co-occurrences to learning causal invariance.

  </details>



- **Beyond Hard Writes and Rigid Preservation: Soft Recursive Least-Squares for Lifelong LLM Editing**  
  Xinyu Wang, Sicheng Lyu, Yu Gu, Jerry Huang, Peng Lu, Yufei Cui, Xiao-Wen Chang  
  _2026-01-22_ · https://arxiv.org/abs/2601.15686v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Model editing updates a pre-trained LLM with new facts or rules without re-training, while preserving unrelated behavior. In real deployment, edits arrive as long streams, and existing editors often face a plasticity-stability dilemma: locate-then-edit "hard writes" can accumulate interference over time, while null-space-style "hard preservation" preserves only what is explicitly constrained, so past edits can be overwritten and unconstrained behaviors may deviate, degrading general capabilities in the many-edits regime. We propose RLSEdit, a recursive least-squares editor for long sequential editing. RLSEdit formulates editing as an online quadratic optimization with soft constraints, minimizing a cumulative key-value fitting objective with two regularizers that control for both deviation from the pre-trained weights and from a designated anchor mapping. The resulting update admits an efficient online recursion via the Woodbury identity, with per-edit cost independent of history length and scaling only with the current edit size. We further provide deviation bounds and an asymptotic characterization of the adherence-preservation trade-off in the many-edits regime. Experiments on multiple model families demonstrate stable scaling to 10K edits, outperforming strong baselines in both edit success and holistic stability -- crucially retaining early edits, and preserving general capabilities on GLUE and held-out reasoning/code benchmarks.

  </details>



- **PyraTok: Language-Aligned Pyramidal Tokenizer for Video Understanding and Generation**  
  Onkar Susladkar, Tushar Prakash, Adheesh Juvekar, Kiet A. Nguyen, Dong-Hwan Jang, Inderjit S Dhillon, Ismini Lourentzou  
  _2026-01-22_ · https://arxiv.org/abs/2601.16210v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Discrete video VAEs underpin modern text-to-video generation and video understanding systems, yet existing tokenizers typically learn visual codebooks at a single scale with limited vocabularies and shallow language supervision, leading to poor cross-modal alignment and zero-shot transfer. We introduce PyraTok, a language-aligned pyramidal tokenizer that learns semantically structured discrete latents across multiple spatiotemporal resolutions. PyraTok builds on a pretrained video VAE and a novel Language aligned Pyramidal Quantization (LaPQ) module that discretizes encoder features at several depths using a shared large binary codebook, yielding compact yet expressive video token sequences. To tightly couple visual tokens with language, PyraTok jointly optimizes multi-scale text-guided quantization and a global autoregressive objective over the token hierarchy. Across ten benchmarks, PyraTok delivers state-of-the-art (SOTA) video reconstruction, consistently improves text-to-video quality, and sets new SOTA zero-shot performance on video segmentation, temporal action localization, and video understanding, scaling robustly to up to 4K/8K resolutions.

  </details>



- **360Anything: Geometry-Free Lifting of Images and Videos to 360°**  
  Ziyi Wu, Daniel Watson, Andrea Tagliasacchi, David J. Fleet, Marcus A. Brubaker, Saurabh Saxena  
  _2026-01-22_ · https://arxiv.org/abs/2601.16192v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Lifting perspective images and videos to 360° panoramas enables immersive 3D world generation. Existing approaches often rely on explicit geometric alignment between the perspective and the equirectangular projection (ERP) space. Yet, this requires known camera metadata, obscuring the application to in-the-wild data where such calibration is typically absent or noisy. We propose 360Anything, a geometry-free framework built upon pre-trained diffusion transformers. By treating the perspective input and the panorama target simply as token sequences, 360Anything learns the perspective-to-equirectangular mapping in a purely data-driven way, eliminating the need for camera information. Our approach achieves state-of-the-art performance on both image and video perspective-to-360° generation, outperforming prior works that use ground-truth camera information. We also trace the root cause of the seam artifacts at ERP boundaries to zero-padding in the VAE encoder, and introduce Circular Latent Encoding to facilitate seamless generation. Finally, we show competitive results in zero-shot camera FoV and orientation estimation benchmarks, demonstrating 360Anything's deep geometric understanding and broader utility in computer vision tasks. Additional results are available at https://360anything.github.io/.

  </details>



- **Phi-SegNet: Phase-Integrated Supervision for Medical Image Segmentation**  
  Shams Nafisa Ali, Taufiq Hasan  
  _2026-01-22_ · https://arxiv.org/abs/2601.16064v1 · `eess.IV`  
  <details><summary>Abstract</summary>

  Deep learning has substantially advanced medical image segmentation, yet achieving robust generalization across diverse imaging modalities and anatomical structures remains a major challenge. A key contributor to this limitation lies in how existing architectures, ranging from CNNs to Transformers and their hybrids, primarily encode spatial information while overlooking frequency-domain representations that capture rich structural and textural cues. Although few recent studies have begun exploring spectral information at the feature level, supervision-level integration of frequency cues-crucial for fine-grained object localization-remains largely untapped. To this end, we propose Phi-SegNet, a CNN-based architecture that incorporates phase-aware information at both architectural and optimization levels. The network integrates Bi-Feature Mask Former (BFMF) modules that blend neighboring encoder features to reduce semantic gaps, and Reverse Fourier Attention (RFA) blocks that refine decoder outputs using phase-regularized features. A dedicated phase-aware loss aligns these features with structural priors, forming a closed feedback loop that emphasizes boundary precision. Evaluated on five public datasets spanning X-ray, US, histopathology, MRI, and colonoscopy, Phi-SegNet consistently achieved state-of-the-art performance, with an average relative improvement of 1.54+/-1.26% in IoU and 0.98+/-0.71% in F1-score over the next best-performing model. In cross-dataset generalization scenarios involving unseen datasets from the known domain, Phi-SegNet also exhibits robust and superior performance, highlighting its adaptability and modality-agnostic design. These findings demonstrate the potential of leveraging spectral priors in both feature representation and supervision, paving the way for generalized segmentation frameworks that excel in fine-grained object localization.

  </details>



- **Data-Driven Conditional Flexibility Index**  
  Moritz Wedemeyer, Eike Cramer, Alexander Mitsos, Manuel Dahmen  
  _2026-01-22_ · https://arxiv.org/abs/2601.16028v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  With the increasing flexibilization of processes, determining robust scheduling decisions has become an important goal. Traditionally, the flexibility index has been used to identify safe operating schedules by approximating the admissible uncertainty region using simple admissible uncertainty sets, such as hypercubes. Presently, available contextual information, such as forecasts, has not been considered to define the admissible uncertainty set when determining the flexibility index. We propose the conditional flexibility index (CFI), which extends the traditional flexibility index in two ways: by learning the parametrized admissible uncertainty set from historical data and by using contextual information to make the admissible uncertainty set conditional. This is achieved using a normalizing flow that learns a bijective mapping from a Gaussian base distribution to the data distribution. The admissible latent uncertainty set is constructed as a hypersphere in the latent space and mapped to the data space. By incorporating contextual information, the CFI provides a more informative estimate of flexibility by defining admissible uncertainty sets in regions that are more likely to be relevant under given conditions. Using an illustrative example, we show that no general statement can be made about data-driven admissible uncertainty sets outperforming simple sets, or conditional sets outperforming unconditional ones. However, both data-driven and conditional admissible uncertainty sets ensure that only regions of the uncertain parameter space containing realizations are considered. We apply the CFI to a security-constrained unit commitment example and demonstrate that the CFI can improve scheduling quality by incorporating temporal information.

  </details>



- **Natural Language-Driven Global Mapping of Martian Landforms**  
  Yiran Wang, Shuoyuan Wang, Zhaoran Wei, Jiannan Zhao, Zhonghua Yao, Zejian Xie, Songxin Zhang, Jun Huang, Bingyi Jing, Hongxin Wei  
  _2026-01-22_ · https://arxiv.org/abs/2601.15949v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Planetary surfaces are typically analyzed using high-level semantic concepts in natural language, yet vast orbital image archives remain organized at the pixel level. This mismatch limits scalable, open-ended exploration of planetary surfaces. Here we present MarScope, a planetary-scale vision-language framework enabling natural language-driven, label-free mapping of Martian landforms. MarScope aligns planetary images and text in a shared semantic space, trained on over 200,000 curated image-text pairs. This framework transforms global geomorphic mapping on Mars by replacing pre-defined classifications with flexible semantic retrieval, enabling arbitrary user queries across the entire planet in 5 seconds with F1 scores up to 0.978. Applications further show that it extends beyond morphological classification to facilitate process-oriented analysis and similarity-based geomorphological mapping at a planetary scale. MarScope establishes a new paradigm where natural language serves as a direct interface for scientific discovery over massive geospatial datasets.

  </details>



- **Dual-Mapping Sparse Vector Transmission for Short Packet URLLC**  
  Yanfeng Zhang, Xu Zhu, Jinkai Zheng, Weiwei Yang, Xianhua Yu, Haiyong Zeng, Yujie Liu, Yong Liang Guan  
  _2026-01-22_ · https://arxiv.org/abs/2601.15819v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Sparse vector coding (SVC) is a promising short-packet transmission method for ultra reliable low latency communication (URLLC) in next generation communication systems. In this paper, a dual-mapping SVC (DM-SVC) based short packet transmission scheme is proposed to further enhance the transmission performance of SVC. The core idea behind the proposed scheme lies in mapping the transmitted information bits onto sparse vectors via block and single-element sparse mappings. The block sparse mapping pattern is able to concentrate the transmit power in a small number of non-zero blocks thus improving the decoding accuracy, while the single-element sparse mapping pattern ensures that the code length does not increase dramatically with the number of transmitted information bits. At the receiver, a two-stage decoding algorithm is proposed to sequentially identify non-zero block indexes and single-element non-zero indexes. Extensive simulation results verify that proposed DM-SVC scheme outperforms the existing SVC schemes in terms of block error rate and spectral efficiency.

  </details>



- **Joint Pilot and Unknown Data-based Localization for OFDM Opportunistic Radar Systems**  
  Mathieu Reniers, Martin Willame, Jérôme Louveaux, Luc Vandendorpe  
  _2026-01-22_ · https://arxiv.org/abs/2601.15785v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Integrated Sensing and Communications (ISAC) has emerged as a promising paradigm for Sixth Generation (6G) and Wi-Fi 7 networks, with the communication-centric approach being particularly attractive due to its compatibility with current standards. Typical communication signals comprise both deterministic known pilot signals and random unknown data payloads. Most existing approaches either rely solely on pilots for positioning, thereby ignoring the radar information present in the received data symbols that constitute the majority of each frame, or rely on data decisions, which bounds positioning performance to that of the communication system. To overcome these limitations, we propose a novel method that extracts positioning information from data payloads without decoding them. We consider an opportunistic scenario in which communication signals from a user are captured by an opportunistic radar equipped with a Uniform Linear Arrays of antennas. We show that, in this setting, the estimation can be efficiently implemented using Fast Fourier Transforms. Finally, we demonstrate superior localization performance compared to existing methods in the literature through numerical simulations.

  </details>



- **Breaking the Resolution Barrier: Arbitrary-resolution Deep Image Steganography Framework**  
  Xinjue Hu, Chi Wang, Boyu Wang, Xiang Zhang, Zhenshan Tan, Zhangjie Fu  
  _2026-01-22_ · https://arxiv.org/abs/2601.15739v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Deep image steganography (DIS) has achieved significant results in capacity and invisibility. However, current paradigms enforce the secret image to maintain the same resolution as the cover image during hiding and revealing. This leads to two challenges: secret images with inconsistent resolutions must undergo resampling beforehand which results in detail loss during recovery, and the secret image cannot be recovered to its original resolution when the resolution value is unknown. To address these, we propose ARDIS, the first Arbitrary Resolution DIS framework, which shifts the paradigm from discrete mapping to reference-guided continuous signal reconstruction. Specifically, to minimize the detail loss caused by resolution mismatch, we first design a Frequency Decoupling Architecture in hiding stage. It disentangles the secret into a resolution-aligned global basis and a resolution-agnostic high-frequency latent to hide in a fixed-resolution cover. Second, for recovery, we propose a Latent-Guided Implicit Reconstructor to perform deterministic restoration. The recovered detail latent code modulates a continuous implicit function to accurately query and render high-frequency residuals onto the recovered global basis, ensuring faithful restoration of original details. Furthermore, to achieve blind recovery, we introduce an Implicit Resolution Coding strategy. By transforming discrete resolution values into dense feature maps and hiding them in the redundant space of the feature domain, the reconstructor can correctly decode the secret's resolution directly from the steganographic representation. Experimental results demonstrate that ARDIS significantly outperforms state-of-the-art methods in both invisibility and cross-resolution recovery fidelity.

  </details>



- **VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning**  
  Chenglin Li, Qianglong Chen, Feng Han, Yikun Wang, Xingxi Yin, Yan Gong, Ruilin Li, Yin Zhang, Jiaqi Wang  
  _2026-01-22_ · https://arxiv.org/abs/2601.15724v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Long-form video understanding remains a fundamental challenge for current Video Large Language Models. Most existing models rely on static reasoning over uniformly sampled frames, which weakens temporal localization and leads to substantial information loss in long videos. Agentic tools such as temporal retrieval, spatial zoom, and temporal zoom offer a natural way to overcome these limitations by enabling adaptive exploration of key moments. However, constructing agentic video understanding data requires models that already possess strong long-form video comprehension, creating a circular dependency. We address this challenge with VideoThinker, an agentic Video Large Language Model trained entirely on synthetic tool interaction trajectories. Our key idea is to convert videos into rich captions and employ a powerful agentic language model to generate multi-step tool use sequences in caption space. These trajectories are subsequently grounded back to video by replacing captions with the corresponding frames, yielding a large-scale interleaved video and tool reasoning dataset without requiring any long-form understanding from the underlying model. Training on this synthetic agentic dataset equips VideoThinker with dynamic reasoning capabilities, adaptive temporal exploration, and multi-step tool use. Remarkably, VideoThinker significantly outperforms both caption-only language model agents and strong video model baselines across long-video benchmarks, demonstrating the effectiveness of tool augmented synthetic data and adaptive retrieval and zoom reasoning for long-form video understanding.

  </details>



- **Skywork UniPic 3.0: Unified Multi-Image Composition via Sequence Modeling**  
  Hongyang Wei, Hongbo Liu, Zidong Wang, Yi Peng, Baixin Xu, Size Wu, Xuying Zhang, Xianglong He, Zexiang Liu, Peiyu Wang, et al.  
  _2026-01-22_ · https://arxiv.org/abs/2601.15664v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The recent surge in popularity of Nano-Banana and Seedream 4.0 underscores the community's strong interest in multi-image composition tasks. Compared to single-image editing, multi-image composition presents significantly greater challenges in terms of consistency and quality, yet existing models have not disclosed specific methodological details for achieving high-quality fusion. Through statistical analysis, we identify Human-Object Interaction (HOI) as the most sought-after category by the community. We therefore systematically analyze and implement a state-of-the-art solution for multi-image composition with a primary focus on HOI-centric tasks. We present Skywork UniPic 3.0, a unified multimodal framework that integrates single-image editing and multi-image composition. Our model supports an arbitrary (1~6) number and resolution of input images, as well as arbitrary output resolutions (within a total pixel budget of 1024x1024). To address the challenges of multi-image composition, we design a comprehensive data collection, filtering, and synthesis pipeline, achieving strong performance with only 700K high-quality training samples. Furthermore, we introduce a novel training paradigm that formulates multi-image composition as a sequence-modeling problem, transforming conditional generation into unified sequence synthesis. To accelerate inference, we integrate trajectory mapping and distribution matching into the post-training stage, enabling the model to produce high-fidelity samples in just 8 steps and achieve a 12.5x speedup over standard synthesis sampling. Skywork UniPic 3.0 achieves state-of-the-art performance on single-image editing benchmark and surpasses both Nano-Banana and Seedream 4.0 on multi-image composition benchmark, thereby validating the effectiveness of our data pipeline and training paradigm. Code, models and dataset are publicly available.

  </details>


