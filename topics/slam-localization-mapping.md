# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-06 07:08 UTC_

Total papers shown: **14**


---

- **A Hybrid Autoencoder for Robust Heightmap Generation from Fused Lidar and Depth Data for Humanoid Robot Locomotion**  
  Dennis Bank, Joost Cordes, Thomas Seel, Simon F. G. Ehlers  
  _2026-02-05_ · https://arxiv.org/abs/2602.05855v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable terrain perception is a critical prerequisite for the deployment of humanoid robots in unstructured, human-centric environments. While traditional systems often rely on manually engineered, single-sensor pipelines, this paper presents a learning-based framework that uses an intermediate, robot-centric heightmap representation. A hybrid Encoder-Decoder Structure (EDS) is introduced, utilizing a Convolutional Neural Network (CNN) for spatial feature extraction fused with a Gated Recurrent Unit (GRU) core for temporal consistency. The architecture integrates multimodal data from an Intel RealSense depth camera, a LIVOX MID-360 LiDAR processed via efficient spherical projection, and an onboard IMU. Quantitative results demonstrate that multimodal fusion improves reconstruction accuracy by 7.2% over depth-only and 9.9% over LiDAR-only configurations. Furthermore, the integration of a 3.2 s temporal context reduces mapping drift.

  </details>



- **ShapeUP: Scalable Image-Conditioned 3D Editing**  
  Inbar Gat, Dana Cohen-Bar, Guy Levy, Elad Richardson, Daniel Cohen-Or  
  _2026-02-05_ · https://arxiv.org/abs/2602.05676v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advancements in 3D foundation models have enabled the generation of high-fidelity assets, yet precise 3D manipulation remains a significant challenge. Existing 3D editing frameworks often face a difficult trade-off between visual controllability, geometric consistency, and scalability. Specifically, optimization-based methods are prohibitively slow, multi-view 2D propagation techniques suffer from visual drift, and training-free latent manipulation methods are inherently bound by frozen priors and cannot directly benefit from scaling. In this work, we present ShapeUP, a scalable, image-conditioned 3D editing framework that formulates editing as a supervised latent-to-latent translation within a native 3D representation. This formulation allows ShapeUP to build on a pretrained 3D foundation model, leveraging its strong generative prior while adapting it to editing through supervised training. In practice, ShapeUP is trained on triplets consisting of a source 3D shape, an edited 2D image, and the corresponding edited 3D shape, and learns a direct mapping using a 3D Diffusion Transformer (DiT). This image-as-prompt approach enables fine-grained visual control over both local and global edits and achieves implicit, mask-free localization, while maintaining strict structural consistency with the original asset. Our extensive evaluations demonstrate that ShapeUP consistently outperforms current trained and training-free baselines in both identity preservation and edit fidelity, offering a robust and scalable paradigm for native 3D content creation.

  </details>



- **VisRefiner: Learning from Visual Differences for Screenshot-to-Code Generation**  
  Jie Deng, Kaichun Yao, Libo Zhang  
  _2026-02-05_ · https://arxiv.org/abs/2602.05998v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Screenshot-to-code generation aims to translate user interface screenshots into executable frontend code that faithfully reproduces the target layout and style. Existing multimodal large language models perform this mapping directly from screenshots but are trained without observing the visual outcomes of their generated code. In contrast, human developers iteratively render their implementation, compare it with the design, and learn how visual differences relate to code changes. Inspired by this process, we propose VisRefiner, a training framework that enables models to learn from visual differences between rendered predictions and reference designs. We construct difference-aligned supervision that associates visual discrepancies with corresponding code edits, allowing the model to understand how appearance variations arise from implementation changes. Building on this, we introduce a reinforcement learning stage for self-refinement, where the model improves its generated code by observing both the rendered output and the target design, identifying their visual differences, and updating the code accordingly. Experiments show that VisRefiner substantially improves single-step generation quality and layout fidelity, while also endowing models with strong self-refinement ability. These results demonstrate the effectiveness of learning from visual differences for advancing screenshot-to-code generation.

  </details>



- **Orthogonal Self-Attention**  
  Leo Zhang, James Martens  
  _2026-02-05_ · https://arxiv.org/abs/2602.05996v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Softmax Self-Attention (SSA) is a key component of Transformer architectures. However, when utilised within skipless architectures, which aim to improve representation learning, recent work has highlighted the inherent instability of SSA due to inducing rank collapse and poorly-conditioned Jacobians. In this work, we design a novel attention mechanism: Orthogonal Self-Attention (OSA), which aims to bypass these issues with SSA, in order to allow for (non-causal) Transformers without skip connections and normalisation layers to be more easily trained. In particular, OSA parametrises the attention matrix to be orthogonal via mapping a skew-symmetric matrix, formed from query-key values, through the matrix exponential. We show that this can be practically implemented, by exploiting the low-rank structure of our query-key values, resulting in the computational complexity and memory cost of OSA scaling linearly with sequence length. Furthermore, we derive an initialisation scheme for which we prove ensures that the Jacobian of OSA is well-conditioned.

  </details>



- **Orthogonal Model Merging**  
  Sihan Yang, Kexuan Shi, Weiyang Liu  
  _2026-02-05_ · https://arxiv.org/abs/2602.05943v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Merging finetuned Large Language Models (LLMs) has become increasingly important for integrating diverse capabilities into a single unified model. However, prevailing model merging methods rely on linear arithmetic in Euclidean space, which often destroys the intrinsic geometric properties of pretrained weights, such as hyperspherical energy. To address this, we propose Orthogonal Model Merging (OrthoMerge), a method that performs merging operations on the Riemannian manifold formed by the orthogonal group to preserve the geometric structure of the model's weights. By mapping task-specific orthogonal matrices learned by Orthogonal Finetuning (OFT) to the Lie algebra, OrthoMerge enables a principled yet efficient integration that takes into account both the direction and intensity of adaptations. In addition to directly leveraging orthogonal matrices obtained by OFT, we further extend this approach to general models finetuned with non-OFT methods (i.e., low-rank finetuning, full finetuning) via an Orthogonal-Residual Decoupling strategy. This technique extracts the orthogonal components of expert models by solving the orthogonal Procrustes problem, which are then merged on the manifold of the orthogonal group, while the remaining linear residuals are processed through standard additive merging. Extensive empirical results demonstrate the effectiveness of OrthoMerge in mitigating catastrophic forgetting and maintaining model performance across diverse tasks.

  </details>



- **CLIP-Map: Structured Matrix Mapping for Parameter-Efficient CLIP Compression**  
  Kangjie Zhang, Wenxuan Huang, Xin Zhou, Boxiang Zhou, Dejia Song, Yuan Xie, Baochang Zhang, Lizhuang Ma, Nemo Chen, Xu Tang, et al.  
  _2026-02-05_ · https://arxiv.org/abs/2602.05909v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Contrastive Language-Image Pre-training (CLIP) has achieved widely applications in various computer vision tasks, e.g., text-to-image generation, Image-Text retrieval and Image captioning. However, CLIP suffers from high memory and computation cost, which prohibits its usage to the resource-limited application scenarios. Existing CLIP compression methods typically reduce the size of pre-trained CLIP weights by selecting their subset as weight inheritance for further retraining via mask optimization or important weight measurement. However, these select-based weight inheritance often compromises the feature presentation ability, especially on the extreme compression. In this paper, we propose a novel mapping-based CLIP compression framework, CLIP-Map. It leverages learnable matrices to map and combine pretrained weights by Full-Mapping with Kronecker Factorization, aiming to preserve as much information from the original weights as possible. To mitigate the optimization challenges introduced by the learnable mapping, we propose Diagonal Inheritance Initialization to reduce the distribution shifting problem for efficient and effective mapping learning. Extensive experimental results demonstrate that the proposed CLIP-Map outperforms select-based frameworks across various compression ratios, with particularly significant gains observed under high compression settings.

  </details>



- **Regularized Calibration with Successive Rounding for Post-Training Quantization**  
  Seohyeon Cha, Huancheng Chen, Dongjun Kim, Haoran Zhang, Kevin Chan, Gustavo de Veciana, Haris Vikalo  
  _2026-02-05_ · https://arxiv.org/abs/2602.05902v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) deliver robust performance across diverse applications, yet their deployment often faces challenges due to the memory and latency costs of storing and accessing billions of parameters. Post-training quantization (PTQ) enables efficient inference by mapping pretrained weights to low-bit formats without retraining, but its effectiveness depends critically on both the quantization objective and the rounding procedure used to obtain low-bit weight representations. In this work, we show that interpolating between symmetric and asymmetric calibration acts as a form of regularization that preserves the standard quadratic structure used in PTQ while providing robustness to activation mismatch. Building on this perspective, we derive a simple successive rounding procedure that naturally incorporates asymmetric calibration, as well as a bounded-search extension that allows for an explicit trade-off between quantization quality and the compute cost. Experiments across multiple LLM families, quantization bit-widths, and benchmarks demonstrate that the proposed bounded search based on a regularized asymmetric calibration objective consistently improves perplexity and accuracy over PTQ baselines, while incurring only modest and controllable additional computational cost.

  </details>



- **Interpreting Manifolds and Graph Neural Embeddings from Internet of Things Traffic Flows**  
  Enrique Feito-Casares, Francisco M. Melgarejo-Meseguer, Elena Casiraghi, Giorgio Valentini, José-Luis Rojo-Álvarez  
  _2026-02-05_ · https://arxiv.org/abs/2602.05817v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  The rapid expansion of Internet of Things (IoT) ecosystems has led to increasingly complex and heterogeneous network topologies. Traditional network monitoring and visualization tools rely on aggregated metrics or static representations, which fail to capture the evolving relationships and structural dependencies between devices. Although Graph Neural Networks (GNNs) offer a powerful way to learn from relational data, their internal representations often remain opaque and difficult to interpret for security-critical operations. Consequently, this work introduces an interpretable pipeline that generates directly visualizable low-dimensional representations by mapping high-dimensional embeddings onto a latent manifold. This projection enables the interpretable monitoring and interoperability of evolving network states, while integrated feature attribution techniques decode the specific characteristics shaping the manifold structure. The framework achieves a classification F1-score of 0.830 for intrusion detection while also highlighting phenomena such as concept drift. Ultimately, the presented approach bridges the gap between high-dimensional GNN embeddings and human-understandable network behavior, offering new insights for network administrators and security analysts.

  </details>



- **Disc-Centric Contrastive Learning for Lumbar Spine Severity Grading**  
  Sajjan Acharya, Pralisha Kansakar  
  _2026-02-05_ · https://arxiv.org/abs/2602.05738v1 · `eess.IV`  
  <details><summary>Abstract</summary>

  This work examines a disc-centric approach for automated severity grading of lumbar spinal stenosis from sagittal T2-weighted MRI. The method combines contrastive pretraining with disc-level fine-tuning, using a single anatomically localized region of interest per intervertebral disc. Contrastive learning is employed to help the model focus on meaningful disc features and reduce sensitivity to irrelevant differences in image appearance. The framework includes an auxiliary regression task for disc localization and applies weighted focal loss to address class imbalance. Experiments demonstrate a 78.1% balanced accuracy and a reduced severe-to-normal misclassification rate of 2.13% compared with supervised training from scratch. Detecting discs with moderate severity can still be challenging, but focusing on disc-level features provides a practical way to assess the lumbar spinal stenosis.

  </details>



- **CSRv2: Unlocking Ultra-Sparse Embeddings**  
  Lixuan Guo, Yifei Wang, Tiansheng Wen, Yifan Wang, Aosong Feng, Bo Chen, Stefanie Jegelka, Chenyu You  
  _2026-02-05_ · https://arxiv.org/abs/2602.05735v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  In the era of large foundation models, the quality of embeddings has become a central determinant of downstream task performance and overall system capability. Yet widely used dense embeddings are often extremely high-dimensional, incurring substantial costs in storage, memory, and inference latency. To address these, Contrastive Sparse Representation (CSR) is recently proposed as a promising direction, mapping dense embeddings into high-dimensional but k-sparse vectors, in contrast to compact dense embeddings such as Matryoshka Representation Learning (MRL). Despite its promise, CSR suffers severe degradation in the ultra-sparse regime, where over 80% of neurons remain inactive, leaving much of its efficiency potential unrealized. In this paper, we introduce CSRv2, a principled training approach designed to make ultra-sparse embeddings viable. CSRv2 stabilizes sparsity learning through progressive k-annealing, enhances representational quality via supervised contrastive objectives, and ensures end-to-end adaptability with full backbone finetuning. CSRv2 reduces dead neurons from 80% to 20% and delivers a 14% accuracy gain at k=2, bringing ultra-sparse embeddings on par with CSR at k=8 and MRL at 32 dimensions, all with only two active features. While maintaining comparable performance, CSRv2 delivers a 7x speedup over MRL, and yields up to 300x improvements in compute and memory efficiency relative to dense embeddings in text representation. Extensive experiments across text and vision demonstrate that CSRv2 makes ultra-sparse embeddings practical without compromising performance, where CSRv2 achieves 7%/4% improvement over CSR when k=4 and further increases this gap to 14%/6% when k=2 in text/vision representation. By making extreme sparsity viable, CSRv2 broadens the design space for real-time and edge-deployable AI systems where both embedding quality and efficiency are critical.

  </details>



- **Exploring the Temporal Consistency for Point-Level Weakly-Supervised Temporal Action Localization**  
  Yunchuan Ma, Laiyun Qing, Guorong Li, Yuqing Liu, Yuankai Qi, Qingming Huang  
  _2026-02-05_ · https://arxiv.org/abs/2602.05718v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Point-supervised Temporal Action Localization (PTAL) adopts a lightly frame-annotated paradigm (\textit{i.e.}, labeling only a single frame per action instance) to train a model to effectively locate action instances within untrimmed videos. Most existing approaches design the task head of models with only a point-supervised snippet-level classification, without explicit modeling of understanding temporal relationships among frames of an action. However, understanding the temporal relationships of frames is crucial because it can help a model understand how an action is defined and therefore benefits localizing the full frames of an action. To this end, in this paper, we design a multi-task learning framework that fully utilizes point supervision to boost the model's temporal understanding capability for action localization. Specifically, we design three self-supervised temporal understanding tasks: (i) Action Completion, (ii) Action Order Understanding, and (iii) Action Regularity Understanding. These tasks help a model understand the temporal consistency of actions across videos. To the best of our knowledge, this is the first attempt to explicitly explore temporal consistency for point supervision action localization. Extensive experimental results on four benchmark datasets demonstrate the effectiveness of the proposed method compared to several state-of-the-art approaches.

  </details>



- **Mode-Dependent Rectification for Stable PPO Training**  
  Mohamad Mohamad, Francesco Ponzio, Xavier Descombes  
  _2026-02-05_ · https://arxiv.org/abs/2602.05619v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Mode-dependent architectural components (layers that behave differently during training and evaluation, such as Batch Normalization or dropout) are commonly used in visual reinforcement learning but can destabilize on-policy optimization. We show that in Proximal Policy Optimization (PPO), discrepancies between training and evaluation behavior induced by Batch Normalization lead to policy mismatch, distributional drift, and reward collapse. We propose Mode-Dependent Rectification (MDR), a lightweight dual-phase training procedure that stabilizes PPO under mode-dependent layers without architectural changes. Experiments across procedurally generated games and real-world patch-localization tasks demonstrate that MDR consistently improves stability and performance, and extends naturally to other mode-dependent layers.

  </details>



- **A Mixed Reality System for Robust Manikin Localization in Childbirth Training**  
  Haojie Cheng, Chang Liu, Abhiram Kanneganti, Mahesh Arjandas Choolani, Arundhati Tushar Gosavi, Eng Tat Khoo  
  _2026-02-05_ · https://arxiv.org/abs/2602.05588v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Opportunities for medical students to gain practical experience in vaginal births are increasingly constrained by shortened clinical rotations, patient reluctance, and the unpredictable nature of labour. To alleviate clinicians' instructional burden and enhance trainees' learning efficiency, we introduce a mixed reality (MR) system for childbirth training that combines virtual guidance with tactile manikin interaction, thereby preserving authentic haptic feedback while enabling independent practice without continuous on-site expert supervision. The system extends the passthrough capability of commercial head-mounted displays (HMDs) by spatially calibrating an external RGB-D camera, allowing real-time visual integration of physical training objects. Building on this capability, we implement a coarse-to-fine localization pipeline that first aligns the maternal manikin with fiducial markers to define a delivery region and then registers the pre-scanned neonatal head within this area. This process enables spatially accurate overlay of virtual guiding hands near the manikin, allowing trainees to follow expert trajectories reinforced by haptic interaction. Experimental evaluations demonstrate that the system achieves accurate and stable manikin localization on a standalone headset, ensuring practical deployment without external computing resources. A large-scale user study involving 83 fourth-year medical students was subsequently conducted to compare MR-based and virtual reality (VR)-based childbirth training. Four senior obstetricians independently assessed performance using standardized criteria. Results showed that MR training achieved significantly higher scores in delivery, post-delivery, and overall task performance, and was consistently preferred by trainees over VR training.

  </details>



- **Geometric Observability Index: An Operator-Theoretic Framework for Per-Feature Sensitivity, Weak Observability, and Dynamic Effects in SE(3) Pose Estimation**  
  Joe-Mei Feng, Sheng-Wei Yu  
  _2026-02-05_ · https://arxiv.org/abs/2602.05582v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present a unified operator-theoretic framework for analyzing per-feature sensitivity in camera pose estimation on the Lie group SE(3). Classical sensitivity tools - conditioning analyses, Euclidean perturbation arguments, and Fisher information bounds - do not explain how individual image features influence the pose estimate, nor why dynamic or inconsistent observations can disproportionately distort modern SLAM and structure-from-motion systems. To address this gap, we extend influence function theory to matrix Lie groups and derive an intrinsic perturbation operator for left-trivialized M-estimators on SE(3). The resulting Geometric Observability Index (GOI) quantifies the contribution of a single measurement through the curvature operator and the Lie algebraic structure of the observable subspace. GOI admits a spectral decomposition along the principal directions of the observable curvature, revealing a direct correspondence between weak observability and amplified sensitivity. In the population regime, GOI coincides with the Fisher information geometry on SE(3), yielding a single-measurement analogue of the Cramer-Rao bound. The same spectral mechanism explains classical degeneracies such as pure rotation and vanishing parallax, as well as dynamic feature amplification along weak curvature directions. Overall, GOI provides a geometrically consistent description of measurement influence that unifies conditioning analysis, Fisher information geometry, influence function theory, and dynamic scene detectability through the spectral geometry of the curvature operator. Because these quantities arise directly within Gauss-Newton pipelines, the curvature spectrum and GOI also yield lightweight, training-free diagnostic signals for identifying dynamic features and detecting weak observability configurations without modifying existing SLAM architectures.

  </details>


