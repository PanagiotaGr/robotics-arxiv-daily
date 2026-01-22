# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-01-22 06:51 UTC_

Total papers shown: **10**


---

- **DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration**  
  Dominik Rößle, Xujun Xie, Adithya Mohan, Venkatesh Thirugnana Sambandham, Daniel Cremers, Torsten Schön  
  _2026-01-21_ · https://arxiv.org/abs/2601.15260v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Perception is a cornerstone of autonomous driving, enabling vehicles to understand their surroundings and make safe, reliable decisions. Developing robust perception algorithms requires large-scale, high-quality datasets that cover diverse driving conditions and support thorough evaluation. Existing datasets often lack a high-fidelity digital twin, limiting systematic testing, edge-case simulation, sensor modification, and sim-to-real evaluations. To address this gap, we present DrivIng, a large-scale multimodal dataset with a complete geo-referenced digital twin of a ~18 km route spanning urban, suburban, and highway segments. Our dataset provides continuous recordings from six RGB cameras, one LiDAR, and high-precision ADMA-based localization, captured across day, dusk, and night. All sequences are annotated at 10 Hz with 3D bounding boxes and track IDs across 12 classes, yielding ~1.2 million annotated instances. Alongside the benefits of a digital twin, DrivIng enables a 1-to-1 transfer of real traffic into simulation, preserving agent interactions while enabling realistic and flexible scenario testing. To support reproducible research and robust validation, we benchmark DrivIng with state-of-the-art perception models and publicly release the dataset, digital twin, HD map, and codebase.

  </details>



- **The Why Behind the Action: Unveiling Internal Drivers via Agentic Attribution**  
  Chen Qian, Peng Wang, Dongrui Liu, Junyao Yang, Dadi Guo, Ling Tang, Jilin Mei, Qihan Ren, Shuai Shao, Yong Liu, et al.  
  _2026-01-21_ · https://arxiv.org/abs/2601.15075v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Model (LLM)-based agents are widely used in real-world applications such as customer service, web navigation, and software engineering. As these systems become more autonomous and are deployed at scale, understanding why an agent takes a particular action becomes increasingly important for accountability and governance. However, existing research predominantly focuses on \textit{failure attribution} to localize explicit errors in unsuccessful trajectories, which is insufficient for explaining the reasoning behind agent behaviors. To bridge this gap, we propose a novel framework for \textbf{general agentic attribution}, designed to identify the internal factors driving agent actions regardless of the task outcome. Our framework operates hierarchically to manage the complexity of agent interactions. Specifically, at the \textit{component level}, we employ temporal likelihood dynamics to identify critical interaction steps; then at the \textit{sentence level}, we refine this localization using perturbation-based analysis to isolate the specific textual evidence. We validate our framework across a diverse suite of agentic scenarios, including standard tool use and subtle reliability risks like memory-induced bias. Experimental results demonstrate that the proposed framework reliably pinpoints pivotal historical events and sentences behind the agent behavior, offering a critical step toward safer and more accountable agentic systems.

  </details>



- **Finite-Sample Inference for Sparsely Permuted Linear Regression**  
  Hirofumi Ota, Masaaki Imaizumi  
  _2026-01-21_ · https://arxiv.org/abs/2601.14872v1 · `math.ST`  
  <details><summary>Abstract</summary>

  We study a noisy linear observation model with an unknown permutation called permuted/shuffled linear regression, where responses and covariates are mismatched and the permutation forms a discrete, factorial-size parameter. This unknown permutation is a key component of the data-generating process, yet its statistical investigation remains challenging due to its discrete nature. In this study, we develop a general statistical inference framework on the permutation and regression coefficients. First, we introduce a localization step that reduces the permutation space to a small candidate set building on recent advances in the repro samples method, whose miscoverage decays polynomially with the number of Monte Carlo samples. Then, based on this localized set, we provide statistical inference procedures: a conditional Monte Carlo test of permutation structures with valid finite-sample Type-I error control. We also develop coefficient inference that remains valid under alignment uncertainty of permutations. For computational purposes, we develop a linear assignment problem computable in polynomial time complexity and demonstrate that its solution asymptotically converges to that of the conventional least squares problem with large computational cost. Extensions to partially permuted designs and ridge regularization are also discussed. Extensive simulations and an application to Beijing air-quality data corroborate finite-sample validity, strong power to detect mismatches, and practical scalability.

  </details>



- **On-the-fly hand-eye calibration for the da Vinci surgical robot**  
  Zejian Cui, Ferdinando Rodriguez y Baena  
  _2026-01-21_ · https://arxiv.org/abs/2601.14871v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In Robot-Assisted Minimally Invasive Surgery (RMIS), accurate tool localization is crucial to ensure patient safety and successful task execution. However, this remains challenging for cable-driven robots, such as the da Vinci robot, because erroneous encoder readings lead to pose estimation errors. In this study, we propose a calibration framework to produce accurate tool localization results through computing the hand-eye transformation matrix on-the-fly. The framework consists of two interrelated algorithms: the feature association block and the hand-eye calibration block, which provide robust correspondences for key points detected on monocular images without pre-training, and offer the versatility to accommodate various surgical scenarios by adopting an array of filter approaches, respectively. To validate its efficacy, we test the framework extensively on publicly available video datasets that feature multiple surgical instruments conducting tasks in both in vitro and ex vivo scenarios, under varying illumination conditions and with different levels of key point measurement accuracy. The results show a significant reduction in tool localization errors under the proposed calibration framework, with accuracies comparable to other state-of-the-art methods while being more time-efficient.

  </details>



- **Tracing 3D Anatomy in 2D Strokes: A Multi-Stage Projection Driven Approach to Cervical Spine Fracture Identification**  
  Fabi Nahian Madhurja, Rusab Sarmun, Muhammad E. H. Chowdhury, Adam Mushtak, Israa Al-Hashimi, Sohaib Bassam Zoghoul  
  _2026-01-21_ · https://arxiv.org/abs/2601.15235v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Cervical spine fractures are critical medical conditions requiring precise and efficient detection for effective clinical management. This study explores the viability of 2D projection-based vertebra segmentation for vertebra-level fracture detection in 3D CT volumes, presenting an end-to-end pipeline for automated analysis of cervical vertebrae (C1-C7). By approximating a 3D volume through optimized 2D axial, sagittal, and coronal projections, regions of interest are identified using the YOLOv8 model from all views and combined to approximate the 3D cervical spine area, achieving a 3D mIoU of 94.45 percent. This projection-based localization strategy reduces computational complexity compared to traditional 3D segmentation methods while maintaining high performance. It is followed by a DenseNet121-Unet-based multi-label segmentation leveraging variance- and energy-based projections, achieving a Dice score of 87.86 percent. Strategic approximation of 3D vertebral masks from these 2D segmentation masks enables the extraction of individual vertebra volumes. The volumes are analyzed for fractures using an ensemble of 2.5D Spatio-Sequential models incorporating both raw slices and projections per vertebra for complementary evaluation. This ensemble achieves vertebra-level and patient-level F1 scores of 68.15 and 82.26, and ROC-AUC scores of 91.62 and 83.04, respectively. We further validate our approach through an explainability study that provides saliency map visualizations highlighting anatomical regions relevant for diagnosis, and an interobserver variability analysis comparing our model's performance with expert radiologists, demonstrating competitive results.

  </details>



- **Federated Transformer-GNN for Privacy-Preserving Brain Tumor Localization with Modality-Level Explainability**  
  Andrea Protani, Riccardo Taiello, Marc Molina Van Den Bosch, Luigi Serio  
  _2026-01-21_ · https://arxiv.org/abs/2601.15042v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Deep learning models for brain tumor analysis require large and diverse datasets that are often siloed across healthcare institutions due to privacy regulations. We present a federated learning framework for brain tumor localization that enables multi-institutional collaboration without sharing sensitive patient data. Our method extends a hybrid Transformer-Graph Neural Network architecture derived from prior decoder-free supervoxel GNNs and is deployed within CAFEIN\textsuperscript{\textregistered}, CERN's federated learning platform designed for healthcare environments. We provide an explainability analysis through Transformer attention mechanisms that reveals which MRI modalities drive the model predictions. Experiments on the BraTS dataset demonstrate a key finding: while isolated training on individual client data triggers early stopping well before reaching full training capacity, federated learning enables continued model improvement by leveraging distributed data, ultimately matching centralized performance. This result provides strong justification for federated learning when dealing with complex tasks and high-dimensional input data, as aggregating knowledge from multiple institutions significantly benefits the learning process. Our explainability analysis, validated through rigorous statistical testing on the full test set (paired t-tests with Bonferroni correction), reveals that deeper network layers significantly increase attention to T2 and FLAIR modalities ($p<0.001$, Cohen's $d$=1.50), aligning with clinical practice.

  </details>



- **Reconstruction-Anchored Diffusion Model for Text-to-Motion Generation**  
  Yifei Liu, Changxing Ding, Ling Guo, Huaiguang Jiang, Qiong Cao  
  _2026-01-21_ · https://arxiv.org/abs/2601.14788v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Diffusion models have seen widespread adoption for text-driven human motion generation and related tasks due to their impressive generative capabilities and flexibility. However, current motion diffusion models face two major limitations: a representational gap caused by pre-trained text encoders that lack motion-specific information, and error propagation during the iterative denoising process. This paper introduces Reconstruction-Anchored Diffusion Model (RAM) to address these challenges. First, RAM leverages a motion latent space as intermediate supervision for text-to-motion generation. To this end, RAM co-trains a motion reconstruction branch with two key objective functions: self-regularization to enhance the discrimination of the motion space and motion-centric latent alignment to enable accurate mapping from text to the motion latent space. Second, we propose Reconstructive Error Guidance (REG), a testing-stage guidance mechanism that exploits the diffusion model's inherent self-correction ability to mitigate error propagation. At each denoising step, REG uses the motion reconstruction branch to reconstruct the previous estimate, reproducing the prior error patterns. By amplifying the residual between the current prediction and the reconstructed estimate, REG highlights the improvements in the current prediction. Extensive experiments demonstrate that RAM achieves significant improvements and state-of-the-art performance. Our code will be released.

  </details>



- **Safeguarding Facial Identity against Diffusion-based Face Swapping via Cascading Pathway Disruption**  
  Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo  
  _2026-01-21_ · https://arxiv.org/abs/2601.14738v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The rapid evolution of diffusion models has democratized face swapping but also raises concerns about privacy and identity security. Existing proactive defenses, often adapted from image editing attacks, prove ineffective in this context. We attribute this failure to an oversight of the structural resilience and the unique static conditional guidance mechanism inherent in face swapping systems. To address this, we propose VoidFace, a systemic defense method that views face swapping as a coupled identity pathway. By injecting perturbations at critical bottlenecks, VoidFace induces cascading disruption throughout the pipeline. Specifically, we first introduce localization disruption and identity erasure to degrade physical regression and semantic embeddings, thereby impairing the accurate modeling of the source face. We then intervene in the generative domain by decoupling attention mechanisms to sever identity injection, and corrupting intermediate diffusion features to prevent the reconstruction of source identity. To ensure visual imperceptibility, we perform adversarial search in the latent manifold, guided by a perceptual adaptive strategy to balance attack potency with image quality. Extensive experiments show that VoidFace outperforms existing defenses across various diffusion-based swapping models, while producing adversarial faces with superior visual quality.

  </details>



- **When Text-as-Vision Meets Semantic IDs in Generative Recommendation: An Empirical Study**  
  Shutong Qiao, Wei Yuan, Tong Chen, Xiangyu Zhao, Quoc Viet Hung Nguyen, Hongzhi Yin  
  _2026-01-21_ · https://arxiv.org/abs/2601.14697v1 · `cs.IR`  
  <details><summary>Abstract</summary>

  Semantic ID learning is a key interface in Generative Recommendation (GR) models, mapping items to discrete identifiers grounded in side information, most commonly via a pretrained text encoder. However, these text encoders are primarily optimized for well-formed natural language. In real-world recommendation data, item descriptions are often symbolic and attribute-centric, containing numerals, units, and abbreviations. These text encoders can break these signals into fragmented tokens, weakening semantic coherence and distorting relationships among attributes. Worse still, when moving to multimodal GR, relying on standard text encoders introduces an additional obstacle: text and image embeddings often exhibit mismatched geometric structures, making cross-modal fusion less effective and less stable. In this paper, we revisit representation design for Semantic ID learning by treating text as a visual signal. We conduct a systematic empirical study of OCR-based text representations, obtained by rendering item descriptions into images and encoding them with vision-based OCR models. Experiments across four datasets and two generative backbones show that OCR-text consistently matches or surpasses standard text embeddings for Semantic ID learning in both unimodal and multimodal settings. Furthermore, we find that OCR-based Semantic IDs remain robust under extreme spatial-resolution compression, indicating strong robustness and efficiency in practical deployments.

  </details>



- **Beyond Error-Based Optimization: Experience-Driven Symbolic Regression with Goal-Conditioned Reinforcement Learning**  
  Jianwen Sun, Xinrui Li, Fuqing Li, Xiaoxuan Shen  
  _2026-01-21_ · https://arxiv.org/abs/2601.14693v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Symbolic Regression aims to automatically identify compact and interpretable mathematical expressions that model the functional relationship between input and output variables. Most existing search-based symbolic regression methods typically rely on the fitting error to inform the search process. However, in the vast expression space, numerous candidate expressions may exhibit similar error values while differing substantially in structure, leading to ambiguous search directions and hindering convergence to the underlying true function. To address this challenge, we propose a novel framework named EGRL-SR (Experience-driven Goal-conditioned Reinforcement Learning for Symbolic Regression). In contrast to traditional error-driven approaches, EGRL-SR introduces a new perspective: leveraging precise historical trajectories and optimizing the action-value network to proactively guide the search process, thereby achieving a more robust expression search. Specifically, we formulate symbolic regression as a goal-conditioned reinforcement learning problem and incorporate hindsight experience replay, allowing the action-value network to generalize common mapping patterns from diverse input-output pairs. Moreover, we design an all-point satisfaction binary reward function that encourages the action-value network to focus on structural patterns rather than low-error expressions, and concurrently propose a structure-guided heuristic exploration strategy to enhance search diversity and space coverage. Experiments on public benchmarks show that EGRL-SR consistently outperforms state-of-the-art methods in recovery rate and robustness, and can recover more complex expressions under the same search budget. Ablation results validate that the action-value network effectively guides the search, with both the reward function and the exploration strategy playing critical roles.

  </details>


