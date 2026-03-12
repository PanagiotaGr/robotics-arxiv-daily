# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-03-12 07:09 UTC_

Total papers shown: **18**


---

- **Learning Bimanual Cloth Manipulation with Vision-based Tactile Sensing via Single Robotic Arm**  
  Dongmyoung Lee, Wei Chen, Xiaoshuai Chen, Rui Zong, Petar Kormushev  
  _2026-03-11_ · https://arxiv.org/abs/2603.10609v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic cloth manipulation remains challenging due to the high-dimensional state space of fabrics, their deformable nature, and frequent occlusions that limit vision-based sensing. Although dual-arm systems can mitigate some of these issues, they increase hardware and control complexity. This paper presents Touch G.O.G., a compact vision-based tactile gripper and perception/control framework for single-arm bimanual cloth manipulation. The proposed framework combines three key components: (1) a novel gripper design and control strategy for in-gripper cloth sliding with a single robot arm, (2) a Vision Foundation Model-backboned Vision Transformer pipeline for cloth part classification (PC-Net) and edge pose estimation (PE-Net) using real and synthetic tactile images, and (3) an encoder-decoder synthetic data generator (SD-Net) that reduces manual annotation by producing high-fidelity tactile images. Experiments show 96% accuracy in distinguishing edges, corners, interior regions, and grasp failures, together with sub-millimeter edge localization and 4.5° orientation error. Real-world results demonstrate reliable cloth unfolding, even for crumpled fabrics, using only a single robotic arm. These results highlight Touch G.O.G. as a compact and cost-effective solution for deformable object manipulation.

  </details>



- **Semantic Landmark Particle Filter for Robot Localisation in Vineyards**  
  Rajitha de Silva, Jonathan Cox, James R. Heselden, Marija Popović, Cesar Cadena, Riccardo Polvara  
  _2026-03-11_ · https://arxiv.org/abs/2603.10847v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable localisation in vineyards is hindered by row-level perceptual aliasing: parallel crop rows produce nearly identical LiDAR observations, causing geometry-only and vision-based SLAM systems to converge towards incorrect corridors, particularly during headland transitions. We present a Semantic Landmark Particle Filter (SLPF) that integrates trunk and pole landmark detections with 2D LiDAR within a probabilistic localisation framework. Detected trunks are converted into semantic walls, forming structural row boundaries embedded in the measurement model to improve discrimination between adjacent rows. GNSS is incorporated as a lightweight prior that stabilises localisation when semantic observations are sparse. Field experiments in a 10-row vineyard demonstrate consistent improvements over geometry-only (AMCL), vision-based (RTAB-Map), and GNSS baselines. Compared to AMCL, SLPF reduces Absolute Pose Error by 22% and 65% across two traversal directions; relative to a NoisyGNSS baseline, APE decreases by 65% and 61%. Row correctness improves from 0.67 to 0.73, while mean cross-track error decreases from 1.40 m to 1.26 m. These results show that embedding row-level structural semantics within the measurement model enables robust localisation in highly repetitive outdoor agricultural environments.

  </details>



- **Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction**  
  Hao Zhou, Lu Qi, Jason Li, Jie Zhang, Yi Liu, Xu Yang, Mingyu Fan, Fei Luo  
  _2026-03-11_ · https://arxiv.org/abs/2603.10597v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Trajectory prediction is critical for autonomous driving, enabling safe and efficient planning in dense, dynamic traffic. Most existing methods optimize prediction accuracy under fixed-length observations. However, real-world driving often yields variable-length, incomplete observations, posing a challenge to these methods. A common strategy is to directly map features from incomplete observations to those from complete ones. This one-shot mapping, however, struggles to learn accurate representations for short trajectories due to significant information gaps. To address this issue, we propose a Progressive Retrospective Framework (PRF), which gradually aligns features from incomplete observations with those from complete ones via a cascade of retrospective units. Each unit consists of a Retrospective Distillation Module (RDM) and a Retrospective Prediction Module (RPM), where RDM distills features and RPM recovers previous timesteps using the distilled features. Moreover, we propose a Rolling-Start Training Strategy (RSTS) that enhances data efficiency during PRF training. PRF is plug-and-play with existing methods. Extensive experiments on datasets Argoverse 2 and Argoverse 1 demonstrate the effectiveness of PRF. Code is available at https://github.com/zhouhao94/PRF.

  </details>



- **Cross-Species Transfer Learning for Electrophysiology-to-Transcriptomics Mapping in Cortical GABAergic Interneurons**  
  Theo Schwider, Ramin Ramezani  
  _2026-03-11_ · https://arxiv.org/abs/2603.11000v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Single-cell electrophysiological recordings provide a powerful window into neuronal functional diversity and offer an interpretable route for linking intrinsic physiology to transcriptomic identity. Here, we replicate and extend the electrophysiology-to-transcriptomics framework introduced by Gouwens et al. (2020) using publicly available Allen Institute Patch-seq datasets from both mouse and human cortex. We focus on GABAergic inhibitory interneurons to target a subclass structure (Lamp5, Pvalb, Sst, Vip) that is comparable and conserved across species. After quality control, we analyzed 3,699 mouse visual cortex neurons and 506 human neocortical neurons from neurosurgical resections. Using standardized electrophysiological features and sparse PCA, we reproduced the major class-level separations reported in the original mouse study. For supervised prediction, a class-balanced random forest provided a strong feature-engineered baseline in mouse data and a reduced but still informative baseline in human data. We then developed an attention-based BiLSTM that operates directly on the structured IPFX feature-family representation, avoiding sPCA and providing feature-family-level interpretability via learned attention weights. Finally, we evaluated a cross-species transfer setting in which the sequence model is pretrained on mouse data and fine-tuned on human data for an aligned 4-class task, improving human macro-F1 relative to a human-only training baseline. Together, these results confirm reproducibility of the Gouwens pipeline in mouse data, demonstrate that sequence models can match feature-engineered baselines, and show that mouse-to-human transfer learning can provide measurable gains for human subclass prediction.

  </details>



- **CUPID: A Plug-in Framework for Joint Aleatoric and Epistemic Uncertainty Estimation with a Single Model**  
  Xinran Xu, Xiuyi Fan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10745v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Accurate estimation of uncertainty in deep learning is critical for deploying models in high-stakes domains such as medical diagnosis and autonomous decision-making, where overconfident predictions can lead to harmful outcomes. In practice, understanding the reason behind a model's uncertainty and the type of uncertainty it represents can support risk-aware decisions, enhance user trust, and guide additional data collection. However, many existing methods only address a single type of uncertainty or require modifications and retraining of the base model, making them difficult to adopt in real-world systems. We introduce CUPID (Comprehensive Uncertainty Plug-in estImation moDel), a general-purpose module that jointly estimates aleatoric and epistemic uncertainty without modifying or retraining the base model. CUPID can be flexibly inserted into any layer of a pretrained network. It models aleatoric uncertainty through a learned Bayesian identity mapping and captures epistemic uncertainty by analyzing the model's internal responses to structured perturbations. We evaluate CUPID across a range of tasks, including classification, regression, and out-of-distribution detection. The results show that it consistently delivers competitive performance while offering layer-wise insights into the origins of uncertainty. By making uncertainty estimation modular, interpretable, and model-agnostic, CUPID supports more transparent and trustworthy AI. Related code and data are available at https://github.com/a-Fomalhaut-a/CUPID.

  </details>



- **TacLoc: Global Tactile Localization on Objects from a Registration Perspective**  
  Zirui Zhang, Boyang Zhang, Fumin Zhang, Huan Yin  
  _2026-03-11_ · https://arxiv.org/abs/2603.10565v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Pose estimation is essential for robotic manipulation, particularly when visual perception is occluded during gripper-object interactions. Existing tactile-based methods generally rely on tactile simulation or pre-trained models, which limits their generalizability and efficiency. In this study, we propose TacLoc, a novel tactile localization framework that formulates the problem as a one-shot point cloud registration task. TacLoc introduces a graph-theoretic partial-to-full registration method, leveraging dense point clouds and surface normals from tactile sensing for efficient and accurate pose estimation. Without requiring rendered data or pre-trained models, TacLoc achieves improved performance through normal-guided graph pruning and a hypothesis-and-verification pipeline. TacLoc is evaluated extensively on the YCB dataset. We further demonstrate TacLoc on real-world objects across two different visual-tactile sensors.

  </details>



- **UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery**  
  Islam Guven, Mehmet Parlak  
  _2026-03-11_ · https://arxiv.org/abs/2603.10528v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Unmanned aerial vehicles (UAVs) are increasingly used to support time-critical medical supply delivery, providing rapid and flexible logistics during emergencies and resource shortages. However, effective deployment of UAV fleets requires coordination mechanisms capable of prioritizing medical requests, allocating limited aerial resources, and adapting delivery schedules under uncertain operational conditions. This paper presents a multi-agent reinforcement learning (MARL) framework for coordinating UAV fleets in stochastic medical delivery scenarios where requests vary in urgency, location, and delivery deadlines. The problem is formulated as a partially observable Markov decision process (POMDP) in which UAV agents maintain awareness of medical delivery demands while having limited visibility of other agents due to communication and localization constraints. The proposed framework employs Proximal Policy Optimization (PPO) as the primary learning algorithm and evaluates several variants, including asynchronous extensions, classical actor--critic methods, and architectural modifications to analyze scalability and performance trade-offs. The model is evaluated using real-world geographic data from selected clinics and hospitals extracted from the OpenStreetMap dataset. The framework provides a decision-support layer that prioritizes medical tasks, reallocates UAV resources in real time, and assists healthcare personnel in managing urgent logistics. Experimental results show that classical PPO achieves superior coordination performance compared to asynchronous and sequential learning strategies, highlighting the potential of reinforcement learning for adaptive and scalable UAV-assisted healthcare logistics.

  </details>



- **Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation**  
  Tao Zhong, Yixun Hu, Dongzhe Zheng, Aditya Sood, Christine Allen-Blanchette  
  _2026-03-11_ · https://arxiv.org/abs/2603.11045v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We propose Neural Field Thermal Tomography (NeFTY), a differentiable physics framework for the quantitative 3D reconstruction of material properties from transient surface temperature measurements. While traditional thermography relies on pixel-wise 1D approximations that neglect lateral diffusion, and soft-constrained Physics-Informed Neural Networks (PINNs) often fail in transient diffusion scenarios due to gradient stiffness, NeFTY parameterizes the 3D diffusivity field as a continuous neural field optimized through a rigorous numerical solver. By leveraging a differentiable physics solver, our approach enforces thermodynamic laws as hard constraints while maintaining the memory efficiency required for high-resolution 3D tomography. Our discretize-then-optimize paradigm effectively mitigates the spectral bias and ill-posedness inherent in inverse heat conduction, enabling the recovery of subsurface defects at arbitrary scales. Experimental validation on synthetic data demonstrates that NeFTY significantly improves the accuracy of subsurface defect localization over baselines. Additional details at https://cab-lab-princeton.github.io/nefty/

  </details>



- **GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations**  
  Boyuan Chen, Minghao Shao, Siddharth Garg, Ramesh Karri, Muhammad Shafique  
  _2026-03-11_ · https://arxiv.org/abs/2603.10978v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision Language Models (VLMs) exhibit persistent hallucinations in counting tasks, with accuracy substantially lower than other visual reasoning tasks (excluding sentiment). This phenomenon persists even in state-of-the-art reasoning-capable VLMs. Conversely, CNN-based object detection models (ODMs) such as YOLO excel at spatial localization and instance counting with minimal computational overhead. We propose GroundCount, a framework that augments VLMs with explicit spatial grounding from ODMs to mitigate counting hallucinations. In the best case, our prompt-based augmentation strategy achieves 81.3% counting accuracy on the best-performing model (Ovis2.5-2B) - a 6.6pp improvement - while reducing inference time by 22% through elimination of hallucination-driven reasoning loops for stronger models. We conduct comprehensive ablation studies demonstrating that positional encoding is a critical component, being beneficial for stronger models but detrimental for weaker ones. Confidence scores, by contrast, introduce noise for most architectures and their removal improves performance in four of five evaluated models. We further evaluate feature-level fusion architectures, finding that explicit symbolic grounding via structured prompts outperforms implicit feature fusion despite sophisticated cross-attention mechanisms. Our approach yields consistent improvements across four of five evaluated VLM architectures (6.2--7.5pp), with one architecture exhibiting degraded performance due to incompatibility between its iterative reflection mechanisms and structured prompts. These results suggest that counting failures stem from fundamental spatial-semantic integration limitations rather than architecture-specific deficiencies, while highlighting the importance of architectural compatibility in augmentation strategies.

  </details>



- **A Hybrid Knowledge-Grounded Framework for Safety and Traceability in Prescription Verification**  
  Yichi Zhu, Kan Ling, Xu Liu, Hengrun Zhang, Huiqun Yu, Guisheng Fan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10891v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Medication errors pose a significant threat to patient safety, making pharmacist verification (PV) a critical, yet heavily burdened, final safeguard. The direct application of Large Language Models (LLMs) to this zero-tolerance domain is untenable due to their inherent factual unreliability, lack of traceability, and weakness in complex reasoning. To address these challenges, we introduce PharmGraph-Auditor, a novel system designed for safe and evidence-grounded prescription auditing. The core of our system is a trustworthy Hybrid Pharmaceutical Knowledge Base (HPKB), implemented under the Virtual Knowledge Graph (VKG) paradigm. This architecture strategically unifies a relational component for set constraint satisfaction and a graph component for topological reasoning via a rigorous mapping layer. To construct this HPKB, we propose the Iterative Schema Refinement (ISR) algorithm, a framework that enables the co-evolution of both graph and relational schemas from medical texts. For auditing, we introduce the KB-grounded Chain of Verification (CoV), a new reasoning paradigm that transforms the LLM from an unreliable generator into a transparent reasoning engine. CoV decomposes the audit task into a sequence of verifiable queries against the HPKB, generating hybrid query plans to retrieve evidence from the most appropriate data store. Experimental results demonstrate robust knowledge extraction capabilities and show promises of using PharmGraph-Auditor to enable pharmacists to achieve safer and faster prescription verification.

  </details>



- **An Extreme Multi-label Text Classification (XMTC) Library Dataset: What if we took "Use of Practical AI in Digital Libraries" seriously?**  
  Jennifer D'Souza, Sameer Sadruddin, Maximilian Kähler, Andrea Salfinger, Luca Zaccagna, Francesca Incitti, Lauro Snidaro, Osma Suominen  
  _2026-03-11_ · https://arxiv.org/abs/2603.10876v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Subject indexing is vital for discovery but hard to sustain at scale and across languages. We release a large bilingual (English/German) corpus of catalog records annotated with the Integrated Authority File (GND), plus a machine-actionable GND taxonomy. The resource enables ontology-aware multi-label classification, mapping text to authority terms, and agent-assisted cataloging with reproducible, authority-grounded evaluation. We provide a brief statistical profile and qualitative error analyses of three systems. We invite the community to assess not only accuracy but usefulness and transparency, toward authority-anchored AI co-pilots that amplify catalogers' work.

  </details>



- **6ABOS: An Open-Source Atmospheric Correction Framework for the EnMAP Hyperspectral Mission Based on 6S**  
  Gabriel Caballero Cañas, Bárbara Alvado Arranz, Xavier Sòria-Perpinyà, Antonio Ruiz-Verdú, Jesús Delegido, José Moreno  
  _2026-03-11_ · https://arxiv.org/abs/2603.10856v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The Environmental Mapping and Analysis Program (EnMAP) mission has opened new frontiers in the monitoring of optically complex environments. However, the accurate retrieval of surface reflectance over water bodies remains a significant challenge, as the water-leaving signal typically accounts for only a small fraction of the total radiance, being easily obscured by atmospheric scattering and surface reflection effects. This paper introduces 6ABOS (6S-based Atmospheric Background Offset Subtraction), a novel open-source Python framework designed to automate the atmospheric correction (AC) of EnMAP hyperspectral imagery. By leveraging the Second Simulation of the Satellite Signal in the Solar Spectrum (6S) radiative transfer model, 6ABOS implements a physically-based inversion scheme that accounts for Rayleigh scattering, aerosol interactions, and gaseous absorption. The framework integrates automated EnMAP metadata parsing with dynamic atmospheric parameter retrieval via the Google Earth Engine (GEE) Application Programming Interface (API). Validation was conducted over two Mediterranean inland water reservoirs with contrasting trophic states: the oligotrophic Benag{'e}ber and the hypertrophic Bell{'u}s. Results demonstrate a high degree of spectral similarity between in situ measurements and EnMAP-derived water-leaving reflectances. The Spectral Angle Mapper (SAM) values remained consistently low (SAM $<$ 10$^\circ$) across both study sites. 6ABOS is distributed via conda-forge, providing the scientific community with a scalable, transparent, and reproducible open-science tool for advancing hyperspectral aquatic research in the cloud-computing era.

  </details>



- **UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis**  
  Yali Zhu, Kang Zhou, Dingbang Wu, Gaofeng Meng  
  _2026-03-11_ · https://arxiv.org/abs/2603.10852v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Breast ultrasound diagnosis typically proceeds from global lesion localization to local sign assessment and then evidence integration to assign a BI-RADS category and determine benignity or malignancy. Many existing methods rely on end-to-end prediction or provide only weakly grounded evidence, which can miss fine-grained lesion cues and limit auditability and clinical review. To align with the clinical workflow and improve evidence traceability, we propose a hierarchical multi-agent framework, termed UltrasoundAgents. A main agent localizes the lesion in the full image and triggers a crop-and-zoom operation. A sub-agent analyzes the local view and predicts four clinically relevant attributes, namely echogenicity pattern, calcification, boundary type, and edge (margin) morphology. The main agent then integrates these structured attributes to perform evidence-based reasoning and output the BI-RADS category and the malignancy prediction, while producing reviewable intermediate evidence. Furthermore, hierarchical multi-agent training often suffers from error propagation, difficult credit assignment, and sparse rewards. To alleviate this and improve training stability, we introduce a decoupled progressive training strategy. We first train the attribute agent, then train the main agent with oracle attributes to learn robust attribute-based reasoning, and finally apply corrective trajectory self-distillation with spatial supervision to build high-quality trajectories for supervised fine-tuning, yielding a deployable end-to-end policy. Experiments show consistent gains over strong vision-language baselines in diagnostic accuracy and attribute agreement, together with structured evidence and traceable reasoning.

  </details>



- **Evaluating Few-Shot Pill Recognition Under Visual Domain Shift**  
  W. I. Chu, G. Tarroni, L. Li  
  _2026-03-11_ · https://arxiv.org/abs/2603.10833v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Adverse drug events are a significant source of preventable harm, which has led to the development of automated pill recognition systems to enhance medication safety. Real-world deployment of these systems is hindered by visually complex conditions, including cluttered scenes, overlapping pills, reflections, and diverse acquisition environments. This study investigates few-shot pill recognition from a deployment-oriented perspective, prioritizing generalization under realistic cross-dataset domain shifts over architectural innovation. A two-stage object detection framework is employed, involving base training followed by few-shot fine-tuning. Models are adapted to novel pill classes using one, five, or ten labeled examples per class and are evaluated on a separate deployment dataset featuring multi-object, cluttered scenes. The evaluation focuses on classification-centric and error-based metrics to address heterogeneous annotation strategies. Findings indicate that semantic pill recognition adapts rapidly with few-shot supervision, with classification performance reaching saturation even with a single labeled example. However, stress testing under overlapping and occluded conditions demonstrates a marked decline in localization and recall, despite robust semantic classification. Models trained on visually realistic, multi-pill data consistently exhibit greater robustness in low-shot scenarios, underscoring the importance of training data realism and the diagnostic utility of few-shot fine-tuning for deployment readiness.

  </details>



- **HanMoVLM: Large Vision-Language Models for Professional Artistic Painting Evaluation**  
  Hongji Yang, Yucheng Zhou, Wencheng Han, Songlian Li, Xiaotong Zhao, Jianbing Shen  
  _2026-03-11_ · https://arxiv.org/abs/2603.10814v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  While Large Vision-Language Models (VLMs) demonstrate impressive general visual capabilities, they remain artistically blind and unable to offer professional evaluation of artworks within specific artistic domains like human experts. To bridge this gap, we transform VLMs into experts capable of professional-grade painting evaluation in the Chinese Artistic Domain, which is more abstract and demands extensive artistic training for evaluation. We introduce HanMo-Bench, a new dataset that features authentic auction-grade masterpieces and AI-generated works, grounded in real-world market valuations. To realize the rigorous judgment, we propose the HanMoVLM and construct a Chain-of-Thought (CoT) validated by experts. This CoT guides the model to perform expert-level reasoning: from content identification and Region of Interest (RoI) localization to professional evaluation, guided by both theme-specific evaluation and typical three-tier evaluation in Chinese paintings. Furthermore, we design a reward function to refine the reasoning process of the HanMoVLM to improve the accuracy. We demonstrate that HanMoVLM can serve as a critical backbone for Test-time Scaling in image generation. By acting as a high-quality verifier, HanMoVLM enables generative models to select the most artistically superior outputs from multiple candidates. Experimental results and human studies confirm that the proposed HanMoVLM effectively bridges the gap, achieving a high consistency with professional experts and significantly improving the quality of Chinese Painting generation.

  </details>



- **Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning**  
  Zhaowei Zhang, Xiaohan Liu, Xuekai Zhu, Junchao Huang, Ceyao Zhang, Zhiyuan Feng, Yaodong Yang, Xiaoyuan Yi, Xing Xie  
  _2026-03-11_ · https://arxiv.org/abs/2603.10588v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Reinforcement learning with verifiable rewards (RLVR) has achieved remarkable success in logical reasoning tasks, yet whether large language model (LLM) alignment requires fundamentally different approaches remains unclear. Given the apparent tolerance for multiple valid responses in moral reasoning, a natural hypothesis is that alignment tasks inherently require diversity-seeking distribution-matching algorithms rather than reward-maximizing policy-based methods. We conduct the first comprehensive empirical study comparing both paradigms on MoReBench. To enable stable RLVR training, we build a rubric-grounded reward pipeline by training a Qwen3-1.7B judge model. Contrary to our hypothesis, we find that distribution-matching approaches do not demonstrate significant advantages over reward-maximizing methods as expected on alignment tasks. Through semantic visualization mapping high-reward responses to semantic space, we demonstrate that moral reasoning exhibits more concentrated high-reward distributions than mathematical reasoning, where diverse solution strategies yield similarly high rewards. This counter-intuitive finding explains why mode-seeking optimization proves equally or more effective for alignment tasks. Our results suggest that alignment tasks do not inherently require diversity-preserving algorithms, and standard reward-maximizing RLVR methods can effectively transfer to moral reasoning without explicit diversity mechanisms.

  </details>



- **Towards Cognitive Defect Analysis in Active Infrared Thermography with Vision-Text Cues**  
  Mohammed Salah, Eman Ouda, Giuseppe Dell'Avvocato, Fabrizio Sarasini, Ester D'Accardi, Jorge Dias, Davor Svetinovic, Stefano Sfarra, Yusra Abdulrahman  
  _2026-03-11_ · https://arxiv.org/abs/2603.10549v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Active infrared thermography (AIRT) is currently witnessing a surge of artificial intelligence (AI) methodologies being deployed for automated subsurface defect analysis of high performance carbon fiber-reinforced polymers (CFRP). Deploying AI-based AIRT methodologies for inspecting CFRPs requires the creation of time consuming and expensive datasets of CFRP inspection sequences to train neural networks. To address this challenge, this work introduces a novel language-guided framework for cognitive defect analysis in CFRPs using AIRT and vision-language models (VLMs). Unlike conventional learning-based approaches, the proposed framework does not require developing training datasets for extensive training of defect detectors, instead it relies solely on pretrained multimodal VLM encoders coupled with a lightweight adapter to enable generative zero-shot understanding and localization of subsurface defects. By leveraging pretrained multimodal encoders, the proposed system enables generative zero-shot understanding of thermographic patterns and automatic detection of subsurface defects. Given the domain gap between thermographic data and natural images used to train VLMs, an AIRT-VLM Adapter is proposed to enhance the visibility of defects while aligning the thermographic domain with the learned representations of VLMs. The proposed framework is validated using three representative VLMs; specifically, GroundingDINO, Qwen-VL-Chat, and CogVLM. Validation is performed on 25 CFRP inspection sequences with impacts introduced at different energy levels, reflecting realistic defects encountered in industrial scenarios. Experimental results demonstrate that the AIRT-VLM adapter achieves signal-to-noise ratio (SNR) gains exceeding 10 dB compared with conventional thermographic dimensionality-reduction methods, while enabling zero-shot defect detection with intersection-over-union values reaching 70%.

  </details>



- **Prompting with the human-touch: evaluating model-sensitivity of foundation models for musculoskeletal CT segmentation**  
  Caroline Magg, Maaike A. ter Wee, Johannes G. G. Dobbe, Geert J. Streekstra, Leendert Blankevoort, Clara I. Sánchez, Hoel Kervadec  
  _2026-03-11_ · https://arxiv.org/abs/2603.10541v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Promptable Foundation Models (FMs), initially introduced for natural image segmentation, have also revolutionized medical image segmentation. The increasing number of models, along with evaluations varying in datasets, metrics, and compared models, makes direct performance comparison between models difficult and complicates the selection of the most suitable model for specific clinical tasks. In our study, 11 promptable FMs are tested using non-iterative 2D and 3D prompting strategies on a private and public dataset focusing on bone and implant segmentation in four anatomical regions (wrist, shoulder, hip and lower leg). The Pareto-optimal models are identified and further analyzed using human prompts collected through a dedicated observer study. Our findings are: 1) The segmentation performance varies a lot between FMs and prompting strategies; 2) The Pareto-optimal models in 2D are SAM and SAM2.1, in 3D nnInteractive and Med-SAM2; 3) Localization accuracy and rater consistency vary with anatomical structures, with higher consistency for simple structures (wrist bones) and lower consistency for complex structures (pelvis, tibia, implants); 4) The segmentation performance drops using human prompts, suggesting that performance reported on "ideal" prompts extracted from reference labels might overestimate the performance in a human-driven setting; 5) All models were sensitive to prompt variations. While two models demonstrated intra-rater robustness, it did not scale to inter-rater settings. We conclude that the selection of the most optimal FM for a human-driven setting remains challenging, with even high-performing FMs being sensitive to variations in human input prompts. Our code base for prompt extraction and model inference is available: https://github.com/CarolineMagg/segmentation-FM-benchmark/

  </details>


