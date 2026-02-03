# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-03 07:06 UTC_

Total papers shown: **16**


---

- **3D Foundation Model-Based Loop Closing for Decentralized Collaborative SLAM**  
  Pierre-Yves Lajoie, Benjamin Ramtoula, Daniele De Martini, Giovanni Beltrame  
  _2026-02-02_ · https://arxiv.org/abs/2602.02430v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Decentralized Collaborative Simultaneous Localization And Mapping (C-SLAM) techniques often struggle to identify map overlaps due to significant viewpoint variations among robots. Motivated by recent advancements in 3D foundation models, which can register images despite large viewpoint differences, we propose a robust loop closing approach that leverages these models to establish inter-robot measurements. In contrast to resource-intensive methods requiring full 3D reconstruction within a centralized map, our approach integrates foundation models into existing SLAM pipelines, yielding scalable and robust multi-robot mapping. Our contributions include: (1) integrating 3D foundation models to reliably estimate relative poses from monocular image pairs within decentralized C-SLAM; (2) introducing robust outlier mitigation techniques critical to the use of these relative poses; and (3) developing specialized pose graph optimization formulations that efficiently resolve scale ambiguities. We evaluate our method against state-of-the-art approaches, demonstrating improvements in localization and mapping accuracy, alongside significant gains in computational and memory efficiency. These results highlight the potential of our approach for deployment in large-scale multi-robot scenarios.

  </details>



- **Relationship-Aware Hierarchical 3D Scene Graph for Task Reasoning**  
  Albert Gassol Puigjaner, Angelos Zacharia, Kostas Alexis  
  _2026-02-02_ · https://arxiv.org/abs/2602.02456v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Representing and understanding 3D environments in a structured manner is crucial for autonomous agents to navigate and reason about their surroundings. While traditional Simultaneous Localization and Mapping (SLAM) methods generate metric reconstructions and can be extended to metric-semantic mapping, they lack a higher level of abstraction and relational reasoning. To address this gap, 3D scene graphs have emerged as a powerful representation for capturing hierarchical structures and object relationships. In this work, we propose an enhanced hierarchical 3D scene graph that integrates open-vocabulary features across multiple abstraction levels and supports object-relational reasoning. Our approach leverages a Vision Language Model (VLM) to infer semantic relationships. Notably, we introduce a task reasoning module that combines Large Language Models (LLM) and a VLM to interpret the scene graph's semantic and relational information, enabling agents to reason about tasks and interact with their environment more intelligently. We validate our method by deploying it on a quadruped robot in multiple environments and tasks, highlighting its ability to reason about them.

  </details>



- **Mapping-Guided Task Discovery and Allocation for Robotic Inspection of Underwater Structures**  
  Marina Ruediger, Ashis G. Banerjee  
  _2026-02-02_ · https://arxiv.org/abs/2602.02389v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Task generation for underwater multi-robot inspections without prior knowledge of existing geometry can be achieved and optimized through examination of simultaneous localization and mapping (SLAM) data. By considering hardware parameters and environmental conditions, a set of tasks is generated from SLAM meshes and optimized through expected keypoint scores and distance-based pruning. In-water tests are used to demonstrate the effectiveness of the algorithm and determine the appropriate parameters. These results are compared to simulated Voronoi partitions and boustrophedon patterns for inspection coverage on a model of the test environment. The key benefits of the presented task discovery method include adaptability to unexpected geometry and distributions that maintain coverage while focusing on areas more likely to present defects or damage.

  </details>



- **Before Autonomy Takes Control: Software Testing in Robotics**  
  Nils Chur, Thiago Santos de Moura, Argentina Ortega, Sven Peldszus, Thorsten Berger, Nico Hochgeschwender, Yannic Noller  
  _2026-02-02_ · https://arxiv.org/abs/2602.02293v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Robotic systems are complex and safety-critical software systems. As such, they need to be tested thoroughly. Unfortunately, robot software is intrinsically hard to test compared to traditional software, mainly since the software needs to closely interact with hardware, account for uncertainty in its operational environment, handle disturbances, and act highly autonomously. However, given the large space in which robots operate, anticipating possible failures when designing tests is challenging. This paper presents a mapping study by considering robotics testing papers and relating them to the software testing theory. We consider 247 robotics testing papers and map them to software testing, discussing the state-of-the-art software testing in robotics with an illustrated example, and discuss current challenges. Forming the basis to introduce both the robotics and software engineering communities to software testing challenges. Finally, we identify open questions and lessons learned.

  </details>



- **Energy-Efficient Neuromorphic Computing for Edge AI: A Framework with Adaptive Spiking Neural Networks and Hardware-Aware Optimization**  
  Olaf Yunus Laitinen Imanov, Derya Umut Kulali, Taner Yilmaz, Duygu Erisken, Rana Irem Turhan  
  _2026-02-02_ · https://arxiv.org/abs/2602.02439v1 · `cs.NE`  
  <details><summary>Abstract</summary>

  Edge AI applications increasingly require ultra-low-power, low-latency inference. Neuromorphic computing based on event-driven spiking neural networks (SNNs) offers an attractive path, but practical deployment on resource-constrained devices is limited by training difficulty, hardware-mapping overheads, and sensitivity to temporal dynamics. We present NeuEdge, a framework that combines adaptive SNN models with hardware-aware optimization for edge deployment. NeuEdge uses a temporal coding scheme that blends rate and spike-timing patterns to reduce spike activity while preserving accuracy, and a hardware-aware training procedure that co-optimizes network structure and on-chip placement to improve utilization on neuromorphic processors. An adaptive threshold mechanism adjusts neuron excitability from input statistics, reducing energy consumption without degrading performance. Across standard vision and audio benchmarks, NeuEdge achieves 91-96% accuracy with up to 2.3 ms inference latency on edge hardware and an estimated 847 GOp/s/W energy efficiency. A case study on an autonomous-drone workload shows up to 312x energy savings relative to conventional deep neural networks while maintaining real-time operation.

  </details>



- **FD-VLA: Force-Distilled Vision-Language-Action Model for Contact-Rich Manipulation**  
  Ruiteng Zhao, Wenshuo Wang, Yicheng Ma, Xiaocong Li, Francis E. H. Tay, Marcelo H. Ang, Haiyue Zhu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02142v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Force sensing is a crucial modality for Vision-Language-Action (VLA) frameworks, as it enables fine-grained perception and dexterous manipulation in contact-rich tasks. We present Force-Distilled VLA (FD-VLA), a novel framework that integrates force awareness into contact-rich manipulation without relying on physical force sensors. The core of our approach is a Force Distillation Module (FDM), which distills force by mapping a learnable query token, conditioned on visual observations and robot states, into a predicted force token aligned with the latent representation of actual force signals. During inference, this distilled force token is injected into the pretrained VLM, enabling force-aware reasoning while preserving the integrity of its vision-language semantics. This design provides two key benefits: first, it allows practical deployment across a wide range of robots that lack expensive or fragile force-torque sensors, thereby reducing hardware cost and complexity; second, the FDM introduces an additional force-vision-state fusion prior to the VLM, which improves cross-modal alignment and enhances perception-action robustness in contact-rich scenarios. Surprisingly, our physical experiments show that the distilled force token outperforms direct sensor force measurements as well as other baselines, which highlights the effectiveness of this force-distilled VLA approach.

  </details>



- **Structure Enables Effective Self-Localization of Errors in LLMs**  
  Ankur Samanta, Akshayaa Magesh, Ayush Jain, Kavosh Asadi, Youliang Yu, Daniel Jiang, Boris Vidolov, Kaveh Hassani, Paul Sajda, Jalaj Bhandari, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02416v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Self-correction in language models remains elusive. In this work, we explore whether language models can explicitly localize errors in incorrect reasoning, as a path toward building AI systems that can effectively correct themselves. We introduce a prompting method that structures reasoning as discrete, semantically coherent thought steps, and show that models are able to reliably localize errors within this structure, while failing to do so in conventional, unstructured chain-of-thought reasoning. Motivated by how the human brain monitors errors at discrete decision points and resamples alternatives, we introduce Iterative Correction Sampling of Thoughts (Thought-ICS), a self-correction framework. Thought-ICS iteratively prompts the model to generate reasoning one discrete and complete thought at a time--where each thought represents a deliberate decision by the model--creating natural boundaries for precise error localization. Upon verification, the model localizes the first erroneous step, and the system backtracks to generate alternative reasoning from the last correct point. When asked to correct reasoning verified as incorrect by an oracle, Thought-ICS achieves 20-40% self-correction lift. In a completely autonomous setting without external verification, it outperforms contemporary self-correction baselines.

  </details>



- **Interpreting and Controlling LLM Reasoning through Integrated Policy Gradient**  
  Changming Li, Kaixing Zhang, Haoyun Xu, Yingdong Shi, Zheng Zhang, Kaitao Song, Kan Ren  
  _2026-02-02_ · https://arxiv.org/abs/2602.02313v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large language models (LLMs) demonstrate strong reasoning abilities in solving complex real-world problems. Yet, the internal mechanisms driving these complex reasoning behaviors remain opaque. Existing interpretability approaches targeting reasoning either identify components (e.g., neurons) correlated with special textual patterns, or rely on human-annotated contrastive pairs to derive control vectors. Consequently, current methods struggle to precisely localize complex reasoning mechanisms or capture sequential influence from model internal workings to the reasoning outputs. In this paper, built on outcome-oriented and sequential-influence-aware principles, we focus on identifying components that have sequential contribution to reasoning behavior where outcomes are cumulated by long-range effects. We propose Integrated Policy Gradient (IPG), a novel framework that attributes reasoning behaviors to model's inner components by propagating compound outcome-based signals such as post reasoning accuracy backward through model inference trajectories. Empirical evaluations demonstrate that our approach achieves more precise localization and enables reliable modulation of reasoning behaviors (e.g., reasoning capability, reasoning strength) across diverse reasoning models.

  </details>



- **AgentRx: Diagnosing AI Agent Failures from Execution Trajectories**  
  Shraddha Barke, Arnav Goyal, Alind Khare, Avaljot Singh, Suman Nath, Chetan Bansal  
  _2026-02-02_ · https://arxiv.org/abs/2602.02475v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  AI agents often fail in ways that are difficult to localize because executions are probabilistic, long-horizon, multi-agent, and mediated by noisy tool outputs. We address this gap by manually annotating failed agent runs and release a novel benchmark of 115 failed trajectories spanning structured API workflows, incident management, and open-ended web/file tasks. Each trajectory is annotated with a critical failure step and a category from a grounded-theory derived, cross-domain failure taxonomy. To mitigate the human cost of failure attribution, we present AGENTRX, an automated domain-agnostic diagnostic framework that pinpoints the critical failure step in a failed agent trajectory. It synthesizes constraints, evaluates them step-by-step, and produces an auditable validation log of constraint violations with associated evidence; an LLM-based judge uses this log to localize the critical step and category. Our framework improves step localization and failure attribution over existing baselines across three domains.

  </details>



- **Self-Supervised Learning from Structural Invariance**  
  Yipeng Zhang, Hafez Ghaemi, Jungyoon Lee, Shahab Bakhtiari, Eilif B. Muller, Laurent Charlin  
  _2026-02-02_ · https://arxiv.org/abs/2602.02381v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Joint-embedding self-supervised learning (SSL), the key paradigm for unsupervised representation learning from visual data, learns from invariances between semantically-related data pairs. We study the one-to-many mapping problem in SSL, where each datum may be mapped to multiple valid targets. This arises when data pairs come from naturally occurring generative processes, e.g., successive video frames. We show that existing methods struggle to flexibly capture this conditional uncertainty. As a remedy, we introduce a latent variable to account for this uncertainty and derive a variational lower bound on the mutual information between paired embeddings. Our derivation yields a simple regularization term for standard SSL objectives. The resulting method, which we call AdaSSL, applies to both contrastive and distillation-based SSL objectives, and we empirically show its versatility in causal representation learning, fine-grained image understanding, and world modeling on videos.

  </details>



- **NAB: Neural Adaptive Binning for Sparse-View CT reconstruction**  
  Wangduo Xie, Matthew B. Blaschko  
  _2026-02-02_ · https://arxiv.org/abs/2602.02356v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Computed Tomography (CT) plays a vital role in inspecting the internal structures of industrial objects. Furthermore, achieving high-quality CT reconstruction from sparse views is essential for reducing production costs. While classic implicit neural networks have shown promising results for sparse reconstruction, they are unable to leverage shape priors of objects. Motivated by the observation that numerous industrial objects exhibit rectangular structures, we propose a novel \textbf{N}eural \textbf{A}daptive \textbf{B}inning (\textbf{NAB}) method that effectively integrates rectangular priors into the reconstruction process. Specifically, our approach first maps coordinate space into a binned vector space. This mapping relies on an innovative binning mechanism based on differences between shifted hyperbolic tangent functions, with our extension enabling rotations around the input-plane normal vector. The resulting representations are then processed by a neural network to predict CT attenuation coefficients. This design enables end-to-end optimization of the encoding parameters -- including position, size, steepness, and rotation -- via gradient flow from the projection data, thus enhancing reconstruction accuracy. By adjusting the smoothness of the binning function, NAB can generalize to objects with more complex geometries. This research provides a new perspective on integrating shape priors into neural network-based reconstruction. Extensive experiments demonstrate that NAB achieves superior performance on two industrial datasets. It also maintains robust on medical datasets when the binning function is extended to more general expression. The code will be made available.

  </details>



- **Spectral Superposition: A Theory of Feature Geometry**  
  Georgi Ivanov, Narmeen Oozeer, Shivam Raval, Tasana Pejovic, Shriyash Upadhyay, Amir Abdullah  
  _2026-02-02_ · https://arxiv.org/abs/2602.02224v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Neural networks represent more features than they have dimensions via superposition, forcing features to share representational space. Current methods decompose activations into sparse linear features but discard geometric structure. We develop a theory for studying the geometric structre of features by analyzing the spectra (eigenvalues, eigenspaces, etc.) of weight derived matrices. In particular, we introduce the frame operator $F = WW^\top$, which gives us a spectral measure that describes how each feature allocates norm across eigenspaces. While previous tools could describe the pairwise interactions between features, spectral methods capture the global geometry (``how do all features interact?''). In toy models of superposition, we use this theory to prove that capacity saturation forces spectral localization: features collapse onto single eigenspaces, organize into tight frames, and admit discrete classification via association schemes, classifying all geometries from prior work (simplices, polygons, antiprisms). The spectral measure formalism applies to arbitrary weight matrices, enabling diagnosis of feature localization beyond toy settings. These results point toward a broader program: applying operator theory to interpretability.

  </details>



- **CIEC: Coupling Implicit and Explicit Cues for Multimodal Weakly Supervised Manipulation Localization**  
  Xinquan Yu, Wei Lu, Xiangyang Luo  
  _2026-02-02_ · https://arxiv.org/abs/2602.02175v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  To mitigate the threat of misinformation, multimodal manipulation localization has garnered growing attention. Consider that current methods rely on costly and time-consuming fine-grained annotations, such as patch/token-level annotations. This paper proposes a novel framework named Coupling Implicit and Explicit Cues (CIEC), which aims to achieve multimodal weakly-supervised manipulation localization for image-text pairs utilizing only coarse-grained image/sentence-level annotations. It comprises two branches, image-based and text-based weakly-supervised localization. For the former, we devise the Textual-guidance Refine Patch Selection (TRPS) module. It integrates forgery cues from both visual and textual perspectives to lock onto suspicious regions aided by spatial priors. Followed by the background silencing and spatial contrast constraints to suppress interference from irrelevant areas. For the latter, we devise the Visual-deviation Calibrated Token Grounding (VCTG) module. It focuses on meaningful content words and leverages relative visual bias to assist token localization. Followed by the asymmetric sparse and semantic consistency constraints to mitigate label noise and ensure reliability. Extensive experiments demonstrate the effectiveness of our CIEC, yielding results comparable to fully supervised methods on several evaluation metrics.

  </details>



- **RIS-Aided Wireless Amodal Sensing for Single-View 3D Reconstruction**  
  Yuhan Wang, Haobo Zhang, Qingyu Liu, Hongliang Zhang, Lingyang Song  
  _2026-02-02_ · https://arxiv.org/abs/2602.02148v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Amodal sensing is critical for various real-world sensing applications because it can recover the complete shapes of partially occluded objects in complex environments. Among various amodal sensing paradigms, wireless amodal sensing is a potential solution due to its advantages of environmental robustness, privacy preservation, and low cost. However, the sensing data obtained by wireless system is sparse for shape reconstruction because of the low spatial resolution, and this issue is further intensified in complex environments with occlusion. To address this issue, we propose a Reconfigurable Intelligent Surface (RIS)-aided wireless amodal sensing scheme that leverages a large-scale RIS to enhance the spatial resolution and create reflection paths that can bypass the obstacles. A generative learning model is also employed to reconstruct the complete shape based on the sensing data captured from the viewpoint of the RIS. In such a system, it is challenging to optimize the RIS phase shifts because the relationship between RIS phase shifts and amodal sensing accuracy is complex and the closed-form expression is unknown. To tackle this challenge, we develop an error prediction model that learns the mapping from RIS phase shifts to amodal sensing accuracy, and optimizes RIS phase shifts based on this mapping. Experimental results on the benchmark dataset show that our method achieves at least a 56.73% reduction in reconstruction error compared to conventional schemes under the same number of RIS configurations.

  </details>



- **Teacher-Guided Student Self-Knowledge Distillation Using Diffusion Model**  
  Yu Wang, Chuanguang Yang, Zhulin An, Weilun Feng, Jiarui Zhao, Chengqing Yu, Libo Huang, Boyu Diao, Yongjun Xu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02107v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Existing Knowledge Distillation (KD) methods often align feature information between teacher and student by exploring meaningful feature processing and loss functions. However, due to the difference in feature distributions between the teacher and student, the student model may learn incompatible information from the teacher. To address this problem, we propose teacher-guided student Diffusion Self-KD, dubbed as DSKD. Instead of the direct teacher-student alignment, we leverage the teacher classifier to guide the sampling process of denoising student features through a light-weight diffusion model. We then propose a novel locality-sensitive hashing (LSH)-guided feature distillation method between the original and denoised student features. The denoised student features encapsulate teacher knowledge and could be regarded as a teacher role. In this way, our DSKD method could eliminate discrepancies in mapping manners and feature distributions between the teacher and student, while learning meaningful knowledge from the teacher. Experiments on visual recognition tasks demonstrate that DSKD significantly outperforms existing KD methods across various models and datasets. Our code is attached in supplementary material.

  </details>



- **WADEPre: A Wavelet-based Decomposition Model for Extreme Precipitation Nowcasting with Multi-Scale Learning**  
  Baitian Liu, Haiping Zhang, Huiling Yuan, Dongjing Wang, Ying Li, Feng Chen, Hao Wu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02096v1 · `physics.ao-ph`  
  <details><summary>Abstract</summary>

  The heavy-tailed nature of precipitation intensity impedes precise precipitation nowcasting. Standard models that optimize pixel-wise losses are prone to regression-to-the-mean bias, which blurs extreme values. Existing Fourier-based methods also lack the spatial localization needed to resolve transient convective cells. To overcome these intrinsic limitations, we propose WADEPre, a wavelet-based decomposition model for extreme precipitation that transitions the modeling into the wavelet domain. By leveraging the Discrete Wavelet Transform for explicit decomposition, WADEPre employs a dual-branch architecture: an Approximation Network to model stable, low-frequency advection, isolating deterministic trends from statistical bias, and a spatially localized Detail Network to capture high-frequency stochastic convection, resolving transient singularities and preserving sharp boundaries. A subsequent Refiner module then dynamically reconstructs these decoupled multi-scale components into the final high-fidelity forecast. To address optimization instability, we introduce a multi-scale curriculum learning strategy that progressively shifts supervision from coarse scales to fine-grained details. Extensive experiments on the SEVIR and Shanghai Radar datasets demonstrate that WADEPre achieves state-of-the-art performance, yielding significant improvements in capturing extreme thresholds and maintaining structural fidelity. Our code is available at https://github.com/sonderlau/WADEPre.

  </details>


