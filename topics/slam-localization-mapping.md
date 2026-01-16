# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-01-16 14:08 UTC_

Total papers shown: **6**


---

- **SVII-3D: Advancing Roadside Infrastructure Inventory with Decimeter-level 3D Localization and Comprehension from Sparse Street Imagery**  
  Chong Liu, Luxuan Fu, Yang Jia, Zhen Dong, Bisheng Yang  
  _2026-01-15_ · https://arxiv.org/abs/2601.10535v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The automated creation of digital twins and precise asset inventories is a critical task in smart city construction and facility lifecycle management. However, utilizing cost-effective sparse imagery remains challenging due to limited robustness, inaccurate localization, and a lack of fine-grained state understanding. To address these limitations, SVII-3D, a unified framework for holistic asset digitization, is proposed. First, LoRA fine-tuned open-set detection is fused with a spatial-attention matching network to robustly associate observations across sparse views. Second, a geometry-guided refinement mechanism is introduced to resolve structural errors, achieving precise decimeter-level 3D localization. Third, transcending static geometric mapping, a Vision-Language Model agent leveraging multi-modal prompting is incorporated to automatically diagnose fine-grained operational states. Experiments demonstrate that SVII-3D significantly improves identification accuracy and minimizes localization errors. Consequently, this framework offers a scalable, cost-effective solution for high-fidelity infrastructure digitization, effectively bridging the gap between sparse perception and automated intelligent maintenance.

  </details>



- **RAG-3DSG: Enhancing 3D Scene Graphs with Re-Shot Guided Retrieval-Augmented Generation**  
  Yue Chang, Rufeng Chen, Zhaofan Zhang, Yi Chen, Sihong Xie  
  _2026-01-15_ · https://arxiv.org/abs/2601.10168v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Open-vocabulary 3D Scene Graph (3DSG) generation can enhance various downstream tasks in robotics, such as manipulation and navigation, by leveraging structured semantic representations. A 3DSG is constructed from multiple images of a scene, where objects are represented as nodes and relationships as edges. However, existing works for open-vocabulary 3DSG generation suffer from both low object-level recognition accuracy and speed, mainly due to constrained viewpoints, occlusions, and redundant surface density. To address these challenges, we propose RAG-3DSG to mitigate aggregation noise through re-shot guided uncertainty estimation and support object-level Retrieval-Augmented Generation (RAG) via reliable low-uncertainty objects. Furthermore, we propose a dynamic downsample-mapping strategy to accelerate cross-image object aggregation with adaptive granularity. Experiments on Replica dataset demonstrate that RAG-3DSG significantly improves node captioning accuracy in 3DSG generation while reducing the mapping time by two-thirds compared to the vanilla version.

  </details>



- **Advancing Adaptive Multi-Stage Video Anomaly Reasoning: A Benchmark Dataset and Method**  
  Chao Huang, Benfeng Wang, Wei Wang, Jie Wen, Li Shen, Wenqi Ren, Yong Xu, Xiaochun Cao  
  _2026-01-15_ · https://arxiv.org/abs/2601.10165v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent progress in reasoning capabilities of Multimodal Large Language Models(MLLMs) has highlighted their potential for performing complex video understanding tasks. However, in the domain of Video Anomaly Detection and Understanding (VAD&U), existing MLLM-based methods are largely limited to anomaly localization or post-hoc description, lacking explicit reasoning processes, risk awareness, and decision-oriented interpretation. To address this gap, we define a new task termed Video Anomaly Reasoning (VAR), which elevates video anomaly analysis from descriptive understanding to structured, multi-stage reasoning. VAR explicitly requires models to perform progressive reasoning over anomalous events before answering anomaly-related questions, encompassing visual perception, causal interpretation, and risk-aware decision making. To support this task, we present a new dataset with 8,641 videos, where each video is annotated with diverse question types corresponding to different reasoning depths, totaling more than 50,000 samples, making it one of the largest datasets for video anomaly. The annotations are based on a structured Perception-Cognition-Action Chain-of-Thought (PerCoAct-CoT), which formalizes domain-specific reasoning priors for video anomaly understanding. This design enables systematic evaluation of multi-stage and adaptive anomaly reasoning. In addition, we propose Anomaly-Aware Group Relative Policy Optimization to further enhance reasoning reliability under weak supervision. Building upon the proposed task and dataset, we develop an end-to-end MLLM-based VAR model termed Vad-R1-Plus, which supports adaptive hierarchical reasoning and risk-aware decision making. Extensive experiments demonstrate that the proposed benchmark and method effectively advance the reasoning capabilities of MLLMs on VAR tasks, outperforming both open-source and proprietary baselines.

  </details>



- **CoGen: Creation of Reusable UI Components in Figma via Textual Commands**  
  Ishani Kanapathipillai, Obhasha Priyankara  
  _2026-01-15_ · https://arxiv.org/abs/2601.10536v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  The evolution of User Interface design has emphasized the need for efficient, reusable, and editable components to ensure an efficient design process. This research introduces CoGen, a system that uses machine learning techniques to generate reusable UI components directly in Figma, one of the most popular UI design tools. Addressing gaps in current systems, CoGen focuses on creating atomic components such as buttons, labels, and input fields using structured JSON and natural language prompts. The project integrates Figma API data extraction, Seq2Seq models, and fine-tuned T5 transformers for component generation. The key results demonstrate the efficiency of the T5 model in prompt generation, with an accuracy of 98% and a BLEU score of 0.2668, which ensures the mapping of JSON to descriptive prompts. For JSON creation, CoGen achieves a success rate of up to 100% in generating simple JSON outputs for specified component types.

  </details>



- **H-EFT-VA: An Effective-Field-Theory Variational Ansatz with Provable Barren Plateau Avoidance**  
  Eyad I. B Hamid  
  _2026-01-15_ · https://arxiv.org/abs/2601.10479v1 · `quant-ph`  
  <details><summary>Abstract</summary>

  Variational Quantum Algorithms (VQAs) are critically threatened by the Barren Plateau (BP) phenomenon. In this work, we introduce the H-EFT Variational Ansatz (H-EFT-VA), an architecture inspired by Effective Field Theory (EFT). By enforcing a hierarchical "UV-cutoff" on initialization, we theoretically restrict the circuit's state exploration, preventing the formation of approximate unitary 2-designs. We provide a rigorous proof that this localization guarantees an inverse-polynomial lower bound on the gradient variance: $Var[\partial θ] \in Ω(1/poly(N))$. Crucially, unlike approaches that avoid BPs by limiting entanglement, we demonstrate that H-EFT-VA maintains volume-law entanglement and near-Haar purity, ensuring sufficient expressibility for complex quantum states. Extensive benchmarking across 16 experiments -- including Transverse Field Ising and Heisenberg XXZ models -- confirms a 109x improvement in energy convergence and a 10.7x increase in ground-state fidelity over standard Hardware-Efficient Ansatze (HEA), with a statistical significance of $p < 10^{-88}$.

  </details>



- **An effective interactive brain cytoarchitectonic parcellation framework using pretrained foundation model**  
  Shiqi Zhang, Fang Xu, Pengcheng Zhou  
  _2026-01-15_ · https://arxiv.org/abs/2601.10412v1 · `eess.IV`  
  <details><summary>Abstract</summary>

  Cytoarchitectonic mapping provides anatomically grounded parcellations of brain structure and forms a foundation for integrative, multi-modal neuroscience analyses. These parcellations are defined based on the shape, density, and spatial arrangement of neuronal cell bodies observed in histological imaging. Recent works have demonstrated the potential of using deep learning models toward fully automatic segmentation of cytoarchitectonic areas in large-scale datasets, but performance is mainly constrained by the scarcity of training labels and the variability of staining and imaging conditions. To address these challenges, we propose an interactive cytoarchitectonic parcellation framework that leverages the strong transferability of the DINOv3 vision transformer. Our framework combines (i) multi-layer DINOv3 feature fusion, (ii) a lightweight segmentation decoder, and (iii) real-time user-guided training from sparse scribbles. This design enables rapid human-in-the-loop refinement while maintaining high segmentation accuracy. Compared with training an nnU-Net from scratch, transfer learning with DINOv3 yields markedly improved performance. We also show that features extracted by DINOv3 exhibit clear anatomical correspondence and demonstrate the method's practical utility for brain region segmentation using sparse labels. These results highlight the potential of foundation-model-driven interactive segmentation for scalable and efficient cytoarchitectonic mapping.

  </details>


