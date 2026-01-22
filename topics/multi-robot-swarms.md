# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-01-22 06:51 UTC_

Total papers shown: **3**


---

- **Integrated Sensing, Communication and Control enabled Agile UAV Swarm**  
  Zhiqing Wei, Yucong Du, Zhiyong Feng, Haotian Liu, Yanpeng Cui, Tao Zhang, Ying Zhou, Huici Wu  
  _2026-01-21_ · https://arxiv.org/abs/2601.14783v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Uncrewed aerial vehicle (UAV) swarms are pivotal in the applications such as disaster relief, aerial base station (BS) and logistics transportation. These scenarios require the capabilities in accurate sensing, efficient communication and flexible control for real-time and reliable task execution. However, sensing, communication and control are studied independently in traditional research, which limits the overall performance of UAV swarms. To overcome this disadvantage, we propose a deeply coupled scheme of integrated sensing, communication and control (ISCC) for UAV swarms, which is a systemic paradigm that transcends traditional isolated designs of sensing, communication and control by establishing a tightly-coupled closed-loop through the co-optimization of sensing, communication and control. In this article, we firstly analyze the requirements of scenarios and key performance metrics. Subsequently, the enabling technologies are proposed, including communication-and-control-enhanced sensing, sensing-and-control-enhanced communication, and sensing-and-communication-enhanced control. Simulation results validate the performance of the proposed ISCC framework, demonstrating its application potential in the future.

  </details>



- **Federated Transformer-GNN for Privacy-Preserving Brain Tumor Localization with Modality-Level Explainability**  
  Andrea Protani, Riccardo Taiello, Marc Molina Van Den Bosch, Luigi Serio  
  _2026-01-21_ · https://arxiv.org/abs/2601.15042v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Deep learning models for brain tumor analysis require large and diverse datasets that are often siloed across healthcare institutions due to privacy regulations. We present a federated learning framework for brain tumor localization that enables multi-institutional collaboration without sharing sensitive patient data. Our method extends a hybrid Transformer-Graph Neural Network architecture derived from prior decoder-free supervoxel GNNs and is deployed within CAFEIN\textsuperscript{\textregistered}, CERN's federated learning platform designed for healthcare environments. We provide an explainability analysis through Transformer attention mechanisms that reveals which MRI modalities drive the model predictions. Experiments on the BraTS dataset demonstrate a key finding: while isolated training on individual client data triggers early stopping well before reaching full training capacity, federated learning enables continued model improvement by leveraging distributed data, ultimately matching centralized performance. This result provides strong justification for federated learning when dealing with complex tasks and high-dimensional input data, as aggregating knowledge from multiple institutions significantly benefits the learning process. Our explainability analysis, validated through rigorous statistical testing on the full test set (paired t-tests with Bonferroni correction), reveals that deeper network layers significantly increase attention to T2 and FLAIR modalities ($p<0.001$, Cohen's $d$=1.50), aligning with clinical practice.

  </details>



- **Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models**  
  Injin Kong, Hyoungjoon Lee, Yohan Jo  
  _2026-01-21_ · https://arxiv.org/abs/2601.14758v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Post-training pretrained Autoregressive models (ARMs) into Masked Diffusion models (MDMs) has emerged as a cost-effective strategy to overcome the limitations of sequential generation. However, the internal algorithmic transformations induced by this paradigm shift remain unexplored, leaving it unclear whether post-trained MDMs acquire genuine bidirectional reasoning capabilities or merely repackage autoregressive heuristics. In this work, we address this question by conducting a comparative circuit analysis of ARMs and their MDM counterparts. Our analysis reveals a systematic "mechanism shift" dependent on the structural nature of the task. Structurally, we observe a distinct divergence: while MDMs largely retain autoregressive circuitry for tasks dominated by local causal dependencies, they abandon initialized pathways for global planning tasks, exhibiting distinct rewiring characterized by increased early-layer processing. Semantically, we identify a transition from sharp, localized specialization in ARMs to distributed integration in MDMs. Through these findings, we conclude that diffusion post-training does not merely adapt model parameters but fundamentally reorganizes internal computation to support non-sequential global planning.

  </details>


