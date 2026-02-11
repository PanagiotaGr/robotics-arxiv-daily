# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-11 07:16 UTC_

Total papers shown: **3**


---

- **Hydra-Nav: Object Navigation via Adaptive Dual-Process Reasoning**  
  Zixuan Wang, Huang Fang, Shaoan Wang, Yuanfei Luo, Heng Dong, Wei Li, Yiming Gan  
  _2026-02-10_ · https://arxiv.org/abs/2602.09972v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While large vision-language models (VLMs) show promise for object goal navigation, current methods still struggle with low success rates and inefficient localization of unseen objects--failures primarily attributed to weak temporal-spatial reasoning. Meanwhile, recent attempts to inject reasoning into VLM-based agents improve success rates but incur substantial computational overhead. To address both the ineffectiveness and inefficiency of existing approaches, we introduce Hydra-Nav, a unified VLM architecture that adaptively switches between a deliberative slow system for analyzing exploration history and formulating high-level plans, and a reactive fast system for efficient execution. We train Hydra-Nav through a three-stage curriculum: (i) spatial-action alignment to strengthen trajectory planning, (ii) memory-reasoning integration to enhance temporal-spatial reasoning over long-horizon exploration, and (iii) iterative rejection fine-tuning to enable selective reasoning at critical decision points. Extensive experiments demonstrate that Hydra-Nav achieves state-of-the-art performance on the HM3D, MP3D, and OVON benchmarks, outperforming the second-best methods by 11.1%, 17.4%, and 21.2%, respectively. Furthermore, we introduce SOT (Success weighted by Operation Time), a new metric to measure search efficiency across VLMs with varying reasoning intensity. Results show that adaptive reasoning significantly enhances search efficiency over fixed-frequency baselines.

  </details>



- **Robust Vision Systems for Connected and Autonomous Vehicles: Security Challenges and Attack Vectors**  
  Sandeep Gupta, Roberto Passerone  
  _2026-02-10_ · https://arxiv.org/abs/2602.09740v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This article investigates the robustness of vision systems in Connected and Autonomous Vehicles (CAVs), which is critical for developing Level-5 autonomous driving capabilities. Safe and reliable CAV navigation undeniably depends on robust vision systems that enable accurate detection of objects, lane markings, and traffic signage. We analyze the key sensors and vision components essential for CAV navigation to derive a reference architecture for CAV vision system (CAVVS). This reference architecture provides a basis for identifying potential attack surfaces of CAVVS. Subsequently, we elaborate on identified attack vectors targeting each attack surface, rigorously evaluating their implications for confidentiality, integrity, and availability (CIA). Our study provides a comprehensive understanding of attack vector dynamics in vision systems, which is crucial for formulating robust security measures that can uphold the principles of the CIA triad.

  </details>



- **Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings**  
  Alexander Fertig, Karthikeyan Chandra Sekaran, Lakshman Balasubramanian, Michael Botsch  
  _2026-02-10_ · https://arxiv.org/abs/2602.09985v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  As autonomous vehicles are rolled out, measures must be taken to ensure their safe operation. In order to supervise a system that is already in operation, monitoring frameworks are frequently employed. These run continuously online in the background, supervising the system status and recording anomalies. This work proposes an online monitoring framework to detect anomalies in object state representations. Thereby, a key challenge is creating a framework for anomaly detection without anomaly labels, which are usually unavailable for unknown anomalies. To address this issue, this work applies a self-supervised embedding method to translate object data into a latent representation space. For this, a JEPA-based self-supervised prediction task is constructed, allowing training without anomaly labels and the creation of rich object embeddings. The resulting expressive JEPA embeddings serve as input for established anomaly detection methods, in order to identify anomalies within object state representations. This framework is particularly useful for applications in real-world environments, where new or unknown anomalies may occur during operation for which there are no labels available. Experiments performed on the publicly available, real-world nuScenes dataset illustrate the framework's capabilities.

  </details>


