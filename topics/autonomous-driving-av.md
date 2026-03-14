# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-03-14 07:02 UTC_

Total papers shown: **2**


---

- **O3N: Omnidirectional Open-Vocabulary Occupancy Prediction**  
  Mengfei Duan, Hao Shi, Fei Teng, Guoqiang Zhao, Yuheng Zhang, Zhiyong Li, Kailun Yang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12144v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Understanding and reconstructing the 3D world through omnidirectional perception is an inevitable trend in the development of autonomous agents and embodied intelligence. However, existing 3D occupancy prediction methods are constrained by limited perspective inputs and predefined training distribution, making them difficult to apply to embodied agents that require comprehensive and safe perception of scenes in open world exploration. To address this, we present O3N, the first purely visual, end-to-end Omnidirectional Open-vocabulary Occupancy predictioN framework. O3N embeds omnidirectional voxels in a polar-spiral topology via the Polar-spiral Mamba (PsM) module, enabling continuous spatial representation and long-range context modeling across 360°. The Occupancy Cost Aggregation (OCA) module introduces a principled mechanism for unifying geometric and semantic supervision within the voxel space, ensuring consistency between the reconstructed geometry and the underlying semantic structure. Moreover, Natural Modality Alignment (NMA) establishes a gradient-free alignment pathway that harmonizes visual features, voxel embeddings, and text semantics, forming a consistent "pixel-voxel-text" representation triad. Extensive experiments on multiple models demonstrate that our method not only achieves state-of-the-art performance on QuadOcc and Human360Occ benchmarks but also exhibits remarkable cross-scene generalization and semantic scalability, paving the way toward universal 3D world modeling. The source code will be made publicly available at https://github.com/MengfeiD/O3N.

  </details>



- **LABSHIELD: A Multimodal Benchmark for Safety-Critical Reasoning and Planning in Scientific Laboratories**  
  Qianpu Sun, Xiaowei Chi, Yuhan Rui, Ying Li, Kuangzhi Ge, Jiajun Li, Sirui Han, Shanghang Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11987v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Artificial intelligence is increasingly catalyzing scientific automation, with multimodal large language model (MLLM) agents evolving from lab assistants into self-driving lab operators. This transition imposes stringent safety requirements on laboratory environments, where fragile glassware, hazardous substances, and high-precision laboratory equipment render planning errors or misinterpreted risks potentially irreversible. However, the safety awareness and decision-making reliability of embodied agents in such high-stakes settings remain insufficiently defined and evaluated. To bridge this gap, we introduce LABSHIELD, a realistic multi-view benchmark designed to assess MLLMs in hazard identification and safety-critical reasoning. Grounded in U.S. Occupational Safety and Health Administration (OSHA) standards and the Globally Harmonized System (GHS), LABSHIELD establishes a rigorous safety taxonomy spanning 164 operational tasks with diverse manipulation complexities and risk profiles. We evaluate 20 proprietary models, 9 open-source models, and 3 embodied models under a dual-track evaluation framework. Our results reveal a systematic gap between general-domain MCQ accuracy and Semi-open QA safety performance, with models exhibiting an average drop of 32.0% in professional laboratory scenarios, particularly in hazard interpretation and safety-aware planning. These findings underscore the urgent necessity for safety-centric reasoning frameworks to ensure reliable autonomous scientific experimentation in embodied laboratory contexts. The full dataset will be released soon.

  </details>


