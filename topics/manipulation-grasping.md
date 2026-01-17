# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-01-17 06:44 UTC_

Total papers shown: **3**


---

- **RAG-3DSG: Enhancing 3D Scene Graphs with Re-Shot Guided Retrieval-Augmented Generation**  
  Yue Chang, Rufeng Chen, Zhaofan Zhang, Yi Chen, Sihong Xie  
  _2026-01-15_ · https://arxiv.org/abs/2601.10168v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Open-vocabulary 3D Scene Graph (3DSG) generation can enhance various downstream tasks in robotics, such as manipulation and navigation, by leveraging structured semantic representations. A 3DSG is constructed from multiple images of a scene, where objects are represented as nodes and relationships as edges. However, existing works for open-vocabulary 3DSG generation suffer from both low object-level recognition accuracy and speed, mainly due to constrained viewpoints, occlusions, and redundant surface density. To address these challenges, we propose RAG-3DSG to mitigate aggregation noise through re-shot guided uncertainty estimation and support object-level Retrieval-Augmented Generation (RAG) via reliable low-uncertainty objects. Furthermore, we propose a dynamic downsample-mapping strategy to accelerate cross-image object aggregation with adaptive granularity. Experiments on Replica dataset demonstrate that RAG-3DSG significantly improves node captioning accuracy in 3DSG generation while reducing the mapping time by two-thirds compared to the vanilla version.

  </details>



- **The impact of tactile sensor configurations on grasp learning efficiency -- a comparative evaluation in simulation**  
  Eszter Birtalan, Miklós Koller  
  _2026-01-15_ · https://arxiv.org/abs/2601.10268v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Tactile sensors are breaking into the field of robotics to provide direct information related to contact surfaces, including contact events, slip events and even texture identification. These events are especially important for robotic hand designs, including prosthetics, as they can greatly improve grasp stability. Most presently published robotic hand designs, however, implement them in vastly different densities and layouts on the hand surface, often reserving the majority of the available space. We used simulations to evaluate 6 different tactile sensor configurations with different densities and layouts, based on their impact on reinforcement learning. Our two-setup system allows for robust results that are not dependent on the use of a given physics simulator, robotic hand model or machine learning algorithm. Our results show setup-specific, as well as generalized effects across the 6 sensorized simulations, and we identify one configuration as consistently yielding the best performance across both setups. These results could help future research aimed at robotic hand designs, including prostheses.

  </details>



- **History Is Not Enough: An Adaptive Dataflow System for Financial Time-Series Synthesis**  
  Haochong Xia, Yao Long Teng, Regan Tan, Molei Qin, Xinrun Wang, Bo An  
  _2026-01-15_ · https://arxiv.org/abs/2601.10143v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  In quantitative finance, the gap between training and real-world performance-driven by concept drift and distributional non-stationarity-remains a critical obstacle for building reliable data-driven systems. Models trained on static historical data often overfit, resulting in poor generalization in dynamic markets. The mantra "History Is Not Enough" underscores the need for adaptive data generation that learns to evolve with the market rather than relying solely on past observations. We present a drift-aware dataflow system that integrates machine learning-based adaptive control into the data curation process. The system couples a parameterized data manipulation module comprising single-stock transformations, multi-stock mix-ups, and curation operations, with an adaptive planner-scheduler that employs gradient-based bi-level optimization to control the system. This design unifies data augmentation, curriculum learning, and data workflow management under a single differentiable framework, enabling provenance-aware replay and continuous data quality monitoring. Extensive experiments on forecasting and reinforcement learning trading tasks demonstrate that our framework enhances model robustness and improves risk-adjusted returns. The system provides a generalizable approach to adaptive data management and learning-guided workflow automation for financial data.

  </details>


