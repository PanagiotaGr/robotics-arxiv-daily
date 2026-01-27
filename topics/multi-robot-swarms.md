# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-01-27 06:52 UTC_

Total papers shown: **5**


---

- **COMETS: Coordinated Multi-Destination Video Transmission with In-Network Rate Adaptation**  
  Yulong Zhang, Ying Cui, Zili Meng, Abhishek Kumar, Dirk Kutscher  
  _2026-01-26_ · https://arxiv.org/abs/2601.18670v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  Large-scale video streaming events attract millions of simultaneous viewers, stressing existing delivery infrastructures. Client-driven adaptation reacts slowly to shared congestion, while server-based coordination introduces scalability bottlenecks and single points of failure. We present COMETS, a coordinated multi-destination video transmission framework that leverages information-centric networking principles such as request aggregation and in-network state awareness to enable scalable, fair, and adaptive rate control. COMETS introduces a novel range-interest protocol and distributed in-network decision process that aligns video quality across receiver groups while minimizing redundant transmissions. To achieve this, we develop a lightweight distributed optimization framework that guides per-hop quality adaptation without centralized control. Extensive emulation shows that COMETS consistently improves bandwidth utilization, fairness, and user-perceived quality of experience over DASH, MoQ, and ICN baselines, particularly under high concurrency. The results highlight COMETS as a practical, deployable approach for next-generation scalable video delivery.

  </details>



- **Advances and Innovations in the Multi-Agent Robotic System (MARS) Challenge**  
  Li Kang, Heng Zhou, Xiufeng Song, Rui Li, Bruno N. Y. Chen, Ziye Wang, Ximeng Meng, Stone Tao, Yiran Qin, Xiaohong Liu, et al.  
  _2026-01-26_ · https://arxiv.org/abs/2601.18733v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advancements in multimodal large language models and vision-languageaction models have significantly driven progress in Embodied AI. As the field transitions toward more complex task scenarios, multi-agent system frameworks are becoming essential for achieving scalable, efficient, and collaborative solutions. This shift is fueled by three primary factors: increasing agent capabilities, enhancing system efficiency through task delegation, and enabling advanced human-agent interactions. To address the challenges posed by multi-agent collaboration, we propose the Multi-Agent Robotic System (MARS) Challenge, held at the NeurIPS 2025 Workshop on SpaVLE. The competition focuses on two critical areas: planning and control, where participants explore multi-agent embodied planning using vision-language models (VLMs) to coordinate tasks and policy execution to perform robotic manipulation in dynamic environments. By evaluating solutions submitted by participants, the challenge provides valuable insights into the design and coordination of embodied multi-agent systems, contributing to the future development of advanced collaborative AI systems.

  </details>



- **3DGesPolicy: Phoneme-Aware Holistic Co-Speech Gesture Generation Based on Action Control**  
  Xuanmeng Sha, Liyun Zhang, Tomohiro Mashita, Naoya Chiba, Yuki Uranishi  
  _2026-01-26_ · https://arxiv.org/abs/2601.18451v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Generating holistic co-speech gestures that integrate full-body motion with facial expressions suffers from semantically incoherent coordination on body motion and spatially unstable meaningless movements due to existing part-decomposed or frame-level regression methods, We introduce 3DGesPolicy, a novel action-based framework that reformulates holistic gesture generation as a continuous trajectory control problem through diffusion policy from robotics. By modeling frame-to-frame variations as unified holistic actions, our method effectively learns inter-frame holistic gesture motion patterns and ensures both spatially and semantically coherent movement trajectories that adhere to realistic motion manifolds. To further bridge the gap in expressive alignment, we propose a Gesture-Audio-Phoneme (GAP) fusion module that can deeply integrate and refine multi-modal signals, ensuring structured and fine-grained alignment between speech semantics, body motion, and facial expressions. Extensive quantitative and qualitative experiments on the BEAT2 dataset demonstrate the effectiveness of our 3DGesPolicy across other state-of-the-art methods in generating natural, expressive, and highly speech-aligned holistic gestures.

  </details>



- **Synchronization and Localization in Ad-Hoc ICAS Networks Using a Two-Stage Kuramoto Method**  
  Dominik Neudert-Schulz, Thomas Dallmann  
  _2026-01-26_ · https://arxiv.org/abs/2601.18643v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  To enable Integrated Communications and Sensing (ICAS) in a peer-to-peer vehicular network, precise synchronization in frequency and phase among the communicating entities is required. In addition, self-driving cars need accurate position estimates of the surrounding vehicles. In this work, we propose a joint, distributed synchronization and localization scheme for a network of communicating entities. Our proposed scheme is mostly signal-agnostic and therefore can be applied to a wide range of possible ICAS signals. We also mitigate the effect of finite sampling frequencies, which otherwise would degrade the synchronization and localization performance severely.

  </details>



- **Scalable Transit Delay Prediction at City Scale: A Systematic Approach with Multi-Resolution Feature Engineering and Deep Learning**  
  Emna Boudabbous, Mohamed Karaa, Lokman Sboui, Julio Montecinos, Omar Alam  
  _2026-01-26_ · https://arxiv.org/abs/2601.18521v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Urban bus transit agencies need reliable, network-wide delay predictions to provide accurate arrival information to passengers and support real-time operational control. Accurate predictions help passengers plan their trips, reduce waiting time, and allow operations staff to adjust headways, dispatch extra vehicles, and manage disruptions. Although real-time feeds such as GTFS-Realtime (GTFS-RT) are now widely available, most existing delay prediction systems handle only a few routes, depend on hand-crafted features, and offer little guidance on how to design a scalable, reusable architecture. We present a city-scale prediction pipeline that combines multi-resolution feature engineering, dimensionality reduction, and deep learning. The framework generates 1,683 spatiotemporal features by exploring 23 aggregation combinations over H3 cells, routes, segments, and temporal patterns, and compresses them into 83 components using Adaptive PCA while preserving 95% of the variance. To avoid the "giant cluster" problem that occurs when dense urban areas fall into a single H3 region, we introduce a hybrid H3+topology clustering method that yields 12 balanced route clusters (coefficient of variation 0.608) and enables efficient distributed training. We compare five model architectures on six months of bus operations from the Société de transport de Montréal (STM) network in Montréal. A global LSTM with cluster-aware features achieves the best trade-off between accuracy and efficiency, outperforming transformer models by 18 to 52% while using 275 times fewer parameters. We also report multi-level evaluation at the elementary segment, segment, and trip level with walk-forward validation and latency analysis, showing that the proposed pipeline is suitable for real-time, city-scale deployment and can be reused for other networks with limited adaptation.

  </details>


