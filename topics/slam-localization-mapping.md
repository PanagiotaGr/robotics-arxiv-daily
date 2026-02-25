# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-25 07:14 UTC_

Total papers shown: **10**


---

- **LST-SLAM: A Stereo Thermal SLAM System for Kilometer-Scale Dynamic Environments**  
  Zeyu Jiang, Kuan Xu, Changhao Chen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20925v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Thermal cameras offer strong potential for robot perception under challenging illumination and weather conditions. However, thermal Simultaneous Localization and Mapping (SLAM) remains difficult due to unreliable feature extraction, unstable motion tracking, and inconsistent global pose and map construction, particularly in dynamic large-scale outdoor environments. To address these challenges, we propose LST-SLAM, a novel large-scale stereo thermal SLAM system that achieves robust performance in complex, dynamic scenes. Our approach combines self-supervised thermal feature learning, stereo dual-level motion tracking, and geometric pose optimization. We also introduce a semantic-geometric hybrid constraint that suppresses potentially dynamic features lacking strong inter-frame geometric consistency. Furthermore, we develop an online incremental bag-of-words model for loop closure detection, coupled with global pose optimization to mitigate accumulated drift. Extensive experiments on kilometer-scale dynamic thermal datasets show that LST-SLAM significantly outperforms recent representative SLAM systems, including AirSLAM and DROID-SLAM, in both robustness and accuracy.

  </details>



- **RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction**  
  Yangfan Zhao, Hanwei Zhang, Ke Huang, Qiufeng Wang, Zhenzhou Shao, Dengyu Wu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20807v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: https://ru4d-slam.github.io

  </details>



- **Efficient Hierarchical Any-Angle Path Planning on Multi-Resolution 3D Grids**  
  Victor Reijgwart, Cesar Cadena, Roland Siegwart, Lionel Ott  
  _2026-02-24_ · https://arxiv.org/abs/2602.21174v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Hierarchical, multi-resolution volumetric mapping approaches are widely used to represent large and complex environments as they can efficiently capture their occupancy and connectivity information. Yet widely used path planning methods such as sampling and trajectory optimization do not exploit this explicit connectivity information, and search-based methods such as A* suffer from scalability issues in large-scale high-resolution maps. In many applications, Euclidean shortest paths form the underpinning of the navigation system. For such applications, any-angle planning methods, which find optimal paths by connecting corners of obstacles with straight-line segments, provide a simple and efficient solution. In this paper, we present a method that has the optimality and completeness properties of any-angle planners while overcoming computational tractability issues common to search-based methods by exploiting multi-resolution representations. Extensive experiments on real and synthetic environments demonstrate the proposed approach's solution quality and speed, outperforming even sampling-based methods. The framework is open-sourced to allow the robotics and planning community to build on our research.

  </details>



- **Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones**  
  Rong Zou, Marco Cannici, Davide Scaramuzza  
  _2026-02-24_ · https://arxiv.org/abs/2602.21101v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.

  </details>



- **Visual Cooperative Drone Tracking for Open-Path Gas Measurements**  
  Marius Schaab, Alisha Kiefer, Thomas Wiedemann, Patrick Hinsen, Achim J. Lilienthal  
  _2026-02-24_ · https://arxiv.org/abs/2602.20768v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Open-path Tunable Diode Laser Absorption Spectroscopy offers an effective method for measuring, mapping, and monitoring gas concentrations, such as leaking CO2 or methane. Compared to spatial sampling of gas distributions using in-situ sensors, open-path sensors in combination with gas tomography algorithms can cover large outdoor environments faster in a non-invasive way. However, the requirement of a dedicated reflection surface for the open-path laser makes automating the spatial sampling process challenging. This publication presents a robotic system for collecting open-path measurements, making use of a sensor mounted on a ground-based pan-tilt unit and a small drone carrying a reflector. By means of a zoom camera, the ground unit visually tracks red LED markers mounted on the drone and aligns the sensor's laser beam with the reflector. Incorporating GNSS position information provided by the drone's flight controller further improves the tracking approach. Outdoor experiments validated the system's performance, demonstrating successful autonomous tracking and valid CO2 measurements at distances up to 60 meters. Furthermore, the system successfully measured a CO2 plume without interference from the drone's propulsion system, demonstrating its superiority compared to flying in-situ sensors.

  </details>



- **Architecting AgentOS: From Token-Level Context to Emergent System-Level Intelligence**  
  ChengYou Li, XiaoDong Liu, XiangBao Meng, XinYu Zhao  
  _2026-02-24_ · https://arxiv.org/abs/2602.20934v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The paradigm of Large Language Models is undergoing a fundamental transition from static inference engines to dynamic autonomous cognitive systems.While current research primarily focuses on scaling context windows or optimizing prompt engineering the theoretical bridge between micro scale token processing and macro scale systemic intelligence remains fragmented.This paper proposes AgentOS,a holistic conceptual framework that redefines the LLM as a "Reasoning Kernel" governed by structured operating system logic.Central to this architecture is Deep Context Management which conceptualizes the context window as an Addressable Semantic Space rather than a passive buffer.We systematically deconstruct the transition from discrete sequences to coherent cognitive states introducing mechanisms for Semantic Slicing and Temporal Alignment to mitigate cognitive drift in multi-agent orchestration.By mapping classical OS abstractions such as memory paging interrupt handling and process scheduling onto LLM native constructs, this review provides a rigorous roadmap for architecting resilient scalable and self-evolving cognitive environments.Our analysis asserts that the next frontier of AGI development lies in the architectural efficiency of system-level coordination.

  </details>



- **Test-Time Training with KV Binding Is Secretly Linear Attention**  
  Junchen Liu, Sven Elflein, Or Litany, Zan Gojcic, Ruilong Li  
  _2026-02-24_ · https://arxiv.org/abs/2602.21204v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Test-time training (TTT) with KV binding as sequence modeling layer is commonly interpreted as a form of online meta-learning that memorizes a key-value mapping at test time. However, our analysis reveals multiple phenomena that contradict this memorization-based interpretation. Motivated by these findings, we revisit the formulation of TTT and show that a broad class of TTT architectures can be expressed as a form of learned linear attention operator. Beyond explaining previously puzzling model behaviors, this perspective yields multiple practical benefits: it enables principled architectural simplifications, admits fully parallel formulations that preserve performance while improving efficiency, and provides a systematic reduction of diverse TTT variants to a standard linear attention form. Overall, our results reframe TTT not as test-time memorization, but as learned linear attention with enhanced representational capacity.

  </details>



- **HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG**  
  Yuqi Huang, Ning Liao, Kai Yang, Anning Hu, Shengchao Hu, Xiaoxing Wang, Junchi Yan  
  _2026-02-24_ · https://arxiv.org/abs/2602.20926v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) often struggle with inherent knowledge boundaries and hallucinations, limiting their reliability in knowledge-intensive tasks. While Retrieval-Augmented Generation (RAG) mitigates these issues, it frequently overlooks structural interdependencies essential for multi-hop reasoning. Graph-based RAG approaches attempt to bridge this gap, yet they typically face trade-offs between accuracy and efficiency due to challenges such as costly graph traversals and semantic noise in LLM-generated summaries. In this paper, we propose HyperNode Expansion and Logical Path-Guided Evidence Localization strategies for GraphRAG (HELP), a novel framework designed to balance accuracy with practical efficiency through two core strategies: 1) HyperNode Expansion, which iteratively chains knowledge triplets into coherent reasoning paths abstracted as HyperNodes to capture complex structural dependencies and ensure retrieval accuracy; and 2) Logical Path-Guided Evidence Localization, which leverages precomputed graph-text correlations to map these paths directly to the corpus for superior efficiency. HELP avoids expensive random walks and semantic distortion, preserving knowledge integrity while drastically reducing retrieval latency. Extensive experiments demonstrate that HELP achieves competitive performance across multiple simple and multi-hop QA benchmarks and up to a 28.8$\times$ speedup over leading Graph-based RAG baselines.

  </details>



- **On the Explainability of Vision-Language Models in Art History**  
  Stefanie Schneider  
  _2026-02-24_ · https://arxiv.org/abs/2602.20853v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) transfer visual and textual data into a shared embedding space. In so doing, they enable a wide range of multimodal tasks, while also raising critical questions about the nature of machine 'understanding.' In this paper, we examine how Explainable Artificial Intelligence (XAI) methods can render the visual reasoning of a VLM - namely, CLIP - legible in art-historical contexts. To this end, we evaluate seven methods, combining zero-shot localization experiments with human interpretability studies. Our results indicate that, while these methods capture some aspects of human interpretation, their effectiveness hinges on the conceptual stability and representational availability of the examined categories.

  </details>



- **Vision-Language Models for Ergonomic Assessment of Manual Lifting Tasks: Estimating Horizontal and Vertical Hand Distances from RGB Video**  
  Mohammad Sadra Rajabi, Aanuoluwapo Ojelade, Sunwook Kim, Maury A. Nussbaum  
  _2026-02-24_ · https://arxiv.org/abs/2602.20658v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Manual lifting tasks are a major contributor to work-related musculoskeletal disorders, and effective ergonomic risk assessment is essential for quantifying physical exposure and informing ergonomic interventions. The Revised NIOSH Lifting Equation (RNLE) is a widely used ergonomic risk assessment tool for lifting tasks that relies on six task variables, including horizontal (H) and vertical (V) hand distances; such distances are typically obtained through manual measurement or specialized sensing systems and are difficult to use in real-world environments. We evaluated the feasibility of using innovative vision-language models (VLMs) to non-invasively estimate H and V from RGB video streams. Two multi-stage VLM-based pipelines were developed: a text-guided detection-only pipeline and a detection-plus-segmentation pipeline. Both pipelines used text-guided localization of task-relevant regions of interest, visual feature extraction from those regions, and transformer-based temporal regression to estimate H and V at the start and end of a lift. For a range of lifting tasks, estimation performance was evaluated using leave-one-subject-out validation across the two pipelines and seven camera view conditions. Results varied significantly across pipelines and camera view conditions, with the segmentation-based, multi-view pipeline consistently yielding the smallest errors, achieving mean absolute errors of approximately 6-8 cm when estimating H and 5-8 cm when estimating V. Across pipelines and camera view configurations, pixel-level segmentation reduced estimation error by approximately 20-30% for H and 35-40% for V relative to the detection-only pipeline. These findings support the feasibility of VLM-based pipelines for video-based estimation of RNLE distance parameters.

  </details>


