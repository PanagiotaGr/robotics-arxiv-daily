# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-12 07:15 UTC_

Total papers shown: **10**


---

- **YOR: Your Own Mobile Manipulator for Generalizable Robotics**  
  Manan H Anjaria, Mehmet Enes Erciyes, Vedant Ghatnekar, Neha Navarkar, Haritheja Etukuru, Xiaole Jiang, Kanad Patel, Dhawal Kabra, Nicholas Wojno, Radhika Ajay Prayage, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.11150v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advances in robot learning have generated significant interest in capable platforms that may eventually approach human-level competence. This interest, combined with the commoditization of actuators, has propelled growth in low-cost robotic platforms. However, the optimal form factor for mobile manipulation, especially on a budget, remains an open question. We introduce YOR, an open-source, low-cost mobile manipulator that integrates an omnidirectional base, a telescopic vertical lift, and two arms with grippers to achieve whole-body mobility and manipulation. Our design emphasizes modularity, ease of assembly using off-the-shelf components, and affordability, with a bill-of-materials cost under 10,000 USD. We demonstrate YOR's capability by completing tasks that require coordinated whole-body control, bimanual manipulation, and autonomous navigation. Overall, YOR offers competitive functionality for mobile manipulation research at a fraction of the cost of existing platforms. Project website: https://www.yourownrobot.ai/

  </details>



- **Safe mobility support system using crowd mapping and avoidance route planning using VLM**  
  Sena Saito, Kenta Tabata, Renato Miyagusuku, Koichi Ozaki  
  _2026-02-11_ · https://arxiv.org/abs/2602.10910v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous mobile robots offer promising solutions for labor shortages and increased operational efficiency. However, navigating safely and effectively in dynamic environments, particularly crowded areas, remains challenging. This paper proposes a novel framework that integrates Vision-Language Models (VLM) and Gaussian Process Regression (GPR) to generate dynamic crowd-density maps (``Abstraction Maps'') for autonomous robot navigation. Our approach utilizes VLM's capability to recognize abstract environmental concepts, such as crowd densities, and represents them probabilistically via GPR. Experimental results from real-world trials on a university campus demonstrated that robots successfully generated routes avoiding both static obstacles and dynamic crowds, enhancing navigation safety and adaptability.

  </details>



- **From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?**  
  Krishna Kanth Nakka, Vedasri Nakka  
  _2026-02-11_ · https://arxiv.org/abs/2602.10771v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems.

  </details>



- **A Unified Experimental Architecture for Informative Path Planning: from Simulation to Deployment with GuadalPlanner**  
  Alejandro Mendoza Barrionuevo, Dame Seck Diop, Alejandro Casado Pérez, Daniel Gutiérrez Reina, Sergio L. Toral Marín, Samuel Yanes Luis  
  _2026-02-11_ · https://arxiv.org/abs/2602.10702v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The evaluation of informative path planning algorithms for autonomous vehicles is often hindered by fragmented execution pipelines and limited transferability between simulation and real-world deployment. This paper introduces a unified architecture that decouples high-level decision-making from vehicle-specific control, enabling algorithms to be evaluated consistently across different abstraction levels without modification. The proposed architecture is realized through GuadalPlanner, which defines standardized interfaces between planning, sensing, and vehicle execution. It is an open and extensible research tool that supports discrete graph-based environments and interchangeable planning strategies, and is built upon widely adopted robotics technologies, including ROS2, MAVLink, and MQTT. Its design allows the same algorithmic logic to be deployed in fully simulated environments, software-in-the-loop configurations, and physical autonomous vehicles using an identical execution pipeline. The approach is validated through a set of experiments, including real-world deployment on an autonomous surface vehicle performing water quality monitoring with real-time sensor feedback.

  </details>



- **Interactive LLM-assisted Curriculum Learning for Multi-Task Evolutionary Policy Search**  
  Berfin Sakallioglu, Giorgia Nadizar, Eric Medvet  
  _2026-02-11_ · https://arxiv.org/abs/2602.10891v1 · `cs.NE`  
  <details><summary>Abstract</summary>

  Multi-task policy search is a challenging problem because policies are required to generalize beyond training cases. Curriculum learning has proven to be effective in this setting, as it introduces complexity progressively. However, designing effective curricula is labor-intensive and requires extensive domain expertise. LLM-based curriculum generation has only recently emerged as a potential solution, but was limited to operate in static, offline modes without leveraging real-time feedback from the optimizer. Here we propose an interactive LLM-assisted framework for online curriculum generation, where the LLM adaptively designs training cases based on real-time feedback from the evolutionary optimization process. We investigate how different feedback modalities, ranging from numeric metrics alone to combinations with plots and behavior visualizations, influence the LLM ability to generate meaningful curricula. Through a 2D robot navigation case study, tackled with genetic programming as optimizer, we evaluate our approach against static LLM-generated curricula and expert-designed baselines. We show that interactive curriculum generation outperforms static approaches, with multimodal feedback incorporating both progression plots and behavior visualizations yielding performance competitive with expert-designed curricula. This work contributes to understanding how LLMs can serve as interactive curriculum designers for embodied AI systems, with potential extensions to broader evolutionary robotics applications.

  </details>



- **(MGS)$^2$-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization**  
  Minglei Li, Mengfan He, Chao Chen, Ziyang Meng  
  _2026-02-11_ · https://arxiv.org/abs/2602.10704v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Cross-view geo-localization (CVGL) is pivotal for GNSS-denied UAV navigation but remains brittle under the drastic geometric misalignment between oblique aerial views and orthographic satellite references. Existing methods predominantly operate within a 2D manifold, neglecting the underlying 3D geometry where view-dependent vertical facades (macro-structure) and scale variations (micro-scale) severely corrupt feature alignment. To bridge this gap, we propose (MGS)$^2$, a geometry-grounded framework. The core of our innovation is the Macro-Geometric Structure Filtering (MGSF) module. Unlike pixel-wise matching sensitive to noise, MGSF leverages dilated geometric gradients to physically filter out high-frequency facade artifacts while enhancing the view-invariant horizontal plane, directly addressing the domain shift. To guarantee robust input for this structural filtering, we explicitly incorporate a Micro-Geometric Scale Adaptation (MGSA) module. MGSA utilizes depth priors to dynamically rectify scale discrepancies via multi-branch feature fusion. Furthermore, a Geometric-Appearance Contrastive Distillation (GACD) loss is designed to strictly discriminate against oblique occlusions. Extensive experiments demonstrate that (MGS)$^2$ achieves state-of-the-art performance, recording a Recall@1 of 97.5\% on University-1652 and 97.02\% on SUES-200. Furthermore, the framework exhibits superior cross-dataset generalization against geometric ambiguity. The code is available at: \href{https://github.com/GabrielLi1473/MGS-Net}{https://github.com/GabrielLi1473/MGS-Net}.

  </details>



- **From Natural Language to Materials Discovery:The Materials Knowledge Navigation Agent**  
  Genmao Zhuang, Amir Barati Farimani  
  _2026-02-11_ · https://arxiv.org/abs/2602.11123v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Accelerating the discovery of high-performance materials remains a central challenge across energy, electronics, and aerospace technologies, where traditional workflows depend heavily on expert intuition and computationally expensive simulations. Here we introduce the Materials Knowledge Navigation Agent (MKNA), a language-driven system that translates natural-language scientific intent into executable actions for database retrieval, property prediction, structure generation, and stability evaluation. Beyond automating tool invocation, MKNA autonomously extracts quantitative thresholds and chemically meaningful design motifs from literature and database evidence, enabling data-grounded hypothesis formation. Applied to the search for high-Debye-temperature ceramics, the agent identifies a literature-supported screening criterion (Theta_D > 800 K), rediscovers canonical ultra-stiff materials such as diamond, SiC, SiN, and BeO, and proposes thermodynamically stable, previously unreported Be-C-rich compounds that populate the sparsely explored 1500-1700 K regime. These results demonstrate that MKNA not only finds stable candidates but also reconstructs interpretable design heuristics, establishing a generalizable platform for autonomous, language-guided materials exploration.

  </details>



- **DeepImageSearch: Benchmarking Multimodal Agents for Context-Aware Image Retrieval in Visual Histories**  
  Chenlong Deng, Mengjie Deng, Junjie Wu, Dun Zeng, Teng Wang, Qingsong Xie, Jiadeng Huang, Shengjie Ma, Changwang Zhang, Zhaoxiang Wang, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10809v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Existing multimodal retrieval systems excel at semantic matching but implicitly assume that query-image relevance can be measured in isolation. This paradigm overlooks the rich dependencies inherent in realistic visual streams, where information is distributed across temporal sequences rather than confined to single snapshots. To bridge this gap, we introduce DeepImageSearch, a novel agentic paradigm that reformulates image retrieval as an autonomous exploration task. Models must plan and perform multi-step reasoning over raw visual histories to locate targets based on implicit contextual cues. We construct DISBench, a challenging benchmark built on interconnected visual data. To address the scalability challenge of creating context-dependent queries, we propose a human-model collaborative pipeline that employs vision-language models to mine latent spatiotemporal associations, effectively offloading intensive context discovery before human verification. Furthermore, we build a robust baseline using a modular agent framework equipped with fine-grained tools and a dual-memory system for long-horizon navigation. Extensive experiments demonstrate that DISBench poses significant challenges to state-of-the-art models, highlighting the necessity of incorporating agentic reasoning into next-generation retrieval systems.

  </details>



- **ContactGaussian-WM: Learning Physics-Grounded World Model from Videos**  
  Meizhong Wang, Wanxin Jin, Kun Cao, Lihua Xie, Yiguang Hong  
  _2026-02-11_ · https://arxiv.org/abs/2602.11021v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Developing world models that understand complex physical interactions is essential for advancing robotic planning and simulation.However, existing methods often struggle to accurately model the environment under conditions of data scarcity and complex contact-rich dynamic motion.To address these challenges, we propose ContactGaussian-WM, a differentiable physics-grounded rigid-body world model capable of learning intricate physical laws directly from sparse and contact-rich video sequences.Our framework consists of two core components: (1) a unified Gaussian representation for both visual appearance and collision geometry, and (2) an end-to-end differentiable learning framework that differentiates through a closed-form physics engine to infer physical properties from sparse visual observations.Extensive simulations and real-world evaluations demonstrate that ContactGaussian-WM outperforms state-of-the-art methods in learning complex scenarios, exhibiting robust generalization capabilities.Furthermore, we showcase the practical utility of our framework in downstream applications, including data synthesis and real-time MPC.

  </details>



- **CMAD: Cooperative Multi-Agent Diffusion via Stochastic Optimal Control**  
  Riccardo Barbano, Alexander Denker, Zeljko Kereta, Runchang Li, Francisco Vargas  
  _2026-02-11_ · https://arxiv.org/abs/2602.10933v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Continuous-time generative models have achieved remarkable success in image restoration and synthesis. However, controlling the composition of multiple pre-trained models remains an open challenge. Current approaches largely treat composition as an algebraic composition of probability densities, such as via products or mixtures of experts. This perspective assumes the target distribution is known explicitly, which is almost never the case. In this work, we propose a different paradigm that formulates compositional generation as a cooperative Stochastic Optimal Control problem. Rather than combining probability densities, we treat pre-trained diffusion models as interacting agents whose diffusion trajectories are jointly steered, via optimal control, toward a shared objective defined on their aggregated output. We validate our framework on conditional MNIST generation and compare it against a naive inference-time DPS-style baseline replacing learned cooperative control with per-step gradient guidance.

  </details>


