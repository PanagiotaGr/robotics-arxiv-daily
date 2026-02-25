# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-25 07:14 UTC_

Total papers shown: **7**


---

- **VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving**  
  Jie Wang, Guang Li, Zhijian Huang, Chenxu Dang, Hangjun Ye, Yahong Han, Long Chen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20794v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The significance of cross-view 3D geometric modeling capabilities for autonomous driving is self-evident, yet existing Vision-Language Models (VLMs) inherently lack this capability, resulting in their mediocre performance. While some promising approaches attempt to mitigate this by constructing Q&A data for auxiliary training, they still fail to fundamentally equip VLMs with the ability to comprehensively handle diverse evaluation protocols. We thus chart a new course, advocating for the infusion of VLMs with the cross-view geometric grounding of mature 3D foundation models, closing this critical capability gap in autonomous driving. In this spirit, we propose a novel architecture, VGGDrive, which empowers Vision-language models with cross-view Geometric Grounding for autonomous Driving. Concretely, to bridge the cross-view 3D geometric features from the frozen visual 3D model with the VLM's 2D visual features, we introduce a plug-and-play Cross-View 3D Geometric Enabler (CVGE). The CVGE decouples the base VLM architecture and effectively empowers the VLM with 3D features through a hierarchical adaptive injection mechanism. Extensive experiments show that VGGDrive enhances base VLM performance across five autonomous driving benchmarks, including tasks like cross-view risk perception, motion prediction, and trajectory planning. It's our belief that mature 3D foundation models can empower autonomous driving tasks through effective integration, and we hope our initial exploration demonstrates the potential of this paradigm to the autonomous driving community.

  </details>



- **Efficient Hierarchical Any-Angle Path Planning on Multi-Resolution 3D Grids**  
  Victor Reijgwart, Cesar Cadena, Roland Siegwart, Lionel Ott  
  _2026-02-24_ · https://arxiv.org/abs/2602.21174v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Hierarchical, multi-resolution volumetric mapping approaches are widely used to represent large and complex environments as they can efficiently capture their occupancy and connectivity information. Yet widely used path planning methods such as sampling and trajectory optimization do not exploit this explicit connectivity information, and search-based methods such as A* suffer from scalability issues in large-scale high-resolution maps. In many applications, Euclidean shortest paths form the underpinning of the navigation system. For such applications, any-angle planning methods, which find optimal paths by connecting corners of obstacles with straight-line segments, provide a simple and efficient solution. In this paper, we present a method that has the optimality and completeness properties of any-angle planners while overcoming computational tractability issues common to search-based methods by exploiting multi-resolution representations. Extensive experiments on real and synthetic environments demonstrate the proposed approach's solution quality and speed, outperforming even sampling-based methods. The framework is open-sourced to allow the robotics and planning community to build on our research.

  </details>



- **Robot Local Planner: A Periodic Sampling-Based Motion Planner with Minimal Waypoints for Home Environments**  
  Keisuke Takeshita, Takahiro Yamazaki, Tomohiro Ono, Takashi Yamamoto  
  _2026-02-24_ · https://arxiv.org/abs/2602.20645v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The objective of this study is to enable fast and safe manipulation tasks in home environments. Specifically, we aim to develop a system that can recognize its surroundings and identify target objects while in motion, enabling it to plan and execute actions accordingly. We propose a periodic sampling-based whole-body trajectory planning method, called the "Robot Local Planner (RLP)." This method leverages unique features of home environments to enhance computational efficiency, motion optimality, and robustness against recognition and control errors, all while ensuring safety. The RLP minimizes computation time by planning with minimal waypoints and generating safe trajectories. Furthermore, overall motion optimality is improved by periodically executing trajectory planning to select more optimal motions. This approach incorporates inverse kinematics that are robust to base position errors, further enhancing robustness. Evaluation experiments demonstrated that the RLP outperformed existing methods in terms of motion planning time, motion duration, and robustness, confirming its effectiveness in home environments. Moreover, application experiments using a tidy-up task achieved high success rates and short operation times, thereby underscoring its practical feasibility.

  </details>



- **NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning**  
  Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan  
  _2026-02-24_ · https://arxiv.org/abs/2602.21172v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models are advancing autonomous driving by replacing modular pipelines with unified end-to-end architectures. However, current VLAs face two expensive requirements: (1) massive dataset collection, and (2) dense reasoning annotations. In this work, we address both challenges with \modelname (\textbf{No} \textbf{R}easoning for \textbf{D}riving). Compared to existing VLAs, \modelname achieves competitive performance while being fine-tuned on $<$60\% of the data and no reasoning annotations, resulting in 3$\times$ fewer tokens. We identify that standard Group Relative Policy Optimization (GRPO) fails to yield significant improvements when applied to policies trained on such small, reasoning-free datasets. We show that this limitation stems from difficulty bias, which disproportionately penalizes reward signals from scenarios that produce high-variance rollouts within GRPO. \modelname overcomes this by incorporating Dr.~GRPO, a recent algorithm designed to mitigate difficulty bias in LLMs. As a result, \modelname achieves competitive performance on Waymo and NAVSIM with a fraction of the training data and no reasoning overhead, enabling more efficient autonomous systems.

  </details>



- **UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling**  
  Kaiyuan Tan, Yingying Shen, Mingfei Tu, Haohui Zhu, Bing Wang, Guang Chen, Hangjun Ye, Haiyang Sun  
  _2026-02-24_ · https://arxiv.org/abs/2602.20943v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Dynamic driving scene reconstruction is critical for autonomous driving simulation and closed-loop learning. While recent feed-forward methods have shown promise for 3D reconstruction, they struggle with long-range driving sequences due to quadratic complexity in sequence length and challenges in modeling dynamic objects over extended durations. We propose UFO, a novel recurrent paradigm that combines the benefits of optimization-based and feed-forward methods for efficient long-range 4D reconstruction. Our approach maintains a 4D scene representation that is iteratively refined as new observations arrive, using a visibility-based filtering mechanism to select informative scene tokens and enable efficient processing of long sequences. For dynamic objects, we introduce an object pose-guided modeling approach that supports accurate long-range motion capture. Experiments on the Waymo Open Dataset demonstrate that our method significantly outperforms both per-scene optimization and existing feed-forward methods across various sequence lengths. Notably, our approach can reconstruct 16-second driving logs within 0.5 second while maintaining superior visual quality and geometric accuracy.

  </details>



- **A Robotic Testing Platform for Pipelined Discovery of Resilient Soft Actuators**  
  Ang, Li, Alexander Yin, Alexander White, Sahib Sandhu, Matthew Francoeur, Victor Jimenez-Santiago, Van Remenar, Codrin Tugui, Mihai Duduta  
  _2026-02-24_ · https://arxiv.org/abs/2602.20963v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Short lifetime under high electrical fields hinders the widespread robotic application of linear dielectric elastomer actuators (DEAs). Systematic scanning is difficult due to time-consuming per-sample testing and the high-dimensional parameter space affecting performance. To address this, we propose an optimization pipeline enabled by a novel testing robot capable of scanning DEA lifetime. The robot integrates electro-mechanical property measurement, programmable voltage input, and multi-channel testing capacity. Using it, we scanned the lifetime of Elastosil-based linear actuators across parameters including input voltage magnitude, frequency, electrode material concentration, and electrical connection filler. The optimal parameter combinations improved operational lifetime under boundary operating conditions by up to 100% and were subsequently scaled up to achieve higher force and displacement output. The final product demonstrated resilience on a modular, scalable quadruped walking robot with payload carrying capacity (>100% of its untethered body weight, and >700% of combined actuator weight). This work is the first to introduce a self-driving lab approach into robotic actuator design.

  </details>



- **GA-Drive: Geometry-Appearance Decoupled Modeling for Free-viewpoint Driving Scene Generatio**  
  Hao Zhang, Lue Fan, Qitai Wang, Wenbo Li, Zehuan Wu, Lewei Lu, Zhaoxiang Zhang, Hongsheng Li  
  _2026-02-24_ · https://arxiv.org/abs/2602.20673v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  A free-viewpoint, editable, and high-fidelity driving simulator is crucial for training and evaluating end-to-end autonomous driving systems. In this paper, we present GA-Drive, a novel simulation framework capable of generating camera views along user-specified novel trajectories through Geometry-Appearance Decoupling and Diffusion-Based Generation. Given a set of images captured along a recorded trajectory and the corresponding scene geometry, GA-Drive synthesizes novel pseudo-views using geometry information. These pseudo-views are then transformed into photorealistic views using a trained video diffusion model. In this way, we decouple the geometry and appearance of scenes. An advantage of such decoupling is its support for appearance editing via state-of-the-art video-to-video editing techniques, while preserving the underlying geometry, enabling consistent edits across both original and novel trajectories. Extensive experiments demonstrate that GA-Drive substantially outperforms existing methods in terms of NTA-IoU, NTL-IoU, and FID scores.

  </details>


