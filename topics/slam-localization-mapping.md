# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-03-04 07:02 UTC_

Total papers shown: **13**


---

- **Exploiting Double-Bounce Paths in Snapshot Radio SLAM: Bounds, Algorithms and Experiments**  
  Xi Zhang, Yu Ge, Ossi Kaltiokallio, Musa Furkan Keskin, Henk Wymeersch, Mikko Valkama  
  _2026-03-03_ · https://arxiv.org/abs/2603.02832v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Radio-based simultaneous localization and mapping (SLAM) has the potential to provide precise user equipment (UE) localization and environmental sensing capabilities by exploiting radio signals. Most existing approaches leverage line-of-sight (LoS) and single-bounce non-line-of-sight (NLoS) paths solely, while higher-order NLoS paths are treated as disturbance. In this paper, we investigate the benefits of leveraging double-bounce NLoS paths for solving the bistatic snapshot radio SLAM problem. We derive the Cramer-Rao bound (CRB) for joint estimation of the UE state and landmark positions when double-bounce NLoS paths are present. In addition, we propose an algorithm to identify double-bounce NLoS paths and leverage them into joint UE and landmarks estimation. The derived bounds are validated through simulated data, and the proposed algorithms are evaluated using experimental millimeter wave (mmWave) measurements harnessing beamformed 5G cellular reference signals. The numerical and experimental results demonstrate that the double-bounce NLoS paths which share at least one incidence point (IP) with the single-bounce NLoS paths improve the estimation accuracy of the UE state and existing IPs of single-bounce NLoS paths. Importantly, exploiting double-bounce NLoS paths enhances environmental mapping capabilities by revealing landmarks that are unobservable with single-bounce NLoS paths alone.

  </details>



- **MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN**  
  Ling Luo, Qianqian Bai  
  _2026-03-03_ · https://arxiv.org/abs/2603.03024v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language Navigation (VLN) aims to empower robots with the ability to perform long-horizon navigation in unfamiliar environments based on complex linguistic instructions. Its success critically hinges on establishing an efficient ``language-understanding -- visual-perception -- embodied-execution'' closed loop. Existing methods often suffer from perceptual distortion and decision drift in complex, long-distance tasks due to the cognitive overload of a single agent. Inspired by distributed cognition theory, this paper proposes MA-CoNav, a Multi-Agent Collaborative Navigation framework. This framework adopts a ``Master-Slave'' hierarchical agent collaboration architecture, decoupling and distributing the perception, planning, execution, and memory functions required for navigation tasks to specialized agents. Specifically, the Master Agent is responsible for global orchestration, while the Subordinate Agent group collaborates through a clear division of labor: an Observation Agent generates environment descriptions, a Planning Agent performs task decomposition and dynamic verification, an Execution Agent handles simultaneous mapping and action, and a Memory Agent manages structured experiences. Furthermore, the framework introduces a ``Local-Global'' dual-stage reflection mechanism to dynamically optimize the entire navigation pipeline. Empirical experiments were conducted using a real-world indoor dataset collected by a Limo Pro robot, with no scene-specific fine-tuning performed on the models throughout the process. The results demonstrate that MA-CoNav comprehensively outperforms existing mainstream VLN methods across multiple metrics.

  </details>



- **RL-Based Coverage Path Planning for Deformable Objects on 3D Surfaces**  
  Yuhang Zhang, Jinming Ma, Feng Wu  
  _2026-03-03_ · https://arxiv.org/abs/2603.03137v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Currently, manipulation tasks for deformable objects often focus on activities like folding clothes, handling ropes, and manipulating bags. However, research on contact-rich tasks involving deformable objects remains relatively underdeveloped. When humans use cloth or sponges to wipe surfaces, they rely on both vision and tactile feedback. Yet, current algorithms still face challenges with issues like occlusion, while research on tactile perception for manipulation is still evolving. Tasks such as covering surfaces with deformable objects demand not only perception but also precise robotic manipulation. To address this, we propose a method that leverages efficient and accessible simulators for task execution. Specifically, we train a reinforcement learning agent in a simulator to manipulate deformable objects for surface wiping tasks. We simplify the state representation of object surfaces using harmonic UV mapping, process contact feedback from the simulator on 2D feature maps, and use scaled grouped convolutions (SGCNN) to extract features efficiently. The agent then outputs actions in a reduced-dimensional action space to generate coverage paths. Experiments demonstrate that our method outperforms previous approaches in key metrics, including total path length and coverage area. We deploy these paths on a Kinova Gen3 manipulator to perform wiping experiments on the back of a torso model, validating the feasibility of our approach.

  </details>



- **DreamFlow: Local Navigation Beyond Observation via Conditional Flow Matching in the Latent Space**  
  Jiwon Park, Dongkyu Lee, I Made Aswin Nahrendra, Jaeyoung Lim, Hyun Myung  
  _2026-03-03_ · https://arxiv.org/abs/2603.02976v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Local navigation in cluttered environments often suffers from dense obstacles and frequent local minima. Conventional local planners rely on heuristics and are prone to failure, while deep reinforcement learning(DRL)based approaches provide adaptability but are constrained by limited onboard sensing. These limitations lead to navigation failures because the robot cannot perceive structures outside its field of view. In this paper, we propose DreamFlow, a DRL-based local navigation framework that extends the robot's perceptual horizon through conditional flow matching(CFM). The proposed CFM based prediction module learns probabilistic mapping between local height map latent representation and broader spatial representation conditioned on navigation context. This enables the navigation policy to predict unobserved environmental features and proactively avoid potential local minima. Experimental results demonstrate that DreamFlow outperforms existing methods in terms of latent prediction accuracy and navigation performance in simulation. The proposed method was further validated in cluttered real world environments with a quadrupedal robot. The project page is available at https://dreamflow-icra.github.io.

  </details>



- **TinyIceNet: Low-Power SAR Sea Ice Segmentation for On-Board FPGA Inference**  
  Mhd Rashed Al Koutayni, Mohamed Selim, Gerd Reis, Alain Pagani, Didier Stricker  
  _2026-03-03_ · https://arxiv.org/abs/2603.03075v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate sea ice mapping is essential for safe maritime navigation in polar regions, where rapidly changing ice conditions require timely and reliable information. While Sentinel-1 Synthetic Aperture Radar (SAR) provides high-resolution, all-weather observations of sea ice, conventional ground-based processing is limited by downlink bandwidth, latency, and energy costs associated with transmitting large volumes of raw data. On-board processing, enabled by dedicated inference chips integrated directly within the satellite payload, offers a transformative alternative by generating actionable sea ice products in orbit. In this context, we present TinyIceNet, a compact semantic segmentation network co-designed for on-board Stage of Development (SOD) mapping from dual-polarized Sentinel-1 SAR imagery under strict hardware and power constraints. Trained on the AI4Arctic dataset, TinyIceNet combines SAR-aware architectural simplifications with low-precision quantization to balance accuracy and efficiency. The model is synthesized using High-Level Synthesis and deployed on a Xilinx Zynq UltraScale+ FPGA platform, demonstrating near-real-time inference with significantly reduced energy consumption. Experimental results show that TinyIceNet achieves 75.216% F1 score on SOD segmentation while reducing energy consumption by 2x compared to full-precision GPU baselines, underscoring the potential of chip-level hardware-algorithm co-design for future spaceborne and edge AI systems.

  </details>



- **Self-supervised Domain Adaptation for Visual 3D Pose Estimation of Nano-drone Racing Gates by Enforcing Geometric Consistency**  
  Nicholas Carlotti, Michele Antonazzi, Elia Cereda, Mirko Nava, Nicola Basilico, Daniele Palossi, Alessandro Giusti  
  _2026-03-03_ · https://arxiv.org/abs/2603.02936v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We consider the task of visually estimating the relative pose of a drone racing gate in front of a nano-quadrotor, using a convolutional neural network pre-trained on simulated data to regress the gate's pose. Due to the sim-to-real gap, the pre-trained model underperforms in the real world and must be adapted to the target domain. We propose an unsupervised domain adaptation (UDA) approach using only real image sequences collected by the drone flying an arbitrary trajectory in front of a gate; sequences are annotated in a self-supervised fashion with the drone's odometry as measured by its onboard sensors. On this dataset, a state consistency loss enforces that two images acquired at different times yield pose predictions that are consistent with the drone's odometry. Results indicate that our approach outperforms other SoA UDA approaches, has a low mean absolute error in position (x=26, y=28, z=10 cm) and orientation ($ψ$=13${^{\circ}}$), an improvement of 40% in position and 37% in orientation over a baseline. The approach's effectiveness is appreciable with as few as 10 minutes of real-world flight data and yields models with an inference time of 30.4ms (33 fps) when deployed aboard the Crazyflie 2.1 Brushless nano-drone.

  </details>



- **Agentic Self-Evolutionary Replanning for Embodied Navigation**  
  Guoliang Li, Ruihua Han, Chengyang Li, He Li, Shuai Wang, Wenchao Ding, Hong Zhang, Chengzhong Xu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02772v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Failure is inevitable for embodied navigation in complex environments. To enhance the resilience, replanning (RP) is a viable option, where the robot is allowed to fail, but is capable of adjusting plan until success. However, existing RP approaches freeze the ego action model and miss the opportunities to explore better plans by upgrading the robot itself. To address this limitation, we propose Self-Evolutionary RePlanning, or SERP for short, which leads to a paradigm shift from frozen models towards evolving models by run-time learning from recent experiences. In contrast to existing model evolution approaches that often get stuck at predefined static parameters, we introduce agentic self-evolving action model that uses in-context learning with auto-differentiation (ILAD) for adaptive function adjustment and global parameter reset. To achieve token-efficient replanning for SERP, we also propose graph chain-of-thought (GCOT) replanning with large language model (LLM) inference over distilled graphs. Extensive simulation and real-world experiments demonstrate that SERP achieves higher success rate with lower token expenditure over various benchmarks, validating its superior robustness and efficiency across diverse environments.

  </details>



- **Inverse Reconstruction of Shock Time Series from Shock Response Spectrum Curves using Machine Learning**  
  Adam Watts, Andrew Jeon, Destry Newton, Ryan Bowering  
  _2026-03-03_ · https://arxiv.org/abs/2603.03229v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The shock response spectrum (SRS) is widely used to characterize the response of single-degree-of-freedom (SDOF) systems to transient accelerations. Because the mapping from acceleration time history to SRS is nonlinear and many-to-one, reconstructing time-domain signals from a target spectrum is inherently ill-posed. Conventional approaches address this problem through iterative optimization, typically representing signals as sums of exponentially decayed sinusoids, but these methods are computationally expensive and constrained by predefined basis functions. We propose a conditional variational autoencoder (CVAE) that learns a data-driven inverse mapping from SRS to acceleration time series. Once trained, the model generates signals consistent with prescribed target spectra without requiring iterative optimization. Experiments demonstrate improved spectral fidelity relative to classical techniques, strong generalization to unseen spectra, and inference speeds three to six orders of magnitude faster. These results establish deep generative modeling as a scalable and efficient approach for inverse SRS reconstruction.

  </details>



- **TrustMH-Bench: A Comprehensive Benchmark for Evaluating the Trustworthiness of Large Language Models in Mental Health**  
  Zixin Xiong, Ziteng Wang, Haotian Fan, Xinjie Zhang, Wenxuan Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.03047v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  While Large Language Models (LLMs) demonstrate significant potential in providing accessible mental health support, their practical deployment raises critical trustworthiness concerns due to the domains high-stakes and safety-sensitive nature. Existing evaluation paradigms for general-purpose LLMs fail to capture mental health-specific requirements, highlighting an urgent need to prioritize and enhance their trustworthiness. To address this, we propose TrustMH-Bench, a holistic framework designed to systematically quantify the trustworthiness of mental health LLMs. By establishing a deep mapping from domain-specific norms to quantitative evaluation metrics, TrustMH-Bench evaluates models across eight core pillars: Reliability, Crisis Identification and Escalation, Safety, Fairness, Privacy, Robustness, Anti-sycophancy, and Ethics. We conduct extensive experiments across six general-purpose LLMs and six specialized mental health models. Experimental results indicate that the evaluated models underperform across various trustworthiness dimensions in mental health scenarios, revealing significant deficiencies. Notably, even generally powerful models (e.g., GPT-5.1) fail to maintain consistently high performance across all dimensions. Consequently, systematically improving the trustworthiness of LLMs has become a critical task. Our data and code are released.

  </details>



- **The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes**  
  Reuben Docea, Rayan Younis, Yonghao Long, Maxime Fleury, Jinjing Xu, Chenyang Li, André Schulze, Ann Wierick, Johannes Bender, Micha Pfeiffer, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02985v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The D4D Dataset provides paired endoscopic video and high-quality structured-light geometry for evaluating 3D reconstruction of deforming abdominal soft tissue in realistic surgical conditions. Data were acquired from six porcine cadaver sessions using a da Vinci Xi stereo endoscope and a Zivid structured-light camera, registered via optical tracking and manually curated iterative alignment methods. Three sequence types - whole deformations, incremental deformations, and moved-camera clips - probe algorithm robustness to non-rigid motion, deformation magnitude, and out-of-view updates. Each clip provides rectified stereo images, per-frame instrument masks, stereo depth, start/end structured-light point clouds, curated camera poses and camera intrinsics. In postprocessing, ICP and semi-automatic registration techniques are used to register data, and instrument masks are created. The dataset enables quantitative geometric evaluation in both visible and occluded regions, alongside photometric view-synthesis baselines. Comprising over 300,000 frames and 369 point clouds across 98 curated recordings, this resource can serve as a comprehensive benchmark for developing and evaluating non-rigid SLAM, 4D reconstruction, and depth estimation methods.

  </details>



- **Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers**  
  Youngjun Jun, Seil Kang, Woojung Han, Seong Jae Hwang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02919v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Video Diffusion Transformers (DiTs) have been synthesizing high-quality video with high fidelity from given text descriptions involving motion. However, understanding how Video DiTs convert motion words into video remains insufficient. Furthermore, while prior studies on interpretable saliency maps primarily target objects, motion-related behavior in Video DiTs remains largely unexplored. In this paper, we investigate concrete motion features that specify when and which object moves for a given motion concept. First, to spatially localize, we introduce GramCol, which adaptively produces per-frame saliency maps for any text concept, including both motion and non-motion. Second, we propose a motion-feature selection algorithm to obtain an Interpretable Motion-Attentive Map (IMAP) that localizes motion spatially and temporally. Our method discovers concept saliency maps without the need for any gradient calculation or parameter update. Experimentally, our method shows outstanding localization capability on the motion localization task and zero-shot video semantic segmentation, providing interpretable and clearer saliency maps for both motion and non-motion concepts.

  </details>



- **3D-DRES: Detailed 3D Referring Expression Segmentation**  
  Qi Chen, Changli Wu, Jiayi Ji, Yiwei Ma, Liujuan Cao  
  _2026-03-03_ · https://arxiv.org/abs/2603.02896v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Current 3D visual grounding tasks only process sentence level detection or segmentation, which critically fails to leverage the rich compositional contextual reasonings within natural language expressions. To address this challenge, we introduce Detailed 3D Referring Expression Segmentation (3D-DRES), a new task that provides a phrase to 3D instance mapping, aiming at enhancing fine-grained 3D vision language understanding. To support 3D-DRES, we present DetailRefer, a new dataset comprising 54,432 descriptions spanning 11,054 distinct objects. Unlike previous datasets, DetailRefer implements a pioneering phrase-instance annotation paradigm where each referenced noun phrase is explicitly mapped to its corresponding 3D elements. Additionally, we introduce DetailBase, a purposefully streamlined yet effective baseline architecture that supports dual-mode segmentation at both sentence and phrase levels. Our experimental results demonstrate that models trained on DetailRefer not only excel at phrase-level segmentation but also show surprising improvements on traditional 3D-RES benchmarks.

  </details>



- **Scores Know Bobs Voice: Speaker Impersonation Attack**  
  Chanwoo Hwang, Sunpill Kim, Yong Kiam Tan, Tianchi Liu, Seunghun Paik, Dongsoo Kim, Mondal Soumik, Khin Mi Mi Aung, Jae Hong Seo  
  _2026-03-03_ · https://arxiv.org/abs/2603.02781v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Advances in deep learning have enabled the widespread deployment of speaker recognition systems (SRSs), yet they remain vulnerable to score-based impersonation attacks. Existing attacks that operate directly on raw waveforms require a large number of queries due to the difficulty of optimizing in high-dimensional audio spaces. Latent-space optimization within generative models offers improved efficiency, but these latent spaces are shaped by data distribution matching and do not inherently capture speaker-discriminative geometry. As a result, optimization trajectories often fail to align with the adversarial direction needed to maximize victim scores. To address this limitation, we propose an inversion-based generative attack framework that explicitly aligns the latent space of the synthesis model with the discriminative feature space of SRSs. We first analyze the requirements of an inverse model for score-based attacks and introduce a feature-aligned inversion strategy that geometrically synchronizes latent representations with speaker embeddings. This alignment ensures that latent updates directly translate into score improvements. Moreover, it enables new attack paradigms, including subspace-projection-based attacks, which were previously infeasible due to the absence of a faithful feature-to-audio mapping. Experiments show that our method significantly improves query efficiency, achieving competitive attack success rates with on average 10x fewer queries than prior approaches. In particular, the enabled subspace-projection-based attack attains up to 91.65% success using only 50 queries. These findings establish feature-aligned inversion as a key tool for evaluating the robustness of modern SRSs against score-based impersonation threats.

  </details>


