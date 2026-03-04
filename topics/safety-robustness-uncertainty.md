# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-03-04 07:02 UTC_

Total papers shown: **11**


---

- **ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation**  
  Xialin He, Sirui Xu, Xinyao Li, Runpei Dong, Liuyu Bian, Yu-Xiong Wang, Liang-Yan Gui  
  _2026-03-03_ · https://arxiv.org/abs/2603.03279v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Achieving autonomous and versatile whole-body loco-manipulation remains a central barrier to making humanoids practically useful. Yet existing approaches are fundamentally constrained: retargeted data are often scarce or low-quality; methods struggle to scale to large skill repertoires; and, most importantly, they rely on tracking predefined motion references rather than generating behavior from perception and high-level task specifications. To address these limitations, we propose ULTRA, a unified framework with two key components. First, we introduce a physics-driven neural retargeting algorithm that translates large-scale motion capture to humanoid embodiments while preserving physical plausibility for contact-rich interactions. Second, we learn a unified multimodal controller that supports both dense references and sparse task specifications, under sensing ranging from accurate motion-capture state to noisy egocentric visual inputs. We distill a universal tracking policy into this controller, compress motor skills into a compact latent space, and apply reinforcement learning finetuning to expand coverage and improve robustness under out-of-distribution scenarios. This enables coordinated whole-body behavior from sparse intent without test-time reference motions. We evaluate ULTRA in simulation and on a real Unitree G1 humanoid. Results show that ULTRA generalizes to autonomous, goal-conditioned whole-body loco-manipulation from egocentric perception, consistently outperforming tracking-only baselines with limited skills.

  </details>



- **Tether: Autonomous Functional Play with Correspondence-Driven Trajectory Warping**  
  William Liang, Sam Wang, Hung-Ju Wang, Osbert Bastani, Yecheng Jason Ma, Dinesh Jayaraman  
  _2026-03-03_ · https://arxiv.org/abs/2603.03278v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The ability to conduct and learn from interaction and experience is a central challenge in robotics, offering a scalable alternative to labor-intensive human demonstrations. However, realizing such "play" requires (1) a policy robust to diverse, potentially out-of-distribution environment states, and (2) a procedure that continuously produces useful robot experience. To address these challenges, we introduce Tether, a method for autonomous functional play involving structured, task-directed interactions. First, we design a novel open-loop policy that warps actions from a small set of source demonstrations (<=10) by anchoring them to semantic keypoint correspondences in the target scene. We show that this design is extremely data-efficient and robust even under significant spatial and semantic variations. Second, we deploy this policy for autonomous functional play in the real world via a continuous cycle of task selection, execution, evaluation, and improvement, guided by the visual understanding capabilities of vision-language models. This procedure generates diverse, high-quality datasets with minimal human intervention. In a household-like multi-object setup, our method is the first to perform many hours of autonomous multi-task play in the real world starting from only a handful of demonstrations. This produces a stream of data that consistently improves the performance of closed-loop imitation policies over time, ultimately yielding over 1000 expert-level trajectories and training policies competitive with those learned from human-collected demonstrations.

  </details>



- **TrustMH-Bench: A Comprehensive Benchmark for Evaluating the Trustworthiness of Large Language Models in Mental Health**  
  Zixin Xiong, Ziteng Wang, Haotian Fan, Xinjie Zhang, Wenxuan Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.03047v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  While Large Language Models (LLMs) demonstrate significant potential in providing accessible mental health support, their practical deployment raises critical trustworthiness concerns due to the domains high-stakes and safety-sensitive nature. Existing evaluation paradigms for general-purpose LLMs fail to capture mental health-specific requirements, highlighting an urgent need to prioritize and enhance their trustworthiness. To address this, we propose TrustMH-Bench, a holistic framework designed to systematically quantify the trustworthiness of mental health LLMs. By establishing a deep mapping from domain-specific norms to quantitative evaluation metrics, TrustMH-Bench evaluates models across eight core pillars: Reliability, Crisis Identification and Escalation, Safety, Fairness, Privacy, Robustness, Anti-sycophancy, and Ethics. We conduct extensive experiments across six general-purpose LLMs and six specialized mental health models. Experimental results indicate that the evaluated models underperform across various trustworthiness dimensions in mental health scenarios, revealing significant deficiencies. Notably, even generally powerful models (e.g., GPT-5.1) fail to maintain consistently high performance across all dimensions. Consequently, systematically improving the trustworthiness of LLMs has become a critical task. Our data and code are released.

  </details>



- **Scores Know Bobs Voice: Speaker Impersonation Attack**  
  Chanwoo Hwang, Sunpill Kim, Yong Kiam Tan, Tianchi Liu, Seunghun Paik, Dongsoo Kim, Mondal Soumik, Khin Mi Mi Aung, Jae Hong Seo  
  _2026-03-03_ · https://arxiv.org/abs/2603.02781v1 · `cs.CR`  
  <details><summary>Abstract</summary>

  Advances in deep learning have enabled the widespread deployment of speaker recognition systems (SRSs), yet they remain vulnerable to score-based impersonation attacks. Existing attacks that operate directly on raw waveforms require a large number of queries due to the difficulty of optimizing in high-dimensional audio spaces. Latent-space optimization within generative models offers improved efficiency, but these latent spaces are shaped by data distribution matching and do not inherently capture speaker-discriminative geometry. As a result, optimization trajectories often fail to align with the adversarial direction needed to maximize victim scores. To address this limitation, we propose an inversion-based generative attack framework that explicitly aligns the latent space of the synthesis model with the discriminative feature space of SRSs. We first analyze the requirements of an inverse model for score-based attacks and introduce a feature-aligned inversion strategy that geometrically synchronizes latent representations with speaker embeddings. This alignment ensures that latent updates directly translate into score improvements. Moreover, it enables new attack paradigms, including subspace-projection-based attacks, which were previously infeasible due to the absence of a faithful feature-to-audio mapping. Experiments show that our method significantly improves query efficiency, achieving competitive attack success rates with on average 10x fewer queries than prior approaches. In particular, the enabled subspace-projection-based attack attains up to 91.65% success using only 50 queries. These findings establish feature-aligned inversion as a key tool for evaluating the robustness of modern SRSs against score-based impersonation threats.

  </details>



- **TinyIceNet: Low-Power SAR Sea Ice Segmentation for On-Board FPGA Inference**  
  Mhd Rashed Al Koutayni, Mohamed Selim, Gerd Reis, Alain Pagani, Didier Stricker  
  _2026-03-03_ · https://arxiv.org/abs/2603.03075v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate sea ice mapping is essential for safe maritime navigation in polar regions, where rapidly changing ice conditions require timely and reliable information. While Sentinel-1 Synthetic Aperture Radar (SAR) provides high-resolution, all-weather observations of sea ice, conventional ground-based processing is limited by downlink bandwidth, latency, and energy costs associated with transmitting large volumes of raw data. On-board processing, enabled by dedicated inference chips integrated directly within the satellite payload, offers a transformative alternative by generating actionable sea ice products in orbit. In this context, we present TinyIceNet, a compact semantic segmentation network co-designed for on-board Stage of Development (SOD) mapping from dual-polarized Sentinel-1 SAR imagery under strict hardware and power constraints. Trained on the AI4Arctic dataset, TinyIceNet combines SAR-aware architectural simplifications with low-precision quantization to balance accuracy and efficiency. The model is synthesized using High-Level Synthesis and deployed on a Xilinx Zynq UltraScale+ FPGA platform, demonstrating near-real-time inference with significantly reduced energy consumption. Experimental results show that TinyIceNet achieves 75.216% F1 score on SOD segmentation while reducing energy consumption by 2x compared to full-precision GPU baselines, underscoring the potential of chip-level hardware-algorithm co-design for future spaceborne and edge AI systems.

  </details>



- **Generative adversarial imitation learning for robot swarms: Learning from human demonstrations and trained policies**  
  Mattes Kraus, Jonas Kuckling  
  _2026-03-03_ · https://arxiv.org/abs/2603.02783v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In imitation learning, robots are supposed to learn from demonstrations of the desired behavior. Most of the work in imitation learning for swarm robotics provides the demonstrations as rollouts of an existing policy. In this work, we provide a framework based on generative adversarial imitation learning that aims to learn collective behaviors from human demonstrations. Our framework is evaluated across six different missions, learning both from manual demonstrations and demonstrations derived from a PPO-trained policy. Results show that the imitation learning process is able to learn qualitatively meaningful behaviors that perform similarly well as the provided demonstrations. Additionally, we deploy the learned policies on a swarm of TurtleBot 4 robots in real-robot experiments. The exhibited behaviors preserved their visually recognizable character and their performance is comparable to the one achieved in simulation.

  </details>



- **Agentic Self-Evolutionary Replanning for Embodied Navigation**  
  Guoliang Li, Ruihua Han, Chengyang Li, He Li, Shuai Wang, Wenchao Ding, Hong Zhang, Chengzhong Xu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02772v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Failure is inevitable for embodied navigation in complex environments. To enhance the resilience, replanning (RP) is a viable option, where the robot is allowed to fail, but is capable of adjusting plan until success. However, existing RP approaches freeze the ego action model and miss the opportunities to explore better plans by upgrading the robot itself. To address this limitation, we propose Self-Evolutionary RePlanning, or SERP for short, which leads to a paradigm shift from frozen models towards evolving models by run-time learning from recent experiences. In contrast to existing model evolution approaches that often get stuck at predefined static parameters, we introduce agentic self-evolving action model that uses in-context learning with auto-differentiation (ILAD) for adaptive function adjustment and global parameter reset. To achieve token-efficient replanning for SERP, we also propose graph chain-of-thought (GCOT) replanning with large language model (LLM) inference over distilled graphs. Extensive simulation and real-world experiments demonstrate that SERP achieves higher success rate with lower token expenditure over various benchmarks, validating its superior robustness and efficiency across diverse environments.

  </details>



- **CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance**  
  Hanyang Wang, Yiyang Liu, Jiawei Chi, Fangfu Liu, Ran Xue, Yueqi Duan  
  _2026-03-03_ · https://arxiv.org/abs/2603.03281v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Classifier-Free Guidance (CFG) has emerged as a central approach for enhancing semantic alignment in flow-based diffusion models. In this paper, we explore a unified framework called CFG-Ctrl, which reinterprets CFG as a control applied to the first-order continuous-time generative flow, using the conditional-unconditional discrepancy as an error signal to adjust the velocity field. From this perspective, we summarize vanilla CFG as a proportional controller (P-control) with fixed gain, and typical follow-up variants develop extended control-law designs derived from it. However, existing methods mainly rely on linear control, inherently leading to instability, overshooting, and degraded semantic fidelity especially on large guidance scales. To address this, we introduce Sliding Mode Control CFG (SMC-CFG), which enforces the generative flow toward a rapidly convergent sliding manifold. Specifically, we define an exponential sliding mode surface over the semantic prediction error and introduce a switching control term to establish nonlinear feedback-guided correction. Moreover, we provide a Lyapunov stability analysis to theoretically support finite-time convergence. Experiments across text-to-image generation models including Stable Diffusion 3.5, Flux, and Qwen-Image demonstrate that SMC-CFG outperforms standard CFG in semantic alignment and enhances robustness across a wide range of guidance scales. Project Page: https://hanyang-21.github.io/CFG-Ctrl

  </details>



- **How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference**  
  Toru Lin, Shuying Deng, Zhao-Heng Yin, Pieter Abbeel, Jitendra Malik  
  _2026-03-03_ · https://arxiv.org/abs/2603.03280v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many essential manipulation tasks - such as food preparation, surgery, and craftsmanship - remain intractable for autonomous robots. These tasks are characterized not only by contact-rich, force-sensitive dynamics, but also by their "implicit" success criteria: unlike pick-and-place, task quality in these domains is continuous and subjective (e.g. how well a potato is peeled), making quantitative evaluation and reward engineering difficult. We present a learning framework for such tasks, using peeling with a knife as a representative example. Our approach follows a two-stage pipeline: first, we learn a robust initial policy via force-aware data collection and imitation learning, enabling generalization across object variations; second, we refine the policy through preference-based finetuning using a learned reward model that combines quantitative task metrics with qualitative human feedback, aligning policy behavior with human notions of task quality. Using only 50-200 peeling trajectories, our system achieves over 90% average success rates on challenging produce including cucumbers, apples, and potatoes, with performance improving by up to 40% through preference-based finetuning. Remarkably, policies trained on a single produce category exhibit strong zero-shot generalization to unseen in-category instances and to out-of-distribution produce from different categories while maintaining over 90% success rates.

  </details>



- **The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes**  
  Reuben Docea, Rayan Younis, Yonghao Long, Maxime Fleury, Jinjing Xu, Chenyang Li, André Schulze, Ann Wierick, Johannes Bender, Micha Pfeiffer, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02985v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The D4D Dataset provides paired endoscopic video and high-quality structured-light geometry for evaluating 3D reconstruction of deforming abdominal soft tissue in realistic surgical conditions. Data were acquired from six porcine cadaver sessions using a da Vinci Xi stereo endoscope and a Zivid structured-light camera, registered via optical tracking and manually curated iterative alignment methods. Three sequence types - whole deformations, incremental deformations, and moved-camera clips - probe algorithm robustness to non-rigid motion, deformation magnitude, and out-of-view updates. Each clip provides rectified stereo images, per-frame instrument masks, stereo depth, start/end structured-light point clouds, curated camera poses and camera intrinsics. In postprocessing, ICP and semi-automatic registration techniques are used to register data, and instrument masks are created. The dataset enables quantitative geometric evaluation in both visible and occluded regions, alongside photometric view-synthesis baselines. Comprising over 300,000 frames and 369 point clouds across 98 curated recordings, this resource can serve as a comprehensive benchmark for developing and evaluating non-rigid SLAM, 4D reconstruction, and depth estimation methods.

  </details>



- **Emerging trends in Cislunar Space for Lunar Science Exploration and Space Robotics aiding Human Spaceflight Safety**  
  Arsalan Muhammad, Yue Wang, Hai Huang, Hao Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02878v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In recent years, the Moon has emerged as an unparalleled extraterrestrial testbed for advancing cuttingedge technological and scientific research critical to enabling sustained human presence on its surface and supporting future interplanetary exploration. This study identifies and investigates two pivotal research domains with substantial transformative potential for accelerating humanity interplanetary aspirations. First is Lunar Science Exploration with Artificial Intelligence and Space Robotics which focusses on AI and Space Robotics redefining the frontiers of space exploration. Second being Space Robotics aid in manned spaceflight to the Moon serving as critical assets for pre-deployment infrastructure development, In-Situ Resource Utilization, surface operations support, and astronaut safety assurance. By integrating autonomy, machine learning, and realtime sensor fusion, space robotics not only augment human capabilities but also serve as force multipliers in achieving sustainable lunar exploration, paving the way for future crewed missions to Mars and beyond.

  </details>


