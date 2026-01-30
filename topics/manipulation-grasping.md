# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-01-30 07:03 UTC_

Total papers shown: **6**


---

- **MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts**  
  Lorenzo Mazza, Ariel Rodriguez, Rayan Younis, Martin Lelis, Ortrun Hellig, Chenpan Li, Sebastian Bodenstedt, Martin Wagner, Stefanie Speidel  
  _2026-01-29_ · https://arxiv.org/abs/2601.21971v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Imitation learning has achieved remarkable success in robotic manipulation, yet its application to surgical robotics remains challenging due to data scarcity, constrained workspaces, and the need for an exceptional level of safety and predictability. We present a supervised Mixture-of-Experts (MoE) architecture designed for phase-structured surgical manipulation tasks, which can be added on top of any autonomous policy. Unlike prior surgical robot learning approaches that rely on multi-camera setups or thousands of demonstrations, we show that a lightweight action decoder policy like Action Chunking Transformer (ACT) can learn complex, long-horizon manipulation from less than 150 demonstrations using solely stereo endoscopic images, when equipped with our architecture. We evaluate our approach on the collaborative surgical task of bowel grasping and retraction, where a robot assistant interprets visual cues from a human surgeon, executes targeted grasping on deformable tissue, and performs sustained retraction. We benchmark our method against state-of-the-art Vision-Language-Action (VLA) models and the standard ACT baseline. Our results show that generalist VLAs fail to acquire the task entirely, even under standard in-distribution conditions. Furthermore, while standard ACT achieves moderate success in-distribution, adopting a supervised MoE architecture significantly boosts its performance, yielding higher success rates in-distribution and demonstrating superior robustness in out-of-distribution scenarios, including novel grasp locations, reduced illumination, and partial occlusions. Notably, it generalizes to unseen testing viewpoints and also transfers zero-shot to ex vivo porcine tissue without additional training, offering a promising pathway toward in vivo deployment. To support this, we present qualitative preliminary results of policy roll-outs during in vivo porcine surgery.

  </details>



- **Multi-Modular MANTA-RAY: A Modular Soft Surface Platform for Distributed Multi-Object Manipulation**  
  Pratik Ingle, Jørn Lambertsen, Kasper Støy, Andres Faina  
  _2026-01-29_ · https://arxiv.org/abs/2601.21884v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Manipulation surfaces control objects by actively deforming their shape rather than directly grasping them. While dense actuator arrays can generate complex deformations, they also introduce high degrees of freedom (DOF), increasing system complexity and limiting scalability. The MANTA-RAY (Manipulation with Adaptive Non-rigid Textile Actuation with Reduced Actuation densitY) platform addresses these challenges by leveraging a soft, fabric-based surface with reduced actuator density to manipulate fragile and heterogeneous objects. Previous studies focused on single-module implementations supported by four actuators, whereas the feasibility and benefits of a scalable, multi-module configuration remain unexplored. In this work, we present a distributed, modular, and scalable variant of the MANTA-RAY platform that maintains manipulation performance with a reduced actuator density. The proposed multi-module MANTA-RAY platform and control strategy employs object passing between modules and a geometric transformation driven PID controller that directly maps tilt-angle control outputs to actuator commands, eliminating the need for extensive data-driven or black-box training. We evaluate system performance in simulation across surface configurations of varying modules (3x3 and 4x4) and validate its feasibility through experiments on a physical 2x2 hardware prototype. The system successfully manipulates objects with diverse geometries, masses, and textures including fragile items such as eggs and apples as well as enabling parallel manipulation. The results demonstrate that the multi-module MANTA-RAY improves scalability and enables coordinated manipulation of multiple objects across larger areas, highlighting its potential for practical, real-world applications.

  </details>



- **DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation**  
  Haozhe Xie, Beichen Wen, Jiarui Zheng, Zhaoxi Chen, Fangzhou Hong, Haiwen Diao, Ziwei Liu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22153v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Manipulating dynamic objects remains an open challenge for Vision-Language-Action (VLA) models, which, despite strong generalization in static manipulation, struggle in dynamic scenarios requiring rapid perception, temporal anticipation, and continuous control. We present DynamicVLA, a framework for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptation through three key designs: 1) a compact 0.4B VLA using a convolutional vision encoder for spatially efficient, structurally faithful encoding, enabling fast multimodal inference; 2) Continuous Inference, enabling overlapping reasoning and execution for lower latency and timely adaptation to object motion; and 3) Latent-aware Action Streaming, which bridges the perception-execution gap by enforcing temporally aligned action execution. To fill the missing foundation of dynamic manipulation data, we introduce the Dynamic Object Manipulation (DOM) benchmark, built from scratch with an auto data collection pipeline that efficiently gathers 200K synthetic episodes across 2.8K scenes and 206 objects, and enables fast collection of 2K real-world episodes without teleoperation. Extensive evaluations demonstrate remarkable improvements in response speed, perception, and generalization, positioning DynamicVLA as a unified framework for general dynamic object manipulation across embodiments.

  </details>



- **Causal World Modeling for Robot Control**  
  Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al.  
  _2026-01-29_ · https://arxiv.org/abs/2601.21998v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This work highlights that video world modeling, alongside vision-language pre-training, establishes a fresh and independent foundation for robot learning. Intuitively, video world models provide the ability to imagine the near future by understanding the causality between actions and visual dynamics. Inspired by this, we introduce LingBot-VA, an autoregressive diffusion framework that learns frame prediction and policy execution simultaneously. Our model features three carefully crafted designs: (1) a shared latent space, integrating vision and action tokens, driven by a Mixture-of-Transformers (MoT) architecture, (2) a closed-loop rollout mechanism, allowing for ongoing acquisition of environmental feedback with ground-truth observations, (3) an asynchronous inference pipeline, parallelizing action prediction and motor execution to support efficient control. We evaluate our model on both simulation benchmarks and real-world scenarios, where it shows significant promise in long-horizon manipulation, data efficiency in post-training, and strong generalizability to novel configurations. The code and model are made publicly available to facilitate the community.

  </details>



- **mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning**  
  Kevin Zakka, Qiayuan Liao, Brent Yi, Louis Le Lay, Koushil Sreenath, Pieter Abbeel  
  _2026-01-29_ · https://arxiv.org/abs/2601.22074v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present mjlab, a lightweight, open-source framework for robot learning that combines GPU-accelerated simulation with composable environments and minimal setup friction. mjlab adopts the manager-based API introduced by Isaac Lab, where users compose modular building blocks for observations, rewards, and events, and pairs it with MuJoCo Warp for GPU-accelerated physics. The result is a framework installable with a single command, requiring minimal dependencies, and providing direct access to native MuJoCo data structures. mjlab ships with reference implementations of velocity tracking, motion imitation, and manipulation tasks.

  </details>



- **Information Filtering via Variational Regularization for Robot Manipulation**  
  Jinhao Zhang, Wenlong Xia, Yaojia Wang, Zhexuan Zhou, Huizhe Li, Yichen Lai, Haoming Song, Youmin Gong, Jie Me  
  _2026-01-29_ · https://arxiv.org/abs/2601.21926v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Diffusion-based visuomotor policies built on 3D visual representations have achieved strong performance in learning complex robotic skills. However, most existing methods employ an oversized denoising decoder. While increasing model capacity can improve denoising, empirical evidence suggests that it also introduces redundancy and noise in intermediate feature blocks. Crucially, we find that randomly masking backbone features at inference time (without changing training) can improve performance, confirming the presence of task-irrelevant noise in intermediate features. To this end, we propose Variational Regularization (VR), a lightweight module that imposes a timestep-conditioned Gaussian over backbone features and applies a KL-divergence regularizer, forming an adaptive information bottleneck. Extensive experiments on three simulation benchmarks (RoboTwin2.0, Adroit, and MetaWorld) show that, compared to the baseline DP3, our approach improves the success rate by 6.1% on RoboTwin2.0 and by 4.1% on Adroit and MetaWorld, achieving new state-of-the-art results. Real-world experiments further demonstrate that our method performs well in practical deployments. Code will released.

  </details>


