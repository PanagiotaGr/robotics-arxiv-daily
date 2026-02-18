# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-18 07:14 UTC_

Total papers shown: **4**


---

- **Dex4D: Task-Agnostic Point Track Policy for Sim-to-Real Dexterous Manipulation**  
  Yuxuan Kuang, Sungjae Park, Katerina Fragkiadaki, Shubham Tulsiani  
  _2026-02-17_ · https://arxiv.org/abs/2602.15828v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning generalist policies capable of accomplishing a plethora of everyday tasks remains an open challenge in dexterous manipulation. In particular, collecting large-scale manipulation data via real-world teleoperation is expensive and difficult to scale. While learning in simulation provides a feasible alternative, designing multiple task-specific environments and rewards for training is similarly challenging. We propose Dex4D, a framework that instead leverages simulation for learning task-agnostic dexterous skills that can be flexibly recomposed to perform diverse real-world manipulation tasks. Specifically, Dex4D learns a domain-agnostic 3D point track conditioned policy capable of manipulating any object to any desired pose. We train this 'Anypose-to-Anypose' policy in simulation across thousands of objects with diverse pose configurations, covering a broad space of robot-object interactions that can be composed at test time. At deployment, this policy can be zero-shot transferred to real-world tasks without finetuning, simply by prompting it with desired object-centric point tracks extracted from generated videos. During execution, Dex4D uses online point tracking for closed-loop perception and control. Extensive experiments in simulation and on real robots show that our method enables zero-shot deployment for diverse dexterous manipulation tasks and yields consistent improvements over prior baselines. Furthermore, we demonstrate strong generalization to novel objects, scene layouts, backgrounds, and trajectories, highlighting the robustness and scalability of the proposed framework.

  </details>



- **Selective Perception for Robot: Task-Aware Attention in Multimodal VLA**  
  Young-Chae Son, Jung-Woo Lee, Yoon-Ji Choi, Dae-Kwan Ko, Soo-Chul Lim  
  _2026-02-17_ · https://arxiv.org/abs/2602.15543v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In robotics, Vision-Language-Action (VLA) models that integrate diverse multimodal signals from multi-view inputs have emerged as an effective approach. However, most prior work adopts static fusion that processes all visual inputs uniformly, which incurs unnecessary computational overhead and allows task-irrelevant background information to act as noise. Inspired by the principles of human active perception, we propose a dynamic information fusion framework designed to maximize the efficiency and robustness of VLA models. Our approach introduces a lightweight adaptive routing architecture that analyzes the current text prompt and observations from a wrist-mounted camera in real-time to predict the task-relevance of multiple camera views. By conditionally attenuating computations for views with low informational utility and selectively providing only essential visual features to the policy network, Our framework achieves computation efficiency proportional to task relevance. Furthermore, to efficiently secure large-scale annotation data for router training, we established an automated labeling pipeline utilizing Vision-Language Models (VLMs) to minimize data collection and annotation costs. Experimental results in real-world robotic manipulation scenarios demonstrate that the proposed approach achieves significant improvements in both inference efficiency and control performance compared to existing VLA models, validating the effectiveness and practicality of dynamic information fusion in resource-constrained, real-time robot control environments.

  </details>



- **Constraining Streaming Flow Models for Adapting Learned Robot Trajectory Distributions**  
  Jieting Long, Dechuan Liu, Weidong Cai, Ian Manchester, Weiming Zhi  
  _2026-02-17_ · https://arxiv.org/abs/2602.15567v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robot motion distributions often exhibit multi-modality and require flexible generative models for accurate representation. Streaming Flow Policies (SFPs) have recently emerged as a powerful paradigm for generating robot trajectories by integrating learned velocity fields directly in action space, enabling smooth and reactive control. However, existing formulations lack mechanisms for adapting trajectories post-training to enforce safety and task-specific constraints. We propose Constraint-Aware Streaming Flow (CASF), a framework that augments streaming flow policies with constraint-dependent metrics that reshape the learned velocity field during execution. CASF models each constraint, defined in either the robot's workspace or configuration space, as a differentiable distance function that is converted into a local metric and pulled back into the robot's control space. Far from restricted regions, the resulting metric reduces to the identity; near constraint boundaries, it smoothly attenuates or redirects motion, effectively deforming the underlying flow to maintain safety. This allows trajectories to be adapted in real time, ensuring that robot actions respect joint limits, avoid collisions, and remain within feasible workspaces, while preserving the multi-modal and reactive properties of streaming flow policies. We demonstrate CASF in simulated and real-world manipulation tasks, showing that it produces constraint-satisfying trajectories that remain smooth, feasible, and dynamically consistent, outperforming standard post-hoc projection baselines.

  </details>



- **PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra**  
  Xiachong Feng, Liang Zhao, Weihong Zhong, Yichong Huang, Yuxuan Gu, Lingpeng Kong, Xiaocheng Feng, Bing Qin  
  _2026-02-17_ · https://arxiv.org/abs/2602.15669v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Current methods for personality control in Large Language Models rely on static prompting or expensive fine-tuning, failing to capture the dynamic and compositional nature of human traits. We introduce PERSONA, a training-free framework that achieves fine-tuning level performance through direct manipulation of personality vectors in activation space. Our key insight is that personality traits appear as extractable, approximately orthogonal directions in the model's representation space that support algebraic operations. The framework operates through three stages: Persona-Base extracts orthogonal trait vectors via contrastive activation analysis; Persona-Algebra enables precise control through vector arithmetic (scalar multiplication for intensity, addition for composition, subtraction for suppression); and Persona-Flow achieves context-aware adaptation by dynamically composing these vectors during inference. On PersonalityBench, our approach achieves a mean score of 9.60, nearly matching the supervised fine-tuning upper bound of 9.61 without any gradient updates. On our proposed Persona-Evolve benchmark for dynamic personality adaptation, we achieve up to 91% win rates across diverse model families. These results provide evidence that aspects of LLM personality are mathematically tractable, opening new directions for interpretable and efficient behavioral control.

  </details>


