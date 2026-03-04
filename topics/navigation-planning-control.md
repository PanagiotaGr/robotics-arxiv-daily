# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-03-04 07:02 UTC_

Total papers shown: **10**


---

- **From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?**  
  Shinas Shaji, Fabian Huppertz, Alex Mitrevski, Sebastian Houben  
  _2026-03-03_ · https://arxiv.org/abs/2603.03148v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In order to flexibly act in an everyday environment, a robotic agent needs a variety of cognitive capabilities that enable it to reason about plans and perform execution recovery. Large language models (LLMs) have been shown to demonstrate emergent cognitive aspects, such as reasoning and language understanding; however, the ability to control embodied robotic agents requires reliably bridging high-level language to low-level functionalities for perception and control. In this paper, we investigate the extent to which an LLM can serve as a core component for planning and execution reasoning in a cognitive robot architecture. For this purpose, we propose a cognitive architecture in which an agentic LLM serves as the core component for planning and reasoning, while components for working and episodic memories support learning from experience and adaptation. An instance of the architecture is then used to control a mobile manipulator in a simulated household environment, where environment interaction is done through a set of high-level tools for perception, reasoning, navigation, grasping, and placement, all of which are made available to the LLM-based agent. We evaluate our proposed system on two household tasks (object placement and object swapping), which evaluate the agent's reasoning, planning, and memory utilisation. The results demonstrate that the LLM-driven agent can complete structured tasks and exhibits emergent adaptation and memory-guided planning, but also reveal significant limitations, such as hallucinations about the task success and poor instruction following by refusing to acknowledge and complete sequential tasks. These findings highlight both the potential and challenges of employing LLMs as embodied cognitive controllers for autonomous robots.

  </details>



- **MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN**  
  Ling Luo, Qianqian Bai  
  _2026-03-03_ · https://arxiv.org/abs/2603.03024v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language Navigation (VLN) aims to empower robots with the ability to perform long-horizon navigation in unfamiliar environments based on complex linguistic instructions. Its success critically hinges on establishing an efficient ``language-understanding -- visual-perception -- embodied-execution'' closed loop. Existing methods often suffer from perceptual distortion and decision drift in complex, long-distance tasks due to the cognitive overload of a single agent. Inspired by distributed cognition theory, this paper proposes MA-CoNav, a Multi-Agent Collaborative Navigation framework. This framework adopts a ``Master-Slave'' hierarchical agent collaboration architecture, decoupling and distributing the perception, planning, execution, and memory functions required for navigation tasks to specialized agents. Specifically, the Master Agent is responsible for global orchestration, while the Subordinate Agent group collaborates through a clear division of labor: an Observation Agent generates environment descriptions, a Planning Agent performs task decomposition and dynamic verification, an Execution Agent handles simultaneous mapping and action, and a Memory Agent manages structured experiences. Furthermore, the framework introduces a ``Local-Global'' dual-stage reflection mechanism to dynamically optimize the entire navigation pipeline. Empirical experiments were conducted using a real-world indoor dataset collected by a Limo Pro robot, with no scene-specific fine-tuning performed on the models throughout the process. The results demonstrate that MA-CoNav comprehensively outperforms existing mainstream VLN methods across multiple metrics.

  </details>



- **Agentic Self-Evolutionary Replanning for Embodied Navigation**  
  Guoliang Li, Ruihua Han, Chengyang Li, He Li, Shuai Wang, Wenchao Ding, Hong Zhang, Chengzhong Xu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02772v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Failure is inevitable for embodied navigation in complex environments. To enhance the resilience, replanning (RP) is a viable option, where the robot is allowed to fail, but is capable of adjusting plan until success. However, existing RP approaches freeze the ego action model and miss the opportunities to explore better plans by upgrading the robot itself. To address this limitation, we propose Self-Evolutionary RePlanning, or SERP for short, which leads to a paradigm shift from frozen models towards evolving models by run-time learning from recent experiences. In contrast to existing model evolution approaches that often get stuck at predefined static parameters, we introduce agentic self-evolving action model that uses in-context learning with auto-differentiation (ILAD) for adaptive function adjustment and global parameter reset. To achieve token-efficient replanning for SERP, we also propose graph chain-of-thought (GCOT) replanning with large language model (LLM) inference over distilled graphs. Extensive simulation and real-world experiments demonstrate that SERP achieves higher success rate with lower token expenditure over various benchmarks, validating its superior robustness and efficiency across diverse environments.

  </details>



- **HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations**  
  Xiaomeng Xu, Jisang Park, Han Zhang, Eric Cousineau, Aditya Bhat, Jose Barreiros, Dian Wang, Shuran Song  
  _2026-03-03_ · https://arxiv.org/abs/2603.03243v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present Whole-Body Mobile Manipulation Interface (HoMMI), a data collection and policy learning framework that learns whole-body mobile manipulation directly from robot-free human demonstrations. We augment UMI interfaces with egocentric sensing to capture the global context required for mobile manipulation, enabling portable, robot-free, and scalable data collection. However, naively incorporating egocentric sensing introduces a larger human-to-robot embodiment gap in both observation and action spaces, making policy transfer difficult. We explicitly bridge this gap with a cross-embodiment hand-eye policy design, including an embodiment agnostic visual representation; a relaxed head action representation; and a whole-body controller that realizes hand-eye trajectories through coordinated whole-body motion under robot-specific physical constraints. Together, these enable long-horizon mobile manipulation tasks requiring bimanual and whole-body coordination, navigation, and active perception. Results are best viewed on: https://hommi-robot.github.io

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



- **CoFL: Continuous Flow Fields for Language-Conditioned Navigation**  
  Haokun Liu, Zhaoqi Ma, Yicheng Chen, Masaki Kitagawa, Wentao Zhang, Jinjie Li, Moju Zhao  
  _2026-03-03_ · https://arxiv.org/abs/2603.02854v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Language-conditioned navigation pipelines often rely on brittle modular components or costly action-sequence generation. To address these limitations, we present CoFL, an end-to-end policy that directly maps a bird's-eye view (BEV) observation and a language instruction to a continuous flow field for navigation. Instead of predicting discrete action tokens or sampling action chunks via iterative denoising, CoFL outputs instantaneous velocities that can be queried at arbitrary 2D projected locations. Trajectories are obtained by numerical integration of the predicted field, producing smooth motion that remains reactive under closed-loop execution. To enable large-scale training, we build a dataset of over 500k BEV image-instruction pairs, each procedurally annotated with a flow field and a trajectory derived from BEV semantic maps built on Matterport3D and ScanNet. By training on a mixed distribution, CoFL significantly outperforms modular Vision-Language Model (VLM)-based planners and generative policy baselines on strictly unseen scenes. Finally, we deploy CoFL zero-shot in real-world experiments with overhead BEV observations across multiple layouts, maintaining reliable closed-loop control and a high success rate.

  </details>



- **SPARC: Spatial-Aware Path Planning via Attentive Robot Communication**  
  Sayang Mu, Xiangyu Wu, Bo An  
  _2026-03-03_ · https://arxiv.org/abs/2603.02845v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Efficient communication is critical for decentralized Multi-Robot Path Planning (MRPP), yet existing learned communication methods treat all neighboring robots equally regardless of their spatial proximity, leading to diluted attention in congested regions where coordination matters most. We propose Relation enhanced Multi Head Attention (RMHA), a communication mechanism that explicitly embeds pairwise Manhattan distances into the attention weight computation, enabling each robot to dynamically prioritize messages from spatially relevant neighbors. Combined with a distance-constrained attention mask and GRU gated message fusion, RMHA integrates seamlessly with MAPPO for stable end-to-end training. In zero-shot generalization from 8 training robots to 128 test robots on 40x40 grids, RMHA achieves approximately 75 percent success rate at 30 percent obstacle density outperforming the best baseline by over 25 percentage points. Ablation studies confirm that distance-relation encoding is the key contributor to success rate improvement in high-density environments. Index Terms-Multi-robot path planning, graph attention mechanism, multi-head attention, communication optimization, cooperative decision-making

  </details>



- **TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language Navigation**  
  Jiaxing Liu, Zexi Zhang, Xiaoyan Li, Boyue Wang, Yongli Hu, Baocai Yin  
  _2026-03-03_ · https://arxiv.org/abs/2603.02972v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language Navigation (VLN) presents a unique challenge for Large Vision-Language Models (VLMs) due to their inherent architectural mismatch: VLMs are primarily pretrained on static, disembodied vision-language tasks, which fundamentally clash with the dynamic, embodied, and spatially-structured nature of navigation. Existing large-model-based methods often resort to converting rich visual and spatial information into text, forcing models to implicitly infer complex visual-topological relationships or limiting their global action capabilities. To bridge this gap, we propose TagaVLM (Topology-Aware Global Action reasoning), an end-to-end framework that explicitly injects topological structures into the VLM backbone. To introduce topological edge information, Spatial Topology Aware Residual Attention (STAR-Att) directly integrates it into the VLM's self-attention mechanism, enabling intrinsic spatial reasoning while preserving pretrained knowledge. To enhance topological node information, an Interleaved Navigation Prompt strengthens node-level visual-text alignment. Finally, with the embedded topological graph, the model is capable of global action reasoning, allowing for robust path correction. On the R2R benchmark, TagaVLM achieves state-of-the-art performance among large-model-based methods, with a Success Rate (SR) of 51.09% and SPL of 47.18 in unseen environments, outperforming prior work by 3.39% in SR and 9.08 in SPL. This demonstrates that, for embodied spatial reasoning, targeted enhancements on smaller open-source VLMs can be more effective than brute-force model scaling. The code will be released upon publication.Project page: https://apex-bjut.github.io/Taga-VLM

  </details>


