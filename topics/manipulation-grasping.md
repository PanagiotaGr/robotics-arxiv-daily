# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-24 07:13 UTC_

Total papers shown: **7**


---

- **To Move or Not to Move: Constraint-based Planning Enables Zero-Shot Generalization for Interactive Navigation**  
  Apoorva Vashisth, Manav Kulshrestha, Pranav Bakshi, Damon Conover, Guillaume Sartoretti, Aniket Bera  
  _2026-02-23_ · https://arxiv.org/abs/2602.20055v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual navigation typically assumes the existence of at least one obstacle-free path between start and goal, which must be discovered/planned by the robot. However, in real-world scenarios, such as home environments and warehouses, clutter can block all routes. Targeted at such cases, we introduce the Lifelong Interactive Navigation problem, where a mobile robot with manipulation abilities can move clutter to forge its own path to complete sequential object- placement tasks - each involving placing an given object (eg. Alarm clock, Pillow) onto a target object (eg. Dining table, Desk, Bed). To address this lifelong setting - where effects of environment changes accumulate and have long-term effects - we propose an LLM-driven, constraint-based planning framework with active perception. Our framework allows the LLM to reason over a structured scene graph of discovered objects and obstacles, deciding which object to move, where to place it, and where to look next to discover task-relevant information. This coupling of reasoning and active perception allows the agent to explore the regions expected to contribute to task completion rather than exhaustively mapping the environment. A standard motion planner then executes the corresponding navigate-pick-place, or detour sequence, ensuring reliable low-level control. Evaluated in physics-enabled ProcTHOR-10k simulator, our approach outperforms non-learning and learning-based baselines. We further demonstrate our approach qualitatively on real-world hardware.

  </details>



- **NovaPlan: Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language Planning**  
  Jiahui Fu, Junyu Nan, Lingfeng Sun, Hongyu Li, Jianing Qian, Jennifer L. Barry, Kris Kitani, George Konidaris  
  _2026-02-23_ · https://arxiv.org/abs/2602.20119v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Solving long-horizon tasks requires robots to integrate high-level semantic reasoning with low-level physical interaction. While vision-language models (VLMs) and video generation models can decompose tasks and imagine outcomes, they often lack the physical grounding necessary for real-world execution. We introduce NovaPlan, a hierarchical framework that unifies closed-loop VLM and video planning with geometrically grounded robot execution for zero-shot long-horizon manipulation. At the high level, a VLM planner decomposes tasks into sub-goals and monitors robot execution in a closed loop, enabling the system to recover from single-step failures through autonomous re-planning. To compute low-level robot actions, we extract and utilize both task-relevant object keypoints and human hand poses as kinematic priors from the generated videos, and employ a switching mechanism to choose the better one as a reference for robot actions, maintaining stable execution even under heavy occlusion or depth inaccuracy. We demonstrate the effectiveness of NovaPlan on three long-horizon tasks and the Functional Manipulation Benchmark (FMB). Our results show that NovaPlan can perform complex assembly tasks and exhibit dexterous error recovery behaviors without any prior demonstrations or training. Project page: https://nova-plan.github.io/

  </details>



- **Towards Dexterous Embodied Manipulation via Deep Multi-Sensory Fusion and Sparse Expert Scaling**  
  Yirui Sun, Guangyu Zhuge, Keliang Liu, Jie Gu, Zhihao xia, Qionglin Ren, Chunxu tian, Zhongxue Ga  
  _2026-02-23_ · https://arxiv.org/abs/2602.19764v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Realizing dexterous embodied manipulation necessitates the deep integration of heterogeneous multimodal sensory inputs. However, current vision-centric paradigms often overlook the critical force and geometric feedback essential for complex tasks. This paper presents DeMUSE, a Deep Multimodal Unified Sparse Experts framework leveraging a Diffusion Transformer to integrate RGB, depth, and 6-axis force into a unified serialized stream. Adaptive Modality-specific Normalization (AdaMN) is employed to recalibrate modality-aware features, mitigating representation imbalance and harmonizing the heterogeneous distributions of multi-sensory signals. To facilitate efficient scaling, the architecture utilizes a Sparse Mixture-of-Experts (MoE) with shared experts, increasing model capacity for physical priors while maintaining the low inference latency required for real-time control. A Joint denoising objective synchronously synthesizes environmental evolution and action sequences to ensure physical consistency. Achieving success rates of 83.2% and 72.5% in simulation and real-world trials, DeMUSE demonstrates state-of-the-art performance, validating the necessity of deep multi-sensory integration for complex physical interactions.

  </details>



- **TactiVerse: Generalizing Multi-Point Tactile Sensing in Soft Robotics Using Single-Point Data**  
  Junhui Lee, Hyosung Kim, Saekwang Nam  
  _2026-02-23_ · https://arxiv.org/abs/2602.19850v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Real-time prediction of deformation in highly compliant soft materials remains a significant challenge in soft robotics. While vision-based soft tactile sensors can track internal marker displacements, learning-based models for 3D contact estimation heavily depend on their training datasets, inherently limiting their ability to generalize to complex scenarios such as multi-point sensing. To address this limitation, we introduce TactiVerse, a U-Net-based framework that formulates contact geometry estimation as a spatial heatmap prediction task. Even when trained exclusively on a limited dataset of single-point indentations, our architecture achieves highly accurate single-point sensing, yielding a superior mean absolute error of 0.0589 mm compared to the 0.0612 mm of a conventional regression-based CNN baseline. Furthermore, we demonstrate that augmenting the training dataset with multi-point contact data substantially enhances the sensor's multi-point sensing capabilities, significantly improving the overall mean MAE for two-point discrimination from 1.214 mm to 0.383 mm. By successfully extrapolating complex contact geometries from fundamental interactions, this methodology unlocks advanced multi-point and large-area shape sensing. Ultimately, it significantly streamlines the development of marker-based soft sensors, offering a highly scalable solution for real-world tactile perception.

  </details>



- **Compositional Planning with Jumpy World Models**  
  Jesse Farebrother, Matteo Pirotta, Andrea Tirinzoni, Marc G. Bellemare, Alessandro Lazaric, Ahmed Touati  
  _2026-02-23_ · https://arxiv.org/abs/2602.19634v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The ability to plan with temporal abstractions is central to intelligent decision-making. Rather than reasoning over primitive actions, we study agents that compose pre-trained policies as temporally extended actions, enabling solutions to complex tasks that no constituent alone can solve. Such compositional planning remains elusive as compounding errors in long-horizon predictions make it challenging to estimate the visitation distribution induced by sequencing policies. Motivated by the geometric policy composition framework introduced in arXiv:2206.08736, we address these challenges by learning predictive models of multi-step dynamics -- so-called jumpy world models -- that capture state occupancies induced by pre-trained policies across multiple timescales in an off-policy manner. Building on Temporal Difference Flows (arXiv:2503.09817), we enhance these models with a novel consistency objective that aligns predictions across timescales, improving long-horizon predictive accuracy. We further demonstrate how to combine these generative predictions to estimate the value of executing arbitrary sequences of policies over varying timescales. Empirically, we find that compositional planning with jumpy world models significantly improves zero-shot performance across a wide range of base policies on challenging manipulation and navigation tasks, yielding, on average, a 200% relative improvement over planning with primitive actions on long-horizon tasks.

  </details>



- **AdaWorldPolicy: World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation**  
  Ge Yuan, Qiyuan Qiao, Jing Zhang, Dong Xu  
  _2026-02-23_ · https://arxiv.org/abs/2602.20057v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. In this work, we introduce a unified framework, World-Model-Driven Diffusion Policy with Online Adaptive Learning (AdaWorldPolicy) to enhance robotic manipulation under dynamic conditions with minimal human involvement. Our core insight is that world models provide strong supervision signals, enabling online adaptive learning in dynamic environments, which can be complemented by force-torque feedback to mitigate dynamic force shifts. Our AdaWorldPolicy integrates a world model, an action expert, and a force predictor-all implemented as interconnected Flow Matching Diffusion Transformers (DiT). They are interconnected via the multi-modal self-attention layers, enabling deep feature exchange for joint learning while preserving their distinct modularity characteristics. We further propose a novel Online Adaptive Learning (AdaOL) strategy that dynamically switches between an Action Generation mode and a Future Imagination mode to drive reactive updates across all three modules. This creates a powerful closed-loop mechanism that adapts to both visual and physical domain shifts with minimal overhead. Across a suite of simulated and real-robot benchmarks, our AdaWorldPolicy achieves state-of-the-art performance, with dynamical adaptive capacity to out-of-distribution scenarios.

  </details>



- **Scalable Low-Density Distributed Manipulation Using an Interconnected Actuator Array**  
  Bailey Dacre, Rodrigo Moreno, Jørn Lambertsen, Kasper Stoy, Andrés Faíña  
  _2026-02-23_ · https://arxiv.org/abs/2602.19653v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Distributed Manipulator Systems, composed of arrays of robotic actuators necessitate dense actuator arrays to effectively manipulate small objects. This paper presents a system composed of modular 3-DoF robotic tiles interconnected by a compliant surface layer, forming a continuous, controllable manipulation surface. The compliant layer permits increased actuator spacing without compromising object manipulation capabilities, significantly reducing actuator density while maintaining robust control, even for smaller objects. We characterize the coupled workspace of the array and develop a manipulation strategy capable of translating objects to arbitrary positions within an N X N array. The approach is validated experimentally using a minimal 2 X 2 prototype, demonstrating the successful manipulation of objects with varied shapes and sizes.

  </details>


