# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-03-04 07:02 UTC_

Total papers shown: **7**


---

- **MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN**  
  Ling Luo, Qianqian Bai  
  _2026-03-03_ · https://arxiv.org/abs/2603.03024v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language Navigation (VLN) aims to empower robots with the ability to perform long-horizon navigation in unfamiliar environments based on complex linguistic instructions. Its success critically hinges on establishing an efficient ``language-understanding -- visual-perception -- embodied-execution'' closed loop. Existing methods often suffer from perceptual distortion and decision drift in complex, long-distance tasks due to the cognitive overload of a single agent. Inspired by distributed cognition theory, this paper proposes MA-CoNav, a Multi-Agent Collaborative Navigation framework. This framework adopts a ``Master-Slave'' hierarchical agent collaboration architecture, decoupling and distributing the perception, planning, execution, and memory functions required for navigation tasks to specialized agents. Specifically, the Master Agent is responsible for global orchestration, while the Subordinate Agent group collaborates through a clear division of labor: an Observation Agent generates environment descriptions, a Planning Agent performs task decomposition and dynamic verification, an Execution Agent handles simultaneous mapping and action, and a Memory Agent manages structured experiences. Furthermore, the framework introduces a ``Local-Global'' dual-stage reflection mechanism to dynamically optimize the entire navigation pipeline. Empirical experiments were conducted using a real-world indoor dataset collected by a Limo Pro robot, with no scene-specific fine-tuning performed on the models throughout the process. The results demonstrate that MA-CoNav comprehensively outperforms existing mainstream VLN methods across multiple metrics.

  </details>



- **SPARC: Spatial-Aware Path Planning via Attentive Robot Communication**  
  Sayang Mu, Xiangyu Wu, Bo An  
  _2026-03-03_ · https://arxiv.org/abs/2603.02845v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Efficient communication is critical for decentralized Multi-Robot Path Planning (MRPP), yet existing learned communication methods treat all neighboring robots equally regardless of their spatial proximity, leading to diluted attention in congested regions where coordination matters most. We propose Relation enhanced Multi Head Attention (RMHA), a communication mechanism that explicitly embeds pairwise Manhattan distances into the attention weight computation, enabling each robot to dynamically prioritize messages from spatially relevant neighbors. Combined with a distance-constrained attention mask and GRU gated message fusion, RMHA integrates seamlessly with MAPPO for stable end-to-end training. In zero-shot generalization from 8 training robots to 128 test robots on 40x40 grids, RMHA achieves approximately 75 percent success rate at 30 percent obstacle density outperforming the best baseline by over 25 percentage points. Ablation studies confirm that distance-relation encoding is the key contributor to success rate improvement in high-density environments. Index Terms-Multi-robot path planning, graph attention mechanism, multi-head attention, communication optimization, cooperative decision-making

  </details>



- **HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations**  
  Xiaomeng Xu, Jisang Park, Han Zhang, Eric Cousineau, Aditya Bhat, Jose Barreiros, Dian Wang, Shuran Song  
  _2026-03-03_ · https://arxiv.org/abs/2603.03243v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present Whole-Body Mobile Manipulation Interface (HoMMI), a data collection and policy learning framework that learns whole-body mobile manipulation directly from robot-free human demonstrations. We augment UMI interfaces with egocentric sensing to capture the global context required for mobile manipulation, enabling portable, robot-free, and scalable data collection. However, naively incorporating egocentric sensing introduces a larger human-to-robot embodiment gap in both observation and action spaces, making policy transfer difficult. We explicitly bridge this gap with a cross-embodiment hand-eye policy design, including an embodiment agnostic visual representation; a relaxed head action representation; and a whole-body controller that realizes hand-eye trajectories through coordinated whole-body motion under robot-specific physical constraints. Together, these enable long-horizon mobile manipulation tasks requiring bimanual and whole-body coordination, navigation, and active perception. Results are best viewed on: https://hommi-robot.github.io

  </details>



- **Architectural HRI: Towards a Robotic Paradigm Shift in Human-Building Interaction**  
  Alex Binh Vinh Duc Nguyen  
  _2026-03-03_ · https://arxiv.org/abs/2603.03052v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Recent advances in sensing, communication, interfaces, control, and robotics are expanding Human-Building Interaction (HBI) beyond adaptive building services and facades toward the physical actuation of architectural space. In parallel, research in robotic furniture, swarm robotics, and shape-changing spaces shows that architectural elements can now be robotically augmented to move, reconfigure, and adapt space. We propose that these advances promise a paradigm shift in HBI, in which multiple building layers physically adapt in synchrony to support occupant needs and sustainability goals more holistically. Conversely, we argue that this emerging paradigm also provides an ideal case for transferring HRI knowledge to unconventional robotic morphologies, including the interpretation of the robot as multiple architectural layers or even as a building. However, this research agenda remains challenged by the temporal, spatial, and social complexity of architectural HRI, and by fragmented knowledge across HCI, environmental psychology, cognitive science, and architecture. We therefore call for interdisciplinary research that unifies the why, what, and how of robotic actuation in architectural forms.

  </details>



- **Generative adversarial imitation learning for robot swarms: Learning from human demonstrations and trained policies**  
  Mattes Kraus, Jonas Kuckling  
  _2026-03-03_ · https://arxiv.org/abs/2603.02783v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In imitation learning, robots are supposed to learn from demonstrations of the desired behavior. Most of the work in imitation learning for swarm robotics provides the demonstrations as rollouts of an existing policy. In this work, we provide a framework based on generative adversarial imitation learning that aims to learn collective behaviors from human demonstrations. Our framework is evaluated across six different missions, learning both from manual demonstrations and demonstrations derived from a PPO-trained policy. Results show that the imitation learning process is able to learn qualitatively meaningful behaviors that perform similarly well as the provided demonstrations. Additionally, we deploy the learned policies on a swarm of TurtleBot 4 robots in real-robot experiments. The exhibited behaviors preserved their visually recognizable character and their performance is comparable to the one achieved in simulation.

  </details>



- **An Optimization-Based User Scheduling Framework for Multiuser MIMO Systems**  
  Victoria Palhares, Christoph Studer  
  _2026-03-03_ · https://arxiv.org/abs/2603.02998v1 · `cs.IT`  
  <details><summary>Abstract</summary>

  Resource allocation is a key factor in multiuser (MU) multiple-input multiple-output (MIMO) wireless systems to provide high quality of service to all user equipments (UEs). In congested scenarios, UE scheduling enables UEs to be distributed over time, frequency, or space in order to mitigate inter-UE interference. Many existing UE scheduling methods rely on greedy algorithms, which fail at treating the resource-allocation problem globally. In this work, we propose a UE scheduling framework for MU-MIMO wireless systems that approximately solves a nonconvex optimization problem that treats scheduling globally. Our UE scheduling framework determines subsets of UEs that should transmit simultaneously in a given resource slot and is flexible in the sense that it (i) supports a variety of objective functions (e.g., post-equalization mean squared error, capacity, and achievable sum rate) and (ii) enables precise control over the minimum and maximum number of resources the UEs should occupy. We demonstrate the efficacy of our UE scheduling framework for millimeter-wave massive MU-MIMO and sub-6-GHz cell-free massive MU-MIMO systems, and we show that it outperforms existing scheduling algorithms while approaching the performance of an exhaustive search.

  </details>



- **NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing**  
  Tianlin Pan, Jiayi Dai, Chenpu Yuan, Zhengyao Lv, Binxin Yang, Hubery Yin, Chen Li, Jing Lyu, Caifeng Shan, Chenyang Si  
  _2026-03-03_ · https://arxiv.org/abs/2603.02802v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent video editing models have achieved impressive results, but most still require large-scale paired datasets. Collecting such naturally aligned pairs at scale remains highly challenging and constitutes a critical bottleneck, especially for local video editing data. Existing workarounds transfer image editing to video through global motion control for pair-free video editing, but such designs struggle with background and temporal consistency. In this paper, we propose NOVA: Sparse Control \& Dense Synthesis, a new framework for unpaired video editing. Specifically, the sparse branch provides semantic guidance through user-edited keyframes distributed across the video, and the dense branch continuously incorporates motion and texture information from the original video to maintain high fidelity and coherence. Moreover, we introduce a degradation-simulation training strategy that enables the model to learn motion reconstruction and temporal consistency by training on artificially degraded videos, thus eliminating the need for paired data. Our extensive experiments demonstrate that NOVA outperforms existing approaches in edit fidelity, motion preservation, and temporal coherence.

  </details>


