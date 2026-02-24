# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-24 07:13 UTC_

Total papers shown: **9**


---

- **To Move or Not to Move: Constraint-based Planning Enables Zero-Shot Generalization for Interactive Navigation**  
  Apoorva Vashisth, Manav Kulshrestha, Pranav Bakshi, Damon Conover, Guillaume Sartoretti, Aniket Bera  
  _2026-02-23_ · https://arxiv.org/abs/2602.20055v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual navigation typically assumes the existence of at least one obstacle-free path between start and goal, which must be discovered/planned by the robot. However, in real-world scenarios, such as home environments and warehouses, clutter can block all routes. Targeted at such cases, we introduce the Lifelong Interactive Navigation problem, where a mobile robot with manipulation abilities can move clutter to forge its own path to complete sequential object- placement tasks - each involving placing an given object (eg. Alarm clock, Pillow) onto a target object (eg. Dining table, Desk, Bed). To address this lifelong setting - where effects of environment changes accumulate and have long-term effects - we propose an LLM-driven, constraint-based planning framework with active perception. Our framework allows the LLM to reason over a structured scene graph of discovered objects and obstacles, deciding which object to move, where to place it, and where to look next to discover task-relevant information. This coupling of reasoning and active perception allows the agent to explore the regions expected to contribute to task completion rather than exhaustively mapping the environment. A standard motion planner then executes the corresponding navigate-pick-place, or detour sequence, ensuring reliable low-level control. Evaluated in physics-enabled ProcTHOR-10k simulator, our approach outperforms non-learning and learning-based baselines. We further demonstrate our approach qualitatively on real-world hardware.

  </details>



- **Chasing Ghosts: A Simulation-to-Real Olfactory Navigation Stack with Optional Vision Augmentation**  
  Kordel K. France, Ovidiu Daescu, Latifur Khan, Rohith Peddi  
  _2026-02-23_ · https://arxiv.org/abs/2602.19577v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous odor source localization remains a challenging problem for aerial robots due to turbulent airflow, sparse and delayed sensory signals, and strict payload and compute constraints. While prior unmanned aerial vehicle (UAV)-based olfaction systems have demonstrated gas distribution mapping or reactive plume tracing, they rely on predefined coverage patterns, external infrastructure, or extensive sensing and coordination. In this work, we present a complete, open-source UAV system for online odor source localization using a minimal sensor suite. The system integrates custom olfaction hardware, onboard sensing, and a learning-based navigation policy trained in simulation and deployed on a real quadrotor. Through our minimal framework, the UAV is able to navigate directly toward an odor source without constructing an explicit gas distribution map or relying on external positioning systems. Vision is incorporated as an optional complementary modality to accelerate navigation under certain conditions. We validate the proposed system through real-world flight experiments in a large indoor environment using an ethanol source, demonstrating consistent source-finding behavior under realistic airflow conditions. The primary contribution of this work is a reproducible system and methodological framework for UAV-based olfactory navigation and source finding under minimal sensing assumptions. We elaborate on our hardware design and open source our UAV firmware, simulation code, olfaction-vision dataset, and circuit board to the community. Code, data, and designs will be made available at https://github.com/KordelFranceTech/ChasingGhosts.

  </details>



- **EEG-Driven Intention Decoding: Offline Deep Learning Benchmarking on a Robotic Rover**  
  Ghadah Alosaimi, Maha Alsayyari, Yixin Sun, Stamos Katsigiannis, Amir Atapour-Abarghouei, Toby P. Breckon  
  _2026-02-23_ · https://arxiv.org/abs/2602.20041v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Brain-computer interfaces (BCIs) provide a hands-free control modality for mobile robotics, yet decoding user intent during real-world navigation remains challenging. This work presents a brain-robot control framework for offline decoding of driving commands during robotic rover operation. A 4WD Rover Pro platform was remotely operated by 12 participants who navigated a predefined route using a joystick, executing the commands forward, reverse, left, right, and stop. Electroencephalogram (EEG) signals were recorded with a 16-channel OpenBCI cap and aligned with motor actions at Delta = 0 ms and future prediction horizons (Delta > 0 ms). After preprocessing, several deep learning models were benchmarked, including convolutional neural networks, recurrent neural networks, and Transformer architectures. ShallowConvNet achieved the highest performance for both action prediction and intent prediction. By combining real-world robotic control with multi-horizon EEG intention decoding, this study introduces a reproducible benchmark and reveals key design insights for predictive deep learning-based BCI systems.

  </details>



- **Contextual Safety Reasoning and Grounding for Open-World Robots**  
  Zachary Ravichadran, David Snyder, Alexander Robey, Hamed Hassani, Vijay Kumar, George J. Pappas  
  _2026-02-23_ · https://arxiv.org/abs/2602.19983v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots are increasingly operating in open-world environments where safe behavior depends on context: the same hallway may require different navigation strategies when crowded versus empty, or during an emergency versus normal operations. Traditional safety approaches enforce fixed constraints in user-specified contexts, limiting their ability to handle the open-ended contextual variability of real-world deployment. We address this gap via CORE, a safety framework that enables online contextual reasoning, grounding, and enforcement without prior knowledge of the environment (e.g., maps or safety specifications). CORE uses a vision-language model (VLM) to continuously reason about context-dependent safety rules directly from visual observations, grounds these rules in the physical environment, and enforces the resulting spatially-defined safe sets via control barrier functions. We provide probabilistic safety guarantees for CORE that account for perceptual uncertainty, and we demonstrate through simulation and real-world experiments that CORE enforces contextually appropriate behavior in unseen environments, significantly outperforming prior semantic safety methods that lack online contextual reasoning. Ablation studies validate our theoretical guarantees and underscore the importance of both VLM-based reasoning and spatial grounding for enforcing contextual safety in novel settings. We provide additional resources at https://zacravichandran.github.io/CORE.

  </details>



- **CACTO-BIC: Scalable Actor-Critic Learning via Biased Sampling and GPU-Accelerated Trajectory Optimization**  
  Elisa Alboni, Pietro Noah Crestaz, Elias Fontanari, Andrea Del Prete  
  _2026-02-23_ · https://arxiv.org/abs/2602.19699v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Trajectory Optimization (TO) and Reinforcement Learning (RL) offer complementary strengths for solving optimal control problems. TO efficiently computes locally optimal solutions but can struggle with non-convexity, while RL is more robust to non-convexity at the cost of significantly higher computational demands. CACTO (Continuous Actor-Critic with Trajectory Optimization) was introduced to combine these advantages by learning a warm-start policy that guides the TO solver towards low-cost trajectories. However, scalability remains a key limitation, as increasing system complexity significantly raises the computational cost of TO. This work introduces CACTO-BIC to address these challenges. CACTO-BIC improves data efficiency by biasing initial-state sampling leveraging a property of the value function associated with locally optimal policies; moreover, it reduces computation time by exploiting GPU acceleration. Empirical evaluations show improved sample efficiency and faster computation compared to CACTO. Comparisons with PPO demonstrate that our approach can achieve similar solutions in less time. Finally, experiments on the AlienGO quadruped robot demonstrate that CACTO-BIC can scale to high-dimensional systems and is suitable for real-time applications.

  </details>



- **Compositional Planning with Jumpy World Models**  
  Jesse Farebrother, Matteo Pirotta, Andrea Tirinzoni, Marc G. Bellemare, Alessandro Lazaric, Ahmed Touati  
  _2026-02-23_ · https://arxiv.org/abs/2602.19634v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The ability to plan with temporal abstractions is central to intelligent decision-making. Rather than reasoning over primitive actions, we study agents that compose pre-trained policies as temporally extended actions, enabling solutions to complex tasks that no constituent alone can solve. Such compositional planning remains elusive as compounding errors in long-horizon predictions make it challenging to estimate the visitation distribution induced by sequencing policies. Motivated by the geometric policy composition framework introduced in arXiv:2206.08736, we address these challenges by learning predictive models of multi-step dynamics -- so-called jumpy world models -- that capture state occupancies induced by pre-trained policies across multiple timescales in an off-policy manner. Building on Temporal Difference Flows (arXiv:2503.09817), we enhance these models with a novel consistency objective that aligns predictions across timescales, improving long-horizon predictive accuracy. We further demonstrate how to combine these generative predictions to estimate the value of executing arbitrary sequences of policies over varying timescales. Empirically, we find that compositional planning with jumpy world models significantly improves zero-shot performance across a wide range of base policies on challenging manipulation and navigation tasks, yielding, on average, a 200% relative improvement over planning with primitive actions on long-horizon tasks.

  </details>



- **CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence**  
  Tarakanath Paipuru  
  _2026-02-23_ · https://arxiv.org/abs/2602.20048v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Modern code intelligence agents operate in contexts exceeding 1 million tokens--far beyond the scale where humans manually locate relevant files. Yet agents consistently fail to discover architecturally critical files when solving real-world coding tasks. We identify the Navigation Paradox: agents perform poorly not due to context limits, but because navigation and retrieval are fundamentally distinct problems. Through 258 automated trials across 30 benchmark tasks on a production FastAPI repository, we demonstrate that graph-based structural navigation via CodeCompass--a Model Context Protocol server exposing dependency graphs--achieves 99.4% task completion on hidden-dependency tasks, a 23.2 percentage-point improvement over vanilla agents (76.2%) and 21.2 points over BM25 retrieval (78.2%).However, we uncover a critical adoption gap: 58% of trials with graph access made zero tool calls, and agents required explicit prompt engineering to adopt the tool consistently. Our findings reveal that the bottleneck is not tool availability but behavioral alignment--agents must be explicitly guided to leverage structural context over lexical heuristics. We contribute: (1) a task taxonomy distinguishing semantic-search, structural, and hidden-dependency scenarios; (2) empirical evidence that graph navigation outperforms retrieval when dependencies lack lexical overlap; and (3) open-source infrastructure for reproducible evaluation of navigation tools.

  </details>



- **Rendezvous and Docking of Mobile Ground Robots for Efficient Transportation Systems**  
  Lars Fischer, Daniel Flögel, Sören Hohmann  
  _2026-02-23_ · https://arxiv.org/abs/2602.19862v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  In-Motion physical coupling of multiple mobile ground robots has the potential to enable new applications like in-motion transfer that improves efficiency in handling and transferring goods, which tackles current challenges in logistics. A key challenge lies in achieving reliable autonomous in-motion physical coupling of two mobile ground robots starting at any initial position. Existing approaches neglect the modeling of the docking interface and the strategy for approaching it, resulting in uncontrolled collisions that make in-motion physical coupling either impossible or inefficient. To address this challenge, we propose a central mpc approach that explicitly models the dynamics and states of two omnidirectional wheeled robots, incorporates constraints related to their docking interface, and implements an approaching strategy for rendezvous and docking. This novel approach enables omnidirectional wheeled robots with a docking interface to physically couple in motion regardless of their initial position. In addition, it makes in-motion transfer possible, which is 19.75% more time- and 21.04% energy-efficient compared to a non-coupling approach in a logistic scenario.

  </details>



- **Iconographic Classification and Content-Based Recommendation for Digitized Artworks**  
  Krzysztof Kutt, Maciej Baczyński  
  _2026-02-23_ · https://arxiv.org/abs/2602.19698v1 · `cs.DL`  
  <details><summary>Abstract</summary>

  We present a proof-of-concept system that automates iconographic classification and content-based recommendation of digitized artworks using the Iconclass vocabulary and selected artificial intelligence methods. The prototype implements a four-stage workflow for classification and recommendation, which integrates YOLOv8 object detection with algorithmic mappings to Iconclass codes, rule-based inference for abstract meanings, and three complementary recommenders (hierarchical proximity, IDF-weighted overlap, and Jaccard similarity). Although more engineering is still needed, the evaluation demonstrates the potential of this solution: Iconclass-aware computer vision and recommendation methods can accelerate cataloging and enhance navigation in large heritage repositories. The key insight is to let computer vision propose visible elements and to use symbolic structures (Iconclass hierarchy) to reach meaning.

  </details>


