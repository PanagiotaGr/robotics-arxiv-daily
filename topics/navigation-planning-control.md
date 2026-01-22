# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-01-22 06:51 UTC_

Total papers shown: **7**


---

- **HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV**  
  Faryal Batool, Iana Zhura, Valerii Serpiva, Roohan Ahmed Khan, Ivan Valuev, Issatay Tokmurziyev, Dzmitry Tsetserukou  
  _2026-01-21_ · https://arxiv.org/abs/2601.14973v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable human--robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments. This paper presents HumanDiffusion, a lightweight image-conditioned diffusion planner that generates human-aware navigation trajectories directly from RGB imagery. The system combines YOLO-11--based human detection with diffusion-driven trajectory generation, enabling a quadrotor to approach a target person and deliver medical assistance without relying on prior maps or computationally intensive planning pipelines. Trajectories are predicted in pixel space, ensuring smooth motion and a consistent safety margin around humans. We evaluate HumanDiffusion in simulation and real-world indoor mock-disaster scenarios. On a 300-sample test set, the model achieves a mean squared error of 0.02 in pixel-space trajectory reconstruction. Real-world experiments demonstrate an overall mission success rate of 80% across accident-response and search-and-locate tasks with partial occlusions. These results indicate that human-conditioned diffusion planning offers a practical and robust solution for human-aware UAV navigation in time-critical assistance settings.

  </details>



- **The Why Behind the Action: Unveiling Internal Drivers via Agentic Attribution**  
  Chen Qian, Peng Wang, Dongrui Liu, Junyao Yang, Dadi Guo, Ling Tang, Jilin Mei, Qihan Ren, Shuai Shao, Yong Liu, et al.  
  _2026-01-21_ · https://arxiv.org/abs/2601.15075v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Large Language Model (LLM)-based agents are widely used in real-world applications such as customer service, web navigation, and software engineering. As these systems become more autonomous and are deployed at scale, understanding why an agent takes a particular action becomes increasingly important for accountability and governance. However, existing research predominantly focuses on \textit{failure attribution} to localize explicit errors in unsuccessful trajectories, which is insufficient for explaining the reasoning behind agent behaviors. To bridge this gap, we propose a novel framework for \textbf{general agentic attribution}, designed to identify the internal factors driving agent actions regardless of the task outcome. Our framework operates hierarchically to manage the complexity of agent interactions. Specifically, at the \textit{component level}, we employ temporal likelihood dynamics to identify critical interaction steps; then at the \textit{sentence level}, we refine this localization using perturbation-based analysis to isolate the specific textual evidence. We validate our framework across a diverse suite of agentic scenarios, including standard tool use and subtle reliability risks like memory-induced bias. Experimental results demonstrate that the proposed framework reliably pinpoints pivotal historical events and sentences behind the agent behavior, offering a critical step toward safer and more accountable agentic systems.

  </details>



- **Moving Beyond Compliance in Soft-Robotic Catheters Through Modularity for Precision Therapies**  
  B. Calmé, N. J. Greenidge, A. Metcalf, A. Bacchetti, G. Loza, D. Kpeglo, P. Lloyd, V. Pensabene, J. H. Chandler, P. Valdastri  
  _2026-01-21_ · https://arxiv.org/abs/2601.14837v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Soft robotic instruments could navigate delicate, tortuous anatomy more safely than rigid tools, but clinical adoption is limited by insufficient tip functionalization and real-time feedback at the tissue interface. Few sensing and therapeutic modules are compact, robust, and adaptable enough to measure, and respond to, subtle physiological cues during intraluminal procedures. We present a 1.47 mm diameter modular soft robotic catheter that integrates sensing, actuation, and therapy while retaining the compliance needed for safe endoluminal navigation. Validated across multiple in vivo settings, we emphasize its utility in endoscopic retrograde cholangiopancreatography (ERCP), a highly technical procedure and a key access route to the pancreas, an organ that is fragile, difficult to instrument, and central to diseases such as pancreatic cancer. Our architecture supports up to four independently controlled functional units, allowing customizable combinations of anchoring, manipulation, sensing, and targeted drug delivery. In a live porcine model, we demonstrate semi-autonomous deployment into the pancreatic duct and 7.5 cm of endoscopic navigation within it, a region currently inaccessible with standard catheters. A closed-loop autonomous/shared-control system that combines a learned model, magnetic actuation, onboard shape sensing, and visual marker tracking further improves cannulation accuracy. Together, these results establish a scalable platform for multifunctional soft robotic catheters and a new paradigm for complex endoluminal interventions, with potential to reduce radiation exposure, shorten training, and accelerate clinical translation of soft robotic technologies.

  </details>



- **DWPP: Dynamic Window Pure Pursuit Considering Velocity and Acceleration Constraints**  
  Fumiya Ohnishi, Masaki Takahashi  
  _2026-01-21_ · https://arxiv.org/abs/2601.15006v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Pure pursuit and its variants are widely used for mobile robot path tracking owing to their simplicity and computational efficiency. However, many conventional approaches do not explicitly account for velocity and acceleration constraints, resulting in discrepancies between commanded and actual velocities that result in overshoot and degraded tracking performance. To address this problem, this paper proposes dynamic window pure pursuit (DWPP), which fundamentally reformulates the command velocity computation process to explicitly incorporate velocity and acceleration constraints. Specifically, DWPP formulates command velocity computation in the velocity space (the $v$-$ω$ plane) and selects the command velocity as the point within the dynamic window that is closest to the line $ω= κv$. Experimental results demonstrate that DWPP avoids constraint-violating commands and achieves superior path-tracking accuracy compared with conventional pure pursuit methods. The proposed method has been integrated into the official Nav2 repository and is publicly available (https://github.com/ros-navigation/navigation2).

  </details>



- **Walk through Paintings: Egocentric World Models from Internet Priors**  
  Anurag Bagchi, Zhipeng Bao, Homanga Bharadhwaj, Yu-Xiong Wang, Pavel Tokmakov, Martial Hebert  
  _2026-01-21_ · https://arxiv.org/abs/2601.15284v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  What if a video generation model could not only imagine a plausible future, but the correct one, accurately reflecting how the world changes with each action? We address this question by presenting the Egocentric World Model (EgoWM), a simple, architecture-agnostic method that transforms any pretrained video diffusion model into an action-conditioned world model, enabling controllable future prediction. Rather than training from scratch, we repurpose the rich world priors of Internet-scale video models and inject motor commands through lightweight conditioning layers. This allows the model to follow actions faithfully while preserving realism and strong generalization. Our approach scales naturally across embodiments and action spaces, ranging from 3-DoF mobile robots to 25-DoF humanoids, where predicting egocentric joint-angle-driven dynamics is substantially more challenging. The model produces coherent rollouts for both navigation and manipulation tasks, requiring only modest fine-tuning. To evaluate physical correctness independently of visual appearance, we introduce the Structural Consistency Score (SCS), which measures whether stable scene elements evolve consistently with the provided actions. EgoWM improves SCS by up to 80 percent over prior state-of-the-art navigation world models, while achieving up to six times lower inference latency and robust generalization to unseen environments, including navigation inside paintings.

  </details>



- **Risk Estimation for Automated Driving**  
  Leon Tolksdorf, Arturo Tejada, Jonas Bauernfeind, Christian Birkner, Nathan van de Wouw  
  _2026-01-21_ · https://arxiv.org/abs/2601.15018v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Safety is a central requirement for automated vehicles. As such, the assessment of risk in automated driving is key in supporting both motion planning technologies and safety evaluation. In automated driving, risk is characterized by two aspects. The first aspect is the uncertainty on the state estimates of other road participants by an automated vehicle. The second aspect is the severity of a collision event with said traffic participants. Here, the uncertainty aspect typically causes the risk to be non-zero for near-collision events. This makes risk particularly useful for automated vehicle motion planning. Namely, constraining or minimizing risk naturally navigates the automated vehicle around traffic participants while keeping a safety distance based on the level of uncertainty and the potential severity of the impending collision. Existing approaches to calculate the risk either resort to empirical modeling or severe approximations, and, hence, lack generalizability and accuracy. In this paper, we combine recent advances in collision probability estimation with the concept of collision severity to develop a general method for accurate risk estimation. The proposed method allows us to assign individual severity functions for different collision constellations, such as, e.g., frontal or side collisions. Furthermore, we show that the proposed approach is computationally efficient, which is beneficial, e.g., in real-time motion planning applications. The programming code for an exemplary implementation of Gaussian uncertainties is also provided.

  </details>



- **SpatialMem: Unified 3D Memory with Metric Anchoring and Fast Retrieval**  
  Xinyi Zheng, Yunze Liu, Chi-Hao Wu, Fan Zhang, Hao Zheng, Wenqi Zhou, Walterio W. Mayol-Cuevas, Junxiao Shen  
  _2026-01-21_ · https://arxiv.org/abs/2601.14895v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present SpatialMem, a memory-centric system that unifies 3D geometry, semantics, and language into a single, queryable representation. Starting from casually captured egocentric RGB video, SpatialMem reconstructs metrically scaled indoor environments, detects structural 3D anchors (walls, doors, windows) as the first-layer scaffold, and populates a hierarchical memory with open-vocabulary object nodes -- linking evidence patches, visual embeddings, and two-layer textual descriptions to 3D coordinates -- for compact storage and fast retrieval. This design enables interpretable reasoning over spatial relations (e.g., distance, direction, visibility) and supports downstream tasks such as language-guided navigation and object retrieval without specialized sensors. Experiments across three real-life indoor scenes demonstrate that SpatialMem maintains strong anchor-description-level navigation completion and hierarchical retrieval accuracy under increasing clutter and occlusion, offering an efficient and extensible framework for embodied spatial intelligence.

  </details>


