# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-03 07:06 UTC_

Total papers shown: **5**


---

- **TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments**  
  Zhiyu Huang, Yun Zhang, Johnson Liu, Rui Song, Chen Tang, Jiaqi Ma  
  _2026-02-02_ · https://arxiv.org/abs/2602.02459v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots in dynamic, human-centric environments must follow language instructions while maintaining real-time reactive control. Vision-language-action (VLA) models offer a promising framework, but they assume temporally aligned reasoning and control, despite semantic inference being inherently delayed relative to real-time action. We introduce Think-in-Control (TIC)-VLA, a latency-aware framework that explicitly models delayed semantic reasoning during action generation. TIC-VLA defines a delayed semantic-control interface that conditions action generation on delayed vision-language semantic states and explicit latency metadata, in addition to current observations, enabling policies to compensate for asynchronous reasoning. We further propose a latency-consistent training pipeline that injects reasoning inference delays during imitation learning and online reinforcement learning, aligning training with asynchronous deployment. To support realistic evaluation, we present DynaNav, a physics-accurate, photo-realistic simulation suite for language-guided navigation in dynamic environments. Extensive experiments in simulation and on a real robot show that TIC-VLA consistently outperforms prior VLA models while maintaining robust real-time control under multi-second reasoning latency. Project website: https://ucla-mobility.github.io/TIC-VLA/

  </details>



- **Traffic-Aware Navigation in Road Networks**  
  Sarah Nassar  
  _2026-02-02_ · https://arxiv.org/abs/2602.02158v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  This project compares three graph search approaches for the task of traffic-aware navigation in Kingston's road network. These approaches include a single-run multi-query preprocessing algorithm (Floyd-Warshall-Ingerman), continuous single-query real-time search (Dijkstra's and A*), and an algorithm combining both approaches to balance between their trade-offs by first finding the top K shortest paths then iterating over them in real time (Yen's). Dijkstra's and A* resulted in the most traffic-aware optimal solutions with minimal preprocessing required. Floyd-Warshall-Ingerman was the fastest in real time but provided distance based paths with no traffic awareness. Yen's algorithm required significant preprocessing but balanced between the other two approaches in terms of runtime speed and optimality. Each approach presents advantages and disadvantages that need to be weighed depending on the circumstances of specific deployment contexts to select the best custom solution. *This project was completed as part of ELEC 844 (Search and Planning Algorithms for Robotics) in the Fall 2025 term.

  </details>



- **SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration**  
  Qingni Wang, Yue Fan, Xin Eric Wang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02419v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Graphical User Interface (GUI) grounding aims to translate natural language instructions into executable screen coordinates, enabling automated GUI interaction. Nevertheless, incorrect grounding can result in costly, hard-to-reverse actions (e.g., erroneous payment approvals), raising concerns about model reliability. In this paper, we introduce SafeGround, an uncertainty-aware framework for GUI grounding models that enables risk-aware predictions through calibrations before testing. SafeGround leverages a distribution-aware uncertainty quantification method to capture the spatial dispersion of stochastic samples from outputs of any given model. Then, through the calibration process, SafeGround derives a test-time decision threshold with statistically guaranteed false discovery rate (FDR) control. We apply SafeGround on multiple GUI grounding models for the challenging ScreenSpot-Pro benchmark. Experimental results show that our uncertainty measure consistently outperforms existing baselines in distinguishing correct from incorrect predictions, while the calibrated threshold reliably enables rigorous risk control and potentials of substantial system-level accuracy improvements. Across multiple GUI grounding models, SafeGround improves system-level accuracy by up to 5.38\% percentage points over Gemini-only inference.

  </details>



- **Well-Posed KL-Regularized Control via Wasserstein and Kalman-Wasserstein KL Divergences**  
  Viktor Stein, Adwait Datar, Nihat Ay  
  _2026-02-02_ · https://arxiv.org/abs/2602.02250v1 · `math.OC`  
  <details><summary>Abstract</summary>

  Kullback-Leibler divergence (KL) regularization is widely used in reinforcement learning, but it becomes infinite under support mismatch and can degenerate in low-noise limits. Utilizing a unified information-geometric framework, we introduce (Kalman)-Wasserstein-based KL analogues by replacing the Fisher-Rao geometry in the dynamical formulation of the KL with transport-based geometries, and we derive closed-form values for common distribution families. These divergences remain finite under support mismatch and yield a geometric interpretation of regularization heuristics used in Kalman ensemble methods. We demonstrate the utility of these divergences in KL-regularized optimal control. In the fully tractable setting of linear time-invariant systems with Gaussian process noise, the classical KL reduces to a quadratic control penalty that becomes singular as process noise vanishes. Our variants remove this singularity, yielding well-posed problems. On a double integrator and a cart-pole example, the resulting controls outperform KL-based regularization.

  </details>



- **LangMap: A Hierarchical Benchmark for Open-Vocabulary Goal Navigation**  
  Bo Miao, Weijia Liu, Jun Luo, Lachlan Shinnick, Jian Liu, Thomas Hamilton-Smith, Yuhe Yang, Zijie Wu, Vanja Videnovic, Feras Dayoub, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02220v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The relationships between objects and language are fundamental to meaningful communication between humans and AI, and to practically useful embodied intelligence. We introduce HieraNav, a multi-granularity, open-vocabulary goal navigation task where agents interpret natural language instructions to reach targets at four semantic levels: scene, room, region, and instance. To this end, we present Language as a Map (LangMap), a large-scale benchmark built on real-world 3D indoor scans with comprehensive human-verified annotations and tasks spanning these levels. LangMap provides region labels, discriminative region descriptions, discriminative instance descriptions covering 414 object categories, and over 18K navigation tasks. Each target features both concise and detailed descriptions, enabling evaluation across different instruction styles. LangMap achieves superior annotation quality, outperforming GOAT-Bench by 23.8% in discriminative accuracy using four times fewer words. Comprehensive evaluations of zero-shot and supervised models on LangMap reveal that richer context and memory improve success, while long-tailed, small, context-dependent, and distant goals, as well as multi-goal completion, remain challenging. HieraNav and LangMap establish a rigorous testbed for advancing language-driven embodied navigation. Project: https://bo-miao.github.io/LangMap

  </details>


