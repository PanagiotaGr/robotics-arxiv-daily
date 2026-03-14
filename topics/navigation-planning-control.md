# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-03-14 07:02 UTC_

Total papers shown: **6**


---

- **Separable neural architectures as a primitive for unified predictive and generative intelligence**  
  Reza T. Batley, Apurba Sarker, Rajib Mostakim, Andrew Klichine, Sourav Saha  
  _2026-03-12_ · https://arxiv.org/abs/2603.12244v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Intelligent systems across physics, language and perception often exhibit factorisable structure, yet are typically modelled by monolithic neural architectures that do not explicitly exploit this structure. The separable neural architecture (SNA) addresses this by formalising a representational class that unifies additive, quadratic and tensor-decomposed neural models. By constraining interaction order and tensor rank, SNAs impose a structural inductive bias that factorises high-dimensional mappings into low-arity components. Separability need not be a property of the system itself: it often emerges in the coordinates or representations through which the system is expressed. Crucially, this coordinate-aware formulation reveals a structural analogy between chaotic spatiotemporal dynamics and linguistic autoregression. By treating continuous physical states as smooth, separable embeddings, SNAs enable distributional modelling of chaotic systems. This approach mitigates the nonphysical drift characteristics of deterministic operators whilst remaining applicable to discrete sequences. The compositional versatility of this approach is demonstrated across four domains: autonomous waypoint navigation via reinforcement learning, inverse generation of multifunctional microstructures, distributional modelling of turbulent flow and neural language modelling. These results establish the separable neural architecture as a domain-agnostic primitive for predictive and generative intelligence, capable of unifying both deterministic and distributional representations.

  </details>



- **ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control**  
  Chetan Borse, Zhixian Xie, Wei-Cheng Huang, Wanxin Jin  
  _2026-03-12_ · https://arxiv.org/abs/2603.12185v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Physics simulation for contact-rich robotics is often bottlenecked by contact resolution: mainstream engines enforce non-penetration and Coulomb friction via complementarity constraints or constrained optimization, requiring per-step iterative solves whose cost grows superlinearly with contact density. We present ComFree-Sim, a GPU-parallelized analytical contact physics engine built on complementarity-free contact modeling. ComFree-Sim computes contact impulses in closed form via an impedance-style prediction--correction update in the dual cone of Coulomb friction. Contact computation decouples across contact pairs and becomes separable across cone facets, mapping naturally to GPU kernels and yielding near-linear runtime scaling with the number of contacts. We further extend the formulation to a unified 6D contact model capturing tangential, torsional, and rolling friction, and introduce a practical dual-cone impedance heuristic. ComFree-Sim is implemented in Warp and exposed through a MuJoCo-compatible interface as a drop-in backend alternative to MuJoCo Warp (MJWarp). Experiments benchmark penetration, friction behaviors, stability, and simulation runtime scaling against MJWarp, demonstrating near-linear scaling and 2--3 times higher throughput in dense contact scenes with comparable physical fidelity. We deploy ComFree-Sim in real-time MPC for in-hand dexterous manipulation on a real-world multi-fingered LEAP hand and in dynamics-aware motion retargeting, demonstrating that low-latency simulation yields higher closed-loop success rates and enables practical high-frequency control in contact-rich tasks.

  </details>



- **AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies**  
  Jennifer Nolan, Travis Driver, John Christian  
  _2026-03-12_ · https://arxiv.org/abs/2603.11969v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.

  </details>



- **Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections**  
  Łukasz Borchmann, Jordy Van Landeghem, Michał Turski, Shreyansh Padarha, Ryan Othniel Kearns, Adam Mahdi, Niels Rogge, Clémentine Fourrier, Siwei Han, Huaxiu Yao, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12180v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Multimodal agents offer a promising path to automating complex document-intensive workflows. Yet, a critical question remains: do these agents demonstrate genuine strategic reasoning, or merely stochastic trial-and-error search? To address this, we introduce MADQA, a benchmark of 2,250 human-authored questions grounded in 800 heterogeneous PDF documents. Guided by Classical Test Theory, we design it to maximize discriminative power across varying levels of agentic abilities. To evaluate agentic behaviour, we introduce a novel evaluation protocol measuring the accuracy-effort trade-off. Using this framework, we show that while the best agents can match human searchers in raw accuracy, they succeed on largely different questions and rely on brute-force search to compensate for weak strategic planning. They fail to close the nearly 20% gap to oracle performance, persisting in unproductive loops. We release the dataset and evaluation harness to help facilitate the transition from brute-force retrieval to calibrated, efficient reasoning.

  </details>



- **Flight through Narrow Gaps with Morphing-Wing Drones**  
  Julius Wanner, Hoang-Vu Phan, Charbel Toumieh, Dario Floreano  
  _2026-03-12_ · https://arxiv.org/abs/2603.12059v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The size of a narrow gap traversable by a fixed-wing drone is limited by its wingspan. Inspired by birds, here, we enable the traversal of a gap of sub-wingspan width and height using a morphing-wing drone capable of temporarily sweeping in its wings mid-flight. This maneuver poses control challenges due to sudden lift loss during gap-passage at low flight speeds and the need for precisely timed wing-sweep actuation ahead of the gap. To address these challenges, we first develop an aerodynamic model for general wing-sweep morphing drone flight including low flight speeds and post-stall angles of attack. We integrate longitudinal drone dynamics into an optimal reference trajectory generation and Nonlinear Model Predictive Control framework with runtime adaptive costs and constraints. Validated on a 130 g wing-sweep-morphing drone, our method achieves an average altitude error of 5 cm during narrow-gap passage at forward speeds between 5 and 7 m/s, whilst enforcing fully swept wings near the gap across variable threshold distances. Trajectory analysis shows that the drone can compensate for lift loss during gap-passage by accelerating and pitching upwards ahead of the gap to an extent that differs between reference trajectory optimization objectives. We show that our strategy also allows for accurate gap passage on hardware whilst maintaining a constant forward flight speed reference and near-constant altitude.

  </details>



- **Modeling Trial-and-Error Navigation With a Sequential Decision Model of Information Scent**  
  Xiaofu Jin, Yunpeng Bai, Antti Oulasvirta  
  _2026-03-12_ · https://arxiv.org/abs/2603.11759v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Users often struggle to locate an item within an information architecture, particularly when links are ambiguous or deeply nested in hierarchies. Information scent has been used to explain why users select incorrect links, but this concept assumes that users see all available links before deciding. In practice, users frequently select a link too quickly, overlook relevant cues, and then rely on backtracking when errors occur. We extend the concept of information scent by framing navigation as a sequential decision-making problem under memory constraints. Specifically, we assume that users do not scan entire pages but instead inspect strategically, looking "just enough" to find the target given their time budget. To choose which item to inspect next, they consider both local (this page) and global (site) scent; however, both are constrained by memory. Trying to avoid wasting time, they occasionally choose the wrong links without inspecting everything on a page. Comparisons with empirical data show that our model replicates key navigation behaviors: premature selections, wrong turns, and recovery from backtracking. We conclude that trial-and-error behavior is well explained by information scent when accounting for the sequential and bounded characteristics of the navigation problem.

  </details>


