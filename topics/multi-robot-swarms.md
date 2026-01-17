# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-01-17 06:44 UTC_

Total papers shown: **3**


---

- **Terrain-Adaptive Mobile 3D Printing with Hierarchical Control**  
  Shuangshan Nors Li, J. Nathan Kutz  
  _2026-01-15_ · https://arxiv.org/abs/2601.10208v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Mobile 3D printing on unstructured terrain remains challenging due to the conflict between platform mobility and deposition precision. Existing gantry-based systems achieve high accuracy but lack mobility, while mobile platforms struggle to maintain print quality on uneven ground. We present a framework that tightly integrates AI-driven disturbance prediction with multi-modal sensor fusion and hierarchical hardware control, forming a closed-loop perception-learning-actuation system. The AI module learns terrain-to-perturbation mappings from IMU, vision, and depth sensors, enabling proactive compensation rather than reactive correction. This intelligence is embedded into a three-layer control architecture: path planning, predictive chassis-manipulator coordination, and precision hardware execution. Through outdoor experiments on terrain with slopes and surface irregularities, we demonstrate sub-centimeter printing accuracy while maintaining full platform mobility. This AI-hardware integration establishes a practical foundation for autonomous construction in unstructured environments.

  </details>



- **CHORAL: Traversal-Aware Planning for Safe and Efficient Heterogeneous Multi-Robot Routing**  
  David Morilla-Cabello, Eduardo Montijano  
  _2026-01-15_ · https://arxiv.org/abs/2601.10340v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Monitoring large, unknown, and complex environments with autonomous robots poses significant navigation challenges, where deploying teams of heterogeneous robots with complementary capabilities can substantially improve both mission performance and feasibility. However, effectively modeling how different robotic platforms interact with the environment requires rich, semantic scene understanding. Despite this, existing approaches often assume homogeneous robot teams or focus on discrete task compatibility rather than continuous routing. Consequently, scene understanding is not fully integrated into routing decisions, limiting their ability to adapt to the environment and to leverage each robot's strengths. In this paper, we propose an integrated semantic-aware framework for coordinating heterogeneous robots. Starting from a reconnaissance flight, we build a metric-semantic map using open-vocabulary vision models and use it to identify regions requiring closer inspection and capability-aware paths for each platform to reach them. These are then incorporated into a heterogeneous vehicle routing formulation that jointly assigns inspection tasks and computes robot trajectories. Experiments in simulation and in a real inspection mission with three robotic platforms demonstrate the effectiveness of our approach in planning safer and more efficient routes by explicitly accounting for each platform's navigation capabilities. We release our framework, CHORAL, as open source to support reproducibility and deployment of diverse robot teams.

  </details>



- **CoCoPlan: Adaptive Coordination and Communication for Multi-robot Systems in Dynamic and Unknown Environments**  
  Xintong Zhang, Junfeng Chen, Yuxiao Zhu, Bing Luo, Meng Guo  
  _2026-01-15_ · https://arxiv.org/abs/2601.10116v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-robot systems can greatly enhance efficiency through coordination and collaboration, yet in practice, full-time communication is rarely available and interactions are constrained to close-range exchanges. Existing methods either maintain all-time connectivity, rely on fixed schedules, or adopt pairwise protocols, but none adapt effectively to dynamic spatio-temporal task distributions under limited communication, resulting in suboptimal coordination. To address this gap, we propose CoCoPlan, a unified framework that co-optimizes collaborative task planning and team-wise intermittent communication. Our approach integrates a branch-and-bound architecture that jointly encodes task assignments and communication events, an adaptive objective function that balances task efficiency against communication latency, and a communication event optimization module that strategically determines when, where and how the global connectivity should be re-established. Extensive experiments demonstrate that it outperforms state-of-the-art methods by achieving a 22.4% higher task completion rate, reducing communication overhead by 58.6%, and improving the scalability by supporting up to 100 robots in dynamic environments. Hardware experiments include the complex 2D office environment and large-scale 3D disaster-response scenario.

  </details>


