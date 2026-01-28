# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-01-28 06:52 UTC_

Total papers shown: **3**


---

- **Judgelight: Trajectory-Level Post-Optimization for Multi-Agent Path Finding via Closed-Subwalk Collapsing**  
  Yimin Tang, Sven Koenig, Erdem Bıyık  
  _2026-01-27_ · https://arxiv.org/abs/2601.19388v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Multi-Agent Path Finding (MAPF) is an NP-hard problem with applications in warehouse automation and multi-robot coordination. Learning-based MAPF solvers offer fast and scalable planning but often produce feasible trajectories that contain unnecessary or oscillatory movements. We propose Judgelight, a post-optimization method that improves trajectory quality after a MAPF solver generates a feasible schedule. Judgelight collapses closed subwalks in agents' trajectories to remove redundant movements while preserving all feasibility constraints. We formalize this process as MAPF-Collapse, prove that it is NP-hard, and present an exact optimization approach by formulating it as integer linear programming (ILP) problem. Experimental results show Judgelight consistently reduces solution cost by around 20%, particularly for learning-based solvers, producing trajectories that are better suited for real-world deployment.

  </details>



- **Tactile Memory with Soft Robot: Robust Object Insertion via Masked Encoding and Soft Wrist**  
  Tatsuya Kamijo, Mai Nishimura, Cristian C. Beltran-Hernandez, Nodoka Shibasaki, Masashi Hamaya  
  _2026-01-27_ · https://arxiv.org/abs/2601.19275v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Tactile memory, the ability to store and retrieve touch-based experience, is critical for contact-rich tasks such as key insertion under uncertainty. To replicate this capability, we introduce Tactile Memory with Soft Robot (TaMeSo-bot), a system that integrates a soft wrist with tactile retrieval-based control to enable safe and robust manipulation. The soft wrist allows safe contact exploration during data collection, while tactile memory reuses past demonstrations via retrieval for flexible adaptation to unseen scenarios. The core of this system is the Masked Tactile Trajectory Transformer (MAT$^\text{3}$), which jointly models spatiotemporal interactions between robot actions, distributed tactile feedback, force-torque measurements, and proprioceptive signals. Through masked-token prediction, MAT$^\text{3}$ learns rich spatiotemporal representations by inferring missing sensory information from context, autonomously extracting task-relevant features without explicit subtask segmentation. We validate our approach on peg-in-hole tasks with diverse pegs and conditions in real-robot experiments. Our extensive evaluation demonstrates that MAT$^\text{3}$ achieves higher success rates than the baselines over all conditions and shows remarkable capability to adapt to unseen pegs and conditions.

  </details>



- **Information-Theoretic Detection of Bimanual Interactions for Dual-Arm Robot Plan Generation**  
  Elena Merlo, Marta Lagomarsino, Arash Ajoudani  
  _2026-01-27_ · https://arxiv.org/abs/2601.19832v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Programming by demonstration is a strategy to simplify the robot programming process for non-experts via human demonstrations. However, its adoption for bimanual tasks is an underexplored problem due to the complexity of hand coordination, which also hinders data recording. This paper presents a novel one-shot method for processing a single RGB video of a bimanual task demonstration to generate an execution plan for a dual-arm robotic system. To detect hand coordination policies, we apply Shannon's information theory to analyze the information flow between scene elements and leverage scene graph properties. The generated plan is a modular behavior tree that assumes different structures based on the desired arms coordination. We validated the effectiveness of this framework through multiple subject video demonstrations, which we collected and made open-source, and exploiting data from an external, publicly available dataset. Comparisons with existing methods revealed significant improvements in generating a centralized execution plan for coordinating two-arm systems.

  </details>


