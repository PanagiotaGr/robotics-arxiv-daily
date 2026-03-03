# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-03-03 07:06 UTC_

Total papers shown: **4**


---

- **Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control**  
  Peng Yu, Xin Wang, Ning Tan  
  _2026-03-02_ · https://arxiv.org/abs/2603.01751v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Continuum robots possess high flexibility and redundancy, making them well suited for safe interaction in complex environments, yet their continuous deformation and nonlinear dynamics pose fundamental challenges to perception, modeling, and control. Existing vision-based control approaches often rely on end-to-end learning, achieving shape regulation without explicit awareness of robot geometry or its interaction with the environment. Here, we introduce a shape-interpretable visual self-modeling framework for continuum robots that enables geometry-aware control. Robot shapes are encoded from multi-view planar images using a Bezier-curve representation, transforming visual observations into a compact and physically meaningful shape space that uniquely characterizes the robot's three-dimensional configuration. Based on this representation, neural ordinary differential equations are employed to self-model both shape and end-effector dynamics directly from data, enabling hybrid shape-position control without analytical models or dense body markers. The explicit geometric structure of the learned shape space allows the robot to reason about its body and surroundings, supporting environment-aware behaviors such as obstacle avoidance and self-motion while maintaining end-effector objectives. Experiments on a cable-driven continuum robot demonstrate accurate shape-position regulation and tracking, with shape errors within 1.56% of image resolution and end-effector errors within 2% of robot length, as well as robust performance in constrained environments. By elevating visual shape representations from two-dimensional observations to an interpretable three-dimensional self-model, this work establishes a principled alternative to vision-based end-to-end control and advances autonomous, geometry-aware manipulation for continuum robots.

  </details>



- **ACDC: Adaptive Curriculum Planning with Dynamic Contrastive Control for Goal-Conditioned Reinforcement Learning in Robotic Manipulation**  
  Xuerui Wang, Guangyu Ren, Tianhong Dai, Bintao Hu, Shuangyao Huang, Wenzhang Zhang, Hengyan Liu  
  _2026-03-02_ · https://arxiv.org/abs/2603.02104v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Goal-conditioned reinforcement learning has shown considerable potential in robotic manipulation; however, existing approaches remain limited by their reliance on prioritizing collected experience, resulting in suboptimal performance across diverse tasks. Inspired by human learning behaviors, we propose a more comprehensive learning paradigm, ACDC, which integrates multidimensional Adaptive Curriculum (AC) Planning with Dynamic Contrastive (DC) Control to guide the agent along a well-designed learning trajectory. More specifically, at the planning level, the AC component schedules the learning curriculum by dynamically balancing diversity-driven exploration and quality-driven exploitation based on the agent's success rate and training progress. At the control level, the DC component implements the curriculum plan through norm-constrained contrastive learning, enabling magnitude-guided experience selection aligned with the current curriculum focus. Extensive experiments on challenging robotic manipulation tasks demonstrate that ACDC consistently outperforms the state-of-the-art baselines in both sample efficiency and final task success rate.

  </details>



- **From Transportation to Manipulation: Transforming Magnetic Levitation to Magnetic Robotics**  
  Lara Bergmann, Noah Greis, Klaus Neumann  
  _2026-03-02_ · https://arxiv.org/abs/2603.01982v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Magnetic Levitation (MagLev) systems fundamentally increase the flexibility of in-machine material flow in industrial automation. Therefore, these systems enable dynamic throughput optimization, which is especially beneficial for high-mix low-volume manufacturing. Until now, MagLev installations have been used primarily for in-machine transport, while their potential for manipulation is largely unexplored. This paper introduces the 6D-Platform MagBot, a low-cost six degrees of freedom parallel kinematic that couples two movers into a composite robotic platform. Experiments show that the 6D-Platform MagBot achieves sub-millimeter positioning accuracy and supports fully autonomous pick up and drop off via a docking station, allowing rapid and repeatable reconfiguration of the machine. Relative to a single mover, the proposed platform substantially expands the reachable workspace, payload, and functional dexterity. By unifying transportation and manipulation, this work advances Magnetic Levitation towards Magnetic Robotics, enabling manufacturing solutions that are more agile, efficient, and adaptable.

  </details>



- **Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation**  
  Han Xue, Nan Min, Xiaotong Liu, Wendi Chen, Yuan Fang, Jun Lv, Cewu Lu, Chuan Wen  
  _2026-03-02_ · https://arxiv.org/abs/2603.02139v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The adoption of fisheye cameras in robotic manipulation, driven by their exceptionally wide Field of View (FoV), is rapidly outpacing a systematic understanding of their downstream effects on policy learning. This paper presents the first comprehensive empirical study to bridge this gap, rigorously analyzing the properties of wrist-mounted fisheye cameras for imitation learning. Through extensive experiments in both simulation and the real world, we investigate three critical research questions: spatial localization, scene generalization, and hardware generalization. Our investigation reveals that: (1) The wide FoV significantly enhances spatial localization, but this benefit is critically contingent on the visual complexity of the environment. (2) Fisheye-trained policies, while prone to overfitting in simple scenes, unlock superior scene generalization when trained with sufficient environmental diversity. (3) While naive cross-camera transfer leads to failures, we identify the root cause as scale overfitting and demonstrate that hardware generalization performance can be improved with a simple Random Scale Augmentation (RSA) strategy. Collectively, our findings provide concrete, actionable guidance for the large-scale collection and effective use of fisheye datasets in robotic learning. More results and videos are available on https://robo-fisheye.github.io/

  </details>


