# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-06 07:08 UTC_

Total papers shown: **5**


---

- **A Hybrid Autoencoder for Robust Heightmap Generation from Fused Lidar and Depth Data for Humanoid Robot Locomotion**  
  Dennis Bank, Joost Cordes, Thomas Seel, Simon F. G. Ehlers  
  _2026-02-05_ · https://arxiv.org/abs/2602.05855v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable terrain perception is a critical prerequisite for the deployment of humanoid robots in unstructured, human-centric environments. While traditional systems often rely on manually engineered, single-sensor pipelines, this paper presents a learning-based framework that uses an intermediate, robot-centric heightmap representation. A hybrid Encoder-Decoder Structure (EDS) is introduced, utilizing a Convolutional Neural Network (CNN) for spatial feature extraction fused with a Gated Recurrent Unit (GRU) core for temporal consistency. The architecture integrates multimodal data from an Intel RealSense depth camera, a LIVOX MID-360 LiDAR processed via efficient spherical projection, and an onboard IMU. Quantitative results demonstrate that multimodal fusion improves reconstruction accuracy by 7.2% over depth-only and 9.9% over LiDAR-only configurations. Furthermore, the integration of a 3.2 s temporal context reduces mapping drift.

  </details>



- **From Vision to Decision: Neuromorphic Control for Autonomous Navigation and Tracking**  
  Chuwei Wang, Eduardo Sebastián, Amanda Prorok, Anastasia Bizyaeva  
  _2026-02-05_ · https://arxiv.org/abs/2602.05683v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic navigation has historically struggled to reconcile reactive, sensor-based control with the decisive capabilities of model-based planners. This duality becomes critical when the absence of a predominant option among goals leads to indecision, challenging reactive systems to break symmetries without computationally-intense planners. We propose a parsimonious neuromorphic control framework that bridges this gap for vision-guided navigation and tracking. Image pixels from an onboard camera are encoded as inputs to dynamic neuronal populations that directly transform visual target excitation into egocentric motion commands. A dynamic bifurcation mechanism resolves indecision by delaying commitment until a critical point induced by the environmental geometry. Inspired by recently proposed mechanistic models of animal cognition and opinion dynamics, the neuromorphic controller provides real-time autonomy with a minimal computational burden, a small number of interpretable parameters, and can be seamlessly integrated with application-specific image processing pipelines. We validate our approach in simulation environments as well as on an experimental quadrotor platform.

  </details>



- **UAV Trajectory Optimization via Improved Noisy Deep Q-Network**  
  Zhang Hengyu, Maryam Cheraghy, Liu Wei, Armin Farhadi, Meysam Soltanpour, Zhong Zhuoqing  
  _2026-02-05_ · https://arxiv.org/abs/2602.05644v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  This paper proposes an Improved Noisy Deep Q-Network (Noisy DQN) to enhance the exploration and stability of Unmanned Aerial Vehicle (UAV) when applying deep reinforcement learning in simulated environments. This method enhances the exploration ability by combining the residual NoisyLinear layer with an adaptive noise scheduling mechanism, while improving training stability through smooth loss and soft target network updates. Experiments show that the proposed model achieves faster convergence and up to $+40$ higher rewards compared to standard DQN and quickly reach to the minimum number of steps required for the task 28 in the 15 * 15 grid navigation environment set up. The results show that our comprehensive improvements to the network structure of NoisyNet, exploration control, and training stability contribute to enhancing the efficiency and reliability of deep Q-learning.

  </details>



- **Reactive Knowledge Representation and Asynchronous Reasoning**  
  Simon Kohaut, Benedict Flade, Julian Eggert, Kristian Kersting, Devendra Singh Dhami  
  _2026-02-05_ · https://arxiv.org/abs/2602.05625v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Exact inference in complex probabilistic models often incurs prohibitive computational costs. This challenge is particularly acute for autonomous agents in dynamic environments that require frequent, real-time belief updates. Existing methods are often inefficient for ongoing reasoning, as they re-evaluate the entire model upon any change, failing to exploit that real-world information streams have heterogeneous update rates. To address this, we approach the problem from a reactive, asynchronous, probabilistic reasoning perspective. We first introduce Resin (Reactive Signal Inference), a probabilistic programming language that merges probabilistic logic with reactive programming. Furthermore, to provide efficient and exact semantics for Resin, we propose Reactive Circuits (RCs). Formulated as a meta-structure over Algebraic Circuits and asynchronous data streams, RCs are time-dynamic Directed Acyclic Graphs that autonomously adapt themselves based on the volatility of input signals. In high-fidelity drone swarm simulations, our approach achieves several orders of magnitude of speedup over frequency-agnostic inference. We demonstrate that RCs' structural adaptations successfully capture environmental dynamics, significantly reducing latency and facilitating reactive real-time reasoning. By partitioning computations based on the estimated Frequency of Change in the asynchronous inputs, large inference tasks can be decomposed into individually memoized sub-problems. This ensures that only the specific components of a model affected by new information are re-evaluated, drastically reducing redundant computation in streaming contexts.

  </details>



- **From Bench to Flight: Translating Drone Impact Tests into Operational Safety Limits**  
  Aziz Mohamed Mili, Louis Catar, Paul Gérard, Ilyass Tabiai, David St-Onge  
  _2026-02-05_ · https://arxiv.org/abs/2602.05922v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Indoor micro-aerial vehicles (MAVs) are increasingly used for tasks that require close proximity to people, yet practitioners lack practical methods to tune motion limits based on measured impact risk. We present an end-to-end, open toolchain that converts benchtop impact tests into deployable safety governors for drones. First, we describe a compact and replicable impact rig and protocol for capturing force-time profiles across drone classes and contact surfaces. Second, we provide data-driven models that map pre-impact speed to impulse and contact duration, enabling direct computation of speed bounds for a target force limit. Third, we release scripts and a ROS2 node that enforce these bounds online and log compliance, with support for facility-specific policies. We validate the workflow on multiple commercial off-the-shelf quadrotors and representative indoor assets, demonstrating that the derived governors preserve task throughput while meeting force constraints specified by safety stakeholders. Our contribution is a practical bridge from measured impacts to runtime limits, with shareable datasets, code, and a repeatable process that teams can adopt to certify indoor MAV operations near humans.

  </details>


