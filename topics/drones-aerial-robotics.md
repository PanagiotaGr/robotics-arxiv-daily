# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-04 07:06 UTC_

Total papers shown: **4**


---

- **Conformal Reachability for Safe Control in Unknown Environments**  
  Xinhang Ma, Junlin Wu, Yiannis Kantaros, Yevgeniy Vorobeychik  
  _2026-02-03_ · https://arxiv.org/abs/2602.03799v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Designing provably safe control is a core problem in trustworthy autonomy. However, most prior work in this regard assumes either that the system dynamics are known or deterministic, or that the state and action space are finite, significantly limiting application scope. We address this limitation by developing a probabilistic verification framework for unknown dynamical systems which combines conformal prediction with reachability analysis. In particular, we use conformal prediction to obtain valid uncertainty intervals for the unknown dynamics at each time step, with reachability then verifying whether safety is maintained within the conformal uncertainty bounds. Next, we develop an algorithmic approach for training control policies that optimize nominal reward while also maximizing the planning horizon with sound probabilistic safety guarantees. We evaluate the proposed approach in seven safe control settings spanning four domains -- cartpole, lane following, drone control, and safe navigation -- for both affine and nonlinear safety specifications. Our experiments show that the policies we learn achieve the strongest provable safety guarantees while still maintaining high average reward.

  </details>



- **HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic**  
  Yu-Hsiang Chen, Wei-Jer Chang, Christian Kotulla, Thomas Keutgens, Steffen Runde, Tobias Moers, Christoph Klas, Wei Zhan, Masayoshi Tomizuka, Yi-Ting Chen  
  _2026-02-03_ · https://arxiv.org/abs/2602.03447v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present HetroD, a dataset and benchmark for developing autonomous driving systems in heterogeneous environments. HetroD targets the critical challenge of navi- gating real-world heterogeneous traffic dominated by vulner- able road users (VRUs), including pedestrians, cyclists, and motorcyclists that interact with vehicles. These mixed agent types exhibit complex behaviors such as hook turns, lane splitting, and informal right-of-way negotiation. Such behaviors pose significant challenges for autonomous vehicles but remain underrepresented in existing datasets focused on structured, lane-disciplined traffic. To bridge the gap, we collect a large- scale drone-based dataset to provide a holistic observation of traffic scenes with centimeter-accurate annotations, HD maps, and traffic signal states. We further develop a modular toolkit for extracting per-agent scenarios to support downstream task development. In total, the dataset comprises over 65.4k high- fidelity agent trajectories, 70% of which are from VRUs. HetroD supports modeling of VRU behaviors in dense, het- erogeneous traffic and provides standardized benchmarks for forecasting, planning, and simulation tasks. Evaluation results reveal that state-of-the-art prediction and planning models struggle with the challenges presented by our dataset: they fail to predict lateral VRU movements, cannot handle unstructured maneuvers, and exhibit limited performance in dense and multi-agent scenarios, highlighting the need for more robust approaches to heterogeneous traffic. See our project page for more examples: https://hetroddata.github.io/HetroD/

  </details>



- **Digital-Twin Empowered Deep Reinforcement Learning For Site-Specific Radio Resource Management in NextG Wireless Aerial Corridor**  
  Pulok Tarafder, Zoheb Hassan, Imtiaz Ahmed, Danda B. Rawat, Kamrul Hasan, Cong Pu  
  _2026-02-03_ · https://arxiv.org/abs/2602.03801v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Joint base station (BS) association and beam selection in multi-UAV aerial corridors constitutes a challenging radio resource management (RRM) problem. It is driven by high-dimensional action spaces, need for substantial overhead to acquire global channel state information (CSI), rapidly varying propagation channels, and stringent latency requirements. Conventional combinatorial optimization methods, while near-optimal, are computationally prohibitive for real-time operation in such dynamic environments. While learning-based approaches can mitigate computational complexity and CSI overhead, the need for extensive site-specific (SS) datasets for model training remains a key challenge. To address these challenges, we develop a Digital Twin (DT)-enabled two-stage optimization framework that couples physics-based beam gain modeling with DRL for scalable online decision-making. In the first stage, a channel twin (CT) is constructed using a high-fidelity ray-tracing solver with geo-spatial contexts, and network information to capture SS propagation characteristics, and dual annealing algorithm is employed to precompute optimal transmission beam directions. In the second stage, a Multi-Head Proximal Policy Optimization (MH-PPO) agent, equipped with a scalable multi-head actor-critic architecture, is trained on the DT-generated channel dataset to directly map complex channel and beam states to jointly execute UAV-BS-beam association decisions. The proposed PPO agent achieves a 44%-121% improvement over DQN and 249%-807% gain over traditional heuristic based optimization schemes in a dense UAV scenario, while reducing inference latency by several orders of magnitude. These results demonstrate that DT-driven training pipelines can deliver high-performance, low-latency RRM policies tailored to SS deployments suitable for real-time resource management in next-generation aerial corridor networks.

  </details>



- **Input-to-State Safe Backstepping: Robust Safety-Critical Control with Unmatched Uncertainties**  
  Max H. Cohen, Pio Ong, Aaron D. Ames  
  _2026-02-03_ · https://arxiv.org/abs/2602.03691v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  Guaranteeing safety in the presence of unmatched disturbances -- uncertainties that cannot be directly canceled by the control input -- remains a key challenge in nonlinear control. This paper presents a constructive approach to safety-critical control of nonlinear systems with unmatched disturbances. We first present a generalization of the input-to-state safety (ISSf) framework for systems with these uncertainties using the recently developed notion of an Optimal Decay CBF, which provides more flexibility for satisfying the associated Lyapunov-like conditions for safety. From there, we outline a procedure for constructing ISSf-CBFs for two relevant classes of systems with unmatched uncertainties: i) strict-feedback systems; ii) dual-relative-degree systems, which are similar to differentially flat systems. Our theoretical results are illustrated via numerical simulations of an inverted pendulum and planar quadrotor.

  </details>


