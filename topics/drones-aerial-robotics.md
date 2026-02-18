# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-18 07:14 UTC_

Total papers shown: **5**


---

- **SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms**  
  Haichao Liu, Yufeng Hu, Shuang Wang, Kangjun Guo, Jun Ma, Jinni Zhou  
  _2026-02-17_ · https://arxiv.org/abs/2602.15633v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous landing of Uncrewed Aerial Vehicles (UAVs) on oscillating marine platforms is severely constrained by wave-induced multi-frequency oscillations, wind disturbances, and prediction phase lags in motion prediction. Existing methods either treat platform motion as a general random process or lack explicit modeling of wave spectral characteristics, leading to suboptimal performance under dynamic sea conditions. To address these limitations, we propose SpecFuse: a novel spectral-temporal fusion predictive control framework that integrates frequency-domain wave decomposition with time-domain recursive state estimation for high-precision 6-DoF motion forecasting of Uncrewed Surface Vehicles (USVs). The framework explicitly models dominant wave harmonics to mitigate phase lags, refining predictions in real time via IMU data without relying on complex calibration. Additionally, we design a hierarchical control architecture featuring a sampling-based HPO-RRT* algorithm for dynamic trajectory planning under non-convex constraints and a learning-augmented predictive controller that fuses data-driven disturbance compensation with optimization-based execution. Extensive validations (2,000 simulations + 8 lake experiments) show our approach achieves a 3.2 cm prediction error, 4.46 cm landing deviation, 98.7% / 87.5% success rates (simulation / real-world), and 82 ms latency on embedded hardware, outperforming state-of-the-art methods by 44%-48% in accuracy. Its robustness to wave-wind coupling disturbances supports critical maritime missions such as search and rescue and environmental monitoring. All code, experimental configurations, and datasets will be released as open-source to facilitate reproducibility.

  </details>



- **Hybrid F' and ROS2 Architecture for Vision-Based Autonomous Flight: Design and Experimental Validation**  
  Abdelrahman Metwally, Monijesu James, Aleksey Fedoseev, Miguel Altamirano Cabrera, Dzmitry Tsetserukou, Andrey Somov  
  _2026-02-17_ · https://arxiv.org/abs/2602.15398v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous aerospace systems require architectures that balance deterministic real-time control with advanced perception capabilities. This paper presents an integrated system combining NASA's F' flight software framework with ROS2 middleware via Protocol Buffers bridging. We evaluate the architecture through a 32.25-minute indoor quadrotor flight test using vision-based navigation. The vision system achieved 87.19 Hz position estimation with 99.90\% data continuity and 11.47 ms mean latency, validating real-time performance requirements. All 15 ground commands executed successfully with 100 % success rate, demonstrating robust F'--PX4 integration. System resource utilization remained low (15.19 % CPU, 1,244 MB RAM) with zero stale telemetry messages, confirming efficient operation on embedded platforms. Results validate the feasibility of hybrid flight-software architectures combining certification-grade determinism with flexible autonomy for autonomous aerial vehicles.

  </details>



- **One Agent to Guide Them All: Empowering MLLMs for Vision-and-Language Navigation via Explicit World Representation**  
  Zerui Li, Hongpei Zheng, Fangguo Zhao, Aidan Chan, Jian Zhou, Sihao Lin, Shijie Li, Qi Wu  
  _2026-02-17_ · https://arxiv.org/abs/2602.15400v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  A navigable agent needs to understand both high-level semantic instructions and precise spatial perceptions. Building navigation agents centered on Multimodal Large Language Models (MLLMs) demonstrates a promising solution due to their powerful generalization ability. However, the current tightly coupled design dramatically limits system performance. In this work, we propose a decoupled design that separates low-level spatial state estimation from high-level semantic planning. Unlike previous methods that rely on predefined, oversimplified textual maps, we introduce an interactive metric world representation that maintains rich and consistent information, allowing MLLMs to interact with and reason on it for decision-making. Furthermore, counterfactual reasoning is introduced to further elicit MLLMs' capacity, while the metric world representation ensures the physical validity of the produced actions. We conduct comprehensive experiments in both simulated and real-world environments. Our method establishes a new zero-shot state-of-the-art, achieving 48.8\% Success Rate (SR) in R2R-CE and 42.2\% in RxR-CE benchmarks. Furthermore, to validate the versatility of our metric representation, we demonstrate zero-shot sim-to-real transfer across diverse embodiments, including a wheeled TurtleBot 4 and a custom-built aerial drone. These real-world deployments verify that our decoupled framework serves as a robust, domain-invariant interface for embodied Vision-and-Language navigation.

  </details>



- **Proactive Conversational Assistant for a Procedural Manual Task based on Audio and IMU**  
  Rehana Mahfuz, Yinyi Guo, Erik Visser, Phanidhar Chinchili  
  _2026-02-17_ · https://arxiv.org/abs/2602.15707v1 · `cs.MM`  
  <details><summary>Abstract</summary>

  Real-time conversational assistants for procedural tasks often depend on video input, which can be computationally expensive and compromise user privacy. For the first time, we propose a real-time conversational assistant that provides comprehensive guidance for a procedural task using only lightweight privacy-preserving modalities such as audio and IMU inputs from a user's wearable device to understand the context. This assistant proactively communicates step-by-step instructions to a user performing a furniture assembly task, and answers user questions. We construct a dataset containing conversations where the assistant guides the user in performing the task. On observing that an off-the-shelf language model is a very talkative assistant, we design a novel User Whim Agnostic (UWA) LoRA finetuning method which improves the model's ability to suppress less informative dialogues, while maintaining its tendency to communicate important instructions. This leads to >30% improvement in the F-score. Finetuning the model also results in a 16x speedup by eliminating the need to provide in-context examples in the prompt. We further describe how such an assistant is implemented on edge devices with no dependence on the cloud.

  </details>



- **Efficient Knowledge Transfer for Jump-Starting Control Policy Learning of Multirotors through Physics-Aware Neural Architectures**  
  Welf Rehberg, Mihir Kulkarni, Philipp Weiss, Kostas Alexis  
  _2026-02-17_ · https://arxiv.org/abs/2602.15533v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Efficiently training control policies for robots is a major challenge that can greatly benefit from utilizing knowledge gained from training similar systems through cross-embodiment knowledge transfer. In this work, we focus on accelerating policy training using a library-based initialization scheme that enables effective knowledge transfer across multirotor configurations. By leveraging a physics-aware neural control architecture that combines a reinforcement learning-based controller and a supervised control allocation network, we enable the reuse of previously trained policies. To this end, we utilize a policy evaluation-based similarity measure that identifies suitable policies for initialization from a library. We demonstrate that this measure correlates with the reduction in environment interactions needed to reach target performance and is therefore suited for initialization. Extensive simulation and real-world experiments confirm that our control architecture achieves state-of-the-art control performance, and that our initialization scheme saves on average up to $73.5\%$ of environment interactions (compared to training a policy from scratch) across diverse quadrotor and hexarotor designs, paving the way for efficient cross-embodiment transfer in reinforcement learning.

  </details>


