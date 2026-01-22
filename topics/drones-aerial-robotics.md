# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-01-22 06:51 UTC_

Total papers shown: **5**


---

- **MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI**  
  Stavrow A. Bahnam, Robin Ferede, Till M. Blaha, Anton E. Lang, Erin Lucassen, Quentin Missinne, Aderik E. C. Verraest, Christophe De Wagter, Guido C. H. E. de Croon  
  _2026-01-21_ · https://arxiv.org/abs/2601.15222v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous drone racing represents a major frontier in robotics research. It requires an Artificial Intelligence (AI) that can run on board light-weight flying robots under tight resource and time constraints, while pushing the physical system to its limits. The state of the art in this area consists of a system with a stereo camera and an inertial measurement unit (IMU) that beat human drone racing champions in a controlled indoor environment. Here, we present MonoRace: an onboard drone racing approach that uses a monocular, rolling-shutter camera and IMU that generalizes to a competition environment without any external motion tracking system. The approach features robust state estimation that combines neural-network-based gate segmentation with a drone model. Moreover, it includes an offline optimization procedure that leverages the known geometry of gates to refine any state estimation parameter. This offline optimization is based purely on onboard flight data and is important for fine-tuning the vital external camera calibration parameters. Furthermore, the guidance and control are performed by a neural network that foregoes inner loop controllers by directly sending motor commands. This small network runs on the flight controller at 500Hz. The proposed approach won the 2025 Abu Dhabi Autonomous Drone Racing Competition (A2RL), outperforming all competing AI teams and three human world champion pilots in a direct knockout tournament. It set a new milestone in autonomous drone racing research, reaching speeds up to 100 km/h on the competition track and successfully coping with problems such as camera interference and IMU saturation.

  </details>



- **HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV**  
  Faryal Batool, Iana Zhura, Valerii Serpiva, Roohan Ahmed Khan, Ivan Valuev, Issatay Tokmurziyev, Dzmitry Tsetserukou  
  _2026-01-21_ · https://arxiv.org/abs/2601.14973v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Reliable human--robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments. This paper presents HumanDiffusion, a lightweight image-conditioned diffusion planner that generates human-aware navigation trajectories directly from RGB imagery. The system combines YOLO-11--based human detection with diffusion-driven trajectory generation, enabling a quadrotor to approach a target person and deliver medical assistance without relying on prior maps or computationally intensive planning pipelines. Trajectories are predicted in pixel space, ensuring smooth motion and a consistent safety margin around humans. We evaluate HumanDiffusion in simulation and real-world indoor mock-disaster scenarios. On a 300-sample test set, the model achieves a mean squared error of 0.02 in pixel-space trajectory reconstruction. Real-world experiments demonstrate an overall mission success rate of 80% across accident-response and search-and-locate tasks with partial occlusions. These results indicate that human-conditioned diffusion planning offers a practical and robust solution for human-aware UAV navigation in time-critical assistance settings.

  </details>



- **Integrated Sensing, Communication and Control enabled Agile UAV Swarm**  
  Zhiqing Wei, Yucong Du, Zhiyong Feng, Haotian Liu, Yanpeng Cui, Tao Zhang, Ying Zhou, Huici Wu  
  _2026-01-21_ · https://arxiv.org/abs/2601.14783v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Uncrewed aerial vehicle (UAV) swarms are pivotal in the applications such as disaster relief, aerial base station (BS) and logistics transportation. These scenarios require the capabilities in accurate sensing, efficient communication and flexible control for real-time and reliable task execution. However, sensing, communication and control are studied independently in traditional research, which limits the overall performance of UAV swarms. To overcome this disadvantage, we propose a deeply coupled scheme of integrated sensing, communication and control (ISCC) for UAV swarms, which is a systemic paradigm that transcends traditional isolated designs of sensing, communication and control by establishing a tightly-coupled closed-loop through the co-optimization of sensing, communication and control. In this article, we firstly analyze the requirements of scenarios and key performance metrics. Subsequently, the enabling technologies are proposed, including communication-and-control-enhanced sensing, sensing-and-control-enhanced communication, and sensing-and-communication-enhanced control. Simulation results validate the performance of the proposed ISCC framework, demonstrating its application potential in the future.

  </details>



- **From Observation to Prediction: LSTM for Vehicle Lane Change Forecasting on Highway On/Off-Ramps**  
  Mohamed Abouras, Catherine M. Elias  
  _2026-01-21_ · https://arxiv.org/abs/2601.14848v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  On and off-ramps are understudied road sections even though they introduce a higher level of variation in highway interactions. Predicting vehicles' behavior in these areas can decrease the impact of uncertainty and increase road safety. In this paper, the difference between this Area of Interest (AoI) and a straight highway section is studied. Multi-layered LSTM architecture to train the AoI model with ExiD drone dataset is utilized. In the process, different prediction horizons and different models' workflow are tested. The results show great promise on horizons up to 4 seconds with prediction accuracy starting from about 76% for the AoI and 94% for the general highway scenarios on the maximum horizon.

  </details>



- **SimD3: A Synthetic drone Dataset with Payload and Bird Distractor Modeling for Robust Detection**  
  Ami Pandat, Kanyala Muvva, Punna Rajasekhar, Gopika Vinod, Rohit Shukla  
  _2026-01-21_ · https://arxiv.org/abs/2601.14742v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Reliable drone detection is challenging due to limited annotated real-world data, large appearance variability, and the presence of visually similar distractors such as birds. To address these challenges, this paper introduces SimD3, a large-scale high-fidelity synthetic dataset designed for robust drone detection in complex aerial environments. Unlike existing synthetic drone datasets, SimD3 explicitly models drones with heterogeneous payloads, incorporates multiple bird species as realistic distractors, and leverages diverse Unreal Engine 5 environments with controlled weather, lighting, and flight trajectories captured using a 360 six-camera rig. Using SimD3, we conduct an extensive experimental evaluation within the YOLOv5 detection framework, including an attention-enhanced variant termed Yolov5m+C3b, where standard bottleneck-based C3 blocks are replaced with C3b modules. Models are evaluated on synthetic data, combined synthetic and real data, and multiple unseen real-world benchmarks to assess robustness and generalization. Experimental results show that SimD3 provides effective supervision for small-object drone detection and that Yolov5m+C3b consistently outperforms the baseline across in-domain and cross-dataset evaluations. These findings highlight the utility of SimD3 for training and benchmarking robust drone detection models under diverse and challenging conditions.

  </details>


