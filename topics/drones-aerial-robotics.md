# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-10 07:19 UTC_

Total papers shown: **3**


---

- **High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning**  
  Jiarui Zhang, Chengyong Lei, Chengjiang Dai, Lijie Wang, Zhichao Han, Fei Gao  
  _2026-02-09_ · https://arxiv.org/abs/2602.08653v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Quadrotor unmanned aerial vehicles (UAVs) are increasingly deployed in complex missions that demand reliable autonomous navigation and robust obstacle avoidance. However, traditional modular pipelines often incur cumulative latency, whereas purely reinforcement learning (RL) approaches typically provide limited formal safety guarantees. To bridge this gap, we propose an end-to-end RL framework augmented with model-based safety mechanisms. We incorporate physical priors in both training and deployment. During training, we design a physics-informed reward structure that provides global navigational guidance. During deployment, we integrate a real-time safety filter that projects the policy outputs onto a provably safe set to enforce strict collision-avoidance constraints. This hybrid architecture reconciles high-speed flight with robust safety assurances. Benchmark evaluations demonstrate that our method outperforms both traditional planners and recent end-to-end obstacle avoidance approaches based on differentiable physics. Extensive experiments demonstrate strong generalization, enabling reliable high-speed navigation in dense clutter and challenging outdoor forest environments at velocities up to 7.5m/s.

  </details>



- **Digital Twin and Agentic AI for Wild Fire Disaster Management: Intelligent Virtual Situation Room**  
  Mohammad Morsali, Siavash H. Khajavi  
  _2026-02-09_ · https://arxiv.org/abs/2602.08949v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  According to the United Nations, wildfire frequency and intensity are projected to increase by approximately 14% by 2030 and 30% by 2050 due to global warming, posing critical threats to life, infrastructure, and ecosystems. Conventional disaster management frameworks rely on static simulations and passive data acquisition, hindering their ability to adapt to arbitrarily evolving wildfire episodes in real-time. To address these limitations, we introduce the Intelligent Virtual Situation Room (IVSR), a bidirectional Digital Twin (DT) platform augmented by autonomous AI agents. The IVSR continuously ingests multisource sensor imagery, weather data, and 3D forest models to create a live virtual replica of the fire environment. A similarity engine powered by AI aligns emerging conditions with a precomputed Disaster Simulation Library, retrieving and calibrating intervention tactics under the watchful eyes of experts. Authorized action-ranging from UAV redeployment to crew reallocation-is cycled back through standardized procedures to the physical layer, completing the loop between response and analysis. We validate IVSR through detailed case-study simulations provided by an industrial partner, demonstrating capabilities in localized incident detection, privacy-preserving playback, collider-based fire-spread projection, and site-specific ML retraining. Our results indicate marked reductions in detection-to-intervention latency and more effective resource coordination versus traditional systems. By uniting real-time bidirectional DTs with agentic AI, IVSR offers a scalable, semi-automated decision-support paradigm for proactive, adaptive wildfire disaster management.

  </details>



- **A Precise Real-Time Force-Aware Grasping System for Robust Aerial Manipulation**  
  Kenghou Hoi, Yuze Wu, Annan Ding, Junjie Wang, Anke Zhao, Chengqian Zhang, Fei Gao  
  _2026-02-09_ · https://arxiv.org/abs/2602.08599v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Aerial manipulation requires force-aware capabilities to enable safe and effective grasping and physical interaction. Previous works often rely on heavy, expensive force sensors unsuitable for typical quadrotor platforms, or perform grasping without force feedback, risking damage to fragile objects. To address these limitations, we propose a novel force-aware grasping framework incorporating six low-cost, sensitive skin-like tactile sensors. We introduce a magnetic-based tactile sensing module that provides high-precision three-dimensional force measurements. We eliminate geomagnetic interference through a reference Hall sensor and simplify the calibration process compared to previous work. The proposed framework enables precise force-aware grasping control, allowing safe manipulation of fragile objects and real-time weight measurement of grasped items. The system is validated through comprehensive real-world experiments, including balloon grasping, dynamic load variation tests, and ablation studies, demonstrating its effectiveness in various aerial manipulation scenarios. Our approach achieves fully onboard operation without external motion capture systems, significantly enhancing the practicality of force-sensitive aerial manipulation. The supplementary video is available at: https://www.youtube.com/watch?v=mbcZkrJEf1I.

  </details>


