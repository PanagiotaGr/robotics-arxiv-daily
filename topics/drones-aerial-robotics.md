# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-03-14 07:02 UTC_

Total papers shown: **3**


---

- **HumDex:Humanoid Dexterous Manipulation Made Easy**  
  Liang Heng, Yihe Tang, Jiajun Xu, Henghui Bao, Di Huang, Yue Wang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12260v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper investigates humanoid whole-body dexterous manipulation, where the efficient collection of high-quality demonstration data remains a central bottleneck. Existing teleoperation systems often suffer from limited portability, occlusion, or insufficient precision, which hinders their applicability to complex whole-body tasks. To address these challenges, we introduce HumDex, a portable teleoperation system designed for humanoid whole-body dexterous manipulation. Our system leverages IMU-based motion tracking to address the portability-precision trade-off, enabling accurate full-body tracking while remaining easy to deploy. For dexterous hand control, we further introduce a learning-based retargeting method that generates smooth and natural hand motions without manual parameter tuning. Beyond teleoperation, HumDex enables efficient collection of human motion data. Building on this capability, we propose a two-stage imitation learning framework that first pre-trains on diverse human motion data to learn generalizable priors, and then fine-tunes on robot data to bridge the embodiment gap for precise execution. We demonstrate that this approach significantly improves generalization to new configurations, objects, and backgrounds with minimal data acquisition costs. The entire system is fully reproducible and open-sourced at https://github.com/physical-superintelligence-lab/HumDex.

  </details>



- **Flight through Narrow Gaps with Morphing-Wing Drones**  
  Julius Wanner, Hoang-Vu Phan, Charbel Toumieh, Dario Floreano  
  _2026-03-12_ · https://arxiv.org/abs/2603.12059v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The size of a narrow gap traversable by a fixed-wing drone is limited by its wingspan. Inspired by birds, here, we enable the traversal of a gap of sub-wingspan width and height using a morphing-wing drone capable of temporarily sweeping in its wings mid-flight. This maneuver poses control challenges due to sudden lift loss during gap-passage at low flight speeds and the need for precisely timed wing-sweep actuation ahead of the gap. To address these challenges, we first develop an aerodynamic model for general wing-sweep morphing drone flight including low flight speeds and post-stall angles of attack. We integrate longitudinal drone dynamics into an optimal reference trajectory generation and Nonlinear Model Predictive Control framework with runtime adaptive costs and constraints. Validated on a 130 g wing-sweep-morphing drone, our method achieves an average altitude error of 5 cm during narrow-gap passage at forward speeds between 5 and 7 m/s, whilst enforcing fully swept wings near the gap across variable threshold distances. Trajectory analysis shows that the drone can compensate for lift loss during gap-passage by accelerating and pitching upwards ahead of the gap to an extent that differs between reference trajectory optimization objectives. We show that our strategy also allows for accurate gap passage on hardware whilst maintaining a constant forward flight speed reference and near-constant altitude.

  </details>



- **HiSync: Spatio-Temporally Aligning Hand Motion from Wearable IMU and On-Robot Camera for Command Source Identification in Long-Range HRI**  
  Chengwen Zhang, Chun Yu, Borong Zhuang, Haopeng Jin, Qingyang Wan, Zhuojun Li, Zhe He, Zhoutong Ye, Yu Mei, Chang Liu, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.11809v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Long-range Human-Robot Interaction (HRI) remains underexplored. Within it, Command Source Identification (CSI) - determining who issued a command - is especially challenging due to multi-user and distance-induced sensor ambiguity. We introduce HiSync, an optical-inertial fusion framework that treats hand motion as binding cues by aligning robot-mounted camera optical flow with hand-worn IMU signals. We first elicit a user-defined (N=12) gesture set and collect a multimodal command gesture dataset (N=38) in long-range multi-user HRI scenarios. Next, HiSync extracts frequency-domain hand motion features from both camera and IMU data, and a learned CSINet denoises IMU readings, temporally aligns modalities, and performs distance-aware multi-window fusion to compute cross-modal similarity of subtle, natural gestures, enabling robust CSI. In three-person scenes up to 34m, HiSync achieves 92.32% CSI accuracy, outperforming the prior SOTA by 48.44%. HiSync is also validated on real-robot deployment. By making CSI reliable and natural, HiSync provides a practical primitive and design guidance for public-space HRI.

  </details>


