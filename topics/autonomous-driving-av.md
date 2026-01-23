# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-01-23 06:51 UTC_

Total papers shown: **3**


---

- **EVolSplat4D: Efficient Volume-based Gaussian Splatting for 4D Urban Scene Synthesis**  
  Sheng Miao, Sijin Li, Pan Wang, Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger, Yiyi Liao  
  _2026-01-22_ · https://arxiv.org/abs/2601.15951v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Novel view synthesis (NVS) of static and dynamic urban scenes is essential for autonomous driving simulation, yet existing methods often struggle to balance reconstruction time with quality. While state-of-the-art neural radiance fields and 3D Gaussian Splatting approaches achieve photorealism, they often rely on time-consuming per-scene optimization. Conversely, emerging feed-forward methods frequently adopt per-pixel Gaussian representations, which lead to 3D inconsistencies when aggregating multi-view predictions in complex, dynamic environments. We propose EvolSplat4D, a feed-forward framework that moves beyond existing per-pixel paradigms by unifying volume-based and pixel-based Gaussian prediction across three specialized branches. For close-range static regions, we predict consistent geometry of 3D Gaussians over multiple frames directly from a 3D feature volume, complemented by a semantically-enhanced image-based rendering module for predicting their appearance. For dynamic actors, we utilize object-centric canonical spaces and a motion-adjusted rendering module to aggregate temporal features, ensuring stable 4D reconstruction despite noisy motion priors. Far-Field scenery is handled by an efficient per-pixel Gaussian branch to ensure full-scene coverage. Experimental results on the KITTI-360, KITTI, Waymo, and PandaSet datasets show that EvolSplat4D reconstructs both static and dynamic environments with superior accuracy and consistency, outperforming both per-scene optimization and state-of-the-art feed-forward baselines.

  </details>



- **DualShield: Safe Model Predictive Diffusion via Reachability Analysis for Interactive Autonomous Driving**  
  Rui Yang, Lei Zheng, Ruoyu Yao, Jun Ma  
  _2026-01-22_ · https://arxiv.org/abs/2601.15729v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Diffusion models have emerged as a powerful approach for multimodal motion planning in autonomous driving. However, their practical deployment is typically hindered by the inherent difficulty in enforcing vehicle dynamics and a critical reliance on accurate predictions of other agents, making them prone to safety issues under uncertain interactions. To address these limitations, we introduce DualShield, a planning and control framework that leverages Hamilton-Jacobi (HJ) reachability value functions in a dual capacity. First, the value functions act as proactive guidance, steering the diffusion denoising process towards safe and dynamically feasible regions. Second, they form a reactive safety shield using control barrier-value functions (CBVFs) to modify the executed actions and ensure safety. This dual mechanism preserves the rich exploration capabilities of diffusion models while providing principled safety assurance under uncertain and even adversarial interactions. Simulations in challenging unprotected U-turn scenarios demonstrate that DualShield significantly improves both safety and task efficiency compared to leading methods from different planning paradigms under uncertainty.

  </details>



- **SuperOcc: Toward Cohesive Temporal Modeling for Superquadric-based Occupancy Prediction**  
  Zichen Yu, Quanli Liu, Wei Wang, Liyong Zhang, Xiaoguang Zhao  
  _2026-01-22_ · https://arxiv.org/abs/2601.15644v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  3D occupancy prediction plays a pivotal role in the realm of autonomous driving, as it provides a comprehensive understanding of the driving environment. Most existing methods construct dense scene representations for occupancy prediction, overlooking the inherent sparsity of real-world driving scenes. Recently, 3D superquadric representation has emerged as a promising sparse alternative to dense scene representations due to the strong geometric expressiveness of superquadrics. However, existing superquadric frameworks still suffer from insufficient temporal modeling, a challenging trade-off between query sparsity and geometric expressiveness, and inefficient superquadric-to-voxel splatting. To address these issues, we propose SuperOcc, a novel framework for superquadric-based 3D occupancy prediction. SuperOcc incorporates three key designs: (1) a cohesive temporal modeling mechanism to simultaneously exploit view-centric and object-centric temporal cues; (2) a multi-superquadric decoding strategy to enhance geometric expressiveness without sacrificing query sparsity; and (3) an efficient superquadric-to-voxel splatting scheme to improve computational efficiency. Extensive experiments on the SurroundOcc and Occ3D benchmarks demonstrate that SuperOcc achieves state-of-the-art performance while maintaining superior efficiency. The code is available at https://github.com/Yzichen/SuperOcc.

  </details>


