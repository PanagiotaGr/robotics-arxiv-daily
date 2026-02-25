# 3D Gaussian Splatting & Neural Rendering (Robotics)

_Robotics arXiv Daily_

_Updated: 2026-02-25 07:14 UTC_

Total papers shown: **4**


---

- **RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction**  
  Yangfan Zhao, Hanwei Zhang, Ke Huang, Qiufeng Wang, Zhenzhou Shao, Dengyu Wu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20807v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: https://ru4d-slam.github.io

  </details>



- **Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones**  
  Rong Zou, Marco Cannici, Davide Scaramuzza  
  _2026-02-24_ · https://arxiv.org/abs/2602.21101v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.

  </details>



- **Monocular Endoscopic Tissue 3D Reconstruction with Multi-Level Geometry Regularization**  
  Yangsen Chen, Hao Wang  
  _2026-02-24_ · https://arxiv.org/abs/2602.20718v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Reconstructing deformable endoscopic tissues is crucial for achieving robot-assisted surgery. However, 3D Gaussian Splatting-based approaches encounter challenges in achieving consistent tissue surface reconstruction, while existing NeRF-based methods lack real-time rendering capabilities. In pursuit of both smooth deformable surfaces and real-time rendering, we introduce a novel approach based on 3D Gaussian Splatting. Specifically, we introduce surface-aware reconstruction, initially employing a Sign Distance Field-based method to construct a mesh, subsequently utilizing this mesh to constrain the Gaussian Splatting reconstruction process. Furthermore, to ensure the generation of physically plausible deformations, we incorporate local rigidity and global non-rigidity restrictions to guide Gaussian deformation, tailored for the highly deformable nature of soft endoscopic tissue. Based on 3D Gaussian Splatting, our proposed method delivers a fast rendering process and smooth surface appearances. Quantitative and qualitative analysis against alternative methodologies shows that our approach achieves solid reconstruction quality in both textures and geometries.

  </details>



- **UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling**  
  Kaiyuan Tan, Yingying Shen, Mingfei Tu, Haohui Zhu, Bing Wang, Guang Chen, Hangjun Ye, Haiyang Sun  
  _2026-02-24_ · https://arxiv.org/abs/2602.20943v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Dynamic driving scene reconstruction is critical for autonomous driving simulation and closed-loop learning. While recent feed-forward methods have shown promise for 3D reconstruction, they struggle with long-range driving sequences due to quadratic complexity in sequence length and challenges in modeling dynamic objects over extended durations. We propose UFO, a novel recurrent paradigm that combines the benefits of optimization-based and feed-forward methods for efficient long-range 4D reconstruction. Our approach maintains a 4D scene representation that is iteratively refined as new observations arrive, using a visibility-based filtering mechanism to select informative scene tokens and enable efficient processing of long sequences. For dynamic objects, we introduce an object pose-guided modeling approach that supports accurate long-range motion capture. Experiments on the Waymo Open Dataset demonstrate that our method significantly outperforms both per-scene optimization and existing feed-forward methods across various sequence lengths. Notably, our approach can reconstruct 16-second driving logs within 0.5 second while maintaining superior visual quality and geometric accuracy.

  </details>


