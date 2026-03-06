# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-03-06 07:04 UTC_

Total papers shown: **3**


---

- **Fusion4CA: Boosting 3D Object Detection via Comprehensive Image Exploitation**  
  Kang Luo, Xin Chen, Yangyi Xiao, Hesheng Wang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05305v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Nowadays, an increasing number of works fuse LiDAR and RGB data in the bird's-eye view (BEV) space for 3D object detection in autonomous driving systems. However, existing methods suffer from over-reliance on the LiDAR branch, with insufficient exploration of RGB information. To tackle this issue, we propose Fusion4CA, which is built upon the classic BEVFusion framework and dedicated to fully exploiting visual input with plug-and-play components. Specifically, a contrastive alignment module is designed to calibrate image features with 3D geometry, and a camera auxiliary branch is introduced to mine RGB information sufficiently during training. For further performance enhancement, we leverage an off-the-shelf cognitive adapter to make the most of pretrained image weights, and integrate a standard coordinate attention module into the fusion stage as a supplementary boost. Experiments on the nuScenes dataset demonstrate that our method achieves 69.7% mAP with only 6 training epochs and a mere 3.48% increase in inference parameters, yielding a 1.2% improvement over the baseline which is fully trained for 20 epochs. Extensive experiments in a simulated lunar environment further validate the effectiveness and generalization of our method. Our code will be released through Fusion4CA.

  </details>



- **CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection**  
  Zhaonian Kuang, Rui Ding, Haotian Wang, Xinhu Zheng, Meng Yang, Gang Hua  
  _2026-03-05_ · https://arxiv.org/abs/2603.05042v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Multi-camera 3D object detection (MC3D) has attracted increasing attention with the growing deployment of multi-sensor physical agents, such as robots and autonomous vehicles. However, MC3D models still struggle to generalize to unseen platforms with new multi-camera configurations. Current solutions simply employ a meta-camera for unified representation but lack comprehensive consideration. In this paper, we revisit this issue and identify that the devil lies in spatial prior discrepancies across source and target configurations, including different intrinsics, extrinsics, and array layouts. To address this, we propose CoIn3D, a generalizable MC3D framework that enables strong transferability from source configurations to unseen target ones. CoIn3D explicitly incorporates all identified spatial priors into both feature embedding and image observation through spatial-aware feature modulation (SFM) and camera-aware data augmentation (CDA), respectively. SFM enriches feature space by integrating four spatial representations, such as focal length, ground depth, ground gradient, and Plücker coordinate. CDA improves observation diversity under various configurations via a training-free dynamic novel-view image synthesis scheme. Extensive experiments demonstrate that CoIn3D achieves strong cross-configuration performance on landmark datasets such as NuScenes, Waymo, and Lyft, under three dominant MC3D paradigms represented by BEVDepth, BEVFormer, and PETR.

  </details>



- **From Code to Road: A Vehicle-in-the-Loop and Digital Twin-Based Framework for Central Car Server Testing in Autonomous Driving**  
  Chengdong Wu, Sven Kirchner, Nils Purschke, Axel Torschmied, Norbert Kroth, Yinglei Song, André Schamschurko, Erik Leo Haß, Kuo-Yi Chao, Yi Zhang, et al.  
  _2026-03-05_ · https://arxiv.org/abs/2603.05279v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Simulation is one of the most essential parts in the development stage of automotive software. However, purely virtual simulations often struggle to accurately capture all real-world factors due to limitations in modeling. To address this challenge, this work presents a test framework for automotive software on the centralized E/E architecture, which is a central car server in our case, based on Vehicle-in-the-Loop (ViL) and digital twin technology. The framework couples a physical test vehicle on a dynamometer test bench with its synchronized virtual counterpart in a simulation environment. Our approach provides a safe, reproducible, realistic, and cost-effective platform for validating autonomous driving algorithms with a centralized architecture. This test method eliminates the need to test individual physical ECUs and their communication protocols separately. In contrast to traditional ViL methods, the proposed framework runs the full autonomous driving software directly on the vehicle hardware after the simulation process, eliminating flashing and intermediate layers while enabling seamless virtual-physical integration and accurately reflecting centralized E/E behavior. In addition, incorporating mixed testing in both simulated and physical environments reduces the need for full hardware integration during the early stages of automotive development. Experimental case studies demonstrate the effectiveness of the framework in different test scenarios. These findings highlight the potential to reduce development and integration efforts for testing autonomous driving pipelines in the future.

  </details>


