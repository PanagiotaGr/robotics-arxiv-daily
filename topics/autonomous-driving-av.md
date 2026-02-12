# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-12 07:15 UTC_

Total papers shown: **8**


---

- **ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving**  
  Jinqing Zhang, Zehua Fu, Zelin Xu, Wenying Dai, Qingjie Liu, Yunhong Wang  
  _2026-02-11_ · https://arxiv.org/abs/2602.10884v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The comprehensive understanding capabilities of world models for driving scenarios have significantly improved the planning accuracy of end-to-end autonomous driving frameworks. However, the redundant modeling of static regions and the lack of deep interaction with trajectories hinder world models from exerting their full effectiveness. In this paper, we propose Temporal Residual World Model (TR-World), which focuses on dynamic object modeling. By calculating the temporal residuals of scene representations, the information of dynamic objects can be extracted without relying on detection and tracking. TR-World takes only temporal residuals as input, thus predicting the future spatial distribution of dynamic objects more precisely. By combining the prediction with the static object information contained in the current BEV features, accurate future BEV features can be obtained. Furthermore, we propose Future-Guided Trajectory Refinement (FGTR) module, which conducts interaction between prior trajectories (predicted from the current scene representation) and the future BEV features. This module can not only utilize future road conditions to refine trajectories, but also provides sparse spatial-temporal supervision on future BEV features to prevent world model collapse. Comprehensive experiments conducted on the nuScenes and NAVSIM datasets demonstrate that our method, namely ResWorld, achieves state-of-the-art planning performance. The code is available at https://github.com/mengtan00/ResWorld.git.

  </details>



- **AurigaNet: A Real-Time Multi-Task Network for Enhanced Urban Driving Perception**  
  Kiarash Ghasemzadeh, Sedigheh Dehghani  
  _2026-02-11_ · https://arxiv.org/abs/2602.10660v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Self-driving cars hold significant potential to reduce traffic accidents, alleviate congestion, and enhance urban mobility. However, developing reliable AI systems for autonomous vehicles remains a substantial challenge. Over the past decade, multi-task learning has emerged as a powerful approach to address complex problems in driving perception. Multi-task networks offer several advantages, including increased computational efficiency, real-time processing capabilities, optimized resource utilization, and improved generalization. In this study, we present AurigaNet, an advanced multi-task network architecture designed to push the boundaries of autonomous driving perception. AurigaNet integrates three critical tasks: object detection, lane detection, and drivable area instance segmentation. The system is trained and evaluated using the BDD100K dataset, renowned for its diversity in driving conditions. Key innovations of AurigaNet include its end-to-end instance segmentation capability, which significantly enhances both accuracy and efficiency in path estimation for autonomous vehicles. Experimental results demonstrate that AurigaNet achieves an 85.2% IoU in drivable area segmentation, outperforming its closest competitor by 0.7%. In lane detection, AurigaNet achieves a remarkable 60.8% IoU, surpassing other models by more than 30%. Furthermore, the network achieves an mAP@0.5:0.95 of 47.6% in traffic object detection, exceeding the next leading model by 2.9%. Additionally, we validate the practical feasibility of AurigaNet by deploying it on embedded devices such as the Jetson Orin NX, where it demonstrates competitive real-time performance. These results underscore AurigaNet's potential as a robust and efficient solution for autonomous driving perception systems. The code can be found here https://github.com/KiaRational/AurigaNet.

  </details>



- **From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?**  
  Krishna Kanth Nakka, Vedasri Nakka  
  _2026-02-11_ · https://arxiv.org/abs/2602.10771v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems.

  </details>



- **Interpretable Vision Transformers in Monocular Depth Estimation via SVDA**  
  Vasileios Arampatzakis, George Pavlidis, Nikolaos Mitianoudis, Nikos Papamarkos  
  _2026-02-11_ · https://arxiv.org/abs/2602.11005v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Monocular depth estimation is a central problem in computer vision with applications in robotics, AR, and autonomous driving, yet the self-attention mechanisms that drive modern Transformer architectures remain opaque. We introduce SVD-Inspired Attention (SVDA) into the Dense Prediction Transformer (DPT), providing the first spectrally structured formulation of attention for dense prediction tasks. SVDA decouples directional alignment from spectral modulation by embedding a learnable diagonal matrix into normalized query-key interactions, enabling attention maps that are intrinsically interpretable rather than post-hoc approximations. Experiments on KITTI and NYU-v2 show that SVDA preserves or slightly improves predictive accuracy while adding only minor computational overhead. More importantly, SVDA unlocks six spectral indicators that quantify entropy, rank, sparsity, alignment, selectivity, and robustness. These reveal consistent cross-dataset and depth-wise patterns in how attention organizes during training, insights that remain inaccessible in standard Transformers. By shifting the role of attention from opaque mechanism to quantifiable descriptor, SVDA redefines interpretability in monocular depth estimation and opens a principled avenue toward transparent dense prediction models.

  </details>



- **Enhancing Predictability of Multi-Tenant DNN Inference for Autonomous Vehicles' Perception**  
  Liangkai Liu, Kang G. Shin, Jinkyu Lee, Chengmo Yang, Weisong Shi  
  _2026-02-11_ · https://arxiv.org/abs/2602.11004v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous vehicles (AVs) rely on sensors and deep neural networks (DNNs) to perceive their surrounding environment and make maneuver decisions in real time. However, achieving real-time DNN inference in the AV's perception pipeline is challenging due to the large gap between the computation requirement and the AV's limited resources. Most, if not all, of existing studies focus on optimizing the DNN inference time to achieve faster perception by compressing the DNN model with pruning and quantization. In contrast, we present a Predictable Perception system with DNNs (PP-DNN) that reduce the amount of image data to be processed while maintaining the same level of accuracy for multi-tenant DNNs by dynamically selecting critical frames and regions of interest (ROIs). PP-DNN is based on our key insight that critical frames and ROIs for AVs vary with the AV's surrounding environment. However, it is challenging to identify and use critical frames and ROIs in multi-tenant DNNs for predictable inference. Given image-frame streams, PP-DNN leverages an ROI generator to identify critical frames and ROIs based on the similarities of consecutive frames and traffic scenarios. PP-DNN then leverages a FLOPs predictor to predict multiply-accumulate operations (MACs) from the dynamic critical frames and ROIs. The ROI scheduler coordinates the processing of critical frames and ROIs with multiple DNN models. Finally, we design a detection predictor for the perception of non-critical frames. We have implemented PP-DNN in an ROS-based AV pipeline and evaluated it with the BDD100K and the nuScenes dataset. PP-DNN is observed to significantly enhance perception predictability, increasing the number of fusion frames by up to 7.3x, reducing the fusion delay by >2.6x and fusion-delay variations by >2.3x, improving detection completeness by 75.4% and the cost-effectiveness by up to 98% over the baseline.

  </details>



- **Towards Learning a Generalizable 3D Scene Representation from 2D Observations**  
  Martin Gromniak, Jan-Gerrit Habekost, Sebastian Kamp, Sven Magg, Stefan Wermter  
  _2026-02-11_ · https://arxiv.org/abs/2602.10943v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We introduce a Generalizable Neural Radiance Field approach for predicting 3D workspace occupancy from egocentric robot observations. Unlike prior methods operating in camera-centric coordinates, our model constructs occupancy representations in a global workspace frame, making it directly applicable to robotic manipulation. The model integrates flexible source views and generalizes to unseen object arrangements without scene-specific finetuning. We demonstrate the approach on a humanoid robot and evaluate predicted geometry against 3D sensor ground truth. Trained on 40 real scenes, our model achieves 26mm reconstruction error, including occluded regions, validating its ability to infer complete 3D occupancy beyond traditional stereo vision methods.

  </details>



- **Viewpoint Recommendation for Point Cloud Labeling through Interaction Cost Modeling**  
  Yu Zhang, Xinyi Zhao, Chongke Bi, Siming Chen  
  _2026-02-11_ · https://arxiv.org/abs/2602.10871v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Semantic segmentation of 3D point clouds is important for many applications, such as autonomous driving. To train semantic segmentation models, labeled point cloud segmentation datasets are essential. Meanwhile, point cloud labeling is time-consuming for annotators, which typically involves tuning the camera viewpoint and selecting points by lasso. To reduce the time cost of point cloud labeling, we propose a viewpoint recommendation approach to reduce annotators' labeling time costs. We adapt Fitts' law to model the time cost of lasso selection in point clouds. Using the modeled time cost, the viewpoint that minimizes the lasso selection time cost is recommended to the annotator. We build a data labeling system for semantic segmentation of 3D point clouds that integrates our viewpoint recommendation approach. The system enables users to navigate to recommended viewpoints for efficient annotation. Through an ablation study, we observed that our approach effectively reduced the data labeling time cost. We also qualitatively compare our approach with previous viewpoint selection approaches on different datasets.

  </details>



- **From Representational Complementarity to Dual Systems: Synergizing VLM and Vision-Only Backbones for End-to-End Driving**  
  Sining Ang, Yuguang Yang, Chenxu Dang, Canyu Chen, Cheng Chi, Haiyan Liu, Xuanyao Mao, Jason Bao, Xuliang, Bingchuan Sun, et al.  
  _2026-02-11_ · https://arxiv.org/abs/2602.10719v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) driving augments end-to-end (E2E) planning with language-enabled backbones, yet it remains unclear what changes beyond the usual accuracy--cost trade-off. We revisit this question with 3--RQ analysis in RecogDrive by instantiating the system with a full VLM and vision-only backbones, all under an identical diffusion Transformer planner. RQ1: At the backbone level, the VLM can introduce additional subspaces upon the vision-only backbones. RQ2: This unique subspace leads to a different behavioral in some long-tail scenario: the VLM tends to be more aggressive whereas ViT is more conservative, and each decisively wins on about 2--3% of test scenarios; With an oracle that selects, per scenario, the better trajectory between the VLM and ViT branches, we obtain an upper bound of 93.58 PDMS. RQ3: To fully harness this observation, we propose HybridDriveVLA, which runs both ViT and VLM branches and selects between their endpoint trajectories using a learned scorer, improving PDMS to 92.10. Finally, DualDriveVLA implements a practical fast--slow policy: it runs ViT by default and invokes the VLM only when the scorer's confidence falls below a threshold; calling the VLM on 15% of scenarios achieves 91.00 PDMS while improving throughput by 3.2x. Code will be released.

  </details>


