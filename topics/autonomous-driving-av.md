# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-10 07:19 UTC_

Total papers shown: **7**


---

- **GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion**  
  Santiago Montiel-Marín, Miguel Antunes-García, Fabio Sánchez-García, Angel Llamazares, Holger Caesar, Luis M. Bergasa  
  _2026-02-09_ · https://arxiv.org/abs/2602.08784v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robust and accurate perception of dynamic objects and map elements is crucial for autonomous vehicles performing safe navigation in complex traffic scenarios. While vision-only methods have become the de facto standard due to their technical advances, they can benefit from effective and cost-efficient fusion with radar measurements. In this work, we advance fusion methods by repurposing Gaussian Splatting as an efficient universal view transformer that bridges the view disparity gap, mapping both image pixels and radar points into a common Bird's-Eye View (BEV) representation. Our main contribution is GaussianCaR, an end-to-end network for BEV segmentation that, unlike prior BEV fusion methods, leverages Gaussian Splatting to map raw sensor information into latent features for efficient camera-radar fusion. Our architecture combines multi-scale fusion with a transformer decoder to efficiently extract BEV features. Experimental results demonstrate that our approach achieves performance on par with, or even surpassing, the state of the art on BEV segmentation tasks (57.3%, 82.9%, and 50.1% IoU for vehicles, roads, and lane dividers) on the nuScenes dataset, while maintaining a 3.2x faster inference runtime. Code and project page are available online.

  </details>



- **A Generic Service-Oriented Function Offloading Framework for Connected Automated Vehicles**  
  Robin Dehler, Michael Buchholz  
  _2026-02-09_ · https://arxiv.org/abs/2602.08799v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Function offloading is a promising solution to address limitations concerning computational capacity and available energy of Connected Automated Vehicles~(CAVs) or other autonomous robots by distributing computational tasks between local and remote computing devices in form of distributed services. This paper presents a generic function offloading framework that can be used to offload an arbitrary set of computational tasks with a focus on autonomous driving. To provide flexibility, the function offloading framework is designed to incorporate different offloading decision making algorithms and quality of service~(QoS) requirements that can be adjusted to different scenarios or the objectives of the CAVs. With a focus on the applicability, we propose an efficient location-based approach, where the decision whether tasks are processed locally or remotely depends on the location of the CAV. We apply the proposed framework on the use case of service-oriented trajectory planning, where we offload the trajectory planning task of CAVs to a Multi-Access Edge Computing~(MEC) server. The evaluation is conducted in both simulation and real-world application. It demonstrates the potential of the function offloading framework to guarantee the QoS for trajectory planning while improving the computational efficiency of the CAVs. Moreover, the simulation results also show the adaptability of the framework to diverse scenarios involving simultaneous offloading requests from multiple CAVs.

  </details>



- **Modeling 3D Pedestrian-Vehicle Interactions for Vehicle-Conditioned Pose Forecasting**  
  Guangxun Zhu, Xuan Liu, Nicolas Pugeault, Chongfeng Wei, Edmond S. L. Ho  
  _2026-02-09_ · https://arxiv.org/abs/2602.08962v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurately predicting pedestrian motion is crucial for safe and reliable autonomous driving in complex urban environments. In this work, we present a 3D vehicle-conditioned pedestrian pose forecasting framework that explicitly incorporates surrounding vehicle information. To support this, we enhance the Waymo-3DSkelMo dataset with aligned 3D vehicle bounding boxes, enabling realistic modeling of multi-agent pedestrian-vehicle interactions. We introduce a sampling scheme to categorize scenes by pedestrian and vehicle count, facilitating training across varying interaction complexities. Our proposed network adapts the TBIFormer architecture with a dedicated vehicle encoder and pedestrian-vehicle interaction cross-attention module to fuse pedestrian and vehicle features, allowing predictions to be conditioned on both historical pedestrian motion and surrounding vehicles. Extensive experiments demonstrate substantial improvements in forecasting accuracy and validate different approaches for modeling pedestrian-vehicle interactions, highlighting the importance of vehicle-aware 3D pose prediction for autonomous driving. Code is available at: https://github.com/GuangxunZhu/VehCondPose3D

  </details>



- **Robustness Is a Function, Not a Number: A Factorized Comprehensive Study of OOD Robustness in Vision-Based Driving**  
  Amir Mallak, Alaa Maalouf  
  _2026-02-09_ · https://arxiv.org/abs/2602.09018v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Out of distribution (OOD) robustness in autonomous driving is often reduced to a single number, hiding what breaks a policy. We decompose environments along five axes: scene (rural/urban), season, weather, time (day/night), and agent mix; and measure performance under controlled $k$-factor perturbations ($k \in \{0,1,2,3\}$). Using closed loop control in VISTA, we benchmark FC, CNN, and ViT policies, train compact ViT heads on frozen foundation-model (FM) features, and vary ID support in scale, diversity, and temporal context. (1) ViT policies are markedly more OOD-robust than comparably sized CNN/FC, and FM features yield state-of-the-art success at a latency cost. (2) Naive temporal inputs (multi-frame) do not beat the best single-frame baseline. (3) The largest single factor drops are rural $\rightarrow$ urban and day $\rightarrow$ night ($\sim 31\%$ each); actor swaps $\sim 10\%$, moderate rain $\sim 7\%$; season shifts can be drastic, and combining a time flip with other changes further degrades performance. (4) FM-feature policies stay above $85\%$ under three simultaneous changes; non-FM single-frame policies take a large first-shift hit, and all no-FM models fall below $50\%$ by three changes. (5) Interactions are non-additive: some pairings partially offset, whereas season-time combinations are especially harmful. (6) Training on winter/snow is most robust to single-factor shifts, while a rural+summer baseline gives the best overall OOD performance. (7) Scaling traces/views improves robustness ($+11.8$ points from $5$ to $14$ traces), yet targeted exposure to hard conditions can substitute for scale. (8) Using multiple ID environments broadens coverage and strengthens weak cases (urban OOD $60.6\% \rightarrow 70.1\%$) with a small ID drop; single-ID preserves peak performance but in a narrow domain. These results yield actionable design rules for OOD-robust driving policies.

  </details>



- **Overview and Comparison of AVS Point Cloud Compression Standard**  
  Wei Gao, Wenxu Gao, Xingming Mu, Changhao Peng, Ge Li  
  _2026-02-09_ · https://arxiv.org/abs/2602.08613v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Point cloud is a prevalent 3D data representation format with significant application values in immersive media, autonomous driving, digital heritage protection, etc. However, the large data size of point clouds poses challenges to transmission and storage, which influences the wide deployments. Therefore, point cloud compression plays a crucial role in practical applications for both human and machine perception optimization. To this end, the Moving Picture Experts Group (MPEG) has established two standards for point cloud compression, including Geometry-based Point Cloud Compression (G-PCC) and Video-based Point Cloud Compression (V-PCC). In the meantime, the Audio Video coding Standard (AVS) Workgroup of China also have launched and completed the development for its first generation point cloud compression standard, namely AVS PCC. This new standardization effort has adopted many new coding tools and techniques, which are different from the other counterpart standards. This paper reviews the AVS PCC standard from two perspectives, i.e., the related technologies and performance comparisons.

  </details>



- **Multi-Staged Framework for Safety Analysis of Offloaded Services in Distributed Intelligent Transportation Systems**  
  Robin Dehler, Oliver Schumann, Jona Ruof, Michael Buchholz  
  _2026-02-09_ · https://arxiv.org/abs/2602.08821v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The integration of service-oriented architectures (SOA) with function offloading for distributed, intelligent transportation systems (ITS) offers the opportunity for connected autonomous vehicles (CAVs) to extend their locally available services. One major goal of offloading a subset of functions in the processing chain of a CAV to remote devices is to reduce the overall computational complexity on the CAV. The extension of using remote services, however, requires careful safety analysis, since the remotely created data are corrupted more easily, e.g., through an attacker on the remote device or by intercepting the wireless transmission. To tackle this problem, we first analyze the concept of SOA for distributed environments. From this, we derive a safety framework that validates the reliability of remote services and the data received locally. Since it is possible for the autonomous driving task to offload multiple different services, we propose a specific multi-staged framework for safety analysis dependent on the service composition of local and remote services. For efficiency reasons, we directly include the multi-staged framework for safety analysis in our service-oriented function offloading framework (SOFOF) that we have proposed in earlier work. The evaluation compares the performance of the extended framework considering computational complexity, with energy savings being a major motivation for function offloading, and its capability to detect data from corrupted remote services.

  </details>



- **Head-to-Head autonomous racing at the limits of handling in the A2RL challenge**  
  Simon Hoffmann, Simon Sagmeister, Tobias Betz, Joscha Bongard, Sascha Büttner, Dominic Ebner, Daniel Esser, Georg Jank, Sven Goblirsch, Alexander Langmann, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08571v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous racing presents a complex challenge involving multi-agent interactions between vehicles operating at the limit of performance and dynamics. As such, it provides a valuable research and testing environment for advancing autonomous driving technology and improving road safety. This article presents the algorithms and deployment strategies developed by the TUM Autonomous Motorsport team for the inaugural Abu Dhabi Autonomous Racing League (A2RL). We showcase how our software emulates human driving behavior, pushing the limits of vehicle handling and multi-vehicle interactions to win the A2RL. Finally, we highlight the key enablers of our success and share our most significant learnings.

  </details>


