# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-01-28 06:52 UTC_

Total papers shown: **9**


---

- **The S3LI Vulcano Dataset: A Dataset for Multi-Modal SLAM in Unstructured Planetary Environments**  
  Riccardo Giubilato, Marcus Gerhard Müller, Marco Sewtz, Laura Alejandra Encinar Gonzalez, John Folkesson, Rudolph Triebel  
  _2026-01-27_ · https://arxiv.org/abs/2601.19557v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We release the S3LI Vulcano dataset, a multi-modal dataset towards development and benchmarking of Simultaneous Localization and Mapping (SLAM) and place recognition algorithms that rely on visual and LiDAR modalities. Several sequences are recorded on the volcanic island of Vulcano, from the Aeolian Islands in Sicily, Italy. The sequences provide users with data from a variety of environments, textures and terrains, including basaltic or iron-rich rocks, geological formations from old lava channels, as well as dry vegetation and water. The data (rmc.dlr.de/s3li_dataset) is accompanied by an open source toolkit (github.com/DLR-RM/s3li-toolkit) providing tools for generating ground truth poses as well as preparation of labelled samples for place recognition tasks.

  </details>



- **VGGT-SLAM 2.0: Real time Dense Feed-forward Scene Reconstruction**  
  Dominic Maggio, Luca Carlone  
  _2026-01-27_ · https://arxiv.org/abs/2601.19887v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present VGGT-SLAM 2.0, a real time RGB feed-forward SLAM system which substantially improves upon VGGT-SLAM for incrementally aligning submaps created from VGGT. Firstly, we remove high-dimensional 15-degree-of-freedom drift and planar degeneracy from VGGT-SLAM by creating a new factor graph design while still addressing the reconstruction ambiguity of VGGT given unknown camera intrinsics. Secondly, by studying the attention layers of VGGT, we show that one of the layers is well suited to assist in image retrieval verification for free without additional training, which enables both rejecting false positive matches and allows for completing more loop closures. Finally, we conduct a suite of experiments which includes showing VGGT-SLAM 2.0 can easily be adapted for open-set object detection and demonstrating real time performance while running online onboard a ground robot using a Jetson Thor. We also test in environments ranging from cluttered indoor apartments and office scenes to a 4,200 square foot barn, and we also demonstrate VGGT-SLAM 2.0 achieves the highest accuracy on the TUM dataset with about 23 percent less pose error than VGGT-SLAM. Code will be released upon publication.

  </details>



- **Enhancing Inverse Perspective Mapping for Automatic Vectorized Road Map Generation**  
  Hongji Liu, Linwei Zheng, Yongjian Li, Mingkai Tang, Xiaoyang Yan, Ming Liu, Jun Ma  
  _2026-01-27_ · https://arxiv.org/abs/2601.19536v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In this study, we present a low-cost and unified framework for vectorized road mapping leveraging enhanced inverse perspective mapping (IPM). In this framework, Catmull-Rom splines are utilized to characterize lane lines, and all the other ground markings are depicted using polygons uniformly. The results from instance segmentation serve as references to refine the three-dimensional position of spline control points and polygon corner points. In conjunction with this process, the homography matrix of IPM and vehicle poses are optimized simultaneously. Our proposed framework significantly reduces the mapping errors associated with IPM. It also improves the accuracy of the initial IPM homography matrix and the predicted vehicle poses. Furthermore, it addresses the limitations imposed by the coplanarity assumption in IPM. These enhancements enable IPM to be effectively applied to vectorized road mapping, which serves a cost-effective solution with enhanced accuracy. In addition, our framework generalizes road map elements to include all common ground markings and lane lines. The proposed framework is evaluated in two different practical scenarios, and the test results show that our method can automatically generate high-precision maps with near-centimeter-level accuracy. Importantly, the optimized IPM matrix achieves an accuracy comparable to that of manual calibration, while the accuracy of vehicle poses is also significantly improved.

  </details>



- **Generative Latent Alignment for Interpretable Radar Based Occupancy Detection in Ambient Assisted Living**  
  Huy Trinh  
  _2026-01-27_ · https://arxiv.org/abs/2601.19853v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  In this work, we study how to make mmWave radar presence detection more interpretable for Ambient Assisted Living (AAL) settings, where camera-based sensing raises privacy concerns. We propose a Generative Latent Alignment (GLA) framework that combines a lightweight convolutional variational autoencoder with a frozen CLIP text encoder to learn a low-dimensional latent representation of radar Range-Angle (RA) heatmaps. The latent space is softly aligned with two semantic anchors corresponding to "empty room" and "person present", and Grad-CAM is applied in this aligned latent space to visualize which spatial regions support each presence decision. On our mmWave radar dataset, we qualitatively observe that the "person present" class produces compact Grad-CAM blobs that coincide with strong RA returns, whereas "empty room" samples yield diffuse or no evidence. We also conduct an ablation study using unrelated text prompts, which degrades both reconstruction and localization, suggesting that radar-specific anchors are important for meaningful explanations in this setting.

  </details>



- **Out-of-Distribution Generalization via Invariant Trajectories for Multimodal Large Language Model Editing**  
  Jiajie Su, Haoyuan Wang, Xiaohua Feng, Yunshan Ma, Xiaobo Xia, Yuyuan Li, Xiaolin Zheng, Jianmao Xiao, Chaochao Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19700v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Knowledge editing emerges as a crucial technique for efficiently correcting incorrect or outdated knowledge in large language models (LLM). Existing editing methods for unimodal LLM rely on a rigid parameter-to-output mapping, which causes causal-underfit and causal-overfit in cascaded reasoning for Multimodal LLM (MLLM). In this paper, we reformulate MLLM editing as an out-of-distribution (OOD) generalization problem, where the goal is to discern semantic shift with factual shift and thus achieve robust editing among diverse cross-modal prompting. The key challenge of this OOD problem lies in identifying invariant causal trajectories that generalize accurately while suppressing spurious correlations. To address it, we propose ODEdit, a plug-and-play invariant learning based framework that optimizes the tripartite OOD risk objective to simultaneously enhance editing reliability, locality, and generality.We further introduce an edit trajectory invariant learning method, which integrates a total variation penalty into the risk minimization objective to stabilize edit trajectories against environmental variations. Theoretical analysis and extensive experiments demonstrate the effectiveness of ODEdit.

  </details>



- **Learning Adaptive Parallel Execution for Efficient Code Localization**  
  Ke Xu, Siyang Xiao, Ming Liang, Yichen Yu, Zhixiang Wang, Jingxuan Xu, Dajun Chen, Wei Jiang, Yong Li  
  _2026-01-27_ · https://arxiv.org/abs/2601.19568v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Code localization constitutes a key bottleneck in automated software development pipelines. While concurrent tool execution can enhance discovery speed, current agents demonstrate a 34.9\% redundant invocation rate, which negates parallelism benefits. We propose \textbf{FuseSearch}, reformulating parallel code localization as a \textbf{joint quality-efficiency optimization} task. Through defining \textbf{tool efficiency} -- the ratio of unique information gain to invocation count -- we utilize a two-phase SFT and RL training approach for learning adaptive parallel strategies. Different from fixed-breadth approaches, FuseSearch dynamically modulates search breadth according to task context, evolving from exploration phases to refinement stages. Evaluated on SWE-bench Verified, FuseSearch-4B achieves SOTA-level performance (84.7\% file-level and 56.4\% function-level $F_1$ scores) with 93.6\% speedup, utilizing 67.7\% fewer turns and 68.9\% fewer tokens. Results indicate that efficiency-aware training naturally improves quality through eliminating noisy redundant signals, enabling high-performance cost-effective localization agents.

  </details>



- **Bridging Information Asymmetry: A Hierarchical Framework for Deterministic Blind Face Restoration**  
  Zhengjian Yao, Jiakui Hu, Kaiwen Li, Hangzhou He, Xinliang Zhang, Shuang Zeng, Lei Zhu, Yanye Lu  
  _2026-01-27_ · https://arxiv.org/abs/2601.19506v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Blind face restoration remains a persistent challenge due to the inherent ill-posedness of reconstructing holistic structures from severely constrained observations. Current generative approaches, while capable of synthesizing realistic textures, often suffer from information asymmetry -- the intrinsic disparity between the information-sparse low quality inputs and the information-dense high quality outputs. This imbalance leads to a one-to-many mapping, where insufficient constraints result in stochastic uncertainty and hallucinatory artifacts. To bridge this gap, we present \textbf{Pref-Restore}, a hierarchical framework that integrates discrete semantic logic with continuous texture generation to achieve deterministic, preference-aligned restoration. Our methodology fundamentally addresses this information disparity through two complementary strategies: (1) Augmenting Input Density: We employ an auto-regressive integrator to reformulate textual instructions into dense latent queries, injecting high-level semantic stability to constrain the degraded signals; (2) Pruning Output Distribution: We pioneer the integration of on-policy reinforcement learning directly into the diffusion restoration loop. By transforming human preferences into differentiable constraints, we explicitly penalize stochastic deviations, thereby sharpening the posterior distribution toward the desired high-fidelity outcomes. Extensive experiments demonstrate that Pref-Restore achieves state-of-the-art performance across synthetic and real-world benchmarks. Furthermore, empirical analysis confirms that our preference-aligned strategy significantly reduces solution entropy, establishing a robust pathway toward reliable and deterministic blind restoration.

  </details>



- **Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction**  
  Ziyu Zhang, Tianle Liu, Diantao Tu, Shuhan Shen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19489v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge. The challenge consists of an initial round using SLAM-generated camera poses (with noisy trajectories) and a final round using COLMAP poses (highly accurate). To robustly handle these heterogeneous settings, we develop a two-stage solution. In the first round, we use reverse per-Gaussian parallel optimization and compact forward splatting based on Taming-GS and Speedy-splat, load-balanced tiling, an anchor-based Neural-Gaussian representation enabling rapid convergence with fewer learnable parameters, initialization from monocular depth and partially from feed-forward 3DGS models, and a global pose refinement module for noisy SLAM trajectories. In the final round, the accurate COLMAP poses change the optimization landscape; we disable pose refinement, revert from Neural-Gaussians back to standard 3DGS to eliminate MLP inference overhead, introduce multi-view consistency-guided Gaussian splitting inspired by Fast-GS, and introduce a depth estimator to supervise the rendered depth. Together, these techniques enable high-fidelity reconstruction under a strict one-minute budget. Our method achieved the top performance with a PSNR of 28.43 and ranked first in the competition.

  </details>



- **Metric $k$-clustering using only Weak Comparison Oracles**  
  Rahul Raychaudhury, Aryan Esmailpour, Sainyam Galhotra, Stavros Sintos  
  _2026-01-27_ · https://arxiv.org/abs/2601.19333v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Clustering is a fundamental primitive in unsupervised learning. However, classical algorithms for $k$-clustering (such as $k$-median and $k$-means) assume access to exact pairwise distances -- an unrealistic requirement in many modern applications. We study clustering in the \emph{Rank-model (R-model)}, where access to distances is entirely replaced by a \emph{quadruplet oracle} that provides only relative distance comparisons. In practice, such an oracle can represent learned models or human feedback, and is expected to be noisy and entail an access cost. Given a metric space with $n$ input items, we design randomized algorithms that, using only a noisy quadruplet oracle, compute a set of $O(k \cdot \mathsf{polylog}(n))$ centers along with a mapping from the input items to the centers such that the clustering cost of the mapping is at most constant times the optimum $k$-clustering cost. Our method achieves a query complexity of $O(n\cdot k \cdot \mathsf{polylog}(n))$ for arbitrary metric spaces and improves to $O((n+k^2) \cdot \mathsf{polylog}(n))$ when the underlying metric has bounded doubling dimension. When the metric has bounded doubling dimension we can further improve the approximation from constant to $1+\varepsilon$, for any arbitrarily small constant $\varepsilon\in(0,1)$, while preserving the same asymptotic query complexity. Our framework demonstrates how noisy, low-cost oracles, such as those derived from large language models, can be systematically integrated into scalable clustering algorithms.

  </details>


