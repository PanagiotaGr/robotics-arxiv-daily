# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-03-04 07:02 UTC_

Total papers shown: **4**


---

- **Utonia: Toward One Encoder for All Point Clouds**  
  Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, Hengshuang Zhao  
  _2026-03-03_ · https://arxiv.org/abs/2603.03283v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We dream of a future where point clouds from all domains can come together to shape a single model that benefits them all. Toward this goal, we present Utonia, a first step toward training a single self-supervised point transformer encoder across diverse domains, spanning remote sensing, outdoor LiDAR, indoor RGB-D sequences, object-centric CAD models, and point clouds lifted from RGB-only videos. Despite their distinct sensing geometries, densities, and priors, Utonia learns a consistent representation space that transfers across domains. This unification improves perception capability while revealing intriguing emergent behaviors that arise only when domains are trained jointly. Beyond perception, we observe that Utonia representations can also benefit embodied and multimodal reasoning: conditioning vision-language-action policies on Utonia features improves robotic manipulation, and integrating them into vision-language models yields gains on spatial reasoning. We hope Utonia can serve as a step toward foundation models for sparse 3D data, and support downstream applications in AR/VR, robotics, and autonomous driving.

  </details>



- **CoFL: Continuous Flow Fields for Language-Conditioned Navigation**  
  Haokun Liu, Zhaoqi Ma, Yicheng Chen, Masaki Kitagawa, Wentao Zhang, Jinjie Li, Moju Zhao  
  _2026-03-03_ · https://arxiv.org/abs/2603.02854v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Language-conditioned navigation pipelines often rely on brittle modular components or costly action-sequence generation. To address these limitations, we present CoFL, an end-to-end policy that directly maps a bird's-eye view (BEV) observation and a language instruction to a continuous flow field for navigation. Instead of predicting discrete action tokens or sampling action chunks via iterative denoising, CoFL outputs instantaneous velocities that can be queried at arbitrary 2D projected locations. Trajectories are obtained by numerical integration of the predicted field, producing smooth motion that remains reactive under closed-loop execution. To enable large-scale training, we build a dataset of over 500k BEV image-instruction pairs, each procedurally annotated with a flow field and a trajectory derived from BEV semantic maps built on Matterport3D and ScanNet. By training on a mixed distribution, CoFL significantly outperforms modular Vision-Language Model (VLM)-based planners and generative policy baselines on strictly unseen scenes. Finally, we deploy CoFL zero-shot in real-world experiments with overhead BEV observations across multiple layouts, maintaining reliable closed-loop control and a high success rate.

  </details>



- **ACE-Brain-0: Spatial Intelligence as a Shared Scaffold for Universal Embodiments**  
  Ziyang Gong, Zehang Luo, Anke Tang, Zhe Liu, Shi Fu, Zhi Hou, Ganlin Yang, Weiyun Wang, Xiaofeng Wang, Jianbo Liu, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.03198v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Universal embodied intelligence demands robust generalization across heterogeneous embodiments, such as autonomous driving, robotics, and unmanned aerial vehicles (UAVs). However, existing embodied brain in training a unified model over diverse embodiments frequently triggers long-tail data, gradient interference, and catastrophic forgetting, making it notoriously difficult to balance universal generalization with domain-specific proficiency. In this report, we introduce ACE-Brain-0, a generalist foundation brain that unifies spatial reasoning, autonomous driving, and embodied manipulation within a single multimodal large language model~(MLLM). Our key insight is that spatial intelligence serves as a universal scaffold across diverse physical embodiments: although vehicles, robots, and UAVs differ drastically in morphology, they share a common need for modeling 3D mental space, making spatial cognition a natural, domain-agnostic foundation for cross-embodiment transfer. Building on this insight, we propose the Scaffold-Specialize-Reconcile~(SSR) paradigm, which first establishes a shared spatial foundation, then cultivates domain-specialized experts, and finally harmonizes them through data-free model merging. Furthermore, we adopt Group Relative Policy Optimization~(GRPO) to strengthen the model's comprehensive capability. Extensive experiments demonstrate that ACE-Brain-0 achieves competitive and even state-of-the-art performance across 24 spatial and embodiment-related benchmarks.

  </details>



- **Context Adaptive Extended Chain Coding for Semantic Map Compression**  
  Runyu Yang, Junqi Liao, Hyomin Choi, Fabien Racapé, Ivan V. Bajić  
  _2026-03-03_ · https://arxiv.org/abs/2603.03073v1 · `eess.IV`  
  <details><summary>Abstract</summary>

  Semantic maps are increasingly utilized in areas such as robotics, autonomous systems, and extended reality, motivating the investigation of efficient compression methods that preserve structured semantic information. This paper studies lossless compression of semantic maps through a novel chain-coding-based framework that explicitly exploits contour topology and shared boundaries between adjacent semantic regions. We propose an extended chain code (ECC) to represent long-range contour transitions more compactly, while retaining a legacy three-orthogonal chain code (3OT) as a fallback mode for further efficiency. To efficiently encode sequences of ECC symbols, a context-adaptive entropy coding scheme based on Markov modeling is employed. Furthermore, a skip-coding mechanism is introduced to eliminate redundant representations of shared contours between adjacent semantic regions, supporting both complete and partial skips via run-length signaling. Experimental results demonstrate that the proposed method achieves an average bitrate reduction of 18\% compared with a state-of-the-art benchmark on semantic map datasets. In addition, the proposed encoder and decoder achieve up to 98\% and 50\% runtime reduction, respectively, relative to a modern generic lossless codec. Extended evaluations on occupancy maps further confirm consistent compression gains across the majority of tested scenarios.

  </details>


