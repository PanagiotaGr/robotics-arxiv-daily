# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-03-03 07:06 UTC_

Total papers shown: **5**


---

- **From Leaderboard to Deployment: Code Quality Challenges in AV Perception Repositories**  
  Mateus Karvat, Bram Adams, Sidney Givigi  
  _2026-03-02_ · https://arxiv.org/abs/2603.02194v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Autonomous vehicle (AV) perception models are typically evaluated solely on benchmark performance metrics, with limited attention to code quality, production readiness and long-term maintainability. This creates a significant gap between research excellence and real-world deployment in safety-critical systems subject to international safety standards. To address this gap, we present the first large-scale empirical study of software quality in AV perception repositories, systematically analyzing 178 unique models from the KITTI and NuScenes 3D Object Detection leaderboards. Using static analysis tools (Pylint, Bandit, and Radon), we evaluated code errors, security vulnerabilities, maintainability, and development practices. Our findings revealed that only 7.3% of the studied repositories meet basic production-readiness criteria, defined as having zero critical errors and no high-severity security vulnerabilities. Security issues are highly concentrated, with the top five issues responsible for almost 80% of occurrences, which prompted us to develop a set of actionable guidelines to prevent them. Additionally, the adoption of Continuous Integration/Continuous Deployment pipelines was correlated with better code maintainability. Our findings highlight that leaderboard performance does not reflect production readiness and that targeted interventions could substantially improve the quality and safety of AV perception code.

  </details>



- **Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling**  
  Alexander Prutsch, David Schinagl, Horst Possegger  
  _2026-03-02_ · https://arxiv.org/abs/2603.01864v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Future trajectories of neighboring traffic agents have a significant influence on the path planning and decision-making of autonomous vehicles. While trajectory forecasting is a well-studied field, research mainly focuses on snapshot-based prediction, where each scenario is treated independently of its global temporal context. However, real-world autonomous driving systems need to operate in a continuous setting, requiring real-time processing of data streams with low latency and consistent predictions over successive timesteps. We leverage this continuous setting to propose a lightweight yet highly accurate streaming-based trajectory forecasting approach. We integrate valuable information from previous predictions with a novel endpoint-aware modeling scheme. Our temporal context propagation uses the trajectory endpoints of the previous forecasts as anchors to extract targeted scenario context encodings. Our approach efficiently guides its scene encoder to extract highly relevant context information without needing refinement iterations or segment-wise decoding. Our experiments highlight that our approach effectively relays information across consecutive timesteps. Unlike methods using multi-stage refinement processing, our approach significantly reduces inference latency, making it well-suited for real-world deployment. We achieve state-of-the-art streaming trajectory prediction results on the Argoverse~2 multi-agent and single-agent benchmarks, while requiring substantially fewer resources.

  </details>



- **LaST-VLA: Thinking in Latent Spatio-Temporal Space for Vision-Language-Action in Autonomous Driving**  
  Yuechen Luo, Fang Li, Shaoqing Xu, Yang Ji, Zehan Zhang, Bing Wang, Yuannan Shen, Jianwei Cui, Long Chen, Guang Chen, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01928v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  While Vision-Language-Action (VLA) models have revolutionized autonomous driving by unifying perception and planning, their reliance on explicit textual Chain-of-Thought (CoT) leads to semantic-perceptual decoupling and perceptual-symbolic conflicts. Recent shifts toward latent reasoning attempt to bypass these bottlenecks by thinking in continuous hidden space. However, without explicit intermediate constraints, standard latent CoT often operates as a physics-agnostic representation. To address this, we propose the Latent Spatio-Temporal VLA (LaST-VLA), a framework shifting the reasoning paradigm from discrete symbolic processing into a physically grounded Latent Spatio-Temporal CoT. By implementing a dual-feature alignment mechanism, we distill geometric constraints from 3D foundation models and dynamic foresight from world models directly into the latent space. Coupled with a progressive SFT training strategy that transitions from feature alignment to trajectory generation, and refined via Reinforcement Learning with Group Relative Policy Optimization (GRPO) to ensure safety and rule compliance. \method~setting a new record on NAVSIM v1 (91.3 PDMS) and NAVSIM v2 (87.1 EPDMS), while excelling in spatial-temporal reasoning on SURDS and NuDynamics benchmarks.

  </details>



- **LAD-Drive: Bridging Language and Trajectory with Action-Aware Diffusion Transformers**  
  Fabian Schmidt, Karol Fedurko, Markus Enzweiler, Abhinav Valada  
  _2026-03-02_ · https://arxiv.org/abs/2603.02035v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While multimodal large language models (MLLMs) provide advanced reasoning for autonomous driving, translating their discrete semantic knowledge into continuous trajectories remains a fundamental challenge. Existing methods often rely on unimodal planning heads that inherently limit their ability to represent multimodal driving behavior. Furthermore, most generative approaches frequently condition on one-hot encoded actions, discarding the nuanced navigational uncertainty critical for complex scenarios. To resolve these limitations, we introduce LAD-Drive, a generative framework that structurally disentangles high-level intention from low-level spatial planning. LAD-Drive employs an action decoder to infer a probabilistic meta-action distribution, establishing an explicit belief state that preserves the nuanced intent typically lost by one-hot encodings. This distribution, fused with the vehicle's kinematic state, conditions an action-aware diffusion decoder that utilizes a truncated denoising process to refine learned motion anchors into safe, kinematically feasible trajectories. Extensive evaluations on the LangAuto benchmark demonstrate that LAD-Drive achieves state-of-the-art results, outperforming competitive baselines by up to 59% in Driving Score while significantly reducing route deviations and collisions. We will publicly release the code and models on https://github.com/iis-esslingen/lad-drive.

  </details>



- **GroupEnsemble: Efficient Uncertainty Estimation for DETR-based Object Detection**  
  Yutong Yang, Katarina Popović, Julian Wiederer, Markus Braun, Vasileios Belagiannis, Bin Yang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01847v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Detection Transformer (DETR) and its variants show strong performance on object detection, a key task for autonomous systems. However, a critical limitation of these models is that their confidence scores only reflect semantic uncertainty, failing to capture the equally important spatial uncertainty. This results in an incomplete assessment of the detection reliability. On the other hand, Deep Ensembles can tackle this by providing high-quality spatial uncertainty estimates. However, their immense memory consumption makes them impractical for real-world applications. A cheaper alternative, Monte Carlo (MC) Dropout, suffers from high latency due to the need of multiple forward passes during inference to estimate uncertainty. To address these limitations, we introduce GroupEnsemble, an efficient and effective uncertainty estimation method for DETR-like models. GroupEnsemble simultaneously predicts multiple individual detection sets by feeding additional diverse groups of object queries to the transformer decoder during inference. Each query group is transformed by the shared decoder in isolation and predicts a complete detection set for the same input. An attention mask is applied to the decoder to prevent inter-group query interactions, ensuring each group detects independently to achieve reliable ensemble-based uncertainty estimation. By leveraging the decoder's inherent parallelism, GroupEnsemble efficiently estimates uncertainty in a single forward pass without sequential repetition. We validated our method under autonomous driving scenes and common daily scenes using the Cityscapes and COCO datasets, respectively. The results show that a hybrid approach combining MC-Dropout and GroupEnsemble outperforms Deep Ensembles on several metrics at a fraction of the cost. The code is available at https://github.com/yutongy98/GroupEnsemble.

  </details>


