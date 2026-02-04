# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-04 07:06 UTC_

Total papers shown: **4**


---

- **HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic**  
  Yu-Hsiang Chen, Wei-Jer Chang, Christian Kotulla, Thomas Keutgens, Steffen Runde, Tobias Moers, Christoph Klas, Wei Zhan, Masayoshi Tomizuka, Yi-Ting Chen  
  _2026-02-03_ · https://arxiv.org/abs/2602.03447v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present HetroD, a dataset and benchmark for developing autonomous driving systems in heterogeneous environments. HetroD targets the critical challenge of navi- gating real-world heterogeneous traffic dominated by vulner- able road users (VRUs), including pedestrians, cyclists, and motorcyclists that interact with vehicles. These mixed agent types exhibit complex behaviors such as hook turns, lane splitting, and informal right-of-way negotiation. Such behaviors pose significant challenges for autonomous vehicles but remain underrepresented in existing datasets focused on structured, lane-disciplined traffic. To bridge the gap, we collect a large- scale drone-based dataset to provide a holistic observation of traffic scenes with centimeter-accurate annotations, HD maps, and traffic signal states. We further develop a modular toolkit for extracting per-agent scenarios to support downstream task development. In total, the dataset comprises over 65.4k high- fidelity agent trajectories, 70% of which are from VRUs. HetroD supports modeling of VRU behaviors in dense, het- erogeneous traffic and provides standardized benchmarks for forecasting, planning, and simulation tasks. Evaluation results reveal that state-of-the-art prediction and planning models struggle with the challenges presented by our dataset: they fail to predict lateral VRU movements, cannot handle unstructured maneuvers, and exhibit limited performance in dense and multi-agent scenarios, highlighting the need for more robust approaches to heterogeneous traffic. See our project page for more examples: https://hetroddata.github.io/HetroD/

  </details>



- **Efficient Sequential Neural Network with Spatial-Temporal Attention and Linear LSTM for Robust Lane Detection Using Multi-Frame Images**  
  Sandeep Patil, Yongqi Dong, Haneen Farah, Hans Hellendoorn  
  _2026-02-03_ · https://arxiv.org/abs/2602.03669v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Lane detection is a crucial perception task for all levels of automated vehicles (AVs) and Advanced Driver Assistance Systems, particularly in mixed-traffic environments where AVs must interact with human-driven vehicles (HDVs) and challenging traffic scenarios. Current methods lack versatility in delivering accurate, robust, and real-time compatible lane detection, especially vision-based methods often neglect critical regions of the image and their spatial-temporal (ST) salience, leading to poor performance in difficult circumstances such as serious occlusion and dazzle lighting. This study introduces a novel sequential neural network model with a spatial-temporal attention mechanism to focus on key features of lane lines and exploit salient ST correlations among continuous image frames. The proposed model, built on a standard encoder-decoder structure and common neural network backbones, is trained and evaluated on three large-scale open-source datasets. Extensive experiments demonstrate the strength and robustness of the proposed model, outperforming state-of-the-art methods in various testing scenarios. Furthermore, with the ST attention mechanism, the developed sequential neural network models exhibit fewer parameters and reduced Multiply-Accumulate Operations (MACs) compared to baseline sequential models, highlighting their computational efficiency. Relevant data, code, and models are released at https://doi.org/10.4121/4619cab6-ae4a-40d5-af77-582a77f3d821.

  </details>



- **Multi-Player, Multi-Strategy Quantum Game Model for Interaction-Aware Decision-Making in Autonomous Driving**  
  Karim Essalmi, Fernando Garrido, Fawzi Nashashibi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03571v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Although significant progress has been made in decision-making for automated driving, challenges remain for deployment in the real world. One challenge lies in addressing interaction-awareness. Most existing approaches oversimplify interactions between the ego vehicle and surrounding agents, and often neglect interactions among the agents themselves. A common solution is to model these interactions using classical game theory. However, its formulation assumes rational players, whereas human behavior is frequently uncertain or irrational. To address these challenges, we propose the Quantum Game Decision-Making (QGDM) model, a novel framework that combines classical game theory with quantum mechanics principles (such as superposition, entanglement, and interference) to tackle multi-player, multi-strategy decision-making problems. To the best of our knowledge, this is one of the first studies to apply quantum game theory to decision-making for automated driving. QGDM runs in real time on a standard computer, without requiring quantum hardware. We evaluate QGDM in simulation across various scenarios, including roundabouts, merging, and highways, and compare its performance with multiple baseline methods. Results show that QGDM significantly improves success rates and reduces collision rates compared to classical approaches, particularly in scenarios with high interaction.

  </details>



- **Inlier-Centric Post-Training Quantization for Object Detection Models**  
  Minsu Kim, Dongyeun Lee, Jaemyung Yu, Jiwan Hur, Giseop Kim, Junmo Kim  
  _2026-02-03_ · https://arxiv.org/abs/2602.03472v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Object detection is pivotal in computer vision, yet its immense computational demands make deployment slow and power-hungry, motivating quantization. However, task-irrelevant morphologies such as background clutter and sensor noise induce redundant activations (or anomalies). These anomalies expand activation ranges and skew activation distributions toward task-irrelevant responses, complicating bit allocation and weakening the preservation of informative features. Without a clear criterion to distinguish anomalies, suppressing them can inadvertently discard useful information. To address this, we present InlierQ, an inlier-centric post-training quantization approach that separates anomalies from informative inliers. InlierQ computes gradient-aware volume saliency scores, classifies each volume as an inlier or anomaly, and fits a posterior distribution over these scores using the Expectation-Maximization (EM) algorithm. This design suppresses anomalies while preserving informative features. InlierQ is label-free, drop-in, and requires only 64 calibration samples. Experiments on the COCO and nuScenes benchmarks show consistent reductions in quantization error for camera-based (2D and 3D) and LiDAR-based (3D) object detection.

  </details>


