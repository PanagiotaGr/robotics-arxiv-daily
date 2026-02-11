# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-11 07:16 UTC_

Total papers shown: **17**


---

- **A Collision-Free Sway Damping Model Predictive Controller for Safe and Reactive Forestry Crane Navigation**  
  Marc-Philip Ecker, Christoph Fröhlich, Johannes Huemer, David Gruber, Bernhard Bischof, Tobias Glück, Wolfgang Kemmetmüller  
  _2026-02-10_ · https://arxiv.org/abs/2602.10035v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Forestry cranes operate in dynamic, unstructured outdoor environments where simultaneous collision avoidance and payload sway control are critical for safe navigation. Existing approaches address these challenges separately, either focusing on sway damping with predefined collision-free paths or performing collision avoidance only at the global planning level. We present the first collision-free, sway-damping model predictive controller (MPC) for a forestry crane that unifies both objectives in a single control framework. Our approach integrates LiDAR-based environment mapping directly into the MPC using online Euclidean distance fields (EDF), enabling real-time environmental adaptation. The controller simultaneously enforces collision constraints while damping payload sway, allowing it to (i) replan upon quasi-static environmental changes, (ii) maintain collision-free operation under disturbances, and (iii) provide safe stopping when no bypass exists. Experimental validation on a real forestry crane demonstrates effective sway damping and successful obstacle avoidance. A video can be found at https://youtu.be/tEXDoeLLTxA.

  </details>



- **RoboSubtaskNet: Temporal Sub-task Segmentation for Human-to-Robot Skill Transfer in Real-World Environments**  
  Dharmendra Sharma, Archit Sharma, John Reberio, Vaibhav Kesharwani, Peeyush Thakur, Narendra Kumar Dhar, Laxmidhar Behera  
  _2026-02-10_ · https://arxiv.org/abs/2602.10015v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Temporally locating and classifying fine-grained sub-task segments in long, untrimmed videos is crucial to safe human-robot collaboration. Unlike generic activity recognition, collaborative manipulation requires sub-task labels that are directly robot-executable. We present RoboSubtaskNet, a multi-stage human-to-robot sub-task segmentation framework that couples attention-enhanced I3D features (RGB plus optical flow) with a modified MS-TCN employing a Fibonacci dilation schedule to capture better short-horizon transitions such as reach-pick-place. The network is trained with a composite objective comprising cross-entropy and temporal regularizers (truncated MSE and a transition-aware term) to reduce over-segmentation and to encourage valid sub-task progressions. To close the gap between vision benchmarks and control, we introduce RoboSubtask, a dataset of healthcare and industrial demonstrations annotated at the sub-task level and designed for deterministic mapping to manipulator primitives. Empirically, RoboSubtaskNet outperforms MS-TCN and MS-TCN++ on GTEA and our RoboSubtask benchmark (boundary-sensitive and sequence metrics), while remaining competitive on the long-horizon Breakfast benchmark. Specifically, RoboSubtaskNet attains F1 @ 50 = 79.5%, Edit = 88.6%, Acc = 78.9% on GTEA; F1 @ 50 = 30.4%, Edit = 52.0%, Acc = 53.5% on Breakfast; and F1 @ 50 = 94.2%, Edit = 95.6%, Acc = 92.2% on RoboSubtask. We further validate the full perception-to-execution pipeline on a 7-DoF Kinova Gen3 manipulator, achieving reliable end-to-end behavior in physical trials (overall task success approx 91.25%). These results demonstrate a practical path from sub-task level video understanding to deployed robotic manipulation in real-world settings.

  </details>



- **A Collaborative Safety Shield for Safe and Efficient CAV Lane Changes in Congested On-Ramp Merging**  
  Bharathkumar Hegde, Melanie Bouroche  
  _2026-02-10_ · https://arxiv.org/abs/2602.10007v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Lane changing in dense traffic is a significant challenge for Connected and Autonomous Vehicles (CAVs). Existing lane change controllers primarily either ensure safety or collaboratively improve traffic efficiency, but do not consider these conflicting objectives together. To address this, we propose the Multi-Agent Safety Shield (MASS), designed using Control Barrier Functions (CBFs) to enable safe and collaborative lane changes. The MASS enables collaboration by capturing multi-agent interactions among CAVs through interaction topologies constructed as a graph using a simple algorithm. Further, a state-of-the-art Multi-Agent Reinforcement Learning (MARL) lane change controller is extended by integrating MASS to ensure safety and defining a customised reward function to prioritise efficiency improvements. As a result, we propose a lane change controller, known as MARL-MASS, and evaluate it in a congested on-ramp merging simulation. The results demonstrate that MASS enables collaborative lane changes with safety guarantees by strictly respecting the safety constraints. Moreover, the proposed custom reward function improves the stability of MARL policies trained with a safety shield. Overall, by encouraging the exploration of a collaborative lane change policy while respecting safety constraints, MARL-MASS effectively balances the trade-off between ensuring safety and improving traffic efficiency in congested traffic. The code for MARL-MASS is available with an open-source licence at https://github.com/hkbharath/MARL-MASS

  </details>



- **Hybrid Responsible AI-Stochastic Approach for SLA Compliance in Multivendor 6G Networks**  
  Emanuel Figetakis, Ahmed Refaey Hussein  
  _2026-02-10_ · https://arxiv.org/abs/2602.09841v1 · `cs.NI`  
  <details><summary>Abstract</summary>

  The convergence of AI and 6G network automation introduces new challenges in maintaining transparency, fairness, and accountability across multivendor management systems. Although closed-loop AI orchestration improves adaptability and self-optimization, it also creates a responsibility gap, where violations of SLAs cannot be causally attributed to specific agents or vendors. This paper presents a hybrid responsible AI-stochastic learning framework that embeds fairness, robustness, and auditability directly into the network control loop. The framework integrates RAI games with stochastic optimization, enabling dynamic adversarial reweighting and probabilistic exploration across heterogeneous vendor domains. An RAAP continuously records AI-driven decision trajectories and produces dual accountability reports: user-level SLA summaries and operator-level responsibility analytics. Experimental evaluations on synthetic two-class multigroup datasets demonstrate that the proposed hybrid model improves the accuracy of the worst group by up to 10.5\%. Specifically, hybrid RAI achieved a WGAcc of 60.5\% and an AvgAcc of 72.7\%, outperforming traditional RAI-GA (50.0\%) and ERM (21.5\%). The audit mechanism successfully traced 99\% simulated SLA violations to the AI entities responsible, producing both vendor and agent-level accountability indices. These results confirm that the proposed hybrid approach enhances fairness and robustness as well as establishes a concrete accountability framework for autonomous SLA assurance in multivendor 6G networks.

  </details>



- **Robust Vision Systems for Connected and Autonomous Vehicles: Security Challenges and Attack Vectors**  
  Sandeep Gupta, Roberto Passerone  
  _2026-02-10_ · https://arxiv.org/abs/2602.09740v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This article investigates the robustness of vision systems in Connected and Autonomous Vehicles (CAVs), which is critical for developing Level-5 autonomous driving capabilities. Safe and reliable CAV navigation undeniably depends on robust vision systems that enable accurate detection of objects, lane markings, and traffic signage. We analyze the key sensors and vision components essential for CAV navigation to derive a reference architecture for CAV vision system (CAVVS). This reference architecture provides a basis for identifying potential attack surfaces of CAVVS. Subsequently, we elaborate on identified attack vectors targeting each attack surface, rigorously evaluating their implications for confidentiality, integrity, and availability (CIA). Our study provides a comprehensive understanding of attack vector dynamics in vision systems, which is crucial for formulating robust security measures that can uphold the principles of the CIA triad.

  </details>



- **Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings**  
  Alexander Fertig, Karthikeyan Chandra Sekaran, Lakshman Balasubramanian, Michael Botsch  
  _2026-02-10_ · https://arxiv.org/abs/2602.09985v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  As autonomous vehicles are rolled out, measures must be taken to ensure their safe operation. In order to supervise a system that is already in operation, monitoring frameworks are frequently employed. These run continuously online in the background, supervising the system status and recording anomalies. This work proposes an online monitoring framework to detect anomalies in object state representations. Thereby, a key challenge is creating a framework for anomaly detection without anomaly labels, which are usually unavailable for unknown anomalies. To address this issue, this work applies a self-supervised embedding method to translate object data into a latent representation space. For this, a JEPA-based self-supervised prediction task is constructed, allowing training without anomaly labels and the creation of rich object embeddings. The resulting expressive JEPA embeddings serve as input for established anomaly detection methods, in order to identify anomalies within object state representations. This framework is particularly useful for applications in real-world environments, where new or unknown anomalies may occur during operation for which there are no labels available. Experiments performed on the publicly available, real-world nuScenes dataset illustrate the framework's capabilities.

  </details>



- **Robust Processing and Learning: Principles, Methods, and Wireless Applications**  
  Shixiong Wang, Wei Dai, Li-Chun Wang, Geoffrey Ye Li  
  _2026-02-10_ · https://arxiv.org/abs/2602.09848v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  This tutorial-style overview article examines the fundamental principles and methods of robustness, using wireless sensing and communication (WSC) as the narrative and exemplifying framework. First, we formalize the conceptual and mathematical foundations of robustness, highlighting the interpretations and relations across robust statistics, optimization, and machine learning. Key techniques, such as robust estimation and testing, distributionally robust optimization, and regularized and adversary training, are investigated. Together, the costs of robustness in system design, for example, the compromised nominal performances and the extra computational burdens, are discussed. Second, we review recent robust signal processing solutions for WSC that address model mismatch, data scarcity, adversarial perturbation, and distributional shift. Specific applications include robust ranging-based localization, modality sensing, channel estimation, receive combining, waveform design, and federated learning. Through this effort, we aim to introduce the classical developments and recent advances in robustness theory to the general signal processing community, exemplifying how robust statistical, optimization, and machine learning approaches can address the uncertainties inherent in WSC systems.

  </details>



- **Humanoid Factors: Design Principles for AI Humanoids in Human Worlds**  
  Xinyuan Liu, Eren Sadikoglu, Ransalu Senanayake, Lixiao Huang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10069v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Human factors research has long focused on optimizing environments, tools, and systems to account for human performance. Yet, as humanoid robots begin to share our workplaces, homes, and public spaces, the design challenge expands. We must now consider not only factors for humans but also factors for humanoids, since both will coexist and interact within the same environments. Unlike conventional machines, humanoids introduce expectations of human-like behavior, communication, and social presence, which reshape usability, trust, and safety considerations. In this article, we introduce the concept of humanoid factors as a framework structured around four pillars - physical, cognitive, social, and ethical - that shape the development of humanoids to help them effectively coexist and collaborate with humans. This framework characterizes the overlap and divergence between human capabilities and those of general-purpose humanoids powered by AI foundation models. To demonstrate our framework's practical utility, we then apply the framework to evaluate a real-world humanoid control algorithm, illustrating how conventional task completion metrics in robotics overlook key human cognitive and interaction principles. We thus position humanoid factors as a foundational framework for designing, evaluating, and governing sustained human-humanoid coexistence.

  </details>



- **Perception with Guarantees: Certified Pose Estimation via Reachability Analysis**  
  Tobias Ladner, Yasser Shoukry, Matthias Althoff  
  _2026-02-10_ · https://arxiv.org/abs/2602.10032v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Agents in cyber-physical systems are increasingly entrusted with safety-critical tasks. Ensuring safety of these agents often requires localizing the pose for subsequent actions. Pose estimates can, e.g., be obtained from various combinations of lidar sensors, cameras, and external services such as GPS. Crucially, in safety-critical domains, a rough estimate is insufficient to formally determine safety, i.e., guaranteeing safety even in the worst-case scenario, and external services might additionally not be trustworthy. We address this problem by presenting a certified pose estimation in 3D solely from a camera image and a well-known target geometry. This is realized by formally bounding the pose, which is computed by leveraging recent results from reachability analysis and formal neural network verification. Our experiments demonstrate that our approach efficiently and accurately localizes agents in both synthetic and real-world experiments.

  </details>



- **ST4VLA: Spatially Guided Training for Vision-Language-Action Models**  
  Jinhui Ye, Fangjing Wang, Ning Gao, Junqiu Yu, Yangkun Zhu, Bin Wang, Jinyu Zhang, Weiyang Jin, Yanwei Fu, Feng Zheng, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.10109v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Large vision-language models (VLMs) excel at multimodal understanding but fall short when extended to embodied tasks, where instructions must be transformed into low-level motor actions. We introduce ST4VLA, a dual-system Vision-Language-Action framework that leverages Spatial Guided Training to align action learning with spatial priors in VLMs. ST4VLA includes two stages: (i) spatial grounding pre-training, which equips the VLM with transferable priors via scalable point, box, and trajectory prediction from both web-scale and robot-specific data, and (ii) spatially guided action post-training, which encourages the model to produce richer spatial priors to guide action generation via spatial prompting. This design preserves spatial grounding during policy learning and promotes consistent optimization across spatial and action objectives. Empirically, ST4VLA achieves substantial improvements over vanilla VLA, with performance increasing from 66.1 -> 84.6 on Google Robot and from 54.7 -> 73.2 on WidowX Robot, establishing new state-of-the-art results on SimplerEnv. It also demonstrates stronger generalization to unseen objects and paraphrased instructions, as well as robustness to long-horizon perturbations in real-world settings. These results highlight scalable spatially guided training as a promising direction for robust, generalizable robot learning. Source code, data and models are released at https://internrobotics.github.io/internvla-m1.github.io/

  </details>



- **Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning**  
  Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, Yuxiong He  
  _2026-02-10_ · https://arxiv.org/abs/2602.10090v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Recent advances in large language model (LLM) have empowered autonomous agents to perform complex tasks that require multi-turn interactions with tools and environments. However, scaling such agent training is limited by the lack of diverse and reliable environments. In this paper, we propose Agent World Model (AWM), a fully synthetic environment generation pipeline. Using this pipeline, we scale to 1,000 environments covering everyday scenarios, in which agents can interact with rich toolsets (35 tools per environment on average) and obtain high-quality observations. Notably, these environments are code-driven and backed by databases, providing more reliable and consistent state transitions than environments simulated by LLMs. Moreover, they enable more efficient agent interaction compared with collecting trajectories from realistic environments. To demonstrate the effectiveness of this resource, we perform large-scale reinforcement learning for multi-turn tool-use agents. Thanks to the fully executable environments and accessible database states, we can also design reliable reward functions. Experiments on three benchmarks show that training exclusively in synthetic environments, rather than benchmark-specific ones, yields strong out-of-distribution generalization. The code is available at https://github.com/Snowflake-Labs/agent-world-model.

  </details>



- **Conformal Prediction Sets for Instance Segmentation**  
  Kerri Lu, Dan M. Kluger, Stephen Bates, Sherrie Wang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10045v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Current instance segmentation models achieve high performance on average predictions, but lack principled uncertainty quantification: their outputs are not calibrated, and there is no guarantee that a predicted mask is close to the ground truth. To address this limitation, we introduce a conformal prediction algorithm to generate adaptive confidence sets for instance segmentation. Given an image and a pixel coordinate query, our algorithm generates a confidence set of instance predictions for that pixel, with a provable guarantee for the probability that at least one of the predictions has high Intersection-Over-Union (IoU) with the true object instance mask. We apply our algorithm to instance segmentation examples in agricultural field delineation, cell segmentation, and vehicle detection. Empirically, we find that our prediction sets vary in size based on query difficulty and attain the target coverage, outperforming existing baselines such as Learn Then Test, Conformal Risk Control, and morphological dilation-based methods. We provide versions of the algorithm with asymptotic and finite sample guarantees.

  </details>



- **Optimistic World Models: Efficient Exploration in Model-Based Deep Reinforcement Learning**  
  Akshay Mete, Shahid Aamir Sheikh, Tzu-Hsiang Lin, Dileep Kalathil, P. R. Kumar  
  _2026-02-10_ · https://arxiv.org/abs/2602.10044v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Efficient exploration remains a central challenge in reinforcement learning (RL), particularly in sparse-reward environments. We introduce Optimistic World Models (OWMs), a principled and scalable framework for optimistic exploration that brings classical reward-biased maximum likelihood estimation (RBMLE) from adaptive control into deep RL. In contrast to upper confidence bound (UCB)-style exploration methods, OWMs incorporate optimism directly into model learning by augmentation with an optimistic dynamics loss that biases imagined transitions toward higher-reward outcomes. This fully gradient-based loss requires neither uncertainty estimates nor constrained optimization. Our approach is plug-and-play with existing world model frameworks, preserving scalability while requiring only minimal modifications to standard training procedures. We instantiate OWMs within two state-of-the-art world model architectures, leading to Optimistic DreamerV3 and Optimistic STORM, which demonstrate significant improvements in sample efficiency and cumulative return compared to their baseline counterparts.

  </details>



- **Learning to Detect Baked Goods with Limited Supervision**  
  Thomas H. Schmitt, Maximilian Bundscherer, Tobias Bocklet  
  _2026-02-10_ · https://arxiv.org/abs/2602.09979v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Monitoring leftover products provides valuable insights that can be used to optimize future production. This is especially important for German bakeries because freshly baked goods have a very short shelf life. Automating this process can reduce labor costs, improve accuracy, and streamline operations. We propose automating this process using an object detection model to identify baked goods from images. However, the large diversity of German baked goods makes fully supervised training prohibitively expensive and limits scalability. Although open-vocabulary detectors (e.g., OWLv2, Grounding DINO) offer lexibility, we demonstrate that they are insufficient for our task. While motivated by bakeries, our work addresses the broader challenges of deploying computer vision in industries, where tasks are specialized and annotated datasets are scarce. We compile dataset splits with varying supervision levels, covering 19 classes of baked goods. We propose two training workflows to train an object detection model with limited supervision. First, we combine OWLv2 and Grounding DINO localization with image-level supervision to train the model in a weakly supervised manner. Second, we improve viewpoint robustness by fine-tuning on video frames annotated using Segment Anything 2 as a pseudo-label propagation model. Using these workflows, we train YOLOv11 for our detection task due to its favorable speed accuracy tradeoff. Relying solely on image-level supervision, the model achieves a mean Average Precision (mAP) of 0.91. Finetuning with pseudo-labels raises model performance by 19.3% under non-ideal deployment conditions. Combining these workflows trains a model that surpasses our fully-supervised baseline model under non-ideal deployment conditions, despite relying only on image-level supervision.

  </details>



- **Contextual and Seasonal LSTMs for Time Series Anomaly Detection**  
  Lingpei Zhang, Qingming Li, Yong Yang, Jiahao Chen, Rui Zeng, Chenyang Lyu, Shouling Ji  
  _2026-02-10_ · https://arxiv.org/abs/2602.09690v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Univariate time series (UTS), where each timestamp records a single variable, serve as crucial indicators in web systems and cloud servers. Anomaly detection in UTS plays an essential role in both data mining and system reliability management. However, existing reconstruction-based and prediction-based methods struggle to capture certain subtle anomalies, particularly small point anomalies and slowly rising anomalies. To address these challenges, we propose a novel prediction-based framework named Contextual and Seasonal LSTMs (CS-LSTMs). CS-LSTMs are built upon a noise decomposition strategy and jointly leverage contextual dependencies and seasonal patterns, thereby strengthening the detection of subtle anomalies. By integrating both time-domain and frequency-domain representations, CS-LSTMs achieve more accurate modeling of periodic trends and anomaly localization. Extensive evaluations on public benchmark datasets demonstrate that CS-LSTMs consistently outperform state-of-the-art methods, highlighting their effectiveness and practical value in robust time series anomaly detection.

  </details>



- **Mitigating the Likelihood Paradox in Flow-based OOD Detection via Entropy Manipulation**  
  Donghwan Kim, Hyunsoo Yoon  
  _2026-02-10_ · https://arxiv.org/abs/2602.09581v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Deep generative models that can tractably compute input likelihoods, including normalizing flows, often assign unexpectedly high likelihoods to out-of-distribution (OOD) inputs. We mitigate this likelihood paradox by manipulating input entropy based on semantic similarity, applying stronger perturbations to inputs that are less similar to an in-distribution memory bank. We provide a theoretical analysis showing that entropy control increases the expected log-likelihood gap between in-distribution and OOD samples in favor of the in-distribution, and we explain why the procedure works without any additional training of the density model. We then evaluate our method against likelihood-based OOD detectors on standard benchmarks and find consistent AUROC improvements over baselines, supporting our explanation.

  </details>



- **Optimal Control of Microswimmers for Trajectory Tracking Using Bayesian Optimization**  
  Lucas Palazzolo, Mickaël Binois, Laëtitia Giraldi  
  _2026-02-10_ · https://arxiv.org/abs/2602.09563v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Trajectory tracking for microswimmers remains a key challenge in microrobotics, where low-Reynolds-number dynamics make control design particularly complex. In this work, we formulate the trajectory tracking problem as an optimal control problem and solve it using a combination of B-spline parametrization with Bayesian optimization, allowing the treatment of high computational costs without requiring complex gradient computations. Applied to a flagellated magnetic swimmer, the proposed method reproduces a variety of target trajectories, including biologically inspired paths observed in experimental studies. We further evaluate the approach on a three-sphere swimmer model, demonstrating that it can adapt to and partially compensate for wall-induced hydrodynamic effects. The proposed optimization strategy can be applied consistently across models of different fidelity, from low-dimensional ODE-based models to high-fidelity PDE-based simulations, showing its robustness and generality. These results highlight the potential of Bayesian optimization as a versatile tool for optimal control strategies in microscale locomotion under complex fluid-structure interactions.

  </details>


