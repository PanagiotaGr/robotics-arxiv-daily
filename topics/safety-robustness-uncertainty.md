# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-02-28 06:54 UTC_

Total papers shown: **12**


---

- **Towards Intelligible Human-Robot Interaction: An Active Inference Approach to Occluded Pedestrian Scenarios**  
  Kai Chen, Yuyao Huang, Guang Chen  
  _2026-02-26_ · https://arxiv.org/abs/2602.23109v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The sudden appearance of occluded pedestrians presents a critical safety challenge in autonomous driving. Conventional rule-based or purely data-driven approaches struggle with the inherent high uncertainty of these long-tail scenarios. To tackle this challenge, we propose a novel framework grounded in Active Inference, which endows the agent with a human-like, belief-driven mechanism. Our framework leverages a Rao-Blackwellized Particle Filter (RBPF) to efficiently estimate the pedestrian's hybrid state. To emulate human-like cognitive processes under uncertainty, we introduce a Conditional Belief Reset mechanism and a Hypothesis Injection technique to explicitly model beliefs about the pedestrian's multiple latent intentions. Planning is achieved via a Cross-Entropy Method (CEM) enhanced Model Predictive Path Integral (MPPI) controller, which synergizes the efficient, iterative search of CEM with the inherent robustness of MPPI. Simulation experiments demonstrate that our approach significantly reduces the collision rate compared to reactive, rule-based, and reinforcement learning (RL) baselines, while also exhibiting explainable and human-like driving behavior that reflects the agent's internal belief state.

  </details>



- **WaterVideoQA: ASV-Centric Perception and Rule-Compliant Reasoning via Multi-Modal Agents**  
  Runwei Guan, Shaofeng Liang, Ningwei Ouyang, Weichen Fei, Shanliang Yao, Wei Dai, Chenhao Ge, Penglei Sun, Xiaohui Zhu, Tao Huang, et al.  
  _2026-02-26_ · https://arxiv.org/abs/2602.22923v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  While autonomous navigation has achieved remarkable success in passive perception (e.g., object detection and segmentation), it remains fundamentally constrained by a void in knowledge-driven, interactive environmental cognition. In the high-stakes domain of maritime navigation, the ability to bridge the gap between raw visual perception and complex cognitive reasoning is not merely an enhancement but a critical prerequisite for Autonomous Surface Vessels to execute safe and precise maneuvers. To this end, we present WaterVideoQA, the first large-scale, comprehensive Video Question Answering benchmark specifically engineered for all-waterway environments. This benchmark encompasses 3,029 video clips across six distinct waterway categories, integrating multifaceted variables such as volatile lighting and dynamic weather to rigorously stress-test ASV capabilities across a five-tier hierarchical cognitive framework. Furthermore, we introduce NaviMind, a pioneering multi-agent neuro-symbolic system designed for open-ended maritime reasoning. By synergizing Adaptive Semantic Routing, Situation-Aware Hierarchical Reasoning, and Autonomous Self-Reflective Verification, NaviMind transitions ASVs from superficial pattern matching to regulation-compliant, interpretable decision-making. Experimental results demonstrate that our framework significantly transcends existing baselines, establishing a new paradigm for intelligent, trustworthy interaction in dynamic maritime environments.

  </details>



- **Deep ensemble graph neural networks for probabilistic cosmic-ray direction and energy reconstruction in autonomous radio arrays**  
  Arsène Ferrière, Aurélien Benoit-Lévy, Olivier Martineau-Huynh, Matías Tueros  
  _2026-02-26_ · https://arxiv.org/abs/2602.23321v1 · `astro-ph.IM`  
  <details><summary>Abstract</summary>

  Using advanced machine learning techniques, we developed a method for reconstructing precisely the arrival direction and energy of ultra-high-energy cosmic rays from the voltage traces they induced on ground-based radio detector arrays. In our approach, triggered antennas are represented as a graph structure, which serves as input for a graph neural network (GNN). By incorporating physical knowledge into both the GNN architecture and the input data, we improve the precision and reduce the required size of the training set with respect to a fully data-driven approach. This method achieves an angular resolution of 0.092° and an electromagnetic energy reconstruction resolution of 16.4% on simulated data with realistic noise conditions. We also employ uncertainty estimation methods to enhance the reliability of our predictions, quantifying the confidence of the GNN's outputs and providing confidence intervals for both direction and energy reconstruction. Finally, we investigate strategies to verify the model's consistency and robustness under real life variations, with the goal of identifying scenarios in which predictions remain reliable despite domain shifts between simulation and reality.

  </details>



- **Accelerated Online Risk-Averse Policy Evaluation in POMDPs with Theoretical Guarantees and Novel CVaR Bounds**  
  Yaacov Pariente, Vadim Indelman  
  _2026-02-26_ · https://arxiv.org/abs/2602.23073v1 · `math.ST`  
  <details><summary>Abstract</summary>

  Risk-averse decision-making under uncertainty in partially observable domains is a central challenge in artificial intelligence and is essential for developing reliable autonomous agents. The formal framework for such problems is the partially observable Markov decision process (POMDP), where risk sensitivity is introduced through a risk measure applied to the value function, with Conditional Value-at-Risk (CVaR) being a particularly significant criterion. However, solving POMDPs is computationally intractable in general, and approximate methods rely on computationally expensive simulations of future agent trajectories. This work introduces a theoretical framework for accelerating CVaR value function evaluation in POMDPs with formal performance guarantees. We derive new bounds on the CVaR of a random variable X using an auxiliary random variable Y, under assumptions relating their cumulative distribution and density functions; these bounds yield interpretable concentration inequalities and converge as the distributional discrepancy vanishes. Building on this, we establish upper and lower bounds on the CVaR value function computable from a simplified belief-MDP, accommodating general simplifications of the transition dynamics. We develop estimators for these bounds within a particle-belief MDP framework with probabilistic guarantees, and employ them for acceleration via action elimination: actions whose bounds indicate suboptimality under the simplified model are safely discarded while ensuring consistency with the original POMDP. Empirical evaluation across multiple POMDP domains confirms that the bounds reliably separate safe from dangerous policies while achieving substantial computational speedups under the simplified model.

  </details>



- **Risk-Aware World Model Predictive Control for Generalizable End-to-End Autonomous Driving**  
  Jiangxin Sun, Feng Xue, Teng Long, Chang Liu, Jian-Fang Hu, Wei-Shi Zheng, Nicu Sebe  
  _2026-02-26_ · https://arxiv.org/abs/2602.23259v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  With advances in imitation learning (IL) and large-scale driving datasets, end-to-end autonomous driving (E2E-AD) has made great progress recently. Currently, IL-based methods have become a mainstream paradigm: models rely on standard driving behaviors given by experts, and learn to minimize the discrepancy between their actions and expert actions. However, this objective of "only driving like the expert" suffers from limited generalization: when encountering rare or unseen long-tail scenarios outside the distribution of expert demonstrations, models tend to produce unsafe decisions in the absence of prior experience. This raises a fundamental question: Can an E2E-AD system make reliable decisions without any expert action supervision? Motivated by this, we propose a unified framework named Risk-aware World Model Predictive Control (RaWMPC) to address this generalization dilemma through robust control, without reliance on expert demonstrations. Practically, RaWMPC leverages a world model to predict the consequences of multiple candidate actions and selects low-risk actions through explicit risk evaluation. To endow the world model with the ability to predict the outcomes of risky driving behaviors, we design a risk-aware interaction strategy that systematically exposes the world model to hazardous behaviors, making catastrophic outcomes predictable and thus avoidable. Furthermore, to generate low-risk candidate actions at test time, we introduce a self-evaluation distillation method to distill riskavoidance capabilities from the well-trained world model into a generative action proposal network without any expert demonstration. Extensive experiments show that RaWMPC outperforms state-of-the-art methods in both in-distribution and out-of-distribution scenarios, while providing superior decision interpretability.

  </details>



- **OSDaR-AR: Enhancing Railway Perception Datasets via Multi-modal Augmented Reality**  
  Federico Nesti, Gianluca D'Amico, Mauro Marinoni, Giorgio Buttazzo  
  _2026-02-26_ · https://arxiv.org/abs/2602.22920v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Although deep learning has significantly advanced the perception capabilities of intelligent transportation systems, railway applications continue to suffer from a scarcity of high-quality, annotated data for safety-critical tasks like obstacle detection. While photorealistic simulators offer a solution, they often struggle with the ``sim-to-real" gap; conversely, simple image-masking techniques lack the spatio-temporal coherence required to obtain augmented single- and multi-frame scenes with the correct appearance and dimensions. This paper introduces a multi-modal augmented reality framework designed to bridge this gap by integrating photorealistic virtual objects into real-world railway sequences from the OSDaR23 dataset. Utilizing Unreal Engine 5 features, our pipeline leverages LiDAR point-clouds and INS/GNSS data to ensure accurate object placement and temporal stability across RGB frames. This paper also proposes a segmentation-based refinement strategy for INS/GNSS data to significantly improve the realism of the augmented sequences, as confirmed by the comparative study presented in the paper. Carefully designed augmented sequences are collected to produce OSDaR-AR, a public dataset designed to support the development of next-generation railway perception systems. The dataset is available at the following page: https://syndra.retis.santannapisa.it/osdarar.html

  </details>



- **Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation**  
  Ismaël Zighed, Andrea Nóvoa, Luca Magri, Taraneh Sayadi  
  _2026-02-26_ · https://arxiv.org/abs/2602.23188v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time and relying solely on sparse observations of the full system. The architecture employs an encode-process-decode structure: a Variational Autoencoder (VAE) to perform dimensionality reduction, and a transformer network to evolve the latent states and model the dynamics. The ROM is parameterized by an external control variable, the Reynolds number in the Navier-Stokes setting, with the transformer exploiting attention mechanisms to capture both temporal dependencies and parameter effects. The probabilistic VAE enables stochastic sampling of trajectory ensembles, providing predictive means and uncertainty quantification through the first two moments. After initial training on a limited set of dynamical regimes, the model is adapted to out-of-sample parameter regions using only sparse data. Its probabilistic formulation naturally supports ensemble generation, which we employ within an ensemble Kalman filtering framework to assimilate data and reconstruct full-state trajectories from minimal observations. We further show that, for the dynamical system considered, the dominant source of error in out-of-sample forecasts stems from distortions of the latent manifold rather than changes in the latent dynamics. Consequently, retraining can be limited to the autoencoder, allowing for a lightweight, computationally efficient, real-time adaptation procedure with very sparse fine-tuning data.

  </details>



- **On Sample-Efficient Generalized Planning via Learned Transition Models**  
  Nitin Gupta, Vishal Pallagani, John A. Aydin, Biplav Srivastava  
  _2026-02-26_ · https://arxiv.org/abs/2602.23148v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Generalized planning studies the construction of solution strategies that generalize across families of planning problems sharing a common domain model, formally defined by a transition function $γ: S \times A \rightarrow S$. Classical approaches achieve such generalization through symbolic abstractions and explicit reasoning over $γ$. In contrast, recent Transformer-based planners, such as PlanGPT and Plansformer, largely cast generalized planning as direct action-sequence prediction, bypassing explicit transition modeling. While effective on in-distribution instances, these approaches typically require large datasets and model sizes, and often suffer from state drift in long-horizon settings due to the absence of explicit world-state evolution. In this work, we formulate generalized planning as a transition-model learning problem, in which a neural model explicitly approximates the successor-state function $\hatγ \approx γ$ and generates plans by rolling out symbolic state trajectories. Instead of predicting actions directly, the model autoregressively predicts intermediate world states, thereby learning the domain dynamics as an implicit world model. To study size-invariant generalization and sample efficiency, we systematically evaluate multiple state representations and neural architectures, including relational graph encodings. Our results show that learning explicit transition models yields higher out-of-distribution satisficing-plan success than direct action-sequence prediction in multiple domains, while achieving these gains with significantly fewer training instances and smaller models. This is an extended version of a short paper accepted at ICAPS 2026 under the same title.

  </details>



- **From Agnostic to Specific: Latent Preference Diffusion for Multi-Behavior Sequential Recommendation**  
  Ruochen Yang, Xiaodong Li, Jiawei Sheng, Jiangxia Cao, Xinkui Lin, Shen Wang, Shuang Yang, Zhaojie Liu, Tingwen Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23132v1 · `cs.IR`  
  <details><summary>Abstract</summary>

  Multi-behavior sequential recommendation (MBSR) aims to learn the dynamic and heterogeneous interactions of users' multi-behavior sequences, so as to capture user preferences under target behavior for the next interacted item prediction. Unlike previous methods that adopt unidirectional modeling by mapping auxiliary behaviors to target behavior, recent concerns are shifting from behavior-fixed to behavior-specific recommendation. However, these methods still ignore the user's latent preference that underlying decision-making, leading to suboptimal solutions. Meanwhile, due to the asymmetric deterministic between items and behaviors, discriminative paradigm based on preference scoring is unsuitable to capture the uncertainty from low-entropy behaviors to high-entropy items, failing to provide efficient and diverse recommendation. To address these challenges, we propose \textbf{FatsMB}, a framework based diffusion model that guides preference generation \textit{\textbf{F}rom Behavior-\textbf{A}gnostic \textbf{T}o Behavior-\textbf{S}pecific} in latent spaces, enabling diverse and accurate \textit{\textbf{M}ulti-\textbf{B}ehavior Sequential Recommendation}. Specifically, we design a Multi-Behavior AutoEncoder (MBAE) to construct a unified user latent preference space, facilitating interaction and collaboration across Behaviors, within Behavior-aware RoPE (BaRoPE) employed for multiple information fusion. Subsequently, we conduct target behavior-specific preference transfer in the latent space, enriching with informative priors. A Multi-Condition Guided Layer Normalization (MCGLN) is introduced for the denoising. Extensive experiments on real-world datasets demonstrate the effectiveness of our model.

  </details>



- **An Empirical Analysis of Cooperative Perception for Occlusion Risk Mitigation**  
  Aihong Wang, Tenghui Xie, Fuxi Wen, Jun Li  
  _2026-02-26_ · https://arxiv.org/abs/2602.23051v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Occlusions present a significant challenge for connected and automated vehicles, as they can obscure critical road users from perception systems. Traditional risk metrics often fail to capture the cumulative nature of these threats over time adequately. In this paper, we propose a novel and universal risk assessment metric, the Risk of Tracking Loss (RTL), which aggregates instantaneous risk intensity throughout occluded periods. This provides a holistic risk profile that encompasses both high-intensity, short-term threats and prolonged exposure. Utilizing diverse and high-fidelity real-world datasets, a large-scale statistical analysis is conducted to characterize occlusion risk and validate the effectiveness of the proposed metric. The metric is applied to evaluate different vehicle-to-everything (V2X) deployment strategies. Our study shows that full V2X penetration theoretically eliminates this risk, the reduction is highly nonlinear; a substantial statistical benefit requires a high penetration threshold of 75-90%. To overcome this limitation, we propose a novel asymmetric communication framework that allows even non-connected vehicles to receive warnings. Experimental results demonstrate that this paradigm achieves better risk mitigation performance. We found that our approach at 25% penetration outperforms the traditional symmetric model at 75%, and benefits saturate at only 50% penetration. This work provides a crucial risk assessment metric and a cost-effective, strategic roadmap for accelerating the safety benefits of V2X deployment.

  </details>



- **Digital Twin-Based Beamforming for Interference Mitigation in AF Relay MIMO Systems**  
  Alexander Bonora, Anna V. Guglielmi, Davide Scazzoli, Marco Giordani, Maurizio Magarini, Vineeth Teeda, Stefano Tomasin  
  _2026-02-26_ · https://arxiv.org/abs/2602.22991v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Beamforming in multiple-input multiple-output (MIMO) systems should take interference mitigation into account. However, for beamform design, accurate channel state information (CSI) is needed, which is often difficult to obtain due to channel variability, feedback overhead, or hardware constraints. For example, amplify-and-forward (AF) relays passively forward signals without measurement, precluding full CSI acquisition to and from the relay. To address these issues, this paper introduces a novel prediction-assisted optimization (PAO) framework for beamform design in AF relay-assisted multiuser MIMO systems. The proposed solution in the AF relay aims at maximizing the signal-plus-interference-to-noise ratio (SINR). Unlike other methods, PAO relies solely on received power measurements, making it suitable for scenarios where CSI is unreliable or unavailable. PAO consists of two stages: a supervised-learning-based neural network (NN) that predicts the positions of transmitters using signal observations, and an optimization algorithm, guided by a digital twin (DT), that iteratively refines the beam direction of the relay in a simulated radio environment. As a key contribution, we validate the proposed framework using realistic measurements collected on a custom-built experimental millimeter wave (mmWave) platform, which enables training of the NN model under practical wireless conditions. The estimated information is then used to update the digital twin with knowledge of the surrounding environment, enabling online optimization. Numerical results show the trade-off between localization accuracy and beamforming performance and confirm that PAO maintains robustness even in the presence of localization errors while reducing the need for real-world measurements.

  </details>



- **Decentralized Ranking Aggregation: Gossip Algorithms for Borda and Copeland Consensus**  
  Anna Van Elst, Kerrian Le Caillec, Igor Colin, Stephan Clémençon  
  _2026-02-26_ · https://arxiv.org/abs/2602.22847v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The concept of ranking aggregation plays a central role in preference analysis, and numerous algorithms for calculating median rankings, often originating in social choice theory, have been documented in the literature, offering theoretical guarantees in a centralized setting, i.e., when all the ranking data to be aggregated can be brought together in a single computing unit. For many technologies (e.g. peer-to-peer networks, IoT, multi-agent systems), extending the ability to calculate consensus rankings with guarantees in a decentralized setting, i.e., when preference data is initially distributed across a communicating network, remains a major methodological challenge. Indeed, in recent years, the literature on decentralized computation has mainly focused on computing or optimizing statistics such as arithmetic means using gossip algorithms. The purpose of this article is precisely to study how to achieve reliable consensus on collective rankings using classical rules (e.g. Borda, Copeland) in a decentralized setting, thereby raising new questions, robustness to corrupted nodes, and scalability through reduced communication costs in particular. The approach proposed and analyzed here relies on random gossip communication, allowing autonomous agents to compute global ranking consensus using only local interactions, without coordination or central authority. We provide rigorous convergence guarantees, including explicit rate bounds, for the Borda and Copeland consensus methods. Beyond these rules, we also provide a decentralized implementation of consensus according to the median rank rule and local Kemenization. Extensive empirical evaluations on various network topologies and real and synthetic ranking datasets demonstrate that our algorithms converge quickly and reliably to the correct ranking aggregation.

  </details>


