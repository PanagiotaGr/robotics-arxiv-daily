# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-28 06:54 UTC_

Total papers shown: **3**


---

- **Decentralized Ranking Aggregation: Gossip Algorithms for Borda and Copeland Consensus**  
  Anna Van Elst, Kerrian Le Caillec, Igor Colin, Stephan Clémençon  
  _2026-02-26_ · https://arxiv.org/abs/2602.22847v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The concept of ranking aggregation plays a central role in preference analysis, and numerous algorithms for calculating median rankings, often originating in social choice theory, have been documented in the literature, offering theoretical guarantees in a centralized setting, i.e., when all the ranking data to be aggregated can be brought together in a single computing unit. For many technologies (e.g. peer-to-peer networks, IoT, multi-agent systems), extending the ability to calculate consensus rankings with guarantees in a decentralized setting, i.e., when preference data is initially distributed across a communicating network, remains a major methodological challenge. Indeed, in recent years, the literature on decentralized computation has mainly focused on computing or optimizing statistics such as arithmetic means using gossip algorithms. The purpose of this article is precisely to study how to achieve reliable consensus on collective rankings using classical rules (e.g. Borda, Copeland) in a decentralized setting, thereby raising new questions, robustness to corrupted nodes, and scalability through reduced communication costs in particular. The approach proposed and analyzed here relies on random gossip communication, allowing autonomous agents to compute global ranking consensus using only local interactions, without coordination or central authority. We provide rigorous convergence guarantees, including explicit rate bounds, for the Borda and Copeland consensus methods. Beyond these rules, we also provide a decentralized implementation of consensus according to the median rank rule and local Kemenization. Extensive empirical evaluations on various network topologies and real and synthetic ranking datasets demonstrate that our algorithms converge quickly and reliably to the correct ranking aggregation.

  </details>



- **InCoM: Intent-Driven Perception and Structured Coordination for Whole-Body Mobile Manipulation**  
  Jiahao Liu, Cui Wenbo, Haoran Li, Dongbin Zhao  
  _2026-02-26_ · https://arxiv.org/abs/2602.23024v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Whole-body mobile manipulation is a fundamental capability for general-purpose robotic agents, requiring both coordinated control of the mobile base and manipulator and robust perception under dynamically changing viewpoints. However, existing approaches face two key challenges: strong coupling between base and arm actions complicates whole-body control optimization, and perceptual attention is often poorly allocated as viewpoints shift during mobile manipulation. We propose InCoM, an intent-driven perception and structured coordination framework for whole-body mobile manipulation. InCoM infers latent motion intent to dynamically reweight multi-scale perceptual features, enabling stage-adaptive allocation of perceptual attention. To support robust cross-modal perception, InCoM further incorporates a geometric-semantic structured alignment mechanism that enhances multimodal correspondence. On the control side, we design a decoupled coordinated flow matching action decoder that explicitly models coordinated base-arm action generation, alleviating optimization difficulties caused by control coupling. Without access to privileged perceptual information, InCoM outperforms state-of-the-art methods on three ManiSkill-HAB scenarios by 28.2%, 26.1%, and 23.6% in success rate, demonstrating strong effectiveness for whole-body mobile manipulation.

  </details>



- **Model Agreement via Anchoring**  
  Eric Eaton, Surbhi Goel, Marcel Hussing, Michael Kearns, Aaron Roth, Sikata Bela Sengupta, Jessica Sorrell  
  _2026-02-26_ · https://arxiv.org/abs/2602.23360v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Numerous lines of aim to control $\textit{model disagreement}$ -- the extent to which two machine learning models disagree in their predictions. We adopt a simple and standard notion of model disagreement in real-valued prediction problems, namely the expected squared difference in predictions between two models trained on independent samples, without any coordination of the training processes. We would like to be able to drive disagreement to zero with some natural parameter(s) of the training procedure using analyses that can be applied to existing training methodologies. We develop a simple general technique for proving bounds on independent model disagreement based on $\textit{anchoring}$ to the average of two models within the analysis. We then apply this technique to prove disagreement bounds for four commonly used machine learning algorithms: (1) stacked aggregation over an arbitrary model class (where disagreement is driven to 0 with the number of models $k$ being stacked) (2) gradient boosting (where disagreement is driven to 0 with the number of iterations $k$) (3) neural network training with architecture search (where disagreement is driven to 0 with the size $n$ of the architecture being optimized over) and (4) regression tree training over all regression trees of fixed depth (where disagreement is driven to 0 with the depth $d$ of the tree architecture). For clarity, we work out our initial bounds in the setting of one-dimensional regression with squared error loss -- but then show that all of our results generalize to multi-dimensional regression with any strongly convex loss.

  </details>


