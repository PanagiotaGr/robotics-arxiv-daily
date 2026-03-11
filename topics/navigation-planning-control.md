# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-03-11 07:08 UTC_

Total papers shown: **6**


---

- **SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments**  
  Shiyi Chen, Mingye Yang, Haiyan Mao, Jiaqi Zhang, Haiyi Liu, Shuheng He, Debing Zhang, Zihao Qiu, Chun Zhang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09460v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Efficiently training quadruped robot navigation in densely cluttered environments remains a significant challenge. Existing methods are either limited by a lack of safety and agility in simple obstacle distributions or suffer from slow locomotion in complex environments, often requiring excessively long training phases. To this end, we propose SEA-Nav (Safe, Efficient, and Agile Navigation), a reinforcement learning framework for quadruped navigation. Within diverse and dense obstacle environments, a differentiable control barrier function (CBF)-based shield constraints the navigation policy to output safe velocity commands. An adaptive collision replay mechanism and hazardous exploration rewards are introduced to increase the probability of learning from critical experiences, guiding efficient exploration and exploitation. Finally, kinematic action constraints are incorporated to ensure safe velocity commands, facilitating successful physical deployment. To the best of our knowledge, this is the first approach that achieves highly challenging quadruped navigation in the real world with minute-level training time.

  </details>



- **BEACON: Language-Conditioned Navigation Affordance Prediction under Occlusion**  
  Xinyu Gao, Gang Chen, Javier Alonso-Mora  
  _2026-03-10_ · https://arxiv.org/abs/2603.09961v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Language-conditioned local navigation requires a robot to infer a nearby traversable target location from its current observation and an open-vocabulary, relational instruction. Existing vision-language spatial grounding methods usually rely on vision-language models (VLMs) to reason in image space, producing 2D predictions tied to visible pixels. As a result, they struggle to infer target locations in occluded regions, typically caused by furniture or moving humans. To address this issue, we propose BEACON, which predicts an ego-centric Bird's-Eye View (BEV) affordance heatmap over a bounded local region including occluded areas. Given an instruction and surround-view RGB-D observations from four directions around the robot, BEACON predicts the BEV heatmap by injecting spatial cues into a VLM and fusing the VLM's output with depth-derived BEV features. Using an occlusion-aware dataset built in the Habitat simulator, we conduct detailed experimental analysis to validate both our BEV space formulation and the design choices of each module. Our method improves the accuracy averaged across geodesic thresholds by 22.74 percentage points over the state-of-the-art image-space baseline on the validation subset with occluded target locations. Our project page is: https://xin-yu-gao.github.io/beacon.

  </details>



- **Let's Reward Step-by-Step: Step-Aware Contrastive Alignment for Vision-Language Navigation in Continuous Environments**  
  Haoyuan Li, Rui Liu, Hehe Fan, Yi Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09740v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Vision-Language Navigation in Continuous Environments (VLN-CE) requires agents to learn complex reasoning from long-horizon human interactions. While Multi-modal Large Language Models (MLLMs) have driven recent progress, current training paradigms struggle to balance generalization capability, error recovery and training stability. Specifically, (i) policies derived from SFT suffer from compounding errors, struggling to recover from out-of-distribution states, and (ii) Reinforcement Fine-Tuning (RFT) methods e.g. GRPO are bottlenecked by sparse outcome rewards. Their binary feedback fails to assign credit to individual steps, leading to gradient signal collapse in failure dominant batches. To address these challenges, we introduce Step-Aware Contrastive Alignment (SACA), a framework designed to extract dense supervision from imperfect trajectories. At its core, the Perception-Grounded Step-Aware auditor evaluates progress step-by-step, disentangling failed trajectories into valid prefixes and exact divergence points. Leveraging these signals, Scenario-Conditioned Group Construction mechanism dynamically routes batches to specialized resampling and optimization strategies. Extensive experiments on VLN-CE benchmarks demonstrate that SACA achieves state-of-the-art performance.

  </details>



- **Declarative Scenario-based Testing with RoadLogic**  
  Ezio Bartocci, Alessio Gambi, Felix Gigler, Cristinel Mateis, Dejan Ničković  
  _2026-03-10_ · https://arxiv.org/abs/2603.09455v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Scenario-based testing is a key method for cost-effective and safe validation of autonomous vehicles (AVs). Existing approaches rely on imperative scenario definitions, requiring developers to manually enumerate numerous variants to achieve coverage. Declarative languages, such as OpenSCENARIO DSL (OS2), raise the abstraction level but lack systematic methods for instantiating concrete, specification-compliant scenarios as simulations. To our knowledge, currently, no open-source solution provides this capability. We present RoadLogic that bridges declarative OS2 specifications and executable simulations. It uses Answer Set Programming to generate abstract plans satisfying scenario constraints, motion planning to refine the plans into feasible trajectories, and specification-based monitoring to verify correctness. We evaluate RoadLogic on instantiating representative OS2 scenarios as simulations in the CommonRoad framework. Results show that RoadLogic consistently produces realistic, specification-satisfying simulations within minutes and captures diverse behavioral variants through parameter sampling, thus opening the door to systematic scenario-based testing for autonomous driving systems.

  </details>



- **An Optimal Control Approach To Transformer Training**  
  Kağan Akman, Naci Saldı, Serdar Yüksel  
  _2026-03-10_ · https://arxiv.org/abs/2603.09571v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  In this paper, we develop a rigorous optimal control-theoretic approach to Transformer training that respects key structural constraints such as (i) realized-input-independence during execution, (ii) the ensemble control nature of the problem, and (iii) positional dependence. We model the Transformer architecture as a discrete-time controlled particle system with shared actions, exhibiting noise-free McKean-Vlasov dynamics. While the resulting dynamics is not Markovian, we show that lifting it to probability measures produces a fully-observed Markov decision process (MDP). Positional encodings are incorporated into the state space to preserve the sequence order under lifting. Using the dynamic programming principle, we establish the existence of globally optimal policies under mild assumptions of compactness. We further prove that closed-loop policies in the lifted is equivalent to an initial-distribution dependent open-loop policy, which are realized-input-independent and compatible with standard Transformer training. To train a Transformer, we propose a triply quantized training procedure for the lifted MDP by quantizing the state space, the space of probability measures, and the action space, and show that any optimal policy for the triply quantized model is near-optimal for the original training problem. Finally, we establish stability and empirical consistency properties of the lifted model by showing that the value function is continuous with respect to the perturbations of the initial empirical measures and convergence of policies as the data size increases. This approach provides a globally optimal and robust alternative to gradient-based training without requiring smoothness or convexity.

  </details>



- **Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation**  
  Won Shik Jang, Ue-Hwan Kim  
  _2026-03-10_ · https://arxiv.org/abs/2603.09506v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Text-goal instance navigation (TGIN) asks an agent to resolve a single, free-form description into actions that reach the correct object instance among same-category distractors. We present \textit{Context-Nav} that elevates long, contextual captions from a local matching cue to a global exploration prior and verifies candidates through 3D spatial reasoning. First, we compute dense text-image alignments for a value map that ranks frontiers -- guiding exploration toward regions consistent with the entire description rather than early detections. Second, upon observing a candidate, we perform a viewpoint-aware relation check: the agent samples plausible observer poses, aligns local frames, and accepts a target only if the spatial relations can be satisfied from at least one viewpoint. The pipeline requires no task-specific training or fine-tuning; we attain state-of-the-art performance on InstanceNav and CoIN-Bench. Ablations show that (i) encoding full captions into the value map avoids wasted motion and (ii) explicit, viewpoint-aware 3D verification prevents semantically plausible but incorrect stops. This suggests that geometry-grounded spatial reasoning is a scalable alternative to heavy policy training or human-in-the-loop interaction for fine-grained instance disambiguation in cluttered 3D scenes.

  </details>


