# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-03 07:06 UTC_

Total papers shown: **7**


---

- **FD-VLA: Force-Distilled Vision-Language-Action Model for Contact-Rich Manipulation**  
  Ruiteng Zhao, Wenshuo Wang, Yicheng Ma, Xiaocong Li, Francis E. H. Tay, Marcelo H. Ang, Haiyue Zhu  
  _2026-02-02_ · https://arxiv.org/abs/2602.02142v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Force sensing is a crucial modality for Vision-Language-Action (VLA) frameworks, as it enables fine-grained perception and dexterous manipulation in contact-rich tasks. We present Force-Distilled VLA (FD-VLA), a novel framework that integrates force awareness into contact-rich manipulation without relying on physical force sensors. The core of our approach is a Force Distillation Module (FDM), which distills force by mapping a learnable query token, conditioned on visual observations and robot states, into a predicted force token aligned with the latent representation of actual force signals. During inference, this distilled force token is injected into the pretrained VLM, enabling force-aware reasoning while preserving the integrity of its vision-language semantics. This design provides two key benefits: first, it allows practical deployment across a wide range of robots that lack expensive or fragile force-torque sensors, thereby reducing hardware cost and complexity; second, the FDM introduces an additional force-vision-state fusion prior to the VLM, which improves cross-modal alignment and enhances perception-action robustness in contact-rich scenarios. Surprisingly, our physical experiments show that the distilled force token outperforms direct sensor force measurements as well as other baselines, which highlights the effectiveness of this force-distilled VLA approach.

  </details>



- **PRISM: Performer RS-IMLE for Single-pass Multisensory Imitation Learning**  
  Amisha Bhaskar, Pratap Tokekar, Stefano Di Cairano, Alexander Schperberg  
  _2026-02-02_ · https://arxiv.org/abs/2602.02396v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic imitation learning typically requires models that capture multimodal action distributions while operating at real-time control rates and accommodating multiple sensing modalities. Although recent generative approaches such as diffusion models, flow matching, and Implicit Maximum Likelihood Estimation (IMLE) have achieved promising results, they often satisfy only a subset of these requirements. To address this, we introduce PRISM, a single-pass policy based on a batch-global rejection-sampling variant of IMLE. PRISM couples a temporal multisensory encoder (integrating RGB, depth, tactile, audio, and proprioception) with a linear-attention generator using a Performer architecture. We demonstrate the efficacy of PRISM on a diverse real-world hardware suite, including loco-manipulation using a Unitree Go2 with a 7-DoF arm D1 and tabletop manipulation with a UR5 manipulator. Across challenging physical tasks such as pre-manipulation parking, high-precision insertion, and multi-object pick-and-place, PRISM outperforms state-of-the-art diffusion policies by 10-25% in success rate while maintaining high-frequency (30-50 Hz) closed-loop control. We further validate our approach on large-scale simulation benchmarks, including CALVIN, MetaWorld, and Robomimic. In CALVIN (10% data split), PRISM improves success rates by approximately 25% over diffusion and approximately 20% over flow matching, while simultaneously reducing trajectory jerk by 20x-50x. These results position PRISM as a fast, accurate, and multisensory imitation policy that retains multimodal action coverage without the latency of iterative sampling.

  </details>



- **Flow Policy Gradients for Robot Control**  
  Brent Yi, Hongsuk Choi, Himanshu Gaurav Singh, Xiaoyu Huang, Takara E. Truong, Carmelo Sferrazza, Yi Ma, Rocky Duan, Pieter Abbeel, Guanya Shi, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02481v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Likelihood-based policy gradient methods are the dominant approach for training robot control policies from rewards. These methods rely on differentiable action likelihoods, which constrain policy outputs to simple distributions like Gaussians. In this work, we show how flow matching policy gradients -- a recent framework that bypasses likelihood computation -- can be made effective for training and fine-tuning more expressive policies in challenging robot control settings. We introduce an improved objective that enables success in legged locomotion, humanoid motion tracking, and manipulation tasks, as well as robust sim-to-real transfer on two humanoid robots. We then present ablations and analysis on training dynamics. Results show how policies can exploit the flow representation for exploration when training from scratch, as well as improved fine-tuning robustness over baselines.

  </details>



- **SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation**  
  Mu Huang, Hui Wang, Kerui Ren, Linning Xu, Yunsong Zhou, Mulin Yu, Bo Dai, Jiangmiao Pang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02402v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Simulating deformable objects under rich interactions remains a fundamental challenge for real-to-sim robot manipulation, with dynamics jointly driven by environmental effects and robot actions. Existing simulators rely on predefined physics or data-driven dynamics without robot-conditioned control, limiting accuracy, stability, and generalization. This paper presents SoMA, a 3D Gaussian Splat simulator for soft-body manipulation. SoMA couples deformable dynamics, environmental forces, and robot joint actions in a unified latent neural space for end-to-end real-to-sim simulation. Modeling interactions over learned Gaussian splats enables controllable, stable long-horizon manipulation and generalization beyond observed trajectories without predefined physical models. SoMA improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding.

  </details>



- **World-Gymnast: Training Robots with Reinforcement Learning in a World Model**  
  Ansh Kumar Sharma, Yixiang Sun, Ninghao Lu, Yunzhe Zhang, Jiarao Liu, Sherry Yang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02454v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robot learning from interacting with the physical world is fundamentally bottlenecked by the cost of physical interaction. The two alternatives, supervised finetuning (SFT) from expert demonstrations and reinforcement learning (RL) in a software-based simulator, are limited by the amount of expert data available and the sim-to-real gap for manipulation. With the recent emergence of world models learned from real-world video-action data, we ask the question of whether training a policy in a world model can be more effective than supervised learning or software simulation in achieving better real-robot performance. We propose World-Gymnast, which performs RL finetuning of a vision-language-action (VLA) policy by rolling out the policy in an action-conditioned video world model and rewarding the rollouts with a vision-language model (VLM). On the Bridge robot setup, World-Gymnast outperforms SFT by as much as 18x and outperforms software simulator by as much as 2x. More importantly, World-Gymnast demonstrates intriguing capabilities of RL with a world model, including training on diverse language instructions and novel scenes from the world model, test-time training in a novel scene, and online iterative world model and policy improvement. Our results suggest learning a world model and training robot policies in the cloud could be the key to bridging the gap between robots that work in demonstrations and robots that can work in anyone's household.

  </details>



- **CIEC: Coupling Implicit and Explicit Cues for Multimodal Weakly Supervised Manipulation Localization**  
  Xinquan Yu, Wei Lu, Xiangyang Luo  
  _2026-02-02_ · https://arxiv.org/abs/2602.02175v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  To mitigate the threat of misinformation, multimodal manipulation localization has garnered growing attention. Consider that current methods rely on costly and time-consuming fine-grained annotations, such as patch/token-level annotations. This paper proposes a novel framework named Coupling Implicit and Explicit Cues (CIEC), which aims to achieve multimodal weakly-supervised manipulation localization for image-text pairs utilizing only coarse-grained image/sentence-level annotations. It comprises two branches, image-based and text-based weakly-supervised localization. For the former, we devise the Textual-guidance Refine Patch Selection (TRPS) module. It integrates forgery cues from both visual and textual perspectives to lock onto suspicious regions aided by spatial priors. Followed by the background silencing and spatial contrast constraints to suppress interference from irrelevant areas. For the latter, we devise the Visual-deviation Calibrated Token Grounding (VCTG) module. It focuses on meaningful content words and leverages relative visual bias to assist token localization. Followed by the asymmetric sparse and semantic consistency constraints to mitigate label noise and ensure reliability. Extensive experiments demonstrate the effectiveness of our CIEC, yielding results comparable to fully supervised methods on several evaluation metrics.

  </details>



- **The Verification Crisis: Expert Perceptions of GenAI Disinformation and the Case for Reproducible Provenance**  
  Alexander Loth, Martin Kappes, Marc-Oliver Pahl  
  _2026-02-02_ · https://arxiv.org/abs/2602.02100v1 · `cs.CY`  
  <details><summary>Abstract</summary>

  The growth of Generative Artificial Intelligence (GenAI) has shifted disinformation production from manual fabrication to automated, large-scale manipulation. This article presents findings from the first wave of a longitudinal expert perception survey (N=21) involving AI researchers, policymakers, and disinformation specialists. It examines the perceived severity of multimodal threats -- text, image, audio, and video -- and evaluates current mitigation strategies. Results indicate that while deepfake video presents immediate "shock" value, large-scale text generation poses a systemic risk of "epistemic fragmentation" and "synthetic consensus," particularly in the political domain. The survey reveals skepticism about technical detection tools, with experts favoring provenance standards and regulatory frameworks despite implementation barriers. GenAI disinformation research requires reproducible methods. The current challenge is measurement: without standardized benchmarks and reproducibility checklists, tracking or countering synthetic media remains difficult. We propose treating information integrity as an infrastructure with rigor in data provenance and methodological reproducibility.

  </details>


