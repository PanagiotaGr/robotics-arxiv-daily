# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-01-22 06:51 UTC_

Total papers shown: **7**


---

- **HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation**  
  Yara Mahmoud, Yasheerah Yaqoot, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-21_ · https://arxiv.org/abs/2601.14874v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Humanoid robots must adapt their contact behavior to diverse objects and tasks, yet most controllers rely on fixed, hand-tuned impedance gains and gripper settings. This paper introduces HumanoidVLM, a vision-language driven retrieval framework that enables the Unitree G1 humanoid to select task-appropriate Cartesian impedance parameters and gripper configurations directly from an egocentric RGB image. The system couples a vision-language model for semantic task inference with a FAISS-based Retrieval-Augmented Generation (RAG) module that retrieves experimentally validated stiffness-damping pairs and object-specific grasp angles from two custom databases, and executes them through a task-space impedance controller for compliant manipulation. We evaluate HumanoidVLM on 14 visual scenarios and achieve a retrieval accuracy of 93%. Real-world experiments show stable interaction dynamics, with z-axis tracking errors typically within 1-3.5 cm and virtual forces consistent with task-dependent impedance settings. These results demonstrate the feasibility of linking semantic perception with retrieval-based control as an interpretable path toward adaptive humanoid manipulation.

  </details>



- **Moving Beyond Compliance in Soft-Robotic Catheters Through Modularity for Precision Therapies**  
  B. Calmé, N. J. Greenidge, A. Metcalf, A. Bacchetti, G. Loza, D. Kpeglo, P. Lloyd, V. Pensabene, J. H. Chandler, P. Valdastri  
  _2026-01-21_ · https://arxiv.org/abs/2601.14837v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Soft robotic instruments could navigate delicate, tortuous anatomy more safely than rigid tools, but clinical adoption is limited by insufficient tip functionalization and real-time feedback at the tissue interface. Few sensing and therapeutic modules are compact, robust, and adaptable enough to measure, and respond to, subtle physiological cues during intraluminal procedures. We present a 1.47 mm diameter modular soft robotic catheter that integrates sensing, actuation, and therapy while retaining the compliance needed for safe endoluminal navigation. Validated across multiple in vivo settings, we emphasize its utility in endoscopic retrograde cholangiopancreatography (ERCP), a highly technical procedure and a key access route to the pancreas, an organ that is fragile, difficult to instrument, and central to diseases such as pancreatic cancer. Our architecture supports up to four independently controlled functional units, allowing customizable combinations of anchoring, manipulation, sensing, and targeted drug delivery. In a live porcine model, we demonstrate semi-autonomous deployment into the pancreatic duct and 7.5 cm of endoscopic navigation within it, a region currently inaccessible with standard catheters. A closed-loop autonomous/shared-control system that combines a learned model, magnetic actuation, onboard shape sensing, and visual marker tracking further improves cannulation accuracy. Together, these results establish a scalable platform for multifunctional soft robotic catheters and a new paradigm for complex endoluminal interventions, with potential to reduce radiation exposure, shorten training, and accelerate clinical translation of soft robotic technologies.

  </details>



- **V-CAGE: Context-Aware Generation and Verification for Scalable Long-Horizon Embodied Tasks**  
  Yaru Liu, Ao-bo Wang, Nanyang Ye  
  _2026-01-21_ · https://arxiv.org/abs/2601.15164v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning long-horizon embodied behaviors from synthetic data remains challenging because generated scenes are often physically implausible, language-driven programs frequently "succeed" without satisfying task semantics, and high-level instructions require grounding into executable action sequences. To address these limitations, we introduce V-CAGE, a closed-loop framework for generating robust, semantically aligned manipulation datasets at scale. First, we propose a context-aware instantiation mechanism that enforces geometric consistency during scene synthesis. By dynamically maintaining a map of prohibited spatial areas as objects are placed, our system prevents interpenetration and ensures reachable, conflict-free configurations in cluttered environments. Second, to bridge the gap between abstract intent and low-level control, we employ a hierarchical instruction decomposition module. This decomposes high-level goals (e.g., "get ready for work") into compositional action primitives, facilitating coherent long-horizon planning. Crucially, we enforce semantic correctness through a VLM-based verification loop. Acting as a visual critic, the VLM performs rigorous rejection sampling after each subtask, filtering out "silent failures" where code executes but fails to achieve the visual goal. Experiments demonstrate that V-CAGE yields datasets with superior physical and semantic fidelity, significantly boosting the success rate and generalization of downstream policies compared to non-verified baselines.

  </details>



- **Walk through Paintings: Egocentric World Models from Internet Priors**  
  Anurag Bagchi, Zhipeng Bao, Homanga Bharadhwaj, Yu-Xiong Wang, Pavel Tokmakov, Martial Hebert  
  _2026-01-21_ · https://arxiv.org/abs/2601.15284v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  What if a video generation model could not only imagine a plausible future, but the correct one, accurately reflecting how the world changes with each action? We address this question by presenting the Egocentric World Model (EgoWM), a simple, architecture-agnostic method that transforms any pretrained video diffusion model into an action-conditioned world model, enabling controllable future prediction. Rather than training from scratch, we repurpose the rich world priors of Internet-scale video models and inject motor commands through lightweight conditioning layers. This allows the model to follow actions faithfully while preserving realism and strong generalization. Our approach scales naturally across embodiments and action spaces, ranging from 3-DoF mobile robots to 25-DoF humanoids, where predicting egocentric joint-angle-driven dynamics is substantially more challenging. The model produces coherent rollouts for both navigation and manipulation tasks, requiring only modest fine-tuning. To evaluate physical correctness independently of visual appearance, we introduce the Structural Consistency Score (SCS), which measures whether stable scene elements evolve consistently with the provided actions. EgoWM improves SCS by up to 80 percent over prior state-of-the-art navigation world models, while achieving up to six times lower inference latency and robust generalization to unseen environments, including navigation inside paintings.

  </details>



- **LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes**  
  Ruofan Liang, Norman Müller, Ethan Weber, Duncan Zauss, Nandita Vijaykumar, Peter Kontschieder, Christian Richardt  
  _2026-01-21_ · https://arxiv.org/abs/2601.15283v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present a novel approach for interactive light editing in indoor scenes from a single multi-view scene capture. Our method leverages a generative image-based light decomposition model that factorizes complex indoor scene illumination into its constituent light sources. This factorization enables independent manipulation of individual light sources, specifically allowing control over their state (on/off), chromaticity, and intensity. We further introduce multi-view lighting harmonization to ensure consistent propagation of the lighting decomposition across all scene views. This is integrated into a relightable 3D Gaussian splatting representation, providing real-time interactive control over the individual light sources. Our results demonstrate highly photorealistic lighting decomposition and relighting outcomes across diverse indoor scenes. We evaluate our method on both synthetic and real-world datasets and provide a quantitative and qualitative comparison to state-of-the-art techniques. For video results and interactive demos, see https://luxremix.github.io.

  </details>



- **BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries**  
  Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, Kai Chen  
  _2026-01-21_ · https://arxiv.org/abs/2601.15197v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.

  </details>



- **HyperNet-Adaptation for Diffusion-Based Test Case Generation**  
  Oliver Weißl, Vincenzo Riccio, Severin Kacianka, Andrea Stocco  
  _2026-01-21_ · https://arxiv.org/abs/2601.15041v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  The increasing deployment of deep learning systems requires systematic evaluation of their reliability in real-world scenarios. Traditional gradient-based adversarial attacks introduce small perturbations that rarely correspond to realistic failures and mainly assess robustness rather than functional behavior. Generative test generation methods offer an alternative but are often limited to simple datasets or constrained input domains. Although diffusion models enable high-fidelity image synthesis, their computational cost and limited controllability restrict their applicability to large-scale testing. We present HyNeA, a generative testing method that enables direct and efficient control over diffusion-based generation. HyNeA provides dataset-free controllability through hypernetworks, allowing targeted manipulation of the generative process without relying on architecture-specific conditioning mechanisms or dataset-driven adaptations such as fine-tuning. HyNeA employs a distinct training strategy that supports instance-level tuning to identify failure-inducing test cases without requiring datasets that explicitly contain examples of similar failures. This approach enables the targeted generation of realistic failure cases at substantially lower computational cost than search-based methods. Experimental results show that HyNeA improves controllability and test diversity compared to existing generative test generators and generalizes to domains where failure-labeled training data is unavailable.

  </details>


