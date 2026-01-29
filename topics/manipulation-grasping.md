# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-01-29 07:02 UTC_

Total papers shown: **6**


---

- **Vibro-Sense: Robust Vibration-based Impulse Response Localization and Trajectory Tracking for Robotic Hands**  
  Wadhah Zai El Amri, Nicolás Navarro-Guerrero  
  _2026-01-28_ · https://arxiv.org/abs/2601.20555v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Rich contact perception is crucial for robotic manipulation, yet traditional tactile skins remain expensive and complex to integrate. This paper presents a scalable alternative: high-accuracy whole-body touch localization via vibro-acoustic sensing. By equipping a robotic hand with seven low-cost piezoelectric microphones and leveraging an Audio Spectrogram Transformer, we decode the vibrational signatures generated during physical interaction. Extensive evaluation across stationary and dynamic tasks reveals a localization error of under 5 mm in static conditions. Furthermore, our analysis highlights the distinct influence of material properties: stiff materials (e.g., metal) excel in impulse response localization due to sharp, high-bandwidth responses, whereas textured materials (e.g., wood) provide superior friction-based features for trajectory tracking. The system demonstrates robustness to the robot's own motion, maintaining effective tracking even during active operation. Our primary contribution is demonstrating that complex physical contact dynamics can be effectively decoded from simple vibrational signals, offering a viable pathway to widespread, affordable contact perception in robotics. To accelerate research, we provide our full datasets, models, and experimental setups as open-source resources.

  </details>



- **Learning From a Steady Hand: A Weakly Supervised Agent for Robot Assistance under Microscopy**  
  Huanyu Tian, Martin Huber, Lingyun Zeng, Zhe Han, Wayne Bennett, Giuseppe Silvestri, Gerardo Mendizabal-Ruiz, Tom Vercauteren, Alejandro Chavez-Badiola, Christos Bergeles  
  _2026-01-28_ · https://arxiv.org/abs/2601.20776v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper rethinks steady-hand robotic manipulation by using a weakly supervised framework that fuses calibration-aware perception with admittance control. Unlike conventional automation that relies on labor-intensive 2D labeling, our framework leverages reusable warm-up trajectories to extract implicit spatial information, thereby achieving calibration-aware, depth-resolved perception without the need for external fiducials or manual depth annotation. By explicitly characterizing residuals from observation and calibration models, the system establishes a task-space error budget from recorded warm-ups. The uncertainty budget yields a lateral closed-loop accuracy of approx. 49 micrometers at 95% confidence (worst-case testing subset) and a depth accuracy of <= 291 micrometers at 95% confidence bound during large in-plane moves. In a within-subject user study (N=8), the learned agent reduces overall NASA-TLX workload by 77.1% relative to the simple steady-hand assistance baseline. These results demonstrate that the weakly supervised agent improves the reliability of microscope-guided biomedical micromanipulation without introducing complex setup requirements, offering a practical framework for microscope-guided intervention.

  </details>



- **STORM: Slot-based Task-aware Object-centric Representation for robotic Manipulation**  
  Alexandre Chapin, Emmanuel Dellandréa, Liming Chen  
  _2026-01-28_ · https://arxiv.org/abs/2601.20381v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual foundation models provide strong perceptual features for robotics, but their dense representations lack explicit object-level structure, limiting robustness and contractility in manipulation tasks. We propose STORM (Slot-based Task-aware Object-centric Representation for robotic Manipulation), a lightweight object-centric adaptation module that augments frozen visual foundation models with a small set of semantic-aware slots for robotic manipulation. Rather than retraining large backbones, STORM employs a multi-phase training strategy: object-centric slots are first stabilized through visual--semantic pretraining using language embeddings, then jointly adapted with a downstream manipulation policy. This staged learning prevents degenerate slot formation and preserves semantic consistency while aligning perception with task objectives. Experiments on object discovery benchmarks and simulated manipulation tasks show that STORM improves generalization to visual distractors, and control performance compared to directly using frozen foundation model features or training object-centric representations end-to-end. Our results highlight multi-phase adaptation as an efficient mechanism for transforming generic foundation model features into task-aware object-centric representations for robotic control.

  </details>



- **Demonstration-Free Robotic Control via LLM Agents**  
  Brian Y. Tsui, Alan Y. Fang, Tiffany J. Hwu  
  _2026-01-28_ · https://arxiv.org/abs/2601.20334v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robotic manipulation has increasingly adopted vision-language-action (VLA) models, which achieve strong performance but typically require task-specific demonstrations and fine-tuning, and often generalize poorly under domain shift. We investigate whether general-purpose large language model (LLM) agent frameworks, originally developed for software engineering, can serve as an alternative control paradigm for embodied manipulation. We introduce FAEA (Frontier Agent as Embodied Agent), which applies an LLM agent framework directly to embodied manipulation without modification. Using the same iterative reasoning that enables software agents to debug code, FAEA enables embodied agents to reason through manipulation strategies. We evaluate an unmodified frontier agent, Claude Agent SDK, across the LIBERO, ManiSkill3, and MetaWorld benchmarks. With privileged environment state access, FAEA achieves success rates of 84.9%, 85.7%, and 96%, respectively. This level of task success approaches that of VLA models trained with less than 100 demonstrations per task, without requiring demonstrations or fine-tuning. With one round of human feedback as an optional optimization, performance increases to 88.2% on LIBERO. This demonstration-free capability has immediate practical value: FAEA can autonomously explore novel scenarios in simulation and generate successful trajectories for training data augmentation in embodied learning. Our results indicate that general-purpose agents are sufficient for a class of manipulation tasks dominated by deliberative, task-level planning. This opens a path for robotics systems to leverage actively maintained agent infrastructure and benefit directly from ongoing advances in frontier models. Code is available at https://github.com/robiemusketeer/faea-sim

  </details>



- **One Step Is Enough: Dispersive MeanFlow Policy Optimization**  
  Guowei Zou, Haitao Wang, Hejun Wu, Yukun Qian, Yuhang Wang, Weibing Li  
  _2026-01-28_ · https://arxiv.org/abs/2601.20701v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Real-time robotic control demands fast action generation. However, existing generative policies based on diffusion and flow matching require multi-step sampling, fundamentally limiting deployment in time-critical scenarios. We propose Dispersive MeanFlow Policy Optimization (DMPO), a unified framework that enables true one-step generation through three key components: MeanFlow for mathematically-derived single-step inference without knowledge distillation, dispersive regularization to prevent representation collapse, and reinforcement learning (RL) fine-tuning to surpass expert demonstrations. Experiments across RoboMimic manipulation and OpenAI Gym locomotion benchmarks demonstrate competitive or superior performance compared to multi-step baselines. With our lightweight model architecture and the three key algorithmic components working in synergy, DMPO exceeds real-time control requirements (>120Hz) with 5-20x inference speedup, reaching hundreds of Hertz on high-performance GPUs. Physical deployment on a Franka-Emika-Panda robot validates real-world applicability.

  </details>



- **Quartet of Diffusions: Structure-Aware Point Cloud Generation through Part and Symmetry Guidance**  
  Chenliang Zhou, Fangcheng Zhong, Weihao Xia, Albert Miao, Canberk Baykal, Cengiz Oztireli  
  _2026-01-28_ · https://arxiv.org/abs/2601.20425v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We introduce the Quartet of Diffusions, a structure-aware point cloud generation framework that explicitly models part composition and symmetry. Unlike prior methods that treat shape generation as a holistic process or only support part composition, our approach leverages four coordinated diffusion models to learn distributions of global shape latents, symmetries, semantic parts, and their spatial assembly. This structured pipeline ensures guaranteed symmetry, coherent part placement, and diverse, high-quality outputs. By disentangling the generative process into interpretable components, our method supports fine-grained control over shape attributes, enabling targeted manipulation of individual parts while preserving global consistency. A central global latent further reinforces structural coherence across assembled parts. Our experiments show that the Quartet achieves state-of-the-art performance. To our best knowledge, this is the first 3D point cloud generation framework that fully integrates and enforces both symmetry and part priors throughout the generative process.

  </details>


