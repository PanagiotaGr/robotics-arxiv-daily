# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-01-23 06:51 UTC_

Total papers shown: **5**


---

- **Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning**  
  Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al.  
  _2026-01-22_ · https://arxiv.org/abs/2601.16163v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Recent video generation models demonstrate remarkable ability to capture complex physical interactions and scene evolution over time. To leverage their spatiotemporal priors, robotics works have adapted video models for policy learning but introduce complexity by requiring multiple stages of post-training and new architectural components for action generation. In this work, we introduce Cosmos Policy, a simple approach for adapting a large pretrained video model (Cosmos-Predict2) into an effective robot policy through a single stage of post-training on the robot demonstration data collected on the target platform, with no architectural modifications. Cosmos Policy learns to directly generate robot actions encoded as latent frames within the video model's latent diffusion process, harnessing the model's pretrained priors and core learning algorithm to capture complex action distributions. Additionally, Cosmos Policy generates future state images and values (expected cumulative rewards), which are similarly encoded as latent frames, enabling test-time planning of action trajectories with higher likelihood of success. In our evaluations, Cosmos Policy achieves state-of-the-art performance on the LIBERO and RoboCasa simulation benchmarks (98.5% and 67.1% average success rates, respectively) and the highest average score in challenging real-world bimanual manipulation tasks, outperforming strong diffusion policies trained from scratch, video model-based policies, and state-of-the-art vision-language-action models fine-tuned on the same robot demonstrations. Furthermore, given policy rollout data, Cosmos Policy can learn from experience to refine its world model and value function and leverage model-based planning to achieve even higher success rates in challenging tasks. We release code, models, and training data at https://research.nvidia.com/labs/dir/cosmos-policy/

  </details>



- **DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning**  
  Junha Lee, Eunha Park, Minsu Cho  
  _2026-01-22_ · https://arxiv.org/abs/2601.16046v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Language-driven dexterous grasp generation requires the models to understand task semantics, 3D geometry, and complex hand-object interactions. While vision-language models have been applied to this problem, existing approaches directly map observations to grasp parameters without intermediate reasoning about physical interactions. We present DextER, Dexterous Grasp Generation with Embodied Reasoning, which introduces contact-based embodied reasoning for multi-finger manipulation. Our key insight is that predicting which hand links contact where on the object surface provides an embodiment-aware intermediate representation bridging task semantics with physical constraints. DextER autoregressively generates embodied contact tokens specifying which finger links contact where on the object surface, followed by grasp tokens encoding the hand configuration. On DexGYS, DextER achieves 67.14% success rate, outperforming state-of-the-art by 3.83%p with 96.4% improvement in intention alignment. We also demonstrate steerable generation through partial contact specification, providing fine-grained control over grasp synthesis.

  </details>



- **Point Bridge: 3D Representations for Cross Domain Policy Learning**  
  Siddhant Haldar, Lars Johannsmeier, Lerrel Pinto, Abhishek Gupta, Dieter Fox, Yashraj Narang, Ajay Mandlekar  
  _2026-01-22_ · https://arxiv.org/abs/2601.16212v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robot foundation models are beginning to deliver on the promise of generalist robotic agents, yet progress remains constrained by the scarcity of large-scale real-world manipulation datasets. Simulation and synthetic data generation offer a scalable alternative, but their usefulness is limited by the visual domain gap between simulation and reality. In this work, we present Point Bridge, a framework that leverages unified, domain-agnostic point-based representations to unlock synthetic datasets for zero-shot sim-to-real policy transfer, without explicit visual or object-level alignment. Point Bridge combines automated point-based representation extraction via Vision-Language Models (VLMs), transformer-based policy learning, and efficient inference-time pipelines to train capable real-world manipulation agents using only synthetic data. With additional co-training on small sets of real demonstrations, Point Bridge further improves performance, substantially outperforming prior vision-based sim-and-real co-training methods. It achieves up to 44% gains in zero-shot sim-to-real transfer and up to 66% with limited real data across both single-task and multitask settings. Videos of the robot are best viewed at: https://pointbridge3d.github.io/

  </details>



- **IVRA: Improving Visual-Token Relations for Robot Action Policy with Training-Free Hint-Based Guidance**  
  Jongwoo Park, Kanchana Ranasinghe, Jinhyeok Jang, Cristina Mata, Yoo Sung Jang, Michael S Ryoo  
  _2026-01-22_ · https://arxiv.org/abs/2601.16207v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many Vision-Language-Action (VLA) models flatten image patches into a 1D token sequence, weakening the 2D spatial cues needed for precise manipulation. We introduce IVRA, a lightweight, training-free method that improves spatial understanding by exploiting affinity hints already available in the model's built-in vision encoder, without requiring any external encoder or retraining. IVRA selectively injects these affinity signals into a language-model layer in which instance-level features reside. This inference-time intervention realigns visual-token interactions and better preserves geometric structure while keeping all model parameters fixed. We demonstrate the generality of IVRA by applying it to diverse VLA architectures (LLaRA, OpenVLA, and FLOWER) across simulated benchmarks spanning both 2D and 3D manipulation (VIMA and LIBERO) and on various real-robot tasks. On 2D VIMA, IVRA improves average success by +4.2% over the baseline LLaRA in a low-data regime. On 3D LIBERO, it yields consistent gains over the OpenVLA and FLOWER baselines, including improvements when baseline accuracy is near saturation (96.3% to 97.1%). All code and models will be released publicly. Visualizations are available at: jongwoopark7978.github.io/IVRA

  </details>



- **DTP: A Simple yet Effective Distracting Token Pruning Framework for Vision-Language Action Models**  
  Chenyang Li, Jieyuan Liu, Bin Li, Bo Gao, Yilin Yuan, Yangfan He, Yuchen Li, Jingqun Tang  
  _2026-01-22_ · https://arxiv.org/abs/2601.16065v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Vision-Language Action (VLA) models have shown remarkable progress in robotic manipulation by leveraging the powerful perception abilities of Vision-Language Models (VLMs) to understand environments and directly output actions. However, by default, VLA models may overly attend to image tokens in the task-irrelevant region, which we describe as 'distracting tokens'. This behavior can disturb the model from the generation of the desired action tokens in each step, affecting the success rate of tasks. In this paper, we introduce a simple yet effective plug-and-play Distracting Token Pruning (DTP) framework, which dynamically detects and prunes these distracting image tokens. By correcting the model's visual attention patterns, we aim to improve the task success rate, as well as exploring the performance upper boundaries of the model without altering its original architecture or adding additional inputs. Experiments on the SIMPLER Benchmark (Li et al., 2024) show that our method consistently achieving relative improvements in task success rates across different types of novel VLA models, demonstrating generalizability to transformer-based VLAs. Further analysis reveals a negative correlation between the task success rate and the amount of attentions in the task-irrelevant region for all models tested, highlighting a common phenomenon of VLA models that could guide future research. We also publish our code at: https://anonymous.4open.science/r/CBD3.

  </details>


