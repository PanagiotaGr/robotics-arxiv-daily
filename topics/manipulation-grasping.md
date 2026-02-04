# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-04 07:06 UTC_

Total papers shown: **5**


---

- **Self-supervised Physics-Informed Manipulation of Deformable Linear Objects with Non-negligible Dynamics**  
  Youyuan Long, Gokhan Solak, Sara Zeynalpour, Heng Zhang, Arash Ajoudani  
  _2026-02-03_ · https://arxiv.org/abs/2602.03623v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We address dynamic manipulation of deformable linear objects by presenting SPiD, a physics-informed self-supervised learning framework that couples an accurate deformable object model with an augmented self-supervised training strategy. On the modeling side, we extend a mass-spring model to more accurately capture object dynamics while remaining lightweight enough for high-throughput rollouts during self-supervised learning. On the learning side, we train a neural controller using a task-oriented cost, enabling end-to-end optimization through interaction with the differentiable object model. In addition, we propose a self-supervised DAgger variant that detects distribution shift during deployment and performs offline self-correction to further enhance robustness without expert supervision. We evaluate our method primarily on the rope stabilization task, where a robot must bring a swinging rope to rest as quickly and smoothly as possible. Extensive experiments in both simulation and the real world demonstrate that the proposed controller achieves fast and smooth rope stabilization, generalizing across unseen initial states, rope lengths, masses, non-uniform mass distributions, and external disturbances. Additionally, we develop an affordable markerless rope perception method and demonstrate that our controller maintains performance with noisy and low-frequency state updates. Furthermore, we demonstrate the generality of the framework by extending it to the rope trajectory tracking task. Overall, SPiD offers a data-efficient, robust, and physically grounded framework for dynamic manipulation of deformable linear objects, featuring strong sim-to-real generalization.

  </details>



- **Deep-Learning-Based Control of a Decoupled Two-Segment Continuum Robot for Endoscopic Submucosal Dissection**  
  Yuancheng Shao, Yao Zhang, Jia Gu, Zixi Chen, Di Wu, Yuqiao Chen, Bo Lu, Wenjie Liu, Cesare Stefanini, Peng Qi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03406v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Manual endoscopic submucosal dissection (ESD) is technically demanding, and existing single-segment robotic tools offer limited dexterity. These limitations motivate the development of more advanced solutions. To address this, DESectBot, a novel dual segment continuum robot with a decoupled structure and integrated surgical forceps, enabling 6 degrees of freedom (DoFs) tip dexterity for improved lesion targeting in ESD, was developed in this work. Deep learning controllers based on gated recurrent units (GRUs) for simultaneous tip position and orientation control, effectively handling the nonlinear coupling between continuum segments, were proposed. The GRU controller was benchmarked against Jacobian based inverse kinematics, model predictive control (MPC), a feedforward neural network (FNN), and a long short-term memory (LSTM) network. In nested-rectangle and Lissajous trajectory tracking tasks, the GRU achieved the lowest position/orientation RMSEs: 1.11 mm/ 4.62° and 0.81 mm/ 2.59°, respectively. For orientation control at a fixed position (four target poses), the GRU attained a mean RMSE of 0.14 mm and 0.72°, outperforming all alternatives. In a peg transfer task, the GRU achieved a 100% success rate (120 success/120 attempts) with an average transfer time of 11.8s, the STD significantly outperforms novice-controlled systems. Additionally, an ex vivo ESD demonstration grasping, elevating, and resecting tissue as the scalpel completed the cut confirmed that DESectBot provides sufficient stiffness to divide thick gastric mucosa and an operative workspace adequate for large lesions.These results confirm that GRU-based control significantly enhances precision, reliability, and usability in ESD surgical training scenarios.

  </details>



- **Continuous Control of Editing Models via Adaptive-Origin Guidance**  
  Alon Wolf, Chen Katzir, Kfir Aberman, Or Patashnik  
  _2026-02-03_ · https://arxiv.org/abs/2602.03826v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Diffusion-based editing models have emerged as a powerful tool for semantic image and video manipulation. However, existing models lack a mechanism for smoothly controlling the intensity of text-guided edits. In standard text-conditioned generation, Classifier-Free Guidance (CFG) impacts prompt adherence, suggesting it as a potential control for edit intensity in editing models. However, we show that scaling CFG in these models does not produce a smooth transition between the input and the edited result. We attribute this behavior to the unconditional prediction, which serves as the guidance origin and dominates the generation at low guidance scales, while representing an arbitrary manipulation of the input content. To enable continuous control, we introduce Adaptive-Origin Guidance (AdaOr), a method that adjusts this standard guidance origin with an identity-conditioned adaptive origin, using an identity instruction corresponding to the identity manipulation. By interpolating this identity prediction with the standard unconditional prediction according to the edit strength, we ensure a continuous transition from the input to the edited result. We evaluate our method on image and video editing tasks, demonstrating that it provides smoother and more consistent control compared to current slider-based editing approaches. Our method incorporates an identity instruction into the standard training framework, enabling fine-grained control at inference time without per-edit procedure or reliance on specialized datasets.

  </details>



- **MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction**  
  Jung Min Lee, Dohyeok Lee, Seokhun Ju, Taehyun Cho, Jin Woo Koo, Li Zhao, Sangwoo Hong, Jungwoo Lee  
  _2026-02-03_ · https://arxiv.org/abs/2602.03668v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning \emph{latent actions} from diverse human videos enables scaling robot learning beyond embodiment-specific robot datasets, and these latent actions have recently been used as pseudo-action labels for vision-language-action (VLA) model pretraining. To make VLA pretraining effective, latent actions should contain information about the underlying agent's actions despite the absence of ground-truth labels. We propose \textbf{M}ulti-\textbf{V}iew\textbf{P}oint \textbf{L}atent \textbf{A}ction \textbf{M}odel (\textbf{MVP-LAM}), which learns discrete latent actions that are highly informative about ground-truth actions from time-synchronized multi-view videos. MVP-LAM trains latent actions with a \emph{cross-viewpoint reconstruction} objective, so that a latent action inferred from one view must explain the future in another view, reducing reliance on viewpoint-specific cues. On Bridge V2, MVP-LAM produces more action-centric latent actions, achieving higher mutual information with ground-truth actions and improved action prediction, including under out-of-distribution evaluation. Finally, pretraining VLAs with MVP-LAM latent actions improves downstream manipulation performance on the SIMPLER and LIBERO-Long benchmarks.

  </details>



- **Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation**  
  Fabian Schramm, Franki Nguimatsia Tiofack, Nicolas Perrin-Gilbert, Marc Toussaint, Justin Carpentier  
  _2026-02-03_ · https://arxiv.org/abs/2602.03639v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Sampling-based controllers, such as Model Predictive Path Integral (MPPI) methods, offer substantial flexibility but often suffer from high variance and low sample efficiency. To address these challenges, we introduce a hybrid variance-reduced MPPI framework that integrates a prior model into the sampling process. Our key insight is to decompose the objective function into a known approximate model and a residual term. Since the residual captures only the discrepancy between the model and the objective, it typically exhibits a smaller magnitude and lower variance than the original objective. Although this principle applies to general modeling choices, we demonstrate that adopting a quadratic approximation enables the derivation of a closed-form, model-guided prior that effectively concentrates samples in informative regions. Crucially, the framework is agnostic to the source of geometric information, allowing the quadratic model to be constructed from exact derivatives, structural approximations (e.g., Gauss- or Quasi-Newton), or gradient-free randomized smoothing. We validate the approach on standard optimization benchmarks, a nonlinear, underactuated cart-pole control task, and a contact-rich manipulation problem with non-smooth dynamics. Across these domains, we achieve faster convergence and superior performance in low-sample regimes compared to standard MPPI. These results suggest that the method can make sample-based control strategies more practical in scenarios where obtaining samples is expensive or limited.

  </details>


