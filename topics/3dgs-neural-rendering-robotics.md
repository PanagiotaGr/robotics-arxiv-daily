# 3D Gaussian Splatting & Neural Rendering (Robotics)

_Robotics arXiv Daily_

_Updated: 2026-03-12 07:09 UTC_

Total papers shown: **2**


---

- **Splat2Real: Novel-view Scaling for Physical AI with 3D Gaussian Splatting**  
  Hansol Lim, Jongseong Brad Choi  
  _2026-03-11_ · https://arxiv.org/abs/2603.10638v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Physical AI faces viewpoint shift between training and deployment, and novel-view robustness is essential for monocular RGB-to-3D perception. We cast Real2Render2Real monocular depth pretraining as imitation-learning-style supervision from a digital twin oracle: a student depth network imitates expert metric depth/visibility rendered from a scene mesh, while 3DGS supplies scalable novel-view observations. We present Splat2Real, centered on novel-view scaling: performance depends more on which views are added than on raw view count. We introduce CN-Coverage, a coverage+novelty curriculum that greedily selects views by geometry gain and an extrapolation penalty, plus a quality-aware guardrail fallback for low-reliability teachers. Across 20 TUM RGB-D sequences with step-matched budgets (N=0 to 2000 additional rendered views, with N unique <= 500 and resampling for larger budgets), naive scaling is unstable; CN-Coverage mitigates worst-case regressions relative to Robot/Coverage policies, and GOL-Gated CN-Coverage provides the strongest medium-high-budget stability with the lowest high-novelty tail error. Downstream control-proxy results versus N provides embodied-relevance evidence by shifting safety/progress trade-offs under viewpoint shift.

  </details>



- **Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation**  
  Tao Zhong, Yixun Hu, Dongzhe Zheng, Aditya Sood, Christine Allen-Blanchette  
  _2026-03-11_ · https://arxiv.org/abs/2603.11045v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We propose Neural Field Thermal Tomography (NeFTY), a differentiable physics framework for the quantitative 3D reconstruction of material properties from transient surface temperature measurements. While traditional thermography relies on pixel-wise 1D approximations that neglect lateral diffusion, and soft-constrained Physics-Informed Neural Networks (PINNs) often fail in transient diffusion scenarios due to gradient stiffness, NeFTY parameterizes the 3D diffusivity field as a continuous neural field optimized through a rigorous numerical solver. By leveraging a differentiable physics solver, our approach enforces thermodynamic laws as hard constraints while maintaining the memory efficiency required for high-resolution 3D tomography. Our discretize-then-optimize paradigm effectively mitigates the spectral bias and ill-posedness inherent in inverse heat conduction, enabling the recovery of subsurface defects at arbitrary scales. Experimental validation on synthetic data demonstrates that NeFTY significantly improves the accuracy of subsurface defect localization over baselines. Additional details at https://cab-lab-princeton.github.io/nefty/

  </details>


