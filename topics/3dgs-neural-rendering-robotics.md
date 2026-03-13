# 3D Gaussian Splatting & Neural Rendering (Robotics)

_Robotics arXiv Daily_

_Updated: 2026-03-13 07:08 UTC_

Total papers shown: **2**


---

- **AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies**  
  Jennifer Nolan, Travis Driver, John Christian  
  _2026-03-12_ · https://arxiv.org/abs/2603.11969v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.

  </details>



- **CEI-3D: Collaborative Explicit-Implicit 3D Reconstruction for Realistic and Fine-Grained Object Editing**  
  Yue Shi, Rui Shi, Yuxuan Xiong, Bingbing Ni, Wenjun Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11810v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Existing 3D editing methods often produce unrealistic and unrefined results due to the deeply integrated nature of their reconstruction networks. To address the challenge, this paper introduces CEI-3D, an editing-oriented reconstruction pipeline designed to facilitate realistic and fine-grained editing. Specifically, we propose a collaborative explicit-implicit reconstruction approach, which represents the target object using an implicit SDF network and a differentially sampled, locally controllable set of handler points. The implicit network provides a smooth and continuous geometry prior, while the explicit handler points offer localized control, enabling mutual guidance between the global 3D structure and user-specified local editing regions. To independently control each attribute of the handler points, we design a physical properties disentangling module to decouple the color of the handler points into separate physical properties. We also propose a dual-diffuse-albedo network in this module to process the edited and non-edited regions through separate branches, thereby preventing undesired interference from editing operations. Building on the reconstructed collaborative explicit-implicit representation with disentangled properties, we introduce a spatial-aware editing module that enables part-wise adjustment of relevant handler points. This module employs a cross-view propagation-based 3D segmentation strategy, which helps users to edit the specified physical attributes of a target part efficiently. Extensive experiments on both real and synthetic datasets demonstrate that our approach achieves more realistic and fine-grained editing results than the state-of-the-art (SOTA) methods while requiring less editing time. Our code is available on https://github.com/shiyue001/CEI-3D.

  </details>


