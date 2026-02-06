# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-06 07:08 UTC_

Total papers shown: **4**


---

- **Visuo-Tactile World Models**  
  Carolina Higuera, Sergio Arnaud, Byron Boots, Mustafa Mukadam, Francois Robert Hogan, Franziska Meier  
  _2026-02-05_ · https://arxiv.org/abs/2602.06001v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We introduce multi-task Visuo-Tactile World Models (VT-WM), which capture the physics of contact through touch reasoning. By complementing vision with tactile sensing, VT-WM better understands robot-object interactions in contact-rich tasks, avoiding common failure modes of vision-only models under occlusion or ambiguous contact states, such as objects disappearing, teleporting, or moving in ways that violate basic physics. Trained across a set of contact-rich manipulation tasks, VT-WM improves physical fidelity in imagination, achieving 33% better performance at maintaining object permanence and 29% better compliance with the laws of motion in autoregressive rollouts. Moreover, experiments show that grounding in contact dynamics also translates to planning. In zero-shot real-robot experiments, VT-WM achieves up to 35% higher success rates, with the largest gains in multi-step, contact-rich tasks. Finally, VT-WM demonstrates significant downstream versatility, effectively adapting its learned contact dynamics to a novel task and achieving reliable planning success with only a limited set of demonstrations.

  </details>



- **ShapeUP: Scalable Image-Conditioned 3D Editing**  
  Inbar Gat, Dana Cohen-Bar, Guy Levy, Elad Richardson, Daniel Cohen-Or  
  _2026-02-05_ · https://arxiv.org/abs/2602.05676v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advancements in 3D foundation models have enabled the generation of high-fidelity assets, yet precise 3D manipulation remains a significant challenge. Existing 3D editing frameworks often face a difficult trade-off between visual controllability, geometric consistency, and scalability. Specifically, optimization-based methods are prohibitively slow, multi-view 2D propagation techniques suffer from visual drift, and training-free latent manipulation methods are inherently bound by frozen priors and cannot directly benefit from scaling. In this work, we present ShapeUP, a scalable, image-conditioned 3D editing framework that formulates editing as a supervised latent-to-latent translation within a native 3D representation. This formulation allows ShapeUP to build on a pretrained 3D foundation model, leveraging its strong generative prior while adapting it to editing through supervised training. In practice, ShapeUP is trained on triplets consisting of a source 3D shape, an edited 2D image, and the corresponding edited 3D shape, and learns a direct mapping using a 3D Diffusion Transformer (DiT). This image-as-prompt approach enables fine-grained visual control over both local and global edits and achieves implicit, mask-free localization, while maintaining strict structural consistency with the original asset. Our extensive evaluations demonstrate that ShapeUP consistently outperforms current trained and training-free baselines in both identity preservation and edit fidelity, offering a robust and scalable paradigm for native 3D content creation.

  </details>



- **InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions**  
  Sirui Xu, Samuel Schulter, Morteza Ziyadi, Xialin He, Xiaohan Fei, Yu-Xiong Wang, Liangyan Gui  
  _2026-02-05_ · https://arxiv.org/abs/2602.06035v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Humans rarely plan whole-body interactions with objects at the level of explicit whole-body movements. High-level intentions, such as affordance, define the goal, while coordinated balance, contact, and manipulation can emerge naturally from underlying physical and motor priors. Scaling such priors is key to enabling humanoids to compose and generalize loco-manipulation skills across diverse contexts while maintaining physically coherent whole-body coordination. To this end, we introduce InterPrior, a scalable framework that learns a unified generative controller through large-scale imitation pretraining and post-training by reinforcement learning. InterPrior first distills a full-reference imitation expert into a versatile, goal-conditioned variational policy that reconstructs motion from multimodal observations and high-level intent. While the distilled policy reconstructs training behaviors, it does not generalize reliably due to the vast configuration space of large-scale human-object interactions. To address this, we apply data augmentation with physical perturbations, and then perform reinforcement learning finetuning to improve competence on unseen goals and initializations. Together, these steps consolidate the reconstructed latent skills into a valid manifold, yielding a motion prior that generalizes beyond the training data, e.g., it can incorporate new behaviors such as interactions with unseen objects. We further demonstrate its effectiveness for user-interactive control and its potential for real robot deployment.

  </details>



- **A Mixed Reality System for Robust Manikin Localization in Childbirth Training**  
  Haojie Cheng, Chang Liu, Abhiram Kanneganti, Mahesh Arjandas Choolani, Arundhati Tushar Gosavi, Eng Tat Khoo  
  _2026-02-05_ · https://arxiv.org/abs/2602.05588v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Opportunities for medical students to gain practical experience in vaginal births are increasingly constrained by shortened clinical rotations, patient reluctance, and the unpredictable nature of labour. To alleviate clinicians' instructional burden and enhance trainees' learning efficiency, we introduce a mixed reality (MR) system for childbirth training that combines virtual guidance with tactile manikin interaction, thereby preserving authentic haptic feedback while enabling independent practice without continuous on-site expert supervision. The system extends the passthrough capability of commercial head-mounted displays (HMDs) by spatially calibrating an external RGB-D camera, allowing real-time visual integration of physical training objects. Building on this capability, we implement a coarse-to-fine localization pipeline that first aligns the maternal manikin with fiducial markers to define a delivery region and then registers the pre-scanned neonatal head within this area. This process enables spatially accurate overlay of virtual guiding hands near the manikin, allowing trainees to follow expert trajectories reinforced by haptic interaction. Experimental evaluations demonstrate that the system achieves accurate and stable manikin localization on a standalone headset, ensuring practical deployment without external computing resources. A large-scale user study involving 83 fourth-year medical students was subsequently conducted to compare MR-based and virtual reality (VR)-based childbirth training. Four senior obstetricians independently assessed performance using standardized criteria. Results showed that MR training achieved significantly higher scores in delivery, post-delivery, and overall task performance, and was consistently preferred by trainees over VR training.

  </details>


