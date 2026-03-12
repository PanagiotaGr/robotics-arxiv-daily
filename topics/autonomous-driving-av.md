# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-03-12 07:09 UTC_

Total papers shown: **5**


---

- **Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction**  
  Hao Zhou, Lu Qi, Jason Li, Jie Zhang, Yi Liu, Xu Yang, Mingyu Fan, Fei Luo  
  _2026-03-11_ · https://arxiv.org/abs/2603.10597v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Trajectory prediction is critical for autonomous driving, enabling safe and efficient planning in dense, dynamic traffic. Most existing methods optimize prediction accuracy under fixed-length observations. However, real-world driving often yields variable-length, incomplete observations, posing a challenge to these methods. A common strategy is to directly map features from incomplete observations to those from complete ones. This one-shot mapping, however, struggles to learn accurate representations for short trajectories due to significant information gaps. To address this issue, we propose a Progressive Retrospective Framework (PRF), which gradually aligns features from incomplete observations with those from complete ones via a cascade of retrospective units. Each unit consists of a Retrospective Distillation Module (RDM) and a Retrospective Prediction Module (RPM), where RDM distills features and RPM recovers previous timesteps using the distilled features. Moreover, we propose a Rolling-Start Training Strategy (RSTS) that enhances data efficiency during PRF training. PRF is plug-and-play with existing methods. Extensive experiments on datasets Argoverse 2 and Argoverse 1 demonstrate the effectiveness of PRF. Code is available at https://github.com/zhouhao94/PRF.

  </details>



- **MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction**  
  Jonas Merkert, Alexander Blumberg, Jan-Hendrik Pauls, Christoph Stiller  
  _2026-03-11_ · https://arxiv.org/abs/2603.10688v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous vehicles rely on map information to understand the world around them. However, the creation and maintenance of offline high-definition (HD) maps remains costly. A more scalable alternative lies in online HD map construction, which only requires map annotations at training time. To further reduce the need for annotating vast training labels, self-supervised training provides an alternative. This work focuses on improving the latent birds-eye-view (BEV) feature grid representation within a vectorized online HD map construction model by enforcing geospatial consistency between overlapping BEV feature grids as part of a contrastive loss function. To ensure geospatial overlap for contrastive pairs, we introduce an approach to analyze the overlap between traversals within a given dataset and generate subsidiary dataset splits following adjustable multi-traversal requirements. We train the same model supervised using a reduced set of single-traversal labeled data and self-supervised on a broader unlabeled set of data following our multi-traversal requirements, effectively implementing a semi-supervised approach. Our approach outperforms the supervised baseline across the board, both quantitatively in terms of the downstream tasks vectorized map perception performance and qualitatively in terms of segmentation in the principal component analysis (PCA) visualization of the BEV feature space.

  </details>



- **DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving**  
  Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, et al.  
  _2026-03-11_ · https://arxiv.org/abs/2603.11041v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT.

  </details>



- **STADA: Specification-based Testing for Autonomous Driving Agents**  
  Joy Saha, Trey Woodlief, Sebastian Elbaum, Matthew B. Dwyer  
  _2026-03-11_ · https://arxiv.org/abs/2603.10940v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Simulation-based testing has become a standard approach to validating autonomous driving agents prior to real-world deployment. A high-quality validation campaign will exercise an agent in diverse contexts comprised of varying static environments, e.g., lanes, intersections, signage, and dynamic elements, e.g., vehicles and pedestrians. To achieve this, existing test generation techniques rely on template-based, manually constructed, or random scenario generation. When applied to validate formally specified safety requirements, such methods either require significant human effort or run the risk of missing important behavior related to the requirement. To address this gap, we present STADA, a Specification-based Test generation framework for Autonomous Driving Agents that systematically generates the space of scenarios defined by a formal specification expressed in temporal logic (LTLf). Given a specification, STADA constructs all distinct initial scenes, a diverse space of continuations of those scenes, and simulations that reflect the behaviors of the specification. Evaluation of STADA on a variety of LTLf specifications formalized in SCENEFLOW using three complementary coverage criteria demonstrates that STADA yields more than 2x higher coverage than the best baseline on the finest criteria and a 75% increase for the coarsest criteria. Moreover, it matches the coverage of the best baseline with 6 times fewer simulations. While set in the context of autonomous driving, the approach is applicable to other domains with rich simulation environments.

  </details>



- **Evaluating randomized smoothing as a defense against adversarial attacks in trajectory prediction**  
  Julian F. Schumann, Eduardo Figueiredo, Frederik Baymler Mathiesen, Luca Laurenti, Jens Kober, Arkady Zgonnikov  
  _2026-03-11_ · https://arxiv.org/abs/2603.10821v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Accurate and robust trajectory prediction is essential for safe and efficient autonomous driving, yet recent work has shown that even state-of-the-art prediction models are highly vulnerable to inputs being mildly perturbed by adversarial attacks. Although model vulnerabilities to such attacks have been studied, work on effective countermeasures remains limited. In this work, we develop and evaluate a new defense mechanism for trajectory prediction models based on randomized smoothing -- an approach previously applied successfully in other domains. We evaluate its ability to improve model robustness through a series of experiments that test different strategies of randomized smoothing. We show that our approach can consistently improve prediction robustness of multiple base trajectory prediction models in various datasets without compromising accuracy in non-adversarial settings. Our results demonstrate that randomized smoothing offers a simple and computationally inexpensive technique for mitigating adversarial attacks in trajectory prediction.

  </details>


