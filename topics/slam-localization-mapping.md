# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-04 07:06 UTC_

Total papers shown: **9**


---

- **FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation**  
  Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Mingjie Zhan, Hongsheng Li  
  _2026-02-03_ · https://arxiv.org/abs/2602.03798v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Assisting non-expert users to develop complex interactive websites has become a popular task for LLM-powered code agents. However, existing code agents tend to only generate frontend web pages, masking the lack of real full-stack data processing and storage with fancy visual effects. Notably, constructing production-level full-stack web applications is far more challenging than only generating frontend web pages, demanding careful control of data flow, comprehensive understanding of constantly updating packages and dependencies, and accurate localization of obscure bugs in the codebase. To address these difficulties, we introduce FullStack-Agent, a unified agent system for full-stack agentic coding that consists of three parts: (1) FullStack-Dev, a multi-agent framework with strong planning, code editing, codebase navigation, and bug localization abilities. (2) FullStack-Learn, an innovative data-scaling and self-improving method that back-translates crawled and synthesized website repositories to improve the backbone LLM of FullStack-Dev. (3) FullStack-Bench, a comprehensive benchmark that systematically tests the frontend, backend and database functionalities of the generated website. Our FullStack-Dev outperforms the previous state-of-the-art method by 8.7%, 38.2%, and 15.9% on the frontend, backend, and database test cases respectively. Additionally, FullStack-Learn raises the performance of a 30B model by 9.7%, 9.5%, and 2.8% on the three sets of test cases through self-improvement, demonstrating the effectiveness of our approach. The code is released at https://github.com/mnluzimu/FullStack-Agent.

  </details>



- **A Scene Graph Backed Approach to Open Set Semantic Mapping**  
  Martin Günther, Felix Igelbrink, Oscar Lima, Lennart Niecksch, Marian Renz, Martin Atzmueller  
  _2026-02-03_ · https://arxiv.org/abs/2602.03781v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  While Open Set Semantic Mapping and 3D Semantic Scene Graphs (3DSSGs) are established paradigms in robotic perception, deploying them effectively to support high-level reasoning in large-scale, real-world environments remains a significant challenge. Most existing approaches decouple perception from representation, treating the scene graph as a derivative layer generated post hoc. This limits both consistency and scalability. In contrast, we propose a mapping architecture where the 3DSSG serves as the foundational backend, acting as the primary knowledge representation for the entire mapping process. Our approach leverages prior work on incremental scene graph prediction to infer and update the graph structure in real-time as the environment is explored. This ensures that the map remains topologically consistent and computationally efficient, even during extended operations in large-scale settings. By maintaining an explicit, spatially grounded representation that supports both flat and hierarchical topologies, we bridge the gap between sub-symbolic raw sensor data and high-level symbolic reasoning. Consequently, this provides a stable, verifiable structure that knowledge-driven frameworks, ranging from knowledge graphs and ontologies to Large Language Models (LLMs), can directly exploit, enabling agents to operate with enhanced interpretability, trustworthiness, and alignment to human concepts.

  </details>



- **CMR: Contractive Mapping Embeddings for Robust Humanoid Locomotion on Unstructured Terrains**  
  Qixin Zeng, Hongyin Zhang, Shangke Lyu, Junxi Jin, Donglin Wang, Chao Huang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03511v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robust disturbance rejection remains a longstanding challenge in humanoid locomotion, particularly on unstructured terrains where sensing is unreliable and model mismatch is pronounced. While perception information, such as height map, enhances terrain awareness, sensor noise and sim-to-real gaps can destabilize policies in practice. In this work, we provide theoretical analysis that bounds the return gap under observation noise, when the induced latent dynamics are contractive. Furthermore, we present Contractive Mapping for Robustness (CMR) framework that maps high-dimensional, disturbance-prone observations into a latent space, where local perturbations are attenuated over time. Specifically, this approach couples contrastive representation learning with Lipschitz regularization to preserve task-relevant geometry while explicitly controlling sensitivity. Notably, the formulation can be incorporated into modern deep reinforcement learning pipelines as an auxiliary loss term with minimal additional technical effort required. Further, our extensive humanoid experiments show that CMR potently outperforms other locomotion algorithms under increased noise.

  </details>



- **FOVI: A biologically-inspired foveated interface for deep vision models**  
  Nicholas M. Blauch, George A. Alvarez, Talia Konkle  
  _2026-02-03_ · https://arxiv.org/abs/2602.03766v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Human vision is foveated, with variable resolution peaking at the center of a large field of view; this reflects an efficient trade-off for active sensing, allowing eye-movements to bring different parts of the world into focus with other parts of the world in context. In contrast, most computer vision systems encode the visual world at a uniform resolution, raising challenges for processing full-field high-resolution images efficiently. We propose a foveated vision interface (FOVI) based on the human retina and primary visual cortex, that reformats a variable-resolution retina-like sensor array into a uniformly dense, V1-like sensor manifold. Receptive fields are defined as k-nearest-neighborhoods (kNNs) on the sensor manifold, enabling kNN-convolution via a novel kernel mapping technique. We demonstrate two use cases: (1) an end-to-end kNN-convolutional architecture, and (2) a foveated adaptation of the foundational DINOv3 ViT model, leveraging low-rank adaptation (LoRA). These models provide competitive performance at a fraction of the computational cost of non-foveated baselines, opening pathways for efficient and scalable active sensing for high-resolution egocentric vision. Code and pre-trained models are available at https://github.com/nblauch/fovi and https://huggingface.co/fovi-pytorch.

  </details>



- **Conditional Flow Matching for Visually-Guided Acoustic Highlighting**  
  Hugo Malard, Gael Le Lan, Daniel Wong, David Lou Alon, Yi-Chiao Wu, Sanjeel Parekh  
  _2026-02-03_ · https://arxiv.org/abs/2602.03762v1 · `eess.AS`  
  <details><summary>Abstract</summary>

  Visually-guided acoustic highlighting seeks to rebalance audio in alignment with the accompanying video, creating a coherent audio-visual experience. While visual saliency and enhancement have been widely studied, acoustic highlighting remains underexplored, often leading to misalignment between visual and auditory focus. Existing approaches use discriminative models, which struggle with the inherent ambiguity in audio remixing, where no natural one-to-one mapping exists between poorly-balanced and well-balanced audio mixes. To address this limitation, we reframe this task as a generative problem and introduce a Conditional Flow Matching (CFM) framework. A key challenge in iterative flow-based generation is that early prediction errors -- in selecting the correct source to enhance -- compound over steps and push trajectories off-manifold. To address this, we introduce a rollout loss that penalizes drift at the final step, encouraging self-correcting trajectories and stabilizing long-range flow integration. We further propose a conditioning module that fuses audio and visual cues before vector field regression, enabling explicit cross-modal source selection. Extensive quantitative and qualitative evaluations show that our method consistently surpasses the previous state-of-the-art discriminative approach, establishing that visually-guided audio remixing is best addressed through generative modeling.

  </details>



- **Tutorial on Reasoning for IR & IR for Reasoning**  
  Mohanna Hoveyda, Panagiotis Efstratiadis, Arjen de Vries, Maarten de Rijke  
  _2026-02-03_ · https://arxiv.org/abs/2602.03640v1 · `cs.IR`  
  <details><summary>Abstract</summary>

  Information retrieval has long focused on ranking documents by semantic relatedness. Yet many real-world information needs demand more: enforcement of logical constraints, multi-step inference, and synthesis of multiple pieces of evidence. Addressing these requirements is, at its core, a problem of reasoning. Across AI communities, researchers are developing diverse solutions for the problem of reasoning, from inference-time strategies and post-training of LLMs, to neuro-symbolic systems, Bayesian and probabilistic frameworks, geometric representations, and energy-based models. These efforts target the same problem: to move beyond pattern-matching systems toward structured, verifiable inference. However, they remain scattered across disciplines, making it difficult for IR researchers to identify the most relevant ideas and opportunities. To help navigate the fragmented landscape of research in reasoning, this tutorial first articulates a working definition of reasoning within the context of information retrieval and derives from it a unified analytical framework. The framework maps existing approaches along axes that reflect the core components of the definition. By providing a comprehensive overview of recent approaches and mapping current methods onto the defined axes, we expose their trade-offs and complementarities, highlight where IR can benefit from cross-disciplinary advances, and illustrate how retrieval process itself can play a central role in broader reasoning systems. The tutorial will equip participants with both a conceptual framework and practical guidance for enhancing reasoning-capable IR systems, while situating IR as a domain that both benefits and contributes to the broader development of reasoning methodologies.

  </details>



- **TIPS Over Tricks: Simple Prompts for Effective Zero-shot Anomaly Detection**  
  Alireza Salehi, Ehsan Karami, Sepehr Noey, Sahand Noey, Makoto Yamada, Reshad Hosseini, Mohammad Sabokrou  
  _2026-02-03_ · https://arxiv.org/abs/2602.03594v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Anomaly detection identifies departures from expected behavior in safety-critical settings. When target-domain normal data are unavailable, zero-shot anomaly detection (ZSAD) leverages vision-language models (VLMs). However, CLIP's coarse image-text alignment limits both localization and detection due to (i) spatial misalignment and (ii) weak sensitivity to fine-grained anomalies; prior work compensates with complex auxiliary modules yet largely overlooks the choice of backbone. We revisit the backbone and use TIPS-a VLM trained with spatially aware objectives. While TIPS alleviates CLIP's issues, it exposes a distributional gap between global and local features. We address this with decoupled prompts-fixed for image-level detection and learnable for pixel-level localization-and by injecting local evidence into the global score. Without CLIP-specific tricks, our TIPS-based pipeline improves image-level performance by 1.1-3.9% and pixel-level by 1.5-6.9% across seven industrial datasets, delivering strong generalization with a lean architecture. Code is available at github.com/AlirezaSalehy/Tipsomaly.

  </details>



- **Soft-Radial Projection for Constrained End-to-End Learning**  
  Philipp J. Schneider, Daniel Kuhn  
  _2026-02-03_ · https://arxiv.org/abs/2602.03461v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Integrating hard constraints into deep learning is essential for safety-critical systems. Yet existing constructive layers that project predictions onto constraint boundaries face a fundamental bottleneck: gradient saturation. By collapsing exterior points onto lower-dimensional surfaces, standard orthogonal projections induce rank-deficient Jacobians, which nullify gradients orthogonal to active constraints and hinder optimization. We introduce Soft-Radial Projection, a differentiable reparameterization layer that circumvents this issue through a radial mapping from Euclidean space into the interior of the feasible set. This construction guarantees strict feasibility while preserving a full-rank Jacobian almost everywhere, thereby preventing the optimization stalls typical of boundary-based methods. We theoretically prove that the architecture retains the universal approximation property and empirically show improved convergence behavior and solution quality over state-of-the-art optimization- and projection-based baselines.

  </details>



- **Causal Inference on Networks under Misspecified Exposure Mappings: A Partial Identification Framework**  
  Maresa Schröder, Miruna Oprescu, Stefan Feuerriegel, Nathan Kallus  
  _2026-02-03_ · https://arxiv.org/abs/2602.03459v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Estimating treatment effects in networks is challenging, as each potential outcome depends on the treatments of all other nodes in the network. To overcome this difficulty, existing methods typically impose an exposure mapping that compresses the treatment assignments in the network into a low-dimensional summary. However, if this mapping is misspecified, standard estimators for direct and spillover effects can be severely biased. We propose a novel partial identification framework for causal inference on networks to assess the robustness of treatment effects under misspecifications of the exposure mapping. Specifically, we derive sharp upper and lower bounds on direct and spillover effects under such misspecifications. As such, our framework presents a novel application of causal sensitivity analysis to exposure mappings. We instantiate our framework for three canonical exposure settings widely used in practice: (i) weighted means of the neighborhood treatments, (ii) threshold-based exposure mappings, and (iii) truncated neighborhood interference in the presence of higher-order spillovers. Furthermore, we develop orthogonal estimators for these bounds and prove that the resulting bound estimates are valid, sharp, and efficient. Our experiments show the bounds remain informative and provide reliable conclusions under misspecification of exposure mappings.

  </details>


