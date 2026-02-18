# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-18 07:14 UTC_

Total papers shown: **8**


---

- **When Remembering and Planning are Worth it: Navigating under Change**  
  Omid Madani, J. Brian Burns, Reza Eghbali, Thomas L. Dean  
  _2026-02-17_ · https://arxiv.org/abs/2602.15274v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  We explore how different types and uses of memory can aid spatial navigation in changing uncertain environments. In the simple foraging task we study, every day, our agent has to find its way from its home, through barriers, to food. Moreover, the world is non-stationary: from day to day, the location of the barriers and food may change, and the agent's sensing such as its location information is uncertain and very limited. Any model construction, such as a map, and use, such as planning, needs to be robust against these challenges, and if any learning is to be useful, it needs to be adequately fast. We look at a range of strategies, from simple to sophisticated, with various uses of memory and learning. We find that an architecture that can incorporate multiple strategies is required to handle (sub)tasks of a different nature, in particular for exploration and search, when food location is not known, and for planning a good path to a remembered (likely) food location. An agent that utilizes non-stationary probability learning techniques to keep updating its (episodic) memories and that uses those memories to build maps and plan on the fly (imperfect maps, i.e. noisy and limited to the agent's experience) can be increasingly and substantially more efficient than the simpler (minimal-memory) agents, as the task difficulties such as distance to goal are raised, as long as the uncertainty, from localization and change, is not too large.

  </details>



- **Measurement-Based Validation of Geometry-Driven RIS Beam Steering in Industrial Environments**  
  Adam Umra, Simon Tewes, Niklas Beckmann, Niels König, Aydin Sezgin, Robert Schmitt  
  _2026-02-17_ · https://arxiv.org/abs/2602.15808v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Reconfigurable intelligent surfaces (RISs) offer programmable control of radio propagation for future wireless systems. For configuration, geometry-driven analytical approaches are appealing for their simplicity and real-time operation, but their performance in challenging environments such as industrial halls with dense multipath and metallic scattering is not well established. To this end, we present a measurement-based evaluation of geometry-driven RIS beam steering in a large industrial hall using a 5 GHz RIS prototype. A novel RIS configuration is proposed in which four patch antennas are mounted in close proximity in front of the RIS to steer the incident field and enable controlled reflection. For this setup, analytically computed, quantized configurations are implemented. Two-dimensional received power maps from two measurement areas reveal consistent, spatially selective focusing. Configurations optimized near the receiver produce clear power maxima, while steering to offset locations triggers a rapid 20-30 dB reduction. With increasing RIS-receiver distance, elevation selectivity broadens due to finite-aperture and geometric constraints, while azimuth steering remains robust. These results confirm the practical viability of geometry-driven RIS beam steering in industrial environments and support its use for spatial field control and localization under non-ideal propagation.

  </details>



- **Criteria-first, semantics-later: reproducible structure discovery in image-based sciences**  
  Jan Bumberger  
  _2026-02-17_ · https://arxiv.org/abs/2602.15712v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Across the natural and life sciences, images have become a primary measurement modality, yet the dominant analytic paradigm remains semantics-first. Structure is recovered by predicting or enforcing domain-specific labels. This paradigm fails systematically under the conditions that make image-based science most valuable, including open-ended scientific discovery, cross-sensor and cross-site comparability, and long-term monitoring in which domain ontologies and associated label sets drift culturally, institutionally, and ecologically. A deductive inversion is proposed in the form of criteria-first and semantics-later. A unified framework for criteria-first structure discovery is introduced. It separates criterion-defined, semantics-free structure extraction from downstream semantic mapping into domain ontologies or vocabularies and provides a domain-general scaffold for reproducible analysis across image-based sciences. Reproducible science requires that the first analytic layer perform criterion-driven, semantics-free structure discovery, yielding stable partitions, structural fields, or hierarchies defined by explicit optimality criteria rather than local domain ontologies. Semantics is not discarded; it is relocated downstream as an explicit mapping from the discovered structural product to a domain ontology or vocabulary, enabling plural interpretations and explicit crosswalks without rewriting upstream extraction. Grounded in cybernetics, observation-as-distinction, and information theory's separation of information from meaning, the argument is supported by cross-domain evidence showing that criteria-first components recur whenever labels do not scale. Finally, consequences are outlined for validation beyond class accuracy and for treating structural products as FAIR, AI-ready digital objects for long-term monitoring and digital twins.

  </details>



- **Revisiting Northrop Frye's Four Myths Theory with Large Language Models**  
  Edirlei Soares de Lima, Marco A. Casanova, Antonio L. Furtado  
  _2026-02-17_ · https://arxiv.org/abs/2602.15678v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Northrop Frye's theory of four fundamental narrative genres (comedy, romance, tragedy, satire) has profoundly influenced literary criticism, yet computational approaches to his framework have focused primarily on narrative patterns rather than character functions. In this paper, we present a new character function framework that complements pattern-based analysis by examining how archetypal roles manifest differently across Frye's genres. Drawing on Jungian archetype theory, we derive four universal character functions (protagonist, mentor, antagonist, companion) by mapping them to Jung's psychic structure components. These functions are then specialized into sixteen genre-specific roles based on prototypical works. To validate this framework, we conducted a multi-model study using six state-of-the-art Large Language Models (LLMs) to evaluate character-role correspondences across 40 narrative works. The validation employed both positive samples (160 valid correspondences) and negative samples (30 invalid correspondences) to evaluate whether models both recognize valid correspondences and reject invalid ones. LLMs achieved substantial performance (mean balanced accuracy of 82.5%) with strong inter-model agreement (Fleiss' $κ$ = 0.600), demonstrating that the proposed correspondences capture systematic structural patterns. Performance varied by genre (ranging from 72.7% to 89.9%) and role (52.5% to 99.2%), with qualitative analysis revealing that variations reflect genuine narrative properties, including functional distribution in romance and deliberate archetypal subversion in satire. This character-based approach demonstrates the potential of LLM-supported methods for computational narratology and provides a foundation for future development of narrative generation methods and interactive storytelling applications.

  </details>



- **The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes**  
  Mohammad Taufeeque, Stefan Heimersheim, Adam Gleave, Chris Cundy  
  _2026-02-17_ · https://arxiv.org/abs/2602.15515v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Training against white-box deception detectors has been proposed as a way to make AI systems honest. However, such training risks models learning to obfuscate their deception to evade the detector. Prior work has studied obfuscation only in artificial settings where models were directly rewarded for harmful output. We construct a realistic coding environment where reward hacking via hardcoding test cases naturally occurs, and show that obfuscation emerges in this setting. We introduce a taxonomy of possible outcomes when training against a deception detector. The model either remains honest, or becomes deceptive via two possible obfuscation strategies. (i) Obfuscated activations: the model outputs deceptive text while modifying its internal representations to no longer trigger the detector. (ii) Obfuscated policy: the model outputs deceptive text that evades the detector, typically by including a justification for the reward hack. Empirically, obfuscated activations arise from representation drift during RL, with or without a detector penalty. The probe penalty only incentivizes obfuscated policies; we theoretically show this is expected for policy gradient methods. Sufficiently high KL regularization and detector penalty can yield honest policies, establishing white-box deception detectors as viable training signals for tasks prone to reward hacking.

  </details>



- **The Skeletal Trap: Mapping Spatial Inequality and Ghost Stops in Ankara's Transit Network**  
  Elifnaz Kancan  
  _2026-02-17_ · https://arxiv.org/abs/2602.15470v1 · `physics.soc-ph`  
  <details><summary>Abstract</summary>

  Ankara's public transport crisis is commonly framed as a shortage of buses or operational inefficiency. This study argues that the problem is fundamentally morphological and structural. The city's leapfrog urban expansion has produced fragmented peripheral clusters disconnected from a rigid, center-oriented bus network. As a result, demand remains intensely concentrated along the Kizilay-Ulus axis and western corridors, while peripheral districts experience either chronic under-service or enforced transfer dependency. The deficiency is therefore not merely quantitative but rooted in the misalignment between urban macroform and network architecture. The empirical analysis draws on a 173-day operational dataset derived from route-level passenger and trip reports published by EGO under the former "Transparent Ankara" initiative. To overcome the absence of stop-level geospatial data, a Connectivity-Based Weighted Distribution Model reallocates passenger volumes to 1 km x 1 km grid cells using network centrality. The findings reveal persistent center-periphery asymmetries, structural bottlenecks, and spatially embedded accessibility inequalities.

  </details>



- **Automatic Funny Scene Extraction from Long-form Cinematic Videos**  
  Sibendu Paul, Haotian Jiang, Caren Chen  
  _2026-02-17_ · https://arxiv.org/abs/2602.15381v1 · `cs.IR`  
  <details><summary>Abstract</summary>

  Automatically extracting engaging and high-quality humorous scenes from cinematic titles is pivotal for creating captivating video previews and snackable content, boosting user engagement on streaming platforms. Long-form cinematic titles, with their extended duration and complex narratives, challenge scene localization, while humor's reliance on diverse modalities and its nuanced style add further complexity. This paper introduces an end-to-end system for automatically identifying and ranking humorous scenes from long-form cinematic titles, featuring shot detection, multimodal scene localization, and humor tagging optimized for cinematic content. Key innovations include a novel scene segmentation approach combining visual and textual cues, improved shot representations via guided triplet mining, and a multimodal humor tagging framework leveraging both audio and text. Our system achieves an 18.3% AP improvement over state-of-the-art scene detection on the OVSD dataset and an F1 score of 0.834 for detecting humor in long text. Extensive evaluations across five cinematic titles demonstrate 87% of clips extracted by our pipeline are intended to be funny, while 98% of scenes are accurately localized. With successful generalization to trailers, these results showcase the pipeline's potential to enhance content creation workflows, improve user engagement, and streamline snackable content generation for diverse cinematic media formats.

  </details>



- **Prescriptive Scaling Reveals the Evolution of Language Model Capabilities**  
  Hanlin Zhang, Jikai Jin, Vasilis Syrgkanis, Sham Kakade  
  _2026-02-17_ · https://arxiv.org/abs/2602.15327v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  For deploying foundation models, practitioners increasingly need prescriptive scaling laws: given a pre training compute budget, what downstream accuracy is attainable with contemporary post training practice, and how stable is that mapping as the field evolves? Using large scale observational evaluations with 5k observational and 2k newly sampled data on model performance, we estimate capability boundaries, high conditional quantiles of benchmark scores as a function of log pre training FLOPs, via smoothed quantile regression with a monotone, saturating sigmoid parameterization. We validate the temporal reliability by fitting on earlier model generations and evaluating on later releases. Across various tasks, the estimated boundaries are mostly stable, with the exception of math reasoning that exhibits a consistently advancing boundary over time. We then extend our approach to analyze task dependent saturation and to probe contamination related shifts on math reasoning tasks. Finally, we introduce an efficient algorithm that recovers near full data frontiers using roughly 20% of evaluation budget. Together, our work releases the Proteus 2k, the latest model performance evaluation dataset, and introduces a practical methodology for translating compute budgets into reliable performance expectations and for monitoring when capability boundaries shift across time.

  </details>


