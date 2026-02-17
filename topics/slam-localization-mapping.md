# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-02-17 07:12 UTC_

Total papers shown: **10**


---

- **Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery**  
  Chandrakanth Gudavalli, Tajuddin Manhar Mohammed, Abhay Yadav, Ananth Vishnu Bhaskar, Hardik Prajapati, Cheng Peng, Rama Chellappa, Shivkumar Chandrasekaran, B. S. Manjunath  
  _2026-02-16_ · https://arxiv.org/abs/2602.14929v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Aligning ground-level imagery with geo-registered satellite maps is crucial for mapping, navigation, and situational awareness, yet remains challenging under large viewpoint gaps or when GPS is unreliable. We introduce Wrivinder, a zero-shot, geometry-driven framework that aggregates multiple ground photographs to reconstruct a consistent 3D scene and align it with overhead satellite imagery. Wrivinder combines SfM reconstruction, 3D Gaussian Splatting, semantic grounding, and monocular depth--based metric cues to produce a stable zenith-view rendering that can be directly matched to satellite context for metrically accurate camera geo-localization. To support systematic evaluation of this task, which lacks suitable benchmarks, we also release MC-Sat, a curated dataset linking multi-view ground imagery with geo-registered satellite tiles across diverse outdoor environments. Together, Wrivinder and MC-Sat provide a first comprehensive baseline and testbed for studying geometry-centered cross-view alignment without paired supervision. In zero-shot experiments, Wrivinder achieves sub-30\,m geolocation accuracy across both dense and large-area scenes, highlighting the promise of geometry-based aggregation for robust ground-to-satellite localization.

  </details>



- **From User Preferences to Base Score Extraction Functions in Gradual Argumentation**  
  Aniol Civit, Antonio Rago, Antonio Andriella, Guillem Alenyà, Francesca Toni  
  _2026-02-16_ · https://arxiv.org/abs/2602.14674v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Gradual argumentation is a field of symbolic AI which is attracting attention for its ability to support transparent and contestable AI systems. It is considered a useful tool in domains such as decision-making, recommendation, debate analysis, and others. The outcomes in such domains are usually dependent on the arguments' base scores, which must be selected carefully. Often, this selection process requires user expertise and may not always be straightforward. On the other hand, organising the arguments by preference could simplify the task. In this work, we introduce \emph{Base Score Extraction Functions}, which provide a mapping from users' preferences over arguments to base scores. These functions can be applied to the arguments of a \emph{Bipolar Argumentation Framework} (BAF), supplemented with preferences, to obtain a \emph{Quantitative Bipolar Argumentation Framework} (QBAF), allowing the use of well-established computational tools in gradual argumentation. We outline the desirable properties of base score extraction functions, discuss some design choices, and provide an algorithm for base score extraction. Our method incorporates an approximation of non-linearities in human preferences to allow for better approximation of the real ones. Finally, we evaluate our approach both theoretically and experimentally in a robotics setting, and offer recommendations for selecting appropriate gradual semantics in practice.

  </details>



- **Diagnosing Knowledge Conflict in Multimodal Long-Chain Reasoning**  
  Jing Tang, Kun Wang, Haolang Lu, Hongjin Chen, KaiTao Chen, Zhongxiang Sun, Qiankun Li, Lingjuan Lyu, Guoshun Nan, Zhigang Zeng  
  _2026-02-16_ · https://arxiv.org/abs/2602.14518v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) in long chain-of-thought reasoning often fail when different knowledge sources provide conflicting signals. We formalize these failures under a unified notion of knowledge conflict, distinguishing input-level objective conflict from process-level effective conflict. Through probing internal representations, we reveal that: (I) Linear Separability: different conflict types are explicitly encoded as linearly separable features rather than entangled; (II) Depth Localization: conflict signals concentrate in mid-to-late layers, indicating a distinct processing stage for conflict encoding; (III) Hierarchical Consistency: aggregating noisy token-level signals along trajectories robustly recovers input-level conflict types; and (IV) Directional Asymmetry: reinforcing the model's implicit source preference under conflict is far easier than enforcing the opposite source. Our findings provide a mechanism-level view of multimodal reasoning under knowledge conflict and enable principled diagnosis and control of long-CoT failures.

  </details>



- **Real-time Range-Angle Estimation and Tag Localization for Multi-static Backscatter Systems**  
  Tara Esmaeilbeig, Kartik Patel, Traian E. Abrudan, John Kimionis, Eleftherios Kampianakis, Michael S. Eggleston  
  _2026-02-16_ · https://arxiv.org/abs/2602.14985v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Multi-static backscatter networks (BNs) are strong candidates for joint communication and localization in the ambient IoT paradigm for 6G. Enabling real-time localization in large-scale multi-static deployments with thousands of devices require highly efficient algorithms for estimating key parameters such as range and angle of arrival (AoA), and for fusing these parameters into location estimates. We propose two low-complexity algorithms, Joint Range-Angle Clustering (JRAC) and Stage-wise Range-Angle Estimation (SRAE). Both deliver range and angle estimation accuracy comparable to FFT- and subspace-based baselines while significantly reducing the computation. We then introduce two real-time localization algorithms that fuse the estimated ranges and AoAs: a maximum-likelihood (ML) method solved via gradient search and an iterative re-weighted least squares (IRLS) method. Both achieve localization accuracy comparable to ML-based brute force search albeit with far lower complexity. Experiments on a real-world large-scale multi-static testbed with 4 illuminators, 1 multi-antenna receiver, and 100 tags show that JRAC and SRAE reduce runtime by up to 40X and IRLS achieves up to 500X reduction over ML-based brute force search without degrading localization accuracy. The proposed methods achieve 3 m median localization error across all 100 tags in a sub-6GHz band with 40 MHz bandwidth. These results demonstrate that multi-static range-angle estimation and localization algorithms can make real-time, scalable backscatter localization practical for next-generation ambient IoT networks.

  </details>



- **Fault Detection in Electrical Distribution System using Autoencoders**  
  Sidharthenee Nayak, Victor Sam Moses Babu, Chandrashekhar Narayan Bhende, Pratyush Chakraborty, Mayukha Pal  
  _2026-02-16_ · https://arxiv.org/abs/2602.14939v1 · `eess.SY`  
  <details><summary>Abstract</summary>

  In recent times, there has been considerable interest in fault detection within electrical power systems, garnering attention from both academic researchers and industry professionals. Despite the development of numerous fault detection methods and their adaptations over the past decade, their practical application remains highly challenging. Given the probabilistic nature of fault occurrences and parameters, certain decision-making tasks could be approached from a probabilistic standpoint. Protective systems are tasked with the detection, classification, and localization of faulty voltage and current line magnitudes, culminating in the activation of circuit breakers to isolate the faulty line. An essential aspect of designing effective fault detection systems lies in obtaining reliable data for training and testing, which is often scarce. Leveraging deep learning techniques, particularly the powerful capabilities of pattern classifiers in learning, generalizing, and parallel processing, offers promising avenues for intelligent fault detection. To address this, our paper proposes an anomaly-based approach for fault detection in electrical power systems, employing deep autoencoders. Additionally, we utilize Convolutional Autoencoders (CAE) for dimensionality reduction, which, due to its fewer parameters, requires less training time compared to conventional autoencoders. The proposed method demonstrates superior performance and accuracy compared to alternative detection approaches by achieving an accuracy of 97.62% and 99.92% on simulated and publicly available datasets.

  </details>



- **Picking the Right Specialist: Attentive Neural Process-based Selection of Task-Specialized Models as Tools for Agentic Healthcare Systems**  
  Pramit Saha, Joshua Strong, Mohammad Alsharid, Divyanshu Mishra, J. Alison Noble  
  _2026-02-16_ · https://arxiv.org/abs/2602.14901v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Task-specialized models form the backbone of agentic healthcare systems, enabling the agents to answer clinical queries across tasks such as disease diagnosis, localization, and report generation. Yet, for a given task, a single "best" model rarely exists. In practice, each task is better served by multiple competing specialist models where different models excel on different data samples. As a result, for any given query, agents must reliably select the right specialist model from a heterogeneous pool of tool candidates. To this end, we introduce ToolSelect, which adaptively learns model selection for tools by minimizing a population risk over sampled specialist tool candidates using a consistent surrogate of the task-conditional selection loss. Concretely, we propose an Attentive Neural Process-based selector conditioned on the query and per-model behavioral summaries to choose among the specialist models. Motivated by the absence of any established testbed, we, for the first time, introduce an agentic Chest X-ray environment equipped with a diverse suite of task-specialized models (17 disease detection, 19 report generation, 6 visual grounding, and 13 VQA) and develop ToolSelectBench, a benchmark of 1448 queries. Our results demonstrate that ToolSelect consistently outperforms 10 SOTA methods across four different task families.

  </details>



- **CT-Bench: A Benchmark for Multimodal Lesion Understanding in Computed Tomography**  
  Qingqing Zhu, Qiao Jin, Tejas S. Mathai, Yin Fang, Zhizheng Wang, Yifan Yang, Maame Sarfo-Gyamfi, Benjamin Hou, Ran Gu, Praveen T. S. Balamuralikrishna, et al.  
  _2026-02-16_ · https://arxiv.org/abs/2602.14879v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Artificial intelligence (AI) can automatically delineate lesions on computed tomography (CT) and generate radiology report content, yet progress is limited by the scarcity of publicly available CT datasets with lesion-level annotations. To bridge this gap, we introduce CT-Bench, a first-of-its-kind benchmark dataset comprising two components: a Lesion Image and Metadata Set containing 20,335 lesions from 7,795 CT studies with bounding boxes, descriptions, and size information, and a multitask visual question answering benchmark with 2,850 QA pairs covering lesion localization, description, size estimation, and attribute categorization. Hard negative examples are included to reflect real-world diagnostic challenges. We evaluate multiple state-of-the-art multimodal models, including vision-language and medical CLIP variants, by comparing their performance to radiologist assessments, demonstrating the value of CT-Bench as a comprehensive benchmark for lesion analysis. Moreover, fine-tuning models on the Lesion Image and Metadata Set yields significant performance gains across both components, underscoring the clinical utility of CT-Bench.

  </details>



- **Bayesian Cosmic Void Finding with Graph Flows**  
  Leander Thiele  
  _2026-02-16_ · https://arxiv.org/abs/2602.14630v1 · `astro-ph.CO`  
  <details><summary>Abstract</summary>

  Cosmic voids contain higher-order cosmological information and are of interest for astroparticle physics. Finding genuine matter underdensities in sparse galaxy surveys is, however, an underconstrained problem. Traditional void finding algorithms produce deterministic void catalogs, neglecting the probabilistic nature of the problem. We present a method to sample from the stochastic mapping from galaxy catalogs to arbitrary void definitions. Our algorithm uses a deep graph neural network to evolve "test particles" according to a flow-matching objective. We demonstrate the method in a simplified example setting but outline steps to generalize it towards practically usable void finders. Trained on a deterministic teacher, the model performs well but has considerable stochasticity which we interpret as regularization. Cosmological information in the predicted void catalogs outperforms the teacher. On the one hand, our method can cheaply emulate existing void finders with apparently useful regularization. More importantly, it also allows us to find the Bayes-optimal mapping between observed galaxies and any void definition. This includes definitions operating at the level of simulated matter density and velocity fields.

  </details>



- **Synthetic Aperture Communication: Principles and Application to Massive IoT Satellite Uplink**  
  Lucas Giroto, Marcus Henninger, Silvio Mandelli  
  _2026-02-16_ · https://arxiv.org/abs/2602.14629v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  While synthetic aperture radar is widely adopted to provide high-resolution imaging at long distances using small arrays, the concept of coherent synthetic aperture communication (SAC) has not yet been explored. This article introduces the principles of SAC for direct satellite-to-device uplink, showcasing precise direction-of-arrival estimation for user equipment (UE) devices, facilitating spatial signal separation, localization, and easing link budget constraints. Simulations for a low Earth orbit satellite at 600 km orbit and two UE devices performing orthogonal frequency-division multiplexing-based transmission with polar coding at 3.5 GHz demonstrate block error rates below 0.1 with transmission powers as low as -10 dBm, even under strong interference when UE devices are resolved but fall on each other's strongest angular sidelobe. These results validate the ability of the proposed scheme to address mutual interference and stringent power limitations, paving the way for massive Internet of Things connectivity in non-terrestrial networks.

  </details>



- **Automated Classification of Source Code Changes Based on Metrics Clustering in the Software Development Process**  
  Evgenii Kniazev  
  _2026-02-16_ · https://arxiv.org/abs/2602.14591v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  This paper presents an automated method for classifying source code changes during the software development process based on clustering of change metrics. The method consists of two steps: clustering of metric vectors computed for each code change, followed by expert mapping of the resulting clusters to predefined change classes. The distribution of changes into clusters is performed automatically, while the mapping of clusters to classes is carried out by an expert. Automation of the distribution step substantially reduces the time required for code change review. The k-means algorithm with a cosine similarity measure between metric vectors is used for clustering. Eleven source code metrics are employed, covering lines of code, cyclomatic complexity, file counts, interface changes, and structural changes. The method was validated on five software systems, including two open-source projects (Subversion and NHibernate), and demonstrated classification purity of P_C = 0.75 +/- 0.05 and entropy of E_C = 0.37 +/- 0.06 at a significance level of 0.05.

  </details>


