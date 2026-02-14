# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-14 07:00 UTC_

Total papers shown: **3**


---

- **DiffPlace: Street View Generation via Place-Controllable Diffusion Model Enhancing Place Recognition**  
  Ji Li, Zhiwei Li, Shihao Li, Zhenjiang Yu, Boyang Wang, Haiou Liu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11875v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Generative models have advanced significantly in realistic image synthesis, with diffusion models excelling in quality and stability. Recent multi-view diffusion models improve 3D-aware street view generation, but they struggle to produce place-aware and background-consistent urban scenes from text, BEV maps, and object bounding boxes. This limits their effectiveness in generating realistic samples for place recognition tasks. To address these challenges, we propose DiffPlace, a novel framework that introduces a place-ID controller to enable place-controllable multi-view image generation. The place-ID controller employs linear projection, perceiver transformer, and contrastive learning to map place-ID embeddings into a fixed CLIP space, allowing the model to synthesize images with consistent background buildings while flexibly modifying foreground objects and weather conditions. Extensive experiments, including quantitative comparisons and augmented training evaluations, demonstrate that DiffPlace outperforms existing methods in both generation quality and training support for visual place recognition. Our results highlight the potential of generative models in enhancing scene-level and place-aware synthesis, providing a valuable approach for improving place recognition in autonomous driving

  </details>



- **Talk2DM: Enabling Natural Language Querying and Commonsense Reasoning for Vehicle-Road-Cloud Integrated Dynamic Maps with Large Language Models**  
  Lu Tao, Jinxuan Luo, Yousuke Watanabe, Zhengshu Zhou, Yuhuan Lu, Shen Ying, Pan Zhang, Fei Zhao, Hiroaki Takada  
  _2026-02-12_ · https://arxiv.org/abs/2602.11860v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Dynamic maps (DM) serve as the fundamental information infrastructure for vehicle-road-cloud (VRC) cooperative autonomous driving in China and Japan. By providing comprehensive traffic scene representations, DM overcome the limitations of standalone autonomous driving systems (ADS), such as physical occlusions. Although DM-enhanced ADS have been successfully deployed in real-world applications in Japan, existing DM systems still lack a natural-language-supported (NLS) human interface, which could substantially enhance human-DM interaction. To address this gap, this paper introduces VRCsim, a VRC cooperative perception (CP) simulation framework designed to generate streaming VRC-CP data. Based on VRCsim, we construct a question-answering data set, VRC-QA, focused on spatial querying and reasoning in mixed-traffic scenes. Building upon VRCsim and VRC-QA, we further propose Talk2DM, a plug-and-play module that extends VRC-DM systems with NLS querying and commonsense reasoning capabilities. Talk2DM is built upon a novel chain-of-prompt (CoP) mechanism that progressively integrates human-defined rules with the commonsense knowledge of large language models (LLMs). Experiments on VRC-QA show that Talk2DM can seamlessly switch across different LLMs while maintaining high NLS query accuracy, demonstrating strong generalization capability. Although larger models tend to achieve higher accuracy, they incur significant efficiency degradation. Our results reveal that Talk2DM, powered by Qwen3:8B, Gemma3:27B, and GPT-oss models, achieves over 93\% NLS query accuracy with an average response time of only 2-5 seconds, indicating strong practical potential.

  </details>



- **Radio Map Prediction from Noisy Environment Information and Sparse Observations**  
  Fabian Jaensch, Çağkan Yapar, Giuseppe Caire, Begüm Demir  
  _2026-02-12_ · https://arxiv.org/abs/2602.11950v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Many works have investigated radio map and path loss prediction in wireless networks using deep learning, in particular using convolutional neural networks. However, most assume perfect environment information, which is unrealistic in practice due to sensor limitations, mapping errors, and temporal changes. We demonstrate that convolutional neural networks trained with task-specific perturbations of geometry, materials, and Tx positions can implicitly compensate for prediction errors caused by inaccurate environment inputs. When tested with noisy inputs on synthetic indoor scenarios, models trained with perturbed environment data reduce error by up to 25\% compared to models trained on clean data. We verify our approach on real-world measurements, achieving 2.1 dB RMSE with noisy input data and 1.3 dB with complete information, compared to 2.3-3.1 dB for classical methods such as ray-tracing and radial basis function interpolation. Additionally, we compare different ways of encoding environment information at varying levels of detail and we find that, in the considered single-room indoor scenarios, binary occupancy encoding performs at least as well as detailed material property information, simplifying practical deployment.

  </details>


