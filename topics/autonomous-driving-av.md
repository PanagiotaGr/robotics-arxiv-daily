# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-01-22 06:51 UTC_

Total papers shown: **2**


---

- **DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration**  
  Dominik Rößle, Xujun Xie, Adithya Mohan, Venkatesh Thirugnana Sambandham, Daniel Cremers, Torsten Schön  
  _2026-01-21_ · https://arxiv.org/abs/2601.15260v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Perception is a cornerstone of autonomous driving, enabling vehicles to understand their surroundings and make safe, reliable decisions. Developing robust perception algorithms requires large-scale, high-quality datasets that cover diverse driving conditions and support thorough evaluation. Existing datasets often lack a high-fidelity digital twin, limiting systematic testing, edge-case simulation, sensor modification, and sim-to-real evaluations. To address this gap, we present DrivIng, a large-scale multimodal dataset with a complete geo-referenced digital twin of a ~18 km route spanning urban, suburban, and highway segments. Our dataset provides continuous recordings from six RGB cameras, one LiDAR, and high-precision ADMA-based localization, captured across day, dusk, and night. All sequences are annotated at 10 Hz with 3D bounding boxes and track IDs across 12 classes, yielding ~1.2 million annotated instances. Alongside the benefits of a digital twin, DrivIng enables a 1-to-1 transfer of real traffic into simulation, preserving agent interactions while enabling realistic and flexible scenario testing. To support reproducible research and robust validation, we benchmark DrivIng with state-of-the-art perception models and publicly release the dataset, digital twin, HD map, and codebase.

  </details>



- **AutoDriDM: An Explainable Benchmark for Decision-Making of Vision-Language Models in Autonomous Driving**  
  Zecong Tang, Zixu Wang, Yifei Wang, Weitong Lian, Tianjian Gao, Haoran Li, Tengju Ru, Lingyi Meng, Zhejun Cui, Yichen Zhu, et al.  
  _2026-01-21_ · https://arxiv.org/abs/2601.14702v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Autonomous driving is a highly challenging domain that requires reliable perception and safe decision-making in complex scenarios. Recent vision-language models (VLMs) demonstrate reasoning and generalization abilities, opening new possibilities for autonomous driving; however, existing benchmarks and metrics overemphasize perceptual competence and fail to adequately assess decision-making processes. In this work, we present AutoDriDM, a decision-centric, progressive benchmark with 6,650 questions across three dimensions - Object, Scene, and Decision. We evaluate mainstream VLMs to delineate the perception-to-decision capability boundary in autonomous driving, and our correlation analysis reveals weak alignment between perception and decision-making performance. We further conduct explainability analyses of models' reasoning processes, identifying key failure modes such as logical reasoning errors, and introduce an analyzer model to automate large-scale annotation. AutoDriDM bridges the gap between perception-centered and decision-centered evaluation, providing guidance toward safer and more reliable VLMs for real-world autonomous driving.

  </details>


