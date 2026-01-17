# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-01-17 06:44 UTC_

Total papers shown: **4**


---

- **SatMap: Revisiting Satellite Maps as Prior for Online HD Map Construction**  
  Kanak Mazumder, Fabian B. Flohr  
  _2026-01-15_ · https://arxiv.org/abs/2601.10512v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Online high-definition (HD) map construction is an essential part of a safe and robust end-to-end autonomous driving (AD) pipeline. Onboard camera-based approaches suffer from limited depth perception and degraded accuracy due to occlusion. In this work, we propose SatMap, an online vectorized HD map estimation method that integrates satellite maps with multi-view camera observations and directly predicts a vectorized HD map for downstream prediction and planning modules. Our method leverages lane-level semantics and texture from satellite imagery captured from a Bird's Eye View (BEV) perspective as a global prior, effectively mitigating depth ambiguity and occlusion. In our experiments on the nuScenes dataset, SatMap achieves 34.8% mAP performance improvement over the camera-only baseline and 8.5% mAP improvement over the camera-LiDAR fusion baseline. Moreover, we evaluate our model in long-range and adverse weather conditions to demonstrate the advantages of using a satellite prior map. Source code will be available at https://iv.ee.hm.edu/satmap/.

  </details>



- **DeepUrban: Interaction-Aware Trajectory Prediction and Planning for Automated Driving by Aerial Imagery**  
  Constantin Selzer, Fabian B. Flohr  
  _2026-01-15_ · https://arxiv.org/abs/2601.10554v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The efficacy of autonomous driving systems hinges critically on robust prediction and planning capabilities. However, current benchmarks are impeded by a notable scarcity of scenarios featuring dense traffic, which is essential for understanding and modeling complex interactions among road users. To address this gap, we collaborated with our industrial partner, DeepScenario, to develop DeepUrban-a new drone dataset designed to enhance trajectory prediction and planning benchmarks focusing on dense urban settings. DeepUrban provides a rich collection of 3D traffic objects, extracted from high-resolution images captured over urban intersections at approximately 100 meters altitude. The dataset is further enriched with comprehensive map and scene information to support advanced modeling and simulation tasks. We evaluate state-of-the-art (SOTA) prediction and planning methods, and conducted experiments on generalization capabilities. Our findings demonstrate that adding DeepUrban to nuScenes can boost the accuracy of vehicle predictions and planning, achieving improvements up to 44.1 % / 44.3% on the ADE / FDE metrics. Website: https://iv.ee.hm.edu/deepurban

  </details>



- **BikeActions: An Open Platform and Benchmark for Cyclist-Centric VRU Action Recognition**  
  Max A. Buettner, Kanak Mazumder, Luca Koecher, Mario Finkbeiner, Sebastian Niebler, Fabian B. Flohr  
  _2026-01-15_ · https://arxiv.org/abs/2601.10521v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Anticipating the intentions of Vulnerable Road Users (VRUs) is a critical challenge for safe autonomous driving (AD) and mobile robotics. While current research predominantly focuses on pedestrian crossing behaviors from a vehicle's perspective, interactions within dense shared spaces remain underexplored. To bridge this gap, we introduce FUSE-Bike, the first fully open perception platform of its kind. Equipped with two LiDARs, a camera, and GNSS, it facilitates high-fidelity, close-range data capture directly from a cyclist's viewpoint. Leveraging this platform, we present BikeActions, a novel multi-modal dataset comprising 852 annotated samples across 5 distinct action classes, specifically tailored to improve VRU behavior modeling. We establish a rigorous benchmark by evaluating state-of-the-art graph convolution and transformer-based models on our publicly released data splits, establishing the first performance baselines for this challenging task. We release the full dataset together with data curation tools, the open hardware design, and the benchmark code to foster future research in VRU action understanding under https://iv.ee.hm.edu/bikeactions/.

  </details>



- **See Less, Drive Better: Generalizable End-to-End Autonomous Driving via Foundation Models Stochastic Patch Selection**  
  Amir Mallak, Erfan Aasi, Shiva Sreeram, Tsun-Hsuan Wang, Daniela Rus, Alaa Maalouf  
  _2026-01-15_ · https://arxiv.org/abs/2601.10707v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Recent advances in end-to-end autonomous driving show that policies trained on patch-aligned features extracted from foundation models generalize better to Out-of-Distribution (OOD). We hypothesize that due to the self-attention mechanism, each patch feature implicitly embeds/contains information from all other patches, represented in a different way and intensity, making these descriptors highly redundant. We quantify redundancy in such (BLIP2) features via PCA and cross-patch similarity: $90$% of variance is captured by $17/64$ principal components, and strong inter-token correlations are pervasive. Training on such overlapping information leads the policy to overfit spurious correlations, hurting OOD robustness. We present Stochastic-Patch-Selection (SPS), a simple yet effective approach for learning policies that are more robust, generalizable, and efficient. For every frame, SPS randomly masks a fraction of patch descriptors, not feeding them to the policy model, while preserving the spatial layout of the remaining patches. Thus, the policy is provided with different stochastic but complete views of the (same) scene: every random subset of patches acts like a different, yet still sensible, coherent projection of the world. The policy thus bases its decisions on features that are invariant to which specific tokens survive. Extensive experiments confirm that across all OOD scenarios, our method outperforms the state of the art (SOTA), achieving a $6.2$% average improvement and up to $20.4$% in closed-loop simulations, while being $2.4\times$ faster. We conduct ablations over masking rates and patch-feature reorganization, training and evaluating 9 systems, with 8 of them surpassing prior SOTA. Finally, we show that the same learned policy transfers to a physical, real-world car without any tuning.

  </details>


