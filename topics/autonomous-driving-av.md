# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-02-18 07:14 UTC_

Total papers shown: **2**


---

- **SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms**  
  Haichao Liu, Yufeng Hu, Shuang Wang, Kangjun Guo, Jun Ma, Jinni Zhou  
  _2026-02-17_ · https://arxiv.org/abs/2602.15633v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Autonomous landing of Uncrewed Aerial Vehicles (UAVs) on oscillating marine platforms is severely constrained by wave-induced multi-frequency oscillations, wind disturbances, and prediction phase lags in motion prediction. Existing methods either treat platform motion as a general random process or lack explicit modeling of wave spectral characteristics, leading to suboptimal performance under dynamic sea conditions. To address these limitations, we propose SpecFuse: a novel spectral-temporal fusion predictive control framework that integrates frequency-domain wave decomposition with time-domain recursive state estimation for high-precision 6-DoF motion forecasting of Uncrewed Surface Vehicles (USVs). The framework explicitly models dominant wave harmonics to mitigate phase lags, refining predictions in real time via IMU data without relying on complex calibration. Additionally, we design a hierarchical control architecture featuring a sampling-based HPO-RRT* algorithm for dynamic trajectory planning under non-convex constraints and a learning-augmented predictive controller that fuses data-driven disturbance compensation with optimization-based execution. Extensive validations (2,000 simulations + 8 lake experiments) show our approach achieves a 3.2 cm prediction error, 4.46 cm landing deviation, 98.7% / 87.5% success rates (simulation / real-world), and 82 ms latency on embedded hardware, outperforming state-of-the-art methods by 44%-48% in accuracy. Its robustness to wave-wind coupling disturbances supports critical maritime missions such as search and rescue and environmental monitoring. All code, experimental configurations, and datasets will be released as open-source to facilitate reproducibility.

  </details>



- **RPT-SR: Regional Prior attention Transformer for infrared image Super-Resolution**  
  Youngwan Jin, Incheol Park, Yagiz Nalcakan, Hyeongjin Ju, Sanghyeop Yeo, Shiho Kim  
  _2026-02-17_ · https://arxiv.org/abs/2602.15490v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  General-purpose super-resolution models, particularly Vision Transformers, have achieved remarkable success but exhibit fundamental inefficiencies in common infrared imaging scenarios like surveillance and autonomous driving, which operate from fixed or nearly-static viewpoints. These models fail to exploit the strong, persistent spatial priors inherent in such scenes, leading to redundant learning and suboptimal performance. To address this, we propose the Regional Prior attention Transformer for infrared image Super-Resolution (RPT-SR), a novel architecture that explicitly encodes scene layout information into the attention mechanism. Our core contribution is a dual-token framework that fuses (1) learnable, regional prior tokens, which act as a persistent memory for the scene's global structure, with (2) local tokens that capture the frame-specific content of the current input. By utilizing these tokens into an attention, our model allows the priors to dynamically modulate the local reconstruction process. Extensive experiments validate our approach. While most prior works focus on a single infrared band, we demonstrate the broad applicability and versatility of RPT-SR by establishing new state-of-the-art performance across diverse datasets covering both Long-Wave (LWIR) and Short-Wave (SWIR) spectra

  </details>


