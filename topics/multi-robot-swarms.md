# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-01-23 06:51 UTC_

Total papers shown: **2**


---

- **AgriPINN: A Process-Informed Neural Network for Interpretable and Scalable Crop Biomass Prediction Under Water Stress**  
  Yue Shi, Liangxiu Han, Xin Zhang, Tam Sobeih, Thomas Gaiser, Nguyen Huu Thuy, Dominik Behrend, Amit Kumar Srivastava, Krishnagopal Halder, Frank Ewert  
  _2026-01-22_ · https://arxiv.org/abs/2601.16045v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Accurate prediction of crop above-ground biomass (AGB) under water stress is critical for monitoring crop productivity, guiding irrigation, and supporting climate-resilient agriculture. Data-driven models scale well but often lack interpretability and degrade under distribution shift, whereas process-based crop models (e.g. DSSAT, APSIM, LINTUL5) require extensive calibration and are difficult to deploy over large spatial domains. To address these limitations, we propose AgriPINN, a process-informed neural network that integrates a biophysical crop-growth differential equation as a differentiable constraint within a deep learning backbone. This design encourages physiologically consistent biomass dynamics under water-stress conditions while preserving model scalability for spatially distributed AGB prediction. AgriPINN recovers latent physiological variables, including leaf area index (LAI), absorbed photosynthetically active radiation (PAR), radiation use efficiency (RUE), and water-stress factors, without requiring direct supervision. We pretrain AgriPINN on 60 years of historical data across 397 regions in Germany and fine-tune it on three years of field experiments under controlled water treatments. Results show that AgriPINN consistently outperforms state-of-the-art deep-learning baselines (ConvLSTM-ViT, SLTF, CNN-Transformer) and the process-based LINTUL5 model in terms of accuracy (RMSE reductions up to $43\%$) and computational efficiency. By combining the scalability of deep learning with the biophysical rigor of process-based modeling, AgriPINN provides a robust and interpretable framework for spatio-temporal AGB prediction, offering practical value for planning of irrigation infrastructure, yield forecasting, and climate-adaptation planning.

  </details>



- **Distributed Multichannel Active Noise Control with Asynchronous Communication**  
  Junwei Ji, Dongyuan Shi, Boxiang Wang, Ziyi Yang, Haowen Li, Woon-Seng Gan  
  _2026-01-22_ · https://arxiv.org/abs/2601.15653v1 · `eess.AS`  
  <details><summary>Abstract</summary>

  Distributed multichannel active noise control (DMCANC) offers effective noise reduction across large spatial areas by distributing the computational load of centralized control to multiple low-cost nodes. Conventional DMCANC methods, however, typically assume synchronous communication and require frequent data exchange, resulting in high communication overhead. To enhance efficiency and adaptability, this work proposes an asynchronous communication strategy where each node executes a weight-constrained filtered-x LMS (WCFxLMS) algorithm and independently requests communication only when its local noise reduction performance degrades. Upon request, other nodes transmit the weight difference between their local control filter and the center point in WCFxLMS, which are then integrated to update both the control filter and the center point. This design enables nodes to operate asynchronously while preserving cooperative behavior. Simulation results demonstrate that the proposed asynchronous communication DMCANC (ACDMCANC) system maintains effective noise reduction with significantly reduced communication load, offering improved scalability for heterogeneous networks.

  </details>


