# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-01-29 07:02 UTC_

Total papers shown: **2**


---

- **A New Dataset and Framework for Robust Road Surface Classification via Camera-IMU Fusion**  
  Willams de Lima Costa, Thifany Ketuli Silva de Souza, Jonas Ferreira Silva, Carlos Gabriel Bezerra Pereira, Bruno Reis Vila Nova, Leonardo Silvino Brito, Rafael Raider Leoni, Juliano Silva, Valter Ferreira, Sibele Miguel Soares Neto, et al.  
  _2026-01-28_ · https://arxiv.org/abs/2601.20847v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Road surface classification (RSC) is a key enabler for environment-aware predictive maintenance systems. However, existing RSC techniques often fail to generalize beyond narrow operational conditions due to limited sensing modalities and datasets that lack environmental diversity. This work addresses these limitations by introducing a multimodal framework that fuses images and inertial measurements using a lightweight bidirectional cross-attention module followed by an adaptive gating layer that adjusts modality contributions under domain shifts. Given the limitations of current benchmarks, especially regarding lack of variability, we introduce ROAD, a new dataset composed of three complementary subsets: (i) real-world multimodal recordings with RGB-IMU streams synchronized using a gold-standard industry datalogger, captured across diverse lighting, weather, and surface conditions; (ii) a large vision-only subset designed to assess robustness under adverse illumination and heterogeneous capture setups; and (iii) a synthetic subset generated to study out-of-distribution generalization in scenarios difficult to obtain in practice. Experiments show that our method achieves a +1.4 pp improvement over the previous state-of-the-art on the PVS benchmark and an +11.6 pp improvement on our multimodal ROAD subset, with consistently higher F1-scores on minority classes. The framework also demonstrates stable performance across challenging visual conditions, including nighttime, heavy rain, and mixed-surface transitions. These findings indicate that combining affordable camera and IMU sensors with multimodal attention mechanisms provides a scalable, robust foundation for road surface understanding, particularly relevant for regions where environmental variability and cost constraints limit the adoption of high-end sensing suites.

  </details>



- **CoBA: Integrated Deep Learning Model for Reliable Low-Altitude UAV Classification in mmWave Radio Networks**  
  Junaid Sajid, Ivo Müürsepp, Luca Reggiani, Davide Scazzoli, Federico Francesco Luigi Mariani, Maurizio Magarini, Rizwan Ahmad, Muhammad Mahtab Alam  
  _2026-01-28_ · https://arxiv.org/abs/2601.20605v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Uncrewed Aerial Vehicles (UAVs) are increasingly used in civilian and industrial applications, making secure low-altitude operations crucial. In dense mmWave environments, accurately classifying low-altitude UAVs as either inside authorized or restricted airspaces remains challenging, requiring models that handle complex propagation and signal variability. This paper proposes a deep learning model, referred to as CoBA, which stands for integrated Convolutional Neural Network (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and Attention which leverages Fifth Generation (5G) millimeter-wave (mmWave) radio measurements to classify UAV operations in authorized and restricted airspaces at low altitude. The proposed CoBA model integrates convolutional, bidirectional recurrent, and attention layers to capture both spatial and temporal patterns in UAV radio measurements. To validate the model, a dedicated dataset is collected using the 5G mmWave network at TalTech, with controlled low altitude UAV flights in authorized and restricted scenarios. The model is evaluated against conventional ML models and a fingerprinting-based benchmark. Experimental results show that CoBA achieves superior accuracy, significantly outperforming all baseline models and demonstrating its potential for reliable and regulated UAV airspace monitoring.

  </details>


