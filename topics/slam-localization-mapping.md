# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-01-29 07:02 UTC_

Total papers shown: **5**


---

- **Vibro-Sense: Robust Vibration-based Impulse Response Localization and Trajectory Tracking for Robotic Hands**  
  Wadhah Zai El Amri, Nicolás Navarro-Guerrero  
  _2026-01-28_ · https://arxiv.org/abs/2601.20555v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Rich contact perception is crucial for robotic manipulation, yet traditional tactile skins remain expensive and complex to integrate. This paper presents a scalable alternative: high-accuracy whole-body touch localization via vibro-acoustic sensing. By equipping a robotic hand with seven low-cost piezoelectric microphones and leveraging an Audio Spectrogram Transformer, we decode the vibrational signatures generated during physical interaction. Extensive evaluation across stationary and dynamic tasks reveals a localization error of under 5 mm in static conditions. Furthermore, our analysis highlights the distinct influence of material properties: stiff materials (e.g., metal) excel in impulse response localization due to sharp, high-bandwidth responses, whereas textured materials (e.g., wood) provide superior friction-based features for trajectory tracking. The system demonstrates robustness to the robot's own motion, maintaining effective tracking even during active operation. Our primary contribution is demonstrating that complex physical contact dynamics can be effectively decoded from simple vibrational signals, offering a viable pathway to widespread, affordable contact perception in robotics. To accelerate research, we provide our full datasets, models, and experimental setups as open-source resources.

  </details>



- **Statistical Properties of Target Localization Using Passive Radar Systems**  
  Mats Viberg, Daniele Gerosa, Tomas McKelvey, Thomas Eriksson  
  _2026-01-28_ · https://arxiv.org/abs/2601.20817v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Passive Radar Systems have received tremendous attention during the past few decades, due to their low cost and ability to remain covert during operation. Such systems do not transmit any energy themselves, but rely on a so-called Illuminator-of-Opportunity (IO), for example a commercial TV station. A network of Receiving Nodes (RN) receive the direct signal as well as reflections from possible targets. The RNs transmit information to a Central Node (CN), that performs the final target detection, localization and tracking. A large number of methods and algorithms for target detection and localization have been proposed in the literature. In the present contribution, the focus is on the seminal Extended Cancelation Algorithm (ECA), in which each RN estimates target parameters after canceling interference from the direct-path as well as clutter from unwanted stationary objects. This is done by exploiting a separate Reference Channel (RC), which captures the IO signal without interference apart from receiver noise. We derive the statistical properties of the ECA parameter estimates under the assumption of a high Signal-to-Noise Ratio (SNR), and we give a sufficient condition for the SNR in the RC to enable statistically efficient estimates. The theoretical results are corroborated through computer simulations, which show that the theory agrees well with empirical results above a certain SNR threshold. The results can be used to predict the performance of passive radar systems in given scenarios, which is useful for feasibility studies as well as system design.

  </details>



- **User Localization via Active Sensing with Electromagnetically Reconfigurable Antennas**  
  Ruizhi Zhang, Yuchen Zhang, Ying Zhang  
  _2026-01-28_ · https://arxiv.org/abs/2601.20501v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  This paper presents an end-to-end deep learning framework for electromagnetically reconfigurable antenna (ERA)-aided user localization with active sensing, where ERAs provide additional electromagnetic reconfigurability to diversify the received measurements and enhance localization informativeness. To balance sensing flexibility and overhead, we adopt a two-timescale design: the digital combiner is updated at each stage, while the ERA patterns are reconfigured at each substage via a spherical-harmonic representation. The proposed mechanism integrates attention-based feature extraction and LSTM-based temporal learning, enabling the system to learn an optimized sensing strategy and progressively refine the UE position estimate from sequential observations. Simulation results show that the proposed approach consistently outperforms conventional digital beamforming-only and single-stage sensing baselines in terms of localization accuracy. These results highlight the effectiveness of ERA-enabled active sensing for user localization in future wireless systems.

  </details>



- **MARE: Multimodal Alignment and Reinforcement for Explainable Deepfake Detection via Vision-Language Models**  
  Wenbo Xu, Wei Lu, Xiangyang Luo, Jiantao Zhou  
  _2026-01-28_ · https://arxiv.org/abs/2601.20433v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Deepfake detection is a widely researched topic that is crucial for combating the spread of malicious content, with existing methods mainly modeling the problem as classification or spatial localization. The rapid advancements in generative models impose new demands on Deepfake detection. In this paper, we propose multimodal alignment and reinforcement for explainable Deepfake detection via vision-language models, termed MARE, which aims to enhance the accuracy and reliability of Vision-Language Models (VLMs) in Deepfake detection and reasoning. Specifically, MARE designs comprehensive reward functions, incorporating reinforcement learning from human feedback (RLHF), to incentivize the generation of text-spatially aligned reasoning content that adheres to human preferences. Besides, MARE introduces a forgery disentanglement module to capture intrinsic forgery traces from high-level facial semantics, thereby improving its authenticity detection capability. We conduct thorough evaluations on the reasoning content generated by MARE. Both quantitative and qualitative experimental results demonstrate that MARE achieves state-of-the-art performance in terms of accuracy and reliability.

  </details>



- **Graph-Structured Deep Learning Framework for Multi-task Contention Identification with High-dimensional Metrics**  
  Xiao Yang, Yinan Ni, Yuqi Tang, Zhimin Qiu, Chen Wang, Tingzhou Yuan  
  _2026-01-28_ · https://arxiv.org/abs/2601.20389v1 · `cs.DC`  
  <details><summary>Abstract</summary>

  This study addresses the challenge of accurately identifying multi-task contention types in high-dimensional system environments and proposes a unified contention classification framework that integrates representation transformation, structural modeling, and a task decoupling mechanism. The method first constructs system state representations from high-dimensional metric sequences, applies nonlinear transformations to extract cross-dimensional dynamic features, and integrates multiple source information such as resource utilization, scheduling behavior, and task load variations within a shared representation space. It then introduces a graph-based modeling mechanism to capture latent dependencies among metrics, allowing the model to learn competitive propagation patterns and structural interference across resource links. On this basis, task-specific mapping structures are designed to model the differences among contention types and enhance the classifier's ability to distinguish multiple contention patterns. To achieve stable performance, the method employs an adaptive multi-task loss weighting strategy that balances shared feature learning with task-specific feature extraction and generates final contention predictions through a standardized inference process. Experiments conducted on a public system trace dataset demonstrate advantages in accuracy, recall, precision, and F1, and sensitivity analyses on batch size, training sample scale, and metric dimensionality further confirm the model's stability and applicability. The study shows that structured representations and multi-task classification based on high-dimensional metrics can significantly improve contention pattern recognition and offer a reliable technical approach for performance management in complex computing environments.

  </details>


