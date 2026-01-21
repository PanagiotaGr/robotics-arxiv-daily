# Safety, Robustness, Uncertainty

_Robotics arXiv Daily_

_Updated: 2026-01-21 06:53 UTC_

Total papers shown: **12**


---

- **GuideTouch: An Obstacle Avoidance Device for Visually Impaired**  
  Timofei Kozlov, Artem Trandofilov, Georgii Gazaryan, Issatay Tokmurziyev, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13813v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Safe navigation for the visually impaired individuals remains a critical challenge, especially concerning head-level obstacles, which traditional mobility aids often fail to detect. We introduce GuideTouch, a compact, affordable, standalone wearable device designed for autonomous obstacle avoidance. The system integrates two vertically aligned Time-of-Flight (ToF) sensors, enabling three-dimensional environmental perception, and four vibrotactile actuators that provide directional haptic feedback. Proximity and direction information is communicated via an intuitive 4-point vibrotactile feedback system located across the user's shoulders and upper chest. For real-world robustness, the device includes a unique centrifugal self-cleaning optical cover mechanism and a sound alarm system for location if the device is dropped. We evaluated the haptic perception accuracy across 22 participants (17 male and 5 female, aged 21-48, mean 25.7, sd 6.1). Statistical analysis confirmed a significant difference between the perception accuracy of different patterns. The system demonstrated high recognition accuracy, achieving an average of 92.9% for single and double motor (primary directional) patterns. Furthermore, preliminary experiments with 14 visually impaired users validated this interface, showing a recognition accuracy of 93.75% for primary directional cues. The results demonstrate that GuideTouch enables intuitive spatial perception and could significantly improve the safety, confidence, and autonomy of users with visual impairments during independent navigation.

  </details>



- **DroneVLA: VLA based Aerial Manipulation**  
  Fawad Mehboob, Monijesu James, Amir Habel, Jeffrin Sam, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13809v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  As aerial platforms evolve from passive observers to active manipulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally. This work introduces a novel concept of autonomous aerial manipulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user. The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom-built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera. VLA performs semantic reasoning to interpret the intent of a user prompt and generates a prioritized task queue for grasping of relevant objects in the scene. Grounding DINO and dynamic A* planning algorithm are used to navigate and safely relocate the object. To ensure safe and natural interaction during the handover phase, the system employs a human-centric controller driven by MediaPipe. This module provides real-time human pose estimation, allowing the drone to employ visual servoing to maintain a stable, distinct position directly in front of the user, facilitating a comfortable handover. We demonstrate the system's efficacy through real-world experiments for localization and navigation, which resulted in a 0.164m, 0.070m, and 0.084m of max, mean euclidean, and root-mean squared errors, respectively, highlighting the feasibility of VLA for aerial manipulation operations.

  </details>



- **Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep Reinforcement Learning**  
  Myong-Yol Choi, Hankyoul Ko, Hanse Cho, Changseung Kim, Seunghwan Kim, Jaemin Seo, Hyondong Oh  
  _2026-01-20_ · https://arxiv.org/abs/2601.13657v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a deep reinforcement learning (DRL) based controller for collective navigation of unmanned aerial vehicle (UAV) swarms in communication-denied environments, enabling robust operation in complex, obstacle-rich environments. Inspired by biological swarms where informed individuals guide groups without explicit communication, we employ an implicit leader-follower framework. In this paradigm, only the leader possesses goal information, while follower UAVs learn robust policies using only onboard LiDAR sensing, without requiring any inter-agent communication or leader identification. Our system utilizes LiDAR point clustering and an extended Kalman filter for stable neighbor tracking, providing reliable perception independent of external positioning systems. The core of our approach is a DRL controller, trained in GPU-accelerated Nvidia Isaac Sim, that enables followers to learn complex emergent behaviors - balancing flocking and obstacle avoidance - using only local perception. This allows the swarm to implicitly follow the leader while robustly addressing perceptual challenges such as occlusion and limited field-of-view. The robustness and sim-to-real transfer of our approach are confirmed through extensive simulations and challenging real-world experiments with a swarm of five UAVs, which successfully demonstrated collective navigation across diverse indoor and outdoor environments without any communication or external localization.

  </details>



- **Zero-shot adaptable task planning for autonomous construction robots: a comparative study of lightweight single and multi-AI agent systems**  
  Hossein Naderi, Alireza Shojaei, Lifu Huang, Philip Agee, Kereshmeh Afsari, Abiola Akanmu  
  _2026-01-20_ · https://arxiv.org/abs/2601.14091v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robots are expected to play a major role in the future construction industry but face challenges due to high costs and difficulty adapting to dynamic tasks. This study explores the potential of foundation models to enhance the adaptability and generalizability of task planning in construction robots. Four models are proposed and implemented using lightweight, open-source large language models (LLMs) and vision language models (VLMs). These models include one single agent and three multi-agent teams that collaborate to create robot action plans. The models are evaluated across three construction roles: Painter, Safety Inspector, and Floor Tiling. Results show that the four-agent team outperforms the state-of-the-art GPT-4o in most metrics while being ten times more cost-effective. Additionally, teams with three and four agents demonstrate the improved generalizability. By discussing how agent behaviors influence outputs, this study enhances the understanding of AI teams and supports future research in diverse unstructured environments beyond construction.

  </details>



- **Diffusion-Guided Backdoor Attacks in Real-World Reinforcement Learning**  
  Tairan Huang, Qingqing Ye, Yulin Jin, Jiawei Lian, Yi Wang, Haibo Hu  
  _2026-01-20_ · https://arxiv.org/abs/2601.14104v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Backdoor attacks embed hidden malicious behaviors in reinforcement learning (RL) policies and activate them using triggers at test time. Most existing attacks are validated only in simulation, while their effectiveness in real-world robotic systems remains unclear. In physical deployment, safety-constrained control pipelines such as velocity limiting, action smoothing, and collision avoidance suppress abnormal actions, causing strong attenuation of conventional backdoor attacks. We study this previously overlooked problem and propose a diffusion-guided backdoor attack framework (DGBA) for real-world RL. We design small printable visual patch triggers placed on the floor and generate them using a conditional diffusion model that produces diverse patch appearances under real-world visual variations. We treat the robot control stack as a black-box system. We further introduce an advantage-based poisoning strategy that injects triggers only at decision-critical training states. We evaluate our method on a TurtleBot3 mobile robot and demonstrate reliable activation of targeted attacks while preserving normal task performance. Demo videos and code are available in the supplementary material.

  </details>



- **HoverAI: An Embodied Aerial Agent for Natural Human-Drone Interaction**  
  Yuhua Jin, Nikita Kuzmin, Georgii Demianchuk, Mariya Lezina, Fawad Mehboob, Issatay Tokmurziyev, Miguel Altamirano Cabrera, Muhammad Ahsan Mustafa, Dzmitry Tsetserukou  
  _2026-01-20_ · https://arxiv.org/abs/2601.13801v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Drones operating in human-occupied spaces suffer from insufficient communication mechanisms that create uncertainty about their intentions. We present HoverAI, an embodied aerial agent that integrates drone mobility, infrastructure-independent visual projection, and real-time conversational AI into a unified platform. Equipped with a MEMS laser projector, onboard semi-rigid screen, and RGB camera, HoverAI perceives users through vision and voice, responding via lip-synced avatars that adapt appearance to user demographics. The system employs a multimodal pipeline combining VAD, ASR (Whisper), LLM-based intent classification, RAG for dialogue, face analysis for personalization, and voice synthesis (XTTS v2). Evaluation demonstrates high accuracy in command recognition (F1: 0.90), demographic estimation (gender F1: 0.89, age MAE: 5.14 years), and speech transcription (WER: 0.181). By uniting aerial robotics with adaptive conversational AI and self-contained visual output, HoverAI introduces a new class of spatially-aware, socially responsive embodied agents for applications in guidance, assistance, and human-centered interaction.

  </details>



- **LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR**  
  Said Taghadouini, Adrien Cavaillès, Baptiste Aubertin  
  _2026-01-20_ · https://arxiv.org/abs/2601.14251v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present \textbf{LightOnOCR-2-1B}, a 1B-parameter end-to-end multilingual vision--language model that converts document images (e.g., PDFs) into clean, naturally ordered text without brittle OCR pipelines. Trained on a large-scale, high-quality distillation mix with strong coverage of scans, French documents, and scientific PDFs, LightOnOCR-2 achieves state-of-the-art results on OlmOCR-Bench while being 9$\times$ smaller and substantially faster than prior best-performing models. We further extend the output format to predict normalized bounding boxes for embedded images, introducing localization during pretraining via a resume strategy and refining it with RLVR using IoU-based rewards. Finally, we improve robustness with checkpoint averaging and task-arithmetic merging. We release model checkpoints under Apache 2.0, and publicly release the dataset and \textbf{LightOnOCR-bbox-bench} evaluation under their respective licenses.

  </details>



- **Robust Localization in OFDM-Based Massive MIMO through Phase Offset Calibration**  
  Qing Zhang, Adham Sakhnini, Robbert Beerten, Haoqiu Xiong, Zhuangzhuang Cui, Yang Miao, Sofie Pollin  
  _2026-01-20_ · https://arxiv.org/abs/2601.14244v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Accurate localization in Orthogonal Frequency Division Multiplexing (OFDM)-based massive Multiple-Input Multiple-Output (MIMO) systems depends critically on phase coherence across subcarriers and antennas. However, practical systems suffer from frequency-dependent and (spatial) antenna-dependent phase offsets, degrading localization accuracy. This paper analytically studies the impact of phase incoherence on localization performance under a static User Equipment (UE) and Line-of-Sight (LoS) scenario. We use two complementary tools. First, we derive the Cramér-Rao Lower Bound (CRLB) to quantify the theoretical limits under phase offsets. Then, we develop a Spatial Ambiguity Function (SAF)-based model to characterize ambiguity patterns. Simulation results reveal that spatial phase offsets severely degrade localization performance, while frequency phase offsets have a minor effect in the considered system configuration. To address this, we propose a robust Channel State Information (CSI) calibration framework and validate it using real-world measurements from a practical massive MIMO testbed. The experimental results confirm that the proposed calibration framework significantly improves the localization Root Mean Squared Error (RMSE) from 5 m to 1.2 cm, aligning well with the theoretical predictions.

  </details>



- **DExTeR: Weakly Semi-Supervised Object Detection with Class and Instance Experts for Medical Imaging**  
  Adrien Meyer, Didier Mutter, Nicolas Padoy  
  _2026-01-20_ · https://arxiv.org/abs/2601.13954v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Detecting anatomical landmarks in medical imaging is essential for diagnosis and intervention guidance. However, object detection models rely on costly bounding box annotations, limiting scalability. Weakly Semi-Supervised Object Detection (WSSOD) with point annotations proposes annotating each instance with a single point, minimizing annotation time while preserving localization signals. A Point-to-Box teacher model, trained on a small box-labeled subset, converts these point annotations into pseudo-box labels to train a student detector. Yet, medical imagery presents unique challenges, including overlapping anatomy, variable object sizes, and elusive structures, which hinder accurate bounding box inference. To overcome these challenges, we introduce DExTeR (DETR with Experts), a transformer-based Point-to-Box regressor tailored for medical imaging. Built upon Point-DETR, DExTeR encodes single-point annotations as object queries, refining feature extraction with the proposed class-guided deformable attention, which guides attention sampling using point coordinates and class labels to capture class-specific characteristics. To improve discrimination in complex structures, it introduces CLICK-MoE (CLass, Instance, and Common Knowledge Mixture of Experts), decoupling class and instance representations to reduce confusion among adjacent or overlapping instances. Finally, we implement a multi-point training strategy which promotes prediction consistency across different point placements, improving robustness to annotation variability. DExTeR achieves state-of-the-art performance across three datasets spanning different medical domains (endoscopy, chest X-rays, and endoscopic ultrasound) highlighting its potential to reduce annotation costs while maintaining high detection accuracy.

  </details>



- **TractRLFusion: A GPT-Based Multi-Critic Policy Fusion Framework for Fiber Tractography**  
  Ankita Joshi, Ashutosh Sharma, Anoushkrit Goel, Ranjeet Ranjan Jha, Chirag Ahuja, Arnav Bhavsar, Aditya Nigam  
  _2026-01-20_ · https://arxiv.org/abs/2601.13897v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Tractography plays a pivotal role in the non-invasive reconstruction of white matter fiber pathways, providing vital information on brain connectivity and supporting precise neurosurgical planning. Although traditional methods relied mainly on classical deterministic and probabilistic approaches, recent progress has benefited from supervised deep learning (DL) and deep reinforcement learning (DRL) to improve tract reconstruction. A persistent challenge in tractography is accurately reconstructing white matter tracts while minimizing spurious connections. To address this, we propose TractRLFusion, a novel GPT-based policy fusion framework that integrates multiple RL policies through a data-driven fusion strategy. Our method employs a two-stage training data selection process for effective policy fusion, followed by a multi-critic fine-tuning phase to enhance robustness and generalization. Experiments on HCP, ISMRM, and TractoInferno datasets demonstrate that TractRLFusion outperforms individual RL policies as well as state-of-the-art classical and DRL methods in accuracy and anatomical reliability.

  </details>



- **DisasterVQA: A Visual Question Answering Benchmark Dataset for Disaster Scenes**  
  Aisha Al-Mohannadi, Ayisha Firoz, Yin Yang, Muhammad Imran, Ferda Ofli  
  _2026-01-20_ · https://arxiv.org/abs/2601.13839v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Social media imagery provides a low-latency source of situational information during natural and human-induced disasters, enabling rapid damage assessment and response. While Visual Question Answering (VQA) has shown strong performance in general-purpose domains, its suitability for the complex and safety-critical reasoning required in disaster response remains unclear. We introduce DisasterVQA, a benchmark dataset designed for perception and reasoning in crisis contexts. DisasterVQA consists of 1,395 real-world images and 4,405 expert-curated question-answer pairs spanning diverse events such as floods, wildfires, and earthquakes. Grounded in humanitarian frameworks including FEMA ESF and OCHA MIRA, the dataset includes binary, multiple-choice, and open-ended questions covering situational awareness and operational decision-making tasks. We benchmark seven state-of-the-art vision-language models and find performance variability across question types, disaster categories, regions, and humanitarian tasks. Although models achieve high accuracy on binary questions, they struggle with fine-grained quantitative reasoning, object counting, and context-sensitive interpretation, particularly for underrepresented disaster scenarios. DisasterVQA provides a challenging and practical benchmark to guide the development of more robust and operationally meaningful vision-language models for disaster response. The dataset is publicly available at https://zenodo.org/records/18267770.

  </details>



- **Towards Token-Level Text Anomaly Detection**  
  Yang Cao, Bicheng Yu, Sikun Yang, Ming Liu, Yujiu Yang  
  _2026-01-20_ · https://arxiv.org/abs/2601.13644v1 · `cs.CL`  
  <details><summary>Abstract</summary>

  Despite significant progress in text anomaly detection for web applications such as spam filtering and fake news detection, existing methods are fundamentally limited to document-level analysis, unable to identify which specific parts of a text are anomalous. We introduce token-level anomaly detection, a novel paradigm that enables fine-grained localization of anomalies within text. We formally define text anomalies at both document and token-levels, and propose a unified detection framework that operates across multiple levels. To facilitate research in this direction, we collect and annotate three benchmark datasets spanning spam, reviews and grammar errors with token-level labels. Experimental results demonstrate that our framework get better performance than other 6 baselines, opening new possibilities for precise anomaly localization in text. All the codes and data are publicly available on https://github.com/charles-cao/TokenCore.

  </details>


