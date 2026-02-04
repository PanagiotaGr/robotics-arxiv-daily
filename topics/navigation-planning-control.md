# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-04 07:06 UTC_

Total papers shown: **7**


---

- **Conformal Reachability for Safe Control in Unknown Environments**  
  Xinhang Ma, Junlin Wu, Yiannis Kantaros, Yevgeniy Vorobeychik  
  _2026-02-03_ · https://arxiv.org/abs/2602.03799v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Designing provably safe control is a core problem in trustworthy autonomy. However, most prior work in this regard assumes either that the system dynamics are known or deterministic, or that the state and action space are finite, significantly limiting application scope. We address this limitation by developing a probabilistic verification framework for unknown dynamical systems which combines conformal prediction with reachability analysis. In particular, we use conformal prediction to obtain valid uncertainty intervals for the unknown dynamics at each time step, with reachability then verifying whether safety is maintained within the conformal uncertainty bounds. Next, we develop an algorithmic approach for training control policies that optimize nominal reward while also maximizing the planning horizon with sound probabilistic safety guarantees. We evaluate the proposed approach in seven safe control settings spanning four domains -- cartpole, lane following, drone control, and safe navigation -- for both affine and nonlinear safety specifications. Our experiments show that the policies we learn achieve the strongest provable safety guarantees while still maintaining high average reward.

  </details>



- **FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation**  
  Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Mingjie Zhan, Hongsheng Li  
  _2026-02-03_ · https://arxiv.org/abs/2602.03798v1 · `cs.SE`  
  <details><summary>Abstract</summary>

  Assisting non-expert users to develop complex interactive websites has become a popular task for LLM-powered code agents. However, existing code agents tend to only generate frontend web pages, masking the lack of real full-stack data processing and storage with fancy visual effects. Notably, constructing production-level full-stack web applications is far more challenging than only generating frontend web pages, demanding careful control of data flow, comprehensive understanding of constantly updating packages and dependencies, and accurate localization of obscure bugs in the codebase. To address these difficulties, we introduce FullStack-Agent, a unified agent system for full-stack agentic coding that consists of three parts: (1) FullStack-Dev, a multi-agent framework with strong planning, code editing, codebase navigation, and bug localization abilities. (2) FullStack-Learn, an innovative data-scaling and self-improving method that back-translates crawled and synthesized website repositories to improve the backbone LLM of FullStack-Dev. (3) FullStack-Bench, a comprehensive benchmark that systematically tests the frontend, backend and database functionalities of the generated website. Our FullStack-Dev outperforms the previous state-of-the-art method by 8.7%, 38.2%, and 15.9% on the frontend, backend, and database test cases respectively. Additionally, FullStack-Learn raises the performance of a 30B model by 9.7%, 9.5%, and 2.8% on the three sets of test cases through self-improvement, demonstrating the effectiveness of our approach. The code is released at https://github.com/mnluzimu/FullStack-Agent.

  </details>



- **Deep-Learning-Based Control of a Decoupled Two-Segment Continuum Robot for Endoscopic Submucosal Dissection**  
  Yuancheng Shao, Yao Zhang, Jia Gu, Zixi Chen, Di Wu, Yuqiao Chen, Bo Lu, Wenjie Liu, Cesare Stefanini, Peng Qi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03406v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Manual endoscopic submucosal dissection (ESD) is technically demanding, and existing single-segment robotic tools offer limited dexterity. These limitations motivate the development of more advanced solutions. To address this, DESectBot, a novel dual segment continuum robot with a decoupled structure and integrated surgical forceps, enabling 6 degrees of freedom (DoFs) tip dexterity for improved lesion targeting in ESD, was developed in this work. Deep learning controllers based on gated recurrent units (GRUs) for simultaneous tip position and orientation control, effectively handling the nonlinear coupling between continuum segments, were proposed. The GRU controller was benchmarked against Jacobian based inverse kinematics, model predictive control (MPC), a feedforward neural network (FNN), and a long short-term memory (LSTM) network. In nested-rectangle and Lissajous trajectory tracking tasks, the GRU achieved the lowest position/orientation RMSEs: 1.11 mm/ 4.62° and 0.81 mm/ 2.59°, respectively. For orientation control at a fixed position (four target poses), the GRU attained a mean RMSE of 0.14 mm and 0.72°, outperforming all alternatives. In a peg transfer task, the GRU achieved a 100% success rate (120 success/120 attempts) with an average transfer time of 11.8s, the STD significantly outperforms novice-controlled systems. Additionally, an ex vivo ESD demonstration grasping, elevating, and resecting tissue as the scalpel completed the cut confirmed that DESectBot provides sufficient stiffness to divide thick gastric mucosa and an operative workspace adequate for large lesions.These results confirm that GRU-based control significantly enhances precision, reliability, and usability in ESD surgical training scenarios.

  </details>



- **A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures**  
  Basile Terver, Randall Balestriero, Megi Dervishi, David Fan, Quentin Garrido, Tushar Nagarajan, Koustuv Sinha, Wancong Zhang, Mike Rabbat, Yann LeCun, et al.  
  _2026-02-03_ · https://arxiv.org/abs/2602.03604v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We present EB-JEPA, an open-source library for learning representations and world models using Joint-Embedding Predictive Architectures (JEPAs). JEPAs learn to predict in representation space rather than pixel space, avoiding the pitfalls of generative modeling while capturing semantically meaningful features suitable for downstream tasks. Our library provides modular, self-contained implementations that illustrate how representation learning techniques developed for image-level self-supervised learning can transfer to video, where temporal dynamics add complexity, and ultimately to action-conditioned world models, where the model must additionally learn to predict the effects of control inputs. Each example is designed for single-GPU training within a few hours, making energy-based self-supervised learning accessible for research and education. We provide ablations of JEA components on CIFAR-10. Probing these representations yields 91% accuracy, indicating that the model learns useful features. Extending to video, we include a multi-step prediction example on Moving MNIST that demonstrates how the same principles scale to temporal modeling. Finally, we show how these representations can drive action-conditioned world models, achieving a 97% planning success rate on the Two Rooms navigation task. Comprehensive ablations reveal the critical importance of each regularization component for preventing representation collapse. Code is available at https://github.com/facebookresearch/eb_jepa.

  </details>



- **Enhancing Navigation Efficiency of Quadruped Robots via Leveraging Personal Transportation Platforms**  
  Minsung Yoon, Sung-Eui Yoon  
  _2026-02-03_ · https://arxiv.org/abs/2602.03397v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Quadruped robots face limitations in long-range navigation efficiency due to their reliance on legs. To ameliorate the limitations, we introduce a Reinforcement Learning-based Active Transporter Riding method (\textit{RL-ATR}), inspired by humans' utilization of personal transporters, including Segways. The \textit{RL-ATR} features a transporter riding policy and two state estimators. The policy devises adequate maneuvering strategies according to transporter-specific control dynamics, while the estimators resolve sensor ambiguities in non-inertial frames by inferring unobservable robot and transporter states. Comprehensive evaluations in simulation validate proficient command tracking abilities across various transporter-robot models and reduced energy consumption compared to legged locomotion. Moreover, we conduct ablation studies to quantify individual component contributions within the \textit{RL-ATR}. This riding ability could broaden the locomotion modalities of quadruped robots, potentially expanding the operational range and efficiency.

  </details>



- **Model-based Optimal Control for Rigid-Soft Underactuated Systems**  
  Daniele Caradonna, Nikhil Nair, Anup Teejo Mathew, Daniel Feliu Talegón, Imran Afgan, Egidio Falotico, Cosimo Della Santina, Federico Renda  
  _2026-02-03_ · https://arxiv.org/abs/2602.03435v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Continuum soft robots are inherently underactuated and subject to intrinsic input constraints, making dynamic control particularly challenging, especially in hybrid rigid-soft robots. While most existing methods focus on quasi-static behaviors, dynamic tasks such as swing-up require accurate exploitation of continuum dynamics. This has led to studies on simple low-order template systems that often fail to capture the complexity of real continuum deformations. Model-based optimal control offers a systematic solution; however, its application to rigid-soft robots is often limited by the computational cost and inaccuracy of numerical differentiation for high-dimensional models. Building on recent advances in the Geometric Variable Strain model that enable analytical derivatives, this work investigates three optimal control strategies for underactuated soft systems-Direct Collocation, Differential Dynamic Programming, and Nonlinear Model Predictive Control-to perform dynamic swing-up tasks. To address stiff continuum dynamics and constrained actuation, implicit integration schemes and warm-start strategies are employed to improve numerical robustness and computational efficiency. The methods are evaluated in simulation on three Rigid-Soft and high-order soft benchmark systems-the Soft Cart-Pole, the Soft Pendubot, and the Soft Furuta Pendulum- highlighting their performance and computational trade-offs.

  </details>



- **Zero-shot large vision-language model prompting for automated bone identification in paleoradiology x-ray archives**  
  Owen Dong, Lily Gao, Manish Kota, Bennett A. Landmana, Jelena Bekvalac, Gaynor Western, Katherine D. Van Schaik  
  _2026-02-03_ · https://arxiv.org/abs/2602.03750v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Paleoradiology, the use of modern imaging technologies to study archaeological and anthropological remains, offers new windows on millennial scale patterns of human health. Unfortunately, the radiographs collected during field campaigns are heterogeneous: bones are disarticulated, positioning is ad hoc, and laterality markers are often absent. Additionally, factors such as age at death, age of bone, sex, and imaging equipment introduce high variability. Thus, content navigation, such as identifying a subset of images with a specific projection view, can be time consuming and difficult, making efficient triaging a bottleneck for expert analysis. We report a zero shot prompting strategy that leverages a state of the art Large Vision Language Model (LVLM) to automatically identify the main bone, projection view, and laterality in such images. Our pipeline converts raw DICOM files to bone windowed PNGs, submits them to the LVLM with a carefully engineered prompt, and receives structured JSON outputs, which are extracted and formatted onto a spreadsheet in preparation for validation. On a random sample of 100 images reviewed by an expert board certified paleoradiologist, the system achieved 92% main bone accuracy, 80% projection view accuracy, and 100% laterality accuracy, with low or medium confidence flags for ambiguous cases. These results suggest that LVLMs can substantially accelerate code word development for large paleoradiology datasets, allowing for efficient content navigation in future anthropology workflows.

  </details>


