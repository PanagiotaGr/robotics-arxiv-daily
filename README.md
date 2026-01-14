# Robotics ArXiv Daily (Autonomous Vehicles • Drones • 3D Gaussian Splatting • More)

A lightweight daily ArXiv digest for robotics-related papers, including:
- Autonomous driving (perception / prediction / planning / BEV / mapping)
- Drones / aerial robotics (navigation, control, SLAM, perception)
- SLAM / Localization / Mapping
- Navigation / Planning / Control
- Manipulation
- Robot Learning (RL/IL/foundation models)
- Multi-robot / swarms
- Safety / robustness / uncertainty
- 3D Gaussian Splatting / Neural Rendering for robotics

This repo updates automatically via GitHub Actions.

---

## How it works (simple)
- Every day, the workflow runs `scripts/fetch_arxiv_daily.py`
- It queries arXiv by category, filters by topic keywords, and produces:
  - `digests/YYYY-MM-DD.md` (daily archive)
  - `topics/<topic>.md` (one file per topic)
  - an updated “Today” block below

---

<!-- BEGIN TODAY -->
## ✅ Today

**Last update:** 2026-01-14  
**Daily archive:** `digests/2026-01-14.md`  

_Auto-generated. Edit `config.yml` to change topics/keywords._

### Browse by topic (links)

- **[Autonomous Driving & AV](topics/autonomous-driving-av.md)**
- **[Drones & Aerial Robotics](topics/drones-aerial-robotics.md)**
- **[SLAM, Localization & Mapping](topics/slam-localization-mapping.md)**
- **[Navigation, Planning & Control](topics/navigation-planning-control.md)**
- **[Manipulation & Grasping](topics/manipulation-grasping.md)**
- **[Robot Learning (RL, IL, Foundation Models)](topics/robot-learning-rl-il-foundation-models.md)**
- **[Multi-Robot & Swarms](topics/multi-robot-swarms.md)**
- **[Safety, Robustness, Uncertainty](topics/safety-robustness-uncertainty.md)**
- **[3D Gaussian Splatting & Neural Rendering (Robotics)](topics/3d-gaussian-splatting-neural-rendering-robotics.md)**

### Autonomous Driving & AV

- **REVNET: Rotation-Equivariant Point Cloud Completion via Vector Neuron Anchor Transformer**
  - Authors: Zhifan Ni, Eckehard Steinbach
  - Published: 2026-01-13 | Category: `cs.CV`
  - Links: [arXiv](https://arxiv.org/abs/2601.08558v1) | [PDF](https://arxiv.org/pdf/2601.08558v1)
  - Matched: kitti
- **Large Multimodal Models for Embodied Intelligent Driving: The Next Frontier in Self-Driving?**
  - Authors: Long Zhang, Yuchen Xia
  - Published: 2026-01-13 | Category: `cs.RO`
  - Links: [arXiv](https://arxiv.org/abs/2601.08434v1) | [PDF](https://arxiv.org/pdf/2601.08434v1)
  - Matched: autonomous driving, self-driving
- **Spiking Neural-Invariant Kalman Fusion for Accurate Localization Using Low-Cost IMUs**
  - Authors: Yaohua Liu, Qiao Xu, Yemin Wang, Hui Yi Leong, Binkai Ou
  - Published: 2026-01-13 | Category: `cs.RO`
  - Links: [arXiv](https://arxiv.org/abs/2601.08248v1) | [PDF](https://arxiv.org/pdf/2601.08248v1)
  - Matched: kitti
- _(See full topic page: [Autonomous Driving & AV](topics/autonomous-driving-av.md))_


### Drones & Aerial Robotics

- **Reasoning Matters for 3D Visual Grounding**
  - Authors: Hsiang-Wei Huang, Kuang-Ming Chen, Wenhao Chai, Cheng-Yen Yang, Jen-Hao Cheng, Jenq-Neng Hwang
  - Published: 2026-01-13 | Category: `cs.CV`
  - Links: [arXiv](https://arxiv.org/abs/2601.08811v1) | [PDF](https://arxiv.org/pdf/2601.08811v1)
  - Matched: vio
- **Uncovering Political Bias in Large Language Models using Parliamentary Voting Records**
  - Authors: Jieying Chen, Karen de Jong, Andreas Poole, Jan Burakowski, Elena Elderson Nosti, Joep Windt, Chendi Wang
  - Published: 2026-01-13 | Category: `cs.AI`
  - Links: [arXiv](https://arxiv.org/abs/2601.08785v1) | [PDF](https://arxiv.org/pdf/2601.08785v1)
  - Matched: vio
- **RMBRec: Robust Multi-Behavior Recommendation towards Target Behaviors**
  - Authors: Miaomiao Cai, Zhijie Zhang, Junfeng Fang, Zhiyong Cheng, Xiang Wang, Meng Wang
  - Published: 2026-01-13 | Category: `cs.IR`
  - Links: [arXiv](https://arxiv.org/abs/2601.08705v1) | [PDF](https://arxiv.org/pdf/2601.08705v1)
  - Matched: vio
- _(See full topic page: [Drones & Aerial Robotics](topics/drones-aerial-robotics.md))_


### SLAM, Localization & Mapping

- **Translating Light-Sheet Microscopy Images to Virtual H&E Using CycleGAN**
  - Authors: Yanhua Zhao
  - Published: 2026-01-13 | Category: `cs.CV`
  - Links: [arXiv](https://arxiv.org/abs/2601.08776v1) | [PDF](https://arxiv.org/pdf/2601.08776v1)
  - Matched: mapping
- **Salience-SGG: Enhancing Unbiased Scene Graph Generation with Iterative Salience Estimation**
  - Authors: Runfeng Qu, Ole Hall, Pia K Bideau, Julie Ouerfelli-Ethier, Martin Rolfs, Klaus Obermayer, Olaf Hellwich
  - Published: 2026-01-13 | Category: `cs.CV`
  - Links: [arXiv](https://arxiv.org/abs/2601.08728v1) | [PDF](https://arxiv.org/pdf/2601.08728v1)
  - Matched: localization
- **Real-Time Localization Framework for Autonomous Basketball Robots**
  - Authors: Naren Medarametla, Sreejon Mondal
  - Published: 2026-01-13 | Category: `cs.RO`
  - Links: [arXiv](https://arxiv.org/abs/2601.08713v1) | [PDF](https://arxiv.org/pdf/2601.08713v1)
  - Matched: localization
- _(See full topic page: [SLAM, Localization & Mapping](topics/slam-localization-mapping.md))_


### Navigation, Planning & Control

- **ToolACE-MCP: Generalizing History-Aware Routing from MCP Tools to the Agent Web**
  - Authors: Zhiyuan Yao, Zishan Xu, Yifu Guo, Zhiguang Han, Cheng Yang, Shuo Zhang, Weinan Zhang, Xingshan Zeng et al.
  - Published: 2026-01-13 | Category: `cs.AI`
  - Links: [arXiv](https://arxiv.org/abs/2601.08276v1) | [PDF](https://arxiv.org/pdf/2601.08276v1)
  - Matched: navigation
- **A brain-inspired information fusion method for enhancing robot GPS outages navigation**
  - Authors: Yaohua Liu, Hengjun Zhang, Binkai Ou
  - Published: 2026-01-13 | Category: `cs.RO`
  - Links: [arXiv](https://arxiv.org/abs/2601.08244v1) | [PDF](https://arxiv.org/pdf/2601.08244v1)
  - Matched: navigation
- _(See full topic page: [Navigation, Planning & Control](topics/navigation-planning-control.md))_


### Manipulation & Grasping

- **FSAG: Enhancing Human-to-Dexterous-Hand Finger-Specific Affordance Grounding via Diffusion Models**
  - Authors: Yifan Han, Pengfei Yi, Junyan Li, Hanqing Wang, Gaojing Zhang, Qi Peng Liu, Wenzhao Lian
  - Published: 2026-01-13 | Category: `cs.RO`
  - Links: [arXiv](https://arxiv.org/abs/2601.08246v1) | [PDF](https://arxiv.org/pdf/2601.08246v1)
  - Matched: manipulation, grasp, grasping, dexterous
- **A Pin-Array Structure for Gripping and Shape Recognition of Convex and Concave Terrain Profiles**
  - Authors: Takuya Kato, Kentaro Uno, Kazuya Yoshida
  - Published: 2026-01-13 | Category: `cs.RO`
  - Links: [arXiv](https://arxiv.org/abs/2601.08143v1) | [PDF](https://arxiv.org/pdf/2601.08143v1)
  - Matched: grasp, grasping
- _(See full topic page: [Manipulation & Grasping](topics/manipulation-grasping.md))_


### Robot Learning (RL, IL, Foundation Models)

- **Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System**
  - Authors: Hsiang-Wei Huang, Junbin Lu, Kuang-Ming Chen, Jenq-Neng Hwang
  - Published: 2026-01-13 | Category: `cs.CL`
  - Links: [arXiv](https://arxiv.org/abs/2601.08829v1) | [PDF](https://arxiv.org/pdf/2601.08829v1)
  - Matched: llm
- **MemRec: Collaborative Memory-Augmented Agentic Recommender System**
  - Authors: Weixin Chen, Yuhan Zhao, Jingyuan Huang, Zihe Ye, Clark Mingxuan Ju, Tong Zhao, Neil Shah, Li Chen et al.
  - Published: 2026-01-13 | Category: `cs.IR`
  - Links: [arXiv](https://arxiv.org/abs/2601.08816v1) | [PDF](https://arxiv.org/pdf/2601.08816v1)
  - Matched: llm
- **Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge**
  - Authors: Yao Tang, Li Dong, Yaru Hao, Qingxiu Dong, Furu Wei, Jiatao Gu
  - Published: 2026-01-13 | Category: `cs.CL`
  - Links: [arXiv](https://arxiv.org/abs/2601.08808v1) | [PDF](https://arxiv.org/pdf/2601.08808v1)
  - Matched: reinforcement learning
- _(See full topic page: [Robot Learning (RL, IL, Foundation Models)](topics/robot-learning-rl-il-foundation-models.md))_


### Multi-Robot & Swarms

- **Out-of-distribution generalization of deep-learning surrogates for 2D PDE-generated dynamics in the small-data regime**
  - Authors: Binh Duong Nguyen, Stefan Sandfeld
  - Published: 2026-01-13 | Category: `cs.LG`
  - Links: [arXiv](https://arxiv.org/abs/2601.08404v1) | [PDF](https://arxiv.org/pdf/2601.08404v1)
  - Matched: distributed
- **Deconstructing Pre-training: Knowledge Attribution Analysis in MoE and Dense Models**
  - Authors: Bo Wang, Junzhuo Li, Hong Chen, Yuanlin Chu, Yuxuan Fan, Xuming Hu
  - Published: 2026-01-13 | Category: `cs.AI`
  - Links: [arXiv](https://arxiv.org/abs/2601.08383v1) | [PDF](https://arxiv.org/pdf/2601.08383v1)
  - Matched: distributed
- **Source-Free Domain Adaptation for Geospatial Point Cloud Semantic Segmentation**
  - Authors: Yuan Gao, Di Cao, Xiaohuan Xi, Sheng Nie, Shaobo Xia, Cheng Wang
  - Published: 2026-01-13 | Category: `cs.CV`
  - Links: [arXiv](https://arxiv.org/abs/2601.08375v1) | [PDF](https://arxiv.org/pdf/2601.08375v1)
  - Matched: consensus
- _(See full topic page: [Multi-Robot & Swarms](topics/multi-robot-swarms.md))_


### Safety, Robustness, Uncertainty

- **APEX-SWE**
  - Authors: Abhi Kottamasu, Akul Datta, Aakash Barthwal, Chirag Mahapatra, Ajay Arun, Adarsh Hiremath, Brendan Foody, Bertie Vidgen
  - Published: 2026-01-13 | Category: `cs.SE`
  - Links: [arXiv](https://arxiv.org/abs/2601.08806v1) | [PDF](https://arxiv.org/pdf/2601.08806v1)
  - Matched: uncertainty
- **Enabling Population-Based Architectures for Neural Combinatorial Optimization**
  - Authors: Andoni Irazusta Garmendia, Josu Ceberio, Alexander Mendiburu
  - Published: 2026-01-13 | Category: `cs.NE`
  - Links: [arXiv](https://arxiv.org/abs/2601.08696v1) | [PDF](https://arxiv.org/pdf/2601.08696v1)
  - Matched: robustness
- **All Required, In Order: Phase-Level Evaluation for AI-Human Dialogue in Healthcare and Beyond**
  - Authors: Shubham Kulkarni, Alexander Lyzhov, Shiva Chaitanya, Preetam Joshi
  - Published: 2026-01-13 | Category: `cs.AI`
  - Links: [arXiv](https://arxiv.org/abs/2601.08690v1) | [PDF](https://arxiv.org/pdf/2601.08690v1)
  - Matched: safe
- _(See full topic page: [Safety, Robustness, Uncertainty](topics/safety-robustness-uncertainty.md))_


### 3D Gaussian Splatting & Neural Rendering (Robotics)

- **RAVEN: Erasing Invisible Watermarks via Novel View Synthesis**
  - Authors: Fahad Shamshad, Nils Lukas, Karthik Nandakumar
  - Published: 2026-01-13 | Category: `cs.CV`
  - Links: [arXiv](https://arxiv.org/abs/2601.08832v1) | [PDF](https://arxiv.org/pdf/2601.08832v1)
  - Matched: novel view synthesis
- **Geo-NVS-w: Geometry-Aware Novel View Synthesis In-the-Wild with an SDF Renderer**
  - Authors: Anastasios Tsalakopoulos, Angelos Kanlis, Evangelos Chatzis, Antonis Karakottas, Dimitrios Zarpalas
  - Published: 2026-01-13 | Category: `cs.CV`
  - Links: [arXiv](https://arxiv.org/abs/2601.08371v1) | [PDF](https://arxiv.org/pdf/2601.08371v1)
  - Matched: novel view synthesis
- _(See full topic page: [3D Gaussian Splatting & Neural Rendering (Robotics)](topics/3d-gaussian-splatting-neural-rendering-robotics.md))_
<!-- END TODAY -->

---

## Configuration
Edit `config.yml` to change:
- categories
- topic names + keywords
- how many days back to consider
- max papers per topic

---

## Local run (optional)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
python scripts/fetch_arxiv_daily.py
