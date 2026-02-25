# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-25 07:14 UTC_

Total papers shown: **4**


---

- **Efficient Hierarchical Any-Angle Path Planning on Multi-Resolution 3D Grids**  
  Victor Reijgwart, Cesar Cadena, Roland Siegwart, Lionel Ott  
  _2026-02-24_ · https://arxiv.org/abs/2602.21174v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Hierarchical, multi-resolution volumetric mapping approaches are widely used to represent large and complex environments as they can efficiently capture their occupancy and connectivity information. Yet widely used path planning methods such as sampling and trajectory optimization do not exploit this explicit connectivity information, and search-based methods such as A* suffer from scalability issues in large-scale high-resolution maps. In many applications, Euclidean shortest paths form the underpinning of the navigation system. For such applications, any-angle planning methods, which find optimal paths by connecting corners of obstacles with straight-line segments, provide a simple and efficient solution. In this paper, we present a method that has the optimality and completeness properties of any-angle planners while overcoming computational tractability issues common to search-based methods by exploiting multi-resolution representations. Extensive experiments on real and synthetic environments demonstrate the proposed approach's solution quality and speed, outperforming even sampling-based methods. The framework is open-sourced to allow the robotics and planning community to build on our research.

  </details>



- **Robot Local Planner: A Periodic Sampling-Based Motion Planner with Minimal Waypoints for Home Environments**  
  Keisuke Takeshita, Takahiro Yamazaki, Tomohiro Ono, Takashi Yamamoto  
  _2026-02-24_ · https://arxiv.org/abs/2602.20645v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The objective of this study is to enable fast and safe manipulation tasks in home environments. Specifically, we aim to develop a system that can recognize its surroundings and identify target objects while in motion, enabling it to plan and execute actions accordingly. We propose a periodic sampling-based whole-body trajectory planning method, called the "Robot Local Planner (RLP)." This method leverages unique features of home environments to enhance computational efficiency, motion optimality, and robustness against recognition and control errors, all while ensuring safety. The RLP minimizes computation time by planning with minimal waypoints and generating safe trajectories. Furthermore, overall motion optimality is improved by periodically executing trajectory planning to select more optimal motions. This approach incorporates inverse kinematics that are robust to base position errors, further enhancing robustness. Evaluation experiments demonstrated that the RLP outperformed existing methods in terms of motion planning time, motion duration, and robustness, confirming its effectiveness in home environments. Moreover, application experiments using a tidy-up task achieved high success rates and short operation times, thereby underscoring its practical feasibility.

  </details>



- **LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding**  
  Jihao Qiu, Lingxi Xie, Xinyue Huo, Qi Tian, Qixiang Ye  
  _2026-02-24_ · https://arxiv.org/abs/2602.20913v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This paper addresses the critical and underexplored challenge of long video understanding with low computational budgets. We propose LongVideo-R1, an active, reasoning-equipped multimodal large language model (MLLM) agent designed for efficient video context navigation, avoiding the redundancy of exhaustive search. At the core of LongVideo-R1 lies a reasoning module that leverages high-level visual cues to infer the most informative video clip for subsequent processing. During inference, the agent initiates traversal from top-level visual summaries and iteratively refines its focus, immediately halting the exploration process upon acquiring sufficient knowledge to answer the query. To facilitate training, we first extract hierarchical video captions from CGBench, a video corpus with grounding annotations, and guide GPT-5 to generate 33K high-quality chain-of-thought-with-tool trajectories. The LongVideo-R1 agent is fine-tuned upon the Qwen-3-8B model through a two-stage paradigm: supervised fine-tuning (SFT) followed by reinforcement learning (RL), where RL employs a specifically designed reward function to maximize selective and efficient clip navigation. Experiments on multiple long video benchmarks validate the effectiveness of name, which enjoys superior tradeoff between QA accuracy and efficiency. All curated data and source code are provided in the supplementary material and will be made publicly available. Code and data are available at: https://github.com/qiujihao19/LongVideo-R1

  </details>



- **Onboard-Targeted Segmentation of Straylight in Space Camera Sensors**  
  Riccardo Gallon, Fabian Schiemenz, Alessandra Menicucci, Eberhard Gill  
  _2026-02-24_ · https://arxiv.org/abs/2602.20709v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  This study details an artificial intelligence (AI)-based methodology for the semantic segmentation of space camera faults. Specifically, we address the segmentation of straylight effects induced by solar presence around the camera's Field of View (FoV). Anomalous images are sourced from our published dataset. Our approach emphasizes generalization across diverse flare textures, leveraging pre-training on a public dataset (Flare7k++) including flares in various non-space contexts to mitigate the scarcity of realistic space-specific data. A DeepLabV3 model with MobileNetV3 backbone performs the segmentation task. The model design targets deployment in spacecraft resource-constrained hardware. Finally, based on a proposed interface between our model and the onboard navigation pipeline, we develop custom metrics to assess the model's performance in the system-level context.

  </details>


