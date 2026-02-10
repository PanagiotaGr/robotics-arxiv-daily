# Navigation, Planning & Control

_Robotics arXiv Daily_

_Updated: 2026-02-10 07:19 UTC_

Total papers shown: **4**


---

- **GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion**  
  Santiago Montiel-Marín, Miguel Antunes-García, Fabio Sánchez-García, Angel Llamazares, Holger Caesar, Luis M. Bergasa  
  _2026-02-09_ · https://arxiv.org/abs/2602.08784v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Robust and accurate perception of dynamic objects and map elements is crucial for autonomous vehicles performing safe navigation in complex traffic scenarios. While vision-only methods have become the de facto standard due to their technical advances, they can benefit from effective and cost-efficient fusion with radar measurements. In this work, we advance fusion methods by repurposing Gaussian Splatting as an efficient universal view transformer that bridges the view disparity gap, mapping both image pixels and radar points into a common Bird's-Eye View (BEV) representation. Our main contribution is GaussianCaR, an end-to-end network for BEV segmentation that, unlike prior BEV fusion methods, leverages Gaussian Splatting to map raw sensor information into latent features for efficient camera-radar fusion. Our architecture combines multi-scale fusion with a transformer decoder to efficiently extract BEV features. Experimental results demonstrate that our approach achieves performance on par with, or even surpassing, the state of the art on BEV segmentation tasks (57.3%, 82.9%, and 50.1% IoU for vehicles, roads, and lane dividers) on the nuScenes dataset, while maintaining a 3.2x faster inference runtime. Code and project page are available online.

  </details>



- **From Obstacles to Etiquette: Robot Social Navigation with VLM-Informed Path Selection**  
  Zilin Fang, Anxing Xiao, David Hsu, Gim Hee Lee  
  _2026-02-09_ · https://arxiv.org/abs/2602.09002v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Navigating socially in human environments requires more than satisfying geometric constraints, as collision-free paths may still interfere with ongoing activities or conflict with social norms. Addressing this challenge calls for analyzing interactions between agents and incorporating common-sense reasoning into planning. This paper presents a social robot navigation framework that integrates geometric planning with contextual social reasoning. The system first extracts obstacles and human dynamics to generate geometrically feasible candidate paths, then leverages a fine-tuned vision-language model (VLM) to evaluate these paths, informed by contextually grounded social expectations, selecting a socially optimized path for the controller. This task-specific VLM distills social reasoning from large foundation models into a smaller and efficient model, allowing the framework to perform real-time adaptation in diverse human-robot interaction contexts. Experiments in four social navigation contexts demonstrate that our method achieves the best overall performance with the lowest personal space violation duration, the minimal pedestrian-facing time, and no social zone intrusions. Project page: https://path-etiquette.github.io

  </details>



- **High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning**  
  Jiarui Zhang, Chengyong Lei, Chengjiang Dai, Lijie Wang, Zhichao Han, Fei Gao  
  _2026-02-09_ · https://arxiv.org/abs/2602.08653v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Quadrotor unmanned aerial vehicles (UAVs) are increasingly deployed in complex missions that demand reliable autonomous navigation and robust obstacle avoidance. However, traditional modular pipelines often incur cumulative latency, whereas purely reinforcement learning (RL) approaches typically provide limited formal safety guarantees. To bridge this gap, we propose an end-to-end RL framework augmented with model-based safety mechanisms. We incorporate physical priors in both training and deployment. During training, we design a physics-informed reward structure that provides global navigational guidance. During deployment, we integrate a real-time safety filter that projects the policy outputs onto a provably safe set to enforce strict collision-avoidance constraints. This hybrid architecture reconciles high-speed flight with robust safety assurances. Benchmark evaluations demonstrate that our method outperforms both traditional planners and recent end-to-end obstacle avoidance approaches based on differentiable physics. Extensive experiments demonstrate strong generalization, enabling reliable high-speed navigation in dense clutter and challenging outdoor forest environments at velocities up to 7.5m/s.

  </details>



- **RFSoC-Based Integrated Navigation and Sensing Using NavIC**  
  Riya Sachdeva, Aakanksha Tewari, Sumit J. Darak, Shobha Sundar Ram, Sanat K. Biswas  
  _2026-02-09_ · https://arxiv.org/abs/2602.08596v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  Prior art has proposed a secondary application for Global Navigation Satellite System (GNSS) infrastructure for remote sensing of ground-based and maritime targets. Here, a passive radar receiver is deployed to detect uncooperative targets on Earth's surface by capturing ground-reflected satellite signals. This work demonstrates a hardware prototype of an L-band Navigation with Indian Constellation (NavIC) satellite-based remote sensing receiver system mounted on an AMD Zynq radio frequency system-on-chip (RFSoC) platform. Two synchronized receiver channels are introduced for capturing the direct signal (DS) from the satellite and ground-reflected signal (GRS) returns from targets. These signals are processed on the ARM processor and field programmable gate array (FPGA) of the RFSoC to generate delay-Doppler maps of the ground-based targets. The performance is first validated in a loop-back configuration of the RFSoC. Next, the DS and GRS signals are emulated by the output from two ports of the Keysight Arbitrary Waveform Generator (AWG) and interfaced with the RFSoC where the signals are subsequently processed to obtain the delay-Doppler maps. The performance is validated for different signal-to-noise ratios (SNR).

  </details>


