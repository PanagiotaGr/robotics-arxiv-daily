# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-03 07:06 UTC_

Total papers shown: **6**


---

- **3D Foundation Model-Based Loop Closing for Decentralized Collaborative SLAM**  
  Pierre-Yves Lajoie, Benjamin Ramtoula, Daniele De Martini, Giovanni Beltrame  
  _2026-02-02_ · https://arxiv.org/abs/2602.02430v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Decentralized Collaborative Simultaneous Localization And Mapping (C-SLAM) techniques often struggle to identify map overlaps due to significant viewpoint variations among robots. Motivated by recent advancements in 3D foundation models, which can register images despite large viewpoint differences, we propose a robust loop closing approach that leverages these models to establish inter-robot measurements. In contrast to resource-intensive methods requiring full 3D reconstruction within a centralized map, our approach integrates foundation models into existing SLAM pipelines, yielding scalable and robust multi-robot mapping. Our contributions include: (1) integrating 3D foundation models to reliably estimate relative poses from monocular image pairs within decentralized C-SLAM; (2) introducing robust outlier mitigation techniques critical to the use of these relative poses; and (3) developing specialized pose graph optimization formulations that efficiently resolve scale ambiguities. We evaluate our method against state-of-the-art approaches, demonstrating improvements in localization and mapping accuracy, alongside significant gains in computational and memory efficiency. These results highlight the potential of our approach for deployment in large-scale multi-robot scenarios.

  </details>



- **Mapping-Guided Task Discovery and Allocation for Robotic Inspection of Underwater Structures**  
  Marina Ruediger, Ashis G. Banerjee  
  _2026-02-02_ · https://arxiv.org/abs/2602.02389v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Task generation for underwater multi-robot inspections without prior knowledge of existing geometry can be achieved and optimized through examination of simultaneous localization and mapping (SLAM) data. By considering hardware parameters and environmental conditions, a set of tasks is generated from SLAM meshes and optimized through expected keypoint scores and distance-based pruning. In-water tests are used to demonstrate the effectiveness of the algorithm and determine the appropriate parameters. These results are compared to simulated Voronoi partitions and boustrophedon patterns for inspection coverage on a model of the test environment. The key benefits of the presented task discovery method include adaptability to unexpected geometry and distributions that maintain coverage while focusing on areas more likely to present defects or damage.

  </details>



- **Bridging the Sim-to-Real Gap with multipanda ros2: A Real-Time ROS2 Framework for Multimanual Systems**  
  Jon Škerlj, Seongjin Bien, Abdeldjallil Naceri, Sami Haddadin  
  _2026-02-02_ · https://arxiv.org/abs/2602.02269v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present $multipanda\_ros2$, a novel open-source ROS2 architecture for multi-robot control of Franka Robotics robots. Leveraging ros2 control, this framework provides native ROS2 interfaces for controlling any number of robots from a single process. Our core contributions address key challenges in real-time torque control, including interaction control and robot-environment modeling. A central focus of this work is sustaining a 1kHz control frequency, a necessity for real-time control and a minimum frequency required by safety standards. Moreover, we introduce a controllet-feature design pattern that enables controller-switching delays of $\le 2$ ms, facilitating reproducible benchmarking and complex multi-robot interaction scenarios. To bridge the simulation-to-reality (sim2real) gap, we integrate a high-fidelity MuJoCo simulation with quantitative metrics for both kinematic accuracy and dynamic consistency (torques, forces, and control errors). Furthermore, we demonstrate that real-world inertial parameter identification can significantly improve force and torque accuracy, providing a methodology for iterative physics refinement. Our work extends approaches from soft robotics to rigid dual-arm, contact-rich tasks, showcasing a promising method to reduce the sim2real gap and providing a robust, reproducible platform for advanced robotics research.

  </details>



- **Extending the Law of Intersegmental Coordination: Implications for Powered Prosthetic Controls**  
  Elad Siman Tov, Nili E. Krausz  
  _2026-02-02_ · https://arxiv.org/abs/2602.02181v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Powered prostheses are capable of providing net positive work to amputees and have advanced in the past two decades. However, reducing amputee metabolic cost of walking remains an open problem. The Law of Intersegmental Coordination (ISC) has been observed across gaits and has been previously implicated in energy expenditure of walking, yet it has rarely been analyzed or applied within the context of lower-limb amputee gait. This law states that the elevation angles of the thigh, shank and foot over the gait cycle are not independent. In this work, we developed a method to analyze intersegmental coordination for lower-limb 3D kinematic data, to simplify ISC analysis. Moreover, inspired by motor control, biomechanics and robotics literature, we used our method to broaden ISC toward a new law of coordination of moments. We find these Elevation Space Moments (ESM), and present results showing a moment-based coordination for able bodied gait. We also analyzed ISC for amputee gait walking with powered and passive prosthesis, and found that while elevation angles remained planar, the ESM showed less coordination. We use ISC as a constraint to predict the shank angles/moments that would compensate for alterations due to a passive foot so as to mimic a healthy thigh angle/moment profile. This may have implications for improving powered prosthetic control. We developed the ISC3d toolbox that is freely available online, which may be used to compute kinematic and kinetic ISC in 3D. This provides a means to further study the role of coordination in gait and may help address fundamental questions of the neural control of human movement.

  </details>



- **Context Learning for Multi-Agent Discussion**  
  Xingyuan Hua, Sheng Yue, Xinyi Li, Yizhe Zhao, Jinrui Zhang, Ju Ren  
  _2026-02-02_ · https://arxiv.org/abs/2602.02350v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Multi-Agent Discussion (MAD) has garnered increasing attention very recently, where multiple LLM instances collaboratively solve problems via structured discussion. However, we find that current MAD methods easily suffer from discussion inconsistency, LLMs fail to reach a coherent solution, due to the misalignment between their individual contexts.In this paper, we introduce a multi-LLM context learning method (M2CL) that learns a context generator for each agent, capable of dynamically generating context instructions per discussion round via automatic information organization and refinement. Specifically, inspired by our theoretical insights on the context instruction, M2CL train the generators to control context coherence and output discrepancies via a carefully crafted self-adaptive mechanism.It enables LLMs to avoid premature convergence on majority noise and progressively reach the correct consensus. We evaluate M2CL on challenging tasks, including academic reasoning, embodied tasks, and mobile control. The results show that the performance of M2CL significantly surpasses existing methods by 20%--50%, while enjoying favorable transferability and computational efficiency.

  </details>



- **The Verification Crisis: Expert Perceptions of GenAI Disinformation and the Case for Reproducible Provenance**  
  Alexander Loth, Martin Kappes, Marc-Oliver Pahl  
  _2026-02-02_ · https://arxiv.org/abs/2602.02100v1 · `cs.CY`  
  <details><summary>Abstract</summary>

  The growth of Generative Artificial Intelligence (GenAI) has shifted disinformation production from manual fabrication to automated, large-scale manipulation. This article presents findings from the first wave of a longitudinal expert perception survey (N=21) involving AI researchers, policymakers, and disinformation specialists. It examines the perceived severity of multimodal threats -- text, image, audio, and video -- and evaluates current mitigation strategies. Results indicate that while deepfake video presents immediate "shock" value, large-scale text generation poses a systemic risk of "epistemic fragmentation" and "synthetic consensus," particularly in the political domain. The survey reveals skepticism about technical detection tools, with experts favoring provenance standards and regulatory frameworks despite implementation barriers. GenAI disinformation research requires reproducible methods. The current challenge is measurement: without standardized benchmarks and reproducibility checklists, tracking or countering synthetic media remains difficult. We propose treating information integrity as an infrastructure with rigor in data provenance and methodological reproducibility.

  </details>


