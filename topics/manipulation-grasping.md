# Manipulation & Grasping

_Robotics arXiv Daily_

_Updated: 2026-02-19 07:13 UTC_

Total papers shown: **3**


---

- **Learning Humanoid End-Effector Control for Open-Vocabulary Visual Loco-Manipulation**  
  Runpei Dong, Ziyan Li, Xialin He, Saurabh Gupta  
  _2026-02-18_ · https://arxiv.org/abs/2602.16705v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Visual loco-manipulation of arbitrary objects in the wild with humanoid robots requires accurate end-effector (EE) control and a generalizable understanding of the scene via visual inputs (e.g., RGB-D images). Existing approaches are based on real-world imitation learning and exhibit limited generalization due to the difficulty in collecting large-scale training datasets. This paper presents a new paradigm, HERO, for object loco-manipulation with humanoid robots that combines the strong generalization and open-vocabulary understanding of large vision models with strong control performance from simulated training. We achieve this by designing an accurate residual-aware EE tracking policy. This EE tracking policy combines classical robotics with machine learning. It uses a) inverse kinematics to convert residual end-effector targets into reference trajectories, b) a learned neural forward model for accurate forward kinematics, c) goal adjustment, and d) replanning. Together, these innovations help us cut down the end-effector tracking error by 3.2x. We use this accurate end-effector tracker to build a modular system for loco-manipulation, where we use open-vocabulary large vision models for strong visual generalization. Our system is able to operate in diverse real-world environments, from offices to coffee shops, where the robot is able to reliably manipulate various everyday objects (e.g., mugs, apples, toys) on surfaces ranging from 43cm to 92cm in height. Systematic modular and end-to-end tests in simulation and the real world demonstrate the effectiveness of our proposed design. We believe the advances in this paper can open up new ways of training humanoid robots to interact with daily objects.

  </details>



- **EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data**  
  Ruijie Zheng, Dantong Niu, Yuqi Xie, Jing Wang, Mengda Xu, Yunfan Jiang, Fernando Castañeda, Fengyuan Hu, You Liang Tan, Letian Fu, et al.  
  _2026-02-18_ · https://arxiv.org/abs/2602.16710v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Human behavior is among the most scalable sources of data for learning physical intelligence, yet how to effectively leverage it for dexterous manipulation remains unclear. While prior work demonstrates human to robot transfer in constrained settings, it is unclear whether large scale human data can support fine grained, high degree of freedom dexterous manipulation. We present EgoScale, a human to dexterous manipulation transfer framework built on large scale egocentric human data. We train a Vision Language Action (VLA) model on over 20,854 hours of action labeled egocentric human video, more than 20 times larger than prior efforts, and uncover a log linear scaling law between human data scale and validation loss. This validation loss strongly correlates with downstream real robot performance, establishing large scale human data as a predictable supervision source. Beyond scale, we introduce a simple two stage transfer recipe: large scale human pretraining followed by lightweight aligned human robot mid training. This enables strong long horizon dexterous manipulation and one shot task adaptation with minimal robot supervision. Our final policy improves average success rate by 54% over a no pretraining baseline using a 22 DoF dexterous robotic hand, and transfers effectively to robots with lower DoF hands, indicating that large scale human motion provides a reusable, embodiment agnostic motor prior.

  </details>



- **Learning to unfold cloth: Scaling up world models to deformable object manipulation**  
  Jack Rome, Stephen James, Subramanian Ramamoorthy  
  _2026-02-18_ · https://arxiv.org/abs/2602.16675v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Learning to manipulate cloth is both a paradigmatic problem for robotic research and a problem of immediate relevance to a variety of applications ranging from assistive care to the service industry. The complex physics of the deformable object makes this problem of cloth manipulation nontrivial. In order to create a general manipulation strategy that addresses a variety of shapes, sizes, fold and wrinkle patterns, in addition to the usual problems of appearance variations, it becomes important to carefully consider model structure and their implications for generalisation performance. In this paper, we present an approach to in-air cloth manipulation that uses a variation of a recently proposed reinforcement learning architecture, DreamerV2. Our implementation modifies this architecture to utilise surface normals input, in addition to modiying the replay buffer and data augmentation procedures. Taken together these modifications represent an enhancement to the world model used by the robot, addressing the physical complexity of the object being manipulated by the robot. We present evaluations both in simulation and in a zero-shot deployment of the trained policies in a physical robot setup, performing in-air unfolding of a variety of different cloth types, demonstrating the generalisation benefits of our proposed architecture.

  </details>


