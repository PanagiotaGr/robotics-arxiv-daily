# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-03-04 07:02 UTC_

Total papers shown: **1**


---

- **Self-supervised Domain Adaptation for Visual 3D Pose Estimation of Nano-drone Racing Gates by Enforcing Geometric Consistency**  
  Nicholas Carlotti, Michele Antonazzi, Elia Cereda, Mirko Nava, Nicola Basilico, Daniele Palossi, Alessandro Giusti  
  _2026-03-03_ · https://arxiv.org/abs/2603.02936v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We consider the task of visually estimating the relative pose of a drone racing gate in front of a nano-quadrotor, using a convolutional neural network pre-trained on simulated data to regress the gate's pose. Due to the sim-to-real gap, the pre-trained model underperforms in the real world and must be adapted to the target domain. We propose an unsupervised domain adaptation (UDA) approach using only real image sequences collected by the drone flying an arbitrary trajectory in front of a gate; sequences are annotated in a self-supervised fashion with the drone's odometry as measured by its onboard sensors. On this dataset, a state consistency loss enforces that two images acquired at different times yield pose predictions that are consistent with the drone's odometry. Results indicate that our approach outperforms other SoA UDA approaches, has a low mean absolute error in position (x=26, y=28, z=10 cm) and orientation ($ψ$=13${^{\circ}}$), an improvement of 40% in position and 37% in orientation over a baseline. The approach's effectiveness is appreciable with as few as 10 minutes of real-world flight data and yields models with an inference time of 30.4ms (33 fps) when deployed aboard the Crazyflie 2.1 Brushless nano-drone.

  </details>


