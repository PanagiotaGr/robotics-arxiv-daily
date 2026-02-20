# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-20 07:10 UTC_

Total papers shown: **2**


---

- **Distributed Virtual Model Control for Scalable Human-Robot Collaboration in Shared Workspace**  
  Yi Zhang, Omar Faris, Chapa Sirithunge, Kai-Fung Chu, Fumiya Iida, Fulvio Forni  
  _2026-02-19_ · https://arxiv.org/abs/2602.17415v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a decentralized, agent agnostic, and safety-aware control framework for human-robot collaboration based on Virtual Model Control (VMC). In our approach, both humans and robots are embedded in the same virtual-component-shaped workspace, where motion is the result of the interaction with virtual springs and dampers rather than explicit trajectory planning. A decentralized, force-based stall detector identifies deadlocks, which are resolved through negotiation. This reduces the probability of robots getting stuck in the block placement task from up to 61.2% to zero in our experiments. The framework scales without structural changes thanks to the distributed implementation: in experiments we demonstrate safe collaboration with up to two robots and two humans, and in simulation up to four robots, maintaining inter-agent separation at around 20 cm. Results show that the method shapes robot behavior intuitively by adjusting control parameters and achieves deadlock-free operation across team sizes in all tested scenarios.

  </details>



- **Geometric Inverse Flight Dynamics on SO(3) and Application to Tethered Fixed-Wing Aircraft**  
  Antonio Franchi, Chiara Gabellieri  
  _2026-02-19_ · https://arxiv.org/abs/2602.17166v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  We present a robotics-oriented, coordinate-free formulation of inverse flight dynamics for fixed-wing aircraft on SO(3). Translational force balance is written in the world frame and rotational dynamics in the body frame; aerodynamic directions (drag, lift, side) are defined geometrically, avoiding local attitude coordinates. Enforcing coordinated flight (no sideslip), we derive a closed-form trajectory-to-input map yielding the attitude, angular velocity, and thrust-angle-of-attack pair, and we recover the aerodynamic moment coefficients component-wise. Applying such a map to tethered flight on spherical parallels, we obtain analytic expressions for the required bank angle and identify a specific zero-bank locus where the tether tension exactly balances centrifugal effects, highlighting the decoupling between aerodynamic coordination and the apparent gravity vector. Under a simple lift/drag law, the minimal-thrust angle of attack admits a closed form. These pointwise quasi-steady inversion solutions become steady-flight trim when the trajectory and rotational dynamics are time-invariant. The framework bridges inverse simulation in aeronautics with geometric modeling in robotics, providing a rigorous building block for trajectory design and feasibility checks.

  </details>


