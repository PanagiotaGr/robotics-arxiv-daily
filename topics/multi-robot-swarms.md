# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-04 07:06 UTC_

Total papers shown: **3**


---

- **When Should Agents Coordinate in Differentiable Sequential Decision Problems?**  
  Caleb Probine, Su Ann Low, David Fridovich-Keil, Ufuk Topcu  
  _2026-02-03_ · https://arxiv.org/abs/2602.03674v1 · `cs.MA`  
  <details><summary>Abstract</summary>

  Multi-robot teams must coordinate to operate effectively. When a team operates in an uncoordinated manner, and agents choose actions that are only individually optimal, the team's outcome can suffer. However, in many domains, coordination requires costly communication. We explore the value of coordination in a broad class of differentiable motion-planning problems. In particular, we model coordinated behavior as a spectrum: at one extreme, agents jointly optimize a common team objective, and at the other, agents make unilaterally optimal decisions given their individual decision variables, i.e., they operate at Nash equilibria. We then demonstrate that reasoning about coordination in differentiable motion-planning problems reduces to reasoning about the second-order properties of agents' objectives, and we provide algorithms that use this second-order reasoning to determine at which times a team of agents should coordinate.

  </details>



- **Human-in-the-Loop Failure Recovery with Adaptive Task Allocation**  
  Lorena Maria Genua, Nikita Boguslavskii, Zhi Li  
  _2026-02-03_ · https://arxiv.org/abs/2602.03603v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Since the recent Covid-19 pandemic, mobile manipulators and humanoid assistive robots with higher levels of autonomy have increasingly been adopted for patient care and living assistance. Despite advancements in autonomy, these robots often struggle to perform reliably in dynamic and unstructured environments and require human intervention to recover from failures. Effective human-robot collaboration is essential to enable robots to receive assistance from the most competent operator, in order to reduce their workload and minimize disruptions in task execution. In this paper, we propose an adaptive method for allocating robotic failures to human operators (ARFA). Our proposed approach models the capabilities of human operators, and continuously updates these beliefs based on their actual performance for failure recovery. For every failure to be resolved, a reward function calculates expected outcomes based on operator capabilities and historical data, task urgency, and current workload distribution. The failure is then assigned to the operator with the highest expected reward. Our simulations and user studies show that ARFA outperforms random allocation, significantly reducing robot idle time, improving overall system performance, and leading to a more distributed workload among operators.

  </details>



- **SAGE-5GC: Security-Aware Guidelines for Evaluating Anomaly Detection in the 5G Core Network**  
  Cristian Manca, Christian Scano, Giorgio Piras, Fabio Brau, Maura Pintor, Battista Biggio  
  _2026-02-03_ · https://arxiv.org/abs/2602.03596v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Machine learning-based anomaly detection systems are increasingly being adopted in 5G Core networks to monitor complex, high-volume traffic. However, most existing approaches are evaluated under strong assumptions that rarely hold in operational environments, notably the availability of independent and identically distributed (IID) data and the absence of adaptive attackers.In this work, we study the problem of detecting 5G attacks \textit{in the wild}, focusing on realistic deployment settings. We propose a set of Security-Aware Guidelines for Evaluating anomaly detectors in 5G Core Network (SAGE-5GC), driven by domain knowledge and consideration of potential adversarial threats. Using a realistic 5G Core dataset, we first train several anomaly detectors and assess their baseline performance against standard 5GC control-plane cyberattacks targeting PFCP-based network services.We then extend the evaluation to adversarial settings, where an attacker tries to manipulate the observable features of the network traffic to evade detection, under the constraint that the intended functionality of the malicious traffic is preserved. Starting from a selected set of controllable features, we analyze model sensitivity and adversarial robustness through randomized perturbations. Finally, we introduce a practical optimization strategy based on genetic algorithms that operates exclusively on attacker-controllable features and does not require prior knowledge of the underlying detection model. Our experimental results show that adversarially crafted attacks can substantially degrade detection performance, underscoring the need for robust, security-aware evaluation methodologies for anomaly detection in 5G networks deployed in the wild.

  </details>


