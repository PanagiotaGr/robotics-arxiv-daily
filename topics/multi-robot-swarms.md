# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-01-21 06:53 UTC_

Total papers shown: **4**


---

- **Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep Reinforcement Learning**  
  Myong-Yol Choi, Hankyoul Ko, Hanse Cho, Changseung Kim, Seunghwan Kim, Jaemin Seo, Hyondong Oh  
  _2026-01-20_ · https://arxiv.org/abs/2601.13657v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  This paper presents a deep reinforcement learning (DRL) based controller for collective navigation of unmanned aerial vehicle (UAV) swarms in communication-denied environments, enabling robust operation in complex, obstacle-rich environments. Inspired by biological swarms where informed individuals guide groups without explicit communication, we employ an implicit leader-follower framework. In this paradigm, only the leader possesses goal information, while follower UAVs learn robust policies using only onboard LiDAR sensing, without requiring any inter-agent communication or leader identification. Our system utilizes LiDAR point clustering and an extended Kalman filter for stable neighbor tracking, providing reliable perception independent of external positioning systems. The core of our approach is a DRL controller, trained in GPU-accelerated Nvidia Isaac Sim, that enables followers to learn complex emergent behaviors - balancing flocking and obstacle avoidance - using only local perception. This allows the swarm to implicitly follow the leader while robustly addressing perceptual challenges such as occlusion and limited field-of-view. The robustness and sim-to-real transfer of our approach are confirmed through extensive simulations and challenging real-world experiments with a swarm of five UAVs, which successfully demonstrated collective navigation across diverse indoor and outdoor environments without any communication or external localization.

  </details>



- **The Orchestration of Multi-Agent Systems: Architectures, Protocols, and Enterprise Adoption**  
  Apoorva Adimulam, Rajesh Gupta, Sumit Kumar  
  _2026-01-20_ · https://arxiv.org/abs/2601.13671v1 · `cs.MA`  
  <details><summary>Abstract</summary>

  Orchestrated multi-agent systems represent the next stage in the evolution of artificial intelligence, where autonomous agents collaborate through structured coordination and communication to achieve complex, shared objectives. This paper consolidates and formalizes the technical composition of such systems, presenting a unified architectural framework that integrates planning, policy enforcement, state management, and quality operations into a coherent orchestration layer. Another primary contribution of this work is the in-depth technical delineation of two complementary communication protocols - the Model Context Protocol, which standardizes how agents access external tools and contextual data, and the Agent2Agent protocol, which governs peer coordination, negotiation, and delegation. Together, these protocols establish an interoperable communication substrate that enables scalable, auditable, and policy-compliant reasoning across distributed agent collectives. Beyond protocol design, the paper details how orchestration logic, governance frameworks, and observability mechanisms collectively sustain system coherence, transparency, and accountability. By synthesizing these elements into a cohesive technical blueprint, this paper provides comprehensive treatments of orchestrated multi-agent systems - bridging conceptual architectures with implementation-ready design principles for enterprise-scale AI ecosystems.

  </details>



- **Remapping and navigation of an embedding space via error minimization: a fundamental organizational principle of cognition in natural and artificial systems**  
  Benedikt Hartl, Léo Pio-Lopez, Chris Fields, Michael Levin  
  _2026-01-20_ · https://arxiv.org/abs/2601.14096v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  The emerging field of diverse intelligence seeks an integrated view of problem-solving in agents of very different provenance, composition, and substrates. From subcellular chemical networks to swarms of organisms, and across evolved, engineered, and chimeric systems, it is hypothesized that scale-invariant principles of decision-making can be discovered. We propose that cognition in both natural and synthetic systems can be characterized and understood by the interplay between two equally important invariants: (1) the remapping of embedding spaces, and (2) the navigation within these spaces. Biological collectives, from single cells to entire organisms (and beyond), remap transcriptional, morphological, physiological, or 3D spaces to maintain homeostasis and regenerate structure, while navigating these spaces through distributed error correction. Modern Artificial Intelligence (AI) systems, including transformers, diffusion models, and neural cellular automata enact analogous processes by remapping data into latent embeddings and refining them iteratively through contextualization. We argue that this dual principle - remapping and navigation of embedding spaces via iterative error minimization - constitutes a substrate-independent invariant of cognition. Recognizing this shared mechanism not only illuminates deep parallels between living systems and artificial models, but also provides a unifying framework for engineering adaptive intelligence across scales.

  </details>



- **TSN-IoT: A Two-Stage NOMA-Enabled Framework for Prioritized Traffic Handling in Dense IoT Networks**  
  Shama Siddiqui, Anwar Ahmed Khan, Nicola Marchetti  
  _2026-01-20_ · https://arxiv.org/abs/2601.13680v1 · `eess.SP`  
  <details><summary>Abstract</summary>

  With the growing applications of the Internet of Things (IoT), a major challenge is to ensure continuous connectivity while providing prioritized access. In dense IoT scenarios, synchronization may be disrupted either by the movement of nodes away from base stations or by the unavailability of reliable Global Navigation Satellite System (GNSS) signals, which can be affected by physical obstructions, multipath fading, or environmental interference, such as such as walls, buildings, moving objects, or electromagnetic noise from surrounding devices. In such contexts, distributed synchronization through Non-Orthogonal Multiple Access (NOMA) offers a promising solution, as it enables simultaneous transmission to multiple users with different power levels, supporting efficient synchronization while minimizing the signaling overhead. Moreover, NOMA also plays a vital role for dynamic priority management in dense and heterogeneous IoT environments. In this article, we proposed a Two-Stage NOMA-Enabled Framework "TSN-IoT" that integrates the mechanisms of conventional Precision Time Protocol (PTP) based synchronization, distributed synchronization and data transmission. The framework is designed as a four-tier architecture that facilitates prioritized data delivery from sensor nodes to the central base station. We demonstrated the performance of "TSN-IoT" through a healthcare use case, where intermittent connectivity and varying data priority levels present key challenges for reliable communication. Synchronization speed and end-to-end delay were evaluated through a series of simulations implemented in Python. Results show that, compared to priority-based Orthogonal Frequency Division Multiple Access (OFDMA), TSN-IoT achieves significantly better performance by offering improved synchronization opportunities and enabling parallel transmissions over the same sub-carrier.

  </details>


