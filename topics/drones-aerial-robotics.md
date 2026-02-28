# Drones & Aerial Robotics

_Robotics arXiv Daily_

_Updated: 2026-02-28 06:54 UTC_

Total papers shown: **3**


---

- **FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time**  
  David Dirnfeld, Fabien Delattre, Pedro Miraldo, Erik Learned-Miller  
  _2026-02-26_ · https://arxiv.org/abs/2602.23115v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Estimating camera motion from monocular video is a fundamental problem in computer vision, central to tasks such as SLAM, visual odometry, and structure-from-motion. Existing methods that recover the camera's heading under known rotation, whether from an IMU or an optimization algorithm, tend to perform well in low-noise, low-outlier conditions, but often decrease in accuracy or become computationally expensive as noise and outlier levels increase. To address these limitations, we propose a novel generalization of the Hough transform on the unit sphere (S(2)) to estimate the camera's heading. First, the method extracts correspondences between two frames and generates a great circle of directions compatible with each pair of correspondences. Then, by discretizing the unit sphere using a Fibonacci lattice as bin centers, each great circle casts votes for a range of directions, ensuring that features unaffected by noise or dynamic objects vote consistently for the correct motion direction. Experimental results on three datasets demonstrate that the proposed method is on the Pareto frontier of accuracy versus efficiency. Additionally, experiments on SLAM show that the proposed method reduces RMSE by correcting the heading during camera pose initialization.

  </details>



- **ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering**  
  Elzo Brito dos Santos Filho  
  _2026-02-26_ · https://arxiv.org/abs/2602.23193v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Autonomous agents based on Large Language Models (LLMs) have evolved from reactive assistants to systems capable of planning, executing actions via tools, and iterating over environment observations. However, they remain vulnerable to structural limitations: lack of native state, context degradation over long horizons, and the gap between probabilistic generation and deterministic execution requirements. This paper presents the ESAA (Event Sourcing for Autonomous Agents) architecture, which separates the agent's cognitive intention from the project's state mutation, inspired by the Event Sourcing pattern. In ESAA, agents emit only structured intentions in validated JSON (agent.result or issue.report); a deterministic orchestrator validates, persists events in an append-only log (activity.jsonl), applies file-writing effects, and projects a verifiable materialized view (roadmap.json). The proposal incorporates boundary contracts (AGENT_CONTRACT.yaml), metaprompting profiles (PARCER), and replay verification with hashing (esaa verify), ensuring the immutability of completed tasks and forensic traceability. Two case studies validate the architecture: (i) a landing page project (9 tasks, 49 events, single-agent composition) and (ii) a clinical dashboard system (50 tasks, 86 events, 4 concurrent agents across 8 phases), both concluding with run.status=success and verify_status=ok. The multi-agent case study demonstrates real concurrent orchestration with heterogeneous LLMs (Claude Sonnet 4.6, Codex GPT-5, Antigravity/Gemini 3 Pro, and Claude Opus 4.6), providing empirical evidence of the architecture's scalability beyond single-agent scenarios.

  </details>



- **No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors**  
  Tao Liu, Gang Wan, Kan Ren, Shibo Wen  
  _2026-02-26_ · https://arxiv.org/abs/2602.23141v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We propose a new unsupervised framework for online video stabilization. Unlike methods based on deep learning that require paired stable and unstable datasets, our approach instantiates the classical stabilization pipeline with three stages and incorporates a multithreaded buffering mechanism. This design addresses three longstanding challenges in end-to-end learning: limited data, poor controllability, and inefficiency on hardware with constrained resources. Existing benchmarks focus mainly on handheld videos with a forward view in visible light, which restricts the applicability of stabilization to domains such as UAV nighttime remote sensing. To fill this gap, we introduce a new multimodal UAV aerial video dataset (UAV-Test). Experiments show that our method consistently outperforms state-of-the-art online stabilizers in both quantitative metrics and visual quality, while achieving performance comparable to offline methods.

  </details>


