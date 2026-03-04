# Robot Learning (RL, IL, Foundation Models)

_Robotics arXiv Daily_

_Updated: 2026-03-04 07:02 UTC_

Total papers shown: **18**


---

- **From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?**  
  Shinas Shaji, Fabian Huppertz, Alex Mitrevski, Sebastian Houben  
  _2026-03-03_ · https://arxiv.org/abs/2603.03148v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In order to flexibly act in an everyday environment, a robotic agent needs a variety of cognitive capabilities that enable it to reason about plans and perform execution recovery. Large language models (LLMs) have been shown to demonstrate emergent cognitive aspects, such as reasoning and language understanding; however, the ability to control embodied robotic agents requires reliably bridging high-level language to low-level functionalities for perception and control. In this paper, we investigate the extent to which an LLM can serve as a core component for planning and execution reasoning in a cognitive robot architecture. For this purpose, we propose a cognitive architecture in which an agentic LLM serves as the core component for planning and reasoning, while components for working and episodic memories support learning from experience and adaptation. An instance of the architecture is then used to control a mobile manipulator in a simulated household environment, where environment interaction is done through a set of high-level tools for perception, reasoning, navigation, grasping, and placement, all of which are made available to the LLM-based agent. We evaluate our proposed system on two household tasks (object placement and object swapping), which evaluate the agent's reasoning, planning, and memory utilisation. The results demonstrate that the LLM-driven agent can complete structured tasks and exhibits emergent adaptation and memory-guided planning, but also reveal significant limitations, such as hallucinations about the task success and poor instruction following by refusing to acknowledge and complete sequential tasks. These findings highlight both the potential and challenges of employing LLMs as embodied cognitive controllers for autonomous robots.

  </details>



- **Utonia: Toward One Encoder for All Point Clouds**  
  Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, Hengshuang Zhao  
  _2026-03-03_ · https://arxiv.org/abs/2603.03283v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  We dream of a future where point clouds from all domains can come together to shape a single model that benefits them all. Toward this goal, we present Utonia, a first step toward training a single self-supervised point transformer encoder across diverse domains, spanning remote sensing, outdoor LiDAR, indoor RGB-D sequences, object-centric CAD models, and point clouds lifted from RGB-only videos. Despite their distinct sensing geometries, densities, and priors, Utonia learns a consistent representation space that transfers across domains. This unification improves perception capability while revealing intriguing emergent behaviors that arise only when domains are trained jointly. Beyond perception, we observe that Utonia representations can also benefit embodied and multimodal reasoning: conditioning vision-language-action policies on Utonia features improves robotic manipulation, and integrating them into vision-language models yields gains on spatial reasoning. We hope Utonia can serve as a step toward foundation models for sparse 3D data, and support downstream applications in AR/VR, robotics, and autonomous driving.

  </details>



- **ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation**  
  Xialin He, Sirui Xu, Xinyao Li, Runpei Dong, Liuyu Bian, Yu-Xiong Wang, Liang-Yan Gui  
  _2026-03-03_ · https://arxiv.org/abs/2603.03279v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Achieving autonomous and versatile whole-body loco-manipulation remains a central barrier to making humanoids practically useful. Yet existing approaches are fundamentally constrained: retargeted data are often scarce or low-quality; methods struggle to scale to large skill repertoires; and, most importantly, they rely on tracking predefined motion references rather than generating behavior from perception and high-level task specifications. To address these limitations, we propose ULTRA, a unified framework with two key components. First, we introduce a physics-driven neural retargeting algorithm that translates large-scale motion capture to humanoid embodiments while preserving physical plausibility for contact-rich interactions. Second, we learn a unified multimodal controller that supports both dense references and sparse task specifications, under sensing ranging from accurate motion-capture state to noisy egocentric visual inputs. We distill a universal tracking policy into this controller, compress motor skills into a compact latent space, and apply reinforcement learning finetuning to expand coverage and improve robustness under out-of-distribution scenarios. This enables coordinated whole-body behavior from sparse intent without test-time reference motions. We evaluate ULTRA in simulation and on a real Unitree G1 humanoid. Results show that ULTRA generalizes to autonomous, goal-conditioned whole-body loco-manipulation from egocentric perception, consistently outperforming tracking-only baselines with limited skills.

  </details>



- **RL-Based Coverage Path Planning for Deformable Objects on 3D Surfaces**  
  Yuhang Zhang, Jinming Ma, Feng Wu  
  _2026-03-03_ · https://arxiv.org/abs/2603.03137v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Currently, manipulation tasks for deformable objects often focus on activities like folding clothes, handling ropes, and manipulating bags. However, research on contact-rich tasks involving deformable objects remains relatively underdeveloped. When humans use cloth or sponges to wipe surfaces, they rely on both vision and tactile feedback. Yet, current algorithms still face challenges with issues like occlusion, while research on tactile perception for manipulation is still evolving. Tasks such as covering surfaces with deformable objects demand not only perception but also precise robotic manipulation. To address this, we propose a method that leverages efficient and accessible simulators for task execution. Specifically, we train a reinforcement learning agent in a simulator to manipulate deformable objects for surface wiping tasks. We simplify the state representation of object surfaces using harmonic UV mapping, process contact feedback from the simulator on 2D feature maps, and use scaled grouped convolutions (SGCNN) to extract features efficiently. The agent then outputs actions in a reduced-dimensional action space to generate coverage paths. Experiments demonstrate that our method outperforms previous approaches in key metrics, including total path length and coverage area. We deploy these paths on a Kinova Gen3 manipulator to perform wiping experiments on the back of a torso model, validating the feasibility of our approach.

  </details>



- **DreamFlow: Local Navigation Beyond Observation via Conditional Flow Matching in the Latent Space**  
  Jiwon Park, Dongkyu Lee, I Made Aswin Nahrendra, Jaeyoung Lim, Hyun Myung  
  _2026-03-03_ · https://arxiv.org/abs/2603.02976v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Local navigation in cluttered environments often suffers from dense obstacles and frequent local minima. Conventional local planners rely on heuristics and are prone to failure, while deep reinforcement learning(DRL)based approaches provide adaptability but are constrained by limited onboard sensing. These limitations lead to navigation failures because the robot cannot perceive structures outside its field of view. In this paper, we propose DreamFlow, a DRL-based local navigation framework that extends the robot's perceptual horizon through conditional flow matching(CFM). The proposed CFM based prediction module learns probabilistic mapping between local height map latent representation and broader spatial representation conditioned on navigation context. This enables the navigation policy to predict unobserved environmental features and proactively avoid potential local minima. Experimental results demonstrate that DreamFlow outperforms existing methods in terms of latent prediction accuracy and navigation performance in simulation. The proposed method was further validated in cluttered real world environments with a quadrupedal robot. The project page is available at https://dreamflow-icra.github.io.

  </details>



- **CMoE: Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots**  
  Shihao Ma, Hongjin Chen, Zijun Xu, Yi Zhao, Ke Wu, Ruichen Yang, Leyao Zou, Zhongxue Gan, Wenchao Ding  
  _2026-03-03_ · https://arxiv.org/abs/2603.03067v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  For effective deployment in real-world environments, humanoid robots must autonomously navigate a diverse range of complex terrains with abrupt transitions. While the Vanilla mixture of experts (MoE) framework is theoretically capable of modeling diverse terrain features, in practice, the gating network exhibits nearly uniform expert activations across different terrains, weakening the expert specialization and limiting the model's expressive power. To address this limitation, we introduce CMoE, a novel single-stage reinforcement learning framework that integrates contrastive learning to refine expert activation distributions. By imposing contrastive constraints, CMoE maximizes the consistency of expert activations within the same terrain while minimizing their similarity across different terrains, thereby encouraging experts to specialize in distinct terrain types. We validated our approach on the Unitree G1 humanoid robot through a series of challenging experiments. Results demonstrate that CMoE enables the robot to traverse continuous steps up to 20 cm high and gaps up to 80 cm wide, while achieving robust and natural gait across diverse mixed terrains, surpassing the limits of existing methods. To support further research and foster community development, we release our code publicly.

  </details>



- **Generative adversarial imitation learning for robot swarms: Learning from human demonstrations and trained policies**  
  Mattes Kraus, Jonas Kuckling  
  _2026-03-03_ · https://arxiv.org/abs/2603.02783v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  In imitation learning, robots are supposed to learn from demonstrations of the desired behavior. Most of the work in imitation learning for swarm robotics provides the demonstrations as rollouts of an existing policy. In this work, we provide a framework based on generative adversarial imitation learning that aims to learn collective behaviors from human demonstrations. Our framework is evaluated across six different missions, learning both from manual demonstrations and demonstrations derived from a PPO-trained policy. Results show that the imitation learning process is able to learn qualitatively meaningful behaviors that perform similarly well as the provided demonstrations. Additionally, we deploy the learned policies on a swarm of TurtleBot 4 robots in real-robot experiments. The exhibited behaviors preserved their visually recognizable character and their performance is comparable to the one achieved in simulation.

  </details>



- **Agentic Self-Evolutionary Replanning for Embodied Navigation**  
  Guoliang Li, Ruihua Han, Chengyang Li, He Li, Shuai Wang, Wenchao Ding, Hong Zhang, Chengzhong Xu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02772v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Failure is inevitable for embodied navigation in complex environments. To enhance the resilience, replanning (RP) is a viable option, where the robot is allowed to fail, but is capable of adjusting plan until success. However, existing RP approaches freeze the ego action model and miss the opportunities to explore better plans by upgrading the robot itself. To address this limitation, we propose Self-Evolutionary RePlanning, or SERP for short, which leads to a paradigm shift from frozen models towards evolving models by run-time learning from recent experiences. In contrast to existing model evolution approaches that often get stuck at predefined static parameters, we introduce agentic self-evolving action model that uses in-context learning with auto-differentiation (ILAD) for adaptive function adjustment and global parameter reset. To achieve token-efficient replanning for SERP, we also propose graph chain-of-thought (GCOT) replanning with large language model (LLM) inference over distilled graphs. Extensive simulation and real-world experiments demonstrate that SERP achieves higher success rate with lower token expenditure over various benchmarks, validating its superior robustness and efficiency across diverse environments.

  </details>



- **How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference**  
  Toru Lin, Shuying Deng, Zhao-Heng Yin, Pieter Abbeel, Jitendra Malik  
  _2026-03-03_ · https://arxiv.org/abs/2603.03280v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Many essential manipulation tasks - such as food preparation, surgery, and craftsmanship - remain intractable for autonomous robots. These tasks are characterized not only by contact-rich, force-sensitive dynamics, but also by their "implicit" success criteria: unlike pick-and-place, task quality in these domains is continuous and subjective (e.g. how well a potato is peeled), making quantitative evaluation and reward engineering difficult. We present a learning framework for such tasks, using peeling with a knife as a representative example. Our approach follows a two-stage pipeline: first, we learn a robust initial policy via force-aware data collection and imitation learning, enabling generalization across object variations; second, we refine the policy through preference-based finetuning using a learned reward model that combines quantitative task metrics with qualitative human feedback, aligning policy behavior with human notions of task quality. Using only 50-200 peeling trajectories, our system achieves over 90% average success rates on challenging produce including cucumbers, apples, and potatoes, with performance improving by up to 40% through preference-based finetuning. Remarkably, policies trained on a single produce category exhibit strong zero-shot generalization to unseen in-category instances and to out-of-distribution produce from different categories while maintaining over 90% success rates.

  </details>



- **MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization**  
  Ashutosh Chaubey, Jiacheng Pang, Mohammad Soleymani  
  _2026-03-03_ · https://arxiv.org/abs/2603.03192v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Omni-modal large language models (omni LLMs) have recently achieved strong performance across audiovisual understanding tasks, yet they remain highly susceptible to cross-modal hallucinations arising from spurious correlations and dominant language priors. In this work, we propose Modality-Decoupled Direct Preference Optimization (MoD-DPO), a simple and effective framework for improving modality grounding in omni LLMs. MoD-DPO introduces modality-aware regularization terms that explicitly enforce invariance to corruptions in irrelevant modalities and sensitivity to perturbations in relevant modalities, thereby reducing unintended cross-modal interactions. To further mitigate over-reliance on textual priors, we incorporate a language-prior debiasing penalty that discourages hallucination-prone text-only responses. Extensive experiments across multiple audiovisual hallucination benchmarks demonstrate that MoD-DPO consistently improves perception accuracy and hallucination resistance, outperforming previous preference optimization baselines under similar training budgets. Our findings underscore the importance of modality-faithful alignment and demonstrate a scalable path toward more reliable and resilient multimodal foundation models.

  </details>



- **REGAL: A Registry-Driven Architecture for Deterministic Grounding of Agentic AI in Enterprise Telemetry**  
  Yuvraj Agrawal  
  _2026-03-03_ · https://arxiv.org/abs/2603.03018v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Enterprise engineering organizations produce high-volume, heterogeneous telemetry from version control systems, CI/CD pipelines, issue trackers, and observability platforms. Large Language Models (LLMs) enable new forms of agentic automation, but grounding such agents on private telemetry raises three practical challenges: limited model context, locally defined semantic concepts, and evolving metric interfaces. We present REGAL, a registry-driven architecture for deterministic grounding of agentic AI systems in enterprise telemetry. REGAL adopts an explicitly architectural approach: deterministic telemetry computation is treated as a first-class primitive, and LLMs operate over a bounded, version-controlled action space rather than raw event streams. The architecture combines (1) a Medallion ELT pipeline that produces replayable, semantically compressed Gold artifacts, and (2) a registry-driven compilation layer that synthesizes Model Context Protocol (MCP) tools from declarative metric definitions. The registry functions as an "interface-as-code" layer, ensuring alignment between tool specification and execution, mitigating tool drift, and embedding governance policies directly at the semantic boundary. A prototype implementation and case study validate the feasibility of deterministic grounding and illustrate its implications for latency, token efficiency, and operational governance. This work systematizes an architectural pattern for enterprise LLM grounding; it does not propose new learning algorithms, but rather elevates deterministic computation and semantic compilation to first-class design primitives for agentic systems.

  </details>



- **Sparse autoencoders reveal organized biological knowledge but minimal regulatory logic in single-cell foundation models: a comparative atlas of Geneformer and scGPT**  
  Ihor Kendiukhov  
  _2026-03-03_ · https://arxiv.org/abs/2603.02952v1 · `q-bio.GN`  
  <details><summary>Abstract</summary>

  Background: Single-cell foundation models such as Geneformer and scGPT encode rich biological information, but whether this includes causal regulatory logic rather than statistical co-expression remains unclear. Sparse autoencoders (SAEs) can resolve superposition in neural networks by decomposing dense activations into interpretable features, yet they have not been systematically applied to biological foundation models. Results: We trained TopK SAEs on residual stream activations from all layers of Geneformer V2-316M (18 layers, d=1152) and scGPT whole-human (12 layers, d=512), producing atlases of 82525 and 24527 features, respectively. Both atlases confirm massive superposition, with 99.8 percent of features invisible to SVD. Systematic characterization reveals rich biological organization: 29 to 59 percent of features annotate to Gene Ontology, KEGG, Reactome, STRING, or TRRUST, with U-shaped layer profiles reflecting hierarchical abstraction. Features organize into co-activation modules (141 in Geneformer, 76 in scGPT), exhibit causal specificity (median 2.36x), and form cross-layer information highways (63 to 99.8 percent). When tested against genome-scale CRISPRi perturbation data, only 3 of 48 transcription factors (6.2 percent) show regulatory-target-specific feature responses. A multi-tissue control yields marginal improvement (10.4 percent, 5 of 48 TFs), establishing model representations as the bottleneck. Conclusions: These models have internalized organized biological knowledge, including pathway membership, protein interactions, functional modules, and hierarchical abstraction, yet they encode minimal causal regulatory logic. We release both feature atlases as interactive web platforms enabling exploration of more than 107000 features across 30 layers of two leading single-cell foundation models.

  </details>



- **CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning**  
  Zhenquan Yao, Zitong Huang, Yihan Zeng, Jianhua Han, Hang Xu, Chun-Mei Feng, Jianwei Ma, Wangmeng Zuo  
  _2026-03-03_ · https://arxiv.org/abs/2603.02951v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Graphical User Interface (GUI) Agents, benefiting from recent advances in multimodal large language models (MLLM), have achieved significant development. However, due to the frequent updates of GUI applications, adapting to new tasks without forgetting old tasks in GUI continual learning remains an open problem. In this work, we reveal that while Supervised Fine-Tuning (SFT) facilitates fast adaptation, it often triggers knowledge overwriting, whereas Reinforcement Learning (RL) demonstrates an inherent resilience that shields prior interaction logic from erasure. Based on this insight, we propose a \textbf{C}ontinual \textbf{G}UI \textbf{L}earning (CGL) framework that dynamically balances adaptation efficiency and skill retention by enhancing the synergy between SFT and RL. Specifically, we introduce an SFT proportion adjustment mechanism guided by policy entropy to dynamically control the weight allocation between the SFT and RL training phases. To resolve explicit gradient interference, we further develop a specialized gradient surgery strategy. By projecting exploratory SFT gradients onto GRPO-based anchor gradients, our method explicitly clips the components of SFT gradients that conflict with GRPO. On top of that, we establish an AndroidControl-CL benchmark, which divides GUI applications into distinct task groups to effectively simulate and evaluate the performance of continual GUI learning. Experimental results demonstrate the effectiveness of our proposed CGL framework across continual learning scenarios. The benchmark, code, and model will be made publicly available.

  </details>



- **Contextual Latent World Models for Offline Meta Reinforcement Learning**  
  Mohammadreza Nakheai, Aidan Scannell, Kevin Luck, Joni Pajarinen  
  _2026-03-03_ · https://arxiv.org/abs/2603.02935v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Offline meta-reinforcement learning seeks to learn policies that generalize across related tasks from fixed datasets. Context-based methods infer a task representation from transition histories, but learning effective task representations without supervision remains a challenge. In parallel, latent world models have demonstrated strong self-supervised representation learning through temporal consistency. We introduce contextual latent world models, which condition latent world models on inferred task representations and train them jointly with the context encoder. This enforces task-conditioned temporal consistency, yielding task representations that capture task-dependent dynamics rather than merely discriminating between tasks. Our method learns more expressive task representations and significantly improves generalization to unseen tasks across MuJoCo, Contextual-DeepMind Control, and Meta-World benchmarks.

  </details>



- **LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval**  
  Minh-Chi Phung, Thien-Bao Le, Cam-Tu Tran-Thi, Thu-Dieu Nguyen-Thi, Vu-Hung Dao  
  _2026-03-03_ · https://arxiv.org/abs/2603.02888v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The increasing diversity and scale of video data demand retrieval systems capable of multimodal understanding, adaptive reasoning, and domain-specific knowledge integration. This paper presents LLandMark, a modular multi-agent framework for landmark-aware multimodal video retrieval to handle real-world complex queries. The framework features specialized agents that collaborate across four stages: query parsing and planning, landmark reasoning, multimodal retrieval, and reranked answer synthesis. A key component, the Landmark Knowledge Agent, detects cultural or spatial landmarks and reformulates them into descriptive visual prompts, enhancing CLIP-based semantic matching for Vietnamese scenes. To expand capabilities, we introduce an LLM-assisted image-to-image pipeline, where a large language model (Gemini 2.5 Flash) autonomously detects landmarks, generates image search queries, retrieves representative images, and performs CLIP-based visual similarity matching, removing the need for manual image input. In addition, an OCR refinement module leveraging Gemini and LlamaIndex improves Vietnamese text recognition. Experimental results show that LLandMark achieves adaptive, culturally grounded, and explainable retrieval performance.

  </details>



- **Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids**  
  Hongjin Chen, Wei Zhang, Pengfei Li, Shihao Ma, Ke Ma, Yujie Jin, Zijun Xu, Xiaohui Wang, Yupeng Zheng, Zining Wang, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02856v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Realizing interactive whole-body control for multi-humanoid systems is critical for unlocking complex collaborative capabilities in shared environments. Although recent advancements have significantly enhanced the agility of individual robots, bridging the gap to physically coupled multi-humanoid interaction remains challenging, primarily due to severe kinematic mismatches and complex contact dynamics. To address this, we introduce Rhythm, the first unified framework enabling real-world deployment of dual-humanoid systems for complex, physically plausible interactions. Our framework integrates three core components: (1) an Interaction-Aware Motion Retargeting (IAMR) module that generates feasible humanoid interaction references from human data; (2) an Interaction-Guided Reinforcement Learning (IGRL) policy that masters coupled dynamics via graph-based rewards; and (3) a real-world deployment system that enables robust transfer of dual-humanoid interaction. Extensive experiments on physical Unitree G1 robots demonstrate that our framework achieves robust interactive whole-body control, successfully transferring diverse behaviors such as hugging and dancing from simulation to reality.

  </details>



- **VSearcher: Long-Horizon Multimodal Search Agent via Reinforcement Learning**  
  Ruiyang Zhang, Qianguo Sun, Chao Song, Yiyan Qi, Zhedong Zheng  
  _2026-03-03_ · https://arxiv.org/abs/2603.02795v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Large models are increasingly becoming autonomous agents that interact with real-world environments and use external tools to augment their static capabilities. However, most recent progress has focused on text-only large language models, which are limited to a single modality and therefore have narrower application scenarios. On the other hand, multimodal large models, while offering stronger perceptual capabilities, remain limited to static knowledge and lack the ability to access and leverage up-to-date web information. In this paper, we propose VSearcher, turning static multimodal model into multimodal search agent capable of long-horizon, multi-turn tool use in real-world web environments, including text search, image search, and web browsing, via reinforcement learning. Specifically, we introduce Iterative Injection Data Synthesis pipeline to generate large-scale, complex multimodal QA questions, which are further filtered with comprehensive metrics to ensure high quality and sufficient difficulty. We then adopt an SFT-then-RL training pipeline to turn base multimodal models to agent capable of multi-turn tool calling in real-world web environments. Besides, we propose a multimodal search benchmark MM-SearchExam dedicated to evaluating search capabilities of multimodal search agents, which proves highly challenging for recent proprietary models. Extensive evaluations across multiple multimodal search benchmarks reveal effectiveness of our method. VSearcher achieves superior performance compared to recent multimodal search agents and even surpasses several proprietary models on multimodal web search tasks.

  </details>



- **Next Embedding Prediction Makes World Models Stronger**  
  George Bredis, Nikita Balagansky, Daniil Gavrilov, Ruslan Rakhimov  
  _2026-03-03_ · https://arxiv.org/abs/2603.02765v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  Capturing temporal dependencies is critical for model-based reinforcement learning (MBRL) in partially observable, high-dimensional domains. We introduce NE-Dreamer, a decoder-free MBRL agent that leverages a temporal transformer to predict next-step encoder embeddings from latent state sequences, directly optimizing temporal predictive alignment in representation space. This approach enables NE-Dreamer to learn coherent, predictive state representations without reconstruction losses or auxiliary supervision. On the DeepMind Control Suite, NE-Dreamer matches or exceeds the performance of DreamerV3 and leading decoder-free agents. On a challenging subset of DMLab tasks involving memory and spatial reasoning, NE-Dreamer achieves substantial gains. These results establish next-embedding prediction with temporal transformers as an effective, scalable framework for MBRL in complex, partially observable environments.

  </details>


