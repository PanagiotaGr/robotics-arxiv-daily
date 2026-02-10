# Multi-Robot & Swarms

_Robotics arXiv Daily_

_Updated: 2026-02-10 07:19 UTC_

Total papers shown: **4**


---

- **Digital Twin and Agentic AI for Wild Fire Disaster Management: Intelligent Virtual Situation Room**  
  Mohammad Morsali, Siavash H. Khajavi  
  _2026-02-09_ · https://arxiv.org/abs/2602.08949v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  According to the United Nations, wildfire frequency and intensity are projected to increase by approximately 14% by 2030 and 30% by 2050 due to global warming, posing critical threats to life, infrastructure, and ecosystems. Conventional disaster management frameworks rely on static simulations and passive data acquisition, hindering their ability to adapt to arbitrarily evolving wildfire episodes in real-time. To address these limitations, we introduce the Intelligent Virtual Situation Room (IVSR), a bidirectional Digital Twin (DT) platform augmented by autonomous AI agents. The IVSR continuously ingests multisource sensor imagery, weather data, and 3D forest models to create a live virtual replica of the fire environment. A similarity engine powered by AI aligns emerging conditions with a precomputed Disaster Simulation Library, retrieving and calibrating intervention tactics under the watchful eyes of experts. Authorized action-ranging from UAV redeployment to crew reallocation-is cycled back through standardized procedures to the physical layer, completing the loop between response and analysis. We validate IVSR through detailed case-study simulations provided by an industrial partner, demonstrating capabilities in localized incident detection, privacy-preserving playback, collider-based fire-spread projection, and site-specific ML retraining. Our results indicate marked reductions in detection-to-intervention latency and more effective resource coordination versus traditional systems. By uniting real-time bidirectional DTs with agentic AI, IVSR offers a scalable, semi-automated decision-support paradigm for proactive, adaptive wildfire disaster management.

  </details>



- **A Generic Service-Oriented Function Offloading Framework for Connected Automated Vehicles**  
  Robin Dehler, Michael Buchholz  
  _2026-02-09_ · https://arxiv.org/abs/2602.08799v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  Function offloading is a promising solution to address limitations concerning computational capacity and available energy of Connected Automated Vehicles~(CAVs) or other autonomous robots by distributing computational tasks between local and remote computing devices in form of distributed services. This paper presents a generic function offloading framework that can be used to offload an arbitrary set of computational tasks with a focus on autonomous driving. To provide flexibility, the function offloading framework is designed to incorporate different offloading decision making algorithms and quality of service~(QoS) requirements that can be adjusted to different scenarios or the objectives of the CAVs. With a focus on the applicability, we propose an efficient location-based approach, where the decision whether tasks are processed locally or remotely depends on the location of the CAV. We apply the proposed framework on the use case of service-oriented trajectory planning, where we offload the trajectory planning task of CAVs to a Multi-Access Edge Computing~(MEC) server. The evaluation is conducted in both simulation and real-world application. It demonstrates the potential of the function offloading framework to guarantee the QoS for trajectory planning while improving the computational efficiency of the CAVs. Moreover, the simulation results also show the adaptability of the framework to diverse scenarios involving simultaneous offloading requests from multiple CAVs.

  </details>



- **Designing Multi-Robot Ground Video Sensemaking with Public Safety Professionals**  
  Puqi Zhou, Ali Asgarov, Aafiya Hussain, Wonjoon Park, Amit Paudyal, Sameep Shrestha, Chia-wei Tang, Michael F. Lighthiser, Michael R. Hieb, Xuesu Xiao, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08882v1 · `cs.HC`  
  <details><summary>Abstract</summary>

  Videos from fleets of ground robots can advance public safety by providing scalable situational awareness and reducing professionals' burden. Yet little is known about how to design and integrate multi-robot videos into public safety workflows. Collaborating with six police agencies, we examined how such videos could be made practical. In Study 1, we presented the first testbed for multi-robot ground video sensemaking. The testbed includes 38 events-of-interest (EoI) relevant to public safety, a dataset of 20 robot patrol videos (10 day/night pairs) covering EoI types, and 6 design requirements aimed at improving current video sensemaking practices. In Study 2, we built MRVS, a tool that augments multi-robot patrol video streams with a prompt-engineered video understanding model. Participants reported reduced manual workload and greater confidence with LLM-based explanations, while noting concerns about false alarms and privacy. We conclude with implications for designing future multi-robot video sensemaking tools. The testbed is available at https://github.com/Puqi7/MRVS\_VideoSensemaking

  </details>



- **Multi-Staged Framework for Safety Analysis of Offloaded Services in Distributed Intelligent Transportation Systems**  
  Robin Dehler, Oliver Schumann, Jona Ruof, Michael Buchholz  
  _2026-02-09_ · https://arxiv.org/abs/2602.08821v1 · `cs.RO`  
  <details><summary>Abstract</summary>

  The integration of service-oriented architectures (SOA) with function offloading for distributed, intelligent transportation systems (ITS) offers the opportunity for connected autonomous vehicles (CAVs) to extend their locally available services. One major goal of offloading a subset of functions in the processing chain of a CAV to remote devices is to reduce the overall computational complexity on the CAV. The extension of using remote services, however, requires careful safety analysis, since the remotely created data are corrupted more easily, e.g., through an attacker on the remote device or by intercepting the wireless transmission. To tackle this problem, we first analyze the concept of SOA for distributed environments. From this, we derive a safety framework that validates the reliability of remote services and the data received locally. Since it is possible for the autonomous driving task to offload multiple different services, we propose a specific multi-staged framework for safety analysis dependent on the service composition of local and remote services. For efficiency reasons, we directly include the multi-staged framework for safety analysis in our service-oriented function offloading framework (SOFOF) that we have proposed in earlier work. The evaluation compares the performance of the extended framework considering computational complexity, with energy savings being a major motivation for function offloading, and its capability to detect data from corrupted remote services.

  </details>


