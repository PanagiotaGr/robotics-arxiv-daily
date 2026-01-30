# SLAM, Localization & Mapping

_Robotics arXiv Daily_

_Updated: 2026-01-30 07:03 UTC_

Total papers shown: **4**


---

- **BLO-Inst: Bi-Level Optimization Based Alignment of YOLO and SAM for Robust Instance Segmentation**  
  Li Zhang, Pengtao Xie  
  _2026-01-29_ · https://arxiv.org/abs/2601.22061v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  The Segment Anything Model has revolutionized image segmentation with its zero-shot capabilities, yet its reliance on manual prompts hinders fully automated deployment. While integrating object detectors as prompt generators offers a pathway to automation, existing pipelines suffer from two fundamental limitations: objective mismatch, where detectors optimized for geometric localization do not correspond to the optimal prompting context required by SAM, and alignment overfitting in standard joint training, where the detector simply memorizes specific prompt adjustments for training samples rather than learning a generalizable policy. To bridge this gap, we introduce BLO-Inst, a unified framework that aligns detection and segmentation objectives by bi-level optimization. We formulate the alignment as a nested optimization problem over disjoint data splits. In the lower level, the SAM is fine-tuned to maximize segmentation fidelity given the current detection proposals on a subset ($D_1$). In the upper level, the detector is updated to generate bounding boxes that explicitly minimize the validation loss of the fine-tuned SAM on a separate subset ($D_2$). This effectively transforms the detector into a segmentation-aware prompt generator, optimizing the bounding boxes not just for localization accuracy, but for downstream mask quality. Extensive experiments demonstrate that BLO-Inst achieves superior performance, outperforming standard baselines on tasks in general and biomedical domains.

  </details>



- **Holographic generative flows with AdS/CFT**  
  Ehsan Mirafzali, Sanjit Shashi, Sanya Murdeshwar, Edgar Shaghoulian, Daniele Venturi, Razvan Marinescu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22033v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  We present a framework for generative machine learning that leverages the holographic principle of quantum gravity, or to be more precise its manifestation as the anti-de Sitter/conformal field theory (AdS/CFT) correspondence, with techniques for deep learning and transport theory. Our proposal is to represent the flow of data from a base distribution to some learned distribution using the bulk-to-boundary mapping of scalar fields in AdS. In the language of machine learning, we are representing and augmenting the flow-matching algorithm with AdS physics. Using a checkerboard toy dataset and MNIST, we find that our model achieves faster and higher quality convergence than comparable physics-free flow-matching models. Our method provides a physically interpretable version of flow matching. More broadly, it establishes the utility of AdS physics and geometry in the development of novel paradigms in generative modeling.

  </details>



- **ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models**  
  Bowen Fang, Wen Ye, Yunyue Su, Jinghao Zhang, Qiang Liu, Yesheng Liu, Xin Sun, Shu Wu, Jiabing Yang, Baole Wei, et al.  
  _2026-01-29_ · https://arxiv.org/abs/2601.21947v1 · `cs.AI`  
  <details><summary>Abstract</summary>

  Prevalent retrieval-based tool-use pipelines struggle with a dual semantic challenge: their retrievers often employ encoders that fail to capture complex semantics, while the Large Language Model (LLM) itself lacks intrinsic tool knowledge from its natural language pretraining. Generative methods offer a powerful alternative by unifying selection and execution, tasking the LLM to directly learn and generate tool identifiers. However, the common practice of mapping each tool to a unique new token introduces substantial limitations: it creates a scalability and generalization crisis, as the vocabulary size explodes and each tool is assigned a semantically isolated token. This approach also creates a semantic bottleneck that hinders the learning of collaborative tool relationships, as the model must infer them from sparse co-occurrences of monolithic tool IDs within a vast library. To address these limitations, we propose ToolWeaver, a novel generative tool learning framework that encodes tools into hierarchical sequences. This approach makes vocabulary expansion logarithmic to the number of tools. Crucially, it enables the model to learn collaborative patterns from the dense co-occurrence of shared codes, rather than the sparse co-occurrence of monolithic tool IDs. We generate these structured codes through a novel tokenization process designed to weave together a tool's intrinsic semantics with its extrinsic co-usage patterns. These structured codes are then integrated into the LLM through a generative alignment stage, where the model is fine-tuned to produce the hierarchical code sequences. Evaluation results with nearly 47,000 tools show that ToolWeaver significantly outperforms state-of-the-art methods, establishing a more scalable, generalizable, and semantically-aware foundation for advanced tool-augmented agents.

  </details>



- **Synthetic-to-Real Domain Bridging for Single-View 3D Reconstruction of Ships for Maritime Monitoring**  
  Borja Carrillo-Perez, Felix Sattler, Angel Bueno Rodriguez, Maurice Stephan, Sarah Barnes  
  _2026-01-29_ · https://arxiv.org/abs/2601.21786v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Three-dimensional (3D) reconstruction of ships is an important part of maritime monitoring, allowing improved visualization, inspection, and decision-making in real-world monitoring environments. However, most state-ofthe-art 3D reconstruction methods require multi-view supervision, annotated 3D ground truth, or are computationally intensive, making them impractical for real-time maritime deployment. In this work, we present an efficient pipeline for single-view 3D reconstruction of real ships by training entirely on synthetic data and requiring only a single view at inference. Our approach uses the Splatter Image network, which represents objects as sparse sets of 3D Gaussians for rapid and accurate reconstruction from single images. The model is first fine-tuned on synthetic ShapeNet vessels and further refined with a diverse custom dataset of 3D ships, bridging the domain gap between synthetic and real-world imagery. We integrate a state-of-the-art segmentation module based on YOLOv8 and custom preprocessing to ensure compatibility with the reconstruction network. Postprocessing steps include real-world scaling, centering, and orientation alignment, followed by georeferenced placement on an interactive web map using AIS metadata and homography-based mapping. Quantitative evaluation on synthetic validation data demonstrates strong reconstruction fidelity, while qualitative results on real maritime images from the ShipSG dataset confirm the potential for transfer to operational maritime settings. The final system provides interactive 3D inspection of real ships without requiring real-world 3D annotations. This pipeline provides an efficient, scalable solution for maritime monitoring and highlights a path toward real-time 3D ship visualization in practical applications. Interactive demo: https://dlr-mi.github.io/ship3d-demo/.

  </details>


