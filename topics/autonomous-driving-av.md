# Autonomous Driving & AV

_Robotics arXiv Daily_

_Updated: 2026-01-21 06:53 UTC_

Total papers shown: **2**


---

- **Correcting and Quantifying Systematic Errors in 3D Box Annotations for Autonomous Driving**  
  Alexandre Justo Miro, Ludvig af Klinteberg, Bogdan Timus, Aron Asefaw, Ajinkya Khoche, Thomas Gustafsson, Sina Sharif Mansouri, Masoud Daneshtalab  
  _2026-01-20_ · https://arxiv.org/abs/2601.14038v1 · `cs.CV`  
  <details><summary>Abstract</summary>

  Accurate ground truth annotations are critical to supervised learning and evaluating the performance of autonomous vehicle systems. These vehicles are typically equipped with active sensors, such as LiDAR, which scan the environment in predefined patterns. 3D box annotation based on data from such sensors is challenging in dynamic scenarios, where objects are observed at different timestamps, hence different positions. Without proper handling of this phenomenon, systematic errors are prone to being introduced in the box annotations. Our work is the first to discover such annotation errors in widely used, publicly available datasets. Through our novel offline estimation method, we correct the annotations so that they follow physically feasible trajectories and achieve spatial and temporal consistency with the sensor data. For the first time, we define metrics for this problem; and we evaluate our method on the Argoverse 2, MAN TruckScenes, and our proprietary datasets. Our approach increases the quality of box annotations by more than 17% in these datasets. Furthermore, we quantify the annotation errors in them and find that the original annotations are misplaced by up to 2.5 m, with highly dynamic objects being the most affected. Finally, we test the impact of the errors in benchmarking and find that the impact is larger than the improvements that state-of-the-art methods typically achieve with respect to the previous state-of-the-art methods; showing that accurate annotations are essential for correct interpretation of performance. Our code is available at https://github.com/alexandre-justo-miro/annotation-correction-3D-boxes.

  </details>



- **PAtt: A Pattern Attention Network for ETA Prediction Using Historical Speed Profiles**  
  ByeoungDo Kim, JunYeop Na, Kyungwook Tak, JunTae Kim, DongHyeon Kim, Duckky Kim  
  _2026-01-20_ · https://arxiv.org/abs/2601.13793v1 · `cs.LG`  
  <details><summary>Abstract</summary>

  In this paper, we propose an ETA model (Estimated Time of Arrival) that leverages an attention mechanism over historical road speed patterns. As autonomous driving and intelligent transportation systems become increasingly prevalent, the need for accurate and reliable ETA estimation has grown, playing a vital role in navigation, mobility planning, and traffic management. However, predicting ETA remains a challenging task due to the dynamic and complex nature of traffic flow. Traditional methods often combine real-time and historical traffic data in simplistic ways, or rely on complex rule-based computations. While recent deep learning models have shown potential, they often require high computational costs and do not effectively capture the spatio-temporal patterns crucial for ETA prediction. ETA prediction inherently involves spatio-temporal causality, and our proposed model addresses this by leveraging attention mechanisms to extract and utilize temporal features accumulated at each spatio-temporal point along a route. This architecture enables efficient and accurate ETA estimation while keeping the model lightweight and scalable. We validate our approach using real-world driving datasets and demonstrate that our approach outperforms existing baselines by effectively integrating road characteristics, real-time traffic conditions, and historical speed patterns in a task-aware manner.

  </details>


