# 3D_Object_Detection
3D Objecet Detection Papers

## Table of Contents
1. Individual Papers
 * [PointPillars CVPR'19](#pointpillars-cvpr19)
 * [PointRCNN CVPR'19](#pointrcnn-cvpr19)
 * [PointNet++ Neurips'17](#pointnet-neurips17)
 * [PointNet CVPR'17](#pointnet-cvpr17)
2. [3D DA, UDA](#3d-da-uda)
3. [Pre-training](#pre-training)
4. [Semi-supervised](#semi-supervised)
5. [Other Papers](#other-papers)

#
#### [PointPillars CVPR'19](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf)
1. Good Intro!
2. A lidar uses a laser scanner to measure the distance to the environment, thus generating a sparse point cloud representation. 
3. Traditionally, a lidar robotics pipeline interprets such point clouds as object detections through a bottomup pipeline involving background subtraction, followed by spatiotemporal clustering and classification [12, 9].
4. While there are many similarities between the modalities, there are two key differences: 1) **the point cloud is a sparse representation**, while an image is dense and 2) the point cloud is 3D, while the image is 2D. As a result, object detection from point clouds does not trivially lend itself to standard image convolutional pipelines.
5. the bird’s eye view tends to be **extremely sparse**. A common workaround to this problem is to partition the ground plane into a regular grid, for example 10 x 10 cm, and then perform a hand-crafted feature encoding method on the points in each grid cell [2, 11, 26, 32]
6. However, such methods may be sub-optimal since the hard-coded feature extraction method **may not generalize to new configurations without significant engineering efforts**.
7. To address these issues, and building on the PointNet design developed by Qi et al. [22], VoxelNet [33] was one of the first methods to truly do end to-end learning in this domain. 
8.  **VoxelNet** divides the space into voxels, applies a PointNet to each voxel, followed by a 3D convolutional middle layer to consolidate the vertical axis, after which a 2D convolutional detection architecture is applied. While the VoxelNet performance is strong, the inference time, at 4.4 Hz, is too slow to deploy in real time.
9.  we propose PointPillars: a method for object detection in 3D that enables end-to-end learning with **only 2D convolutional layers**. 
10.  PointPillars uses a novel encoder that learns features on pillars (vertical columns) of the point cloud to predict 3D oriented boxes for objects. by learning
features instead of relying on fixed encoders, PointPillars can leverage the full information represented by the point cloud. Further, by operating on pillars instead of voxels there is no need to tune the binning of the vertical direction by hand. Finally, pillars are fast because all key operations can be formulated as 2D convolutions which are extremely efficient to compute on a GPU. An additional benefit of learning features is that PointPillars requires no hand-tuning to use different point cloud configurations such as multiple lidar scans or even radar point clouds.

![image](https://user-images.githubusercontent.com/13063395/139265900-80aa9814-f851-4da3-98c5-9179b1e97220.png)

#### [PointRCNN CVPR'19](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.pdf)

![image](https://user-images.githubusercontent.com/13063395/139439993-b0e36ec3-5b07-4725-ab60-eda23f88f672.png)

#### [PointNet++ Neurips'17](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)
https://medium.com/@sanketgujar95/https-medium-com-sanketgujar95-pointnetplus-5d2642560c0d
* by design PointNet does not capture **local structures** induced by the metric space points live in, limiting its ability to recognize fine-grained patterns and **generalizability** to complex scenes.
* In this work, we introduce a **hierarchical neural network** that applies PointNet **recursively** on a **nested partitioning** of the input point set.
* With further observation that point sets are usually sampled with **varying densities**, which results in greatly **decreased performance** for networks trained on uniform densities, we propose novel **set learning layers** to adaptively combine features from **multiple scales**.
* The basic idea of PointNet is to **learn a spatial encoding of each point** and then aggregate all individual point features to a global point cloud signature.
* The ability to abstract local patterns along the hierarchy allows better generalizability to unseen cases.
* **The general idea of PointNet++** is simple. We first **partition** the set of points into **overlapping local regions** by the distance metric of the underlying space. Similar to CNNs, we **extract local features** capturing fine geometric structures from small neighborhoods; **such local features are further grouped into larger units** and processed to produce higher level features. This process is repeated until we obtain the features of the whole point set.
* The design of PointNet++ has to address **two issues**: how to generate the **partitioning** of the point set, and **how to abstract sets of points** or local features through a local feature learner.
* **Abstract sets of points**: PointNet
* **Partitioning**: Each partition is defined as a **neighborhood ball** in the underlying **Euclidean** space, whose parameters include **centroid** location and **scale**. To evenly cover the whole set, the centroids are selected among input point set by a farthest point sampling (FPS) algorithm. Compared with volumetric CNNs that scan the space with fixed strides, our local receptive fields are dependent on both the input data and the metric, and thus more efficient and effective.
* Asignificant contribution of our paper is that PointNet++ leverages neighborhoods at multiple scales to achieve both robustness and detail capture. Assisted with **random input dropout** during training, the network learns to adaptively weight patterns detected at different scales and combine multi-scale features according to the input data.
* Our hierarchical structure is composed by a number of **set abstraction levels**. At each level, a set of points is processed and abstracted to produce a new set with fewer elements. The set abstraction level is made of three key layers: **Sampling layer, Grouping layer and PointNet layer**. The Sampling layer selects a set of points from input points, which defines the centroids of local regions. Grouping layer then constructs local region sets by finding “neighboring” points around the centroids. PointNet layer uses a mini-PointNet to encode local region patterns into feature vectors.
* **Sampling layer**. Given input points {x1,x2,...,xn}, we use iterative farthest point sampling (FPS) to choose a subset of points {xi1,xi2,...,xim}, such that xij is the most distant point (in metric distance) from the set {xi1,xi2,...,xij−1} with regard to the rest points. In contrast to CNNs that scan the vector space agnostic of data distribution, **our sampling strategy generates receptive fields in a data dependent manner**.
* **Grouping layer.** The input to this layer is a point set of size **N × (d + C)** and the coordinates of a set of centroids of size **N'× d**. The output are groups of point sets of size **N'× K × (d + C)**, where each group corresponds to a local region and K is the number of points in the neighborhood of centroid points. Note that K varies across groups but the succeeding PointNet layer is able to convert flexible number of points into a fixed length local region feature vector. **Ball query** finds all points that are within a radius to the query point (an upper limit of K is set in implementation). An alternative range query is **K nearest neighbor** (kNN) search which finds a fixed number of neighboring points. Compared with kNN, ball query’s local neighborhood guarantees a fixed region scale thus making local region feature more generalizable across space, which is preferred for tasks requiring local pattern recognition (e.g. semantic point labeling).
* **PointNet layer**. In this layer, the input are N' local regions of points with data size N'×K×(d+C). Each local region in the output is abstracted by its centroid and local feature that encodes the centroid’s neighborhood. Output data size is N'× (d + C'). The coordinates of points in a local region are firstly translated into a local frame relative to the centroid point. We use PointNet [20] as described in Sec. 3.1 as the basic building block for local pattern learning. By using relative coordinates together with point features we can capture point-to-point relations in the local region.
* **Multi-scale grouping (MSG).**, a simple but effective way to capture multiscale patterns is to apply grouping layers with different scales followed by according PointNets to extract features of each scale. Features at different scales are concatenated to form a multi-scale feature.
* **Multi-resolution grouping (MRG).**
* 

#### [PointNet CVPR'17](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)

[Very good summary by the authors](https://www.youtube.com/watch?v=Cge-hot0Oc0&ab_channel=ComputerVisionFoundationVideos)

[another good explanation](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a)

[another explanation with some code](https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263)

#### [ST3D CVPR'21](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_ST3D_Self-Training_for_Unsupervised_Domain_Adaptation_on_3D_Object_Detection_CVPR_2021_paper.pdf)

1. Random Object Scaling (ROS), in pre-training phase, to overcome object size bias on the labeled source domain
 * it points to point densities bias but this module does not handle this
  

2. Pseudo Label generation using Quality-aware Triplet Memory Bank (QTMB), using IOU based scores, keeping a history, ensemble and voting
 * **Good explanation:** _First_, the **confidence** of object category prediction may not necessarily reflect the **precision** of location as shown by the blue line in Fig. 3 (a). _Second_, the fraction of false labels is much increased in confidence score intervals with **medium values** as illustrated in Fig. 3 (b). _Third_, model fluctuations induce inconsistent pseudo labels as demonstrated in Fig. 3 (c). The above factors will undoubtedly have negative impacts on the pseudo-labeled objects, leading to noisy supervisory information and instability for self-training.
 * Generating pseudo labels using the pretrained model from prev. stage. then caching these pseudo labels in the memory bank for later usage. updating these seudo labels based on IoU score. try to avoid ambiguous samples by setting thresholds for ignoring noisy labels. noisy labels are those ones with medium values for IoU or ..? **Two IoU thresholds for matching**
 * memory voting: when updating the memory bank, if current generated pseudo labels have enough overlaps with old labels with higher confidence, they will be used for the next training stage. If they do not have match with older labels we have to decide to ignore or discard them. So we keep history for the number of times that a new generated label is not matched with old ones if it is exceeded than a threshold we discard it. if it is less than another threshold we cach it otherwise we ignore it during training because it is noisy. **one threshold for the confidence (updating the bank) and Two thresholds for voting part**

3. Curriculum Data Augmentation (CDA), progressively increasing the intensity of augmentation, preventing from overfitting to easy examples from previous stage. prev. stage produces only easy examples and training on them leads to overfitting so the authors use data augmentation to avoide this. 

4. Experiments on: Waymo -> kitti, Waymo -> Lyft, Waymo -> nuScenes, nuScenes -> kitti 
 * using kitti eval metrics
 * comaring with Oracle, source only and train in germany test in usa paper
 * using two point cloud architectures: SECOND and pvrcnn
 * codebase based on openpcdet
 * for all datasets [-75,75]m for X and Y axis, [-2, 4] for Z

5. Ablattions: size normalization from train in germany test in usa brings a large improvement in Waymo -> kitti and nuScenes -> kitti
 * even naive self-training is useful at least on 3d eval setting.
 * 


#### [3DIoUMatch CVPR'21](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_3DIoUMatch_Leveraging_IoU_Prediction_for_Semi-Supervised_3D_Object_Detection_CVPR_2021_paper.pdf)

1. Using votenet and pvrcnn
2. In the pretraining phase the authors train a votenet with same losses plus a 3d IoU loss on the labeled set.
3. Then they train student teacher models on both labeled and unlabeled data and adapt an EMA teacher. 
4. using asymetric data augmentation and pseudo label filtering

# 
## 3D DA, UDA
* ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection CVPR'21
  * First, we pre-train the 3D detector on the source domain with our proposed random object scaling strategy for mitigating the negative effects of source domain bias. 
  * Then, the detector is iteratively improved on the target domain by alternatively conducting two steps, which are the **pseudo label updating** with the developed **quality-aware triplet memory bank** and the model training with **curriculum data augmentation**.   
* Unsupervised Domain Adaptive 3D Detection with Multi-Level Consistency ICCV'21
* Adversarial Training on Point Clouds for Sim-to-Real 3D Object Detection
* PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation Neurips'19
* FAST3D: Flow-Aware Self-Training for 3D Object Detectors
* A Survey on Deep Domain Adaptation for LiDAR Perception 2021
* Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds CVPR'21
* Self-Supervised Learning for Domain Adaptation on Point Clouds WACV'21
#

## Pre-training
* Self-supervised Pretraining of 3d Features on any point-cloud ICCV'21
* Guided Point Contrastive Learning for Semi-supervised Point Cloud Semantic Segmentation ICCV'21
#

## Semi-supervised 
* Multimodal Semi-Supervised Learning for 3D Objects
* Multimodal virtual point 3d detection
* 3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection CVPR'21
  * Although SESS brings noticeable improvements upon a vanilla VoteNet when using only a small portion of labeled data, we find their consistency regularization suboptimal, as it is uniformly enforced on all the student and teacher predictions. In this work, we instead propose to apply confidence-based filtering to improve the quality of pseudo-labels from the teacher predictions and we are the first (in both 2D and 3D object detection) to introduce IoU estimation for localization filtering. 
  * **IoU estimation** was first proposed in a 2D object detection work IoU-Net [12], which proposed an IoU head that runs in parallel to bounding box refinement and is **differentiable** w.r.t. bounding box parameters. IoUNet adds an IoU estimation head to several off-the-shelf 2D detectors and uses IoU estimation instead of classification confidence to guide NMS, which improves the performance consistently over different backbones. 
  * For 3D object detection, STD [32] follows IoU-Net to add a simple IoU estimation branch parallel with the box estimation branch and to guide NMS with IoU estimation. PV-RCNN [24] devises a similar 3D IoU estimation module and use it at IoU-guided NMS stage. These two modules, unfortunately, are not suitable for IoU optimization **as the features fed to the IoU estimation branch are not differentiable w.r.t. the bounding box size**.
* SESS: Self-Ensembling Semi-Supervised 3D Object Detection CVPR'20
  * SESS is built upon VoteNet [18] and adopts a two-stage training scheme. It leverages a mutual learning framework composed of an EMA teacher and a student, uses asymmetric data augmentation, and enforces three kinds of consistency losses between the teacher and student outputs. 
  * The idea behind self-ensembling approaches is to improve the generalization of a model b**y encouraging consensus among ensemble predictions of unknown samples under small perturbations of inputs or network parameters.** For instance, Γ model [13], a variation of ladder network [23], consists of two identical parallel branches that respectively take one image and the corrupted version of the image as input. The consistency loss is computed based on the difference between the (pre-activated) predictions from the clean branch and the (pre-activated) corrupted branches processed by an explicit denoising layer. In contrast to Γ model, Π model [6] discards the explicit denoising layer and inputs the same image with different corruption conditions into a single branch. Virtual Adversarial Training [10] **shares similar idea with the Π model but uses adversarial perturbation instead of independent noise.** Temporal model [6], an extension of Π model, forces the consistency between the recent network output and the aggregation of network predictions over multiple previous training epochs rather than predictions from auxiliary corrupted input. However, this model becomes cumbersome when applied to large dataset because it needs to maintain a per-sample moving average of the historical network predictions. 
* PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection CVPR'20
  * PV-RCNN[24] is a high-performance and efficient LiDAR point cloud detector that deeply integrates both 3D voxel CNNs and PointNet++- style set abstraction to learn more discriminative point cloud features. Specifically, PV-RCNN first passes the 3D scene through a novel voxel set abstraction module based on sparse 3D CNN to get a set of keypoints with representative scene features. Then RoI grid pooling is then applied to the keypoints to abstract proposal-specific features into RoI grid points. The RoI grid points containing rich context information are finally used to accurately estimate bounding box parameters. PV-RCNN itself incorporates an IoU-estimation module which can predict the IoU of each bounding box and use it to guide the sorting of the boxes. 
#

## Uncategorized papers
* [SemanticKITTI: A Dataset for Semantic Scene Understanding
of LiDAR Sequences ICCV'19](https://arxiv.org/pdf/1904.01416.pdf)
* [Towards 3D LiDAR-based semantic scene understanding of 3D point cloud sequences: The SemanticKITTI Dataset 2021](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/behley2021ijrr.pdf)
* [Frustum PointNets for 3D Object Detection from RGB-D Data CVPR'18](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf)

## Other related work but in other task like 2d detection or classification
* Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision CVPR'21
* FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence Neurips'20
* Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results Neurips'17
  * Mean Teacher [22] tackles the weakness of temporal model by replacing network prediction average with network parameter average. It contains two network branches - teacher and student with the same architecture. The parameters of the teacher are the exponential moving average of the student network parameters that are updated by stochastic gradient descent. The student network is trained to yield consistent predictions with the teacher network. 

## Theoritical General Papers
* THEORETICAL ANALYSIS OF SELF-TRAINING WITH DEEP NETWORKS ON UNLABELED DATA ICLR'21
