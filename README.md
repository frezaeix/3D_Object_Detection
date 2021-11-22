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


# 
#### [PointRCNN CVPR'19](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.pdf)

![image](https://user-images.githubusercontent.com/13063395/139439993-b0e36ec3-5b07-4725-ab60-eda23f88f672.png)

# 

#### [PointNet++ Neurips'17](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)
https://medium.com/@sanketgujar95/https-medium-com-sanketgujar95-pointnetplus-5d2642560c0d

# 
#### [PointNet CVPR'17](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)

[Very good summary by the authors](https://www.youtube.com/watch?v=Cge-hot0Oc0&ab_channel=ComputerVisionFoundationVideos)

[another good explanation](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a)

[another explanation with some code](https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263)

# 
#### [ST3D CVPR'21](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_ST3D_Self-Training_for_Unsupervised_Domain_Adaptation_on_3D_Object_Detection_CVPR_2021_paper.pdf)

1. Random Object Scaling (ROS), in pre-training phase, to overcome object size bias on the labeled source domain
 1.1. it points to point densities bias but this module does not handle this 

2. Pseudo Label generation using Quality-aware Triplet Memory Bank (QTMB), using IOU based scores, keeping a history, ensemble and voting
 2.1 **Good explanation:** _First_, the **confidence** of object category prediction may not necessarily reflect the **precision** of location as shown by the blue line in Fig. 3 (a). _Second_, the fraction of false labels is much increased in confidence score intervals with **medium values** as illustrated in Fig. 3 (b). _Third_, model fluctuations induce inconsistent pseudo labels as demonstrated in Fig. 3 (c). The above factors will undoubtedly have negative impacts on the pseudo-labeled objects, leading to noisy supervisory information and instability for self-training.

3. Curriculum Data Augmentation (CDA), progressively increasing the intensity of augmentation, preventing from overfitting to easy examples

4. Experiments on: Waymo -> kitti, Waymo -> Lyft, Waymo -> nuScenes, nuScenes -> kitti 

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
#
## Other Papers
* [SemanticKITTI: A Dataset for Semantic Scene Understanding
of LiDAR Sequences ICCV'19](https://arxiv.org/pdf/1904.01416.pdf)
* [Towards 3D LiDAR-based semantic scene understanding of 3D point cloud sequences: The SemanticKITTI Dataset 2021](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/behley2021ijrr.pdf)
* [Frustum PointNets for 3D Object Detection from RGB-D Data CVPR'18](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf)
