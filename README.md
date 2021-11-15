# 3D_Object_Detection
3D Objecet Detection Papers

## Table of Contents
* [PointPillars CVPR'19](https://github.com/frezaeix/3D_Object_Detection/blob/main/README.md#pointpillars-cvpr19)

#
### TEMPLATE

#### [Paper's Title](https://)

**My Own Abstract**

**Intro and Related Work**

**Method**

**Implementation Details**

**Experimental Setup**

**Ablations**
# 
#### [PointPillars CVPR'19](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf)

**My Own Abstract**

**Intro and Related Work**
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

**Method**
![image](https://user-images.githubusercontent.com/13063395/139265900-80aa9814-f851-4da3-98c5-9179b1e97220.png)


**Implementation Details**

**Experimental Setup**

**Ablations**

# 
#### [PointRCNN CVPR'19](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.pdf)

**My Own Abstract**

**Intro and Related Work**


**Method**
![image](https://user-images.githubusercontent.com/13063395/139439993-b0e36ec3-5b07-4725-ab60-eda23f88f672.png)


**Implementation Details**

**Experimental Setup**

**Ablations**
# 

#### [PointNet++ Neurips'17](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)
https://medium.com/@sanketgujar95/https-medium-com-sanketgujar95-pointnetplus-5d2642560c0d

**My Own Abstract**

**Intro and Related Work**

**Method**

**Implementation Details**

**Experimental Setup**

**Ablations**
# 
#### [PointNet CVPR'17](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)

[Very good summary by the authors](https://www.youtube.com/watch?v=Cge-hot0Oc0&ab_channel=ComputerVisionFoundationVideos)
[another good explanation](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a)
# 
