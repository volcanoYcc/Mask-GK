# Mask-GK
This repository is the official implementation of our paper:  
[Mask-GK: An Efficient Method Based on Mask Gaussian Kernel for Segmentation and Counting of Grape Berries in Field](https://temp)  

## GBISC dataset
The GBISC dataset was captured by us using an iPhone 12 smartphone, an iPhone 13 smartphone, and an HUAWEI mate 40 pro smartphone within vineyards in Longquanyi and Shuangliu districts in Chengdu, Sichuan province, China, in July 2023. This dataset comprises 150 high-resolution RGB images of three distinct grape varieties: Kyoho, Shine Muscat and Summer Black. All images were captured in a frontal pose with approximately the same distance to the grape vines. We add instance segmentation annotations for a total of 50718 grape berries in these images in COCO format. The original size of images are 4032×3024 and 4096×3072, and are resized to 2048×1536 in model training and testing.

Our dataset can be obtained from the following links:  
The [GBISC dataset](https://pan.baidu.com/s/1Or85K8Q46wnZXq1awpDQxw), code: j7tc (or for anyone outside China: [GBISC dataset](https://drive.google.com/temp)).

1. Kyoho

<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_1.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_2.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_3.jpg" width="260px" />
<details>
<summary>click to show more</summary>
  
2. Shine Muscat
  
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_1.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_2.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_3.jpg" width="260px" />

3. Summer Black

<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_1.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_2.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_3.jpg" width="260px" />
</details>

## Mask Gaussian kernels
Our codes for generating Gaussian kernels with grape berry mask, bbox and point annotations are shown in  
`'datasets/GBISC/generate_probmaps_mask.py'`  
`'datasets/GBISC/generate_probmaps_bbox.py'`  
`'datasets/GBISC/generate_probmaps_point.py'`

Example probability maps generated using our proposed mask Gaussian kernel method:

<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_30.jpg" width="390px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_30_mask.jpg" width="390px" />
<details>
<summary>click to show more</summary>
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_13.jpg" width="390px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_13_mask.jpg" width="390px" />
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_8.jpg" width="390px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_8_mask.jpg" width="390px" />
</details>

## The Mask-GK method
In our paper, we introduce a novel high-accuracy detection pipeline named Mask-GK for in field grape berry segmentation and counting.

Project setup  
1. To setup our project on your own device, you need to download all the following files:  
(1). All the code in this repository.  
(2). The GBISC dataset.  
(3). Our [pretrained model weight](https://pan.baidu.com/s/1v8CnsqS5bxd2URVDNqIAbg), code: wygn (or for anyone outside China: [pretrained model weight](https://drive.google.com/file/d/temp)).

2. Extract the GBISC dataset into `'datasets/GBISC'`, and put the model weight into `'run/paper_weight'`.

3. Run `'datasets/GBISC/generate_probmaps_mask.py'` to generate ground truth mask-based probability maps. This process can take several hours, give yourself a short break first.

4. Run `'eval_vis.py'` to evaluate the Mask-GK method and visualize the detection results, run `'train.py'` to train Mask-GK yourself.

Example grape berry instance segmention results of Mask-GK:

<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_16_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_21_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_48_pred.jpg" width="260px" />
<details>
<summary>click to show more</summary>
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_28_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_30_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_39_pred.jpg" width="260px" />
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_20_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_33_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_42_pred.jpg" width="260px" />
</details>
