# Segmentation and Counting of Grape Berries in Field
This repository is the official implementation of our paper:  
[Mask-GK: An Efficient Method Based on Mask Gaussian Kernel for Segmentation and Counting of Grape Berries in Field](https://temp)  

## Chengdu grape berry dataset
Chengdu dataset was captured by us with an iPhone 12 smartphone, an iPhone 13 smartphone, and an HUAWEI mate 40 pro smartphone in vineyards in Longquanyi and Shuangliu districts, Chengdu, China, in July 2023. It contains a total of 150 RGB images of three grape varieties: Kyoho, Shine Muscat and Summer Black. All the images are captured in a frontal pose with approximately the same distance to the grape vines. We add instance segmentation annotations for a total of 50718 grape berries in these images in COCO format. The original size of images and annotations are 4032×3024 and 4096×3072, and are resized to 2048×1536 in model training and testing.

You can download our dataset from the following links:  
The [GBISC dataset](https://pan.baidu.com/temp), code: temp (or for anyone outside China: [GBISC dataset](https://drive.google.com/temp)).

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
In file `'generate_probmaps_mask.py'`, `'generate_probmaps_bbox.py'`, and `'generate_probmaps_point.py'` we show our codes of generating Gaussian kernels with grape berry mask, bbox and point annotations in COCO format.

Example probability maps generated using our proposed mask Gaussian kernel method:

<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_30.jpg" width="390px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_30_mask.jpg" width="390px" />
<details>
<summary>click to show more</summary>
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_13.jpg" width="390px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_13_mask.jpg" width="390px" />
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_8.jpg" width="390px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_8_mask.jpg" width="390px" />
</details>

## Grape berry segmentation and counting
Grape berry instance segmention results obtained from the probability maps predicted be the neural network using watershed algorithm:

<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_16_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_21_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/Kyoho_48_pred.jpg" width="260px" />
<details>
<summary>click to show more</summary>
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_28_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_30_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/ShineMuscat_39_pred.jpg" width="260px" />
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_20_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_33_pred.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/run/README_images/SummerBlack_42_pred.jpg" width="260px" />
</details>
