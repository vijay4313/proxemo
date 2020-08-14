# ProxEmo: Gait-based Emotion Learning and Multi-view Proxemic Fusion for Socially-Aware Robot Navigation

ProxEmo is a novel end-to-end emotion prediction algorithm for socially aware robot navigation among pedestrians. The approach predicts the perceived emotions of a pedestrian from walking gaits, which is then used for emotion-guided navigation taking into account social and proxemic constraints. Multi-view skeleton graph convolution-based model uses commodity camera mounted onto a moving robot to classify emotions. Our emotion recognition is integrated into a mapless navigation scheme and makes no assumptions about the environment of pedestrian motion.

<div align="center">
  <img src="https://github.com/vijay4313/emotion-bot/blob/inference/misc/front_image.png"/>
</div>

## Overview

We first capture an RGB video from an onboard camera and extract pedestrian poses and track them at each frame. These tracked poses over a predefined time period are embedded into an image, which is then passed into our ProxEmo model for classifying emotions into four classes. The obtained emotions then undergo proxemic fusion with the LIDAR data and are finally passed into the navigation stack.

<div align="center">
  <img src="https://github.com/vijay4313/emotion-bot/blob/inference/misc/Overview.png"/>
</div>

## Model Architecture

The network is trained on image embeddings of the 5D gait set G, which are scaled up to 244×244. The architecture consists of four group convolution (GC) layers. Each GC layer consists of four groups that have been stacked together. This represents the four group convolution outcomes for each of the four emotion labels. The group convolutions are stacked in two stages represented by Stage 1 and Stage 2. The output of the network has a dimension of 4 × 4 after passing through a sof tmax layer. The final predicted emotion is given by the maxima of this 4×4 output.

<div align="center">
  <img src="https://github.com/vijay4313/emotion-bot/blob/inference/misc/deepFeel.png"/>
</div>

## Prerequisites

The code is implemented in Python and has the following dependency:

1. Python3
2. Pytorch >= 1.4

To run the demo with intel realsense D435 camera following libraries are required:

1. opencv2
2. pyrealsense2
3. cubemos SDK (works with UBUNTU 18)

## Before running the code

### Dataset

Sample dataset can be downloaded from [EWalk: Emotion Walk](http://gamma.cs.unc.edu/GAIT/#EWalk). Sample H5 files can be found in [GitHub](https://github.com/vijay4313/emotion-bot/tree/inference/emotion_classification/sample_data). For full dataset, please contact the authors.

### Pretrained model

VS-GCNN model trained on the above dataset can be downloaded from [google drive](https://drive.google.com/file/d/14SB-wNVW-2Wp9738UWbh_3K3mYO_tSsn/view?usp=sharing)

### Config file changes

Below are the basic changes to be made in config file. Open config file from `[proxemo folder]/emotion_classification/modeling/config` and make following changes.

1. Set the mode

```bash
GENERAL : MODE : ['train' | 'inference' | 'test' ]
```

2. Specify pretrained model path if running in *inferece* or *test* mode or warm starting the training 

```bash
MODEL : PRETRAIN_PATH : <path to model dir>
MODEL : PRETRAIN_NAME : <model file name>
```

3. Specify features and labels H5 files.

```bash
DATA : FEATURES_FILE : <path to features file>
DATA : LABELS_FILE : <path to lables file>
```

## Running the code

Clone the repo.

```bash
git clone https://github.com/vijay4313/emotion-bot.git
cd <proxemo directory>
```

Find the latest release tag from [released versions](https://github.com/vijay4313/emotion-bot/releases) and checkout the latest release.

```bash
git checkout tags/<latest_tag_name>
```

example

```bash
git fetch --all --tags
git checkout tags/v1.0.0
```

All the settings are configured as yaml file from *[proxemo folder]/emotion_classification/modeling/config*. We have provided two settings file one for inference and one to train the model.

To run the code with specific settings file, run the below command

```bash
python3 main.py --settings infer
```

To run the demo, connect intel realsense D435 camera with above mentioned pre requsites and execute below command

```bash
python3 demo.py --model ./emotion_classification/modeling/config/infer.yaml
```

## Links

### [Paper Arxiv version](https://arxiv.org/abs/2003.01062)

### [Demo and Video](https://gamma.umd.edu/researchdirections/affectivecomputing/proxemo/)

## Results

We use the emotions detected by ProxEmo along with the LIDAR data to perform
Proxemic Fusion. This gives us a comfort distance around a pedestrian for emotionally-guided navigation. The green arrows represent the path after accounting for comfort distance while the violet arrows indicate the path without considering this distance. Observe the significant change in the path taken in the sad case. Note that the overhead image is representational, and ProxEmo works entirely from a egocentric camera on a robot.

<div align="center">
  <img src="https://github.com/vijay4313/emotion-bot/blob/inference/misc/CompNavigation.png"/>
</div>

Comparison of ProxEmo with other state-of-theartemotion classification algorithms.

<div align="center">
  <img src="https://github.com/vijay4313/emotion-bot/blob/inference/misc/CompTable.png"/>
</div>

Here we present the performance metrics of our ProxEmo network compared to the state-of-the-art arbitrary view action recognition models. We perform a comprehensive comparison of models across multiple distances of skeletal gaits from the camera and across multiple view-groups. It can be seen that
our ProxEmo network outperforms other state-of-the-art network by 50% at an average in terms of prediction accuracy.

<div align="center">
  <img src="https://github.com/vijay4313/emotion-bot/blob/inference/misc/DeepFeel-model-results.png"/>
</div>

Confusion Matrix

<div align="center">
  <img src="https://github.com/vijay4313/emotion-bot/blob/inference/misc/ConfusionMatrix.png"/>
</div>


## Cite this paper

```bash
@article{narayanan2020proxemo,
  title={ProxEmo: Gait-based Emotion Learning and Multi-view Proxemic Fusion for Socially-Aware Robot Navigation},
  author={Narayanan, Venkatraman and Manoghar, Bala Murali and Dorbala, Vishnu Sashank and Manocha, Dinesh and Bera, Aniket},
  journal={arXiv preprint arXiv:2003.01062},
  year={2020}
}
```

## Contact authors

Venkatraman Narayanan <vnarayan@terpmail.umd.edu>

Bala Murali Manoghar Sai Sudhakar <bsaisudh@umd.edu>

Vishnu Sashank Dorbala <vdorbala@terpmail.umd.edu>

Aniket Bera <ab@cs.umd.edu>