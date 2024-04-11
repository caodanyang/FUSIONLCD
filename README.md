# FUSION

## Table of Contents
- [Paper](#paper)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Running the Code](#running-the-code)
- [Evaluation](#evaluation)
## Paper
If you find the poject helps you, you can cite our paper:
Cao D, Yue H, Liu Z, et al. BEVLCD+: Real-Time and Rotation-Invariant Loop Closure Detection Based on BEV of Point Cloud[J]. IEEE Transactions on Instrumentation and Measurement, 2023.
Yue H, Cao D, Liu Z, et al. Cross Fusion of Point Cloud and Learned Image for Loop Closure Detection[J]. IEEE Robotics and Automation Letters, 2024.

## Overview
We provide code for BEV mode and fusion mode, so you can easily train and test.

## Prerequisites
Before you can use this project, you'll need to do the following:

1. **Download Datasets**: Download the [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/download.php).

2. **Prepare Dataset Structure**: Use `preparedataset.py` to construct a dataset structure that complies with the project's requirements. Make sure to update the necessary paths in the code.

3. **Prepare environment**: Use the commonds on `env.txt` to create your environment. Windows and Ubuntu is OK.
 
## Running the Code
To run the code, follow these steps:

1. Configure the code to run in either BEV mode or fusion mode using the settings in `config.yaml`.

2. If you want to load a trained model used in the paper, ensure that you update the file path accordingly.

3. Run `python train.py`


## Evaluation
Evaluate the saved data using the evaluation script.

## Others
If you have any questions please feel free to contact us.
