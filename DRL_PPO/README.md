# PPO_RoboschoolHalfCheetah-v1 Pytorch
This implementation uses low dimensional input to control the RoboschoolHalfCheetah environment 

## Installation

This software is tested on Ubuntu 16.04(x64) using python3.6, gym==0.10, roboschool=1.0 and Pytorch 0.4.1 with cuda-9.0 and cudnn-9.2 . 

The following methods are provided to install dependencies:

## Conda

You can create a conda environment with the required dependencies using: 

```
conda env create -f environment.yml
```
## Test model
The folder contains play.py that can loads the pretrained network paramters and performs the control task in the given environment.
Specify the path to the pretrained model. e.g.,

```
python play.py -m /DRL_PPO/saves/ppo-train/best_+2549.112_1496000.dat
```

## Train modelThe train.py trains the PPO agent  and saves the logs in runs directory 
The Config.py contains the hyperparameter to be tuned.


# Reference
- https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter15
- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
- https://github.com/ShangtongZhang/DeepRL
- https://github.com/udacity/deep-reinforcement-learning

