# DDPG_Pendulum-v0- Pytorch
This implementation uses low dimensional input to control the inverted pendulum Gym environment 

## Installation

This software is tested on Ubuntu 16.04(x64) using python3.6, gym==0.10 and Pytorch 0.4.1 with cuda-9.0, cudnn-9.2 and a GTX-1050 GPU. 

The following methods are provided to install dependencies:

## Conda

You can create a conda environment with the required dependencies using: 

```
conda env create -f environment.yml
```

## Test model
The folder contains testagent.py that loads the pretrained network paramters and performs the control task in the given environment.

## Train model
The train.py trains the DDPG agent.
The Config.py contains the hyperparameter to be tuned.

ToDo:
Scale up to another environment
Include wrapper to monitor the model.
Hyperparameter tuning is needed.




#Reference
https://github.com/ShangtongZhang/DeepRL
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
