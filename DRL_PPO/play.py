#!/usr/bin/env python3
import argparse
import gym
import roboschool

from ppo import *

import numpy as np
import torch


ENV_ID = "RoboschoolHalfCheetah-v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", default= "monitor", help="Sets the recording directory")
    args = parser.parse_args()
    env = gym.make(args.env)
    net = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))
    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor(obs)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        env.render("human")
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            env.close()
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
