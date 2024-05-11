import gym
import mlagents_envs
# from baselines import deepq
# from baselines import logger
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from gym_trainer.a2c import MyA2C

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from torch.utils.tensorboard import SummaryWriter
import sys
import torch
import numpy as np
import random
from pathlib import Path 
def main():
  # unity_env = UnityEnvironment("/Users/rishimalhotra/projects/checker2_one_agent.app")
  unity_env = UnityEnvironment()
  torch.manual_seed(0)
  np.random.seed(0)
  random.seed(0)
  vec_env = UnityToGymWrapper(unity_env, uint8_visual=True)


  # avg length: 43.
  policy_network_checkpoint_path = "checkpoints/using_log_exp_with_lower_lr_for_sigma/step_18003/policy_network.pth"

  model = MyA2C(vec_env, None, n_steps=10, policy_network_checkpoint_path=policy_network_checkpoint_path)
  average_episode_length, std_episode_length = model.collect_rollouts_for_inference(num_episodes=100, deterministic=True)
  print('average episode length:', average_episode_length)
  # model.save("a2c_cartpole")
  



if __name__ == '__main__':
  main()