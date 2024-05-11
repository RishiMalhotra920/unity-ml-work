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
def main(run_id):
  # unity_env = UnityEnvironment("/Users/rishimalhotra/projects/checker2_one_agent.app")
  unity_env = UnityEnvironment()
  torch.manual_seed(0)
  np.random.seed(0)
  random.seed(0)
  vec_env = UnityToGymWrapper(unity_env, uint8_visual=True)
  print('hi', vec_env._observation_space, isinstance(vec_env.action_space, gym.spaces.Box), vec_env.action_space)
  writer = SummaryWriter(f"runs/{run_id}")
  print('run id: ', run_id)

  checkpoint_path = Path(f"checkpoints/{run_id}")
  checkpoint_path.mkdir(parents=True, exist_ok=True)

  model = MyA2C(vec_env, writer, n_steps=10)
  model.learn(total_timesteps=25000, policy_network_lr=7e-4, value_network_lr=7e-4, sigma_lr=1e-4, ent_coef=10e-6, checkpoint_path=checkpoint_path)
  vec_env.close()
  # model.save("a2c_cartpole")
  

  # logger.configure('./logs')  # Change to log in a different directory
  # act = deepq.learn(
  #   env,
  #   "cnn",  # For visual inputs
  #   lr=2.5e-4,
  #   total_timesteps=1000000,
  #   buffer_size=50000,
  #   exploration_fraction=0.05,
  #   exploration_final_eps=0.1,
  #   print_freq=20,
  #   train_freq=5,
  #   learning_starts=20000,
  #   target_network_update_freq=50,
  #   gamma=0.99,
  #   prioritized_replay=False,
  #   checkpoint_freq=1000,
  #   checkpoint_path='./logs',  # Change to save model in a different directory
  #   dueling=True
  # )
  print("Saving model to unity_model.pkl")
  # act.save("unity_model.pkl")


if __name__ == '__main__':
  if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
      if arg.startswith("--run-id="):
        run_id = arg[9:]
        main(run_id)
