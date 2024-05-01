
from a2c.generate_data_from_environment import generate_data_from_environment
from a2c.train import train
from pathlib import Path
from a2c.inference import epsilon_greedy_infer, infer
from a2c.model import load_actor, load_critic
from mlagents_envs.environment import UnityEnvironment
import torch
import numpy as np

if __name__ == "__main__":
  # env = UnityEnvironment(file_name="/Users/rishimalhotra/projects/checker3.app")
  env = UnityEnvironment()
  torch.manual_seed(42)
  np.random.seed(42)
  env = UnityEnvironment(file_name="apps//many_agents.app")
  env.reset()
  
  checkpoints_dir = Path("checkpoints")
  dataset_dir = Path("datasets")

  actor = load_actor(checkpoints_dir / "actor_step_9.pth")
  critic = load_critic(checkpoints_dir / "critic_step_9.pth")

  num_steps_in_rl_loop = 10
  num_epochs = 2
  num_episodes_for_data_generation=200
  epsilon_min = 0.1 
  epsilon = 0.8
  epsilon_decay = 0.9

  policy = lambda x: infer(actor, x)
  generate_data_from_environment(policy, env, num_episodes=num_episodes_for_data_generation, save_path=None)

  env.close()