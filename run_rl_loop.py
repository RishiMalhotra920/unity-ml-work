
from a2c.generate_data_from_environment import generate_data_from_environment
from a2c.train import train
from a2c.model import actor, critic
from pathlib import Path
from a2c.inference import epsilon_greedy_infer
from mlagents_envs.environment import UnityEnvironment
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
  # env = UnityEnvironment(file_name="/Users/rishimalhotra/projects/checker3.app")
  env = UnityEnvironment()
  torch.manual_seed(42)
  np.random.seed(42)
  # env = UnityEnvironment(file_name="apps/many_agents.app")
  env.reset()

  checkpoints_dir = Path("checkpoints")
  dataset_dir = Path("datasets")

  num_steps_in_rl_loop = 50
  num_epochs = 5
  num_episodes_for_data_generation=500
  epsilon_min = 0.1 
  epsilon = 1.0
  epsilon_decay = 0.95
  
  time_now = datetime.now()
  for rl_loop_step in range(num_steps_in_rl_loop):
    print(f"====rl loop step {rl_loop_step}====")
    writer = SummaryWriter(f"runs/{time_now}/rl_loop_step_{rl_loop_step}")
    policy = lambda x: epsilon_greedy_infer(actor, x, epsilon)
    data = generate_data_from_environment(policy, env, num_episodes=num_episodes_for_data_generation, save_path=dataset_dir / f"step_{rl_loop_step}.pkl")
    actor, critic = train(data, actor, critic, num_epochs, writer, checkpoints_dir / f"actor_step_{rl_loop_step}.pth", checkpoints_dir / f"critic_step_{rl_loop_step}.pth")
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

  env.close()

  
