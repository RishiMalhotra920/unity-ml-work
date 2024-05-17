
from generate_data_from_environment import generate_data_from_environment
from pathlib import Path
from inference import infer
from model import load_pi_network
from mlagents_envs.environment import UnityEnvironment
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

if __name__ == "__main__":
  # env = UnityEnvironment(file_name="/Users/rishimalhotra/projects/checker3.app")
  env = UnityEnvironment()
  torch.manual_seed(42)
  np.random.seed(42)
  # env = UnityEnvironment(file_name="apps/many_agents.app")
  env.reset()
  
  writer = SummaryWriter(f"runs/inference_{datetime.now()}")
  # good_checkpoints_dir = Path("good_checkpoints")
  good_checkpoints_dir = Path("good_checkpoints")
  dataset_dir = Path("datasets")

  actor = load_pi_network(good_checkpoints_dir / "step_26.pth")

  num_episodes_for_data_generation = 100
  policy = lambda x: infer(actor, x, mode='greedy')
  generate_data_from_environment(policy, env, num_episodes_for_data_generation, writer, 0, save_path=None)

  env.close()