
from a2c.generate_data_from_environment import generate_data_from_environment
from a2c.train import train
from a2c.model import pi_network, load_pi_network
from pathlib import Path
from a2c.inference import infer
from mlagents_envs.environment import UnityEnvironment
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys
import sys

if __name__ == "__main__":

  # get run-id from command line

  if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
      if arg.startswith("--run-id="):
        run_id = arg[9:]
        print(run_id)

  # env = UnityEnvironment(file_name="/Users/rishimalhotra/projects/checker3.app")
  # env = UnityEnvironment(file_name="apps/many_agents.app")
  env = UnityEnvironment()
  torch.manual_seed(42)
  np.random.seed(42)
  # env = UnityEnvironment(file_name="apps/many_agents.app")
  env.reset()

  checkpoints_dir = Path("checkpoints")
  dataset_dir = Path("datasets")

  gamma = 0.95 #good one 0.95
  num_steps_in_rl_loop = 2000
  num_epochs = 1
  num_episodes_for_data_generation=12
  # num_episode_per_agent = 1
  # num_episodes_for_data_generation_decay = 0.8
  lr = 1e-8
  epsilon_min = 0.1 
  epsilon = 1.0
  epsilon_decay = 0.0010
  max_episode_length = 100 #good value 10
  # max_episode_length_increase = 1.01
  top_max_episode_length = 100
  lr_decay = 0.99

  writer = SummaryWriter(f"runs/{run_id}")

  # load good_checkpoints/step_32.pth 
  # pi_network = load_pi_network("good_checkpoints/step_32.pth")

  # link it all to the run id. best checkpoints should be stored under run id
  
  for rl_loop_step in range(num_steps_in_rl_loop):
    print(f"====rl loop step {rl_loop_step}====")
    
    policy = lambda x: infer(pi_network, x, epsilon)
    data = generate_data_from_environment(policy, env, num_episodes=num_episodes_for_data_generation, max_episode_length=int(max_episode_length), writer=writer, step=rl_loop_step, save_path=dataset_dir / f"step_{rl_loop_step}.txt")
    pi_network = train(data, pi_network, gamma, lr, num_epochs, writer, rl_loop_step, checkpoints_dir / f"step_{rl_loop_step}.pth")

    # max_episode_length = min(top_max_episode_length, max_episode_length * max_episode_length_increase)
    # epsilon = max(epsilon_min, epsilon * epsilon_decay)
    epsilon = max(epsilon_min, epsilon - epsilon_decay )
    lr = max(lr * lr_decay, 0.00001)
    writer.add_scalar("max_episode_length", int(max_episode_length), rl_loop_step)
    writer.add_scalar("epsilon", epsilon, rl_loop_step)
    # num_episodes_for_data_generation = max(50, int(num_episodes_for_data_generation * num_episodes_for_data_generation_decay))
    # epsilon = max(epsilon_min, epsilon * epsilon_decay)

  env.close()

  
