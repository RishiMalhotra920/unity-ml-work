
from generate_data_from_environment import generate_data_from_environment
from train import train
from model import pi_network, v_network, load_pi_network
from pathlib import Path
from inference import infer
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

    gamma = 0.99
    num_steps_in_rl_loop = 500
    num_epochs = 1
    num_episodes_for_data_generation_per_agent = 1
    # num_episode_per_agent = 1
    # num_episodes_for_data_generation_decay = 0.8
    pi_network_lr = 1e-8
    v_network_lr = 1e-3
    lr_min = 1e-10
    epsilon_min = 0.1
    epsilon = 1.0
    epsilon_decay = 0.99
    max_episode_length = 1000
    # max_episode_length_increase = 1.01
    top_max_episode_length = 100
    lr_decay = 0.99
    num_agents = 12

    writer = SummaryWriter(f"runs/{run_id}")

    # load good_checkpoints/step_32.pth
    # pi_network = load_pi_network("good_checkpoints/step_32.pth")

    # link it all to the run id. best checkpoints should be stored under run id

    # epsilon annealing formula
    # epsilon = 1 - 0.9 * (t/1m)
    # note you have to change this in the unity env as well.
    assert num_agents == 12
    for rl_loop_step in range(num_steps_in_rl_loop):
        print(f"====rl loop step {rl_loop_step}====")

        def policy(x): return infer(pi_network, x, epsilon)
        data = generate_data_from_environment(policy, env, num_agents, num_episodes_per_agent=num_episodes_for_data_generation_per_agent, max_episode_length=int(
            max_episode_length), writer=writer, step=rl_loop_step, save_path=dataset_dir / f"step_{rl_loop_step}.txt")
        pi_network = train(data, pi_network, v_network, gamma, pi_network_lr, v_network_lr,
                           num_epochs, writer, rl_loop_step, checkpoints_dir / f"step_{rl_loop_step}.pth")

        # max_episode_length = min(top_max_episode_length, max_episode_length * max_episode_length_increase)
        # epsilon = max(epsilon_min, epsilon * epsilon_decay)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        # max_episode_length = min(top_max_episode_length, max_episode_length * max_episode_length_increase)
        pi_network_lr = max(pi_network_lr * lr_decay, lr_min)
        v_network_lr = max(v_network_lr * lr_decay, lr_min)
        writer.add_scalar("pi_network_lr", pi_network_lr, rl_loop_step)
        writer.add_scalar("v_network_lr", v_network_lr, rl_loop_step)
        writer.add_scalar("max_episode_length", int(
            max_episode_length), rl_loop_step)
        writer.add_scalar("epsilon", epsilon, rl_loop_step)
        # num_episodes_for_data_generation = max(50, int(num_episodes_for_data_generation * num_episodes_for_data_generation_decay))
        # epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()
