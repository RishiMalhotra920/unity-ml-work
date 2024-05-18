import gym
import mlagents_envs
# from baselines import deepq
# from baselines import logger
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from gym_trainer.ppo import PPO

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from torch.utils.tensorboard import SummaryWriter
import sys
import torch
import numpy as np
import random
from pathlib import Path
from env import Environment
import traceback

def main(run_id):
    # unity_env = UnityEnvironment("/Users/rishimalhotra/projects/checker2_one_agent.app")
    # unity_env = UnityEnvironment()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    writer = SummaryWriter(f"runs/{run_id}")
    print('run id: ', run_id)
    print('Waiting for you to hit play on unity')

    # load_policy_network_checkpoint_path = "good_checkpoints/increasing_n_steps_to_15/step_25000/policy_network.pth"
    # load_value_network_checkpoint_path = "good_checkpoints/increasing_n_steps_to_15/step_25000/value_network.pth"
    load_policy_network_checkpoint_path = None
    load_value_network_checkpoint_path = None

    checkpoint_path = Path(f"checkpoints/PPO/{run_id}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    env = Environment(0)
    try:
      model = PPO(env,
                  writer,
                  n_steps=150,
                  load_policy_network_checkpoint_path=load_policy_network_checkpoint_path,
                  load_value_network_checkpoint_path=load_value_network_checkpoint_path)

      model.learn(total_timesteps=50000,
                  policy_network_lr=7e-4,
                  value_network_lr=7e-4,
                  sigma_lr=1e-4,
                  value_coef=1,
                  ent_coef=1e-2,
                  checkpoint_path=checkpoint_path)
    except Exception as e:
      print(e)
      print(traceback.format_exc())
      env.close()
                  
    # model = MyA2C(vec_env,
    #               writer,
    #               n_steps=15,
    #               num_rollouts_per_update=10,
    #               load_policy_network_checkpoint_path=load_policy_network_checkpoint_path,
    #               load_value_network_checkpoint_path=load_value_network_checkpoint_path)
    # model.learn(total_timesteps=50000,
    #             policy_network_lr=7e-4,
    #             value_network_lr=7e-4,
    #             sigma_lr=1e-4,
    #             ent_coef=1e-2,
    #             checkpoint_path=checkpoint_path)
    # act.save("unity_model.pkl")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--run-id="):
                run_id = arg[9:]
                main(run_id)
