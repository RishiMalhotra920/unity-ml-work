import gym
import gym_unity

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env


from gym_unity.envs.unity_env import UnityEnv

def main():
    env = UnityEnv("./envs/GridWorld", 0, use_visual=True, uint8_visual=True)
    logger.configure('./logs') # Çhange to log in a different directory

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("checkers")
    print("logging done.")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # env.render("human")


if __name__ == '__main__':
    main()