# Balancing 3d Ball

![Cover](assets/Cover.png)

[Demo Video](assets/3DBallBalancingDemo.mov)

## About

The goal for the agent is to balance the ball on his head.

The observation space is an 8 dimensional vector in continuous space: [-1, 1]

- The agent's rotation in the z plane. 1 dimension
- The agent's rotation in the x plane. 1 dimension
- The ball's relative position (to the agent's head). 3 dimensions
- The ball's velocity. 3 dimensions

Rotations in Unity are expressed as [quaternions](https://en.wikipedia.org/wiki/Quaternion). The z and x components represent the z and x components in the equation below: q=w+xi+yj+zk

The action space is 2 dimensional.

- The force to apply to the agent in the z plane
- The force to apply to the agent in the x plane.

The forces manipulate the agent's rotation in the respective planes.

## Method

- After a lot of experimentation, I was able to make the agent balance the ball on his head for 40 timesteps.
- I used the A2C algorithm. The policy was an MLP with two hidden layers with 64 neurons, a linear layer that produced the mean and a linear layer followed by softplus to produce the variance of a gaussian distribution. When collecting rollouts, an action was sampled from this gaussian distribution
- An entropy bonus ensured high standard deviation early on - leading to exploration. The proportion of entropy bonus to reward decreased over time, which lowered standard deviation and hence exploration.
- I collected 10 rollouts per update and batched these rollouts. Each rollout used an n_step of 15 timesteps. This meant that the return was `G = r_0 + gamma * r_1 + ... * gamma**14 * r_15 + V(s_15)`.
- I used only one agent to collect all 10 rollouts. I did not have the time to set up multiple agents with the unity gym environment.
- I found gradient clipping and constraining every action to [-1, 1] helpful.

## How to improve the solution

- I ran out of time but a solution with less bootstrapping, more agents collecting trajectories, more num_rollouts_per_update and high entropy bonus would have improved the model's performance.

## To run

- Install unity and ml agents with this [tutorial](https://github.com/Unity-Technologies/ml-agents/blob/release_21_docs/docs/Installation.md#advanced-local-installation-for-development-1)
- Follow this [getting started guide](https://github.com/Unity-Technologies/ml-agents/blob/release_21_docs/docs/Getting-Started.md)
- clone this repo and activate the mlagents virtual environment you created in the tutorial
- run `python training_with_gym.py --run-id=YOUR_RUN_ID` to train the model. You need to have the unity editor opened and need to hit the play button.
- run `python inference_with_gym.py` to run a trained model

## References

- The code base architecture and algorithm design was inspired by [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/)
- [A3C](https://arxiv.org/pdf/1602.01783). The synchronous version of the A3C algorithm is called A2C. I also referenced Section 9 of the appendix to design the hyperparameters.
