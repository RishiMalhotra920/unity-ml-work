import torch
from torch import nn
import random
import numpy as np


def train(data, pi_network, v_network, gamma, pi_network_lr, v_network_lr, num_epochs, writer, step, save_path):
    # Define the loss functions
    pi_criterion = nn.MSELoss()
    v_criterion = nn.MSELoss()
    
    # Define the optimizers
    pi_optimizer = torch.optim.Adam(pi_network.parameters(), lr=pi_network_lr)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=v_network_lr)

    print("Training...")

    # shuffle the order of data
    random.shuffle(data) # shuffle whole trajectories, maintaining the order of each trajectory

    for epoch in range(num_epochs):
        total_pi_loss = 0
        total_v_loss = 0
        num_updates = 0
        total_G = 0
        for episode in data:
            last_reward = episode[-1][2]
            if last_reward == -1:
                G = 0
                print('not bootstrapping... G:', G)
            else:
                s_last = torch.from_numpy(episode[-1][3]).float()
                G = v_network(s_last).detach().item()
                print('bootstrapping... G:', G)

            for t in range(len(episode)-1, -1, -1):
                (s, a, r, s_prime) = episode[t]
                s = torch.from_numpy(s).float()

                G = r + gamma * G
                # Compute value for the current state
                v_s = v_network(s)
                
                # Forward pass for the policy network
                (a1_mean, a2_mean), (a1_var, a2_var) = pi_network(s)
                softplus = torch.nn.Softplus()
                a1_var = softplus(a1_var)
                a2_var = softplus(a2_var)

                actual_a1, actual_a2 = torch.tensor(a[0][0]), torch.tensor(a[0][1])
                pi_a_given_s = torch.distributions.Normal(a1_mean, a1_var).log_prob(actual_a1) + torch.distributions.Normal(a2_mean, a2_var).log_prob(actual_a2)
                # Adjusted return with baseline
                adjusted_G = G - v_s.detach()  # detach v_s to avoid computing gradients for it

                # Loss for the policy network
                steps_to_go = len(episode) - 1 - t
                pi_loss = -(gamma**steps_to_go) * adjusted_G * pi_a_given_s

                # Loss for the value network
                v_loss = v_criterion(v_s, torch.tensor([G], dtype=torch.float32))

                # Update policy network
                pi_optimizer.zero_grad()
                pi_loss.backward()
                pi_optimizer.step()

                # Update value network
                v_optimizer.zero_grad()
                v_loss.backward()
                v_optimizer.step()

                total_pi_loss += pi_loss.item()
                total_v_loss += v_loss.item()
                num_updates += 1

            total_G += G

        avg_pi_loss = total_pi_loss / num_updates
        avg_v_loss = total_v_loss / num_updates
        step_epoch = step * num_epochs + epoch
        writer.add_scalar('episode return', total_G / len(data), step_epoch)
        writer.add_scalar('training loss', avg_pi_loss, step_epoch)
        writer.add_scalar('value loss', avg_v_loss, step_epoch)

    print('Finished Training')
    print(f"Writing model to {save_path}")
    torch.save(pi_network.state_dict(), save_path)

    return pi_network
