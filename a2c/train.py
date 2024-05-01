import torch
from torch import nn
import random
def train(data, actor, critic, num_epochs, writer, actor_save_path, critic_save_path):
    
    # Define the loss function
    criterion = nn.MSELoss()
    # Define the optimizer
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)

    print("Training...")
    # there should be a dataset, dataloader and what not here
    # Train the network
    

    for epoch in range(num_epochs):
        total_actor_loss = 0
        total_critic_loss = 0
        total_r = 0
        num_actor_updates = 0
        num_critic_updates = 0
        for i, (s, a, r, s_prime) in enumerate(data):
            s = torch.from_numpy(s)

            # Forward pass
            a = actor(s)
            s_and_a = torch.cat((s, a), 0)
            r_pred = critic(s_and_a)
            
            critic_loss = criterion(r_pred, torch.tensor([r], dtype=torch.float32))
            actor_loss = -r_pred.mean()
            scaled_actor_loss = actor_loss
            prob = random.uniform(0,1)
            total_r += r
            if prob > 0.5:

                # optimize the actor
                actor_optimizer.zero_grad()
                for param in critic.parameters():
                    param.requires_grad = False
                

                for param in actor.parameters():
                    param.requires_grad = True
                
                
                scaled_actor_loss.backward(retain_graph=True)

                actor_optimizer.step()
                total_actor_loss += scaled_actor_loss.item()
                num_actor_updates += 1

            else:
                # optimize the critic

                critic_optimizer.zero_grad()

                for param in critic.parameters():
                    param.requires_grad = True

                for param in actor.parameters():
                    param.requires_grad = False

                # print("trace 9")
                critic_loss.backward()
                critic_optimizer.step()
                # print("trace 10")
                
                total_critic_loss += critic_loss.item()
                num_critic_updates += 1
        
        avg_actor_loss = total_actor_loss / (num_actor_updates+1)
        avg_critic_loss = total_critic_loss / (num_critic_updates+1)
        avg_r = total_r / len(data)
        
        writer.add_scalar('training reward', avg_r, epoch)
        writer.add_scalar('training actor loss', avg_actor_loss, epoch)
        writer.add_scalar('training critic loss', avg_critic_loss, epoch)
        

    print('Finished Training')
    print(f"Writing actor to {actor_save_path} and critic to {critic_save_path}")
    torch.save(actor.state_dict(), actor_save_path)
    torch.save(critic.state_dict(), critic_save_path)
    
    return actor, critic
