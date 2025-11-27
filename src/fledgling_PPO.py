import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from fledgling import FlappyBirdMuJoCoEnv
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modified network architectures for continuous actions
class ActorNet(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=128):
        super().__init__()
        
        self.hidden1 = nn.Linear(state_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, s):
        x = F.relu(self.hidden1(s))
        x = F.relu(self.hidden2(x))
        mean = torch.tanh(self.mean(x)) * 1.5  # Scale to action range [-1.5, 1.5]
        return mean, self.log_std.exp()

class ValueNet(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=128):
        super().__init__()
        
        self.hidden1 = nn.Linear(state_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, s):
        x = F.relu(self.hidden1(s))
        x = F.relu(self.hidden2(x))
        value = self.output(x)
        return value
    
def load_checkpoint(checkpoint_path, actor, value_func, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    value_func.load_state_dict(checkpoint['value_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    episode = checkpoint.get('episode', 0)
    rewards = checkpoint.get('rewards', [])
    
    print(f"Loaded checkpoint from episode {episode}")
    return episode, rewards

def moving_average(data, window_size):
    """Calculate the moving average of the data array"""
    if len(data) < window_size:
        return data  # Not enough data points
    
    window = np.ones(window_size) / window_size
    # Use 'valid' mode to avoid edge effects
    ma_data = np.convolve(data, window, mode='valid')
    # Pad the beginning with NaNs to maintain original data length
    return np.concatenate([np.full(window_size-1, np.nan), ma_data])

def main():
    # Hyperparameters
    gamma = 0.995       # discount factor
    kl_coeff = 0.20    # weight coefficient for KL-divergence loss
    vf_coeff = 0.50    # weight coefficient for value loss
    learning_rate = 0.0005
    max_episodes = 3000
    
    # Checkpoint settings
    checkpoint_dir = "checkpoints/"
    checkpoint_frequency = 500  # Save every 100 episodes
    best_reward = -float('inf')
    
    # Create checkpoint directory if it doesn't exist
    import os
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Initialize environment, networks, and optimizer
    env = FlappyBirdMuJoCoEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = ActorNet(state_dim, action_dim).to(device)
    value_func = ValueNet(state_dim).to(device)
    all_params = list(actor.parameters()) + list(value_func.parameters())
    opt = torch.optim.AdamW(all_params, lr=learning_rate)
    
    # Function to select action
    def pick_action_and_logp(s):
        with torch.no_grad():
            s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
            
            # Get mean and std from actor network
            mean, std = actor(s_batch)
            
            # Sample from normal distribution
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1)
            
            # Clamp action to valid range
            action = torch.clamp(action, -1.5, 1.5)
            
            return action.squeeze().cpu().numpy(), mean.squeeze().cpu().numpy(), std.squeeze().cpu().numpy(), log_prob.item()
    
    # Training loop
    reward_records = []
    policy_loss_records = []
    value_loss_records = []
    kl_div_records = []
    
    for i_episode in range(max_episodes):
        # Run episode
        states = []
        actions = []
        means = []
        stds = []
        log_probs = []
        rewards = []
        
        s, _ = env.reset()
        done = False
        
        while not done:
            states.append(s)
            action, mean, std, log_prob = pick_action_and_logp(s)
            s_next, r, term, trunc, _ = env.step(action)
            
            actions.append(action)
            means.append(mean)
            stds.append(std)
            log_probs.append(log_prob)
            rewards.append(r)
            
            s = s_next
            done = term or trunc
        
        # Calculate cumulative rewards
        cum_rewards = np.zeros_like(rewards)
        reward_len = len(rewards)
        for j in reversed(range(reward_len)):
            cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float).to(device)
        means_old = torch.tensor(np.array(means), dtype=torch.float).to(device)
        stds_old = torch.tensor(np.array(stds), dtype=torch.float).to(device)
        log_probs_old = torch.tensor(np.array(log_probs), dtype=torch.float).to(device).unsqueeze(1)
        cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device).unsqueeze(1)
        
        # Training step
        opt.zero_grad()
        
        # Get values and new distributions
        values_new = value_func(states)
        means_new, stds_new = actor(states)
        
        # Advantages
        advantages = cum_rewards - values_new.detach()
        
        # Create distributions
        old_dist = torch.distributions.Normal(means_old, stds_old)
        new_dist = torch.distributions.Normal(means_new, stds_new)
        
        # New log probs
        log_probs_new = new_dist.log_prob(actions).sum(1, keepdim=True)
        
        # Ratio and policy loss
        ratio = torch.exp(log_probs_new - log_probs_old)
        policy_loss = -torch.mean(ratio * advantages)
        
        # KL divergence for continuous actions
        kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_dist, new_dist).sum(1))
        
        # Value loss
        value_loss = F.mse_loss(values_new, cum_rewards)
        
        # Total loss
        total_loss = policy_loss + kl_coeff * kl_div + vf_coeff * value_loss
        
        # Optimize
        total_loss.backward()
        opt.step()
        
        # Record and report
        # Record all metrics
        policy_loss_records.append(policy_loss.item())
        value_loss_records.append(value_loss.item())
        kl_div_records.append(kl_div.item())
        episode_reward = np.sum(rewards)
        reward_records.append(episode_reward)
        print(f"Episode {i_episode}, Total Reward: {episode_reward:.2f}", end="\r")
        
        # # Early stopping if good enough
        # if i_episode > 50 and np.mean(reward_records[-50:]) > 100:  # Adjust threshold as needed
        #     break
                # Save checkpoint periodically
        if (i_episode + 1) % checkpoint_frequency == 0:
            checkpoint = {
                'episode': i_episode,
                'actor_state_dict': actor.state_dict(),
                'value_state_dict': value_func.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'rewards': reward_records,
                'best_reward': best_reward
            }
            torch.save(checkpoint, f"{checkpoint_dir}checkpoint_episode_{i_episode+1}.pt")
            print(f"\nCheckpoint saved at episode {i_episode+1}")
            
        # # Save best model when we achieve better performance
        # if episode_reward > best_reward:
        #     best_reward = episode_reward
        #     best_checkpoint = {
        #         'episode': i_episode,
        #         'actor_state_dict': actor.state_dict(),
        #         'value_state_dict': value_func.state_dict(),
        #         'optimizer_state_dict': opt.state_dict(),
        #         'reward': episode_reward
        #     }
        #     torch.save(best_checkpoint, f"{checkpoint_dir}best_model.pt")

    print("\nTraining complete!")
    


    # Save final model
    final_checkpoint = {
        'episode': max_episodes,
        'actor_state_dict': actor.state_dict(),
        'value_state_dict': value_func.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'rewards': reward_records,
        'best_reward': best_reward
    }
    torch.save(final_checkpoint, f"{checkpoint_dir}final_model.pt")
    print(f"Final model saved to {checkpoint_dir}final_model.pt")
    
    # # Plot learning curve
    # plt.figure(figsize=(10, 5))
    
    # # Plot learning curve
    # plt.figure(figsize=(10, 5))
    # plt.plot(reward_records)
    # plt.title('Learning Curve')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.savefig('fledgling_learning_curve.png')


        # Enhanced plotting with multiple subplots
    plt.figure(figsize=(15, 12))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(reward_records, color='blue', label='Total Reward')
    plt.plot(moving_average(reward_records, 50), color='orange', label='Moving Avg (50)')
    plt.title('Learning Curve (Rewards)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    # Plot policy loss
    plt.subplot(2, 2, 2)
    plt.plot(policy_loss_records, color='red')
    plt.plot(moving_average(policy_loss_records, 50), color='purple', label='Moving Avg (50)')
    plt.title('Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot value loss
    plt.subplot(2, 2, 3)
    plt.plot(value_loss_records, color='green')
    plt.title('Value Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot KL divergence
    plt.subplot(2, 2, 4)
    plt.plot(kl_div_records, color='purple')
    plt.title('KL Divergence')
    plt.xlabel('Episode')
    plt.ylabel('KL Div')
    
    plt.tight_layout()
    plt.savefig('fledgling_training_metrics.png')
    plt.show()
    
    # # Test and visualize trained policy
    print("Testing policy...")
    s, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _, _, _ = pick_action_and_logp(s)
        s, r, term, trunc, _ = env.step(action)
        env.render()
        total_reward += r
        done = term or trunc
        # time.sleep(0.01)  # Slow down visualization
    
    print(f"Test episode complete, Total Reward: {total_reward:.2f}")
    env.close()


def test_checkpoint(checkpoint_path, episodes=5, render=True):
    """
    Load a checkpoint and test it for multiple episodes.
    
    Args:
        checkpoint_path: Path to the saved checkpoint
        episodes: Number of test episodes to run
        render: Whether to render the environment
    """
    # Setup environment
    env = FlappyBirdMuJoCoEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create models
    actor = ActorNet(state_dim, action_dim).to(device)
    value_func = ValueNet(state_dim).to(device)
    
    # Load checkpoint
    episode, rewards = load_checkpoint(checkpoint_path, actor, value_func)
    print(f"Loaded model from episode {episode}")
    
    # Function to select action
    def pick_action(s):
        with torch.no_grad():
            s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
            mean, std = actor(s_batch)
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            action = torch.clamp(action, -1.5, 1.5)
            return action.squeeze().cpu().numpy()
    
    # Test for multiple episodes
    test_rewards = []
    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action = pick_action(s)
            s, r, term, trunc, _ = env.step(action)
            if render:
                env.render()
                # time.sleep(0.01)  # Slow down visualization
            
            total_reward += r
            step_count += 1
            done = term or trunc
        
        test_rewards.append(total_reward)
        print(f"Test episode {ep+1}/{episodes}, Reward: {total_reward:.2f}, Steps: {step_count}")
    
    env.close()
    
    print(f"\nAverage test reward: {np.mean(test_rewards):.2f}")
    return test_rewards


if __name__ == "__main__":
    main()

    # checkpoint_path = "checkpoints_fantastic/checkpoint_episode_2000.pt"  # or any other checkpoint
    # checkpoint_path = "checkpoints_fantastic/final_model.pt"  # or any other checkpoint

    # test_rewards = test_checkpoint(checkpoint_path, episodes=5, render=True)