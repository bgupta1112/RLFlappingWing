import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fledgling import FlappyBirdMuJoCoEnv
import time
import random
import matplotlib.pyplot as plt
import os

# CRITICAL OPTIMIZATION: Switch to float32 for 2-3x speedup
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for saving models and plots
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Optimized Replay Buffer using pre-allocated NumPy arrays
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idxs]).to(device),
            torch.FloatTensor(self.actions[idxs]).to(device),
            torch.FloatTensor(self.rewards[idxs]).reshape(-1, 1).to(device),
            torch.FloatTensor(self.next_states[idxs]).to(device),
            torch.FloatTensor(self.dones[idxs]).reshape(-1, 1).to(device)
        )

    def __len__(self):
        return self.size

# State Normalization for better stability and faster convergence
class RunningNormalizer:
    def __init__(self, epsilon=1e-5):
        self.mean = None
        self.std = None
        self.epsilon = epsilon
        self.count = 0
        
    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.mean is None:
            return x  # Pass through if not initialized
        return (x - self.mean) / (self.std + self.epsilon)
    
    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        batch_mean = np.mean(x, axis=0)
        batch_std = np.std(x, axis=0)
        batch_count = x.shape[0]
        
        if self.mean is None:
            self.mean = batch_mean
            self.std = batch_std
            self.count = batch_count
        else:
            total_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * batch_count / total_count
            
            # Update variance efficiently
            m_a = self.std ** 2 * self.count
            m_b = batch_std ** 2 * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
            self.std = np.sqrt(M2 / total_count)
            self.count = total_count

# Neural Networks with smaller hidden dimensions
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128): 
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return 1.5*torch.tanh(self.out(x))

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# Orthogonal initialization for faster convergence
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=0.8)
        nn.init.constant_(m.bias.data, 0)

# TD3 Setup
state_dim = 6
action_dim = 2
hidden_dim = 128 
lr = 0.0003
gamma = 0.995
tau = 0.005

pi_model = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
pi_target = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
pi_model.apply(init_weights)
pi_target.load_state_dict(pi_model.state_dict())
pi_target.requires_grad_(False)

q1_model = QNet(state_dim, action_dim, hidden_dim).to(device)
q2_model = QNet(state_dim, action_dim, hidden_dim).to(device)
q1_model.apply(init_weights)
q2_model.apply(init_weights)
q1_target = QNet(state_dim, action_dim, hidden_dim).to(device)
q2_target = QNet(state_dim, action_dim, hidden_dim).to(device)
q1_target.load_state_dict(q1_model.state_dict())
q2_target.load_state_dict(q2_model.state_dict())
q1_target.requires_grad_(False)
q2_target.requires_grad_(False)

# Add weight decay for better regularization
pi_optimizer = torch.optim.Adam(pi_model.parameters(), lr=lr, weight_decay=1e-5)
q1_optimizer = torch.optim.Adam(q1_model.parameters(), lr=lr, weight_decay=1e-5)
q2_optimizer = torch.optim.Adam(q2_model.parameters(), lr=lr, weight_decay=1e-5)

def select_action(state, normalizer=None, noise_scale=0.1):
    with torch.no_grad():
        if normalizer is not None:
            state = normalizer(state)
        state = torch.FloatTensor(np.array([state])).to(device)
        action = pi_model(state).cpu().numpy()[0]
        action += noise_scale * np.random.randn(action_dim)
        return np.clip(action, -1.5, 1.5)

# Optimized target network updates with in-place operations
def update_td3_targets():
    with torch.no_grad():
        for target, source in zip(q1_target.parameters(), q1_model.parameters()):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
        for target, source in zip(q2_target.parameters(), q2_model.parameters()):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
        for target, source in zip(pi_target.parameters(), pi_model.parameters()):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)

def update_td3(states, actions, rewards, next_states, dones, policy_delay=2, step=0):
    with torch.no_grad():
        next_actions = pi_target(next_states)
        noise = (0.2 * torch.randn_like(next_actions)).clamp(-0.5, 0.5)
        next_actions = (next_actions + noise).clamp(-1.5, 1.5)

        target_q1 = q1_target(next_states, next_actions)
        target_q2 = q2_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target = rewards + gamma * (1 - dones) * target_q

    q1 = q1_model(states, actions)
    q2 = q2_model(states, actions)
    q1_loss = F.mse_loss(q1, target)
    q2_loss = F.mse_loss(q2, target)

    q1_optimizer.zero_grad()
    q1_loss.backward()
    q1_optimizer.step()

    q2_optimizer.zero_grad()
    q2_loss.backward()
    q2_optimizer.step()

    pi_loss = torch.tensor(0.0)
    if step % policy_delay == 0:
        pi_actions = pi_model(states)
        pi_loss = -q1_model(states, pi_actions).mean()
        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()
        update_td3_targets()

    return q1_loss.item(), q2_loss.item(), pi_loss.item()

# Functions for saving and loading models
def save_checkpoint(episode, total_steps):
    checkpoint = {
        'pi_model': pi_model.state_dict(),
        'pi_target': pi_target.state_dict(),
        'q1_model': q1_model.state_dict(),
        'q2_model': q2_model.state_dict(),
        'q1_target': q1_target.state_dict(),
        'q2_target': q2_target.state_dict(),
        'pi_optimizer': pi_optimizer.state_dict(),
        'q1_optimizer': q1_optimizer.state_dict(),
        'q2_optimizer': q2_optimizer.state_dict(),
        'episode': episode,
        'total_steps': total_steps
    }
    torch.save(checkpoint, f"models/td3_checkpoint_ep{episode}.pt")
    print(f"Checkpoint saved at episode {episode}")

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    pi_model.load_state_dict(checkpoint['pi_model'])
    pi_target.load_state_dict(checkpoint['pi_target'])
    q1_model.load_state_dict(checkpoint['q1_model'])
    q2_model.load_state_dict(checkpoint['q2_model'])
    q1_target.load_state_dict(checkpoint['q1_target'])
    q2_target.load_state_dict(checkpoint['q2_target'])
    pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
    q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
    q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
    return checkpoint['episode'], checkpoint['total_steps']

def test_checkpoint(checkpoint_path, num_episodes=5, render=True):
    """Test a saved agent from checkpoint"""
    load_checkpoint(checkpoint_path)
    
    env = FlappyBirdMuJoCoEnv()
    normalizer = RunningNormalizer()
    test_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        normalizer.update(np.array([state]))
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Use policy without exploration noise
            with torch.no_grad():
                state_normalized = normalizer(state)
                state_tensor = torch.FloatTensor(np.array([state_normalized])).to(device)
                action = pi_model(state_tensor).cpu().numpy()[0]
            
            next_state, reward, done, _, _ = env.step(action)
            normalizer.update(np.array([next_state]))
            
            episode_reward += reward
            state = next_state
            step += 1
            if render:
                env.render()
            
        test_rewards.append(episode_reward)
        print(f"Test Episode {ep+1}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(test_rewards)
    print(f"Average Test Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward, test_rewards

def moving_average(data, window_size):
    """Calculate the moving average of the data array"""
    if len(data) < window_size:
        return data  # Not enough data points
    
    window = np.ones(window_size) / window_size
    # Use 'valid' mode to avoid edge effects
    ma_data = np.convolve(data, window, mode='valid')
    # Pad the beginning with NaNs to maintain original data length
    return np.concatenate([np.full(window_size-1, np.nan), ma_data])
# Training with optimized parameters
def train(max_episodes=300, max_steps=1000, start_steps=5000, batch_size=512, 
          save_freq=100, eval_freq=100, render=False):
    env = FlappyBirdMuJoCoEnv()
    buffer = ReplayBuffer(capacity=1_000_000, state_dim=state_dim, action_dim=action_dim)
    normalizer = RunningNormalizer()
    
    total_steps = 0
    episode_rewards = []
    q1_losses, q2_losses, pi_losses = [], [], []
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        normalizer.update(np.array([state]))
        
        episode_reward = 0
        episode_q1_loss, episode_q2_loss, episode_pi_loss = 0, 0, 0
        updates = 0
        
        for step in range(max_steps):
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = select_action(state, normalizer)

            next_state, reward, done, _, _ = env.step(action)
            normalizer.update(np.array([next_state]))
            
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            total_steps += 1

            # More efficient update schedule
            if len(buffer) > batch_size: #and total_steps % 2 == 0:  # Update every 2 steps
                batch = buffer.sample(batch_size)
                q1_loss, q2_loss, pi_loss = update_td3(*batch, step=total_steps)
                episode_q1_loss += q1_loss
                episode_q2_loss += q2_loss
                episode_pi_loss += pi_loss
                updates += 1

            if done:
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        if updates > 0:
            avg_q1_loss = episode_q1_loss / updates
            avg_q2_loss = episode_q2_loss / updates
            avg_pi_loss = episode_pi_loss / updates
            q1_losses.append(avg_q1_loss)
            q2_losses.append(avg_q2_loss)
            pi_losses.append(avg_pi_loss)
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step+1}, "
                  f"Q1 Loss = {avg_q1_loss:.4f}, Q2 Loss = {avg_q2_loss:.4f}, Pi Loss = {avg_pi_loss:.4f}")
        else:
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step+1}, No updates yet")
        
        # Save checkpoint periodically
        if (episode + 1) % save_freq == 0:
            save_checkpoint(episode + 1, total_steps)
        
        # Evaluate periodically
        if (episode + 1) % eval_freq == 0:
            plot_training_progress(episode_rewards, q1_losses, q2_losses, pi_losses)
        
        # # Early stopping if we're doing well
        # if episode > 50 and len(episode_rewards) > 10:
        #     recent_avg = np.mean(episode_rewards[-10:])
        #     if recent_avg > 500:  # Adjust threshold based on your environment
        #         print(f"Early stopping at episode {episode+1} with avg reward {recent_avg:.2f}")
        #         save_checkpoint(episode + 1, total_steps)
        #         break
    
    # Save final model
    save_checkpoint(max_episodes, total_steps)
    
    # Plot final training progress
    plot_training_progress(episode_rewards, q1_losses, q2_losses, pi_losses, final=True)
    
    return episode_rewards, q1_losses, q2_losses, pi_losses

def plot_training_progress(rewards, q1_losses, q2_losses, pi_losses, final=False):
    """Plot training progress"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards, color='blue', label='Episode Reward')
    plt.plot(moving_average(rewards, 5), color='orange', label='Moving Avg (10)', linestyle='--')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot Q1 losses
    plt.subplot(2, 2, 2)
    plt.plot(q1_losses)
    plt.title('Q1 Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot Q2 losses
    plt.subplot(2, 2, 3)
    plt.plot(q2_losses)
    plt.title('Q2 Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot policy losses
    plt.subplot(2, 2, 4)
    plt.plot(pi_losses)
    plt.title('Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    
    # Save the plot
    suffix = "final" if final else f"ep{len(rewards)}"
    plt.savefig(f"plots/training_progress_{suffix}.png")
    plt.close()

def main():
    # Train the agent
    train(max_episodes=300, max_steps=3000, render=False)
    
    # Uncomment the line below to test a saved checkpoint instead of training
    # test_checkpoint("models/td3_checkpoint_ep200.pt", num_episodes=5, render=True)

if __name__ == "__main__":
    main()