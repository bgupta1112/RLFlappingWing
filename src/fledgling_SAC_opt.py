import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from fledgling import FlappyBirdMuJoCoEnv
import time
import os
try:
    import imageio
except ImportError:
    imageio = None

# CRITICAL OPTIMIZATION: Switch to float32 for ~2x speedup
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimized Policy Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=128): 
        super().__init__()

        self.hidden1 = nn.Linear(state_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Action range for output scaling
        self.action_scale = torch.tensor(1.5).to(device)
        self.action_bias = torch.tensor(0.0).to(device)

    def forward(self, s):
        x = F.relu(self.hidden1(s))
        x = F.relu(self.hidden2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, s):
        mean, log_std = self.forward(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Sample using reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Compute log probability, scaling, etc.
        log_prob = normal.log_prob(x_t)
        
        # Apply correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Scale and shift actions
        scaled_action = action * self.action_scale + self.action_bias
        
        return scaled_action, log_prob

# Optimized Q-Network
class QNet(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=128):  # Smaller hidden dim
        super().__init__()

        self.hidden1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        q = self.output(x)
        return q

# State normalization for stability
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
            
            # Update variance (using parallel algorithm)
            m_a = self.std ** 2 * self.count
            m_b = batch_std ** 2 * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
            self.std = np.sqrt(M2 / total_count)
            self.count = total_count

# Optimized Replay Buffer - faster sampling using numpy arrays
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

# Initialize networks with orthogonal initialization for faster convergence
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=0.8)
        nn.init.constant_(m.bias.data, 0)

# Create networks
state_dim = 6
action_dim = 2
hidden_dim = 128

pi_model = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
q_origin_model1 = QNet(state_dim, action_dim, hidden_dim).to(device)
q_origin_model2 = QNet(state_dim, action_dim, hidden_dim).to(device)
q_target_model1 = QNet(state_dim, action_dim, hidden_dim).to(device)
q_target_model2 = QNet(state_dim, action_dim, hidden_dim).to(device)

# Apply weight initialization
pi_model.apply(init_weights)
q_origin_model1.apply(init_weights)
q_origin_model2.apply(init_weights)
q_target_model1.apply(init_weights)
q_target_model2.apply(init_weights)

# Initialize target networks with same weights
for target_param, param in zip(q_target_model1.parameters(), q_origin_model1.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(q_target_model2.parameters(), q_origin_model2.parameters()):
    target_param.data.copy_(param.data)

# Disable gradient for target networks
q_target_model1.requires_grad_(False)
q_target_model2.requires_grad_(False)

# Hyperparameters
gamma = 0.99  # discount factor
tau = 0.001   # target network update rate
alpha = 0.4   # temperature parameter for entropy
lr = 0.0001   # learning rate

# Optimizers with weight decay for better regularization
pi_optimizer = torch.optim.Adam(pi_model.parameters(), lr=lr, weight_decay=1e-5)
q1_optimizer = torch.optim.Adam(q_origin_model1.parameters(), lr=lr, weight_decay=1e-5)
q2_optimizer = torch.optim.Adam(q_origin_model2.parameters(), lr=lr, weight_decay=1e-5)

# Function to sample action for environment interaction
def select_action(state, normalizer=None):
    with torch.no_grad():
        if normalizer is not None:
            state = normalizer(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _ = pi_model.sample(state)
        return action.cpu().numpy()[0]

# Function to update policy
def update_policy(states):
    # Get actions and log probs from current policy
    actions, log_probs = pi_model.sample(states)
    
    # Calculate Q-values for these actions
    q1 = q_origin_model1(states, actions)
    q2 = q_origin_model2(states, actions)
    q = torch.min(q1, q2)
    
    # Policy loss: maximize Q-value while maintaining entropy
    policy_loss = (alpha * log_probs - q).mean()
    
    # Update policy
    pi_optimizer.zero_grad()
    policy_loss.backward()
    pi_optimizer.step()
    
    return policy_loss.item()

# Function to update Q-networks
def update_q_networks(states, actions, rewards, next_states, dones):
    with torch.no_grad():
        # Get next actions and log probs from current policy
        next_actions, next_log_probs = pi_model.sample(next_states)
        
        # Get Q-values from target networks
        next_q1 = q_target_model1(next_states, next_actions)
        next_q2 = q_target_model2(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2)
        
        # Calculate target with entropy term
        target_q = rewards + gamma * (1 - dones) * (next_q - alpha * next_log_probs)
    
    # Current Q-values
    curr_q1 = q_origin_model1(states, actions)
    curr_q2 = q_origin_model2(states, actions)
    
    # Calculate losses
    q1_loss = F.mse_loss(curr_q1, target_q)
    q2_loss = F.mse_loss(curr_q2, target_q)
    
    # Update Q1
    q1_optimizer.zero_grad()
    q1_loss.backward()
    q1_optimizer.step()
    
    # Update Q2
    q2_optimizer.zero_grad()
    q2_loss.backward()
    q2_optimizer.step()
    
    return q1_loss.item(), q2_loss.item()

# Optimized function to update target networks with in-place operations
def update_targets():
    with torch.no_grad():
        for target_param, param in zip(q_target_model1.parameters(), q_origin_model1.parameters()):
            target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)
        for target_param, param in zip(q_target_model2.parameters(), q_origin_model2.parameters()):
            target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)

def moving_average(data, window_size):
    """Calculate the moving average of the data array"""
    if len(data) < window_size:
        return data  # Not enough data points
    
    window = np.ones(window_size) / window_size
    # Use 'valid' mode to avoid edge effects
    ma_data = np.convolve(data, window, mode='valid')
    # Pad the beginning with NaNs to maintain original data length
    return np.concatenate([np.full(window_size-1, np.nan), ma_data])

def train():
    # Environment and buffer setup
    env = FlappyBirdMuJoCoEnv()
    buffer = ReplayBuffer(capacity=1000000, state_dim=state_dim, action_dim=action_dim)
    normalizer = RunningNormalizer()
    
    # Training hyperparameters
    batch_size = 512  # Increased for better GPU utilization
    start_steps = 5000  # Reduced random steps (was 10000)
    update_after = 1000  # Start updating earlier
    update_every = 20  # Update less frequently but more efficiently
    max_episodes = 300
    max_steps_per_episode = 3500  # Reduced from 3000
    
    # Logging
    rewards_history = []
    avg_q_values = []
    policy_losses = []
    q_losses = []
    episode_steps = []
    
    # Training loop
    total_steps = 0
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False
        
        # Update normalizer with initial state
        normalizer.update(np.array([state]))
        
        while not done and episode_step < max_steps_per_episode:
            # Determine action
            if total_steps < start_steps:
                action = env.action_space.sample()  # Random action
            else:
                action = select_action(state, normalizer)
                action = np.clip(action, -1.5, 1.5)
            
            # Take step in environment
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            # Update normalizer with new state
            normalizer.update(np.array([next_state]))
            
            # Store transition
            buffer.push(state, action, reward, next_state, float(done))
            
            # Update state
            state = next_state
            episode_step += 1
            total_steps += 1
            
            # Update networks - more efficient update schedule
            if total_steps >= update_after and total_steps % update_every == 0:
                # Do a fixed number of updates instead of update_every
                for _ in range(update_every):  # Just 4 updates per cycle
                    if buffer.size >= batch_size:
                        batch = buffer.sample(batch_size)
                        q1_loss, q2_loss = update_q_networks(*batch)
                        
                        # # Update policy and targets less frequently 
                        # if _ % 2 == 0:
                        policy_loss = update_policy(batch[0])
                        update_targets()
                        
                        # Log losses only when updating policy
                        q_losses.append((q1_loss + q2_loss) / 2)
                        policy_losses.append(policy_loss)
            
            # Don't render during training - major performance boost
            # Only uncomment for visualization after training is complete
            # if episode > 90:  
            #     env.render()
        
        # Log episode info
        rewards_history.append(episode_reward)
        episode_steps.append(episode_step)
        
        # Print progress
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(rewards_history[-5:])
            print(f"Episode {episode+1}/{max_episodes}, " 
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Steps: {episode_step}, "
                  f"Total Steps: {total_steps}"
                  f"Policy Loss: {policy_losses[-1]:.4f}, "
                  f"Q Loss: {q_losses[-1]:.4f}")
        
        # # Early stopping if we're converging
        # if episode > 50 and len(rewards_history) > 10:
        #     recent_avg = np.mean(rewards_history[-10:])
        #     if recent_avg > 500:  # Adjust threshold based on your environment
        #         print(f"Early stopping at episode {episode+1} with avg reward {recent_avg:.2f}")
        #         break
    
        # Save model
        if (episode+1) % 100 == 0 or episode == max_episodes - 1:
            save_checkpoint(episode+1, total_steps, normalizer)
    # torch.save(pi_model.state_dict(), f"flappy_sac_policy_{episode}.pt")

    # Plot training progress
    plt.figure(figsize=(12, 8))
    window_size = 5
    smoothed_rewards = moving_average(rewards_history, window_size)    
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history, 'b-', alpha=0.4, label='Raw Rewards')
    plt.plot(smoothed_rewards, 'r-', linewidth=2, label=f'{window_size}-Ep Moving Avg')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(episode_steps)
    plt.title('Episode Duration')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    if q_losses:
        plt.subplot(2, 2, 3)
        plt.plot(q_losses)
        plt.title('Q-Network Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
    
    if policy_losses:
        plt.subplot(2, 2, 4)
        plt.plot(policy_losses)
        plt.title('Policy Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    os.makedirs('plots_SAC', exist_ok=True)
    plt.savefig('plots_SAC/flappy_sac_training.png')
    print("Training plot saved to plots_SAC/flappy_sac_training.png")
    plt.show()
    
    return pi_model

def save_checkpoint(episode, total_steps, normalizer=None):
    """Save all model weights and training state to resume training later"""
    checkpoint = {
        'pi_model': pi_model.state_dict(),
        'q_origin_model1': q_origin_model1.state_dict(),
        'q_origin_model2': q_origin_model2.state_dict(),
        'q_target_model1': q_target_model1.state_dict(),
        'q_target_model2': q_target_model2.state_dict(),
        'pi_optimizer': pi_optimizer.state_dict(),
        'q1_optimizer': q1_optimizer.state_dict(),
        'q2_optimizer': q2_optimizer.state_dict(),
        'episode': episode,
        'total_steps': total_steps,
        'alpha': alpha
    }
    
    # Save normalizer state if provided
    if normalizer is not None:
        checkpoint['normalizer_mean'] = normalizer.mean
        checkpoint['normalizer_std'] = normalizer.std
        checkpoint['normalizer_count'] = normalizer.count
        
    os.makedirs("models_SAC", exist_ok=True)
    torch.save(checkpoint, f"models_SAC/sac_checkpoint_ep{episode}.pt")
    print(f"Checkpoint saved at episode {episode}")

def load_checkpoint(checkpoint_path):
    """Load a saved checkpoint to resume training or for testing"""
    # Use weights_only=False to allow loading numpy arrays in checkpoints
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Load model weights
    pi_model.load_state_dict(checkpoint['pi_model'])
    q_origin_model1.load_state_dict(checkpoint['q_origin_model1'])
    q_origin_model2.load_state_dict(checkpoint['q_origin_model2'])
    q_target_model1.load_state_dict(checkpoint['q_target_model1'])
    q_target_model2.load_state_dict(checkpoint['q_target_model2'])
    
    # Load optimizer states
    pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
    q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
    q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
    
    # Load normalizer state if available
    normalizer = RunningNormalizer()
    if 'normalizer_mean' in checkpoint:
        normalizer.mean = checkpoint['normalizer_mean']
        normalizer.std = checkpoint['normalizer_std']
        normalizer.count = checkpoint['normalizer_count']
    
    # Load alpha (temperature parameter) if available
    global alpha
    if 'alpha' in checkpoint:
        alpha = checkpoint['alpha']
        
    print(f"Loaded checkpoint from episode {checkpoint['episode']}")
    return checkpoint['episode'], checkpoint['total_steps'], normalizer

def record_video_episode(checkpoint_path=None, output_path="videos", episode_num=1, fps=30):
    """Record a single episode as a video using mujoco viewer"""
    env = FlappyBirdMuJoCoEnv()
    normalizer = None
    frames = []
    
    # Load from checkpoint if provided
    if checkpoint_path:
        _, _, normalizer = load_checkpoint(checkpoint_path)
    
    state, _ = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    os.makedirs(output_path, exist_ok=True)
    
    while not done and steps < 3500:  # max_steps_per_episode
        action = select_action(state, normalizer)
        next_state, reward, done, _, _ = env.step(action)
        
        if normalizer:
            normalizer.update(np.array([next_state]))
        
        # Try to capture frame from MuJoCo renderer
        try:
            if hasattr(env, 'renderer'):
                frame = env.renderer.render()
                if frame is not None:
                    frames.append(frame)
        except Exception as e:
            pass  # Silently continue if rendering fails
        
        episode_reward += reward
        state = next_state
        steps += 1
    
    # Save video if we captured frames and imageio is available
    if frames and imageio is not None:
        video_path = os.path.join(output_path, f"episode_{episode_num:04d}.mp4")
        try:
            imageio.mimsave(video_path, frames, fps=fps)
            print(f"Video saved: {video_path}")
        except Exception as e:
            print(f"Failed to save video: {e}")
    else:
        print(f"No frames captured (rendering may not be available). Install imageio for video support: pip install imageio")
    
    env.close()
    return episode_reward, steps

def test(checkpoint_path=None, render=True, episodes=10, plot_rewards=True, save_video=False):
    """Test a model loaded from checkpoint"""
    env = FlappyBirdMuJoCoEnv()
    normalizer = None
    rewards = []
    steps_list = []
    
    # Load from checkpoint if provided
    if checkpoint_path:
        _, _, normalizer = load_checkpoint(checkpoint_path)
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        # Record video for this episode if requested
        if save_video and episode == 0:
            print(f"Recording video for test episode 1...")
            record_video_episode(checkpoint_path, "videos", episode_num=1)
        
        while not done:
            # Use the loaded normalizer from checkpoint
            action = select_action(state, normalizer)
            next_state, reward, done, _, _ = env.step(action)
            
            # Update normalizer with new observation
            if normalizer:
                normalizer.update(np.array([next_state]))
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if render:
                env.render()
        
        print(f"Test Episode {episode+1}, Reward: {episode_reward:.2f}, Steps: {steps}")
        rewards.append(episode_reward)
        steps_list.append(steps)
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Average Test Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Plot rewards if requested
    if plot_rewards:
        plt.figure(figsize=(12, 6))
        window_size = 5
        smoothed_rewards = moving_average(rewards, window_size)
        plt.plot(rewards, 'b-', alpha=0.4, label='Raw Rewards')
        plt.plot(smoothed_rewards, 'r-', linewidth=2, label=f'{window_size}-Ep Moving Avg')
        plt.title('Test Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot individual episode rewards
        # plt.subplot(1, 2, 1)
        # episodes_x = np.arange(1, episodes+1)
        # plt.plot(episodes_x, rewards, 'bo-', label='Episode Reward')
        # # plt.axhline(y=mean_reward, color='r', linestyle='--', 
        # #            label=f'Mean: {mean_reward:.2f}')
        # plt.title('Test Episode Rewards')
        # plt.xlabel('Episode')
        # plt.ylabel('Total Reward')
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        
        # Plot steps per episode
        # plt.subplot(1, 2, 2)
        # plt.plot(episodes_x, steps_list, 'go-', label='Episode Length')
        # plt.axhline(y=np.mean(steps_list), color='r', linestyle='--', 
        #            label=f'Mean: {np.mean(steps_list):.0f} steps')
        # plt.title('Episode Duration')
        # plt.xlabel('Episode')
        # plt.ylabel('Steps')
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        
        # plt.tight_layout()
        os.makedirs("plots_SAC", exist_ok=True)
        plt.savefig('plots_SAC/sac_test_results.png')
        print("Test plot saved to plots_SAC/sac_test_results.png")
        plt.show()
    
    env.close()
    return mean_reward, rewards, steps_list

if __name__ == "__main__":
    print(f"Training SAC on FlappyBird environment (device: {device})")
    trained_model = train()
    # trained_model = load_checkpoint("models_SAC/sac_checkpoint_ep500.pt")
    # trained_model = torch.load(f"flappy_sac_policy_{2999}.pt")
    # Test the trained model
    # print("Testing trained policy...")
    # avg_reward = test("models_SAC/sac_checkpoint_ep300.pt", render=True)
    # print(f"Average test reward: {avg_reward:.2f}")