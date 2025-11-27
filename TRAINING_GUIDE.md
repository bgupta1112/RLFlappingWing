# RL Flapping Wing - Training with Plots & Video Recording

## Quick Setup

### 1. Install Dependencies
```bash
cd /home/bibek/Documents/Github/RLFlappingWing
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train and Save Everything
Run the automated script that trains, saves plots, and records videos:
```bash
python3 train_and_record.py
```

## Features

### 📊 Plots Saved Automatically
- **Training plots**: `plots_SAC/flappy_sac_training.png`
  - Episode rewards (with 5-episode moving average)
  - Episode duration
  - Q-network loss
  - Policy loss

- **Test plots**: `plots_SAC/sac_test_results.png`
  - Test episode rewards with smoothing
  - Statistics (mean ± std)

### 🎥 Videos Saved Automatically
- **Test episodes**: `videos/episode_*.mp4`
  - 30 FPS MP4 videos
  - Recorded from MuJoCo renderer
  - Shows agent behavior during testing

### 🤖 Model Checkpoints Saved Automatically
- **Training checkpoints**: `models_SAC/sac_checkpoint_ep*.pt`
  - Saves every 100 episodes
  - Contains: policy, Q-networks, optimizers, normalizer state
  - Can resume training or test from any checkpoint

## Usage Examples

### Train and Get All Outputs
```python
from src.fledgling_SAC_opt import train

# Start training
model = train()
# Automatically saves plots to plots_SAC/flappy_sac_training.png
# Automatically saves checkpoints to models_SAC/sac_checkpoint_ep*.pt
```

### Test a Checkpoint with Plots
```python
from src.fledgling_SAC_opt import test

# Test with plots
mean_reward, rewards, steps = test(
    checkpoint_path="models_SAC/sac_checkpoint_ep300.pt",
    episodes=10,
    plot_rewards=True,  # Saves to plots_SAC/sac_test_results.png
    save_video=False
)
print(f"Average test reward: {mean_reward:.2f}")
```

### Record a Video of Test Episode
```python
from src.fledgling_SAC_opt import record_video_episode

# Record one episode as video
reward, steps = record_video_episode(
    checkpoint_path="models_SAC/sac_checkpoint_ep300.pt",
    output_path="videos",
    episode_num=1,
    fps=30
)
print(f"Video saved! Reward: {reward:.2f}")
```

## Output Directory Structure
```
RLFlappingWing/
├── plots_SAC/
│   ├── flappy_sac_training.png       (generated during training)
│   └── sac_test_results.png          (generated after test)
├── models_SAC/
│   ├── sac_checkpoint_ep100.pt
│   ├── sac_checkpoint_ep200.pt
│   └── sac_checkpoint_ep300.pt
├── videos/
│   └── episode_0001.mp4              (generated during testing)
└── src/
    └── fledgling_SAC_opt.py
```

## Configuration

### Training Parameters (in `fledgling_SAC_opt.py`)
- `max_episodes`: Default 300 (change in `train()` function)
- `batch_size`: Default 512
- `start_steps`: Default 5000 random steps before learning
- `update_every`: Default 20 steps between updates

### Video Parameters (in recording functions)
- `fps`: Frames per second (default 30)
- `output_path`: Directory to save videos (default "videos")
- `max_steps_per_episode`: Default 3500 steps

## Troubleshooting

### Videos Not Saving
```bash
# Check if imageio is installed
python3 -c "import imageio; print(imageio.__version__)"

# If not, install it
pip install imageio imageio-ffmpeg
```

### Plots Not Showing
- Plots are always saved to disk regardless of display
- Check `plots_SAC/` directory for PNG files
- If using headless server, rendering to display will fail but files are still saved

### Missing Checkpoints
- Checkpoints are saved every 100 episodes
- First checkpoint appears after 100 episodes of training
- Check `models_SAC/` directory for available checkpoints

## File Descriptions

| File | Purpose |
|------|---------|
| `fledgling_SAC_opt.py` | Main SAC implementation with train/test/video functions |
| `fledgling.py` | FlappyBird MuJoCo environment |
| `FlappyBird.xml` | MuJoCo physics model definition |
| `train_and_record.py` | Automated training script with all features |
| `requirements.txt` | Python package dependencies |

## Tips for Best Results

1. **First Run**: Training takes time. Let it run for all 300 episodes to get good results.
2. **Monitoring**: Watch `plots_SAC/flappy_sac_training.png` to monitor training progress.
3. **Testing**: Test after training completes to see how well the agent performs.
4. **Videos**: Record videos to visualize agent behavior and debug if needed.
5. **Resuming**: Use `load_checkpoint()` to resume training from a checkpoint.

## System Requirements

- Python 3.8+
- 4GB RAM (8GB+ recommended)
- GPU (CUDA) for faster training, CPU works too
- FFmpeg (installed via `imageio-ffmpeg`)

Enjoy training! 🚀
