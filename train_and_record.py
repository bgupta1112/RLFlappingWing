#!/usr/bin/env python3
"""
Example script to train, save plots, and record videos of the SAC RL agent.
"""

import sys
import os
sys.path.insert(0, 'src')

from fledgling_SAC_opt import train, test, record_video_episode

def main():
    print("=" * 60)
    print("SAC Training with Plots and Video Recording")
    print("=" * 60)
    
    # Option 1: Train the model
    print("\n[1/3] Training SAC agent...")
    print("      Training will save:")
    print("      - Checkpoints to: models_SAC/sac_checkpoint_ep*.pt")
    print("      - Training plots to: plots_SAC/flappy_sac_training.png")
    trained_model = train()
    
    print("\n✓ Training complete!")
    print("  Checkpoints saved in: models_SAC/")
    print("  Plots saved in: plots_SAC/")
    
    # Option 2: Test and save plots
    print("\n[2/3] Testing trained model and saving plots...")
    latest_checkpoint = "models_SAC/sac_checkpoint_ep300.pt"
    if os.path.exists(latest_checkpoint):
        mean_reward, rewards, steps = test(
            checkpoint_path=latest_checkpoint,
            render=False,  # Set to True if you want visual rendering
            episodes=10,
            plot_rewards=True,
            save_video=False  # Set to True to record first episode
        )
        print(f"\n✓ Test complete! Average reward: {mean_reward:.2f}")
        print(f"  Test plots saved to: plots_SAC/sac_test_results.png")
    else:
        print(f"  Checkpoint not found: {latest_checkpoint}")
        print(f"  Available checkpoints: {os.listdir('models_SAC') if os.path.exists('models_SAC') else 'None'}")
    
    # Option 3: Record a video
    print("\n[3/3] Recording test episode video...")
    if os.path.exists(latest_checkpoint):
        reward, steps = record_video_episode(
            checkpoint_path=latest_checkpoint,
            output_path="videos",
            episode_num=1,
            fps=30
        )
        print(f"\n✓ Video recorded!")
        print(f"  Episode reward: {reward:.2f}, Steps: {steps}")
        print(f"  Video saved to: videos/episode_0001.mp4")
    
    print("\n" + "=" * 60)
    print("All files saved:")
    print("  - Training plots: plots_SAC/")
    print("  - Model checkpoints: models_SAC/")
    print("  - Test videos: videos/")
    print("=" * 60)

if __name__ == "__main__":
    main()
