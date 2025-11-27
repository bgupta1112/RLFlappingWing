import gymnasium as gym
from gymnasium import spaces
import mujoco

# Try to import mujoco viewer, fall back gracefully if not available
try:
    import mujoco.viewer as mujoco_viewer
except ImportError:
    try:
        import mujoco_viewer
    except ImportError:
        mujoco_viewer = None

import numpy as np
import time
import os

class FlappyBirdMuJoCoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Load MuJoCo model - find FlappyBird.xml relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, "FlappyBird.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # Define action space (flapping force: -1 to 1)
        # self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(2,), dtype=np.float64)
        # [flap_L, flap_R, pitch_L, pitch_R, tail_angle]
        # self.action_space = spaces.Box(
        #     low=np.array([-1.5, -1.5, -1.5, -1.5, -0.7]),   # Min values for left wing flap, right wing flap, and tail angle
        #     high=np.array([1.5, 1.5, 1.5, 1.5, 0.7]),    # Max values for left wing flap, right wing flap, and tail angle
        #     dtype=np.float64
        # )

        self.action_space = spaces.Box(
            low=np.array([-1.5, -1.5]),#, -0.7]),   # Min values for left wing flap, right wing flap, and tail angle
            high=np.array([1.5, 1.5]),#, 0.7]),    # Max values for left wing flap, right wing flap, and tail angle
            dtype=np.float64
        )

        # Define observation space (position, velocity, and angle)
        # self.observation_space = spaces.Box(
        #     low=np.array([-5, -5, -np.pi, -2, -2, -1]),  # Min values
        #     high=np.array([5, 5, np.pi, 2, 2, 1]),  # Max values
        #     dtype=np.float64
        # )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),  # 3 pos + 4 quat + 3 lin vel + 3 ang vel
            dtype=np.float64
        )
    # def calculate_reward(self):
    #     # Target position
    #     target_pos = np.array([8.0, 0.0, 7.0])
        
    #     # Current position and velocity
    #     current_pos = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]])
    #     current_vel = np.array([self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]])
        
    #     # 1. Distance reward (keep but reduce weight)
    #     distance = np.linalg.norm(current_pos - target_pos)
    #     distance_reward = 5.0 * np.exp(-0.1 * distance)
        
    #     # 2. Forward velocity reward (critical for horizontal flight)
    #     velocity_reward = 4.0 * max(0, current_vel[0])  # Reward positive x-direction velocity
        
    #     # 3. Height maintenance reward (keep at 7.0)
    #     target_height = 7.0
    #     height_diff = abs(current_pos[2] - target_height)
    #     height_reward = 3.0 * np.exp(-0.5 * height_diff)  # Reward being close to target height
        
    #     # 4. Progressive x-position reward (provides gradient throughout journey)
    #     # Scales from 0 to 8 as bird moves from -8 to +8
    #     x_progress = (current_pos[0] + 8.0) / 16.0  # Normalized from 0 to 1
    #     progress_reward = 3.0 * x_progress  # Linear reward based on progress
        
    #     # 5. Height penalty (stronger and starts earlier)
    #     height_penalty = 0.0
    #     if current_pos[2] < 3.0:  # Start penalty earlier
    #         height_penalty = 5.0 * (5.0 - current_pos[2])
        
    #     # 6. Orientation stability (keep from original)
    #     qw, qx, qy, qz = self.data.qpos[3:7]
    #     pitch = np.arctan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy))
    #     roll = np.arcsin(2.0 * (qw*qy - qz*qx))
    #     orientation_penalty = -1.0 * (abs(roll) + abs(pitch))
        
    #     # Combined reward
    #     reward = (
    #         distance_reward + 
    #         velocity_reward + 
    #         height_reward + 
    #         progress_reward - 
    #         height_penalty + 
    #         orientation_penalty + 
    #         0.1  # Small living bonus
    #     )
        
    #     return reward


    # def calculate_reward(self):
    #     # Target position
    #     target_pos = np.array([8.0, 0.0, 7.0])
        
    #     # Current position and velocity
    #     current_pos = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]])
    #     current_vel = np.array([self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]])
        
    #     # Get orientation
    #     qw, qx, qy, qz = self.data.qpos[3:7]
    #     pitch = np.arctan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy))
    #     roll = np.arcsin(2.0 * (qw*qy - qz*qx))
        
    #     # 1. Orientation stability - INCREASED WEIGHT SIGNIFICANTLY
    #     # This is now the primary control factor
    #     orientation_factor = np.exp(-2.0 * (abs(roll) + abs(pitch)))  # Value between 0-1 based on stability
    #     orientation_penalty = -(abs(roll) + abs(pitch))  # Increased from -1.0 to -8.0
        
    #     # 2. Forward velocity reward - NOW TIED TO ORIENTATION
    #     # Only rewards forward velocity when orientation is stable
    #     velocity_reward = 4.0 * max(0, current_vel[0]) * orientation_factor
        
    #     # 3. Height maintenance reward (keep at 7.0)
    #     target_height = 7.0
    #     height_diff = abs(current_pos[2] - target_height)
    #     height_reward = 3.0 * np.exp(-0.5 * height_diff) * orientation_factor  # Only rewards height when stable
        
    #     # 4. Progressive x-position reward - REDUCED WEIGHT
    #     x_progress = (current_pos[0] + 8.0) / 16.0  # Normalized from 0 to 1
    #     progress_reward = 2.0 * x_progress  # Reduced from 3.0 to 2.0
        
    #     # 5. Height penalty (stronger and starts earlier)
    #     height_penalty = 0.0
    #     if current_pos[2] < 4.0:  # Start penalty earlier
    #         height_penalty = 4.0 * (4.0 - current_pos[2])
        
    #     # 6. Distance reward - REDUCED WEIGHT
    #     distance = np.linalg.norm(current_pos - target_pos)
    #     distance_reward = 2.0 * np.exp(-0.1 * distance)  # Reduced from 5.0 to 2.0
        
    #     # Combined reward
    #     reward = (
    #         distance_reward + 
    #         velocity_reward + 
    #         height_reward + 
    #         progress_reward - 
    #         height_penalty + 
    #         orientation_penalty + 
    #         0.1  # Small living bonus
    #     )
        
    #     return reward

    ##uncomment here
    def calculate_reward(self):
        # Define target position
        # target_pos = np.array([8.0, 0.0, 7.0])  # Target x, y, z position
        # target_distance = 8.0
        start_x = -8.0
        # Get current position
        # current_pos = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]])
        
        # Calculate Euclidean distance to target
        # distance = np.linalg.norm(current_pos - target_pos)
        distance = self.data.qpos[0] - start_x #np.linalg.norm(current_pos - target_pos)
        # Convert distance to reward (closer = better)
        # distance_reward = 5.0 * np.exp(-0.1 * distance)
        reward = np.exp(0.1 * distance)
        
        # # Penalty for falling below a certain height
        height_penalty = 0.0
        if self.data.qpos[2] < 3.0:
            height_penalty = (3.0 - self.data.qpos[2])
        
        # Orientation stability reward
        # qw, qx, qy, qz = self.data.qpos[3:7]
        # pitch = np.arctan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy))
        # roll = np.arcsin(2.0 * (qw*qy - qz*qx))
        # orientation_penalty = -1.0 * (abs(roll) + abs(pitch))
        
        # Final reward
        # reward = distance_reward - height_penalty + orientation_penalty + 0.1  # Small living bonus
        return reward + 0.1 - height_penalty
    ###Uncomment here

    # def calculate_reward(self): # osayar bs ye7ayar
    #     # Get current position and velocity
    #     current_pos = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]])
    #     current_vel = np.array([self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]])
        
    #     # 1. Progress reward (normalized from 0-1 as bird moves from -8 to +8)
    #     x_progress = min(1.0, max(0.0, (current_pos[0] + 8.0) / 16.0))
    #     progress_reward = 2.0 * x_progress
        
    #     # 2. Forward velocity reward (small bonus for moving in the right direction)
    #     velocity_reward = 0.8 * max(0, current_vel[0])  # Only reward positive x velocity
        
    #     # 3. Basic height penalty (only activates when too low)
    #     height_penalty = 0.0
    #     if current_pos[2] < 3.0:
    #         height_penalty = 3.0 *(3.0 - current_pos[2])
        
    #     # Final simple reward
    #     reward = progress_reward + velocity_reward - height_penalty + 0.1
        
    #     # Clip reward to reasonable range for stable learning
    #     reward = np.clip(reward, -5.0, 5.0)
        
    #     return reward
    # def calculate_reward(self): #this reward made the bird fly and then btfqd el shaghaf
    #     # Define target position
    #     start_pos = np.array([-8.0, 0.0, 7.0])  # Target x, y, z position
        
    #     # Get current position
    #     current_pos = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]])
        
    #     # Calculate Euclidean distance to target
    #     distance = current_pos[0] - start_pos[0] #np.linalg.norm(current_pos - target_pos)
    #     x_progress = min(1.0, max(0.0, (current_pos[0] + 8.0) / 16.0))
        
    #     # Convert distance to reward (closer = better)
    #     distance_reward = 1.0 * x_progress #np.exp(-0.1 * distance)
        
    #     # Penalty for falling below a certain height
    #     # height_penalty = 0.0
    #     # if self.data.qpos[2] < 3.0:
    #     #     height_penalty = 10.0 * (3.0 - self.data.qpos[2])
    #     # if current_pos[2] < 2.0:
    #     #     # Calculate progress factor (0 at start, approaching 1 at target)
    #     #     # x_progress = min(1.0, max(0.0, (self.data.qpos[0] + 8.0) / 16.0))
            
    #     #     # Reduce penalty as bird progresses further (multiply by (1 - progress_factor))
    #     #     # This creates a penalty that starts at full strength but diminishes as x increases
    #     #     penalty_factor = 1.0 - (0.5 * x_progress)  # At most reduce penalty by 50%
    #     #     height_penalty = 5.0 * (2.0 - current_pos[2]) * penalty_factor
        
    #     # # Orientation stability reward
    #     # qw, qx, qy, qz = self.data.qpos[3:7]
    #     # pitch = np.arctan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy))
    #     # roll = np.arcsin(2.0 * (qw*qy - qz*qx))
    #     # orientation_penalty = -5.0 * (abs(roll) + abs(pitch))
        
    #     # Final reward
    #     # reward = distance_reward - height_penalty + orientation_penalty + 0.1  # Small living bonus
    #     reward = distance_reward + 0.1  # Small living bonus
    #     print(f"Distance: {distance}, Reward: {reward}")
    #     return reward

        # target_dist = 8.0
        # forward = self.data.qpos[0] - 8.0
        # #add the reward for forward distance exponentially
        # reward = 2.0 * np.exp(0.1 * forward)
        #calcluate euclidean distance
        # # Penalty for falling below a certain height
        # height = self.data.qpos[2]
        # if height < 1.1:
        #     reward -= 10.0
        # return reward
    # def calculate_reward(self): #undo comment
    #     # Basic measurements
    #     target_dist = 8.0
    #     height = self.data.qpos[2]
    #     forward = self.data.qpos[0] - self.prev_x
    #     self.prev_x = self.data.qpos[0]
        
    #     # Track wing positions for cycle detection
    #     left_wing_pos = self.data.qpos[7]  # J_flap_L position
    #     right_wing_pos = self.data.qpos[9]  # J_flap_R position
        
    #     # Track wing position changes for flapping detection
    #     if not hasattr(self, 'prev_left_pos'):
    #         self.prev_left_pos = left_wing_pos
    #         self.prev_right_pos = right_wing_pos
    #         self.cycle_timer = 0
    #         self.last_cycle_time = 0
        
    #     # Update cycle timer
    #     self.cycle_timer += 1
        
    #     # Detect wing movement patterns (zero crossings)
    #     frequency_reward = 0
    #     if (self.prev_left_pos * left_wing_pos <= 0) or (self.prev_right_pos * right_wing_pos <= 0):
    #         cycle_duration = self.cycle_timer - self.last_cycle_time
    #         self.last_cycle_time = self.cycle_timer
            
    #         # Reward shorter cycles (higher frequency)
    #         frequency_reward = 2.0 * np.exp(-0.05 * max(cycle_duration, 1))
        
    #     # Update previous positions
    #     self.prev_left_pos = left_wing_pos
    #     self.prev_right_pos = right_wing_pos
        
    #     # Track velocity changes for acceleration reward
    #     current_vel_x = self.data.qvel[0]
    #     if not hasattr(self, 'prev_vel_x'):
    #         self.prev_vel_x = current_vel_x
        
    #     # Reward effective acceleration
    #     accel_x = current_vel_x - self.prev_vel_x
    #     accel_reward = 1.5 * max(0, accel_x)
    #     self.prev_vel_x = current_vel_x
        
    #     # Primary rewards
    #     height_reward = 1.5 * height
    #     forward_reward = 4.0 * forward
        
    #     # Detect wing pattern (proper flapping)
    #     wing_sync = 0.5 * np.exp(-2.0 * (left_wing_pos + right_wing_pos)**2)
        
    #     # Penalties
    #     angular_vel = np.abs(self.data.qvel[3:6]).sum()
    #     stability_penalty = 0.2 * angular_vel
    #     crash_penalty = 15.0 if height < 1.1 else 0.0
        
    #     # Combined reward
    #     reward = (
    #         forward_reward +     # Main goal
    #         height_reward +      # Stay airborne
    #         frequency_reward +   # Flap faster
    #         accel_reward +       # Generate effective propulsion
    #         wing_sync -          # Synchronize wings properly
    #         stability_penalty -  # Don't tumble
    #         crash_penalty        # Don't crash
    #     )
    
    #     return reward    
    # def calculate_reward(self):
    #     # Height component: Reward for staying at an optimal height
    #     height = self.data.qpos[2]
    #     forward = self.data.qpos[0] + 8
    #     # forward = (self.data.qpos[0] - self.prev_x)
    #     # self.prev_x = self.data.qpos[0]
    #     reward = height + 2*forward

    #     # reward = 2*height + 1.25 * forward

        
    #     # Add a small living reward
    #     reward += 0.1
        
    #     return reward
    
    # def calculate_reward(self): #simple reward
    #     height = self.data.qpos[2]
    #     forward = self.data.qpos[0] - self.prev_x
    #     self.prev_x = self.data.qpos[0]

    #     # Base rewards
    #     reward = 2.0 * height + 2.5 * forward  # Encourage staying high and going far

    #     # Penalty for flipping/tumbling too much
    #     angular_vel = np.abs(self.data.qvel[3:6]).sum()
    #     reward -= 0.1 * angular_vel

    #     # # Optional: Encourage smoother control (less spammy input)
    #     # reward -= 0.01 * np.linalg.norm(action)

    #     # Crash penalty
    #     if height < 1.1:
    #         reward -= 10.0

    #     return reward

    # def calculate_reward(self): very simple forward reward
    #     forward = self.data.qpos[0] 




    # def calculate_reward(self): #complex reward
    #     height = self.data.qpos[2]
    #     forward = self.data.qpos[0] - self.prev_x
    #     self.prev_x = self.data.qpos[0]
        
    #     # Track wing positions and velocities
    #     left_wing_pos = self.data.qpos[7]  # J_flap_L position
    #     right_wing_pos = self.data.qpos[9]  # J_flap_R position
    #     left_wing_vel = self.data.qvel[7]  # J_flap_L velocity
    #     right_wing_vel = self.data.qvel[9]  # J_flap_R velocity
        
    #     # Base rewards
    #     height_reward = 1.5 * height  # Still important but slightly reduced
    #     forward_reward = 3.0 * forward  # Increased to prioritize forward motion
        
    #     # NEW: Flapping frequency reward
    #     # Higher absolute velocities = faster flapping
    #     flap_velocity = (np.abs(left_wing_vel) + np.abs(right_wing_vel)) / 2.0
    #     flap_reward = 1.0 * min(flap_velocity, 5.0)  # Cap to prevent excessive values
        
    #     # NEW: Synchronized flapping reward
    #     # Wings should move in opposite directions (proper flapping)
    #     sync_reward = 0.5 * np.exp(-2.0 * (left_wing_pos + right_wing_pos)**2)
        
    #     # Efficiency reward: forward progress per energy used
    #     efficiency = forward / (np.abs(self.data.ctrl).sum() + 0.1)  # Avoid division by zero
    #     efficiency_reward = 0.5 * efficiency
        
    #     # Stability penalty (keep this)
    #     angular_vel = np.abs(self.data.qvel[3:6]).sum()
    #     stability_penalty = 0.1 * angular_vel
        
    #     # Crash penalty (keep this)
    #     crash_penalty = 10.0 if height < 1.1 else 0.0
        
    #     # Combined reward
    #     reward = (height_reward + 
    #             forward_reward + 
    #             flap_reward + 
    #             sync_reward + 
    #             efficiency_reward - 
    #             stability_penalty - 
    #             crash_penalty)
        
        #return reward
        # height = self.data.qpos[2]
        # height_target = 7.0  # Start height
        # height_reward = 2.0 * np.exp(-0.5 * ((height - height_target) / 2.0) ** 2)
        
        # # Strong penalty for falling too low - this is critical
        # ground_penalty = 0.0
        # if height < 4.0:  # If below 5 units height
        #     ground_penalty = -6.0 * (5.0 - height)**2  # Quadratic penalty gets stronger as it falls
        
        # # # 2. Wing synchronization reward - Encourage coordinated flapping
        # # left_wing_pos = self.data.qpos[7]  # J_flap_L position
        # # right_wing_pos = self.data.qpos[9]  # J_flap_R position
        # # left_wing_vel = self.data.qvel[7]  # J_flap_L velocity
        # # right_wing_vel = self.data.qvel[9]  # J_flap_R velocity
        
        # # # Reward when wings move in sync (similar positions but opposite sign)
        # # # This encourages proper flapping (both wings up or both wings down)
        # # wing_sync_reward = 3.0 * np.exp(-5.0 * (left_wing_pos - right_wing_pos)**2)
        
        # # # 3. Flapping motion reward - Encourage actual flapping
        # # # Reward higher velocities but with same sign (moving in same direction)
        # # flapping_magnitude = (np.abs(left_wing_vel) + np.abs(right_wing_vel)) / 2.0
        # # flapping_sync = np.exp(-2.0 * (np.sign(left_wing_vel) - np.sign(right_wing_vel))**2)
        # # flapping_reward = 2.0 * flapping_magnitude * flapping_sync
        
        # # Cap the reward for extremely fast movements
        # # flapping_reward = min(flapping_reward, 3.0)
        
        # # 4. Forward position - Progress from starting position
        # # Bird starts at x = -8, reward movement to the right
        # forward_pos = self.data.qpos[0] + 8.0  # Zero at start position
        # forward_reward = 2.0 * forward_pos
        
        # # # 5. Energy efficiency - Slight penalty for control inputs
        # # energy_penalty = -0.01 * (
        # #     abs(self.data.ctrl[0]) + abs(self.data.ctrl[1]) + 
        # #     abs(self.data.ctrl[2]) + abs(self.data.ctrl[3]) + 
        # #     abs(self.data.ctrl[8])
        # # )
        
        # # 6. Orientation stability - Use quaternion to get proper orientation
        # # Extract roll, pitch from quaternion
        # qw, qx, qy, qz = self.data.qpos[3:7]
        
        # # Calculate pitch and roll from quaternion
        # pitch = np.arctan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy))
        # roll = np.arcsin(2.0 * (qw*qy - qz*qx))
        # yaw = np.arctan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz))
        
        # # Penalize non-horizontal orientation
        # orientation_penalty = -2.0 * (abs(roll) + abs(pitch))#(abs(pitch) + abs(roll))
        
        # # 7. Living bonus - Small reward for survival
        # survival_bonus = 0.2
        
        # # Combine rewards with appropriate weights
        # reward = (
        #     height_reward + 
        #     # wing_sync_reward + 
        #     # flapping_reward + 
        #     forward_reward + 
        #     ground_penalty +
        #     #energy_penalty + 
        #     orientation_penalty + 
        #     survival_bonus
        # )
        
        #return reward

    def step(self, action):
        # Apply action (flapping forces)
        # self.data.ctrl[4] = 1.0
        # self.data.ctrl[6] = 1.0
        self.data.ctrl[0] = action[0]  # Left wing
        self.data.ctrl[1] = 0 # Left wing pitch
        self.data.ctrl[2] = action[1] # Right wing
        self.data.ctrl[3] = 0 # Right wing pitch
        #self.data.ctrl[8] = action[2] # tail angle
        # Step simulation
        mujoco.mj_step(self.model, self.data)

        quat = self.data.qpos[3:7]  
        lin_vel = self.data.qvel[:3]  
        ang_vel = self.data.qvel[3:6]

        # obs = np.concatenate([
        #     self.data.qpos[0:3],  # x, y, z
        #     quat,                
        #     lin_vel,
        #     ang_vel
        # ])
        # # Get observation (bird position, velocity, and angle)
        obs = np.array([
            self.data.qpos[0],  # X position
            self.data.qpos[2],  # Z position (height)
            self.data.qpos[3],  # Angle (rotation)
            self.data.qvel[0],  # X velocity
            self.data.qvel[2],  # Z velocity
            self.data.qvel[3],  # Angular velocity
        ])

        # print(f"Obs: {obs}")
        # Reward: Keep flapping & moving forward
        reward = self.calculate_reward() # Reward for forward movement


        # Done if bird falls below a certain height
        done = self.data.qpos[2] < 2

        return obs, reward, done, False, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.prev_x = self.data.qpos[0]
        self.prev_left_pos = self.data.qpos[7]
        self.prev_right_pos = self.data.qpos[9]
        self.prev_vel_x = self.data.qvel[0]
        self.cycle_timer = 0
        self.last_cycle_time = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.data.qpos[0], self.data.qpos[2], self.data.qpos[3],
            self.data.qvel[0], self.data.qvel[2], self.data.qvel[3]
        ])

    def _get_obs(self):
        pos_x = self.data.qpos[0] / 10.0       # assuming range [-10, 10]
        pos_z = self.data.qpos[2] / 10.0
        angle = self.data.qpos[3]              # already between [-1, 1] if quaternion
        vel_x = self.data.qvel[0] / 5.0        # normalize velocities
        vel_z = self.data.qvel[2] / 5.0
        ang_vel = self.data.qvel[3] / 5.0

        return np.array([pos_x, pos_z, angle, vel_x, vel_z, ang_vel], dtype=np.float64)


    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None